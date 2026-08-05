"""Weight synchronisation, ``docs/train.md`` §8.1 — and the silent failure it invites.

The network exists in two representations. Training needs a backward pass, so the
gradient phase runs ``nn/model.py`` in torch with fp32 master weights. Self-play runs
the fused encoder, whose weights are pre-permuted into mma B-fragment order by
``pack_b`` — that permutation is the whole reason the kernel reaches 61.6k evals/s —
and ``cuda_impl.FusedEncoder.__init__(source)`` builds it by ``.detach()``ing from a
torch module. **It is a snapshot.**

⚠️ **A snapshot built once is a working system that never learns.** Construct the
fused encoder in a constructor and self-play runs generation-0 weights for the entire
run, while every loss curve looks healthy, every counter looks healthy, and the Elo
curve is flat for a reason no diagnostic in §11 reports. This is the same shape as the
C1 bug where ``env.push_history`` returned a *new* ring and a dict still pointed at the
old one; the symptom appeared three moves and eight thousand simulations later.

So the countermeasure is not a comment. The packed encoder carries the generation it
was built from, the self-play phase asserts that it equals the loop's current
generation before a single search runs, and a fingerprint of the master weights is
compared against the one taken at pack time so a rebuild that packed the wrong module
fails loudly too. Cost is one integer compare and one 6.4M-element reduction per
phase, against a failure mode that costs a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from brokefish.nn import encoder_impl


@torch.no_grad()
def weight_fingerprint(net) -> float:
    """One float that changes if any parameter does.

    An index-weighted double-precision sum, so it is sensitive to a permutation of
    the same values as well as to a change in them — which matters here, because the
    failure being guarded against is precisely a permutation (``pack_b``) applied to
    the wrong tensor. Deterministic for a fixed device and shape, which is what lets
    it be stored in a checkpoint and compared after a resume.
    """
    acc = 0.0
    for i, (_, p) in enumerate(sorted(net.named_parameters(), key=lambda kv: kv[0])):
        f = p.detach().reshape(-1).double()
        idx = torch.arange(1, f.numel() + 1, device=f.device, dtype=torch.float64)
        acc += float(((i + 1) * f * (idx % 9973.0 + 1.0)).sum())
    return acc


@dataclass
class PackedWeights:
    """A fused encoder, plus the two facts that say whether it is still the right one."""

    encoder: object
    weight_gen: int
    fingerprint: float

    @classmethod
    def pack(cls, net, weight_gen: int, impl: str = "cuda",
             fp8: bool = False) -> "PackedWeights":
        """Permute the master weights into the kernel's order. Milliseconds.

        ⚠️ `fp8` puts the FFN's two matmuls in e4m3 (`csrc/fp8_gemm.cuh`). It is
        **inference only** -- the gradient step runs the fp32 master weights through
        the torch module -- so it cannot destabilise the optimiser; the whole effect
        is slightly noisier self-play. Measured: 1.15x encoder throughput at 1.05 %
        max prior-space error.
        """
        kw = {"fp8": True} if fp8 else {}
        if fp8 and impl != "cuda":
            raise ValueError(f"fp8 is a CUDA-kernel feature; impl is {impl!r}")
        return cls(encoder=encoder_impl(impl)(net, **kw), weight_gen=int(weight_gen),
                   fingerprint=weight_fingerprint(net))

    def evaluate(self):
        """The evaluator the search wants: ``(boards, control, rep) -> fp16 logits``."""
        return self.encoder.forward_full

    def assert_current(self, net, weight_gen: int) -> None:
        """§8.1. Raises rather than silently self-playing a stale network."""
        if self.weight_gen != int(weight_gen):
            raise AssertionError(
                f"train.md §8.1: self-play is about to run weights packed at generation "
                f"{self.weight_gen} while the loop is at generation {weight_gen}. The "
                f"packed encoder is a snapshot and has to be rebuilt after every "
                f"gradient phase; a run that does not is a run that never learns.")
        got = weight_fingerprint(net)
        if got != self.fingerprint:
            raise AssertionError(
                f"train.md §8.1: the master weights changed since they were packed "
                f"(fingerprint {self.fingerprint!r} at pack time, {got!r} now) without "
                f"the generation moving. Either a gradient step ran outside the loop's "
                f"accounting, or the wrong module was packed.")
