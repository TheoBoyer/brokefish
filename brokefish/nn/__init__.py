"""The encoder, and the implementations of it.

Every implementation exposes the same class name and the same contract, so a
caller switches by changing one string rather than one import. There are two
entry points, and which one you get depends on what you construct it from.

The whole network, which is what a self-play step wants — boards in, logits out
(spec §7). Fixed by :mod:`tests.test_b2`::

    net = BrokefishNet().cuda().half().eval()
    model = encoder_impl("cuda")(net)
    policy_logits, promo, value = model.forward_full(boards, control, rep)
    # boards [n, 32] int16, control [n] int16, rep [n] uint8
    # policy_logits [n, 32, 64] fp16 RAW -- the legality mask is the search's
    # promo [n, 32, 4] fp16, value [n] fp32 already through tanh

The encoder stack alone, which is the A/B control and the historical
measurement. Fixed by :mod:`tests.test_model`::

    enc = torch.nn.TransformerEncoder(layer, num_layers=8).cuda().half().eval()
    model = encoder_impl("cuda")(enc)
    out = model(x, alive)          # x [n_boards, 32, 256] fp16, alive [n_boards, 32]

Either way: constructed from a ``norm_first`` ReLU stack in fp16 on CUDA, holding
its own buffers, allocating nothing per call on the backbone path, returning views
that the next call may overwrite. How an implementation lays its weights out
internally is its own business, and so is how much of the work it fuses — the CUDA
one runs the gather, the final norm and the heads inside its single launch, the
Triton one runs them as torch ops around its kernel. The source module is the same
for both, which is what makes them comparable.

The point of the registry is measurement. Both implementations live in one
process, so ``bench/bench_model.py`` can interleave them in an order-balanced
duel. A cross-run before/after on this machine is worth nothing: clocks drift by
±3 % and that is larger than most deltas worth chasing (docs/perf.md).
"""

from __future__ import annotations

IMPLEMENTATIONS = ("triton", "cuda")

_WHY: dict[str, str] = {}


def encoder_impl(name: str):
    """The ``FusedEncoder`` class of one implementation.

    Raises ``ValueError`` for an unknown name and lets the implementation's own
    import error through otherwise -- a CUDA kernel that fails to compile has to
    say so, not disappear.
    """
    if name == "triton":
        from brokefish.nn.triton_impl import FusedEncoder
        return FusedEncoder
    if name == "cuda":
        from brokefish.nn.cuda_impl import FusedEncoder
        return FusedEncoder
    raise ValueError(f"unknown implementation {name!r}, expected one of {IMPLEMENTATIONS}")


def available() -> list[str]:
    """The implementations importable here, in registry order.

    An implementation that fails to import is skipped rather than fatal, since
    the CUDA one needs a toolchain that a fresh machine may not have. The reason
    is kept and :func:`why_unavailable` prints it, so a silent skip is never the
    end of the story -- the harnesses report what they did not run.
    """
    found = []
    for name in IMPLEMENTATIONS:
        try:
            encoder_impl(name)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            _WHY[name] = f"{type(exc).__name__}: {exc}"
        else:
            found.append(name)
            _WHY.pop(name, None)
    return found


def why_unavailable() -> dict[str, str]:
    """Why each missing implementation is missing. Call :func:`available` first."""
    return dict(_WHY)


__all__ = ["IMPLEMENTATIONS", "encoder_impl", "available", "why_unavailable"]
