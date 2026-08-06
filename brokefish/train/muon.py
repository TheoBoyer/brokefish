"""Muon, matching `torch.optim.Muon` exactly, plus the granularity knob it lacks.

Ablation 2. `docs/journal/2026-08-05-muon-survey.md` is the survey this implements;
read its addendum before touching :func:`chunk_spec`.

Muon replaces SGD-momentum's update with the *nearest semi-orthogonal matrix* to the
momentum buffer -- `U V^T` from its SVD -- computed by a quintic Newton-Schulz
iteration rather than by an SVD. It applies to hidden matmul weights only; embeddings,
the readout heads, and every scalar/vector stay on AdamW, which is why this file
carries an AdamW too. One optimiser object, so one `state_dict`, so resume keeps
working.

## The oracle

⚠️ **`torch.optim.Muon` exists in our stack** (torch 2.13,
`.venv/.../torch/optim/_muon.py`) and this file reproduces it line for line on the
unchunked path -- momentum, Nesterov, the Newton-Schulz iteration, the epsilon
placement, the learning-rate adjustment, the weight-decay ordering, and the bf16
return dtype. `tests/test_muon.py` asserts bit-equality against it. The same test
asserts bit-equality of the auxiliary AdamW against `torch.optim.AdamW`.

That is the whole point: **the only thing here that torch does not do is the
chunking**, so the only thing a test has to reason about is the chunking. Everything
else has an oracle, which is how the rest of this repository is built.

## The chunking, and the one formula that had to be generalised

`chunk_spec` cuts a fused tensor into functionally distinct blocks before
orthogonalising. Jordan's post is the primary source and it is unambiguous:
*"Muon works better for optimizing transformers if it is applied to their Q, K, V
parameters separately, rather than together."* CMuon (arXiv 2608.02502) names the
mechanism -- one fused tensor gets one shared preconditioner `(sum_j G_j^T G_j)^-1/2`
across blocks whose principal directions need not align -- and Kimi K3 / GLM-5 push
the same idea down to individual attention heads.

⚠️ **`_adjust_ratio` is the piece that is genuinely ours and it is easy to get wrong.**
torch's `_adjust_lr` multiplies by `sqrt(max(1, A/B))`, whose *purpose* (its docstring
says so) is to make the update's RMS a consistent `1 / sqrt(fan_in)` whatever the
matrix's aspect ratio -- that consistency is exactly what makes a learning rate
transfer. Chunking breaks the derivation, because the Frobenius norm of a
concatenation of orthogonalised blocks is `sqrt(sum_i min(r_i, c_i))`, not
`sqrt(min(R, C))`. Re-deriving under the same target RMS gives

    ratio = sqrt( R / sum_i min(r_i, c_i) )

which **reduces to torch's `sqrt(max(1, A/B))` exactly when there is one chunk**
(test `test_adjust_ratio_matches_torch_unchunked`), and which is the version that
still transfers. Applying torch's formula per chunk instead is the trap: on
`out_proj` split into 8 head-blocks of (256, 32) it inflates the step by `sqrt(8)`,
so a `head_group` sweep would secretly also be a learning-rate sweep.

A consequence worth stating, because it is reassuring and it is testable: for
row-chunking into blocks at least as wide as they are tall -- which is every split
this network does -- the ranks add up exactly, the ratio is 1, and **the QKV split
changes the update's direction without changing its size.**

## What is deliberately not here

*Polar Express coefficients* (arXiv 2505.16932) are strictly better than the fixed
quintic. They are absent because I do not have their per-step coefficient table from
a primary source and inventing one would be worse than shipping the legacy triple.
`ns_coefficients` is per-iteration for exactly that reason: pass five triples instead
of one and nothing else changes.

*Gram Newton-Schulz* (42 % fewer FLOPs) is absent because the whole iteration costs
~1.3 ms against an 18.8 s self-play phase -- under 0.01 % of a step. Buying 42 % of
nothing at the price of a documented half-precision instability is a bad trade.

*Momentum warmup* (modded-nanogpt ramps `beta` 0 -> 300 steps) and *NorMuon* (a
current speedrun record, PR #144) are one variable too many for a first paired run
against AdamW. NorMuon is implemented and off; momentum warmup is not implemented.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

# torch/optim/_muon.py's constants, which are Keller Jordan's. Not minimax-optimal per
# step -- see the module docstring on Polar Express.
NS_COEFFS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_STEPS: int = 5
NS_EPS: float = 1e-7


def newton_schulz(G: torch.Tensor, ns_steps: int = NS_STEPS,
                  ns_coefficients=NS_COEFFS, eps: float = NS_EPS) -> torch.Tensor:
    """`phi(x) = a x + b x^3 + c x^5` on the singular values. Returns **bfloat16**.

    Line-for-line `torch.optim._muon._zeropower_via_newtonschulz`, with the single
    extension that `ns_coefficients` may be a sequence of `ns_steps` triples so a
    per-iteration schedule drops in.

    ⚠️ **The bf16 return dtype is not an oversight, it is torch's behaviour**, and it
    is load-bearing for the oracle test: `param.add_(update, alpha=-lr)` promotes, so
    the applied update is bf16-rounded. Casting back to fp32 here would be *more*
    precise and would no longer be Muon-as-shipped.

    The transpose is not cosmetic. `X @ X.T` is `min_dim x min_dim`, so putting the
    short side on the rows makes the loop's two matmuls as small as possible -- for a
    (1024, 256) tensor that is 256x256 instead of 1024x1024, 16x less work per
    iteration for an identical result.
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz wants a matrix, got {tuple(G.shape)}")
    coeffs = list(ns_coefficients)
    if not isinstance(coeffs[0], (list, tuple)):
        coeffs = [tuple(coeffs)] * ns_steps
    if len(coeffs) < ns_steps:
        raise ValueError(f"{len(coeffs)} coefficient triples for {ns_steps} steps")

    transposed = G.size(0) > G.size(1)
    # `.contiguous()` first: chunk views are strided, and the in-place `div_` below
    # then writes through a view of a view. It is a copy either way, so this costs
    # nothing and removes a class of aliasing bug.
    ortho = G.contiguous().bfloat16()
    if transposed:
        ortho = ortho.T
    # Ensure the spectral norm is at most 1 -- Frobenius bounds it and needs no
    # iteration of its own. `clamp`, not `+ eps`: torch's choice, and it leaves a
    # well-scaled matrix untouched instead of shrinking it slightly.
    ortho = ortho / ortho.norm().clamp(min=eps)
    for i in range(ns_steps):
        a, b, c = coeffs[i]
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)   # b*A + c*A@A
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)         # a*X + B@X
    if transposed:
        ortho = ortho.T
    return ortho


# -- which parameters are Muon-shaped, and how finely they are cut ----------- #

def chunk_spec(name: str, n_heads: int = 8, qkv_split: bool = True,
               head_group: int = 8) -> Tuple[int, int]:
    """`(dim, n_chunks)` -- the axis to cut along and into how many pieces.

    ⚠️ **`in_proj` and `out_proj` chunk on opposite axes, and getting this backwards
    produces a running, converging, wrong optimiser.**

    `in_proj_weight` is `[Q; K; V]` stacked on **rows**, and inside each 256-row block
    head `h` owns rows `32h .. 32h+32`. So `dim = 0`, and `3 * (n_heads // g)` equal
    contiguous chunks land exactly on the (projection, head-group) boundaries.

    `out_proj.weight` carries its head structure on the **input** dimension: columns
    `32h .. 32h+32` are head `h`'s contribution to the residual stream. So `dim = 1`.

    `head_group = 8` means one group holding all eight heads, i.e. no head splitting;
    `head_group = 1` is per-head, which is Kimi K3's and GLM-5's "Muon Split". The
    ladder is `g in {1, 2, 4, 8}` and arXiv 2605.08933 shows the optimum *moves during
    training* and that both extremes lose somewhere -- so it is a hyperparameter, not
    a constant.
    """
    if n_heads % head_group != 0:
        raise ValueError(f"head_group {head_group} does not divide n_heads {n_heads}")
    groups = n_heads // head_group

    if name.endswith("in_proj_weight"):
        if not qkv_split:
            if groups != 1:
                raise ValueError(
                    "head_group < n_heads needs qkv_split=True: cutting 768 rows into "
                    f"{groups} pieces straddles the Q/K/V boundaries and means nothing")
            return 0, 1
        return 0, 3 * groups
    if name.endswith("out_proj.weight"):
        return 1, groups
    # linear1 / linear2 have no head structure and are fused with nothing -- our FFN
    # is two separate modules, so CMuon's gate+up chunking has no analogue here.
    return 0, 1


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    """Hidden matmul weights only: 6,291,456 parameters, 98.6 % of this network.

    Embeddings and the three readout heads are `ndim == 2` and are still excluded,
    which is Jordan's rule (*"the input and output layers should be optimized by a
    standard method such as AdamW"*). It matters more here than usual because
    `value.weight` is `(1, 256)`, and the nearest semi-orthogonal matrix to a single
    row is that row's direction with its magnitude discarded.
    """
    return p.ndim == 2 and (
        name.endswith("in_proj_weight")
        or name.endswith("out_proj.weight")
        or name.endswith("linear1.weight")
        or name.endswith("linear2.weight"))


def muon_param_groups(net: torch.nn.Module, *, lr: float, aux_lr: float, wd: float,
                      n_heads: int = 8, qkv_split: bool = True, head_group: int = 8,
                      betas: Tuple[float, float] = (0.9, 0.95), momentum: float = 0.95,
                      eps: float = 1e-8) -> List[dict]:
    """Three groups: Muon matrices, AdamW matrices, AdamW flats.

    `aux_lr` is carried as `lr_scale` rather than as an absolute rate, so
    `TrainConfig.lr_at`'s one number keeps driving every group and the warmup and the
    cosine apply to all three. `Trainer.train_step` multiplies by `lr_scale`.

    ⚠️ **`lr` here must be the schedule's *peak*, not `lr_at(0)`.** With warmup the
    step-0 rate is `peak / warmup_steps`, and a ratio taken against that would give
    the auxiliary group a rate `warmup_steps` times too large for the whole run.

    The `ndim < 2` group keeps `weight_decay = 0` for the reason `build_optimizer`
    already gives: shrinking a per-channel LayerNorm gain toward zero is not
    regularisation, it scales the layer's output down and the next layer scales it
    back up.
    """
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    muon, aux_decay, aux_flat, specs = [], [], [], []
    for n, p in named:
        if is_muon_param(n, p):
            muon.append(p)
            specs.append(chunk_spec(n, n_heads, qkv_split, head_group))
        elif p.ndim >= 2:
            aux_decay.append(p)
        else:
            aux_flat.append(p)

    scale = (aux_lr / lr) if lr else 1.0
    return [
        {"params": muon, "use_muon": True, "chunks": specs, "lr": lr,
         "lr_scale": 1.0, "weight_decay": wd, "momentum": momentum, "nesterov": True},
        {"params": aux_decay, "use_muon": False, "lr": aux_lr, "lr_scale": scale,
         "weight_decay": wd, "betas": betas, "eps": eps},
        {"params": aux_flat, "use_muon": False, "lr": aux_lr, "lr_scale": scale,
         "weight_decay": 0.0, "betas": betas, "eps": eps},
    ]


def adjust_ratio(rows: int, cols: int, dim: int, n_chunks: int) -> float:
    """The learning-rate multiplier that keeps the update's RMS at `1 / sqrt(fan_in)`.

    `sqrt(R / sum_i min(r_i, c_i))`. See the module docstring for the derivation and
    for why the obvious alternative -- torch's formula applied per chunk -- silently
    couples `head_group` to the effective learning rate.
    """
    if dim == 0:
        r, c = rows // n_chunks, cols
    else:
        r, c = rows, cols // n_chunks
    return math.sqrt(rows / (n_chunks * min(r, c)))


# -- the optimiser ---------------------------------------------------------- #

class Muon(torch.optim.Optimizer):
    """Muon on the flagged groups, AdamW on the rest. One object, one state dict.

    ⚠️ **`normuon` is my reconstruction of arXiv 2510.05491 from a search summary, not
    from the paper.** The mechanism is right -- Muon flattens the *matrix* condition
    number but leaves neuron norms non-uniform, so a second moment is kept per output
    row and the result renormalised to preserve the Frobenius norm Muon would have
    produced -- but the bias correction and the epsilon placement are mine. Off by
    default; read the paper before turning it on for a run that matters.
    """

    def __init__(self, param_groups: Iterable[dict], *, ns_steps: int = NS_STEPS,
                 ns_coefficients=NS_COEFFS, ns_eps: float = NS_EPS,
                 normuon: bool = False, normuon_beta: float = 0.95):
        defaults = dict(lr=0.0, lr_scale=1.0, weight_decay=0.0, momentum=0.95,
                        nesterov=True, betas=(0.9, 0.95), eps=1e-8,
                        use_muon=False, chunks=None)
        super().__init__(list(param_groups), defaults)
        self.ns_steps, self.ns_coefficients, self.ns_eps = ns_steps, ns_coefficients, ns_eps
        self.normuon, self.normuon_beta = normuon, normuon_beta

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon"):
                self._muon_group(group)
            else:
                self._adamw_group(group)
        return loss

    # -- Muon: torch/optim/_muon.py's `_single_tensor_muon`, plus chunking ---- #

    def _muon_group(self, group: dict) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        momentum, nesterov = group["momentum"], group["nesterov"]
        for p, (dim, n_chunks) in zip(group["params"], group["chunks"]):
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]
            # EMA, not heavy-ball -- torch's *code* does this even though its docstring
            # writes `B_t = mu B_{t-1} + g_t`. The two differ by a constant 1/(1-mu)
            # that Newton-Schulz normalises away, so the orthogonalised direction is
            # identical; the EMA form is followed because it is what actually ships.
            buf.lerp_(p.grad, 1 - momentum)
            update = p.grad.lerp(buf, momentum) if nesterov else buf

            update = self._orthogonalise(update, dim, n_chunks)
            if self.normuon:
                update = self._normuon(state, update)
            adjusted_lr = lr * adjust_ratio(p.size(0), p.size(1), dim, n_chunks)
            # ⚠️ Decay uses the *unadjusted* `lr`, and lands before the update. Both
            # are torch's ordering; swapping either is a different optimiser.
            p.mul_(1 - lr * wd)
            p.add_(update, alpha=-adjusted_lr)

    def _orthogonalise(self, g: torch.Tensor, dim: int, n_chunks: int) -> torch.Tensor:
        """Cut, orthogonalise each piece independently, reassemble.

        Independently is the entire point: one Newton-Schulz over a fused tensor
        applies one shared preconditioner across blocks whose principal directions
        need not align -- CMuon's "subspace interference", and Kimi K3's "heads with
        larger momentum scale dominate the shared update direction".
        """
        ns = lambda m: newton_schulz(m, self.ns_steps, self.ns_coefficients, self.ns_eps)
        if n_chunks == 1:
            return ns(g)
        return torch.cat([ns(c) for c in g.chunk(n_chunks, dim=dim)], dim=dim)

    def _normuon(self, state: dict, o: torch.Tensor) -> torch.Tensor:
        if "normuon_v" not in state:
            state["normuon_v"] = torch.zeros(o.size(0), device=o.device,
                                             dtype=torch.float32)
            state["normuon_step"] = 0
        state["normuon_step"] += 1
        v, b, t = state["normuon_v"], self.normuon_beta, state["normuon_step"]
        v.mul_(b).add_(o.float().pow(2).mean(dim=1), alpha=1 - b)
        scaled = o / (v / (1 - b ** t)).sqrt().add(1e-8).unsqueeze(1).to(o.dtype)
        # Renormalise to the Frobenius norm plain Muon would have produced, so NorMuon
        # redistributes the update across neurons without resizing it -- and so a
        # learning rate found for plain Muon still means the same thing.
        return scaled * (o.norm() / scaled.norm().clamp(min=1e-12))

    # -- AdamW, for the 91,904-parameter residue ----------------------------- #

    def _adamw_group(self, group: dict) -> None:
        """`torch.optim.AdamW`'s single-tensor path, reproduced exactly.

        ⚠️ The epsilon sits *outside* the second-moment bias correction --
        `sqrt(v)/sqrt(bc2) + eps`, not `sqrt(v/bc2 + eps)` and not `sqrt(vhat) + eps`.
        Jordan's `MuonWithAuxAdam` writes the third form with `eps = 1e-10`; torch
        writes the first with `1e-8`. This follows torch, because the control run
        `t7h-fp8` was `torch.optim.AdamW` and the residue has to be the same optimiser
        in both arms or the ablation has two variables.
        """
        lr, wd, (b1, b2), eps = (group["lr"], group["weight_decay"],
                                 group["betas"], group["eps"])
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
            m, v, t = state["exp_avg"], state["exp_avg_sq"], state["step"]
            p.mul_(1 - lr * wd)
            m.lerp_(p.grad, 1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
            denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
            p.addcdiv_(m, denom, value=-lr / bc1)
