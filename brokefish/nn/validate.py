"""Numerical QA for a **trained** checkpoint: does the fused encoder still agree?

    python -m brokefish.nn.validate --checkpoint checkpoints/t7h-n128-collapse-002006.pt \
        --buffer t7h-n128-collapse

⚠️ **Why this exists at all.** Every correctness test the repository had ran on an
*untrained* net, and two of them do not run a net at all: `tests/test_model.py`
drives the megakernel with `torch.randn` activations, and `tests/test_b2.py` runs the
full boards-to-logits path on `BrokefishNet()` at random init. Both then measure in
**logit space**, relative to the largest magnitude, at `REL_TOL = 5e-3`.

That bar loosens as training sharpens the policy, and it does so exponentially.
`_rel` permits an absolute logit error of `5e-3 * max|logit|`, and the softmax turns
an absolute logit error `d` into a ratio error up to `exp(2d)`:

    max|logit|      permitted error in probability space
       3  (random init)                ~3 %
      10                              ~10 %
      30                              ~35 %

So a green `tests/` is consistent with a third of the prior mass being wrong on a
trained checkpoint, and nothing would say so. This module measures the quantity the
search actually consumes -- the **prior over the node's legal edges** -- instead.

**The prior is not recomputed here.** It comes out of `Search._expand` by swapping
the evaluator under `root_init`, so it is the same softmax, the same promotion
`log_softmax`, the same canonical edge order and the same §4.3 truncation the search
runs. Re-implementing that expression is precisely how this repository ended up with
four independent copies of the first-play-urgency constant, one of them wrong for
three days.

**Four logit sets, and the differences between them mean different things.**

* ``fp32``   -- the oracle: `BrokefishNet` in fp32 arithmetic, on weights that have
  *already been rounded to fp16*, so weight rounding is common to every variant and
  only the arithmetic differs. Same construction as `tests/test_model.build`.
* ``store``  -- the oracle's logits cast to fp16 and back. `cuda_impl` **refuses** an
  fp32 policy (§6.3: casting would change the numbers the reference computes), so
  fp16 storage is a property of the search contract. This is the floor no kernel fix
  can go below, and charging it to the kernel would be measuring the design.
* ``torch16`` -- the same module in fp16 arithmetic. The fair yardstick: a fused
  kernel is doing its job if it sits no further from ``fp32`` than torch's own fp16
  path does, which is `tests/test_model.py`'s `FP32_TOL` logic lifted into
  probability space.
* one entry per fused implementation.

⚠️ **Not a training signal.** Positions are drawn from a run's replay buffer because
that is the distribution the network is actually evaluated on -- a 150-ply rook
endgame is a different activation regime from anything `random_positions` reaches.
Nothing here forms an opinion about a chess move or flows back into training; it is
numerical validation of a kernel against a reference. The tabula rasa boundary is
about where *knowledge* comes from, and no knowledge crosses here.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from brokefish.env import torch_impl as env
from brokefish.nn import available, encoder_impl, why_unavailable
from brokefish.nn.model import BrokefishNet
from brokefish.search.torch_impl import Search, SearchConfig

# IEEE binary16. `MAX` is where an activation becomes `inf` -- fp16 has no
# saturating mode, so an overflow anywhere inside the stack surfaces as a
# non-finite output and the finiteness check below catches it. `SUBNORMAL_MIN` is
# where a *prior* silently becomes zero, and a zero prior has an exploration term of
# zero at every simulation budget (`search.md` §6.6), so it is unreachable forever.
FP16_MAX = 65504.0
FP16_SUBNORMAL_MIN = 5.9604644775390625e-08

# Positions that no random playout produces and that stress a different part of the
# stack: 218 legal moves (the edge cap and the widest softmax), a board with almost
# every slot dead (the padding mask), and a promotion race (the promo head).
ADVERSARIAL: Tuple[Tuple[str, str], ...] = (
    ("max-mobility", "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"),
    ("bare-kings-and-a-rook", "4k3/8/8/8/8/8/8/4K2R w K - 0 1"),
    ("promotion-race", "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 0 1"),
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("endgame-kq-vs-k", "8/8/8/4k3/8/8/4KQ2/8 w - - 0 1"),
)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_net(path: Optional[str], device: str = "cuda") -> Tuple[BrokefishNet, str]:
    """The fp16 inference copy, and what it was built from.

    ``.half()`` **after** loading, so a checkpoint carrying fp32 master weights is
    rounded exactly once and one carrying fp16 already is unchanged. Both the oracle
    and the fused implementation are then built from this same rounded module, which
    is what makes the comparison below about arithmetic rather than about weights.
    """
    net = BrokefishNet()
    if path is None:
        return net.to(device).half().eval(), "random init (no checkpoint)"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob
    if isinstance(blob, dict) and "net" in blob and isinstance(blob["net"], dict):
        state = blob["net"]          # a resumable checkpoint; a curve one is bare
    net.load_state_dict(state)
    return net.to(device).half().eval(), path


def positions_from_buffer(run: str, count: int, buffer_dir: str = "data/replay",
                          device: str = "cuda"):
    """The newest `count` records of a run's replay buffer, newest game first.

    ⚠️ Read through `BufferView.game`, never off `.data` directly: games wrap the
    ring, and a zeroed record decodes as a live white pawn on a1 rather than as an
    error. `game` returns exactly the record count the index says, which is what
    stops the scan at its own end.
    """
    import numpy as np

    from brokefish.train.buffer import BufferView

    view = BufferView.for_run(run, buffer_dir)
    boards, control, rep, got = [], [], [], 0
    for i in range(len(view) - 1, -1, -1):
        g = view.game(i)
        if g.shape[0] == 0:
            continue
        boards.append(g["board"])
        control.append(g["control"])
        rep.append(g["rep"])
        got += int(g.shape[0])
        if got >= count:
            break
    if not boards:
        raise RuntimeError(f"the buffer for {run!r} holds no games")
    b = torch.from_numpy(np.concatenate(boards)[:count].astype(np.int16))
    c = torch.from_numpy(np.concatenate(control)[:count].astype(np.int16))
    r = torch.from_numpy(np.concatenate(rep)[:count].astype(np.uint8))
    return b.to(device), c.to(device), r.to(device)


def positions_random(count: int, plies: int = 60, seed: int = 0, device: str = "cuda"):
    from tests.boards import random_positions          # noqa: PLC0415 - test-only dep

    boards, control, _ = random_positions(count, plies=plies, seed=seed, device=device)
    return boards, control, torch.zeros(boards.shape[0], dtype=torch.uint8, device=device)


def positions_adversarial(device: str = "cuda"):
    boards, control, names = [], [], []
    for name, fen in ADVERSARIAL:
        b, c = env.from_fen(fen)
        boards.append(b.reshape(1, 32))
        control.append(c.reshape(1))
        names.append(name)
    b = torch.cat(boards).to(device)
    c = torch.cat(control).to(device)
    # Every clamped repetition value of spec §7.2, so the embedding row that a real
    # game reaches is exercised rather than only row 0.
    out_b, out_c, out_r = [], [], []
    for rep in (0, 1, 2):
        out_b.append(b)
        out_c.append(c)
        out_r.append(torch.full((b.shape[0],), rep, dtype=torch.uint8, device=device))
    return torch.cat(out_b), torch.cat(out_c), torch.cat(out_r)


def drop_terminal(boards, control, rep):
    """A terminal position has no candidate edges, so its prior is not a distribution."""
    mask, in_check = env.movegen(boards, control)
    code, _ = env.terminal(mask, in_check, control, boards)
    keep = (code == 0).nonzero(as_tuple=True)[0]
    return boards[keep].contiguous(), control[keep].contiguous(), rep[keep].contiguous()


# --------------------------------------------------------------------------- #
# The two things being compared
# --------------------------------------------------------------------------- #

_INT_OF_WIDTH = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality, which `torch.equal` is not in the presence of NaN.

    ⚠️ `NaN != NaN`, so `torch.equal` on a tensor holding NaN is False even against
    itself -- and a forward that overflows reports "this implementation reuses its
    buffers" when it does nothing of the kind. Comparing the raw bits answers the
    question actually being asked: *did these bytes change*.
    """
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    kind = _INT_OF_WIDTH[a.element_size()]
    return bool((a.contiguous().view(kind) == b.contiguous().view(kind)).all())


@torch.no_grad()
def aliases_across_calls(fn, boards, control, rep) -> bool:
    """Does a second call overwrite the tensors the first one returned?

    ⚠️ **`cuda` does and `triton` does not**, and nothing said so until this suite
    was written. The CUDA encoder returns *views into buffers it owns*, which is a
    defensible choice on a path the search runs 800 times a move -- but it means any
    caller that accumulates outputs across calls silently gets the last batch
    repeated. It cost this file its first measurement: chunking 4111 positions into
    five calls and concatenating produced four copies of chunk four, which read as a
    950x kernel error, a 59 % top-1 disagreement and a value head off by 1.811 on a
    [-1, 1] output. Every one of those numbers was the harness, not the kernel.
    `.contiguous()` does not save you -- on an already-contiguous tensor it returns
    the same object.

    Production is unaffected today: `Search.simulate` hands the output straight to
    `expand` in the same step, and the puzzle and suite probes consume it inside the
    loop that produced it. This is a trap for the *next* caller, so it is measured
    rather than assumed.
    """
    n = boards.shape[0]
    if n < 2:
        return False
    half = max(n // 2, 1)
    p1, q1, v1 = fn(boards[:half], control[:half], rep[:half])
    keep = (p1.clone(), q1.clone(), v1.clone())
    fn(boards[half:half * 2], control[half:half * 2], rep[half:half * 2])
    return not all(_same_bits(k, t) for k, t in zip(keep, (p1, q1, v1)))


@torch.no_grad()
def run_logits(fn, boards, control, rep, chunk: int = 1024):
    """`(policy [N,32,64], promo [N,32,4], value [N])` from any forward, chunked.

    ⚠️ **`.clone()` on every chunk, and it is load-bearing** -- see
    :func:`aliases_across_calls` for what leaving it out measured.
    """
    pol, pro, val = [], [], []
    for lo in range(0, boards.shape[0], chunk):
        hi = min(lo + chunk, boards.shape[0])
        p, q, v = fn(boards[lo:hi], control[lo:hi], rep[lo:hi])
        pol.append(p.clone())
        pro.append(q.clone())
        val.append(v.clone())
    return torch.cat(pol), torch.cat(pro), torch.cat(val)


@torch.no_grad()
def run_priors(boards, control, policy, promo, value, e_cap: int, chunk: int = 512):
    """The prior over each position's legal edges, **through the search itself**.

    Returns ``(prior [N, E] fp32, move [N, E] int16, nedges [N])``.

    The evaluator is a closure over precomputed logits and ignores its arguments,
    which is deliberate: `_expand` reads only `(node, mask, do, policy, promo, value)`,
    so the repetition feature the search would have computed for itself never enters
    the comparison and the caller keeps control of it. `eps = 0` so §6.1's Dirichlet
    returns without touching the generator, and the sweep is off because it writes
    `edge_N`/`edge_Q` and never a prior -- between them, the three passes differ in
    the logits and in nothing else.
    """
    N = boards.shape[0]
    device = boards.device
    priors = torch.zeros((N, e_cap), dtype=torch.float32, device=device)
    moves = torch.zeros((N, e_cap), dtype=torch.int16, device=device)
    nedges = torch.zeros((N,), dtype=torch.int64, device=device)

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        cfg = SearchConfig(n=1, B=hi - lo, E=e_cap, eps=0.0,
                           root_terminal_sweep=False, terminal_collapse=False)
        p, q, v = policy[lo:hi], promo[lo:hi], value[lo:hi]
        search = Search(cfg, lambda _b, _c, _r: (p, q, v), device=device,
                        check_invariants=False)
        search.reset(boards[lo:hi].clone(), control[lo:hi].clone())
        search.root_init()
        priors[lo:hi] = search.edge_prior[:, 0].float()
        moves[lo:hi] = search.edge_move[:, 0]
        nedges[lo:hi] = search.node_nedges[:, 0].long()
    return priors, moves, nedges


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class Delta:
    """One variant scored against the fp32 oracle, in probability space."""

    name: str
    max_abs: float          # the headline: max |p - p_ref| over legal edges
    p95_abs: float
    max_rel: float          # on edges the oracle gives at least `rel_floor`
    max_kl: float           # KL(p_ref || p), per position
    mean_kl: float
    top1_disagree: float    # fraction of positions whose best prior moved
    top3_miss: float        # ... and fell out of the top three
    flip_risk: float        # fraction where max |dp| exceeds the top-two gap
    flushed: int            # legal edges the oracle gives > 0 and this gives 0
    edge_set_differs: int   # positions where §4.3 truncation kept a different set
    max_abs_logit: float
    nonfinite: int
    max_abs_value: float
    aliases_across_calls: bool = False

    def line(self) -> str:
        return (f"  {self.name:<10} |dp| max {self.max_abs:.3e}  p95 {self.p95_abs:.3e}"
                f"   rel {self.max_rel:7.2%}   KL max {self.max_kl:.2e}"
                f"   top1 {self.top1_disagree:6.3%}  flip {self.flip_risk:6.3%}"
                f"   flush {self.flushed:5d}")


def compare(name: str, ref, got, rel_floor: float = 1e-3) -> Delta:
    """`ref` and `got` are each `(prior, move, nedges, policy, promo, value)`."""
    p_ref, m_ref, ne_ref, pol_ref, _pro_ref, v_ref = ref
    p_got, m_got, ne_got, pol_got, pro_got, v_got = got

    E = p_ref.shape[1]
    idx = torch.arange(E, device=p_ref.device)
    valid = idx[None, :] < ne_ref[:, None]

    # §4.3 keeps the E highest priors, so a large enough logit difference can change
    # *which* moves survive. That is a different and worse failure than a shifted
    # probability, so it is counted separately rather than being averaged in.
    same_set = (ne_ref == ne_got) & ((m_ref == m_got) | ~valid).all(-1)
    edge_set_differs = int((~same_set).sum())

    cmp_rows = same_set
    v = valid & cmp_rows[:, None]
    d = (p_got - p_ref).abs()
    d = torch.where(v, d, torch.zeros_like(d))
    # ⚠️ `inf`, not 0, when nothing is comparable. A variant broken badly enough to
    # change every position's edge set leaves no rows to difference, and reporting a
    # max of 0.0 for it reads as a *perfect* result sitting next to the non-finite
    # count that is the real story. Measured: `W+A+acc16@448` overflowed the fp16
    # accumulator on 171,983 outputs and this line printed `0.000e+00`.
    max_abs = float(d.max()) if bool(v.any()) else float("inf")
    flat = d[v]
    p95 = float(torch.quantile(flat, 0.95)) if flat.numel() else float("inf")

    big = v & (p_ref >= rel_floor)
    max_rel = float((d[big] / p_ref[big]).max()) if bool(big.any()) else float("inf")

    # KL(p_ref || p_got). A flushed entry would make this infinite, which is true but
    # useless as a summary, so the clamp is at the smallest positive fp16 -- the
    # actual resolution of the array these numbers were read out of.
    q = p_got.clamp(min=FP16_SUBNORMAL_MIN)
    terms = torch.where(v & (p_ref > 0), p_ref * (p_ref.clamp(min=1e-30) / q).log(),
                        torch.zeros_like(p_ref))
    kl = terms.sum(-1)[cmp_rows]
    max_kl = float(kl.max()) if kl.numel() else 0.0
    mean_kl = float(kl.mean()) if kl.numel() else 0.0

    neg = torch.finfo(torch.float32).min
    a = torch.where(v, p_ref, torch.full_like(p_ref, neg))
    b = torch.where(v, p_got, torch.full_like(p_got, neg))
    rows = cmp_rows.nonzero(as_tuple=True)[0]
    top1_ref = a.argmax(-1)
    top1_got = b.argmax(-1)
    top1_bad = (top1_ref != top1_got)[rows]
    k = min(3, E)
    top3_got = b.topk(k, dim=-1).indices
    top3_miss = (~(top3_got == top1_ref[:, None]).any(-1))[rows]

    # The decision-relevant one. A prior error only matters if it can change what the
    # search does, and at a fresh node the first descent is decided by the prior gap
    # alone -- the same margin-versus-perturbation argument `tests/test_search_cuda.py`
    # makes for the selection score.
    two = a.topk(min(2, E), dim=-1).values
    gap = (two[:, 0] - two[:, 1]) if two.shape[1] == 2 else torch.full_like(two[:, 0], 1.0)
    flip = (d.max(-1).values > gap)[rows]

    flushed = int((v & (p_ref > 0) & (p_got == 0)).sum())
    nonfinite = int((~torch.isfinite(pol_got.float())).sum()
                    + (~torch.isfinite(pro_got.float())).sum()
                    + (~torch.isfinite(v_got.float())).sum())

    n = max(int(cmp_rows.sum()), 1)
    return Delta(
        name=name, max_abs=max_abs, p95_abs=p95, max_rel=max_rel,
        max_kl=max_kl, mean_kl=mean_kl,
        top1_disagree=float(top1_bad.sum()) / n, top3_miss=float(top3_miss.sum()) / n,
        flip_risk=float(flip.sum()) / n, flushed=flushed,
        edge_set_differs=edge_set_differs,
        max_abs_logit=float(pol_got.float().abs().max()),
        nonfinite=nonfinite,
        max_abs_value=float((v_got.float() - v_ref.float()).abs().max()))


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #

def validate(checkpoint: Optional[str], boards, control, rep, impls: Sequence[str],
             e_cap: int = 96, rel_floor: float = 1e-3, device: str = "cuda",
             verbose: bool = True,
             forwards: Optional[Dict[str, "object"]] = None,
             net: Optional[BrokefishNet] = None) -> Dict[str, Delta]:
    """Every variant against the fp32 oracle. Returns the deltas by name.

    ``forwards`` replaces the encoder lookup for the named implementations with a
    plain ``(boards, control, rep) -> (policy, promo, value)`` callable. It exists so
    `tests/test_validate.py` can feed this a *deliberately wrong* encoder and prove
    the suite bites -- a validator nobody has watched fail is an assertion, not a
    measurement.

    ⚠️ ``net`` must be passed whenever ``forwards`` is, and the two must be the same
    module. ``checkpoint=None`` builds a **fresh random** `BrokefishNet`, so a caller
    that wraps its own encoder and lets this build the oracle is comparing two
    unrelated networks. That is not hypothetical: it made the first version of the
    injection test report a 0.85 prior delta for a 0.002 logit bump, non-monotone in
    the bump, and pass its assertions for entirely the wrong reason.
    """
    if forwards and net is None:
        raise ValueError(
            "pass `net` alongside `forwards`: otherwise the oracle is built from a "
            "different random module than the injected forward wraps, and every "
            "delta below is the gap between two unrelated networks")
    if net is not None:
        net16, source = net, "caller-supplied module"
    else:
        net16, source = load_net(checkpoint, device)
    net32 = copy.deepcopy(net16).float()

    boards, control, rep = drop_terminal(boards, control, rep)
    n = int(boards.shape[0])
    if n == 0:
        raise RuntimeError("every position was terminal; nothing to validate")

    def bundle(policy, promo, value):
        pri, mov, ne = run_priors(boards, control, policy, promo, value, e_cap)
        return (pri, mov, ne, policy, promo, value)

    ref_logits = run_logits(net32, boards, control, rep)
    ref = bundle(*ref_logits)

    variants: Dict[str, tuple] = {
        # The search contract stores fp16 logits and `cuda_impl` refuses anything
        # else, so this is the floor: what the *format* costs with exact arithmetic.
        "store": bundle(ref_logits[0].half().float(), ref_logits[1].half().float(),
                        ref_logits[2]),
        "torch16": bundle(*run_logits(net16, boards, control, rep)),
    }
    aliasing: Dict[str, bool] = {}
    for impl in impls:
        # A fresh module per implementation. They do not mutate it today, but two
        # kernels packing weights out of one instance is a coupling nobody declared.
        if forwards and impl in forwards:
            fn = forwards[impl]
        else:
            fused = encoder_impl(impl)(copy.deepcopy(net16))
            fn = fused.forward_full
        aliasing[impl] = aliases_across_calls(fn, boards, control, rep)
        variants[impl] = bundle(*run_logits(fn, boards, control, rep))

    if verbose:
        print(f"\ncheckpoint  {source}")
        print(f"positions   {n} non-terminal, E = {e_cap}, "
              f"max |policy logit| {float(ref_logits[0].abs().max()):.1f} "
              f"({100 * float(ref_logits[0].abs().max()) / FP16_MAX:.3f} % of fp16 max)")
        print(f"            mean edges {float(ref[2].float().mean()):.1f}, "
              f"max {int(ref[2].max())}")
        print()

    out = {}
    for name, got in variants.items():
        out[name] = compare(name, ref, got, rel_floor=rel_floor)
        out[name].aliases_across_calls = bool(aliasing.get(name, False))
        if verbose:
            print(out[name].line())
    if verbose:
        for impl, al in aliasing.items():
            if al:
                print(f"\n  note: {impl}.forward_full returns views into buffers it "
                      f"reuses; a caller that keeps them across calls gets the last "
                      f"batch repeated (see aliases_across_calls)")
    return out


def report(deltas: Dict[str, Delta], impls: Sequence[str], max_abs_dp: Optional[float],
           fp32_tol: Optional[float] = 2.0) -> List[str]:
    """Hard failures first, then the relative bar. Returns the failures.

    ⚠️ ``fp32_tol=None`` skips the relative bar, and there is exactly one situation
    that calls for it: a path whose precision is **deliberately** lower than fp16.
    The bar asks "is this kernel no further from fp32 than torch's own fp16 is", which
    is the right question for an fp16 kernel and a meaningless one for e4m3 — the fp8
    FFN measures 10x torch fp16's p95 by construction, and that number is the feature
    rather than a defect. Such a path is judged by ``max_abs_dp`` against a budget
    that was agreed in advance, and by the emulation in `nn/quant.py` that predicted it.
    """
    bad: List[str] = []
    torch16 = deltas.get("torch16")
    for impl in impls:
        d = deltas[impl]
        if d.nonfinite:
            bad.append(f"{impl}: {d.nonfinite} non-finite outputs -- an overflow "
                       f"inside the stack, since fp16 has no saturating mode")
        if d.edge_set_differs:
            bad.append(f"{impl}: §4.3 truncation kept a different edge set on "
                       f"{d.edge_set_differs} positions")
        if max_abs_dp is not None and d.max_abs > max_abs_dp:
            bad.append(f"{impl}: max |dp| {d.max_abs:.3e} over {max_abs_dp:.3e}")
        # The yardstick that does not need a threshold invented for it: torch's own
        # fp16 path is doing the same job with the same weights, so a fused kernel
        # materially further from fp32 than that is doing something else.
        #
        # ⚠️ On **p95, not on the max**, and the reason is statistical rather than
        # convenient. Both sides of that ratio are the single largest of ~10^5
        # samples, so the ratio is an estimate built from one observation each and
        # moves by 2x on noise: measured on a trained checkpoint, all three fp16
        # paths had an identical p95 of 1.22e-4 while their maxima spread over 2.2x.
        # The max is still gated -- by `max_abs_dp`, in absolute terms, where a
        # single bad edge is exactly what you want to catch -- and still printed.
        if fp32_tol is not None and torch16 is not None and torch16.p95_abs > 0 and \
                d.p95_abs > fp32_tol * torch16.p95_abs:
            bad.append(f"{impl}: p95 |dp| {d.p95_abs:.3e} is "
                       f"{d.p95_abs / torch16.p95_abs:.1f}x torch fp16's "
                       f"{torch16.p95_abs:.3e}, over the {fp32_tol}x bar")
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=None,
                   help="a curve checkpoint or a resumable one; omit for random init")
    p.add_argument("--buffer", default=None, help="a run whose replay buffer supplies positions")
    p.add_argument("--buffer-dir", default="data/replay")
    p.add_argument("--positions", type=int, default=2048)
    p.add_argument("--random-plies", type=int, default=0,
                   help="add this many plies of random playout positions instead of "
                        "(or as well as) the buffer")
    p.add_argument("--no-adversarial", action="store_true")
    p.add_argument("--impl", default=None, help="comma separated; default is every available")
    p.add_argument("--e-cap", type=int, default=SearchConfig().E)
    p.add_argument("--rel-floor", type=float, default=1e-3,
                   help="the relative delta is reported on priors at least this large; "
                        "1 %% of 1e-6 is not a number anyone can act on")
    p.add_argument("--max-abs-dp", type=float, default=0.01,
                   help="hard bar on max |dp| over the legal edges. 0.01 is the 1 %% "
                        "target; the measured value on a trained checkpoint is "
                        "2.2e-3, so this clears by ~5x and is a ceiling rather than "
                        "a fitted threshold")
    p.add_argument("--fp32-tol", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quant", default=None,
                   help="comma separated subset of fp8,int8 -- the CUDA kernel's "
                        "quantised FFN paths. ⚠️ Without this the tool cannot reach "
                        "them at all: `--impl` names a *registry* entry and both "
                        "quantised paths are constructor flags on the CUDA one, which "
                        "is why the fp8 kernel that shipped in August had no coverage "
                        "here for its first ten days")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    device = "cuda"
    impls = [s for s in (a.impl.split(",") if a.impl else available()) if s]
    if not impls:
        print(f"no encoder implementation available: {why_unavailable()}")
        return 1

    parts = []
    if a.buffer:
        parts.append(positions_from_buffer(a.buffer, a.positions, a.buffer_dir))
    if a.random_plies or not a.buffer:
        parts.append(positions_random(a.positions, plies=a.random_plies or 60,
                                      seed=a.seed))
    if not a.no_adversarial:
        parts.append(positions_adversarial())
    boards = torch.cat([x[0] for x in parts])
    control = torch.cat([x[1] for x in parts])
    rep = torch.cat([x[2] for x in parts])

    forwards, net = None, None
    quant = [q for q in (a.quant.split(",") if a.quant else []) if q]
    if quant:
        import copy as _copy

        from brokefish.nn.cuda_impl import FusedEncoder
        # `int8a` is int8 on all four matmuls, not just the FFN's two.
        kwargs = {"fp8": {"fp8": True}, "int8": {"int8": True}}
        bad_q = [q for q in quant if q not in kwargs]
        if bad_q:
            print(f"unknown quant mode {bad_q}, expected some of {sorted(kwargs)}")
            return 1
        net, _src = load_net(a.checkpoint, device)
        # ⚠️ Built from the same module the oracle is, per `validate`'s own contract:
        # letting it construct a second random net would compare two unrelated
        # networks and report a delta that is the gap between them.
        forwards = {q: FusedEncoder(_copy.deepcopy(net), **kwargs[q]).forward_full
                    for q in quant}
        impls = impls + quant

    deltas = validate(a.checkpoint, boards, control, rep, impls,
                      e_cap=a.e_cap, rel_floor=a.rel_floor,
                      forwards=forwards, net=net)
    # ⚠️ `fp32_tol=None` for the quantised paths: the bar asks "is this no further from
    # fp32 than torch's own fp16 is", which is the right question for an fp16 kernel and
    # a meaningless one for a path whose lower precision is the feature. They are judged
    # by `--max-abs-dp` against a budget agreed in advance, as `report` documents.
    bad = report(deltas, [i for i in impls if i not in quant], a.max_abs_dp, a.fp32_tol)
    bad += report(deltas, quant, a.max_abs_dp, None)

    print()
    t16 = deltas.get("torch16")
    for d in deltas.values():
        ratio = (d.max_abs / t16.max_abs) if t16 and t16.max_abs else float("nan")
        print(f"  {d.name:<10} max |dv| {d.max_abs_value:.3e}   "
              f"max |logit| {d.max_abs_logit:8.2f}   non-finite {d.nonfinite}   "
              f"max|dp| vs torch16 {ratio:5.2f}x")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({k: vars(v) for k, v in deltas.items()}, fh, indent=2)
        print(f"\n  wrote {a.json}")

    print()
    for line in bad:
        print(f"  FAIL  {line}")
    print("  all checks passed" if not bad else f"  FAILED ({len(bad)})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
