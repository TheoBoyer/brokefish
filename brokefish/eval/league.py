"""The league: every checkpoint of a run, rated on one scale. `evaluation.md` §5.4.

    uv run --no-project --python .venv/bin/python -m brokefish.eval.league \
        --run t4h-n64 --games 36 --sims 64

What it does, in order: build the pool (the frozen anchor plus the run's
checkpoints), generate one fixed set of openings, play the fixed SAI calendar of
pairings, fit one global Bradley-Terry model over the whole graph, join the euro
counter from the training log, and write one record per checkpoint.

Three things are deliberate.

**The zero is uniformly random legal play** (rebased 2026-08-08, `evaluation.md`
§5.1). It used to be a frozen random-init network at 64 simulations, held in
`checkpoints/anchor.pt` — a file, so that the zero did not depend on a torch version
or on nobody editing `BrokefishNet`'s constructor. That fixed the wrong half of the
problem. The zero was never a *network*, it was a network **plus a search**: the day
§6.1a's root terminal sweep landed, the anchor got stronger while its file was
untouched, and the origin of the curve moved in silence. The file was also gitignored,
so every published number hung off one untracked blob.

Random play has none of that. It is defined by the rules of chess and a uniform draw,
so it is the same player on every commit, every architecture and every future search
budget — and it cannot be deleted. `Search.random_move` implements it *outside* the
search on purpose: a one-simulation search would inherit the root terminal sweep and
find every mate in one.

The old anchor is still in the pool as `init:n64`, which is what lets this scale be
related to the previous ones by a measured offset rather than by assertion. ⚠️ That
relates *future* leagues; the three leagues already published stay on their own
scales, because different fits have different units and only replaying their
checkpoints here would change that.

**A player is a `(network, budget)` pair.** The simulation count used to be a
league-wide constant that quietly belonged to the scale, which is the reason every
report so far carries "comparable to nothing else". As a per-player field it becomes
a measured axis instead: one checkpoint at 16, 64 and 256 sims is three players in one
fit, and the gap between them is what a doubling of search is worth. `--ladder` uses
it to bridge random play up to the training curve; `--grid` uses it to measure the
search-versus-training exchange rate.

**The calendar is fixed.** Each checkpoint plays generation offsets ±1, ±2, ±3,
±6, ±8, ±12 (SAI's schedule, §5.2), plus the anchor plays a spread across the whole
run. The long-range edges are the load-bearing part: without them the graph is a
path and the global fit degenerates back into the chain §5.2 exists to avoid.
Variance-proportional sampling — spend games where `p(1-p)` is largest — is the
upgrade, and `EloFit.predict` is the function it needs; it is not in v1 because it
needs an online fit and the fixed calendar needs nothing.

**Nothing here may select a checkpoint** (`evaluation.md` §2). This module writes
to the run's own folder; nothing under `brokefish/train/` reads what it writes, and
`tests/test_league.py` asserts that rather than trusting it. Keeping the checkpoint
with the best league rating is distillation through a one-bit channel and it is the
prohibition most likely to be violated by accident, because it looks like good
practice.

⚠️ **`--keep-checkpoints` writes a bare `state_dict`** with no step and no euro
count in it, so the euro axis is joined from `runs/<run>/<run>.jsonl` by step. If that
log is missing, the curve still has an Elo axis and its `euros_spent` is `null` —
which is honest, and visible, rather than a zero that looks like a measurement.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .elo import Edge, EloFit, elo_half_width, fit_elo
from .match import MatchResult, play_match, random_openings

# SAI's fixed schedule, `evaluation.md` §5.2. Their graph ran ~13 000 edges over
# ~1000 nodes, i.e. ~13 pairings per checkpoint, which is where this shape and the
# sizing arithmetic in §5.4 both come from.
SAI_OFFSETS: Tuple[int, ...] = (1, 2, 3, 6, 8, 12)

# §5.1, rebased 2026-08-08. **The zero is uniformly random legal play**, not the
# frozen random-init network — see the module docstring. `ANCHOR_NAME` is what
# `fit_elo` pins, and it is kept as the name of that concept rather than of that
# file, so nothing outside this module has to know the zero moved.
RANDOM_NAME = "random"
ANCHOR_NAME = RANDOM_NAME
INIT_NAME = "init"
from brokefish import paths

#: ⚠️ Shared, not per-run: every Elo scale is anchored to this one file, and a copy
#: in each run's folder would let two leagues anchor to two different networks.
DEFAULT_ANCHOR_PATH = paths.ANCHOR_PATH

# §5.1a's bottom rungs: the untrained network at four budgets. These exist to make
# the zero *estimable* — random play loses 36-0 to anything past the first few
# hundred steps, so without intermediate strengths the scale would hang off a
# saturated edge. `init:n64` is exactly the player that was the anchor before
# 2026-08-08, which is what lets this scale be related to the previous one.
DEFAULT_LADDER: Tuple[int, ...] = (1, 4, 16, 64)


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PoolEntry:
    """One rated player: a name, a place on the x-axis, weights, and a budget.

    ⚠️ **A player is a `(network, budget)` pair, not a network** (§5.1a, 2026-08-08).
    The search budget used to be a league-wide constant that silently belonged to the
    scale — "ratings at 64 sims" — which is why no two leagues were comparable and why
    every report carried that caveat. Making it a per-player field turns it into a
    measured axis: the same checkpoint at 16, 64 and 256 simulations is three players
    in one fit, and the distance between them is the Elo value of a doubling of search.
    """

    name: str
    step: int
    path: Optional[str]       # None for `random`, which has no network at all
    run: str = ""             # which training run it came from; "" for the ladder
    sims: int = 64            # 0 = uniformly random legal play, no tree, no network


ANCHOR_SEED = 20260731


def ensure_anchor(path: str = DEFAULT_ANCHOR_PATH, seed: int = ANCHOR_SEED) -> str:
    """The frozen random-init network, created once and never again.

    ⚠️ Written only if the file is absent. Overwriting it would silently move the
    zero of the cost-versus-Elo curve and make every previously published point
    incomparable, so this refuses to touch an existing file.

    ⚠️ **`checkpoints/` is gitignored, so this file is not version-controlled**, and
    a curve whose origin can be deleted by `rm -rf checkpoints` is a curve with no
    origin. Two things stand in for that. The seed is fixed, so the file is
    *usually* reproducible; and `anchor_digest` is written into every league report
    and into `docs/ledger/state.md`, so if it is ever regenerated on a different
    torch version and comes out different, that is visible in one comparison
    instead of being an unexplained shift in every rating.
    """
    if os.path.exists(path):
        return path
    from brokefish.nn.model import BrokefishNet

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        net = BrokefishNet()
    finally:
        torch.random.set_rng_state(gen_state)
    torch.save(net.state_dict(), path)
    return path


def anchor_digest(path: str = DEFAULT_ANCHOR_PATH) -> Optional[str]:
    """SHA-256 of the anchor file, or None if it is not there.

    The file's bytes rather than a weight fingerprint, because the question this
    answers is "is this the same anchor" and nothing subtler.
    """
    import hashlib

    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_STEP_RE = re.compile(r"-(\d{6})\.pt$")


def discover_checkpoints(run: str, checkpoint_dir: Optional[str] = None
                         ) -> List[Tuple[int, str]]:
    """`(step, path)` for every ``{run}-{step:06d}.pt`` snapshot, in step order.

    ⚠️ `checkpoint_dir=None` resolves **per run** to `runs/<run>/checkpoints`, which is
    the whole point of the layout: a league over several runs reads each one's own
    folder rather than a shared directory that happened to hold them all.
    """
    checkpoint_dir = checkpoint_dir or paths.checkpoint_dir(run)
    out = []
    for path in glob.glob(os.path.join(checkpoint_dir, f"{run}-*.pt")):
        m = _STEP_RE.search(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def subsample(items: Sequence, limit: Optional[int]) -> List:
    """At most `limit` items, evenly spaced, keeping the first and the last.

    ⚠️ **Spacing by index, never by score.** Which checkpoints get rated is a cost
    decision; *which one is best* is what §2 forbids this package from acting on.
    An even sweep over the run answers "what does the curve look like" and cannot
    smuggle a selection in.
    """
    n = len(items)
    if limit is None or n <= limit or limit <= 0:
        return list(items)
    if limit == 1:
        return [items[-1]]
    idx = [round(i * (n - 1) / (limit - 1)) for i in range(limit)]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def build_pool(runs, checkpoint_dir: Optional[str] = None,
               anchor_path: str = DEFAULT_ANCHOR_PATH,
               limit: Optional[int] = 32, sims: int = 64,
               ladder: Sequence[int] = DEFAULT_LADDER,
               grid: Sequence[int] = (), grid_points: int = 0) -> List[PoolEntry]:
    """Random play, then the untrained ladder, then the checkpoints by step.

    The pool is ordered by *approximate strength* — `(step, sims)` — because that is
    what the SAI calendar consumes: it pairs index `i` with `i + {1,2,3,6,8,12}`, so
    the ordering decides which players actually meet. Two useful consequences fall
    out of sorting on `sims` within a step: the four `init:n*` rungs land at indices
    1-4 and so are bridged to `random` by the small offsets, and a checkpoint's own
    budget variants are adjacent and therefore always play each other.

    `ladder` is the untrained network's budgets (§5.1a). `grid` is the set of budgets
    to also rate `grid_points` checkpoints at, evenly spaced over each run — that is
    the search-versus-training exchange rate, and it is off by default because it
    multiplies the pool.

    Index 0 is the anchor on purpose: the SAI calendar then connects it to
    players 1, 2, 3, 6, 8 and 12 for free, so the zero of the scale is a real
    node in the graph rather than an assumption bolted on afterwards.

    ⚠️ **Several runs are sorted together, not concatenated, and that is the whole
    point.** Two runs rated in two leagues share only the anchor, which pins the zero
    of each scale but not its slope, so the difference between them is confounded with
    whatever the two Bradley-Terry fits did to the units — measured on 2026-08-05, a
    +24 Elo mean gap between `t7h-fp8` and its control that could not be told from a
    fit artefact. Concatenating the pools would barely help: `sai_pairings` connects
    index i to i+{1,2,3,6,8,12}, so appending run B after run A gives cross-run edges
    only at the seam. Sorting by step interleaves them, and then **every** offset is a
    cross-run edge for half its length. That is what makes one scale one scale.

    `limit` is the number of **checkpoints per run** (it excluded the anchor before
    2026-08-08; the ladder is a fixed size now, so counting it in was only confusing),
    and the calendar grows linearly with the pool.
    """
    if isinstance(runs, str):
        runs = [runs]
    runs = list(dict.fromkeys(runs))          # de-duplicate, keep order
    init_path = ensure_anchor(anchor_path)

    entries: List[PoolEntry] = []
    for run in runs:
        found = discover_checkpoints(run, checkpoint_dir)
        if not found:
            raise FileNotFoundError(
                f"no {run}-NNNNNN.pt snapshots in {checkpoint_dir}/. A run only writes "
                f"them with --keep-checkpoints; the rolling {run}.pt is one endpoint "
                f"and a curve needs a history.")
        chosen = subsample(found, limit)
        for step, path in chosen:
            entries.append(PoolEntry(name=f"{run}@{step}:n{sims}", step=step,
                                     path=path, run=run, sims=sims))
        # The budget grid, on a spread of the checkpoints that were already chosen —
        # a subsample of a subsample, so the grid points sit *on* the training curve
        # and the vertical distance at a shared step is exactly the search value.
        for step, path in subsample(chosen, grid_points) if grid_points else []:
            for k in grid:
                if k == sims:
                    continue
                entries.append(PoolEntry(name=f"{run}@{step}:n{k}", step=step,
                                         path=path, run=run, sims=k))
    entries.sort(key=lambda e: (e.step, e.sims, e.run))

    head = [PoolEntry(name=RANDOM_NAME, step=0, path=None, run="", sims=0)]
    head += [PoolEntry(name=f"{INIT_NAME}:n{k}", step=0, path=init_path, run="", sims=k)
             for k in sorted(ladder)]
    return head + entries


def budget_ladder_pairs(pool: Sequence[PoolEntry]) -> List[Tuple[int, int]]:
    """Every pair of players that share a network and differ only in budget.

    The SAI calendar orders by step and would pair a checkpoint's budget variants only
    by accident of adjacency. This makes the edge that actually measures §5.1a's axis
    explicit: `ckpt:n16` vs `ckpt:n64` vs `ckpt:n256` is a direct, same-network
    read of what a doubling of search is worth, with the network held exactly fixed.

    It also produces the bottom of the ladder for free, since the `init:n*` rungs are
    one network at four budgets.
    """
    by_net: Dict[tuple, List[int]] = {}
    for i, e in enumerate(pool):
        if e.path is None:
            continue                       # `random` has no network to vary
        by_net.setdefault((e.run, e.step, e.path), []).append(i)
    out = set()
    for idx in by_net.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out.add((min(idx[a], idx[b]), max(idx[a], idx[b])))
    return sorted(out)


# --------------------------------------------------------------------------- #
# The calendar
# --------------------------------------------------------------------------- #

def sai_pairings(n: int, offsets: Sequence[int] = SAI_OFFSETS,
                 anchor_every: int = 4,
                 anchor_span: Optional[int] = None) -> List[Tuple[int, int]]:
    """Which pairs of pool indices play. `evaluation.md` §5.2.

    Each index plays the ones `offsets` ahead of it, which gives every checkpoint
    up to `2 * len(offsets)` edges — six forward, six backward.

    `anchor_every` adds an edge from index 0 to every `anchor_every`-th checkpoint.
    The calendar alone only connects the anchor to the first twelve, and a scale
    whose zero is measured only against the start of the run is one whose late
    points are reached through a long chain of intermediate fits — the exact
    failure mode §5.2 is about. Set it to 0 to get the bare SAI schedule.

    ⚠️ **`anchor_span` bounds those edges, and it exists because they were measuring
    nothing.** Against the pre-2026-08-08 anchor, `anchor vs t12h-pcr@1405` and every
    pairing above it came back `0-0-36`: a saturated edge costs a full 36 games and
    contributes no information beyond "further apart than this league can resolve".
    With random play as the zero the saturation starts even earlier. So the anchor's
    spread is capped at the first `anchor_span` players — where it can still lose
    games — and one edge to the far end is kept deliberately, as the check that it
    really is saturated rather than as a measurement. `None` is the old behaviour.

    The right fix is variance-proportional sampling (spend games where `p(1-p)` is
    largest), which `EloFit.predict` already has the machinery for; this is the cheap
    version of it and says so.
    """
    pairs = {(i, i + d) for i in range(n) for d in offsets if i + d < n}
    if anchor_every > 0:
        top = n if anchor_span is None else min(n, anchor_span + 1)
        pairs |= {(0, j) for j in range(1, top, anchor_every)}
        if n > 1:
            pairs.add((0, n - 1))          # the far end always meets the anchor
    return sorted(pairs)


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

def load_engine(path: str, impl: Optional[str] = "cuda",
                device: str = "cuda", quant: Optional[str] = None) -> Callable:
    """An evaluator for one checkpoint: ``(boards, control, rep) -> logits``.

    ⚠️ The master network is built and loaded on the **CPU** and only the packed
    fp16 weights go to the card. A league holds thirty of these at once, and thirty
    fp32 masters is a gigabyte of an eight-gigabyte card that is also driving the
    display. `FusedEncoder` does its own `.half()` on what it packs, so an fp32 CPU
    module is a perfectly good source.

    ``quant`` is ``"int8"`` or ``"fp8"`` -- the same two names the training CLI
    uses, and ``"int8"`` means whatever `cuda_impl.SCHEME` currently is.

    ⚠️ **It defaults to ``None`` -- fp16 -- and that default is load-bearing, not
    laziness.** Every rating this project has ever produced, including the frozen
    anchor that pins the scale at Elo 0, was measured with every player in fp16.
    Precision is a property of the *player*, exactly as the search budget is
    (`evaluation.md` §5.1a), so flipping this default silently rebases every scale
    and makes new numbers incomparable to the whole ledger. Turning it on is a
    decision to re-rate, and the cost of doing so is the fp16-vs-quantised head to
    head, which is a measurement and not an assumption.
    """
    from brokefish.nn.model import net_for_state

    from .layer0 import load_net_state

    # ⚠️ **Sized from the file, not from the constructor's defaults.** A checkpoint is
    # a bare `state_dict` and carries no architecture, so the value head's width is
    # read back off `value.weight`. That is what lets a scalar-head checkpoint from
    # August and a win/draw/loss one sit in the *same* Bradley-Terry fit -- and the
    # shared `checkpoints/anchor.pt`, which every Elo scale anchors to, is a
    # scalar-head net that must keep loading forever.
    state = load_net_state(path, device="cpu")
    net = net_for_state(state)
    net.load_state_dict(state)
    net.eval()
    if impl is None:
        net = net.to(device)
        return lambda b, c, r: net(b, c, r)
    from brokefish.nn import encoder_impl

    kw = {"fp8": {"fp8": True}, "int8": {"int8": True}}.get(quant or "", {})
    if quant and not kw:
        raise ValueError(f"unknown quant {quant!r}, expected fp8 or int8")
    if kw and impl != "cuda":
        raise ValueError(f"quant={quant!r} is a property of the CUDA kernel, "
                         f"but impl={impl!r}")
    return encoder_impl(impl)(net, **kw).forward_full


# --------------------------------------------------------------------------- #
# The euro axis
# --------------------------------------------------------------------------- #

def _first(rec: dict, *keys: str):
    """The first key present in `rec`, or None. For log-schema changes."""
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def training_series(jsonl_path: str) -> List[Tuple[int, dict]]:
    """`(step, {euros, training_seconds})` from a training log, in step order.

    `evaluation.md` §5.3: the cost counter and the Elo are written by the same
    writer, which is this module — so the join happens here, once, rather than by
    hand in a notebook six weeks later.

    ⚠️ **Seconds as well as euros.** Every run so far was launched at the default
    `--euros-per-hour 0`, which makes the euro axis identically zero and the curve
    a vertical line. `training_seconds` is the same quantity before the rate is
    applied — `train.md` §10's self-play plus gradient and nothing else — so it is
    a usable x-axis even for a run that never priced itself.
    """
    out: List[Tuple[int, dict]] = []
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            # ⚠️ **Two spellings, on purpose.** `train/loop.py` dropped its `phase/`
            # wrapper on 2026-07-31 because wandb groups on the first path component
            # only, so `gradient/kl` is now what a new run writes where `t4h-n64` and
            # everything before it wrote `phase/gradient/kl`. Reading only the new one
            # would make the cost axis silently null for every log already on disk —
            # and a null x-axis looks like "the run had no euros", not like a bug.
            euros = _first(rec, "euros/training", "phase/euros/training")
            seconds = _first(rec, "euros/training_seconds",
                             "phase/euros/training_seconds")
            if step is None or (euros is None and seconds is None):
                continue
            out.append((int(step), {"euros": None if euros is None else float(euros),
                                    "training_seconds": None if seconds is None
                                    else float(seconds)}))
    return sorted(out, key=lambda kv: kv[0])


def series_at(series: Sequence[Tuple[int, dict]], step: int) -> dict:
    """The cost counters as of `step`: the last record at or before it.

    Step 0 is the frozen anchor, which is a network nobody trained, so its cost is
    exactly zero rather than unknown — and a zero there is what makes it the origin
    of the curve rather than a point with a missing x.
    """
    if step <= 0:
        return {"euros": 0.0, "training_seconds": 0.0}
    best: dict = {}
    for s, rec in series:
        if s <= step:
            best = rec
        else:
            break
    return best


# --------------------------------------------------------------------------- #
# Running one
# --------------------------------------------------------------------------- #

class _Log:
    """stdout plus a `tail -f`-able file. `CLAUDE.md` rule 7."""

    def __init__(self, path: Optional[str]) -> None:
        self.fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.fh = open(path, "w", buffering=1)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        if self.fh is not None:
            self.fh.write(line + "\n")

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


def endpoint_pairs(pool: Sequence[PoolEntry]) -> List[Tuple[int, int]]:
    """Each run's last checkpoint against every other run's, at each shared budget.

    ⚠️ **The SAI calendar orders by step, and two runs of different lengths do not
    finish next to each other.** Rating `t24h-fp8` (8 416 steps) beside `t12h-pcr`
    (4 213) put their final checkpoints 14 indices apart, and the largest offset is
    12 — so the one pairing the league was commissioned to resolve, "which run ends
    stronger", had no edge at all and would have been answered through a chain of
    intermediates. That is precisely what §5.2 says not to do.

    Adding an edge can only help: a global Bradley-Terry fit is not biased by extra
    games, it is sharpened where they are played.
    """
    best: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for i, e in enumerate(pool):
        if not e.run:
            continue
        key = (e.run, e.sims)
        if key not in best or e.step > best[key][0]:
            best[key] = (e.step, i)
    by_budget: Dict[int, List[int]] = {}
    for (_run, sims), (_step, i) in best.items():
        by_budget.setdefault(sims, []).append(i)
    out = set()
    for idx in by_budget.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out.add((min(idx[a], idx[b]), max(idx[a], idx[b])))
    return sorted(out)


@torch.no_grad()
def run_league(pool: Sequence[PoolEntry], games: int = 36, n_sims: int = 64,
               opening_plies: int = 8, max_plies: int = 512,
               offsets: Sequence[int] = SAI_OFFSETS, anchor_every: int = 4,
               anchor_span: Optional[int] = None,
               impl: Optional[str] = "cuda", search_impl: str = "cuda",
               quant: Optional[str] = None,
               device: str = "cuda", seed: int = 0, prior: float = 1.0,
               cost=None,
               log: Optional[Callable[[str], None]] = None,
               engine_loader: Callable[[str], Callable] = None,
               search_kw: Optional[dict] = None) -> dict:
    """Play the calendar, fit the ratings, return the whole record.

    `games` is the number of games **per pairing** and must be even: the unit of
    measurement is the colour-swapped game pair, so an odd count would leave one
    opening played from one side only, which is the imbalance the pairing exists to
    remove.
    """
    if games % 2:
        raise ValueError(f"games per pairing must be even — the unit is the "
                         f"colour-swapped pair, not the game. Got {games}")
    # ⚠️ `fit_elo` unions the anchor into its name list, so an absent anchor does not
    # raise there — it invents a phantom player at Elo 0 that played nothing, and
    # every rating is then pinned to the `prior` instead of to a real opponent. Every
    # number in the report would look normal. So the pool is checked here.
    if not any(p.name == ANCHOR_NAME for p in pool):
        raise ValueError(
            f"no player named {ANCHOR_NAME!r} in the pool, so the zero of the scale "
            f"would be a phantom that played no games (evaluation.md §5.1). "
            f"`build_pool` always puts it at index 0.")
    log = log or (lambda line: None)

    n = len(pool)
    # Two calendars unioned: SAI's step schedule, and the same-network budget edges
    # that are the only direct read of §5.1a's axis.
    ladder_pairs = budget_ladder_pairs(pool)
    end_pairs = endpoint_pairs(pool)
    pairs = sorted(set(sai_pairings(n, offsets=offsets, anchor_every=anchor_every,
                                    anchor_span=anchor_span))
                   | set(ladder_pairs) | set(end_pairs))
    openings, opening_control = random_openings(games // 2, plies=opening_plies,
                                                seed=seed, device=device)
    budgets = sorted({e.sims for e in pool})
    # A pairing costs roughly the mean of its two budgets, so this is the league's
    # size in units of "one pairing at the reference budget" — the honest cost line,
    # since a 256-sim player is four times a 64-sim one and a flat pairing count hides it.
    units = sum((pool[i].sims + pool[j].sims) / (2.0 * n_sims) for i, j in pairs)

    log(f"  pool     {n} players, {pool[0].name} .. {pool[-1].name}")
    log(f"  calendar {len(pairs)} pairings x {games} games = {len(pairs) * games} games"
        f"  ({len(ladder_pairs)} same-network budget edges, "
        f"{len(end_pairs)} run-endpoint edges)")
    log(f"  openings {games // 2} distinct, {opening_plies} random legal plies")
    log(f"  budgets  {budgets} sims (0 = uniformly random legal play, the zero of the "
        f"scale); reference {n_sims}; batch = {games // 2} games per half")
    log(f"  cost     {units:.0f} pairing-equivalents at n = {n_sims}")
    log(f"  sizing   one edge resolves +-{elo_half_width(games, 0.8):.0f} Elo at d = 0.8, "
        f"+-{elo_half_width(games, 0.5):.0f} at d = 0.5 (arithmetic, evaluation.md §9)")
    log("")

    loader = engine_loader or (
        lambda path: load_engine(path, impl=impl, device=device, quant=quant))
    engines: Dict[object, Callable] = {}

    def _refuse(*_a, **_k):
        """The random player's evaluator. It must never be called.

        `_swap_evaluator` points the search at *some* evaluator every ply, including
        on the random player's turns, and `random_move` does not consult it. If that
        ever stops being true this raises instead of quietly rating a network that
        was supposed to be a coin."""
        raise AssertionError(
            "evaluation.md §5.1: the random player was asked to evaluate a position. "
            "It has no network — `random_move` must not reach the evaluator.")

    def engine(entry: PoolEntry) -> Callable:
        # Keyed by **path**, not by name: the budget grid rates one network at several
        # budgets, and loading the same 6.4M-parameter master three times would put
        # three copies of it on an 8 GB card that is also driving the display.
        key = entry.path
        if key is None:
            return _refuse
        if key not in engines:
            engines[key] = loader(key)
        return engines[key]

    edges: List[Edge] = []
    raw: List[dict] = []
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        a, b = pool[i], pool[j]
        t = time.time()
        result = play_match(engine(a), engine(b), openings, opening_control,
                            n_sims=n_sims, max_plies=max_plies,
                            sims_a=a.sims, sims_b=b.sims,
                            search_impl=search_impl, device=device, seed=seed,
                            search_kw=search_kw)
        edges.append(Edge(a=a.name, b=b.name, a_wins=result.a_wins,
                          draws=result.draws, b_wins=result.b_wins))
        raw.append({"a": a.name, "b": b.name, **result.as_dict(),
                    "seconds": time.time() - t})
        log(f"  [{k + 1:4d}/{len(pairs)}] {a.name:>24s} vs {b.name:<24s} "
            f"{result.a_wins:3d}-{result.draws:3d}-{result.b_wins:<3d} "
            f"score {result.a_score:.3f}  draws {result.draw_rate:5.1%}  "
            f"{result.mean_plies:5.1f} plies  "
            f"{result.unfinished:2d} unfinished  {time.time() - t:6.1f}s")

    fit = fit_elo(edges, anchor=ANCHOR_NAME, prior=prior)
    seconds = time.time() - t0

    curve = _curve_points(pool, fit, edges, n_sims=n_sims, cost=cost)
    # The ladder rung that *is* the pre-2026-08-08 anchor, i.e. the one at the
    # reference budget — that is the player this scale relates to the old ones
    # through, so it is the one worth naming. Any rung falls back, since they are all
    # the same file and the digest is the file's.
    rungs = [p for p in pool if p.run == "" and p.path is not None]
    init = next((p for p in rungs if p.sims == n_sims), rungs[0] if rungs else None)
    return {
        "config": {"games_per_pairing": games, "n_sims": n_sims,
                   "opening_plies": opening_plies, "max_plies": max_plies,
                   "offsets": list(offsets), "anchor_every": anchor_every,
                   "anchor_span": anchor_span, "budgets": budgets,
                   "budget_edges": len(ladder_pairs), "endpoint_edges": len(end_pairs),
                   "pairing_equivalents": units,
                   "impl": impl, "search_impl": search_impl, "seed": seed,
                   # ⚠️ **Part of the scale, like `quant`.** Until 2026-08-22 every
                   # league rated under plain PUCT with no terminal collapse, because
                   # `SearchConfig` defaults both off and `eval_config` never set
                   # them -- while every run since `t12h-gumbel` was *trained* under
                   # `--gumbel --gumbel-m 16 --terminal-collapse`. Ratings from a
                   # Gumbel league and a PUCT league are **not** joinable, and without
                   # this field nothing downstream could tell them apart.
                   "search_kw": dict(search_kw or {}),
                   # ⚠️ Recorded in the report because it is part of the scale. A
                   # league fitted at one precision cannot be joined to one fitted at
                   # another, and without this field nothing downstream could tell.
                   "quant": quant,
                   "prior": prior, "pairings": len(pairs),
                   "games_played": sum(e.games for e in edges),
                   "seconds": seconds},
        "pool": [asdict(p) for p in pool],
        "matches": raw,
        "fit": {"anchor": fit.anchor, "dispersion": fit.dispersion,
                "dispersion_applied": fit.dispersion_applied,
                "decisive": fit.decisive,
                "iterations": fit.iterations, "converged": fit.converged,
                "fit_version": fit.fit_version,
                # ⚠️ Since 2026-08-08 the zero is `random`, which is defined by the
                # rules and has no file to digest. This is the *ladder's* network —
                # `init:n64` is the player that used to be the anchor, so the digest
                # still says which untrained net relates this scale to the old ones.
                "anchor_digest": anchor_digest(init.path) if init is not None else None,
                "init_player": init.name if init is not None else None},
        "curve": curve,
    }


def _curve_points(pool: Sequence[PoolEntry], fit: EloFit, edges: Sequence[Edge],
                  n_sims: int,
                  cost) -> List[dict]:
    """One record per player, in the shape `evaluation.md` §5.3 fixes."""
    stamp = time.time()
    opponents: Dict[str, List[str]] = {}
    played: Dict[str, int] = {}
    drawn: Dict[str, int] = {}
    for e in edges:
        for x, y in ((e.a, e.b), (e.b, e.a)):
            opponents.setdefault(x, []).append(y)
            played[x] = played.get(x, 0) + e.games
            drawn[x] = drawn.get(x, 0) + e.draws

    out = []
    for entry in pool:
        name = entry.name
        n = played.get(name, 0)
        # ⚠️ `cost` is a dict of run -> series for a joint league, but a *bare* series
        # is the older single-run contract and several callers still pass one. Taking
        # only the dict would have made the cost axis silently null for them, which is
        # a column of `None` rather than an exception -- so both are accepted here.
        series = cost.get(entry.run) if isinstance(cost, dict) else cost
        if not entry.run:
            # ⚠️ The anchor and the §5.1a ladder belong to no run, so a per-run `cost`
            # dict has no series for them and `cost.get("")` is None — which wrote a
            # **null** x for the origin of the curve. Confirmed in
            # `logs/curve-joint-pcr.csv`, whose `anchor` row has an empty
            # `euros_spent` and `training_seconds`; every joint league since
            # 2026-08-05 lost its zero that way, and a missing x reads as "not
            # measured" rather than as the exact zero it is. They cost nothing to
            # train because nobody trained them, so it is 0.0 and not unknown.
            at = {"euros": 0.0, "training_seconds": 0.0}
        else:
            at = {} if not series else series_at(series, entry.step)
        out.append({
            "checkpoint_id": name,
            "run": entry.run,
            "step": entry.step,
            "euros_spent": at.get("euros"),
            "training_seconds": at.get("training_seconds"),
            "games_played": n,
            "elo": fit.elo.get(name),
            "ci95": None if name not in fit.se else fit.ci95(name),
            "se": fit.se.get(name),
            "se_raw": fit.se_raw.get(name),
            # ⚠️ The player's **own** budget, not the league's reference. This column
            # used to be a constant restated per row; it is now the second axis.
            "n_sims": entry.sims,
            "draw_rate": (drawn.get(name, 0) / n) if n else None,
            "opponents": sorted(opponents.get(name, [])),
            "fit_version": fit.fit_version,
            "timestamp": stamp,
        })
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="the D2 league, docs/reference/evaluation.md §5.4")
    p.add_argument("--run", required=True, action="append", metavar="RUN",
                   help="the training run whose checkpoints are rated. Repeatable: "
                        "several runs go into ONE Bradley-Terry fit, interleaved by "
                        "step so they actually play each other. Two runs rated "
                        "separately share only the anchor, which pins each scale's "
                        "zero but not its units, and the difference between them is "
                        "then confounded with the fits")
    p.add_argument("--games", type=int, default=36,
                   help="games per pairing, even; half that many distinct openings")
    p.add_argument("--sims", type=int, default=64,
                   help="the REFERENCE budget: the one the training curve is rated "
                        "at. Other budgets are players in the same fit, not other "
                        "leagues -- see --ladder and --grid")
    p.add_argument("--ladder", type=int, nargs="*", default=list(DEFAULT_LADDER),
                   metavar="N",
                   help="budgets to rate the untrained network at. These bridge "
                        "uniformly random play (the zero) to the start of the "
                        "training curve, which is 700+ Elo away and unmeasurable in "
                        "one pairing. Empty for none")
    p.add_argument("--grid", type=int, nargs="*", default=[], metavar="N",
                   help="also rate --grid-points checkpoints at these budgets. This "
                        "is the search-versus-training exchange rate: how much "
                        "training a doubling of search is worth, measured on one "
                        "scale. ⚠️ a 256-sim player costs 4x a 64-sim one")
    p.add_argument("--grid-points", type=int, default=0,
                   help="how many checkpoints per run get the --grid budgets, evenly "
                        "spaced over the run")
    p.add_argument("--anchor-span", type=int, default=None,
                   help="cap the anchor's spread at the first N players. Its edges "
                        "against anything far stronger come back 0-0-N and cost a "
                        "full pairing to learn nothing")
    p.add_argument("--opening-plies", type=int, default=8,
                   help="random legal plies per opening, even so White is to move")
    p.add_argument("--max-plies", type=int, default=512)
    p.add_argument("--limit", type=int, default=32,
                   help="rate at most this many players **per run**, evenly spaced by step")
    p.add_argument("--anchor-every", type=int, default=4,
                   help="the anchor also plays every Nth checkpoint; 0 for bare SAI")
    p.add_argument("--checkpoints", default=None,
                   help="runs/<run>/checkpoints per run by default")
    p.add_argument("--anchor", default=DEFAULT_ANCHOR_PATH)
    p.add_argument("--train-log", default=None,
                   help="runs/<run>/<run>.jsonl by default; the euro axis is joined "
                        "from it")
    p.add_argument("--impl", default="cuda", help="the fused encoder; 'none' for the torch module")
    p.add_argument("--quant", default=None, choices=("fp8", "int8"),
                   help="rate the players on a quantised kernel. ⚠️ This is part of "
                        "the Elo scale: a league run with it cannot be joined to the "
                        "existing ones, which are all fp16. Default off for that "
                        "reason, not because it is slow")
    p.add_argument("--search-impl", default="cuda", choices=("cuda", "torch"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gumbel", action="store_true",
                   help="rate under Gumbel MuZero root selection instead of PUCT. "
                        "⚠️ This is the protocol every run since t12h-gumbel was "
                        "TRAINED under; the league has always rated under PUCT. "
                        "gumbel_scale stays 0 (eval_config), so it is deterministic "
                        "sequential halving over the top-m prior actions, ranked by "
                        "completed Q. Not joinable with a PUCT league.")
    p.add_argument("--gumbel-m", type=int, default=16,
                   help="root actions considered under --gumbel. 16 is mctx's default "
                        "and what training used")
    p.add_argument("--terminal-collapse", action="store_true",
                   help="§6.6a's collapse onto proved-winning edges, as training ran "
                        "it. ⚠️ Also off in every league before 2026-08-22")
    p.add_argument("--prior", type=float, default=1.0,
                   help="drawn games against a phantom at Elo 0, per player")
    p.add_argument("--out", default=None,
                   help="runs/<run>/league-<run>.json. For several runs it lands in "
                        "the FIRST run's folder as league-joint-<a>+<b>.json")
    return p


def _search_kw(args) -> dict:
    """The non-default search settings, as a dict for `SearchConfig`.

    Empty when neither flag is given, so a league run without them is byte-identical
    to every league before 2026-08-22 and the old ratings stay valid.
    """
    kw = {}
    if getattr(args, "gumbel", False):
        kw["gumbel"] = True
        kw["gumbel_m"] = int(args.gumbel_m)
    if getattr(args, "terminal_collapse", False):
        kw["terminal_collapse"] = True
    return kw


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    runs = list(dict.fromkeys(args.run))
    stem = runs[0] if len(runs) == 1 else "joint-" + "+".join(runs)
    # ⚠️ A joint league lands in the **first** run's folder, by the convention
    # `brokefish/paths.py` documents. The order is the order given on the command line,
    # so the choice is visible in the invocation rather than decided by sorting.
    out_path = args.out or paths.artifact(runs[0], f"league-{stem}.json", create=True)
    log = _Log(os.path.splitext(out_path)[0] + ".log")

    log(f"league  run={'+'.join(runs)}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  tail -f {os.path.splitext(out_path)[0] + '.log'}")
    pool = build_pool(runs, checkpoint_dir=args.checkpoints,
                      anchor_path=args.anchor, limit=args.limit, sims=args.sims,
                      ladder=args.ladder, grid=args.grid,
                      grid_points=args.grid_points)
    if len(runs) > 1:
        log(f"  ⚠️ {len(runs)} runs in ONE fit, interleaved by step so the calendar's "
            f"offsets cross between them. Ratings from this report are comparable "
            f"across runs; ratings from two separate leagues are not.")
    # ⚠️ One series per run. `--train-log` names a single file and so only makes sense
    # for a single run; with several, each is joined from its own run folder.
    cost = {}
    for run in runs:
        path = args.train_log if (args.train_log and len(runs) == 1) \
            else paths.train_log(run)
        series = training_series(path)
        if not series:
            log(f"  ⚠️ no training log at {path}; {run}'s cost axis will be null")
        else:
            cost[run] = series

    report = run_league(
        pool, games=args.games, n_sims=args.sims, opening_plies=args.opening_plies,
        max_plies=args.max_plies, anchor_every=args.anchor_every,
        anchor_span=args.anchor_span,
        impl=None if args.impl == "none" else args.impl, quant=args.quant,
        search_impl=args.search_impl, device=args.device, seed=args.seed,
        search_kw=_search_kw(args),
        prior=args.prior, cost=cost or None, log=log)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1)

    log("")
    from .curve import format_curve
    log(format_curve(report))
    log("")
    log(f"  zero is {report['fit']['anchor']} — uniformly random legal play "
        f"(evaluation.md §5.1). Ladder net {report['fit']['init_player']} "
        f"sha256 {report['fit']['anchor_digest']}")
    fit_info = report["fit"]
    if fit_info["dispersion_applied"]:
        log(f"  dispersion {fit_info['dispersion']:.3f} applied over "
            f"{fit_info['decisive']} decisive games  "
            f"(1 - draw rate is what to compare it to; see elo.py)")
    else:
        log(f"  ⚠️ dispersion {fit_info['dispersion']:.3f} NOT applied: only "
            f"{fit_info['decisive']} decisive games. The intervals below are the "
            f"uncorrected Bradley-Terry ones, which are too wide by ~1/sqrt(1-d)")
    log(f"  {report['config']['games_played']} games in "
        f"{report['config']['seconds'] / 60:.1f} min -> {out_path}")
    log.close()


if __name__ == "__main__":
    main()
