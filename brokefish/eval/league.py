"""The league: every checkpoint of a run, rated on one scale. `evaluation.md` §5.4.

    uv run --no-project --python .venv/bin/python -m brokefish.eval.league \
        --run t4h-n64 --games 36 --sims 64

What it does, in order: build the pool (the frozen anchor plus the run's
checkpoints), generate one fixed set of openings, play the fixed SAI calendar of
pairings, fit one global Bradley-Terry model over the whole graph, join the euro
counter from the training log, and write one record per checkpoint.

Three things are deliberate.

**The anchor is a file, not a seed.** `evaluation.md` §5.1 makes the random-init
network the zero of the scale forever. A seed is not forever — it depends on the
torch version, on the initialisation order, and on nobody editing `BrokefishNet`'s
constructor. So the first run writes `checkpoints/anchor.pt` and every run after
that loads it. Delete that file and the whole curve moves.

**The calendar is fixed.** Each checkpoint plays generation offsets ±1, ±2, ±3,
±6, ±8, ±12 (SAI's schedule, §5.2), plus the anchor plays a spread across the whole
run. The long-range edges are the load-bearing part: without them the graph is a
path and the global fit degenerates back into the chain §5.2 exists to avoid.
Variance-proportional sampling — spend games where `p(1-p)` is largest — is the
upgrade, and `EloFit.predict` is the function it needs; it is not in v1 because it
needs an online fit and the fixed calendar needs nothing.

**Nothing here may select a checkpoint** (`evaluation.md` §2). This module writes
to `logs/`; nothing under `brokefish/train/` reads what it writes, and
`tests/test_league.py` asserts that rather than trusting it. Keeping the checkpoint
with the best league rating is distillation through a one-bit channel and it is the
prohibition most likely to be violated by accident, because it looks like good
practice.

⚠️ **`--keep-checkpoints` writes a bare `state_dict`** with no step and no euro
count in it, so the euro axis is joined from `logs/<run>.jsonl` by step. If that
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

ANCHOR_NAME = "anchor"
DEFAULT_ANCHOR_PATH = os.path.join("checkpoints", "anchor.pt")


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PoolEntry:
    """One rated player: a name, a place on the x-axis, and where its weights are."""

    name: str
    step: int
    path: Optional[str]       # None only for a freshly minted anchor


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


def discover_checkpoints(run: str, checkpoint_dir: str = "checkpoints") -> List[Tuple[int, str]]:
    """`(step, path)` for every ``{run}-{step:06d}.pt`` snapshot, in step order."""
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


def build_pool(run: str, checkpoint_dir: str = "checkpoints",
               anchor_path: str = DEFAULT_ANCHOR_PATH,
               limit: Optional[int] = 32) -> List[PoolEntry]:
    """The anchor at index 0, then the run's checkpoints in step order.

    Index 0 is the anchor on purpose: the SAI calendar then connects it to
    checkpoints 1, 2, 3, 6, 8 and 12 for free, so the zero of the scale is a real
    node in the graph rather than an assumption bolted on afterwards.
    """
    pool = [PoolEntry(name=ANCHOR_NAME, step=0, path=ensure_anchor(anchor_path))]
    found = discover_checkpoints(run, checkpoint_dir)
    if not found:
        raise FileNotFoundError(
            f"no {run}-NNNNNN.pt snapshots in {checkpoint_dir}/. A run only writes "
            f"them with --keep-checkpoints; the rolling {run}.pt is one endpoint and "
            f"a curve needs a history.")
    for step, path in subsample(found, None if limit is None else max(1, limit - 1)):
        pool.append(PoolEntry(name=f"{run}@{step}", step=step, path=path))
    return pool


# --------------------------------------------------------------------------- #
# The calendar
# --------------------------------------------------------------------------- #

def sai_pairings(n: int, offsets: Sequence[int] = SAI_OFFSETS,
                 anchor_every: int = 4) -> List[Tuple[int, int]]:
    """Which pairs of pool indices play. `evaluation.md` §5.2.

    Each index plays the ones `offsets` ahead of it, which gives every checkpoint
    up to `2 * len(offsets)` edges — six forward, six backward.

    `anchor_every` adds an edge from index 0 to every `anchor_every`-th checkpoint.
    The calendar alone only connects the anchor to the first twelve, and a scale
    whose zero is measured only against the start of the run is one whose late
    points are reached through a long chain of intermediate fits — the exact
    failure mode §5.2 is about. Set it to 0 to get the bare SAI schedule.
    """
    pairs = {(i, i + d) for i in range(n) for d in offsets if i + d < n}
    if anchor_every > 0:
        pairs |= {(0, j) for j in range(1, n, anchor_every)}
        if n > 1:
            pairs.add((0, n - 1))          # the far end always meets the anchor
    return sorted(pairs)


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

def load_engine(path: str, impl: Optional[str] = "cuda",
                device: str = "cuda") -> Callable:
    """An evaluator for one checkpoint: ``(boards, control, rep) -> logits``.

    ⚠️ The master network is built and loaded on the **CPU** and only the packed
    fp16 weights go to the card. A league holds thirty of these at once, and thirty
    fp32 masters is a gigabyte of an eight-gigabyte card that is also driving the
    display. `FusedEncoder` does its own `.half()` on what it packs, so an fp32 CPU
    module is a perfectly good source.
    """
    from brokefish.nn.model import BrokefishNet

    from .layer0 import load_net_state

    net = BrokefishNet()
    net.load_state_dict(load_net_state(path, device="cpu"))
    net.eval()
    if impl is None:
        net = net.to(device)
        return lambda b, c, r: net(b, c, r)
    from brokefish.nn import encoder_impl

    return encoder_impl(impl)(net).forward_full


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


@torch.no_grad()
def run_league(pool: Sequence[PoolEntry], games: int = 36, n_sims: int = 64,
               opening_plies: int = 8, max_plies: int = 512,
               offsets: Sequence[int] = SAI_OFFSETS, anchor_every: int = 4,
               impl: Optional[str] = "cuda", search_impl: str = "cuda",
               device: str = "cuda", seed: int = 0, prior: float = 1.0,
               cost: Optional[Sequence[Tuple[int, dict]]] = None,
               log: Optional[Callable[[str], None]] = None,
               engine_loader: Callable[[str], Callable] = None) -> dict:
    """Play the calendar, fit the ratings, return the whole record.

    `games` is the number of games **per pairing** and must be even: the unit of
    measurement is the colour-swapped game pair, so an odd count would leave one
    opening played from one side only, which is the imbalance the pairing exists to
    remove.
    """
    if games % 2:
        raise ValueError(f"games per pairing must be even — the unit is the "
                         f"colour-swapped pair, not the game. Got {games}")
    log = log or (lambda line: None)

    n = len(pool)
    pairs = sai_pairings(n, offsets=offsets, anchor_every=anchor_every)
    openings, opening_control = random_openings(games // 2, plies=opening_plies,
                                                seed=seed, device=device)

    log(f"  pool     {n} players, {pool[0].name} .. {pool[-1].name}")
    log(f"  calendar {len(pairs)} pairings x {games} games = {len(pairs) * games} games")
    log(f"  openings {games // 2} distinct, {opening_plies} random legal plies")
    log(f"  budget   n = {n_sims} sims, batch = {games // 2} games per half")
    log(f"  sizing   one edge resolves +-{elo_half_width(games, 0.8):.0f} Elo at d = 0.8, "
        f"+-{elo_half_width(games, 0.5):.0f} at d = 0.5 (arithmetic, evaluation.md §9)")
    log("")

    loader = engine_loader or (lambda path: load_engine(path, impl=impl, device=device))
    engines: Dict[str, Callable] = {}

    def engine(entry: PoolEntry) -> Callable:
        if entry.name not in engines:
            engines[entry.name] = loader(entry.path)
        return engines[entry.name]

    edges: List[Edge] = []
    raw: List[dict] = []
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        a, b = pool[i], pool[j]
        t = time.time()
        result = play_match(engine(a), engine(b), openings, opening_control,
                            n_sims=n_sims, max_plies=max_plies,
                            search_impl=search_impl, device=device, seed=seed)
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
    return {
        "config": {"games_per_pairing": games, "n_sims": n_sims,
                   "opening_plies": opening_plies, "max_plies": max_plies,
                   "offsets": list(offsets), "anchor_every": anchor_every,
                   "impl": impl, "search_impl": search_impl, "seed": seed,
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
                # Which anchor this scale's zero actually was. `checkpoints/` is
                # gitignored, so this digest is the only durable record of it.
                "anchor_digest": anchor_digest(pool[0].path) if pool[0].path else None},
        "curve": curve,
    }


def _curve_points(pool: Sequence[PoolEntry], fit: EloFit, edges: Sequence[Edge],
                  n_sims: int,
                  cost: Optional[Sequence[Tuple[int, dict]]]) -> List[dict]:
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
        at = {} if cost is None else series_at(cost, entry.step)
        out.append({
            "checkpoint_id": name,
            "step": entry.step,
            "euros_spent": at.get("euros"),
            "training_seconds": at.get("training_seconds"),
            "games_played": n,
            "elo": fit.elo.get(name),
            "ci95": None if name not in fit.se else fit.ci95(name),
            "se": fit.se.get(name),
            "se_raw": fit.se_raw.get(name),
            "n_sims": n_sims,
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
    p.add_argument("--run", required=True, help="the training run whose checkpoints are rated")
    p.add_argument("--games", type=int, default=36,
                   help="games per pairing, even; half that many distinct openings")
    p.add_argument("--sims", type=int, default=64, help="simulations per move")
    p.add_argument("--opening-plies", type=int, default=8,
                   help="random legal plies per opening, even so White is to move")
    p.add_argument("--max-plies", type=int, default=512)
    p.add_argument("--limit", type=int, default=32,
                   help="rate at most this many players, evenly spaced by step")
    p.add_argument("--anchor-every", type=int, default=4,
                   help="the anchor also plays every Nth checkpoint; 0 for bare SAI")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--anchor", default=DEFAULT_ANCHOR_PATH)
    p.add_argument("--train-log", default=None,
                   help="logs/<run>.jsonl by default; the euro axis is joined from it")
    p.add_argument("--impl", default="cuda", help="the fused encoder; 'none' for the torch module")
    p.add_argument("--search-impl", default="cuda", choices=("cuda", "torch"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prior", type=float, default=1.0,
                   help="drawn games against a phantom at Elo 0, per player")
    p.add_argument("--out", default=None, help="logs/league-<run>.json by default")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    out_path = args.out or os.path.join("logs", f"league-{args.run}.json")
    log = _Log(os.path.splitext(out_path)[0] + ".log")

    log(f"league  run={args.run}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  tail -f {os.path.splitext(out_path)[0] + '.log'}")
    pool = build_pool(args.run, checkpoint_dir=args.checkpoints,
                      anchor_path=args.anchor, limit=args.limit)
    cost = training_series(args.train_log or os.path.join("logs", f"{args.run}.jsonl"))
    if not cost:
        log("  ⚠️ no training log found; the curve will have a null cost axis")

    report = run_league(
        pool, games=args.games, n_sims=args.sims, opening_plies=args.opening_plies,
        max_plies=args.max_plies, anchor_every=args.anchor_every,
        impl=None if args.impl == "none" else args.impl,
        search_impl=args.search_impl, device=args.device, seed=args.seed,
        prior=args.prior, cost=cost or None, log=log)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1)

    log("")
    from .curve import format_curve
    log(format_curve(report))
    log("")
    log(f"  anchor sha256 {report['fit']['anchor_digest']}  "
        f"(evaluation.md §5.1: the zero of this scale)")
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
