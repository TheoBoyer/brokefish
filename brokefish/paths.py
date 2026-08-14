"""Where a run's artifacts live. **The layout is defined here and nowhere else.**

Before 2026-08-14 a run scattered itself across four directories — `logs/<run>.log`
and `logs/<run>.jsonl` beside every other log the repository has ever written,
`checkpoints/<run>-*.pt` beside every other run's, `data/replay/<run>.dat`, and a
`wandb/run-*` directory named after wandb's own id rather than the run. Nothing was
wrong with any single one of them; together they meant that "everything `t12h-gumbel`
produced" was a question you answered with `ls | grep`.

Now::

    runs/<run>/
        <run>.log               the human-readable tail -f target
        <run>.jsonl             one record per generation, the euro axis
        <run>-puzzles.jsonl     the layer-1 probe's per-puzzle detail
        checkpoints/
            <run>.pt            rolling, resumable: weights + optimiser + buffer index
            <run>-NNNNNN.pt     the history, weights only, under --keep-checkpoints
        replay/
            <run>.dat           the ring, and its .index.npz
        wandb/run-*/            wandb's own directory, placed here by `dir=`
        league-<run>.json       and the curve csv/png beside it
        ...                     anything else an evaluation writes for this run

⚠️ **Two things deliberately stay outside.**

`checkpoints/anchor.pt` is the league's zero point — the fixed random network every
Elo scale is anchored to (`evaluation.md` §5.1a). It belongs to no run, and putting a
copy in each run's folder would let two leagues silently anchor to two different files.

`logs/` survives for what it was always good at: ad-hoc investigation output that is
not a run. `logs/fp8-attn.log` is a measurement of a kernel, not of a training run,
and it has no run folder to go in.

⚠️ **Artifacts spanning several runs go in the first run's folder** — a joint league
over `t12h-muong` and `t12h-gumbel` writes into `runs/t12h-muong/`. That is a
convention, not a deduction: the alternative was a combined folder per pair, and the
combinatorics of that get ugly the moment a third run joins.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

#: Everything a run produces hangs off here.
RUNS_ROOT = "runs"

#: ⚠️ Shared, not per-run. See the module docstring.
ANCHOR_PATH = os.path.join("checkpoints", "anchor.pt")

#: Where investigation output that is not a run still goes.
LOGS_ROOT = "logs"


def _maybe_make(path: str, create: bool) -> str:
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def run_dir(run: str, create: bool = False) -> str:
    """`runs/<run>`. Also the directory wandb is pointed at."""
    return _maybe_make(os.path.join(RUNS_ROOT, run), create)


def checkpoint_dir(run: str, create: bool = False) -> str:
    return _maybe_make(os.path.join(RUNS_ROOT, run, "checkpoints"), create)


def buffer_dir(run: str, create: bool = False) -> str:
    return _maybe_make(os.path.join(RUNS_ROOT, run, "replay"), create)


def artifact(run: str, name: str, create: bool = False) -> str:
    """A named file inside a run's folder — a league report, a curve, a PGN."""
    _maybe_make(os.path.join(RUNS_ROOT, run), create)
    return os.path.join(RUNS_ROOT, run, name)


def joint_dir(runs: Iterable[str], create: bool = False) -> str:
    """The folder for an artifact spanning several runs: **the first one's**.

    Takes the runs in the order the caller listed them, which is the order they were
    given on the command line, so the choice is visible in the invocation rather than
    decided by sorting.
    """
    first = next(iter(runs))
    return run_dir(first, create)


def train_log(run: str) -> str:
    """`runs/<run>/<run>.jsonl` — the record stream the euro axis is joined from."""
    return os.path.join(RUNS_ROOT, run, f"{run}.jsonl")


def resolve(explicit: Optional[str], run: str, kind: str, create: bool = False) -> str:
    """An explicit path if given, else this run's default for `kind`.

    Every CLI keeps its `--checkpoints` / `--buffer-dir` override, because a rented
    box with a fast scratch disk is a real case and so is pointing an evaluation at
    somebody else's checkpoints. The default is what changed, not the flag.
    """
    if explicit:
        return _maybe_make(explicit, create)
    return {"checkpoints": checkpoint_dir, "replay": buffer_dir,
            "run": run_dir}[kind](run, create)
