"""Move pre-2026-08-14 artifacts into the `runs/<run>/` layout.

    python scripts/migrate_runs.py            # print the plan, move nothing
    python scripts/migrate_runs.py --apply

`brokefish/paths.py` owns the layout; this only relocates what already exists.

⚠️ **Conservative on purpose.** Anything this cannot attribute to a run with certainty
is *left where it is* and listed at the end. `logs/` is not meant to end up empty: it
keeps the investigation output that belongs to no run — `logs/fp8-attn.log` is a
measurement of a kernel, `logs/gate1a.log` is the evidence behind a ledger row.
Guessing at those and scattering them into run folders would be worse than leaving a
short list for a human to decide about.

The attribution rules, strongest first:

* **checkpoints** — `checkpoints/<run>-NNNNNN.pt` and `<run>.pt`, by filename. The
  shared `anchor.pt` never moves.
* **replay** — `data/replay/<run>.dat` and its index, by filename.
* **wandb** — each `wandb/run-*/files/wandb-metadata.json` records the argv that
  started it, so the run comes from its own `--run` flag rather than from a guess.
* **league reports** — read the JSON and take the runs it actually rates; a joint
  report goes to the **first** of them, per `paths.joint_dir`.
* **curves** — beside the league report whose stem they share.
* **per-run logs** — `<run>.log`, `<run>.jsonl`, `<run>-puzzles.jsonl`, and
  `chain-*.log` where the suffix names a known run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brokefish import paths  # noqa: E402

STEP_RE = re.compile(r"^(?P<run>.+)-(?P<step>\d{6})\.pt$")


def known_runs() -> List[str]:
    """What counts as a run, longest name first so `t15-adamw` beats `t15`.

    ⚠️ **Two signals, and both are deliberately strict.** A checkpoint is proof: only
    the training loop writes `<run>.pt`. Failing that, a name qualifies only if it has
    *both* `<name>.log` and `<name>.jsonl`, which is what `train.log.Logger` opens as a
    pair and what nothing else does.

    The first draft accepted any `<name>.jsonl` and produced 47 folders, a dozen of
    them holding one file: `layer0_cuda`, `probe-agc`, `gate2-roots` are evaluation
    output, not runs, and giving each a run folder would have made the directory it is
    supposed to tidy harder to read rather than easier.
    """
    runs = set()
    for p in glob.glob(os.path.join("checkpoints", "*.pt")):
        base = os.path.basename(p)
        if base == "anchor.pt":
            continue
        m = STEP_RE.match(base)
        runs.add(m.group("run") if m else base[:-3])
    for p in glob.glob(os.path.join("logs", "*.jsonl")):
        name = os.path.basename(p)[: -len(".jsonl")]
        if name.endswith("-puzzles"):
            name = name[: -len("-puzzles")]
        if os.path.exists(os.path.join("logs", f"{name}.log")) and \
                os.path.exists(os.path.join("logs", f"{name}.jsonl")):
            runs.add(name)
    return sorted(runs, key=len, reverse=True)


def runs_in_report(path: str) -> List[str]:
    """The runs a league report actually rates, in the order it lists them."""
    try:
        with open(path) as fh:
            rep = json.load(fh)
    except Exception:
        return []
    out: List[str] = []
    for key in ("players", "entries", "pool"):
        for row in rep.get(key, []) or []:
            r = row.get("run") if isinstance(row, dict) else None
            if r and r not in out and r not in ("random", "anchor"):
                out.append(r)
    return out


def wandb_run(d: str) -> Optional[str]:
    meta = os.path.join(d, "files", "wandb-metadata.json")
    try:
        with open(meta) as fh:
            args = json.load(fh).get("args", [])
    except Exception:
        return None
    if "--run" in args:
        i = args.index("--run")
        if i + 1 < len(args):
            return args[i + 1]
    return None


def plan() -> Tuple[List[Tuple[str, str]], List[str]]:
    runs = known_runs()
    moves: List[Tuple[str, str]] = []
    skipped: List[str] = []

    # -- checkpoints --------------------------------------------------------- #
    for p in sorted(glob.glob(os.path.join("checkpoints", "*"))):
        base = os.path.basename(p)
        if base == "anchor.pt":
            continue                                   # shared; see paths.py
        m = STEP_RE.match(base)
        run = m.group("run") if m else base.split(".pt")[0]
        if run in runs:
            moves.append((p, os.path.join(paths.checkpoint_dir(run), base)))
        else:
            skipped.append(p)

    # -- replay buffers ------------------------------------------------------ #
    for p in sorted(glob.glob(os.path.join("data", "replay", "*"))):
        base = os.path.basename(p)
        run = next((r for r in runs if base.startswith(r + ".")), None)
        if run:
            moves.append((p, os.path.join(paths.buffer_dir(run), base)))
        else:
            skipped.append(p)

    # -- wandb --------------------------------------------------------------- #
    for d in sorted(glob.glob(os.path.join("wandb", "*run-*"))):
        run = wandb_run(d)
        if run:
            moves.append((d, os.path.join(paths.run_dir(run), "wandb",
                                          os.path.basename(d))))
        else:
            skipped.append(d)

    # -- league reports first: curves follow them ---------------------------- #
    league_home: Dict[str, str] = {}                   # stem -> run
    for p in sorted(glob.glob(os.path.join("logs", "league-*.json"))):
        stem = os.path.basename(p)[len("league-"): -len(".json")]
        found = runs_in_report(p)
        run = found[0] if found else next((r for r in runs if stem == r), None)
        if not run:
            skipped.append(p)
            continue
        league_home[stem] = run
        moves.append((p, paths.artifact(run, os.path.basename(p))))
        side = os.path.join("logs", f"league-{stem}.log")
        if os.path.exists(side):
            moves.append((side, paths.artifact(run, os.path.basename(side))))

    for p in sorted(glob.glob(os.path.join("logs", "curve-*"))):
        stem = os.path.basename(p)[len("curve-"):]
        stem = os.path.splitext(stem)[0]
        run = league_home.get(stem) or next((r for r in runs if stem == r), None)
        if run:
            moves.append((p, paths.artifact(run, os.path.basename(p))))
        else:
            skipped.append(p)

    # -- per-run logs -------------------------------------------------------- #
    already = {a for a, _ in moves}
    for p in sorted(glob.glob(os.path.join("logs", "*"))):
        if p in already or os.path.isdir(p):
            continue
        base = os.path.basename(p)
        run = None
        for r in runs:                                 # longest first
            if base in (f"{r}.log", f"{r}.jsonl", f"{r}-puzzles.jsonl",
                        f"chain-{r}.log", f"{r}.diff"):
                run = r
                break
        if run is None and base.startswith("chain-") and base.endswith(".log"):
            # ⚠️ Chain scripts were named by hand, so `chain-muong.log` belongs to
            # `t12h-muong` and `chain-agrepro.log` to `t12h-agrepro`. A **unique**
            # suffix match is safe; `chain-24h.log` matches three runs and is left
            # alone rather than assigned to whichever sorts first.
            tag = base[len("chain-"): -len(".log")]
            hits = [r for r in runs if r.endswith(tag)]
            if len(hits) == 1:
                run = hits[0]
        if run:
            moves.append((p, paths.artifact(run, base)))
        else:
            skipped.append(p)
    return moves, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="move; otherwise print only")
    a = ap.parse_args()

    moves, skipped = plan()
    # The curve pass and the catch-all pass both see an unattributed file.
    seen = set()
    skipped = [x for x in skipped if not (x in seen or seen.add(x))]
    by_run: Dict[str, int] = defaultdict(int)
    for _, dst in moves:
        by_run[dst.split(os.sep)[1]] += 1

    print(f"{len(moves)} files/dirs to move into {len(by_run)} run folders\n")
    for run in sorted(by_run):
        print(f"  runs/{run:<24} {by_run[run]:>4}")
    print(f"\n{len(skipped)} left where they are (no confident run attribution):")
    for p in skipped[:40]:
        print(f"  {p}")
    if len(skipped) > 40:
        print(f"  ... and {len(skipped) - 40} more")

    if not a.apply:
        print("\ndry run -- nothing moved. Re-run with --apply.")
        return 0

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"  ⚠️ exists, skipped: {dst}")
            continue
        shutil.move(src, dst)
    print(f"\nmoved {len(moves)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
