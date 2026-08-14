"""Score a network on Lichess puzzles — from inside the training loop, or outside it.

Two entry points, same measurement:

- **`PuzzleProbe`** — called by `train/loop.py` right after each checkpoint. This is
  the one to use. ~20 s per checkpoint against ~40 min of training between them.
- **`watch`** — a CLI that polls a run's checkpoint directory from a second process,
  for scoring a run that has already finished, or one launched without the probe::

      uv run --no-project --python .venv/bin/python -m brokefish.eval.watch --run t24h-n256

What it measures is the one thing self-play cannot: an **absolute** score against a
fixed external reference, **independent of the evaluation simulation budget**. Every
other number we have is either relative to an opponent that moves with the network
(self-play statistics) or tied to a search budget that makes two leagues incomparable
(Elo, `evaluation.md` §3). The puzzle metrics are on neither footing.

⚠️ **How `evaluation.md` §2 is enforced here.** The prohibition is that evaluation
output may not reach a training decision. Until 2026-08-02 that was enforced by
banning `brokefish/train/` from importing `brokefish/eval/` at all — a module-graph
proxy for a dataflow rule, which also forced this measurement into a second process.
`PuzzleProbe.run` now **returns `None`**: it writes to the logger and hands the caller
nothing, so no score exists inside the training process to threshold or select on.
That is a strictly stronger guarantee than the import ban, and
`tests/test_league.py::TestAntiSelection` asserts both the signature and the call site.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from brokefish import paths

import torch

# Coarse buckets for the dashboard. The full 14-bin curve lives in the JSONL; 56
# wandb series is a chart nobody reads, and these four are the shape.
BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("easy", 0, 1000), ("mid", 1000, 1600),
    ("hard", 1600, 2200), ("expert", 2200, 10_000),
)


def score_checkpoint(path: str, puzzles, impl: Optional[str] = None,
                     ks: Sequence[int] = (1, 3, 5), batch: int = 1024,
                     device: str = "cuda") -> dict:
    """Every puzzle metric for one checkpoint, as flat keys ready for wandb."""
    from brokefish.nn.model import BrokefishNet

    from .layer0 import load_net_state
    from .puzzles import score_puzzles_line

    net = BrokefishNet()
    net.load_state_dict(load_net_state(path, device="cpu"))
    net = net.to(device).eval()
    try:
        r = score_puzzles_line(puzzles, net, impl=impl, batch=batch,
                               ks=tuple(ks), device=device)
    finally:
        del net
        torch.cuda.empty_cache()

    out = {"puzzles/solve_rate": r["solve_rate"],
           "puzzles/solve_ci95_lo": r["ci95"][0],
           "puzzles/solve_ci95_hi": r["ci95"][1],
           "puzzles/mean_line": r["mean_line"],
           "puzzles/n_turns": r["n_turns"]}
    for k in ks:
        out[f"puzzles/move_pass@{k}"] = r[f"move_pass@{k}"]
    for name, lo, hi in BUCKETS:
        sel = [b for b in r["bins"] if lo <= b["rating_lo"] < hi]
        n = sum(b["n"] for b in sel)
        t = sum(b["n_turns"] for b in sel)
        if not n:
            continue
        out[f"puzzles/solve_rate_{name}"] = sum(b["solved"] for b in sel) / n
        for k in ks:
            hits = sum(b[f"move_pass@{k}"] * b["n_turns"] for b in sel)
            out[f"puzzles/move_pass@{k}_{name}"] = hits / t if t else 0.0
    out["_bins"] = r["bins"]
    return out


class PuzzleProbe:
    """Score the live network on puzzles from **inside** the training loop.

    ⚠️ **`run` returns `None`, and that is the safety mechanism.** `evaluation.md`
    §2 forbids evaluation output from reaching a training decision. Until 2026-08-02
    that was enforced by forbidding `brokefish/train/` to import `brokefish/eval/`
    at all — a module-graph proxy for a dataflow rule, and one that forced this
    measurement out into a second process nobody wanted to babysit.

    The rule is now enforced where it actually lives: the probe writes its numbers
    straight to the logger and hands the caller nothing. There is no value in the
    training process to compare, threshold, or select on — "keep the checkpoint with
    the best puzzle score" cannot be written, because the score does not exist there.
    `tests/test_league.py::TestAntiSelection` asserts the signature and the call site.

    ⚠️ **The probe's time is not training time.** `train.md` §10 defines the curve's
    x-axis as self-play plus gradient and nothing else, so this must never be added
    to `Trainer.seconds`. It is logged as `puzzles/seconds` instead, which keeps it
    visible without letting it contaminate the cost axis.

    The puzzle set loads once (~16 s) and is reused for the run's lifetime.
    """

    def __init__(self, limit: Optional[int] = 20_000, ks: Sequence[int] = (1, 3, 5),
                 impl: Optional[str] = None, batch: int = 1024,
                 device: str = "cuda", detail_path: Optional[str] = None) -> None:
        self.limit, self.ks, self.impl = limit, tuple(ks), impl
        self.batch, self.device = batch, device
        self.detail_path = detail_path
        self._puzzles = None
        self.unavailable: Optional[str] = None

    def _load(self):
        if self._puzzles is None and self.unavailable is None:
            from .puzzles import load_puzzles
            self._puzzles = load_puzzles(limit=self.limit, device=self.device)
        return self._puzzles

    @torch.no_grad()
    def run(self, net, log, step: int) -> None:
        """Score `net` and log it. **Returns nothing, by design — see the class.**

        ⚠️ Restores `net.training`: the loop hands over a module in train mode and
        would otherwise get it back in eval mode, which disables nothing today (there
        is no dropout) and would silently disable it the day there is.
        """
        from .puzzles import score_puzzles_line

        if self.unavailable is not None:
            return
        was_training = net.training
        t0 = time.time()
        try:
            puzzles = self._load()
            net.eval()
            r = score_puzzles_line(puzzles, net, impl=self.impl, batch=self.batch,
                                   ks=self.ks, device=self.device)
        except FileNotFoundError as exc:
            # The 300 MB CSV is not in the repository. Say so once and stay quiet.
            self.unavailable = str(exc)
            log.note(f"  ⚠️ puzzle probe disabled: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - a diagnostic may never kill a run
            log.note(f"  ⚠️ puzzle probe failed at step {step}: "
                     f"{type(exc).__name__}: {exc}")
            return
        finally:
            if was_training:
                net.train()

        out = {"puzzles/solve_rate": r["solve_rate"],
               "puzzles/mean_line": r["mean_line"],
               "puzzles/seconds": time.time() - t0}
        for k in self.ks:
            out[f"puzzles/move_pass@{k}"] = r[f"move_pass@{k}"]
        for name, lo, hi in BUCKETS:
            sel = [b for b in r["bins"] if lo <= b["rating_lo"] < hi]
            n = sum(b["n"] for b in sel)
            t = sum(b["n_turns"] for b in sel)
            if not n:
                continue
            out[f"puzzles/solve_rate_{name}"] = sum(b["solved"] for b in sel) / n
            for k in self.ks:
                hits = sum(b[f"move_pass@{k}"] * b["n_turns"] for b in sel)
                out[f"puzzles/move_pass@{k}_{name}"] = hits / t if t else 0.0
        # ⚠️ The move-kind split is the diagnostic that names *what* is missing where
        # the solve rate says only *that* something is. `asked` is a property of the
        # puzzle set and constant across checkpoints, so it is noted once rather than
        # logged as a flat series; `played` and the pass rates are the network's.
        for name, d in r.get("kinds", {}).items():
            out[f"puzzles/played_{name}"] = d["played"]
            for k in self.ks:
                out[f"puzzles/pass@{k}_{name}"] = d[f"pass@{k}"]
        log.log(out, step=step)
        # The full 14-band curve and the kind table go to their own file: 56 wandb
        # series is a chart nobody reads, and losing the fine bins would mean
        # re-scoring every checkpoint to get them back.
        self._write_detail(step, r)
        k = r.get("kinds", {})
        log.note(f"  puzzles  solve {r['solve_rate']:.4f}  "
                 f"pass@1 {r['move_pass@1']:.4f}  pass@5 {r['move_pass@5']:.4f}  "
                 f"| capture {k.get('capture', {}).get('pass@1', 0):.3f} "
                 f"check {k.get('check', {}).get('pass@1', 0):.3f} "
                 f"mate {k.get('mate', {}).get('pass@1', 0):.3f}  "
                 f"[{out['puzzles/seconds']:.1f}s, not training time]")

    def _write_detail(self, step: int, r: dict, log_dir: str = "logs") -> None:
        """Per-band and per-kind detail, appended to its own JSONL."""
        if self.detail_path is None:
            return
        try:
            os.makedirs(os.path.dirname(self.detail_path) or ".", exist_ok=True)
            with open(self.detail_path, "a", buffering=1) as fh:
                fh.write(json.dumps({"step": step, "bins": r["bins"],
                                     "kinds": r.get("kinds", {})}, default=float) + "\n")
        except Exception:  # noqa: BLE001 - a detail file may never kill a run
            self.detail_path = None


def watch(run: str, checkpoint_dir: Optional[str] = None, interval: float = 120.0,
          limit: Optional[int] = 20_000, impl: Optional[str] = None,
          ks: Sequence[int] = (1, 3, 5), device: str = "cuda",
          use_wandb: bool = True, project: str = "brokefish",
          once: bool = False, log_dir: str = "logs") -> None:
    """Poll for new ``{run}-NNNNNN.pt`` snapshots and score each one exactly once."""
    from .league import discover_checkpoints
    from .puzzles import load_puzzles

    os.makedirs(log_dir, exist_ok=True)
    jsonl = open(os.path.join(log_dir, f"{run}-puzzles.jsonl"), "a", buffering=1)

    t0 = time.time()
    puzzles = load_puzzles(limit=limit, device=device)
    print(f"  {len(puzzles)} puzzles, mean line {float(puzzles.step_len.float().mean()):.2f} "
          f"solver moves, loaded in {time.time() - t0:.1f}s", flush=True)

    wb = None
    if use_wandb:
        try:
            import wandb
            wb = wandb.init(project=project, name=f"{run}-puzzles",
                            config={"watching": run, "n_puzzles": len(puzzles),
                                    "ks": list(ks)}, reinit=True)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            print(f"  ⚠️ wandb off, JSONL only: {type(exc).__name__}: {exc}", flush=True)

    seen: set = set()
    while True:
        found = discover_checkpoints(run, checkpoint_dir)
        todo = [(s, p) for s, p in found if s not in seen]
        for step, path in todo:
            t = time.time()
            try:
                rec = score_checkpoint(path, puzzles, impl=impl, ks=ks, device=device)
            except Exception as exc:  # noqa: BLE001
                # A checkpoint caught mid-write is the expected failure; it will be
                # picked up on the next poll rather than killing the watcher.
                print(f"  ⚠️ step {step}: {type(exc).__name__}: {exc}", flush=True)
                continue
            seen.add(step)
            bins = rec.pop("_bins")
            rec["step"] = step
            rec["seconds"] = time.time() - t
            jsonl.write(json.dumps({**rec, "bins": bins}, default=float) + "\n")
            if wb is not None:
                wb.log(rec, step=step)
            print(f"  step {step:6d}  solve {rec['puzzles/solve_rate']:.4f}  "
                  f"move@1 {rec['puzzles/move_pass@1']:.4f}  "
                  f"move@5 {rec['puzzles/move_pass@5']:.4f}  "
                  f"[{rec['seconds']:.1f}s]", flush=True)
        if once:
            break
        time.sleep(interval)

    if wb is not None:
        wb.finish()
    jsonl.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="score a run's checkpoints on Lichess puzzles, out of process")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoints", default=None,
                    help="runs/<run>/checkpoints by default")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=20_000, help="puzzles to load")
    ap.add_argument("--impl", default=None, help="fused encoder; None is the torch module")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--once", action="store_true", help="score what exists and exit")
    args = ap.parse_args(argv)

    print(f"watching {args.run} in {args.checkpoints}/  "
          f"-> {paths.artifact(args.run, args.run + '-puzzles.jsonl')}", flush=True)
    watch(args.run, checkpoint_dir=args.checkpoints, interval=args.interval,
          limit=args.limit, impl=args.impl, ks=args.ks, device=args.device,
          use_wandb=not args.no_wandb, once=args.once)


if __name__ == "__main__":
    main()
