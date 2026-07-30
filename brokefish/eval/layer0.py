"""Layer 0 assembled: one call per checkpoint, one JSON record out.

`evals.md` §4. Everything here is self-contained — no second process, no opening
book, no external data — which is what lets it run on every checkpoint. It
alarms; it never blocks.

⚠️ **The return value may not select anything** (§2's third prohibition). It is a
record of what a checkpoint was, not an input to deciding whether to keep it.
`diverged` is the one exception the boundary allows: a liveness flag tripped by
NaNs, which are a fact about arithmetic rather than an opinion about chess.

Two things `evals.md` §4 lists are deliberately **not here**. The ~50 games against
the frozen anchor need two nets in one game, which is the league machinery of D2;
layer 0 gains that row when D2 lands. And `max |post-scale attention logit|` is an
fp16 overflow watch on the kernels rather than a measurement of a net — it belongs
to the training loop, `docs/train.md` §11, and was dropped from here on 2026-07-30.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Optional

import torch

from .metrics import (game_statistics, policy_entropy, self_play_run,
                      value_calibration)
from .suites import (DEFAULT_SUITE_PATH, Suite, load_suites, score_suite,
                     score_suite_policy)

@torch.no_grad()
def layer0_report(net, suites: Optional[Dict[str, Suite]] = None,
                  games: int = 64, n_sims: int = 100, max_plies: int = 300,
                  suite_sims: int = 128, impl: Optional[str] = None,
                  seed: int = 0, device: str = "cuda",
                  step: Optional[int] = None, suite_path: Optional[str] = None,
                  ) -> dict:
    """Every layer-0 number for one checkpoint, as a plain dict.

    ``suites`` may be passed in already loaded, which is how a training loop
    should use it: the harvest is minutes and the suites never change, so paying
    for it once per run rather than once per checkpoint is the difference between
    layer 0 costing seconds and costing minutes.
    """
    t0 = time.time()
    if suites is None:
        suites = load_suites(suite_path or DEFAULT_SUITE_PATH, device=device)

    run = self_play_run(net, games=games, n_sims=n_sims, max_plies=max_plies,
                        impl=impl, seed=seed, device=device)

    suite_scores, policy_scores = {}, {}
    for name, suite in suites.items():
        suite_scores[name] = score_suite(suite, net, n=suite_sims, impl=impl,
                                         seed=seed, device=device)
        policy_scores[name] = score_suite_policy(suite, net, impl=impl, device=device)

    calib = value_calibration(run)
    entropy = policy_entropy(run)
    stats = game_statistics(run)

    diverged, reasons = _liveness(calib, entropy)
    return {
        "step": step,
        "seconds": time.time() - t0,
        "config": {"games": games, "n_sims": n_sims, "suite_sims": suite_sims,
                   "impl": impl, "seed": seed},
        "value_calibration": calib,
        "policy_entropy": entropy,
        "games": stats,
        "suites": suite_scores,
        "suites_policy_only": policy_scores,
        "diverged": diverged,
        "diverged_reasons": reasons,
    }


def _liveness(calib: dict, entropy: dict):
    """The only evaluation output allowed to act, and only by aborting (§2).

    Every trigger is a fact about arithmetic. "The value head is badly
    calibrated" is not here and must not be: a young net is badly calibrated and
    that is what training is for.
    """
    reasons = []
    for name, value in (("brier", calib.get("brier")),
                        ("mean_pred", calib.get("mean_pred")),
                        ("entropy_nats", entropy.get("entropy_nats"))):
        if value is not None and value != value:      # NaN
            reasons.append(f"{name} is NaN")
    return bool(reasons), reasons


def main() -> None:
    import argparse

    from brokefish.nn.model import BrokefishNet

    ap = argparse.ArgumentParser(description="layer 0 for one checkpoint")
    ap.add_argument("checkpoint", nargs="?", help="a torch state_dict; random init if absent")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--suite-sims", type=int, default=128)
    ap.add_argument("--suites", default=DEFAULT_SUITE_PATH)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None, help="append the record to this file")
    args = ap.parse_args()

    net = BrokefishNet().to(args.device)
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    net.eval()

    report = layer0_report(net, games=args.games, n_sims=args.sims,
                           suite_sims=args.suite_sims, impl=args.impl,
                           device=args.device, suite_path=args.suites)
    line = json.dumps(report)
    print(line)
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
