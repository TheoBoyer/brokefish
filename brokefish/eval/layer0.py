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
                  search_impl: str = "cuda",
                  ) -> dict:
    """Every layer-0 number for one checkpoint, as a plain dict.

    ``suites`` may be passed in already loaded, which is how a training loop
    should use it: the harvest is minutes and the suites never change, so paying
    for it once per run rather than once per checkpoint is the difference between
    layer 0 costing seconds and costing minutes.

    ⚠️ **``impl`` is the encoder and ``search_impl`` is the search**, and they are
    two different axes. ``impl`` picks the fused evaluator; ``search_impl`` picks
    the tree. Until 2026-07-31 this function had no ``search_impl`` at all, so every
    layer-0 run took ``runner.make_search``'s default of ``"torch"`` and drove the
    *reference* search — with no flag to change it. The reference exists to be the
    oracle the kernel is checked against (`tests/test_search_cuda.py` holds the two
    together tree for tree), not to be the thing a measurement runs on. The default
    is now ``"cuda"``; pass ``"torch"`` deliberately when the point is the oracle.
    """
    t0 = time.time()
    if suites is None:
        suites = load_suites(suite_path or DEFAULT_SUITE_PATH, device=device)

    run = self_play_run(net, games=games, n_sims=n_sims, max_plies=max_plies,
                        impl=impl, seed=seed, device=device, search_impl=search_impl)

    suite_scores, policy_scores = {}, {}
    for name, suite in suites.items():
        suite_scores[name] = score_suite(suite, net, n=suite_sims, impl=impl,
                                         seed=seed, device=device,
                                         search_impl=search_impl)
        policy_scores[name] = score_suite_policy(suite, net, impl=impl, device=device)

    calib = value_calibration(run)
    entropy = policy_entropy(run)
    stats = game_statistics(run)

    diverged, reasons = _liveness(calib, entropy)
    return {
        "step": step,
        "seconds": time.time() - t0,
        "config": {"games": games, "n_sims": n_sims, "suite_sims": suite_sims,
                   "impl": impl, "search_impl": search_impl, "seed": seed},
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


def load_net_state(path: str, device: str = "cuda") -> dict:
    """The network's weights, from either a bare ``state_dict`` or a C2 checkpoint.

    ⚠️ **A training checkpoint is not a ``state_dict``.** ``train.md`` §9 stores the
    optimiser, the RNG states, the config and the in-flight game state alongside the
    weights, so handing one straight to ``load_state_dict`` fails on the extra keys —
    which made this CLI unable to evaluate the only artifact the training loop
    produces. Added 2026-07-31.

    ``weights_only=False`` is required for that form because §9's payload holds a
    config dataclass, so this must only ever be pointed at a file this repository
    wrote. A bare ``state_dict`` is still loaded the safe way.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "net" in blob and isinstance(blob["net"], dict):
        return blob["net"]
    return blob


def main() -> None:
    import argparse

    from brokefish.nn.model import BrokefishNet

    ap = argparse.ArgumentParser(description="layer 0 for one checkpoint")
    ap.add_argument("checkpoint", nargs="?", help="a torch state_dict; random init if absent")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--suite-sims", type=int, default=128)
    ap.add_argument("--suites", default=DEFAULT_SUITE_PATH)
    ap.add_argument("--impl", default=None,
                    help="the fused encoder: cuda or triton. NOT the search")
    ap.add_argument("--search-impl", default="cuda", choices=("cuda", "torch"),
                    help="the tree. cuda is the kernel; torch is the reference oracle "
                         "and is roughly an order of magnitude slower")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None, help="append the record to this file")
    args = ap.parse_args()

    net = BrokefishNet().to(args.device)
    if args.checkpoint:
        net.load_state_dict(load_net_state(args.checkpoint, args.device))
    net.eval()

    report = layer0_report(net, games=args.games, n_sims=args.sims,
                           suite_sims=args.suite_sims, impl=args.impl,
                           device=args.device, suite_path=args.suites,
                           search_impl=args.search_impl)
    line = json.dumps(report)
    print(line)
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
