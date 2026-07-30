"""What layer 0 costs, phase by phase, on both search implementations.

`evals.md` §4 predicted "~6 s per checkpoint at the loop rate C1 measured". That
prediction assumed the fused encoder and the CUDA search; the default arguments to
`layer0_report` are the torch reference and the torch oracle model, which is three
orders of magnitude away from it. This benchmark measures both, so the claim in
the document is a number and not an aspiration, and so the gap between them is
attributed rather than averaged away.

    .venv/bin/python -m bench.bench_eval --games 64 --sims 100 | tee logs/d1_layer0.log

⚠️ Not an interleaved A/B. The two paths differ by orders of magnitude, not by
percent, so the thermal-drift protocol that `bench_model.py` needs would only add
noise to a comparison that does not need it. Do not copy this file's structure
into a kernel benchmark.
"""

from __future__ import annotations

import argparse
import time

import torch

from brokefish.eval.metrics import (game_statistics, policy_entropy, self_play_run,
                                    value_calibration)
from brokefish.eval.suites import (DEFAULT_SUITE_PATH, load_suites, score_suite,
                                   score_suite_policy)
from brokefish.nn.model import BrokefishNet


def phase(name: str, fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  {name:<34s} {dt:8.2f} s", flush=True)
    return out, dt


def run(net, suites, games: int, sims: int, suite_sims: int, impl, search_impl,
        max_plies: int, device: str, batch=None) -> float:
    label = (f"encoder={impl or 'torch oracle'}  search={search_impl}"
             f"  games={games} batch={batch or games}")
    print(f"\n{label}", flush=True)
    total = 0.0

    run_, dt = phase("self-play games", lambda: self_play_run(
        net, games=games, n_sims=sims, max_plies=max_plies, batch=batch, impl=impl,
        search_impl=search_impl, device=device))
    total += dt
    _c, dt = phase("value calibration", lambda: value_calibration(run_)); total += dt
    _e, dt = phase("policy entropy", lambda: policy_entropy(run_)); total += dt
    _g, dt = phase("game statistics", lambda: game_statistics(run_)); total += dt

    for name, suite in suites.items():
        _s, dt = phase(f"suite {name} (search)", lambda s=suite: score_suite(
            s, net, n=suite_sims, impl=impl, search_impl=search_impl, device=device))
        total += dt
    for name, suite in suites.items():
        _s, dt = phase(f"suite {name} (policy)", lambda s=suite: score_suite_policy(
            s, net, impl=impl, device=device))
        total += dt

    print(f"  {'TOTAL':<34s} {total:8.2f} s"
          f"   ({run_.meta['games']} games, {int(run_.value_pred.numel())} scored plies,"
          f" {run_.n_abandoned} abandoned, {run_.n_in_flight} in flight)", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--suite-sims", type=int, default=128)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--suites", default=DEFAULT_SUITE_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", choices=["fast", "reference", "both"], default="both")
    args = ap.parse_args()

    torch.manual_seed(0)
    net = BrokefishNet().to(args.device).eval()
    suites = load_suites(args.suites, device=args.device)
    print(f"suites: " + ", ".join(f"{k}={len(v)}" for k, v in suites.items()))

    paths = []
    if args.only in ("fast", "both"):
        paths.append(("cuda", "cuda"))
    if args.only in ("reference", "both"):
        paths.append((None, "torch"))
    for impl, search_impl in paths:
        run(net, suites, args.games, args.sims, args.suite_sims, impl, search_impl,
            args.max_plies, args.device, batch=args.batch)


if __name__ == "__main__":
    main()
