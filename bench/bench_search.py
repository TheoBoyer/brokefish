"""Gate 1a: what the tree costs, measured inside the real search.

`bench/bench_loop.py` timed one node's work with no tree and got 62.9k/s, which
is a ceiling on the loop rate and was never Gate 1. This runs the whole of
`docs/mcts.md` §6 on device: `root_init`, `n` simulations of descent, encoder,
expansion and backup, and `select_and_advance`, over a real batch of games.

**The gate number is useful evaluations per second.** A simulation whose descent
ended on a stored terminal, or that created a terminal child, has nothing for the
network to say; §6.3 evaluates it anyway to keep the launch shape static and
discards the result. Those rows cost time and produce no training signal, so they
are excluded from the numerator and reported separately. `raw` counts every row
the encoder actually ran, which is the hardware's view and always `n * B` per move.

Three things are reported alongside, because the gate alone hides where the time
goes. `s/move` is what a self-play generation is priced in. The encoder-only rate
is the same batch through `forward_full` and nothing else, so `tree overhead` is
the honest cost of everything §6 adds. And the sweep over `n` says whether that
overhead is flat, which it should not quite be: a larger `n` means a deeper tree,
and the descent, the repetition scan and the backup all grow with depth.

Protocol. Clocks fall to 1.38-1.5 GHz under sustained load and drift by ±3 %, so
the encoder baseline and the full search are **interleaved with the order reversed
on alternate rounds**; a naive before/after on this card has already produced a
fake 4.5 % gain once. Only the averages at the end are quotable.

⚠️ The batch is built by random legal play, so it starts at a realistic ply rather
than at the start position, where trees are shallow and the tree cost is
understated. `--plies` controls that and is reported with every number.

Run from the repository root::

    python -m bench.bench_search                       # the gate: n=800, B=4096
    python -m bench.bench_search --sweep 32,128,800    # the n sweep
    python -m bench.bench_search --n 128 --batch 1024  # a quick pass
"""

from __future__ import annotations

import argparse
import time

import torch

from brokefish.env import cuda_impl as env
from brokefish.nn import available, encoder_impl, why_unavailable
from brokefish.nn.model import BrokefishNet
from brokefish.search import SearchConfig, search_impl


def build_batch(B: int, plies: int, seed: int):
    """`B` live positions from random legal play, terminals replaced.

    Invariant 8 forbids searching a finished position, and a batch built by random
    play contains some, so the terminal rows are refilled from the ones that are
    still going rather than dropped: `B` is a shape the encoder is measured at and
    changing it would change the number.
    """
    from tests.boards import random_positions

    boards, control, _ = random_positions(B, plies=plies, seed=seed, device="cuda")
    mask, in_check = env.movegen(boards, control)
    code, _ = env.terminal(mask, in_check, control, boards)
    live = (code == 0).nonzero(as_tuple=True)[0]
    if live.numel() == 0:
        raise SystemExit(f"every position at {plies} plies is terminal")
    fill = live[torch.arange(B, device="cuda") % live.numel()]
    return boards[fill].contiguous(), control[fill].contiguous(), int(live.numel())


def timed(fn) -> float:
    """One timed block. Warm-up is the caller's, once per phase rather than once
    per block: at n = 800 a block is nearly a minute and a per-block warm-up would
    double the campaign for nothing after the first."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def run_point(model, boards, control, n: int, B: int, rounds: int, seed: int,
              log) -> dict:
    """One `(n, B)` point: the search and the encoder baseline, interleaved."""
    cfg = SearchConfig(n=n, B=B)
    search = search_impl("cuda")(
        cfg, model.forward_full, env=env, seed=seed,
        # Both off on purpose. `check_invariants` costs a host synchronisation per
        # move and the counters cost a device atomic per simulation; the gate is
        # the search, not the instrumentation. The counters are turned back on for
        # one extra move at the end so the useful-evaluation count is available.
        check_invariants=False, collect_stats=False)
    search.reset(boards.clone(), control.clone())
    rep = torch.zeros(B, dtype=torch.uint8, device="cuda")

    def one_move():
        search.reset_finished()
        search.self_play_move()

    # `n + 1`, not `n`. A move is `root_init` plus `n` simulations and each of those
    # calls the encoder once, so a baseline of `n` charges the tree for one extra
    # evaluation: 3.1 % at n = 32 and 0.1 % at n = 800, which is most of the
    # apparent fall in overhead with `n` and none of it real.
    def encoder_only():
        for _ in range(n + 1):
            model.forward_full(boards, control, rep)

    phases = [("search", one_move), ("encoder", encoder_only)]
    totals = {name: 0.0 for name, _ in phases}
    for name, fn in phases:
        log(f"    warm-up {name}")
        fn()
    for r in range(rounds):
        for name, fn in (phases if r % 2 == 0 else phases[::-1]):
            ms = timed(fn)
            totals[name] += ms
            log(f"    round {r} {name:8s} {ms / 1e3:8.3f} s (provisional)")

    search_ms = totals["search"] / rounds
    encoder_ms = totals["encoder"] / rounds

    # One more move with the counters on, to find out how many of the `n * B`
    # encoder rows were leaves the network was actually asked about.
    search.collect_stats = True
    search.reset_counters()
    search.reset_finished()
    search.self_play_move()
    counters = search.device_counters()
    wasted = counters["terminal_descents"] + counters["terminal_children"]
    raw = float(n) * B
    useful = raw - wasted
    calls = n + 1

    return {
        "n": n, "B": B,
        "search_ms": search_ms, "encoder_ms": encoder_ms,
        "raw_per_s": raw / search_ms * 1e3,
        "useful_per_s": useful / search_ms * 1e3,
        "encoder_per_s": calls * B / encoder_ms * 1e3,
        "overhead_ms": (search_ms - encoder_ms) / calls,
        "moves_per_s": 1e3 / search_ms,
        "overhead": search_ms / encoder_ms - 1.0,
        "useful_frac": useful / raw,
        "max_depth": counters["max_depth"],
        "mean_depth": counters["mean_depth"],
        "max_edges": counters["max_edges"],
        "truncated": counters["truncated_nodes"],
        "truncated_mass": counters["truncated_mass"],
        "max_nodes": counters["max_nodes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800, help="simulations per move")
    parser.add_argument("--batch", type=int, default=4096, help="games in flight")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--plies", type=int, default=30,
                        help="random legal plies used to build the batch")
    parser.add_argument("--sweep", type=str, default="",
                        help="comma-separated values of n to sweep instead of --n")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    impls = available()
    if "cuda" not in impls:
        raise SystemExit(f"the fused CUDA encoder is required: {why_unavailable()}")

    handle = open(args.log, "w", buffering=1) if args.log else None

    def log(line: str = "") -> None:
        print(line, flush=True)
        if handle:
            handle.write(line + "\n")

    torch.manual_seed(args.seed)
    net = BrokefishNet().cuda().half().eval()
    model = encoder_impl("cuda")(net)
    boards, control, n_live = build_batch(args.batch, args.plies, args.seed)

    free, total = torch.cuda.mem_get_info()
    log(f"brokefish Gate 1a: the whole of docs/mcts.md §6 on device")
    log(f"  B = {args.batch} games at {args.plies} random plies "
        f"({n_live} distinct live positions), {args.rounds} interleaved rounds")
    log(f"  {torch.cuda.get_device_name(0)}, {(total - free) / 2**30:.2f} of "
        f"{total / 2**30:.2f} GiB in use before the tree")

    values = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.n]
    rows = []
    for n in values:
        tree_bytes = 856 * (n + 1) * args.batch
        log(f"\n  n = {n}  (tree {tree_bytes / 2**30:.2f} GiB, "
            f"{4 * n + 2} launches per move)")
        rows.append(run_point(model, boards, control, n, args.batch, args.rounds,
                              args.seed, log))
        torch.cuda.empty_cache()

    log("\n" + "=" * 78)
    log(f"{'n':>5} {'useful/s':>10} {'raw/s':>10} {'encoder/s':>10} {'tree':>7} "
        f"{'tree ms':>8} {'s/move':>8} {'useful':>7} {'depth':>11}")
    for r in rows:
        log(f"{r['n']:5d} {r['useful_per_s']:10.0f} {r['raw_per_s']:10.0f} "
            f"{r['encoder_per_s']:10.0f} {r['overhead'] * 100:6.1f}% "
            f"{r['overhead_ms']:8.2f} {1 / r['moves_per_s']:8.2f} "
            f"{r['useful_frac'] * 100:6.1f}% "
            f"{r['mean_depth']:5.1f}/{r['max_depth']:<5.0f}")
    log("=" * 78)
    log("  useful/s gates. raw/s counts every row the encoder ran, including the")
    log("  terminal leaves §6.3 evaluates and discards; encoder/s is the same")
    log("  n+1 calls through forward_full alone, so 'tree' is what all of §6 adds")
    log("  on top of the evaluations a move has to run anyway, in per-simulation ms.")

    for r in rows:
        log(f"\n  n = {r['n']}: max edges {r['max_edges']:.0f} (E = 64), "
            f"{r['truncated']:.0f} nodes truncated dropping {r['truncated_mass']:.4g} "
            f"of prior mass, pool high water {r['max_nodes']:.0f} of {r['n'] + 1}")

    gate = max(r["useful_per_s"] for r in rows)
    log(f"\n  Gate 1: 45-50k evals/s is GO, under 15k is NO-GO. Best here: "
        f"{gate:.0f}/s at n = {max(rows, key=lambda r: r['useful_per_s'])['n']}.")
    if handle:
        handle.close()


if __name__ == "__main__":
    main()
