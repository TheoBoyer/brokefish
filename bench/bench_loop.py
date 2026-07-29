"""What the environment charges the network, measured rather than argued.

The two halves of the system now exist and have been timed separately: the
encoder at ~62k evals/s (`bench/bench_model.py`) and the movegen at 5.01M
positions/s (`csrc/tests/tperft.cu`). Those were measured on different days at
different batch sizes, so composing them arithmetically is a guess. This script
runs them against each other on the same boards in the same process.

⚠️ **This is not Gate 1.** Gate 1 is 45-50k evals/s sustained inside a synthetic
MCTS loop, and there is no tree here: no descent, no backup, no selection, no
per-node memory traffic. What is here is one node's worth of work, which is the
*floor* on the loop's cost and an upper bound on its speed. C1 produces the real
number, and it will be lower.

Protocol, and it is not optional on this card: clocks fall to 1.38-1.5 GHz under
load and drift by ±3 %, which is larger than the effect being measured. Phases are
therefore run interleaved with the order reversed on alternate rounds, so the
position penalty cancels instead of being read as a cost. Only the averages at
the end are quotable.

Run from the repository root::

    python -m bench.bench_loop [--boards 16384] [--rounds 5]
"""

import argparse
import time

import torch

from brokefish.env import cuda_impl as env
from brokefish.nn import available, encoder_impl, why_unavailable
from brokefish.nn.model import BrokefishNet


def time_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--plies", type=int, default=12,
                        help="depth of random legal play used to build the batch")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    impls = available()
    if not impls:
        raise SystemExit(f"no encoder implementation available: {why_unavailable()}")
    name = "cuda" if "cuda" in impls else impls[0]

    from tests.boards import random_positions

    torch.manual_seed(0)
    net = BrokefishNet().cuda().half().eval()
    model = encoder_impl(name)(net)
    boards, control, rep = random_positions(args.boards, plies=args.plies, seed=0)
    n = boards.shape[0]

    # One legal move per position, chosen once. Picking it inside the timed block
    # would measure torch indexing, which is not part of either half.
    mask, _ = env.movegen(boards, control)
    flat = env.bitset_to_bool(mask).reshape(n, 2048)
    any_legal = flat.any(-1)
    move = torch.where(any_legal, flat.to(torch.uint8).argmax(-1),
                       torch.full_like(any_legal, -1, dtype=torch.long)).to(torch.int16)
    hash0 = env.hash_position(boards, control)
    print(f"{n} positions at {args.plies} plies, {int(any_legal.sum())} with a legal move, "
          f"encoder impl '{name}'")

    def evaluate():
        return model.forward_full(boards, control, rep)

    def legality():
        return env.movegen(boards, control)

    def advance():
        return env.step(boards, control, move, hash=hash0)

    def advance_no_hash():
        return env.step(boards, control, move)

    def evaluate_and_legality():
        evaluate()
        legality()

    def whole_node():
        evaluate()
        legality()
        advance()

    # The environment phases are listed on their own as well as fused, because the
    # interesting quantity is a difference and a difference of two averages is
    # only trustworthy if both were measured under the same protocol.
    phases = [
        ("evaluate", evaluate),
        ("legality (movegen)", legality),
        ("advance (step + hash)", advance),
        ("advance, no hash", advance_no_hash),
        ("evaluate + legality", evaluate_and_legality),
        ("whole node", whole_node),
    ]

    totals = {label: 0.0 for label, _ in phases}
    n_calls = args.rounds * len(phases)
    done = 0
    t0 = time.perf_counter()
    print(f"\n{n_calls} timed blocks ({args.rounds} rounds x {len(phases)} phases).")
    with torch.no_grad():
        for r in range(args.rounds):
            order = phases if r % 2 == 0 else phases[::-1]
            for label, fn in order:
                ms = time_ms(fn)
                totals[label] += ms
                done += 1
                elapsed = time.perf_counter() - t0
                print(f"  [{done:2d}/{n_calls}] round {r} {label:24s} {ms:8.3f} ms "
                      f"(provisional)   elapsed {elapsed:5.0f}s  "
                      f"eta {elapsed / done * (n_calls - done):5.0f}s", flush=True)

    avg = {label: totals[label] / args.rounds for label, _ in phases}
    print(f"\n{n} boards, fp16, {args.rounds} interleaved rounds:")
    for label, _ in phases:
        ms = avg[label]
        print(f"  {label:24s} {ms:8.3f} ms   {n / ms:8.1f}k/s")

    # The two numbers this script exists for.
    overhead_legality = avg["evaluate + legality"] / avg["evaluate"] - 1
    overhead_node = avg["whole node"] / avg["evaluate"] - 1
    print(f"\n  environment cost over a bare evaluation:")
    print(f"    legality only        +{overhead_legality * 100:5.1f}%")
    print(f"    legality and advance +{overhead_node * 100:5.1f}%")
    print(f"  sustained node rate    {n / avg['whole node']:8.1f}k/s")
    print("\n  Gate 1 wants 45-50k evals/s inside a real MCTS loop. This is one")
    print("  node's work with no tree, so it is a ceiling on that number, not it.")


if __name__ == "__main__":
    main()
