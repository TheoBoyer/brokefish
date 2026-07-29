"""Throughput of the PyTorch environment, in boards per second.

Not a target: the CUDA movegen is the one with a bar to clear (50-100M boards/s,
`csrc/README.md`). This exists so that the cost of the reference implementation
is a number rather than a guess, and so that A1 has something to be compared to.

    .venv/bin/python -m bench.bench_env [--device cuda] [--positions 4096]

Memory is what caps the batch size: the second order is brute force, so a batch
of N positions expands into roughly 35N boards, and the intermediates of
`first_order_mask` are `[35N, 32] int64` each.
"""

import argparse
import time

import torch

from brokefish.env import initial_boards, movegen, step
from brokefish.env.torch_impl import bitset_to_bool

BATCHES = [64, 256, 1024, 4096]


def make_batch(n: int, device: str, plies: int = 12, seed: int = 0):
    """A batch of distinct mid-game positions, from random legal play."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    boards, control = initial_boards(n, device=device)
    for _ in range(plies):
        legal = bitset_to_bool(movegen(boards, control)[0]).view(n, 2048).float().cpu()
        alive = legal.sum(-1) > 0
        if not alive.all():
            legal = legal[alive]
            boards, control, n = boards[alive], control[alive], int(alive.sum())
        move = torch.multinomial(legal, 1, generator=g).squeeze(-1).to(boards.device)
        boards, control, _, _ = step(boards, control, move)
    return boards, control


def timeit(fn, warmup: int = 2, repeats: int = 5, device: str = "cpu") -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    print(f"device: {args.device}")
    print(f"{'positions':>10} {'movegen':>12} {'step':>12} {'boards/s':>14}")
    for n in BATCHES:
        boards, control = make_batch(n, args.device)
        n = boards.shape[0]
        mask, _ = movegen(boards, control)
        move = (mask != 0).float().argmax(-1) * 64  # any slot with a move; target unused
        legal = bitset_to_bool(mask)
        b_idx, p_idx, s_idx = legal.nonzero(as_tuple=True)
        first = torch.zeros(n, dtype=torch.long, device=boards.device)
        first.scatter_reduce_(0, b_idx, p_idx * 64 + s_idx, reduce="amax")

        t_movegen = timeit(lambda: movegen(boards, control), repeats=args.repeats, device=args.device)
        t_step = timeit(lambda: step(boards, control, first), repeats=args.repeats, device=args.device)
        print(f"{n:>10} {t_movegen * 1e3:>10.2f}ms {t_step * 1e3:>10.2f}ms "
              f"{n / t_movegen:>14,.0f}")


if __name__ == "__main__":
    main()
