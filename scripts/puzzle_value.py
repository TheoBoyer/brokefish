"""Score the puzzle suite through the **value head**, as a one-ply search.

    python -m scripts.puzzle_value --ckpt a.pt b.pt --limit 20000

The policy probe (`eval/puzzles.py`, `n_sims = 0`) grades one head and one head only.
This grades the other: every legal move is played, each resulting position is evaluated,
and the move chosen is the one minimising the **opponent's** value -- `argmax -v(child)`.
The value head is being used exactly as logits over moves.

⚠️ **Why it matters.** Measured 2026-08-17: `t12h-muon9-int8-005010` beats
`t12h-gumbel-004009` by +0.061 pass@1 on the policy probe and *loses to it* 62-45-93 at
n = 128. On shared outcome-labelled positions its value head scores 0.615 sign accuracy
against 0.783. Games at n >= 64 are decided by the value head -- Gumbel proposes 16
candidates from the prior and then ranks them by completed-Q -- so a policy-only probe
cannot see the thing that decides the game. This closes that gap.

⚠️ A terminal child is worth +1 to the mover by the rules, not whatever the network says
about it, so mates are scored from the environment rather than from the head. Without
that the metric would punish a net for a position it is never asked to evaluate.

⚠️ Evaluation output (`evaluation.md`): it may never flow backwards into training or
checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


# ⚠️ The implementation lives in `brokefish/eval/puzzles.py` and is imported, not copied.
# It moved there on 2026-08-19 so the training loop's `PuzzleProbe` could run the same
# metric in-loop; a second copy here would be the thing that drifts.
from brokefish.eval.puzzles import score_puzzles_value, value_pick  # noqa: E402,F401


def score_ckpt(path, ps, device, bin_width=200, impl="cuda"):
    """One checkpoint, through `brokefish.eval.puzzles.score_puzzles_value`.

    ⚠️ `impl="cuda"` is the default here as it is in the probe: measured 2026-08-19 on
    the full 20 000 puzzles, the fused encoder is **21.5 s against 234.8 s** and agrees
    with the fp32 master weights to **0.0001**, which is twenty times inside the binomial
    standard error. Pass `--impl none` to score on the master weights instead.
    """
    from brokefish.eval.layer0 import load_net_state
    from brokefish.nn.model import net_for_state

    state = load_net_state(path, device="cpu")
    net = net_for_state(state)
    net.load_state_dict(state)
    net = net.to(device).eval()
    try:
        r = score_puzzles_value(ps, net, impl=impl, bin_width=bin_width, device=device)
    finally:
        del net
        torch.cuda.empty_cache()
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--impl", default="cuda",
                   help="fused encoder; 'none' scores on the fp32 master weights")
    a = p.parse_args()

    from brokefish.eval.puzzles import load_puzzles
    ps = load_puzzles(limit=a.limit, device=a.device)
    print(f"  {len(ps)} puzzles, mean line {float(ps.step_len.float().mean()):.2f}\n")

    res = {}
    for path in a.ckpt:
        t = time.perf_counter()
        r = score_ckpt(path, ps, a.device,
                       impl=None if a.impl == 'none' else a.impl)
        res[path] = r
        print(f"  {path}\n     value_pass@1 {r['value_pass@1']:.4f}   "
              f"solve_rate {r['solve_rate']:.4f}   ({time.perf_counter() - t:.0f}s)\n",
              flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
