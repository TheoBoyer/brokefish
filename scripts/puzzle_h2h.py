"""Score checkpoints on the puzzle suite **from disk**, and score an HTTP engine too.

    python -m scripts.puzzle_h2h --ckpt a.pt b.pt --limit 20000            # ours
    python -m scripts.puzzle_h2h --ckpt a.pt --engine http://127.0.0.1:8083 --limit 500

⚠️ **From disk, on purpose.** The in-run probe scores `self.net` in process; the league
and the match harness both load the `.pt`. If a saved checkpoint were stale or partial
you would see exactly the discrepancy this script exists to test -- excellent logged
puzzle numbers, ordinary match results. Comparing the two is the check.

⚠️ **`--sims` is the whole comparison.** `n = 0` is the raw policy argmax, which is what
the training probe reports and what no game is ever decided by. Anything else runs the
search; to compare against a foreign engine over HTTP the budget must match on both
sides, and so must `gumbel_m` -- our training used `--gumbel-m 16 --sims 128`, and
`mctx.gumbel_muzero_policy` defaults `max_num_considered_actions = 16`, so at n = 128
with m = 16 the two searches consider the same number of root moves.

⚠️ Evaluation output (`evaluation.md`): it may never flow backwards into training or
checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import torch


def ask_engine(url: str, fens, n: int, timeout: float = 3600.0, k: int = 1):
    body = json.dumps({"fens": list(fens), "n": int(n), "k": int(k)}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/move", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read())["moves"]


def score_engine(url: str, limit: int, n: int, device: str, chunk: int,
                 ks=(1, 3, 5), bin_width: int = 200):
    """Every puzzle metric for a foreign engine, over the same suite and the same bins.

    ⚠️ **The aggregate is not enough.** A single `move_pass@1` cannot separate "better
    at easy tactics" from "better at hard ones", and it hides `solve_rate`, which is the
    stricter question -- every turn of the line correct, not just the first. Both are
    recorded here, per rating bin, in the same shape `eval/puzzles.py` produces for our
    own nets, so the two sides can be compared bin for bin.

    `pass@k` for k > 1 needs the engine's top-k, which the one-move contract does not
    carry; `--k` asks for it and an engine that ignores the field simply reports
    `pass@1` for every k, which is visible in the output rather than silent.
    """
    import chess

    from brokefish.env import notation
    from brokefish.eval.puzzles import load_puzzles

    ps = load_puzzles(limit=limit, device=device)
    N = len(ps)
    S = int(ps.step_len.max()) if ps.step_len is not None else 1
    kmax = max(ks)
    # [puzzle][turn] -> rank of the right move in the engine's list, or -1
    rank = -torch.ones((N, S), dtype=torch.int32)
    t0 = time.perf_counter()
    for s_ in range(S):
        live = (ps.step_len > s_).nonzero(as_tuple=True)[0]
        if live.numel() == 0:
            break
        b = ps.step_boards[live, s_]
        c = ps.step_control[live, s_]
        lab = ps.step_answer[live, s_].to(torch.int64)
        want = notation.to_uci(b, lab & 0x7FF, (lab >> 11) & 0b11)
        fens = [notation.to_fen(b[i:i + 1], c[i:i + 1])[0] for i in range(b.shape[0])]
        got = []
        for lo in range(0, len(fens), chunk):
            got += ask_engine(url, fens[lo:lo + chunk], n, k=kmax)
            print(f"    turn {s_}: {min(lo + chunk, len(fens))}/{len(fens)}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        for idx, g, w in zip(live.tolist(), got, want):
            lst = g if isinstance(g, list) else [g]
            rank[idx, s_] = lst.index(w) if w in lst else -1

    solved_turn = (rank >= 0) & (rank < 1)
    valid = torch.arange(S)[None, :] < ps.step_len[:, None].cpu()
    out = {"n_puzzles": N, "sims": n, "n_turns": int(valid.sum())}
    for k in ks:
        hit = (rank >= 0) & (rank < k) & valid
        out[f"move_pass@{k}"] = float(hit.sum()) / max(int(valid.sum()), 1)
    # ⚠️ solve_rate is every solver turn of the line correct, not the first one.
    line_ok = ((solved_turn | ~valid).all(dim=1))
    out["solve_rate"] = float(line_ok.float().mean())
    out["mean_line"] = float(ps.step_len.float().mean())

    rating = ps.rating.cpu()
    bins = []
    lo = int(rating.min()) // bin_width * bin_width
    while lo <= int(rating.max()):
        m = (rating >= lo) & (rating < lo + bin_width)
        if int(m.sum()):
            v = valid[m]
            row = {"rating_lo": lo, "rating_hi": lo + bin_width,
                   "n": int(m.sum()), "n_turns": int(v.sum()),
                   "solve_rate": float(line_ok[m].float().mean()),
                   "mean_line": float(ps.step_len[m].float().mean())}
            for k in ks:
                h = (rank[m] >= 0) & (rank[m] < k) & v
                row[f"move_pass@{k}"] = float(h.sum()) / max(int(v.sum()), 1)
            bins.append(row)
        lo += bin_width
    out["bins"] = bins
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", nargs="*", default=[])
    p.add_argument("--engine", default=None, help="base URL of an HTTP engine")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--sims", type=int, default=0, help="0 = raw policy, as the probe")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from brokefish.eval.puzzles import load_puzzles, score_puzzles, score_puzzles_line
    from brokefish.eval.layer0 import load_net_state
    from brokefish.nn.model import net_for_state

    out = {}
    ps = None
    for path in a.ckpt:
        if ps is None:
            ps = load_puzzles(limit=a.limit, device=a.device)
        state = load_net_state(path, device="cpu")
        net = net_for_state(state)
        net.load_state_dict(state)
        net = net.to(a.device).eval()
        t = time.perf_counter()
        if a.sims == 0:
            r = score_puzzles_line(ps, net, impl="cuda", device=a.device)
            key = "move_pass@1"
        else:
            r = score_puzzles(ps, net, n=a.sims, impl="cuda", search_impl="cuda",
                              device=a.device)
            key = "solve_rate"
        out[path] = r
        bits = "  ".join(f"{m} {r[m]:.4f}" for m in
                         ("move_pass@1", "move_pass@3", "move_pass@5", "solve_rate")
                         if m in r)
        print(f"  {path}\n     {bits or f'{key} {r[key]:.4f}'}   "
              f"({time.perf_counter() - t:.0f}s, sims={a.sims})", flush=True)
        del net
        torch.cuda.empty_cache()

    if a.engine:
        print(f"  engine {a.engine} at n={a.sims}, {a.limit} puzzles", flush=True)
        r = score_engine(a.engine, a.limit, a.sims, a.device, a.chunk)
        out[a.engine] = r
        print(f"     move_pass@1 {r['move_pass@1']:.4f}  @3 {r['move_pass@3']:.4f}  "
              f"@5 {r['move_pass@5']:.4f}  solve {r['solve_rate']:.4f}  "
              f"mean_line {r['mean_line']:.2f}  ({r['n_turns']} turns)")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2, default=str)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
