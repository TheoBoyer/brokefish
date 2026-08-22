"""Live comparison of a running head-to-head against a previous one, paired by opening.

    python -m scripts.h2h_compare --now logs/gate2-h2h-muon9int8.pgn \
                                  --ref logs/gate2-h2h.pgn

⚠️ **Two comparisons, and only one of them is fair.** A partial score is a *biased*
estimate: short games finish first, decisive games are short, and draws take longest,
so any prefix underestimates the final score and climbs monotonically toward it. In the
2026-08-13 match `t12h-gumbel` lost its first six games and finished at 0.4000.

  * `same N` compares this run's score after N finished games with the reference's
    score after *its* first N. Both carry the same bias, so the difference is readable
    even though neither level is.
  * `paired` matches games by (opening FEN, colour) and scores only the openings both
    runs have completed. That removes the opening draw entirely and is the number to
    trust -- at the cost of being defined on fewer games.
"""
from __future__ import annotations

import argparse
import re


def parse(path):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return []
    out = []
    for g in txt.split("[Event ")[1:]:
        def f(k, d=None):
            m = re.search(r'\[' + k + r' "(.*?)"\]', g)
            return m.group(1) if m else d
        w = (f("White") or "")
        # A is whichever side is not alphagateau; the arbiter writes the real names.
        a_white = not w.startswith("alphagateau")
        r = f("Result", "*")
        s = 0.5 if r == "1/2-1/2" else (1.0 if (r == "1-0") == a_white else 0.0)
        out.append({"fen": f("FEN"), "a_white": a_white, "score": s,
                    "plies": int(f("PlyCount", "0")), "result": r})
    return out


def tally(games):
    n = len(games)
    if not n:
        return 0, 0.0, 0, 0, 0
    s = sum(g["score"] for g in games)
    w = sum(1 for g in games if g["score"] == 1)
    d = sum(1 for g in games if g["score"] == 0.5)
    return n, s / n, w, d, n - w - d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--now", required=True)
    p.add_argument("--ref", default="logs/gate2-h2h.pgn")
    a = p.parse_args()

    cur, ref = parse(a.now), parse(a.ref)
    n, sc, w, d, l = tally(cur)
    rn, rs, rw, rd, rl = tally(ref[:n])
    fn, fs, fw, fd, fl = tally(ref)

    print(f"  this run    {n:3d} done   {w}-{d}-{l}   score {sc:.4f}"
          f"   mean {sum(g['plies'] for g in cur) / max(n, 1):.1f} plies")
    print(f"  ref @same N {rn:3d}        {rw}-{rd}-{rl}   score {rs:.4f}"
          f"   -> delta {sc - rs:+.4f}")
    print(f"  ref FINAL   {fn:3d}        {fw}-{fd}-{fl}   score {fs:.4f}"
          f"   mean {sum(g['plies'] for g in ref) / max(fn, 1):.1f} plies")

    key = lambda g: (g["fen"], g["a_white"])
    rmap = {key(g): g for g in ref}
    both = [(g, rmap[key(g)]) for g in cur if key(g) in rmap]
    if both:
        cs = sum(x["score"] for x, _ in both) / len(both)
        rr = sum(y["score"] for _, y in both) / len(both)
        agree = sum(1 for x, y in both if x["score"] == y["score"])
        print(f"  paired      {len(both):3d} openings both finished:"
              f"  this {cs:.4f}  vs ref {rr:.4f}   delta {cs - rr:+.4f}"
              f"   ({agree} same outcome)")


if __name__ == "__main__":
    main()
