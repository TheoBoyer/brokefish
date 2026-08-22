"""Where the live run stands against every comparable run, at matched wall clock.

⚠️ **Matched wall clock, not matched step.** Runs differ in kernel, optimiser and now in
sample reuse, so steps/hour differs by up to 1.8x between them; comparing at matched step
silently rewards whichever recipe takes the most steps per hour, and comparing at matched
runtime is the axis the cost curve is actually denominated in. Both are printed, because a
gap that exists on one and not the other is itself the finding.

⚠️ The policy puzzle probe is the weak instrument here: across five paired leagues its
correlation with dElo is +0.79 at n=16, +0.58 at n=64 and **+0.08 at n=256**. The value
column only exists for runs started after 2026-08-19, when the probe was wired in.

    uv run --no-project --python .venv/bin/python scripts/status.py [live-run]
"""
from __future__ import annotations

import datetime
import json
import os
import sys

LIVE = sys.argv[1] if len(sys.argv) > 1 else "t12h-wdl"
#: The live run's `--minutes` budget, in hours. Only used for the projection line.
HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
# ⚠️ **The peer list is discovered, not written down, and that is the fix for a bug that
# recurred three times.** A hand-maintained list goes stale silently: on 2026-08-21 it
# was missing `t12h-flat` and `t12h-reuse2`, and on 2026-08-22 it was missing
# `t12h-wdl` — which was the *only* run differing from the live one by a single flag,
# i.e. the one comparison the table existed to make. A missing peer is not a smaller
# table, it is a **wrong** table that reads like a complete one.
#
# So: every run under `runs/` with a puzzle series is a peer, minus `DROP`. Anything new
# appears automatically; anything unlabelled is printed with a ⚠️ rather than hidden.
#: Runs that are not comparable on this axis at all — a smoke test, a 15-minute probe,
#: and AlphaGateau's own cadence (batch 256, so its "steps" are not our steps).
DROP = {"probe-smoke", "t12h-n32", "t12h-agc", "t12h-agrepro", "t9h-n128-sweep",
        "t7h-n128-collapse"}


def _discover(live: str) -> list:
    import glob
    found = []
    for d in glob.glob("runs/*/"):
        r = os.path.basename(d.rstrip("/"))
        if r == live or r in DROP:
            continue
        log = os.path.join(d, f"{r}.log")
        if not os.path.exists(log):
            continue
        # A run counts as a peer once it has produced at least one puzzle probe.
        with open(log, errors="ignore") as fh:
            if any("puzzles  solve" in line for line in fh):
                found.append((os.path.getmtime(log), r))
    return [r for _, r in sorted(found, reverse=True)]


LABEL = {
    "t12h-muon-wdl": "Muon wd.09, flat lr, WDL head",
    "t12h-reuse2": "AdamW int8-all, reuse 1.63, cosine",
    "t12h-flat": "AdamW int8-all, CONSTANT lr",
    "t12h-nsched": "AdamW int8-all, constant lr, n 32-256",
    "t12h-wdl": "AdamW int8-all, flat lr, WDL head",
    "t24h-adamw-int8": "AdamW int8-all gumbel, 24 h",
    "t12h-int8": "AdamW int8-FFN gumbel",
    "t12h-gumbel": "AdamW fp8 gumbel",
    "t12h-muon9-int8": "Muon wd.09 int8-all gumbel",
    "t12h-muon9": "Muon wd.09 int8-FFN gumbel",
    "t12h-muong": "Muon wd.04 fp8 gumbel",
    "t12h-vw2": "Muon wd.09 vw2.0 int8-all gumbel",
    "t24h-fp8": "AdamW fp8 PUCT, 24 h",
    "t24h-muon": "Muon wd.01 fp8 PUCT, 24 h",
}


def probes(run):
    """step -> (policy pass@1, policy solve, value pass@1 or None, value solve)."""
    out = {}
    p = f"runs/{run}/{run}-puzzles.jsonl"
    if not os.path.exists(p):
        return out
    for line in open(p):
        x = json.loads(line)
        k, b = x["kinds"], x["bins"]
        n = sum(v["n_turns"] for v in k.values())
        N = sum(v["n"] for v in b)
        vb = x.get("value_bins")
        vp = vs = None
        if vb:
            vt = sum(v["n_turns"] for v in vb)
            vp = sum(v["value_pass@1"] * v["n_turns"] for v in vb) / vt if vt else None
            vs = sum(v["solved"] for v in vb) / sum(v["n"] for v in vb)
        out[x["step"]] = (sum(v["pass@1"] * v["n_turns"] for v in k.values()) / n,
                          sum(v["solved"] for v in b) / N, vp, vs)
    return out


def wall(run):
    w = {}
    p = f"runs/{run}/{run}.jsonl"
    if not os.path.exists(p):
        return w
    for line in open(p):
        x = json.loads(line)
        if "euros/wall_seconds" in x and "step" in x:
            w[x["step"]] = x["euros/wall_seconds"] / 3600
    return w


def main():
    PEERS = _discover(LIVE)
    P = {r: probes(r) for r in [LIVE] + PEERS}
    W = {r: wall(r) for r in [LIVE] + PEERS}
    if not P[LIVE]:
        print(f"  {LIVE}: no puzzle probe yet")
        return
    step = max(P[LIVE])
    h = W[LIVE][min(W[LIVE], key=lambda t: abs(t - step))] if W[LIVE] else float("nan")
    live = P[LIVE][step]
    tot = max(W[LIVE].values()) if W[LIVE] else h
    print(f"  {LIVE}  step {step}, {tot:.2f} h elapsed; latest probe at {h:.2f} h\n")
    print(f"  {'run':18s} {'configuration':34s} {'@step':>6s} {'@h':>5s} "
          f"{'pass@1':>7s} {'solve':>7s} {'val@1':>7s} {'vsolve':>7s} {'d p@1':>7s}")
    rows = [(LIVE, step, h, live)]
    for r in PEERS:
        if not P[r] or not W[r]:
            continue
        hof = lambda s: W[r][min(W[r], key=lambda t: abs(t - s))]
        k = min(P[r], key=lambda t: abs(hof(t) - h))
        rows.append((r, k, hof(k), P[r][k]))
    for r, k, hh, (p1, sv, vp, vs) in rows:
        mark = "  <- LIVE" if r == LIVE else ""
        # ⚠️ An unlabelled peer is shown with a marker rather than dropped: the whole
        # point of discovery is that a new run can never go missing from this table.
        if r != LIVE and r not in LABEL:
            LABEL[r] = "⚠️ unlabelled — add it to LABEL"
        v1 = f"{vp:7.4f}" if vp is not None else "      -"
        v2 = f"{vs:7.4f}" if vs is not None else "      -"
        d = "" if r == LIVE else f"{live[0] - p1:+7.4f}"
        print(f"  {r:18s} {LABEL.get(r, ''):34s} {k:6d} {hh:5.2f} "
              f"{p1:7.4f} {sv:7.4f} {v1} {v2} {d:>7s}{mark}")
    print("\n  finals for reference: t24h-adamw-int8 0.4602/0.1638   "
          "t12h-muon9-int8 0.4545/0.1573   t12h-int8 0.4056/0.1174   AG 0.5051/0.2212")
    if len(W[LIVE]) > 40:
        # ⚠️ The *recent* rate, not step/elapsed. `--min-records` holds the gradient
        # phase off during the buffer fill, so the lifetime average understates the
        # steady state badly early on -- 351 steps/h against a true 758 at 0.6 h.
        ks = sorted(W[LIVE])
        a, b = ks[-40], ks[-1]
        rate = (b - a) / max(W[LIVE][b] - W[LIVE][a], 1e-9)
        cur = max(W[LIVE].values())
        # ⚠️ Projected *final step at 12 h*, not an ETA to a step target: every run
        # here is bounded by `--minutes`, so `--total-steps` is the schedule's length
        # and not a stopping condition. Under a flat rate it is inert entirely.
        left = max(HOURS - cur, 0.0)
        print(f"  {rate:.0f} steps/h over the last 40 logged points "
              f"({step / max(cur, 1e-9):.0f} lifetime, held down by the buffer fill); "
              f"~{step + rate * left:,.0f} steps at {HOURS:g} h, ending "
              f"{(datetime.datetime.now() + datetime.timedelta(hours=left)).strftime('%F %H:%M')}")


if __name__ == "__main__":
    main()
