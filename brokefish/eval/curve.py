"""The cost-versus-Elo curve, which is the deliverable. `evaluation.md` §5.3.

    uv run --no-project --python .venv/bin/python -m brokefish.eval.curve \
        logs/league-t4h-n64.json --csv docs/ledger/curve-t4h-n64.csv

This module presents; `league.py` measures. In particular the euro axis is
**joined by the league**, not here — §5.3's warning is that if the two axes are
written by different processes they get joined by hand later, and by then nobody
remembers whether the euro counter included the failed runs. So a curve point that
arrives here with `euros_spent = null` is reported as null and is not repaired.

⚠️ **This is the self-anchored scale** (`evaluation.md` §5.1, §6): Elo 0 is
**uniformly random legal play** — rebased 2026-08-08 from the frozen random-init
network, which was a network *plus a search* and therefore moved whenever the search
did. It is not a CCRL rating and not a Lichess rating. Every table this writes says so
in its header, because a number on this scale and a number on a published one are
different quantities and the whole point of §6 is that they must never be silently
mixed.

⚠️ **`n_sims` is per player, not per league** (§5.1a). A curve may hold the same
checkpoint at several budgets; sorting by step alone then puts them on top of each
other, so the table carries the budget as a column and the reader should group by it
before reading a slope.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Optional, Sequence, Tuple

SCALE_NOTE = ("Elo is on the self-anchored scale of evaluation.md §5.1: 0 is uniformly "
              "random legal play, not a published rating.")

COLUMNS = ("checkpoint_id", "run", "step", "euros_spent", "training_seconds", "games_played",
           "elo", "ci95", "se", "se_raw", "n_sims", "draw_rate", "fit_version")


def load_report(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def curve_rows(report: dict) -> List[dict]:
    """Curve points in step order, then budget order.

    ⚠️ Step alone is no longer a key (§5.1a, 2026-08-08): a pool that rates one
    checkpoint at several budgets has several rows per step, and sorting on step alone
    leaves their order to the sort's stability rather than to anything meaningful.
    """
    return sorted(report.get("curve", []),
                  key=lambda r: (r.get("step") or 0, r.get("n_sims") or 0))


def format_curve(report: dict) -> str:
    """A markdown table of the curve, for a log and for `docs/ledger/`."""
    rows = curve_rows(report)
    cfg = report.get("config", {})
    head = (f"  {cfg.get('pairings', '?')} pairings, {cfg.get('games_played', '?')} games; "
            f"reference n = {cfg.get('n_sims', '?')} sims, budgets in the pool "
            f"{cfg.get('budgets', '?')}. {SCALE_NOTE}")
    out = [head, "",
           "| checkpoint | step | sims | train s | € | games | Elo | ±95% | draws |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        secs = "—" if r.get("training_seconds") is None else f"{r['training_seconds']:.0f}"
        euro = "—" if r.get("euros_spent") is None else f"{r['euros_spent']:.2f}"
        elo = "—" if r.get("elo") is None else f"{r['elo']:+.0f}"
        ci = "pinned" if r["checkpoint_id"] == report.get("fit", {}).get("anchor") else (
            "—" if r.get("ci95") is None else f"±{r['ci95']:.0f}")
        draw = "—" if r.get("draw_rate") is None else f"{r['draw_rate']:.1%}"
        out.append(f"| {r['checkpoint_id']} | {r.get('step', '')} | "
                   f"{r.get('n_sims', '')} | {secs} | {euro} | "
                   f"{r.get('games_played', 0)} | {elo} | {ci} | {draw} |")
    return "\n".join(out)


def write_csv(report: dict, path: str) -> None:
    """The curve as data. One row per checkpoint, the columns of §5.3."""
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in curve_rows(report):
            w.writerow(r)


def cost_axis(report: dict) -> Optional[str]:
    """Which compute axis this report can actually plot: euros, seconds, or neither.

    Euros are preferred because they are the paper's x-axis, but a run launched at
    the default `--euros-per-hour 0` has an identically-zero euro column, and a
    slope against a constant x is not a small number — it is not a number. Training
    seconds are the same quantity before the rate is applied, so they are the
    fallback rather than a different measurement.
    """
    rows = curve_rows(report)
    for key in ("euros_spent", "training_seconds"):
        vals = [r.get(key) for r in rows if r.get("elo") is not None]
        good = [v for v in vals if v is not None and v > 0]
        if len(good) >= 2 and max(good) > min(good):
            return key
    return None


def row_run(r: dict) -> str:
    """Which training run a curve point belongs to.

    `league.py` writes a `run` field, but reports from before 2026-08-05 do not have
    one, so it falls back to the `<run>@<step>` naming that `build_pool` has always
    used. The anchor belongs to no run and comes back as `""`.
    """
    if r.get("run"):
        return r["run"]
    cid = r.get("checkpoint_id", "")
    return cid.split("@")[0] if "@" in cid else ""


def group_by_run(report: dict) -> List[Tuple[str, List[dict]]]:
    """`(run, rows)` in first-appearance order; the anchor is dropped.

    ⚠️ The anchor is in every run's graph and is pinned at Elo 0, so folding it into
    one run's least squares would drag that run's intercept and nobody else's.
    """
    out: dict = {}
    for r in curve_rows(report):
        run = row_run(r)
        if run:
            out.setdefault(run, []).append(r)
    return list(out.items())


def slope_per_decade(report: dict, rows: Optional[Sequence[dict]] = None) -> Optional[float]:
    """Elo gained per 10× of training compute — Jones' law, `CLAUDE.md`'s Gate 2.

    Least squares on `(log10 cost, Elo)` over the points that have both, on
    whichever axis `cost_axis` says exists. ⚠️ Returns `None` rather than a number
    when there is no axis with any spread.
    """
    key = cost_axis(report)
    if key is None:
        return None
    # ⚠️ One run at a time. A joint report holds several runs on one scale, and a
    # single least squares over all of them fits the *envelope* of two trajectories,
    # which is not any run's slope and is not what Gate 2 asks for.
    pts = [(r[key], r["elo"]) for r in (curve_rows(report) if rows is None else rows)
           if r.get(key) and r.get("elo") is not None and r[key] > 0]
    if len(pts) < 2:
        return None
    xs = [math.log10(x) for x, _ in pts]
    ys = [y for _, y in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def plot(report: dict, path: str) -> Optional[str]:
    """A PNG of the curve, if matplotlib is here. Returns the path, or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    groups = [(run, [r for r in rs if r.get("elo") is not None])
              for run, rs in group_by_run(report)]
    groups = [(run, rs) for run, rs in groups if rs]
    if not groups:
        return None
    key = cost_axis(report)
    label = {"euros_spent": "training euros",
             "training_seconds": "training seconds (self-play + gradient)",
             None: "optimiser steps"}[key]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for run, rows in groups:
        xs = [(r.get(key) if key else None) or r["step"] for r in rows]
        ys = [r["elo"] for r in rows]
        es = [r.get("ci95") or 0.0 for r in rows]
        sl = slope_per_decade(report, rows)
        tag = run if sl is None else f"{run}  ({sl:+.0f} Elo/decade)"
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1, capsize=2,
                    label=tag if len(groups) > 1 else None)
    if len(groups) > 1:
        ax.legend(fontsize=7)
    ax.axhline(0.0, lw=0.8, ls="--", color="grey")
    ax.set_xlabel(label)
    ax.set_ylabel("Elo (self-anchored, 0 = random init)")
    ax.set_title(f"brokefish cost-vs-Elo, n = {report.get('config', {}).get('n_sims', '?')}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="present a league report as the curve")
    ap.add_argument("report", help="logs/league-<run>.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args(argv)

    report = load_report(args.report)
    print(format_curve(report))
    axis = cost_axis(report)
    groups = group_by_run(report)
    print()
    # ⚠️ Per run, never pooled. A joint report puts several runs on one scale, and one
    # least squares across them fits the envelope of two trajectories rather than
    # either one's slope -- which is the quantity Gate 2 is about.
    for run, rows in groups:
        slope = slope_per_decade(report, rows)
        if slope is None:
            print(f"  {run:<24} Elo per 10x compute: —  (no cost axis with spread)")
        else:
            print(f"  {run:<24} Elo per 10x compute: {slope:+.0f}  on {axis}   "
                  f"Gate 2 wants +500 (CLAUDE.md)")
    if len(groups) > 1:
        print("  one Bradley-Terry fit, so these ratings are comparable across runs")
    if args.csv:
        write_csv(report, args.csv)
        print(f"  wrote {args.csv}")
    if args.png:
        got = plot(report, args.png)
        print(f"  wrote {got}" if got else "  matplotlib is not installed; no PNG")


if __name__ == "__main__":
    main()
