"""The cost-versus-Elo curve, which is the deliverable. `evaluation.md` §5.3.

    uv run --no-project --python .venv/bin/python -m brokefish.eval.curve \
        runs/t4h-n64/league-t4h-n64.json

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

    ⚠️ **Since §5.1a this is not a plottable series.** A run now holds several search
    budgets, and a "run" group mixes them: several points share an x and the line
    zigzags between budgets. Use :func:`group_by_series`.
    """
    out: dict = {}
    for r in curve_rows(report):
        run = row_run(r)
        if run:
            out.setdefault(run, []).append(r)
    return list(out.items())


def group_by_series(report: dict) -> List[Tuple[Tuple[str, int], List[dict]]]:
    """`((run, sims), rows)` — the unit that is actually a curve.

    §5.1a made a player a `(network, budget)` pair, so a training curve is a run **at
    one budget**. Grouping by run alone puts three budgets on one line: the x-axis
    repeats, the line zigzags, and — much worse — `slope_per_decade` fits a straight
    line through points that differ in search rather than in training and reports the
    result as Elo per decade of *compute*. That number would be wrong and would look
    entirely normal.
    """
    out: dict = {}
    for r in curve_rows(report):
        run = row_run(r)
        if run:
            out.setdefault((run, r.get("n_sims")), []).append(r)
    return list(out.items())


def by_training_level(report: dict) -> List[Tuple[Tuple[str, int], List[dict]]]:
    """`((run, step), rows sorted by budget)` for the checkpoints rated at 2+ budgets.

    The transpose of :func:`group_by_series`, and the view that answers the question
    the budget grid exists for: **does search get more or less valuable as the network
    trains?** Each series is one network, held exactly fixed, at several budgets, so
    its slope is Elo per doubling of search and nothing else.

    The untrained ladder is included as the `("init", 0)` level, because "what is
    search worth to a network that knows nothing" is the left-hand end of that curve.
    """
    out: dict = {}
    for r in report.get("curve", []):
        cid = r.get("checkpoint_id", "")
        if r.get("elo") is None or r.get("n_sims") in (None, 0):
            continue
        key = (row_run(r), r.get("step", 0)) if row_run(r) else ("init", 0)
        if key[0] == "init" and not cid.startswith("init"):
            continue                       # `random` has no budget to vary
        out.setdefault(key, []).append(r)
    return [(k, sorted(v, key=lambda r: r["n_sims"]))
            for k, v in out.items() if len(v) >= 2]


def elo_per_doubling(rows: Sequence[dict]) -> Optional[float]:
    """Least squares of Elo on `log2(sims)` — the value of a doubling of search."""
    pts = [(math.log2(r["n_sims"]), r["elo"]) for r in rows
           if r.get("n_sims") and r.get("elo") is not None]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


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
    """A PNG of the curve, if matplotlib is here. Returns the path, or None.

    **Two panels sharing the Elo axis, and the reason is not aesthetic.** Elo,
    training cost and search budget are three quantities, and the tempting move —
    collapse them onto one axis of "total compute" — is wrong here: training is a
    one-off cost and simulations are a *recurring* cost paid per move by whoever runs
    the thing, so adding them requires assuming how many games will ever be played.
    Nothing in this project fixes that number. So they stay on separate axes.

    * **Left: Elo against training cost, one line per budget.** The vertical distance
      between two lines at the same x *is* the Elo value of the extra search, read
      directly off the plot. Each line is one `(run, budget)` series, which is the
      only grouping whose slope is Elo per decade of *training*.
    * **Right: Elo against simulations, one line per training level.** Whether those
      lines fan out or converge as training proceeds is the question the budget grid
      exists to answer — is search worth more or less to a sharper policy? The left
      panel cannot show it, because there the budget is the *series* and not the axis.

    ⚠️ Contours of Elo over `(training, sims)` were considered and rejected: the grid
    is a handful of checkpoints at a handful of budgets, and a contour drawn through
    it would render interpolation as if it were measurement.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    series = [((run, sims), [r for r in rs if r.get("elo") is not None])
              for (run, sims), rs in group_by_series(report)]
    series = [(k, rs) for k, rs in series if rs]
    if not series:
        return None
    key = cost_axis(report)
    label = {"euros_spent": "training euros",
             "training_seconds": "training seconds (self-play + gradient)",
             None: "optimiser steps"}[key]
    levels = by_training_level(report)
    ncols = 2 if levels else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.6), dpi=140,
                             squeeze=False)
    ax = axes[0][0]

    # -- left: the training curve, one line per budget ---------------------- #
    runs = sorted({run for (run, _), _ in series})
    for (run, sims), rows in sorted(series, key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        xs = [(r.get(key) if key else None) or r["step"] for r in rows]
        ys = [r["elo"] for r in rows]
        es = [r.get("ci95") or 0.0 for r in rows]
        sl = slope_per_decade(report, rows)
        name = f"{run} " if len(runs) > 1 else ""
        tag = f"{name}n={sims}" + ("" if sl is None else f"  ({sl:+.0f} Elo/decade)")
        # Dashed for the sparse grid budgets, solid for the reference: they carry
        # five points against twenty-four and should not read as equally resolved.
        dense = len(rows) > 8
        ax.errorbar(xs, ys, yerr=es, marker="o" if dense else "s", ms=3,
                    lw=1.4 if dense else 1.0, ls="-" if dense else "--",
                    capsize=2, label=tag)
    # ⚠️ **Linear**, which is what every plot in this repository has used. A log x was
    # tried on 2026-08-08 and reverted: the run spans 1 146 s to 41 587 s, i.e. 1.6
    # decades, so a log axis gives half the plot width to the first tenth of the run —
    # the noisy, uninteresting part — and compresses the part anybody is reading it
    # for. The Elo-per-decade slope does not need the axis to be logarithmic; it is
    # fitted on `log10(cost)` regardless and printed in the legend.
    ax.axhline(0.0, lw=0.8, ls="--", color="grey")
    ax.set_xlabel(label)
    ax.set_ylabel("Elo (self-anchored, 0 = uniformly random legal play)")
    ax.set_title("training cost")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # -- right: the value of search, one line per training level ------------ #
    if levels:
        ax2 = axes[0][1]
        for (run, step), rows in sorted(levels, key=lambda kv: kv[0][1]):
            xs = [r["n_sims"] for r in rows]
            ys = [r["elo"] for r in rows]
            es = [r.get("ci95") or 0.0 for r in rows]
            d = elo_per_doubling(rows)
            tag = ("untrained" if run == "init" else f"step {step}")
            tag += "" if d is None else f"  ({d:+.0f}/2x)"
            ax2.errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1.2, capsize=2, label=tag)
        ax2.set_xscale("log", base=2)
        ax2.axhline(0.0, lw=0.8, ls="--", color="grey")
        ax2.set_xlabel("simulations per move (inference)")
        ax2.set_title("value of search, network held fixed")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(ax.get_ylim())

    fig.suptitle("brokefish: Elo against training cost and against search budget, "
                 "one Bradley-Terry fit", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="present a league report as the curve")
    ap.add_argument("report", help="runs/<run>/league-<run>.json")
    ap.add_argument("--csv", default=None,
                    help="beside the report, as curve-<stem>.csv, unless named")
    ap.add_argument("--png", default=None,
                    help="beside the report, as curve-<stem>.png, unless named")
    args = ap.parse_args(argv)

    # ⚠️ Default beside the report rather than into a fixed directory: a curve belongs
    # to the league that produced it, and the league already lives in the right run's
    # folder. Naming either flag still overrides.
    stem = os.path.basename(args.report)
    stem = stem[len("league-"):] if stem.startswith("league-") else stem
    stem = os.path.splitext(stem)[0]
    here = os.path.dirname(args.report) or "."
    if args.csv is None:
        args.csv = os.path.join(here, f"curve-{stem}.csv")
    if args.png is None:
        args.png = os.path.join(here, f"curve-{stem}.png")

    report = load_report(args.report)
    print(format_curve(report))
    axis = cost_axis(report)
    groups = group_by_series(report)
    print()
    # ⚠️ Per run **and per budget**, never pooled. A joint report puts several runs on
    # one scale, and one least squares across them fits the envelope of two
    # trajectories rather than either one's slope -- which is the quantity Gate 2 is
    # about. Since §5.1a the same is true across budgets, and worse: those points
    # differ in *search*, so a slope through them is not Elo per decade of training at
    # all, and nothing about the number would look wrong.
    for (run, sims), rows in sorted(groups, key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        slope = slope_per_decade(report, rows)
        tag = f"{run} @ n={sims}"
        if slope is None:
            print(f"  {tag:<28} Elo per 10x training: —  (no cost axis with spread)")
        else:
            print(f"  {tag:<28} Elo per 10x training: {slope:+.0f}  on {axis}   "
                  f"Gate 2 wants +500 (CLAUDE.md)")
    levels = by_training_level(report)
    if levels:
        print()
        print("  the other axis — Elo per doubling of *search*, network held fixed:")
        for (run, step), rows in sorted(levels, key=lambda kv: kv[0][1]):
            d = elo_per_doubling(rows)
            budgets = "/".join(str(r["n_sims"]) for r in rows)
            name = "untrained ladder" if run == "init" else f"{run}@{step}"
            print(f"  {name:<28} {d:+.0f} per 2x   over n = {budgets}")
        print("  ⚠️ inference cost is recurring and training cost is one-off; they are")
        print("     not added here, because nothing fixes how many games get played.")
    if len({run for (run, _), _ in groups}) > 1:
        print("  one Bradley-Terry fit, so these ratings are comparable across runs")
    if args.csv:
        write_csv(report, args.csv)
        print(f"  wrote {args.csv}")
    if args.png:
        got = plot(report, args.png)
        print(f"  wrote {got}" if got else "  matplotlib is not installed; no PNG")


if __name__ == "__main__":
    main()
