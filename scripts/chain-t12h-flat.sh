#!/usr/bin/env bash
# 12 h: `t24h-adamw-int8`'s recipe with a **constant learning rate**.
#
#   1/3  train t12h-flat, 12 h
#   2/3  ONE joint league: t12h-flat + t24h-adamw-int8
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS THE SCHEDULE.** `--decay step` with a single-entry
# `lr_schedule` makes `lr_at` return the same rate at every step (`loop.py:381`), so this
# is a flat 1e-3 after the 30-step warmup. Everything else is copied from
# `chain-t24h-adamw-int8.sh`: AdamW lr 1e-3, `--adam-wd` at its 0.01 default, gumbel
# m=16, n=128, 1024 games, 10 moves/phase, window 20000 / mean-plies 350, warmup 30,
# clip 1.0, terminal collapse, `--int8`, reuse at its 0.815 default.
#
# ⚠️ `--total-steps` is **inert here** and is set only so the log's schedule line is not
# nonsense: under `step` decay with one entry, `lr_at` never consults it. That removes the
# sizing risk that every cosine run carries.
#
# ## Why this has never been measured
#
# Every run with an Elo number on this ledger uses `--decay cosine`. The runs that did not
# -- `t15-adamw`, `t15-adamw-warm`, `t4h-n64`, `t60-n256` -- are 15-minute to 4-hour probes
# from July on recipes that no longer exist, and **no league has ever compared a constant
# rate with a cosine one**. The cosine was adopted and never validated against the
# alternative.
#
# ## The argument that has nothing to do with final Elo
#
# ⚠️ **Under cosine, an intermediate checkpoint is not a valid point on the cost curve.**
# It is mid-anneal and systematically understates what its compute budget is worth. That
# is not hypothetical: on 2026-08-20 `t24h-adamw-int8` at its 12 h mark was 53 % through a
# 9800-step cosine, so reading it as "the same recipe at 12 h" was wrong, and the
# `t12h-reuse2` comparison inherited the ambiguity. The whole class of "the control had a
# different schedule" objections exists because of this.
#
# With a flat rate **every checkpoint is a finished checkpoint**. A 12 h run and a 24 h run
# become comparable at every hour, `docs/roadmap.md`'s cost-vs-Elo curve becomes readable
# at every point rather than only at its end, and future arms stop needing a matched
# `--total-steps` to be interpretable.
#
# ⚠️ The cost is whatever annealing was buying. Measured on `t24h-adamw-int8`, the last
# 20 % of its cosine was worth about **+0.01 policy pass@1** -- small, but that is one
# observation on the weak instrument, not a measurement of what this run gives up.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * Control is **`t24h-adamw-int8`** in ONE joint Bradley-Terry fit. Its ladder gives
#     both a matched-step and a matched-wall-clock comparison -- and ⚠️ the matched-wall-
#     clock one is exactly the reading this run exists to make interpretable, so it is
#     reported *with* the caveat that the control is unannealed there, not instead of it.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ **The league is not the verdict.** Measured 2026-08-19: `t24h-adamw-int8` is
#     +237 +/- 79 (z = 3.0) over `t12h-int8` internally and **+12 +/- 34** against
#     AlphaGateau. Self-anchored Elo does not transfer across engine families.
#   * ⚠️ Reported alongside: **value-head puzzles, logged in-loop at every checkpoint**
#     (`puzzles/value_pass@1`, `puzzles/value_solve_rate`) -- the value head answering
#     alone, no terminal collapse, no rules override, no search. On `t12h-reuse2` that
#     metric traced a U: 0.311 -> 0.358 at step 1609 -> 0.259 at 5831 -> 0.333 at the end,
#     while the policy went 0.057 -> 0.414. ⚠️ **A flat rate is the clean test of whether
#     that U is the cosine or the recipe**, because there is no cosine here to blame.
#   * ⚠️ The policy puzzle probe is reported and not weighted: correlation with dElo is
#     +0.79 at n=16, +0.58 at n=64 and **+0.08 at n=256** across five paired leagues.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best -- and under a flat rate that
# distinction matters more, because every checkpoint is a candidate.
#
# ⚠️ No AlphaGateau match in this chain. Run it afterwards if the league warrants:
#     bash scripts/h2h-vs-ag.sh runs/t12h-flat/checkpoints/t12h-flat.pt t12h-flat-vs-ag 200 128 0 0
#
#   tail -f runs/t12h-flat/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=t12h-flat
CTRL=t24h-adamw-int8
mkdir -p runs/$RUN
LOG=runs/$RUN/chain.log
PY="uv run --no-project --python .venv/bin/python"

say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "chain start.  tail -f $LOG"
git rev-parse HEAD | tee -a "$LOG"
git diff > runs/$RUN/$RUN.diff; wc -l runs/$RUN/$RUN.diff | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader | tee -a "$LOG"

# 1/3 -----------------------------------------------------------------------
say "1/3 train $RUN  (12 h, adamw CONSTANT lr 1e-3, int8 all four, gumbel m=16, n=128)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.001 --decay step --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
say "2/3 joint league: $RUN + $CTRL"
$PY -m brokefish.eval.league \
  --run $RUN --run $CTRL \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+$CTRL.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
