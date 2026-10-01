#!/usr/bin/env bash
# 12 h: `t12h-wdl`'s recipe with **`--target-mix 0.5`**, one variable.
#
#   1/4  train t12h-target-mix, 12 h
#   2/4  the headline: 200 games at n=128 against t24h-adamw-int8 (scripts/h2h-bf.sh)
#   3/4  ONE joint league under Gumbel+collapse: t12h-target-mix + t12h-wdl
#   4/4  the curve
#
# ⚠️ ONE VARIABLE MOVES AND IT IS `--target-mix`, 0 -> 0.5. Every other flag is copied
# verbatim from `chain-t12h-wdl.sh`: WDL head, AdamW lr 1e-3 `--decay step`, warmup 30,
# clip 1.0, gumbel m=16, n=128, 1024 games, 10 moves/phase, window 20000 / mean-plies
# 350, terminal collapse, `--int8`, reuse at its 0.815 default.
#
# ## What the flag is
#
# The value target becomes `0.5 z + 0.5 root_value`: half the game's outcome, half the
# search's own root value at the time the position was played. `root_value` is written
# by the search into every record already (buffer.py) and is the search's number, not
# anyone's opinion, so it sits inside the tabula rasa boundary. On the WDL head the
# label is a soft W/D/L distribution: the outcome's one-hot mixed with the maximum-
# entropy distribution whose expectation is the root value (loss.py:wdl_max_entropy).
#
# ## Why this knob (docs/journal/2026-08-24-the-value-head-is-a-calibration-failure.md)
#
# The trunk carries the value information in every run measured and the head throws
# it away by fitting the one outcome bit of a game, repeated over its ~118 positions.
# On the offline harness of 2026-08-24 (fixed data, same init) this target took Muon's
# held-out game correlation from 0.494 to 0.591 and AdamW's from 0.594 to 0.604; the
# 2026-09-09 dissection puts the value head first among the causes of our 2.1x hanging
# rate against AlphaGateau (41 % of hanging moves do not lower the raw value). Nothing
# in the loop has tried it before this run.
#
# ⚠️ The honest prior. The last six 12 h single-knob arms measured inside their error
# bars or lost (reinjection x4, int8, WDL under PUCT). The base rate is "no measurable
# effect", and the offline gain for AdamW was +0.010 of correlation, not Muon's +0.097.
#
# ⚠️ PREREGISTERED MEASUREMENTS (evaluation.md §3). These are what gets reported; what
# they mean for the line is Theo's call, not this file's.
#   * Headline: the final checkpoint's score against t24h-adamw-int8 over 200 games at
#     n=128, Gumbel m=16 both sides, with its 95 % Wilson interval. For scale, the 12 h
#     arms measured on this instrument score 0.4225 (reinject lr3), 0.4525 (lr6),
#     0.3675 (reuse2) and t12h-int8 0.400; t12h-wdl itself has not been played on it and
#     is played here in the same stage, so the two 12 h scores are read side by side.
#   * The joint Gumbel league with t12h-wdl, all three budgets; a sign flip across
#     budgets is itself the finding. ⚠️ Every joint fit's levels are compressed by the
#     phantom prior (evaluation.md §5.4.5); the neighbouring difference is exact.
#   * In-run: `gradient/value` is a soft-label CE and is NOT comparable to t12h-wdl's
#     hard-label number; `value_saturated_frac` and the value puzzle probe are.
#
# tail -f runs/t12h-target-mix/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=t12h-target-mix
CTRL=t12h-wdl
REF=runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt
mkdir -p runs/$RUN
LOG=runs/$RUN/chain.log
PY="uv run --no-project --python .venv/bin/python"
say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
say "chain start.  tail -f $LOG"
git rev-parse HEAD | tee -a "$LOG"
git diff > runs/$RUN/$RUN.diff; wc -l runs/$RUN/$RUN.diff | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader | tee -a "$LOG"

# 1/4 -----------------------------------------------------------------------
say "1/4 train $RUN  (12 h, t12h-wdl's recipe + --target-mix 0.5)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --target-mix 0.5 --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.001 --decay step --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/4 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/4 -----------------------------------------------------------------------
say "2/4 headline: $RUN vs t24h-adamw-int8, 200 games at n=128; then $CTRL on the same instrument"
scripts/h2h-bf.sh runs/$RUN/checkpoints/$RUN.pt $REF $RUN-vs-t24h 200 128 >> "$LOG" 2>&1
grep "A scores" logs/h2h-$RUN-vs-t24h.log | tee -a "$LOG"
scripts/h2h-bf.sh runs/$CTRL/checkpoints/$CTRL.pt $REF $CTRL-vs-t24h 200 128 >> "$LOG" 2>&1
grep "A scores" logs/h2h-$CTRL-vs-t24h.log | tee -a "$LOG"
say "2/4 done"

# 3/4 -----------------------------------------------------------------------
say "3/4 joint league UNDER GUMBEL+COLLAPSE: $RUN + $CTRL"
$PY -m brokefish.eval.league \
  --run $RUN --run $CTRL \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "3/4 done (exit $?)"

# 4/4 -----------------------------------------------------------------------
say "4/4 the curve"
$PY -m brokefish.eval.curve runs/$RUN/league-joint-$RUN+$CTRL.json >> "$LOG" 2>&1
say "4/4 done (exit $?)"
say "chain finished"
