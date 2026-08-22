#!/usr/bin/env bash
# 12 h: `t24h-adamw-int8`'s recipe with the sample reuse doubled.
#
#   1/4  train t12h-reuse2, 12 h
#   2/4  ONE joint league: t12h-reuse2 + t24h-adamw-int8
#   3/4  the curve
#   4/4  direct head to head against AlphaGateau's released checkpoint
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--samples-per-position`**, 0.815 -> 1.63. Every
# other flag is copied from `chain-t24h-adamw-int8.sh`: AdamW lr 1e-3, `--adam-wd` at its
# 0.01 default, gumbel m=16, n=128, 1024 games, 10 moves/phase, window 20000 /
# mean-plies 350, cosine to `lr_min` 5e-5, warmup 30, clip 1.0, terminal collapse,
# `--int8`. (Second flag, and it is the *implementation* of the first: see `--min-records`
# below.)
#
# ## Why this knob
#
# Measured reuse has been **0.813 in every run on the ledger** -- each generated position
# is trained on *less than once*, and ~19 % are evicted from the 20 000-game window
# without ever being sampled. Self-play is 87 % of a generation (14.7 s of 16.9 s), so the
# loop generates data faster than the gradient phase consumes it. The knob has never been
# swept.
#
# At 1.63 the arithmetic, from the measured 1.08 s per optimiser step: steps/generation
# 2.04 -> 4.08, gradient 2.2 s -> 4.4 s, generation 16.9 s -> 19.1 s. So **~1.77x
# steps/hour (429 -> ~769) and ~11 % fewer games**. `--total-steps 9100` sizes the cosine
# at ~98 % of 12 h at that rate; past `total_steps` the schedule clamps to `lr_min`
# (`loop.py:357`), so the run anneals fully and finishes flat rather than being cut off.
#
# ⚠️ **`--min-records 200000` is not a free choice, it is what high reuse requires.**
# `loop.py:743` gates the gradient phase on `max(batch, min_records)` and `min_records`
# defaults to 0, so the floor is the batch -- which *scales with the batch and therefore
# shrinks exactly when a high-reuse recipe needs it larger*. Measured 2026-08-12 at
# AlphaGateau's settings (batch 256, 7.6 samples/position) that produced **304 steps over
# a 465-record buffer by generation 3**, loss collapsing to 1.68 and KL to 0.17, with
# every counter reporting a healthy run. 200 000 records is ~20 generations, under 2 % of
# the run, and it binds only during the fill.
#
# ## The honest prior, and the argument against my own scepticism
#
# ⚠️ I argued against this arm and the argument was thin. It rested on **one** external
# measurement -- 12 h -> 24 h bought **+12 +/- 34 Elo** against AlphaGateau -- generalised
# to a different axis. Reuse is not more wall clock; it is more gradient steps on the same
# data, which is *sample efficiency*, and this project's two largest Track E results are
# both on that axis: **E0**, n=128 reaching the landmark on 2.7-3.0x fewer positions
# (slope 25 -> 332 Elo/decade), and **E1.2/PCR**, 3.7-6.0x on positions for +36 +/- 70.
#
# The reservation that survives: 1.77x steps comes with **0.89x games**, so if the gain
# from 12 h -> 24 h came from *data* rather than *steps*, this returns nothing. That is
# the question the run answers, and nobody has asked it.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * Control is **`t24h-adamw-int8`**, in ONE joint Bradley-Terry fit. Its ladder
#     contains a ~step-4600 checkpoint at the 12 h mark, so the pairing gives both a
#     **matched wall clock** and a **matched step** comparison from the same fit.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ **The league is no longer the verdict.** Measured 2026-08-19: `t24h-adamw-int8`
#     is **+237 +/- 79 (z = 3.0)** over `t12h-int8` internally and **+12 +/- 34** against
#     AlphaGateau. Self-anchored Elo does not transfer. Step 4 is the number that counts.
#   * ⚠️ Reported alongside, and cheaper than they were: **value-head puzzles now run
#     in-loop at every checkpoint** (`puzzles/value_pass@1`, `puzzles/value_solve_rate`),
#     the value head answering alone -- no terminal collapse, no rules override, no
#     search. 20.6 s per checkpoint on the fused encoder.
#   * ⚠️ The policy puzzle probe is reported and **not** weighted: across five paired
#     leagues its correlation with dElo is +0.79 at n=16, +0.58 at n=64 and **+0.08 at
#     n=256**.
#   * ⚠️ In-buffer value MSE is an **anti-signal**: three cases now have it inverted
#     against held-out value quality, most recently `t12h-vw2` (19 % better buffer MSE,
#     -0.045 value-puzzle, -231 Elo).
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t12h-reuse2/chain.log
set -u
cd ~/brokefish
RUN=t12h-reuse2
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

# 1/4 -----------------------------------------------------------------------
say "1/4 train $RUN  (12 h, adamw, int8 all four, gumbel m=16, n=128, reuse 1.63)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 9100 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.001 --lr-min 5e-5 --decay cosine --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --samples-per-position 1.63 --min-records 200000 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/4 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/4 -----------------------------------------------------------------------
say "2/4 joint league: $RUN + $CTRL"
$PY -m brokefish.eval.league \
  --run $RUN --run $CTRL \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/4 done (exit $?)"

# 3/4 -----------------------------------------------------------------------
say "3/4 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+$CTRL.json >> "$LOG" 2>&1
say "3/4 done (exit $?)"

# 4/4 -----------------------------------------------------------------------
# ⚠️ Same protocol as `logs/gate2-h2h-alphagateau.json` so the numbers compare directly:
# seed 0, 100 openings, n = 128 both sides, 8 opening plies, 300-ply cap. Reference
# points on it: `t12h-gumbel-004009` 0.400 (-70 Elo), `t24h-adamw-int8` 0.4175 (-58).
say "4/4 head to head vs alphagateau"
bash scripts/h2h-vs-ag.sh runs/$RUN/checkpoints/$RUN.pt $RUN-vs-ag 200 128 0 0 \
  >> "$LOG" 2>&1
say "4/4 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
