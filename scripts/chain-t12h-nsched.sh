#!/usr/bin/env bash
# 12 h: constant learning rate, with the search budget stepped 32 -> 64 -> 128 -> 256.
#
#   1/3  train t12h-nsched, 12 h
#   2/3  ONE joint league: t12h-nsched + t12h-flat
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--sims-schedule`.** Everything else is copied from
# `chain-t12h-flat.sh`: AdamW at a **constant** 1e-3 (`--decay step`, single-entry
# schedule), `--adam-wd` at its 0.01 default, gumbel m=16, 1024 games, 10 moves/phase,
# reuse at its 0.815 default, window 20000 / mean-plies 350, warmup 30, clip 1.0,
# terminal collapse, `--int8`.
#
# ## Why this is only interpretable now
#
# ⚠️ A stepped `n` was **untestable under a cosine**, and that is what wrecked
# `t12h-anneal` (the only previous attempt, which lost 45-70 Elo to `t12h-pcr`). Records
# per generation are fixed at 10 240 whatever `n` is, so only the *generation time*
# changes -- and it changes by 5.4x across 32..256. Measured rates: 1250 / 769 / 435 / 232
# steps per hour. Under a step-keyed cosine that puts **46 % of the run's gradient steps
# in the first three hours** and leaves the schedule 91 % annealed before n = 256 begins,
# so the best targets arrive when the learning rate is already at its floor.
#
# A constant rate removes that entirely: it does not matter how the steps distribute in
# time, because every step gets the same rate. This run and `t12h-flat` differ in exactly
# one thing.
#
# ## The switch points
#
# `sims_schedule` keys on **training seconds** (self-play + gradient), deliberately --
# `loop.py:150`: "wall clock would include the puzzle probe, which §10 keeps off the
# axis". The probe costs ~47 s per checkpoint (26 policy + 21 value) every 200 steps, and
# that overhead is largest exactly where the step rate is highest, so equal *training*
# seconds are not equal wall clock. Solving for 3 h of wall clock per phase:
#
#     phase   steps/h   probe s per training hour   training seconds for 3 h wall
#     n=32       1250            294                        9 985
#     n=64        769            181                       10 284
#     n=128       435            102                       10 502
#     n=256       232             55                       10 639
#
# which gives the keys 0 / 9985 / 20269 / 30771, rounded to 0 / 10000 / 20300 / 30800.
# ⚠️ `--sims` is raised to the largest N automatically (`loop.py`'s CLI note), so the tree
# is allocated for 256 from the start.
#
# ⚠️ **`pcr_p` stays 0**, so this is a plain stepped cap, not KataGo's playout cap
# randomisation. The `/n` half of each entry is therefore inert and is set equal to `N`
# so the log does not imply a fast cap that is never used.
#
# ⚠️ **The one thing I would change if it were my call, and am not:** at `gumbel_m = 16`,
# n = 32 gives each candidate **2 visits total and 0.5 in the first halving phase**.
# `CLAUDE.md` records m = n as degenerate -- m=16 at n=16 measured **-92 Elo against
# PUCT**. The first three hours therefore train on near-degenerate policy targets. Scaling
# m with n (8 at n=32) would fix it and would also be a second variable, so it is flagged
# rather than done.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * Control is **`t12h-flat`**: same recipe, same constant rate, same 12 h, fixed
#     n = 128. One joint Bradley-Terry fit, so matched step and matched wall clock both
#     come from the same scale.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ **The league is not the verdict.** `t24h-adamw-int8` measured +237 +/- 79
#     internally over `t12h-int8` and **+12 +/- 34** against AlphaGateau.
#   * ⚠️ Reported alongside: value-head puzzles, in-loop at every checkpoint. On
#     `t12h-flat`, 25 probes over 12 h with no schedule anywhere: **policy 0.0844 ->
#     0.4087 (4.8x), value 0.3530 -> 0.3480 (0.99x)**. If a stepped budget moves the value
#     head at all it will show against that flat line.
#   * ⚠️ The policy puzzle probe is reported and **not** weighted: correlation with dElo
#     is +0.79 at n=16, +0.58 at n=64 and **+0.08 at n=256** across five paired leagues.
#     ⚠️ It is also **not comparable across phases here** -- the probe runs at
#     `n_sims = 0`, so it grades the policy head alone and is blind to the budget that
#     produced its targets.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint.
#
#   tail -f runs/t12h-nsched/chain.log
set -u
cd ~/brokefish
RUN=t12h-nsched
CTRL=t12h-flat
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
say "1/3 train $RUN  (12 h, constant lr 1e-3, n stepped 32/64/128/256 every 3 h)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --sims-schedule '0:32/32,10000:64/64,20300:128/128,30800:256/256' \
  --minutes 720 --total-steps 8000 \
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
