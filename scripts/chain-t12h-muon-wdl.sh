#!/usr/bin/env bash
# 12 h: `t12h-muon9-int8`'s Muon recipe with a **constant learning rate** and the
# **win/draw/loss value head**.
#
#   1/3  train t12h-muon-wdl, 12 h
#   2/3  ONE joint league, UNDER GUMBEL + COLLAPSE: + t12h-wdl + t12h-muon9-int8
#   3/3  the curve
#
# ⚠️ **TWO variables move against `t12h-muon9-int8`** -- the schedule and the value head
# -- and that is deliberate, because the pairing that isolates each one already exists:
#   * against **`t12h-wdl`** only the *optimiser* differs (both flat lr, both WDL);
#   * against **`t12h-muon9-int8`** only the schedule and the head differ.
# Both are in the same Bradley-Terry fit, so both readings come out of one league.
#
# Everything else is copied from `t12h-muon9-int8`'s recovered config: Muon peak
# lr 0.02, polar Newton-Schulz 5 steps, head_group 8, qkv_split, aux AdamW at 1e-3,
# `--adam-wd 0.09`, gumbel m=16, n=128, 1024 games, 10 moves/phase, window 20000 /
# mean-plies 350, warmup 30, clip 1.0, terminal collapse, `--int8`, reuse 0.815.
#
# ## Why this arm
#
# Measured 2026-08-22 on a **file-pinned** 40 000-position yardstick
# (`journal/2026-08-21-value-head-audit.md`, retraction section): the Muon value head
# **rises to step ~2200 and then degrades**, while AdamW's rises to its last checkpoint.
#
# | run | peak corr(v,z) | at step | final | final calibration excess |
# |---|---:|---:|---:|---:|
# | t12h-flat (AdamW, scalar) | 0.5693 | **5010** | 0.5693 | +5 % |
# | t12h-wdl (AdamW, WDL) | 0.5652 | 3408 | 0.5626 | +5 % |
# | t12h-muon9-int8 (Muon, scalar) | 0.5356 | **2204** | 0.5290 | **+24 %** |
# | t12h-vw2 (Muon, scalar, vw 2.0) | 0.5097 | 2204 | **0.4686** | **+51 %** |
#
# Muon has the best *policy* of any 12 h arm (0.4545 pass@1) and the worst value head.
# The WDL head is the one thing measured to hold calibration. **This run asks whether
# it fixes Muon's half of the problem.** Nothing here says it will: the mechanism of the
# Muon degradation is unmeasured, and a cross-entropy also pushes toward one-hot.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * **The league runs under `--gumbel --gumbel-m 16 --terminal-collapse`** -- the
#     protocol these networks are TRAINED under. Measured 2026-08-22
#     (`journal/2026-08-22-the-league-was-on-the-wrong-protocol.md`): rating `t12h-wdl`
#     under PUCT gave **-14 Elo** and under Gumbel **+121** on the identical calendar.
#     ⚠️ These ratings are therefore **not joinable** with any league before that date.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * Primary diagnostic = **held-out `corr(v, z)` on the pinned yardstick**, not the
#     value-puzzle probe. Measured 2026-08-21: the probe's within-run movement is
#     **0.59x its own noise**, while it tracks value quality *across* runs at Spearman
#     +0.96. So it is read final-vs-final only.
#   * ⚠️ **`gradient/value` and `value_saturated_frac` are not comparable to any Muon
#     run before this one** (MSE vs a 3-class cross-entropy; `ln 3 = 1.0986` is the
#     start, not 1.0). In-buffer value MSE has been an **anti-signal** three times.
#   * ⚠️ `--value-weight` stays 1.0 and is inherited, not calibrated, for a
#     cross-entropy. KataGo runs c_value = 1.5. A different weight is a separate run.
#   * ⚠️ The policy puzzle probe is reported and **not** weighted: its correlation with
#     dElo is +0.79 at n=16, +0.58 at n=64 and **+0.08 at n=256**. The Muon arms have
#     topped it and lost the head-to-head before (`t12h-muon9-int8` vs `t12h-gumbel`,
#     +0.061 probe, 62-45-93 at n=128).
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t12h-muon-wdl/chain.log
set -u
cd ~/brokefish
RUN=t12h-muon-wdl
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
say "1/3 train $RUN  (12 h, muon lr 0.02 CONSTANT, wd 0.09, WDL head, int8, gumbel m=16, n=128)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 10 \
  --optimizer muon --lr 0.02 --aux-lr 0.001 --adam-wd 0.09 \
  --ns-scheme polar --head-group 8 \
  --decay step --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
# ⚠️ Three runs, ONE fit. `t12h-wdl` isolates the optimiser; `t12h-muon9-int8` is the
# Muon baseline this recipe came from. The fit spans the scalar/WDL head change, which
# works only because a checkpoint's head width is read off `value.weight`
# (`nn/model.py:n_value_of`).
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-wdl + t12h-muon9-int8"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-wdl --run t12h-muon9-int8 \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-wdl+t12h-muon9-int8.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
