#!/usr/bin/env bash
# 12 h: `t12h-wdl`'s recipe with the value head pooling the **raw** residual and
# applying `norm_f` to the pooled vector -- `LN(mean(h))` instead of `mean(LN(h))`.
#
#   1/3  train t12h-prenorm, 12 h
#   2/3  ONE joint league, UNDER GUMBEL + COLLAPSE: + t12h-wdb + t12h-wdl
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--value-head`**, pooled -> prenorm, against
# `t12h-wdb`. Every other flag is copied from `chain-t12h-wdb.sh`, which copied
# `chain-t12h-wdl.sh`: AdamW lr 1e-3 with `--decay step` (flat after the 30-step
# warmup), `--adam-wd` 0.01, `--value-classes 3`, gumbel m=16, n=128, 1024 games,
# 10 moves/phase, window 20000 / mean-plies 350, clip 1.0, terminal collapse, `--int8`,
# reuse 0.815.
#
# ## Why, and it is one measured defect rather than a theory
#
# A mean of normed vectors is **not** normed. Measured on `t12h-wdb`:
#
#   |mean(LN(h))|            15.47 -> 10.86   over the run (token alignment 0.975 -> 0.765)
#   |value.weight|           1.645 ->  2.328   +42 %, compensating
#   value_saturated_frac     0.000 ->  0.111
#   held-out corr(v, z)      peak 0.5516 @2805 -> 0.4956 final
#
# The head was chasing an input whose scale shrank 30 % underneath it. `LN(mean(h))`
# has its scale set by the norm and cannot do that. Pooling before the norm also lets a
# token with a larger residual count for more -- a learned weighting that an unweighted
# mean of normed vectors cannot express, and the thing AlphaZero's per-square value
# head has.
#
# ⚠️ What was ruled out first, so this is not the fourth guess in a row:
#   * the gradients are correct -- directional finite differences against autograd on
#     all three heads, 5.9e-6 / 5.5e-5 / 2.5e-5 relative;
#   * weight decay is not it -- AdamW decoupled at 0.01 shrinks `value.weight` by 5 %
#     over 5000 steps, and the measurement shows it *growing* 42 %;
#   * the trunk did not lose the information -- probed frame-matched, `pool -> white`
#     is flat at 0.4466 / 0.4465 / 0.4451 across the entire wdb run;
#   * it is not memorisation -- the clean weight_gen split shows the head is *better*
#     on records it never trained on, by a constant margin.
#
# ⚠️ And what this arm does NOT claim: probed on frozen trunks, `LN(mean(h))` carries
# the same linearly decodable outcome as `mean(LN(h))`, within +/-0.015 at every
# checkpoint. The whole argument is about training **dynamics**. If the scale drift was
# not what cost the Elo, this comes back level with `t12h-wdb` and that is the finding.
#
# ⚠️ Cost, predicted then measured: predicted under 0.5 %, measured **+0.49 %**
# ABBA-interleaved at B = 4096 -- less than the pooled head's +0.80 %.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * **The league runs under `--gumbel --gumbel-m 16 --terminal-collapse`**, the
#     protocol these networks are trained under. Rating `t12h-wdl` under PUCT gave
#     **-14 Elo** and under Gumbel **+121** on the identical calendar
#     (`journal/2026-08-22-the-league-was-on-the-wrong-protocol.md`). ⚠️ Not joinable
#     with any league before 2026-08-22.
#   * Three runs, ONE fit. `t12h-wdb` is the **one-flag control**; `t12h-wdl` is the
#     king head, which is currently the best of the three at -0 Elo.
#   * Headline = final checkpoint at **n = 256**; the full 16/64/256 grid is reported.
#   * The mechanism check is `|mean|`, `|value.weight|` and held-out `corr(v, z)` on
#     the **file-pinned** yardstick over the ladder. ⚠️ That yardstick is `t12h-wdl`'s
#     own late self-play, so it is played at wdl's home ground; the Elo is not.
#   * ⚠️ The value-puzzle probe moves 0.59x its own noise within a run and is read
#     **final-vs-final only**. The policy probe correlates with dElo at +0.08 at n=256.
#   * ⚠️ The honest base rate: of six single-knob 12 h arms, one came back +121, one
#     -231, one -339, and three inside their bars.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint.
#
#   tail -f runs/t12h-prenorm/chain.log
set -u
cd ~/brokefish
RUN=t12h-prenorm
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
say "1/3 train $RUN  (12 h, adamw flat lr 1e-3, int8, gumbel m=16, n=128, POOLED W/D/B head)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --value-head prenorm \
  --gumbel --gumbel-m 16 --sims 128 \
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
# ⚠️ The fit spans two head changes at once (input and frame), which works only because
# a checkpoint's head is recovered from the file: its width from `value.weight` and its
# input from the presence of a `value_mode` buffer (`nn/model.py`). `t12h-flat`'s
# scalar checkpoints and the shared `checkpoints/anchor.pt` load unchanged.
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-wdb + t12h-wdl"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-wdb --run t12h-wdl \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-wdb+t12h-wdl.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
