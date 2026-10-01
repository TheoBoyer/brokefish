#!/usr/bin/env bash
# 12 h: muon at the weight decay the two measured points actually imply, on int8.
#
#   1/3  train t12h-muon9, 12 h
#   2/3  ONE joint league: t12h-muon9 + t12h-int8
#   3/3  the curve
#
# ⚠️ **THIS IS THE THIRD MUON ATTEMPT AND THE SECOND ONE'S PREREGISTRATION SAID NOT TO
# RUN IT.** `2026-08-14-muon-with-decay.md`: *"The preregistration said a loss closes
# the optimiser line rather than inviting a third attempt, and by the pinned headline
# this is a loss. Honouring that is the whole reason for pinning it in advance."* That
# sentence is still true and this run is being launched anyway, on Théo's call. It is
# recorded here rather than quietly ignored, because the value of a preregistration is
# exactly that overriding it costs something visible.
#
# The argument for reopening, and it is not "we did not like the answer". `wd = 0.04`
# was sized to bring muon's weight norm to AdamW's 131 and **missed by 2.2×** -- it
# peaked at 292.9. So the hypothesis it was built to test was never tested at the value
# that would test it. The same entry says so: *"Whether `wd ~= 0.09` would finish the
# job. The norm target was missed by 2.2x, and nobody has run the arm that hits it."*
#
# `wd = 0.09` is not a guess either. Two points now fix `||W||* = c/wd`:
#
#     wd 0.01 -> 529 (still rising)      wd 0.04 -> 292.9 (peaked, then fell to 247.8)
#     c ~= 0.04 x 292.9 = 11.7   ->   at wd 0.09, ||W||* ~= 130
#
# which is AdamW's 131. ⚠️ **Watch `weight_norm` in the first two hours.** Peaking near
# 130 means the model is right and the rest of the run is the real test. Peaking well
# above means `||W||* = c/wd` is the wrong law and the mechanism is not what we think,
# which is worth knowing by hour three rather than hour twelve.
#
# ⚠️ Second thing to watch: `wd = 0.09` makes the equilibrium relative step `lr/||W||`
# roughly 2.2× larger than the `wd = 0.04` arm's, i.e. effectively a higher learning
# rate late in the run. If `kl` or `grad_norm` blows up early, that is the cause.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3, roadmap D5).
#   * Control is **`t12h-int8`**, not `t12h-gumbel`: same kernel, same 12 h, same cosine
#     shape, same Gumbel m=16, same n=128, same window, same batch, same collapse. Only
#     the optimiser and its decay move, which is what makes this one variable.
#   * Headline = final checkpoint at **n = 256**, ledger continuity.
#   * ⚠️ The headline is NOT the verdict if the budgets disagree; the full 16/64/256
#     grid is reported and a sign flip is itself the finding. `t12h-muong` gave
#     +22 / +156 / -51 and that disagreement was the result.
#   * Success = beat `t12h-int8` at equal wall clock.
#   * ⚠️ **A loss closes the optimiser line for good.** Three arms at three decays,
#     none of them beating AdamW, is not an open question any more -- and unlike last
#     time there is no "but the intervention undershot" left to appeal to, because
#     hitting the norm target is precisely what this arm does.
#
# ⚠️ Disk: 9.9 GB free at launch, and the run needs ~3.7 GB (3.1 buffer + 0.6 of kept
# checkpoints). It fits. `runs/t12h-int8/replay` is another 3.1 GB that can go if it
# gets tight -- that run is complete and journalled.
#
#   tail -f runs/t12h-muon9/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=t12h-muon9
mkdir -p runs/$RUN
LOG=runs/$RUN/chain.log
PY="uv run --no-project --python .venv/bin/python"

say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "chain start.  tail -f $LOG"
git rev-parse HEAD | tee -a "$LOG"
git diff > runs/$RUN/$RUN.diff; wc -l runs/$RUN/$RUN.diff | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"

# 1/3 -----------------------------------------------------------------------
# `--total-steps 4500` is `t12h-int8`'s, and correct for the same reason: same kernel,
# so the same ~16.8 s play + 2.2 s grad, so the same ~4 586 actual steps. The cosine
# then completes just before the clock and the run **ends annealed**, which is the
# property that makes it comparable to a control that also did.
#
# ⚠️ `--adam-wd` reaches the matrices, not just the AdamW side: `muon.py:279` and `:281`
# set `weight_decay: wd` on the muon group and the aux 2D group, and only `ndim < 2`
# keeps 0. That is the intent -- the norm growth is in the matrices.
say "1/3 train $RUN  (12 h, int8, muon lr=0.02 polar, wd 0.09, gumbel m=16, n=128)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 4500 \
  --games 1024 --moves-per-phase 10 \
  --optimizer muon --lr 0.02 --aux-lr 0.001 --head-group 8 --ns-scheme polar \
  --adam-wd 0.09 --lr-min 0.001 --decay cosine --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
say "2/3 joint league: $RUN + t12h-int8"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-int8 \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-int8.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
