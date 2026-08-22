#!/usr/bin/env bash
# 12 h: t12h-muon9-int8's recipe exactly, with the value term weighted 2.0.
#
#   1/3  train t12h-vw2, 12 h
#   2/3  ONE joint league: t12h-vw2 + t12h-muon9-int8
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--value-weight`.** Every other flag is copied
# character for character from `chain-t12h-muon9-int8.sh`: muon lr 0.02 polar,
# `--adam-wd 0.09`, aux lr 0.001, head-group 8, gumbel m=16, n=128, 1024 games, 10
# moves/phase, window 20000 / mean-plies 350, cosine to `lr_min` 0.001, warmup 30, clip
# 1.0, terminal collapse, `--int8`, 5000 steps. Same kernel, so the step target does not
# need resizing.
#
# ## Why this knob, measured 2026-08-17
#
# `az_loss` is `policy_CE + value_weight * MSE` with `value_weight = 1.0`, AlphaZero's,
# never swept in this project (`loss.py:216`, `loop.py:286`). The value term's *share*
# of the loss is not configured, though -- it emerges -- and it differs between the arms:
# mean over the last 200 gradient steps, `value/policy` is **0.342** in `t12h-gumbel`
# (AdamW) and **0.243** in both muon arms, while `grad_value_head` falls 0.379 -> 0.294
# and `grad_policy_head` is flat at 0.127-0.133.
#
# Three diagnostics say that share matters and locate the damage:
#
#   * On **outcome prediction** (361 games, split by game, ridge on frozen `h_king`) the
#     representation is equally good everywhere: probe 0.807-0.836 across all six runs
#     measured. Muon's trained value head is *not* behind AdamW's on this task
#     (0.7905 vs 0.7754). The coarse signal is intact.
#   * On the **value-head puzzle task** -- ranking ~30 sibling positions one ply apart,
#     which is where the 0.08 pass@1 gap actually is -- refitting the readout on the
#     frozen trunk with the ranking loss, initialised at the run's own `value.weight`,
#     recovers **+0.035** for `t12h-muon9` and **+0.031** for `t12h-muon9-int8`, and
#     **-0.012** for `t12h-gumbel`. AdamW's head is already at its trunk's ceiling; the
#     muon heads are not. That half of the gap is a fitting deficit.
#   * The other half is representational and it is **not the weight decay**. At matched
#     `--adam-wd 0.01`, same PUCT self-play, same fp8: `t24h-fp8` (AdamW) refit ceiling
#     0.4800, `t24h-muon` (Muon) 0.4569. The decay does something else and it is
#     separable -- participation ratio of `h_king` is 14-15 at wd 0.01 (both optimisers)
#     and 9.8-10.6 at wd 0.04-0.09, and a **pooled** readout recovers most of it
#     (`t12h-muong` king 0.4423 -> pool 0.4670). So wd concentrates the king token;
#     Muon costs ~0.023 of ceiling on top, at any wd.
#
# `--value-weight` is the direct lever on the emergent share, and it is the only one
# that costs nothing but a run.
#
# ⚠️ **2.0 and not 1.41.** Restoring parity with the AdamW arm's loss share would be
# 0.342/0.243 = 1.41; 2.0 is chosen instead because a 12 h arm should move the knob far
# enough to be readable against a league whose 1-sigma has been 70-129 Elo on this axis.
# A monotone effect is then measurable in the direction, and 0.5 -- AlphaGateau's actual
# value, `optax.l2_loss` being `0.5(x-y)^2` -- is the opposite arm if this one reads.
#
# ⚠️ **The honest prior is that the league cannot resolve this.** Every intervention
# measured on this axis has come back inside its own error bar: `t12h-int8` +20 +/- 129,
# `t12h-pcr` +36 +/- 70, `t12h-muon9-int8` a rerun that inverted its own puzzle ranking
# in games. What this run *can* resolve, because the instruments are much tighter than
# the league, is the **value-head half**: the refit-ceiling gap and the puzzle value
# score have standard errors near 0.008, so a change of 0.03 is visible.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * Control is **`t12h-muon9-int8`**: same optimiser, lr, decay, scheme, search,
#     window, batch, collapse, cosine shape, kernel and step target. Only
#     `--value-weight` moves.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ Reported alongside, and these are the tighter instruments here:
#       - value-head puzzle pass@1 and solve_rate (`scripts/puzzle_value.py`, 20 000),
#       - the refit-ceiling gap of `_vpuz3.py` (does the head reach its trunk's ceiling),
#       - `gradient/value`, `gradient/policy` and their ratio, `grad_value_head`,
#       - participation ratio of `h_king` (`_vpool.py`).
#   * ⚠️ The policy puzzle probe is **reported and not weighted**: `n_sims = 0` is the
#     raw policy and it is 0-for-4 at predicting n = 256. It is an instrument for the
#     small-n end and is read as that.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: none of the above may select a checkpoint. The final checkpoint is
# the headline because it is the final one, not because it scored best.
#
#   tail -f runs/t12h-vw2/chain.log
set -u
cd ~/brokefish
RUN=t12h-vw2
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
say "1/3 train $RUN  (12 h, int8 all four, muon lr=0.02 polar, wd 0.09, gumbel m=16, n=128, value-weight 2.0)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5000 \
  --games 1024 --moves-per-phase 10 \
  --optimizer muon --lr 0.02 --aux-lr 0.001 --head-group 8 --ns-scheme polar \
  --adam-wd 0.09 --lr-min 0.001 --decay cosine --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 --value-weight 2.0 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
say "2/3 joint league: $RUN + t12h-muon9-int8"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-muon9-int8 \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-muon9-int8.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
