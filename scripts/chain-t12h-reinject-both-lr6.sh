#!/usr/bin/env bash
# 12 h: `t12h-reinject-both-lr3`'s recipe at **6x the original rate, with the ramp slope
# held constant**.
#
#   1/4  train t12h-reinject-both-lr6, 12 h
#   2/4  ONE joint league UNDER GUMBEL + COLLAPSE, FIVE arms
#   3/4  the curve
#   4/4  fit-free h2h against t24h-adamw-int8, the best checkpoint we have
#
# ⚠️ **TWO FLAGS MOVE AND THEY ARE ONE IDEA**: `--lr` 0.003 -> 0.006 and `--warmup`
# 30 -> 180. `lr_at` ramps as `peak * (step + 1) / warmup_steps`, so `--warmup 30`
# fixes the number of steps and *not* the slope: at 6x the peak the ramp climbs 6x
# steeper and hands over to a rate the network has never seen, after the same 30 steps.
# 180 restores the baseline's climb exactly -- 0.001/30 = 3.33e-05 of lr per step, the
# same per-step increment `t12h-wdl` used -- and reaches the peak at 3.5 % of the run.
# Holding warmup at 30 would have been the *bigger* change, not the smaller one.
#
# ## Why the slope and not the step count
#
# ⚠️ The ramp exists for a **measured** Adam failure, not as a formality. Adam's update
# is gradient-*normalised*, so every parameter moves about `lr` per step however small
# its gradient; the value head's pre-tanh activation sums 256 of those. At `lr = 1e-3`
# **step 1 alone** drove it into saturation -- `value_saturated_frac = 1.0`,
# `grad_value_head = 0.000`, dead for 62 of 116 steps (`t15-adamw`, 2026-07-31).
# At peak 0.006 with warmup 30, step 1's rate is 2.0e-04: six times the dose that
# caused that failure. With warmup 180 it is 3.33e-05, the dose that did not.
#
# ## Why 6x at all
#
# 0.001 -> 0.003 moved the value puzzle probe **0.3690 -> 0.4091**, the best on the
# ledger, and the four-way fit put `lr3` +43.2 over `t12h-wdl` at n = 256. Whether that
# continues is the question.
#
# ⚠️ **The honest prior, and it is bad.** `lr3` scored **0.4225 (-54 Elo)** against
# `t24h-adamw-int8` fit-free over 200 games, and ~0.336 against AlphaGateau at 76 games
# before that match was stopped. Its weight norm ran **1.83x** the lr-0.001 control's
# and climbing linearly, and `journal/2026-08-24-...` ties exactly that quantity to the
# value head's over-confidence failure. 6x is the axis most likely to break, and the
# value probe has now joined the internal signals that do not transfer (+237 -> +12,
# Muon +143 -> negative, lr3's probe -> -54).
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md 3).
#   * Control is **`t12h-reinject-both-lr3`** in ONE joint fit. The other three arms ride
#     along so all five sit on one scale.
#   * **Primary = Elo at n = 256, final checkpoint.**
#   * ⚠️ **Stage 4/4 is the reading I would trust if the two disagree.** 200 games at
#     n = 128 against our best checkpoint, no Bradley-Terry anywhere, 60 s of GPU. On
#     2026-08-30 the four-way fit said `lr3` was +43.2 and the fit-free match said -54;
#     the fit's pool had moved a headline 100 Elo the day before. `lr3`'s 0.4225 is the
#     number to beat and it is measured on this exact protocol.
#   * The value-head and policy puzzle probes are reported and **not** weighted, and are
#     read at the **final checkpoint** -- not averaged over checkpoints and not over a
#     window chosen after seeing the data. Both of those were mine and both flattered a
#     result that then reversed.
#   * ⚠️ Report the 80 coefficients and their correlation with `lr3`'s. At 3x they came
#     back r = 0.88 against the 1x run with 2.3x the magnitude; whether 6x keeps the
#     structure or destroys it is a real question independent of the Elo.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
#   tail -f runs/t12h-reinject-both-lr6/chain.log
set -u
cd ~/brokefish
RUN=t12h-reinject-both-lr6
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
say "1/3 train $RUN  (12 h, adamw CONSTANT lr 6e-3, warmup 180, int8, gumbel m=16, n=128, WDL head, reinject BOTH)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --reinject both \
  --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.006 --decay step --warmup 180 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# The 80 numbers. `reinject_c` is [16, 5]: site 2*L is block L's attention norm and
# 2*L+1 is its FFN norm; columns are square, type_special, color_turn, clock, rep.
say "1/3b the learned coefficients"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch
from brokefish.paths import run_dir
p = f"{run_dir('t12h-reinject-both-lr6')}/checkpoints/t12h-reinject-both-lr6.pt"
c = torch.load(p, map_location="cpu", weights_only=False)["net"]["reinject_c"].float()
print("reinject_c  [block][norm]  square  type_special  color_turn  clock  rep")
for i, row in enumerate(c):
    tag = f"block {i // 2} {'ln1' if i % 2 == 0 else 'ln2'}"
    print(f"  {tag:<13}: " + "  ".join(f"{v:+.4f}" for v in row))
ln1, ln2 = c[0::2], c[1::2]
print(f"  |c| mean {c.abs().mean():.4f}  max {c.abs().max():.4f}")
print(f"  attention sites |c| mean {ln1.abs().mean():.4f}   "
      f"FFN sites |c| mean {ln2.abs().mean():.4f}")
PYEOF

# 2/3 -----------------------------------------------------------------------
# ⚠️ Three arms, one fit. Each checkpoint's re-injection mode is read back off its own
# marker (`nn/model.py:reinject_of`), so `none` / `ln1` / `both` checkpoints and the
# shared `checkpoints/anchor.pt` all load into the right architecture from one loop.
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-reinject-both-lr3 + t12h-reinject-both + t12h-reinject + t12h-wdl"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-reinject-both-lr3 --run t12h-reinject-both --run t12h-reinject --run t12h-wdl \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-reinject-both-lr3+t12h-reinject-both+t12h-reinject+t12h-wdl.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"

# 4/4 -----------------------------------------------------------------------
# ⚠️ **Fit-free, and it is the stage that mattered on 2026-08-30.** 200 games at
# n = 128 against the best checkpoint we have, no Bradley-Terry anywhere. `lr3` read
# +43.2 over `t12h-wdl` in the four-way fit and scored **0.4225 (-54 Elo)** here; the
# two disagree because the fit's pool moved a headline by 100 Elo the day before.
# 60 seconds of GPU, directly comparable to lr3's number.
say "4/4 h2h vs t24h-adamw-int8 (fit-free, 200 games, n=128)"
bash scripts/h2h-bf.sh \
  runs/$RUN/checkpoints/$RUN.pt \
  runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt \
  $RUN-vs-t24h 200 128 >> "$LOG" 2>&1
say "4/4 done (exit $?)"
grep -E "^A scores" logs/h2h-$RUN-vs-t24h.log | tee -a "$LOG"
