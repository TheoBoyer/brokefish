#!/usr/bin/env bash
# 24 h: `t12h-reinject-both-lr6`'s recipe, doubled. **ONE VARIABLE MOVES: `--minutes`.**
#
#   1/4  train t24h-reinject-lr6, 24 h
#   2/4  ONE joint league UNDER GUMBEL + COLLAPSE, four arms
#   3/4  the curve
#   4/4  fit-free h2h against t24h-adamw-int8, the best checkpoint we have
#
# ## Why this run and not another 12 h
#
# ⚠️ Every re-injection arm so far has been **compute-handicapped against the number it
# was being judged by**. `lr6` scored 0.4525 (-33 Elo) against `t24h-adamw-int8` over
# 200 fit-free games -- a CI that includes parity -- but it is a 12 h run and that is a
# 24 h run. Reading it as "-33 Elo behind" is reading a 0.3-decade compute deficit as an
# architecture result. On this ledger a 12 h -> 24 h step has previously been worth
# ~218 Elo (`t12h-int8` -> `t24h-adamw-int8`).
#
# This run removes that confound. `lr6` reached **step 4926** in 12 h and
# `t24h-adamw-int8` reached **step 10218** in 24 h, so at 24 h this arm should land near
# 10 000 steps: **matched in wall clock and approximately matched in steps** against the
# strongest checkpoint we have. That has not been true of any comparison in this series.
#
# ## What is held fixed
#
# Every flag is copied verbatim from `chain-t12h-reinject-both-lr6.sh`: `--reinject
# both`, AdamW peak 0.006 with `--warmup 180` (the ramp slope of the 0.001 baseline,
# 3.33e-05 of lr per step -- see that script for the measured Adam saturation failure
# the ramp exists for), `--decay step` so the rate is flat after the ramp, gumbel m=16,
# n=128, 1024 games, 10 moves/phase, window 20000 / mean-plies 350, clip 1.0, terminal
# collapse, `--int8`, `reinject_c` exempt from weight decay (`train/loop.py:no_decay`).
#
# `--total-steps 10400` is 2x the 12 h arm's cap and deliberately does not bind: the run
# stops on its 1440-minute cap, as `lr6` stopped on its 720.
#
# ⚠️ **`--decay step` means there is no anneal.** `t24h-adamw-int8` ran a cosine to
# `lr_min`; this run ends at 0.006 flat. That is a real difference between the two 24 h
# runs and it is inherited from the 12 h ladder, not chosen here. It cuts against this
# arm at the finish line, and a future anneal arm is the obvious follow-up if the
# coefficients hold up.
#
# ⚠️ **The honest prior.** The weight norm at 3x ran 1.83x the 0.001 control's and was
# climbing linearly; `journal/2026-08-24-...` ties exactly that quantity to a value-head
# over-confidence failure. At 6x for twice as long, a blow-up somewhere past step 5000 is
# the single most likely way this run ends badly, and no 6x arm has ever run that far.
# The coefficients themselves doubled at every rate step (|c| mean 1.27 -> 2.25, max
# 6.49 -> 11.76) with r = 0.91 structure retained; whether that saturates or diverges
# over 10 000 steps is unmeasured.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md 3).
#   * **Stage 4/4 is the headline.** 200 games at n = 128 against `t24h-adamw-int8`,
#     no Bradley-Terry anywhere, compute-matched for the first time. The numbers to beat
#     on this exact protocol are `lr3` 0.4225 (-54) and `lr6` 0.4525 (-33), both at 12 h.
#     **0.500 is parity with the best checkpoint in the project.**
#   * Secondary = Elo at n = 256, final checkpoint, in ONE joint fit with
#     `t24h-adamw-int8` (the 24 h control), `t12h-reinject-both-lr6` (the time
#     doubling) and `t12h-wdl` (the ladder's origin). The 16/64/256 grid is reported.
#   * ⚠️ **If the fit and the match disagree, the match wins.** They disagreed on `lr3`
#     (+43.2 in the four-way fit, -54 fit-free) and the fit's pool had moved a headline
#     100 Elo the day before. The five-way fit's +187.9 for `lr6` at n=256 against
#     +36.3 at n=64, with `converged: False`, is the same instrument misbehaving.
#   * The value-head and policy puzzle probes are reported and **not** weighted, read at
#     the **final checkpoint** -- not averaged, not over a window chosen after seeing the
#     data. ⚠️ The value probe has now failed as a proxy in both directions: it went
#     0.4091 -> 0.4028 from `lr3` to `lr6` while strength went -54 -> -33.
#   * ⚠️ Report the 80 coefficients, their correlation with `lr6`'s, and the weight norm
#     trace. "Did 10 000 steps at 6x saturate or diverge" is a question this run answers
#     directly and independently of the Elo.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t24h-reinject-lr6/chain.log
set -u
cd ~/brokefish
RUN=t24h-reinject-lr6
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
say "1/4 train $RUN  (24 h, adamw CONSTANT lr 6e-3, warmup 180, int8, gumbel m=16, n=128, WDL head, reinject BOTH)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --reinject both \
  --gumbel --gumbel-m 16 --sims 128 \
  --minutes 1440 --total-steps 10400 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.006 --decay step --warmup 180 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/4 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# The 80 numbers, and whether 24 h at 6x kept `lr6`'s structure. `reinject_c` is
# [16, 5]: site 2*L is block L's attention norm and 2*L+1 is its FFN norm; columns are
# square, type_special, color_turn, clock, rep.
say "1/4b the learned coefficients"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch
from brokefish.paths import run_dir
p = f"{run_dir('t24h-reinject-lr6')}/checkpoints/t24h-reinject-lr6.pt"
c = torch.load(p, map_location="cpu", weights_only=False)["net"]["reinject_c"].float()
print("reinject_c  [block][norm]  square  type_special  color_turn  clock  rep")
for i, row in enumerate(c):
    tag = f"block {i // 2} {'ln1' if i % 2 == 0 else 'ln2'}"
    print(f"  {tag:<13}: " + "  ".join(f"{v:+.4f}" for v in row))
ln1, ln2 = c[0::2], c[1::2]
print(f"  |c| mean {c.abs().mean():.4f}  max {c.abs().max():.4f}")
print(f"  attention sites |c| mean {ln1.abs().mean():.4f}   "
      f"FFN sites |c| mean {ln2.abs().mean():.4f}")

# Structure against the 12 h arm at the same rate: did doubling the time keep the
# shape and only scale it, as every rate step did, or did it reorganise?
q = f"{run_dir('t12h-reinject-both-lr6')}/checkpoints/t12h-reinject-both-lr6.pt"
d = torch.load(q, map_location="cpu", weights_only=False)["net"]["reinject_c"].float()
a, b = c.flatten(), d.flatten()
r = ((a - a.mean()) * (b - b.mean())).sum() / (a.std(unbiased=False) * b.std(unbiased=False) * a.numel())
print(f"  vs t12h-reinject-both-lr6: Pearson r {r:.4f}   "
      f"|c| mean ratio {(c.abs().mean() / d.abs().mean()):.4f}")
PYEOF

# 2/4 -----------------------------------------------------------------------
# ⚠️ Four arms, one fit, one scale. Each checkpoint's re-injection mode is read back off
# its own marker (`nn/model.py:reinject_of`), so the `none` checkpoints of
# `t24h-adamw-int8` / `t12h-wdl`, the `both` checkpoints here, and the shared
# `checkpoints/anchor.pt` all load into the right architecture from one loop.
say "2/4 joint league UNDER GUMBEL+COLLAPSE: $RUN + t24h-adamw-int8 + t12h-reinject-both-lr6 + t12h-wdl"
$PY -m brokefish.eval.league \
  --run $RUN --run t24h-adamw-int8 --run t12h-reinject-both-lr6 --run t12h-wdl \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/4 done (exit $?)"

# 3/4 -----------------------------------------------------------------------
say "3/4 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t24h-adamw-int8+t12h-reinject-both-lr6+t12h-wdl.json >> "$LOG" 2>&1
say "3/4 done (exit $?)"

# 4/4 -----------------------------------------------------------------------
# ⚠️ **This is the headline of the run.** 200 games at n = 128 against the best
# checkpoint we have, no Bradley-Terry anywhere, and for the first time in this series
# the two sides had the same wall clock. 60 seconds of GPU.
say "4/4 h2h vs t24h-adamw-int8 (fit-free, 200 games, n=128) -- compute-matched"
bash scripts/h2h-bf.sh \
  runs/$RUN/checkpoints/$RUN.pt \
  runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt \
  $RUN-vs-t24h 200 128 >> "$LOG" 2>&1
say "4/4 done (exit $?).  chain finished"
grep -E "^A scores" logs/h2h-$RUN-vs-t24h.log | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
