#!/usr/bin/env bash
# 12 h: `t12h-reinject`'s recipe with re-injection at **both** norms of each block.
#
#   1/3  train t12h-reinject-both, 12 h
#   2/3  ONE joint league UNDER GUMBEL + COLLAPSE: this + t12h-reinject + t12h-wdl
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--reinject`**, ln1 -> both. Every other flag is
# copied verbatim from `chain-t12h-reinject.sh`, which copied them from
# `chain-t12h-wdl.sh`. The three arms therefore sit on a **ladder of the same knob**:
# 0 sites (`t12h-wdl`), 8 sites (`t12h-reinject`), 16 sites (this). That is what makes
# the three-way joint fit below a dose-response reading rather than three comparisons.
#
# ## What changes
#
# `ln1` injects at the attention norm of each block, 8 sites and 40 parameters. `both`
# adds the FFN norm: 16 sites, 80 parameters. The FFN then reads the raw inputs
# directly instead of only through attention's output.
#
# ⚠️ The coefficients still start at exactly zero, so this arm also **begins as the
# same function as `t12h-wdl`** and as `t12h-reinject`. All three start from one point.
#
# ## What it costs
#
# **+3.9 % of useful evals/s against no injection**, against `ln1`'s +1.86 %
# (`ledger/perf.md`, interleaved and order-balanced with identical trees). So expect
# roughly 2 % fewer steps in 12 h than `t12h-reinject` and ~3 % fewer than `t12h-wdl`.
# ⚠️ That is a real compute difference between the arms and the reason the step-matched
# reading below is not optional here.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md 3).
#   * Controls are **`t12h-reinject` and `t12h-wdl`** in ONE joint Bradley-Terry fit,
#     all three re-rated under Gumbel + collapse. One variable: the number of sites.
#   * **Primary = Elo at n = 256, final checkpoint**, ledger continuity, same as the
#     `ln1` arm. The full 16/64/256 grid is reported.
#   * ⚠️ **Secondary and reported beside it: the same fit read at matched steps.** This
#     arm is ~3 % slower per step than `t12h-wdl` by construction, so final-vs-final is
#     *not* compute-matched here the way it happened to be for `ln1`. Every 200-step
#     checkpoint is in the same fit, so both readings cost nothing extra.
#   * ⚠️ **The dose-response is the finding, not the pairwise sign.** Three points on
#     one knob: if 16 sites beats 8 beats 0, that is a different statement from any two
#     of them differing. If 8 beats both 0 and 16, that is also a statement.
#   * The value-head and policy puzzle probes are reported and **not** weighted. The
#     policy probe's correlation with dElo is +0.79 at n=16, +0.58 at n=64 and **+0.08
#     at n=256**.
#   * ⚠️ `reinject_c` is exempt from weight decay (`train/loop.py:no_decay`). It is
#     `[16, 5]` here rather than `[8, 5]`; the exemption is by name and covers both.
#   * ⚠️ Report the 80 learned coefficients. On the `ln1` arm they went large -- `square`
#     reached an effective 3.1x in the middle blocks and `clock` was cut to 0.25x in
#     block 0 -- so "did the FFN sites ask for anything the attention sites did not" is
#     a real question this run can answer directly.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t12h-reinject-both/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=t12h-reinject-both
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
say "1/3 train $RUN  (12 h, adamw CONSTANT lr 1e-3, int8, gumbel m=16, n=128, WDL head, reinject BOTH)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --reinject both \
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

# The 80 numbers. `reinject_c` is [16, 5]: site 2*L is block L's attention norm and
# 2*L+1 is its FFN norm; columns are square, type_special, color_turn, clock, rep.
say "1/3b the learned coefficients"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch
from brokefish.paths import run_dir
p = f"{run_dir('t12h-reinject-both')}/checkpoints/t12h-reinject-both.pt"
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
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-reinject + t12h-wdl"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-reinject --run t12h-wdl \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-reinject+t12h-wdl.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
