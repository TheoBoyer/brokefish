#!/usr/bin/env bash
# 12 h: `t12h-reinject-both`'s recipe at **3x the learning rate**.
#
#   1/3  train t12h-reinject-both-lr3, 12 h
#   2/3  ONE joint league UNDER GUMBEL + COLLAPSE, FOUR arms
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--lr`**, 0.001 -> 0.003. Every other flag is
# copied verbatim from `chain-t12h-reinject-both.sh`. The control is therefore
# **`t12h-reinject-both`**, not `t12h-wdl`: against `t12h-wdl` two things differ and
# nothing about this run is interpretable that way.
#
# ## Why
#
# Re-injection adds a large direct path from the inputs to every block, and the
# coefficients it learned were not small -- `square` reached an effective **3.5x** in the
# middle blocks. A network whose conditioning has changed that much plausibly wants a
# different rate, and 0.001 was inherited from `chain-t12h-wdl.sh` rather than tuned for
# this architecture. That is the hypothesis: **not that the coefficients were
# rate-limited** (they travelled a long way: |c| mean 0.57, max 2.59), but that the
# whole network under re-injection is not at its best rate.
#
# ⚠️ **The honest prior.** `--reinject ln1` and `--reinject both` both came back with no
# measurable Elo (+13.4 and -42.5 at n=256, against CIs of ~±100), and lr is a knob this
# ledger has already found sensitive -- `docs/ledger/state.md` records `lr = 0.002`
# beating AZ's 0.2. 0.003 is 1.5x the value that note prefers and 3x what every 12 h arm
# here has run. It may simply train worse. `--grad-clip 1.0` is the only guard.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md 3).
#   * Control is **`t12h-reinject-both`** in ONE joint Bradley-Terry fit. One variable.
#   * **Primary = Elo at n = 256, final checkpoint**, ledger continuity.
#   * ⚠️ **Four arms in one fit, on purpose.** On 2026-08-29 the *same two runs, same
#     checkpoints, same protocol* moved **-101.0** and **-72.7** Elo purely from adding a
#     third arm to the pool -- `t12h-reinject`'s headline flipped -14.9 -> +13.4 with no
#     new information about it. Ratings are only comparable inside one fit, so all four
#     arms go in one, and the three older arms are re-read on this fit's scale.
#   * ⚠️ **A positive result here does not transfer to `t12h-wdl`.** If lr 0.003 helps,
#     the next question is whether it helps *without* re-injection, and that is a
#     separate 12 h arm that this one cannot substitute for.
#   * ⚠️ Both prior leagues report `converged: False` at the 10,000-iteration cap with
#     dispersion 0.82-0.84; nine of 32 leagues on this ledger do. A four-arm fit will not
#     be better behaved. The cap itself is worth investigating and has not been.
#   * The value-head and policy puzzle probes are reported and **not** weighted.
#   * ⚠️ Report the 80 learned coefficients, and compare them to
#     `t12h-reinject-both`'s: whether a 3x rate finds the *same* structure (square up,
#     clock suppressed at block 0, rep ignored) is a real question this run answers even
#     if the Elo does not move.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t12h-reinject-both-lr3/chain.log
set -u
cd ~/brokefish
RUN=t12h-reinject-both-lr3
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
say "1/3 train $RUN  (12 h, adamw CONSTANT lr 3e-3, int8, gumbel m=16, n=128, WDL head, reinject BOTH)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --reinject both \
  --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.003 --decay step --warmup 30 --grad-clip 1.0 \
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
p = f"{run_dir('t12h-reinject-both-lr3')}/checkpoints/t12h-reinject-both-lr3.pt"
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
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-reinject-both + t12h-reinject + t12h-wdl"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-reinject-both --run t12h-reinject --run t12h-wdl \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-reinject-both+t12h-reinject+t12h-wdl.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
