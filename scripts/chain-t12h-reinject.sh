#!/usr/bin/env bash
# 12 h: `t12h-wdl`'s recipe with **per-layer input re-injection**.
#
#   1/3  train t12h-reinject, 12 h
#   2/3  ONE joint league UNDER GUMBEL + COLLAPSE: this + t12h-wdl
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--reinject`**, none -> ln1. Every other flag is
# copied verbatim from `chain-t12h-wdl.sh`: `--value-classes 3`, AdamW lr 1e-3 with
# `--decay step` (a flat rate after the 30-step warmup), `--adam-wd` at its 0.01
# default, gumbel m=16, n=128, 1024 games, 10 moves/phase, window 20000 /
# mean-plies 350, clip 1.0, terminal collapse, `--int8`, reuse at its 0.815 default.
#
# ⚠️ **The league runs under Gumbel + collapse and `chain-t12h-wdl.sh`'s did not.**
# That script predates `journal/2026-08-22-the-league-was-on-the-wrong-protocol.md`,
# which found every league on this ledger rating PUCT while every run trained Gumbel.
# The joint fit below replays both arms from their checkpoints, so `t12h-wdl` is
# re-rated on the right protocol here rather than carried over. Its PUCT numbers do
# not transfer: the same calendar moved it from -14 to +121 at the headline.
#
# ## What the knob is
#
# `--reinject ln1` makes each block's LayerNorm see
#
#     h + sum_k c[site, k] * E_k
#
# instead of `h`, over the five embedding tables of spec 7.2 -- square, type_special,
# color_turn, clock, rep -- gathered again at the same indices, with a learned scalar
# per (site, source). 8 sites, **40 parameters**. The residual stream is untouched: the
# mix enters the norm's argument and nothing else. `spec.md` 7.1a is normative;
# `journal/2026-08-28-input-reinjection.md` is how it was built.
#
# ⚠️ **The coefficients start at exactly zero and the net is then the old net to the
# bit** -- verified on both the fp16 and int8 kernels at an odd batch size
# (`tests/test_b2.py::test_reinject_at_zero_is_the_old_kernel`). So this arm and its
# control start from the same function, not from two random draws, and the only thing
# that can separate them is what the optimiser does with 40 numbers.
#
# ## Why this knob
#
# Under pre-norm, everything a block knows about the position it must read out of the
# residual stream, and the stream is also carrying eight layers of accumulated
# computation. The per-board features are the sharp end of that: **the 50-move clock and
# the repetition count enter once, at layer 0, and have to survive to the value head**.
# Re-injection gives every block a direct read.
#
# ⚠️ **The honest prior.** The base rate for a 12 h single-knob arm on this ledger is
# "no measurable effect": the last three bought +8, +18 and +40 Elo, all inside their
# error bars, and `--value-weight 2.0` cost a run and came back **-231**. 40 parameters
# out of 6,383,360 is the smallest intervention yet attempted here.
#
# ## What it costs
#
# **+1.86 % of useful evals/s**, measured interleaved and order-balanced in the real
# MCTS with identical trees in both arms (`ledger/perf.md`). Against the *un-injected*
# kernel of 2026-08-28 morning it is net **-2.2 %**, because the profiling done for this
# feature also found 1.86 % in the int8 GEMM k-loop and 1.54 % in the quantiser.
#
# ⚠️ **Which is exactly why the headline is read twice.** `t12h-wdl` ran to **step 5055**
# and stopped on its 720-minute cap, not its 5200-step cap. The kernel is now ~4 %
# faster than it was that night, so 12 h no longer buys the compute it bought then and a
# final-vs-final reading silently hands this arm more steps. `--checkpoint-every 200`
# with `--keep-checkpoints` puts every intermediate checkpoint in the same joint fit, so
# both readings come out of one league at no extra cost.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md 3).
#   * Control is **`t12h-wdl`** in ONE joint Bradley-Terry fit, both re-rated under
#     Gumbel + collapse. One variable: `--reinject`.
#   * **Primary = Elo at n = 256, final checkpoint**, ledger continuity. The full
#     16/64/256 grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ **Secondary and reported beside it: the same fit read at ~step 5055**, the
#     control's own final step. If the two readings disagree the compute difference is
#     the explanation and the step-matched one is the architecture's number.
#   * The value-head puzzle probe and the policy probe are reported and **not**
#     weighted. The policy probe's correlation with dElo is +0.79 at n=16, +0.58 at
#     n=64 and **+0.08 at n=256**.
#   * ⚠️ **`reinject_c` is exempt from weight decay** (`train/loop.py:no_decay`) because
#     it is a table of gains starting at zero. Without that, AdamW at lr 1e-3 / wd 0.01
#     shrinks it 5 % over the run, and a Muon arm at lr 0.02 / wd 0.09 would multiply it
#     by exp(-9). Any future arm that moves the optimiser inherits this or is not
#     comparable to this one.
#   * ⚠️ Report the learned coefficients themselves. 40 numbers is few enough to print,
#     and "which sources did which depths ask for" is the only direct read on the
#     mechanism this run has. A run where they stay near zero has answered the question
#     as cleanly as one where they do not.
#   * ⚠️ Self-anchored Elo does not transfer: `t24h-adamw-int8` is +237 +/- 79 (z = 3.0)
#     over `t12h-int8` internally and **+12 +/- 34** against AlphaGateau.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
# ⚠️ No AlphaGateau match in this chain. Run it afterwards if the league warrants:
#     bash scripts/h2h-vs-ag.sh runs/t12h-reinject/checkpoints/t12h-reinject.pt \
#          t12h-reinject-vs-ag 200 128 0 0
#
#   tail -f runs/t12h-reinject/chain.log
set -u
cd ~/brokefish
RUN=t12h-reinject
CTRL=t12h-wdl
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
say "1/3 train $RUN  (12 h, adamw CONSTANT lr 1e-3, int8, gumbel m=16, n=128, WDL head, reinject ln1)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --reinject ln1 \
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

# The 40 numbers, printed. `reinject_c` is [8, 5]: rows are blocks 0-7, columns are
# square, type_special, color_turn, clock, rep -- the normative order of `EmbOff`.
say "1/3b the learned coefficients"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch
from brokefish.paths import run_dir
p = f"{run_dir('t12h-reinject')}/checkpoints/t12h-reinject.pt"
# ⚠️ The trainer's checkpoint nests the weights under "net" beside the optimiser
# state and the config; `reinject_c` is not at the top level.
c = torch.load(p, map_location="cpu", weights_only=False)["net"]["reinject_c"].float()
print("reinject_c  [block][square, type_special, color_turn, clock, rep]")
for i, row in enumerate(c):
    print(f"  block {i}: " + "  ".join(f"{v:+.4f}" for v in row))
print(f"  |c| mean {c.abs().mean():.4f}  max {c.abs().max():.4f}")
PYEOF

# 2/3 -----------------------------------------------------------------------
# ⚠️ This fit spans the architecture change, which is possible because a checkpoint's
# re-injection mode is read back off its own marker (`nn/model.py:reinject_of`), exactly
# as the value head's width is. `t12h-wdl`'s checkpoints and the shared
# `checkpoints/anchor.pt` carry no marker and load as `reinject = "none"`.
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + $CTRL"
$PY -m brokefish.eval.league \
  --run $RUN --run $CTRL \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+$CTRL.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
