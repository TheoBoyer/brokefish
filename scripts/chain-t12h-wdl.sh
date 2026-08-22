#!/usr/bin/env bash
# 12 h: `t12h-flat`'s recipe with a **win/draw/loss value head**.
#
#   1/3  train t12h-wdl, 12 h
#   2/3  ONE joint league: t12h-wdl + t12h-flat
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--value-classes`**, 1 -> 3. Every other flag is
# copied verbatim from `chain-t12h-flat.sh`: AdamW lr 1e-3 with `--decay step` (a flat
# rate after the 30-step warmup), `--adam-wd` at its 0.01 default, gumbel m=16, n=128,
# 1024 games, 10 moves/phase, window 20000 / mean-plies 350, clip 1.0, terminal
# collapse, `--int8`, reuse at its 0.815 default.
#
# ⚠️ **The control is `t12h-flat` and not `t12h-nsched`**, even though nsched is the
# better run. Under a flat rate every checkpoint is a finished checkpoint, so the
# matched-wall-clock reading is interpretable at every hour; nsched carries a search
# schedule that confounds the compute axis (its Elo-per-decade came out *lower* at
# every budget for exactly that reason). One variable, and the schedule is not it.
#
# ## What the head is
#
# `--value-classes 3` trains the value head as a 3-way classifier -- 0 = loss,
# 1 = draw, 2 = win, from the side to move, class index `z + 1` -- with a
# cross-entropy, instead of one `tanh` logit with a squared error. It still hands the
# search a single fp32 in [-1, 1], `p(win) - p(loss)`, so the search, the terminal
# collapse, the probes and the whole Elo pipeline are bit-untouched.
# `docs/journal/2026-08-21-the-wdl-value-head.md`; measured free
# (`ledger/perf.md`: 1.0004x at B = 4096, interleaved).
#
# ## Why this knob
#
# The value head is the one part of this network never observed to train. Four 12 h
# runs, ~100 probes: the held-out value-puzzle probe went **0.3530 -> 0.3480 (x0.99)**
# while the policy probe went **0.0844 -> 0.4087 (x4.8)**, and that was independent of
# the learning-rate schedule (`t12h-flat`), of the sample reuse (`t12h-reuse2`) and of
# the search budget (`t12h-nsched` -- its n=256 phase did not move it either).
#
# The three mechanisms, none of them measured here: a squared error through a `tanh`
# has a gradient that vanishes as |x| grows *whether or not the prediction is right*,
# which is the documented `lr = 0.2` collapse; a draw becomes a class with its own mass
# instead of a coincidence at zero, in leagues that run 2-100 % draws; and KataGo and
# Leela both do it.
#
# ⚠️ **The honest prior.** This is the same class of hypothesis as `--value-weight 2.0`,
# which cost a 12 h run and came back **-231 Elo**. And the last three 12 h arms bought
# +8, +18 and +40 Elo -- all inside their error bars -- so the base rate for a 12 h
# single-knob arm on this ledger is "no measurable effect".
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * Control is **`t12h-flat`** in ONE joint Bradley-Terry fit. Both are 12 h at a
#     constant rate, so final-vs-final is matched wall clock by construction and every
#     intermediate checkpoint is a valid point on the cost curve.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ **The primary metric is the value-head puzzle probe, not Elo**, and this is the
#     only arm on this ledger for which that is true. `puzzles/value_pass@1` and
#     `puzzles/value_solve_rate` run in-loop at every checkpoint -- the value head
#     answering alone, no terminal collapse, no rules override, no search. `t12h-flat`
#     traced 0.3530 -> 0.3480 over its 12 h. **If this head does not move that number,
#     it did not do the thing it was built to do**, whatever the Elo says.
#   * ⚠️ The policy puzzle probe is reported and **not** weighted: its correlation with
#     dElo is +0.79 at n=16, +0.58 at n=64 and **+0.08 at n=256** across five paired
#     leagues.
#   * ⚠️ **`gradient/value` is not comparable to any run before this one.** An MSE
#     against z in {-1, 0, 1} starts near 1.0; a 3-class cross-entropy starts at
#     ln 3 = 1.0986 and floors at 0. `value_saturated_frac` likewise keeps its name and
#     changes its meaning to "a confident classifier". Neither may be read across the
#     branch, and in-buffer value MSE was already an **anti-signal** -- three cases
#     inverted against held-out value quality.
#   * ⚠️ `--value-weight` stays at 1.0 and is **inherited, not calibrated**. AGZ's 1:1
#     is justified for a unit-scaled squared error and that justification does not
#     transfer to a cross-entropy. KataGo runs c_value = 1.5. A second arm at a
#     different weight is a separate run, not a rescue of this one.
#   * ⚠️ Self-anchored Elo does not transfer: `t24h-adamw-int8` is +237 +/- 79 (z = 3.0)
#     over `t12h-int8` internally and **+12 +/- 34** against AlphaGateau.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
# ⚠️ No AlphaGateau match in this chain. Run it afterwards if the league warrants:
#     bash scripts/h2h-vs-ag.sh runs/t12h-wdl/checkpoints/t12h-wdl.pt t12h-wdl-vs-ag 200 128 0 0
#
#   tail -f runs/t12h-wdl/chain.log
set -u
cd ~/brokefish
RUN=t12h-wdl
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
say "1/3 train $RUN  (12 h, adamw CONSTANT lr 1e-3, int8, gumbel m=16, n=128, WDL head)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --gumbel --gumbel-m 16 --sims 128 \
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
# ⚠️ This fit spans the architecture change, which is only possible because a
# checkpoint's value-head width is read back off `value.weight` rather than assumed
# (`nn/model.py:n_value_of`). `t12h-flat`'s scalar-head checkpoints and the shared
# `checkpoints/anchor.pt` load unchanged.
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
