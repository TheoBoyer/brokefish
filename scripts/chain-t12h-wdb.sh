#!/usr/bin/env bash
# 12 h: `t12h-wdl`'s recipe with the value head **pooled over every live token** and
# predicting **White / draw / Black** instead of the mover's result.
#
#   1/3  train t12h-wdb, 12 h
#   2/3  ONE joint league, UNDER GUMBEL + COLLAPSE: + t12h-wdl + t12h-flat
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS `--value-head`**, king -> pooled. Every other flag
# is copied from `chain-t12h-wdl.sh`: AdamW lr 1e-3 with `--decay step` (flat after the
# 30-step warmup), `--adam-wd` at its 0.01 default, `--value-classes 3`, gumbel m=16,
# n=128, 1024 games, 10 moves/phase, window 20000 / mean-plies 350, clip 1.0, terminal
# collapse, `--int8`, reuse at its 0.815 default.
#
# ## The hypothesis, in Theo's words
#
# The **policy** head is applied to all 32 tokens, so every token takes policy gradient
# directly. The **value** head is a row select of one token -- the side-to-move king --
# so value gradient reaches the other 31 only *through that token's attention*. That
# asymmetry is a candidate for why the value head has been the weak half.
#
# ⚠️ It is a bottleneck, not a disconnection: 8 non-causal layers do carry the gradient
# to every token. The asymmetry alone does not prove the bottleneck binds.
#
# `pooled` averages the final normed tokens over every **live** slot of **both**
# colours and feeds that to the head. Dead slots are excluded: a captured slot is
# `1 << 11` with colour, type and square wiped, so it decodes as a live white pawn on
# a1 (CLAUDE.md) and averaging it in would make the value track material lost by an
# accident of the encoding.
#
# A symmetric mean carries no notion of whose turn it is in *which token it read*, so
# the head predicts White's frame -- W/D/B -- and `heads` flips it into the mover's by
# the sign of the control word. Nothing downstream sees the difference.
#
# ## What it costs: nothing, and for a reason worth writing down
#
# `norm_f` is per token and sits **upstream** of the pool, and the heads are biasless,
# so `W @ mean(hn) == mean(W @ hn)` exactly. The kernel therefore averages the 32
# per-token value logits it *already computes* (warp 2 has produced all 32 aux columns
# on every forward since B2) instead of running a second GEMM on a pooled vector. No
# extra mma, no extra SMEM, no extra register. Measured kernel-vs-torch: the pooled
# path is **tighter** than the row select in fp16, 9.4e-4 against 3.9e-3, because
# averaging 32 logits cancels rounding.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * **The league runs under `--gumbel --gumbel-m 16 --terminal-collapse`**, the
#     protocol these networks are TRAINED under. Measured 2026-08-22: rating `t12h-wdl`
#     under PUCT gave **-14 Elo** and under Gumbel **+121** on the identical calendar
#     (`journal/2026-08-22-the-league-was-on-the-wrong-protocol.md`). ⚠️ Not joinable
#     with any league before that date.
#   * Three runs, ONE fit. `t12h-wdl` is the **one-flag control**; `t12h-flat` is the
#     original scalar king head, so the whole scalar -> WDL -> pooled path lands on one
#     scale.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * ⚠️ Reported alongside and **not** weighted: the policy puzzle probe correlates
#     with dElo at +0.79 (n=16), +0.58 (n=64), **+0.08 (n=256)**. The value-puzzle
#     probe moves **0.59x its own noise** within a run and is read **final-vs-final
#     only**, where it tracks held-out value quality at Spearman +0.96.
#   * The diagnostic that decides the *mechanism* is held-out `corr(v, z)` on the
#     **file-pinned** yardstick, run on the checkpoint ladder after the fact. ⚠️ Pinned
#     to a file because two ladders built from the same live ring hours apart are not
#     comparable and nothing in the numbers says so (2026-08-21).
#   * ⚠️ **New counter**: `gradient/value_logit_rms`. The pooled head's input is not
#     unit-scale -- `norm_f` normalises each token, the mean of normed vectors is not
#     normed, and its magnitude moves with how aligned the tokens are and with **how
#     many pieces are alive**. This is the cheap downstream proxy for that drift.
#   * ⚠️ `gradient/value` is not comparable across heads (MSE vs a 3-class CE starting
#     at ln 3 = 1.0986), and in-buffer value MSE has been an **anti-signal** three
#     times.
#   * ⚠️ `--value-weight` stays 1.0, inherited rather than calibrated for a
#     cross-entropy. KataGo runs c_value = 1.5.
#   * ⚠️ The honest base rate: of the last five single-knob 12 h arms, four came back
#     inside their error bars and one (`t12h-vw2`) came back -231 Elo.
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t12h-wdb/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=t12h-wdb
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
  --run $RUN --value-classes 3 --value-head pooled \
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
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-wdl + t12h-flat"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-wdl --run t12h-flat \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-wdl+t12h-flat.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
