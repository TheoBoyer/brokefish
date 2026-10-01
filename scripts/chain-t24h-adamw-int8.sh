#!/usr/bin/env bash
# 24 h: the best AdamW recipe we have, on the int8-everywhere kernel, then AlphaGateau.
#
#   1/4  train t24h-adamw-int8, 24 h
#   2/4  ONE joint league: t24h-adamw-int8 + t12h-int8 + t12h-gumbel
#   3/4  the curve
#   4/4  direct head to head against AlphaGateau's released checkpoint
#
# ⚠️ **This pauses the Muon track.** The reason is not yesterday's diagnostics, it is
# the one clean optimiser comparison in the project: `t24h-fp8` (AdamW) against
# `t24h-muon` (Muon) at matched `--adam-wd 0.01`, same PUCT self-play, same fp8, same
# 128 sims, ~8 000 steps each, measured **+84 / +75 / +152 Elo to AdamW** at
# n = 16/64/256 -- the largest margin in the paired-league table and 2.6 sigma at 256.
# Muon's founding claim (+143 Elo, `docs/journal/2026-08-07-muon-curve.md`) was measured
# at **step 1905** against an unscreened AdamW rate, and CLAUDE.md already flags it as
# provisional. The long horizon inverts it. Muon has never had an lr or wd sweep and
# that remains a fair defence; it is several arms and it is not the road to AlphaGateau.
#
# ## Why this configuration
#
# The base is `chain-t12h-int8.sh` -- AdamW lr 1e-3, `--adam-wd` at its 0.01 default,
# gumbel m=16, n=128 -- because `t12h-int8` beat `t12h-gumbel` at n = 64 and 256 and is
# our strongest AdamW arm. Two things move from it, both deliberate:
#   * **24 h instead of 12 h**, which is the point of the run.
#   * **`--int8` now selects all four weight matmuls** rather than the FFN's two. The
#     kernel measured 1.174x inside the real MCTS (92 639 against 78 890 evals/s), and
#     `t12h-muon9-int8` got 5 010 steps in 12 h on it, so 24 h should reach ~10 000.
#     `--total-steps 9800` deliberately **under**-shoots so the cosine completes and the
#     tail runs flat at `lr_min` rather than leaving the run un-annealed.
#
# ## Why step 4 exists and why it is last
#
# ⚠️ `logs/gate2-h2h-alphagateau.json`: `t12h-gumbel-004009` already played AlphaGateau's
# released checkpoint at n = 128 both sides, 200 games, and scored **0.400 -- 20 W /
# 120 D / 60 L, -70 Elo** (CI 0.335-0.469). That is a 12 h AdamW run 70 Elo behind. At
# the measured +592 to +707 Elo per 10x training time, 12 h -> 24 h is 0.30 decades and
# ~+180-210 Elo. So this run has a real chance of passing that checkpoint, and the head
# to head is the **only** measurement that says so without a rating conversion --
# CLAUDE.md's restated Gate 2 level half, exactly.
#
# It runs **after** the curve so that a server crash costs nothing that already exists.
# It is also the step most likely to fail unattended: the AG server has died mid-sweep
# before, hence the retry loop, and two models share an 8 GB card.
#
# ⚠️ Beating this checkpoint is **not** "beating AlphaGateau". Their `rankings.json`
# lists a stronger 8-layer model at 2366 +/- 25 which they do not release; the one we
# play tops out near 2156. The claim available to us is the released checkpoint.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3).
#   * Controls are **`t12h-int8`** (same recipe, half the time, int8 FFN only) and
#     **`t12h-gumbel`** (same search, AdamW, fp8), both in ONE joint Bradley-Terry fit
#     so all three sit on one scale.
#   * Headline = final checkpoint at **n = 256**, ledger continuity; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding.
#   * The **AlphaGateau head to head** is reported as its own number, on its own
#     protocol, and is not converted into the league's scale.
#   * ⚠️ Reported and NOT weighted: the policy puzzle probe. Measured over five paired
#     leagues its correlation with dElo is +0.79 at n=16, +0.58 at n=64 and **+0.08 at
#     n=256** -- at the headline budget it is a coin flip. It also mispredicts the AG
#     gap badly: AG is +0.11 pass@1 over `t12h-gumbel` and only +70 Elo.
#   * ⚠️ In-buffer value MSE is reported as an **anti-signal**. Three cases now have it
#     inverted against held-out value quality, most recently `t12h-vw2` (19 % better
#     buffer MSE, -0.045 value-puzzle, -231 Elo).
#   * ⚠️ No result here closes or opens a line. That call is Theo's.
#
# ⚠️ `evaluation.md`: nothing here may select a checkpoint. The final checkpoint is the
# headline because it is final, not because it scored best.
#
#   tail -f runs/t24h-adamw-int8/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
REPO=$PWD
AG_DIR=${AG_DIR:-$REPO/../alphagateau}
RUN=t24h-adamw-int8
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
say "1/4 train $RUN  (24 h, adamw lr=1e-3, int8 all four, gumbel m=16, n=128)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 1440 --total-steps 9800 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.001 --lr-min 5e-5 --decay cosine --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/4 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/4 -----------------------------------------------------------------------
say "2/4 joint league: $RUN + t12h-int8 + t12h-gumbel"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-int8 --run t12h-gumbel \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/4 done (exit $?)"

# 3/4 -----------------------------------------------------------------------
say "3/4 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-int8+t12h-gumbel.json >> "$LOG" 2>&1
say "3/4 done (exit $?)"

# 4/4 -----------------------------------------------------------------------
# Same protocol as logs/gate2-h2h-alphagateau.json so the two numbers compare directly:
# n = 128 both sides, 200 games over 100 openings, 8 opening plies, 300-ply cap.
say "4/4 head to head vs alphagateau (released checkpoint, n=128 both sides)"
CKPT=runs/$RUN/checkpoints/$RUN.pt

start_servers() {
  systemctl --user stop ag-h2h.scope bf-h2h.scope 2>/dev/null
  pkill -f serve_ag 2>/dev/null; pkill -f serve_brokefish 2>/dev/null; sleep 3
  : > logs/serve-ag-h2h.log; : > logs/serve-bf-h2h.log
  ( cd "$AG_DIR" && systemd-run --user --scope -p MemoryMax=8G \
      --unit=ag-h2h --quiet env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 .venv/bin/python "$REPO/scripts/alphagateau/serve_ag.py" \
      --ckpt models/chess_2024-08-20:00h13/000499.ckpt --port 8085 --pad 64 \
      --warm-n 0 > "$REPO"/logs/serve-ag-h2h.log 2>&1 & )
  ( systemd-run --user --scope -p MemoryMax=6G --unit=bf-h2h --quiet \
      $PY scripts/serve_brokefish.py --ckpt "$CKPT" --port 8081 \
      > logs/serve-bf-h2h.log 2>&1 & )
  for i in $(seq 1 60); do
    grep -q "ready on" logs/serve-ag-h2h.log 2>/dev/null \
      && grep -q "ready on" logs/serve-bf-h2h.log 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

for attempt in 1 2 3; do
  say "4/4 attempt $attempt"
  if start_servers; then
    tail -1 logs/serve-ag-h2h.log | tee -a "$LOG"
    tail -1 logs/serve-bf-h2h.log | tee -a "$LOG"
    $PY -m scripts.arbiter \
      --a http://127.0.0.1:8081 --b http://127.0.0.1:8085 \
      --games 200 --n 128 --width 64 --opening-plies 8 --max-plies 300 \
      --out logs/h2h-$RUN-vs-ag.json --pgn logs/h2h-$RUN-vs-ag.pgn \
      >> "$LOG" 2>&1
    [ -s logs/h2h-$RUN-vs-ag.json ] && break
  else
    echo "servers did not come up" | tee -a "$LOG"
  fi
done
pkill -f serve_ag 2>/dev/null; pkill -f serve_brokefish 2>/dev/null
say "4/4 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
