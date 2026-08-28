#!/usr/bin/env bash
# 12 h: t12h-muon-wdl's recipe + playout cap randomisation, one variable.
#
#   1/3  train t12h-muon-wdl-pcr, 12 h
#   2/3  ONE joint league under Gumbel+collapse: this + t12h-wdl (AdamW) + t12h-muon-wdl (Muon, no PCR)
#   3/3  the curve
#
# Why (2026-08-24, scratchpad offline harness, see the journal when it is written):
# on a FIXED replay buffer, same init, Muon fits the per-game outcome bit of *seen*
# games (train corr 0.81 vs AdamW 0.62) and loses it on unseen games (0.49 vs 0.59) --
# game-level memorisation of `z`, one bit repeated on ~118 positions of each game.
# Not weight decay (wd .01 = .09), not the NS scheme, not the lr, not the head, not
# the kernel. PCR stores only the full-search 25 % of positions, so each game's bit
# is repeated ~30 times instead of ~118, and games are ~2x cheaper (more distinct bits
# per hour).
#
# ⚠️ Cadence rides positions *generated* (`loop.py:670`), so at pcr_p = 0.25 each
# stored record is trained ~3.3x rather than 0.815x. Théo's call (steps-matched, as
# `t12h-pcr` ran): the repetition is partly put back through reuse. If the value probe
# still declines here, the reuse-matched arm (`--samples-per-position 0.204`) is next.
#
# ⚠️ PREREGISTERED: the measurement is (a) the in-run value probe -- does it stop
# declining (t12h-muon-wdl: 0.343 @1002 -> 0.287 @5010; t12h-wdl: flat 0.35-0.38) --
# and (b) the joint Gumbel league at n = 16/64/256 against both controls. What counts
# as success is Théo's call, not this file's.
#
# ⚠️ Disk: 6.3 GB free at launch, run needs ~3.7 GB. runs/t12h-prenorm/replay (3.1 GB)
# is the offline harness's dataset -- do not delete it to make room.
#
#   tail -f runs/t12h-muon-wdl-pcr/chain.log
set -u
cd ~/brokefish
RUN=t12h-muon-wdl-pcr
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
# t12h-muon-wdl's command verbatim + `--pcr-p 0.25 --pcr-fast-sims 32`. `--decay step`
# holds lr = 0.02 for the whole run (total-steps is inert under it, as in the base).
# `--moves-per-phase 12`, not the base's 10: PCR is stratified per phase and
# round(0.25 x 10) = 2 would realise p = 0.2; 3 of 12 is exactly 0.25.
say "1/3 train $RUN  (12 h, muon lr 0.02 CONSTANT, wd 0.09, WDL head, int8, gumbel m=16, PCR p=0.25 N=128 n=32)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --gumbel --gumbel-m 16 --sims 128 \
  --pcr-p 0.25 --pcr-fast-sims 32 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 12 \
  --optimizer muon --lr 0.02 --aux-lr 0.001 --adam-wd 0.09 \
  --ns-scheme polar --head-group 8 \
  --decay step --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
# Three runs, ONE fit, on the protocol they trained under. t12h-wdl isolates the
# optimiser; t12h-muon-wdl isolates PCR (its own league never ran, so this is also its
# first rating).
say "2/3 joint league UNDER GUMBEL+COLLAPSE: $RUN + t12h-wdl + t12h-muon-wdl"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-wdl --run t12h-muon-wdl \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 \
  --gumbel --gumbel-m 16 --terminal-collapse >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-wdl+t12h-muon-wdl.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
