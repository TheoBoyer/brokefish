#!/usr/bin/env bash
# 12 h: t12h-muon-wdl's recipe + `--value-subsample 4`, one variable.
#
#   1/3  train t12h-muon-wdl-vs4, 12 h
#   2/3  ONE joint league under Gumbel+collapse: this + t12h-wdl (AdamW) + t12h-muon-wdl (Muon)
#   3/3  the curve
#
# Why (2026-08-25): the Muon value pathology is game-level memorisation of `z`, one
# bit repeated on ~118 positions of a game (offline harness, 2026-08-24). PCR moved
# that count only at the price of 1/p more reuse per stored record, and the
# steps-matched arm (t12h-muon-wdl-pcr) flattened the value probe at a LOWER level
# (0.25 vs the control's 0.34 peak) while the policy probe fell 0.13 -> 0.10:
# confounded, not read. `--value-subsample 4` moves the same count -- the value loss
# is taken over every 4th ply of each game, ~118 -> ~30 labels per game -- and
# NOTHING else: same records, same cadence, same 0.68 exposures per record, same
# optimiser, same policy gradient, same throughput.
#
# ⚠️ PREREGISTERED measurement, decision Théo's: (a) the in-run value probe against
# t12h-muon-wdl (0.343 @1002 -> 0.287 @5010) and t12h-wdl (flat 0.35-0.38) -- does
# the decline go and where does the level sit; (b) the policy probe against
# t12h-muon-wdl's 0.129 @5010 -- the policy gradient is untouched, so it should not
# move; (c) the joint Gumbel league at n = 16/64/256 against both controls.
# Prediction written before the run: value holds near the peak, policy within noise.
# If value still decays, the repetition count is not the mechanism. If value holds
# and policy drops, the trunk needed the value gradient volume and the KataGo-style
# auxiliary head (main head on z, aux on root_value, search reads only the main) is
# the next arm.
#
# ⚠️ Disk: the buffer memmap preallocates 20000 x 350 x 463 B = 3.2 GB up front.
# The chain WAITS for 5 GB free before starting (runs/t12h-prenorm/replay and
# runs/t12h-muon-wdl-pcr/replay + intermediate checkpoints are the space to free).
#
#   tail -f runs/t12h-muon-wdl-vs4/chain.log
set -u
cd ~/brokefish
RUN=t12h-muon-wdl-vs4
mkdir -p runs/$RUN
LOG=runs/$RUN/chain.log
PY="uv run --no-project --python .venv/bin/python"

say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "chain start.  tail -f $LOG"
git rev-parse HEAD | tee -a "$LOG"
git diff > runs/$RUN/$RUN.diff; wc -l runs/$RUN/$RUN.diff | tee -a "$LOG"

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
while [ "$(free_gb)" -lt 5 ]; do
  say "waiting for disk: $(free_gb) GB free, need 5 (delete the old replays / checkpoints)"
  sleep 60
done
df -h / | tail -1 | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader | tee -a "$LOG"

# 1/3 -----------------------------------------------------------------------
# t12h-muon-wdl's command verbatim + `--value-subsample 4`. `--decay step` holds
# lr = 0.02 for the whole run; the run ends on --minutes, not on --total-steps.
say "1/3 train $RUN  (12 h, muon lr 0.02 CONSTANT, wd 0.09, WDL head, int8, gumbel m=16, value loss on 1 in 4 plies)"
$PY -m brokefish.train.loop \
  --run $RUN --value-classes 3 --gumbel --gumbel-m 16 --sims 128 \
  --value-subsample 4 \
  --minutes 720 --total-steps 5200 \
  --games 1024 --moves-per-phase 10 \
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
