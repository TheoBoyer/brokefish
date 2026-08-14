#!/usr/bin/env bash
# 12 h: the best-known recipe with the int8 kernel.
#
#   1/3  train t12h-int8, 12 h
#   2/3  ONE joint league: t12h-int8 + t12h-gumbel
#   3/3  the curve
#
# The bet, and it is the first run of this kernel. `t12h-int8` is `t12h-gumbel`'s
# command with **one flag changed**, `--fp8` -> `--int8`. That flag moves two things at
# once and both point the same way:
#
#   * **accuracy.** Prior-space error against the fp32 oracle falls from 4.37e-2 to
#     1.71e-2 max and flip risk from 4.43 % to 1.83 % on `t12h-gumbel-004009` -- the
#     fraction of positions where the error can change the move played. Self-play
#     targets are the search's output, so a less corrupted search writes better targets.
#   * **throughput.** 78 258 evals/s against 70 367 inside the real MCTS, so ~1.11x the
#     generations in the same 12 hours.
#
# ⚠️ **They are not separable in this run and nothing here claims they are.** A win is
# a win for "int8 at equal wall clock", not for either mechanism. Separating them would
# need a step-matched arm, which costs another 12 hours and is not what this is.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3, roadmap D5).
#   * Control is **`t12h-gumbel`**, the best model: same 12 h, same cosine shape, same
#     Gumbel m=16, same n=128, same window, same batch, same collapse. Only the FFN's
#     precision differs.
#   * Headline = final checkpoint at **n = 256**, for ledger continuity.
#   * ⚠️ The headline is NOT the verdict if the budgets disagree; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding -- that is
#     exactly what happened to `t12h-muong` (+22 / +156 / -51) and it is why the rule
#     is written down before the number exists.
#   * Success = beat `t12h-gumbel` at equal wall clock. A loss says the kernel's
#     accuracy and speed together do not buy Elo, which would be worth knowing and
#     would close the precision line rather than invite a third attempt.
#
# ⚠️ `--total-steps 4500`, not `t12h-gumbel`'s 4150. That parameter is the **cosine
# horizon**, not a stopping condition, and the run is stopped by `--minutes 720`.
# `t12h-gumbel` reached ~4185 actual steps at 20.7 s/gen; at 1.11x the self-play rate
# this one should reach ~4600, so 4500 has the schedule finish just before the clock
# does. Undershooting is the safe direction: the tail then sits at `lr_min` and the run
# still **ends annealed**, which is the property that makes it comparable to the
# control. Overshooting would stop it mid-anneal and it would not be.
#
# ⚠️ Disk: 13 GB free at launch. The buffer lands near 3.1 GB (`t12h-muong`'s) and the
# 22 kept checkpoints near 0.6 GB. It fits; it does not fit twice.
#
#   tail -f runs/t12h-int8/chain.log
set -u
cd ~/brokefish
RUN=t12h-int8
mkdir -p runs/$RUN
LOG=runs/$RUN/chain.log
PY="uv run --no-project --python .venv/bin/python"

say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "chain start.  tail -f $LOG"
git rev-parse HEAD | tee -a "$LOG"
git diff > runs/$RUN/$RUN.diff; wc -l runs/$RUN/$RUN.diff | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"

# 1/3 -----------------------------------------------------------------------
say "1/3 train $RUN  (12 h, int8 FFN + 2 boards/CTA, gumbel m=16, n=128, adamw 1e-3)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 4500 \
  --games 1024 --moves-per-phase 10 \
  --optimizer adamw --lr 0.001 --lr-min 5e-5 --decay cosine --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
# Two runs, ~53 players, 36 games/pairing -- the interval that resolved a z = 3.8
# effect in the Gumbel league. Expect ~2 h. The report lands in the FIRST run's folder,
# `runs/t12h-int8/`, by the convention brokefish/paths.py documents.
say "2/3 joint league: $RUN + t12h-gumbel"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-gumbel \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-gumbel.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
