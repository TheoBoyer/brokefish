#!/usr/bin/env bash
# 12 h: t12h-muon9's recipe exactly, on the int8-everywhere kernel.
#
#   1/3  train t12h-muon9-int8, 12 h
#   2/3  ONE joint league: t12h-muon9-int8 + t12h-muon9
#   3/3  the curve
#
# ⚠️ **ONE VARIABLE MOVES AND IT IS THE KERNEL.** Every hyperparameter is copied from
# `chain-t12h-muon9.sh` -- muon lr 0.02 polar, `--adam-wd 0.09`, aux lr 0.001,
# head-group 8, gumbel m=16, n=128, 1024 games, 10 moves/phase, window 20000, cosine to
# `lr_min` 0.001, warmup 30, clip 1.0, terminal collapse. `--int8` now selects all four
# weight matmuls instead of the FFN's two, and `--total-steps` is resized for the
# resulting throughput. Nothing else differs.
#
# So this run asks exactly one question: **what does 1.14x more training inside a fixed
# 12 hours buy?**
#
# ⚠️ The honest prior is that it buys nothing this league can resolve, and it is
# recorded before the fact rather than after. `t12h-int8` asked the same question at
# 1.096x steps / 1.115x games and measured **+20 +/- 129 Elo** at n = 256, with the
# largest |z| in its grid at 0.45. `t12h-pcr` asked a larger version of it -- 3.7-6.0x
# on positions at equal wall clock -- and measured **+36 +/- 70**. Two nulls on the same
# axis; this is a third point on it at 1.14x, which is *between* those two interventions
# in size and closer to the smaller one.
#
# What makes it worth a run anyway, and this is the part that is not about the kernel:
# it is a **second, independent estimate of the wd = 0.09 arm itself**. `t12h-muon9` was
# measured once, against `t12h-int8`, and read +76 / +63 / -5 at n = 16/64/256 -- a shape
# falling with budget, and a headline of -5 +/- 129 that is indistinguishable from zero
# in either direction. A rerun of the same recipe against that same checkpoint says how
# much of that grid was the recipe and how much was one sample of a noisy league.
#
# ⚠️ And it leaves one term open. `eval/league.py:381` builds every player in fp16 and
# always has, so no rating in this project has ever been measured on the kernel the run
# actually used. The quantisation's cost *in games* is unmeasured, and until the
# same-checkpoint fp16-vs-int8 head to head is run, a negative result here cannot be
# separated into "more training bought nothing" and "the extra quantisation cost
# something". That head to head is cheap and should precede reading this run's number.
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3, roadmap D5).
#   * Control is **`t12h-muon9`**: same optimiser, same lr, same decay, same scheme,
#     same search, same window, same batch, same collapse, same cosine shape. Only the
#     kernel and the step target move.
#   * Headline = final checkpoint at **n = 256**, ledger continuity.
#   * ⚠️ The headline is NOT the verdict if the budgets disagree; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding. Both muon
#     arms so far fell with n (+22/+156/-51 and +76/+63/-5), and whether that shape
#     reappears on a rerun of one of them is itself informative.
#   * Success = beat `t12h-muon9` at equal wall clock.
#   * ⚠️ The puzzle probe is **reported and not weighted**. It runs at `n_sims = 0`,
#     which is the raw policy, and it is 0-for-4 at predicting n = 256. It remains a
#     good instrument for the small-n end and should be read as that.
#
# ⚠️ `--total-steps 5000` and not 4500. The kernel is measured 1.174x faster inside the
# real MCTS (92 639 against 78 890 evals/s) and the first generations of the aborted
# launch measured play at 13.8-14.3 s against `t12h-int8`'s 16.8 s, so a generation goes
# ~19.0 s -> ~16.6 s and 4 539 steps becomes ~5 190. 5 000 deliberately **under**-shoots:
# a cosine that completes early still anneals and the last steps run flat at `lr_min`,
# where one that does not complete leaves the run un-annealed and not comparable to a
# control that was. `t12h-int8` undershot by 1.9 % for the same reason.
#
#   tail -f runs/t12h-muon9-int8/chain.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=t12h-muon9-int8
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
# ⚠️ `--adam-wd` reaches the matrices, not just the AdamW side: `muon.py:279` and `:281`
# set `weight_decay: wd` on the muon group and the aux 2D group, and only `ndim < 2`
# keeps 0. That is the intent -- the norm growth is in the matrices.
say "1/3 train $RUN  (12 h, int8 all four, muon lr=0.02 polar, wd 0.09, gumbel m=16, n=128)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5000 \
  --games 1024 --moves-per-phase 10 \
  --optimizer muon --lr 0.02 --aux-lr 0.001 --head-group 8 --ns-scheme polar \
  --adam-wd 0.09 --lr-min 0.001 --decay cosine --warmup 30 --grad-clip 1.0 \
  --window-games 20000 --mean-plies 350 \
  --terminal-collapse --int8 \
  --keep-checkpoints --checkpoint-every 200 \
  >> "$LOG" 2>&1
say "1/3 done (exit $?)"
df -h / | tail -1 | tee -a "$LOG"

# 2/3 -----------------------------------------------------------------------
say "2/3 joint league: $RUN + t12h-muon9"
$PY -m brokefish.eval.league \
  --run $RUN --run t12h-muon9 \
  --games 36 --sims 64 --limit 16 \
  --ladder 1 4 16 64 --grid 16 256 --grid-points 4 \
  --anchor-every 4 --anchor-span 12 >> "$LOG" 2>&1
say "2/3 done (exit $?)"

# 3/3 -----------------------------------------------------------------------
say "3/3 the curve"
$PY -m brokefish.eval.curve \
  runs/$RUN/league-joint-$RUN+t12h-muon9.json >> "$LOG" 2>&1
say "3/3 done (exit $?).  chain finished"
df -h / | tail -1 | tee -a "$LOG"
