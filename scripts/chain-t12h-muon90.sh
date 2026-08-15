#!/usr/bin/env bash
# 12 h: muon at wd = 0.9, on the int8-everywhere kernel.
#
#   1/3  train t12h-muon90, 12 h
#   2/3  ONE joint league: t12h-muon90 + t12h-muon9
#   3/3  the curve
#
# ⚠️ **TWO THINGS MOVE IN THIS RUN AND THAT IS DELIBERATE, NOT AN OVERSIGHT.** The
# weight decay goes 0.09 -> 0.9 *and* the kernel goes int8-FFN -> int8-all-four. It is
# recorded here because a preregistration that does not say so is worthless later.
#
# The argument for spending one run on two changes: the kernel half is expected to be
# Elo-neutral at fixed wall clock and is being taken for its **throughput**. That
# expectation is measured, not assumed -- `t12h-int8` moved the same axis (1.096x
# steps, 1.115x games) and produced **+20 +/- 129 Elo**, i.e. nothing this league can
# resolve. What is *not* measured is what the extra quantisation costs at play time,
# because `eval/league.py:381` builds every player in fp16 and always has. That gap is
# bounded separately by a same-checkpoint fp16-vs-int8a head to head; until it lands,
# the honest statement is that this run's kernel term is **unbounded below by anything
# we have measured**, and a negative result cannot be cleanly attributed to `wd`.
#
# ⚠️ What the norm law says about wd = 0.9, recorded so the result can be read against
# a prediction rather than after the fact. Measured final `train/weight_norm`:
#
#     AdamW (t12h-int8)  134.3        muon wd 0.04 (t12h-muong)  247.8
#     muon wd 0.09 (t12h-muon9)  145.3   <- peaked 210.0 at step 999
#
# So `wd = 0.09` **hit AdamW's norm to within 8 %**, and produced -5 Elo at the
# preregistered n = 256. The norm-gap hypothesis has therefore been tested at the value
# that tests it. `||W||* = c/wd` with c from the final norms (9.9 at 0.04, 13.1 at 0.09,
# so c is not constant and rises with wd) predicts **||W|| ~ 11-15 at wd = 0.9** -- an
# order of magnitude *below* AdamW rather than closer to it.
#
# That makes this arm a different experiment from the last two, and it should be read
# as one: not "does closing the norm gap recover muon's +143", which is answered, but
# **"does a much smaller equilibrium relative step help"** -- `lr/||W||` here is ~10x
# `t12h-muon9`'s, i.e. effectively a far more aggressive late-run learning rate.
#
# ⚠️ Watch in the first two hours: `weight_norm` heading for ~11-15 rather than ~130,
# and `grad_norm` / `kl` for a blow-up. A 10x larger relative step is the regime where
# `--grad-clip 1.0` starts firing continuously, which would make this an implicit lr
# schedule rather than the intervention it is meant to be. `t4h-n64` failed exactly
# that way (clip firing on 99 % of the first 300 steps).
#
# ⚠️ PREREGISTERED BEFORE LAUNCH (evaluation.md §3, roadmap D5).
#   * Control is **`t12h-muon9`**: same optimiser, same lr, same polar scheme, same
#     Gumbel m=16, same n=128, same window, same batch, same collapse, same cosine
#     shape. `wd` and the kernel move; nothing else does.
#   * Headline = final checkpoint at **n = 256**, ledger continuity.
#   * ⚠️ The headline is NOT the verdict if the budgets disagree; the full 16/64/256
#     grid is reported and a sign flip across budgets is itself the finding. Both
#     previous muon arms fell with n (+22/+156/-51 and +76/+63/-5) and that shape is
#     the thing to look for first.
#   * Success = beat `t12h-muon9` at equal wall clock.
#   * ⚠️ The puzzle probe is **reported and not weighted**. It runs at `n_sims = 0`,
#     which is the raw policy, and it is 0-for-4 at predicting n = 256. It remains a
#     good instrument for the small-n end and should be read as that.
#
# ⚠️ `--total-steps 5000` and not 4500. The kernel is measured 1.142x faster inside the
# real MCTS (90 123 against 78 890 evals/s), which moves a generation from ~19.0 s to
# ~16.9 s, so 4 539 steps becomes ~5 100. 5 000 deliberately **under**-shoots: a cosine
# that completes early still anneals and the last steps run flat at `lr_min`, where one
# that does not complete leaves the run un-annealed and not comparable to a control
# that was. `t12h-int8` undershot by 1.9 % for the same reason.
#
#   tail -f runs/t12h-muon90/chain.log
set -u
cd ~/brokefish
RUN=t12h-muon90
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
say "1/3 train $RUN  (12 h, int8 all four, muon lr=0.02 polar, wd 0.9, gumbel m=16, n=128)"
$PY -m brokefish.train.loop \
  --run $RUN --gumbel --gumbel-m 16 --sims 128 \
  --minutes 720 --total-steps 5000 \
  --games 1024 --moves-per-phase 10 \
  --optimizer muon --lr 0.02 --aux-lr 0.001 --head-group 8 --ns-scheme polar \
  --adam-wd 0.9 --lr-min 0.001 --decay cosine --warmup 30 --grad-clip 1.0 \
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
