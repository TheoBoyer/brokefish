# Brokefish

Brokefish learns chess from self-play alone, on one RTX 4060 Laptop GPU. As of
2026-09-09, the network from its 24-hour run scores 0.4175 against the final checkpoint
of [AlphaGateau](https://arxiv.org/abs/2410.23753)'s 500-iteration run, which trained
for 13 days and 16 hours on eight RTX A5000s: 13 wins, 141 draws and 46 losses over 200
games, −58 Elo, at 128 simulations a move on both sides. That is 24 GPU-hours of
training against about 2,620 on different GPUs, a total that includes 8 × 48.6 hours
of their evaluation
([the match](journal/2026-08-19-the-league-does-not-transfer.md),
[their checkpoint's provenance](journal/2026-08-13-alphagateau-cadence.md)).

The 2105 Elo the AlphaGateau paper reports for that network is self-anchored, a fit
over their own checkpoints with the pool mean pinned at 1000, so it cannot be compared
with any rating here; the match above needs no rating conversion
([reading the paper](journal/2026-08-09-alphagateau-read.md)).

!!! info "Interim report, state as of 2026-09-09 (commit 5b24ad9)"

    The goal is to build the cheapest superhuman chess engine that learns from scratch,
    and to measure on the way a curve of Elo against euros spent, which has not been
    published for chess. Neither has been reached.

    Done: the CUDA chess environment (track A), the network and its fused encoder kernel
    (B), the on-device search with PUCT and Gumbel (C1), the self-play training loop
    (C2), the evaluation stack from rule suites to the rating league (D0-D2), int8
    inference, and a match harness against AlphaGateau's released network.

    Not done: tying our ratings to a published Elo scale (D3-D5), the pilot run that
    would produce the first cost-against-Elo points (C4), and the full run (C5). Every
    Elo in this repository outside the AlphaGateau matches is measured against our own
    checkpoints.

    Everything runs on a single GPU. The project does no distributed training.

## What I found

1. Self-anchored Elo does not transfer to an outside opponent. The 24 h run rated
   +237 ± 79 over the 12 h `t12h-int8` in our joint league, and +12.5 ± 34 over the
   12 h `t12h-gumbel` when both played AlphaGateau. A direct 200-game match against
   `t12h-int8`, under the search both trained with, gives 0.600 (+70 Elo, 95 % interval
   [0.531, 0.665]). The league fit behind +237 did not converge (dispersion 1.148)
   and was played under PUCT (next item). [2026-08-19](journal/2026-08-19-the-league-does-not-transfer.md),
   [2026-09-09](journal/2026-09-09-two-matches-owed.md)
2. The league rated a different search from the one training used. Every league
   before 2026-08-22 played PUCT, while every run since `t12h-gumbel` trained with
   Gumbel; re-rating `t12h-wdl` against its control under Gumbel moved its headline
   from −14 to +121 Elo. Every rating of a Gumbel-trained run recorded before that date
   carries the mismatch, Gate 2's slope included.
   [2026-08-22](journal/2026-08-22-the-league-was-on-the-wrong-protocol.md)
3. Micro-batch accumulation was not exact under a value mask. The value loss is a
   mean over the supervised rows and `train_step` scaled it by the row count; each term
   is now scaled by the count it averages over, and one run trained under the old
   weighting. The same review found Gumbel's prior floor at the smallest normal fp16,
   2^-14, which merged every prior below 6e-5 onto one logit; it is now 2^-24. None of
   the review's four fixes was measured for its effect on playing strength.
   [2026-09-09](journal/2026-09-09-core-algorithm-review-fixes.md)
4. The value head loses on calibration while the trunk carries the information.
   Over eleven runs scored on one pinned set of 40,000 positions, a linear readout of
   the head's own input reaches a correlation of 0.596-0.607 at every final checkpoint,
   and the gap between that ceiling and the trained head orders every Elo outcome
   measured, from 0.006 for the best 12 h recipe to 0.078 for a run that lost 231 Elo
   to its control. A thousand-step refit of the head alone recovers the gap. The gap
   may not be used to select checkpoints.
   [2026-08-24](journal/2026-08-24-the-value-head-is-a-calibration-failure.md)
5. Muon's early lead over AdamW did not hold. Muon led by +143 ± 26 Elo at step
   1905 and reached ratings 200-400 in about 1.4× less wall clock, against an AdamW
   learning rate that was never re-screened, in a run that died at that step when the
   disk filled; at 24 h it trailed by 152. With the weight decay it needed, the gap read
   +22 / +156 / −51 at n = 16 / 64 / 256. All of these are on the PUCT league, and Muon
   is not the default. [2026-08-07](journal/2026-08-07-muon-curve.md),
   [2026-08-14](journal/2026-08-14-muon-with-decay.md)
6. We hang material at 2.15-2.21× AlphaGateau's rate. Of the reproduced blunders
   (uncompensated losses of 3 pawns or more), a third are flagged one ply late, a
   1024-simulation search fixes 14 %, and the rest split between a collapse of the
   interior search and a value head that does not see the hanging piece (raw value not
   lower after the blunder in 41 % of cases). The 12 h checkpoint that played is gone,
   so that side replays 170 of 269 positions, and our blunder rate against ourselves is
   unmeasured. [2026-09-09](journal/2026-09-09-blunder-dissection.md)
7. Several expected gains measured as nulls.
   - Input re-injection: five runs, fit deltas whose sign flips between search budgets,
     and 0.4300 for the compute-matched 24 h arm in a direct match against the baseline
     ([2026-09-09](journal/2026-09-09-reinjection-is-a-null.md)).
   - The int8 kernel in a 12 h run: 1.10× the training steps and +20 ± 129 Elo. It
     ships because it costs nothing
     ([2026-08-15](journal/2026-08-15-int8-run.md)).
   - Fusing the FFN handoff removed 29 memory instructions and added about 110
     arithmetic ones in an issue-limited kernel, and measured neutral
     ([2026-08-05](journal/2026-08-05-ffn-handoff-negative.md)).
   - A 4.5 % kernel speedup from a naive before-and-after came from an order effect of
     about 3.3 ms against a true difference of 0.35 ms; order-balanced interleaved A/B
     has been the protocol since ([perf ledger](ledger/perf.md)).

## How it is measured

- **Strength** comes from a Bradley-Terry league over checkpoints, fitted by Newton's
  method and anchored on the random-init network. Every rating carries the search
  budget it was played at, since the Elo gained per 10× of training compute measured
  +408 / +703 / +793 at n = 16 / 64 / 256 on the PUCT league. A large league compresses ratings by about
  a quarter through the fit's prior, so levels are compared only inside one league
  ([`evaluation.md`](reference/evaluation.md)).
- **Matches against another engine** go through `scripts/arbiter.py`, with
  python-chess as the only authority on legality and result, and every opening played
  once with each colour.
- **Checkpoint selection**: the headline of a run is its final checkpoint, fixed
  before launch; choosing the checkpoint with the best evaluation
  score would leak the evaluation into the result.
- **Speed** is compared by order-balanced interleaved A/B inside one process, because
  the card's clocks drift by ±3 % under load ([perf ledger](ledger/perf.md)).
- **Correctness** rests on oracles: the PyTorch implementations for the CUDA ones,
  python-chess and perft for the rules (green on the six standard positions, the
  start position to depth 6). `pytest tests/ --mutation` checks that the tests fail
  when the code under them is broken.
- **Documentation**: `docs/reference/` says what the code must do, `docs/ledger/`
  holds the measurements, and `docs/journal/` is dated and append-only,
  including the entries that were later proved wrong.

## How it is built

The environment is a chess engine in CUDA. A position is 32 `uint16` words, one per
piece slot, plus a control word carrying side to move and the halfmove clock, and move
generation runs on the GPU. Slots are stable for a whole game, so the network's 32
piece tokens line up with the engine's 32 legality masks and the policy is masked by a
device-side AND inside the search. The environment costs 2.2 % of a search node.

The network is a transformer over those 32 tokens: `d = 256`, 8 layers, 8 heads, FFN
1024, 6,383,360 parameters, about 400 MFLOPs per evaluation. It outputs raw 32×64
policy logits, promotion logits and a value.

The search is an MCTS that runs on the device, with PUCT as specified in AlphaGo Zero
and Gumbel MuZero at the root, the interior and the training target. Self-play runs
Gumbel over the top 16 moves at 128 simulations a move; 800 simulations is the
configuration the throughput benchmarks use ([`search.md`](reference/search.md)).

Training generates games and takes gradient steps on the same card, with AdamW by
default and Muon and a win/draw/loss value head as options
([`training.md`](reference/training.md)).

The encoder kernel runs boards to logits in one launch. Its fp16 version loads weights
from global memory straight into tensor-core fragments, never staging them in shared
memory, and reached 25.2 TFLOPS on the backbone benchmark with fp16 accumulation, 71 %
of the 35.5 TFLOPS that mode issues on this card; cuBLAS reaches 16.1-16.5 TFLOPS on the same four per-layer
shapes with fp32 accumulation, whose ceiling here is 18, so the two figures use
different accumulator precisions. The shipping kernel runs int8 on all four matmuls
with two boards per CTA: 88,148 evaluations/s at 4096 boards and 78,258 useful
evaluations/s inside the search (`bench_search`, n = 800, B = 4096), 1.37× the 56,996
that cleared the first engineering gate. Absolute numbers drift about 3 % with card
temperature ([perf ledger](ledger/perf.md),
[writing the kernel](journal/2026-07-29-encoder-kernel.md)).

## What's next

- The gap to AlphaGateau will be attacked where the dissection found it, in the value
  head's calibration and the interior search. The representation question (piece
  tokens against a graph over squares) remains untested.
- D3 to D5 will tie our ratings to a published scale: a harness for matches against
  CPU engines, calibration against three or four CCRL-rated engines, then a gate whose
  threshold, opponents, time control, hardware and search budget are fixed before the
  run ([roadmap](roadmap.md)).
- C4, a pilot run, will give the first points of the cost-against-Elo curve and a check
  of its slope against Jones' scaling law (+500 Elo per 10× compute).
- C5, the full run, will produce the curve.

## What training may use

Training data comes from games Brokefish plays against itself. Excluded as training
input: human game databases, games or evaluations produced by any other engine,
pretrained networks, distillation from a stronger model, opening books, and
hand-written chess heuristics such as piece-square tables. The environment encodes the
rules of chess, and anything that encodes a judgment about which positions are better
has to be learned.

Randomised openings, temperature sampling and Dirichlet noise diversify self-play
without importing knowledge. Standard opening suites such as UHO and TCEC are used
during evaluation to reduce variance between paired games; evaluation is measurement
and sits outside the training boundary.

## Where this sits

lc0 is the only project that has reached superhuman play from scratch, at roughly
eight GPU-years per run on volunteer compute that was never costed. Solo attempts
published since then plateau between 1200 and 2000 Elo, or start from a supervised
network ([prior art](journal/2026-07-28-prior-art.md)). AlphaGateau is the nearest
reference with a full accounting of its cost: 65.5M positions (256 streams of 512 plies
× 500 iterations), and released checkpoints a direct match can be played against.

## How this documentation is organised

Three kinds of page, and the section it sits under is the kind.

**Reference** is what the code must do. It has no dates in the body and no
measurements, and it is rewritten whenever the code moves.
[`spec.md`](reference/spec.md) is the frozen engine-network contract; the others are
normative for their component.

**Ledger** is the numbers, each in one place. [`state.md`](ledger/state.md) is one
bullet per landed component, [`perf.md`](ledger/perf.md) one row per measurement.
Everything else cites them rather than restating them.

**[Journal](journal/index.md)** is dated and append-only: the experiments, the
audits, the dead ends, and the reasoning behind decisions. An entry has to be right as
of its date and is never edited afterwards. Where an entry and a reference page
disagree, the reference page is current.

[`roadmap.md`](roadmap.md) sits on top of all three and says what is next.
