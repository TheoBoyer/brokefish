# Brokefish

Brokefish is a chess engine that learns only from games it plays against itself, in the
style of AlphaZero, built to train on a single consumer GPU. The aim is to find out how
cheaply a superhuman engine can be trained from zero, and to plot playing strength
against the money spent on compute.

!!! info "Where it stands, 2026-09-09"

    - After 24 hours on an RTX 4060 Laptop, the network scores 0.4175 against
      [AlphaGateau](https://arxiv.org/abs/2410.23753), a published model trained for
      13.7 days on eight GPUs: 58 Elo short.
    - The training loop, the GPU chess engine, the search and the evaluation tools are
      built and tested. The goal is not reached, and our ratings are not yet tied to a
      public Elo scale.
    - Everything runs on one GPU. Claude Code wrote most of the code and documentation
      under my direction. I designed the board representation that keeps the whole loop
      on the GPU, and the scope, the training rules, the measurement method and the
      decisions are also mine.

The code, installation and usage are in the
[repository](https://github.com/TheoBoyer/brokefish). Trained weights are on
[Hugging Face](https://huggingface.co/Theob/Brokefish).

## Findings

The full match against AlphaGateau, 200 games at 128 simulations a move on both sides:
13 wins, 141 draws, 46 losses. AlphaGateau's paper reports 2105 Elo for that network,
on a scale fitted to its own checkpoints, so that number cannot be compared with ours;
the direct match needs no conversion
([reading the paper](journal/2026-08-09-alphagateau-read.md)).

### How strength is measured here

Ratings come from games between checkpoints of our own runs, plus a fixed randomly
initialised network as the zero point, fitted to Elo by maximum likelihood. They are
only comparable inside one pool of players, and they depend on how many search
simulations each engine gets per move, so every number below states it. The only
outside reference is a direct match against
[AlphaGateau](https://arxiv.org/abs/2410.23753): 200 games, each opening played once
with each colour, python-chess as referee.

### 1. Ratings against our own checkpoints overstate progress

The 24-hour run rated 237 Elo above a 12-hour run in our internal pool. Against
AlphaGateau it scored 0.4175, and a 12-hour run that differs only in inference
precision scored 0.400, which puts them 12 ± 34 Elo apart. Played directly against the
first 12-hour run, with the search both were trained with, the 24-hour run scores
0.600, or +70 Elo (95 % interval 0.531 to 0.665). Doubling the training time
helped, by far less than the internal rating said.
([journal](journal/2026-08-19-the-league-does-not-transfer.md),
[direct match](journal/2026-09-09-two-matches-owed.md))

### 2. For eleven days, evaluation used a different search than training

Runs trained with Gumbel MuZero search from 2026-08-11, while the rating pool kept
playing with PUCT, the AlphaZero search. Re-rating one comparison with the search the
networks were trained for moved its result from −14 to +121 Elo. Every comparison of
Gumbel-trained runs made before the fix carries this bias.
([journal](journal/2026-08-22-the-league-was-on-the-wrong-protocol.md))

### 3. Gradient accumulation was not exact

The value loss averages over the positions that carry a value target, and when a batch
was split into micro-batches each part was weighted by the wrong count. A code review
found it along with three other bugs, among them a floor on move probabilities in the
Gumbel search set at 2⁻¹⁴ instead of 2⁻²⁴, which made every move below 6·10⁻⁵ look
equally likely. All four are fixed. One run trained with the wrong weighting, and the
effect of the fixes on playing strength was not measured.
([journal](journal/2026-09-09-core-algorithm-review-fixes.md))

### 4. The value head is badly calibrated, and the network underneath it is fine

Across eleven runs scored on the same 40,000 positions, a linear readout fitted on the
value head's own input reaches a 0.60 correlation with game outcomes at every final
checkpoint. How far the trained head falls below that ranks the runs in the same order
as their Elo, and retraining the head alone for 1,000 steps closes the gap. The
network learns what the outcome depends on; the last layer is overconfident.
([journal](journal/2026-08-24-the-value-head-is-a-calibration-failure.md))

### 5. Muon against AdamW: an early lead that did not last

Muon led AdamW by 143 ± 26 Elo at step 1905 and reached the same ratings about 1.4×
faster, against an AdamW learning rate that had not been tuned, in a Muon run that
crashed at that step when the disk filled. Over 24 hours Muon ended 152 Elo behind.
With more weight decay the results were mixed: +22, +156 and −51 Elo at 16, 64 and 256
simulations. All of this was rated with PUCT (point 2), and AdamW remains the default.
([first run](journal/2026-08-07-muon-curve.md),
[with weight decay](journal/2026-08-14-muon-with-decay.md))

### 6. We hang pieces at twice AlphaGateau's rate

In the two AlphaGateau matches our network lost material without compensation 2.2
times as often as its opponent. Replaying each of those positions: in a third, the
counting rule fires one move after the real mistake; searching 8 times longer fixes
14 %; the rest split between the search never visiting the refutation, and a value
head that does not see the hanging piece (in 41 % of cases its evaluation does not drop
after the blunder).
([journal](journal/2026-09-09-blunder-dissection.md))

### 7. Things that did not help

- Feeding the input embeddings back into every layer: five runs, gains whose sign
  flips with the search budget, and a direct-match score of 0.43 for the
  compute-matched 24-hour run.
  ([journal](journal/2026-09-09-reinjection-is-a-null.md))
- int8 inference during self-play: 1.10× more training steps in 12 hours and
  +20 ± 129 Elo, which is no measurable gain. It stays on because it costs nothing.
  ([journal](journal/2026-08-15-int8-run.md))
- Fusing a step of the inference kernel removed 29 memory instructions and added about
  110 arithmetic ones, in a kernel limited by instruction issue: no speedup.
  ([journal](journal/2026-08-05-ffn-handoff-negative.md))
- An early 4.5 % kernel speedup came from measuring one version after the other while
  the GPU warmed up. Every benchmark since interleaves the variants in one process.
  ([performance log](ledger/perf.md))

## How it works

**Board.** A position is 32 16-bit words, one per piece, plus a control word for side
to move and the move counters. A piece keeps the same slot for the whole game. Move
generation runs in CUDA, one warp per position. Because a slot always holds the same
piece, the network's i-th token, the i-th row of the legal-move mask and the i-th row
of the policy all refer to that piece, so applying the rules to the network's output
is a bitwise AND on the GPU, with no gather and no round trip to the CPU.

**Network.** A transformer whose 32 input tokens are the 32 pieces: 8 layers, width
256, 8 heads, 6.4 M parameters, about 400 MFLOPs per evaluation. For each piece it
outputs 64 logits, one per destination square, so the legal-move mask from the board
lines up with the policy output directly.

**Search.** Monte Carlo tree search on the GPU, with PUCT as in AlphaGo Zero and Gumbel
MuZero. Self-play uses Gumbel at 128 simulations per move.

**Inference kernel.** The whole network runs in one CUDA kernel launch, from board
words to logits. Its fp16 version reached 25.2 TFLOPS with fp16 accumulation, 71 % of
what the card's tensor cores issue in that mode; cuBLAS reaches 16.1 to 16.5 TFLOPS on
the same matrix shapes with fp32 accumulation, so the two use different precisions.
The int8 version used in training reaches 78,000 network evaluations per second inside
the search ([performance log](ledger/perf.md),
[writing the kernel](journal/2026-07-29-encoder-kernel.md)).

**Correctness.** Each CUDA component has a PyTorch reference that the tests compare it
against. The move generator is also checked against python-chess and published perft
counts.

## What comes next

- Work on the two causes of the blunders: the value head's calibration and the search.
- Rate the network against established CPU engines, so that its Elo can be placed on a
  public scale.
- A pilot run that produces the first points of the strength-against-cost curve, then
  the full run.

## What training may use

Training uses only games Brokefish plays against itself: no human games, no games or
evaluations from other engines, no pretrained weights, no opening books and no
hand-written chess knowledge. The code knows the rules and nothing about which positions
are good. Random openings, sampling temperature and Dirichlet noise add variety to
self-play. Standard opening suites are allowed in evaluation only.

## How this documentation is organised

**Reference** pages say what the code must do, and are rewritten when the code changes.
[`spec.md`](reference/spec.md) is the contract between the chess engine and the
network.

**Ledger** pages hold the measured numbers: [`state.md`](ledger/state.md) per
component, [`perf.md`](ledger/perf.md) per measurement.

The **[journal](journal/index.md)** is dated and append-only: experiments, audits and
dead ends, each correct as of its date. Where an entry and a reference page disagree,
the reference page is current. [`roadmap.md`](roadmap.md) is the plan.
