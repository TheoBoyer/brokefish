# What the experiments have found so far

State as of 2026-09-09. Each result links to the journal entry that holds the
measurement and its caveats.

In short: the training loop works and a 24-hour run on a laptop GPU gets within 58 Elo
of a published model that trained for two weeks on eight GPUs. The most useful lessons
so far are about measurement: internal ratings overstate progress, and for eleven days
the evaluation ran a different search than training did.

## How strength is measured here

Ratings come from games between checkpoints of our own runs, plus a fixed randomly
initialised network as the zero point, fitted to Elo by maximum likelihood. They are
only comparable inside one pool of players, and they depend on how many search
simulations each engine gets per move, so every number below states it. The only
outside reference is a direct match against
[AlphaGateau](https://arxiv.org/abs/2410.23753): 200 games, each opening played once
with each colour, python-chess as referee.

## 1. Ratings against our own checkpoints overstate progress

The 24-hour run rated 237 Elo above a 12-hour run in our internal pool. Against
AlphaGateau it scored 0.4175, and a 12-hour run that differs only in inference
precision scored 0.400, which puts them 12 ± 34 Elo apart. Played directly against the
first 12-hour run, with the search both were trained with, the 24-hour run scores
0.600, or +70 Elo (95 % interval 0.531 to 0.665). Doubling the training time
helped, by far less than the internal rating said.
([journal](docs/journal/2026-08-19-the-league-does-not-transfer.md),
[direct match](docs/journal/2026-09-09-two-matches-owed.md))

## 2. For eleven days, evaluation used a different search than training

Runs trained with Gumbel MuZero search from 2026-08-11, while the rating pool kept
playing with PUCT, the AlphaZero search. Re-rating one comparison with the search the
networks were trained for moved its result from −14 to +121 Elo. Every comparison of
Gumbel-trained runs made before the fix carries this bias.
([journal](docs/journal/2026-08-22-the-league-was-on-the-wrong-protocol.md))

## 3. Gradient accumulation was not exact

The value loss averages over the positions that carry a value target, and when a batch
was split into micro-batches each part was weighted by the wrong count. A code review
found it along with three other bugs, among them a floor on move probabilities in the
Gumbel search set at 2⁻¹⁴ instead of 2⁻²⁴, which made every move below 6·10⁻⁵ look
equally likely. All four are fixed. One run trained with the wrong weighting, and the
effect of the fixes on playing strength was not measured.
([journal](docs/journal/2026-09-09-core-algorithm-review-fixes.md))

## 4. The value head is badly calibrated, and the network underneath it is fine

Across eleven runs scored on the same 40,000 positions, a linear readout fitted on the
value head's own input reaches a 0.60 correlation with game outcomes at every final
checkpoint. How far the trained head falls below that ranks the runs in the same order
as their Elo, and retraining the head alone for 1,000 steps closes the gap. The
network learns what the outcome depends on; the last layer is overconfident.
([journal](docs/journal/2026-08-24-the-value-head-is-a-calibration-failure.md))

## 5. Muon against AdamW: an early lead that did not last

Muon led AdamW by 143 ± 26 Elo at step 1905 and reached the same ratings about 1.4×
faster, against an AdamW learning rate that had not been tuned, in a Muon run that
crashed at that step when the disk filled. Over 24 hours Muon ended 152 Elo behind.
With more weight decay the results were mixed: +22, +156 and −51 Elo at 16, 64 and 256
simulations. All of this was rated with PUCT (point 2), and AdamW remains the default.
([first run](docs/journal/2026-08-07-muon-curve.md),
[with weight decay](docs/journal/2026-08-14-muon-with-decay.md))

## 6. We hang pieces at twice AlphaGateau's rate

In the two AlphaGateau matches our network lost material without compensation 2.2
times as often as its opponent. Replaying each of those positions: in a third, the
counting rule fires one move after the real mistake; searching 8 times longer fixes
14 %; the rest split between the search never visiting the refutation, and a value
head that does not see the hanging piece (in 41 % of cases its evaluation does not drop
after the blunder).
([journal](docs/journal/2026-09-09-blunder-dissection.md))

## 7. Things that did not help

- Feeding the input embeddings back into every layer: five runs, gains whose sign
  flips with the search budget, and a direct-match score of 0.43 for the
  compute-matched 24-hour run.
  ([journal](docs/journal/2026-09-09-reinjection-is-a-null.md))
- int8 inference during self-play: 1.10× more training steps in 12 hours and
  +20 ± 129 Elo, which is no measurable gain. It stays on because it costs nothing.
  ([journal](docs/journal/2026-08-15-int8-run.md))
- Fusing a step of the inference kernel removed 29 memory instructions and added about
  110 arithmetic ones, in a kernel limited by instruction issue: no speedup.
  ([journal](docs/journal/2026-08-05-ffn-handoff-negative.md))
- An early 4.5 % kernel speedup came from measuring one version after the other while
  the GPU warmed up. Every benchmark since interleaves the variants in one process.
  ([performance log](docs/ledger/perf.md))
