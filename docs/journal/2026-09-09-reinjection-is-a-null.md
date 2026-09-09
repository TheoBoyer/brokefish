# Input re-injection: five runs, no Elo

*2026-09-09. Written eight days after the last of these runs ended, because nothing
had been written. The five chains of 2026-08-28 to 09-01 all finished cleanly and
none of them has a journal entry, an index row or a ledger line. This is the record.*

## The arms

Every arm is `t12h-wdl`'s recipe with one variable moved, all under Gumbel m=16, n=128,
int8, `--terminal-collapse`, `--value-classes 3`, AdamW with `--decay step`. Each chain
ended in a joint league under the training protocol, and the three later ones in a
200-game direct match at n=128 against `t24h-adamw-int8`, the network the ledger
records as our strongest.

| arm | the variable | steps at the clock cap |
|---|---|---:|
| `t12h-reinject` | `--reinject ln1`, 8 sites, 40 scalars | 5065 |
| `t12h-reinject-both` | `--reinject both`, 16 sites, 80 scalars | 4918 |
| `t12h-reinject-both-lr3` | lr 0.001 → 0.003 | 4908 |
| `t12h-reinject-both-lr6` | lr 0.003 → 0.006, warmup 30 → 180 | 4926 |
| `t24h-reinject-lr6` | the same arm at 24 h, compute-matched | 9427 |

`t24h-adamw-int8` reached 10218 steps in its 24 h, so the compute-matched arm is 8 %
short in steps at equal wall clock, which is the re-injection's own throughput cost.
No arm diverged. The 6× rate the 2026-08-30 script header expected to blow up past
step 5000 held to the end with a flat loss.

## The fits

Each chain fits its own joint league, so the rows below are five scales and are
comparable only within a block. Final checkpoints, ±ci95.

| fit | at n=256 | Δ read off the fit |
|---|---|---:|
| 1: `ln1` vs `wdl` | 1713 ±104 vs 1728 ±106 | −15 |
| 2: `both`, `ln1`, `wdl` | 1584 vs 1640 vs 1627 | `both` −43, `ln1` +13 |
| 3: `lr3` vs `both` | 1651 vs 1582 | +69 |
| 4: `lr6` vs `lr3` | 1628 vs 1517 | +111, and **−26 at n=64** |
| 5: `t24h-reinject-lr6` vs `t24h-adamw-int8` | 1754 ±100 vs 1831 ±104 | **−77** |

Three things in that table are about the instrument rather than the arms. The `ln1`
headline moved from −15 to +13 when a third arm joined the pool. Fit 4's +111 at
n=256 is −26 at n=64, so the sign depends on the budget. All five fits report
`converged: False` at the 10,000-iteration cap, as every joint fit in `runs/` does.
Measured the same day against a converged Newton solve: the cap cost under 0.2 Elo on
any level and under 0.002 on any difference, so the +187.9 that the lr6 script header
blamed on the instrument was the fit's actual answer. The instrument problem is
elsewhere: the phantom prior compresses a 100-player league by about a quarter
(`evaluation.md` §5.4.5), which does not touch these deltas but does touch every
slope read inside one league.

At step 6412 the 24 h arm was ahead of `t24h-adamw-int8@6813` by +33, and it lost that
by the finish. The 2026-08-31 script header predicted this shape from `--decay step`
having no anneal.

## The matches

The chain scripts prescribe the direct match as the reading that wins when the fit
disagrees. 200 games at n=128, both sides Gumbel m=16, against `t24h-adamw-int8`:

| arm | W-D-L | score | 95 % Wilson | Elo |
|---|---|---:|---|---:|
| `t12h-reinject-both-lr3` | 56-57-87 | 0.4225 | [0.356, 0.492] | −54 |
| `t12h-reinject-both-lr6` | 62-57-81 | 0.4525 | [0.385, 0.522] | −33 |
| `t24h-reinject-lr6` | 60-52-88 | **0.4300** | [0.363, 0.499] | **−49** |

The compute-matched 24 h arm scores below the 12 h arm it doubled, and its interval
excludes parity. Fit 4 said +111 for `lr6` against `lr3`; the matches say +21 between
the same two networks, inside one interval. The two `ln1` and `both` arms never got
a match, since the fit-free protocol only started with `lr3` on 08-30.

The puzzle probes point the other way: `t24h-reinject-lr6` ends at 0.4763 pass@1 and
0.1795 solve against `t24h-adamw-int8`'s 0.4602 and 0.1638. A probe lead with a match
loss is the pattern the 2026-08-15 entry counted as 1-for-4, now 1-for-5.

## The coefficients

The learned scalars grow with the rate and with time and never settle:

| arm | mean \|c\| | max \|c\| |
|---|---:|---:|
| `ln1` | 0.50 | 2.59 |
| `both` | 0.57 | 2.50 |
| `lr3` | 1.27 | 6.49 |
| `lr6` | 2.25 | 11.76 |
| `t24h-lr6` | 3.22 | 13.50 |

The 24 h vector is 1.43× the 12 h one at the same rate and correlates with it at
r = 0.965, so the growth is structured. FFN sites ask for about 1.3× what attention
sites do, in every arm. The `square` table dominates from block 1 on (+2.4 to +2.6 in
blocks 1 and 2 of `ln1`) while `clock` and `rep` stay near zero. The network takes
the inputs and keeps taking more of them, and its play does not change.

## What this settles

Re-injecting the model's inputs at every block, at any rate tried and at twice the
clock, does not raise Elo on the direct match, and the fit cannot see a gain either
once the budget is varied. The feature stays in the kernel as a mode at its measured
cost, +3.8 % of evals/s at `ln1`, with this null attached. The `both` mode at +5.2 %
has no reason left to exist beyond the A/B and should not be the default anywhere.

What it does not settle is whether the inputs the network lacks are the ones it was
given. `square`, `type_special` and `color_turn` are already in the residual stream;
re-reading them buys a fresh copy, not new information. The 2026-08-19 blunder
finding asks for something the tokens do not carry, and that is the next thing to
test, on evidence, before another architectural run.

## Method notes

Five chains ran to completion and the results sat in `runs/` for eight days. The
chain scripts compute no deltas, so every "+187.9" in a script header was a hand
subtraction from a rating table, and the two earliest arms never got the match that
the later scripts declared the headline. Two rules for the next line of work: a chain
writes its own journal stub when it finishes, and no arm is compared without the
direct match.
