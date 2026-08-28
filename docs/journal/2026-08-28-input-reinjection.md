# Every block sees the inputs again: `--reinject`, +3.8 % of evals/s

**2026-08-28.** Théo asked for it directly: *"make every input of each block a linear
combination of all model inputs plus residual stream, with a learnable scalar factor
for each element"*, and then *"try to add this capability for less than 5 % cost of
evals/sec"*. This entry is what it is, what it cost, and the two things I got wrong on
the way.

## What it is

A block's LayerNorm sees

    h + sum_k c[site, k] * E_k

instead of `h`, where the five `E_k` are the embedding tables of spec §7.2 -- square,
type_special, color_turn, clock, rep -- gathered again at the site's own indices, and
`c` is a learned scalar per (site, source). `--reinject ln1` is the attention norm of
each block, 8 sites and 40 parameters; `--reinject both` adds the FFN norm, 16 sites
and 80. `none` is the default and is every network before today.

**The residual stream is untouched.** The mix enters the norm's argument and nothing
else, so `h` still carries exactly `h + attn(.) + ffn(.)`. This is not DenseFormer's
depth-weighted average, which mixes the *stream*; it is a fresh read of the model's own
inputs at every depth, and the coefficients on the residual side are fixed at 1 because
LayerNorm is scale-invariant and only the ratio is observable.

**Zero-initialised.** A `reinject` net that has not trained is the network it modifies
-- and not approximately: the kernel is bit-identical at `c = 0` on both the fp16 and
the int8 path, at an odd batch size, verified in
`tests/test_b2.py::test_reinject_at_zero_is_the_old_kernel`. That is what makes an A/B
of this flag an A/B of one variable rather than of a re-randomised network, and it is
why a `t12h` checkpoint can be loaded into one and continued.

## What it cost

**+3.80 %** of useful evals/s at `ln1`, **+5.20 %** at `both`, interleaved and
order-balanced in the real MCTS at `n = 800`, `B = 4096`. A second run of the same
binary gave +4.19 % and +5.75 %. The full table, the protocol and the design that got
there are in `docs/ledger/perf.md`. The training step pays the same order, +4.34 %.

`ln1` meets the budget. `both` does not, and it is left in as a mode with its number
attached rather than removed, because whether the FFN norm is worth 5.5 % is a question
about Elo and not about the kernel.

## The two things I had wrong

**I costed the wrong number of gathers, twice.** The first answer said five tables, five
loads per row. `clock` and `rep` are indexed by the *board*, so they are one vector for
all 32 rows -- `embed_board` has folded them into a single `uint4` since B2 and I had
read that code. Worse, `color_turn` is `color * 2 + stm` and the side to move is fixed
within a board, so it takes **two** of its four rows, not 32 per-token gathers. Three of
the five sources are loop-invariant across a LayerNorm's rows. Pre-mixing them into two
slices per board takes the per-row cost from five 16-byte loads to **two**, and that is
most of the difference between a feature that fits the budget and one that does not.

**I argued against precomputing on numbers that were wrong.** Théo proposed writing the
per-site mixes to global once in the prologue and streaming them back with prefetch, and
I called it not worth it, citing an L2 working set and a VRAM figure. He was right on
both counts he raised: his scheme moves fewer bytes and issues fewer loads than the
five-gather version I was defending, the VRAM was never a constraint on a card at half
occupancy, and I had assumed a 24 MB L2 where the part has 32. What actually settles it
is neither argument: once three of the five sources are free, the gather moves 32 KB per
board per site against precompute's 16 KB read plus 16 KB written, and it reads 40 KB of
device-shared table that every CTA hits in L2 rather than per-board data with no reuse.
The basis is smaller than any combination of it. That is the whole reason to recompute,
and it is not the reason I first gave.

## One measured dead end

Guarding the per-row decode under `if (inject)`, so that a site which does not inject
would not pay for the shuffle, is the obvious cleanup and it is a **regression**: 0.7 %
on `ln1`, 3.0 % on `both`, and 250 -> 254 registers
(`logs/reinject-ab-branch-guarded.log`). A branch around the two loads is a scheduling
barrier -- ptxas can no longer lift them out of the unrolled row loop, so their L2
latency stops hiding behind the previous row's shuffle reduction. The dead decode is
cheaper than the lost pipelining, and the reason is now written where the branch would
go back.

## What is not measured

**Nothing about Elo.** No run has trained with this. The hypothesis it exists to test is
Théo's and this entry does not restate it as a prediction. The kernel is instantiated for
two configurations only -- int8 quant 3 at two boards per CTA, which is what self-play
runs, and fp16 at one board, which is what the tests run -- and refuses the rest rather
than silently dropping the injection. The Triton backbone refuses it outright: the mix
enters every block's norm and a backbone handed `x` once cannot express it.
