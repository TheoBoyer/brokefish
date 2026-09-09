# Two matches that were owed

*2026-09-09. Two direct matches the ledger was missing, 200 games each at n=128,
Gumbel m=16 on both sides, `scripts/h2h-bf.sh`, a minute of card each.*

## `t24h-adamw-int8` against `t12h-int8`, on the protocol they trained for

The ledger's strongest claim about training longer is the 2026-08-19 entry's
**+237 ± 79** for this pair, from a joint league that rated PUCT while both networks
trained Gumbel (2026-08-22). A joint league under Gumbel cannot be rerun, since
`t12h-int8`'s step snapshots were deleted and only its rolling checkpoint remains, so
this is the fit-free reading of the same two final networks:

| | W-D-L | score | 95 % Wilson | Elo |
|---|---|---:|---|---:|
| `t24h-adamw-int8` vs `t12h-int8`, Gumbel | 98-44-58 | **0.600** | [0.531, 0.665] | **+70** |

The 08-19 league's own head-to-head inside the pool was 77-14-17 under PUCT, a score
of 0.78. On the protocol the networks play in self-play the second 12 hours are worth
+70 Elo with an interval that excludes zero and excludes +237. This is the third pair
where the PUCT rating moved a headline by more than its error bar, after `t12h-wdl`
(−14 → +121) and the reinjection arms, and it moved this one down.

## `t12h-reuse2` against `t24h-adamw-int8`

`t12h-reuse2` doubled the sample reuse (`--samples-per-position 1.63`, the
2026-08-20 arm) and its match log stopped at "attempt 1" on the day the AlphaGateau
server was leaking. Played today:

| | W-D-L | score | 95 % Wilson | Elo |
|---|---|---:|---|---:|
| `t12h-reuse2` vs `t24h-adamw-int8`, Gumbel | 45-57-98 | 0.3675 | [0.304, 0.436] | −94 |

For scale, the 12 h reinjection arms score 0.4225 to 0.4525 against the same
opponent (`2026-09-09-reinjection-is-a-null.md`) and `t12h-int8` scores 0.400 by the
match above. Doubling the reuse did not buy a stronger 12 h network on this
instrument; it is the weakest 12 h arm measured against `t24h-adamw-int8` so far. The
2026-08-19 note that "samples per position 1.63 is ~1.77× steps/h, not 2×" stands as
the cost side; the benefit side is this number.
