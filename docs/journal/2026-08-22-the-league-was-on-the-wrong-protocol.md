# Every league rated PUCT. Every run since t12h-gumbel trained Gumbel.

**2026-08-22.** `t12h-wdl` finished level with its control on Elo (−14 at the headline)
while beating it on both puzzle probes, and Théo asked the question that unlocked it:
*"in the league, do they play with gumbel search?"*

They did not. **`SearchConfig.gumbel` and `terminal_collapse` both default to `False`,
`runner.eval_config` never set either, and nothing in `brokefish/eval/` did** — while
every run since `t12h-gumbel` was **trained** under `--gumbel --gumbel-m 16
--terminal-collapse`.

Re-running the same 297-pairing calendar with the protocol the nets were trained for:

| | t12h-flat | t12h-wdl | Δ |
|---|---:|---:|---:|
| **Gumbel + collapse**, final @ n=256 | +1750 ±108 | **+1871 ±111** | **+121** |
| PUCT, same calendar and seed | +1668 ±92 | +1654 ±92 | **−14** |

Direct head-to-head, final vs final at n=256: **23-1-12 for WDL** under Gumbel,
16-3-17 (dead heat) under PUCT.

## The whole grid, both protocols

| step | n | PUCT ΔElo | Gumbel ΔElo |
|---:|---:|---:|---:|
| 201 | 256 | +196 | +290 |
| 601 | 64 | +88 | +183 |
| 2404 | 64 | −31 | −58 |
| 3408 | 256 | +11 | +49 |
| 3808 | 64 | −10 | +26 |
| 5010 | 64 | +5 | +72 |
| **5010** | **256** | **−14** | **+121** |

| | PUCT | Gumbel |
|---|---:|---:|
| mean Δ over 24 matched pairs | +39.8 | **+57.8** |
| median | +25.0 | **+38.5** |
| WDL ahead | 21/24 | **22/24** |

Aggregate direct head-to-head over all matched pairs: **438-128-298, score 0.581 over
864 games**, ~**+57 Elo ± 22**. ⚠️ **That paired aggregate is what carries the claim**,
not the headline: the two marginal ±111 intervals overlap, and the single 5010:n256 edge
alone is +110 ± 119.

## Why the protocol moves it, and why it moves *this* experiment most

Gumbel samples m = 16 root candidates from the prior and runs sequential halving, ranking
them by **completed Q** — the value head chooses the move. PUCT lets the prior drive
exploration through `P·sqrt(N)/(1+N)` and the value only enters through backed-up Q. **A
value-head experiment rated under PUCT is the wrong instrument**, and I built the run
that way and preregistered it that way.

## What this costs beyond one run

⚠️ **Every Elo number in `docs/ledger/` is on a protocol its network never trained
under.** That includes the Muon arms, `t24h-adamw-int8`'s **+237 ± 79** over
`t12h-int8`, `t12h-vw2`'s −231, and every Gate 2 slope. It is not a bug — both arms of
a pairing were always rated identically, so each delta is internally fair — but it is a
systematic mismatch that has never been written down, and this run shows it can hide an
effect entirely and flip a headline's sign.

Cheapest way to find out how much it matters: re-rate `t24h-adamw-int8` vs `t12h-int8`
under Gumbel. That is one league.

⚠️ A Gumbel league and a PUCT league are **not joinable into one fit**. The report now
records `search_kw` in its config for exactly that reason, alongside `quant`.

## Also measured on the day

The Gumbel league ran **86.1 min against PUCT's 123.0** on the identical calendar, with
shorter games (collapse ends proved wins) and a higher dispersion correction (**0.824**
against 0.703). The scales are not interchangeable.

## The optimisation audit, since the league is 2 hours every time

Measured on a free card, n = 64, Gumbel + collapse:

| B | ms/move | row-moves/s |
|---:|---:|---:|
| 1 | 31.9 | 31 |
| **18 ← the league** | **34.3** | **525** |
| 36 | 54.5 | 660 |
| 144 | 153.8 | 936 |
| 288 | 299.3 | 962 |

**A move costs ~31 ms whether the batch holds 1 game or 18.** `--games 36` gives 18 rows
per half, which is 6x below where marginal cost even begins. And the encoder is the
reason:

| B | ms per encoder call | boards/s |
|---:|---:|---:|
| 1 | 0.3771 | 2 651 |
| **18** | **0.3803** | **47 328** |
| 36 | 0.3812 | 94 448 |
| 4096 | 37.72 | 108 577 |

**One encoder call costs 0.377 ms whether it serves 1 board or 36** — the fixed cost of
bringing 6.4M parameters onto the SMs, paid **per call, per weight set**. It is ~70 % of
a simulation at B = 18.

⚠️ **So a batch cannot usefully mix networks.** Splitting a batch across K nets means K
calls at 0.377 ms each; a per-row weight pointer pays the same weight traffic. **The unit
that must be large is boards-per-network-per-call**, and the knee is only B ≈ 36.

Second loss: **62.4 % of searched row-plies are on throwaway positions.** The loop runs
until the *longest* game in the batch ends — measured 280 plies with a median of 4 live
rows out of 18.

Combined ceiling **~4.9x** (123 min -> ~25), and it needs one change: a scheduler that
keeps ~36-144 live games *per network* in flight across pairings. The calendar already
gives each checkpoint ~13 pairings, i.e. ~234 same-network rows.

Measured dead ends, so nobody retries them: **compacting live rows 0 %** (B = 4 costs what
B = 18 costs), **`check_invariants=False` 1.2 %**, **multi-process negative** (a pairing
went 1.2 s -> 3.6 s sharing the card). ⚠️ `nvidia-smi` reported 92 % "utilization"
throughout — that counts a kernel being resident, not SM occupancy; the tell was 1950 MHz
and 223 MiB, where a real load drops this card to 1230-1500 MHz.

**Not built.** It is a rewrite of the path every Elo number comes through, and
`_play_half`'s lockstep assertion is what currently guarantees one network per batch.
