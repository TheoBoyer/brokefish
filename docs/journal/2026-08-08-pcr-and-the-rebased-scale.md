# 2026-08-08 (ninth entry) — playout cap randomisation, and a scale that can hold a second axis

Two things landed on the same day and the second one changed how the first should be
read. `t12h-pcr` is E1.2 built and run; the rebased league is `evaluation.md` §5.1/§5.1a,
which moved the zero of the Elo scale onto uniformly random legal play and made the
search budget a per-player field instead of a constant hidden inside the scale.

## Part 1 — playout cap randomisation

KataGo §3.1, built as specified: on a proportion `p` of turns the search runs the full
cap `N` and the position is recorded; on the rest it runs a fast cap `n` with the root
Dirichlet noise off, and **nothing is recorded**.

`t12h-pcr`: `p = 0.3`, `(N, n) = (256, 64)`, realised mean **121.6 sims/move in every one
of 2124 generations**. Muon at `lr = 2e-2` polar, 1024 games in flight, batch 4096,
cosine over 4200 steps. 12 h wall clock, ended at step 4311 fully annealed (the cosine
reached `lr_min` at 4200 and it ran 111 steps there). Control: `t7h-muon`, the same
optimiser and rate at uniform `n = 128`, which died at 5.65 h — so the comparison is
*controlled* over the first 20 006 training seconds and solo after that.

### Three things the `search.md` §11 seam did not cover

⚠️ **`budget[b]` alone buys correctness and none of the saving.** `simulate` hands all `B`
staged leaves to the encoder every iteration whether or not they are active, so filling
`budget` with 64 and leaving the loop at `range(config.n)` produces a perfectly correct
tree of 64 simulations **at the price of 256** — and every counter in §15 says it worked.
`self_play_move` now owns `budget` and the iteration count together. The budget is tied
across the batch, because sorting the batch so the active set is a contiguous prefix
buys nothing either: measured 2026-08-07, encoder throughput per board is flat from 4096
boards down to 128 (97.2 %).

⚠️ **§4's value parity counted positions in the pending block**, which is the distance
from the end of the *game* only while the recorded plies are consecutive. It now reads
the side to move off the control word: `z = -r · sign(control) · sign(control_T)`.
Identical on a dense game; on an 8-ply game recorded at plies 0, 3, 6 the old rule
**inverts every sign**.

⚠️ **`buffer.append` does two jobs**, and "do not record cheap turns" written as "do not
call it" loses the closure of every game that *ends* on a cheap turn — 70 % of them at
`p = 0.3`. Those records sit in `_pending`, the slot resets, and the next game to finish
there flushes them under **its** result. No exception, no counter.

### The result, on the 2026-08-07 joint scale

| axis | gap, PCR − control |
|---|---|
| equal step, last paired point (1805) | **+36 ± 70** |
| equal training seconds, last three points | +45, +46, **+57** |
| equal recorded positions | **+413 to +560** |

Positions to reach a rating: **6.0× fewer at Elo 200, 4.4× at 400, 3.7× at 600**.

**And most of that is the record rate coming back out.** PCR records 30 % of plies and
runs generations 3 % cheaper, so at equal seconds the control gets **3.24×** more
positions; if PCR's positions were worth exactly 3.24× each the two would tie on the
clock. They are worth 3.7–6.0×, so the *net* advantage is **1.14× to 1.85×, declining
with rating**. That is why the wall-clock gap is +36 to +57 and not +400.

⚠️ **+36 ± 70 is not significant on its own.** What carries weight is that the sign is
positive at all 10 matched steps and all 9 equal-second points, not the size of any one.

⚠️ **KataGo's "more games are played" benefit is essentially absent at this operating
point, and that is our choice, not a property of the method.** `moves_per_phase ×
games_in_flight` fixes plies per generation, so extra games come only from cheaper
generations — and 121.6 against 128 is 5 %. KataGo's own recipe is `0.375 N`; ours is
`0.475 N`, because it was deliberately cost-matched to the control. So this run is a
**pure policy-target experiment**: is a 256-sim target on 30 % of positions worth more
than a 128-sim target on all of them, at equal wall clock? It is, slightly.

Health over the whole run, trailing-20 windows: decisive 0.229 → **0.825**, threefold
0.770 → **0.057**, root `max_pi` 0.225 → 0.328 monotonically, KL bottoming at generation
860 and **rising to 0.375** — the search pulling further ahead of the policy, which is the
healthy direction. `weight_norm` ran 139 → 446 by generation 1260 and then **went flat**,
which answers the muon entry's standing warning that a long run "should not be launched
without deciding what bounds it": the cosine anneal bounds it, at least to 11.5 h.

⚠️ **The `self_play/*` counters mix both tiers** (63 % of *simulations* come from the
expensive tier, 30 % of *moves*), so `mean_root_edges_visited`, `search_disagrees_frac`
and the depth statistics do not describe the training targets and are not comparable to a
uniform run's. `gradient/*`, `buffer/*` and the terminal counters are unaffected. The
recorded tier's root statistics are recoverable exactly from the replay buffer after the
fact; `search_disagrees_frac` is not, because the record does not carry the prior.

## Part 2 — the rebased scale, and the axis it made room for

### Why the old zero had to go

It was the frozen random-init network at 64 simulations, held as a file so that it could
not move with a torch version. That fixed the wrong half:

- **it was never a network, it was a network plus a search.** When §6.1a's root terminal
  sweep landed the anchor started finding every mate in one — stronger, with its file
  untouched, and the origin of the curve moved in silence;
- `checkpoints/` is gitignored, so the origin of every published number was one untracked
  blob;
- it was saturated: `anchor vs t12h-pcr@1405` and every pairing above it returned
  `0-0-36`, a full 36 games each for no information.

`Search.random_move` is implemented **outside** the search on purpose — a one-simulation
search would inherit §6.1a and reintroduce exactly the coupling being removed. Verified:
0 illegal moves in 7 680 plies, χ² = 13.5 on 19 df against the flat distribution, and
byte-identical move sequences under five different `SearchConfig`s.

Two silent bugs fell out on the way. A pool without the anchor made `fit_elo` **invent a
phantom** at Elo 0 that had played nothing — it unions the anchor into its name list, so
its own guard is dead code — and every rating would then be pinned to the `prior`. And in
a joint league the anchor's cost axis was written **null instead of 0.0**, because `cost`
is keyed by run and the anchor belongs to none; it is visible as the empty `euros_spent`
on the `anchor` row of `logs/curve-joint-pcr.csv`.

### The rebase moved the zero by about 30 Elo, which is a null result worth having

39 players, 205 pairings × 36 games = 7 380 games, 79.5 min, dispersion 0.673 over 5 538
decisive games. `logs/league-t12h-pcr-rebased.{json,log}`, `logs/curve-t12h-pcr-rebased.{csv,png}`.

| player | Elo | ±95 % |
|---|---:|---:|
| `random` | +0 | pinned |
| `init:n1` | +30 | ±50 |
| `init:n4` | +16 | ±49 |
| `init:n16` | +10 | ±48 |
| `init:n64` | **−30** | ±53 |

`init:n64` **is** the pre-2026-08-08 anchor. It sits 30 Elo *below* random play and
indistinguishable from it, so the old scale was not inflated in level — the rebase buys
durability, not a correction. It also means the ladder rungs each resolve to ±50 instead
of saturating, which is what makes the bottom of the scale estimable at all.

### Search is worth nothing to an untrained network, and ~+155 Elo/doubling to a trained one

| network | Elo per doubling of search |
|---|---:|
| untrained ladder (n = 1/4/16/64) | **−9** |
| `t12h-pcr@101` | +13 |
| `@1202` | +125 |
| `@2206` | +156 |
| `@3109` | +152 |
| `@4213` | +155 |

It climbs from nothing and **plateaus by step ~2200**. A random value head propagates
noise deeper, so deeper search buys nothing and mildly hurts; the network has to be able
to use the search before the search is worth anything. Nothing else in this project could
see that, because the budget was a constant.

### ⚠️ Gate 2's slope is a function of the evaluation budget

| eval budget | Elo per decade of training seconds |
|---|---:|
| n = 16 | **+408** |
| n = 64 | **+703** |
| n = 256 | **+793** |

Gate 2 wants +500. It clears at 64 and 256 and **fails at 16**. Training and search are
complementary — the n=64 to n=256 gap widens from +44 at step 101 to +288 at step 4213 —
so "Elo per decade" is not a property of a training run. **Every Gate 2 claim has to carry
its `n`**, which is §3's own rule finally biting.

### The exchange rate

| step | train s | Elo @64 | Elo @256 | seconds for n=64 to match | ratio |
|---:|---:|---:|---:|---:|---:|
| 101 | 1 146 | +57 | +101 | 1 660 | 1.45× |
| 1 202 | 11 998 | +674 | +872 | 22 685 | 1.89× |
| 2 206 | 21 849 | +865 | +1 123 | — | beyond the run |
| 3 109 | 30 692 | +924 | +1 243 | — | beyond the run |
| 4 213 | 41 587 | +1 035 | +1 323 | — | beyond the run |

`n = 64` tops out at **+1 050** after the whole 11.5 h. So past step ~2 200, **quadrupling
the search beats any amount of training this run reached**. Early on it is a trade at
1.5–1.9×; late it stops being a trade.

⚠️ The `n = 16` and `n = 256` series are five points each against `n = 64`'s twenty-four,
so their slopes are the least resolved numbers here, and the exchange ratios interpolate
the `n = 64` curve.

## What this does not say

⚠️ **This is a fresh fit.** Comparable within this report and to nothing else — including
the +954 the same checkpoint scored on the morning's joint scale. Different
Bradley-Terry fits share a zero but not their units, so the three leagues published
before today stay on their own scales. `init:n64` is in the pool precisely so that
*future* leagues relate by a measured offset rather than by assertion.

⚠️ **Nothing here is an absolute rating.** +1 323 self-anchored is not 1 323 Elo on any
published scale, and no measurement in this project constrains what it is. `evaluation.md`
§6 (layer 2b) is the missing step and it has never been run.

⚠️ **The control died at step 1905**, so PCR's late behaviour is measured against its own
early slope and not against anything running beside it.

## One thing that should be re-measured before it is acted on

`roadmap.md` E2's case rests on *"threefold is 45–85 % of how our games end"*. In
`t12h-pcr` the trailing-20 threefold rate ended at **5.7 %** with the decisive rate at
82.5 %. Whatever is true of the policy's threefold blind spot on the layer-0 suite, its
*consequence* has collapsed over a long run, and E2's priority should not be argued from
the old number.
