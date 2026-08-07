# Roadmap

What is left to build, and the decisions still open. Written 2026-07-29 after the
specification was frozen; estimates are in working days with both of us on the task.

Everything already delivered — A0 through D1, with the annotations each step
collected as it landed — moved to
[the build log](journal/2026-07-31-build-log.md) on 2026-07-31, because a roadmap
that is four-fifths history stops being readable as a plan.

⚠️ **Both engine tracks, the search and the training loop are closed.** A0, A1, A2,
B0, B1, B2, C1 and C2 are done, and so are D0 and D1. The CUDA engine is perft-green,
the network runs boards to logits in one launch, **the environment costs 2.2 % of a
node** and the tree 4.4 % on top of it, and **Gate 1a cleared at 56 996 useful evals/s
at `n = 800`, `B = 4096`**. What remains is C4's pilot, C5's full run, and the four
Track D steps that need checkpoints or a second process.

⚠️ **The shape of the project changed at C2.** Everything up to it had a written
specification and an oracle before a line was typed, which is why the estimates held.
What is left has neither in the same sense: a training run's failure mode is a curve
that is merely worse than it should have been, and no perft catches that. Treat the
estimates below as estimates.

## Joining the tracks

### ~~C3. Evaluation protocol~~ → [Track D](#track-d-evaluation)

**Promoted to its own track on 2026-07-30.** C3 was one 3-4 day line item covering
everything from a PGN writer to a preregistered superhuman claim, which is six
pieces with different dependencies, and two of them can start today while two cannot
start until C2 produces checkpoints. It is now Track D, specified in
[`evaluation.md`](reference/evaluation.md), and the honest estimate is **8-12 days**, not 3-4.

### C4. Pilot, 3-5 days

Small-budget run for the first curve points and the Jones slope check. The
simulations-per-move sweep runs here, since it needs both a working loop and an Elo
protocol to be measured against. Gate 2 falls here.

### C5. Full run and write-up, 10-20 days

Mostly waiting. The write-up starts during D2, as soon as the first curve points
have error bars.

## Track D: evaluation

Specified in [`evaluation.md`](reference/evaluation.md), which is the contract for this track the way
`spec.md` is for A and B. It is a **draft**: §2 and §3 are ready to freeze, the rest
firms up as the track is built. [The eval prior-art pass](journal/2026-07-30-eval-prior-art.md)
holds the verification of every claim it borrows from AlphaZero, KataGo, SAI and lc0.

The deliverable is not a rating. It is a **sequence of (euros, Elo) points with error
bars and a stated configuration**, which is a harder object than "measure how strong
it is" and drives the whole decomposition.

Four evaluation layers, four different questions, four different costs — `evaluation.md`
§1 has the table. The steps below are ordered by dependency, not by layer number.

### ~~D2. The league and the rating fit — layer 2~~ ✅ landed 2026-07-31

`brokefish/eval/{match,elo,league,curve}.py`, specified in `evaluation.md` §5.4,
built in [the D2 journal entry](journal/2026-07-31-d2-league.md), 56 checks in
`tests/test_league.py`. Everything below was the plan; what shipped differs in three
places, all recorded in §5.4: **no opening book** (8 random legal plies instead —
UHO breaks draws between engines strong enough to hold a balanced position, which is
not our failure mode), **the fixed SAI calendar rather than variance-proportional
pairing** (which needs an online fit and buys nothing yet; `EloFit.predict` is the
function it will need), and **a score-based Elo scale** rather than BayesElo's
draw-model one.

✅ **Three curves measured, 2026-07-31 and 08-01**: `t4h-n64` +74 ± 17, `t4h-fix`
+53 ± 13 (both at 64 eval sims), `t12h-n128` +451 ± 39 (at 128 eval sims, so a
different scale — §3 makes the simulation count part of it). 4 677 / 6 027 / 5 186
games; **9.3 s per pairing at 64 sims and 38 s at 128**, the latter because decisive
games run 123-138 plies against 60-80. The cost arithmetic below assumed ~17 min for
1000 games at `n = 800` and is superseded by those measurements.

The curve itself. A checkpoint league where the opponent is our own past self, so the
opponent strength escalates for free and no external process is involved.

- the **frozen anchor**: the random-init network, pinned at Elo 0 in the fit. It is
  the origin of the cost-versus-Elo curve by construction and it detects scale drift.
  Precedent: lc0 anchors its chart at "the first net".
- one **global** Bradley-Terry / BayesElo fit over the whole graph of games, never a
  chain of pairwise deltas (`evaluation.md` §5.2).
- **variance-proportional pairing**: play a pairing with frequency proportional to
  `p(1−p)` under the current fit. KataGo's rule; SAI's fixed ±1,2,3,6,8,12 schedule
  is the fallback if the online fit is not ready.
- the curve-point record of §5.3, in which **`euros_spent` is written by the same
  writer as `elo`** — two files means a hand join six weeks later.

⚠️ **Buildable before C2.** The league needs a set of nets of differing strength, not
a training run: a random-init net plus a few deliberately-degraded copies gives a
synthetic ladder with a known ordering, which is a better test of the fit than real
checkpoints because the right answer is known in advance.

Cost arithmetic: 1000 league games at `n = 800` ≈ 6.4×10⁷ evals ≈ 17 minutes, both
players on device. Layer 2 is not a budget problem.

### D3. The external match harness, 2-3 days

Layers 2b and 1 need something this repository has never had: **one game at a time,
against a process**. Everything here is throughput-shaped — `B = 4096` in the search,
`B = 16384` in `bench_loop` — and at `B = 1` the encoder runs one CTA on 24 SMs.

So this is an **async scheduler**, not a loop: `N` games in flight against `N` UCI
processes, our side batching whichever games are currently waiting on us. Needs D0.

⚠️ **Our opponents are CPU-only, which is a gift.** CCRL-rated engines run on the
CPU while our net runs on the GPU, so the process pool and the self-play loop do not
contend. The 8 GB budget is untouched by this track.

### D4. Calibration — layer 2b, 1-2 days plus wall clock

Converting the self-anchored scale to a published one. `evaluation.md` §6 and §7.1.

⚠️ **Submission to CCRL is closed** (§7.1): the list is CPU-only and refused a GPU
exception for Lc0. The anchor is therefore a rating we *borrow*, not one we are
awarded, and a borrowed rating only transfers if our match reproduces the conditions
it was measured under.

Two parts, because "reproducible" and "rated" are different properties:

- **the ladder** — one strong engine at fixed node counts. Reproducible on any
  machine, spans roughly 1500-3000 monotonically, and keeps every internal number
  free of a time control. ⚠️ Not `UCI_Elo`: its mechanism is a randomised bias over
  MultiPV candidates, so it plays strong-with-blunders, and beating a blunderer is a
  different skill from beating a 2000-rated engine — the *measurement* does not
  transfer, whatever the label says.
- **the anchors** — 3-4 distinct CCRL-rated engines at ~300-400 Elo spacing, run at
  their rated configuration. These convert node counts to absolute Elo.

Each anchor also plays one match under CCRL conditions, which measures the offset
between our protocol and theirs. Measured once, own error bar, converts the ladder.

**Which engines is the open decision**, and it is the next one due (`evaluation.md` §11).
Selection criteria: frozen public release, single-threaded, CPU-only, ≥150 games in
CCRL's "pure" list, UCI, open source.

### D5. The gate — layer 1, 1 day plus match time

The preregistered claim: threshold, list, time control, hardware and search budget,
fixed **before** the run. A basket of three or four rated engines rather than one,
since a basket costs the same per game and cannot be defeated by a single
anti-computer blind spot.

Sizing from `evaluation.md` §9: `se(Elo) ≈ 347·√(1−d)/√N`. At the top of the scale draws
dominate, so 100 games resolves ±37 Elo — enough only if the claimed margin clears
~40. A gate expecting to land near its threshold needs ~750 games for ±25.

### What Track D has already settled

| | |
|---|---|
| **no checkpoint gating** | AZ's choice. It keeps the x-axis well defined — a rejected candidate costs euros and yields no curve point — and SAI names gating as an aggravating factor for Elo inflation. **C2 therefore does not depend on this track.** KataGo's 200-game check still runs in layer 0 as a non-blocking logged diagnostic |
| **fixed simulations per move**, not time control, in layers 0/2/2b | thermal drift, hardware independence, and it makes `n` an axis rather than a confound. ⚠️ AZ is *not* precedent — it rated at 1 s/move |
| **greedy move selection in evaluation** | temperature 0, no Dirichlet, following AZ. Not the same protocol as self-play |
| **target a CI width, not a game count** | the draw fraction drifts over the run, so fixed `N` over-measures early points and under-measures late ones |
| **two cost numbers** | training compute only on the curve's x-axis, which is what makes it comparable to AlphaGateau; total project cost published separately |
| **tablebases: never ours, whatever the opponent is rated with** | §7.2, superseding the two-option sentence in [Measuring strength](#measuring-strength) below |

## Track E: sample efficiency

Opened 2026-08-01, after the `t12h-n128` run.

⚠️ **The objective is Elo per hour. Sample efficiency is a proxy for it, and the two
are not the same** — corrected 2026-08-01, because the first draft of this section
stated the proxy as the objective and that inflates every conclusion in it.

The gap is exactly the cost of the thing that bought the sample efficiency. Measured
on the same landmark:

```
25% decisive at        positions        wall clock
t4h-n64                 7.29 M           2.79 h
t4h-fix                 8.03 M           2.97 h
t12h-n128               2.69 M           1.80 h
                       2.7-3.0x         1.55-1.65x
```

**A 3.0× sample-efficiency gain is a 1.6× Elo/h gain**, because doubling `sims` costs
2× per position. So the standing rule for this track: *an intervention that buys
sample efficiency by spending compute per position must beat its own cost factor to
be worth anything at all.* Sample efficiency is the right proxy for interventions
that are **free per position** — auxiliary targets, SWA, better inputs, a better
optimiser — and a misleading one for anything that buys quality with FLOPs.

### E0. The lever, and what it is actually worth

⚠️ **`records per generation is 10 240 in every run so far`** — it is
`moves_per_phase × games_in_flight` and **does not depend on `sims`**. So doubling
the simulation budget costs zero extra positions; it costs FLOPs per position. That
makes the n=64 → n=128 comparison a clean sample-efficiency experiment, and it is the
sharpest result the project has:

The landmark is the **25 % decisive-game crossing**, over a 20-generation trailing
window. Positions are counted as `generations × 10 240`, which is the one measure
valid across all three runs — `t4h-n64` predates the position-denominated cadence, so
its steps and its samples are not in a fixed ratio.

| | total positions | steps | **25 % decisive at** | league |
|---|---:|---:|---|---|
| `t4h-n64` | 10.4 M | 1292 | step 1003 = **7.29 M positions** | +74 ± 17 @64 sims |
| `t4h-fix` | 10.6 M | 2106 | step 1587 = **8.03 M positions** | +53 ± 13 @64 sims |
| `t12h-n128` | 18.1 M | 3583 | step 525 = **2.69 M positions** | +451 ± 39 @128 sims |

**2.7–3.0× fewer samples to the same behavioural landmark**, and the final decisive
rate is 66.1 % against 32.1 % / 32.3 %. The Elo slope went from +24.6 ± 6.2 to
+332 ± 26 per decade of compute — 66 % of Jones' law against 5 %.

The other half of the same measurement is the KL direction, which is what tells us
the mechanism rather than the size:

```
n=64 : KL falls 0.443 -> 0.140    the search converges TO the policy -- no signal left
n=128: KL bottoms 0.123, RISES to 0.186    the search stays AHEAD of the policy
```

⚠️ **The scales are not comparable.** `t12h-n128`'s league ran at 128 simulations and
the other two at 64, and `evaluation.md` §3 makes the simulation count part of the
scale. The anchor's own draw rate proves the scale moved: 41.4 % at 128 sims against
82.8 % at 64. The `+378` internal rise is measured within one scale and stands; the
comparison to `+74` is not licensed until a 64-sim league is run on `t12h-n128`.

⚠️ **A measurement that argued the wrong way, recorded so it is not repeated.** A
sims sweep on the *final `t4h-fix` checkpoint* showed more simulations making the
target **flatter** (max π 0.154 → 0.132 from n=32 to n=256), and was used to argue
against raising `sims`. That checkpoint's value head had already collapsed to
std 0.023, and in that regime Q carries nothing so PUCT's asymptote is π → P. **The
collapse is a consequence of training at n=64.** Measuring the endpoint of a
degenerate run and using it to justify continuing the degenerate run is circular; a
sims sweep is only informative on a healthy net.

### E1. More signal per position

⚠️ The rows marked **free** cost nothing per position, so sample efficiency and Elo/h
move together for them. The rows marked **paid** buy quality with FLOPs and have to
clear their own cost factor.

| | what | source | cost |
|---|---|---|---|
| **E1.1** | **paid** — **`sims = 256`.** Two points on this curve both moved hard; this is the third. ⚠️ It costs 2× per position, so on the real axis it must reach the landmark **under 1.80 h** — i.e. under ~1.41 M positions against n=128's 2.69 M, a **1.9× sample-efficiency gain merely to break even**. The previous doubling gave 2.7×. If this one gives under 1.9× the lever has saturated and the track's weight moves to E1.2/E1.4/E1.5 | ours, E0 | one run |
| **E1.2** | **paid, but cheaply** — **Playout cap randomisation.** Large `N` on a proportion `p` of turns, a small `n` on the rest. ⚠️ **Corrected 2026-08-07 against the primary text**: *"Only turns with a full search are recorded for training"* — the cheap turns are **not** recorded, for value or for anything else. The value target gains **indirectly**, because the same compute plays more games and the outcome is *"one noisy binary result per entire game"*. An earlier draft of this row said the cheap turns feed the value target; they do not. Their main run: `p = 0.25`, `(N, n) = (600, 100)`, annealed to `(1000, 200)` after two days, **Dirichlet noise and every explorative setting disabled on the fast searches** | KataGo §3.1; ablation Table 2: removing it costs **1.37×**, over fixed N ∈ {100,150,200,250,600} | medium; `Search.budget` is already the per-game tensor and [`search.md`](reference/search.md) §11 prices it. ⚠️ §11's "five lines" is correctness only — the encoder evaluates all `B` staged leaves every simulation, so a batch holding mixed budgets **costs the maximum, not the mean** |
| **E1.3** | **free** — **Forced playouts + policy target pruning.** `n_forced(c) = (k·P(c)·ΣN(c'))^½`, k=2, PUCT set to ∞ until met; then subtract those playouts from the *target* unless the move proved good | KataGo §3.2, ablated as NoForcedTP. ⚠️ Their motivation is our FPU bug stated in general form: *"even if a Dirichlet noise move was good, its initial evaluation might be negative, preventing further search"* | medium |
| **E1.4** | **free** — **Auxiliary policy target** — predict the **opponent's next** policy, `w_opp = 0.15` | KataGo §4.1: *"modest but clear benefit… nearly costless… deserves attention"*. **Fully game-agnostic**, unlike ownership and score | low |
| **E1.5** | **free** — **Moves-left head** | lc0 ships one with its own loss weight; it is the chess analogue of E1.4 on the value side | low |
| **E1.6** | **free** — **Stochastic weight averaging** — snapshot per ~250k samples, EMA of 4 at decay 0.75 | KataGo, main run | very low |
| **E1.7** | **free** — **Balanced win/loss sampling from the buffer** | ELF OpenGo App. C: the value head over-estimated one colour, causing premature resignation and collapsed replay diversity | low |

KataGo's own summary of why this tier is the tier: *"enriching the training data with
additional targets is valuable when data is limited or expensive."*

### E2. Raise the ceiling — positions that currently teach nothing

**Total move count and a slice of history** ([`spec.md`](reference/spec.md) §7.2 inputs). AZ's Table S1 carries
`T = 8` stacked positions and a total-move-count plane; we have `T = 1` and no move
count.

⚠️ **Measured consequence, not a suspicion**: the policy scores **0.035 on
`avoid_threefold` against a 0.036–0.044 chance baseline** at every checkpoint, while
the search does it at 0.86 — and threefold is **45–85 % of how our games end**. With
`T = 1` and only the current position's `rep` flag, nothing in the input distinguishes
the repeating move from any other. It is an information-theoretic impossibility, and
no quantity of samples fixes one. Sample efficiency is capped at zero for that slice
of the data.

Cheapest faithful step: total move count as one embedding table (parallel to
`emb_clock`), plus the last move as two 64-row per-board tables. The full `T = 8` is
256 tokens and a real architecture change.

### E3. The exchange rate — compute per sample, not samples

These do **not** improve sample efficiency. They change what a unit of sample
efficiency costs, which is why they follow E1 rather than lead it.

- **fp8 for the rollout matmuls.** Measured on this card (`CLAUDE.md`): **e4m3→fp32 at
  41.6 TFLOPS against fp16→fp32 at 18.0**, a 2.3×. It buys E1.1 and E1.2 outright.
  ⚠️ Ordering matters: if the curve saturates at n=128 this is worth much less, so it
  follows E1.1's answer. ⚠️ It also needs its own equivalence story — the tree is
  currently checked bit-for-bit against the fp32 reference, and an fp8 rollout can
  only promise agreement on the *move*, not on the numbers.
- **muP, then progressive network sizing.** [`model.py`](../brokefish/nn/model.py) already defers this in
  place: *"Real muP treats embeddings and readouts differently and that is a training
  decision (C2), not a forward-pass one."* muP's payoff is hyperparameter transfer
  across width, which is what makes KataGo's progressive sizing — (6,96) → (10,128) →
  (15,192) → (20,256) — safe rather than a re-tune at every step. They belong together.
  A smaller net early also has a real sample-efficiency argument: fewer parameters to
  fit per position.
- ~~**Muon.**~~ ✅ **Measured 2026-08-07 and it belongs in E1, not here** — it is *free*
  per position (Newton-Schulz is under 0.01 % of a step), so it never needed a cost
  factor to beat. `t7h-muon` against an AdamW control at 128 sims: **+143 ± 26 Elo at
  step 1905** in one joint league, **~1.4× less wall clock to any fixed rating** between
  Elo 200 and 400, KL rising in both.
  [The entry](journal/2026-08-07-muon-curve.md) has the protocol and four caveats, of
  which two matter for what comes next: **the AdamW rate was never re-screened** under
  the same 780 s guard that chose muon's `2.0e-2`, so this is +1.4× against the baseline
  we have been running rather than against a tuned one; and **`weight_norm` goes
  108.6 → 369.3** where the control's goes 108.6 → 117.2, since a fixed-RMS update makes
  `adam_wd` on the aux group irrelevant to the matrices. ⚠️ **Decide what bounds the
  norm before a 24 h muon run.** The run died at 5.65 h of 7 h on a full disk, so nothing
  is known about the cosine tail.

### E4. Looped transformer and HRM-adjacent recurrence

More computation per position with **no extra parameters and no extra positions** —
which is the architectural form of exactly what raising `sims` just bought. On a
project whose x-axis is compute rather than parameters, a 6.4M-parameter network that
thinks longer is on-thesis. Adaptive depth per position also fits chess, where
tactical positions want more work than quiet ones.

Last only because it is the largest bet, not because the axis is wrong.

### E5. The blocker that gates E2 and E4

**Anchor versioning, ~1 hour, and it comes first of anything architectural.**
`eval/league.py`'s `load_engine` builds a fixed `BrokefishNet`, so any change to the
input or the architecture makes `checkpoints/anchor.pt` unloadable and breaks the
entire cost-versus-Elo scale. The match harness takes *evaluators* rather than
architectures, so the frozen anchor can still play — it needs a versioned constructor
and nothing more. Without it, every architectural experiment starts a new scale and
none of them can be compared.

### What Track E has already settled

| | |
|---|---|
| **the objective is Elo/h, not Elo per sample** | the same measurement reads 3.0× on samples and 1.6× on the clock. An intervention that buys quality with FLOPs per position must beat its own cost factor |
| **simulations per target is a first-order lever** | 2.7-3.0× fewer positions and **1.6× less wall clock** to the same landmark, n=64 → n=128, with positions-per-generation held constant at 10 240 |
| **the optimiser is a free 1.4× on the clock** | Muon + Polar Express on the 98.6 % of parameters that are matrices: **+143 ± 26 Elo at step 1905**, ~1.4× less wall clock to any rating from 200 to 400, at zero cost per position. Against an untuned AdamW rate, and with the weight norm tripling — [2026-08-07](journal/2026-08-07-muon-curve.md) |
| **a behavioural landmark overstates a rate gain** | the 25 %-decisive crossing reads **3.5×** for muon where the Elo curve reads **1.4×**, because the landmark fires at Elo ≈ 30, inside the transient. E0's 2.7–3.0× on `n = 64 → n = 128` is the same measurement and carries the same optimism until it is re-read against an Elo curve. ⚠️ The landmark also has an early false crossing at ~generation 20 in every run; take the **last** up-crossing |
| **the repetition problem was a symptom, not a cause** | threefold fell **82.8 % → 4.6 %** of terminations over `t12h-n128` with no change to the inputs. E2's representation gap is real but it was not what capped the previous runs — the search simply stopped needing a draw |
| **game length and the ply cap are settled at n=128** | mean plateaus at ~180 plies, the cap fires on **0.50 %** and is flat. `--max-plies 512` is comfortable; `--mean-plies 250` is the right buffer sizing, since `window_games` binds first |
| **the gradient is well behaved at n=128** | `grad_norm` settles at ~0.33 and stays, saturation 0.0000 throughout, `--grad-clip 5.0` fires 16.7 % in the transient and 0 % after. Contrast `t4h-n64`: monotone collapse to 0.19, and a 1.0 clip firing on 99 % of the first 300 steps — an implicit lr schedule rather than a safety net. ⚠️ `weight_norm` drifts +5 % at `adam_wd = 0.01`, flat at 0.1; watch over 24 h |
| **the KL direction is the diagnostic** | falling KL means the search has converged to the policy and the loop is an expensive identity function; rising KL after an initial fall is what a working loop looks like |
| **do not down-weight the value loss** | KataGo *up*-weights it (`c_g = 1.5`); AGZ's 0.01 was supervised-only and its own text conditions the 1:1 choice on data abundance |
| **do not starve the policy loss** | ELF OpenGo: cross-entropy coefficient at 1/362 capped the model at amateur level |
| **global pooling does not apply** | KataGo §3.3 gives *convolutional* nets the global context our 32-token full-attention transformer already has |
| **initialisation is unspecified in the literature** | neither AGZ, AZ, KataGo nor ELF states a scheme (all four texts checked 2026-08-01). lc0's source uses `glorot_normal` throughout, which for our head geometry is **1.27× sharper** than our `d^-0.5`. Ours is not an outlier and this is not an ablation worth spending a run on |

## Measuring strength

⚠️ **Partly superseded by [`evaluation.md`](reference/evaluation.md) as of 2026-07-30.** Two statements
below no longer hold: submission to CCRL is not available (§7.1 — the list is
CPU-only and refused a GPU exception for Lc0), and "no tablebases on either side or
the same on both" omits the option AlphaZero actually used and that §7.2 adopts. The
"SPRT or fixed-N" phrasing is also split in §9, since they answer different
questions. What survives unchanged is everything about the scale conversion, which is
why the section stays.

Engine rating lists and FIDE ratings are separate scales, and the conversion between
them is contested. Two rules of thumb in common use disagree materially: one holds
that the scales intersect at 2800 with CCRL points worth about 0.70 FIDE points, and
another proposes `FIDE = (CCRL - 1600) × 0.85 + 1600`. Applied to CCRL 3000 they
give roughly FIDE 2940 and FIDE 2790 respectively. The reported agreement is best
near 2900 and worst below 1800 and above 3500.

Three consequences for the protocol.

**Preregister the threshold and the configuration.** Fix what counts as superhuman,
on which list, at which time control, on which inference hardware, at which search
budget, before the run rather than after it. Strength varies with all of them.

**Anchor to a third party.** ⚠️ **Only the fallback is available.** Getting listed
would remove the methodology argument entirely, but CCRL is CPU-only and declined a
GPU exception for Lc0, so the remaining option is the second one: reproduce their
conditions and say so. `evaluation.md` §7.1 and D4 build the protocol-offset measurement
that makes a borrowed rating transfer.

**Choose the threshold with margin.** Since the conversion is disputed by roughly
150 Elo at the relevant point, a target that clears the highest FIDE rating ever
achieved under the pessimistic conversion is worth more than one that clears it
under the optimistic one.

The measurement itself: paired games from UHO or TCEC books with both colours from
each opening, error bars from SPRT or fixed-N with a stated confidence interval,
enough games that the interval excludes the threshold, no tablebases and no
pondering on either side or the same on both, hardware and time control reported for
both engines, and all PGNs published. An assertion in the code that the anchor
engine never touches a training tensor.

## Decisions with deadlines

Only the open ones. A decision that has been made lives in the document that owns
the parameter — [`spec.md`](reference/spec.md) §11 says which — and the record of
when and why it was made is in [the build log](journal/2026-07-31-build-log.md).

| decision | needed by |
|---|---|
| **`lr` and its three drop points** | the first ablation on top of the AZ reproduction, and the next one due. `training.md` §12 check 1 is the instrument, and the first sweep already contradicts AZ's `0.2`. Provisional until a rate reaches `KL ≈ 0` |
| **which calibration engines** | start of D4, and also due now. Criteria in D4 above; `evaluation.md` §11 |
| whether the external anchor is a node in the league fit or a separate affine map | start of D2. Folding it in deletes the two-scale hazard but pulls D3 earlier |
| preregistered superhuman threshold and configuration | start of D5, and it cannot move afterwards |
| **simulations per move** | measured in C4. This is the one that decides whether the run fits the budget at all — see the Calendar |
| adaptive budget by KL on completed Q | after C4, as an optimisation |
| playout cap randomisation | unscheduled. `search.md` §11 prices it |

## Risks

**The 5-50M games anchor.** A five-fold error in how many games are needed is a
five-fold budget error, and it multiplies every number in the Calendar below. The
pilot's Jones slope is what recalibrates it. This is the largest open risk.

**The training loop has no oracle.** C2 is the first phase where being wrong is
silent: there is no perft and no independent implementation to be differentially
tested against, and the failure mode is a curve that is merely worse than it should
have been. `training.md` §12's nine checks are what replace one, and
passing all nine is consistent with training a subtly wrong objective competently.

**The untrained anchor is seed-dependent.** `evaluation.md` §5.1 pins one
random-init network at Elo 0 as the origin of the whole curve. Over six inits its
puzzle score moves by 0.29 (`state.md`), so the anchor measures the seed as
much as the architecture. Whether that matters to the *rating* fit, as opposed to the
diagnostic, is a D2 decision that has not been taken.

**The 8 GB budget.** Measured 2026-07-29 and closed for the loop as specified: peak
near 2.4 GB, because self-play and training alternate and the two activation blocks
never coexist. It binds in exactly one place, the replay buffer, which is why the
policy target is stored sparse over the legal moves in mask order and the mask is
recomputed at training time — 156 bytes a position against 4.2 KB, so 312 MB for 2M
positions rather than 8.4 GB. The cost is one movegen call per training sample.
`training.md` §5.1 owns the schema.

**Second-order legality.** ✅ Retired 2026-07-30 by perft on both engines and 26
single-rule mutations. The story is in the build log; it is listed here only because
it was the risk that would have invalidated every downstream Elo number silently.

## The critical path

Everything with a specification and an oracle is done. What is left is one loop, a
pilot, and the parts of Track D that need either a checkpoint or a second process.

```
C2 ✅ ──> C4 (pilot) ──> C5 (full run)
D0 ✅ ──> D1 ✅ ──> D2 ──> D3 ──> D4 ──> D5
```

**On the path**: C4, then C5. C4 needs D2, because a pilot with no rating fit
produces GPU time and no curve point.

⚠️ **Revised 2026-08-01: [Track E](#track-e-sample-efficiency) now sits between D2 and C4.** The
pilot's job is to fix `n` and the config for the full run, and E0 measured a **3.0×
sample-efficiency difference** between two values of `n` that the pilot would
otherwise have had to discover at full-run prices. Every hour spent on E1 lowers the
cost of C5 by more than it costs.

**Off it, and startable now**: D3. (D2 landed 2026-07-31; the synthetic ladder is in
`tests/test_league.py`, where the fit recovers ratings it was never told.) D3 is an
async scheduler against UCI processes and shares nothing with the
training loop; our opponents are CPU-only, so the two do not contend for the card.

**Decisions rather than code**: the calibration engines (D4) and the preregistration
(D5). Both are listed above and neither is blocked by anything.

**Not parallelisable**: C5 is GPU time and does not compress.

## Calendar

⚠️ **The 1.3 hours per estimated day that held through A0-B2 no longer applies.**
It held because every one of those phases had a written specification and an oracle
before a line was typed. Nothing left has both. Treat what follows as ordering, not
as scheduling.

| milestone | state |
|---|---|
| environment, network, search, training loop | ✅ 2026-07-29 to 2026-07-31, A0-B2, C1, C2 |
| layer 0 and layer 3 diagnostics | ✅ 2026-07-30, D0 and D1, 31.7 s per checkpoint |
| the league and the rating fit | ✅ 2026-07-31, D2, 56 checks; three curves measured 2026-07-31/08-01 |
| **sample efficiency** | **[Track E](#track-e-sample-efficiency), opened 2026-08-01.** E0 measured; E1.1 is one run; E1.2-E1.7 are days, not weeks. **Sits before C4** |
| the external match harness | D3, 2-3 days |
| calibration and the gate | D4 and D5, both blocked on a decision, not on code |
| first curve points, Gate 2 | end of C4, plus the pilot's own GPU time |
| full run | see below — this is the open question, not a schedule |

⚠️ **The full run does not fit the stated budget at `n = 800`, and that is known
rather than overlooked.** The €100-500 figure and the 113-hour estimate this document
used to carry both assume **32 simulations per move**, from the superseded Gumbel
sizing; `search.md` froze `n = 800`, which is 25× the evaluations:

| | evals | at the measured 57.0k/s |
|---|---|---|
| 10M games × 80 plies × **32** sims | 2.56×10¹⁰ | ~113 h |
| 10M games × 80 plies × **800** sims | 6.4×10¹¹ | **~3,120 h**, 130 days on the 4060 |

On an H100 the same work is roughly 26 days at $3.95/h ≈ $2,500, against a stated
€100-500. `training.md` §13 owns this arithmetic and the decision that goes
with it: **establish that the AZ configuration hill-climbs at all at `n = 800` first,
then bring the simulation count down** with the Gumbel seam, playout cap
randomisation and C4's sweep. Demonstrating the climb is the scientific claim; making
it cheap is engineering that follows it, and doing them in the other order means
debugging a learning failure and a search approximation at the same time.

The first hill-climbing evidence therefore comes from a **truncated run**, not from
10M games.
