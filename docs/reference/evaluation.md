# Evaluation

**Status: draft, 2026-07-30.** This document proposes the evaluation protocol. It is
**not normative yet**: §11 lists the decisions it does not have the authority to make
and who has to make them. Everything outside §11 is a recommendation with its reason
attached, so that disagreeing with one part does not require re-deriving the rest.

Scope: how strength is measured, and the harness that measures it. The training loop
and the replay buffer are C2. The search is [`search.md`](search.md). The engine and
network contracts are [`spec.md`](spec.md); nothing here modifies them.

The deliverable this serves is a **cost-versus-Elo curve**, so the object being
produced is not a rating but a *sequence of (euros, Elo) points with error bars and a
stated configuration*. That is a stronger requirement than "measure how strong it is",
and it drives most of what follows.

---

## 1. Four layers, because they answer four questions

A single evaluation cannot serve all of them: they differ by two orders of magnitude
in latency and by three in what they cost.

| layer | question | opponent | budget | when | latency |
|---|---|---|---|---|---|
| **0 — regression** | is the run alive? | frozen anchor + generated suites | low `n`, ~50 games | every checkpoint | seconds |
| **2 — progress** | what is the curve? | our own checkpoints | fixed sims/move | continuous | minutes |
| **2b — calibration** | where is the curve on an absolute scale? | rated external engines | fixed nodes | every ~10× compute | hours |
| **1 — gate** | did the project succeed? | third-party-rated basket | their conditions | once | hours to days |
| **3 — diagnostic** | *why* is this net bad? | positions, not games | n/a | on demand | seconds |

Layers 0, 2 and 3 need no external process, no opening book and no second engine.
That is deliberate: the cheap layers are the frequent ones.

⚠️ **The numbering is not the order they run in.** Layer 1 is numbered first because
it is the claim; it executes last and exactly once.

---

## 2. Where evaluation sits relative to the tabula rasa boundary

Evaluation is measurement and sits **outside** the boundary. Allowed here and nowhere
else: standard opening books (UHO, TCEC), engines with published ratings, human-curated
puzzle sets. Precedent: the AlphaZero-Stockfish match in the *Science* paper started
from TCEC openings.

The boundary is not "evaluation may touch this data" but "**evaluation output may not
flow backwards**". Three concrete prohibitions:

1. No opponent's move, evaluation, or game ever enters the replay buffer.
2. No book position ever seeds a self-play game.
3. **No evaluation result selects a checkpoint, stops a run early, or picks a
   hyperparameter.**

⚠️ **The third is the one that will actually get violated.** "We kept the checkpoint
with the best puzzle score" is distillation from a human-curated set through a
one-bit channel, and it is not visibly different from ordinary good practice. The
curve is a record of what training produced, not of what evaluation selected from.
The only permitted use of an evaluation result inside the loop is **abort on
divergence** (§4), which is a liveness check and carries no chess opinion.

Two asserts to write, both cheap:

- the anchor engine's process handle and any book/puzzle tensor are unreachable from
  the training tensors — enforce by construction, not by convention;
- the replay buffer rejects any record whose provenance tag is not `selfplay`.

---

## 3. The unit of a game: fixed simulations, not time

**Recommendation, applies to layers 0, 2 and 3.** A game is played at a fixed number
of MCTS simulations per move. Four reasons:

- **Reproducibility.** Clocks on the 4060 drop to 1.38-1.5 GHz under sustained load
  and drift ±3 %. Every benchmark in `perf.md` is order-balanced interleaved
  because of it. A time-controlled match has no such defence, and a rating measured
  under one thermal state does not compare to one measured under another.
- **Hardware independence.** A rating at `n = 800` means the same thing on the 4060
  and on a rented H100. A rating at "one minute per move" does not.
- **It removes a confound and creates an axis.** `n` becomes a measured variable
  rather than a hidden one, which is exactly the simulations-per-move sweep the
  roadmap wants in C4.
- **It costs nothing to report.** The Elo of a checkpoint at `n = 800` and at
  `n = 100` are two different numbers, and both are interesting.

⚠️ **A rating is a function of `(net, n)`, never of `net` alone.** Every rating this
project reports carries its `n`. A curve point with an unstated `n` is not a curve
point.

⚠️ **AlphaZero is not precedent for this.** Its training-progress ratings came from a
**1 second per move** tournament, i.e. a time control. The four reasons above stand on
their own, but the recommendation is ours and not inherited — see
`2026-07-30-eval-prior-art.md` §2.4.

**Move selection during evaluation is greedy on the root visit count**: temperature 0,
no Dirichlet noise, in every layer. This follows AZ, which does the same, and it is
not the same protocol as self-play, which needs both for diversity.

**Exception: layer 1.** The gate is a claim staked against a third party's scale, so
it runs under whatever conditions that third party publishes — which today means time
control. That is a cost of anchoring, accepted knowingly, and it is the only place it
is paid.

---

## 4. Layer 0 — regression

The job is to notice within minutes that a run has diverged, not to measure strength.
It runs on every checkpoint and it never blocks; it alarms.

Contents, all self-contained:

- ~50 games against the **frozen anchor** (§5) at low `n`;
- the generated rule suites of §8.1;
- three scalars logged per checkpoint:
  - **value calibration** — predicted value against realised outcome, as a reliability
    curve. A value head that has stopped tracking outcomes is the earliest visible
    symptom of a diverging run and it precedes any Elo drop.
  - **policy entropy** — collapse to a delta and drift to uniform are different
    failures and both are visible here.
  - **draw rate and mean game length** — a self-play population that has collapsed
    to one line shows up here before it shows up in Elo.

⚠️ **`max |post-scale attention logit|` was a fourth scalar here and is not any
more**, removed 2026-07-30 after D1 implemented it and the implementing showed why
it does not fit. It is an fp16 overflow watch on the fused kernels, not a
measurement of a net: it has to reach inside `net.encoder.layers`, where the rest
of layer 0 treats the network as `(boards, control, rep) -> (policy, promo, value)`
and nothing else; it can only read the *torch oracle's* logits in fp32 rather than
the fp16 accumulator that actually overflows, so it reports health for a kernel it
never touched; and `CLAUDE.md` asks for it **during training**, where a NaN is
found at the step that caused it rather than at the next checkpoint. It landed in
this section because layer 0 was the only per-checkpoint hook that existed when the
section was written — placement by availability, not by fit. Its home is
`training.md` §11, which already carries it, and `search.md` §15.3 lists it among
the search's numerical-health counters.

**Cost: 44.4 s per checkpoint**, measured on 2026-07-30 for 64 games at `n = 100`
plus the six suites of §8.1 at `n = 128`, on the fused encoder and the CUDA search
(`bench/bench_eval.py`, `logs/d1_layer0.log`). The layer still never blocks — a
checkpoint takes minutes — but it is not free, and it is **7.4× the ~6 s this
section predicted**, which was arithmetic and is now a measurement.

⚠️ Two of the three factors in that gap are estimation errors worth keeping. The
prediction assumed ~80 plies a game; a random-init net plays **125**, and layer 0
runs at the *start* of training where games are longest. And it costed the mean
game when the loop pays for the **longest** one in the batch, since every slot
keeps searching until the last game in its wave ends. The third factor is that
`bench_loop.py`'s 62.3k evals/s is a `B = 16384` number and layer 0 runs at 64.

Both attempts to close the gap failed, and the failures are recorded in
`self_play_run`'s docstring rather than quietly dropped: collecting the first `N`
games *to finish* is 1.5× cheaper and biases game length short by 24 plies, which
disqualifies it for a metric that is game length; and shrinking the batch to
overlap the waves came out marginally *worse*, the search losing on 16 boards
exactly what the tail saved. The knobs that work are `games` and `n_sims`.

The suites are not the problem: **2.8 s of the 44.4**, all six of them, search and
raw policy both. Self-play is 41.4 s of it.

---

## 5. Layer 2 — progress, and the curve itself

**The opponent is our own past self.** AlphaZero tracks training progress
self-anchored, not against a ladder of graded external engines (§12), and the
argument is stronger here than it was there: self-play games run 4096-wide on device
with no host round trip, while games against an external process run roughly one per
core with a host round trip per move. That is two to three orders of magnitude in
games per hour, in exchange for a signal that is *noisier*, because an external
ladder adds calibration drift on top of match variance.

A self-anchored league also answers "how do you gradually improve the opponents"
for free: the opponent is the previous checkpoint, so it escalates automatically.

### 5.1 The anchor is uniformly random legal play

**The zero of the scale is a player that picks uniformly at random among the legal
moves.** No network, no tree, no search configuration. It is the origin of the
cost-versus-Elo curve by construction, it costs nothing to keep, it carries no chess
opinion, and any drift in the fitted scale shows up directly as its rating moving off
zero.

⚠️ **Rebased 2026-08-08. The zero used to be the frozen random-init network**, held as
a file (`checkpoints/anchor.pt`) rather than as a seed, precisely so that it could not
move with a torch version or an edit to `BrokefishNet.__init__`. That fixed the wrong
half of the problem, and the reasoning is worth keeping because the same mistake is
easy to make again:

- **It was never a network, it was a network plus a search.** The anchor was played at
  64 simulations with whatever `SearchConfig` the league defaulted to. When §6.1a's
  root terminal sweep landed, the anchor started finding every mate in one — it got
  stronger while its file was untouched, and the origin of the curve moved in silence.
  A zero that moves when the *search* changes cannot anchor a project whose whole
  Track E is changing the search.
- **`checkpoints/` is gitignored**, so the origin of every published number was one
  untracked blob, and `league.py`'s own docstring said "delete that file and the whole
  curve moves".
- **It was saturated anyway.** Measured 2026-08-08 on `runs/t12h-pcr/league-joint-pcr.log`:
  `anchor vs t12h-pcr@1405` and every pairing above it returned `0-0-36`. Its edges
  cost a full 36 games each and carried no information.

Random play has none of those properties. It is defined by the rules of chess and a
uniform draw, so it is the same player on every commit, on every architecture, and at
every future search budget, and there is no file to lose. It is implemented as
`Search.random_move` **outside** the search rather than as a one-simulation search,
because a one-simulation search inherits §6.1a and would find every mate in one — which
is exactly the dependence being removed.

Uniform over the **move set including promotion type**: a pawn reaching the last rank
offers four moves, not one. Weighting it as one would make the anchor quietly
underpromotion-averse, which is an opinion about chess.

⚠️ **This does not make the previously published leagues comparable.** Different
Bradley-Terry fits share a zero but not their units, so `t24h-fp8`'s +905, the muon
league's +542 and `league-joint-pcr`'s +954 remain three scales. What the rebase fixes
is everything from here on, plus one bridge: the old anchor stays in the pool as
`init:n64`, so its rating on the new scale is a *measured* offset rather than an
assertion.

Precedent: lc0's training chart "sets 'the first net' to Elo 0", with the explicit
warning that it is therefore "not comparable, even between different training runs"
(lc0 FAQ, quoted in `2026-07-30-eval-prior-art.md` §4.1). KataGo instead anchors externally, at
ELF ≈ 0. The difference matters only for §6: an internal anchor makes the calibration
step mandatory, an external one folds it into the fit.

### 5.1a A player is a `(network, budget)` pair

§3 already fixes the principle — *"a rating is a function of `(net, n)`, never of `net`
alone"*, and *"the Elo of a checkpoint at `n = 800` and at `n = 100` are two different
numbers, and both are interesting."* Until 2026-08-08 the implementation honoured only
the first half: `n` was a league-wide constant, reported per row, and therefore a
**hidden term in the scale**. That is the reason every report so far carries "comparable
to nothing else".

The budget is now a per-player field. One checkpoint at 16, 64 and 256 simulations is
**three players in one fit**, and the distance between them is the Elo value of a
doubling of search, measured with the network held exactly fixed.

Two things this buys that nothing else in the project measures:

- **The bottom of the scale becomes estimable.** Random play loses 36-0 to anything
  past the first few hundred training steps, so a scale hung directly off it would rest
  on a saturated edge. The untrained network at `n ∈ {1, 4, 16, 64}` supplies the
  intermediate rungs — `init:n1` is essentially the raw policy argmax — and the ladder
  carries the zero up to the start of the training curve.
- **The search-versus-training exchange rate.** Is `ckpt@2000` at 256 sims stronger
  than `ckpt@4000` at 64? The x-axis of the curve is training seconds only, which for
  a bot anyone actually *runs* is half the cost. This is the other half.

⚠️ **Cost is not flat across the pool.** A pairing costs roughly the mean of its two
budgets, so a 256-sim player is four times a 64-sim one and a pairing *count* hides it.
The league reports `pairing_equivalents` — the calendar's size in units of one pairing
at the reference budget — and that is the number to size a run against.

⚠️ **The Bradley-Terry model assumes one latent strength per player.** A `(net, budget)`
pair is a legitimate player so the model is sound, but the pool's strength range roughly
doubles, and a wider range strains a single-parameter fit. Read `dispersion`.

The calendar gets one addition for this: **every pair of players that share a network
and differ only in budget plays**, since SAI's schedule orders by step and would pair
budget variants only by accident of adjacency.

### 5.2 Do not chain

⚠️ **Rating checkpoint `k` only against `k−1` and summing the deltas inflates the
scale.** Two mechanisms compound: pairwise estimation error accumulates along the
chain, and non-transitivity — checkpoint cycles where A beats B beats C beats A —
adds a positive bias at every link. The canonical statement is SAI §3.5: "when this
is limited to matches between newly promoted networks and their predecessor, as in
Leela Zero, the resulting estimates are believed to give an Elo rating inflation, in
particular in combination with gating." lc0's own FAQ concedes the symptom.

⚠️ Note "believed to". Nobody in this literature appears to have *measured* it;
it is the consensus explanation for a real discrepancy. Treated here as a risk worth
designing around, not as an established quantity.

The fix, from KataGo (the fit) and SAI/CloudyGo (the pairings):

- fit **one** Bradley-Terry / BayesElo model over the whole *graph* of games played,
  not a sequence of pairwise deltas. KataGo: "a global Bayesian maximum-likelihood
  Elo based on all game results so far".
- choose pairings by **variance-proportional sampling** — play a pairing with
  frequency proportional to `p(1−p)`, where `p` is the win probability the current
  global fit predicts. This spends games where the fit is uncertain and stops
  burning them on pairings already resolved, and it needs nothing the online fit
  does not already compute. KataGo's rule.
- fallback if the online fit is not ready: SAI's fixed schedule — each new
  checkpoint plays generation offsets ±1, ±2, ±3, ±6, ±8, ±12, plus a slowly-changing
  reference panel. Their graph ran ~13 000 edges over ~1000 nodes, i.e. ~13 pairings
  per checkpoint, which is a usable sizing number.
- the anchor is pinned at 0 in the fit.

The long-range edges are the load-bearing part either way: without them the graph is
a path and the global fit degenerates back into a chain.

### 5.3 What a curve point is

A record, not a number:

```
{checkpoint_id, euros_spent, games_played, elo, ci95, n_sims,
 draw_rate, opponents[], fit_version, timestamp}
```

⚠️ **`euros_spent` is logged by the same writer that logs `elo`.** If the two axes
live in different files they get joined by hand six weeks later, and by then nobody
remembers whether the euro counter included the failed runs.

Cost arithmetic, not a measurement: 1000 league games × 80 plies × `n = 800` =
6.4×10⁷ evals ≈ 17 min at the 62.3k ceiling, with both players on device. Layer 2 is
not a budget problem.

### 5.4 The harness — D2, specified 2026-07-31

`brokefish/eval/match.py`, `elo.py`, `league.py`, `curve.py`; `tests/test_league.py`.
Four modules and one command:

```
uv run --no-project --python .venv/bin/python -m brokefish.eval.league \
    --run t4h-n64 --games 36 --sims 64
```

#### 5.4.1 Openings: self-generated, not a book

Each opening is **8 uniformly random legal plies** from the start position,
rejected if terminal or if fewer than two legal replies remain, deduplicated by
Zobrist hash, and driven by a CPU generator so a seed reproduces the book on any
machine.

⚠️ **A UHO/TCEC book was considered and declined for v1**, and the reason is not
the tabula rasa boundary — §2 explicitly permits standard books in *evaluation*.
It is that the book does not address our failure mode. UHO exists to break draws
between engines that are strong enough to hold a balanced position; the draws here
come from a near-uniform policy shuffling into threefold repetition, which no
opening prevents. Random openings are also naturally *unbalanced*, which produces
**more** decisive games, and §5.4.2 is what makes that unbalance fair. A published
book becomes worth its dependency at §6, when the scale has to be comparable to
CCRL's — not before.

⚠️ **Evaluation is deterministic** (§3: `eps = 0`, `tau_plies = 0`), so the
openings are the **only** source of diversity in the entire league. Two engines
replaying one opening produce one game, every time. A pairing of `G` games
therefore needs `G/2` genuinely distinct openings, which is why the deduplication
above is a hash and not an assumption.

#### 5.4.2 Colours are paired, and the unit is the pair

Every opening is played twice with the colours swapped. Without it, a league on
unbalanced openings measures who drew the favourable side. §9 already calls paired
openings "mandatory anyway" for the variance; here they are also what makes an
unbalanced book *fair*, which is what allows the book to be unbalanced at all.

The property this buys is exact rather than statistical: **a network played
against itself scores exactly 0.5 per pair, whatever the games do.** Both halves
are literally the same game, so one half's point is the other half's zero. That
identity is `tests/test_league.py`'s end-to-end oracle — it catches a sign flip, a
swapped assignment and a double-counted half with one assertion and no sample size
to argue about. `--games` must be even for the same reason.

#### 5.4.3 One network per batch, kept in lockstep

The two networks alternate by ply, so a batch mixing both colour assignments needs
two evaluations per position — either both networks over the whole batch (2× the
encoder) or a gather/scatter with dynamic shapes. Neither is necessary. The two
colour assignments are played as **two separate batches**, and inside one batch
every row is at the same ply, so at every moment every row wants the same network:
one evaluator call, static shapes, 1× the encoder.

⚠️ That lockstep is one edit away from being false, and if it breaks the games stay
legal, the results stay plausible and the curve is quietly meaningless. So it is
**asserted every move**, not documented — `match._swap_evaluator`. It holds because
every opening has the same length and because a finished row is restarted *in
phase*: a row reset to the ordinary start position would be a ply out of step with
the rest of the batch.

⚠️ **A finished game may not be searched again** (`search.md` §6.7, invariant 8),
so a finished row plays a throwaway game whose moves are ignored. The batch
therefore does real work for games that have ended. With games running 40 to 250
plies that tail is this harness's known inefficiency; it is a cost, not a
correctness problem, and refilling dead slots from a queue is the named seam if it
ever matters.

#### 5.4.4 The calendar is fixed

SAI's schedule (§5.2): each pool index plays the ones at offsets +1, +2, +3, +6,
+8, +12, giving up to twelve edges per checkpoint. The **anchor sits at pool index
0**, so the calendar connects it to checkpoints 1, 2, 3, 6, 8 and 12 for free; it
additionally plays every 4th checkpoint and always the last one, because a zero
measured only against the start of the run is reached from the far end by a chain,
which is the failure §5.2 exists to avoid.

| pool | pairings | mean degree | games at `--games 36` |
|---|---|---|---|
| 16 | 68 | 8.5 | 2 448 |
| 24 | 118 | 9.8 | 4 248 |
| 32 | 168 | 10.5 | 6 048 |

Variance-proportional sampling — spend games where `p(1−p)` is largest, KataGo's
rule — is the upgrade and `EloFit.predict` is the function it needs. It is not in
v1 because it requires an online fit and the fixed calendar requires nothing.

#### 5.4.5 The fit

One global Bradley-Terry model over the whole graph, maximised by Newton's method on
the exact log-likelihood after a short minorization-maximization warm start
(`bt-newton/v2`, 2026-09-09; the MM scheme alone crawled along the pool's soft mode
and hit its iteration cap on every real league, at a cost under 0.2 Elo), never a
chain (§5.2). Two departures from the textbook, both load-bearing:

- **the anchor is pinned, not fitted** — held at `gamma = 1`, i.e. Elo 0, so the
  scale means something across runs and "the anchor drifted" is a detectable event;
- **a phantom opponent regularizes** — every player also plays `--prior` drawn
  games against a phantom at Elo 0. ⚠️ **The pull is not small for the pool**
  (measured 2026-09-09, `elo.py`'s docstring): a phantom game against a player 1000
  Elo above zero is saturated and pulls with its full half point, and a hundred of
  them act on the pool's soft mode against the anchor's ~200 informative games.
  `t24h-adamw-int8@10218:n256` reads 1831 at `prior = 1` and 2469 at 0.01, weak
  players move ~170 and the top ~640, so it is a compression by about a quarter on
  a 100-player league, scaling with pool size. Differences between neighbouring
  players are unaffected (−76.7 at every prior). Levels from leagues of different
  sizes are therefore not comparable even on one anchor, and an Elo-per-decade slope
  read inside a big league is compressed. The default stays at 1 because changing it
  changes what every recorded rating means; the decision is Théo's. A player who won every game has an infinite
  maximum-likelihood rating, and one will: the first real checkpoint against the
  random-init anchor is plausibly 100 %. At `prior = 1` against several hundred
  real games it moves a rating by well under an Elo point, and it shrinks *toward*
  zero, so it is conservative.

**The scale is score-based** — expected score `1/(1 + 10^(−Δ/400))`, which is
§9's convention, FIDE's, CCRL's and Ordo's default. ⚠️ It is **not** BayesElo's
draw-model Elo, which factors draws out and is a larger number for the same games.
The two must never be compared, and §6's affine map is between this scale and a
published one, not between two conventions.

**The interval.** The Fisher information of Bradley-Terry is a weighted graph
Laplacian; inverting it gives the covariance, which is the point of a global fit —
a checkpoint borrows precision from every path through the graph, not only from its
own edges. But BT with half-points is misspecified: it models a game's score as
having variance `p(1−p)` where a match at draw fraction `d` has `(1−d)/4`. The raw
interval is therefore too **wide**, by about `1/√(1−d)` — a factor of 2.2 at
`d = 0.8`, which is not a rounding error. The correction is the standard
quasi-likelihood one: estimate the dispersion from the Pearson residuals and scale
the covariance by it. `dispersion` is reported, and lands near `1 − d`; `se_raw`
keeps the uncorrected number so the correction is visible rather than baked in.

#### 5.4.6 Unfinished games are dropped, never adjudicated

A game past `--max-plies` (512) has an unknown result. Calling it a draw is the one
adjudication this harness could make without an engine, and it is exactly the bias
the draw rate exists to detect. The count is reported per pairing instead, so a run
where it is not ≈ 0 is visible rather than absorbed.

#### 5.4.7 Sizing

`se(Elo) ≈ 347·√(1−d)/√N` per edge (§9), and the global fit does better than that
because of §5.4.5. At `--games 36` and `d = 0.8` one edge resolves ±51 Elo; pooled
over ~11 edges a checkpoint lands near ±15.

#### 5.4.8 The boundary, enforced rather than documented

The league writes to `logs/` and nothing under `brokefish/train/` imports
`brokefish.eval` or names a league artefact — asserted by
`tests/test_league.py::TestAntiSelection`. §2's prohibition that will actually get
violated is checkpoint selection, because "keep the checkpoint with the best league
rating" looks like good practice; the structural guarantee is that it cannot be
written without deleting a test. Which checkpoints get *rated* is a cost decision
and is spaced evenly by index, never by score.

---

## 6. Layer 2b — calibration

Layer 2 produces a scale whose zero is our own random-init net. Layer 2b's only job
is to produce the **affine map from that scale to a published one**, by playing a
handful of checkpoints spread across the run against externally rated engines under
fixed nodes per move.

Frequency: roughly every 10× of compute, i.e. once per curve decade, which is a
handful of matches over the whole project.

⚠️ **Two scales exist and must never be silently mixed.** A number on the
self-anchored scale and a number on CCRL's scale are different quantities. Every
plot and table states which one it is on; the mapping is reported with its own
uncertainty, and that uncertainty propagates into any absolute claim.

---

## 7. Layer 1 — the gate

The preregistered claim: a threshold, on a named list, at a named time control, on
named inference hardware, at a named search budget, **fixed before the run rather
than after it** (`roadmap.md`, "Measuring strength").

Two design points:

**A basket, not one opponent.** Three or four rated engines cost the same per game as
one and cannot be defeated by a single anti-computer blind spot. A gate that rests on
one opponent's style is a weaker claim for the same compute.

⚠️ **"Stockfish limited to 3000 Elo" is not a well-defined object.**
`UCI_LimitStrength`/`UCI_Elo` is Stockfish's own scale, not CCRL's, and is poorly
calibrated near the top of its range. Fixed depth, fixed nodes, an older version, and
reduced hardware each produce a different "3000". Choosing the throttling mechanism
*is* choosing the threshold, and it lands squarely inside the ~150 Elo scale dispute
the roadmap already flags. Choosing the throttling mechanism *is* choosing the
threshold.

### 7.1 Submission to CCRL is not available — decided 2026-07-30

⚠️ **CCRL is CPU-only and has declined to make an exception for a GPU engine.** Lc0
is the precedent: the list's position is that every other engine runs on CPU, and no
GPU allowance was granted. TCEC by contrast split into separate CPU and GPU leagues
from Season 17. So the roadmap's first option — "get listed, which removes the
methodology argument entirely" — is closed to us, and only its fallback remains:
**reproduce their conditions and say so.**

That changes what the anchor *is*. It is no longer a rating we are awarded; it is a
rating we **borrow** from a published list by playing an engine that already carries
one. A borrowed rating only transfers if our match reproduces the conditions under
which it was measured, so the protocol question is not cosmetic.

CCRL 40/15's actual conditions: 40 moves in 15 minutes referenced to an Intel
i7-4770k (equivalence derived from a Stockfish benchmark), pondering off, 4-6 man
tablebases permitted, hash 256 or 512 MB identical across the match, and a **generic
12-move book — not UHO or TCEC**. Their time control being CPU-benchmark-referenced
leaves normalisation for a GPU engine undefined, which is now our problem to state
rather than theirs to rule on.

**Resolution: run both protocols and measure the offset.** The ladder runs at our own
protocol (fixed nodes for the opponent, fixed simulations for us, UHO/TCEC books),
which is what every internal number uses. Separately, each anchor engine plays one
match under CCRL conditions, which measures the offset between the two protocols for
that engine. The offset is measured once, carries its own error bar, and converts the
whole ladder. This is strictly better than picking one protocol and costs a handful
of matches.

### 7.2 Tablebases and pondering — decided 2026-07-30

Supersedes `roadmap.md`'s "no tablebases and no pondering on either side or the
same on both", which offers only two options where three exist.

- **We never use tablebases, in any layer.** Not a boundary question — a
  claim-integrity one. An engine that consults a tablebase is not the engine the
  curve is about.
- **The opponent uses whatever its rated configuration uses**, including 4-6 man
  tablebases at CCRL conditions. Demanding otherwise means the opponent is not
  running as rated, which throws away the anchor we went there for. AlphaZero's
  *Science* match did exactly this: 6-man Syzygy for Stockfish, none for AZ.
- **Pondering off on both sides**, which matches CCRL and costs nothing.

Also required, from the roadmap: hardware and time control reported for both engines;
all PGNs published; an assertion that the anchor engine never touched a training
tensor.

---

## 8. Layer 3 — diagnostics

Split in two, because half of it needs no external data at all.

### 8.1 Rule-level, self-generated

Generated by our own move generator, so they import zero chess opinion and are
inside the boundary even during training. **Built in D1**, `brokefish/eval/suites.py`,
200 items each, cached in `data/suites.pt`.

Every suite has one shape: a position, a set of moves that are right **by the
rules**, and a set that are wrong. The answer key is always a spec §4.3 terminal
code, never a material count — *mate beats stalemate* is a rule, *a knight is worth
three pawns* is an opinion — and that is what makes them runnable during training
rather than only after it. It is also why there is no "best move" suite: naming the
best move in a quiet position needs an opinion we are not allowed to have.

| suite | item | the wrong move |
|---|---|---|
| `mate_in_1` | a mate in 1 exists | — |
| `avoid_stalemate` | a mate and a stalemate both exist | stalemating |
| `avoid_fifty` | a mate exists at halfmove clock 99 | letting the clock run out |
| `avoid_threefold` | a mate exists and one reply repeats a twice-seen position | repeating |
| `avoid_insufficient` | a mate exists and one reply strips the board to a dead draw | trading into it |
| `underpromotion` | **every** mating move promotes to a knight, bishop or rook | promoting to a queen |

Each is scored twice: once through the search, once on the raw policy argmax. The
comparison is the diagnostic — a policy that is right where the search is wrong is
a different bug from both being wrong, and the second costs one forward pass.

⚠️ **The original wording of this section did not survive contact with the
generator.** "Stalemate avoidance — winning *material* positions" needed a material
count, so the suite requires a mate instead, which is decidable by the rules alone.
"Underpromotion — the only *non-losing* move promotes to a knight or rook" needed a
search to decide what loses, so the suite requires the underpromotion to be the
only *mate*. And "insufficient material and threefold — `terminal` already computes
both" described an engine property, not a test of the search; both became
avoid-the-draw suites in the same shape as the others.

⚠️ **Where the positions come from is a measurement.** Random legal play from the
start position, the source `tests/boards.py` uses, is nearly useless here: 170 924
such positions contain 85 stalemate-avoidance items, 3 threefold, and **0**
underpromotions. Random play lives in a full-board middlegame and every motif but
plain mate-in-1 lives in a sparse endgame. `brokefish/eval/positions.py` samples
endgames directly — two kings and a handful of pieces, with our own move generator
rejecting whatever is not a legal position — which is 18× the yield, and it is a
sampling bias, never a bias in the answer key.

⚠️ **Two suites are constructed, not sampled**, and this is where to look first if
one of them ever behaves oddly. `avoid_fifty` sets the halfmove clock to 99, which
sampling never reaches with a mate on the board. `avoid_threefold` **plants the
repetition ring**: it writes the hash of one reversible reply's position into the
ring twice, so playing that reply draws. The ring is exactly "positions already
seen" and two copies is what a real repetition looks like, but the path that
produced them is not replayed, so the suite measures the right thing without
proving such a game exists. Nothing else in the repository plants a ring.

These test the search and the value head against the *rules*, which is a different
question from strength, and they answer it in 2.8 s for all six.

### 8.2 Skill-level, external

**Lichess's puzzle database, not chess.com's.** CC0, **6 014 381 puzzles** in the
June 2026 export, updated monthly. Each carries a rating obtained by treating every
solve attempt as a **Glicko-2 game between the player and the puzzle**, plus a
`RatingDeviation` field — which is the point: it gives a calibrated *difficulty axis*,
so the output is a solve-rate-versus-difficulty curve rather than a single opaque
percentage. Filter on low rating deviation; the tail of rarely-attempted puzzles is
noise on that axis.

⚠️ **Puzzle accuracy is not strength.** It is tactics-heavy, single-best-move, and
says nothing about positional understanding or endgame technique. Layer 3 answers
"why is this net bad". It never answers "how strong is this net", and per §2 it never
selects anything.

Built in D1, `brokefish/eval/puzzles.py`. The CSV is not in the repository — 300 MB
of zstd, and not ours — so `load_puzzles` raises with the `curl` line rather than
silently returning nothing.

⚠️ **The schema is the one unverified thing in Track D's code.** The reader, the
setup move, the deviation and rating filters and the rating binning are all tested,
but against a CSV this repository writes itself in the format `puzzles.py` claims.
That proves the reader and not the claim. `2026-07-30-eval-prior-art.md` §8 verified the
licence, the 6 014 381 count and the Glicko-2 deviation field against
database.lichess.org; the column names and the two conventions below came from
memory and stay unconfirmed until the export is on disk.

⚠️ **Lichess's `FEN` column is not the position to solve.** The first move in
`Moves` is the opponent's and is played for you; the solution starts at the second.
Scoring the first move scores the opponent, passes every test, and measures
nothing.

⚠️ **A puzzle is a line and we score its first move only.** Multi-move scoring
needs the opponent's replies played out of `Moves`, which is another layer of
harness. The first-move rate is an upper bound on line accuracy and is reported as
what it is.

---

## 9. How many games

For a match at score `p` per game with draw fraction `d` and roughly equal win rates:

```
Var(score per game) = (1 − d)/4
dElo/dp |p=0.5   = 400/(ln10 · 0.25) ≈ 695

se(Elo) ≈ 347 · √(1 − d) / √N            95 % half-width = 1.96 · se
```

⚠️ **Arithmetic, not measurement.** It also assumes independent games; paired
openings (both colours from each position) reduce the variance further and are
mandatory anyway.

95 % half-width in Elo:

| games | `d` = 0.1 | `d` = 0.4 | `d` = 0.7 |
|---|---|---|---|
| 100 | ±65 | ±53 | ±37 |
| 500 | ±29 | ±24 | ±17 |
| 1000 | ±20 | ±17 | ±12 |
| 2000 | ±14 | ±12 | ±8 |

Two consequences.

**The draw fraction moves this by 1.8×, and it is not constant over the run.** Early
play is decisive (`d ≈ 0.1-0.2`); near the top it is not (AlphaZero-Stockfish was 72
draws in 100). So games-per-curve-point *falls* as the net improves, and a fixed
game count over-measures the early points and under-measures the late ones.

**100 games is a defensible gate size at the top and nowhere else.** At `d = 0.7` it
resolves ±37 Elo, which is why AlphaZero could publish a 100-game match — 28 W / 72 D
/ 0 L, a margin far larger than the interval. It is only sound if the claimed margin
exceeds roughly 40 Elo; a gate that expects to land *near* its threshold needs ~750
games for ±25. AZ's own *Science* rerun went to 1000 games at `d ≈ 0.84`
(155 W / 6 L), i.e. ±9.

For reference, AlphaGo Zero's gating test was **400 games at a >55 % margin**, which
is a ~35 Elo bar; KataGo's is **100 wins out of 200**, i.e. a 0 Elo bar, which rejects
regressions rather than requiring improvement.

**SPRT and fixed-N are not interchangeable.** SPRT is an accept/reject test — it
belongs to checkpoint gating (§11) if gating is adopted at all. A curve point is an
*estimate* and needs fixed-N with a stated interval. `docs/roadmap.md:467` currently
says "SPRT or fixed-N" as if they were alternatives for the same job; they are not.

---

## 10. What the harness has to be

### 10.1 The shape problem

Everything in this repository is throughput-shaped: `B = 4096` in the search
contract, `B = 16384` in `bench/bench_loop.py`. A match against an external engine is
the opposite shape. At `B = 1` the encoder runs one CTA on 24 SMs and 800 sequential
simulations per move — the wrong instrument by roughly the SM count.

So layers 1 and 2b are an **async scheduler**, not a loop: `N` games in flight, `N`
opponent processes, our side batching whichever games are currently waiting on us.
Layer 2 has no such problem — both players are ours, and a league match is just
self-play with two different weight sets.

### 10.2 Three exporters that do not exist

`brokefish/env/torch_impl.py` has `from_fen`, `parse_san` and `from_pgn`. It has **no
`to_fen`, no move→UCI-string, and no PGN writer.** All three are required to speak to
any external engine or to publish PGNs as the roadmap promises, all three are cheap,
and all three are immediately useful for debugging C1. They are the first thing to
build and they have no dependency on any decision in §11.

### 10.3 Ordering

Layer 0 and layer 3.1 depend only on C1. Layer 2 depends on C2 producing checkpoints.
Layers 2b and 1 depend on the exporters and the scheduler. Nothing here needs the
opponent-ladder decision to be made before it starts.

---

## 11. Open decisions

| decision | who/when | note |
|---|---|---|
| the preregistered threshold, list, time control, hardware and search budget | start of D5 | this is the claim; it cannot move afterwards |
| the gate basket: which engines, throttled how | start of D5 | §7 argues for CCRL-listed at CCRL conditions |
| ~~**checkpoint gating in the training loop**~~ | **settled 2026-07-30: no gating** | AZ's choice. Gating makes the x-axis ambiguous — a rejected candidate costs euros and yields no curve point — and SAI names it as an aggravating factor for the §5.2 inflation. C2 therefore does **not** depend on the eval harness. KataGo's 200-game check is still run in layer 0 as a **non-blocking logged diagnostic**, so the write-up can say whether gating would have fired |
| ~~**games per curve point**~~ | **settled 2026-07-30: target a CI width, not a game count** | `d` drifts over the run (§9), so a fixed `N` over-measures early points and under-measures late ones. Fix the ±Elo, let `N` follow; variance-proportional sampling (§5.2) is the same statistic |
| ~~**what the euro counter counts**~~ | **settled 2026-07-30: two numbers** | the curve's x-axis is **training compute only**, which is what makes it comparable to AlphaGateau's 13.7 days × 8 A5000s. ⚠️ **The cost axis is comparable; their Elo axis is not** — it is self-anchored with the pool mean at 1000 (2026-08-09).  Total project cost — eval, failed runs, development — is published as a separate headline figure. Only the first is incomparable; only the second is dishonest for a cost paper |
| ~~**submission to a third-party list**~~ | **closed 2026-07-30: not available** | CCRL is CPU-only (§7.1). The anchor is a *borrowed* rating, not an awarded one |
| **which calibration engines** | start of D4, and the next one to decide | §6 |
| eval `n` for the reported curve | start of C4 | need not equal training's 800 |
| rating fit: BayesElo, Ordo, or our own BT fit | start of D2 | all three are fine; the pairing schedule matters more than the fitter |
| whether the external anchor is a node in the league fit or a separate affine map | start of D2 | folding it in deletes §6's two-scale hazard but pulls §10.2's exporters earlier |

---

## 12. Prior art

Verified against the papers on 2026-07-30. The quotes, links, the full scorecard of
what was checked and what turned out to be wrong, and the four things still
unverified are in **[the eval prior-art pass](../journal/2026-07-30-eval-prior-art.md)**.

The five results that this document rests on:

| | |
|---|---|
| **AZ removed gating** | explicit in its list of differences from AGZ: one network "updated continually", and the 55 % replacement margin omitted |
| **AZ rated at 1 s/move** | BayesElo over a tournament among AZ iterations **and a baseline player** — so not purely self-anchored, and a time control rather than a simulation count (§3) |
| **KataGo fits globally** | "a global Bayesian maximum-likelihood Elo based on all game results so far", with variance-proportional opponent sampling (§5.2) |
| **KataGo keeps gating** | 100 wins out of 200 against the current net. The references disagree with each other on this, which is why §11 has a row for it |
| **SAI §3.5 states the inflation** | chained predecessor-only matches, "in particular in combination with gating"; fix credited to CloudyGo (§5.2) |

⚠️ **One claim from the first draft is withdrawn, not softened**: that lc0 ran gating
early and dropped it. It could not be sourced, and what evidence there is points the
other way. `2026-07-30-eval-prior-art.md` §4.2.

---

## Changelog

**2026-08-08.** §5.1 rebased and §5.1a added. **The zero of the scale is uniformly
random legal play**, not the frozen random-init network — because the old zero was
never a network but a network *plus a search*, and it moved in silence when §6.1a's
root terminal sweep landed. Its file was also gitignored, and by 2026-08-08 it was
saturated: `anchor vs t12h-pcr@1405` and everything above it came back `0-0-36`.
Random play is defined by the rules alone, cannot be deleted, and is implemented
outside the search (`Search.random_move`) precisely so that no future change to
`SearchConfig` can move it. **§5.1a makes the search budget a per-player field**,
which is §3's own principle — *"a rating is a function of `(net, n)`"* — finally
reaching the implementation rather than being restated per row; the ladder
`init:n{1,4,16,64}` bridges random play up to the training curve, and a budget grid on
the checkpoints measures the search-versus-training exchange rate. ⚠️ This makes future
leagues comparable, **not past ones**: different fits share a zero but not their units,
so the three leagues already published stay on their own scales. The old anchor remains
in the pool as `init:n64`, which turns the relation between the scales into a measured
offset. Two bugs found on the way: a pool without the anchor made `fit_elo` invent a
**phantom** player at Elo 0 that had played nothing, and in a joint league the anchor's
cost axis was written as **null** rather than zero because `cost` is keyed by run and
the anchor belongs to none — visible in `logs/curve-joint-pcr.csv`.

**2026-07-31.** §5.4 added: D2's harness, specified and built the same day
(`brokefish/eval/{match,elo,league,curve}.py`, `tests/test_league.py`). Three things
in it depart from what §5 previously assumed, and each is argued in place rather
than quietly adopted: **no opening book** (§5.4.1 — UHO addresses a draw mechanism
that is not ours, and random openings are more decisive, which is the direction we
need); **the fixed SAI calendar rather than variance-proportional pairing** (§5.4.4
— the latter needs an online fit and buys nothing while the calendar costs nothing);
and **an explicit statement that the scale is score-based, not BayesElo's draw-model
Elo** (§5.4.5), which §9's own arithmetic already implied and which §6's affine map
now has to name. §5.4.5 also corrects an interval that would otherwise have been too
wide by `1/√(1−d)` — 2.2× at the draw rates observed — in the conservative
direction, which is the kind that is never noticed. The story is in
`journal/2026-07-31-d2-league.md`.

**draft, 2026-07-30.** First version. Proposes four layers against the three Théo
framed, the addition being layer 0, which exists because layers 1-3 are all
minutes-to-hours and none of them notices a diverged run inside a training step.
Two things move relative to the framing: Stockfish moves out of continuous progress
tracking into §6 and §7, on the throughput asymmetry and the AZ precedent of §12; and
the "SPRT or fixed-N" of `docs/roadmap.md:467` is split, since they answer different
questions. The sample-size arithmetic of §9 supersedes the ~2000-3000 games per curve
point figure given in conversation on 2026-07-30, which assumed no draws and was
therefore pessimistic by up to 1.8×.

**draft, 2026-07-30, verification pass.** §12 was written from memory and has been
checked against the papers; `2026-07-30-eval-prior-art.md` holds the quotes and the scorecard.
Six things changed as a result: AZ is no longer cited as precedent for fixed
simulations (it rated at 1 s/move, §3); §5.2 attributes the inflation claim to SAI
§3.5 rather than KataGo and adopts KataGo's variance-proportional pairing over a
hand-specified schedule; §5.1 gains lc0's first-net-at-zero as precedent; §7 records
that AZ gave Stockfish tablebases and that CCRL's book is not UHO/TCEC; §8.2's puzzle
count goes from ~4M to 6 014 381; and §11's gating row now records that AZ and KataGo
**disagree**, where the first draft implied consensus.

**draft, 2026-07-30, first decisions.** Four rows leave §11: **no gating** (AZ's
choice, and it keeps the x-axis well defined); **target a CI width rather than a game
count**; **two cost numbers**, with training-only on the curve's x-axis; and
**CCRL submission is closed** — the list is CPU-only and refused a GPU exception for
Lc0, so §7 was rewritten around a *borrowed* rating instead of an awarded one, with
§7.1 resolving the protocol question by running both and measuring the offset. §7.2
settles tablebases (never ours, whatever the opponent is rated with) and supersedes
`roadmap.md`'s two-option sentence. Still open and now next: **which calibration
engines**.

**draft, 2026-07-30, D1 built.** §4 and §8 stop being proposals: `brokefish/eval/`
implements layer 0 and layer 3, `tests/test_eval.py` checks the answer keys against
python-chess, and the suites are cached in `data/suites.pt`. Three things in this
document turned out to be wrong and are corrected in place rather than softened.
**§4's ~6 s is 45.0 s measured** (44.4 once the attention scan left, later the same
day) — the estimate assumed 80-ply games where a
random-init net plays 125, and costed the mean game where the loop pays for the
longest in the batch; two attempts to close the gap failed and are recorded.
**§8.1's suite definitions needed a material count and a search** to decide their
answers, so all three were restated in terms of terminal codes. And **random legal
play cannot generate them**: 170 924 positions from the start yield zero
underpromotion items, which is why `positions.py` exists.

**draft, 2026-07-30, one scalar removed.** `max |post-scale attention logit|` leaves
§4's list. Implementing it in D1 is what showed it does not belong: it is an fp16
overflow watch on the fused kernels rather than a measurement of a network, it is
the only thing in `brokefish/eval/` that would have to open up `net.encoder.layers`,
and it can only read the torch oracle's logits in fp32 rather than the fp16
accumulator that actually overflows. `CLAUDE.md` asks for it during *training*, and
`training.md` §11 and `search.md` §15.3 both already carry it. It was in §4 because
layer 0 was the only per-checkpoint hook that existed when §4 was written. Layer 0
now logs three scalars, not four.
