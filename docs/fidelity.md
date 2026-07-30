# Fidelity: where we might be wrong about the things we are copying

Brokefish implements two specifications it did not write. The chess rules are
FIDE's. The search is AlphaZero's. Both are reproduced from documents, and a
reproduction can fail in a way no test in this repository can see: the tests check
that the code does what [`spec.md`](spec.md) and [`mcts.md`](mcts.md) say, and this
document is about whether those two say the right thing.

Every other document here records what was measured. This one records what was
**not**, and what would change if we are wrong. It exists because the two audits
below found things worth writing down, and because the alternative to writing them
down is rediscovering them during a training run.

Read it as a betting sheet, not a bug list. Nothing here is a known defect. Each
entry is something whose truth rests on a reading, a convention or an unmeasured
assumption rather than on a test.

Written 2026-07-30, after C1.

---

## 1. What the evidence actually covers

Two layers, and they are worth separating because their strength is very
different.

**Internal consistency: does the code do what our documents say.** Very strong,
and this is most of what the test suite measures.

| | evidence |
|---|---|
| CUDA search against the torch reference | 1 488 112 selections compared, trees identical field for field after every simulation, up to `n = 800` and `B = 1024` |
| torch reference against `mcts.md` | 15 of 16 mutations killed, five independent test angles |
| torch reference against an independent AGZ transcription | `tests/oracle.py`, built four ways round, exact at 64 and 512 simulations |
| CUDA engine against the torch engine | bit-exact over 10 000 positions and 322 246 transitions |
| torch engine against python-chess | 7 185 positions, full `chess.Move` sets, terminal codes, hashes as a partition |

**External faithfulness: do our documents match FIDE and AlphaZero.** Weaker, and
unevenly so. Perft is the one place where an oracle outside this repository settles
the question completely; everywhere else the oracle is a reading of a paper.

⚠️ **The oracles for the search and for the terminal rules are not independent of
us.** `tests/oracle.py` is a second implementation of *our reading* of AGZ, so it
catches implementation divergence and not misreading. python-chess is genuinely
independent, but it implements the same engine conventions we do, so a place where
every engine departs from FIDE is a place where agreeing with python-chess proves
nothing about agreeing with FIDE. Section 3 is that list.

---

## 2. The search against AlphaZero

Layered, with the confidence I would actually bet at.

| claim | confidence | why it stops there |
|---|---|---|
| the kernels compute what the reference computes | ~99.9 % | measured, above. Residual: `B = 4096` itself has never been compared tree-for-tree, only 1024 |
| the reference implements `mcts.md` | ~99 % | measured. Residual: a mistake shared by reference and oracle |
| `mcts.md` is a correct reading of AGZ | ~85 % | section 2.1 |
| AGZ-as-read is what DeepMind ran | ~70 % | `c_puct` is published nowhere. This has a floor no test can raise |

### 2.1 The four open items, ranked

**(a) Virtual loss may be on the wrong list.** [`mcts.md`](mcts.md) §1.1 files
virtual loss under "not present, and an addition rather than a deviation", beside
Gumbel and WDL. But AGZ's search ran multiple threads with virtual loss, and AZ says
its search is identical to AGZ, so AlphaZero almost certainly had it. A search with
virtual loss builds a *different tree* from the same number of sequential
simulations, deliberately, since virtual loss pushes concurrent descents apart. §3.1's
argument that our batching does not need it is correct and is a different question
from whether AlphaZero had it.

If this is right, virtual loss belongs beside tree reuse (§3.6) and resignation
(§3.7) as a **deviation from AlphaZero**, not an addition to it, and v0's search is
slightly more concentrated than AlphaZero's at the same `n`.

*What would settle it*: ten minutes with the AGZ Methods, "Search Algorithm". Not
done because it was noticed after C1 closed, and it changes a classification rather
than a line of code.

**(b) `sqrt(sum_b N(s,b))` against `sqrt(parent visits)`.** Genuinely open, already
in `mcts.md` §13. AGZ's formula says the first, the released pseudocode does the
second, and they differ by one at every interior node. Under AGZ's form the first
descent below every new node ignores the policy and takes the lowest-index edge,
about 2 % of all selections at `n = 800`, spent systematically on the same move.

*What would settle it*: an A/B once Elo is measurable. One line either way.

**(c) The temperature schedule's unit.** v0 uses `tau = 1` for 30 **plies**. AGZ
says "the first 30 moves"; in Go a move is a ply, in chess "move" usually means a
pair. If AZ meant 30 full moves, our exploration window is half the intended one and
self-play openings are less diverse than AlphaZero's.

*What would settle it*: the same ten minutes, plus a look at what the released
pseudocode counts.

**(d) `c_puct` rests on an unrefereed file, and this one cannot be closed.** AGZ
calls it "a constant determining the level of exploration" and says the search
parameters came from Gaussian process optimisation. It publishes no value. The
logarithmic form `log((N + 19652 + 1) / 19652) + 1.25` exists only in the released
pseudocode, which contradicts AGZ in three other places. `mcts.md` §1.1 already says
this; it is repeated here because it puts a ceiling on every other number in the
table above. No amount of testing raises it.

### 2.1a Raised and closed: the temperature does not enter the training target

Recorded because the papers overload one symbol in a way that invites a wrong reading,
and someone will re-derive it.

AGZ names **one** object, `π_a ∝ N(s,a)^(1/τ)`, and uses it in both of the sentences
that matter: the move is played "by sampling the search probabilities `π_t`", and "the
data for each time-step `t` is stored as `(s_t, π_t, z_t)`". Methods, Self-Play then
sets `τ = 1` for the first 30 moves and `τ → 0` afterwards. Read literally, that would
make the *training target* one-hot on the most-visited move for the rest of every
game — about 60 % of all positions at our game lengths.

**It does not, and three first-hand sources agree.**

- AGZ's own gloss on the schedule gives `τ` one job: "the temperature is set to
  `τ = 1`; this **selects moves** proportionally to their visit count in MCTS, and
  ensures a diverse set of positions are encountered".
- AGZ frames MCTS as a **policy improvement operator** whose output *is* the improved
  distribution, and trains "to maximise the similarity of the policy vector `p_t` to
  the search probabilities `π_t`".
- the released pseudocode separates the two explicitly. `store_search_statistics`
  records `visit_count / sum_visits` — raw counts, **no temperature** — `make_target`
  returns exactly that, and `num_sampling_moves = 30` appears only inside
  `select_action`.

So [`mcts.md`](mcts.md) §6.7's `pi(a) = N(a)/n` as the stored target, with `tau_plies`
affecting only which move gets played, is what AlphaZero did. **No deviation.**
[`train.md`](train.md) §3.1.

⚠️ The same file bears on (c) above: `select_action` tests
`len(game.history) < config.num_sampling_moves`, and `history` holds one entry per
*action*, so the released code counts **30 plies**, not 30 move pairs — which is what
v0 does. That does not settle what AGZ's prose meant, since in Go a move is a ply, but
it removes the reading under which our exploration window is half the intended one.
(c) stays open on the paper and is closed on the code.

### 2.2 One thing that is ours and has no AlphaZero analogue

**fp16 priors.** [`mcts.md`](mcts.md) §4.2 stores `edge_prior` as fp16, where
AlphaZero ran in bfloat16 or fp32 on TPUs. With an untrained policy this is
harmless and measured: the worst kernel-to-reference disagreement over the whole
suite is 0.5 fp16 ULP.

⚠️ With a **trained** policy the tail changes character. fp16's smallest subnormal
is 6e-8, so a prior below that flushes to zero, and an edge with prior exactly zero
has no exploration term at all: `pb_c * P * sqrt(N_v)` is zero forever, and the edge
can only ever be reached by the tie-break at a freshly created node. At 31 edges and
a flat policy that is unreachable. At a 218-move position with a confident policy it
could silently remove real moves from consideration.

This is **unmeasured and has no counter**. The cheap fix is two more fields in
`mcts.md` §15's block: the smallest nonzero prior written, and the count of priors
that underflowed to zero. Both are one `atomicMin` and one `atomicAdd` in
`expand_kernel`.

### 2.3 Deviations that are already documented and are not in doubt

Listed so the sheet above is not mistaken for the whole difference: tree reuse
(§3.6), resignation (§3.7), the game-length cap (§13), the 32×64 action space
against AZ's 8×8×73, one position plus a clock and a repetition feature against AZ's
eight stacked positions (§3.3), and the `E = 64` edge cap (§3.2, §4.3), which
AlphaZero had no analogue of since its children were a dictionary.

---

## 3. The engine against FIDE

Stronger than the search, because perft is a real external oracle and it is
exhaustive over exactly the part of the rules that is hardest to get right.

### 3.1 What perft settles, and it settles it completely

Move generation, legality, castling, en passant, promotion, pins and discovered
checks are validated against published node counts on the six standard positions:

| position | depth | nodes |
|---|---|---|
| startpos | 6 | 119 060 324 |
| kiwipete | 5 | 193 690 690 |
| position 3 | 6 | 11 030 083 |
| position 4 | 5 | 15 833 292 |
| position 5 | 5 | 89 941 194 |
| position 6 | 5 | 164 075 551 |

**598 million nodes, every count matching, 4.8 s on the 4060.** The five
non-startpos positions went a ply deeper on 2026-07-30: they had stopped at depth 4
because the PyTorch engine needs about twenty minutes for kiwipete at depth 5, and
the CUDA engine does the whole table in under five seconds, so the only reason to
stay shallow had already gone. A node count that matches at these depths is not
consistent with any rule being wrong.

Two things this rules out that are worth naming, because they are the classic
silent bugs. The en passant capture that exposes the king along a rank by removing
**two** pawns at once, which defeats naive pin detection and which the brute-force
second-order replay handles by construction. And underpromotion, which position 5
carries inside its own node count rather than only in the differential harness.

### 3.2 Where the engine departs from the official rules, all deliberate

None of these is a defect and every engine makes the same choices. They are here
because "matches python-chess" does not imply "matches FIDE", and because self-play
results depend on them.

**(a) Draws are automatic where FIDE makes them claimable.** FIDE Article 9.2 and
9.3 make threefold repetition and the fifty-move rule *claims*: absent a claim the
game continues, and the automatic thresholds are fivefold (9.6.1) and seventy-five
moves (9.6.2). `terminal` ends the game at three occurrences and at 100 plies
without a pawn move or capture.

The consequence is real and not cosmetic: a position a player would decline to draw
and play on is scored 0 here. Since both sides of every self-play game are the same
network, the effect is symmetric, and it shortens games in a way that is good for
throughput. It also means our fivefold and seventy-five-move rules never fire, which
is consistent rather than missing.

**(b) FIDE's "dead position" is reduced to a material rule.** Article 5.2.2 draws
the game when no series of legal moves can deliver mate, which includes blocked pawn
structures with plenty of material. `insufficient_material` implements the standard
material approximation, matching python-chess: no pawns, rooks or queens, and either
every bishop on a single colour complex with no knights, or exactly one knight and
no bishops.

The approximation errs in the **safe** direction. K+N+N against K is not called a
draw, which matches FIDE, since mate there is possible with cooperation. Blocked
positions are not called dead, so they run to the fifty-move rule instead. Nothing
is declared drawn that FIDE would allow to continue, so no game is cut short
incorrectly; some are longer than they need to be.

**(c) `irreversible` omits python-chess's en passant clause.** Already in
[`env.md`](env.md), repeated here because it is the one deviation that touches the
repetition rule. `Board.is_irreversible` includes `has_legal_en_passant()`; spec §6.2
does not. Dropping it can only lengthen the repetition window, never shorten it, so
it cannot cause a missed repetition, and a position with a legal en passant always
follows a double push, which is a pawn move and therefore already irreversible.

**(d) Promotion is one mask bit and four edges.** Spec §3's action space fixes it;
the caller expands. Also in [`env.md`](env.md).

### 3.3 What is checked by a differential test rather than by an oracle

The terminal codes, the repetition count, `irreversible`, the castling-rights
vector and the hash are checked against python-chess over 7 185 positions, with the
hash checked as a **partition** rather than position by position, which is the only
form that catches two identical positions receiving different keys.

⚠️ **Threefold repetition is the one rule perft cannot reach**, since perft counts
leaves and never asks whether a game is over. It is covered by a constructed forced
line, two knight round trips putting the start position on the board three times,
asserting the code fires on the third occurrence and not the second. That is one
line rather than a population.

⚠️ **The positions in the differential harness come from human games and random
play**, where checkmate and stalemate are rare and threefold is close to
nonexistent. The test asserts the direction that matters, that a finished game is
never missed, over whatever positions it sees. It does not assert that it saw any of
each.

*What would strengthen it*: assert a non-empty count per terminal code, the way
`test_search.py` already does for the search's forced sweep. Cheap and not done.

### 3.4 Two latent silent behaviours, neither reachable today

**The repetition ring truncates silently at 100.** `push_history` writes at
`slot = length.clamp(max=MAX_HISTORY - 1)`, so a 101st push would overwrite the last
entry instead of failing. It is unreachable: the repetition window resets on a
superset of the fifty-move clock's conditions, and the game ends at 100 plies, so
the ring holds at most 100. The bound is exact rather than generous, which is worth
knowing before anything changes the fifty-move threshold.

**The castling three-square test reads the attack map after the move.** Marked
`LOOKS WRONG, IS NOT` in `csrc/movegen.cuh` and validated by perft. The argument:
any attacker whose ray to the crossed squares was screened by the departing king
must pass through the king's origin square, that square is one of the three tested,
and it is empty in the post-move map. The rook lands inside the ray it would
otherwise block, and control-mode attack maps include the blocker's own square. Both
legs check out and 598M perft nodes agree.

### 3.5 What the engine does not attempt

Chess960, draw by agreement, resignation, time forfeit, and the fifty-move and
threefold **claim** mechanics of section 3.2(a). None is needed for self-play and
none is a silent omission.

---

## 4. The training loop against AlphaZero

Written 2026-07-30 alongside [`train.md`](train.md), before any C2 code exists, so
every row is a statement about the *specification* and none of them is yet a
statement about an implementation.

⚠️ **The evidence here is weaker than anywhere else in this file, and structurally so.**
§2's search had an independent AGZ transcription to check against and §3's engine has
perft. A training loop has neither. [`train.md`](train.md) §12 lists eight necessary
conditions that replace an oracle, and passing all eight is consistent with training a
subtly wrong objective competently.

### 4.1 Not in doubt

Closed by the papers' own text rather than by a choice of ours, so they are recorded
here only so the sheet below is not mistaken for the whole difference.

**The value target** is the final game outcome in `{−1, 0, +1}` from the mover's point
of view, quoted verbatim in `train.md` §4. **The loss** is AZ eq. (1) with the
cross-entropy and mean-squared error weighted equally, which AGZ Methods states
explicitly. **No checkpoint gating**: AZ removed AGZ's evaluator, and `evals.md` §11
arrived at the same place independently. **Batch 4096 by gradient accumulation** is
not a deviation at all — with LayerNorm and a mean loss it is arithmetically the same
batch (`train.md` §7.2), which is what lets AZ's learning rate schedule be inherited
rather than invented.

### 4.2 Deviations, ranked by how much they could matter

| | what | confidence it is harmless | why it stops there |
|---|---|---|---|
| a | **`lr = 0.2` under SGD+momentum** | low | AZ's value for a 20-block, ~46M-parameter convolutional resnet at batch 4096. Ours is a 6.38M-parameter pre-norm transformer initialised at `std = 1/sqrt(d)`. Nothing about that tuning transfers, and it will be re-tuned. **The re-tuned value is a deviation from AZ and must be recorded here with the number actually used** |
| b | **the learning rate drop points** | medium | Published in neither paper. AZ p.14 says the rate "was dropped three times" and gives no steps; Figure 1's axis is not annotated. AGZ's Extended Data Table 3 is verified but has two drops at different values (57 % and 86 % of 700k steps). `train.md` §7.3 takes the released pseudocode's `learning_rate_schedule`, read 2026-07-30 as `{0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4}` — three drops at AZ's four published values, i.e. **14 %, 43 %, 71 %**. Still the file §1.1 of `mcts.md` refuses as an authority, but here it *agrees* with the paper and supplies only what the paper omits |
| c | ~~the replay window~~ | **not a deviation** | Settled 2026-07-30: the full AGZ figure of **500,000 games** is kept. ≈13.2 GB, which no machine here holds in RAM, but the buffer is a disk-backed memory mapping and the access pattern needs 1.6 MB/s — so the constraint was never RAM. `train.md` §5.2. Two things stay live: 13.2 GB of free disk is a precondition of a long run, and the sizing assumes an 80-ply mean game length that is an assumption rather than a measurement |
| d | **strict phase alternation** | medium | AZ runs self-play and training concurrently and continuously, 5000 TPUs against 64, so self-play always holds the newest parameters. We have one GPU and the 8 GB budget was measured on the premise that the two activation blocks never coexist. Staleness is therefore bounded by a generation rather than by parameter-server lag. Measurable after the fact: `weight_gen` is stamped into every record |
| e | **the game-length cap at 512 plies** | high | AZ's Domain Knowledge item 5 caps chess games and scores them drawn but does not give the number; 512 is the pseudocode's. Our rules are *stricter* than AZ's plane encoding implies (we implement the fifty-move rule and threefold), so the cap should rarely fire. `train.md` §5.4 makes its firing rate a logged counter rather than an assumption |
| f | **`samples_per_game = 65.2`** | high | Not a published parameter — a ratio *derived* from two published counts (700,000 × 4,096 minibatch positions against 44M training games, AZ Table S3 and p.4). It reproduces AZ's data economics exactly if our mean game length matches theirs, and approximately if it does not, since the per-position reuse is the derived quantity and the per-game one is what we hold fixed |
| g | **the masked policy softmax** | high | `train.md` §3.3. AZ's Representation section says illegal moves are masked and renormalised, and our search's prior is a masked softmax, so training on the same support is the faithful reading. The unmasked alternative is *coherent* rather than wrong — softmax is consistent under restriction — so this is a choice between two defensible readings, not a departure |
| h | **SGD+momentum, if it is ever replaced** | n/a today | AGZ Methods: momentum 0.9, `c = 10⁻⁴`, and AZ defers to it. `train.md` §7.1 specifies it. Our measured 5000 positions/s was AdamW; **if AdamW is kept for throughput or stability that is a real deviation and belongs in this table with the reason** |

⚠️ **(a) and (b) are the two the project will actually depart on**, and they compound:
a re-tuned learning rate makes the inherited drop *points* mean something different
again, since both were AZ's for a schedule we are no longer running. The honest
description of what ships is "AZ's schedule shape, our magnitudes", and that sentence
should appear next to any Elo number that depends on it.

---

## 5. How to use this document

An entry here should either be closed or be re-argued when the thing it depends on
changes. In particular:

* §2.1(a) and (c) are two readings that cost ten minutes each to settle and have not
  been settled. They are the cheapest confidence available anywhere in this file.
* §2.2's fp16 prior underflow becomes measurable the moment a trained network
  exists, and the counter that would measure it does not exist yet. That is the one
  entry that turns from a caveat into a bug silently.
* §4.2(a) is the entry whose *value* is not yet known: the re-tuned learning rate. It
  must be written back here with the number actually used, or the table records an
  intention rather than the run.
* §3.2(a) is fixed by choice and should be revisited only if evaluation against an
  external engine ever needs FIDE claim semantics.
* §3.3's coverage assertions are cheap and would remove the only place where the
  engine's terminal rules could pass vacuously.
