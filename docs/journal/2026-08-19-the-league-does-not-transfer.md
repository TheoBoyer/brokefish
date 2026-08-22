# 2026-08-19 — +237 Elo inside the pool, +12 against AlphaGateau

`t24h-adamw-int8` is the first run on this ledger to move the league decisively, and the
first to be checked against an outside opponent on the same day. The two numbers do not
agree, and the gap between them is the entry.

| | measured against | Elo | z |
|---|---|---|---|
| t24h-adamw-int8 − t12h-int8 | our own joint league, n = 256 | **+237 ± 79** | 3.0 |
| t24h-adamw-int8 − t12h-gumbel | AlphaGateau, 200 games, n = 128 | **+12.5 ± 34** | 0.36 |

The internal number is not an artefact of the Bradley-Terry fit: the two final checkpoints
played each other directly inside that league and the 24 h run won **77-14-17**, a 0.778
score, ≈ +218 Elo on games alone. The gain is real and it is large. It simply does not
appear against a different engine family.

## The two matches

Same protocol both times — seed 0, 100 openings from 8 plies of uniform random play, each
played both colours, n = 128 for both engines, 300-ply cap, `scripts/arbiter.py` as
referee with `python-chess` as the only authority on legality and result.

| | W-D-L | score | Elo | mean plies | draw rate |
|---|---|---|---|---|---|
| `t12h-gumbel-004009` (12 h), 2026-08-13 | 20-120-60 | 0.4000 | −70.4 | 92.5 | 60 % |
| `t24h-adamw-int8` (24 h), 2026-08-19 | 13-141-46 | 0.4175 | −57.9 | 96.9 | **70.5 %** |

Fewer wins, fewer losses, more draws: at this level the extra 12 hours bought *stability*,
not strength. The score difference is 0.0175 with SE ≈ 0.049.

⚠️ **The draw rates are the tell.** Against AlphaGateau our nets draw 60-70 % of games.
Against *each other*, in the same league, the 12 h and 24 h checkpoints drew **14 of 108**.
Our pool is decisive about differences that AlphaGateau is indifferent to.

## Where the Elo actually goes: we hang pieces at twice their rate

Counting *uncompensated* material losses only — our move, their reply, our recapture, and
still at least 3 pawns down. No engine is consulted; this is piece values and the rules.

| | score vs AG | **our blunders / 1000 plies** | AG's / 1000 plies | our games with ≥1 |
|---|---|---|---|---|
| t12h-gumbel (12 h) | 0.4000 | **13.30** | 6.17 | 129/200 (65 %) |
| t24h-adamw-int8 (24 h) | 0.4175 | **12.49** | 5.83 | 117/200 (59 %) |

**Doubling the training budget moved the blunder rate 6 %.** The median ply of the first
blunder went 36 → 32, i.e. nowhere. And in both matches we hang material at **2.1× their
rate**. That is where the −60 Elo lives, and it is the one quantity twelve extra hours of
self-play did not touch.

The losses are not positional grinds. Twenty of twenty-one sampled losses end between 5
and 28 pawns down; the median game is materially decided **38 % of the way through**, then
played out. Even **54 % of the draws** contain an uncompensated piece loss — the draw wall
is AlphaGateau failing to convert, not us holding.

## The search cap is symmetric, so it is not the explanation

The obvious suspect was `gumbel_m = 16`: search only ever considers the top 16 moves by
prior, so a refutation outside that 16 is unreachable at any `n`, and a better prior would
improve *ranking within the 16* — exactly the kind of gain that shows up against ourselves
and not against a tactical opponent.

⚠️ **That story is dead, and Théo killed it in one sentence: AlphaGateau has the same
cap.** `mctx.gumbel_muzero_policy` defaults `max_num_considered_actions = 16`
(`mctx/_src/policies.py:136`) and neither `serve_ag.py` nor their `mcts.py` overrides it.
Both engines search 128 simulations over their own top-16. Ours blunders twice as often.
A symmetric constraint cannot produce an asymmetric failure rate.

So the difference is **in the network**, not in the search wrapped around it. What remains
of the search angle is smaller and separate: PUCT over all root moves measured **+42 Elo
over Gumbel m=16 at n = 128 on identical weights**, which is worth having but is a third of
the gap and applies to us and to them alike.

## What that leaves, ranked by how testable it is

The comparison is not size or data. AlphaGateau's released checkpoint is **5 layers, inner
128, 65.5 M frames**; ours is 8 layers, d = 256, **6.38 M parameters, ~52 M records**. A
smaller net on comparable data blunders half as much.

1. **Representation.** Ours is a 32-piece-token transformer; theirs is a graph network over
   squares. "Is this square defended" is a statement about squares, and a piece-token model
   has to reconstruct it through attention. Hanging a piece is precisely a failure to
   evaluate square control. This is the hypothesis I would test first and it is the most
   expensive to test.
2. **The value head.** Measured the same week: our value head sits 0.014-0.062 below what a
   refitted linear readout of its own trunk achieves, and `--value-weight 2.0` made it
   catastrophically worse (−231 Elo). AlphaGateau weights value at **0.5**, half the policy,
   in the code that produced this checkpoint.
3. **Self-play diversity.** Our openings come from temperature and Dirichlet noise on our
   own policy; a distribution that never presents a tactic cannot teach it. The blunder
   rate against *ourselves* is unmeasured and should not be assumed equal to 12.5/1000.

## What this does to the ledger

⚠️ **Every Elo in `docs/ledger/` is self-anchored, and this entry is the first direct
evidence of how badly that can diverge from external strength.** A 3-sigma, +237 Elo
internal result corresponds to +12 ± 34 externally. The league is not wrong — it measures
what it measures, and the 77-14-17 head to head is a fact — but "Elo" in this repository
means "Elo against our own lineage" and the two are not interchangeable. Gate 2's level
half was already restated (2026-08-09) to require a direct head to head for exactly this
reason; this is the measurement that shows why.

⚠️ It also puts a number on the cost of a cheap proxy. The policy puzzle probe correlates
+0.08 with ΔElo at n = 256 across five paired leagues, and the league itself now correlates
poorly with the only external anchor we have. Two proxies deep, the signal is gone.

## Also settled today

`t24h-adamw-int8` trained the full 24.00 h, 10 303 steps, 430 122 games, final puzzle
**0.4602 pass@1 / 0.1638 solve** — +0.055/+0.046 over the best 12 h AdamW arm, level with
the best 12 h Muon arm, and −0.045/−0.057 against AlphaGateau. The Muon track is paused,
on the strength of `t24h-fp8` vs `t24h-muon` at matched `--adam-wd 0.01` (AdamW by
+84/+75/+152). ⚠️ That decision now looks weaker than it did this morning: if internal Elo
does not transfer, then the league result that justified pausing Muon is denominated in the
same suspect unit.

⚠️ The league fit reports `converged: False` at 10 000 iterations with **dispersion 1.148**
against 0.33-0.74 for every other league in the project. The direct 77-14-17 pairing
supports the sign and rough size of +237 independently, but the exact magnitude carries
more uncertainty than the ±79 states.
