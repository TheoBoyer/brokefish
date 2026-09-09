# The training loop

**Status: draft, 2026-07-30.** Normative for C2: the replay buffer, the loss, the
optimiser, the alternation between self-play and gradient phases, checkpointing and
the euro counter. It does not modify [`search.md`](search.md), which owns the search, or
[`spec.md`](spec.md), which owns the engine and network contracts. Where it needs a
quantity those documents leave open, it closes it here and says on whose authority.

---

## 1. What this is, and which document is the authority

C2 is the outer loop: play games with the current weights, store what the search
produced, sample from the store, take gradient steps, publish new weights, repeat.
Everything inside one move is already specified and built — `search.md` §6 for the
search, `spec.md` §7 for the network.

⚠️ **The authority for training is the AlphaZero paper (arXiv:1712.01815) for what it
states, and AlphaGo Zero (Nature 550:354-359) for everything AZ defers to it on.** AZ
says explicitly, p.4: *"Unless otherwise specified, the training and search algorithm
and parameters are identical to AlphaGo Zero."* So AZ owns the loss form, the value
target, the batch size, the learning rate values and the number of steps; AGZ owns the
optimiser, the momentum, the L2 constant, the loss weighting and the replay window.

The same warning `search.md` §1.1 carries applies here: **the released `pseudocode.py`
is not an authority.** It is used below exactly once, for the learning rate drop
points, which neither paper publishes, and it is labelled where it is used.

⚠️ **C2 is the first phase with no oracle.** Perft settled the engine, `nn/model.py`
settled the encoder, an independently written AGZ search settled C1. There is no
external artifact that says whether a training loop is correct, and its failure mode
is not a crash — it is a curve that is merely worse than it should have been, which is
indistinguishable from a project whose premise was wrong. §12 is what replaces the
oracle, and it is the part of this document to argue with.

---

## 2. Where v0 differs from AlphaZero, and why

Everything else in this document reproduces AZ. These are the departures, all forced,
all with a named cause. Each one also belongs in [the fidelity audit](../journal/2026-07-30-fidelity.md).

| | AZ | v0 | cause |
|---|---|---|---|
| **phase structure** | self-play and training run concurrently and continuously, 5000 TPUs against 64 | strict alternation on one device | one GPU. §8 |
| **weight publication** | continuous; self-play always holds the newest parameters | once per generation | follows from the above |
| **games** | 44M | 10M ± a factor of five | budget. `CLAUDE.md` |
| **network** | 20-block resnet on 8×8 planes with a T=8 history | 6.38M-param piece-token transformer, no history | `spec.md` §7 |
| **game-length cap** | "maximum number of steps ... determined by typical game length", scored drawn | **512 plies, scored drawn** (§5.4). AZ's Domain Knowledge item 5; the number is the pseudocode's | v0 had no cap at all, which a replay buffer cannot tolerate |
| **batch composition** | one 4096 minibatch | 4 accumulated micro-batches of 1024 | 8 GB. §7.2, and it is **exact**, not an approximation |

Not a departure, and worth stating because it looks like one: **there is no
checkpoint gating.** AZ removed AGZ's evaluator and updates a single network
continually. `evaluation.md` §11 settled this independently on the grounds that gating
makes the curve's x-axis ill-defined. The two agree.

---

## 3. The loss

AZ eq. (1), unchanged:

$$l = (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c\lVert\theta\rVert^2$$

AGZ Methods, Optimisation: *"The cross-entropy and mean-squared error losses are
weighted equally (this is reasonable because rewards are unit scaled)."* Our value
head is `tanh` into `[-1, 1]` (`spec.md` §7.4) and the target is in `{-1, 0, +1}`, so
the same justification holds and the weighting stays 1:1. **Do not add a value-loss
weight**; AGZ used 0.01 only in its *supervised* comparison, on a different data
distribution, and importing it here would be reproducing the wrong experiment.

`c = 10⁻⁴`, AGZ Methods. §7.1 says where it is applied.

### 3.0a The value term has a second form, off by default

⚠️ Added 2026-08-21, as an **option**: `--value-classes 3` replaces the squared error
with a **3-class cross-entropy** on a win/draw/loss head (`spec.md` §7.4). Nothing
above changes at the default `--value-classes 1`, which is the arithmetic every Elo
number on the ledger was produced by.

$$l_v = -\log \hat{p}(\text{class}(z)), \qquad \text{class}(z) = z + 1$$

⚠️ **With `--value-head pooled` the target is `z` in *White's* frame**, i.e.
`z · sign(control)`, because the pooled head predicts White/draw/Black rather than the
mover's result (`spec.md` §7.4). Getting that backwards does not crash: it trains a
head that is exactly right where White is to move and exactly wrong everywhere else,
which reads as a head that learns nothing rather than as a bug. `az_loss` takes the
frame from the network's own `value_absolute`, and `tests/test_train.py` pins it
against a hand transcription in double.

Why it is worth a run. The value head is the one part of this network that has not
been observed to train: across four 12-hour runs the held-out value-puzzle probe moved
`0.3530 -> 0.3480` (×0.99) while the policy probe moved ×4.8, and that was independent
of the schedule, the sample reuse and the search budget. Three mechanisms make a
classifier a candidate rather than a cosmetic change:

1. **A `tanh` saturates and a softmax does not saturate the same way.** The documented
   death of the value head at `lr = 0.2` was `value_saturated_frac -> 1.0` with
   `grad_value_head -> 0.000` (`journal/2026-07-31-value-collapse.md`); the squared
   error through a `tanh` has a gradient that vanishes as `|x|` grows *whether or not
   the prediction is right*, and a cross-entropy's does not.
2. **A draw is a class, not a coincidence at zero.** Under the scalar head, "draw" is
   whatever pre-image of 0 the head finds; under the classifier it is a target with its
   own mass. Our leagues are draw-heavy (2 % to 100 % depending on the pairing).
3. **It is what the field does.** KataGo and Leela both train win/draw/loss, and
   KataGo's `c_value = 1.5` exists precisely because the classifier's scale is not the
   squared error's.

⚠️ **Against it**, stated plainly: nothing above is a measurement, this is the same
class of hypothesis as `--value-weight 2.0` (which cost a 12-hour run and came back
−231 Elo), and the head is *cheap* to change but not free to evaluate — a run is 12
hours plus a league.

⚠️ **`--value-weight` is not calibrated for it.** The squared error against
`z in {-1, 0, 1}` starts near 1.0; the cross-entropy starts at `ln 3 = 1.0986` and has
a floor of 0 rather than of the target's own entropy. `gradient/value` is therefore
**not comparable across the two branches**, and neither is `value_saturated_frac`,
which keeps its name and comes to mean "a confident classifier" instead of "a
saturated tanh". The 1:1 weighting is inherited, not justified, on this branch.

### 3.1 What the target is

`π` is the **root visit distribution**, `π(a) = N(a) / n` over the root's edges —
`search.md` §6.7, already normalised and already stored in the record as
`(policy_move, policy_prob)`.

**Not the move that was played.** AZ p.3 trains "to maximise the similarity of the
policy vector `p_t` to the search probabilities `π_t`", and the played move is a
*sample from* `π` below `tau_plies` — a one-hot target would be a single noisy draw
from a distribution we already hold exactly. The point of the loop is that MCTS is a
policy improvement operator whose output is the improved distribution; the sample is
what makes the game diverse, not what carries the signal.

⚠️ **`τ` does not enter the target**, only the move choice — despite AGZ naming a
single `π_a ∝ N^(1/τ)` and using it in both roles, which invites the opposite reading.
Three first-hand sources say otherwise: AGZ's own gloss ("this **selects moves**
proportionally to their visit count"), the policy-improvement framing, and the
released pseudocode, where `store_search_statistics` stores
`visit_count / sum_visits` with **no temperature**, `make_target` returns that, and
`num_sampling_moves` appears only in `select_action`. `2026-07-30-fidelity.md` §2.1a has the
quotes; it is **not** a deviation, and `search.md` §6.7 already does the right thing.

### 3.2 Cross-entropy or KL — the same thing, and log the other one

The paper's term is `−πᵀ log p`, which is the cross-entropy `H(π, p)`. The
KL divergence is `KL(π‖p) = H(π, p) − H(π)`, and `H(π)` is the entropy of a **stored
target**, constant with respect to `θ`. **The two have identical gradients**; they
differ by a per-sample constant. So "cross-entropy" and "KL against the visit
distribution" describe the same optimisation and the paper's phrasing is the
cross-entropy one.

The difference is only visible when you *read* the number. Cross-entropy has a floor
of `H(π) > 0` that varies batch to batch, so a perfectly fitted policy shows a loss
that is neither zero nor stable. KL goes to zero at a perfect fit.

**Train on the cross-entropy (the paper's form); additionally log the KL**, which is
`CE − H(π)` and therefore free once `H(π)` is computed. §12's overfit-one-batch check
in particular is much easier to read against a quantity whose target is zero.

### 3.3 What `p` is a distribution over

The network emits 2048 raw policy logits plus 4 promotion logits and applies no mask
(`spec.md` §7.4, deliberately). So the loss has to say what the softmax denominator
runs over, and there are two coherent answers.

**Decision: over the legal moves, not over all 2048.** Three reasons, in order of
weight:

- AZ Methods, Representation: *"Illegal moves are masked out by setting their
  probabilities to zero, and re-normalising the probabilities for remaining moves."*
- it is the distribution the **search** consumes — `expand` builds its prior as a
  softmax over the surviving edges (`search.md` §6.4) — so training and search calibrate
  the same object;
- the ~2000 permanently-illegal logits then receive no gradient at all, instead of
  being pushed towards `−∞` forever for no benefit.

⚠️ **The alternative is not wrong, and it is worth knowing why.** Softmax is
consistent under restriction: a softmax over a subset is exactly the conditional of
the full softmax given that subset. So a network trained unmasked and searched masked
is coherent — the search's masked prior is the correct conditional of what was
trained. That variant needs no legality mask at training time at all. It is in §14.

### 3.4 What that costs, and what it does *not* require

Nothing, as of §3.5: the record carries its own support. For one record, decode each
stored `policy_move` label to its logit, take `log_softmax` over the stored edges, and
dot with `policy_prob`.

⚠️ The first draft of this section costed a `movegen` per training sample — 3.8M
positions/s in-process against ~5000 positions/s of training, **0.13 %** of the phase —
and rejected storing the edge labels because it "adds 128 B to 330 B". That arithmetic
was wrong: §10's arrays are `E` wide and zero-padded whether or not the entries are
used, so storing the unvisited edges costs **nothing**. The cheap option was available
the whole time and the recompute was never necessary.

⚠️ **The unvisited legal moves matter.** They contribute nothing to `−Σ π log p`
directly and everything through the denominator. Restricting the softmax to the
`policy_len` moves that were actually visited optimises a different objective, and
every loss value it produces looks reasonable.

**One thing must match the search exactly, and it is not the enumeration.**

**The label decode.** A `u16` label is `(move, promo)` per `spec.md` §3; `move`
indexes the `[32, 64]` policy head as `(slot, square)` and `promo` selects one of the
4 promotion logits, with a promotion edge's logit being
`policy_logits[slot, square] + log_softmax(promo)[k]`. This map is shared with
`expand` and **a mismatch permutes the target silently** — the loss still falls,
because the network happily learns a consistently shuffled labelling. §12 check 3 is
for this.

⚠️ Corrected from the first draft, which required the training path to rebuild the
search's edge array "in the same canonical order". It does not. The denominator is a
sum over a *set*, and indexing logits by label is order-free.

### 3.5 The record carries the support, so the training path recomputes nothing

Revised 2026-07-31, twice, and the second revision deleted the problem the first one
was solving. The question is what the softmax denominator runs over when a position
has more than `E` legal moves and the search kept only `E` of them.

**The first draft said: recompute it.** Rebuild the legality mask with `movegen` and
re-derive the top-`E`. **That cannot be done correctly** — the search truncated with the
*generating* weights and the training path holds the *current* ones — and it fails in
practice, not merely in principle: within eight generations of the first smoke run, on

```
r6r/1ppq1kpp/5n2/2n1pbP1/7P/1P1P1N2/PpP1PK2/2R2Q1R b - - 1 1
```

the search kept `b2b1=N` and a recomputed top-`E` dropped it, which turns a stored
target into an infinite loss.

**The right question is why the training path is recomputing anything at all.** The
search knew its support exactly. §10's policy arrays are already `E` wide and
zero-padded, so storing **every edge** rather than only the visited ones — `π = 0`
where the search never went — costs **zero extra bytes**. `policy_len` becomes the
root's edge count instead of its visit count and the record is complete.

**Decision: the denominator is the stored edge set.** The consequences are all in one
direction:

- the truncation question disappears. The stored edges *are* the support, exactly, by
  construction, whichever weights chose them;
- the loss touches **no engine**: no `movegen`, no legality mask, no `bitset_to_bool`.
  `az_loss` does not take an `env` argument;
- it is cheaper by two orders of magnitude in the tensor that dominates it — an
  `[N, 96]` gather where the masked form built two `[N, 8192]` fp32 tensors;
- the canonical enumeration order of `search.md` §6.4 never enters the training path.
  No sort, no `topk`, and no host synchronisation to discover whether a sort was
  needed.

⚠️ **What is given up, and where it went.** The recomputed mask was also an
independent cross-check on the record. That does not vanish, it moves: `audit_labels`
verifies every stored edge is legal in the recomputed position, and §12's `audit_every`
runs it periodically — at 0.13 % of a step it is affordable every hundredth one rather
than never. ⚠️ It catches an *illegal* label and not a *permuted* one; only the
comparison against a live search settles the ordering, and that is check 3.

⚠️ **The residual difference from AZ** is that on a position with more than 64 legal
moves the denominator is the search's 64 rather than every legal move, which is what AZ
Methods renormalises over. Softmax is consistent under restriction, so training exactly
the distribution the search consumes is coherent — but it is ours and not the paper's,
and it is in [the fidelity audit](../journal/2026-07-30-fidelity.md) §4.2(i).

⚠️ **One thing the label alone cannot tell you.** `promo == 0` does not mean "not a
promotion": spec §3 numbers the types `0:N 1:B 2:R 3:Q` and a quiet move carries field
0 too. Whether the `log_softmax(promo)` term belongs in an edge's logit is decided from
the **piece word and the target rank**, which the record's board carries — never from
the label. Getting it backwards deletes the entire underpromotion motif.

---

## 4. The value target

Closed by the paper, not open. AZ p.3:

> At the end of the game, the terminal position `s_T` is scored according to the rules
> of the game to compute the game outcome `z`: −1 for a loss, 0 for a draw, and +1 for
> a win.

So: **the final game result, one number per game, written into every record of that
game, from the point of view of the side to move in that record's position.** No
bootstrapping, no mixing with the search's root value, no discounting by ply.

⚠️ This is what makes §3.0a's classifier legal at all: `z` is *exactly* `{-1, 0, +1}`,
so `z + 1` is a class index and not a rounding. `az_loss` asserts it rather than
assuming it — a bootstrapped or averaged target would need a soft-label loss, and
rounding one into a class would train a confident wrong answer.

This closes the "value target" row of `spec.md` §11. The row was open because KataGo
reports an Elo gain from mixing in the search value; that remains a legitimate later
experiment and it is not what AZ did.

⚠️ **Perspective, written down because getting it backwards is a working system that
plays to lose.** `MoveRecord.result` is `int8` *from the new mover's point of view*
after the move was played (`search/torch_impl.py`). The record's own position has the
*previous* mover to move. The buffer therefore stores, for a game that ended at ply
`T` with result `r` from the point of view of the player to move at ply `T`:

```
z(t) = r  if (T - t) is even, else -r
```

with `z = 0` for every draw regardless of parity. This is the same parity argument as
`search.md` §6.5's backup flip and fails the same way if inverted.

⚠️ **That form assumes the recorded plies are consecutive, and under §15 they are
not.** Playout cap randomisation records only the full-search turns, so `t` indexes a
*sparse subset* of the game and the distance from the end of the pending block is no
longer the distance from the end of the game. The rule is therefore stated on the
quantity it was always about — which side is to move — and reads the control word, whose
sign is the side to move (`env.md` §2.2):

```
z = -r * sign(control_record) * sign(control_T)
```

`control_T` is the control word of the position the last move was played from. The two
forms are **identical on a dense game** — the last record *is* ply `T`, so its own sign
gives `-r` and every earlier one alternates — and only the second survives a sparse one.
It is also strictly more robust in the case §9 already relied on: a game whose earlier
records were lost to a resume still gets correct values, because nothing counts anything.
`tests/test_train.py` checks both forms agree densely and that the sparse one is exact.

**The ablation seam stays open at zero cost.** `search.md` §10 already reserves
`weight_gen` plus one `f32` for a bootstrapped target. C2 writes the search's root
value into that field and trains on nothing but `z`. Two bytes of a 330 B record buys
the ability to run the KataGo mix later without regenerating a corpus.

---

## 5. The replay buffer

### 5.1 Schema

The record is `search.md` §10, unchanged, produced by
`search.select_and_advance()` as a `MoveRecord`, one row per game per move:

```
board       [32] u16    the position searched
control     i16
rep         u8          min(rep - 1, 2)
policy      [96] (u16 move, f16 prob)   the root's WHOLE edge set, pi = 0 where unvisited
policy_len  u8          the root's edge count, NOT its visit count (§3.5)
value       f32         §4, written when the game ends
weight_gen  u16         the generation that produced the search
root_value  f32         reserved, written, trained on by nothing (§4)
```

**462 B per position** — `K_POLICY = 96` since 2026-08-02, when `E` moved 64 -> 96
(`search.md` §4.3). It was 334 B at `K_POLICY = 64`.

⚠️ **A `.dat` written at the old width cannot be read at the new one**, and it would
not fail loudly on its own: reinterpreting the bytes yields records whose all-zero
board word decodes as a live white pawn on a1 rather than as an error. `buffer.py`
therefore checks `RECORD.itemsize` against the file and refuses the mismatch.

✅ **`root_value` landed 2026-07-31**, and it was C2's one edit into a module it does
not own: `MoveRecord` in `search/torch_impl.py`, computed in `select_and_advance`, and
added to the field-by-field comparison in `tests/test_search_cuda.py`.
`search/cuda_impl.py` needed no change — it inherits `select_and_advance` and its
kernels already write `edge_N` and `edge_Q`.

⚠️ It is the **search's** value of the root, `Σ π(a) Q(a)` over the root edges, not
the raw network evaluation the root started from. The improved estimate is what a
KataGo-style mix would bootstrap against; the raw one is already recoverable from the
network. Stored in `[-1, 1]` to match `z`, while the tree works in `[0, 1]` (§3.5 of
`search.md`).

### 5.2 Window

**The most recent 500,000 games**, sampled uniformly at random over all positions in
them. AGZ Methods, Optimisation, inherited by AZ. Decided 2026-07-30 to take the
literal AZ figure rather than shrink it: `spec.md` §8 argues that a bounded window is
the *only* thing controlling both the policy staleness and the value bias, so a
reduction would trade one measured quantity for another.

500,000 games × ~80 plies × 462 B ≈ **18.5 GB**.

⚠️ **The buffer is a memory-mapped file on disk, not a RAM allocation, and that is
what makes the AZ window affordable.** 15.7 GB of RAM cannot hold it; the access
pattern does not need RAM to. At ~1.2 optimiser steps/s, sampling 4096 records is
**1.6 MB/s of random reads**, which the OS page cache absorbs almost entirely and
NVMe would serve regardless. It never occupies VRAM either — micro-batches are staged
to device from the mapping.

⚠️ **Disk is the resource to check before a long run, not RAM.** 18.5 GB has to be
free, and the loop should refuse to start rather than discover it at generation 40.
`~80 plies` is an assumption, not a measurement — AZ does not publish its mean game
length and ours is unmeasured — so **size the file from the measured mean once the
first generation exists**, and log the achieved bytes-per-game.

Eviction is by game, oldest first, once the game count exceeds the window.

### 5.3 Pending records

A record is not sampleable until its `value` exists, which is when its game ends. The
buffer therefore holds two populations, and only the completed one is drawn from.

Each game in flight owns an index of its own pending records; when the game
terminates, `z` is written to all of them by the parity rule of §4 and they become
sampleable together. `search.reset_finished()` already restarts finished games in
place, so the hand-off point is unambiguous: the same step that resets a row closes
out its pending records.

⚠️ **A game that never terminates leaks its whole pending list.** This is why §5.4
exists.

### 5.4 The game-length cap

**512 plies, scored as a draw, counted as a completed game.** This closes the row
`search.md` §13 leaves open ("absent in v0").

AZ's Domain Knowledge item 5 terminates over-long chess games and assigns them a drawn
outcome; the paper does not give the number and the pseudocode uses 512. Our rules are
stricter than AZ's — we implement the fifty-move rule and threefold, which AZ's plane
encoding also carries — so the cap should fire rarely and its firing **rate is a
logged counter**, not an assumption. If it fires often, something upstream is wrong.

### 5.5 Startup

No gradient step runs until the buffer holds at least one full batch of *sampleable*
records (4096). Before that the loop is pure self-play. AGZ has the same property
implicitly, since its window is "the most recent 500,000 games" however few exist.

---

## 6. Cadence: how much training per game

AZ does not publish a reuse factor. It publishes the two numbers that determine one
(Table S3 and p.4, chess): **700,000 mini-batches of 4,096** against **44 million
training games**.

$$\frac{700{,}000 \times 4{,}096}{44 \times 10^{6}} = 65.2 \text{ positions sampled per game generated}$$

⚠️ **Revised 2026-07-31: the cadence rides positions, not games.** The paragraph below
was the original reading and it is wrong in a way that is invisible until game length
moves.

`65.2 per game` is only a reuse factor **at a fixed game length**. The conversion this
document used — "at a mean game length near 80 plies it corresponds to a per-position
reuse just under 1" — is a division: `reuse = 65.2 / mean_plies`. So a per-*game* rule
lets the quantity that actually acts on training float with something the training
does not control.

Measured on run `t4h-n64`: the mean game grew **101 → 144 plies** over 1 292 steps, and
the per-position reuse fell **0.641 → 0.374** — a 42 % drop in how much each generated
example is trained on, *inside one run*, while the data rate was constant at 10 240
records per generation throughout. Records are produced at `moves_per_phase ×
batch_games` and do not depend on game length; only the number of games *closing* per
phase does, so the step budget collapsed while the data did not.

⚠️ **The 80-ply figure is ours, not AlphaZero's.** Checked against both paper texts on
2026-07-31: AZ publishes 44 million training games and 700 000 × 4096 samples, and
states that games past a maximum length were terminated as draws, but **no game
length anywhere**. So AZ's true per-position reuse is unknown and 0.815 is our estimate
of it.

**The constant to reproduce is therefore `samples_per_position = 65.2 / 80 = 0.815`**,
and `steps_owed` takes the records a phase appended rather than the games it closed. A
run whose reuse halves partway through cannot be compared with itself, let alone with
AZ. `samples_per_game = 65.2` stays in the config as the published ratio it is derived
from, and `--samples-per-position 65.2/mean_plies` reproduces the old behaviour for an
A/B.

At `moves_per_phase = 10` and 1024 games in flight this is 10 240 records and **2.04
steps per generation, constant** — against 0.94 and falling under the old rule.

It closes the "reuse factor R" row of `spec.md` §11.

⚠️ **It is 65 per game, not 1 per game.** The one-position-per-game rule is real and
belongs to **AlphaGo (Nature 2016)**, whose *value network* was trained on 30 million
positions each drawn from a separate game, because training on full games overfitted
(train MSE 0.19 against test 0.37). AGZ and AZ dropped that rule and decorrelate with
the 500,000-game window instead. Importing it here would be reproducing AlphaGo 2016's
workaround inside an AlphaZero loop, and would discard 98 % of generated data.

**Implementation.** After each self-play phase, with `G` games completed since the last
gradient phase, run

```
steps = floor((carry + 0.815 * R) / 4096)
carry = (carry + 0.815 * R) - 4096 * steps
```

where `R` is the number of records the phase appended. The carry makes the long-run
ratio exact regardless of phase size, and is still needed even though `R` is fixed
today: `moves_per_phase × batch_games` is config, the last phase of a run can be
short, and `0.815 × 10240 / 4096 = 2.04` is not an integer.

⚠️ Expressed per **game**, not per position, because that is the ratio AZ's numbers
give directly; the per-position form additionally depends on a mean game length AZ
does not publish and ours does not match.

---

## 7. The optimiser

### 7.1 Values

| | value | authority |
|---|---|---|
| optimiser | **SGD with momentum**, not Adam or AdamW | AGZ Methods, Optimisation |
| momentum | 0.9 | AGZ Methods |
| L2 | `c = 10⁻⁴` | AGZ Methods |
| batch | 4,096 | AZ p.4 |
| learning rate | 0.2 → 0.02 → 0.002 → 0.0002 | AZ p.14 |
| total steps | see §6; AZ's own was 700,000 | AZ Table S3 |

⚠️ **The L2 term is applied once per optimiser step, not once per micro-batch.** Leave
`c‖θ‖²` out of the accumulated loss entirely and express it as the optimiser's
`weight_decay`, which is applied at step time by construction. Putting it inside an
accumulated loss makes its effective strength depend on the accumulation count, which
is a bug that shows up as a mysterious dependence on a memory parameter.

⚠️ **`weight_decay = 2c`, not `c`** (noted 2026-07-31, `train/loss.py`'s
`weight_decay_for`). AGZ writes the penalty as `c‖θ‖²` with **no factor of a half**,
so its gradient is `2cθ`; `torch.optim.SGD(weight_decay=w)` adds `wθ`. Passing `c`
straight through halves the regularisation the paper specifies, and no check in §12
would notice — it changes nothing except a curve, months later. The literature's
frequent `½c‖θ‖²` convention is what makes this easy to get wrong in either direction,
so the value is derived from AGZ's formula rather than from a habit.

⚠️ **Our measured 5000 positions/s was fwd+bwd+AdamW.** SGD+momentum is cheaper in
both time and state (one buffer instead of two: ~26 MB rather than ~51 MB), so the
figure is a lower bound, but it has not been re-measured. Do that before it is quoted.

### 7.2 Batch 4096 by accumulation, and why it is exact

Four micro-batches of 1024, gradients accumulated, one optimiser step. This is **not**
an approximation of batch 4096 — it is batch 4096:

- the network is pre-norm LayerNorm with a final `norm_f` (`spec.md` §7); there are no
  batch statistics anywhere, so every per-sample activation is independent of batch
  composition;
- the loss is a mean over samples, so four micro-batch losses each scaled by ¼ sum to
  the gradient of the mean over 4096, up to floating-point accumulation order;
  ⚠️ **each term by the count it is a mean over.** The policy term is a mean over the
  micro-batch's rows; the value term is a mean over its *value-supervised* rows
  (`value_mask`, §5.1). Its weight is therefore that micro-batch's supervised count
  over the whole batch's, not ¼: with one scale for both, a micro-batch holding one
  supervised row weighed as much as one holding two, and the accumulated value gradient
  was not the whole batch's (`loss.py:micro_batch_weights`, fixed 2026-09-09,
  `docs/core-algorithm-review.md` §3). With every row supervised the two scales coincide;
- cost is one persistent fp32 gradient buffer, 6,383,360 × 4 B = **25.5 MB**, and
  *fewer* optimiser steps than batch 1024.

The measured 1773 MB of training activations at batch 1024 (bf16) is what makes this
necessary; the same tensor at 4096 is ~7.1 GB in an 8 GB card.

**The consequence that matters**: AZ's learning rate schedule transfers unchanged,
instead of needing a rescaling we would have had to invent. In a tabula rasa run,
every hyperparameter that can be inherited rather than chosen is one fewer thing the
result depends on us for.

### 7.3 The schedule, and the one number neither paper gives

AZ p.14: *"The learning rate was set to 0.2 for each game, and was dropped three times
(to 0.02, 0.002 and 0.0002 respectively) during the course of training."* **The drop
points are not in the paper**, and Figure 1's x-axis is not annotated with them.

Express the schedule as **fractions of total training**, since our run is ~159,000
steps and AZ's was 700,000 (§13). Two candidate sets:

- AGZ Extended Data Table 3, verified, reinforcement-learning column: `10⁻²` over
  0–400k, `10⁻³` over 400–600k, `10⁻⁴` beyond, i.e. drops at **57 %** and **86 %** of
  700k steps. Two drops, not three, and different absolute values.
- the released `pseudocode.py`'s `learning_rate_schedule`, **read 2026-07-30** as
  `{0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4}`, i.e. **14 %, 43 %, 71 %** of
  700k steps. Three drops, matching AZ's prose, at exactly AZ's four values. ⚠️ Still
  the file `search.md` §1.1 refuses as an authority — but here it agrees with the paper
  rather than contradicting it, and supplies only the numbers the paper omits.

**Default: the pseudocode fractions**, because they are the only source with three
drops and AZ's prose says three. Recorded in `2026-07-30-fidelity.md` as a parameter taken from
an unrefereed file.

⚠️ **`lr = 0.2` under plain SGD is the single most likely value in this document to be
wrong for our network.** AZ's was a 20-block, ~46M-parameter convolutional resnet;
ours is a 6.38M-parameter pre-norm transformer initialised at `std = 1/sqrt(d)`.
Nothing about AZ's tuning transfers to that. §12's overfit-one-batch check will show
divergence immediately, and a learning-rate sweep is the first legitimate ablation on
top of the AZ reproduction — not a change to it.

**First measurement, 2026-07-31.** `python -m brokefish.train.overfit --sweep
0.2,0.02,0.002,0.0002`, 1024 positions harvested at `n = 16`, 300 steps each from one
initialisation and one frozen batch:

| lr | KL at 0 | KL at 300 | value at 0 | value at 300 |
|---|---|---|---|---|
| 0.2 | 1.585 | 0.876 | 1.047 | **1.383** |
| 0.02 | 1.585 | 0.750 | 1.047 | 0.703 |
| **0.002** | 1.585 | **0.571** | 1.047 | **0.178** |
| 0.0002 | 1.585 | 0.946 | 1.047 | 0.215 |

`lr = 0.002` wins on both heads, **two orders of magnitude below AZ's value**. At
`lr = 0.2` the value loss *rises* and then sits at exactly 1.383 while the gradient
norm falls to 0.1: that is a **saturated `tanh`**, which stops producing gradient
entirely — `search.md` §15.3 already watches for it, and here it is a training pathology
rather than a search one.

⚠️ **This is a rate comparison and not yet a pass of check 1.** No rate reached
`KL ≈ 0`, so it does not yet say the loop *can* memorise a batch; 300 steps of
SGD+momentum over 1024 positions is not enough to conclude either way. And the batch
was harvested at `n = 16`, where the mean `policy_len` is 3.5 — the targets are far
sparser than the `n = 800` ones a real run produces. The verdict on `lr` is provisional
until check 1 is run at §12's own size, and the row in [the fidelity audit](../journal/2026-07-30-fidelity.md)
§4.2(a) stays open until it is.

### 7.4 Ablation 2 — Muon, and the granularity that comes with it

`--optimizer muon`, `brokefish/train/muon.py`. Off by default: the SGD baseline is the
thing being ablated against, so it does not move.

Muon replaces SGD-momentum's update with the nearest semi-orthogonal matrix to the
momentum buffer, computed by a quintic Newton–Schulz iteration rather than an SVD. It
applies to hidden matmul weights only. Here that is **6,291,456 of 6,383,360
parameters — 98.6 %** — because this network has no tied embedding and its three
readout heads are 17,664 parameters between them. The remaining 91,904 (five embedding
tables, three heads, every bias and LayerNorm gain) run AdamW.

| | value | authority |
|---|---|---|
| momentum | 0.95, Nesterov | `torch.optim.Muon`; Jordan's post reports Nesterov better in every case tested |
| Newton–Schulz | 5 steps, Polar Express per-iteration coefficients | arXiv 2505.16932 Alg. 1 |
| lr adjustment | `√max(1, d_out/d_in)`, generalised to chunks | `torch.optim._muon._adjust_lr` |
| aux group | AdamW, `betas` and `eps` as §7.1's ablation 1 | Jordan's rule for embeddings/heads |
| chunking | Q, K, V separately; head groups optional | Jordan's post; CMuon; Kimi K3 |

**`torch.optim.Muon` is the oracle and it is in our stack.** `tests/test_muon.py`
asserts bit-equality against it on the unchunked path, and against `torch.optim.AdamW`
(`foreach=False`) for the auxiliary group. Everything except the chunking therefore has
an external reference, which is the only reason a hand-written optimiser belongs in
this repository at all.

⚠️ **The learning rate does not transfer from ablation 1 and must be swept.** Muon's
update has RMS `1/√fan_in` = 1/16 here; AdamW's is `≈ lr`. So the RMS-matched
equivalent of the control's `1e-3` is `≈ 0.016`, which is where `torch.optim.Muon`'s
own 0.02 default sits. `aux_lr` is carried as a *ratio* to `lr` so one schedule — one
warmup, one cosine — drives both groups.

⚠️ **Q, K and V are orthogonalised separately, and this is not an option.** Jordan's
post states it plainly: *"Muon works better for optimizing transformers if it is
applied to their Q, K, V parameters separately, rather than together."* CMuon
(arXiv 2608.02502) names the mechanism — one fused tensor gets one shared
preconditioner `(Σ_j G_jᵀG_j)^{-1/2}` across blocks whose principal directions need not
align. `nn.MultiheadAttention` hands us the fused `(768, 256)` form by default, so the
*default* is the wrong algorithm. `--no-qkv-split` exists to measure that, not to use it.

⚠️ **`in_proj` and `out_proj` chunk on opposite axes.** `in_proj_weight` is `[Q; K; V]`
on **rows**, and within each 256-row block head `h` owns rows `32h .. 32h+32`.
`out_proj.weight` carries its head structure on the **input** dimension: columns
`32h .. 32h+32` are head `h`. Getting this backwards produces an optimiser that runs,
converges, and is a different algorithm; `test_out_proj_chunks_columns_not_rows` is the
check that bites.

⚠️ **The lr adjustment had to be re-derived for chunks, and the obvious version is
wrong.** torch's factor exists to hold the update's RMS at `1/√fan_in` whatever the
aspect ratio — that consistency *is* what makes the rate transfer. A concatenation of
orthogonalised blocks has Frobenius norm `√Σ_i min(r_i, c_i)`, not `√min(R, C)`, so the
correct factor is

    ratio = √( R / Σ_i min(r_i, c_i) )

which reduces to `√max(1, A/B)` exactly at one chunk. Applying torch's formula *per
chunk* instead inflates `out_proj` at `head_group = 1` by `√8`, which would make a
granularity sweep secretly a learning-rate sweep.

⚠️ **`grad_clip` means much less under Muon.** A global clip rescales every gradient by
one factor and Newton–Schulz divides each matrix by its own norm, so the clip is very
nearly invisible to 98.6 % of the parameters. "The clip never fires" stops being
evidence of stability; watch the update-to-weight norm ratio instead.

⚠️ **Polar Express is the default, and it is what makes the rate actually transfer.**
The `√max(1, A/B)` arithmetic is exact, so whether the promised RMS consistency is
*delivered* depends entirely on how well the iteration converges — and Jordan's fixed
quintic converges differently depending on the block's aspect ratio. A square Gaussian
block has Marchenko–Pastur mass near zero that five steps do not lift; a very
rectangular one has a concentrated spectrum that they do. Measured 2026-08-07 over six
real shapes × five seeds, achieved RMS as a fraction of the `1/√fan_in` target:

| scheme | mean | min | max | **spread** | \|σ−1\| max |
|---|---|---|---|---|---|
| jordan | 0.9160 | 0.8473 | 0.9579 | **0.1106** | 0.4222 |
| polar | 1.0053 | 0.9883 | 1.0157 | **0.0274** | 0.2547 |

**The legacy quintic breaks learning-rate transfer across our own layers by 11 %;
Polar Express cuts that to 2.7 %.** Cost, all 32 matrices at five steps: **15.36 ms
against 15.03**, +2.2 %, both ~0.07 % of a 21 s step. `--ns-scheme jordan` reverts to
torch's algorithm and is what the oracle test pins.

Two choices there are ours and both are measured, not inherited. **The 1.01 safety
margin applies on every executed step**, where the paper's 8-step recipe drops it on
the last — at five steps we truncate mid-schedule, and keeping it throughout measured
better (spread 0.027 vs 0.030). **bfloat16 is kept**: fp32 moves the achieved RMS in
the fourth decimal and doubles the cost, 30.0 ms vs 15.4.

⚠️ The coefficient table was transcribed from arXiv:2505.16932's HTML by a small model,
so it is **not** trusted on provenance — it is trusted because its steady state
`(1.875, −1.25, 0.375)` is the closed form `(15x − 10x³ + 3x⁵)/8` for the matrix sign,
which a garbled transcription does not land on, and because it behaves under
measurement exactly as the paper claims. Both facts are asserted in `tests/test_muon.py`.

**`head_group` is a hyperparameter, not a constant.** 8 = no head splitting, 1 =
per-head (Kimi K3, GLM-5 "Muon Split"). arXiv 2605.08933 turns the choice into a
gain-versus-cost inequality and finds the optimum *moves during training* — per-head
early, coarser late — with both extremes losing somewhere. It is one integer, so it is
a sweep, and the ladder is `{1, 2, 4, 8}`.

**Not implemented, deliberately:** Gram Newton–Schulz (42 % of an iteration that costs
0.07 % of a step, at the price of a documented half-precision instability), momentum
warmup. NorMuon is
implemented, off, and flagged in its docstring as reconstructed from a search summary
rather than the paper.

The survey behind all of this is
[`journal/2026-08-05-muon-survey.md`](../journal/2026-08-05-muon-survey.md).

---

## 8. The loop

One **generation** is one self-play phase followed by one gradient phase.

```
loop:
    self-play phase   B = 4096 games in flight, M move-steps
                      harvest one MoveRecord per move-step
                      close out pending records of games that ended
                      reset_finished()
    gradient phase    steps from §6, batch 4096 by accumulation
                      weight_gen += 1
    every K steps     checkpoint (§9)
```

`M` is a throughput knob, not a contract: it trades phase-switch overhead against how
stale the self-play weights are within a generation. A phase of a few minutes is the
target. The **only** thing the ratio of §6 requires is that it be honoured in the long
run, which the carry guarantees for any `M`.

`weight_gen` is stamped into every record by the search, so staleness is measurable
after the fact rather than assumed: the mean of `current_gen − record.weight_gen` over
a sampled batch is a logged scalar (§11).

### 8.1 Weight synchronisation, and the silent failure it invites

**The network exists in two representations and the loop must convert between them
every generation.**

Training needs a backward pass, so the gradient phase runs `nn/model.py` in torch with
fp32 master weights. Self-play runs the fused encoder, whose weights are pre-permuted
into mma B-fragment order by `pack_b` — that permutation is the whole reason the
kernel reaches 61.6k evals/s (`CLAUDE.md`), and `cuda_impl.Encoder.__init__(source)`
builds it by `.detach()`ing from a torch module. **It is a snapshot.**

So each generation ends with: gradient phase → rebuild the packed encoder from the
updated torch model → `weight_gen += 1` → next self-play phase. The rebuild is a
permutation of 6.4M parameters and costs nothing worth measuring.

⚠️ **A snapshot built once is a working system that never learns.** Construct the
fused encoder in a constructor and self-play runs generation-0 weights for the entire
run, while every loss curve looks healthy, every counter looks healthy, and the Elo
curve is flat for a reason no diagnostic in §11 reports. This is the same shape as the
C1 bug where `env.push_history` returned a *new* ring and a dict still pointed at the
old one — the symptom appeared three moves and eight thousand simulations later.

The countermeasure is the same one C1 adopted, and it is not a comment:

- the packed encoder carries the `weight_gen` it was built from;
- the self-play phase **asserts** that it equals the loop's current generation before
  a single search runs, and raises otherwise;
- a fingerprint of the torch master weights is compared against the fingerprint stored
  at pack time, so a rebuild that silently packed the wrong module fails loudly too.

Cost is one integer compare and one hash per phase, against a failure mode that costs
a run.

### 8.2 Precision

| | |
|---|---|
| master weights | **fp32** |
| forward and backward in the gradient phase | **bf16 autocast** — the 1773 MB activation measurement was taken this way, so the 8 GB budget assumes it |
| the policy log-softmax and the loss reduction | **fp32**. The term is a reduction over ≤64 logits and one position; doing it in bf16 saves nothing measurable and costs precision on the quantity being optimised |
| SGD momentum buffer | fp32, 25.5 MB |
| accumulated gradients | fp32, 25.5 MB (§7.2) |
| self-play forward | fp16, unchanged — the fused encoder's own format (`spec.md` §7) |

⚠️ The two phases therefore run the network at **different precisions on purpose**,
and `tests/test_b2.py` already holds the two paths to agreement. What is *not* covered
is agreement after a weight update, which is what §12 check 9 adds.

⚠️ **Self-play must run under `torch.no_grad()` and the training phase must not hold a
self-play tree.** The 8 GB budget was measured on the premise that the two activation
blocks never coexist (`roadmap.md`, Risks). Interleaving them at finer grain than a
phase breaks that measurement, which is the real reason the loop alternates rather
than a preference for simplicity.

---

## 9. Checkpoint and resume

A checkpoint must make the run reconstructible, which is stronger than making the
weights loadable. It holds:

- fp32 master weights, and the SGD momentum buffers;
- `step`, `weight_gen`, total games completed, total positions generated, total
  samples drawn, and the §6 `carry`;
- the learning-rate schedule position (derived from `step`, stored anyway so a
  changed schedule is detectable);
- **every RNG state**: the search's generator (Dirichlet at the root, the `multinomial`
  move sampling below `tau_plies`) and the buffer's sampling generator;
- the euro counter and the €/h rate it was accumulated at (§10);
- a config hash covering everything in this document, so a resume under changed
  parameters fails loudly instead of silently producing a hybrid run.

The replay buffer is checkpointed **separately and less often**, because it is ~13 GB
against ~50 MB of weights. A resume that finds no buffer snapshot restarts the buffer
empty and says so; that is a real discontinuity in the run and it goes in the log, not
in a comment.

⚠️ **Resume must be bit-exact.** Checkpoint at step `k`, continue to `k + 100`; then
resume from `k` and run 100 steps. The two weight tensors must be identical. This is
one of the few checks in C2 with a definite right answer, and it catches the whole
class of "something in the loop is not in the checkpoint" bugs, which otherwise appear
weeks later as an unexplained kink in the Elo curve.

---

## 10. The euro counter

`evaluation.md` §5.3 requires `euros_spent` to be written by the same writer as `elo`,
because two files means a hand join six weeks later. The training loop is that writer
for the cost half.

Accumulate **device-seconds**, split by phase (self-play, gradient), multiply by a
configured €/h stored in the checkpoint. Wall clock, not an estimate from FLOPs: the
card throttles to 1230-1290 MHz inside the loop ([the Gate 1a measurement](../ledger/perf.md#what-the-tree-actually-costs-measured-2026-07-30)) and any FLOP-derived
figure would be an upper bound on work and a lower bound on cost.

**Two numbers, per Track D's settled decisions:**

- **the curve's x-axis** — self-play plus gradient phases of the run itself. This is
  what makes the curve comparable to AlphaGateau. It **excludes** league games,
  evaluation matches and all development.
- **total project cost** — published separately, includes everything.

Both are written per curve point; neither is derived from the other.

---

## 11. Instrumentation

wandb, **online by default** (corrected 2026-07-30 — the previous default was offline,
justified by a failure mode that does not exist: wandb writes every record to
`wandb/run-*/` on disk before uploading, and syncs from a background thread that
retries, so a dropped network stalls the upload and never the run). `--wandb-mode
offline` is for a machine with no credentials; `python -m wandb sync wandb/offline-run-*`
pushes it afterwards. The JSONL sink is unconditional either way.

**Metric names are `self_play/…`, `gradient/…`, `buffer/…`, `euros/…`** — one level of
nesting and no more (corrected 2026-07-31). ⚠️ wandb groups a run's charts on the
**first** path component only, so the previous `phase/` wrapper put every metric in the
run into a single folder and threw away the grouping the second component would have
given for free. The wrapper carried no information: every record logged there was a
phase. This changed the JSONL keys too, so a reader of a pre-2026-07-31 log wants
`phase/gradient/kl` where a current one has `gradient/kl`.

Terminal codes are logged **by name** — `self_play/terminal_checkmate`,
`self_play/terminal_threefold` — from [spec §4.3](spec.md#43-terminal)'s table via
`env.TERMINAL_NAMES`, not as `terminal_codes/4`.

**Per gradient step**: total loss, and the policy, value and L2 terms separately — a
single scalar hides which head has stopped learning. Learning rate. Global gradient
norm and weight norm.

**Per gradient phase**: mean staleness `current_gen − record.weight_gen` over sampled
batches; buffer occupancy in games and in sampleable positions; achieved
samples-per-game against the 65.2 target; positions/s.

**Per self-play phase**: the whole of `search.md` §15's counter block, which
`SearchStats.snapshot()` already produces. Plus the target distribution — win/draw/loss
fractions of completed games and mean game length, which is the earliest signal that
self-play has collapsed — and the **game-length cap firing rate** (§5.4).

**Continuously**: euros, both numbers.

**Deferred by decision, not oversight**: `max |post-scale attention logit|`, the fp16
overflow ceiling of `perf.md`. One line to add when wanted. The risk it covers is a
run that NaNs and is lost rather than one that is silently wrong — overflow is loud.

---

## 12. Validation, in the absence of an oracle

Ordered by what they catch, not by cost. All of them are exit criteria for C2, in the
sense `roadmap.md` uses for perft.

1. **Overfit one batch.** 4096 fixed positions, no buffer, no self-play, train to
   near-zero loss. Exercises the entire forward/backward/optimiser/accumulation path
   in seconds. If it cannot memorise 4096 positions, nothing downstream is worth
   running. Also the fastest detector of a wrong `lr`.
2. **The loss transcribed twice.** Once in the training path, once as a scalar
   reference reading the AZ formula directly, compared on random tensors — the same
   method `test_search.py` used on `ucb_score`. Includes the accumulation identity:
   four micro-batches of 1024 against one batch of 4096, gradients equal to fp
   tolerance.
3. **The label decode, against a live search.** Run one search; for each root edge,
   the logit the *training* path selects for that edge's stored label must be the same
   scalar `expand` used to build the edge's prior. Plus the truncation case: on a
   position with more than 64 legal moves, every stored label must still be a
   candidate — §3.5's exact invariant, replacing the first draft's "the same 64-move
   support on both sides", which is unachievable. ⚠️ **This is the most important
   check in C2.** A permuted target trains a network that learns a shuffled
   labelling, the loss falls the whole time, and every other check in this list
   passes while it happens. It is cheap enough to assert on every micro-batch, and
   `az_loss(strict=True)` does.
4. **Resume bit-exactness**, §9.
5. **Buffer invariants**, asserted continuously and cheaply: every pending record is
   filled exactly once; no record is sampled before it is filled; eviction never
   removes a game with live pending records; occupancy never exceeds the window.
6. **Value-parity on a hand-built game.** A short game with a known result, records
   checked ply by ply against the §4 parity rule. The `L >= 3` warning from
   `search.md` applies: use a game of odd *and* even length, since a two-ply game cannot
   distinguish the correct rule from its inverse.
7. **A rules-only supervised smoke test.** Train the network to predict the legality
   mask, or material balance, from the position — quantities the *rules* define, so it
   is inside the tabula rasa boundary as a diagnostic and never as a training run. It
   separates "can this network and this loop learn anything" from "is self-play
   working", which are otherwise confounded for the first several generations.
8. **D1's layer-0 scalars**, once training runs: value calibration against realised
   outcomes, policy entropy, draw rate. These are the only in-flight evidence that the
   loop is healthy rather than merely running.
9. **Weights actually propagate.** Take one gradient step large enough to move the
   network measurably, rebuild the packed encoder, and require the fused encoder's
   output on a fixed batch to *change* — and to match the updated torch model to
   `test_b2.py`'s tolerance. ⚠️ Every other check in this list passes on a loop whose
   self-play is frozen at generation 0 (§8.1). This one is the only one that does not.

⚠️ None of these is an oracle. They are necessary conditions, and a loop that passes
all eight can still be training the wrong objective competently. What would come
closest to an oracle is the **rules-only supervised task at scale**: if the network
cannot fit a quantity the rules determine exactly, the fault is ours and not the idea's.

---

## 13. Cost arithmetic

Arithmetic over measured numbers, not measurements.

**The gradient phases.** 10M games × 65.2 samples/game = 6.52×10⁸ samples = **~159,000
optimiser steps** at batch 4096. At the measured 5000 positions/s (AdamW; SGD is
cheaper and unmeasured), that is **~36 hours**.

That our step count lands near AZ's own 700,000 would be a coincidence at batch 1024
and is not one here: we run 4.4× fewer games at the same batch, so we run 4.4× fewer
steps.

**⚠️ The self-play side does not fit the project's stated budget at `n = 800`, and the
plan is to find out whether the configuration hill-climbs before making it cheap.**
`roadmap.md`'s full-run estimate — 113 hours, and the €100-500 figure derived from
it — assumes **32 simulations per move**, from the superseded Gumbel sizing.
`search.md` froze `n = 800`. At 800:

| | evals | at 57.0k/s |
|---|---|---|
| 10M games × 80 plies × **32** sims | 2.56×10¹⁰ | ~113 h (`roadmap.md`) |
| 10M games × 80 plies × **800** sims | 6.4×10¹¹ | **~3,120 h** |

which is 130 days on the 4060, and puts training at **1.1 %** of the run rather than
the 24 % it would be at `n = 32`. On an H100 the same work is roughly 26 days at
$3.95/h ≈ $2,500, against a stated €100-500.

**Sequenced, not unreconciled (decided 2026-07-30).** The order is: establish that the
AZ configuration *hill-climbs at all* at `n = 800`, then bring the simulation count
down with the throughput work — `search.md` §11's Gumbel seam, playout cap
randomisation, and the sims sweep `roadmap.md` schedules in C4 — until the wall-clock
is acceptable. Demonstrating the climb is the scientific claim; making it cheap is
engineering that follows it, and doing them in the other order risks debugging a
learning failure and a search approximation at the same time.

⚠️ **What C2 owes that plan is not to bake `n` into the training contract.** §6's
cadence is per *game* and invariant to `n`; nothing else in this document reads it.
The first hill-climbing evidence therefore comes from a **truncated run**, not the
full 10M games, and the buffer, the schedule fractions (§7.3) and the euro counter all
have to behave sensibly when the run is stopped early. That is a requirement, not an
aside.

---

## 14. Not frozen

| | why it is open |
|---|---|
| **`lr = 0.2`** | AZ's value for a 46M-param resnet, on a 6.38M-param transformer. §7.3. First legitimate ablation |
| **the drop points** | published nowhere; §7.3 takes them from an unrefereed file |
| **bytes per game** | §5.2 sizes the mapping from an assumed 80-ply mean. Measure it in generation 1 and resize |
| **the phase size `M`** | a throughput knob, tuned once the loop runs |
| **checkpoint frequency `K`** | AGZ used every 1,000 steps; ours has no evaluator to feed, so it is a resume-granularity choice |
| **the unmasked policy softmax** | §3.3's alternative: denominator over all 2048 logits, no movegen at training time. Coherent with a masked search by conditioning, and cheaper. Not the default because AZ masks and because it spends gradient on permanently illegal moves |
| **the bootstrapped value ablation** | KataGo's mix. The field is written and unused from generation 0, so it costs nothing to defer |
| ~~**playout cap randomisation**~~ | closed 2026-08-07 — see §15 |
| **simulations per move** | §13. `n = 800` for the hill-climbing run, reduced afterwards by the `search.md` §11 seams and C4's sweep. Not a C2 parameter either way |

---

## 15. Playout cap randomisation

**Off by default** (`pcr_p = 0`), so a run started today reproduces every number
measured before 2026-08-07. The authority is KataGo §3.1 (arXiv 1902.10565), read
directly rather than from memory.

**The tension.** The value target is one noisy binary result per *entire game*, so
value training wants many cheap games. The policy target is `N / n` over a tree, and
unless `n` is large the search does not deviate much from the prior, so the policy
does not readily improve. Uniform `n` has to pick one. E1.1 measured the policy side
of that on this project: doubling `128 → 256` was worth **+275 ± 67 Elo at equal
positions** (`journal/2026-08-07-e11-sims-256.md`), which is what makes the expensive
tier worth spending on a fraction of turns.

**The rule.** On a proportion `p` of turns the search runs the full cap `N` and the
position is recorded; on the other `1 - p` it runs a fast cap `n < N` with the §6.1
Dirichlet noise **off**, and **nothing is recorded**. KataGo, verbatim:

> On a small proportion p of turns, we perform a full search, stopping when the tree
> reaches a cap of N nodes, and for all other turns we perform a fast search with a
> much smaller cap of n < N. **Only turns with a full search are recorded for
> training.** For fast searches, we also disable Dirichlet noise and other explorative
> settings, maximizing strength.

⚠️ **The cheap turns are not recorded for the value target either.** They are still
*played*, and the extra games that produces is the whole value gain — more independent
outcomes, not more rows per outcome. KataGo's main run used `p = 0.25` and
`(N, n) = (600, 100)`, annealed to `(1000, 200)`; their ablation puts it at **1.37×**.

**One budget for the whole batch, per move.** `search.md` §11's hook is per *game*, and
the per-game gating is real and tested — but `simulate` hands all `B` staged leaves to
the encoder every iteration whether or not they are active, so a batch holding mixed
budgets costs the **maximum**, not the mean. Measured 2026-08-07: encoder throughput per
board is flat from 4096 boards down to 128 (97.2 %), so narrowing the batch to the
active set buys nothing either. A homogeneous batch is what makes an average cap an
actual saving, and `Search.self_play_move(sims=...)` therefore owns `budget` and the
iteration count together.

⚠️ **`n_sims` is the FULL cap, never the mean.** It sizes the node pool
(`n_max = n + 1`) and the path arrays, so a fast cap may only go *down*; the loop
refuses a `pcr_fast_sims` above it at construction. The realised mean is logged per
generation as `self_play/sims_mean` — measured, because it is the cost axis.

**The schedule is stratified, not Bernoulli**: exactly `round(p × moves_per_phase)`
turns per phase, positions drawn without replacement, redrawn every generation and
seeded from `(seed, generation)` so §9's bit-exact resume gets it for free. Bernoulli
draws give the same turns in expectation but a per-generation cost wandering by ±11 %
at `moves_per_phase = 10`, and that cost is the x-axis. Redrawing matters because a
game's ply offset within the phase is fixed for its whole life, so a constant pattern
would record one residue class of each game's plies and nothing else.

**Two things it changes that are easy to miss.**

⚠️ **The window is denominated in games and only `p` of each game's plies reach it.**
At `p = 0.3`, `window_games = 20 000` holds ~760 k records where a uniform run held
~2.5 M — a 3.3× smaller and far more correlated window. This is the failure KataGo
itself hit: their Fixed-600 ablation needed *"the window size doubled, as an informal
test without doubling showed major overfitting due to lack of data."* **Raise
`--window-games` by `1/p`**, and lower `--mean-plies` by the same factor, since the ring
is sized in records and only the recorded ones are written.

⚠️ **§6's cadence rides positions *generated*, not stored**, which is deliberate: it
holds the gradient steps per generation equal to a uniform-`n` control's, so a
difference in the curve is the data and not the step count. The price is that each
stored position is trained on `1/p` times more often — reuse `0.815 → 2.72` at
`p = 0.3` — which is part of the intervention and is watched through
`gradient/staleness`.

**The trap in the buffer, which is silent.** `ReplayBuffer.append` does two unrelated
jobs: it accumulates the move's rows, and it *closes* every game that just finished.
The natural way to write "do not record cheap turns" is to not call it on them — which
loses the closure of every game that **ends** on a cheap turn, i.e. ~`1 - p` of all
games. Those games' earlier records stay in `_pending`, the search resets the slot, and
the next game to finish there flushes them under **its** result: no exception, no
counter, and label noise on most of the buffer. So the two jobs are separated by a
`store` flag rather than by the call, and the closure path is unconditional. §4's value
rule was rewritten in the same change, for the same reason.

---

## Changelog

**draft, 2026-07-30.** First version. Two things it closes that `spec.md` §11 listed
as open — the value target (§4, closed by the AZ paper's own text rather than by a
choice) and the reuse factor (§6, derived from AZ's published step and game counts) —
and one that `search.md` §13 listed as absent, the game-length cap (§5.4). The
accumulation identity of §7.2 is what lets AZ's learning-rate schedule be inherited
rather than invented, and is the reason batch 4096 is a reproduction and not an
aspiration. §12 exists because C2 is the first phase with no external oracle, and §13
records that `n = 800` and the project's cost model have not been reconciled.

§3 was revised the same day. The first draft required the training path to rebuild the
search's edge array "in the same canonical order", which is not necessary: the softmax
denominator is a sum over a set. What must match is the label-to-logit decode, and the
truncation when a position exceeds 64 legal moves. §3.2 was added because the loss is
often described as a KL against the visit distribution, which is the same optimisation
— the two differ by the target's own entropy, a constant — and because the KL form is
the one worth logging. Every deviation this document introduces is tabulated in
[the fidelity audit](../journal/2026-07-30-fidelity.md) §4.

§3.1 was also revised the same day, in the other direction. The first draft read AGZ
literally and concluded that `τ → 0` past move 30 made the *training target* one-hot,
making v0's stored `N/n` a deviation affecting ~60 % of positions. It is not a
deviation: the released pseudocode stores untempered visit counts and applies
`num_sampling_moves` only in `select_action`, and AGZ's own gloss on the schedule says
`τ` selects moves. Retracted to `2026-07-30-fidelity.md` §2.1a, where it is kept as a closed item
because the symbol overloading will invite the same wrong reading again. The same read
of `select_action` narrowed §2.1(c): the released code counts 30 **plies**.

**§3.5 revised again, 2026-07-31, and the second revision deleted the problem.** The
first two drafts both had the training path *recomputing* the softmax support — one by
rebuilding the search's truncation, one by rebuilding the full legal set — and neither
asked why it was recomputing anything the rollout already knew. It was: `search.md` §10's
policy arrays are `E` wide and zero-padded, so storing the root's whole edge set rather
than only its visited edges costs **no bytes**. The record now carries its own support,
`az_loss` takes no `env`, the dominant tensor went from two `[N, 8192]` fp32 arrays to
one `[N, 96]` gather, and the truncation question stops existing. The engine-side
cross-check the recompute used to provide moves to `audit_labels` on a schedule.

**Implemented 2026-07-31**, `brokefish/train/` and `tests/test_train.py` (33 checks).
Four things the document said turned out to be wrong or incomplete once they were
code, and all four are corrected above rather than annotated:

- **§3.5, the truncation.** The contract asked the training path to reproduce the
  search's `E` support. It cannot — the search truncated with the generating
  weights — and the disagreement is not theoretical: it fired in the eighth generation
  of the first smoke run. The denominator is now the full legal set, which is AZ's own
  rule, is the coherent choice under softmax restriction, and turns §12 check 3 into an
  exact invariant.
- **§7.1, `weight_decay = 2c`.** AGZ's `c‖θ‖²` has no half in it.
- **§5.1, `root_value`** is in, and it is the visit-weighted root `Q`, not the raw
  network evaluation.
- **§9's checkpoint** carries the games in flight as well as the counters, because a
  fresh tree is built every move and the games are therefore the whole of self-play's
  state. Resume is bit-exact, tested.

Two things measured rather than assumed while building it. The backward pass is
deterministic run to run on this card, so §9's bit-exactness does not need
`use_deterministic_algorithms` to be more than a belt; and §7.2's accumulation
identity holds to 1e-11 relative **in float64**, while in fp32 it lands at 1e-4 to
1e-6 and moves between runs, because a matmul at `N` and a matmul at `N/4` select
different cuBLAS split-k reductions. The identity is exact; fp32 is not. The test
proves it in double and pins the fp32 case loosely, with the reason named.

Five decisions were taken on 2026-07-30 and are folded in above rather than left as
open rows: the replay window keeps AZ's literal 500,000 games and lives in a
disk-backed mapping (§5.2), `n = 800` stands for a hill-climbing run with the
simulation count reduced afterwards (§13), the weight-synchronisation step and its
currency assertion are specified (§8.1) because a stale packed snapshot is a loop that
trains forever and learns nothing, training precision is pinned (§8.2), and
`root_value` is acknowledged as an edit to *search* code rather than training code
(§5.1).
