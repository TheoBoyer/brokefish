# MCTS v0

**Status: draft, 2026-07-30.** This document is normative for the v0 search and for
the torch reference it is validated against. It does not modify
[`spec.md`](spec.md), which owns the engine and network contracts; where it needs
something §7 leaves implicit, it says so and the clarification is owed back to §7.

Scope: one self-play move, from a position to a chosen move plus a training record.
The training loop, the replay buffer's lifetime and the evaluation protocol are C2
and C3 and are outside this document. The one thing it fixes for them is the record
schema in §10.

The design goal is the simplest search that can produce a curve point, laid out so
that every published refinement is a replacement of one named function rather than a
change to a data structure. §11 lists those functions and what plugs into each.

---

## 1. What this is

v0 is the AlphaZero search, adapted to fixed-shape batched execution on one GPU.

Faithfulness to AlphaZero is a deliberate correctness lever rather than a homage. It
gives the reference implementation of §12 something published to be checked against,
in the same way `perft` gave the move generator an oracle outside this repository.
Every place v0 departs from it is listed in §3 with the reason.

### 1.1 Which document is the authority

Three artefacts describe this algorithm and they do not agree with each other, so
this document names its source at every step.

| | what it settles |
|---|---|
| **AGZ**, Silver et al., *Mastering the game of Go without human knowledge*, Nature 550:354-359 (2017), Methods, "Search Algorithm" | the search itself. AZ says "unless otherwise specified, the training and search algorithm and parameters are identical to AlphaGo Zero", so every formula comes from here |
| **AZ**, Silver et al., arXiv:1712.01815 and Science 362:1140-1144 (2018) | the parameters that change for chess, and the list of differences from AGZ. It gives **no** PUCT formula |
| the `pseudocode.py` released with the Science paper | nothing normative. It is the only artefact that makes the value perspective explicit, and it contradicts AGZ in three places |

⚠️ **The pseudocode is not the specification, and §13 lists the three places v0
follows AGZ against it.** Anyone porting from the pseudocode inherits an
un-normalised Dirichlet, a visit count that is one too large at every interior node,
and a value read that maximises the opponent's outcome.

Identical to AGZ, with the source for each: per-edge statistics
`{N(s,a), W(s,a), Q(s,a), P(s,a)}`; selection by `argmax(Q + U)` with
`U(s,a) = c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))`; `Q` initialised to 0 on
expansion; the backup `N += 1`, `W += v`, `Q = W/N` over the path; Dirichlet noise
`P(s,a) = (1-eps) p_a + eps eta_a` with `eta ~ Dir(alpha)` and `eps = 0.25`; and
`pi(a|s0) = N^(1/tau) / sum_b N^(1/tau)` at `tau = 1` for the first 30 plies then
`tau -> 0`. From AZ: `n = 800` simulations, `alpha = 0.3` for chess, a value head in
place of rollouts, the expected outcome rather than a win probability, and the final
game outcome as the value target.

⚠️ **`c_puct`'s value is not published.** AGZ calls it "a constant determining the
level of exploration" and says the search parameters were chosen by Gaussian process
optimisation. The logarithmic form `log((N + pb_c_base + 1)/pb_c_base) + pb_c_init`
and the numbers 19652 and 1.25 exist only in the released pseudocode. Over
`n = 800` that form moves between 1.25 and 1.29, so v0 is running a constant
`c_puct` of about 1.25 with a 3 % drift, and any claim that this reproduces
AlphaZero's exploration rests on an unrefereed file.

Not present, and each one is an addition to this rather than a deviation from it:
virtual loss, first-play-urgency reduction, WDL values, a moves-left head, certainty
propagation, transposition merging, playout cap randomisation, forced playouts,
lower-confidence-bound root selection, Gumbel root selection.
[the prior-art survey](../journal/2026-07-28-prior-art.md) covers where each comes from and
§11 says which function it replaces.

⚠️ **Tree reuse and resignation are deviations, not additions**, and this section
used to have them on the wrong list. AGZ reuses the subtree under the played move
and resigns clearly lost games, and AZ does not list either among its differences,
so both are part of AlphaZero. §3.6 and §3.7 carry them.

---

## 2. Design criteria

**Every array shape is known at launch.** No allocation, no compaction, no
data-dependent shape, and no host synchronisation anywhere inside a move. The
tree is a fixed pool per game, indexed by integers. This is the same constraint
[spec §4](spec.md#4-the-engine-contract) puts on the engine, for the same reason.

**One warp per game, one lane per slot.** This is what `csrc/movegen.cuh` and
`csrc/step.cuh` already require of their callers, and reusing their warp-collective
device functions from inside the search kernels is why A1 had to land before C1.

**One simulation per game per step.** The network batch is the game batch, so a
batch of `B` leaves comes from `B` distinct trees and no leaf is ever selected
twice inside a step. §3.1 explains what this removes.

**Static sparsity.** A node holds a capped list of edges over its legal moves, not
a dense array over the 2048-entry action space. The cap is a compile-time constant,
so the sparsity costs nothing in shape flexibility; the only runtime-varying
quantity in the entire design is a loop trip count.

---

## 3. Where v0 differs from AlphaZero, and why

### 3.1 Parallelism, and the absence of virtual loss

AlphaZero ran one game per worker across thousands of accelerators. A single tree
therefore had to produce a batch of leaves before calling the network, and PUCT is
deterministic given a tree, so selecting twice in a row returns the same leaf.
Virtual loss is the standard fix: on selecting a leaf, immediately record a pretend
lost visit on every edge along the path, which makes that path look worse so the
next selection goes elsewhere; when the real value arrives, the pretend visit is
removed and the real one applied. It costs bookkeeping, it biases selection by an
amount that has to be tuned, it is the main source of race conditions in
multithreaded implementations, and lc0 layers a collision counter on top of it
because forcing full diversity turns out to be worse than tolerating some
duplicates.

None of that applies here. `B` games advance in lockstep, one simulation each, so
the batch is `B` leaves from `B` trees by construction. Virtual loss, collision
counting and collision caps do not exist in v0 and have no seam reserved for them.

### 3.1a Three places the released pseudocode contradicts AGZ

v0 follows AGZ. Recorded here because the pseudocode is what most reimplementations
start from, and each of these is silent.

**`sqrt(sum_b N(s,b))` and not `sqrt(parent.visit_count)`.** AGZ's `U(s,a)` divides
by the square root of the sum over the node's own edges. The pseudocode uses the
node's visit count, which is one larger at every node below the root: the simulation
that created a node incremented that node's count but took no edge there. The two
differ most at a node's first revisit, where AGZ has `sum_b N = 0`, so `U = 0` for
every edge, every score is `Q = 0`, and the tie goes to the lowest edge index. Under
the pseudocode the count is 1 and the highest-prior edge is taken instead.

⚠️ **So under AGZ the first descent below a newly created node ignores the policy**,
and in v0's canonical order it always takes the lowest slot's lowest square. That is
one arbitrary visit per node, about 2 % of all selections at `n = 800`, spent
systematically on the same move. Measured: 31 to 43 exact ties per position over 64
simulations, which is one per node created. It is a faithful reading of the published
formula and it is the one place where faithfulness costs something real. §13 has it
as an open question rather than a settled choice.

**A normalised Dirichlet.** AGZ writes `eta ~ Dir(alpha)`, which sums to 1. The
pseudocode mixes raw `numpy.random.gamma(alpha, 1, k)` draws without dividing by
their sum; at `alpha = 0.3` and 31 legal moves those sum to about 9.3, so the priors
come out summing to roughly 3 instead of 1 and the exploration term is inflated
several-fold. §6.1 follows AGZ.

**A value that is not read from the opponent's point of view.** The pseudocode stores
`value_sum` per node, and its `backpropagate` puts each node's sum in *that node's
own mover's* frame. Its `ucb_score` then reads `child.value()` and adds it to the
parent's score unmodified, where the parent's mover is the other player. Taken
literally the search maximises the opponent's outcome. AGZ's per-edge `Q(s,a)` is an
action value for the player at `s` and has no such ambiguity, which is why §4.2
stores per edge. §6.5 is the whole argument.

### 3.2 Memory

AlphaZero allocated nodes and gave each an unbounded child dictionary. v0 uses a
fixed pool of `n + 1` nodes per game and caps edges per node at `E` (§4). A
position with more than `E` legal edges keeps the `E` highest-prior ones and the
kernel counts the event. §4.3 carries the measurement that sets `E`.

### 3.3 Network input

AlphaZero's chess input was eight stacked positions plus repetition planes.
[spec §7.2](spec.md#72-token-features) gives the network one position plus a clock
feature and a three-valued repetition feature, so the search has to supply `rep`
rather than a history stack. §7 defines it.

### 3.4 Terminal detection

AlphaZero's pseudocode asked the game whether it was over. Our engine returns a
terminal code and a result from [spec §4.3](spec.md#43-terminal), including
threefold repetition against a per-game ring, which is state the search owns and
which A1 deliberately left unported. §7 is that half.

### 3.5 Value convention

The value head emits `tanh` output in `[-1, 1]` from the mover's point of view
([spec §7.4](spec.md#74-heads)). AlphaZero's pseudocode works in `[0, 1]` and
backs up `1 - v` for the opponent instead of `-v`.

⚠️ The two are equivalent under `q01 = (v + 1) / 2` **only if the constant is
rescaled with them.** In PUCT the value term and the exploration term are added, so
halving the range of `Q` doubles the effective exploration. v0 therefore stores `Q`
in `[0, 1]` and converts once at backup, which keeps `pb_c_init = 1.25` meaning
what it means in the published pseudocode. An implementation that keeps `Q` in
`[-1, 1]` and reuses 1.25 is running a search with **half** the intended exploration
and will not reproduce anything.

⚠️ **Direction corrected 2026-07-31** — this said "twice". Substituting
`q01 = (v + 1) / 2` into the score and dropping the per-node constant:

```
score[-1,1] = pb_c * P + (2 * q01 - 1) = 2 * ( (pb_c / 2) * P + q01 ) - 1
```

so `[-1, 1]` at `pb_c_init = c` is `[0, 1]` at `c / 2`. The equivalent constants are
therefore 1.25 in `[0, 1]` and **2.5** in `[-1, 1]`.

⚠️ **Every constant in Q-space rides this remap, and the one that did not was the
first-play urgency** (§6.6). A draw is `0` in `[-1, 1]` and `0.5` in `[0, 1]`; v0
carried the literal `0`, which is a certain loss. That paragraph is the rule this
section states, broken one constant over.

**Why the flip and not a second readout.** `W_value` is `[256, 1]` applied to one
token, so a position's evaluation yields one number, the mover's. Reading the *other*
king's token would yield the opponent's estimate directly and appear to remove the
`1 - q`.

⚠️ **It does not remove the parity.** The flip is not a consequence of the network's
convention; it is a consequence of one leaf evaluation having to be attributed to edges
owned by alternating movers. A path `R` (White to move) → `C` (Black to move) → `L`
(White to move) evaluates only `L`. `edge_Q[C][e]` is read by Black and
`edge_Q[R][e]` by White, so something has to alternate along the path whatever the
network returns. With two readouts the code selects `v_black(L)` at odd levels and
`v_white(L)` at even ones, which is the same parity computation selecting between two
arrays instead of computing `1 - q`. The motivation for the change dissolves.

Given that, three reasons to prefer the flip. The second readout is off-distribution:
[spec §8](spec.md#8-loss-and-targets) trains the value against the outcome from the
mover's point of view, so on a white-to-move position only slot 15's readout receives
gradient, and training both would be an extra loss term. Chess is zero-sum, so
`v_black = 1 - v_white` holds exactly and the flip imposes it for free, where two
readouts would hope the network learns it. And two unconstrained readouts make
`edge_Q` at a node and at its parent fail to sum to 1, so the tree stops being a
consistent game tree and PUCT compares values on drifting scales.

Two per-side value functions is the right design in a general-sum game. In a zero-sum
one it is redundant. The idea does have a use that costs nothing: `v_white + v_black - 1`
is a free calibration diagnostic, and a KataGo-style auxiliary loss on it is a
legitimate later experiment.

The place this becomes worth reopening is a WDL head, where the flip becomes a
permutation of (win, draw, loss) rather than `1 - q`. §11 lists that as a spec change
rather than a seam.

---

### 3.6 Tree reuse, which AlphaZero has and v0 does not

AGZ: "The search tree is reused at subsequent time-steps: the child node
corresponding to the played action becomes the new root node; the subtree below this
child is retained along with all its statistics, while the remainder of the tree is
discarded." AZ does not list this among its differences from AGZ, so AlphaZero
reuses. The released pseudocode builds a fresh root every move and is wrong about it.

v0 discards the whole tree every move, which is a real loss and not a simplification
with no cost. The subtree under the played move typically holds a large share of the
`n` simulations already spent, so reuse is worth some fraction of `n` for free, and
`n` is the most expensive number in this project ([the cost arithmetic](../ledger/perf.md#the-search-in-the-loop)). What it costs to implement is
node lifetime: a bump pointer becomes a free list, or a copy-compact pass moves the
retained subtree to the front of the pool. §11 has it as a real change rather than a
seam.

### 3.7 Resignation, which AlphaZero has and v0 does not

AGZ: "In order to save computation, clearly lost games are resigned. The resignation
threshold `v_resign` is selected automatically to keep the fraction of false
positives (games that could have been won if AlphaGo had not resigned) below 5 %. To
measure false positives, we disable resignation in 10 % of self-play games and play
until termination."

This is a cost mechanism and this is a cost project, so it belongs on the list of
things to build rather than the list of refinements. Early networks lose slowly and
end games by the fifty-move rule, so the tail of a decided game is a large share of
the positions generated and almost none of the information. The self-calibrating
threshold is the part worth copying: it needs no tuning and it measures its own
damage. C2 owns it, `pick_move` in §11 is where it plugs in, and §15's terminal-code
histogram is what would tell us how much it is worth.

---

## 4. The tree

### 4.1 Parameters

| symbol | meaning | v0 value |
|---|---|---|
| `n` | simulations per move | 800, a parameter, see §4.4 |
| `B` | games in flight, and therefore the network batch | 4096, a parameter, see §4.4 |
| `E` | edge cap per node | 64 |
| `Nmax` | nodes per game, `n + 1` | 801 |
| `Dmax` | path buffer depth, `n` | 800 |
| `pb_c_base` | PUCT exploration base | 19652 |
| `pb_c_init` | PUCT exploration constant | 1.25 |
| `alpha` | Dirichlet concentration | 0.3 |
| `eps` | Dirichlet mixing weight | 0.25 |
| `tau_plies` | plies played at temperature 1 | 30 |

`Nmax = n + 1` because the root is created once and each simulation creates at most
one node.

⚠️ **`n` is a memory parameter, not only a time parameter.** Each simulation adds a
node to a tree that has to persist for the whole move, so `n` sets `Nmax` and the
tree costs `856 · (n+1) · B` bytes. That is what couples `n` to `B` in §4.4. The
compute cost is `n` network evaluations per move and scales the same way, so `n` is
paid twice.

`Dmax = n` is the exact bound and not a cap. A path through a tree of `n + 1` nodes
traverses at most `n` edges, so the path array of §4.2 at 3 B per entry is 2.4 KB per
game and 9.8 MB at `B = 4096`. Since the exact bound is affordable, v0 does not cap
the descent depth, does not truncate a path, and has no behavioural change or counter
to go with one. Real tree depth will be far below `Dmax`; the array is sized for the
worst case because the worst case is cheap.

### 4.2 Arrays

Structure of arrays, all resident in VRAM for the whole self-play phase, all
allocated once.

```
node_board     [B, Nmax, 32]  u16    the position at this node, spec §2.1
node_control   [B, Nmax]      i16    spec §2.2
node_hash      [B, Nmax]      u64    spec §6.1
node_value     [B, Nmax]      f16    the network's v, or the terminal result
node_nedges    [B, Nmax]      u8     0 for a terminal node
node_flags     [B, Nmax]      u8     bits 0-2 terminal code, bit 3 expanded,
                                     bit 4 the move that created it was irreversible
node_parent    [B, Nmax]      i16    -1 at the root
node_pedge     [B, Nmax]      u8     the edge index in the parent that reaches here

edge_move      [B, Nmax, E]   u16    (slot * 64 + square) | (promo << 11), spec §3
edge_prior     [B, Nmax, E]   f16    P(a), normalised over this node's edges
edge_child     [B, Nmax, E]   i16    -1 when unexpanded
edge_N         [B, Nmax, E]   u16    visit count
edge_Q         [B, Nmax, E]   f32    running mean in [0,1], see §3.5

path_node      [B, Dmax]      i16    the current simulation's path, root first
path_edge      [B, Dmax]      u8
path_len       [B]            i16    not u8: Dmax is 800

game_ring      [B, 100]       u64    spec §6.3, per game, survives across moves
game_ring_len  [B]            i32
game_ply       [B]            i32    for the temperature schedule
node_count     [B]            i32    bump pointer into the node pool
```

Per node that is 88 B of node fields and 768 B of edge fields, so **856 B per
node**. At `n = 800` a tree is 686 KB per game, plus 384 B of path and 800 B of
ring. §4.4 has the totals. Arithmetic from the field widths, not a measurement.

**The path array removes two of the three depth-proportional walks, not all three.**
A simulation traverses its path three times: down during selection, and up twice, once
to count repetitions before the terminal test and once to back the value up.

The descent is irreducibly serial. Node `d + 1` is not known until selection has run
at node `d`, so it is a dependent chain of load-edges, argmax, load-child, and its
latency grows with depth no matter how the path is stored. Nothing here helps it.

The two upward walks are a different matter. `node_parent` and `node_pedge` do encode
every path, but following them is another dependent chain: each address is the value
just loaded, so one lane does 40 serialised round trips for a depth-40 path. Writing
the path down on the way down costs nothing, since the descent already holds `v` and
`e`, and it turns both walks into independent accesses across 32 lanes. The
repetition scan of §7 becomes a ballot and the backup of §6.5 becomes a parallel
scatter, since each level of a path touches a distinct edge and the point of view at
each level is fixed by parity.

`node_parent` and `node_pedge` are kept for invariant 3 and for debugging rather than
for traversal.

⚠️ [the cost arithmetic](../ledger/perf.md#the-search-in-the-loop) argues that none of this is likely to matter for throughput, because the
network dominates a node by three orders of magnitude. The reason to keep the path
array anyway is that a parallel scatter over an array is simpler code than a pointer
chase, and Gate 1a will say whether the argument holds.

There is no stored legality mask. Expansion converts the mask into edges and the
edges carry it from then on.

### 4.3 Why `E = 96`

**64 until 2026-08-02, then 96.** Both the original bet and the reason it lost are
kept here, because the way it lost is the interesting part: the histogram was right
and the decision was still wrong.

The original measurement, over the 10 000 positions of `data/cuda_testset`, which
come from random legal playouts, counting a promotion as four edges:

| statistic | edges |
|---|---|
| mean | 31.2 |
| median | 32 |
| p99 | 52 |
| p99.9 | 59 |
| max | 65 |
| fraction above 64 | 0.01 % |

Three reasons for 64 rather than 48 or 128. The measured tail put truncation at one
position in 10⁴. A warp has 32 lanes, so 64 edges is exactly two per lane in the
selection scan with no predicate on the second pass. And the edge arrays are 90 % of
a node, so 128 would nearly double the tree, from 2.81 GB to 5.34 GB at the sizing
of §4.4, to cover a range the histogram says is empty.

**What the bet missed: truncation is not uniform over the positions that matter.**
Re-measured on 2026-08-02 over 80 000 roots from the trained `t24h-n256` policy —
mean 23.8, p99 58, p99.9 68, max 83, and 0 % above 96, so the *aggregate* histogram
had if anything moved the right way. But conditioning on the roots where a forced
mate exists:

| | all roots | roots with a mate available |
|---|---|---|
| mean edges | 23.6 | 45.1 |
| expansions at the cap | 0.317 % | 4.80 % |

Mating positions are exactly the wide ones — the losing king is in the open, and the
attacker's queen and rooks have their full mobility. So the 0.317 % aggregate rate
was hiding a **15× higher** truncation rate on the one class of node where dropping
an edge is not a small loss of precision but the loss of a *proved win*. A cap chosen
against the mean was being applied to the tail that carries the signal.

96 rather than 128: 96 = 3 × 32 is still exactly `kE/32` edges per lane with no tail
predicate (the scan is written as `kEPerLane` and `static_assert`s `kE % 32 == 0`),
it clears the measured max of 83 with margin, and it costs 1.5× the tree rather than
2×. 128 would have bought nothing the histogram says is occupied.

⚠️ The distribution is still measured on a policy that changes for the whole run, so
`E = 96` remains a bet on a moving distribution. §15.1 is the monitoring that keeps
it a bet rather than an assumption.

⚠️ Playouts under-represent open middlegames and the constructed maximum in chess is
218, so treat any measured fraction as a floor. Truncation is therefore **counted at
runtime** and the count is reported with every self-play phase — `truncated_nodes`
and `truncated_mass` generally, plus `terminals_truncated`, which counts the strictly
worse event of the cap dropping an edge the rules had already proved terminal. If
they rise, §11 has the evolution.

Truncating by prior rank stays inside the training boundary, since the ordering
comes from the learned policy and carries no chess opinion. What it costs is
strength and label fidelity on the truncated positions, which is why it is measured
rather than assumed.

### 4.4 Why `n = 800` and `B = 4096`

`n` is chosen first, and it is chosen for convergence rather than for throughput.
800 is AlphaZero's number for chess, so v0 runs the published algorithm at the
published budget and a failure to learn cannot be blamed on an under-powered search.
Throughput at that budget is expected to miss Gate 1's band, and the intended
response is to bring `n` down and the optimisations of §11 in afterwards, against
a baseline that is known to converge. Choosing `n` for throughput first would leave
no such baseline.

`B` is then chosen freely, because the encoder turns out to be flat in batch size.
Measured 2026-07-30, `impl 'cuda'`, `forward_full`, four interleaved rounds with the
order reversed on alternate rounds:

| `B` | ms | evals/s | against 16384 |
|---|---|---|---|
| 1024 | 15.98 | 64 064 | 100.1 % |
| 2048 | 31.89 | 64 227 | 100.3 % |
| 4096 | 63.86 | 64 142 | 100.2 % |
| 8192 | 127.89 | 64 055 | 100.0 % |
| 16384 | 255.89 | 64 028 | 100.0 % |

One CTA per board over 24 SMs means even 1024 boards is more than twenty waves, so
there is nothing to saturate that a larger batch would saturate better. Every
throughput number in [`perf.md`](../ledger/perf.md) was taken at 16384 and the table says they
transfer unchanged.

The tree cost is `856 · (n+1) · B`:

| | `B` = 1024 | 2048 | 4096 | 16384 |
|---|---|---|---|---|
| `n` = 128 | 113 MB | 226 MB | 452 MB | 1.81 GB |
| `n` = 400 | 352 MB | 703 MB | 1.41 GB | 5.63 GB |
| `n` = 800 | 703 MB | 1.41 GB | 2.81 GB | 11.2 GB |

`n = 800` with `B = 16384` does not fit in 8 GB, which is the whole reason this
section exists. `B = 4096` costs 2.81 GB of tree and gives 512 blocks of work per
kernel, comfortably enough to amortise the block tail of §9.

**4096 is AlphaZero's SGD batch, and matching it is a starting point rather than an
optimum.** AlphaZero's learner drew 4096 games from a one-million-game window and took
**one position from each**, so a gradient batch was 4096 positions from 4096 distinct
games. At `B = 4096` a single move step produces exactly that shape, one record per
game, with no reconstruction by the replay buffer. Under one-position-per-game
sampling the ratio of games generated to batch size does not matter, so nothing forces
`B` to equal the learner's batch; the equality is chosen so that the first
configuration is AlphaZero's and any later divergence from it is deliberate.

⚠️ The correspondence is exact at "one position per game" and cosmetic beyond it. The
4096 records from one move step all sit at the same ply, which AlphaZero's random-ply
sampling does not. The replay buffer samples across plies, so this matters only if C2
ever trains directly on a step's output.

⚠️ The learner runs on the same card, alternating with self-play
([`roadmap.md`](../roadmap.md)), and its peak footprint at a gradient batch of 4096 has
never been measured. Backward activations at 4096 × 32 tokens × 256 across 8 layers are
plausibly a few GB against 2.81 GB of tree. If the two do not fit together, `B` is one
line and is the first thing to cut.

`B` also sets the granularity of the alternation between the two phases, which is C2's
concern and is the one real lower bound on it. A generation produces `B · 80 ≈ 328k`
positions, about 80 steps of 4096 at a reuse factor of 1. A much smaller `B` would make
each generation too small to be worth a learner phase.

---

## 5. Invariants

An implementation may assert all of these; the reference implementation does.

1. `node_count[b] <= Nmax` at all times.
2. Node 0 is the root, `node_parent[b][0] == -1`.
3. `edge_child[b][v][e] != -1` implies `node_parent[b][child] == v` and
   `node_pedge[b][child] == e`.
4. A node with terminal code nonzero has `node_nedges == 0` and is never expanded.
5. A node with `node_nedges == 0` and terminal code 0 cannot exist. An empty mask
   is checkmate or stalemate by [spec §4.3](spec.md#43-terminal).
6. `sum_e edge_N[b][0][e] == s` after `s` simulations.
7. `edge_N[b][v][e] == 0` implies `edge_child[b][v][e] == -1`, and for a
   non-terminal node `c` reached by edge `(v, e)`,
   `edge_N[b][v][e] == sum_f edge_N[b][c][f] + 1`. The `+ 1` is the simulation
   that created `c`, which incremented the edge into it and found no statistics
   below it to walk through. A terminal node is excluded, having no edges to sum.
8. The root is never terminal. A terminal position ends the game and is never
   searched.
9. `edge_Q` lies in `[0, 1]`.

Invariant 5 is the one that catches a movegen or masking regression, and it is the
reason [spec §7.4](spec.md#74-heads) requires asserting `mask.any(-1)` before a
masked softmax.

---

## 6. The algorithm

One self-play move is `root_init`, then `n` iterations of three kernels, then
`select_and_advance`. No host synchronisation at any point, and every launch has a
static shape.

### 6.1 `root_init`

For each game: copy the current position into node 0, reset `node_count[b]` to 1,
zero node 0's edge arrays, set `node_parent[b][0] = -1`.

The root's expansion needs a network evaluation, so `root_init` runs the evaluate
and expand steps of §6.3 and §6.4 on node 0, then mixes Dirichlet noise into its
priors:

```
P(a) <- (1 - eps) * P(a) + eps * eta(a),   eta ~ Dir(alpha, ..., alpha)
```

over the root's `node_nedges` edges, with `alpha = 0.3` and `eps = 0.25`.

⚠️ cuRAND has no device-side gamma generator. `Dir(alpha)` with `alpha < 1` is
generated as `eta_i = g_i / sum_j g_j` with `g_i ~ Gamma(alpha, 1)`, and
`Gamma(alpha < 1, 1)` is drawn as `Gamma(alpha + 1, 1) * U^(1/alpha)` with the
Marsaglia-Tsang squeeze for the `alpha + 1` part. The reference implementation must
use the same construction and the same stream so the two can be compared tree for
tree (§12).

### 6.2 `descent`

One warp per game. Starting at node 0, repeat: score every edge of the current node
by §6.6, take the argmax, and follow it. If the chosen edge has `edge_child == -1`,
stop. If the current node is terminal, stop.

```
v = 0                                    // current node
d = 0
while d < Dmax:                          // Dmax = n is the exact bound, never reached
    if terminal_code(v) != 0: break
    e = argmax_edge(b, v)
    path_node[b][d] = v                  // written as we go, see §4.2
    path_edge[b][d] = e
    d += 1
    c = edge_child[b][v][e]
    if c == -1: break
    v = c
path_len[b] = d
```

The path is written on the way down, which costs nothing because the descent already
holds `v` and `e`, and it is what makes §6.5 and §7 parallel.

On stopping at an unexpanded edge `(v, e)`:

1. `apply` the move with `step_full` from `csrc/zobrist.cuh`, producing the child
   board, control word and hash.
2. Compute the child's repetition count against the game ring **and** the hashes on
   the path from `v` up to the root, per §7.
3. Call `terminal` from `csrc/terminal.cuh` with the child's mask, `in_check`,
   control word and board, and **no history**, then fold code 4 in where the
   count from step 2 is 3 or more. `terminal`'s history argument is spec §6.3's
   `[N, 100]` ring, which cannot express the tree half, so the search owns the
   count and hands the engine only what the engine's own signature covers. The
   fold sits at spec §4.3's priority, above insufficient material and below
   everything else; both are draws, so `result` does not move.
4. Allocate `c = atomicAdd(&node_count[b], 1)`, write the child's board, control,
   hash, parent and parent edge, and set `edge_child[b][v][e] = c`.
5. If the child is terminal, set `node_value[b][c]` from its `result` converted to
   `[0, 1]` and mark it needing no evaluation.

On stopping at a terminal node, no node is created and the terminal node's stored
value is what gets backed up.

⚠️ `movegen` is called here for the terminal test and again in `expand` for the
edges. Calling it once and staging the mask in shared memory is an optimisation, and
the mask is 256 B per game so it fits; v0 does not do it because the two calls sit in
different kernels and the environment is 2.2 % of a node.

### 6.3 `evaluate`

The encoder, `forward_full(boards, control, rep)`, over all `B` new nodes.

⚠️ Games whose descent created no node, or created a terminal one, have nothing to
evaluate. v0 evaluates them anyway and discards the result, which keeps the launch
shape static and avoids a compaction pass. The waste is bounded by the terminal
rate and is deliberate.

An implementation must not feed a terminal position to the encoder if it also
applies the legality mask there, because an all-illegal row softmaxes to `NaN`
([spec §7.4](spec.md#74-heads)). Feeding it and discarding the output is safe only
because v0 masks during expansion, on nodes it has already established are not
terminal.

### 6.4 `expand`

For a non-terminal new node `c`:

1. `movegen` gives the 32-word mask.
2. Enumerate set bits in canonical order, lane `p` owning mask word `p`, ascending
   by `p` then by square, and emit one edge per bit. A bit whose move is a
   promotion emits four edges, one per promotion type in the
   [spec §3](spec.md#3-moves-and-the-action-space) order `N B R Q`.
3. If the count exceeds `E`, keep the `E` largest unnormalised priors and
   increment the truncation counter. Ties are broken by canonical order and the
   survivors are written in canonical order, not in prior order, so that the
   ordering in step 2 is the only one an implementation has to reproduce.
4. The unnormalised prior of an edge is `policy_logits[c][slot][square]`, plus
   `log softmax(promo[c][slot])[promo]` for a promotion edge. This is the
   factorisation `P(target | piece) · P(type | piece)` that
   [spec §7.4](spec.md#74-heads) requires, in log space so that a single softmax
   over the node's edges normalises both at once.
5. `edge_prior <- softmax` over the node's edges. `edge_N`, `edge_Q` and
   `edge_child` are zeroed and set to -1.
6. `node_value[b][c] <- (value[c] + 1) / 2`, and set the expanded bit.

The canonical enumeration order in step 2 is normative. It is what makes a tree
comparable between the reference and the kernel, since PUCT ties are broken by edge
index.

### 6.5 `backup`

One warp per game, over the path array rather than up the parent links. Every level
of a path touches a distinct edge, and the point of view at each level is fixed by
the parity of its distance from the leaf, so the updates are independent and lane `i`
takes level `i`:

```
L = path_len[b]
qleaf = node_value[b][leaf]        // in [0,1], side-to-move-at-the-leaf's point of view
for d = lane; d < L; d += 32:
    v = path_node[b][d]
    e = path_edge[b][d]
    q = ((L - d) % 2 == 0) ? qleaf : 1 - qleaf
    edge_N[b][v][e] += 1
    edge_Q[b][v][e] += (q - edge_Q[b][v][e]) / edge_N[b][v][e]
```

At a real depth of 40 that is two iterations instead of 40 dependent loads. The
running mean rather than a sum-and-divide keeps `edge_Q` in `[0, 1]` and its error
bounded, so fp32 needs no wide-accumulator argument.

The parity: `edge_Q[v][e]` is by definition the value of move `e` seen by the player
to move at `v`, because that is the player choosing among `v`'s edges in §6.6.
`node_board[d]` sits `L - d` plies before the leaf, so the player to move there is
the same as the player to move at the leaf exactly when `L - d` is even. At
`d = L - 1`, the edge that reaches the leaf, `L - d` is 1, so that edge takes
`1 - qleaf`, which is correct: the player who moved into the leaf is the one who is
*not* to move there.

⚠️ Inverting this parity produces a search that reliably plays the worst available
move, and it is the single most common bug in reimplementations of this algorithm.
The forced-mate check of §12 is what catches it.

A worked case, because "the network already returns the mover's value" is true and is
not sufficient. Path `R` (White to move) → `C` (Black to move) → `L` (White to move),
with only `L` evaluated, and the network returning `qleaf = 0.9`, meaning White wins
90 % from `L`.

| level `d` | node | who chooses here | what `edge_Q` must hold | `L - d` |
|---|---|---|---|---|
| 1 | `C` | Black | 0.1, because Black moved into a position White wins | 1, odd |
| 0 | `R` | White | 0.9 | 2, even |

Storing 0.9 on `C`'s edge would tell Black that moving into a 90 %-losing position is
Black's best available move. The network's output is from the perspective of the mover
**at the leaf**, and a single simulation attributes that one number to edges owned by
players who alternate along the path, so it has to be re-expressed at every other
level. The edge into the leaf always flips, because the player who chose it is by
definition not the player to move at the leaf.

The flip can equivalently live at read time, with `Q` stored per node and negated
whenever a parent reads a child. That is the same computation moved across the
read-write boundary and it does not remove it. v0 stores per edge and flips at write
time.

### 6.6 The selection score

Exactly the AlphaZero pseudocode, on a node `v` with parent visit count
`N_v = sum_e edge_N[b][v][e]`:

```
pb_c = log((N_v + pb_c_base + 1) / pb_c_base) + pb_c_init
pb_c = pb_c * sqrt(N_v) / (edge_N[b][v][e] + 1)
score(e) = pb_c * edge_prior[b][v][e] + Q(e)
Q(e)    = edge_N[b][v][e] == 0 ? 0.5 : edge_Q[b][v][e]
```

`pb_c_init` is **added** to the logarithm rather than multiplied by it. Ties are
broken by the lowest edge index. An unvisited edge scores `Q = 0.5`, which is AGZ's
first-play urgency — a **draw** from the mover's point of view — expressed in the
`[0, 1]` convention of §3.5; §11 has the alternatives.

⚠️ **Corrected 2026-07-31; v0 shipped `0` here and it is an absorbing state.** AGZ
scores an untried move at `Q = 0` in `[-1, 1]`, where 0 is a draw. §3.5 remaps the
tree to `[0, 1]` and warns that the two are equivalent *only if the constants are
rescaled with them* — and then rescaled the value and not this. A literal `0` in
`[0, 1]` is a **certain loss**, so against a value head near the draw value an untried
move needs `pb_c * prior > 0.5`, i.e. `prior > 0.0145` at `n = 800`. Anything below
that is unreachable **at every simulation budget**, so its prior can never be
corrected upward and can only fall further.

Measured on run `c2-8h` (177,669 games, 2,827 steps): 3 of 20 root moves visited at
`n = 64` *and* at `n = 800`, `KL(pi||P) = 0.0855` at both; decisive games fell
5.13 % → 1.76 % over seven hours while the policy collapsed onto one
position-independent move. The same network with `Q = 0.5` visits 20 of 20 at
`n = 800`. The story is in
[the 2026-07-31 collapse post-mortem](../journal/2026-07-31-value-collapse.md).

### 6.6a The terminal collapse

`terminal_collapse`, **off by default**. On a node with at least one edge whose child
the rules have proved is checkmate, the selection score of §6.6 is replaced by

```
win(e)   = the child of e is checkmate for THIS node's mover
score(e) = win(e) ? -edge_N[b][v][e] : -infinity      if any win(e)
```

so the simulations round-robin over the winning edges and reach nothing else. Ties
break to the lowest edge index, as in §6.6.

**Why.** §6.1a guarantees a mate at the root is *present* in the edge set; it does
not guarantee it is *chosen*. Measured over run `t9h-n128-sweep`'s finished buffer —
88,322 positions in the newest 600 games, of which **785 had a mate in one (0.89 %)**:

| | |
|---|---|
| the mating move was in the record's support | **100.0 %** |
| median `pi` on it | **0.302** (mean 0.401, p10 0.039) |
| targets that were a point mass (`pi > 0.99`) | **0 of 785** |
| the mate was the argmax | **56.1 %** (53.6 % past `tau_plies`, where the argmax *is* the move played) |
| mean visits on the mating edge | 51.3 of 128 |

Those roots are already won — `root_value` mean **0.886** — so a proved `1.0` leads its
rivals by ~0.11 while a fresh edge's exploration term is worth ~0.42, and 128
simulations over 42.4 edges is 3.0 visits each. Exploration wins. The effect is
monotone in the edge count, which is the same statement said twice: under 30 edges the
mate takes 0.631 of the visits, over 60 edges it takes 0.203.

⚠️ **The parity is the whole risk.** `win(e)` means the mover *at this node* delivers
mate, i.e. the child's `result` is `-1` in the child's own frame, i.e.
`node_value[child] == 0.0` — a certain loss for whoever moves there. It is the same
flip as §6.5's backup and as §6.1a's seeding. Inverted, the collapse steers onto the
move that loses on the spot, and nothing in the aggregates would say so.
`tests/test_search.py::test_the_collapse_gets_the_parity_right_below_the_root` checks
both directions of the mask against the rules over every allocated node.

**What it is not.** There is no proof propagation, no solved-node state, and no
handling of proved draws. A proved *draw* is good only when everything else is proved
lost, which needs the resolved-children logic of a full solver; §6.1a already seeds a
stalemate child at 0.5 precisely so that it stays buried, and the collapse must not
promote it. Only WIN collapses.

**Placement in the literature.** MCTS-Solver (Winands, Björnsson & Saito 2008)
backpropagates proved wins and losses; Leela's `certainty propagation` states the
play-time half of this exactly — *"if we have a certain win at root we can play it
immediately, regardless of the visits that move received"* — and CrazyAra's
`Exact-win-MCTS 2.0` prunes a proved loss with `Q = -inf`. None of them publish what
becomes of the **training target**; KataGo's policy target pruning is the precedent
that a target may legitimately differ from the visit distribution, and Gumbel
AlphaZero is the general form of the complaint — at a small simulation budget the
normalised visit count is not a policy improvement operator at all.

* Winands, Björnsson & Saito, *Monte-Carlo Tree Search Solver*, Computers and Games 2008 — [pdf](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf)
* [lc0 MCTS-Solver / certainty propagation](https://github.com/Videodr0me/leela-chess-experimental/wiki/MCTS-Solver---Certainty-Propagation-and-Autoextending), and [PR #487](https://github.com/LeelaChessZero/lc0/pull/487/files)
* Czech, Korus & Kersting, *Monte-Carlo Graph Search for AlphaZero*, ICAPS 2021 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2012.11045)
* Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2019 — [ar5iv](https://ar5iv.labs.arxiv.org/html/1902.10565)
* Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022 — [pdf](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

**Cost.** One `[B, Nmax, E]` uint8 mask, allocated only when the flag is on — 315 MB at
the Gate 1a shape against 3.8 GB for the edge arrays. Selection gains one warp ballot.
No simulations are saved: `descent` and the encoder launch on the full batch whatever
the descent did, so this is a quality change and not a throughput one.

### 6.7 `select_and_advance`

At the root, form `pi(a) = N(a) / n` and pick the move:

* `game_ply[b] < tau_plies`: sample from `pi`.
* otherwise: `argmax N(a)`, ties by lowest edge index.

⚠️ **With §6.6a on there is one exception, and it is the only place `pi` is not the
visit distribution.** If any root edge is a proved win, `pi` is set to **uniform over
exactly those edges** and zero elsewhere. Two reasons it is not left to the visit
counts alone: §6.1a seeds one visit on each *drawing* terminal edge too, and a
stalemate child at a won root is the worst move on the board — it would carry ~0.008 of
the target; and `budget + seeded` rarely divides evenly among the winners, so they
would sit at ±1 visit of each other. Every proved win at the root is a mate **in one**,
since §6.1a's sweep is depth 1, so the winners are equally optimal and uniform is exact
rather than an approximation. 72.5 % of the time there is exactly one of them.

⚠️ The moment a proof reaches the root from *below*, that stops being true — wins of
different lengths are not equally good and this line would need CrazyAra's `END_IN_PLY`
to prefer the shortest. `edge_N` is left untouched, so invariant 6 still reads the
visits the simulations actually made.

Then emit the training record of §10, push the pre-move hash into the game ring per
[spec §6.3](spec.md#63-detecting-a-repetition-on-device), apply the move to the real game position with
`step_full`, call `terminal` on the result, and mark finished games.

A finished game is replaced by a fresh one at the start of the next move, which
keeps `B` constant and is why `game_ply` and the ring are per game rather than
global.

---

## 7. Repetition during the descent

This is the half of [spec §6.3](spec.md#63-detecting-a-repetition-on-device) that A1 left to the search,
and it is the only genuinely new mechanism in v0.

A position's repetition count is the number of times its hash has occurred in the
game so far, including itself. Inside the tree, "the game so far" is the game ring
plus the hashes on the path from the root to the node's parent.

```
rep(child) = 1
           + count of child_hash in game_ring[b][0 : game_ring_len[b]]
           + count of child_hash in the path hashes from node 0 to v inclusive
```

`repetition_count` in `csrc/terminal.cuh` already does the ring half, warp
collective over 100 entries. The path half is the same shape: lane `i` reads
`node_hash[path_node[b][i]]`, compares against the child's hash, and
`__popc(__ballot_sync(...))` gives the count, in `ceil(path_len / 32)` iterations.
This is the second reason the path array of §4.2 exists; without it the same scan is a
chain of dependent loads up `node_parent`. The gather it does perform lands inside the
game's own `node_hash` block, which is 6.4 KB and therefore cache-resident.

⚠️ The repetition count is needed before the terminal test, which is needed before
the evaluation, which is needed before the backup, so the path is read twice per
simulation. That is inherent to the ordering and not a missed fusion.

The value fed to [spec §7.2](spec.md#72-token-features)'s `emb_rep` is
`min(rep - 1, 2)`, giving 0 for a first occurrence, 1 for a second and 2 for a third
or beyond. `emb_rep` has three entries, which fixes the clamp;
[spec §7.2](spec.md#72-token-features) does not state the mapping and should.

An irreversible move empties the ring ([spec §6.2](spec.md#62-the-repetition-window)).
Inside the tree the equivalent is that the path walk stops at the first node whose
incoming move was irreversible, since no earlier position can recur. v0 implements
the stop, because it is one predicate and it bounds the walk in the common case.

---

## 8. Terminal nodes

A terminal node is created, stored and never expanded. Its `node_value` is its
`result` from [spec §4.3](spec.md#43-terminal), converted to the `[0, 1]`
convention: `-1` becomes 0 and `0` becomes 0.5. A `result` of `+1` never occurs.

Every simulation that reaches a terminal node backs up its stored value again, so a
proven mate accumulates visits at the correct value without any special handling.
v0 does no certainty propagation, so a mate at depth 3 is learned by the search
statistically rather than proven and propagated. §11 has that evolution.

### 8.1 The search plays chess, with no exceptions

The engine implements the rules and the search does not get to disagree with it.
Repetition is a draw at the third occurrence, per
[spec §4.3](spec.md#43-terminal), and not at the second as engines conventionally
score it inside a search.

Three reasons, and they generalise to any future proposal of this kind.

The value target would stop being the game's outcome. A network trained on
second-occurrence draws has learned a game that is not chess, and it will misplay the
positions where the difference bites, which are exactly the positions where one side
is trying to escape a repetition.

Evaluation is against engines playing real chess under real arbitration
([`roadmap.md`](../roadmap.md#measuring-strength)). A divergence that helps during
training has to be unlearned or specially handled at evaluation time, and a
train-test mismatch on a rule is a hard bug to attribute.

It is an imported opinion about how to search, so it sits on the wrong side of the
training boundary. The rules are allowed; judgements about them are what has to be
learned.

⚠️ Not a rules divergence, and worth distinguishing: `irreversible` omits
python-chess's en passant clause ([`environment.md`](environment.md#three-places-it-departs-from-python-chess-all-deliberate)).
That only ever makes the repetition window longer, never shorter, so no repetition
can be missed and no rule is bent. It is a divergence from another implementation,
not from chess.

---

## 9. Kernel decomposition

Four launches per simulation, and `4n + 2` per move. Built 2026-07-30 in
`csrc/search.cuh` and `csrc/search.cu`, driven by
`brokefish/search/cuda_impl.py`; the register and shared-memory columns are what
`-Xptxas -v` reports.

| kernel | grid | registers | SMEM | what |
|---|---|---|---|---|
| `root_init` | `B / W` blocks | 38 | none | node 0 and the encoder staging, once per move |
| `descent` | `B / W` blocks of `W * 32` threads | 96 | 17 512 B | select, `step_full`, repetition, `terminal`, allocate |
| `evaluate` | one CTA per board | | | `forward_full`, the existing encoder kernel |
| `expand` | `B / W` blocks | 72 | 11 264 B | `movegen`, edge enumeration, prior softmax |
| `backup` | `B / W` blocks | 20 | none | the walk of §6.5 |
| `select_and_advance` | torch | | | §6.7, once per move |

`W` is warps per block, 8 in the engine kernels. **Nothing spills.** Fusing the
whole of §6.2 into `descent` was the thing most likely to: `movegen` alone is 71
registers and `step_full` is 80. At 96 registers and 256 threads it fits twice on an
SM, and the fallback of splitting the terminal test into a fifth launch was not
needed.

Two departures from the table as it was written, both taken deliberately.

**`select_and_advance` stays in torch.** It runs once per move, has no tree state in
it, and needs a `multinomial` for §6.7's temperature branch. Two launches out of
3202 at `n = 800`, and it means the CUDA search inherits the reference's §6.7 and
§10 rather than reimplementing them.

**The root's noise is drawn in torch** and handed over as a tensor, for the reason
§12 gives: one seed then drives both implementations and the differential test runs
at the real `eps`.

⚠️ **`descent` writes the leaf out to a contiguous `[B, 32]` buffer.** The encoder is
one CTA per board over a contiguous batch and a simulation's leaf sits at a different
node index in every game, so the descent stages it as it creates it rather than the
encoder learning to gather. That buffer is also what carries `rep`, and comparing it
against the reference's gather is what catches a wrong repetition count, which is
otherwise invisible in the tree until it changes the evaluator's output several
simulations later.

⚠️ **`E = 96` is compile-time and `B` and `Nmax` are runtime.** §6.6's scan is then
exactly two edges per lane with no predicate on the second pass, and §12's harness
still runs at `B = 8`.

### 9.1 The fusion v0 does not take

Only two of the three per-simulation kernels are separated by a real dependency. The
encoder has to run between `descent` and `expand`, because expansion needs its logits.
`backup` does not: it consumes the leaf's value, which the encoder also produced, so
`expand` and `backup` read the same output and touch disjoint state. They can be one
kernel.

That kernel can then absorb the next descent, since simulation `s + 1`'s selection
needs exactly the statistics `s`'s backup just wrote. A move becomes 800 iterations of
two launches:

```
A:  expand(leaf of s-1) ; backup(path of s-1) ; descent(s) ; repetition ; terminal ; allocate
B:  encoder
```

`2n + 2` launches per move instead of `3n + 2`, with the path array staying in
registers across the backup and the descent instead of round-tripping through global
memory.

v0 does not do it. At `n = 800` the saving is 800 launches, about 4 ms against a
25.5 s move by [the tree-cost prediction](../ledger/perf.md#how-much-the-tree-is-likely-to-cost), so it buys nothing measurable, and it costs the property that
every kernel is separately checkable against the reference implementation, which §12
depends on. The fusion is a late optimisation with a known shape, which is the right
state for it to be in.

⚠️ Trip counts vary between warps in `descent` and `backup`, since games are at
different depths. This is not intra-warp divergence: all 32 lanes of a warp work on
the same game and the same node, and they agree on every branch. The block-level
consequence is that warps which finish early sit idle while the block's registers
and shared memory stay allocated until the last warp exits, so a new block cannot
take that slot. With a grid of `B / W = 512` blocks against a handful of resident
slots per SM, that tail is amortised across thousands of blocks and should be small.
The design does not depend on it being small: the fallback is a persistent kernel
over a work queue, which changes one kernel and no data structure. There is no
`__syncthreads` in either kernel, so no warp ever waits on another.

---

## 10. The training record

One record per real move per game, `B` records per move step.

```
board       [32] u16    the position searched, spec §2.1
control     i16         spec §2.2
rep         u8          min(rep - 1, 2), spec §7.2
policy      [E] (u16 move, f16 prob)   the root's whole edge set
policy_len  u8          the root's edge count
value       f32         filled in when the game ends
weight_gen  u16         the network generation that produced the search
root_value  f32         the search's value of the root, Σ π(a) Q(a) in [-1, 1]
```

⚠️ **`policy` is the root's whole edge array, not only the edges a simulation
reached** (revised 2026-07-31). An edge with `N(a) = 0` is stored with `π = 0`. It
carries nothing for the training *target* and everything for the training
*denominator*: it is what tells C2 which moves the softmax normalises over, without
which the training path has to recompute `movegen` and re-derive a truncation it
cannot reproduce (`training.md` §3.5). The array is `E` wide and zero-padded either way,
so this costs **no bytes at all** — which is why the earlier `[min(E, n)]` width, and
the "at most `min(E, n)` nonzero entries" reasoning behind it, was a false economy.

A record is 64 + 2 + 1 + 192 + 192 + 1 + 4 + 2 + 4 = **462 B** independently of `n`
(the two 192s are `E = 96` moves at u16 and `E = 96` probabilities at f16).

`value` is the final game outcome from the point of view of the side to move in
`board`, in `[-1, 1]`. It is written when the game terminates, so a record is
incomplete until then and the buffer needs a per-game index of its own pending
records. The spare fields for a bootstrapped or mixed value target are `weight_gen`
plus one reserved `f32`; [spec §11](spec.md#11-the-rl-layer) leaves that choice open
and this schema is meant not to force a migration when it is made.

---

## 11. The seams

Every entry in the right column is a replacement of the named function and touches
no array in §4.2.

| function | v0 | what plugs in |
|---|---|---|
| `select_root` | §6.6, same as interior | Gumbel top-`m` plus sequential halving; forced playouts; root-specific FPU |
| `select_interior` | §6.6 | FPU reduction; `pi' - N/(1+sum N)`; dynamic `cpuct` |
| `backup` | §6.5 running mean | uncertainty weighting; subtree value bias correction |
| `policy_target` | §6.7, `N / n` | `softmax(logits + sigma(completedQ))`; policy target pruning |
| `pick_move` | §6.7, `tau` then argmax | LCB selection; resignation; playout cap randomisation |
| `edge_base(b, v)` | `(b * Nmax + v) * E` | a bump-allocated edge pool, if §4.3's counter demands it |

Two structural hooks, both free in v0 and awkward to retrofit:

**The simulation loop reads `budget[b]`** rather than a constant `n`, with `descent`
returning immediately for a game whose budget is exhausted. In v0 every entry is `n`.
This is what playout cap randomisation needs.

⚠️ **"and it is five lines" was wrong, and the way it was wrong is the interesting
part** (corrected 2026-08-07, when it was built — `training.md` §15). The hook gives
*correctness* for a heterogeneous budget and none of the *saving*: `simulate` hands all
`B` staged leaves to the encoder every iteration whether they are active or not, so a
batch holding mixed budgets pays the largest budget in it. Filling `budget` and leaving
the loop at `range(n)` produces a perfectly correct tree of 64 simulations at the price
of `n`, and every counter in §15 says it worked. Two further lines were needed —
`self_play_move` fills `budget` and iterates the same number — and one design decision:
**the budget is tied across the batch**, since sorting the budgets so the active set is
a contiguous prefix buys nothing either (encoder throughput per board is flat from 4096
boards down to 128, measured 2026-08-07). `sims` may only go *down* from `config.n`,
which sizes the node pool and the path arrays.

**The fast search also drops the §6.1 root noise**, which is a `root_init(noise=False)`
argument rather than a seam: KataGo disables Dirichlet on turns nobody records, to
maximise strength on a move that is played for real.

**The record schema carries `weight_gen` and a reserved value field** so that a
bootstrapped or mixed value target does not force a buffer migration.

Not reachable through a seam, and each one is a real change: WDL values (a head in
[spec §7.4](spec.md#74-heads) and a wider record), a moves-left head (same), tree
reuse across moves (node lifetime, so a free list instead of a bump pointer), and
transposition merging (invariant 3 and the visit accounting).

---

## 12. The reference implementation and how v0 is validated

The order of work is the reference first, then the differential harness, then the
kernels. This is the same shape that made A1 land, and the roadmap notes that C1
lacking an oracle is why its estimate cannot be converted.

**`brokefish/search/torch_impl.py`** implements §6 over
`brokefish/env/torch_impl.py`, with the arrays of §4.2 as torch tensors. **Done
2026-07-30**, `tests/test_search.py`, 22 checks.

It is batched over games and looped over depth and simulations: every step is one
torch op with a leading `B` axis, and the only Python loops are the two the kernel
also runs serially. A per-game Python loop would transcribe the pseudocode more
literally and would take a day per move at `B = 4096`.

Both the environment and the network are injected, so the same search code runs
over `env/torch_impl.py` or `env/cuda_impl.py` and against the model of
`nn/model.py` or a fused encoder. All four combinations produce the same tree on
the positions tested, which is a free differential check on A1 and B2 as well as
on this. Measured 2026-07-30 with both fused and the invariant checks off, one
move at `n = 32`: **27.4k evals/s at `B = 256` and 36.8k at `B = 1024`**.

⚠️ That is not Gate 1a and must not be read as it. It is a Python reference with a
host synchronisation per descent level, measured at `n = 32` where trees are
shallow, and it says nothing about `n = 800`. What it does say is that C2 can
develop against this rather than waiting for the kernel.

Three departures in structure, none in values, and each is marked in the source
where it happens. The legality mask is carried from the terminal test into
expansion in a local variable rather than recomputed, since §6.2's warning
describes a kernel boundary this module does not have. `edge_move` and `edge_N`
are `int16` rather than `u16`, and `path_len` `int32`, because torch has no
unsigned arithmetic worth using. And the repetition count is computed by the
search rather than passed through the engine's ring, per §6.2 step 3.

**The differential test** runs both implementations on the same positions with the
same seed and asserts equality of the whole tree, node for node and edge for edge:
`node_board`, `node_hash`, `node_flags`, `edge_move`, `edge_N`, and `edge_Q` to a
stated tolerance. `edge_prior` and `edge_Q` are the only floating-point quantities
and both derive from the encoder, which `tests/test_b2.py` already pins.

Three things make exact tree equality achievable rather than aspirational: the
canonical edge order of §6.4, tie-breaking by lowest edge index in §6.6 and §6.7,
and the shared Dirichlet construction of §6.1. Any of the three left loose turns the
comparison into a distributional one, which is much weaker.

The third was open until the kernel landed and is now closed, by the decision of
2026-07-30 that the root noise is drawn in torch and handed to the kernel as a
`[B, E]` tensor rather than generated on device. Marsaglia-Tsang with the
`alpha < 1` boost needs a normal variate and therefore a Box-Muller of its own,
which is forty lines of hard-to-test device code for one launch out of 3202 per
move; drawing it in torch means the CUDA search inherits §6.1 unchanged and the two
implementations consume one stream from one seed. The differential test therefore
runs at the real `eps = 0.25` rather than needing `eps = 0`, and device generation
stays a named seam that nothing depends on.

**Independent checks** the reference cannot provide, because it is the same
algorithm. `tests/test_search.py` is these, and it comes at the search from five
sides rather than one, because no single one of them is an oracle:

*Component oracles*, where a published formula can be transcribed twice and
compared. §6.6 against a scalar Python transcription of AlphaZero's `ucb_score`
over 1200 random tree states, which is the strongest check in the file: a wrong
`sqrt`, a multiplied `pb_c_init`, a parent count taken over invalid edges or a
tie broken the other way fails here and nowhere else. And §6.1's noise against the
moments of `Dir(alpha)`, whose marginal is `Beta(alpha, (k-1) alpha)`.

*Exact arithmetic on hand-built state*, where equalities replace tolerances. The
parity of §6.5 at path lengths 1 to 5, and the running mean over repeated visits.

⚠️ At `L = 2` the correct parity and the inverted `d % 2` agree on both levels, so
a two-level test cannot distinguish them. `L = 3` is where they become opposites.
An implementation that tests the parity only at depth 2 has not tested it.

*Forced descents*, which pin one real ply against the engine with no dependence on
the exploration schedule. Every root edge of four positions visited exactly once in
one batch, asserting `edge_Q == 1 - node_value(child)` and, for terminal children,
`node_value == (result + 1) / 2` against the engine replaying the move rather than
against what the search stored. Plus a mate at path length 2, where the two levels
must disagree and a draw would prove nothing, since 0.5 is its own complement.

*A constant evaluator*, which makes the tree a function of the rules alone. With
uniform priors and one value, every backed-up number must be exactly 0.5, and
identical positions must produce byte-identical trees, which is where a dependence
on row index or an uninitialised read shows up.

*Properties of chess*, which no reimplementation of the algorithm can supply: a
forced mate found, invariants 1 to 9, `sum_e edge_N[0][e] == n` after every move,
the truncation counter at zero where §4.3 says it should be, and the terminal-code
histogram of the forced sweep asserted non-empty for checkmate, stalemate and
insufficient material, so that a vacuous pass is impossible.

⚠️ **Whether the search *reaches* a node is a property of the exploration
schedule, not of correctness.** With first-play urgency at 0 an unvisited edge is
scored as a loss, so a mate whose prior is 0.017 is not reached until
`pb_c * P * sqrt(N_v)` clears the visited edges' `Q`. The forced-mate check needs
`n = 800` and `eps = 0` to pass with an untrained network, and the same position
mirrored and colour-swapped fails at `n = 800` because its prior is 0.013 instead.
Any test that waits for the search to find something is measuring that schedule,
which is why the checks above force the descent instead.

⚠️ **One check this section used to propose is not implementable.** "A symmetric
position where the visit distribution must be symmetric under the board's
symmetry, with the network fixed" does not hold: the network has a per-square
embedding and no reflection equivariance, so a mirrored position gives different
priors and different values, and the visits do not mirror. The constant evaluator
above buys the same sensitivity from a property the implementation does have.

**Both sides of the interface are swapped as part of the suite.** The environment
and the network are injected, so the reference runs over `env/torch_impl.py` or
`env/cuda_impl.py` and against `nn/model.py` or a fused encoder. All four produce
the same tree, which checks A1 and B2 against each other as a side effect.

### 12.1 What the suite is measured to catch

A passing suite says nothing until something has tried to defeat it, so
`tests/test_mutation_search.py` edits the reference one plausible mistake at a
time and runs the whole suite against each variant, in the same shape
`test_mutation_mask.py` already uses for the encoder. Every mutation is either a
documented ⚠️ in this document or something that actually went wrong while the
reference was written. Measured 2026-07-30, **15 of 16 killed**:

| mutation | tests that kill it |
|---|---|
| parity `(L-d)%2` to `d%2` | 25 |
| parity: never flip | 6 |
| `pb_c_init` multiplied rather than added | 6 |
| `N_v` in place of `sqrt(N_v)` | 2 |
| PUCT denominator `N` rather than `N+1` | 4 |
| first-play urgency 0 to 0.5 | 2 |
| terminal value `(result+1)/2` to `(1-result)/2` | 3 |
| value head `(v+1)/2` to `(1-v)/2` | 2 |
| truncation keeps the lowest priors | 1 |
| repetition ignores the irreversible cut | 2 |
| repetition drops the game ring | 2 |
| repetition drops the in-tree half | 3 |
| promotion order reversed | 4 |
| `Gamma(alpha<1)` without the `U^(1/alpha)` boost | 1 |
| argmax before `tau` rather than sampling | 1 |
| **tie-break by `torch.argmax`** | **0, and expected** |

The survivor is the control. `torch.argmax` returns the lowest index on both CPU
and CUDA, so the mutation is behaviourally equivalent and the harness asserts that
it survives; a suite that killed it would be reacting to an implementation detail
torch does not promise. `_lowest_argmax` stays because the tie-break is a contract
the kernel has to match, and `test_ties_go_to_the_lowest_edge_index` pins the rule
so a change of *rule* is caught even though a change of *backend* is not.

Every mutation is killed by at least one test written *for* it, which was not true
on the first sweep: the value-range mutation was caught only by
`test_threefold_inside_the_tree_ends_the_line`, a canary that fails because the
search path changes rather than because it is looking at the value convention.

⚠️ **A constant evaluator at value 0 cannot see the value convention at all**,
since `(0+1)/2` and the mistaken `(1-0)/2` are both 0.5 and the whole tree agrees
under either. 0.5 is the smallest value that separates them, and
`test_constant_evaluator_maps_the_value_range` uses it. A single-test kill is worth
checking for this shape of blindness before trusting it. §12.2 is what thickens it.

### 12.2 The oracle

`tests/oracle.py`, **done 2026-07-30**: a second MCTS written from AGZ's Methods, one
game at a time, in objects and Python lists. `tests/test_oracle.py` runs both on the
same positions with the same numbers and compares the two trees node for node and edge
for edge. This is the perft analogue for the search, and the only check here that
looks at the *trajectory* rather than at one piece of it.

It is built four ways round from the reference on purpose, so a shared mistake cannot
hide:

* **values per node, flipped at read time**, where the reference stores per edge and
  flips at write time. §6.5 claims those are equivalent; the comparison asserts
  `edge_Q[v][e] == 1 - child.value()` at every edge, which is what checks it
* **the repetition window walked up the parent links**, not read from a path array
* **the path as a list of node references**, not an index array
* **node indices from a counter**, and node `i` of one has to be node `i` of the
  other, which pins the allocation order as well as the contents

The evaluator is shared and is integer arithmetic: a splitmix64 mix of the board
words, the control word and `rep`, scaled so every float it produces is a small
integer over a power of two and therefore exact in fp32 and fp64 alike. The real
network cannot serve here, since it is called on a batch of `B` in the reference and on
one position in the oracle, and an fp32 reduction need not give the same last bit at
two batch sizes. It reads `rep`, which is what makes §7's count testable end to end: a
wrong repetition count changes a node's logits and the trees diverge below it.

**What agreement is worth, measured rather than asserted.** Two trees matching says
nothing unless a mismatch would also have meant something, and it would not if the two
implementations' scores were closer together than their arithmetic error. So every run
measures both sides of that. At 64 simulations over seven positions: priors and node
values come out **bit-identical**, `Q` disagrees by at most **1e-7** (an fp32 running
mean against an fp64 mean), and the closest selection is **3.4e-6** apart, 34 times the
error. At 512 simulations the closest margin falls to **9e-8** and **4 of 15,014**
selections land inside the floor, so the deep run reports that count instead of
demanding zero; two priors in about 15,000 edges also land on opposite sides of an fp16
rounding boundary, one ULP apart. Those are the honest limits of the comparison.

The seven positions are one per behaviour: startpos, black to move, a mate available,
every child terminal, a promotion available, an endgame that repeats inside the tree,
and a midgame with a large branching factor. Truncation at `E = 6`, root noise with a
fixed `eta`, and both halves of §7's window each get their own run.

⚠️ **The oracle checks implementation faithfulness, not design choices.** Where both
follow the same reading of AGZ, agreement proves the reading was implemented twice and
not that it was the right reading. §3.1a's `sqrt(sum_b N(s,b))` is the case in point:
both take the lowest-index edge on the first descent below a new node, so the
comparison passes and says nothing about whether that is what AlphaZero did.

⚠️ **The comparison itself must be able to fail.**
`test_the_comparison_is_not_vacuous` perturbs each of the thirteen fields it claims to
check, one at a time, plus the node count, and requires every perturbation to be
reported.

### 12.3 The kernels against the reference

`tests/test_search_cuda.py`, **done 2026-07-30**, 19 checks. Both implementations
run the same batch from the same seed and every array of §4.2 is compared **after
every simulation**, so a divergence names the simulation that caused it rather than
the four hundred that followed. `brokefish/search/cuda_impl.py` is a subclass of
the reference and shares its tensors, which is what makes that cheap.

**What comes out bit-identical, and what does not.** Every integer field, `edge_Q`
and `node_value` agree exactly, at every simulation of every run below, including
one move at `n = 800` over 64 games and 462,720 selections. `edge_prior` does not
always: the kernel's softmax denominator is a warp reduction over at most 64
surviving candidates and the reference's is a torch reduction over the 8192-wide
masked candidate row, so the two land on opposite sides of an fp16 rounding
boundary on roughly one edge in a hundred.

That difference is one fp16 ULP and it is **not** provably too small to change a
selection. Every run therefore measures both sides of the question, the same way
§12.2 does: the closest gap between the best and the second-best PUCT score at any
selection, and the largest score perturbation the prior rounding could have caused.

| run | selections | closest non-tie | rounding worth |
|---|---|---|---|
| start position, `eps = 0` | 6 008 | 5.1e-5 | 0 |
| root noise, `eps = 0.25` | 8 504 | 2.7e-5 | 0 |
| three moves through the temperature switch | 12 064 | 1.5e-5 | 0 |
| 218 legal moves, truncating | 384 | 1.3e-3 | 0 |
| random positions, 189 promotion edges | 48 696 | 5.3e-6 | 1.9e-5 |
| a threefold inside the tree | 6 176 | 8.6e-6 | 5.6e-4 |
| endgames, 300 random plies | 84 892 | 1.1e-6 | 3.0e-4 |
| `n = 800`, `B = 64` | 462 720 | 6.0e-8 | 2.5e-4 |

Where the last column is zero the agreement is guaranteed and where it exceeds the
margin the agreement is empirical: the rounding *could* have changed a decision and
over those runs it did not. Saying so is the point of the table.

**One test has a hard guarantee.** With constant logits every prior is an exact
`1/k` on both sides, because the kernel's denominator is a sum of `k` ones and so is
the reference's, and an integer sum in fp32 does not care about reduction order.
That comparison is exact, with no tolerance anywhere in it. It runs on **pawnless
positions**, since a promotion edge picks up `log_softmax` of four equal numbers,
which is `-log 4` and not zero. It is also the hardest exercise of §6.6's tie-break
available, since equal priors make every score at a fresh node tie: 4 196 exact ties
in 8 516 selections.

⚠️ **The root's priors get twice the tolerance, and it is not a fudge.** They are
quantised to fp16 twice: the kernel writes them, then §6.1's noise reads them back,
mixes `0.75 p + 0.25 eta` in fp32 and writes fp16 again. A one-ULP difference in `p`
survives the 0.75 and picks up the output's own half-ULP rounding, so the two can
land two fp16 steps apart. Interior edges go through fp16 once and stay inside one.
Measured at exactly 2.00 ULP, at the root, on about one edge in 10⁵ with the noise
on, and it only appears at batch sizes large enough to see a 10⁻⁵ event.

⚠️ **The batch axis needs its own test and did not have one until it found this.**
Everything else here runs at `B <= 64`, because the reference costs a torch launch
per descent level and the comparison a synchronisation per simulation, which leaves
untested the axis where an index that fits in 32 bits at `B = 8` stops fitting, a
grid is sized wrong, or blocks race. `test_agrees_at_a_self_play_batch` runs
`B = 1024` for two moves and is the only thing in the file that saw the paragraph
above.

⚠️ **A tie is not a rounding risk and the margin excludes it.** At a freshly created
node `sqrt(N_v)` is zero, so every score is `0 * prior + 0`, exactly zero whatever
the prior rounded to, and the tie-break decides it identically on both sides by
construction. That is also why ties are so common: one per node created, about a
quarter of all selections.

Coverage the start position cannot reach gets its own run: the fifty-move boundary
at clock 99, a threefold whose hash is already twice in the game ring, checkmate and
insufficient material 300 random plies in, 218 legal moves for the truncation of
§4.3, 189 promotion edges for §6.4's four-per-move fan-out, and a 45-ply path so
that §7's scan and §6.5's backup loop more than once per lane.

Three guards, because the rest of the file would pass without them.
`test_the_comparison_is_not_vacuous` perturbs each of the 21 fields it claims to
check and requires every perturbation to be reported.
`test_a_rebound_tree_tensor_is_caught` covers the one bug this harness found:
`env.push_history` is a pure function and returns a *new* ring, so a dict of tensors
built once in the constructor points at the ring that stopped being updated after
move 0, and the symptom is a repetition count that is too low three moves and eight
thousand simulations later. And `test_every_tree_field_reaches_the_kernel` checks
that the name-keyed tree the binding reads covers every array of §4.2.

**`csrc/tests/tselect.cu`** pins the two warp reductions on their own, with no
Python and no torch, against a host reference in double written from §6.4 and §6.6
rather than copied from the kernel. The tree comparison is the stronger check and
cannot say *which* reduction was wrong; this one can, and it reaches inputs a real
position would take years to produce. 20 000 random selections across four regimes
with 9 587 exact ties and none inside the fp32 floor, six all-tied nodes where the
lowest index has to win as an equality, and 400 enumerations of which 199 truncate
and 80 have every logit identical, so the tie-break inside the radix select is
exercised rather than assumed.

---

## 13. Not frozen

| open question | state |
|---|---|
| `n`, simulations per move | 800 in v0, AlphaZero's number, chosen so convergence is not in question. [the cost arithmetic](../ledger/perf.md#the-search-in-the-loop) makes it the most expensive number in the project and the sweep downward is where the throughput work starts |
| `B`, games in flight | 4096 in v0, matching AlphaZero's SGD batch per §4.4, since throughput is flat in `B` and the tree fits |
| value target | final outcome in v0. [spec §11](spec.md#11-the-rl-layer) owns the choice; §10 is shaped not to force a migration |
| game start positions | startpos in v0, with Dirichlet and temperature as the only diversity, as in AlphaZero. Randomised openings are allowed by the training boundary and are a change to `select_and_advance` |
| whether to revisit Gumbel | deferred, not rejected. [the cost arithmetic](../ledger/perf.md#the-search-in-the-loop) is the argument for revisiting it, and §11 says it costs two functions |
| `sqrt(sum_b N(s,b))` or `sqrt(node visits)` | AGZ's formula in v0, which makes the first descent below every new node ignore the policy and take the lowest-index edge (§3.1a). The released pseudocode's form removes that. One line either way, and worth an A/B once Elo can be measured, because AGZ's form spends about 2 % of all selections on a systematically chosen move |
| tree reuse | absent in v0 and present in AlphaZero (§3.6). Worth some fraction of `n` for free, which is the most expensive number here, and it costs a free list |
| resignation | absent in v0 and present in AlphaZero (§3.7). A cost mechanism with a self-calibrating threshold, which C2 should have before any long run |
| virtual loss | `search.md` §1.1 files it as an addition, and [the fidelity audit](../journal/2026-07-30-fidelity.md) §2.1(a) argues it is a deviation: AGZ's search ran threaded with virtual loss and AZ defers to AGZ, so v0's tree is slightly more concentrated than AlphaZero's at the same `n`. Ten minutes with the AGZ Methods settles it |
| `tau_plies` in plies or moves | 30 plies in v0. AGZ says "the first 30 moves", which is a ply in Go and a pair in chess. [the fidelity audit](../journal/2026-07-30-fidelity.md) §2.1(c) |
| fp16 priors under a trained policy | §4.2 stores priors as fp16, where a prior below 6e-8 flushes to zero and its exploration term is zero forever. Harmless at today's flat policy, unmeasured at a sharp one, and §15 has no counter for it. [the fidelity audit](../journal/2026-07-30-fidelity.md) §2.2 |
| how far the terminal collapse should reach | depth 1 in §6.6a: the root's proved wins come from §6.1a's exhaustive sweep, and below the root a win is only found where ordinary exploration happens to land on it. Measured on `t9h-n128-sweep`, **~0.25 terminal children are discovered per search of 128 simulations** below the root, so there is little raw material to propagate at this budget and the case for a full MCTS-Solver is a bet on `n` going *up*. Extending it needs a solved-node state, the DRAW logic §6.6a deliberately omits, and `END_IN_PLY` to prefer the shorter mate |
| the game-length cap | absent in v0. AZ's Domain Knowledge item 5 terminates chess games "exceeding a maximum number of steps (determined by typical game length)" and scores them drawn; the pseudocode uses 512 plies. Our rules end games only as chess does, which is stricter and can produce longer games |

---
## 14. Cost arithmetic

Moved to [`perf.md`](../ledger/perf.md#the-search-in-the-loop) on 2026-07-31: the
prediction of what the tree would cost, and the Gate 1a measurement that settled it,
are numbers and belong in the ledger. What §13 needs from them is one sentence —
**`n` is the most expensive number in the project**, linearly, and the sweep downward
is where the throughput work starts.


## 15. Instrumentation

Every fixed size in §4 is a bet that a distribution stays where it was measured, and
those distributions depend on the policy, which changes for the whole run. A bet that
is not monitored becomes an assumption. The search therefore returns a counter block
per self-play phase, accumulated on device with `atomicAdd` into a small struct and
read once per generation, so nothing here costs a host synchronisation.

### 15.1 The fixed sizes, and whether they still hold

| statistic | why | what it means if it moves |
|---|---|---|
| `max_edges` over all expanded nodes | the `E = 96` bet of §4.3 | a trained policy steers into different position types, and open middlegames carry more legal moves than the random playouts `E` was measured on |
| `n_truncated` nodes, and the sum of prior mass discarded | the cost of the bet, not just its frequency | the mass matters more than the count: truncating 66 moves whose tail holds 0.1 % of the prior is harmless, truncating one that holds 20 % is not |
| `max_depth`, plus p50 and p99 | descent is the one irreducibly serial walk (§4.2) and its latency is proportional to depth | a sharpening policy concentrates visits and deepens trees, so this grows over a run and is the term that would invalidate [the tree-cost prediction](../ledger/perf.md#how-much-the-tree-is-likely-to-cost) |
| `max_nodes_used` and the mean pool fill | `Nmax = n + 1` is exact, so the interesting quantity is the shortfall | a low fill means many descents ended at terminal nodes, which is wasted encoder batch (§6.3) |
| `n_terminal_descents` | the waste §6.3 accepts deliberately | rising means the deliberate waste stopped being small, and compaction becomes worth its complexity |
| `n_empty_mask_expansions` | invariant 5 | must be exactly zero. Anything else is a movegen or masking regression, not a statistic |

### 15.2 Whether the search is doing anything

| statistic | why |
|---|---|
| `max_N / n` at the root, and the root visit entropy | the direct measure of how peaked the policy target is, and the concrete form of the low-`n` problem in [the low-`n` policy target](../ledger/perf.md#the-low-n-policy-target) |
| number of root edges with `N > 0` | at small `n` the root cannot even cover its own edges; this is the number that says so |
| fraction of moves where `argmax N` differs from `argmax prior` | search that never disagrees with the policy is search that is not earning its cost, and this is the cheapest signal that `n` is too small or `cpuct` is wrong |
| root `Q` of the chosen move against the eventual game result | value calibration. Cheap here, and C3 needs it anyway |

### 15.3 Numerical and rules health

| statistic | why |
|---|---|
| **max abs post-scale attention logit** | the single quantity `CLAUDE.md` names for the fp16 accumulation ceiling, at 3.1 today against a limit near 12× weight growth. Overflow is loud rather than silent, and this is the early warning |
| fraction of `\|value\|` above 0.99 | a saturated `tanh` stops producing gradient and makes `edge_Q` unable to order moves |
| terminal code histogram at game end | early networks end nearly every game by repetition or the fifty-move rule, and watching `terminal_threefold` and `terminal_fifty_move` fall is the earliest sign that anything is being learned. **Emitted by name, not by index** — `terminal_checkmate`, `terminal_threefold`, … from [spec §4.3](spec.md#43-terminal)'s table via `env.TERMINAL_NAMES`, because `terminal_codes/4` in a dashboard tells the reader nothing (changed 2026-07-31) |
| game length distribution | the same signal, and it is what converts positions per second into games per second for the cost accounting |

### 15.4 Cost

Evaluations per second, moves per second, wall clock and energy per generation, and
the cumulative euro counter. C2 owns the counter; the search owns the throughput
terms that feed it, and Gate 1a is the first read.

---

## Changelog

**2026-07-30, the kernels land and Gate 1a is measured.** `csrc/search.cuh`,
`csrc/search.cu` and `brokefish/search/cuda_impl.py` implement §9's four kernels;
§12.3 is the differential harness and [the Gate 1a measurement](../ledger/perf.md#what-the-tree-actually-costs-measured-2026-07-30) the measurement. Trees agree with the
reference field for field after every simulation, including one move at `n = 800`
over 462,720 selections.

Four things §9 had wrong or open, and each is now decided in the section: it is
four launches per simulation and not three; `select_and_advance` stays in torch,
since it runs once per move and needs a `multinomial`; the root's Dirichlet is drawn
in torch and handed over as a tensor, which is what closes §12's open caveat and
lets the differential test run at the real `eps`; and `descent` stages the leaf into
a contiguous buffer for the encoder, which is also where `rep` is checkable.

The tree costs less than [the tree-cost prediction](../ledger/perf.md#how-much-the-tree-is-likely-to-cost) guessed and more than nothing. [the Gate 1a measurement](../ledger/perf.md#what-the-tree-actually-costs-measured-2026-07-30) has the numbers
and where the estimate was wrong.

**2026-07-30, the oracle lands, and the papers get read properly.**
`tests/oracle.py` and `tests/test_oracle.py`: a second MCTS written from AGZ's
Methods, four ways round from the reference, compared tree for tree. §12.2 has what
it is and what its precision limits are. The trees agree exactly on seven positions
at 64 and at 512 simulations.

Reading the papers instead of the released pseudocode changed six things in this
document. §1.1 now names which artefact is the authority for what, because the three
of them disagree: AZ gives no PUCT formula at all and defers to AGZ, and `c_puct`'s
value is published nowhere, so the logarithmic form and the numbers 19652 and 1.25
rest on an unrefereed file. §3.1a lists the three places the pseudocode contradicts
AGZ, all of them silent: an un-normalised Dirichlet, a visit count one too large at
every interior node, and a value read that maximises the opponent's outcome. §3.6 and
§3.7 move tree reuse and resignation from "refinements v0 has not taken" to
"deviations from AlphaZero", since AGZ has both and AZ does not list either among its
differences. §13 gains three open questions, including the one §3.1a exposes: under
AGZ's formula the first descent below every new node ignores the policy and takes the
lowest-index edge, which is about 2 % of all selections at `n = 800`.

The reference needed no correction. Every formula in §6 survived the comparison
against the papers and against the oracle.

**2026-07-30, the reference lands.** `brokefish/search/torch_impl.py` and
`tests/test_search.py`, 35 checks, with 15 of 16 mutations killed by
`tests/test_mutation_search.py`; §12 has what they are and §12.1 what they catch. Six things the
implementation forced, all of them small and all of them in this document now:
`node_flags` bit 4 carries irreversibility, which §7's walk needs and §4.2 did not
provide; `path_len` is `i16`, since `u8` cannot hold a depth of 800; invariant 7's
"equals" was off by one, and the `+ 1` is the simulation that created the node;
§6.4 step 3 fixes the truncation ordering, which the differential test needs
pinned; §6.2 step 3 splits the repetition count from the terminal call, because
spec §6.3's ring is `[N, 100]` and the tree half does not fit in it; and §6.1's
Dirichlet stream is not yet reproducible across implementations, which §12 records.

The forced-mate check of §12 wanted `n = 800` and `eps = 0` to pass. With
first-play urgency at 0 an unvisited edge is scored as a loss, so a mate whose
prior is 0.017 under an untrained network is not reached until
`pb_c * P * sqrt(N_v)` clears the visited edges' `Q`, which happens between 400
and 800 simulations. That is AlphaZero's behaviour rather than a defect, and it is
[the low-`n` policy target](../ledger/perf.md#the-low-n-policy-target)'s low-`n` problem in its most concrete form.

**draft, 2026-07-30.** First version. `n = 800` and `B = 4096` from §4.4, chosen
convergence-first after the encoder was measured flat in batch size. `Dmax` is the
exact bound `n` rather than a cap, so v0 truncates no path. §15 is the counter block,
which exists because every fixed size in §4 is a bet on a policy-dependent
distribution. Supersedes the
"Search is Gumbel MCTS" statement in `index.md` and `CLAUDE.md`, which record
the intent formed during sizing and not the v0 design.
