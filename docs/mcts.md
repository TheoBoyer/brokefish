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
[`due_diligence.md`](due_diligence.md) covers where each comes from and
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
`[-1, 1]` and reuses 1.25 is running a search with twice the intended exploration
and will not reproduce anything.

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
`n` is the most expensive number in this project (§14). What it costs to implement is
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

⚠️ §14 argues that none of this is likely to matter for throughput, because the
network dominates a node by three orders of magnitude. The reason to keep the path
array anyway is that a parallel scatter over an array is simpler code than a pointer
chase, and Gate 1a will say whether the argument holds.

There is no stored legality mask. Expansion converts the mask into edges and the
edges carry it from then on.

### 4.3 Why `E = 64`

Measured over the 10 000 positions of `data/cuda_testset`, which come from random
legal playouts, counting a promotion as four edges:

| statistic | edges |
|---|---|
| mean | 31.2 |
| median | 32 |
| p99 | 52 |
| p99.9 | 59 |
| max | 65 |
| fraction above 64 | 0.01 % |

Three reasons for 64 rather than 48 or 128. The measured tail puts truncation at
one position in 10⁴. A warp has 32 lanes, so 64 edges is exactly two per lane in
the selection scan with no predicate on the second pass. And the edge arrays are 90 %
of a node, so 128 would nearly double the tree, from 2.81 GB to 5.34 GB at the
sizing of §4.4, to cover a range the histogram says is empty.

⚠️ The distribution above was measured on random playouts and the policy that
generates positions changes for the whole run, so `E = 64` is a bet on a moving
distribution. §15.1 is the monitoring that keeps it a bet rather than an assumption.

⚠️ Playouts under-represent open middlegames and the constructed maximum in chess
is 218, so treat 0.01 % as a floor. Truncation is therefore **counted at runtime**
and the count is reported with every self-play phase. If it rises, §11 has the
evolution.

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
throughput number in [`perf.md`](perf.md) was taken at 16384 and the table says they
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
([`roadmap.md`](roadmap.md)), and its peak footprint at a gradient batch of 4096 has
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
Q(e)    = edge_N[b][v][e] == 0 ? 0 : edge_Q[b][v][e]
```

`pb_c_init` is **added** to the logarithm rather than multiplied by it. Ties are
broken by the lowest edge index. An unvisited edge scores `Q = 0`, which is
AlphaZero's first-play urgency and is a loss from the mover's point of view in the
`[0, 1]` convention; §11 has the alternatives.

### 6.7 `select_and_advance`

At the root, form `pi(a) = N(a) / n` and pick the move:

* `game_ply[b] < tau_plies`: sample from `pi`.
* otherwise: `argmax N(a)`, ties by lowest edge index.

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
([`roadmap.md`](roadmap.md#measuring-strength)). A divergence that helps during
training has to be unlearned or specially handled at evaluation time, and a
train-test mismatch on a rule is a hard bug to attribute.

It is an imported opinion about how to search, so it sits on the wrong side of the
training boundary. The rules are allowed; judgements about them are what has to be
learned.

⚠️ Not a rules divergence, and worth distinguishing: `irreversible` omits
python-chess's en passant clause ([`env.md`](env.md#three-places-it-departs-from-python-chess-all-deliberate)).
That only ever makes the repetition window longer, never shorter, so no repetition
can be missed and no rule is bent. It is a divergence from another implementation,
not from chess.

---

## 9. Kernel decomposition

Per simulation, three launches, and per move `3n + 2`:

| kernel | grid | what |
|---|---|---|
| `root_init` | `B / W` blocks | once per move, then reuses `evaluate` and `expand` |
| `descent` | `B / W` blocks of `W * 32` threads | select, `step_full`, repetition, `terminal`, allocate |
| `evaluate` | one CTA per board | `forward_full`, the existing encoder kernel |
| `expand` | `B / W` blocks | `movegen`, edge enumeration, prior softmax |
| `backup` | `B / W` blocks | the walk of §6.5 |
| `select_and_advance` | `B / W` blocks | once per move |

`W` is warps per block, 8 in the engine kernels.

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
25.5 s move by §14.1, so it buys nothing measurable, and it costs the property that
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
policy      [min(E, n)] (u16 move, f16 prob)
policy_len  u8
value       f32         filled in when the game ends
weight_gen  u16         the network generation that produced the search
```

The policy target holds at most `min(E, n)` nonzero entries: at most `n` because the
visit counts sum to `n`, and at most `E` because the root has at most `E` edges. At
`n >= 64` the binding limit is `E = 64`, so a record is
64 + 2 + 1 + 256 + 1 + 4 + 2 ≈ 330 B independently of `n`, and a 2M-position window
is 660 MB in host RAM.

`value` is the final game outcome from the point of view of the side to move in
`board`, in `[-1, 1]`. It is written when the game terminates, so a record is
incomplete until then and the buffer needs a per-game index of its own pending
records. The spare fields for a bootstrapped or mixed value target are `weight_gen`
plus one reserved `f32`; [spec §11](spec.md#11-not-frozen) leaves that choice open
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
returning immediately for a game whose budget is exhausted. In v0 every entry is
`n`. This is what playout cap randomisation needs, and it is five lines.

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

⚠️ The third is the one that is not yet closed. The reference builds its Dirichlet
noise by §6.1's construction, but torch draws one global sequence where a kernel
gives each lane its own cuRAND state, so the same seed gives different noise. The
comparison therefore has to drive both implementations from the same numbers, which
is why the noise is a method on the reference rather than inline code. A
counter-based generator on both sides would remove the caveat and is the cleaner
fix if the differential test needs the root's noise rather than `eps = 0`.

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

---

## 13. Not frozen

| open question | state |
|---|---|
| `n`, simulations per move | 800 in v0, AlphaZero's number, chosen so convergence is not in question. §14 makes it the most expensive number in the project and the sweep downward is where the throughput work starts |
| `B`, games in flight | 4096 in v0, matching AlphaZero's SGD batch per §4.4, since throughput is flat in `B` and the tree fits |
| value target | final outcome in v0. [spec §11](spec.md#11-not-frozen) owns the choice; §10 is shaped not to force a migration |
| game start positions | startpos in v0, with Dirichlet and temperature as the only diversity, as in AlphaZero. Randomised openings are allowed by the training boundary and are a change to `select_and_advance` |
| whether to revisit Gumbel | deferred, not rejected. §14 is the argument for revisiting it, and §11 says it costs two functions |
| `sqrt(sum_b N(s,b))` or `sqrt(node visits)` | AGZ's formula in v0, which makes the first descent below every new node ignore the policy and take the lowest-index edge (§3.1a). The released pseudocode's form removes that. One line either way, and worth an A/B once Elo can be measured, because AGZ's form spends about 2 % of all selections on a systematically chosen move |
| tree reuse | absent in v0 and present in AlphaZero (§3.6). Worth some fraction of `n` for free, which is the most expensive number here, and it costs a free list |
| resignation | absent in v0 and present in AlphaZero (§3.7). A cost mechanism with a self-calibrating threshold, which C2 should have before any long run |
| the game-length cap | absent in v0. AZ's Domain Knowledge item 5 terminates chess games "exceeding a maximum number of steps (determined by typical game length)" and scores them drawn; the pseudocode uses 512 plies. Our rules end games only as chess does, which is stricter and can produce longer games |

---

## 14. Cost arithmetic

From the measured 64.2k evals/s, ignoring the learner's share and assuming the tree
costs nothing. A move costs `n` evaluations per game, so the rate of real moves is
`64.2e3 / n` across the whole batch:

| `n` | moves/s | 10k games (80 plies) | 128k games | 10M games |
|---|---|---|---|---|
| 32 | 2006 | 6.6 min | 1.4 h | 111 h |
| 128 | 502 | 27 min | 5.7 h | 444 h |
| 400 | 161 | 1.4 h | 17.7 h | 1386 h |
| 800 | 80 | 2.8 h | 35.4 h | 2772 h |

⚠️ Arithmetic from a measurement and not a measurement. It is a floor: the tree's
cost is what Gate 1a exists to measure, and it comes out of these numbers rather than
being added to them.

The `n = 800` column is what makes the ordering in §4.4 workable rather than
reckless. A convergence check does not need 10M games; the first evidence that the
loss is falling and that self-play Elo is rising arrives in the low thousands of
games, which is under three hours. A full run at `n = 800` is out of the question on
this card, and that is the point: the search budget is the first thing the
optimisation work will attack, with a converging baseline to regress against.

The reduction path, in the order it should be tried: bring `n` down and watch Elo per
game, since AlphaZero's 800 was chosen on hardware where evaluations were nearly
free; then Gumbel, whose entire purpose is to make small `n` behave, and which §11
prices at two functions; then playout cap randomisation, which pays the large budget
on only a fraction of moves.

### 14.1 How much the tree is likely to cost

Rough arithmetic, to set expectations for Gate 1a rather than to substitute for it.

A descent step reads one node's edge statistics, 64 × (2 + 2 + 4) = 512 B, and picks
an argmax. Taking 300 cycles for the load-and-reduce and a real depth of 40, one
simulation's descent is about 12k cycles, and a move's 800 simulations about 9.6M
cycles per game. At `B = 4096` over roughly 1150 concurrent warp slots that is about
3.6 waves, so 34M cycles, or **24 ms per move** at 1.4 GHz.

The same move costs 800 encoder launches at 63.9 ms each, which is **51.1 s**.

So the arithmetic puts the tree near 0.05 % of a move, against the 2.2 % the
environment measured. If that survives contact with a profiler, then the block tail of
§9, the path array of §4.2 and the parallel backup of §6.5 are all decisions about
code clarity and none of them is a throughput decision.

⚠️ The 300 cycles is a guess, not a measurement, and the estimate ignores the
repetition scan, the expansion and every launch overhead. Treat it as an order of
magnitude. Gate 1a is what settles it, and the reason to write it down now is that it
argues against optimising any of this before measuring.

### 14.2 The low-`n` policy target

At small `n` the policy target degrades in a specific way worth watching for: with
about 31 edges at the root and `n` visits to spread over them, `N / n` approaches
uniform-with-noise. The reference implementation shows this long before the kernel
exists, which is a reason to look at it there.

---

## 15. Instrumentation

Every fixed size in §4 is a bet that a distribution stays where it was measured, and
those distributions depend on the policy, which changes for the whole run. A bet that
is not monitored becomes an assumption. The search therefore returns a counter block
per self-play phase, accumulated on device with `atomicAdd` into a small struct and
read once per generation, so nothing here costs a host synchronisation.

### 15.1 The fixed sizes, and whether they still hold

| statistic | why | what it means if it moves |
|---|---|---|
| `max_edges` over all expanded nodes | the `E = 64` bet of §4.3 | a trained policy steers into different position types, and open middlegames carry more legal moves than the random playouts `E` was measured on |
| `n_truncated` nodes, and the sum of prior mass discarded | the cost of the bet, not just its frequency | the mass matters more than the count: truncating 66 moves whose tail holds 0.1 % of the prior is harmless, truncating one that holds 20 % is not |
| `max_depth`, plus p50 and p99 | descent is the one irreducibly serial walk (§4.2) and its latency is proportional to depth | a sharpening policy concentrates visits and deepens trees, so this grows over a run and is the term that would invalidate §14.1 |
| `max_nodes_used` and the mean pool fill | `Nmax = n + 1` is exact, so the interesting quantity is the shortfall | a low fill means many descents ended at terminal nodes, which is wasted encoder batch (§6.3) |
| `n_terminal_descents` | the waste §6.3 accepts deliberately | rising means the deliberate waste stopped being small, and compaction becomes worth its complexity |
| `n_empty_mask_expansions` | invariant 5 | must be exactly zero. Anything else is a movegen or masking regression, not a statistic |

### 15.2 Whether the search is doing anything

| statistic | why |
|---|---|
| `max_N / n` at the root, and the root visit entropy | the direct measure of how peaked the policy target is, and the concrete form of the low-`n` problem in §14.2 |
| number of root edges with `N > 0` | at small `n` the root cannot even cover its own edges; this is the number that says so |
| fraction of moves where `argmax N` differs from `argmax prior` | search that never disagrees with the policy is search that is not earning its cost, and this is the cheapest signal that `n` is too small or `cpuct` is wrong |
| root `Q` of the chosen move against the eventual game result | value calibration. Cheap here, and C3 needs it anyway |

### 15.3 Numerical and rules health

| statistic | why |
|---|---|
| **max abs post-scale attention logit** | the single quantity `CLAUDE.md` names for the fp16 accumulation ceiling, at 3.1 today against a limit near 12× weight growth. Overflow is loud rather than silent, and this is the early warning |
| fraction of `\|value\|` above 0.99 | a saturated `tanh` stops producing gradient and makes `edge_Q` unable to order moves |
| terminal code histogram at game end | early networks end nearly every game by repetition or the fifty-move rule, and watching codes 3 and 4 fall is the earliest sign that anything is being learned |
| game length distribution | the same signal, and it is what converts positions per second into games per second for the cost accounting |

### 15.4 Cost

Evaluations per second, moves per second, wall clock and energy per generation, and
the cumulative euro counter. C2 owns the counter; the search owns the throughput
terms that feed it, and Gate 1a is the first read.

---

## Changelog

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
§14.2's low-`n` problem in its most concrete form.

**draft, 2026-07-30.** First version. `n = 800` and `B = 4096` from §4.4, chosen
convergence-first after the encoder was measured flat in batch size. `Dmax` is the
exact bound `n` rather than a cap, so v0 truncates no path. §15 is the counter block,
which exists because every fixed size in §4 is a bet on a policy-dependent
distribution. Supersedes the
"Search is Gumbel MCTS" statement in `docs/index.md` and `CLAUDE.md`, which record
the intent formed during sizing and not the v0 design.
