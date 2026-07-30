# The search debugger

**Status: implemented, 2026-07-30.** Normative for the trace format, the recorder and
the server API. It does not modify [`mcts.md`](mcts.md), which owns the search, or
[`spec.md`](spec.md), which owns the engine and network contracts. Where it needs a
quantity those documents do not name, it derives it and says from what.

Scope: one search, recorded and browsed. A human plays a checkpoint, opens the tree
the checkpoint built, and steps through the simulations that built it.

Out of scope: training monitoring. Scalars over generations go to wandb, which
already exists and is better at it. Nothing in this document produces a time series.

---

## 1. What this is for

The search is the part of the system whose failures are silent. A wrong kernel
crashes or disagrees with the reference; a search that spends 780 of 800 simulations
on a move it was already certain about produces a perfectly valid tree and a bad
game. The counters in `mcts.md` §15 detect that in aggregate. They do not answer the
question a human actually asks, which is why *this* move in *this* position.

Answering it needs three things visible at once: the root's edges with `Q` and `U`
separated, the order in which the simulations arrived, and the leaf each one reached.
None of the three is recoverable from a finished tree, because backup is destructive:
`edge_Q` is a running mean and the sequence that produced it is gone.

So the debugger records the sequence.

## 2. The trace is the artifact

A search is a deterministic function of (position, seed, checkpoint, config), so a
recording of one is complete. The web application is a viewer over a recorded trace
and never drives a live search to render a frame.

Three consequences, and they are the reason for the split.

A trace is a file. It can be kept, attached to a bug, compared against a trace from
another generation, and reopened in six months when the checkpoint that produced it
has been deleted.

Browsing needs no GPU and no torch process. The viewer is static files and a JSON
document.

A pathological search seen once is reproducible instead of gone.

The server retains a live half for the work a trace cannot do: starting a new search,
evaluating an arbitrary position, and playing a game. Every one of those emits a
trace, and the viewer then reads it like any other.

## 3. What is recorded, and what is derived

Recording a full tree snapshot per simulation is 800 copies of a 686 KB tree
(`mcts.md` §4.2), which is not a format, it is a memory dump.

Backup touches only the edges on the current path, adding 1 to `N` and folding the
leaf value into a running mean. Expansion touches only the node being created. So the
per-simulation delta is bounded by the path length plus one node, and replaying the
deltas from an empty tree reconstructs the exact tree state after any prefix of the
simulations.

**Everything derivable from a replayed tree state is derived and not stored.** In
particular the PUCT scores are derived. `Q + U` at node `v` is a pure function of
`edge_N`, `edge_Q`, `edge_prior` and `N_v`, all of which the replay has, so storing
64 floats per level per simulation (about 1.5M numbers at `n = 800`) would be storing
a function of data already present.

⚠️ **The viewer's arithmetic is float64 and the reference's is float32.** A replayed
`edge_Q` agrees with the torch tree to about 1e-7, not exactly, and a derived PUCT
score inherits that. Visit counts are integers and agree exactly. Anywhere the viewer
displays a near-tie between two edges it is displaying two numbers whose order it
computed in a different precision from the search that chose between them, and §7.2
requires it to say so.

## 4. Trace format

Format version 1. One JSON document per search. Field order is not significant;
absent optional fields mean absent, never a default.

### 4.1 Header

```jsonc
{
  "format": 1,
  "created": "2026-07-30T14:22:31Z",     // host clock, informational
  "kind": "search" | "move",             // "move" when part of a played game
  "config": {                            // mcts.md §4.1, verbatim
    "n": 800, "B": 1, "E": 64,
    "pb_c_base": 19652.0, "pb_c_init": 1.25,
    "alpha": 0.3, "eps": 0.25, "tau_plies": 30,
    "seed": 0
  },
  "impl": {"env": "cuda", "encoder": "cuda", "search": "torch"},
  "checkpoint": {
    "path": "runs/pilot/gen_0041.pt",
    "sha256": "…",                       // of the weight file, so a trace names its net
    "weight_gen": 41,
    "params": 6383360
  }
}
```

⚠️ `"search"` is `"torch"` in v1 and the field exists so a future CUDA trace is
distinguishable rather than silently assumed. §10 has why the CUDA search is not
traced.

### 4.2 Nodes

A node is created by exactly one simulation and expanded by the same one, so node
records live inside the simulation that created them rather than in a parallel array.
The root is the exception and sits at the top level.

```jsonc
"root": { …node record, id 0… },
```

A node record:

```jsonc
{
  "id": 0,
  "parent": -1,          // node id, -1 at the root
  "pedge": 0,            // the parent's edge index that reaches here
  "depth": 0,
  "board": [ …32 ints… ],   // spec §2.1 piece words, as unsigned
  "control": 1,             // spec §2.2
  "fen": "rnbq…",           // derived, see §6; present so the viewer needs no engine
  "hash": "0x…",            // spec §6.1, hex because JSON numbers are float64
  "rep": 0,                 // spec §7.2's clamped feature, the network's input
  "rep_raw": 1,             // mcts.md §7's count including this occurrence
  "flags": 16,              // mcts.md §4.2, raw
  "code": 0,                // terminal code, spec §4.3, broken out from flags
  "value": 0.5123,          // node_value, in [0,1], mcts.md §3.5
  "value_source": "net" | "terminal",
  "edges": {                // absent on a terminal node
    "move": [ …ints… ],     // spec §3 label with the promo field, edge order
    "uci":  [ …strings… ],  // derived, §6
    "san":  [ …strings… ],  // derived, §6
    "prior":[ …floats… ]    // edge_prior as stored, fp16 values widened
  },
  "n_legal": 34,            // before §4.3 truncation
  "truncated_mass": 0.0     // prior mass dropped, mcts.md §15.1
}
```

The root additionally carries the noise, because how far Dirichlet moved the priors is
a question the viewer must be able to answer and the mixture is destructive:

```jsonc
"prior_pre_noise": [ …floats… ],
"noise":           [ …floats… ]    // eta, absent when eps == 0
```

⚠️ The root's stored `prior` is quantised to fp16 twice, once by expansion and once
by the noise mixing. It is the number the search used, so it is the number stored;
`prior_pre_noise` and `noise` are provided for display and do not reconstruct it
bit-exactly.

### 4.3 Simulations

```jsonc
"sims": [
  {
    "s": 0,
    "path": [[0, 17], [3, 2]],   // (node, edge) pairs, root first, length L
    "leaf": 12,                  // node id the value came from
    "created": { …node record… } | null,
    "value": 0.4871,             // q at the leaf, in [0,1], before any flip
    "ended": "expanded" | "terminal_child" | "terminal_stored"
  },
  …
]
```

`created` is `null` exactly when `ended` is `"terminal_stored"`, which is a descent
that walked into a node already known to be terminal. `"terminal_child"` created a
node whose value came from the game result rather than the network.

`path` has `L` entries and the leaf sits below the last one, so the leaf is
`edge_child` of `path[L-1]` when a node was created, and equals it otherwise.

### 4.4 The tail

```jsonc
"final_root": {
  "N": [ …ints… ],
  "Q": [ …floats… ]
},
"pi":       [ …floats… ],   // mcts.md §6.7's visit distribution over root edges
"played":   17,             // root edge index actually played
"temperature": "sampled" | "argmax",
"stats": { …SearchStats.snapshot()… }
```

`final_root` exists so the viewer can check its own replay (§5). The full final tree
is not stored; the exact whole-tree check lives in the test suite, where it can be
done in float32 (§8).

### 4.5 Size

Arithmetic from the field widths, not a measurement. At `n = 800` with a mean depth
around 30 and 32 edges per created node: a simulation is roughly 60 numbers of path,
70 of edges, 32 of board, plus the FEN and the move strings, so 2 to 3 KB of JSON.
The whole trace lands near 2 MB, and gzip on the wire takes it well below that.

⚠️ The mean depth is a guess and the trace size is linear in it. If a trained policy
drives deeper trees this grows, and the fallback is dropping `san`/`uci`/`fen` from
node records and deriving them in the viewer, which costs a chess implementation in
JavaScript. Measure before assuming the format holds.

**Measured, 2026-07-30, on the untrained network.** A search from the start position
at `n = 256` writes 340 KB, **1360 B per simulation**, over a tree of 257 nodes whose
deepest path is 10. The promotion position of `tests/test_trace.py` at `n = 200`
writes 181 KB, 928 B per simulation. Both are below the estimate above, which assumed
a mean depth of 30 that an untrained policy does not produce; the number to watch is
the depth, since a trained policy is what makes the trees deep and the paths long.

## 5. Replay

The viewer holds a tree of the shape `mcts.md` §4.2 describes and applies simulations
in order. To show the state after simulation `k`, replay `0..k`.

```
apply(sim):
    if sim.created: insert the node, set edge_child[parent][pedge] = id
    L = len(sim.path)
    for d, (v, e) in enumerate(sim.path):
        q = (L - d) % 2 == 0 ? sim.value : 1 - sim.value
        N[v][e] += 1
        Q[v][e] += (q - Q[v][e]) / N[v][e]
```

This is `mcts.md` §6.5 transcribed. The parity flip is the same one, and the same
warning applies: inverting it produces a viewer that shows the search preferring its
worst moves, and at `L = 2` the wrong parity agrees with the right one, so nothing
shallow catches it.

Scrubbing backwards replays from zero rather than undoing, because a running mean
does not invert stably. At 800 simulations a full replay is microseconds of
JavaScript, so no incremental structure is warranted.

The viewer checks its final state against `final_root`: `N` exactly, `Q` within 1e-5.
A mismatch is displayed as a banner rather than swallowed, since a viewer that
silently disagrees with the search it is displaying is worse than no viewer.

## 6. Moves, positions and python-chess

The engine's action space is `(slot, square)` with a promotion field
(`spec.md` §3), and slot indices are meaningless to a human. The server owns every
translation and the viewer never sees a slot.

`brokefish/env/interop.py` already provides `to_chess_board`, `list_legal_moves` and
`move_to_args`. The debugger needs one addition, `label_to_move(words, label) ->
chess.Move`, which is the single-label form of `list_legal_moves`: source square from
the slot's own word, target from the low bits, promotion type from the field in spec
§3's `N B R Q` order.

⚠️ `interop.py`'s docstring says nothing in the training loop may depend on it. The
debugger is not the training loop and the dependency is correct. The rule stays: the
self-play path never imports python-chess.

FEN comes from `to_chess_board(...).fen()`. Two fields it cannot fill correctly are
worth knowing before they are believed: the fullmove number, which the engine does
not carry, and the halfmove clock at the root of a search, which is the real game's
and is correct, against a node deep in a tree, where it is correct as well because
`step_full` maintains it. The fullmove number is emitted as 1 everywhere and the
viewer must not display it.

Board rendering is `chess.svg.board()` server-side, one request per position, with
the last move and a check highlighted. The viewer overlays 64 transparent rectangles
for click targets. No chess logic runs in the browser.

## 7. Views

Every view reads the replayed state at the current simulation index `k`, which is
global to the page. Changing `k` changes everything at once.

### 7.1 Root table

One row per root edge, sorted by `N` descending, ties by edge index.

| column | source |
|---|---|
| move | `san` |
| `N` | replayed |
| `π` | `N / ΣN` at the current `k` |
| `Q` | replayed, shown in [0,1] with the [-1,1] value in parentheses |
| `P` | `prior` |
| `U` | derived, `pb_c(N_v) · P · √N_v / (1 + N)` |
| `Q + U` | derived, the number §6.6 maximises |
| child | node id, or empty when unexpanded |

`Q` and `U` are separate columns because almost every "why that move" resolves into
one term dominating, and a combined score hides which.

⚠️ An edge with `N = 0` displays `Q` as blank, not as 0. First-play urgency scores it
as 0 (`mcts.md` §6.6), and a blank says "no estimate" where a 0 says "estimated as a
loss". Both readings are used in the literature and confusing them is how people
misdiagnose exploration.

### 7.2 Simulation scrubber

A slider over `1..n` plus step and play controls. At `k` it shows the path as a move
sequence from the root, the leaf board, the leaf value, and which of §4.3's three
endings it was.

The path is displayed with the score that decided each step, recomputed from the
state before the simulation was applied, alongside the runner-up and the margin
between them. When the margin falls below 1e-5 the row is marked, because §3's
precision caveat means the viewer's ordering is not the search's guarantee there.

### 7.3 Node inspector

Click any node in any view. Shows its board, its FEN, its depth, its terminal code,
its `value` and where the value came from, its edges with the same columns as §7.1,
its repetition count, and the list of simulation indices that passed through it.

The simulation list is an inverted index over the paths, built once at load. It is
what makes "this node was abandoned after simulation 200" and "this node absorbed
everything after simulation 340" visible.

### 7.4 Run summary

Not per-`k`. Depth histogram, terminal codes reached, `n_legal` maximum against
`E = 64`, truncated prior mass, node pool fill against `Nmax`, and whether
argmax-visits agrees with argmax-`Q` at the root.

That last one earns its place: `edge_Q` lives in [0,1] and PUCT adds it to the
exploration term, so a `pb_c_init` tuned for a [-1,1] convention explores half as
much as intended (`mcts.md` §3.5). Visits and `Q` disagreeing at the root is the
cheapest visible symptom.

### 7.5 Play

A board, the human's legal moves from `list_legal_moves`, and the engine replying at
the configured `n`. Every engine reply emits a trace and links to it, so a move that
looked wrong is one click from the tree that produced it.

Move input is click-source then click-target, with a promotion picker when the target
needs one. Takeback and "search this position without playing it" are both required,
because the common interaction is noticing a mistake two moves later.

## 8. What must not happen

**Recording must not change the numbers.** `TracingSearch` subclasses the reference,
draws no random numbers of its own, and adds no branch that depends on what it
observed. The test that enforces it compares a traced search against an untraced one
from the same seed and requires every array of §4.2 to be bit-identical, which is the
same standard `tests/test_search_cuda.py` holds the kernels to.

⚠️ **Values only. Speed is not part of the contract.** The reference is not on the
self-play path and nothing gates on its throughput, so the recorder may copy every
tensor it wants, synchronise per simulation, and build derived structures inline. A
recorder that halves the reference's speed is acceptable; one that moves a single bit
of `edge_Q` is not. The constraint that does bind is trace size (§4.5), which is a
property of the file rather than of the recording.

**No external engine evaluation, anywhere in this application.** Not Stockfish, not a
tablebase probe, not an opening name. `index.md`'s boundary forbids engine opinion as
training input, and a debugger is precisely where it would enter instead through the
operator: look at an engine disagreeing, change a hyperparameter, and the curve is
contaminated by a route no test covers. The network's own value and policy are the
only evaluations displayed, which is also all a human needs to answer why the
*network* played a move.

**The package never imports the debugger.** `debugger/` depends on `brokefish`; the
reverse import is a lint failure. FastAPI and uvicorn stay out of
`requirements.lock`'s runtime set.

## 9. Layout

```
brokefish/search/trace.py   TracingSearch, the schema constants, `write_trace`,
                            and `replay` (the Python twin of §5, for the tests)
debugger/server.py          FastAPI: static files, trace listing, board SVG,
                            live search, evaluation, play sessions
debugger/static/            index.html, app.js, style.css. ES modules, no build
                            step, no bundler, no CDN, works offline
debugger/README.md          how to run it
tests/test_trace.py         §8's identity test and §10's schema checks
traces/                     emitted traces, gitignored
```

`trace.py` sits in the package rather than in `debugger/` because it subclasses the
reference search and is tested with the rest of the suite. The web application is the
only part that is not importable from `brokefish`.

### 9.1 Server API

```
GET    /                        the viewer
GET    /api/traces              [{id, created, kind, n, checkpoint, fen, played}]
GET    /api/trace/{id}          the trace document of §4
DELETE /api/trace/{id}          traces are cheap to make and pile up
GET    /api/board.svg           ?fen=&lastmove=&check=&flipped=&size=
POST   /api/search              {fen, n, eps, seed, checkpoint} -> {trace_id, seconds}
POST   /api/eval                {fen, checkpoint, top} -> {value, top: [{san, uci, p}]}
POST   /api/game                {checkpoint, n, human_color, fen} -> {game_id, …state}
GET    /api/game/{id}           the state, for a reload
POST   /api/game/{id}/move      {uci} -> {…state, reply_uci, trace_id}
POST   /api/game/{id}/takeback  -> the state
```

The game state is `{fen, code, ply, legal, human_white, last, traces}`, where `legal`
is the UCI list the engine's own mask produced and `traces` is one trace id per ply,
`null` on the human's. `/api/eval` builds its priors by running §6.4's expansion on a
one-simulation search rather than by reading the policy head a second way, so what it
shows is what a search would start from, promotion split and truncation included.

`POST /api/search` and `/api/game/{id}/move` are synchronous and slow (§11). They
return when the search finishes.

## 10. Torch only, and why

The debugger drives `brokefish/search/torch_impl.py` and not the kernels.

The reference exposes every intermediate the trace needs as a tensor already, so
recording is reading fields between calls, and its speed is nobody's concern (§11).
The kernels are the opposite on both counts: recording from the CUDA search would mean
copying out of device buffers every simulation and adding a code path to the driver of
the one search that Gate 1a measures, for the benefit of a debugger. The split follows
the freedom, with all the instrumentation on the implementation nothing is measured
against.

The two are validated to produce the same tree node for node
(`tests/test_search_cuda.py`, `mcts.md` §12.3), so a trace of the reference is a
trace of what the kernel does, to within the one fp16 ULP §12.3 records on
`edge_prior`.

⚠️ That equivalence is the whole justification and it holds only where the
differential test covers. A CUDA-only bug outside its coverage is invisible here by
construction, so the debugger is not evidence about the kernels.

## 11. Cost

The only cost that matters here is how long a human waits, since the reference search
is not on the self-play path and no gate, benchmark or curve point depends on its
speed. Slowing it down to record a trace is free.

**Measured, 2026-07-30, recorder included: 18 ms per simulation** at `B = 1` on the
4060, from the start position with the untrained network (`n = 256` in 4.62 s,
`n = 200` in 3.25 s). That extrapolates to about **15 s for one `n = 800` move**,
which is a wait but not a job queue, and it puts an interactive game at `n = 128` at
roughly two seconds a move.

The reference runs at 27.4k evals/s at `B = 256` (`mcts.md` §12) and this is 55/s, so
the per-simulation cost at `B = 1` is not that number divided by anything: a `B = 1`
search is bound by per-op launch overhead rather than by the network.

**The recorder itself costs 11 %**, five interleaved order-balanced rounds at
`n = 128`, medians 15.2 ms per simulation untraced against 16.8 traced. Per §8 that
number is not a constraint on anything, and it is recorded so that a later change
that makes it 3× is visible as a change rather than as the way it always was.

The default `n` in the server's forms is 128 for that reason. If a trained policy makes
the trees deeper and the descent longer, the honest response is a lower `n` for play
with the `n` recorded in the trace, which the format already carries.

⚠️ **No optimisation of `torch_impl.py` may be justified by this document.** The
reference exists to be readable and to be the thing the kernels are checked against.
A change that makes it faster and harder to compare against `mcts.md` §6 costs more
than the seconds it saves.

## 12. Validation

Five checks, in `tests/test_trace.py` unless noted.

1. **Identity.** Traced and untraced searches from the same seed produce
   bit-identical trees and identical move records (§8).
2. **Replay.** `replay()` over the trace reproduces the final torch tree: `edge_N`
   exactly at every node, `edge_Q` within 1e-6, over the whole tree rather than just
   the root.
3. **Move round-trip.** Every edge label in every node record converts to a
   `chess.Move` that python-chess calls legal in that node's FEN, and the edge count
   matches python-chess's legal move count for the position when the node was not
   truncated. This validates the whole `(slot, square)` translation layer against the
   external oracle.
4. **Position round-trip.** Applying a node's `san` path from the root in
   python-chess reproduces that node's FEN.
5. **Schema.** An emitted trace validates against §4, including the invariant that
   `created` is `null` exactly when `ended == "terminal_stored"`.

Check 3 is the one that matters most, because a debugger that mislabels moves is a
debugger that lies fluently, and every other check would pass while it did.

Three more landed with the code, in the same file. A replay to `k` has to equal a
search that was only ever run for `k` simulations, which is what says the prefix is a
real state and not just an accumulator. The derived PUCT score of §3 has to pick, at
every `k`, the root edge that simulation `k+1` actually took, which is the one check
that the derive-do-not-store decision does not quietly break the scrubber. And the
truncation case is asserted where it fires rather than waited for: 218 legal moves
against `E = 64`, with the dropped mass recorded.

⚠️ **The viewer's replay is a second implementation and the suite does not run it.**
`static/trace.js` was checked against `trace.py`'s `replay` and `puct` over three real
traces, agreeing to 1e-12 on every node's `N`, `Q` and child, and on the derived
scores at every third `k`. That check needs node and lives outside the suite, so it is
a thing that was true on 2026-07-30 rather than a thing that stays true.

## 13. Not frozen

- **Generation diff.** Two checkpoints, one position, root tables side by side. The
  format supports it already (a trace names its checkpoint); the view does not exist.
- **Tracing a chosen row of a `B > 1` search.** The recorder takes row 0 of a `B = 1`
  search. Recording row `r` of a self-play batch would let a trace come from a real
  generation rather than a reconstruction, at the cost of copying a slice per
  simulation.
- **Tree graph layout.** A visits-weighted node-link diagram. Tables first, because
  they answer the questions and a graph of 800 nodes mostly does not.
- **Trace diff.** Same position, same seed, two encoder implementations or two
  `pb_c_init` values, with the first divergent simulation reported. Cheap given the
  format, and it turns the viewer into a bisection tool.
- **UCI adapter.** Track D needs one for the external match harness (`evals.md`
  §10.2). It shares the move translation of §6 and nothing else, and belongs there.

## Changelog

**implemented, 2026-07-30.** `brokefish/search/trace.py`, `debugger/` and
`tests/test_trace.py` land, 18 checks green. The format needed no change; what the
code added to this document is measurements in place of two estimates (§4.5's trace
size, §11's cost and the recorder's 11 % overhead) and the endpoints §9.1 was missing.
`TracingSearch` turned out to need no copy of the reference's simulation body: every
quantity §4 asks for leaves an existing method as an argument or a return value, and
`n_legal` and the truncated mass leave `_expand` as arguments to the counter block,
which is why `_RecordingStats` exists.

**draft, 2026-07-30.** First version, written before any code. The trace-as-artifact
split of §2 and the derive-do-not-store rule of §3 are the two decisions everything
else follows from. Torch only (§10), justified by the differential test rather than by
convenience, and stated as a limitation on what the debugger is evidence about.
Training monitoring is explicitly out of scope and goes to wandb.
