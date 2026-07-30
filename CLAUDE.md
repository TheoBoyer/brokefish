# CLAUDE.md — brokefish

**"What's the cheapest way to have a superhuman chessbot from scratch?"**
A cost experiment, not an attempt to beat Stockfish. The deliverable is a clean
cost-vs-Elo curve on consumer hardware — a curve that does not exist in the
literature. A negative result is publishable too.

## Hard rules

1. **Answer in the chat.** Writing to a file never substitutes for answering. The
   complete answer is the **last message of the turn**; tool calls and note-taking
   come first. Never end a turn on "recorded".
2. **No flattery, no complacency.** "I don't know" is valued; bluffing is the only
   real failure. Never claim an unmeasured performance number.
3. **Predict → measure → explain the gap**, in that order, for every kernel.
4. **Kernels**: the goal is a world-class kernel. Comment the hardware reasoning,
   not the C++.
5. **A step is finished only when its number is written down.**
6. **Strict scope**: only comment on or fix what was asked.
7. For any long experiment campaign: give a `tail -f`-able log path up front, and a
   detailed summary at the end.

## Tabula rasa boundary — non-negotiable, it *is* the project

Forbidden in training: human games, engine games or labels, pretrained nets,
distillation, opening books, any inherited chess heuristic. Rules are allowed;
opinions about chess are not. (If distillation were allowed the flag is already
planted — CF-6M, NNUE, and the degenerate "just download Stockfish".)
Allowed: self-generated diversity (temperature, Dirichlet, random openings).
Allowed in evaluation only: standard books (UHO/TCEC) — evaluation is measurement.

## Architecture and sizing (decided)

- Env: GPU chess engine on a 12-bit piece-list representation (32×`uint16` slots
  + one `int16` side-to-move word), fully device-side. Slots are stable for the
  whole game, which is what index-aligns the 32 piece tokens with the 32 legality
  masks. Spec: `docs/spec.md`. The PyTorch reference engine (the
  correctness oracle, not a dependency) lives in `~/steakfish`.
- Net: piece-token transformer, 32 tokens = 32 pieces, summed position/type/turn
  embeddings, non-causal, 32×64 policy logits index-aligned with the engine's
  legality mask (same shape, so masking is a device-side AND with no gather and no
  host round-trip). ⚠️ **The network does not do that masking** — it emits raw
  logits and the search applies the mask, decided 2026-07-30, spec §7.4.
- Size **d=256 / L=8 / H=8 / FFN=1024 = 6.32M params in the stack ≈ 400 MFLOPs/eval**
  (1M params was refuted by scaling laws; AlphaGateau plateaus at ~2100 with 1M).
  With the embeddings (47,104), `norm_f` (512) and the three heads (17,664) the
  whole network is **6,383,360 params**, which is the number for the curve.
- Search: **AlphaZero PUCT, n=800**, spec `docs/mcts.md` (2026-07-30, superseding
  the "Gumbel MCTS, n=32" that sizing had assumed). `n` is chosen for convergence
  first; the sweep downward is where the throughput work starts, and Gumbel is a
  named seam for it.
  ⚠️ **The authority for the search is AlphaGo Zero (Nature 550:354-359), not the
  AlphaZero paper and not the released `pseudocode.py`.** AZ gives no PUCT formula and
  says its search is "identical to AlphaGo Zero"; `c_puct`'s value is published
  nowhere, so the logarithmic form and 19652/1.25 come from an unrefereed file. The
  pseudocode contradicts AGZ in three silent ways (un-normalised Dirichlet, a visit
  count one too large at every interior node, a value read that maximises the
  opponent's outcome), and AGZ has tree reuse and resignation, which v0 does not.
  `docs/mcts.md` §1.1, §3.1a, §3.6, §3.7 and §13.
- ~10M games × 80 plies → ~325 µs/move → **100k evals/s** wanted; ~10¹⁹ FLOPs,
  ~100-500 € spot. **The system is NN-bound, not env-bound.**

**Gate 1 (engineering)**: ≥45-50k evals/s sustained **inside a synthetic MCTS loop**
on the 4060 → GO; <15k after real effort → NO-GO. A pure forward benchmark does not
count. **Cleared 2026-07-30, and in the real loop rather than a synthetic one:
56 996 useful evals/s at n=800, B=4096** (`bench/bench_search.py`,
`logs/gate1a.log`, `docs/mcts.md` §14.3).
**Gate 2 (science)**: beat AlphaGateau (~2100 Elo) with an Elo slope matching
Jones' law (+500 Elo per 10× compute).

## Layout

```
brokefish/   Python package — nn/ (model.py is the whole network and the oracle;
             one file per fused implementation, selected by name through
             `encoder_impl`), env/ (the PyTorch engine), search/ (the MCTS);
             training will land here
csrc/        CUDA C++ — chess.cuh (the 12-bit representation) and encoder.cu (the
             network) are settled; movegen.cuh holds the first order, contract for
             the rest in csrc/README.md. Device tests in csrc/tests/, one nvcc line
             each, no Python and no torch
tests/       correctness (torch is the oracle)      python -m tests.test_model
             the B2 surface, boards to logits        python -m tests.test_b2
             the MCTS v0 reference                   python -m tests.test_search
             FEN/UCI/SAN/PGN, python-chess oracle    python -m tests.test_notation
             it against an independent AZ oracle     python -m tests.test_oracle
             the CUDA search against the reference   python -m tests.test_search_cuda
             the debugger's recorder and format      python -m tests.test_trace
             do those tests bite?                    pytest tests/ --mutation
             boards.py generates positions by random legal play — rules only
bench/       throughput, interleaved A/B protocol   python -m bench.bench_model
             --path full (default, B2) or backbone (reproduces perf.md rows 0-9)
             Gate 1a, the whole of §6 on device      python -m bench.bench_search
docs/        spec.md (normative engine/network contract), mcts.md (normative
             search contract), evals.md (the evaluation contract, draft — Track D
             in roadmap.md), perf.md (ledger), due_diligence.md (prior art),
             fidelity.md (where we may be wrong about FIDE and AlphaZero),
             debugger.md (normative trace format and viewer contract)
debugger/    the web viewer, the only part not importable from `brokefish`
             FastAPI + ES modules, no build step   README.md has the two commands
logs/        campaign output, tail -f-able while it runs; gate1a.log is C1's
```

This is a production repository: no staging areas, no step-by-step ladders, no
snapshots of work happening elsewhere. Code lands here when it is correct and
tested. Exploratory kernels and learning exercises belong in a scratch directory.

## State

- `brokefish/nn/model.py` — **the whole network of spec §7, and the oracle**: five
  embedding tables, the pre-norm stack, `norm_f`, and three biasless heads
  (policy `[256,64]`, promo `[256,4]`, value `[256,1]`). fp32 master weights,
  init std `1/sqrt(d)`. **B2 is done** (2026-07-30): `forward_full(boards,
  control, rep) -> (policy_logits, promo, value)` runs boards to logits in one
  CUDA launch at **62.3k evals/s**, ×3.25 over the torch full model — faster than
  the backbone benchmark, because the activation tile stops crossing HBM.
  `tests/test_b2.py` (14 tests) and `bench/bench_model.py --path full`.
  ⚠️ **The policy output is NOT masked.** Raw logits; the legality mask belongs to
  the search, which needs a masked softmax anyway. An all-illegal row means a
  terminal got expanded — `-inf` gives NaN, `-65504` gives uniform, so use `-inf`
  *and* assert `mask.any(dim=-1)`. Spec §7.4 carries this.
  ⚠️ The embedding sum's **order is normative** and the CUDA gather is tested
  bit-identical, not to a tolerance: `(square + type_special + color_turn) +
  (clock + rep)`, with clock and rep per position, not per token.
  ⚠️ `rep` was an argument with no producer until C1. `csrc/search.cuh`'s descent
  now computes it (§7) and stages it for the encoder; `tests/boards.py` still
  synthesises one for the tests that do not run a search.
- `brokefish/nn/cuda_impl.py` + `csrc/encoder.cu` — **the shipping kernel.
  61.6k evals/s masked on the backbone benchmark = ×3.02 over torch eager, ×1.43
  over Triton. B1 GO target (45-50k) is cleared**; 71 % of the measured fp16 mma
  issue ceiling. B2's prologue and epilogue are `__noinline__`, which is
  **measured, not stylistic**: they sit outside the k-loop but not outside its
  register budget, and inlined they pushed 56 bytes of spill into the mma blocks
  (12 as ABI calls, throughput unchanged). Anything else added to the prologue or
  epilogue has to be checked the same way.
  8 layers in one launch, one CTA per board, 8 warps, fp16 accumulation.
  **Weights never enter shared memory**: the host pre-permutes every matrix into
  mma B-fragment order (`pack_b`) and each lane pulls a whole fragment pair with
  one 128-bit load straight into the mma's registers. That is the whole trick —
  14 barriers per layer. The first CUDA build staged weights through SMEM like
  the Triton one and was *slower* than Triton (202 barriers/layer, 255
  registers); `docs/perf.md` has both rows and the static census, because the
  failed prediction is the instructive part.
  ⚠️ **Two hard budgets, both binding, both measured.** SMEM must stay at or
  under **50,176 B** or the block stops fitting twice on an SM (worth 5.4 %),
  which is why `bufA` alone is unpadded and the residual is staged and added
  row-wise instead of in place. Two CTAs of 256 threads then cap registers at
  **128/thread** (the kernel uses 116). Every further idea — fusing Q/K/V into
  one k-loop, deeper weight prefetch, `HCHUNK` 512 — breaks one of the two, so
  check both before writing code.
  ⚠️ `#pragma unroll 1` on the k-loop is **measured, not incidental** — unroll 2
  spills (−6.9 %), full unroll goes to local memory (−0.8 %).
  ⚠️ `acc_dtype="fp32"` **raises NotImplementedError here**: the kernel only
  emits `f16.f16.f16.f16`, and silently returning an fp16-accumulated result
  would pass every tolerance test while defeating the flag's only purpose. Use
  the Triton implementation for the wide accumulator until it is ported.
- `brokefish/nn/triton_impl.py` — 8 layers in one Triton kernel, 42.6k evals/s,
  validated 2.6e-3 vs torch, with the mandatory dead-token mask of spec §7.3
  and fp16 accumulation (`acc_dtype="fp32"` reverts it, and works). Kept as the
  A/B control in every benchmark and as the only fp32-accumulate path.
  ⚠️ fp16 accumulation moves the overflow ceiling to about 12× weight growth
  (the residual stream, already fp16, fails at ~17×). The single quantity to log
  during training is **max |post-scale attention logit|**, at 3.1 today; overflow
  is loud (inf → NaN), not silent. `docs/perf.md` has the headroom table.
  Implementations are selected by name (`brokefish.nn.encoder_impl`);
  `tests/test_model.py` (11 tests) and `bench/bench_model.py` run every
  available one, and `brokefish/nn/_build.py` compiles CUDA against torch's own
  CUDA 13.2.
- `csrc/tests/` — device-level unit tests, no Python and no torch: one `nvcc`
  line each, pinning the fragment layouts and the packed weight order against a
  host reference. `tdirect.cu` owns an independent reimplementation of `pack_b`,
  because a kernel/host packing mismatch is silent and produces plausible
  numbers eight layers downstream.
- `brokefish/env/` — the PyTorch engine, ported from steakfish and now the oracle
  for the CUDA movegen. **Spec §§2-6 are complete** as of A2: promotion, the
  fifty-move reset, Zobrist and the repetition ring, `in_check`, the terminal
  codes and the null move. Perft green on all six standard positions, startpos
  to depth 6 (119 060 324 nodes, 1220 s under `pytest --slow`); differential
  fuzzing against python-chess asserts the full `chess.Move` set, the hash
  partition, the terminal code and the repetition count. See `docs/env.md`,
  which also lists the three deliberate departures from python-chess.
- `csrc/movegen.cuh` — **the first order of the move generator, A1.1, done**: the
  per-(type, square) table, pawns with en passant, slider occlusion, castling
  rights, the friendly-occupancy filter, the control-mode attack map and
  `in_check`. Bit-exact against the PyTorch engine over the 10 000 positions of
  `data/cuda_testset`, stage by stage (`csrc/tests/tmovegen.cu`), at **109.7M
  boards/s**, 39 registers, no spill, 11 264 B SMEM. One warp per position, lane
  `i` owning slot `i`, which is what makes piece-list → bitboard a
  `__reduce_or_sync` and "unmoved rook on h1" a single ballot.
  ⚠️ That is the cheap half. The second order replays ~35 candidates per position,
  so the finished kernel lands near 3M boards/s — arithmetic, not a measurement.
  ⚠️ A header and not a `.cu` **on purpose**: C1's descent expands nodes mid-walk
  and has to call this from inside a kernel, never across a launch boundary.
  ⚠️ Torch tolerates shifts by a negative or out-of-range amount and C++ does not.
  Three of them are marked `DEVIATION` in the file; do not "simplify" them back.
- `csrc/step.cuh` — **the board mutation of spec §4.2, A1.2a, done**: the piece, the
  captured slot (including the en passant victim, which does not stand on the target
  square), the promotion type bits, the rook leg of a castle, the `special` bits and
  the fifty-move clock. Bit-exact over the 322 246 `(position, move, promo)` cases of
  the dump (`csrc/tests/tstep.cu`), at **738M moves/s**, 19 registers, no SMEM.
  Warp-collective with the same lane-owns-slot layout as `movegen.cuh`, so the second
  order can apply a candidate and build the resulting attack map in one warp.
  ⚠️ Two things in it **look wrong and are not**, both marked `LOOKS WRONG, IS NOT`:
  the rook does not get its `special` bit set when it castles (the king's is, and a
  right needs both), and the per-ply pawn-flag clear runs after the promotion so a
  just-promoted piece is skipped (unreachable: a flagged pawn stands on rank 4).
  ⚠️ `step.cuh` does **not** include `movegen.cuh` and must not: A1.2b's incremental
  Zobrist needs the attack map (spec §6.1 hashes the ep file only when the capture is
  legal), so that half needs a third home rather than a cycle.
- `movegen()` and `movegen_kernel` in `csrc/movegen.cuh` — **full legality, A1.3,
  done**. Brute-force second order: every pseudo-legal move applied and the resulting
  position asked whether our king stands on an attacked square, the whole warp on one
  candidate at a time so the mutated board never leaves registers. **Perft green on
  all six standard positions, startpos to depth 6 = 119 060 324 nodes in 1.01 s** at
  5.01M movegen/s (the reference takes 1220 s), 71 registers, no spill.
  ⚠️ 71 registers caps it at 3 blocks/SM, so occupancy is 50 %. Untested lever, and
  the margin over the loop's 45k is 111×, so it waits.
  ⚠️ The castling three-square test reads the attack map **after** the move, rook
  already on f1. That matches the reference and is validated by perft and by
  python-chess; it agrees with FIDE because neither leg can hide an attack on those
  three squares. Do not "fix" it to the pre-move map.
  ⚠️ The promotion choice cannot change legality (the opponent's map sees our
  promoted piece only through occupancy, never its type), so one replay settles all
  four edges. That is why `movegen` passes an arbitrary promo.
  Pruning to king moves, pieces on a line through the king, and en passant would cut
  ~35 replays to a handful. Deliberately not done: A1 ports the algorithm.
- `csrc/zobrist.cuh` + `csrc/terminal.cuh` — **spec §6 and §4.3, A1.2b, done**
  (2026-07-30): `castling_rights`, `legal_ep_file`, `hash_position`, `step_full` (the
  mutation plus clock, incremental Zobrist and `irreversible`), `insufficient_material`,
  `repetition_count`, `terminal`. **134.5M steps/s** for `step_full`, 80 registers, no
  spill, 17 512 B SMEM; `hash_position` 169-206M/s. At 5.01M movegen/s the hash is
  **3.6 % of the per-move cost**, so it does not need optimising.
  ⚠️ **The keys are generated on device, never loaded.** splitmix64's state advance is
  a plain addition, so key `i` is `mix(seed + (i+1)·φ)` and all 781 are independent.
  That is why `luts.py` picked splitmix64, and `tzobrist.cu` pins the device table
  against the Python one so they cannot drift.
  ⚠️ **`zobrist.cuh` includes `movegen.cuh`; `step.cuh` must never do so.** spec §6.1
  hashes the ep file only when the capture is actually legal, so the hash needs an
  attack map. That is the whole reason the engine is four headers: chess.cuh →
  step.cuh → movegen.cuh → zobrist.cuh, with terminal.cuh on chess.cuh alone.
  ⚠️ `step_full` costs 5.5× `apply_move` because `castling_rights` and
  `legal_ep_file` are each paid **twice**, before and after. That is inherent, not
  sloppiness: both en passant states enter the key.
  ⚠️ The per-game repetition ring is **C1's**, not here. `repetition_count` is
  implemented and unit-tested on a synthetic ring; a static dump can never carry
  terminal code 4.
- `csrc/engine.cu` + `brokefish/env/cuda_impl.py` — **the torch binding, A1 closed
  2026-07-30**. Swaps in for `torch_impl` behind identical signatures.
  `tests/test_cuda_env.py` (18 tests) covers what the device tests cannot: dtypes,
  optional args, the null move, the empty batch, argument order, and perft to depth 4
  through Python. Stateless on purpose: the tables arrive as tensors per call, since a
  cached device pointer would be wrong exactly once, on a device change.
  ⚠️ **Nothing in `cuda_impl` is on the self-play path.** One launch per call is a
  host-driven shape and spec §4 forbids host sync inside a step. C1 calls the
  warp-collective device functions in `csrc/*.cuh` from inside its own kernel.
- **`bench/bench_loop.py`: the environment is 2.2 % of a node.** 16 384 boards, five
  interleaved rounds: evaluate 255.1 ms (64.2k/s), + legality 259.9 ms (63.0k/s),
  whole node 260.6 ms (**62.9k/s**). The founding "NN-bound, not env-bound" claim is
  now a measurement, and the old 37.7k serialisation arithmetic was wrong by 1.7×.
  ⚠️ **62.9k/s is NOT Gate 1** and must never be quoted as it. No tree: no descent,
  no backup, no selection, no per-node traffic. It is a **ceiling** on the loop rate,
  and `bench/bench_search.py` has since produced the real number, 57.0k, of which the
  tree is 4.4 % and the card's throttling under a 57-second move is most of the rest.
- ⚠️ **`mask.sum(-1)` is a bug, not just `> 0` and sorting.** Mask words are int64
  carrying uint64 patterns, so summing overflows: two pieces able to reach h8 give
  2 × 2⁶³ = 0. `terminal` used it and called a position with four legal replies
  checkmate. Fixed 2026-07-29, pinned by `test_terminal_survives_mask_overflow`, full
  story in `docs/env.md`. Test emptiness with `(mask == 0).all(-1)`.
- `csrc/tests/harness.cuh` — shared reading and reporting for the device tests. The
  engine tests print the offending position as a board plus a slot-by-slot diff, and
  `tstep.cu` **fails when a rule is uncovered** rather than passing vacuously; the
  dump is random playouts, where en passant is 39 cases out of 322 246.
- `docs/mcts.md` — **the normative v0 search contract**, written 2026-07-30. v0 is
  the AlphaZero search adapted to fixed shapes: PUCT with the logarithmic
  exploration term, Dirichlet root noise, visit counts as the policy target,
  `n=800`, `B=4096`, `E=64` edges per node, a fresh tree per move. Faithfulness is
  a correctness lever, since the released pseudocode is an external oracle the way
  perft was. Gumbel is deferred, not rejected, and §11 prices it at two functions.
  ⚠️ **Not virtual loss, and there is no seam for it.** One simulation per game per
  step means the batch is `B` leaves from `B` trees, so there is nothing to
  deduplicate. §3.1 has the whole argument.
  ⚠️ `Q` is stored in **[0,1]**, not [-1,1]. PUCT *adds* the value and exploration
  terms, so halving `Q`'s range doubles effective exploration and `pb_c_init=1.25`
  stops meaning what the pseudocode means by it.
  ⚠️ The backup flips by parity: `q = ((L-d) % 2 == 0) ? qleaf : 1 - qleaf`. One
  leaf evaluation updates edges owned by players who alternate, so the flip is not
  about the network's convention and does not disappear if the value is read from
  the other king. Inverting it produces a search that plays the worst move.
- `brokefish/search/torch_impl.py` — **the MCTS v0 reference, done 2026-07-30**.
  §6 end to end: `root_init` with Dirichlet, descent, evaluate, expand, backup,
  `select_and_advance`, the §10 record and the §15 counter block. Batched over
  games, looped over depth and simulations. `tests/test_search.py`, 35 checks from
  five angles, because no one of them is an oracle: §6.6 against a scalar
  transcription of AlphaZero's `ucb_score` over 1200 random tree states and §6.1's
  noise against the moments of `Dir(alpha)`; exact arithmetic on hand-built state;
  forced descents over every root edge of four positions; a constant evaluator,
  under which the tree is a function of the rules alone; and properties of chess.
  **15 of 16 mutations killed** by `tests/test_mutation_search.py`, the survivor
  being the designed control (`torch.argmax` happens to return the lowest index, so
  swapping it for `_lowest_argmax` changes nothing measurable). `docs/mcts.md` §12.1
  has the table and which mutations only one test catches.
  ⚠️ **Test the parity at `L >= 3`.** At `L = 2` the correct `(L-d) % 2` and the
  inverted `d % 2` agree on both levels, so a two-level test cannot see the bug it
  was written for. The first version of that test could not.
  ⚠️ **Whether the search reaches a node is the exploration schedule, not
  correctness.** With FPU at 0 a mate at prior 0.017 needs `n = 800`; the same
  position mirrored and colour-swapped has prior 0.013 and is missed at 800. Force
  the descent (zero the other priors and their `Q`) rather than waiting for it, and
  give the node one visit first, since at `N_v = 0` every score is 0 and §6.6 takes
  edge 0 whatever the priors say.
  ⚠️ A vertical mirror is **not** a chess symmetry: pawns are not mirror-symmetric,
  so `6k1/5ppp/...` mirrored gives White an interposing promotion and stops being
  mate. Mirror *and* swap colours. This cost an hour of chasing a bug that was in
  the FEN.
  Both the environment and the network are injected, so it runs over `env/torch_impl`
  or `env/cuda_impl` and against `nn/model.py` or a fused encoder, and all four
  combinations give the same tree — a free differential check on A1 and B2.
  **27.4k evals/s at B=256, 36.8k at B=1024**, one move at `n=32`, both fused,
  invariant checks off.
  ⚠️ **That is not Gate 1a.** A Python reference with a host sync per descent level,
  measured where trees are shallow. It says C2 can develop against this instead of
  waiting for the kernel, and nothing about `n=800`.
  ⚠️ The forced-mate check needs `n=800` and `eps=0`. With first-play urgency at 0
  an unvisited edge is scored as a loss, so a mate at prior 0.017 is not reached
  until `pb_c * P * sqrt(N_v)` clears the visited edges' `Q`, between 400 and 800
  simulations. AlphaZero's behaviour, and §14.2's low-`n` problem in concrete form.
  ⚠️ The Dirichlet **stream** is now shared: `cuda_impl` subclasses this and
  inherits §6.1, so one seed drives both and §12's comparison runs at the real `eps`.
- `csrc/search.cuh` + `csrc/search.cu` + `brokefish/search/cuda_impl.py`: **the CUDA
  search, C1 closed 2026-07-30. Gate 1a: 56 996 useful evals/s at n=800, B=4096**,
  a 4.4 % tree overhead over the encoder alone, 57.2 s per move
  (`bench/bench_search.py`, `logs/gate1a.log`, `docs/mcts.md` §14.3). `descent` fuses
  select, `step_full`, the repetition scan, `terminal` and the allocation into one
  launch at **96 registers with no spill**; `expand` is 72, `backup` 20.
  `cuda_impl.Search` subclasses the reference and replaces only the four
  per-simulation steps, so the two share tensors and `select_and_advance`, the root
  noise and `reset` stay in torch on purpose.
  ⚠️ **The evaluator must return fp16 logits.** `expand` reads `policy` and `promo`
  as fp16, which is what both fused encoders emit; `nn/model.py` keeps fp32 master
  weights and returns fp32, and casting it in the driver would silently change the
  numbers the reference computes in fp32. It raises instead.
  ⚠️ **The tree reaches the kernel as a dict keyed by name, rebuilt once per move.**
  `env.push_history` is a pure function and returns a *new* ring, so a dict built once
  in the constructor points at the ring that stopped being updated after move 0. The
  symptom was a repetition count too low three moves and eight thousand simulations
  later. `_check_tree_is_current` turns any future rebinding into a loud failure.
  ⚠️ `edge_prior` is **not** bit-identical to the reference: the kernel's softmax
  denominator is a warp reduction over ≤64 survivors and torch's is over the
  8192-wide masked row, so they differ by one fp16 ULP on about one edge in a
  hundred. Everything else, `edge_Q` and `node_value` included, is exact.
  `docs/mcts.md` §12.3 has the margin table and which runs are guaranteed rather
  than empirical.
  ⚠️ `E = 64` is compile-time (§6.6's scan is two edges per lane), `B` and `Nmax`
  runtime. Changing `E` means editing `kE` in `csrc/search.cuh`.
  ⚠️ **The card throttles harder here than in any earlier benchmark**: 1230-1290 MHz
  at 82 °C through a 57-second move, so the same encoder call reads 59.9k inside the
  loop against the 64.1k `docs/perf.md` quotes. Every absolute number in that file is
  an upper bound on what the kernel does inside a generation, by about 6.7 %.
- `tests/test_search_cuda.py`: **the kernels against the reference, 19 checks**.
  both implementations from one seed, every array of §4.2 compared after **every**
  simulation, up to one move at `n=800` over 462 720 selections. Coverage the start
  position cannot reach gets its own run: the fifty-move boundary, a threefold from
  the game ring, checkmate and insufficient material 300 plies in, 218 legal moves for
  truncation, 189 promotion edges, and a 45-ply path so §7's scan loops twice per lane.
  `csrc/tests/tselect.cu` pins the PUCT scan and the canonical enumeration on their
  own against a host reference in double.
  ⚠️ The bit-exact test needs **pawnless positions**. Constant logits make every prior
  an exact `1/k`, but a promotion edge picks up `log_softmax` of four equal numbers,
  which is `-log 4` and not zero.
  ⚠️ An untouched node-pool slot is all zeros, which decodes as **a live white pawn on
  a1**. Any scan over the pool has to stop at each game's own `node_count`, not at the
  batch maximum.
  ⚠️ **The root's priors are fp16-quantised twice**, by the kernel and again by the
  torch noise mixing, so they get a 2-ULP tolerance where interior edges get 1. It is
  a 10⁻⁵ event and only `test_agrees_at_a_self_play_batch` (B=1024, slow) is large
  enough to see it. Everything else in the file runs at B ≤ 64, which leaves the
  batch axis otherwise untested.
- `tests/oracle.py` + `tests/test_oracle.py` — **the oracle, done 2026-07-30**: a
  second MCTS written from the AlphaGo Zero Methods, one game at a time in objects and
  lists, built four ways round from the reference (values per node flipped at read
  time, the repetition window walked up the parent links, the path as node references,
  node indices from a counter). Trees compared node for node and edge for edge on seven
  positions; **they agree exactly at 64 and at 512 simulations**. `docs/mcts.md` §12.2.
  ⚠️ **The evaluator has to be integer arithmetic, not the network.** The reference
  calls the net on a batch of B and the oracle on one position, and an fp32 reduction
  need not give the same last bit at two batch sizes. The shared evaluator is a
  splitmix64 mix whose outputs are small integers over powers of two, exact in fp32 and
  fp64, and it reads `rep`, which is what finally tests §7's count end to end.
  ⚠️ **Agreement is only worth what the precision floor allows.** Measured: priors and
  node values bit-identical, `Q` within 1e-7 (fp32 running mean vs fp64 mean), closest
  selection margin 3.4e-6 at n=64 (34x the error) but 9e-8 at n=512, where 4 of 15014
  selections could have been decided by rounding. The guard is self-calibrating and the
  deep run reports the count rather than demanding zero.
  ⚠️ **It checks faithfulness of implementation, not of reading.** Where both follow the
  same reading of the paper, agreement proves nothing about the reading.
- `brokefish/search/trace.py` + `debugger/` — **the search debugger, done 2026-07-30**.
  `TracingSearch` records one `B = 1` reference search as the per-simulation deltas of
  `docs/debugger.md` §4, and the web app replays them: root table with `Q` and `U`
  split, a scrubber over the `n` simulations with the score that decided each descent
  step, a node inspector, a run summary, and play against a checkpoint.
  `tests/test_trace.py`, 18 checks. **1360 B per simulation, 18 ms per simulation** at
  `B = 1`, so an `n = 800` move is about 15 s and the forms default to `n = 128`.
  ⚠️ **FastAPI and uvicorn are not installed and not in `requirements.lock`**, because
  nothing in the package imports them: `.venv/bin/pip install fastapi uvicorn`. The
  reverse import, `brokefish` reaching into `debugger/`, is a defect.
  ⚠️ **Every override in `TracingSearch` captures and returns unchanged.** Nothing
  copies the reference's simulation body, and `_RecordingStats` exists because
  `n_legal` and the truncated mass leave `_expand` only as arguments to the counter
  block. `test_recording_does_not_change_the_search` compares a traced run against an
  untraced one tensor for tensor; a recorder that halves the reference's speed is fine
  and one that moves a bit of `edge_Q` is not.
  ⚠️ **The viewer's replay is a second implementation the suite does not run.**
  `static/trace.js` was checked against `trace.py` over three traces to 1e-12 with
  node, which makes it a thing that was true on 2026-07-30. Change one, check both.
  ⚠️ Torch only. The kernels agree with the reference tree for tree, so a trace is a
  trace of what the kernel does *wherever `tests/test_search_cuda.py` covers*, and the
  debugger is not evidence about the kernels.
  ⚠️ No external engine evaluation in it, ever: not Stockfish, not a tablebase, not an
  opening name. The operator is the route by which engine opinion would enter, and no
  test covers that route (`docs/debugger.md` §8).
- `docs/fidelity.md` — **where the reproduction may be wrong, written 2026-07-30**.
  Everything else in `docs/` records what was measured; this records what was not.
  Two audits: the search against AlphaZero, the engine against FIDE. Nothing in it is
  a known defect, and every entry names what would settle it.
  ⚠️ **The two audits have very different strength.** perft is a real external oracle
  and settles move generation completely: **598M nodes across the six standard
  positions, all matching published counts, 4.8 s** (deepened 2026-07-30, the five
  non-startpos positions went from depth 4 to 5-6 because the CUDA engine made it
  free). The search has no oracle outside our own reading of AGZ.
  ⚠️ The three entries worth acting on: virtual loss is probably misfiled as an
  addition when AGZ had it, so it is a *deviation*; `tau_plies` may be plies where
  AGZ meant move pairs; and **fp16 priors flush to zero below 6e-8**, which is
  harmless at today's flat policy, unmeasured at a sharp one, and has no counter.
  That last one turns from a caveat into a bug silently.
- `brokefish/env/notation.py` — **D0, the output side, done 2026-07-30**: `to_fen`,
  `to_uci`, `to_san`, `to_pgn` and a batched `GameRecorder`, the inverses of the
  `from_fen`/`parse_san`/`from_pgn` that already existed. Nothing could emit a game
  before this, which blocked every external-opponent layer of `docs/evals.md`.
  `tests/test_notation.py`, 30 tests, python-chess as the oracle **string for
  string** over ~600 positions and ~13 000 moves of random legal play, plus a PGN
  handed back to `chess.pgn` and replayed.
  ⚠️ **SAN is the only hard part.** Three independent ways to be plausibly wrong:
  disambiguation, the en passant capture landing on an empty square, and the
  check/mate suffix. The disambiguation rule inverts easily — a rival on our *rank*
  forces the *file* — and the test carries the case neither hint alone resolves.
  ⚠️ **A FEN round trip does not preserve slots**: `from_fen` assigns them in scan
  order, the engine keeps a piece where it started, so after 1. Nh3 the piece lists
  are a permutation of each other. A FEN cannot resume anything indexed by slot.
  ⚠️ En passant uses the **legal** convention (`legal_ep_file`, spec §6.1), matching
  python-chess's default; the full-move number is an argument, since spec §2.2 does
  not carry one. Binds to `torch_impl` on purpose: host-side I/O for evaluation,
  never on the self-play path, so a PGN writer needs no CUDA toolchain.
- `docs/evals.md` — the evaluation contract, **draft**, and Track D in the roadmap.
  Four layers: regression (every checkpoint, ~6 s), the self-anchored checkpoint
  league that produces the curve, external calibration, and the preregistered gate.
  `eval_prior_art.md` at the root verifies every borrowed claim against the papers.
  Settled so far: **no checkpoint gating** (so C2 does not depend on Track D), fixed
  simulations per move rather than a time control, greedy move selection in
  evaluation, target a CI width rather than a game count, and two cost numbers with
  training-only on the curve's x-axis.
  ⚠️ **Submission to CCRL is closed** — the list is CPU-only and refused a GPU
  exception for Lc0 — so the anchor is a *borrowed* rating and our matches have to
  reproduce the conditions it was measured under.
  ⚠️ Evaluation output may never flow backwards. The prohibition that will actually
  get violated is checkpoint selection: "keep the checkpoint with the best puzzle
  score" is distillation through a one-bit channel and looks like good practice.
- The rest of the RL layer (training loop, replay buffer, cost accounting) is
  **untouched** — no line written, no decision frozen.

## Environment

`.venv` is a self-contained environment: torch 2.13.0+cu132, triton 3.7.1,
python 3.13, the exact stack every number in `docs/perf.md` was measured on.
Run `.venv/bin/python -m tests.test_model` from the repository root.

It was cloned with `cp -al` from a sibling environment rather than installed,
because uv's cache entries for the `nvidia-*` wheels have lost their HTTP
revalidation metadata (only the unpacked archive survives), so any `uv pip
install` re-downloads ~3 GB even with every version pinned. Hardlinks mean the
clone costs no disk and stays independent: installing or removing a package
replaces files rather than editing them in place.

On a fresh machine (rented GPU, CI) use `requirements.lock` — the frozen wheel
set, pinned down to torch's whole transitive closure. Hand over the install command, never run it.

## Hardware

**RTX 4060 Laptop, 8 GB, sm89 (Ada)**: fp16/fp8 tensor cores via `mma.sync`,
`cp.async` — but no TMA, no wgmma, no clusters. 99 KB SMEM/block, 64K regs/SM.
⚠️ Clocks drop to **1.38-1.5 GHz** under sustained load, and **fp32 accumulation is
half rate on GeForce**. Measured `mma.sync` issue rates, 2026-07-29:
**18.0 TFLOPS fp16→fp32**, 35.5 fp16→fp16, 41.6 e4m3→fp32. The "35-40 TFLOPS fp16"
that used to sit here is the fp16-*accumulate* rate and does not apply to a kernel
that accumulates in fp32 — see `docs/perf.md`, it moves the GO target. Thermal drift ±3 % → **only order-balanced interleaved A/B
comparisons are valid** (a naive before/after already produced a fake -4.5 % gain).
Heavy phases: Modal ($30 free/month, H100 ≈ $3.95/h).

## Don't redo these (measured dead ends)

`torch.compile`/CUDA graphs (0 %), fp16 accumulation (0 %), cache/evict hints
(0 to -8 %), fp16 LN affine (noise), smaller GEMM tiles via CUTLASS (2× worse),
reductions via `tl.dot` instead of `tl.sum` (**1.85× worse** — layout conversion
costs more than the reduction), `Var = E[x²]-µ²` (noise), head dimension (noise).

Tooling traps: Triton has no lists and no dynamic register indexing (hence the
hand-unrolled `x0..x3` tiles, which pin `d_model = 4 × 64`); ncu and torch SDPA
crash together (`Cannot load symbol cudnnGetVersion`) so profile without torch
attention in-process; `nsys stats` needs `--force-export=true`; never truncate an
error message.

⚠️ Running `.venv/bin/python -m pytest` **without activating the venv** fails any
test that builds a CUDA extension: torch's `is_ninja_available()` shells out to
`ninja --version`, so it needs `.venv/bin` on `PATH`, not just the `ninja` package
importable. The error says "Ninja is required to load C++ extensions", which reads
like a missing install and is not one. Use `PATH="$PWD/.venv/bin:$PATH"` or activate.


---
`docs/cuda_walkthrough.md` is the pedagogic write-up of how `csrc/encoder.cu` was
arrived at — fragment layouts, the reuse analysis that decides the design, the
version that was slower than Triton, and the measurement method. Read that before
touching the kernel.

Deeper detail on any of the above — full sizing derivation, prior-art due diligence,
the complete CUDA kernel spec, the perf ladder: `briefing.md`, `docs/`. Consult the
section you need when you need the *why* — not every session.
