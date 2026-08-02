# State — what is built, what it measured, what will bite you

Split out of `CLAUDE.md` on 2026-07-30, when that file crossed 600 lines. This is
the per-component ledger: one bullet per landed component, the number it was
measured at, and the traps that cost time. `CLAUDE.md` keeps the summary table and
the traps that belong to no single component; everything else lives here.

Every ⚠️ is a mistake that was actually made, or a design choice that reads as a bug
and is not. Deleting one because it looks obvious is how it gets made again.

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
  registers); `perf.md` has both rows and the static census, because the
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
  is loud (inf → NaN), not silent. `perf.md` has the headroom table.
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
  partition, the terminal code and the repetition count. See `environment.md`,
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
  story in `environment.md`. Test emptiness with `(mask == 0).all(-1)`.
- `csrc/tests/harness.cuh` — shared reading and reporting for the device tests. The
  engine tests print the offending position as a board plus a slot-by-slot diff, and
  `tstep.cu` **fails when a rule is uncovered** rather than passing vacuously; the
  dump is random playouts, where en passant is 39 cases out of 322 246.
- `search.md` — **the normative v0 search contract**, written 2026-07-30. v0 is
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
  games, looped over depth and simulations. `tests/test_search.py`, 37 checks from
  five angles, because no one of them is an oracle: §6.6 against a scalar
  transcription of AlphaZero's `ucb_score` over 1200 random tree states and §6.1's
  noise against the moments of `Dir(alpha)`; exact arithmetic on hand-built state;
  forced descents over every root edge of four positions; a constant evaluator,
  under which the tree is a function of the rules alone; and properties of chess.
  **15 of 16 mutations killed** by `tests/test_mutation_search.py`, the survivor
  being the designed control (`torch.argmax` happens to return the lowest index, so
  swapping it for `_lowest_argmax` changes nothing measurable). `search.md` §12.1
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
  simulations. AlphaZero's behaviour, and [the low-`n` policy target](perf.md#the-low-n-policy-target)'s low-`n` problem in concrete form.
  ⚠️ The Dirichlet **stream** is now shared: `cuda_impl` subclasses this and
  inherits §6.1, so one seed drives both and §12's comparison runs at the real `eps`.
- `csrc/search.cuh` + `csrc/search.cu` + `brokefish/search/cuda_impl.py`: **the CUDA
  search, C1 closed 2026-07-30. Gate 1a: 56 996 useful evals/s at n=800, B=4096**,
  a 4.4 % tree overhead over the encoder alone, 57.2 s per move
  (`bench/bench_search.py`, `logs/gate1a.log`, [the Gate 1a measurement](perf.md#what-the-tree-actually-costs-measured-2026-07-30)). `descent` fuses
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
  `search.md` §12.3 has the margin table and which runs are guaranteed rather
  than empirical.
  ⚠️ `E = 96` is compile-time (§6.6's scan is `kE/32` edges per lane), `B` and `Nmax`
  runtime. Changing `E` means editing `kE` in `csrc/search.cuh`.
  ⚠️ **The card throttles harder here than in any earlier benchmark**: 1230-1290 MHz
  at 82 °C through a 57-second move, so the same encoder call reads 59.9k inside the
  loop against the 64.1k `perf.md` quotes. Every absolute number in that file is
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
  positions; **they agree exactly at 64 and at 512 simulations**. `search.md` §12.2.
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
  `debugger.md` §4, and the web app replays them: root table with `Q` and `U`
  split, a scrubber over the `n` simulations with the score that decided each descent
  step, a node inspector, a run summary, and play against a checkpoint.
  `tests/test_trace.py`, 18 checks. **1360 B per simulation, 18 ms per simulation** at
  `B = 1`, so an `n = 800` move is about 15 s and the forms default to `n = 128`.
  ⚠️ **FastAPI and uvicorn are in `.venv` but still not in `requirements.lock`**
  (installed 2026-07-30, fastapi 0.141.1 / uvicorn 0.52.0), because nothing in the
  package imports them — so a fresh machine built from the lock will not have them.
  The reverse import, `brokefish` reaching into `debugger/`, is a defect.
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
  test covers that route (`debugger.md` §8).
- `2026-07-30-fidelity.md` — **where the reproduction may be wrong, written 2026-07-30**.
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
  before this, which blocked every external-opponent layer of `evaluation.md`.
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
- `evaluation.md` — the evaluation contract, **draft**, and Track D in the roadmap.
  Four layers: regression (every checkpoint, **44.4 s measured**), the self-anchored
  checkpoint league that produces the curve, external calibration, and the
  preregistered gate.
  `2026-07-30-eval-prior-art.md` at the root verifies every borrowed claim against the papers.
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
- `brokefish/eval/` — **layers 0 and 3 of `evaluation.md`, D1 done 2026-07-30**. The six
  rule suites (200 items each, cached in `data/suites.pt`), the layer-0 scalars, the
  Lichess puzzle curve, and `layer0_report()`, one call per checkpoint and one JSON
  record out. `tests/test_eval.py` (33 tests) checks every answer key against
  python-chess; `bench/bench_eval.py` times it phase by phase into
  `logs/d1_layer0.log`.
  ⚠️ **The Lichess CSV's schema was the one claim with no oracle, and is now
  settled**: `data/lichess_db_puzzle.csv` is on disk, and both the column names and
  the "the first move is the opponent's" convention were checked against real rows
  on 2026-07-30. `2026-07-30-eval-prior-art.md` §8 covers the licence, the count and the
  Glicko-2 deviation field.
  ⚠️ **A random-init net scores 0.147 on the puzzle curve against a 0.051 uniform
  null, and none of it is chess knowledge.** Decomposed: **0.448** where the solution
  is mate in 1, **0.054** everywhere else — chance to three decimals. Terminal nodes
  carry their result in place of an evaluation, so MCTS proves mates with no
  evaluation function and is blind to everything else. It finds only ~57 % of the
  available mates (137 of 279 mating edges get **zero** visits in 800 sims; PUCT's
  exploration is proportional to the prior and the smallest mating prior was
  0.00115), and in 17 cases it found the mate and played something else, because root
  selection is greedy on visit count rather than on `Q`. Over six random inits the
  mate rate is **0.568 ± 0.100** (0.448-0.738), so the untrained anchor's puzzle score
  measures the seed as much as the architecture — which matters because `evaluation.md`
  §5.1 pins that net as the frozen Elo-0 anchor. Measured 2026-07-30,
  `logs/d1_puzzles.log` and `logs/d1_why.log`.
  ⚠️ **`max |post-scale attention logit|` is not here**, though `evaluation.md` §4 listed
  it as a fourth layer-0 scalar. It is an fp16 overflow watch on the kernels rather
  than a measurement of a net, it would be the only thing in `eval/` reaching inside
  `net.encoder.layers`, and it can only read the torch oracle's logits rather than
  the accumulator that overflows. It belongs to C2 and `training.md` §11 already
  carries it. Dropped 2026-07-30.
  ⚠️ **Every suite's answer key is a spec §4.3 terminal code**, never a material
  count and never an evaluation, which is what keeps them inside the tabula rasa
  boundary during training. `evaluation.md`'s original definitions of stalemate
  avoidance and underpromotion both needed something else — a material count and a
  search — and were restated one ply deep: a mate must exist, the wrong move must be
  a named draw. There is deliberately no "best move" suite.
  ⚠️ **`tests/boards.py`'s random legal play cannot generate five of the six.** In
  170 924 positions from the start: 2 150 mate-in-1, 85 stalemate-avoidance, 3
  threefold, **0** underpromotion. `eval/positions.py` samples sparse endgames
  directly and lets `movegen` reject the illegal ones — 18× the yield, a bias in
  which positions are looked at and never in the key. `avoid_fifty` sets the clock
  to 99 and `avoid_threefold` **plants the repetition ring**; nothing else in the
  repository plants a ring.
  ⚠️ **The promotion field does not identify a promotion**: spec §3 is `0:N 1:B 2:R
  3:Q` and a non-promotion move also carries field 0, so testing `promo == 0` for
  "knight promotion" silently deletes the whole underpromotion motif. Ask the board,
  through `promotion_targets`.
  **31.7 s per checkpoint** (2026-07-31, `logs/layer0_cuda.jsonl`, 64 games at
  n = 100, six suites at n = 128), against §4's predicted ~6 s. ⚠️ The cost is a
  function of the checkpoint, not a constant: self-play dominates and it pays for
  the longest game in the batch, so a random init at 149 mean plies costs **46.9 s**
  where an 83-step net at 99 plies costs 31.7. Quote the range, not the number. The estimate assumed
  80-ply games where a random-init net plays 99–125, and costed the mean game where
  the loop pays for the longest in the batch. Both fixes failed and are written
  down: collecting the first `N` games *to finish* is 1.5× cheaper and biases mean
  game length short by 24 plies — disqualifying, since game length is one of the
  three metrics — and `batch < games` came out marginally worse. Layer 0 never
  blocks.
  ⚠️ **The old 44.4 s was measured on the reference tree, not the kernel, and the
  line here said otherwise.** `runner.make_search` defaulted to `search_impl="torch"`
  and `layer0.py` had no flag to change it, so no layer-0 run had ever driven the
  CUDA search. Same checkpoint, same seed, 2026-07-31: **519.6 s on the torch tree
  against 31.7 s on the kernel, 16.4×**, with suite accuracies agreeing to within
  one or two items of 200 (`mate_in_1` 0.860 both ways) — which is what makes it one
  measurement rather than two. Every record now carries `config.search_impl`.
  ⚠️ The two axes are coupled: `impl` is the encoder, `search_impl` is the tree, and
  the kernel's `_expand` refuses fp32 logits, so a CUDA search needs a fused encoder.
  `make_search` now pairs them. `TestSearchImplPairing` covers both directions and is
  deliberately not `@slow` — the flip that exposed this left the default suite green
  because every test driving a real search was slow-marked and skipped.
  ⚠️ Evaluation must run under `no_grad`. The first version did not, and the fp32
  master weights built an autograd graph across a 300-ply run that OOM'd an 8 GB
  card at 8 games.
- `brokefish/eval/{match,elo,league,curve}.py` — **D2, the league and the curve,
  `evaluation.md` §5.4, landed 2026-07-31**. `match.py` plays one pairing,
  `elo.py` fits one global Bradley-Terry model over the whole graph, `league.py`
  runs the fixed SAI calendar and joins the cost axis, `curve.py` presents.
  `tests/test_league.py`, 56 checks, all of them CPU-only on stub evaluators, 58 s.
  The frozen anchor is `checkpoints/anchor.pt`, **sha256
  `a27a099aa62940f13cb859a2adf16900c3934282f13ac27ccb680792fab4da57`** — recorded
  here because `checkpoints/` is gitignored and this digest is the only durable
  record of what the zero of the scale was.
  ⚠️ **No opening book, and the reason is not the tabula rasa boundary** — §2 permits
  standard books in evaluation. UHO exists to break draws between engines strong
  enough to hold a balanced position; our draws are a near-uniform policy shuffling
  into threefold repetition, which no opening prevents. 8 random legal plies,
  deduplicated by Zobrist hash. A book earns its dependency at §6, not before.
  ⚠️ **Evaluation is deterministic** (`eps = 0`, `tau_plies = 0`), so the openings are
  the *only* source of diversity in the league. Same pairing + same opening = the same
  game, every time. `G` games need `G/2` genuinely distinct openings, which is why the
  deduplication is a hash rather than an assumption.
  ⚠️ **A net against itself scores exactly 0.5 per colour-swapped pair** — both halves
  are literally the same game, so one half's point is the other's zero. That identity
  is the end-to-end oracle and it catches a sign flip, a swapped colour assignment and
  a double-counted half with no sample size to argue about.
  ⚠️ **One network per batch, and the lockstep is asserted every move.** The two nets
  alternate by ply, so a batch mixing both colour assignments would need two encoder
  calls per position. Playing the two assignments as two batches makes every row of a
  batch want the same net — 1× the encoder, static shapes — but only while every row
  is at the same ply. A finished row is therefore restarted **in phase**; a row reset
  to the ordinary start position would be a ply out of step, half the batch would be
  evaluated by the wrong network, and the games would stay legal and the results
  plausible. `match._swap_evaluator` raises instead.
  ⚠️ **The scale is score-based Elo, not BayesElo's draw-model Elo.** Expected score
  `1/(1+10^(−Δ/400))`, which is §9's convention and Ordo's default; BayesElo factors
  draws out and reports a larger number for the same games. Never compare the two.
  ⚠️ **The plain Bradley-Terry interval is too wide by ~`1/√(1−d)`** — a factor of 2.2
  at `d = 0.8`, so not a rounding error. BT models a game's score as having variance
  `p(1−p)` where a match at draw fraction `d` has `(1−d)/4`. Corrected by the standard
  quasi-likelihood dispersion, which lands near `1−d`; `se_raw` keeps the uncorrected
  number and `dispersion` is reported so the correction is visible.
  ⚠️ **A phantom opponent is what keeps an undefeated player finite**, and one will
  occur: the first real checkpoint against the random-init anchor is plausibly 100 %.
  One drawn game against a phantom at Elo 0, per player. It shrinks toward zero, so
  it is conservative.
  ⚠️ **The anchor is a file, not a seed.** `checkpoints/anchor.pt`, written once and
  never overwritten. A seed is not forever — it depends on the torch version and on
  nobody editing `BrokefishNet.__init__`. Delete that file and the whole curve moves.
  ⚠️ **A synthetic ladder that draws first and decides the rest by Δ is a different
  model**, whose expected score is `0.5 + (1−d)(p−0.5)`; a fitter that recovers Δ
  correctly then looks like it compresses the scale by `1−d`. Carve the draw band out
  of the score instead. Cost half an hour on 2026-07-31 and is written into
  `tests/test_league.py::_simulate`.
  ⚠️ Unfinished games are **dropped and counted**, never adjudicated as draws — that
  is exactly the bias the draw rate exists to detect.
  ⚠️ **An all-draw league reported `±0 Elo`**, found by the first real smoke league
  on 2026-07-31 and not by any unit test — every synthetic ladder has decisive games
  in it, because one without them has nothing to recover. All draws means zero
  Pearson residual, so the dispersion estimate is exactly 0 and the corrected
  interval collapses to 0: a false claim of certainty produced by the correction that
  exists to remove a false claim of imprecision. Guarded by `min_decisive = 30`,
  below which the uncorrected interval is reported and `dispersion_applied` says so.
  ⚠️ **`load_engine` packs the fused encoder from a CPU fp32 master** — thirty
  checkpoints then cost 13 MB each on an 8 GB card that is also the display, instead
  of 38. Checked against the torch oracle 2026-07-31: policy to 6.2e-3, value to
  1.7e-3, and the policy comes back fp16 as the kernel's `_expand` demands.
  ⚠️ **Throughput is unmeasured.** The smoke league ran on the kernel, but beside
  `t4h-n64` using the card at 99 %, so its 6 s per pairing measures contention.
  Batch is `--games / 2` per
  half (18 rows at the default), which is 576 encoder tokens — a prediction that small
  batches are acceptable *because a board is 32 tokens*, and a prediction is what it
  stays until a league runs. The tail waste from finished rows playing throwaway games
  is real and unquantified; refilling dead slots from a queue is the named seam.
- `brokefish/train/` — **the C2 training loop, `training.md`, done 2026-07-31**.
  `loss.py` (AZ eq. 1 and the label decode), `buffer.py` (the 500k-game window as a
  memory-mapped ring), `sync.py` (§8.1's two weight representations), `log.py`,
  `loop.py` (the alternation, the cadence, the optimiser, checkpoint/resume, the euro
  counter) and `overfit.py` (§12 check 1). `tests/test_train.py`, 31 checks.
  ⚠️ **The cadence rides positions, not games** (changed 2026-07-31, `training.md` §6).
  AZ's published `65.2 samples per game` is a reuse factor only at a fixed game length,
  and `t4h-n64` measured the mean game growing 101 → 144 plies while the per-position
  reuse fell **0.641 → 0.374 inside one run** — a 42 % drop in how much each example is
  trained on, with the data rate constant at 10 240 records per generation throughout.
  Records are produced at `moves_per_phase × batch_games` and do not depend on game
  length; only the games *closing* per phase do. `samples_per_position = 65.2/80 =
  0.815` now drives `steps_owed`, which gives 2.04 steps per generation constant
  against 0.94 and falling. ⚠️ The 80-ply divisor is **ours** — neither AZ nor AGZ
  publishes a game length, checked against both texts.
  ⚠️ **Log keys lost their `phase/` prefix** on 2026-07-31: wandb groups on the first
  path component only, so the wrapper put every metric of a run in one folder and threw
  away the `self_play` / `gradient` / `buffer` / `euros` grouping. Terminal codes are
  logged by name (`self_play/terminal_threefold`), from `env.TERMINAL_NAMES` — spec
  §4.3's table, now the one copy in code. A reader of a pre-2026-07-31 JSONL wants
  `phase/gradient/kl` where a current one has `gradient/kl`; `eval/league.py` reads both
  spellings, because a renamed key would have made the curve's cost axis silently null
  for every log already on disk.
  ⚠️ **C2 is the first phase with no oracle.** Perft settled the engine, `model.py`
  the encoder, an independent AGZ search settled C1. Nothing external says a training
  loop is correct, and its failure mode is a curve that is merely worse than it should
  have been. `training.md` §12 is what replaces the oracle and is the part to argue with.
  ⚠️ **The record stores the root's whole edge set, so the training path recomputes
  nothing** (`search.md` §10, `training.md` §3.5, revised 2026-07-31). `policy_len` is the
  edge count, not the visit count, and an unvisited edge is stored with `π = 0`. It
  carries nothing for the target and everything for the *denominator*. The arrays were
  already `E` wide and zero-padded, so this costs **no bytes** — the first draft
  recomputed `movegen` at training time to rediscover a support the search already knew,
  and rejected storing it on an arithmetic that was simply wrong.
  Consequences: `az_loss` takes no `env`; the dominant tensor went from two `[N, 8192]`
  fp32 arrays to one `[N, 96]` gather; the `E` truncation question disappears
  rather than being answered — it could not be answered, since the search truncated with
  the *generating* weights and training holds the *current* ones, and the two disagreed
  in the eighth generation of the first smoke run. The engine-side cross-check moves to
  `audit_labels`, run every `audit_every` steps (0.13 % of a step, so 100 is free).
  ⚠️ **`promo == 0` does not mean "not a promotion"** — spec §3 is `0:N 1:B 2:R 3:Q`
  and a quiet move carries field 0 too. Whether the `log_softmax(promo)` term belongs
  in an edge's logit is read from the *piece word and target rank*, never from the
  label. `is_promotion_edge` is that decision.
  ⚠️ **`weight_decay = 2c`, not `c`.** AGZ writes the penalty as `c‖θ‖²` with no half,
  so its gradient is `2cθ` and torch's `weight_decay=w` adds `wθ`. Passing `c` through
  halves the regularisation and nothing in §12 would notice.
  ⚠️ **A packed encoder is a snapshot.** Build it once in a constructor and self-play
  runs generation-0 weights forever while every loss curve looks healthy. Same shape
  as the C1 `push_history` bug. `PackedWeights` carries its `weight_gen` and a weight
  fingerprint, and `assert_current` runs before a single search of every phase.
  ⚠️ **§7.2's accumulation identity is exact and fp32 does not show it.** In float64 it
  holds to 1e-11 relative; in fp32 it lands at 1e-4 to 1e-6 and *moves between runs*,
  because a matmul at `N` and at `N/4` select different cuBLAS split-k reductions. The
  test proves it in double and pins fp32 loosely on purpose. Backward is deterministic
  run to run on this card, which is what makes §9's resume bit-exact.
  ⚠️ An in-memory buffer saves its **records** in its snapshot and a mapped one does
  not. Restoring an index over a fresh array samples zeroed records, and a zeroed
  record decodes as a live white pawn on a1 rather than as an error.
  ⚠️ **First §12 check-1 sweep, 2026-07-31: `lr = 0.002` beats AZ's `0.2` on both
  heads**, two orders of magnitude down (KL 0.571 against 0.876, value 0.178 against
  1.383; 1024 positions harvested at `n = 16`, 300 steps, one init and one frozen
  batch). At `lr = 0.2` the value loss *rises* and then pins at exactly 1.383 with the
  gradient norm at 0.1 — a **saturated `tanh`**, which produces no gradient at all.
  Provisional: no rate reached `KL ≈ 0`, so this is a rate comparison and not yet a
  pass of check 1, and `n = 16` targets have a mean `policy_len` of 3.5 against the
  much richer ones `n = 800` produces. `training.md` §7.3 has the table.
- Cost accounting is in (`loop.py`'s euro counter, §10, two numbers: the curve's
  x-axis and the project total). What is still **untouched** is everything downstream
  of C2 — the checkpoint league, the Elo curve, C4's sims sweep.
