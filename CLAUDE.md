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
- Search: Gumbel MCTS, n=32 sims in training, 800+ at evaluation.
- ~10M games × 80 plies → ~325 µs/move → **100k evals/s** wanted; ~10¹⁹ FLOPs,
  ~100-500 € spot. **The system is NN-bound, not env-bound.**

**Gate 1 (engineering)**: ≥45-50k evals/s sustained **inside a synthetic MCTS loop**
on the 4060 → GO; <15k after real effort → NO-GO. A pure forward benchmark does not
count. **Gate 2 (science)**: beat AlphaGateau (~2100 Elo) with an Elo slope matching
Jones' law (+500 Elo per 10× compute).

## Layout

```
brokefish/   Python package — nn/ (model.py is the whole network and the oracle;
             one file per fused implementation, selected by name through
             `encoder_impl`), env/ (the PyTorch engine); MCTS and training will
             land here
csrc/        CUDA C++ — chess.cuh (the 12-bit representation) and encoder.cu (the
             network) are settled; movegen.cuh holds the first order, contract for
             the rest in csrc/README.md. Device tests in csrc/tests/, one nvcc line
             each, no Python and no torch
tests/       correctness (torch is the oracle)      python -m tests.test_model
             the B2 surface, boards to logits        python -m tests.test_b2
             boards.py generates positions by random legal play — rules only
bench/       throughput, interleaved A/B protocol   python -m bench.bench_model
             --path full (default, B2) or backbone (reproduces perf.md rows 0-9)
docs/        spec.md (normative engine/network contract), perf.md (ledger),
             due_diligence.md (prior art)
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
  ⚠️ `rep` has no producer until C1. It is an argument; tests synthesise it.
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
  no backup, no selection, no per-node traffic. It is a **ceiling** on the loop rate.
  What Gate 1a now measures is what the tree costs, the only unmeasured term left.
- ⚠️ **`mask.sum(-1)` is a bug, not just `> 0` and sorting.** Mask words are int64
  carrying uint64 patterns, so summing overflows: two pieces able to reach h8 give
  2 × 2⁶³ = 0. `terminal` used it and called a position with four legal replies
  checkmate. Fixed 2026-07-29, pinned by `test_terminal_survives_mask_overflow`, full
  story in `docs/env.md`. Test emptiness with `(mask == 0).all(-1)`.
- `csrc/tests/harness.cuh` — shared reading and reporting for the device tests. The
  engine tests print the offending position as a board plus a slot-by-slot diff, and
  `tstep.cu` **fails when a rule is uncovered** rather than passing vacuously; the
  dump is random playouts, where en passant is 39 cases out of 322 246.
- The whole RL layer (MCTS, tree, training loop, Elo protocol, cost accounting) is
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
