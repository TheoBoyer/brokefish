# Roadmap

Written 2026-07-29, after the specification was frozen. Estimates are in working
days with both of us on the task, and they assume co-implementation with no split
of ownership between kernels and harnesses.

⚠️ **Both tracks are closed as of 2026-07-30.** A0, A1, A2, B0, B1 and B2 are done:
the CUDA engine is perft-green and the network runs boards to logits in one launch.
The measurement that mattered most is that **the environment costs 2.2 % of a node**,
which turns the project's founding assumption, "the system is NN-bound, not
env-bound", from an argument about FLOP counts into a number. C1 is the only phase
left before the pilot, and the only one with no specification.

⚠️ **The day unit calibrated at about 1.3 wall hours, and that calibration has now
expired.** A0, A2, B0 and B1 were delivered between 12:00 and 21:00 on 2026-07-29,
and A1 and B2 on 2026-07-30: 17 to 23 estimated days across two sessions and two
parallel tracks. It held because every one of those phases had a written
specification and an oracle to check against before a line was typed. C1 has
neither, [spec §11](spec.md#11-not-frozen) lists its parameters as explicitly not
frozen, and there is no reference implementation to be differentially tested against.
Do not convert the C1 estimate.

The two tracks shared only the engine interface in
[spec §4](spec.md#4-the-engine-contract), which is frozen, and never shared a file.
What is left does not split that cleanly: C1 and C3 both live in `brokefish/`.

## What steakfish provides

The PyTorch engine in `~/steakfish` is the ancestor of this project. It is not a
dependency, and python-chess is the correctness oracle, but several pieces are worth
importing rather than rebuilding.

### Taken as-is

| what | where | note |
|---|---|---|
| FEN and `chess.Board` to 32 words | `engine.py`: `from_fen`, `from_board`, `from_boards` | already sets `magic = ±(halfmove_clock+1)`, matching spec v1 |
| 32 words back to `chess.Board` | `utils.py`: `to_chess_package` | reconstructs castling rights, ep square, turn, and `halfmove_clock = abs(magic)-1` |
| mask to `chess.Move` and back | `utils.py`: `list_legal_moves`, `move_to_args` | `move_to_args` already parses `.promotion` |
| the four movegen lookup tables | `utils/generate_bitset_cache.py` | `move_bitsets`, occlusion offsets and masks, `filled_lines`, which is exactly what the CUDA kernel indexes |
| differential dump pipeline | `utils/dump_cuda_testset.py` | 21 hand-picked FENs covering castling through check, pinned en passant, promotions, mate, stalemate and sparse endgames, plus per-stage snapshots for validating a kernel incrementally |
| fuzzing harness | `tests/test_moves.py` | random positions, legal-move set equality against python-chess, then random walks re-asserting every ply, single and batched |
| position sampler | `data.py` | draws real positions from PGN parquet shards |
| board and bitmask printers | `utils.py`: `plot_board`, `plot_bitmask` | |
| movegen throughput benchmark | `benchmark.py` | |

The import path already implements the spec-v1 clock semantics. Only `apply_move`
carries the old unconditional ply counter.

`data.py` samples positions from human games. Those positions are test coverage for
the move generator and never reach training, which keeps them outside the training
boundary. The dependency on `data/*.parquet` has to be carried or replaced with
generated positions.

### Not provided, and therefore work ✅ closed by A2

Everything in this list was a gap in the ancestor and is now implemented, except
the last two lines. Kept as the record of what the port owed.

Promotion in the mask, where `pawn_moves` carried a bare `# Promotion #TODO`. The
halfmove clock reset in `apply_move`. Zobrist hashing, repetition, insufficient
material, `in_check`, and terminal codes. Mixed batches containing `-1`.
Underpromotion in the tests, which were gated behind `IGNORE_PROMOTIONS = True`.

Still open: the pinned-set second-order pass, since only the brute-force version
exists, deferred by A1's design. The two open bugs recorded in `todo.txt` replay
clean on both engines and are stale.

## Track A: the environment

### A0. Port and inventory, 1 day ✅ 2026-07-29

Done, and written up in [the environment](env.md): the engine lives in
`brokefish/env/`. Perft is green on startpos, Kiwipete and position 3, to 197 281
nodes in the default test run and to 4 865 609 under `pytest --slow`. Differential
fuzzing against python-chess is clean over 3032 legal-move-set comparisons drawn
from 256 positions. `csrc/chess.cuh` now exposes `advance_control(control, reset)`
in place of `advance_ply()`.

The dumps under `~/learncuda/` were kept rather than deleted, on the grounds that
the clock reset which would invalidate them was deferred to A2. That reset has now
landed, so they are dead and `scripts/dump_cuda_testset.py` has been re-run. The two
open bugs in `~/steakfish/todo.txt` replay clean on both engines and are stale.

### A1. Transliterate to CUDA, 3-5 days ✅ 2026-07-30

Port `first_order_mask` and `step` directly, keeping the brute-force second-order
pass that replays every candidate move. Validate against python-chess.

Three sub-steps, in this order because each one is the previous one's test oracle
and because a composite of two unvalidated halves cannot be debugged:

| | what | state |
|---|---|---|
| A1.1 | `first_order_mask`, both modes, and `in_check` | ✅ 2026-07-29, `csrc/movegen.cuh` |
| A1.2a | `step`, the board mutation and the fifty-move clock | ✅ 2026-07-29, `csrc/step.cuh` |
| A1.3 | the second order, the `movegen` entry point, perft | ✅ 2026-07-29 |
| A1.2b | the incremental Zobrist, `irreversible` and the terminal codes | ✅ 2026-07-30 |

The hash comes last because **perft never reads it**, which is what the reference's
optional `hash` argument exists for. That buys the correctness gate that matters
sooner, at the cost of an engine that is perft-green while still unable to detect a
threefold repetition. Decided with Théo on 2026-07-29.

**A1.1 landed at 109.7M boards/s** on the 4060 at a batch of 260 000, 39 registers,
no spill, 11 264 B of shared memory, bit-exact against the PyTorch engine over the
10 000 positions of `data/cuda_testset` at each of the four rule stages plus the
friendly-occupancy filter, the control-mode attack map and `in_check`
(`csrc/tests/tmovegen.cu`). That number is the cheap half of the movegen and says
nothing about the finished kernel. It is not in `docs/perf.md`, which is the model
inference ledger.

`scripts/dump_cuda_testset.py` now also writes `fo_control.bin`, the attack map of
the side not to move, which is what let `in_check` be validated at A1.1 instead of
at the end of A1.3.

**A1.2a landed at 738M moves/s** over the 322 246 cases of the dump, 19 registers,
no spill, no shared memory, with the 32 slot words and the control word bit-exact
against the reference. The clock rides along with the mutation because it is one
select over two flags the mutation already computes; splitting it out would have
meant a second kernel to flip a sign.

That number is a pessimistic proxy for the cost inside the second order, where the
mutated board stays in registers and feeds `first_order_mask` in the same warp
instead of going to memory. Composing the two measurements gives **3.0M positions
per second** for the finished movegen: 9.1 ns per first-order warp plus 1.4 ns per
mutation, times the measured 31.1 legal moves per position. That is arithmetic over
two separate measurements and it is a prediction, not a result. The loop needs 45k.

⚠️ **The hash is not in there, and its dependency is easy to miss.** Spec §6.1
hashes the en passant file only when the capture is actually legal, king safety
included, so `legal_ep_file` runs a king-safety test of its own and therefore needs
the control-mode attack map. A1.1 is a prerequisite for A1.2b rather than merely
convenient before it, and paying it 35 times per position would put a second-order
pass inside a second-order pass.

**A1.3 landed, and perft is green on all six standard positions including startpos
to depth 6**: 119 060 324 nodes in **1.01 s** at 5.01M movegen/s, against 1220 s for
the same count in the PyTorch reference. 71 registers, no spill. The full legality
mask and `in_check` are bit-exact against the reference over the 10 000 positions of
the dump, and perft is the one check in the suite whose oracle is not this
repository, so a bug the two engines share would still show up there.

Measured movegen rate by position type: **5.01M/s** at full chunks in the deepest
level, 2.0-3.7M/s at partial chunks where launch overhead dominates, and 10.4M/s on
position 3, a sparse endgame with few replays per position. The loop needs 45k. The
predicted 3.0M was low and the reason is that the shared-memory tables are loaded
once per block and amortised over 8 positions × ~35 replays rather than over 8
positions.

⚠️ **The port found a bug in the reference, and it was a real one.** `terminal` read
"no legal move" as `mask.sum(-1) == 0`, and mask words are int64 carrying uint64 bit
patterns, so the sum overflows: four pieces able to reach g8 give 4 × 2⁶² = 2⁶⁴ = 0,
and two able to reach h8 do the same. One position in 10 000 hit it, and
`1r1r2Rk/pp5p/4b2n/1P5P/n1Ppp3/R4P1p/1B1KP3/8 b - - 1 1` was reported as checkmate
with result −1 while having four legal replies, every one capturing the checking
rook. That is a false game end and a −1 value target on a live position. Fixed to
`(mask == 0).all(-1)`, pinned by `test_terminal_survives_mask_overflow`, and written
up in [the environment](env.md). The differential fuzzing missed it because it needs
a specific high-bit coincidence, and 1 280 fuzzed positions were not enough.

`csrc/tests/tstep.cu` reports rule coverage and fails when any rule is uncovered,
because the dump is drawn from random playouts where castling and en passant are
rare: 42 343 captures, 39 en passant across all four geometries, 297 castles both
sides, 1 556 promotions, 16 597 double pushes, 10 000 null moves. All eight
single-rule mutations of `step.cuh` are caught, each failing exactly the number of
cases its rule covers, and all six of `movegen.cuh`'s second order are caught by the
differential test and by perft independently, the castling ones at Kiwipete depth 2.

**A1.2b landed at 134.5M steps/s** for the full `step_full` (mutation, clock,
incremental Zobrist, `irreversible`), 80 registers, no spill, 17 512 B of shared
memory. `hash_position` from scratch runs at 169-206M/s and `terminal` needs no
launch of its own worth measuring. In `csrc/zobrist.cuh` and `csrc/terminal.cuh`.

Predicted 150-300M and measured 134.5M, and the decomposition says why: 1.35 ns for
the mutation plus 2 × 3.05 ns, because `castling_rights` and `legal_ep_file` are each
paid twice, once for the position before the move and once for the position after.
An incremental hash cannot avoid that, since both en passant states enter the key.
The number that matters is the composition: at 5.01M movegen/s a real move costs
199.6 ns of movegen against 7.4 ns of hashing, so **the hash is 3.6 % of the per-move
cost**. spec §6.1 says the update is "four to six XORs per move", which is true and
is not where the time goes; the en passant legality test that §6.1 also mandates is.

The keys are **generated on device**, not loaded, which is the property `luts.py`
chose splitmix64 for. splitmix64 advances its state by adding a constant, so key `i`
is `mix(seed + (i+1)·φ)` and the 781 keys are independent: a parallel map, not a
serial loop. `tzobrist.cu` pins the device table against the Python one, so the two
cannot drift.

The strongest check needs no oracle at all: hashing a child position from scratch has
to agree with having arrived at it incrementally, over all 322 182 transitions. An
incremental update wrong in a way the reference is also wrong in still fails it.
Twelve single-rule mutations across the two headers are caught, including
re-introducing the int64 sum bug, which now fails on 2 positions.

⚠️ **The coverage check found a second gap, this one in the test set.** The dump had
no fifty-move position at all, because the playouts run 100 plies and the clock keeps
resetting, so that branch of `terminal` was untested by construction. Five positions
were added to `SPECIAL_FENS`: the fifty-move boundary on both sides (clock 99 gives
none, clock 100 gives drawn), two bishops on one colour complex and two on opposite
complexes, which is the pair spec §4.3 was amended for, and the overflow position
itself. Every code the dump can carry now appears.

**A1 is closed.** `csrc/engine.cu` and `brokefish/env/cuda_impl.py` put the four
entry points of spec §4 plus the hash behind `torch_impl`'s signatures, so
`from . import cuda_impl as env` swaps the implementation and nothing else. The
bindings are stateless: the tables arrive as tensors on every call rather than
living in a global device pointer that would be wrong exactly once, on a device
change, and would then read someone else's memory. `tests/test_cuda_env.py` (18
tests) covers everything between the kernel and the caller, which is what the
device tests cannot: dtypes, the optional arguments, the null move, the empty
batch, the argument order in the six-tensor signatures, and perft to depth 4
through the Python API.

⚠️ Nothing in `cuda_impl` is on the self-play path. spec §4 forbids the engine from
synchronising with the host inside a step, and one launch per call is a host-driven
shape. C1's descent calls the device functions in `csrc/*.cuh` from inside its own
kernel. The module exists for tests, benchmarks and notebooks.

**The combined-throughput number, `bench/bench_loop.py`, 16 384 boards, five
interleaved rounds:**

| phase | ms | rate |
|---|---|---|
| evaluate (B2 full path) | 255.067 | 64.2k/s |
| legality (movegen) | 4.307 | 3 803.8k/s |
| advance (step + hash) | 0.158 | 103 900k/s |
| advance, no hash | 0.035 | 469 096k/s |
| evaluate + legality | 259.948 | 63.0k/s |
| whole node | 260.565 | **62.9k/s** |

**The environment costs 2.2 % of a node.** Legality alone is 1.9 % and advancing
adds 0.3 %. That is the first measurement behind the sizing claim that has been in
`CLAUDE.md` since the start, "the system is NN-bound, not env-bound": it was an
argument from FLOP counts and it is now a number. It also retires the 37.7k
serialisation arithmetic this section used to carry, which assumed the two phases
serialise at their separately-measured rates and was wrong by a factor of 1.7.

⚠️ **62.9k/s is not Gate 1 and must not be quoted as it.** Gate 1 is 45-50k evals/s
sustained inside a synthetic MCTS loop, and `bench_loop.py` has no tree: no descent,
no backup, no selection, no per-node memory traffic. One node's work is a **ceiling**
on the loop rate, so the real number is lower and C1 produces it. What this does
settle is that the environment is not what will decide it.

The movegen reads 3.80M positions/s here against 5.01M in `tperft.cu`. Different
batch size, different residency, and in-process alongside a 6.3M-parameter network,
so the two are not the same measurement and the lower one is the relevant one.

Repetition is implemented (`repetition_count`, spec §6.3) and unit-tested on a
synthetic ring, but the per-game ring itself is C1's: a static dump holds positions,
and repetition is a property of the game that reached one, so `terminal_code.bin` can
never carry code 4.

The 50-100M boards/s figure in the specification stays a write-up claim rather than a
gate. The environment is 2.2 % of a node, so optimising it can win at most that, and
nothing here needs it.

⚠️ **A1 is unconditionally on the critical path, ahead of C1**, which this section
used to treat as optional.

The reason is architectural rather than a throughput margin. C1's descent is a
device-side kernel walking a tree pool in VRAM, and it expands nodes mid-descent.
A device-side kernel cannot call a torch program, so an environment that lives in
Python forces the loop to return to the host at every expansion, which is a
different loop from the one we intend to ship. Building C1 against the torch
engine would mean writing the integration twice and learning nothing transferable
from the first version.

The throughput arithmetic points the same way without being the argument. MCTS
needs one movegen per node expansion, so at 45k evals/s it wants 45k movegens/s
against the torch engine's measured 98k boards/s; two phases at 61.3k and 98k that
serialise combine to **37.7k**, under the 45-50k GO band. That is arithmetic over
two measurements taken separately at different batch sizes, assuming strict
serialisation, and it is worth nothing as a prediction about the shipped system,
where the CUDA engine is expected near 1.6M boards/s.

**Exit criteria**: perft green against the PyTorch oracle on all six standard
positions, the differential test set in `data/cuda_testset` reproduced bit for bit
including `in_check`, the hash and the terminal codes, and one interleaved
combined-throughput number with the encoder. That last one costs an hour once both
kernels exist and is the first honest estimate of what the environment charges the
loop.

### A2. Complete the rules, 3-4 days ✅ 2026-07-29

Spec §§2-6 are implemented in the PyTorch engine. Promotion through the `promo`
field, the fifty-move reset, the incremental Zobrist hash, the `irreversible` flag,
the repetition ring, `in_check`, the terminal codes with the amended
insufficient-material rule, and the null move.

Every exit condition met. `IGNORE_PROMOTIONS` is deleted and the differential
harness compares full `chess.Move` objects. `in_check`, the terminal code, the
repetition count, `irreversible`, the castling-rights vector, the legal en passant
file and the hash are all asserted against python-chess. Perft is green on all six
standard positions, startpos to depth 6.

Two findings changed the contract rather than the code. `interop.list_legal_moves`
had to expand a promoting mask bit into four moves, which exposed that the perft
harness was counting mask bits where perft counts moves: position 5 is 44 moves and
41 bits. And spec §4.3's four-case material rule was amended to match python-chess,
which calls K+2B against K a draw when both bishops share a colour complex.

The signatures are now `movegen -> (mask, in_check)` and
`step -> (boards, control, hash, irreversible)`, with `hash` optional in both
directions. Details in [the environment](env.md).

## Track B: the network

### B0. Dead-token mask on the current Triton kernel, done 2026-07-29

The 37.2k evals/s baseline and the 45-50k GO target were both measured without the
attention mask that [spec §7.3](spec.md#73-dead-tokens) makes mandatory, so the
target was being chased against a kernel we do not ship.

The mask costs **0.97 %**, measured over a 12-round interleaved duel, which puts the
production kernel at 36.4k evals/s and ×1.78 over torch eager. Four implementations
land within 0.44 % of each other; the [performance ledger](perf.md#the-dead-token-mask-2026-07-29)
has the table and the reason the prediction of 3 to 6 % was wrong.

The GO target is unchanged and the gap to it is now ×1.24.

### B1. CUDA encoder rewrite, 6-9 days ✅ 2026-07-29

Warp-specialised kernel with `cp.async`, no register staging, and per-tile
mbarriers, following the plan in `csrc/README.md` and the diagnosis in the
[performance ledger](perf.md).

The dead-token mask goes in from the start. It lives inside the attention inner loop
and changes register pressure, and the current tuning sits on a cliff where
`maxnreg=168` is the exact threshold that forces the third CTA. Adding it after
tuning would mean re-running the sweep.

**Re-baselined 2026-07-29 by the roofline measurement.** fp32 accumulation is half
rate on GeForce Ada, so the ceiling for the kernel as it stood was 18.0 TFLOPS and
the 45k target sat at 103 % of it — unreachable by any amount of engineering. fp16
accumulation lifts the ceiling to 35.5 and was ported to the Triton kernel
immediately: **42.6k evals/s, ×2.12 over eager**, leaving ×1.06 to GO. The CUDA
kernel now starts from 50 % of its ceiling rather than 83 %, which is the regime
where removing the 90 barriers per layer actually pays. It accumulates in fp16 by
default with an fp32 flag, and the split-GEMM alternative is refuted (see
[the ledger](perf.md#the-roofline-measured-2026-07-29)).

**Landed at 61.3k evals/s masked**, ×3.03 over eager and ×1.43 over Triton, at
71 % of the measured fp16 issue ceiling. The B1 GO band of 45-50k is cleared on a
pure forward benchmark, which is the part Gate 1 explicitly says does not count;
the in-loop number is C1's to produce. The warp-specialised plan was not what won:
the kernel keeps weights out of shared memory entirely and pulls B-fragments
straight into registers from a host-side permutation, and the first build that did
stage through SMEM was slower than Triton. `docs/cuda_walkthrough.md` is the
write-up, and the two binding budgets (50 176 B of SMEM, 128 registers) are in
`CLAUDE.md`.

### B2. Embeddings and heads, 1-2 days ✅ 2026-07-30

The five embedding tables from spec §7.2, the value head on the king token, and the
promotion head on the pawn token. Both are prologue and epilogue work that leaves
the inner loop untouched, so they follow the optimisation rather than precede it.

**Landed at 62.3k evals/s boards-to-logits**, ×3.25 over the torch full model and
faster than the 61.6k the backbone benchmark reads on the same day — the activation
tile no longer crosses HBM in either direction. `brokefish/nn/model.py` is the
oracle, `tests/test_b2.py` has 14 tests, and the embedding gather is held to
bit-identical rather than to a tolerance.

Four decisions changed the spec rather than just implementing it, and §7.4/§7.2
carry them: a **final LayerNorm** before the heads (the pre-norm stack was feeding a
raw fp16 residual stream to a linear head), the single padded aux head `W_a : [256,8]`
**split into `W_promo : [256,4]` and `W_value : [256,1]`** with all three heads —
policy included, and it was always separate — made **biasless**, the **policy mask
moved out to the search** so the encoder is a
pure function of the position, and the **output tensor shapes** — including `value`
as a ready `[N] fp32` — pinned as contract because the search is written against them.

The instructive failure was a register one: the prologue and epilogue sit outside the
k-loop but not outside its *register budget*, and inlined they pushed 56 bytes of
spill into the mma blocks. `__noinline__` on both fixes it. `docs/perf.md` and
`docs/cuda_walkthrough.md` §15 have the table.

⚠️ `rep` is a required input (§7.2) whose producer is descent-time detection in C1,
which does not exist. B2 takes it as an argument and the tests synthesise it; the
contract will not change when C1 fills it, but nothing yet checks that the number
arriving there is right.

## Joining the tracks

Both tracks closed on 2026-07-30, so this is where the project now lives. The
engine and the network are done, tested and measured; C1 is the only thing between
here and a first curve point, and it is the only phase with no specification.

### C1. MCTS loop, specified 2026-07-30

Specified in [`mcts.md`](mcts.md), which is normative and supersedes the
"Gumbel root selection with sequential halving" this section used to name: v0 is
AlphaZero PUCT at `n = 800`, and Gumbel is a named seam (§11) for the throughput
work rather than part of the first design. The reference implementation in
`brokefish/search/torch_impl.py` is done; the CUDA kernels are not.

Gate 1a falls at the end of it: the first sustained evals/s measured inside the
loop, with the environment live.

⚠️ **What Gate 1a still has to decide changed on 2026-07-30.** It was going to answer
"is the encoder the bottleneck it is assumed to be", and `bench/bench_loop.py` has
now answered that part: the environment is 2.2 % of a node, so the encoder is. What
Gate 1a measures instead is **what the tree costs**, which is the only unmeasured
term left. `bench_loop.py` reads 62.9k node/s with no tree at all, and that is a
ceiling: the descent, the backup, the selection and the per-node memory traffic all
come out of it. Gate 1's band is 45-50k, so the question is whether the tree costs
more than about 25 % of a node.

Everything C1 needs now exists and is tested: `csrc/movegen.cuh`, `csrc/step.cuh`,
`csrc/zobrist.cuh` and `csrc/terminal.cuh` expose warp-collective device functions
that can be called from inside a kernel, which was the constraint that put A1 ahead
of C1 in the first place. Two pieces were deliberately left here rather than in A1:
the per-game repetition ring of spec §6.3, and `rep` for spec §7.2's embedding, which
B2 currently takes as a synthesised argument.

### C2. Training loop, 3-4 days

Replay buffer, alternating self-play and gradient phases, weight versioning,
checkpoint and resume, and the euro counter. The buffer schema depends on the value
target question below.

Learner placement is settled: GPU, alternating with self-play. Measured throughput
is 5000 positions per second on the 4060 against 130 on the CPU, both fwd+bwd+AdamW
at d=256 and L=8.

### C3. Evaluation protocol, 3-4 days

See [Measuring strength](#measuring-strength) below. This is a build-once item, and
it has to span 1500 to 3000 Elo because the pilot targets AlphaGateau's range while
the full run targets the top.

### C4. Pilot, 3-5 days

Small-budget run for the first curve points and the Jones slope check. The
simulations-per-move sweep runs here, since it needs both a working loop and an Elo
protocol to be measured against. Gate 2 falls here.

### C5. Full run and write-up, 10-20 days

Mostly waiting. The write-up starts during C3.

## Measuring strength

Engine rating lists and FIDE ratings are separate scales, and the conversion between
them is contested. Two rules of thumb in common use disagree materially: one holds
that the scales intersect at 2800 with CCRL points worth about 0.70 FIDE points, and
another proposes `FIDE = (CCRL - 1600) × 0.85 + 1600`. Applied to CCRL 3000 they
give roughly FIDE 2940 and FIDE 2790 respectively. The reported agreement is best
near 2900 and worst below 1800 and above 3500.

Three consequences for the protocol.

**Preregister the threshold and the configuration.** Fix what counts as superhuman,
on which list, at which time control, on which inference hardware, at which search
budget, before the run rather than after it. Strength varies with all of them.

**Anchor to a third party.** Getting listed on CCRL or CEGT removes the methodology
argument entirely, since their procedure is established and independent. Failing
that, reproduce their conditions and say so.

**Choose the threshold with margin.** Since the conversion is disputed by roughly
150 Elo at the relevant point, a target that clears the highest FIDE rating ever
achieved under the pessimistic conversion is worth more than one that clears it
under the optimistic one.

The measurement itself: paired games from UHO or TCEC books with both colours from
each opening, error bars from SPRT or fixed-N with a stated confidence interval,
enough games that the interval excludes the threshold, no tablebases and no
pondering on either side or the same on both, hardware and time control reported for
both engines, and all PGNs published. An assertion in the code that the anchor
engine never touches a training tensor.

## Decisions with deadlines

| decision | needed by |
|---|---|
| ~~**A1/A2 order**~~ | settled: A2 first, done 2026-07-29, so A1 has a complete oracle |
| ~~**A1 priority**~~ | settled: A1 blocks C1, since a device-side descent cannot call a torch program |
| ~~**is the system env-bound?**~~ | settled by measurement 2026-07-30: no, the environment is **2.2 %** of a node |
| tree node layout | start of C1, and now the next thing to decide |
| the per-game repetition ring (spec §6.3) | start of C1; A1 left it to the search on purpose |
| value target: outcome, bootstrapped search value, or a mix | start of C2 |
| reuse factor R and training window | start of C2 |
| preregistered superhuman threshold and configuration | start of C3 |
| simulations per move | measured in C4 |
| adaptive budget by KL on completed Q | after C4, as an optimisation |

## Risks

**Second-order legality.** ✅ **Retired 2026-07-30.** Silent when wrong, and it would
have invalidated every downstream Elo number. Both engines clear perft on all six
standard positions, startpos to depth 6, and the CUDA side is bit-exact against the
PyTorch one over the whole differential set. Twenty-six single-rule mutations across
the four engine headers are each caught by at least two independent checks. What the
port also did was find a real bug in the *reference*, an int64 overflow in `terminal`
that reported a position with four legal replies as checkmate, which is the argument
for having two implementations rather than one careful one.

**The encoder rewrite.** Six to nine days is the estimate, with roughly five days of
uncertainty in either direction.

**The 8 GB budget.** Measured 2026-07-29 and closed. Self-play and training
alternate, so the two activation blocks never coexist and the peak sits near 2.4 GB:
13 MB of fp16 weights, 540 MB of inference activations at B=16384, 115 MB of trees
and history rings for 16384 games in flight, 1773 MB of training activations at
batch 1024 bf16 (measured, against 1.9 GB predicted), 90 MB of optimiser state and
about 500 MB of context and allocator overhead.

One place it does bind. A dense `[32,64]` fp16 policy target costs 4.2 KB per
position and caps the replay buffer near 1M positions. The v0 target is dense over
legal moves rather than over all 2048, and chess positions carry 30 to 40 of those,
so storing the probabilities alone in the order the legality mask produces, and
recomputing that mask from the stored board at training time, brings a position to
about 156 bytes: 312 MB for 2M positions against 8.4 GB. The cost is one movegen
call per training sample. This fixes the buffer schema in C2.

**The 5-50M games anchor.** A five-fold error is a five-fold budget error. The
pilot's Jones slope is what recalibrates it.

## The critical path

**Both tracks are complete.** A1 closed 2026-07-30 and B2 the same day, so every
kernel the system needs exists, is tested against an oracle, and has a number. The
shape of the project changes here: what is left is one loop, and almost nothing is
parallel any more.

```
C1 (5-8 d, unscoped) ──> C2 (3-4 d) ──> C4 ──> C5
C3 (3-4 d) ──────────────┘   off the path, startable now
```

**On the path**: C1, then C2, then C4, then C5. C1 is now the *only* item on it
before the pilot, it is the long pole, and it is the one thing in the project with
no written specification. That is the whole risk profile as of tonight.

**Off it, and parallelisable now**:

* C3, the evaluation protocol, which needs a network to play games and nothing
  else. Both halves of the engine now exist behind a Python API, so C3 can be
  built and debugged against random-weight self-play without waiting for C1. It is
  the largest block of work that can start immediately, and the preregistration it
  demands is a decision rather than code.
* The C1 design: the tree node layout, the descent's interface to the three
  kernels and the per-game repetition ring of spec §6.3, which A1 deliberately
  left to the search. All three are settled in [`mcts.md`](mcts.md) and built in
  `brokefish/search/torch_impl.py`; what remains is the kernels.

**Not parallelisable**: C2 against C1, since the replay buffer schema depends on
what the search emits. C4 and C5 are GPU time and do not compress.

**Ownership**: with `csrc/` quiet, the natural split is C1 to one instance and C3
plus the open decisions to the other. Both will be editing `brokefish/`, which A1
and B2 never had to share, so that is the first time the two tracks actually
collide on files.

## Calendar

Converted at the calibrated 1.3 hours per estimated day, per track. Hours are
session hours, not calendar hours.

| milestone | estimate | note |
|---|---|---|
| environment correct, perft green | ✅ 2026-07-29 | A0 and A2, day 1 |
| encoder at target with mask | ✅ 2026-07-29 | B0 and B1, day 1, forward benchmark only |
| CUDA engine complete, perft green | ✅ 2026-07-30 | A1, all four sub-steps and the binding |
| network complete, boards to logits | ✅ 2026-07-30 | B2, 62.3k evals/s |
| environment cost measured | ✅ 2026-07-30 | **2.2 % of a node**, `bench/bench_loop.py` |
| first in-loop number, Gate 1a | +12 to 20 h | end of C1, two or three sessions |
| training and evaluation in place | +8 to 12 h after C1 | C2 and C3 in parallel |
| first curve points, Gate 2 | +30 h from now | end of C4, plus the pilot's own GPU time |
| full run | 5 days of GPU, floor | see below |

⚠️ **The calibration no longer applies to what is left.** The 1.3 hours per estimated
day held across A0, A1, A2, B0, B1 and B2, which is now 17 to 23 estimated days
delivered in two sessions. Every one of them had a written specification and an
oracle to check against before a line was typed. C1 has neither: `docs/spec.md` §11
lists the search parameters as explicitly not frozen, and there is no reference
implementation to be differentially tested against. Treat the C1 estimate as an
estimate, not as 6 to 10 hours.

The full run does not follow the calibration either. 10M games × 80 plies × 32
simulations is 2.56e10 evaluations, and at the measured 62.9k node/s that is 113
hours at 100 % duty on the 4060, before the tree, the training phases or any
duty-cycle loss. Five days is a floor, the 5-50M games anchor carries a five-fold
uncertainty that applies to it, and 62.9k is a ceiling rather than the loop rate.
The same work on an H100 is under a day at $3.95/h, which is where the €100-500
estimate comes from.
