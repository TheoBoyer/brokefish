# csrc: CUDA sources

Two kernels will live here. Neither exists yet in production form; this file is
the contract they have to meet, so that whoever writes them does not have to
rediscover it.

Target architecture is **sm89 (Ada)**: `mma.sync` and `cp.async` are available,
TMA, `wgmma`, `setmaxnreg` and clusters are not. Build with
`-arch=sm_89 -O3 -Xptxas -v -lineinfo`.

## Building

`brokefish/nn/_build.py` compiles these sources into a loadable extension, and
`python scripts/check_cuda_build.py` exercises the whole path on a trivial
kernel — run it first on a new machine.

The toolkit's major version has to match the one torch was built against, which
is **13.2** for this venv. `torch.utils.cpp_extension` picks its toolkit from
whichever `nvcc` is first on `PATH` and then refuses to build on a mismatch; on
Théo's laptop that would find Ubuntu's `nvidia-cuda-toolkit` (12.0) and fail, so
`_build.py` ignores `PATH` and searches for a matching toolkit instead —
`/usr/local/cuda-13.*`, the `nvidia-cuda-nvcc-cu13` wheel inside the venv, or
`BROKEFISH_CUDA_HOME`. Installing 13.2 side by side is enough; nothing has to be
removed or resymlinked:

```
sudo apt install cuda-nvcc-13-2 cuda-cudart-dev-13-2 cuda-cccl-13-2
uv pip install -p .venv/bin/python ninja
```

`-Xptxas -v` is on by default and the build tree is kept under
`~/.cache/brokefish/ext`, because registers, spill bytes and shared memory per
kernel are the occupancy story on this card and a silent build throws them away.

## `chess.cuh`, the representation ✅

The 12-bit piece-list word, its decode/encode, and the two device helpers every
kernel needs. This is the one piece of the engine that is settled: the slot-stable
layout is what makes the network's 32 piece tokens index-aligned with the engine's
32 legality masks, so the policy mask is a device-side AND with no gather and no
host round-trip.

## `movegen`, legal move generation ✅ perft green

`movegen.cuh` holds the first order: the per-(type, square) table, pawn pushes and
captures and en passant, slider occlusion, castling rights, and the removal of
friendly targets. The same code with `kControl = true` is the attack map, which is
where `in_check` comes from. All seven outputs are bit-exact against the PyTorch
engine over the 10 000 positions of `data/cuda_testset` (`tests/tmovegen.cu`), and
one launch runs at **109.7M boards/s** on the 4060 at a batch of 260 000, with 39
registers, no spill and 11 264 B of shared memory.

⚠️ That number is the first order alone, and the first order is the cheap half.
The second-order pass replays roughly 35 candidate moves per position, so the same
design finished lands near 3M boards/s. That is arithmetic on one measurement, not
a measurement.

`movegen()` adds the brute-force second order on top: every pseudo-legal move is
applied and the resulting position is asked whether our own king stands on a square
the opponent attacks, with the whole warp working one candidate at a time so the
mutated board never leaves registers. Castling additionally tests the three squares
the king crosses. **Perft is green on all six standard positions, startpos to depth
6 = 119 060 324 nodes in 1.01 s** at 5.01M movegen/s, where the PyTorch reference
needs 1220 s; 71 registers, no spill.

Perft is the only test here whose oracle is not this repository, which is why it
gates the kernel: the differential tests compare against the PyTorch engine, so a
bug the two share is invisible to them. Porting this kernel found exactly such a
bug in the reference, an int64 overflow in `terminal`; see `docs/env.md`.

⚠️ 71 registers caps the kernel at 3 blocks per SM, so occupancy is 50 %. The
margin over what the loop needs is 111×, so that stays a note rather than a task,
and so does the pruning that would cut ~35 replays per position to a handful.

A header rather than a `.cu` because C1's descent kernel expands nodes in the
middle of a tree walk and has to call this from inside a kernel.

Requirements the finished kernel has to meet:

- **Input** `[n_positions, 32] uint16` boards plus the `[n_positions] int16`
  control word, **output** `[n_positions, 32] uint64` legality masks, bit `s` set
  iff the move `slot -> square s` is legal, and `[n_positions] uint8 in_check`.
  `in_check` falls out of the second-order pass and spec §4.1 requires it: an
  all-zero mask is checkmate when it is set and stalemate otherwise, and nothing
  else tells the two apart.
- **Oracle**: `brokefish/env/torch_impl.py`, which implements the same contract in
  PyTorch and is itself checked against python-chess (`docs/env.md`).
  `scripts/dump_cuda_testset.py` writes the flat binaries, including per-stage
  snapshots so a kernel can be validated one rule at a time.
- **Fully device-side.** Nothing about the search or the environment may touch
  the host inside a self-play step.
- Full legality, not pseudo-legality: sliders with occlusion, castling, en
  passant, promotions, pin and check filtering (the second-order pass only has to
  consider pieces that can block or expose the king).
- **Correctness bar**: differential test against python-chess, then perft to
  known node counts. There is no partial credit, and a movegen that is 99% right
  makes every downstream Elo number meaningless.
- **Performance bar**: 50-100M boards/s on an RTX 4060, which is 2-3 orders of
  magnitude above pgx on an A100 and keeps the environment under 0.5% of the
  self-play budget. Benchmark in nodes/s against Ankan Banerjee's `perft_gpu`
  (recompile it locally: its published figures are from a 2013 GTX 780 and use
  bulk counting, so they are not comparable as published) and against Stockfish
  on CPU.

The step-by-step version that was being written elsewhere as a learning exercise
is superseded and no longer feeds this file; its pawn logic survives in
`movegen.cuh`, with three shifts by a negative or out-of-range amount corrected.
What lands here is one production kernel with its own test, and the perft gate
applies to the entry point, not to the first order.

## `step`, applying one move ✅ mutation landed

`step.cuh` holds the board mutation of spec §4.2 and the fifty-move clock, at
**738M moves/s** with 19 registers, no spill and no shared memory, bit-exact over
the 322 246 `(position, move, promo)` cases of the dump (`tests/tstep.cu`). Same
warp-collective layout as the movegen, so the second order can apply a candidate
move and build the resulting attack map inside one warp with the board never
leaving registers.

⚠️ The 738M figure writes every result to memory, which the second order will not
do, so read it as an upper bound on the cost rather than the cost.

The incremental Zobrist and `irreversible` are deliberately elsewhere. Spec §6.1
hashes the en passant file only when the capture is actually legal, so the hash
needs its own king-safety test and therefore the attack map; paying it once per
candidate move would nest a second-order pass inside a second-order pass. Perft
never reads a hash, which is why the reference takes it as an optional argument and
why that half is scheduled after perft rather than before.

## `zobrist` and `terminal`, spec §6 and §4.3 ✅

`zobrist.cuh` carries `castling_rights`, `legal_ep_file`, `hash_position` and
`step_full`, which is the mutation plus the clock, the incremental Zobrist and
`irreversible`. **134.5M steps/s**, 80 registers, no spill, 17 512 B of shared
memory; `hash_position` from scratch runs at 169-206M/s. At the movegen's 5.01M
positions/s a real move costs 199.6 ns of movegen against 7.4 ns of hashing, so the
hash is 3.6 % of the per-move cost and does not need optimising.

`terminal.cuh` carries `insufficient_material`, `repetition_count` and `terminal`,
and sits on `chess.cuh` alone since it needs no attack map.

The keys are **generated on device rather than loaded**. splitmix64 advances its
state by adding a constant, so key `i` is `mix(seed + (i+1)·φ)` and all 781 are
independent, which makes the table a parallel map instead of a serial loop. That is
the property `brokefish/env/luts.py` chose splitmix64 for, and `tests/tzobrist.cu`
pins the device table against the Python one so the two cannot drift.

⚠️ **The include graph is load-bearing**: `chess.cuh` → `step.cuh` → `movegen.cuh` →
`zobrist.cuh`. spec §6.1 hashes the en passant file only when the capture is
actually legal, king safety included, so the hash needs an attack map and
`step.cuh` must stay free of `movegen.cuh`. Putting the hash in `step.cuh` makes a
cycle; putting it in the second-order pass nests a second order inside a second
order.

⚠️ `step_full` costs 5.5× `apply_move` because `castling_rights` and
`legal_ep_file` are each paid twice, before and after the move. That is inherent to
an incremental hash: both en passant states enter the key.

## `encoder`, fused transformer forward ⬜ not written

The CUDA C++ replacement for `brokefish/nn/triton_impl.py`, and the only identified path
to Gate 1 (36.4k evals/s today with the mandatory dead-token mask of spec §7.3,
45-50k needed).

The Triton version is bounded by **instruction-mix serialization under phase
synchronization**: generalist warps plus CTA barriers align every warp on the
same instruction type at the same instant, so one pipe saturates while the others
idle. The PTX shows 90 `bar.sync` per layer, because every weight tile travels
`global -> registers -> SMEM -> ldmatrix -> mma`, and each transit through shared
memory costs a barrier.

The fix, which Triton cannot express:

- **specialized warps** (loader / consumer / storer) instead of generalist ones
- **`cp.async` global -> shared with no register staging**
- **per-tile mbarriers** instead of CTA-wide barriers, so warps dephase instead
  of marching in lockstep
- SMEM paging with early release, from Hazy's "No Bubbles" megakernel; tile
  shapes and operand alternation from FlashAttention-2

⚠️ **`cp.async.mbarrier.arrive` hangs the GPU on sm89.** The validated idiom is
`commit_group` + `wait_group` + a plain `mbarrier.arrive`. A warp-specialized
prototype has already been made to run correctly on this card, so the mechanism
carries no hardware risk, and only the writing effort remains.

`docs/perf.md` has the full ledger, including the levers already measured to be
worthless. Do not re-test those.
