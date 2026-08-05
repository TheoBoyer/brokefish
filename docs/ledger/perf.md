# Performance ledger

Every measured number in the project, in one place, so the rest of `docs/` cites
rather than restates. Inference first, then the search inside the loop.

All measurements come from the RTX 4060 Laptop, 8 GB, sm89, running the
d=256 / L=8 / H=8 / FFN=1024 model (6,383,360 parameters, about 400 MFLOPs per
evaluation) in fp16 at B=16384 boards with T=32 tokens, between 2026-07-27 and
2026-07-30. Every version was validated numerically against torch at a relative
error below 5e-3. Row 11 is at B=4096, which §4.4 of `search.md` measures as flat.

⚠️ Protocol: clocks fall to 1.38-1.5 GHz under load and drift by ±3 %, so every
comparison is an order-balanced interleaved A/B. A naive before-and-after produced a
fake 4.5 % gain early in the project, where the position penalty of about 3.3 ms
dwarfed the true delta of 0.35 ms.

## The ladder

| # | version | ms | evals/s | vs baseline |
|---|---|---|---|---|
| 0 | torch eager (`nn.TransformerEncoder`, default flash attention) | ~700 | **22-24k** | ×1.00 |
| 0b | `torch.compile` max-autotune (CUDA graphs active, verified in the profiler) | ~700 | 21-23k | **×1.00, zero gain** |
| 1 | `EFFICIENT` attention instead of flash (one line) | ~658 | 24.9k | ×1.04 |
| 2 | fused block v2 (3 Triton kernels) | ~482 | ~34k | ×1.45 |
| 3 | fused encoder, BC=128 (1 CTA/SM) | 482 | 34.0k | ×1.45 |
| 4 | fused encoder, BC=64 (2 CTAs/SM) | 453 | 36.2k | ×1.55 |
| 5 | + `maxnreg=168` | 440.8 | 37.2k | ×1.79 vs eager |
| 6 | + dead-token mask (§7.3) | **449.6** | **36.4k** | **×1.78** vs eager |
| 7 | + fp16 accumulation (§ roofline), best Triton | 384.4 | 42.6k | ×2.12 vs eager |
| 7b | CUDA C++, weights staged through SMEM (first correct build) | 417.7 | 39.2k | ×1.97 — **slower than Triton** |
| 8 | CUDA C++, weights direct to registers | 297.8 | 55.0k | ×2.71 vs eager |
| 9 | **+ ping-pong prefetch, 2 CTAs/SM, row-wise residual, current default** | **267.5** | **61.3k** | **×3.03** vs eager |
| | GO target | ~350 | 45-50k | **cleared** |
| 10 | B2: boards in, policy/promo/value out — **a different measurement**, see below | **263.0** | **62.3k** | ×3.25 vs the torch full model |
| 11 | **Gate 1a: the whole MCTS of `search.md` §6 on device** at `n=800`, `B=4096`. **Another different measurement**, see below | 57 220 ms/move | **57.0k useful** | tree costs 4.4 % |
| 12 | B2 + **e4m3 FFN**, activation scale per 128-element k-tile | — | — | **×1.154** over row 10 |
| 13 | B2 + **e4m3 FFN, one scale per row** (`gemm_fp8_row`), current fp8 default | — | **82.2k** | **×1.214** over row 10 |
| 14 | **Gate 1 with the fp8 encoder**, the whole of `search.md` §6 on device at `n=800`, `B=4096` | 45 470 ms/move | **71.5k useful** | tree costs 5.4 % |
| 14b | the same at `n=128`, which is what self-play ships since Track E | 6 910 ms/move | **75.4k useful** | tree costs 6.1 % |

Row 14 replaces row 11 as the gate number and is the same measurement: `bench_search`,
B = 4096 at 30 random plies, 4087 distinct live positions, 3 interleaved rounds. The
same-day fp16 control was run rather than differencing against July:

| n | fp16 useful/s | fp8 useful/s | in-loop ratio | fp16 encoder | fp8 encoder |
|---|---|---|---|---|---|
| 128 | 58 948 | **75 401** | **×1.279** | 62 646 | 81 068 |
| 800 | 57 492 | **71 542** | **×1.244** | 60 682 | 76 036 |

Today's fp16 at n=800 reads 57 492 against row 11's 56 996 from 2026-07-30 — 0.9 %
apart, which is what validates comparing the two dates at all.

⚠️ Two things about row 11 that are **not** the encoder, both confirmed by the control
reproducing them. Mean depth fell from 5.8/28 to 2.7/9: that is the FPU fix of
2026-07-31, since AGZ's literal `Q = 0` in a `[0, 1]` tree makes an unvisited child look
like a loss and drives the search deep. And §4.3 truncation went from "443 nodes
dropping 2.317 of prior mass" to zero, because `SearchConfig.E` moved 64 → 96.
`bench_search` had that cap hardcoded in its own log line and so printed `(E = 64)`
next to 76 live edges and zero truncations, which cannot both be true; now it prints
`SearchConfig().E`.

The in-loop fp8 ratio (×1.244 to ×1.279) is **higher** than `speed.py`'s ×1.215 to
×1.220 on the same kernels. Unexplained. The runs differ in heat soak (45 s rounds here
against 20 tight reps) and in the net (random init here, a trained checkpoint there);
neither should move throughput and I have not measured which does.

Rows 12 and 13 are ratios and not absolutes on purpose: they come from `speed.py`, an
order-balanced six-round A/B against the fp16 kernel in the *same process*, and the
absolute number moves with the card's temperature far more than the delta does. Two
consecutive runs read ×1.214 (fp16 67.7k, fp8 82.2k) and ×1.218 (fp16 65.8k, fp8
80.2k) — the ratio is stable to 0.3 % while the absolute moved 2.8 %.

### Where the time goes, measured rather than subtracted

The 79.4 / 20.6 split below is a **residual** — the whole kernel's 25.5 TFLOPS against
32.1 for the four GEMM shapes alone. `bench/bench_phases.py` measures the split
directly, from `clock64` deltas at barriers the kernel already has, and the two agree
for the kernel the residual described:

| group | fp16 | fp8 (ships) |
|---|---|---|
| matmul | **81.3 %** | **72.9 %** |
| staging + barriers | 8.9 % | 8.7 % |
| normalisation | 5.4 % | 6.4 % |
| attention (2.0 % of the flops) | 3.1 % | 3.8 % |
| fp8 quantisation | — | **6.8 %** |
| prologue + epilogue | 1.2 % | 1.4 % |

fp16 measures 81.3 % against the 79.4 % subtraction — two points apart, the direct
number higher, which is what `shapes_ab.cu`'s hot L2 and 48 CTAs would flatter.

⚠️ **These are shares of elapsed time, not marginal costs, and the difference is not
academic.** A CTA that stalls at a barrier or on shared memory hands its SM to the
other resident CTA, so a phase's cycles include work that is not its own and deleting
that work frees nothing unless the SM had idle issue slots — which it does not. Fusing
the FFN handoff removed 11.3 % of measured phase cycles and **0.0 % of runtime**
(`docs/journal/2026-08-05-ffn-handoff-negative.md`). The currency for a speedup on this
kernel is **issue slots**: `encoder_kernel<true>` is 4056 SASS instructions of which
192 are `HMMA`, and an opcode histogram from `cuobjdump -sass` is what decides whether
a candidate the table below flags is real. Use the table to find candidates, not to
size them.

⚠️ **The subtraction does not describe the shipping kernel.** fp8 is **72.9 %** matmul,
because it halves the FFN's matmul time and then pays 6.8 % of quantisation the fp16
kernel never sees. Sizing any plan for the fp8 kernel against 79.4 / 20.6 gets the
answer wrong in both directions; `docs/journal/2026-08-05-phase-decomposition.md` shows
which. Stable to 0.1 point between 512 and 4096 CTAs. The instrumented kernel is 1.03x
to 1.06x slower than the shipped one and `PROF=false` is byte-identical SASS, verified
with `cuobjdump`.

### What binds it, measured with `ncu` rather than argued

`/usr/local/cuda-12/bin/ncu` on `encoder_kernel<true>`, 512 CTAs, fp8:

| metric | value |
|---|---|
| `gpu__dram_throughput` | **0.50 %** |
| `lts__t_sector_hit_rate` | 99.81 % |
| `lts__throughput` | 48.69 % |
| `l1tex__throughput` | 63.85 % |
| `sm__throughput` | **65.31 %** |
| `smsp__inst_executed` | 248,705,024 = **60,720 per warp** |

⚠️ **DRAM is idle at 0.5 % and L2 hits 99.8 %, so weight *bandwidth* is not the
constraint.** But `--set full` names what is. Stall cycles per issued instruction, of
13.03 total:

| stall reason | cycles | share | what it is |
|---|---|---|---|
| `long_scoreboard` | **3.61** | **27.7 %** | global loads that miss L1 |
| `math_pipe_throttle` | **3.01** | **23.1 %** | the tensor pipe, busy |
| `wait` | 2.37 | 18.2 % | fixed-latency register dependencies |
| *selected* (issuing) | 1.00 | 7.7 % | |
| `short_scoreboard` | 0.92 | 7.1 % | shared memory |
| `barrier` | 0.77 | 5.9 % | `__syncthreads` |
| `not_selected` | 0.63 | 4.8 % | |
| `mio_throttle` | 0.41 | 3.1 % | |

and the two numbers that explain them:

| | |
|---|---|
| `l1tex__t_sector_hit_rate` | **1.70 %** — 98.3 % of global load sectors miss L1 |
| `sm__pipe_tensor_cycles_active` | **63.81 %** of peak, and it *is* `sm__throughput` |
| `smsp__issue_active` | 29.58 % of cycles |
| `sm__warps_active` | 32.1 % of peak = 16 of 48 warps |
| local (spill) sectors | 1.6 % of LSU sectors — **not** a factor |

**The bottleneck is the weight stream and the tensor pipe, in that order and of
comparable size.** Every CTA marches the whole 8.4 MB fp8 weight slab through an L1 that
holds ~28 KB, because 2 CTAs x 50,176 B of shared memory take 100,352 of the SM's 128 KB
unified L1+SMEM — so the shared-memory budget that buys the second CTA is *also* what
starves L1, and 98 % of weight loads pay L2 latency. Meanwhile the mma is at 64 % of
peak, which caps the whole kernel at about **1.57x** however perfect everything else gets.

⚠️ That is a complete explanation of three failed attempts in one day: removing shared
memory work (`short_scoreboard`, 7.1 %), arithmetic (`math_pipe_throttle` is the tensor
pipe, not the ALU — `pipe_fma` is 10.9 % and `pipe_alu` 15.3 %) or spills (1.6 % of
sectors) touches none of the top two. Occupancy would, and `launch__occupancy_limit`
reports **registers and shared memory both cap at 2 blocks** — so 3 CTAs/SM needs 85
registers *and* 34 KB of shared memory, against today's 128 and 49 KB.

⚠️ **Row 13's win is a register story, not an arithmetic one.** fp8 halves the mma
work either way; what row 12 could not spend was registers. `__launch_bounds__(THREADS,
2)` allows exactly 65536 / (2 × 256) = 128 of them, and row 12's `float acc[2][4][4]`
running total — needed only because the scale changed every 128 k-elements — spilled
960 B against the fp16 path's 64 B. **Giving it the registers instead is a measured
loss**: at 1 CTA/SM it uses 200 registers, spills nothing, and runs **×0.978**, because
16 warps per SM is the whole of this kernel's latency hiding. One scale per row removes
the second accumulator instead of feeding it: 200 B of spill, ×1.214.
`docs/journal/2026-08-05-fp8-per-row.md`.

Rows 0-9 all measure the same thing, activations in and activations out, and are
comparable to each other. **Row 10 is not on that ladder**: it takes 64 bytes of
board instead of a 16 KB activation tile and emits logits instead of activations, so
its baseline is the whole torch model rather than `nn.TransformerEncoder`. It is
faster than row 9 while doing more arithmetic, for reasons in the B2 section below.
`--path backbone` still reproduces rows 0-9 unchanged.

Row 9 is a 5-round interleaved run in which torch eager read 19.4k masked and the
Triton kernel 42.7k, so the transferable figures are **×3.03 over eager and ×1.43
over Triton**. At 0.411 GFLOP/eval that is 25.2 TFLOPS, or **71 % of the 35.5
TFLOPS** fp16-accumulate issue rate measured on this card.

Row 8 was a 5-round interleaved run in which torch eager read 20.3k and the Triton
kernel 42.8k: ×2.71 over eager, ×1.29 over Triton, 22.6 TFLOPS, 64 % of ceiling.

Row 7 is a 6-round interleaved run where eager read 20.1k on a hot card; the
isolated A/B that produced it, against the fp32-accumulate kernel on the same
weights, put it at 43.7k and ×1.19. Quote the ratio, not the absolute.

The ladder spans three sessions, so only the final ×2.12 is a same-run A/B against the
same-day eager baseline. Rows 5 and 6 come from the same interleaved run, where eager
read 20.5k on a hot card against the 22-24k it reaches cold. Absolute evals/s move
with temperature by more than most of the deltas here, so the ratios are the
transferable part.

## Row 11, Gate 1a, and the clock 2026-07-30

Row 11 is not on the ladder either, and for a larger reason than row 10: it is a
whole self-play move rather than a forward pass, so it runs `n + 1` encoder calls
plus the descent, the expansion and the backup of `search.md` §6, and its unit is
the number the project is gated on. The search in the loop, below, has the full table, the
sweep over `n` and what the §15 counters said. Three things belong here.

**The tree costs 4.4 %, flat in `n`.** 2.84 ms per simulation at `n = 32`, 2.88 at
128, 3.00 at 800, against an encoder that takes 65 ms for the same batch. The 5 %
rise across the sweep is the mean tree depth going from 4.1 to 5.8. The baseline is
`n + 1` encoder calls and not `n`, since a move is `root_init` plus `n` simulations;
charging the tree for the extra evaluation makes the overhead look like it falls
with `n`, which it does not.

**The encoder is slower inside the loop than in its own benchmark, and this is the
transferable finding.** §4.4 of `search.md` measured 64 142 evals/s at `B = 4096`. The
same call in the same process, run 801 times back to back as part of this campaign,
gives **59 852/s**, 6.7 % lower. The card sits at **1230-1290 MHz and 82 °C** through
a 57-second block where a short encoder benchmark holds the 1.38-1.5 GHz this
document assumes. Nothing about the kernel changed.

⚠️ **Every absolute number in this file was taken in blocks of a few hundred
milliseconds.** They are all upper bounds on what the same kernel does inside a
self-play generation, and the discount measured here is 6.7 %. The interleaved
protocol keeps the *ratios* valid regardless, which is the reason it exists.

**Predicted 1.7 ms of tree, measured 3.00.** Scaled to the clock it ran at the
prediction was 1.84 ms, so the gap is 1.6×. The prediction's dominant term was the
two `movegen` calls, priced from the perft kernel's 5.01M boards/s. That kernel uses
71 registers and fits three blocks per SM; inside `descent` the same code runs in a
96-register kernel at two blocks per SM. Lower occupancy on the dominant term is the
leading candidate and it is **not measured**: a 1.2 ms gap inside a 4.4 % overhead
does not justify the profiler run, and saying which it is without running one would
be a guess dressed as a finding.

## Why the first CUDA kernel was slower than Triton, 2026-07-29

Row 7b is worth keeping in the ladder because the rewrite was justified by a
prediction that turned out to have the wrong sign. The argument for leaving
Triton was that it spends 90 `bar.sync` per layer staging weight tiles, and that
hand-written staging would cut that to 14. The first correct CUDA kernel ran
**202 barriers per layer** — more than twice what it was replacing — and lost by
7.7 %.

The mechanism was four lines in `gemm_full`: a single weight buffer in shared
memory, so every k-tile needed one barrier to stop the readers and another to
publish the new tile, and `KTILE=32` against `kdim=256` made that eight tiles per
GEMM. Nothing overlapped. Counting from the SASS rather than from the design
document would have caught it before the benchmark did.

The static census of that build, from `ptxas -v` and `cuobjdump -sass`:

| | row 7b | row 8 |
|---|---|---|
| barriers / layer | 202 | **14** |
| registers / thread | 255 (register file exhausted) | 134 |
| STS / layer | 284 | 12 |
| shared memory / block | 89.5 KB | 49.5 KB |

## Weights never enter shared memory, 2026-07-29

The fix was to stop moving weights through shared memory at all.

A weight element is read once per CTA and never reused inside it — a board is 32
tokens, so each weight participates in 32 MACs and then is done. Staging it in
shared memory buys nothing and costs three passes over the LSU (`LDG`, `STS`,
`LDSM`) where one would do, plus the barriers protecting the buffer. Measured
against 1.58 MB of weights actually consumed per layer, the staged version moved
**4.7 MB per layer through shared memory**.

An mma B fragment is only 4 bytes per lane and the lane→element map is fixed at
compile time, so the host pre-permutes each matrix into exactly that order
(`pack_b` in `brokefish/nn/cuda_impl.py`) and the kernel reads a whole fragment
pair with one coalesced 128-bit load per lane, straight into the registers the
mma consumes. No shared memory, no `ldmatrix`, no barrier. The A operand still
lives in shared memory, because all eight warps contract over the same
activations and there is nowhere cheaper to put it.

Second-order consequences, all of which mattered:

* Shared memory per block fell to 49.5 KB, because the weight tile is gone and
  the V-transpose scratch now aliases the FFN hidden buffer.
* Registers fell from 255 to 134, since nothing stages through them any more.
* The 14 remaining barriers are the real cross-warp handoffs: LayerNorm writes
  the activation buffer by row while the GEMMs read it by column, and vice versa
  for `store_frags`. `gemm_full` had been providing these implicitly with its
  leading `__syncthreads()`; removing it silently broke all three handoffs at
  once, which is worth remembering — a barrier that is load-bearing for someone
  else's dependency is not documented anywhere.

### Unrolling the k-loop makes it slower

Pinned with `#pragma unroll 1` after measurement, not left to the compiler:

| unroll | registers | stack frame | cuda + mask | vs Triton, same run |
|---|---|---|---|---|
| 1 | 134 | 0 | **297.8 ms** | ×1.29 |
| 2 | 136 | 0 | 298.9 ms | ×1.29 |
| 4 | 166 | 0 | 357.5 ms | ×1.07 |
| full | 246 | 192 bytes | not measured | — |

Eight hoisted B loads have live ranges long enough to cost more than the overlap
they buy, and full unroll spills into local memory outright.

### Unrolling it still makes it slower, for a new reason

Re-measured at row 9, where 2 CTAs/SM caps the budget at 128 registers/thread:

| outer unroll | registers | stack frame | spills | vs row 9 |
|---|---|---|---|---|
| 1 | 116 | 0 | none | **baseline** |
| 2 | 128 | 8 bytes | 4 B store / 4 B load | −6.85 % |
| full | 128 | 192 bytes | none (local instead) | −0.84 % |

At row 8 the argument was live ranges; now it is simply that there is no room.

## Row 9 — the optimization round, 2026-07-29

Five changes, each measured on its own with an order-balanced interleaved A/B in
a standalone harness (no torch in the process, so `ncu` attaches cleanly and a
variant rebuilds in seconds). Every one is **bit-identical** to row 8 on 16384
boards, which is the correctness gate: row 8 is already validated against torch,
so a variant is compared against row 8 directly and a moved fragment shows up as
a large error rather than a small one.

| # | change | delta | cumulative |
|---|---|---|---|
| a | ping-pong the weight prefetch; packed row pitch as a template argument | **+4.63 %** | +4.6 % |
| b | unpad `bufA` to 50,176 B, which fits two CTAs per SM | **+5.43 %** | +10.3 % |
| c | residual staged in fragment order and added row-wise | **+0.90 %** | +11.3 % |
| d | `gamma`/`beta` and the alive mask as vector loads | +0.42 % | **+11.77 %** |

The measured noise floor of the harness is ±0.25 % on a 4-round average
(identical kernels compiled twice read −0.23 %).

### (a) The k-loop was 71 % overhead instructions

The row 8 loop body, from the SASS, per k-step: **16 HMMA against 15 MOV, 7 IMAD,
4 LEA, 2 ISETP, 1 SHF, 1 IADD3**, plus 4 LDSM and 4 LDG. The MOVs were the
`cur[p] = pre[p]` copy that kept the prefetched weights alive while the next
group loaded over them; the integer ops were weight-address arithmetic that never
got strength-reduced because the packed row pitch `nk32` arrived as a runtime
argument. Two buffers alternating instead of one buffer plus a copy costs the same
32 registers and deletes the MOVs; `NK32` as a template parameter turns
`p * NK32 * 32` into an LDG immediate offset. `sm__inst_executed_pipe_alu.sum`
fell by 66 % (209.5M → 71.6M) and the tensor pipe went from 69.4 % to 72.4 %.

### (b) 2 CTAs/SM, which row 8 measured as worthless

Row 8's note said occupancy had stopped being the constraint. That was true of
row 8 and false of row 9a: once the ALU work was gone, `long_scoreboard` — waiting
on global loads — became the largest removable stall, and occupancy is exactly
what hides memory latency.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` puts the threshold at **50,176 B**
per block (128 B granularity, so 100,352 B usable of the SM's 102,400). Row 8 sat
at 50,688 — over by 512 bytes. The 512 bytes come out of `bufA`, the residual
stream, which is the only buffer no `ldmatrix` ever reads and therefore the only
one whose 8-half row pad is not required: at pitch 264 the eight rows a lane group
touches land in eight bank groups, and `bufA` needed that only for the scattered
residual read-modify-write.

Measured: occupancy 16.66 % → 33.10 %, tensor pipe 72.4 % → **81.0 %**.

This also inverts the register budget. Two blocks of 256 threads on a 65,536-entry
register file is **128 registers per thread**, and the kernel uses 116. Every
remaining traffic optimization — fusing Q/K/V into one k-loop, double-buffering
the A fragments, deeper weight prefetch — costs more registers than that leaves,
so the occupancy win closes those doors. That is the trade, and it is worth it:
2 CTAs/SM is +5.4 % and the largest of them was worth under 1 %.

### (c) The residual had to stop being scattered

Unpadding `bufA` made the in-place fragment-order residual collide 8 ways: bank
conflicts went from 7.2M to 46.2M and L1 wavefronts rose 14 %. Staging the
projection with `store_frags` into a padded buffer, where the write is
conflict-free, and adding it into `bufA` a row at a time costs 20 shared accesses
per warp against 32, with no conflicts — and needs **no extra barrier**, because
every row of `bufA` belongs to exactly one warp and the scratch buffer is dead at
both points where it is needed. The layer keeps its 14 barriers.

### What did not work, measured

| idea | result |
|---|---|
| reorder the mma so no accumulator is revisited within 8 instructions | ±0 — ptxas already scheduled it |
| double-buffer the A fragments (`ldmatrix` one k-step ahead) | +0.56 % at 1 CTA/SM, **±0 at 2** — the extra warps already cover LDSM latency |
| unroll the k-loop by 2 or fully | −6.85 % / −0.84 %, see the table above |
| deeper weight prefetch, fuse Q/K/V, `HCHUNK` 512 | not measurable: all exceed the 128-register or 50,176-byte budget |

### What is left

At **71 % of the mma issue ceiling** the kernel is bounded by
`math_pipe_throttle` (8.61 cycles per issue-active, the good stall — the tensor
pipe refusing new work) ahead of `long_scoreboard` (5.39) and `wait` (3.23). The
absolute headroom left in this structure is +23 %, and every lever that would
reach it is blocked by the register budget that 2 CTAs/SM imposes.

Two boards per CTA remains the one change that would break the deadlock: it
halves the weight traffic *per board* rather than trading one resource for
another. It needs 4 m-tiles of accumulators and a `bufA`/`bufB`/scratch triple at
64 rows, which is 101,376 B — one CTA per SM again, so it is a bet against this
round's main result and a redesign rather than a tweak.

## B2, and the trap in the register cap, 2026-07-30

The embedding gather and the head epilogue run **once per board against eight
layers of loop**, so their own speed is irrelevant. Their register demand is not,
and that is not obvious: ptxas allocates one budget for the whole kernel, so cold
code that wants registers does not slow *itself* down — it shrinks what the k-loop
has left and makes **that** spill.

Measured with a standalone probe (`nvcc` on the kernel alone, no torch headers, so
a variant compiles in seconds), each row adding to the one above:

| variant | registers | spill stores | stack frame |
|---|---|---|---|
| pre-B2 kernel, new signature only | 113 | 0 | 0 B |
| + final norm and heads, inlined | 128 | 12 B | 208 B |
| + embedding prologue, inlined | 128 | **56 B** | 240 B |
| both as `__noinline__` ABI calls | 128 | 12 B | 208 B |

The prologue alone accounted for 44 of the 56 bytes, and the spilled loads and
stores landed *inside the mma blocks* — the k-loop paying for code it never
executes. `__noinline__` gives each an ABI frame of its own and hands the loop its
allocation back. The remaining 12 bytes are the cost of the kernel containing any
ABI call at all; they were not worth chasing further, because the throughput
measurement below says the whole change is free.

Two smaller things fell out of the same pressure. The gather accumulates one table
at a time instead of loading all five and then adding (16 live registers instead of
40, same arithmetic), and the two per-position tables — clock and repetition — are
summed into one vector per board rather than per token. That second one is why spec
§7.2's summation order is `(square + type_special + color_turn) + (clock + rep)`
rather than a plain left fold: in fp16 the grouping is part of the answer, so it had
to become normative rather than incidental. `tests/test_b2.py` holds the gather to
**bit-identical**, not to a tolerance, precisely because nothing here rounds.

The prediction was "+0 to +4 registers and no spill", made on the grounds that both
blocks sit outside the loop. Wrong, and instructively so: *outside the loop* is not
the same as *outside the loop's register budget*.

### What it costs, measured

The backbone path — activations in, activations out — is unchanged code, so it is
the control for the register question. In one interleaved run against the unchanged
Triton kernel:

| | before B2 | after B2 |
|---|---|---|
| cuda + mask | 267.48 ms, 61.3k | 266.02 ms, 61.6k |
| triton + mask | 383.34 ms, 42.7k | 379.21 ms, 43.2k |
| cuda / triton | ×1.433 | ×1.426 |

Both implementations read 1.1 % faster on this run than on the row-9 one, which is
the clock drift this protocol exists to cancel and the reason only the ratio is
quotable across runs. That ratio moved by 0.5 %, inside the ±0.25 % per-candidate
noise floor once compounded. The register change is free.

### The B2 path is faster than the backbone benchmark

`bench/bench_model.py --path full`, 16384 boards, five interleaved rounds:

| candidate | ms | evals/s | vs torch |
|---|---|---|---|
| torch full model | 853.42 | 19.2k | ×1.00 |
| triton full | 407.50 | 40.2k | ×2.09 |
| **cuda full** | **262.99** | **62.3k** | **×3.25** |

62.3k boards-to-logits against 61.6k for activations-in-activations-out, doing
strictly more arithmetic. Two things pay for it: the `[32,256]` activation tile no
longer crosses HBM in either direction, and the backbone benchmark's `out.copy_(x)`
— unavoidable for an in-place kernel — goes away with it. Roughly 4 ms of the batch,
which is about what 536 MB of device-to-device copy costs here.

⚠️ **The Triton column is 7.5 % slower than its own backbone row (407.5 against
379.2), not the "well under 1 %" predicted.** Running the gather, the final norm and
the three heads as torch ops costs 28 ms per 16384 boards: several launches plus a
full `[N,32,256]` round trip plus a cuBLAS `[32,256]x[256,96]`. The decision stands —
Triton exists as the backbone A/B control and the only fp32-accumulate path, and
`--path backbone` still compares like with like — but the asymmetry is 7.5 %, not
noise, and any claim that reads the two full rows as a kernel-to-kernel comparison
is reading 0.3 % of arithmetic plus 7.5 % of torch overhead.

## The dead-token mask, 2026-07-29

[Spec §7.3](../reference/spec.md#73-dead-tokens) makes the mask mandatory, and rows 0 through 5
were all measured without it, so the GO target was being chased against a kernel
that was not the one we ship.

Four ways of killing dead keys in the `[32,32]` score tile were compiled from one
source through a `MASK_MODE` constexpr, so the unmasked path stayed bit-identical
and the A/B stayed honest. All four are numerically equal to torch under
`src_key_padding_mask` at 2.4e-3 on live tokens.

### Equivalence

`tests/test_model.py` sweeps eleven occupancies from full boards down to bare kings,
covering contiguous, alternating, scattered and per-board-varying dead sets, since
those put different bit patterns in each lane's slice of the predicate tile and a
single random mask would miss a layout mistake. Board counts run 32 to 4096.

On live tokens the maximum absolute delta against torch fp16 is **2.34e-2**, flat
across every occupancy, on outputs whose largest magnitude is 12.1. That is 3 fp16
ulps, and the relative error is 1.5e-3 to 1.9e-3.

Running the same rounded weights through an fp32 encoder separates the two error
sources. The kernel sits 1.75e-2 from fp32 at worst, torch fp16 sits 2.22e-2 from
it, so the fp16 delta above is arithmetic noise and the kernel is on the better side
of it. That comparison is an assertion rather than an observation.

Two further properties are asserted. Dead rows are inert: overwriting them with
values up to 65504, the fp16 ceiling, returns every live output bit-equal. And with
every slot alive the masked kernel reproduces the unmasked one to 2 ulps rather than
bit-exactly, because the dynamic predicate changes instruction scheduling, which
changes which multiply-adds get contracted.

### Dead tokens have to be finite

The mask zeroes the attention weight of a dead key, and `dot(p, v)` then multiplies
by that zero rather than selecting on it, so `0 * inf` is `NaN` and one infinite dead
token poisons every live token on its board. Measured: dead rows at 65504 are inert,
dead rows at `inf` or `NaN` return `NaN` everywhere.

Spec §7.2 builds a dead token by summing the same embedding tables as a live one, so
finiteness holds by construction. It is a constraint on whatever writes dead slots in
the B2 prologue, and `test_dead_tokens_must_be_finite` pins it.

### Mutation testing

Nine variants of `model.py`, each a plausible mistake, are compiled and run against
the suite by `tests/test_mutation_mask.py`. All eight defects are killed and the
control survives. It recompiles the kernel once per mutant, so it is opt-in:
`pytest tests/ --mutation`.

| mutant | killed by |
|---|---|
| mask rows instead of columns | 5 of 7 tests |
| mask rows as well as columns | 5 |
| predicate inverted | 6 |
| alive read one slot off | 6 |
| every CTA reads board 0 | 5 |
| mask applied on layer 0 only | 5 |
| mask applied on head 0 only | 5 |
| dead values zeroed instead of dead keys masked | 5 |
| dead tokens attend to themselves (control) | survives, correctly |

Two results changed how the tests are written.

Masking rows produces `NaN` rather than a wrong number, because a fully masked row
softmaxes `-inf` against a `-inf` maximum. That is the same mechanism as the
finiteness constraint above, arriving from the other direction.

The per-CTA read of board 0 is caught by only 4 of the 11 occupancies, namely the
five random ones and the per-board-varying one. Every occupancy that applies the same
pattern to all boards misses it, which is what the pattern variety buys: a suite
built on `kings only` or `contiguous dead block` alone would ship this bug.

The control matters as much as the kills. Letting a dead token attend to itself
changes dead rows and nothing else, and the suite lets it through, so the tests are
keyed to what the spec makes observable rather than to the kernel's incidental
output.

| variant | regs | spill bytes | delta |
|---|---|---|---|
| no mask, the row-5 kernel | 168 | 66 | reference |
| `tl.where` on a `[BM,BM]` predicate built once from a `[BM]` alive vector | 168 | 76 | -1.24 % |
| additive `-inf` bias, `[BM]` fp32 broadcast into the score tile | 168 | 84 | -1.62 % |
| additive `-inf` bias, `[BM,BM]` loaded broadcast to skip the layout conversion | 168 | 76 | -1.68 % |
| multiplicative 0/1 after the `exp`, max left over all keys | 168 | 76 | -1.58 % |

The four masked variants sit inside a 0.44 % band, which is below this machine's
drift, so the choice between them is free. A 12-round duel of the first two put the
cost at **0.97 %**, and the production benchmark at 0.77 %. Call it 1 %.

The prediction on record was 3 to 6 %. The measurement is 1 %, and the reason is the
third bottleneck below: at No-Eligible 89.9 % the attention phase is stalled on
barriers rather than on issue slots, so 1024 extra `selp` per head land in existing
stall shadow. What the mask does cost shows up as spill bytes, 66 to 76, which is
`maxnreg=168` paying for the extra live values.

`tl.where` won on grounds other than speed. It merges into the block-diagonal
predicate that the kernel already evaluates, so it stays one select per element when
`BM > T` puts several boards in a CTA, where the additive variants would need both a
select and an add.

## The roofline, measured 2026-07-29

Taken before writing the CUDA kernel, to find out what the ceiling actually is
rather than assume it. Two microbenchmarks, both reproducible from the scratch
directory: raw `mma.sync` in a register-resident loop with eight independent
accumulators, and cuBLAS through `torch.mm` on the exact shapes.

| what | rate |
|---|---|
| `mma.sync.m16n8k16.f32.f16.f16.f32` (what we run today) | **18.0 TFLOPS** |
| `mma.sync.m16n8k16.f16.f16.f16.f16` | 35.5 TFLOPS |
| `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` | 41.6 TFLOPS |
| cuBLAS 8192³ fp16, the friendliest shape there is | 19.0 TFLOPS |
| cuBLAS on the four real per-layer shapes at M = 524288 | 16.1-16.5 TFLOPS |
| HBM, plain copy | ~155 GB/s |

**fp32 accumulation is half rate on GeForce Ada.** The 35-40 TFLOPS this project
had been reasoning with is the fp16-accumulate rate; the kernel we ship
accumulates in fp32 for torch parity, so its ceiling is 18, not 36.

### What that does to the numbers

The network is 0.411 GFLOP per evaluation, so evals/s converts to TFLOPS by
×0.411. The current kernel at 36.4k evals/s is running at **15.0 TFLOPS, which is
83 % of the raw mma issue rate** and 91 % of what cuBLAS extracts from the same
shapes with a batch dimension 16384× friendlier.

| evals/s | TFLOPS needed | as a fraction of the 18.0 ceiling |
|---|---|---|
| 36.4k, today | 15.0 | 83 % |
| 40k | 16.4 | 91 % |
| 45k, the GO target | 18.5 | **103 %** |
| 50k | 20.6 | 114 % |

**The GO target is above the machine's fp32-accumulate tensor-core issue rate.**
No amount of warp specialisation reaches it: removing every barrier, every
`ldmatrix` and every byte of memory traffic still leaves 43.8k as the arithmetic
limit. A realistic perfect kernel lands near 40-44k.

Three ways out, and they are the real content of B1: fp16 accumulation (ceiling
35.5), fp8 e4m3 (ceiling 41.6, a bigger change), or a re-scoped gate. The
accumulator type is a template parameter in the CUDA kernel precisely so the
choice is made on measurement.

### fp16 accumulation costs less error than fp16 storage already does

Measured by emulating `mma...f16.f16.f16.f16` through the whole 8-layer stack —
each k=16 product delivered into an fp16 accumulator — against an fp32 encoder
holding the same rounded weights. The control is what makes the row readable: the
same harness with an fp32 accumulator has to come back clean, and does.

| accumulator | max abs vs fp32 | relative |
|---|---|---|
| torch fp16 itself, fp32 accumulate | 1.56e-2 | 1.36e-3 |
| fp32 accumulate, same harness (control) | 1.53e-3 | 1.34e-4 |
| fp16, promoted to fp32 every 2 k-steps | 2.71e-3 | 2.36e-4 |
| fp16, promoted every 4 | 2.87e-3 | 2.50e-4 |
| fp16, promoted every 16 | 5.56e-3 | 4.85e-4 |
| **fp16, never promoted (K=1024 in one go)** | **9.59e-3** | **8.37e-4** |

Even the worst case sits **6× under the 5e-3 bar**, and below what torch's own
fp16 forward costs. The contraction length is not the problem: activations are
LayerNormed, so the summands are O(1) and K=1024 accumulates to O(30), nowhere
near fp16's range.

⚠️ The emulation keeps activations in fp32 between stages where the kernel stores
fp16, so the absolute figures are optimistic. The transferable part is the
comparison: switching the accumulator costs about 6× the control's error, which
is still less than one fp16 store per stage costs. Confirm on the real kernel.

⚠️ The "fp16 accumulation, 0 %" row in the dead-ends table below does **not**
refute this. It was measured through `tl.dot`, and a 0 % delta at 83 % of the
fp32 ceiling is itself the anomaly: either Triton kept accumulating in fp32 and
only cast the result, or the kernel was barrier-bound at the time. Re-measure it
in CUDA before believing it.

### fp16 accumulation on the current Triton kernel, measured

The emulation above said it should be numerically safe. This is the kernel
itself, one source compiled twice through an `ACC` constexpr so the two paths
differ in nothing else, at B=16384 masked, 8 interleaved rounds with the order
reversed every other round.

| variant | ms | evals/s | vs production | TFLOPS |
|---|---|---|---|---|
| production, fp32 accumulate | 445.5 | 36.8k | ×1.00 | 15.1 |
| variant, fp32 accumulate (restructure control) | 446.8 | 36.7k | ×1.00 | 15.1 |
| **variant, fp16 accumulate** | **375.3** | **43.7k** | **×1.19** | **17.9** |
| variant, fp16 accumulate, `maxnreg=128` | 408.0 | 40.2k | ×1.09 | 16.5 |

The control matters: the phase-3 restructure the variant needs costs 0.3 %, below
this machine's drift, so the ×1.19 is the accumulator and nothing else. Dropping
`maxnreg` to 128 to exploit the smaller accumulators loses 9 %, so 168 stays.

Accuracy over five weight draws, 1024 boards, worst live-token delta against an
fp32 encoder holding the same rounded weights:

| seed | torch fp16 | fp16 accumulate | relative | ratio to torch |
|---|---|---|---|---|
| 0 | 1.74e-2 | 1.84e-2 | 1.50e-3 | 1.06 |
| 1 | 1.91e-2 | 1.62e-2 | 1.28e-3 | 0.85 |
| 2 | 1.70e-2 | 1.85e-2 | 1.50e-3 | 1.09 |
| 3 | 2.36e-2 | 1.81e-2 | 1.30e-3 | 0.77 |
| 4 | 1.80e-2 | 1.86e-2 | 1.31e-3 | 1.04 |

**fp16 accumulation lands in the same error band as torch's own fp16 forward**,
at 1.3-1.5e-3 relative against a 5e-3 bar, and `test_kernel_is_no_further_from_
fp32_than_torch_is` passes at a ratio near 1.0 against its tolerance of 2.0.

⚠️ This **refutes the "fp16 accumulation, 0 %" row** in the dead-ends table. The
earlier attempt cast the result; it did not change the accumulator, so ptx still
held `mma...f32.f16.f16.f32`. What emits the fast instruction is an fp16
accumulator threaded through the call — `tl.dot(a, b, acc, out_dtype=tl.float16)`
with `acc = tl.zeros(..., dtype=tl.float16)`. Verified by counting mma opcodes in
the PTX: 272 of each kind across the two compilations.

### The overflow ceiling, and how we would know

fp16 tops out at 65504, and activations grow through training, so the question is
which accumulator gets there first. The structure answers most of it: every
accumulator in a pre-norm transformer contracts a **LayerNorm output** against a
weight tile, and LayerNorm is scale-invariant in the residual stream, so a
growing residual never reaches them. What reaches them is γ and the weights.

Peaks at initialisation, over 8 layers, against the 65504 ceiling:

| accumulator | peak | headroom |
|---|---|---|
| QKV | 3.6 | 18 356× |
| W_o | 1.1 | 61 803× |
| FFN1 | 3.1 | 21 362× |
| FFN2, the K=1024 one | 1.2 | 54 177× |
| **Q·Kᵀ before the 1/√dₕ scale** | **17.4** | **3 764×** |
| the residual stream, already fp16 today | 11.5 | 5 715× |

Scaling γ and every weight matrix by `s`, a crude stand-in for training growth,
the attention score is the only thing that ever overflows, and it does so at
s≈8 while every other accumulator still has 100× left. It grows as s⁴ where the
others grow as s² or s³, because it contracts two projections instead of one.

The fix is exact and free: **the host folds 1/√dₕ into W_q**, so the kernel
accumulates post-scale logits — 21 293× headroom instead of 3 764×, moving
overflow to s≈12. For scale, the fp16 residual stream we already store today
fails around s≈17, so fp16 accumulation moves the breaking point from 17 to 12
rather than introducing a new failure mode.

**It is loud, not silent.** Overflow gives inf, `LayerNorm(inf)` is NaN, and the
`0*inf` path that `test_dead_tokens_must_be_finite` pins spreads it across the
board. A finiteness check on the value head catches it on the step it happens,
and the early-warning quantity to log during training is max |post-scale
attention logit|, which sits at 3.1 today. `acc_dtype="fp32"` is the escape
hatch, at −16 %.

### The architecture question, settled

A split implementation — big cuBLAS GEMMs plus a custom attention kernel plus
fused epilogues — is refuted. Its GEMMs *alone*, at M = 524288 where cuBLAS is at
its best, cost 50.85 ms per layer, so 406.8 ms for eight layers, or 40.3k evals/s
before attention (13.6 ms per layer measured), before LayerNorm, and before any
of the ~32 GB of activation traffic that fusion avoids. The fused megakernel
already does the whole forward in 449.6 ms. Fusion is the right architecture and
the remaining work is inside it.

## Diagnosis

Three successive bottlenecks, each measured.

**DRAM and kernel boundaries**, versions 0 and 1. Activations round-tripped to HBM
between every op, and layernorm and softmax ran as thin kernels. Fusion solved it
and DRAM dropped to 1.5 %.

**Occupancy locked by registers**, versions 2 and 3. At 255 registers per thread the
kernel ran 2 to 3 CTAs/SM where shared memory at 28.7 KB would allow 3. Any
`maxnreg` below 168 spills. Forcing the third CTA bought 2.5 %.

**Instruction-mix serialisation under phase synchronisation**, version 5 and
current. No-Eligible sits at 89.9 % and math-pipe stalls at 47.9 %. Warps are
generalist and sequential, and CTA barriers align them all on the same instruction
type at the same instant, so one pipe saturates while the others idle in rotation.

### The proof in the PTX, body of one layer

```
mma.sync    272     ld.global   135     bar.sync     90   ← 3 intended
ldmatrix    187     st.shared   193     shfl        136
cp.async      0     ld.shared    60     st.global    11
```

Every weight tile follows `global → registers → SMEM → ldmatrix → mma`, and any
transit through warp-shared SMEM forces a CTA barrier, giving 90 per layer and 720
across the kernel. `cp.async` is at 0 because `num_stages=1` won the sweep. This is
the concrete mechanism behind the third bottleneck, and what `cp.async` plus
per-tile mbarriers remove in CUDA.

## Winning settings

Joint sweep over 126 configurations on 2026-07-29: `BM=32, num_warps=4,
num_stages=1, maxnreg=168`, with weight tiles of [64,64] at 8 KB.

The joint sweep confirmed Théo's hunch that the parameters are not orthogonal.
`maxnreg=168`, the third-CTA threshold, only appears as a winner jointly and never
in isolated sweeps. It is worth a real 2.5 %, confirmed in 3 interleaved duels
out of 3.

## Remaining levers

| lever | expected gain | state |
|---|---|---|
| **warp-specialised CUDA C++ kernel** | **×1.25-2.5** | the path to GO; full spec in `csrc/README.md` |
| fp8 e4m3 | ×1.5-2 theoretical | untested, deferred by Théo |
| weight-stationary (weights resident in SMEM, activations streamed) | ~200 GB of weight re-reads down to ~26 GB | untested |
| 2-head attention pingpong (FA3 style) | 2-5 % | untested |
| fold LN γ/β into the following weights (`W'=diag(γ)W`, `b'=b+βW`) | ≤1 %, exact | untested, precomputable at init |
| parameterise the hardcoded loop bounds (`range(12)` to `range(3*D//BC)`) | 0 % | hygiene; the kernel is silently wrong if D or DFF change |

## Measured dead ends

Do not retest these.

| attempt | result |
|---|---|
| `torch.compile` / CUDA graphs | 0 %, with graphs genuinely active (3 `cudaGraphLaunch` in the profiler) |
| fp16 accumulation in the `dot`s | ~~0 %~~ **refuted 2026-07-29: +19 %.** The original test cast the result instead of changing the accumulator, so the PTX still held `f32.f16.f16.f32`. See the roofline section |
| cache and evict hints (`.ca`, `.cg`, evict_last…) | 0 to -8 % |
| fp16 LN affine, 8 sites | 0.1 %, noise |
| `maxnreg` below 168 | spills, regression |
| smaller GEMM tiles via CUTLASS | 2× worse |
| head dimension (dh=32/64/128) | ≤5 %, noise |
| `bias=False` | noise |
| **reductions via `tl.dot(ones)` instead of `tl.sum`** | **1.85× worse**. `tl.sum` costs 48 `shfl` and 0 barriers; `tl.dot` costs 64 mma plus 24 `bar.sync` and 32 `st.shared` of layout conversion, so the conversion outweighs the reduction it replaces. Reducing on tensor cores pays only when the data already sits in mma layout, as in an FA epilogue, never straight from a load. |
| `Var = E[x²] - µ²` for independent reductions and more ILP | +0.8 %, noise, so the µ-to-variance dependency is off the critical path |
| attention block-diagonal mask | dead at BM=T=32, and the compiler already eliminates it (0 `-inf`, 0 `setp.eq.s32` in the PTX), so there is nothing to gain. The dynamic liveness mask now rides on the same predicate and costs 1 % |
| epilogue reparameterisation (deferring `rstd`, folding µ via `colsum(W)`) | unfavourable here, since the output side is 3-4× wider than the input (3D=768, DFF=1024 against D=256) |

## FlashAttention lessons at T=32

FA's core ideas of KV tiling, online softmax and non-materialisation are moot at
T=32, where everything fits in SMEM and there is nothing to avoid materialising.
Three things transfer.

Operand delivery without register staging or CTA barriers, meaning `cp.async` plus
mbarrier plus `ldmatrix`, addresses the 90 barriers above and drives the CUDA plan.
The FA3 pingpong overlaps the softmax and SFU pipe with the mma pipe. Resident
operands motivate the weight-stationary lever.

One micro-decision already goes the right way here: we divide `p` by the sum before
the PV matmul, which is correct because T=32 is smaller than DH=64. FA2 divides
after because its ratio runs the other way.

From Hazy's "No Bubbles" megakernel, the loader/consumer/storer warp roles and SMEM
paging with early release all port to sm89, with the exception of TMA, `setmaxnreg`
and clusters.

## sm89 hardware trap

`cp.async.mbarrier.arrive` hangs the GPU on sm89. The validated idiom is
`commit_group` plus `wait_group` plus a plain `mbarrier.arrive`. A working
warp-specialised prototype exists, built in Gluon, so the mechanism is proven on
this card and the CUDA version carries no hardware risk.

---

## The search in the loop

Moved from `search.md` §14 on 2026-07-31. The prediction came first and the
measurement that settled it is below, which is why the two are kept together.

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

### How much the tree is likely to cost

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

### The low-`n` policy target

At small `n` the policy target degrades in a specific way worth watching for: with
about 31 edges at the root and `n` visits to spread over them, `N / n` approaches
uniform-with-noise. The reference implementation shows this long before the kernel
exists, which is a reason to look at it there.

### What the tree actually costs, measured 2026-07-30

`bench/bench_search.py`, `logs/gate1a.log`. `B = 4096` games at 30 random plies,
three interleaved rounds with the order reversed on alternate ones. The whole of §6
runs on device: `root_init`, `n` simulations of descent, encoder, expansion and
backup, and `select_and_advance`.

| `n` | useful evals/s | raw evals/s | encoder alone | tree | tree, ms/simulation | s per move | mean/max depth |
|---|---|---|---|---|---|---|---|
| 32 | **58 942** | 59 021 | 63 547 | 4.4 % | 2.84 | 2.22 | 4.1 / 15 |
| 128 | **58 763** | 58 852 | 61 890 | 4.3 % | 2.88 | 8.91 | 5.1 / 21 |
| 800 | **56 996** | 57 265 | 59 852 | 4.4 % | 3.00 | 57.22 | 5.8 / 28 |

**Gate 1 wanted 45-50k and gets 57.0k at the full `n = 800`.** The gated number is
*useful* evaluations: a descent that ended on a stored terminal, or that created a
terminal child, has nothing for the network to say, and §6.3 evaluates it anyway to
keep the launch shape static. That waste is 0.5 % at `n = 800` and 0.1 % below it.

The baseline is `n + 1` calls through `forward_full` and nothing else, because a
move is `root_init` plus `n` simulations and each calls the encoder once. Charging
the tree for one extra evaluation makes the overhead look like it falls with `n`
(7.5 % → 4.5 %); it does not, it is flat.

**The tree costs 3.00 ms per simulation at `B = 4096`, 0.73 µs per game.** Nearly
flat in `n`: the 5 % rise from 32 to 800 is the mean depth going from 4.1 to 5.8,
which is the only term that should grow and does.

**Predicted 1.7 ms and 2.7 %, measured 3.00 ms and 4.4 %.** Two separate gaps.

The absolute rate misses its 62.5k prediction because the encoder in this campaign
is **59 852/s, not the 64 142/s of §4.4**. A 57-second block holds the card at
1230-1290 MHz where the shorter encoder benchmark sits at 1.38-1.5 GHz, and 82 °C is
where it settles. Sustained MCTS load throttles harder than the benchmark that sized
it, which is a fact about the card and not about the search. The interleaved protocol
is what keeps the *ratio* meaningful anyway.

Scaled to the clock it actually ran at, the prediction was 1.84 ms against 3.00
measured, so the tree is 1.6× more expensive than the arithmetic said. The
prediction's dominant term was the two `movegen` calls, priced at the perft kernel's
5.01M boards/s. That kernel uses 71 registers and fits three blocks per SM; inside
`descent` the same code runs in a 96-register kernel at two blocks per SM, and inside
`expand` at 72 with a different instruction mix. Lower occupancy on the dominant term
is the leading candidate and it is **not measured**, because a 1.2 ms gap inside a
4.4 % overhead is not worth a profiler run.

⚠️ **The prediction was wrong by two orders of magnitude, in the optimistic direction.** It
put the tree near 0.05 % of a move; it is 4.4 %. Its error is not the 300-cycle guess
for a descent level, which if anything was generous, since the measured mean depth is
5.8 and not the 40 it assumed. Its error is scope: it costed the descent alone and
explicitly set aside the two `movegen` calls, the expansion and the launches, which
are the bulk of the 3.00 ms. The conclusion it drew still holds, since 4.4 % is not
worth attacking, but the number it drew it from should not be quoted.

**What the cost table above becomes**, now that the tree is in it. The measured rate of
real plies is `4096 / (s per move)`:

| `n` | plies/s, arithmetic | plies/s, measured | 10M games (80 plies) |
|---|---|---|---|
| 32 | 2006 | 1845 | 120 h |
| 128 | 502 | 460 | 483 h |
| 800 | 80 | 71.6 | 3103 h |

12 % above the arithmetic at every point, which is the tree plus the throttling.
The ordering is unchanged: a full run at `n = 800` is out of the question on this
card and the search budget is still the first thing to attack.

**What the counters say about §4's fixed sizes**, over one instrumented move at each
point. `E = 64` holds: 443 of 3 276 800 expansions truncated at `n = 800`, 0.0135 %
against §4.3's estimate of 0.01 %, dropping 2.32 of prior mass in total across those
443 nodes. The largest candidate count seen was **77**, above the 65 maximum in
`data/cuda_testset`, which is the reminder §4.3 asks for that the histogram was
measured on random playouts. And the node pool ends every move at exactly
`801 of 801`: `Nmax = n + 1` is tight, not generous, and every simulation created a
node.

---
