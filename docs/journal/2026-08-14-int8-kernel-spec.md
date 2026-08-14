# 2026-08-14 (seventeenth entry) — int8 instead of e4m3, two boards per CTA: the spec, and the register check that gates it

A week of fp8 work has been asking the wrong question. The question was *how much
more of the network can we quantise*, and the answer kept being "more speed, more
error, pick a point". Changing the **format** breaks that frontier: int8 is more
accurate *and* the same speed, and the reason is a property of our activations that
this repository has now measured three separate ways.

This entry specifies the kernel, records the Step 0 register measurement that decides
whether it is buildable, and preregisters the gate.

## The case, in one table

`t12h-gumbel-004009`, 1039 non-terminal positions, prior space through
`Search._expand`, against the fp32 oracle. `flip` is the fraction of positions where
`max |dp|` exceeds the top-two prior gap — the metric that says a *move* can change.

| | max \|Δp\| | p95 | flip | issue rate |
|---|---:|---:|---:|---|
| `cuda-fp8` — **ships today** (FFN e4m3) | 4.37e-2 | 5.19e-3 | 4.43 % | 72.2 TFLOPS |
| **FFN int8** | **1.47e-2** | 1.95e-3 | **1.93 %** | **72.2 TFLOPS** |
| FFN int8 + u8 hidden | 1.71e-2 | 1.92e-3 | **1.64 %** | 72.2 TFLOPS |
| all-four-matmul int8 | 3.76e-2 | 3.39e-3 | 3.18 % | 72.2 TFLOPS |
| all-four-matmul e4m3 | 8.15e-2 | 1.05e-2 | 9.35 % | 72.2 TFLOPS |

On `t24h-muon-008215`, where the error is worst: FFN e4m3 **11.20 %** flip → FFN int8
**5.88 %**; all-four e4m3 19.98 % → int8 7.71 %.

**int8 on all four matmuls is more accurate than e4m3 on two** (3.18 % against
4.43 %). The issue rates are in `perf.md`; they are identical and so is the wall
clock.

### Why, and it is our own distribution

e4m3 spends four bits on an exponent and keeps three of mantissa; its error is
**relative**, ~2⁻⁴ everywhere across eighteen binades. int8 spends everything on
magnitude and its error is **absolute** inside a block. Which wins is decided by how
much dynamic range the block actually has, and ours has almost none — measured here
three times over:

- `q_max` is **flat from 4 to 448** (2026-08-14): only 0.6-0.9 % of Q entries go
  subnormal at `q_max = 4`, and 0.09 % at 32.
- **A Hadamard rotation bought nothing** (2026-08-05): amax/rms 5.1 at qkv and 6.7 at
  ffn2, against a Gaussian's 2.9, and flattening them changed nothing.
- Q/K/V span **~2 binades of e4m3's 18**.

[Baalen et al., *FP8 versus INT8 for efficient deep learning inference*](https://arxiv.org/abs/2303.17951)
predicts exactly this: INT8 best on uniform data, FP8-**E2** best on Gaussian with
INT8 a close second, FP8-**E4** best only on outlier-heavy distributions. We have been
running E4M3 — the outlier format — on data with no outliers. They also report
FP8-E4-with-fp16-accumulator at **53 % more gates** than INT8, and DeepSeek's ISCA '25
paper reaches the same corner from a third direction: their **LogFMT** beats E4M3 and
E5M2 at the same 8 bits, and they dropped it for register pressure and log/exp
throughput.

## ⚠️ The cost: integer mma has no narrow accumulator

`f16`, `s16` and `f32` accumulators all fail to compile with `s8` operands; e4m3 has
both `f16` and `f32`. This is arithmetic, not omission. fp16 accumulation is
affordable for e4m3 only because `kQMax = 8` discards range we never use
(`q_max² × 256 ≤ 65504`), and **an integer's range is its precision** — the same trick
would discard mantissa. So the mma accumulator goes from 2 registers per tile to 4.

The precedent for what that costs is already in this repository, in
`fp8_gemm.cuh:257`: adding a ping-pong prefetch — **16 registers** — took the encoder
from **1.147× to 0.835×**, a 27 % regression, because the path sits against the
128-register cap. int8 at today's occupancy adds exactly 16.

**Which is why the format change and the occupancy change are one design, not two.**

## Step 0 — the measurement that gates the build

`nvcc -c -O3 -arch=sm_89 -Xptxas -v` on a transformed copy of `csrc/encoder.cu`, with
the m-tile count and the accumulator type parameterised. Full table in
`perf.md`; the shape of it:

| | regs | spill loads |
|---|---:|---:|
| 1 board, 2 CTA/SM, e4m3 — **ships, validates the harness** | 128 | 240 B |
| 1 board, 2 CTA/SM, int8 | 128 | **336 B** |
| **2 boards, 1 CTA/SM, int8 — the proposal** | **223** | **0 B** |

**223 registers of 255, and no spill whatsoever.** The proposal is buildable, with 32
registers spare — enough for the double-buffered A fragments that measured **+0.56 %
at 1 CTA/SM and ±0 at 2**, i.e. the one optimisation that only pays in this
configuration.

⚠️ **And Step 0 inverted the build order I had written.** I expected the int8 drop-in
at current occupancy to be the safe first step. It is the *only* configuration that
gets worse: ptxas clamps to 128 and spill loads go **240 → 336 B (+40 %)**. Every
configuration with 255 registers available spills nothing. The occupancy change is not
a follow-on to the format change; it is what makes the format change free.

## Resource equations

Device limits queried, not assumed: `maxSharedMemoryPerBlockOptin` **101 376 B**,
65 536 registers/SM, 24 SMs, 48 warps/SM.

|                       | SMEM/CTA | boards/SM | regs/thread | warps/SM | weight traffic/board |
|---|---:|---:|---:|---:|---:|
| ships | 50 176 | 2 | 128 | 16 | 1.00× |
| **proposed** | **100 352** | 2 | **255** | **8** | **0.50×** |

⚠️ The 2026-08-09 retrospective says two boards per CTA "needs 101 376 B, which is
exactly the 99 KB limit". That sentence is what made this look blocked and it was
quoting the *ceiling*, not the requirement: 2 × 50 176 = **100 352 B**, which fits
with 1 024 B spare.

**Why traffic is the target.** The shipped kernel is memory-latency bound — the stall
order has flipped since the fp16 profile the ledger quotes: `long_scoreboard` **3.62**
over `math_pipe_throttle` **3.00**, where fp16 measured 5.53/8.16. L2→SM traffic is
**8.63 MB/board** at a 99.61 % L2 hit rate — i.e. every CTA re-reads the entire weight
set for its one board, confirming the retrospective's arithmetic (8.14 MB predicted).
Two boards per CTA halves that by construction.

## The scheme — normative once built

Defined in `brokefish/nn/quant.py` **and nowhere else**, transcribed once into
`csrc/int8_gemm.cuh`, and pinned by a test that re-derives rather than re-runs.

- **Weights**: one scale per 128 output channels over the whole K — the same shape as
  `Fp8SOff`, for the same reason (`gemm_*_row` reads `w_scale[n8/16]` with no k index)
  — `scale = amax/127`, round to nearest.
- **Activations**: one scale per row, `scale = amax/127`, stored in the row padding at
  `fp8cfg::kScaleOff`, exactly as today.
- **ff2's activation is `u8`**: post-ReLU and therefore non-negative, `scale = amax/255`,
  through `mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32`. 256 levels instead of 127.
  A float cannot trade its sign bit for mantissa, so this has no e4m3 analogue. ⚠️ Its
  measured effect is **within noise** (flip 1.93 % → 1.64 % on gumbel, 3.18 % → 2.99 %
  on all-four, slightly worse on one of four rows); it is in the spec because it is
  free, not because it was shown to help.
- **`pack_b_fp8` is reused unchanged.** The m16n8k32 B-fragment layout depends on
  element *width*, not type, so the permutation, `csrc/tests/tfp8.cu`'s independent C++
  packing and `test_the_packed_layout_decodes_back` all carry over.
- **Descale** `float(acc) × sa × sw` in fp32, for the reason `gemm_fp8_row` already
  documents: the product of two scales is routinely below fp16's smallest normal.

### ⚠️ `kQMax` disappears, and that is a robustness result

The worst s32 partial is `127 × 127 × 256 = 4.13e6` against `2.147e9` — a **520×
margin**. int8 has no infinity, no NaN, no saturating convert and no reachable
overflow. The failure that cost this project a day — `q_max` disagreeing across two
languages, an infinite partial, `inf * 0` → NaN on 116 boards of 128 while every
isolated component tested clean — is **structurally impossible** here. Both
`static_assert`s, the `q_max` sweep, and `fp8_q_max()`'s cross-language pin all become
unnecessary on this path.

## Two boards per CTA is `M = 64`

The four weight matmuls are per-token and both boards share every weight, so stacking
them is a taller A and one B load for twice the work. Allocate `bufA`, `bufB` and
`scratch` at `2T` rows so one stride serves both boards.

- **qkv, out, ff1, ff2**: m-tiles 2 → 4. This is the entire traffic saving.
- ⚠️ **Attention does not share.** The score matrix is per-board `[32,32]` per head, so
  it runs twice — rows 0-31, then 32-63. It is 3.8 % of the budget.
- **LayerNorm, residual adds, ReLU, `quantise_row`**: per-token, 64 rows instead of 32.
- **`alive`**: two `__ballot_sync` reductions. `T == warpSize` still makes each ballot
  the whole reduction.
- **Indexing**: `blockIdx.x` becomes the board *pair*, grid `ceil(B/2)`. ⚠️ Odd `B`
  needs a tail guard, and `simulate` hands a fixed `B` — this touches the staging
  contract, which is the one interface change.

## The flag

`--int8`, mirroring `--fp8` (`brokefish/train/loop.py:1119`). **The fp8 path stays.**
Both are compile-time template parameters on `encoder_kernel`, for the reason the
existing comment gives: a runtime branch makes ptxas allocate for the union and spills
the path every number in `perf.md` was measured on. `FusedEncoder(net, int8=True)`
parallels `fp8=True`; the two are mutually exclusive and asserted so.

## Build order, revised by Step 0

1. **int8 at 2 boards/CTA, 1 CTA/SM, `ff1`/`ff2` only.** Do not stage through the
   1-board version — Step 0 says that is the only configuration that spills.
   Verification: `csrc/tests/tint8.cu` against a host product, then `nn/quant.py`'s
   emulation against the kernel in prior space, then `nn/validate.py`.
2. **A/B the throughput**, interleaved and order-balanced over six rounds, never
   before/after.
3. Only then consider extending int8 past the FFN — `all4` measures 3.18 % flip
   against the FFN-only 1.93 %, so it is a further trade, not a free one.

## Preregistered gate

Agreed before the build: **runtime ≥ 1.0× today's, at a max |Δp| at or near the
measured int8 level (1.47e-2 against the shipped 4.37e-2) — an absolute win.** Runtime
below 1.0× is judged on the curve, not on this entry.

⚠️ **The traffic halving is certain** — it is arithmetic on the loop structure. **The
throughput is not.** Compute SOL 64.0 % and L1/TEX SOL 64.9 % are equal today, so
relieving memory alone runs into compute quickly, and 8 warps against 16 pushes the
other way by an unknown amount. I would not bet inside 0.95×-1.2× for the occupancy
change on its own.

## Built, same day — the format half

⚠️ **Half the spec shipped and half did not.** `--int8` is landed, tested and
measured; **two boards per CTA is not built**. The build was staged deliberately —
changing format and layout together would have made a slow result unattributable — and
the format half cleared the gate on its own, so the layout half is now a separate
question rather than a dependency.

What landed: `csrc/int8_gemm.cuh`, `mma_s8`/`mma_u8s8`/`cvt_int8x4` in `mma.cuh`, an
`INT8` template parameter on `encoder_kernel` beside `FP8`, an explicit `quant` mode on
`model_forward`, `FusedEncoder(int8=True)`, `PackedWeights.pack(int8=…)`, and `--int8`
on the training loop. The fp8 path is untouched; **531 tests pass**.

**The gate is cleared on both axes at once**, which is not what a quantisation change
usually does: **1.016× e4m3's throughput** and **2.56× lower max prior error**
(1.71e-2 against 4.37e-2), flip risk **4.43 % → 1.83 %**. Numbers and protocol in
`docs/ledger/perf.md`.

⚠️ **The throughput prediction was wrong in the good direction and remains
unexplained.** Step 0 said int8 at 2 CTAs/SM would spill more (240 → 336 B) and
`fp8_gemm.cuh` prices 16 registers at 27 %, so a small regression was expected.
It measured +1.6 %. Two candidate mechanisms — a cheaper descale, a dearer quantise —
and no measurement separating them.

Three things the spec asserted that the build confirmed: `pack_b_fp8` transferred
verbatim (the fragment map really is a property of element width); the emulation
predicted the kernel to four digits, where the e4m3 path missed by 1.6× on its first
landing; and `kQMax` genuinely disappeared — there is no accumulator constant to
co-ordinate across two languages on this path, and the test that pins the int8 maxima
says so in its own docstring rather than guarding a live hazard.

## QA, and the one thing nothing catches

Validated on **nine networks** including random init — int8 beats e4m3 on max |Δp| and
on flip risk on **every one**, ratio 1.29× (`t24h-fp8-008416`) to 6.18× (random init),
zero non-finite outputs anywhere. `csrc/tests/tint8.cu` pins the convert, both mma
fragment layouts, `quantise_row_int8` in the aliasing in-place case, and `gemm_s8_row`
against an exact host product. 541 tests pass; ten are new.

⚠️ **The shipped validator could not reach either quantised path.** `--impl` names a
registry entry and both are constructor flags on the CUDA one, which is why the fp8
kernel ran for ten days with **no test coverage at all** — only its emulation was
tested. `nn/validate.py --quant fp8,int8` now reaches them.

**Five deliberate faults, five different answers:**

| mutation | `tint8.cu` | `test_quant.py` |
|---|---|---|
| swap the `sa_lo`/`sa_hi` row split | caught | caught |
| signed mma on the unsigned hidden | caught | caught |
| `kQMaxS` 127 → 126 | missed | caught |
| **drop the aliasing `__syncwarp`** | **missed** | **missed** |
| weight-scale block index off by one | missed | caught |

The first two were **missed by the device test until this exercise added
`test_gemm_row`** — the primitives were pinned and the function composing them was not.

⚠️ **Nothing catches the missing `__syncwarp`, and it is the exact hazard that produced
NaN on 116 boards of 128 on the e4m3 path.** It is a scheduling race, not a value
error: with no divergence the warp stays in lockstep and the wrong code gives the right
answer. `compute-sanitizer --tool racecheck` reports the int8 kernel clean (its 8
hazards are all inside torch's own `reduce_kernel`), but a clean racecheck on correct
code does not prove it would flag the broken version. **That hazard is guarded by a
comment and by nothing else, on both the e4m3 and the int8 path.**

## What is not established

- **No flip-risk number anywhere in this week's work has been converted to Elo.** Not
  today's 4.43 %, not int8's 1.93 %. The entire accuracy case is prior-space.
- Whether 8 warps can hide the memory latency that 16 currently hide.
- Whether the u8 hidden helps at all — measured within noise.

Artefacts: `logs/fp8-attn.log`, `logs/fp8-store.log`, `logs/fp8-vs-step.log`,
`logs/fp8-fidelity-vs-training.png`, and the Step 0 numbers in `docs/ledger/perf.md`.
