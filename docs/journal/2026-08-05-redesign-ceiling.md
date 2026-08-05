# 2026-08-05 (fifth entry) — what a redesign could buy, and the ceiling model that says so

Not an experiment. A costing, written down because the day's three attempts all failed
for the same reason and the question "would a major redesign unlock 100k+" deserves an
arithmetic answer rather than an opinion. Nothing here was built or measured today;
every number below is either quoted from an existing measurement or derived from one,
and the derivations are marked.

## The model, and it validates against a counter it never saw

Measured mma issue rates, 2026-08-04, on a harness that reproduces `perf.md`'s fp16 row
to 0.3 %:

    mma.m16n8k16.f16.f16.f16.f16      35.6 TFLOPS    qkv, attention, out-proj
    mma.m16n8k32.f16.e4m3.e4m3.f16    72.0 TFLOPS    ff1, ff2         (2.02x)

FLOP census per eval, from the shapes (T=32, d=256, dff=1024, L=8). It sums to
0.411 GFLOP, which is the ledger's number, so the census is right:

| | MFLOP/layer | share | precision today |
|---|---|---|---|
| qkv | 12.58 | 24.5 % | fp16 |
| attention (S = QK^T, P·V) | 1.05 | 2.0 % | fp16 |
| out-proj | 4.19 | 8.2 % | fp16 |
| ff1 + ff2 | 33.55 | **65.3 %** | **fp8** |

Normalising to fp16-equivalent issue time, `0.653/2 + 0.347 = 0.6735`. So the
mma-issue ceiling for the algorithm as it stands is

    35.6 / 0.6735 = 52.9 TFLOPS = 128.7k evals/s

⚠️ **The cross-check is what makes this worth writing down.** The kernel runs 82.8k =
34.0 TFLOPS effective, so it sits at **34.0 / 52.9 = 64.3 %** of that ceiling. `ncu`
measured `sm__pipe_tensor_cycles_active` = **63.81 %**. A FLOP census and a hardware
counter, sharing no input, agree to half a point. The model can be trusted for
extrapolation, which is exactly what it is about to be used for.

**100k from here needs 77.7 % tensor-pipe efficiency, up from 63.8 % — a 1.21x on
efficiency alone, inside a fixed 128.7k ceiling.** That is the same fight the day's
three attempts lost, and they returned 0.8 % combined.

## Lever 1 — qkv and out-proj to e4m3. 1.32x, and it moves the ceiling

The only candidate that reduces mma *work* rather than competing for issue slots.
Attention stays fp16: P is a probability distribution, and it is 2.0 % of FLOPs, so the
risk is all cost and no benefit.

    fp8 share    65.3 %  ->  98.0 %
    time units   0.6735  ->  0.5102        1.32x
    ceiling      128.7k  ->  169.8k evals/s

⚠️ **At a 170k ceiling, 100k needs 58.9 % efficiency — lower than the 64.3 % the kernel
already achieves.** That is a qualitatively different problem from the one this day was
spent on. Every previous attempt tried to raise efficiency under a fixed ceiling; this
one buys headroom and can afford to *lose* 8 % relative efficiency and still clear 100k.

The plumbing is nearly free, which is the surprising part. `K = 256` for both qkv and
out-proj is the same depth `ff1` already runs, so `q_max = 8`'s accumulator bound
(`q_max^2 * 256 <= 65504/3`) holds with no re-derivation, and `quantise_row` and
`gemm_fp8_row` apply verbatim. `w_qkv` is [768][256]: 768/128 = 6 scale blocks, and a
warp's 32 columns are 32-aligned so they never straddle one — the existing descale is
correct as written.

The cost is **two more `quantise_row` per layer**, after `norm1` (feeding qkv) and after
the attention output (feeding out-proj). `quant_h` measured at 5.4 % of the CTA
timeline in the phase decomposition, so this is roughly +5-10 % instructions against a
-32 % mma reduction.

## Lever 2 — G boards per CTA, and a correction to my own candidate list

⚠️ **`2026-08-05-coda-fold.md`'s candidate 2 — "shrink the shared-memory budget to give
L1 back" — rests on a wrong diagnosis and should not be attempted.** In `gemm_direct`
warp `w` reads n-tiles `4w..4w+4`; every warp reads different bytes and each layer's
weights are read **exactly once per CTA**. There is no reuse to capture. The 98.3 % L1
miss rate is **compulsory, not capacity**, and no L1 size in the 28-64 KB range changes
a stream with zero reuse.

The fix for a compulsory stream is to make it shorter — more boards per CTA:

    G = 2, 512 threads = 16 warps
    warp w  ->  board (w/8), head (w%8), output columns [32(w%8), +32)

2 boards x 8 heads = 16 warps **exactly**, so attention stays entirely warp-local and
the property the whole design rests on survives untouched.

| | today | G = 2 |
|---|---|---|
| SMEM | 2 x 49.5 KB (2 CTAs) | 99 KB (1 CTA) |
| registers/thread | 128 | 128 |
| warps/SM | 16 | 16 |
| weight L2->SM traffic | 1x | **0.5x** |

Warps 0-7 and 8-15 request the same weight tiles in the same cycles, so the second set
hits L1 with perfect temporal locality — the one access pattern a 28 KB L1 does serve.
And it spends **no registers**, which is what killed both prefetch attempts (register
ping-pong measured 0.835x).

The tight spot is 99 KB against Ada's 101,376 B opt-in maximum, and 16 warps per
barrier instead of 8 — the 14 barriers per layer get wider and I cannot size that
without measuring.

## Lever 3 — cp.async. Real, and deferred

Ada has `cp.async` and no TMA. It stages global -> shared with **zero registers**,
which is precisely what register prefetch could not do. `coda-fold.md` dismissed it as
"there is no shared memory spare"; the budget is actually small — one k-group of four
n-tiles per warp in fp8 is 32 lanes x 8 B = 256 B, double-buffered across 8 warps is
**4 KB**. And it is warp-private, so `cp.async.wait_group` + `__syncwarp()`: **it does
not reintroduce the CTA barrier the original design was built to kill.**

Deferred because it converts a stall into instructions, in a kernel where instructions
are the currency, and I have been wrong about that trade three times this week.

## The verdict, with the discount applied

**Predicted 100-115k, 100k more likely than not, confidence moderate.**

Against it, and weighted heavily: **three wrong denominators in three attempts** —
elapsed cycles predicted 11.3 % and delivered 0.0 %, static instructions predicted
4.1 % and delivered 0.8 %. The observed pattern is that this kernel returns about half
of what any instruction-level model predicts. Applying that haircut to 1.32x gives
1.16x = 96k, just under.

⚠️ **The gate is accuracy, not speed, and it comes first.** `2026-08-04-fp8-encoder.md`
excluded qkv and out-proj from fp8 because *that is where three quarters of the error
came from*. Today's max |dp| is 1.64e-2 on `t7h-fp8-002206`. All-fp8 could be 3-4x
that. This has to be settled in prior space through `brokefish/nn/validate.py` on a
trained checkpoint **before a line of kernel is written** — if it fails, lever 1 fails
and the redesign is worth nothing. The two mitigations are per-row scaling (which the
earlier failed experiment did not have) and FA3's incoherent processing, a random-sign
Hadamard to flatten outliers before quantising; `scratchpad/hadamard.py` exists.

## What this is worth to the project, which is the part that argues against doing it

Gate 1 is cleared at **71.5k useful evals/s in the live loop** against a 45-50k bar.
This redesign is about 1.25x on generation throughput for an estimated 2-3 days of
kernel work plus an accuracy campaign that may veto it.

Track E's n=64 -> n=128 finding was worth **1.6x wall clock to the same landmark**, and
it came out of one sweep. A search-side lever beat this entire redesign, and **n=128
against n=256 is still unmeasured** — the sweep stopped after one step of a lever it
had just shown to be first-order.

So the honest ranking is: the untested arm of the sims sweep is the better buy. Lever 1
is the one worth building if the kernel is being built for its own sake, and the
prior-space check is its first step, not its last.

## Ledger bug found while doing this

`docs/ledger/perf.md:627` still lists `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` at **41.6
TFLOPS**. `2026-08-04-fp8-encoder.md` measured **36.0** on a harness that reproduces
the fp16 row exactly, and asked for the 41.6 row to be "re-derived or retracted". It
was not. Any ceiling computed from that row is 13 % optimistic.
