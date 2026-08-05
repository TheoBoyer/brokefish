# 2026-08-05 (second entry) — what the 20.6 % is made of, and the budget I was using was the wrong one

## Why this needed measuring rather than estimating

`docs/ledger/perf.md` gets the non-matmul share by **subtraction**: the four real GEMM
shapes run at 32.1 TFLOPS in isolation (`csrc/shapes_ab.cu`) and the whole kernel at
25.5, so 20.6 % is "something else". That residual cannot name the something else, and
it cannot tell overhead apart from the GEMMs running slower in situ. The morning's
entry sized a plan against it anyway, which is the mistake this one corrects.

## The instrument

`csrc/encoder.cu` grew a `namespace prof`: `clock64` deltas at boundaries the kernel
**already has**, so no barrier is added — adding one would serialise warps that
currently overlap and inflate the very total it is supposed to divide. Each warp's
deltas sum to its own lifetime, so the fractions are exact by construction and nothing
has to be assumed about overlap. Two CTAs share an SM, so a phase's cycles include time
the SM spent on the other CTA, which is what makes them fractions of *elapsed* time —
the denomination a speedup is quoted in.

Three details it turns on:

* `PROF` is a **template parameter**, like `FP8` and for the same reason: a runtime
  branch would make ptxas allocate for the union and spill the path every number in
  perf.md was measured on. Verified rather than asserted — `cuobjdump -sass` on
  `encoder_kernel<false,false>` and `<true,false>` against the same kernels at HEAD is
  **byte-identical, 3424 and 4056 instructions**.
* The clock read is `asm volatile` with a `"memory"` clobber. Without both, ptxas may
  move loads and stores across it, which is exactly what makes a phase boundary a
  fiction.
* Counters are **sharded 32 ways by CTA**. A mark fires ~250 times per warp per CTA;
  unsharded, that is millions of atomics onto 36 addresses, and same-address atomics
  serialise — the profiler would have been measuring its own contention and charging it
  to whichever phase it landed in.

⚠️ **The instrumented kernel is not the shipped kernel.** The clock reads are
scheduling barriers, and they change register pressure: the fp8 path's spill *drops*
from 40/200 B to 8/48 B under `PROF=true`. Wall-clock cost measured at **1.027× to
1.062×** depending on arm and batch. Every fraction below carries that caveat.

## The decomposition

512 boards, random-init net, `bench/bench_phases.py`. Cycles per board, warp 0.
Re-run at **4096 boards every group lands within 0.1 point**, so these are not an
artefact of the operating point.

| phase | fp16 cycles | % | fp8 cycles | % |
|---|---|---|---|---|
| prologue | 2,755 | 0.3 | 2,427 | 0.3 |
| ln1 | 18,794 | 1.8 | 19,167 | 2.3 |
| qkv_gemm | 171,140 | 16.8 | 149,580 | 17.9 |
| qkv_bias | 6,281 | 0.6 | 4,691 | 0.6 |
| attention | 31,262 | 3.1 | 31,420 | 3.8 |
| attn_store | 9,328 | 0.9 | 8,000 | 1.0 |
| proj_gemm | 64,034 | 6.3 | 64,164 | 7.7 |
| proj_epi | 5,425 | 0.5 | 5,017 | 0.6 |
| res1_ln2 | 27,469 | 2.7 | 25,902 | 3.1 |
| quant_a | — | — | 11,625 | 1.4 |
| **ff1_gemm** | 332,771 | 32.6 | 193,597 | 23.2 |
| ff1_epi | 50,448 | 4.9 | 33,520 | 4.0 |
| hid_store | 13,360 | 1.3 | 16,090 | 1.9 |
| **quant_h** | — | — | 45,221 | **5.4** |
| **ff2_gemm** | 261,607 | 25.6 | 200,278 | 24.0 |
| ff2_epi | 6,273 | 0.6 | 5,594 | 0.7 |
| res2 | 9,197 | 0.9 | 8,602 | 1.0 |
| epilogue | 9,811 | 1.0 | 8,962 | 1.1 |

| group | fp16 | fp8 |
|---|---|---|
| matmul | **81.3 %** | **72.9 %** |
| staging + barriers | 8.9 % | 8.7 % |
| normalisation | 5.4 % | 6.4 % |
| attention | 3.1 % | 3.8 % |
| fp8 quantisation | — | **6.8 %** |
| prologue + epilogue | 1.2 % | 1.4 % |

## The residual method was right, for the kernel it described

fp16 measures **81.3 %** matmul against perf.md's 79.4 % by subtraction. Two points
apart, and the direct number is the higher one — consistent with `shapes_ab.cu`'s 48
CTAs and hot L2 flattering the isolated GEMM rate slightly. The subtraction was sound.

## And it does not describe the kernel that ships

⚠️ **The fp8 kernel is 72.9 % matmul, not 79.4 %**, because fp8 halves the FFN's matmul
time and then adds 6.8 % of quantisation that the fp16 kernel never pays. Every piece
of Stage-3 arithmetic in this morning's entry used the fp16 budget for an fp8 kernel.
The conclusions that came out of it —

> FFN-only fp8 tops out near **84k** […] 90k needs the non-matmul 20.6 % cut to 13.0 %,
> which is the CODA reparametrisation

— are wrong in both directions, and the correction goes the good way.

**90k is 1.095× from here, not 1.452×.** The `1.452×` was measured from the *fp16*
62.0k baseline; Stage 1 has since delivered 82.2k, so 90k/82.2k = 1.095 and the job is
to remove **8.7 points out of a 27.1-point budget**. (100k is 1.217×, i.e. 17.8 points.)

**And CODA is not the biggest lever.** I named it because I assumed "other" was mostly
normalisation. Measured, normalisation is 6.4 % and the largest single item is the
FFN's internal handoff:

| lever | measured size | what it is |
|---|---|---|
| **ff1 → ff2 handoff** | **11.3 %** | `ff1_epi` 4.0 + `hid_store` 1.9 + `quant_h` 5.4 |
| normalisation (CODA) | 6.4 % | two LayerNorms and two residual adds per layer |
| attention | 3.9 % | 2.0 % of the FLOPs, so mostly softmax and the V transpose |
| staging elsewhere | 2.9 % | `attn_store`, `proj_epi`, `ff2_epi`, `qkv_bias` |
| `quant_a` | 1.4 % | one per layer, against `quant_h`'s four |
| prologue + epilogue | 1.3 % | the embedding gather and the three heads |

The handoff writes ReLU'd fragments to SMEM as fp16, reads all 256 of them back to find
a row maximum, and writes them again as bytes. The values were in registers when the
ReLU ran. Per call `quant_h` and `quant_a` cost the same (1,413 vs 1,453 cycles) — the
5.4 % is simply four calls a layer against one, which is also why fusing it into the
ff1 epilogue is worth four times as much as fusing `quant_a` would be.

**8.7 points is inside the handoff alone.** CODA is now the second item on the list,
not the first.

## A second lead, unexplained

`ff1_gemm` gets **1.72×** from fp8 and `ff2_gemm` only **1.31×**, on identical FLOP
counts. In fp16 the asymmetry runs the other way — ff1 332,771 cycles against ff2's
261,607 — so it is not simply "ff2 is the hard one". Candidates: `ADD=true`'s extra
fp16 round trip per output, and ff2's 16 KB stride between n8-tiles against ff1's 4 KB.
Unmeasured, and worth roughly 3 points if ff2 could be brought to ff1's ratio.

## Reproducing

    uv run --no-project --python .venv/bin/python -m bench.bench_phases \
        --boards 512 --reps 4 [--impl fp8|fp16|both] [--checkpoint …]

`ext.set_profile(True)` recompiles nothing — both instantiations are in the module, and
the flag picks one at launch.
