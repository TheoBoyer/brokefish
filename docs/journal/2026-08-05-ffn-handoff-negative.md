# 2026-08-05 (third entry) — the FFN handoff fuses cleanly and buys nothing, and the reason invalidates how I sized it

## What was attempted

The phase decomposition from this morning put the `linear1` → `linear2` handoff at
**11.3 % of the fp8 kernel** — `ff1_epi` 4.0 + `hid_store` 1.9 + `quant_h` 5.4 — and
called it the largest single item in the 27.1 % that is not matmul. The handoff writes
ReLU'd fragments to shared memory as fp16, reads all 256 columns back to find a row
maximum, and reads them a second time to convert. The values were in registers when the
ReLU ran.

So: compute the row maximum from the fragments, cross warps with one fp16 partial per
(row, warp) through the 16 bytes of padding each `hid` row already carries, and write
e4m3 bytes straight out. `store_frags` disappears, both read passes disappear, one of
the three barriers disappears, and — because the four rows a lane's D fragments hold
are exactly the four rows `load_a_frag` will ask for — the scale never needs to enter
shared memory either; it goes to the GEMM in registers.

It works. Byte-exact against a host reference, eight warps, mutation-tested.

## And it is worth nothing

| | fp8 vs fp16 |
|---|---|
| Stage 1, two runs | 1.214×, 1.218× |
| fused, 2-byte stores | 1.217× |
| fused, stores paired across `t` | 1.210× |

Four order-balanced six-round A/Bs in the same process, so temperature is controlled.
The fused kernel is **neutral**, and the second variant is if anything marginally worse.

## Why, measured

`cuobjdump -sass`, opcode histogram of `encoder_kernel<true>`:

```
                 TOTAL   STS   LDS  SHFL  FMNMX  LOP3  IMAD  F2FP
Stage 1           4056   144    89   146    122   121    55   128
fused             4096   130    74   167    159   147    83   146
```

**The fusion removes 29 memory instructions and adds about 110 arithmetic ones, for a
net +40.** The kernel is issue-limited, so instructions are the currency and this is a
small loss dressed as a large win.

Where the arithmetic comes from is structural, not sloppy:

* the register-side maximum needs 32 `FMNMX` over the fragments **plus** 8 shuffles to
  reduce across the four lanes of a row-group, where the shared-memory side got its 32
  values with 16 `LDS.32` and needed 3 shuffles;
* combining the eight warps' partials is another 32 `FMNMX` and 4 `LDS.128`, which the
  old path did not need at all because the row was already whole in memory;
* and the 2-byte stores have to be paired across `t` with 16 more shuffles, because
  shared-memory **writes to a bank serialise** — reads broadcast, writes do not — so
  the natural 2 bytes per lane puts two lanes in one word and conflicts on half the
  banks. Measured before pairing: 16,251 cycles per board came out of `hid_store` and
  16,709 went straight into `quant_h`.

Being generous — dropping the `fabsf` (the input is post-ReLU, so it is redundant) and
doing the eight-partial combine in `half2` — gets the count back to about where it
started. **The fusion is a wash on this hardware by construction, not by a bad day.**

## The correction, which is the real result

⚠️ **Per-CTA elapsed cycles are not marginal costs when two CTAs share an SM.** The
decomposition measures a CTA's own timeline, and when that CTA stalls at a barrier or
on shared-memory latency the SM runs the *other* one. Removing work from a phase
shortens that CTA's timeline only if the SM had idle issue slots — and it does not, it
is saturated by the pair.

So this morning's sentence

> **8.7 points is inside the handoff alone.** CODA is now the second item on the list,
> not the first.

was arithmetic on the wrong denominator. Elapsed-cycle shares say **where a CTA's wall
clock goes**, which is what they were built to say and what they still say correctly.
They do not say what removing a phase buys. For that the currency is **issue slots**,
and the only way to spend fewer of them is to execute fewer instructions.

That reframes the whole remaining plan. `bench_phases.py` remains the right tool for
finding *candidates*; the opcode histogram is what decides whether a candidate is real:

    /usr/local/cuda-12.8/bin/cuobjdump -sass now.cubin | \
        awk '…count opcodes inside the target function…'

(`cuobjdump` is at `/usr/bin` and `/usr/local/cuda-12.8`; the 13.2 toolkit `_build.py`
resolves ships `nvcc` and `ptxas` only.)

## What this leaves for 90k

90k is 1.095× from 82.2k, i.e. **8.7 % fewer issue slots**, and the kernel spends 4056
instructions of which 192 are `HMMA`. The three biggest non-mma opcodes are `HADD2`
549, `FADD` 405 and `FMUL` 404 — 1,358 instructions, a third of the kernel, and none of
them is in a phase the decomposition flagged. `FADD`/`FMUL` at that count is
LayerNorm's fp32 arithmetic and the fp8 descale; `HADD2` is bias, residual and the mma
accumulate chain.

That is a different search than the one I set out on this morning, and it points back
at **CODA** after all — not because normalisation is a large share of elapsed time
(6.4 %, it is not), but because folding γ into the next GEMM's weights and deferring
1/σ and μ deletes `FADD`/`FMUL`/`HADD2` *instructions*, which is the thing that
actually costs.

⚠️ Nothing here is a plan yet. The next honest step is to attribute those 1,358
arithmetic instructions to source lines — `nvdisasm -g` or `ncu --set SourceCounters`
on the standalone `core.cuh` driver — before promising what deleting any of them buys.

## Landed

Nothing. The kernel is reverted to Stage 1; `csrc/` is byte-identical to `cb92886`.
`bench/bench_phases.py` and the `prof` instrumentation, which are from Stage 2, stay.
