# Writing the encoder kernel: a case study

How `csrc/encoder.cu` got to 55.0k evals/s, including the version that was
*slower* than the Triton kernel it replaced, and why.

This is written for someone fluent in Triton and PyTorch who has not spent much
time in CUDA C++. It assumes you know what a warp is, what coalescing means and
how a roofline works. It does not assume you have met `mma.sync`, `ldmatrix`,
inline PTX, `ptxas -v`, or the register-file arithmetic that decides occupancy —
those get spelled out.

The thing worth taking away is not the final code. It is the *reuse analysis* in
§7, which is three lines long, decides the entire design, and which I got wrong
the first time.

---

## 0. The problem, and the unit conversion that makes it legible

We evaluate a transformer: 32 tokens, d=256, 8 layers, 8 heads, FFN 1024. Per
evaluation:

| | FLOPs |
|---|---|
| QKV projection, 32×256 × 256×768 | 12.58 M |
| attention (QKᵀ and PV, 8 heads) | 1.05 M |
| output projection | 4.19 M |
| FFN up + down | 33.55 M |
| **per layer** | **51.4 M** |
| **× 8 layers** | **411 M** |

So **0.411 GFLOP per evaluation**, and the conversion you should keep in your
head for the rest of this document is

```
evals/s × 0.411 = TFLOPS
```

55.0k evals/s is 22.6 TFLOPS. That single number is what turns "is this fast?"
into a question with an answer, because it can be compared against what the
silicon can actually issue.

Target: ≥45k, ideally >50k. Baselines: torch eager 19.4k, our Triton kernel
42.8k.

## 1. The machine, measured rather than looked up

Everything here was measured on the actual card, and two of the numbers
contradict what you would assume from marketing material.

```
NVIDIA GeForce RTX 4060 Laptop GPU, sm_89 (Ada)
24 SMs · 32 MB L2 · 256 GB/s DRAM
65,536 registers per SM · 102,400 B shared memory per SM (101,376 max per block)
1536 threads per SM max
clocks: 1.38-1.5 GHz sustained (not the 2055 MHz boost)
```

Tensor core issue rates, measured with a microbenchmark that does nothing but
issue back-to-back `mma.sync`:

| instruction | TFLOPS |
|---|---|
| `m16n8k16.f32.f16.f16.f32` (fp16 in, **fp32 accumulate**) | **18.0** |
| `m16n8k16.f16.f16.f16.f16` (fp16 in, **fp16 accumulate**) | **35.5** |
| `m16n8k32.f32.e4m3.e4m3.f32` (fp8) | 41.6 |

**fp32 accumulation is half rate on GeForce.** This is a market-segmentation
fuse, not a physical limit — the same die in a datacentre SKU does not do this.
It reframes the whole project: at fp32 accumulate the ceiling is 18.0 TFLOPS =
43.8k evals/s, so **the 45k target would have been literally unreachable**, and
every optimisation would have been fighting for a share of something that
couldn't win. At fp16 accumulate the ceiling is 35.5 TFLOPS = 86k evals/s and
45k is 52% of issue rate — demanding but ordinary.

That is why this kernel accumulates in fp16, and why the first thing to do on any
new card is measure the instruction you intend to build the kernel out of. The
number that used to sit in our notes ("35-40 TFLOPS fp16") was the fp16-*accumulate*
figure being quoted for a kernel that accumulated in fp32. It was off by 2×, in
the direction that makes you attempt the impossible.

**Is fp16 accumulation safe?** Separate analysis (`docs/perf.md`): in a pre-norm
transformer every accumulator's A operand is a LayerNorm output, so the residual
stream — the thing that actually grows during training — never feeds a
contraction. The single exposed accumulator is Q·Kᵀ, and we fold 1/√d_head into
W_q at construction so the scale applies *before* the contraction rather than
after, which buys 5.7× of headroom for one multiply on the host. Overflow is
loud (inf → NaN), not silent.

## 2. The mental model: what one CTA is actually doing

One CTA (thread block) takes one board through all 8 layers and never writes
activations to global memory. 256 threads = 8 warps. Warp `w` owns head `w` and
model dimensions `[32w, 32w+32)` of every output.

That last sentence is the load-bearing design decision and it comes from
attention: if warp `w` owns head `w` entirely, then Q·Kᵀ, the softmax and P·V for
that head are computed by one warp from start to finish, with no cross-warp
communication and therefore **no CTA barrier anywhere inside attention**. The
softmax reduction is over 32 keys, which one warp holds in registers, so it is
two `__shfl_xor_sync` and done.

Now the number that should immediately bother you:

```
work per CTA per layer:     51.4 MFLOP
weights read per layer:     1.58 MB
                            ----------
arithmetic intensity:       32.5 FLOP/byte
```

**32 MACs per weight element.** That is the whole game. A weight is loaded, used
by 32 tokens, and never touched again inside this CTA. Hold onto that; §7 is
entirely about its consequences.

For context, a normal training GEMM has intensity in the hundreds or thousands,
because M is thousands of rows. Ours is 32 because a board is 32 pieces. We are
running a GEMM with a pathologically small M, and that shapes everything.

## 3. The tensor core contract

In Triton you write `tl.dot(a, b)` and the compiler picks instructions and
layouts. In CUDA C++ there is no such layer. You issue the instruction yourself,
in inline PTX, and **you are responsible for having the right bytes in the right
lane's registers beforehand**. That is the entire difficulty of this kind of
kernel; everything else is bookkeeping.

The instruction:

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16  {d0,d1}, {a0,a1,a2,a3}, {b0,b1}, {d0,d1}
```

It computes `D = A·B + D` for a 16×8 output tile with a contraction depth of 16,
**cooperatively across all 32 lanes of one warp**. It is not a per-thread
operation. Each lane supplies a few registers and receives a few back, and the
mapping from (lane, register) to (row, column) is fixed by the ISA.

Here is how you write that in CUDA C++, which is worth reading closely if inline
asm is new:

```cpp
__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
```

* `uint32_t (&d)[2]` is a *reference to an array of 2*, not a pointer. It keeps
  the array in registers and lets the compiler see the size. A plain `uint32_t*`
  would work but invites the compiler to place the array in local memory.
* The operands are `uint32_t` even though the data is fp16: each register holds
  **two packed halves**. That is why counts look halved everywhere.
* `"+r"` means read-write (the accumulator, which appears both as output `%0,%1`
  and as the last input), `"r"` means read-only, `%N` are positional slots
  numbered across outputs then inputs.
* `asm volatile` stops the compiler from deleting or reordering it based on the
  (from its perspective, opaque) side effects.
* `__forceinline__` matters: without it you get a real function call per mma.

**The fragment layouts.** With `g = lane/4` (0..7) and `t = lane%4` (0..3):

```
A (16×16, row-major)     a[0] = A[g  ][2t], A[g  ][2t+1]
                         a[1] = A[g+8][2t], A[g+8][2t+1]
                         a[2] = A[g  ][2t+8], A[g  ][2t+9]
                         a[3] = A[g+8][2t+8], A[g+8][2t+9]

B (16×8, k-major)        b[0] = B[2t  ][g], B[2t+1][g]
                         b[1] = B[2t+8][g], B[2t+9][g]

D (16×8, fp16 accum)     d[0] = D[g  ][2t], D[g  ][2t+1]
                         d[1] = D[g+8][2t], D[g+8][2t+1]
```

Stare at those until two facts jump out, because the kernel is built on them:

**Fact 1 — D and A agree.** The D fragment of one 16×8 output tile has exactly
the shape of half an A fragment. So the D fragments of two *adjacent* output
tiles concatenate into one complete A fragment:

```cpp
a[0] = lo[0];  a[1] = lo[1];  a[2] = hi[0];  a[3] = hi[1];
```

That is a *register rename* — zero instructions, zero memory. It is how a GEMM
result feeds directly into the next `mma` as the A operand. In this kernel it
carries S → P → O inside attention with no round trip through memory.

**Fact 2 — D and B agree along the other axis.** Look at `b[0] = B[2t][g],
B[2t+1][g]`. For `S = Q·Kᵀ` the B operand needs `B[k][n] = K[n][k]`. Substituting,
lane L needs `K[g][2t], K[g][2t+1]` — which is exactly the `d[0]` layout that K
came out of the QKV projection in. **K needs no transpose.** V does, and gets one
from `ldmatrix.trans`.

These two facts are why the kernel can do attention entirely in registers.

## 4. `ldmatrix`, and the trap that cost a day

`mma` wants operands scattered across lanes in that specific pattern. Loading it
with ordinary `half` loads would take many instructions and hit awful bank
conflicts. `ldmatrix` exists for exactly this:

```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
```

It loads **four 8×8 fp16 matrices** from shared memory. Each lane supplies *one
address*, and the instruction distributes the result so that each lane ends up
holding the two elements the `mma` will want. Which lane addresses which matrix
is fixed:

```
lanes  0-7  give the 8 row addresses of matrix 0 → r0
lanes  8-15                          matrix 1 → r1
lanes 16-23                          matrix 2 → r2
lanes 24-31                          matrix 3 → r3
```

So to fill an A fragment (16×16 = four 8×8 blocks) you compute, per lane:

```cpp
int r = row0 + (lane & 7) + ((lane & 8) ? 8 : 0);   // which row
int c = col0 +             ((lane & 16) ? 8 : 0);   // which column half
```

giving matrix 0 = rows 0-7/cols 0-7, matrix 1 = rows 8-15/cols 0-7, matrix 2 =
rows 0-7/cols 8-15, matrix 3 = rows 8-15/cols 8-15 — which is precisely
`a[0..3]`.

**The trap.** For the B operand, whose source is `W[n][k]`, the roles of
`(lane & 8)` and `(lane & 16)` swap: the *k* axis moves with `& 8` and the *n*
axis with `& 16`. Get it backwards and `r1`/`r2` are silently exchanged. Every
address stays in range. Every `mma` still issues. Nothing crashes. You get
plausible wrong numbers, and if — as we did — you have a unit test that passes,
you will look everywhere else first.

Our unit test passed because it contained *its own copy* of the address
function, and that copy was correct. **A unit test that reimplements the thing it
tests pins the idea, not the code.** The tests now compile the real header
(`csrc/tests/core.cuh` is `encoder.cu` with the torch bindings stripped by
`sed`), so there is only one copy of anything.

**Shared memory row pitch.** `ldmatrix` reads 8 rows of 16 bytes. For those to
land in 8 different bank groups you need the row pitch to be odd in units of 16
bytes. We use `AROW = 256 + 8 = 264` halves = 528 bytes; `528/16 = 33`, and
`33 mod 8 = 1`, so consecutive rows step through bank groups one at a time and
the 8 rows never collide.

*Alternative considered:* XOR swizzling, which costs 0 bytes of padding instead
of our 8 halves per row. We chose padding because at these sizes the 16 KB it
costs is affordable, and swizzle bugs are silent in the same way the `ldmatrix`
bug above is. Padding is a fully checkable arithmetic property; a swizzle is an
invariant you have to maintain at every access site.

## 5. Design v1: transplanting the Triton mental model

The reasoning that produced the first version:

> The Triton kernel spends **90 `bar.sync` per layer**, because every weight tile
> goes global → registers → shared → `ldmatrix` → `mma`, and each transit through
> shared memory costs a CTA barrier. In CUDA we can stage explicitly with
> `cp.async` and per-tile barriers and cut that to ~14.

So v1 did what a Triton kernel does: cooperatively stage a weight tile into
shared memory, barrier, everyone reads it with `ldmatrix`, barrier, next tile.

```cpp
for (int k0 = 0; k0 < kdim; k0 += KTILE) {
    __syncthreads();                       // everyone stopped reading the old tile
    load_wtile(wtile, w_global, k0, ...);  // LDG into registers, STS into shared
    __syncthreads();                       // the new tile is complete
    gemm_tile(acc, a_base + k0, wtile, warp * 32, lane);
}
```

**It was 7.7% slower than Triton.** 39.2k against 42.8k.

## 6. The measurement that killed it

Not a profiler — just the compiler and the disassembler. This workflow is worth
internalising because it is fast, needs no GPU, and answers structural questions
exactly:

```bash
# register and shared-memory usage, plus spills
nvcc -arch=sm_89 -O3 -std=c++17 -Xptxas -v -c stub.cu -o stub.o

# what actually got emitted (SASS, the real machine code, not PTX)
cuobjdump -sass stub.o | grep -c '\bBAR\.'
```

Results for v1:

```
Used 255 registers, 0 bytes spill
101 BAR (static)   286 LDG   284 STS   388 LDSM   800 HMMA
```

Three things, each fatal.

**(a) The barrier count went the wrong way.** 101 is the *static* count; the FFN
chunk loop is not unrolled, so the dynamic count per layer is:

| site | k-tiles | barriers |
|---|---|---|
| Q, K, V — three separate `gemm_full` | 8 each | 48 |
| output projection | 8 | 16 |
| FFN, 4 chunks × (ff1 + ff2 + 1) | 8 + 8 | 132 |
| LayerNorm / residual / attention | — | ~6 |
| **total** | | **≈202** |

Against Triton's 90. The rewrite whose entire justification was cutting 90 to 14
shipped **202**. Not a missed prediction — a prediction with the wrong sign.

The mechanism is visible in the four lines above: single-buffered staging means
two barriers per k-tile, and `KTILE=32` against `kdim=256` makes that 8 tiles per
GEMM. Nothing overlaps: load, stop, compute, stop.

**(b) 255 registers.** 255 × 256 threads = 65,280, essentially the SM's entire
64K register file. Combined with 89.5 KB of shared memory, that pins the kernel
to **1 CTA per SM = 8 warps out of a possible 48**. Every barrier is fully
exposed, because there is no other block resident to run during the stall. On the
Triton side the same stalls get covered by a second block.

**(c) 286 LDG + 284 STS.** Every weight still travels global → registers →
shared. `cp.async` — the thing the whole design was chosen for — was in the
design document and not in the kernel. I had benchmarked the unpipelined skeleton
as though it were the design.

**The time budget.** Per CTA: 16384 boards / 24 SMs = 683 CTAs per SM, so at
417.7 ms each CTA occupies its SM for ~611 µs. The arithmetic floor is 411 MFLOP
at 35.5/24 = 1.479 TFLOPS per SM = **278 µs**. So 55% of the time was overhead.

Modelling that overhead (shared memory and L1 both sustain ~128 B/cycle/SM):

| traffic per layer per CTA | bytes | ≈ cycles |
|---|---|---|
| weights: LDG through L1 | 1.58 MB | 12,300 |
| weights: STS into shared | 1.58 MB | 12,300 |
| weights: LDSM out of shared | 1.58 MB | 12,300 |
| A operand: LDSM out of shared | 1.5 MB | 11,700 |
| **total** | **6.2 MB** | **≈48,600** |

against an arithmetic floor of ~50,400 cycles per layer. **The kernel was moving
almost exactly as many bytes through the LSU as it was doing arithmetic**, and
with 8 warps and no double buffering, very little of it overlapped.

## 7. The reuse analysis

Here is the whole thing:

> A weight element is read **once per CTA and never reused inside it**. A board
> is 32 tokens, so each weight participates in 32 MACs and is then finished.

Shared memory exists to hold data that will be read *more than once* by
*different warps*. Ask, for each operand, who reads it and how often:

| operand | read by | times | belongs in |
|---|---|---|---|
| activations (A) | all 8 warps | 8× | **shared memory** |
| weights (B) | exactly one warp, once | 1× | **nowhere — go straight to registers** |

Warp `w` owns output columns `[32w, 32w+32)`, so it reads weight rows `32w..32w+31`
and no other warp ever touches them. Staging them in shared memory buys **zero
reuse** and costs three LSU passes where one would do, plus the barriers that
exist only to protect the buffer.

This is why the `cp.async` plan was the wrong fix. `cp.async` makes staging
cheaper and overlappable — a real improvement to a step that **should not exist**.
The right move is not to pipeline the copy. It is to delete it.

Two objections worth answering, because they are the reason this isn't obvious:

*"Won't the global loads be badly coalesced?"* They would be, in the natural
`[out][in]` layout — a B fragment wants 16 bytes from each of 8 rows that are 512
bytes apart. But the layout is ours to choose. See §8.

*"Isn't shared memory much faster than global?"* For a *reused* value, yes. For a
value read once, a global load that hits L2 and a shared-memory load cost
comparable LSU throughput, and the global path skips the store entirely. Our
weights are 12.6 MB total against 32 MB of L2, and all 24 SMs march through them
roughly in step, so essentially every weight load is an L2 hit. DRAM sits at ~1.5%
utilisation.

## 8. Design v2: pre-swizzled weights, straight into registers

A B fragment is **4 bytes per lane**, and the lane → element map is fixed at
compile time. So permute the weights on the host, once at construction, into
exactly the order the lanes want them. Then loading a B fragment is one
*perfectly coalesced* 128-bit load per lane, direct into the registers `mma`
consumes. No shared memory. No `ldmatrix`. No barrier.

The packed layout for a matrix `[N][K]`, in halves:

```
packed[n8][k32][lane][8]          n8 = n/8,  k32 = k/32
```

where lane L (`g = L/4`, `t = L%4`) holds, for `j = 0..7`:

```
j = 0,1   W[8*n8 + g][32*k32      + 2t], +1   → b[0] of k-step 2*k32
j = 2,3   W[8*n8 + g][32*k32 +  8 + 2t], +1   → b[1] of k-step 2*k32
j = 4,5   W[8*n8 + g][32*k32 + 16 + 2t], +1   → b[0] of k-step 2*k32+1
j = 6,7   W[8*n8 + g][32*k32 + 24 + 2t], +1   → b[1] of k-step 2*k32+1
```

One `uint4` per lane is **two complete B fragments**, and the warp's 32 lanes
read 512 contiguous bytes. On the host this is one `view` + `permute` + `reshape`:

```python
v = w.reshape(n // 8, 8, k // 32, 2, 2, 4, 2)
#             n8      g  k32      k16 h8 t  pair
return v.permute(0, 2, 1, 5, 3, 4, 6).reshape(-1)
```

Why two k-steps per load rather than one: it makes the per-lane load 16 bytes
(`LDG.128`) instead of 8. At 8 bytes the instruction *count* would double and
land at roughly 768 loads per warp per layer, against 384 for the staged version
— we would have traded LSU bytes for issue slots. At 16 bytes it is 384, the same
count as before, with the STS and LDSM passes deleted outright.

The inner loop:

```cpp
template <int K32N>
__device__ __forceinline__ void gemm_direct(uint32_t acc[2][4][2], const half* a_base,
                                            int a_stride, const half* w, int n8_0,
                                            int k32_0, int nk32, int lane) {
    const uint4* wp = reinterpret_cast<const uint4*>(w);

    uint4 pre[4];                                   // one group kept in flight
    for (int p = 0; p < 4; ++p)
        pre[p] = wp[((size_t)(n8_0 + p) * nk32 + k32_0) * 32 + lane];

    #pragma unroll 1
    for (int j = 0; j < K32N; ++j) {
        uint32_t af[2][2][4];                       // A still comes from shared
        for (int m = 0; m < 2; ++m)
            for (int kk = 0; kk < 2; ++kk)
                ldmatrix_x4(af[m][kk],
                            a_frag_addr(a_base, m*16, j*32 + kk*16, a_stride, lane));

        uint4 cur[4];
        for (int p = 0; p < 4; ++p) cur[p] = pre[p];
        if (j + 1 < K32N)                           // prefetch the next group
            for (int p = 0; p < 4; ++p)
                pre[p] = wp[((size_t)(n8_0 + p) * nk32 + k32_0 + j + 1) * 32 + lane];

        for (int p = 0; p < 4; ++p) {
            uint32_t b0[2] = {cur[p].x, cur[p].y};
            uint32_t b1[2] = {cur[p].z, cur[p].w};
            for (int m = 0; m < 2; ++m) {
                mma_f16(acc[m][p], af[m][0], b0);
                mma_f16(acc[m][p], af[m][1], b1);
            }
        }
    }
}
```

Per iteration: 4 `ldmatrix` (A), 4 `LDG.128` (B), 16 `mma`. The `pre`/`cur`
copy is a one-deep software pipeline — the loads for group `j+1` are issued
before the 16 `mma` of group `j`, so the ~300-cycle L2 hit is covered by
arithmetic instead of stalling. Four independent loads per warp × 8 warps is
ample memory-level parallelism; a deeper pipeline is unaffordable anyway, since
the accumulators are the register budget.

`template <int K32N>` rather than a runtime argument: with a runtime trip count
ptxas keeps the loop rolled and cannot fold the `j + 1 < K32N` guard. Every call
site knows its depth. (`#pragma unroll 1` on top of that is deliberate and
measured — see §12.)

**Results, from the same static analysis:**

| | v1 (staged) | v2 (direct) |
|---|---|---|
| barriers / layer | 202 | **14** |
| registers / thread | 255 | **134** |
| STS / layer | 284 | **12** |
| shared memory / block | 89.5 KB | **49.5 KB** |
| **evals/s, masked** | **39.2k** | **55.0k** |

## 9. The parts that are not the GEMM

**LayerNorm** is warp-local by a different assignment than the GEMMs: warp `w`
owns *rows* `w, w+8, w+16, w+24`, and one lane owns 8 contiguous columns of a
row. The reduction is then two `__shfl_xor_sync` and never leaves the warp.

This is the reason the residual stream lives in shared memory rather than in
registers. A register-resident stream would spread each row across all 8 warps
(since warps own *columns* for the GEMM), so every norm would need a cross-warp
reduction and a CTA barrier. Putting the stream in shared memory lets each phase
choose the assignment that suits it, and pay only for the handoff.

**The variance trap.** The one-pass form `Var = E[x²] − µ²` is one reduction
instead of two, and we used it. It produces **NaN**. A dead token is allowed to
hold any finite value, including 65504; at that magnitude `sq/Dm` and `mean*mean`
are both 4.29e9, where an fp32 ulp is 512, so the true zero rounds *negative* and
`rsqrtf` returns NaN. The attention mask multiplies dead V by an exact zero, and
`0 * NaN = NaN`, so one dead row poisons all 32 tokens of the board. The fix is
`fmaxf(sq/Dm - mean*mean, 0.f)` — one instruction. This was caught by a test that
sets dead rows to the fp16 ceiling and requires live outputs to be *bit-identical*.

**Attention** runs entirely in one warp on registers, using Facts 1 and 2 from
§3: S = Q·Kᵀ takes its B operand directly out of K's D fragments, softmax is two
shuffles, and P (an fp16 D fragment) becomes the A operand of P·V by register
rename. Only V needs a memory round trip, for the transpose, via
`ldmatrix.x4.trans` — and since it is warp-private that costs `__syncwarp()`, not
a CTA barrier.

**The dead-token mask** is a 32-bit word, one bit per slot, tested against each
lane's columns to force `-inf` before the softmax max. Dead *keys* are masked;
dead *rows* are computed and never read.

## 10. Choosing the warp tiling

Eight warps have to divide an M×N output tile. Let the warps form a `Wm × Wn`
grid with `Wm·Wn = 8`. Each warp reads the full K extent of its M-slice and its
N-slice, so shared-memory read traffic per k-slab is

```
traffic = M·K·Wn + N·K·Wm
```

For our GEMMs (M=32, N=256, K=256):

| Wm × Wn | A traffic | B traffic | total |
|---|---|---|---|
| **1 × 8** | 32·256·8 = 65,536 | 256·256·1 = 65,536 | **131,072** |
| 2 × 4 | 32,768 | 131,072 | 163,840 |
| 4 × 2 | 16,384 | 262,144 | 278,528 |

`1 × 8` wins, because N ≫ M — a warp should stretch along the *long* axis. This
is also the assignment attention wants (§2), so there is no conflict. With M=32
there was never much choice, but it is worth knowing the formula rather than
guessing: for a squarer tile the answer flips.

## 11. The barrier trap, and other second-order effects

Deleting `gemm_full` broke the kernel in a way worth describing, because it is a
class of bug rather than a typo.

`gemm_full` opened with `__syncthreads()`. That barrier was there for its *own*
buffer. But three other handoffs had come to depend on it:

* LayerNorm writes the activation buffer **by row** (warp `w` owns rows `w, w+8, …`)
  and the next GEMM reads **all** rows.
* `store_frags` writes it **by column** (warp `w` owns columns `32w..32w+31`) and
  the next GEMM reads **all** columns.

Both are cross-warp handoffs with no barrier of their own — they were free-riding
on a barrier that belonged to something else. Remove it and you get an error of
4.14 on outputs of magnitude ~4.5.

There is no tooling for this. `compute-sanitizer --tool racecheck` will find some
of it; what actually protects you is stating the dependency at the site that
needs it, with a comment explaining *which* two accesses it separates, so the
next person deleting a function knows what they are taking with them.

**Shared memory and aliasing.** With the weight tile gone, the block needs only
three activation buffers. The V-transpose scratch and the FFN hidden chunk have
disjoint lifetimes, so they share one allocation. 89.5 KB → 49.5 KB.

**The occupancy arithmetic**, which is worth doing by hand once:

```
per SM: 65,536 registers, 102,400 B shared memory
our block: 256 threads

134 regs → 134 × 256 = 34,304 regs;  2 blocks = 68,608 > 65,536  ✗  1 block/SM
128 regs → 128 × 256 = 32,768 regs;  2 blocks = 65,536 = 65,536  ✓
                       49.5 KB × 2 = 101,376 B ≤ 102,400 B        ✓
```

So 2 blocks/SM is reachable — and §12 says what happened when we took it.

## 12. Things that were tried and did not work

Kept because knowing the shape of the failures is most of the value.

**Unrolling the k-loop.** Intuition says unroll for overlap. Measured:

| unroll | registers | stack frame | cuda + mask |
|---|---|---|---|
| **1** | 134 | 0 | **297.8 ms** |
| 2 | 136 | 0 | 298.9 ms |
| 4 | 166 | 0 | 357.5 ms (**+20%**) |
| full | 246 | **192 bytes** | spilled to local memory |

Eight hoisted `LDG.128` have live ranges long enough that keeping them alive
costs more than the overlap they buy. Pinned at `#pragma unroll 1` *with the
table in a comment*, because it looks like an oversight otherwise.

**Forcing 2 CTAs/SM** via `__launch_bounds__(THREADS, 2)`. ptxas obliged at 122
registers with **zero spills**, everything fit — and there was **no speedup at
all**. Occupancy had stopped being the constraint the moment the barriers went
away. Reverted to `1`.

**`cp.async` multistage staging.** The plan v1 was built to enable. Made
unnecessary by §7: it optimises a step that no longer exists.

**Fusing Q, K, V into one k-loop.** They are three GEMMs over the same A operand,
so fusing would load A fragments once instead of three times: ~256 KB/layer,
about 2,000 cycles, ~2.5%. Not done — it needs three accumulator sets and three
prefetch groups live simultaneously, and registers are the binding budget. On the
list, not obviously positive.

**Earlier dead ends, from the Triton era:** `torch.compile`/CUDA graphs (0%),
cache/evict hints (0 to −8%), smaller GEMM tiles via CUTLASS (2× worse),
reductions via `tl.dot` instead of `tl.sum` (1.85× worse — the layout conversion
costs more than the reduction).

## 13. How to measure on this laptop

Sustained clocks drift ±3%, which is larger than most deltas worth chasing. A
naive before/after once produced a fake −4.5% "regression" in this project.

* **All candidates in one process, interleaved, with the order reversed on
  alternate rounds**, so the position penalty cancels rather than being read as a
  result.
* **Quote ratios, not absolutes.** Absolute evals/s from a cold card and a hot
  one differ by more than the deltas. The Triton kernel stays in the benchmark
  permanently as an in-process control: comparing `cuda/triton` within one run is
  valid; comparing 294 ms from one process against 296 ms from another is not.
* **Static analysis first.** `ptxas -v` and `cuobjdump -sass` need no GPU, take
  seconds, and answer "how many barriers", "how many registers", "did it spill"
  exactly. Every real finding in this document came from there first.
* **Device-level unit tests with no Python and no torch.** `csrc/tests/*.cu`,
  one `nvcc` line each. They let you bisect a layout bug in seconds instead of
  minutes, and they eliminated four hypotheses by measurement during the
  `b_frag_addr` hunt.

Traps: `ncu` and torch SDPA crash together (`Cannot load symbol
cudnnGetVersion`), so profile without torch attention in-process. Anything over
48 KB of dynamic shared memory needs `cudaFuncSetAttribute`, and without it the
launch fails while `cudaDeviceSynchronize()` still reports "no error" — check
`cudaGetLastError()` too, or you will compare against an untouched buffer and get
a small, believable, entirely fictional delta. (This happened, and cost an
afternoon of chasing a nonexistent kernel bug.)

## 14. Where it stands and what is left

61.3k evals/s = 25.2 TFLOPS = **71% of the measured fp16 mma issue rate**.

### The model in this section was checked, and it was roughly right

The first version of this document predicted the ~28,600 cycles/layer above the
arithmetic floor as LSU traffic, at 128 B/cycle/SM:

| per layer per CTA | bytes | predicted cycles |
|---|---|---|
| A fragments out of shared | 1.5 MB | 11,700 |
| weights out of L1/L2 | 1.58 MB | 12,300 |
| **total** | **3.1 MB** | **≈24,000** |

and said the honest next step was `ncu` stall reasons rather than more
arithmetic. That step was taken. `l1tex__data_pipe_lsu_wavefronts.sum` measured
**29,100 wavefronts per CTA-layer** against the 24,000 predicted — the model was
sound, and the missing ~5,000 are the stores, the residual, and bank conflicts
the model ignored.

But the model was answering the wrong question. The stall breakdown showed the
top removable cost was not bandwidth at all:

| stall (cycles per issue-active) | at 55.0k | at 61.3k |
|---|---|---|
| `math_pipe_throttle` — tensor pipe full, the *good* stall | 2.75 | **8.61** |
| `long_scoreboard` — waiting on a global load | 1.71 | 5.39 |
| `wait` — fixed-latency dependency | 2.69 | 3.23 |
| `barrier` | 0.40 | 1.93 |

and the SASS showed the k-loop issuing **16 HMMA against 31 non-memory
instructions** — 15 register MOVs from a prefetch buffer copy, and integer
address arithmetic that never got strength-reduced because the packed row pitch
was a runtime argument. Neither shows up in a bandwidth model. Deleting them
(ping-pong buffers, pitch as a template parameter) cut ALU instructions by 66%
and was worth 4.6%.

### The lesson of the second round: budgets, not bandwidth

The larger win, 5.4%, was fitting two CTAs on an SM — which the previous round
had *measured as worthless* and written down as such. Both measurements were
correct. Occupancy was worthless when the kernel was ALU-bound and valuable once
it was memory-latency-bound. **A negative result about a resource is only valid
against the bottleneck it was measured under**, which is an argument for
re-running cheap experiments after any change that moves the bottleneck, not for
trusting the ledger.

Fitting required 512 bytes. Shared memory has a hard cliff at **50,176 B/block**
on this card — measured with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
not read off a spec sheet — and the kernel sat at 50,688. The 512 bytes came from
`bufA`, the residual stream, which is the only buffer no `ldmatrix` ever reads
and so the only one that does not need its 8-half row pad. Unpadding it made the
in-place residual collide 8 ways (bank conflicts 7.2M → 46.2M), which was fixed
by staging the fragment block in a padded buffer and adding it row-wise, for no
extra barrier.

Two CTAs of 256 threads on a 65,536-entry register file is then **128 registers
per thread**. The kernel uses 116. That budget is now what blocks everything
else: fusing Q/K/V into one k-loop, prefetching weights two steps ahead, and
double-buffering the A fragments are all correct optimizations that each need
more registers than 12 leaves. The occupancy win bought 5.4% and cost the
ability to spend registers — and the things it locked out were measured at under
1% each, so it was the right trade, but it is a trade and not a free lunch.

### What is actually left

The kernel is now bounded by `math_pipe_throttle`: the tensor pipe refusing new
work is the largest single stall, which is the condition you want. The absolute
headroom in this structure is +23%, and reaching it means loosening one of the
two budgets rather than finding another instruction to delete.

**Two boards per CTA** is still the one change that breaks the deadlock instead
of trading inside it: the same weights would serve 64 tokens instead of 32, so
weight traffic per board halves and arithmetic intensity goes from 32.5 to 65
FLOP/byte. It needs 4 m-tiles of accumulators and a 64-row buffer triple —
101,376 B, one CTA per SM again. So it is a direct bet against this round's main
result, and it has to beat it by more than the 5.4% it gives back.

The deeper point stands and has now been made twice. The first round was fast
because of an analysis that fits on one line — *weights have no intra-CTA reuse,
so they must not enter shared memory* — and slow for a day because a mental model
was ported instead of derived. The second round found 11.8% by reading the SASS
and the stall counters and disbelieving a written-down negative result. Neither
came from theory about the machine; both came from measuring the specific thing
in front of us.

## 15. B2: the prologue and epilogue, and one thing I got wrong

B2 puts the embedding gather in front of the stack and the final norm plus the
three heads behind it, so that a self-play step hands the kernel 64 bytes of board
and gets 4 KB of logits back, with no activation ever reaching HBM.

Two predictions to check against, both made before writing anything.

**Bandwidth.** Before B2 the kernel read a `[32,256]` fp16 tile and wrote one back:
32 KB per board, which at 61.3k evals/s is 2.0 GB/s. After B2 the input is 64 B of
board plus a control word plus a repetition byte, and the output is 4 KB of policy,
512 B of promo and one float — about 4.9 KB, 6.5× less. Both numbers are noise on a
250 GB/s bus. The prediction was "bandwidth is irrelevant in both directions" and
that held: nothing about the change is visible in the timing.

**Arithmetic.** Policy is `2·32·256·64` = 1.05 MFLOP against 411 MFLOP per
evaluation, so 0.26 %; the promo and value heads are 0.03 %; the five gathers are
41k adds. Total
0.3 %, amortised over eight layers. That held too.

**Registers.** This is the one I got wrong, and it is the interesting part. The
prediction was "+0 to +4 registers, no spill, because both blocks sit outside the
loop". What actually happened: 113 registers and no spill became the 128-register
cap with 56 bytes of spill stores, and the spilled loads and stores appeared
*interleaved with the mma blocks*. The loop was paying for code it never runs.

The mechanism is that ptxas allocates one register budget for the whole kernel.
Cold code that wants registers does not slow itself down — it reduces what the hot
loop has left. *Outside the loop* is not the same as *outside the loop's register
budget*, and with only 12 registers of headroom under the two hard budgets, there
was no slack to absorb it.

The fix is to stop inlining. `__device__ __noinline__` on the gather and on the head
epilogue gives each an ABI frame of its own; the call costs a handful of cycles once
per board, and the loop gets its allocation back. Spill stores fell from 56 bytes to
12, and the backbone path measures identical to before B2 against the unchanged
Triton control. `docs/perf.md` has the four-row table.

Two smaller consequences worth knowing, because both changed the *spec* rather than
the code. The gather accumulates one table at a time rather than loading five and
then adding, which is 16 live registers instead of 40. And the clock and repetition
tables, being per position rather than per token, are summed into one vector per
board — which fixes the summation order as `(square + type_special + color_turn) +
(clock + rep)`. In fp16 that grouping is part of the answer, so §7.2 now states it,
and `tests/test_b2.py` holds the gather to bit-identical rather than to a tolerance.
An approximate test there would have passed with the adds regrouped, and then the
two implementations would have drifted on the one quantity in the network that has
no rounding of its own to hide behind.

Where the head GEMM reads from is the last detail, and it is a consequence of the
previous round. The final residual lives in `bufA`, which is the one buffer with no
row padding — dropping those 512 bytes is what bought the second CTA per SM — so a
head GEMM reading it through `ldmatrix` would collide eight ways. The final norm has
to happen anyway; it writes `bufB`, which is padded, and the problem never arises.
Two independent reasons for one LayerNorm, which is usually a sign the design is
sitting where it belongs.
