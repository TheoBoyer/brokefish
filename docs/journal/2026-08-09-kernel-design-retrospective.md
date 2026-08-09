# 2026-08-09 (eleventh entry) — what I would change about the kernels, and what I would not

Written from `docs/ledger/perf.md`'s measurements rather than from taste, after a week
spent on the loop above the kernels rather than in them. Everything below is either a
number from the ledger, arithmetic **labelled as arithmetic**, or an opinion labelled as
one. Nothing here has been tried.

## The design as it stands, and why it is right

`csrc/encoder.cu` is a **board-per-CTA, activation-resident pipeline**: one board's
32×256 activations stay on chip through all eight layers and never round-trip to HBM,
while the 12.2 MB of weights stream past them.

That bet is validated twice over. Versions 0 and 1 round-tripped activations to HBM and
were slower. And the kernel reaches **25.2 TFLOPS against cuBLAS's 16.1-16.5** on the
same per-layer shapes at `M = 524288` — a 1.5× win from fusion alone, on the friendliest
possible batch dimension for the library. **I would keep this.** It is the load-bearing
idea and it was correct.

## Where it is stuck, in the ledger's own words

| | |
|---|---|
| achieved | 25.2 TFLOPS, **71 % of the 35.5 TFLOPS** fp16-accumulate issue rate |
| top stall | `math_pipe_throttle` **8.61** cycles per issue-active — the tensor pipe refusing work |
| then | `long_scoreboard` 5.39, `wait` 3.23 |
| occupancy | `sm__warps_active` 32.1 % = **16 of 48 warps**, 2 CTAs/SM, 128 registers, 50 176 B SMEM |
| headroom | **+23 %**, and every lever that reaches it is blocked by the register budget |

The identified fix is **two boards per CTA** — it halves weight traffic *per board*
rather than trading one resource for another — and it needs **101 376 B**, which is
exactly the 99 KB limit. So it forces 1 CTA/SM and swaps a register problem for an
occupancy one. That is the deadlock.

⚠️ **Arithmetic, not a measurement:** each CTA re-reads the whole weight set for its one
board, so a `B = 4096` call moves ~4096 × 12.2 MB ≈ **50 GB** through L2 (the weights are
32 MB-L2-resident, so this is L2→SM traffic, not HBM). Against ~67 ms of compute for the
same call at 25.2 TFLOPS, that is roughly a third of the time budget — which is
consistent with `long_scoreboard` sitting second, and is the quantity two-boards-per-CTA
would halve.

---

## What I would change

### 1. Native fp8 weights, not fp8 retrofitted into the FFN

The one I would do first, because it attacks **both** limiters at once. On Ada, e4m3
issues at **41.6 TFLOPS against fp16's 35.5**, *and* halves the weight traffic that the
arithmetic above puts at a third of the budget.

Today fp8 covers the FFN's two matmuls only — two thirds of the FLOPs — and measured
**1.15×**, which is about what halving a third of the traffic predicts. Extending it to
QKV and the output projection should be worth most of another ~1.1×, and more
importantly it shrinks the weight tiles, which is the SMEM pressure blocking
two-boards-per-CTA. **The two levers may come for the price of one.**

Designing for fp8 from the start also fixes the accumulator layout and the scaling
strategy up front. Retrofitting it is exactly what produced NaN on 116 boards of 128
from a `q_max` that disagreed across two languages, while every isolated component
tested clean.

> ⚠️ **Correction, same day, after reading `encoder.cu` instead of reasoning about it.**
> The paragraph above originally continued *"and it shrinks the weight tiles, which is
> the SMEM pressure blocking two-boards-per-CTA — the two levers may come for the price
> of one."* **That is false.** `encoder.cu:52`: *"Three activation buffers and nothing
> else — no weight tile, because the weights never pass through shared memory."*
> Weights go global → registers → mma and never occupy SMEM at all, so making them fp8
> frees **no** shared memory and unlocks no extra board per CTA. What fp8 weights buy is
> real but narrower: half the L2→register traffic, and the 41.6 TFLOPS e4m3 issue rate
> instead of 35.5. The SMEM budget is **activations**, and only an fp8 *residual stream*
> would move it. See the occupancy arithmetic below.

### 2. Solve the occupancy equation on paper before writing the kernel

The current design arrived at 1 board/CTA and 2 CTAs/SM, and *then* discovered that
every remaining improvement is register-blocked: double-buffered A fragments, k-loop
unrolling, fused QKV and `HCHUNK 512` all "exceed the 128-register or 50 176-byte
budget".

Enumerating `(boards/CTA, CTAs/SM, weight precision)` against 99 KB of SMEM and 64K
registers per SM is a small search, and it is the decision every later tuning inherits.
Doing it after the kernel exists means discovering the constraint one blocked idea at a
time.

**Solved, from `encoder.cu`'s own constants.** `T = 32`, `Dm = 256`, `NWARPS = 8` so
`THREADS = 256`; `SM_A = T·256`, `SM_B = SM_S = T·264` halves, totalling
**50 176 B** per board. Per SM the machine offers 65 536 registers, 102 400 B of
shared memory and 48 warps. At the shipped point of 2 CTAs/SM:

| resource | used | available | |
|---|---:|---:|---|
| registers | 2 × 256 × 128 = **65 536** | 65 536 | **100 %** |
| shared memory | 2 × 50 176 = **100 352** | 102 400 | **98 %** |
| warps | 2 × 8 = **16** | 48 | 33 % |

⚠️ **Both hard budgets are saturated to the byte and to the register.** That is not a
coincidence — `AROWA` drops bufA's padding precisely to save the 512 B that buys the
second CTA — but it does mean the design has no slack anywhere, and it is why every
listed optimisation came back "exceeds the 128-register or 50 176-byte budget".

The reachable points, with weight traffic per board relative to today:

| variant | SMEM/board | boards/CTA | CTAs/SM | boards/SM | regs/thread | wt traffic/board |
|---|---:|---:|---:|---:|---:|---:|
| shipped (fp16 activations) | 50 176 | 1 | 2 | 2 | 128 | 1.00× |
| fp8 weights everywhere | 50 176 | 1 | 2 | 2 | 128 | **0.50×** |
| two boards, fp16 activations | 100 352 | 2 | **1** | 2 | **255** | 0.50× |
| **fp8 activations too** | 25 088 | 2 | 2 | **4** | 128 | **0.25×** |

Two conclusions fall out.

**The ledger's two-boards-per-CTA proposal is really a register play, not a memory
one.** At 1 CTA/SM the register file divides by 256 threads instead of 512, giving the
hardware maximum of **255 registers per thread** — which is exactly what every blocked
lever needed. The price is 8 warps of 48 instead of 16, and the ledger already measured
that latency hiding suffers there: double-buffering the A fragments is worth *+0.56 % at
1 CTA/SM and ±0 at 2*, i.e. at 2 CTAs the extra warps already cover LDSM latency and at
1 CTA they do not.

**And the only variant that raises boards per SM without paying occupancy is the one
that puts the *activations* in fp8.** That is a numerics decision about the residual
stream across eight layers, not a layout decision, and it is a much larger accuracy
question than fp8 weights: `2026-08-04-fp8-encoder.md` measured 0.66 % max prior-space
error for FFN *weights*, and says three quarters of the error in the full-fp8 variant
came from the other two matmuls. Half measures do not fit either — keeping the residual
in fp16 and demoting only `SM_B`/`SM_S` gives 33 280 B per board, and two of those is
66 560 B, which still does not fit twice on an SM.

### 3. Do not let Triton's limitations pin the architecture

`CLAUDE.md`: Triton has no lists and no dynamic register indexing, "hence the
hand-unrolled `x0..x3` tiles, **which pin `d_model = 4 × 64`**".

An architecture parameter is downstream of a tooling limitation in a language this
kernel is no longer written in. `d_model` should be set by the scaling law and the
parameter budget; if it happens to be 256, fine, but it should not be 256 *because of an
unroll*. Writing it in CUDA C++ from the start removes the constraint entirely.

### 4. Compacted leaf staging instead of a fixed-`B` tensor

`simulate` hands the encoder all `B` staged leaves every iteration whether or not they
are active. That single fact shaped E1.2: a batch holding mixed budgets costs the
**maximum**, not the mean, so playout cap randomisation had to tie one budget across the
whole batch.

If the encoder took a **compacted leaf list plus a count**, then variable per-game
budgets, early termination, Gumbel's sequential halving and playout cap randomisation
would all be free instead of each needing a workaround. The compaction pass is cheap,
and per-board throughput is **flat from 4096 boards down to 128** (97.2 %, measured
2026-08-07), so a shrunken batch really does run shorter.

**This is the change that would most enlarge what the search is allowed to do**, and it
is a staging-interface change rather than a numerics one.

### 5. Grid-stride loops in the tree kernels

`root_init_kernel`, `descent_kernel`, `expand_kernel` and `backup_kernel` all index
`b = blockIdx.x * kWarps + (threadIdx.x >> 5)` and launch `blocks_for(B)` blocks — one
pass, grid pinned to the batch. A grid-stride loop costs nothing, decouples grid size
from `B`, and hands over an SM-footprint cap for free: launching at
`blocks_for(B) * 9 / 10` confines self-play to 90 % of the resident slots without MPS,
green contexts, or a second process.

### 6. Make the kernel measure itself against the ceiling

"71 % of the mma issue ceiling" comes from one `ncu` session on 2026-07-29, **before the
fp8 FFN landed**. The stall breakdown for the kernel actually running today is unknown.
A benchmark that reports achieved TFLOPS against the measured ceiling on every run keeps
that number alive rather than letting it decay into folklore — which it now has.

---

## What I would not change

- **The activation-resident fusion.** Measured, twice.
- **One warp per game in the tree kernels.** The tree ops are branchy and irregular, and
  a warp per game makes divergence per-game, which is the right granularity.
- **The index-aligned 32×64 policy.** Masking is a device-side AND, no gather, no host
  round trip. It is why the environment is 2.2 % of a node.

## ⚠️ A stale entry in the top-level dead-end list

`CLAUDE.md` lists **"fp16 accumulation (0 %)"** among the measured dead ends. `perf.md`
records the opposite: *"~~0 %~~ **refuted 2026-07-29: +19 %.** The original test cast the
result instead of changing the accumulator, so the PTX still held
`f32.f16.f16.f32`."*

The top-level file every session reads is telling future work not to try one of the
larger wins the project found. It is also the difference between an 18.0 TFLOPS ceiling
and a 35.5 one, which is the number the whole design is judged against.
