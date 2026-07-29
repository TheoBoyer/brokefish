# Brokefish — project dossier

*Last updated 2026-07-29.* The thesis, the numbers behind it, the engineering
state, and the decisions already taken. `CLAUDE.md` is the working summary; this is
where the reasoning lives, so consult the section you need rather than the whole
file. The only external dependency is the PyTorch reference implementation of the
chess engine, in `~/steakfish` — it is the correctness oracle, never a runtime
dependency.

---

## 1. What Brokefish is

**"What's the cheapest way to have a superhuman chessbot from scratch?"**

`README.md` (written by Théo) sets the tone and the rules; this briefing gives the
engineering state, the numbers, and the decisions already made.

The project is **a cost experiment**, not an attempt to beat Stockfish. The
scientific deliverable is a **clean cost-vs-Elo curve**, obtained starting strictly
from zero, on consumer hardware. A negative result ("we plateau at X Elo for Y €")
is publishable too — that's the point: the curve does not exist anywhere in the
literature (see §6).

### The tabula rasa boundary — non-negotiable, it *is* the project

Forbidden in **training**: human games, engine-generated games, engine labels,
pretrained networks, distillation, opening books, any inherited chess heuristic
(piece-value tables, PSTs, …). The **rules** of chess are allowed; **opinions**
about chess are not.

Why the strictness: if distillation is accepted, **the flag is already planted** —
CF-6M (arXiv:2409.12272) does 6M params on 1×A100 from public Lc0 data, and NNUE
has made superhuman play a consumer commodity since 2020. The degenerate case
("download Stockfish, it's free") closes the debate. The *only* version of the
claim still open is the strictly tabula rasa one, and it must be stated in the
title.

Allowed and necessary: **self-generated** position diversity — temperature over the
first ~30 plies, Dirichlet noise at the root, uniform random openings (NNUE-datagen
style), KataGo-style branching. That's generation, not imported knowledge.

Allowed in **evaluation**: standard opening books (UHO, TCEC), paired games with
both colors, Elo-calibrated opponents. Evaluation is *measurement*, it sits outside
the boundary (precedent: the AZ-SF match in the Science paper started from TCEC
openings).

---

## 2. The intended architecture (decided)

- **Env**: GPU chess engine on a 12-bit **piece-list** representation
  (32 × `uint16` per position, one slot per piece, plus one `int16` side-to-move
  word), not classical bitboards. Move generation entirely device-side. Full
  specification in `docs/spec.md`.
- **Network**: **piece-token transformer** — 32 tokens = 32 pieces, **summed**
  position/type/turn embeddings, **non-causal** backbone, **32×64 logits** policy
  head (piece × destination square) index-aligned with the legality mask the engine
  produces (32×`uint64` — exactly the same shape, so masking to -inf is a device-side
  AND with **zero host round-trip in the loop**). The network emits raw logits and
  the *search* applies that AND, decided 2026-07-30; spec §7.4 has the three reasons
  and the all-illegal-row hazard.
- **Size**: `d=256, L=8, H=8, FFN=1024` = **6.32M params ≈ 400 MFLOPs/eval**.
  This number is the result of a correction (§6): the original 1M-param bet is
  contradicted by scaling laws.
- **Search**: **Gumbel MCTS**, n=32 simulations in training (800+ at evaluation).
  Batched, no host synchronization.

---

## 3. Sizing (derived games-first, at Théo's insistence)

The chain of reasoning, worth keeping for the write-up:

| step | value | anchor |
|---|---|---|
| games to superhuman | **~10M** | lc0: 10M games → +2900 Elo in its first 10 months; AZ 44M @ 800 sims; KataGo 4.2M. Range 5-50M = **the weakest link in the estimate** |
| moves | ~80 plies/game → **8×10⁸ moves** | |
| aggregate latency target | **~325 µs/move** for a 3-GPU-day run | |
| NN evals | n=32 sims → **~10 µs/eval = 100k evals/s** | |
| total compute | 2.6×10¹⁰ evals × 400 MFLOPs ≈ **10¹⁹ FLOPs** | a few days of 4090/H100, 2-4 weeks of 4060 |
| budget | **~100-500 €** spot | |
| env | 100k steps/s needed vs 50-100M boards/s targeted | **<0.5 % of the budget** — the env will not be the bottleneck |
| training | ~10 % of self-play cost | 1 position/move ×3 vs 32 evals/move |

Structural consequence: **the system is NN-bound, not env-bound.** All optimization
effort goes into network inference in the tiny-token regime (T=32 tokens only — a
regime nothing in the kernel literature optimizes for).

---

## 4. GO/NO-GO gates (decided with Théo on 2026-07-28)

**Gate 1 — engineering.** Sustained evals/s of the d=256/L8 model, **measured
inside a synthetic MCTS loop** (live tree, real leaf batches, env in the loop,
bf16) on the RTX 4060 Laptop.
- **GO ≥ 45-50k evals/s** (≈30 % of tensor peak → full self-play ~7 days locally,
  or 1-3 days / 100-300 € on rented GPUs)
- **NO-GO < 15k** after a genuine optimization effort
- in between: decide on cost
- ⚠️ **a pure forward benchmark does not count** — the gate is decided in the loop.

**Gate 2 — science.** Pilot run: beat **AlphaGateau (~1830-2100 Elo)** in < X
GPU-days, with an Elo slope consistent with Jones' law (+500 Elo per 10× compute,
arXiv:2104.03113). Otherwise: a negative result, publishable at low cost.

Sequencing: engine (movegen + perft) → Gate 1 → Gate 2 → final run.

---

## 5. Engineering state as of 2026-07-29

### 5.1 The model kernel — `brokefish/nn/triton_impl.py` ✅ working deliverable

Full forward of all 8 layers of an `nn.TransformerEncoder` (norm_first, relu, fp16)
in **a single Triton kernel**. Numerically validated against torch
(rel 1.8e-3 < 5e-3) by `tests/test_model.py`.

**Measured ladder on RTX 4060 Laptop, B=16384 boards:**

| version | evals/s | note |
|---|---|---|
| torch eager (baseline) | 22-24k | ~15-20 % of tensor peak |
| `torch.compile` max-autotune | 21-23k | **zero gain**, verified under challenge |
| + `EFFICIENT` attention (instead of flash) | 24.9k | one line |
| fused block v2 (3 kernels) | ~34k | |
| fused encoder | 37.2k (440.8 ms) = ×1.79 | BM=32, warps=4, stages=1, maxnreg=168 |
| + dead-token mask | 36.4k (449.6 ms) = ×1.78 | spec §7.3, mandatory, costs 0.97 % |
| **+ fp16 accumulation (current default)** | **42.6k (384.4 ms) = ×2.12** | fp32 accumulate is half rate on GeForce Ada; +19 % in an isolated A/B |
| **GO** | **45-50k** | **×1.06 still missing** |

Run from the repository root: `python -m tests.test_model` (correctness) and
`python -m bench.bench_model` (throughput, interleaved A/B). Reference venv is
Triton 3.7.1; **Théo does the installs himself**, see §8.

**Diagnosed bottleneck** (ncu + nsys + PTX reading):
- DRAM at **1.5 %** — memory is no longer the issue, fusion did its job
- only 2 CTAs/SM, **register-locked** (255/thread; SMEM at 28.7 KB would allow 3);
  `maxnreg=168` forces the 3rd CTA and buys +2.5 %
- **No-Eligible 89.9 %**, math-pipe stalls 47.9 %
- Precise name for it: **instruction-mix serialization under phase
  synchronization**. Warps are generalist and sequential, and CTA barriers align
  them all on the same instruction type at the same moment → one pipe saturated
  while the others idle, in rotation.
- **The number that matters** (measured in PTX, body of one layer): 272
  `mma.sync`, 187 `ldmatrix`, 193 `st.shared`, **90 `bar.sync`**, 0 `cp.async`.
  We wanted 3 barriers per layer, there are 90: every weight tile goes
  `global → registers → SMEM → ldmatrix → mma`, and any transit through shared
  SMEM requires a CTA barrier. **720 barriers across the kernel.**

**The next step is written: a warp-specialized CUDA C++ kernel.** Théo's decision
on 07-29 (over a Gluon variant): learning CUDA is a goal in itself, and stacking a
beta tool on an already-experimental project doubles the risk. The spec is complete
and rests on measured facts:
- specialized warps (loader / consumer / storer) instead of generalist warps
- `cp.async` global→SMEM **with no register staging**, **per-tile mbarriers**
  instead of CTA barriers → warps dephase, phase alignment ends
- sm89-portable recipes from FA2 and Hazy's "No Bubbles" megakernel (SMEM paging
  with early release); TMA / `setmaxnreg` / clusters are unavailable on Ada and
  are not needed
- **known hardware trap**: `cp.async.mbarrier.arrive` **hangs** the GPU on sm89;
  the validated idiom is `commit_group` + `wait_group` + plain `mbarrier.arrive`
- **feasibility already proven**: a warp-specialized prototype runs correctly on
  this card (via Gluon) — the mechanism carries **no hardware risk**, only the
  writing effort remains.

**Remaining levers** (see `docs/perf.md` for numbers):
1. warp-specialized CUDA C++ kernel — **the path to GO**
2. fp8 e4m3 — untested, deferred by Théo ("we'll see later")
3. weight-stationary restructure (resident weights, streamed activations:
   ~200 GB of weight re-reads → ~26 GB of activations)
4. interleaving 2 attention heads, FA3-pingpong style (2-5 % expected, untested)
5. folding LayerNorm's γ/β into the following weights (≤1 %, exact, precomputable)

**Already-measured dead ends — do not redo them**: `torch.compile`/CUDA graphs
(0 %), fp16 accumulation (0 %), cache/evict hints (0 to -8 %), fp16 LN affine
(0.1 %), smaller GEMM tiles via CUTLASS (2× worse), reductions via `tl.dot` instead
of `tl.sum` (**1.85× worse** — the layout conversion costs more than the reduction),
`Var = E[x²]-µ²` for ILP (noise), head dimension (≤5 % = noise).

### 5.2 The engine — `csrc/` ⬜ specified, not written

Only `csrc/chess.cuh` is settled: the 12-bit piece-list word with its decode/encode
and the device helpers. The slot-stable layout is the load-bearing decision — it is
what makes the network's 32 piece tokens index-aligned with the engine's 32
legality masks, so masking the policy is a device-side AND with no gather and no
host round-trip.

The move generator itself is **not written here**. `csrc/README.md` holds its
contract: full legality (sliders with occlusion, castling, en passant, promotions,
pin and check filtering), fully device-side, validated differentially against
python-chess and then by perft, at 50-100M boards/s on the 4060.

That last figure is what keeps the environment under 0.5 % of the self-play budget,
and it is 2-3 orders of magnitude above pgx on an A100 — the technical edge of the
whole project. The benchmark of record is nodes/s against Ankan Banerjee's
`perft_gpu` (recompiled locally: his published numbers are from a 2013 GTX 780 and
use *bulk counting*, so they are not comparable as published) and against Stockfish
on CPU.

> A step-by-step version of this kernel is being written elsewhere as a CUDA
> learning exercise, against reference dumps from the Python engine. **It is not
> staged into this repository.** What lands in `csrc/` is one production kernel
> with its own test, once it passes perft.

---

## 6. Due diligence — what already exists (4 research agents, 07-28)

**Verdict: the flag is not planted, the niche is real.** Details in
`docs/due_diligence.md`. The points not to re-learn:

- **Only lc0 is superhuman from scratch.** 10M games → +2900 Elo over its first 10
  months (which validates our anchor), but distributed compute was never counted,
  ~8 GPU-years per run. Every known solo attempt dies at 1200-2000 Elo, or cheats
  via supervised bootstrap (CrazyAra, BadGyal, searchless 270M).
- **The budget framing has never been executed.** No cost-vs-Elo curve for chess
  has been published. Closest: a 2018 lczero forum estimate (~$15k), and the
  *search-contempt* paper (Apr 2025, arXiv:2504.07757) which **proposes**
  consumer-GPU feasibility without executing it. **People are circling: the clock
  is running.**
- **Bar to beat**: **AlphaGateau** (NeurIPS 2024, arXiv:2410.23753) — pgx + Gumbel
  128 sims + 1M-param GNN, 128k games, 13.7 days × 8×A5000 → **1830-2100 Elo**.
  Beatable.
- **Existing envs**: `pgx` (JAX) is the only mature full-chess GPU env —
  **~2×10⁵ steps/s on an A100**. Our target (50-100M boards/s on a 4060) is 2-3
  orders of magnitude above, on 20× cheaper hardware. `torchess` (CUDA) is a broken
  PoC (no mate detection, wrong repetition).
- **THE correction from due diligence**: the "1M params is enough" bet is **wrong**.
  Scaling laws (Neumann & Gros, arXiv:2210.00849) show an Elo plateau in
  parameters; the smallest ~superhuman networks known are ~5-6M and go through
  distillation + search. AlphaGateau, at exactly 1M params, plateaus at ~2100.
  → network revised to **5-10M params**, hence d=256/L8.
- Reassuring independent convergence: Jones' law applied from AlphaGateau
  (2100 Elo @ 128k games, +500 Elo/10×) → ~2850-2900 Elo around 5-50M games.
  Consistent with the lc0 anchor.
- **On pufferlib** (the original inspiration, verified in the code): their gains
  come from a C env, shared/pinned memory, ~150k-param networks, `torch.compile`
  (+20 %) and a single custom kernel (GAE). **Their bottleneck — the CPU↔GPU
  bridge — does not exist for us** (everything is device-side), and their network
  recipe is not enough for MCTS: forwards are sequential and dependent, hence the
  need for CUDA graphs / a persistent kernel.

---

## 7. What is NOT decided (the real remaining work)

The entire RL layer is untouched. No line written, no decision frozen:
- batched device-side Gumbel MCTS, sync-free (identified as **harder than the
  movegen**)
- tree layout in GPU memory, allocation, reuse between moves
- training loop, replay buffer, actor/learner cadence
- Elo evaluation protocol (calibrated opponents, books, time control)
- cost accounting (the € counter is a deliverable, not a footnote)

---

## 8. Hardware and tooling

### Hardware realities to never forget

**RTX 4060 Laptop, 8 GB, sm89 (Ada).** No TMA, no wgmma, no clusters — but fp16/fp8
tensor cores via `mma.sync` and `cp.async` are available. 99 KB SMEM per block, 64K
registers/SM.

⚠️ **Clocks drop to 1.38-1.5 GHz under sustained load** (power limit; 2055 MHz is
only observed between runs). All roofline reasoning uses **~35-40 TFLOPS fp16
sustained**, not the nameplate. Thermal drift reaches ±3 %: **only order-balanced
interleaved A/B comparisons are valid** (a naive before/after already produced a
fake -4.5 % gain in this project).

For heavy phases: Modal, $30 free credits/month, per-second billing (H100 ≈ $3.95/h,
B200 ≈ $6.25/h).

### Tooling traps already paid for

- **Triton has no lists**: comprehensions only work over existing tuples, no
  `tuple()` in jit scope, no dynamic register indexing. Hence the hand-unrolled
  `x0..x3` tiles in `brokefish/nn/triton_impl.py`, and the structural specialization to
  D/BC=4. Loop **bounds do accept constexpr expressions** though
  (`range(3*D//BC)` compiles) — the hardcoded `range(12)` in the kernel is
  sloppiness to fix, not a tool limit.
- **ncu and torch SDPA crash together** (`Cannot load symbol cudnnGetVersion`).
  Workaround: profile with no torch attention in the process at all (build the
  reference outside the profiled run).
- `nsys stats` serves a stale sqlite → always `--force-export=true`.
- **Never truncate an error message** (a 100-character truncation cost 10 minutes
  of blind debugging).

---

## 9. Repository layout, and what does not belong here

```
brokefish/   Python package — model.py today; MCTS, env bindings and training
             will land here
csrc/        CUDA C++ — chess.cuh (settled); movegen and the fused encoder
             kernel are specified in csrc/README.md but not written
tests/       correctness, torch as the oracle       python -m tests.test_model
bench/       throughput, interleaved A/B protocol   python -m bench.bench_model
docs/        perf.md (ledger + measured dead ends), due_diligence.md (prior art)
```

**This is a production repository.** No staging areas, no step-by-step ladders, no
snapshots of work happening somewhere else, no exploration scripts. Code lands here
when it is correct and tested; everything upstream of that belongs in a scratch
directory.

The PyTorch reference implementation of the chess engine lives in `~/steakfish`.
It is the correctness oracle for the movegen and nothing else: no Brokefish code
imports it, and `docs/spec.md` restates everything a kernel needs.

**This repo is not under git yet** — Théo's call, don't run `git init` for him. It
is meant to become partially public through the write-ups.

---

## 10. If you don't know where to start

In decreasing order of value, given the state above:

1. **Land the movegen** in `csrc/`, to the contract in `csrc/README.md`: full
   legality, device-side, differential test against python-chess, then perft. It is
   the blocking artifact, and it is CUDA written by Théo.
2. **Write the warp-specialized CUDA C++ encoder kernel** — the only identified path
   to GO (×1.25 to find). The §5.1 spec is complete: the bottleneck is named, the
   recipes are known, the mechanism is proven on this hardware, both failure modes
   are mapped.
3. **Build the synthetic MCTS loop** — without it Gate 1 cannot be cleared (a pure
   forward doesn't count). It is also the prototype of the real self-play loop.
4. Everything else (§7) comes after.
