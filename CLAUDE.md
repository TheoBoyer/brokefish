# CLAUDE.md — brokefish

**"What's the cheapest way to have a superhuman chessbot from scratch?"** A cost
experiment, not an attempt to beat Stockfish. The deliverable is a clean cost-vs-Elo
curve on consumer hardware — one that does not exist in the literature. A negative
result is publishable too.

## Hard rules
1. **Answer in the chat.** Writing to a file never substitutes for answering. The
   complete answer is the **last message of the turn**; tool calls and note-taking
   come first. Never end a turn on "recorded".
2. **No flattery, no complacency.** "I don't know" is valued; bluffing is the only
   real failure. Never claim an unmeasured performance number.
3. **Predict → measure → explain the gap**, in that order, for every kernel.
4. **Kernels**: the goal is a world-class kernel. Comment the hardware reasoning,
   not the C++.
5. **A step is finished only when its number is written down.**
6. **Strict scope**: only comment on or fix what was asked.
7. Long campaign: a `tail -f`-able log path up front, a detailed summary at the end.

## Tabula rasa boundary — non-negotiable, it *is* the project
Forbidden in training: human games, engine games or labels, pretrained nets,
distillation, opening books, any inherited chess heuristic. Rules are allowed,
opinions about chess are not. Allowed: self-generated diversity (temperature,
Dirichlet, random openings). Allowed in evaluation only: standard books (UHO/TCEC),
since evaluation is measurement.
⚠️ Evaluation output may never flow backwards. The prohibition that will actually get
violated is checkpoint selection: "keep the checkpoint with the best puzzle score" is
distillation through a one-bit channel and looks like good practice.

## Architecture and sizing (decided)
- **Env**: 12-bit piece-list board (32×`uint16` slots + one `int16` control word),
  fully device-side. Slots are stable for the whole game, which index-aligns the 32
  piece tokens with the 32 legality masks. Normative: `docs/reference/spec.md`.
- **Net**: piece-token transformer, 32 tokens = 32 pieces, non-causal, 32×64 policy
  logits index-aligned with the mask, so masking is a device-side AND — no gather, no
  host round-trip. ⚠️ **The network does not do that masking**: raw logits out, the
  search applies it (spec §7.4). d=256 / L=8 / H=8 / FFN=1024, **6,383,360 params ≈
  400 MFLOPs/eval**, the number for the curve (1M was refuted by scaling laws).
- **Search**: AlphaZero PUCT, **n=800**, `docs/reference/search.md`. `n` is chosen for convergence
  first; the sweep downward is where throughput work starts, Gumbel a named seam.
  ⚠️ **The authority is AlphaGo Zero (Nature 550:354-359)** — not the AZ paper, not
  the released `pseudocode.py`, which publishes no PUCT formula and no `c_puct`, so
  the logarithmic form and 19652/1.25 come from an unrefereed file that contradicts
  AGZ in three silent ways. `docs/reference/search.md` §1.1, §3.1a, §3.6, §3.7, §13.
- ~10M games × 80 plies → **100k evals/s** wanted; ~10¹⁹ FLOPs, ~100-500 € spot.
  **NN-bound, not env-bound** — measured: the environment is 2.2 % of a node.
- **Gate 1 (engineering)**: ≥45-50k evals/s **inside a real MCTS loop** on the 4060.
  **Cleared 2026-07-30: 56 996 useful evals/s at n=800, B=4096** (`bench_search.py`,
  `logs/gate1a.log`, `docs/ledger/perf.md`). **Gate 2 (science)**: beat AlphaGateau
  (~2100 Elo) with an Elo slope matching Jones' law (+500 Elo per 10× compute).

## Layout
```
brokefish/  nn/ (model.py is the network and the oracle; one file per fused impl,
            selected by name through `encoder_impl`), env/, search/, eval/, train/
csrc/       include order is load-bearing: chess.cuh → step.cuh → movegen.cuh →
            zobrist.cuh, terminal.cuh on chess.cuh alone. Device tests in csrc/tests/,
            one nvcc line each, no Python and no torch; build contract in README.md
tests/      torch and python-chess are the oracles; `pytest tests/ --mutation` asks
            whether the tests bite. boards.py makes positions by random legal play
bench/      bench_model (--path full|backbone), bench_search (Gate 1a), bench_loop,
            bench_env, bench_eval. Interleaved A/B protocol, never before/after
docs/       three kinds, and the kind is the directory. reference/ is normative —
            what the code must do: spec, search, training, evaluation, environment,
            debugger. ledger/ is the numbers — state.md per component, perf.md per
            measurement. journal/ is dated and append-only: experiments, audits,
            lessons, never edited after the fact. roadmap.md and index.md sit on top
debugger/   the web viewer, the only part not importable from `brokefish`
data/       cuda_testset (A1's dump), suites.pt (D1), lichess_db_puzzle.csv
logs/       campaign output, tail -f-able while it runs
```
Production repository: no staging areas, no ladders, no snapshots of work happening
elsewhere. Code lands here correct and tested; exploratory kernels belong in scratch.

## State — the summary. **`docs/ledger/state.md` is the real ledger; read it.**
Every row has a bullet there with its measured number and the traps that cost time.
Track E (sample efficiency) is the live track and sits before C4; `docs/roadmap.md`
is the plan and the open decisions, and `docs/journal/` is how each row was reached.

| done | what | the number |
|---|---|---|
| A1, A2 | engine, PyTorch and CUDA, `csrc/*.cuh` + `env/` | perft green, 598M nodes / 4.8 s; 5.01M movegen/s |
| B1, B2 | the encoder, `csrc/encoder.cu` + `nn/` | 62.3k evals/s boards-to-logits, ×3.25 over torch |
| C1 | the MCTS, `csrc/search.cu` + `search/` | **Gate 1a: 56 996 evals/s**, tree is 4.4 % |
| C2 | training, `brokefish/train/` + `docs/reference/training.md` | 31 checks; `lr = 0.002` beats AZ's 0.2, provisional |
| D0, D1 | notation, rule suites, layer 0, puzzles | layer 0 = 31.7 s per checkpoint |
| D2 | the league and the curve, `eval/{match,elo,league,curve}.py` | 56 checks; three curves measured |
| E0 | **sims per target is a first-order lever**, `docs/roadmap.md` Track E | n=128 reaches the landmark on **2.7-3.0× fewer positions**; slope 25 → 332 Elo/decade |
| — | the debugger, the AGZ oracle, `docs/journal/2026-07-30-fidelity.md` | trees agree exactly at n=512 |

Four traps belonging to no single component, each with its story in `docs/ledger/state.md`:

- ⚠️ **`mask.sum(-1)` is a bug** — int64 words carrying uint64 patterns overflow (two
  pieces reaching h8 give 2 × 2⁶³ = 0). Use `(mask == 0).all(-1)`.
- ⚠️ **An all-zero board word decodes as a live white pawn on a1**, not as an error:
  any scan over a pool or a buffer must stop at its own count.
- ⚠️ **A snapshot built in a constructor goes stale silently** — bit C1's tree dict
  and C2's packed weights; a third will look as innocent.
- ⚠️ **Evaluation and self-play run under `no_grad`**, or the fp32 master weights
  build an autograd graph across the run and OOM the card.

⚠️ A second Claude session works this repository in parallel: check `git status` and
mtimes before rewriting a shared file (`CLAUDE.md` was edited mid-turn on 2026-07-30),
and `nvidia-smi` before taking the GPU.

## Environment and hardware
`.venv` is self-contained: torch 2.13.0+cu132, triton 3.7.1, python 3.13 — the exact
stack every number in `docs/ledger/perf.md` was measured on. Run modules from the repository
root through uv: `uv run --no-project --python .venv/bin/python -m tests.test_model`.
On a fresh machine use `requirements.lock`; hand over the install command, never run it.
⚠️ `--no-project` and no `--sync` are **load-bearing**: `.venv` was cloned with
`cp -al` rather than installed, because uv's cache entries for the `nvidia-*` wheels
lost their HTTP revalidation metadata, so anything that re-resolves re-downloads ~3 GB
even with every version pinned. Never let a `uv run` turn into an install.
✅ **No `PATH=` prefix is needed for anything, ever** (fixed 2026-07-31). torch's
`is_ninja_available()` shells out to `ninja --version`, so the console script has to be
on `PATH`; `nn/_build.py` now puts it there itself, from `ninja.BIN_DIR`, immediately
before torch looks — verified by a from-scratch `nvcc` compile under `PATH=/usr/bin:/bin`.
`uv run --python .venv/bin/python` also prepends `.venv/bin` on its own, so the old
advice was only ever about invoking `.venv/bin/python` directly. If it ever regresses
the error now names PATH instead of saying "Ninja is required to load C++ extensions",
which looked like a missing install and was not.

**RTX 4060 Laptop, 8 GB, sm89 (Ada)**: fp16/fp8 tensor cores via `mma.sync`,
`cp.async` — no TMA, no wgmma, no clusters. 99 KB SMEM/block, 64K regs/SM. Heavy
phases: Modal ($30 free/month, H100 ≈ $3.95/h).
⚠️ It is also the display adapter. A multi-minute pinned GPU job can take the desktop
down; check `nvidia-smi` and ask before launching one.
⚠️ Clocks drop to **1.38-1.5 GHz** under sustained load and **fp32 accumulation is
half rate on GeForce**. Measured `mma.sync` issue rates, 2026-07-29: **18.0 TFLOPS
fp16→fp32**, 35.5 fp16→fp16, 41.6 e4m3→fp32. Under a 57-second move it sits at
1230-1290 MHz, so every absolute number in `docs/ledger/perf.md` is an upper bound on what
the kernel does inside a generation, by ~6.7 %. Thermal drift ±3 % → **only
order-balanced interleaved A/B comparisons are valid** (a naive before/after already
produced a fake −4.5 % gain).

## Don't redo these (measured dead ends)
`torch.compile`/CUDA graphs (0 %), fp16 accumulation (0 %), cache/evict hints
(0 to −8 %), fp16 LN affine (noise), smaller GEMM tiles via CUTLASS (2× worse),
reductions via `tl.dot` instead of `tl.sum` (**1.85× worse** — layout conversion
costs more than the reduction), `Var = E[x²]−µ²` (noise), head dimension (noise).
Tooling: Triton has no lists and no dynamic register indexing (hence the hand-unrolled
`x0..x3` tiles, which pin `d_model = 4 × 64`); ncu and torch SDPA crash together
(`Cannot load symbol cudnnGetVersion`), so profile without torch attention in-process;
`nsys stats` needs `--force-export=true`; never truncate an error message.

---
`docs/ledger/state.md` first, for what exists and what will bite; `docs/journal/2026-07-29-encoder-kernel.md`
before touching `csrc/encoder.cu`; `docs/` for the *why*.
