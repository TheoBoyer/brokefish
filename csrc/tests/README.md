# Device-level tests for the CUDA kernels

No Python, no torch, one `nvcc` line each. They exist because a wrong fragment
layout is *silent*: every address stays in range, every mma still issues, and the
error only shows up as plausible-looking wrong numbers eight layers downstream.

    nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tdirect.cu -o tdirect && ./tdirect

⚠️ **Nothing runs these automatically.** `pytest tests/` does not reach them, so a
check here can be red for days and say nothing. It has happened: `tselect.cu`'s host
reference kept AGZ's literal `Q = 0` first-play urgency through the fix of
2026-07-31 and failed **71 of 20 000 cases** until 2026-08-03 — the fourth
independent copy of that expression to carry the `[-1, 1]` constant into the `[0, 1]`
tree, after `tests/oracle.py`, `search/trace.py` and `debugger/static/trace.js`. Build
and run every binary here after touching a kernel, and before believing a green
`pytest`.

`core.cuh` is `encoder.cu` with the torch bindings stripped, so the tests compile
the real code rather than a copy of it. Regenerate it after editing the kernel:

    sed -e '/#include <torch\/extension.h>/d' -e '/#include <c10\/cuda\/CUDAException.h>/d' \
        -e '/^namespace {$/,$d' ../encoder.cu > core.cuh

(The cut used to be at `^void encoder_forward`; B2 put an anonymous namespace of
host helpers in front of it, so the marker moved.)

| file | pins | state |
|---|---|---|
| `tzobrist.cu` | **spec §6**: the device-generated key table against Python's, castling rights, the legal en passant file, the full hash, `step_full`'s incremental hash and `irreversible`, and incremental against from-scratch over every transition | bit-exact over 322 182 transitions |
| `tterminal.cu` | **spec §4.3**: the terminal code and result, plus `repetition_count` on a synthetic ring | bit-exact over 10 000 positions |
| `tperft.cu` | **perft against the published node counts**, all six standard positions, startpos to depth 6. The only oracle here that is not this repository | green, 1.01 s |
| `tstep.cu` | **the board mutation of spec §4.2**: the 32 slot words and the control word after every legal move of every position, promotions expanded to four, one null move each | bit-exact over 322 246 cases |
| `tmovegen.cu` | **the first-order move generator**, stage by stage against the PyTorch engine's own dump: base bitsets, pawns, sliders, castling, the friendly filter, the control-mode attack map and `in_check` | bit-exact over 10 000 positions |
| `tselect.cu` | **docs/mcts.md §6.6, §6.6a and §6.4**: the PUCT scan and its tie-break, §6.6a's terminal collapse in both branches, and the canonical edge enumeration with promotions and truncation, against a host reference in double | 20 000 selections, 20 000 collapse cases, 400 enumerations, no dump needed |
| `tfp8.cu` | **docs/journal/2026-08-04-fp8-encoder.md**: the e4m3 primitives the FFN kernel rests on — `cvt.e4m3x2.f16x2`'s byte order against NVIDIA's host converter, `mma.m16n8k32`'s 8-bit fragment layout against a host product, `quantise_tile`, and `gemm_fp8` both on exactly-representable values and on realistic ones | layout and packing exact; realistic values 1.6e-3 relative, which is the fp16 accumulator |
| `tdirect.cu` | **the packed weight layout**: `gemm_direct` against a host reference that reimplements `pack_b` from the fragment definition | bit-exact |
| `tgemm.cu` | A, B and D fragment layouts for one 32x32 tile | bit-exact |
| `tg2.cu` | the same at A row pitches 40 and 264 | bit-exact |
| `tload.cu` | the global→shared staging path | bit-exact (path no longer used) |
| `tk.cu`, `tk2.cu`, `tfull.cu` | the retired SMEM-staged `gemm_full` | superseded; kept for the history in docs/perf.md |

`tdirect.cu` is the one that matters now. Its host-side `pack_b_host` is written
from the fragment definition rather than copied from `brokefish/nn/cuda_impl.py`,
so a divergence between the kernel and the Python packer fails *here*, loudly,
instead of in the middle of a training run.

`tmovegen.cu` reads the flat binaries `scripts/dump_cuda_testset.py` writes, so it
needs a dump that is current:

    .venv/bin/python scripts/dump_cuda_testset.py
    nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tmovegen.cu -o tmovegen && ./tmovegen

It compares one rule at a time because the final mask cannot tell a wrong slider
from a wrong pawn, and it prints the offending position as a board with the two
bitboards and their xor side by side. `tstep.cu` reads the same dump and reports
its own rule coverage, failing when any rule is uncovered: the positions come from
random playouts, where en passant is 39 cases out of 322 246 and would otherwise
pass vacuously.

`harness.cuh` holds the reading and the printing both engine tests share, so the
board diagram and the slot-by-slot word diff exist once.

`tselect.cu` needs no dump: it generates its own cases from a host splitmix64 and
compares against a reference written from `docs/reference/search.md` rather than copied from
the kernel. The stronger check on those two functions is the tree-for-tree
comparison in `tests/test_search_cuda.py`; this one exists because that comparison
reports "the trees diverged at simulation 43" and cannot say which of the two
reductions was wrong. It also reaches inputs a real position does not: every score
tied, every logit tied, 218 candidates, exactly 64 candidates.

`tperft.cu` matters for a reason the other two cannot cover. They compare the kernel
to the PyTorch engine, so a bug the two engines share passes both. Perft's counts are
published values. Porting the movegen found exactly such a shared bug, an int64
overflow in `terminal` that `docs/reference/environment.md` records.

`tzobrist.cu`'s last check is the one that cannot be faked, and it uses no oracle:
hashing a child position from scratch has to agree with having arrived at it by an
incremental update, over all 322 182 transitions in the dump. An incremental update
that is wrong in a way the reference is also wrong in still fails it.

The coverage checks have now found two gaps rather than one. `tterminal.cu` reported
that the dump held **no fifty-move position**, because the playouts run 100 plies and
the clock keeps resetting, so that branch was untested by construction. Five positions
were added to `SPECIAL_FENS` to close it and to pin the boundary on both sides.

All five engine tests have been mutation-tested. Every single-rule break of `step.cuh`
(the en passant victim, the rook leg, the promotion type, the double-push flag, the
king/rook right, the null-move restore, the clock reset, the per-ply flag clear) is
caught, each failing exactly the number of cases its rule covers. That is the check
that a coverage count of 39 actually means something. All six single-rule breaks of
`movegen.cuh`'s second order (dropping it, castling by destination only, the wrong
three castling squares, the pre-move king square, the attacker's colour, the wrong
king slot) are caught by `tmovegen.cu` and by `tperft.cu` independently, the castling
pair at Kiwipete depth 2. Twelve more across `zobrist.cuh` and `terminal.cuh` are
caught too, among them re-introducing the int64 sum bug, which fails on 2 positions
now that the offending FEN is in the dump.

Two traps this directory has already paid for:

* `ldmatrix.x4` fixes which lane group addresses which of its four 8x8 matrices
  (0-7 → `raw[0]`, 8-15 → `raw[1]`, 16-23 → `raw[2]`, 24-31 → `raw[3]`). Swapping
  the n and k axes in the address computation costs a day.
* Anything over 48 KB of dynamic shared memory needs `cudaFuncSetAttribute`, and
  without it the launch fails while `cudaDeviceSynchronize()` still says "no
  error" — the comparison then reads an untouched buffer and reports a small,
  believable delta. Always check `cudaGetLastError()` too.
