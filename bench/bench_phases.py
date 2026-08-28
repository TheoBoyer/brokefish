"""Where the encoder megakernel's time actually goes, phase by phase.

`docs/ledger/perf.md` gets the non-matmul share by **subtraction**: the four real GEMM
shapes run at 32.1 TFLOPS in isolation (`csrc/shapes_ab.cu`) and the whole kernel at
25.5, so 20.6 % is "something else". That residual cannot say *which* something else,
and it cannot tell overhead apart from the GEMMs themselves running slower in situ.
This measures directly, with `clock64` deltas at boundaries the kernel already has.

Read `csrc/encoder.cu`'s `namespace prof` for why per-warp deltas are a sound
decomposition. The short version: each warp's deltas sum to its own lifetime, so the
fractions are exact by construction and nothing is assumed about overlap.

Three things this script is careful about, because each of them can turn the answer
into a number that is confidently wrong:

* **The instrumented kernel is not the shipped kernel.** The clock reads carry a
  memory clobber, which is a scheduling barrier, which changes register pressure --
  measured, the fp8 path's spill *drops* from 40/200 B to 8/48 B under `PROF=true`.
  So the run reports the wall-clock ratio between the two and you should not trust a
  decomposition whose total moved much.
* **Cycles, not seconds.** This card's clock falls from 2055 MHz to 1230-1290 under a
  sustained load, so a wall-clock denominator drifts while the decomposition does not.
* **The warp-0 column exists because warps are not symmetric in the epilogue**, where
  warps 0-2 do the three heads and 3-7 idle. Warp 0 works in every phase, so its
  column is the honest one for elapsed time; the all-warp column shows the idling.

Run from the repository root::

    python -m bench.bench_phases [--boards 512] [--reps 4] [--impl fp8|fp16|both]
"""

from __future__ import annotations

import argparse
import copy
import time

import torch

# Which phases are pure matmul, for the cross-check against the residual method.
# `attention` is deliberately not in here: it is two small mma wrapped in a softmax
# and a V transpose, and it is 2.0 % of the body's FLOPs, so calling it either way
# would be a claim rather than a measurement.
MATMUL = ("qkv_gemm", "proj_gemm", "ff1_gemm", "ff2_gemm")

#: The groups the plan is denominated in. `docs/journal/2026-08-05-fp8-per-row.md`
#: needs the non-matmul 20.6 % cut to 13.0 % for 90k evals/s, and the CODA
#: reparametrisation only touches normalisation and the staging around it -- so what
#: matters is not the size of "other" but how much of it those two groups hold.
GROUPS = {
    "matmul": MATMUL,
    "normalisation": ("ln1", "res1_ln2", "res2"),
    "attention": ("attention",),
    "staging + barriers": ("attn_store", "proj_epi", "ff1_epi", "hid_store", "ff2_epi",
                           "qkv_bias"),
    "fp8 quantisation": ("quant_a", "quant_h"),
    "prologue + epilogue": ("prologue", "epilogue"),
}


def _positions(n: int, seed: int = 0):
    from tests.boards import random_positions

    b, c, _ = random_positions(n, plies=40, seed=seed, device="cuda")
    return b, c, torch.zeros(b.shape[0], dtype=torch.uint8, device="cuda")


def _timed(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def profile(fused, ext, boards, control, rep, reps: int):
    """Cycles per phase, [rows, phases], summed over every CTA of every rep."""
    run = lambda: fused.forward_full(boards, control, rep)
    with torch.no_grad():
        run()                                   # warm up, reach clocks
    torch.cuda.synchronize()

    # ⚠️ Order-balanced, because the two arms differ by a recompiled kernel and this
    # card drifts 3 % under sustained load. The point of the pair is the perturbation
    # the instrumentation causes, and a naive before/after has already manufactured a
    # 4.5 % effect in this project.
    off = on = 0.0
    for i in range(4):
        if i % 2 == 0:
            ext.set_profile(False); off += _timed(run, reps)
            ext.set_profile(True);  on += _timed(run, reps)
        else:
            ext.set_profile(True);  on += _timed(run, reps)
            ext.set_profile(False); off += _timed(run, reps)
    off, on = off / 4, on / 4

    ext.set_profile(True)
    ext.reset_profile()
    with torch.no_grad():
        run()
    torch.cuda.synchronize()
    counts = ext.read_profile().double()
    ext.set_profile(False)
    return counts, off, on


def report(name, counts, names, ctas, t_off, t_on, evals_per_s):
    all_warps, warp0 = counts[0], counts[1]
    tot0 = float(warp0.sum())
    tot_all = float(all_warps.sum())
    if tot0 <= 0:
        raise RuntimeError("the profiler recorded nothing -- is set_profile wired?")

    print(f"\n=== {name} ===")
    print(f"  {ctas} CTAs, warp 0 spent {tot0 / ctas:,.0f} cycles per board")
    print(f"  instrumentation cost {t_on / t_off:.3f}x wall clock "
          f"({t_off * 1e3:.2f} ms -> {t_on * 1e3:.2f} ms)")
    print(f"\n  {'phase':<22}{'cycles/board':>14}{'warp 0':>9}{'all warps':>11}")
    for i, nm in enumerate(names):
        c0, ca = float(warp0[i]), float(all_warps[i])
        if c0 == 0 and ca == 0:
            continue
        print(f"  {nm:<22}{c0 / ctas:>14,.0f}{c0 / tot0:>8.1%}{ca / tot_all:>11.1%}")

    print(f"\n  {'group':<22}{'cycles/board':>14}{'warp 0':>9}")
    idx = {nm: i for i, nm in enumerate(names)}
    seen = set()
    for label, members in GROUPS.items():
        sel = [idx[m] for m in members if m in idx]
        seen.update(members)
        c0 = float(warp0[sel].sum())
        if c0 == 0:
            continue
        print(f"  {label:<22}{c0 / ctas:>14,.0f}{c0 / tot0:>8.1%}")
    rest = [i for nm, i in idx.items() if nm not in seen]
    if rest:
        print(f"  {'unclassified':<22}{float(warp0[rest].sum()) / ctas:>14,.0f}"
              f"{float(warp0[rest].sum()) / tot0:>8.1%}")

    mm = float(warp0[[idx[m] for m in MATMUL]].sum()) / tot0
    print(f"\n  matmul share, measured directly       {mm:>7.1%}")
    print(f"  matmul share, perf.md by subtraction    79.4%   "
          f"(25.5 / 32.1 TFLOPS, shapes_ab.cu)")
    print(f"  -> non-matmul is {1 - mm:.1%} here against 20.6 % there")
    if evals_per_s:
        print(f"  reference throughput this run: {evals_per_s / 1e3:.1f}k evals/s "
              f"(uninstrumented)")


def main() -> int:
    from brokefish.nn._build import load_extension
    from brokefish.nn.cuda_impl import FusedEncoder
    from brokefish.nn.model import BrokefishNet

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boards", type=int, default=512,
                    help="CTAs per launch. 512 is ~10 waves on 24 SMs at 2 CTAs each, "
                         "which fills the machine without making the counters a "
                         "contention benchmark")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--impl", default="both",
                    choices=("fp8", "fp16", "int8", "both"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--reinject", default="none", choices=("none", "ln1", "both"),
                    help="profile the per-layer input re-injection of spec 7.1a. The "
                         "coefficients stay at zero, so the arithmetic is the shipped "
                         "one and the phases are comparable to `none` row for row")
    a = ap.parse_args()

    ext = load_extension("brokefish_encoder", ["encoder.cu"])
    if not hasattr(ext, "set_profile"):
        print("this extension has no profiling hook; rebuild csrc/encoder.cu")
        return 1
    names = list(ext.profile_phases())

    if a.checkpoint:
        from brokefish.nn.validate import load_net
        net, src = load_net(a.checkpoint)
    else:
        torch.manual_seed(0)
        net = BrokefishNet(reinject=a.reinject).cuda().eval().half()
        src = f"random init, reinject={a.reinject}"
    print(f"net: {src}\nboards per launch: {a.boards}")

    boards, control, rep = _positions(a.boards)
    KW = {"fp16": {}, "fp8": {"fp8": True}, "int8": {"int8": True}}
    arms = [("fp16", KW["fp16"]), ("fp8", KW["fp8"])] if a.impl == "both" \
        else [(a.impl, KW[a.impl])]
    for label, kw in arms:
        fused = FusedEncoder(copy.deepcopy(net), **kw)
        counts, t_off, t_on = profile(fused, ext, boards, control, rep, a.reps)
        report(label, counts, names, a.boards, t_off, t_on, a.boards / t_off)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
