"""Throughput of the fused encoder against the PyTorch baselines.

Gate 1 wants 45-50k sustained evals/s for the d=256/L8 network. This script does
not clear that gate -- a pure forward pass never can, the gate is defined inside
a live MCTS loop -- but it is the fastest signal on whether a kernel change
helped or hurt.

The masked row is the one that counts. Spec 7.3 makes the dead-token mask
mandatory, so the unmasked row is kept only as the historical reference the
37.2k baseline was measured on.

Measurement protocol, and it is not optional on a laptop GPU: clocks fall from
2055 MHz to 1.38-1.5 GHz under sustained load, and the resulting drift reaches
+-3%, which is larger than most of the deltas worth chasing. Candidates are
therefore run interleaved, with the order reversed on every other round, so that
the position penalty cancels instead of being read as a speedup. A naive
before/after once produced a fake -4.5% "regression" in this project.

Absolute evals/s from a cold GPU and from a hot one differ by more than the
deltas being chased, so quote the ratios across a run and the absolutes only
alongside the run that produced them.

Run from the repository root::

    python -m bench.bench_model [--boards 16384] [--rounds 5]
"""

import argparse
import time

import torch

from brokefish.nn import available, encoder_impl, why_unavailable

D, H, T, DFF, N_LAYERS = 256, 8, 32, 1024, 8
KING_SLOTS = (15, 31)


def make_alive(n_boards: int, seed: int = 1, dead_fraction: float = 0.45):
    """A plausible mid-game occupancy. Kings hold slots 15 and 31 and never die.

    The mask is data-independent in cost -- every lane evaluates the same select
    whatever the bits say -- so the exact fraction does not move the timing.
    """
    g = torch.Generator().manual_seed(seed)
    alive = torch.rand(n_boards, T, generator=g) >= dead_fraction
    for slot in KING_SLOTS:
        alive[:, slot] = True
    return alive.cuda()


def time_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1e3


def full_candidates(args):
    """The B2 path: boards in, policy/promo/value out.

    This is the number the project now cares about, and it is not the same
    measurement as the backbone one. The CUDA implementation fuses the gather,
    the final norm and the three heads into its single launch; the Triton one
    runs them as torch ops around its kernel, by choice, because they are 0.3 %
    of the FLOPs and writing a Triton prologue would produce no new information.
    Timing both end to end is what makes that choice auditable rather than a
    hidden asymmetry.
    """
    from brokefish.nn.model import BrokefishNet
    from tests.boards import random_positions

    torch.manual_seed(0)
    net = BrokefishNet().cuda().half().eval()
    boards, control, rep = random_positions(args.boards, plies=12, seed=0)

    impls = available()
    fused = {name: encoder_impl(name)(net) for name in impls}

    with torch.no_grad():
        pol_w, pro_w, val_w = net(boards, control, rep)
    for name, model in fused.items():
        pol, _, val = model.forward_full(boards, control, rep)
        rel = ((pol.float() - pol_w.float()).abs().max() / pol_w.float().abs().max()).item()
        dv = (val - val_w).abs().max().item()
        if rel >= 5e-3 or dv >= 8e-3:
            raise SystemExit(f"{name} full path is wrong (policy rel {rel:.1e}, "
                             f"value {dv:.1e}) -- aborted")
        print(f"[OK] {name:6s} full      policy rel {rel:.1e}, value {dv:.1e} vs torch")
    del pol_w, pro_w, val_w
    torch.cuda.empty_cache()

    cands = [("torch full", lambda: net(boards, control, rep))]
    for name, model in fused.items():
        cands.append((f"{name} full", lambda m=model: m.forward_full(boards, control, rep)))
    return cands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--path", choices=("full", "backbone", "both"), default="full",
                        help="full: boards to logits, the B2 deliverable. backbone: "
                             "activations in and out, which is what every row of "
                             "docs/perf.md was measured on.")
    args = parser.parse_args()

    impls = available()
    if not impls:
        raise SystemExit(f"no encoder implementation available: {why_unavailable()}")
    for name, reason in why_unavailable().items():
        print(f"[--] skipping {name}: {reason}")

    candidates = []

    if args.path in ("backbone", "both"):
        torch.manual_seed(0)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=DFF,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        encoder = torch.nn.TransformerEncoder(layer, num_layers=N_LAYERS).cuda().half().eval()
        x = torch.randn(args.boards, T, D, device="cuda", dtype=torch.half)
        alive = make_alive(args.boards)
        pad = ~alive
        fused = {name: encoder_impl(name)(encoder) for name in impls}

        # Correctness gates the benchmark: a fast wrong kernel is not a result.
        with torch.no_grad():
            plain, masked = encoder(x), encoder(x, src_key_padding_mask=pad)
        # One at a time: forward() hands back its internal buffer, so holding two
        # results at once would compare the second against itself.
        for name, model in fused.items():
            for label, run, want, sel in (
                ("unmasked", lambda m=model: m(x), plain, torch.ones_like(alive)),
                ("masked", lambda m=model: m(x, alive), masked, alive),
            ):
                ref = want[sel].float()
                rel = ((run()[sel].float() - ref).abs().max() / ref.abs().max()).item()
                if rel >= 5e-3:
                    raise SystemExit(f"{name} encoder is wrong, {label} (rel {rel:.1e}) -- aborted")
                print(f"[OK] {name:6s} {label:9s} rel {rel:.1e} vs torch")
        del plain, masked
        torch.cuda.empty_cache()
        print(f"     {args.boards} boards, {N_LAYERS} layers, "
              f"alive fraction {alive.float().mean():.2f}")

        # The masked rows are the ones that count -- spec 7.3 makes the mask
        # mandatory -- and the unmasked ones stay because the 37.2k baseline in
        # docs/perf.md was measured without it.
        candidates += [
            ("torch eager", lambda: encoder(x)),
            ("torch eager + mask", lambda: encoder(x, src_key_padding_mask=pad)),
        ]
        for name, model in fused.items():
            candidates.append((name, lambda m=model: m(x)))
            candidates.append((f"{name} + mask", lambda m=model: m(x, alive)))

    if args.path in ("full", "both"):
        candidates += full_candidates(args)

    totals = {name: 0.0 for name, _ in candidates}
    n_calls = args.rounds * len(candidates)
    done = 0
    t0 = time.perf_counter()
    print(f"\n{n_calls} timed blocks ({args.rounds} rounds x {len(candidates)} candidates).")
    with torch.no_grad():
        for r in range(args.rounds):
            order = candidates if r % 2 == 0 else candidates[::-1]
            for name, fn in order:
                ms = time_ms(fn)
                totals[name] += ms
                done += 1
                # Per-round numbers, printed as they land so a long run is
                # visibly alive. They are NOT results: a single round carries the
                # full +-3% clock drift and the order penalty this protocol
                # exists to cancel. Only the averages at the end are quotable.
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (n_calls - done)
                print(f"  [{done:2d}/{n_calls}] round {r} {name:20s} {ms:8.2f} ms "
                      f"(provisional)   elapsed {elapsed:5.0f}s  eta {eta:5.0f}s",
                      flush=True)

    baseline = totals[candidates[0][0]] / args.rounds
    print(f"\n{args.boards} boards, fp16, {args.rounds} interleaved rounds:")
    for name, _ in candidates:
        ms = totals[name] / args.rounds
        print(f"  {name:20s} {ms:8.2f} ms   {args.boards / ms:6.1f}k evals/s   x{baseline / ms:4.2f}")
    print("\n  B1 target: >45k evals/s masked, fp16, on this benchmark.")
    print("  Gate 1 itself is the same number measured inside a live MCTS loop.")


if __name__ == "__main__":
    main()
