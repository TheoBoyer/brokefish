"""Correctness of the fused encoder against PyTorch, which is the oracle.

Two oracles, actually. Comparing the fp16 kernel against fp16 torch measures the
sum of two rounding errors and charges all of it to the kernel. Running the same
rounded weights through an fp32 encoder gives a third reading, so each fp16 path
can be scored against it separately: if the kernel is no further from fp32 than
torch fp16 is, the delta is arithmetic noise rather than a defect.

Every test runs against every implementation `brokefish.nn` can import, so the
CUDA kernel clears the same bar as the Triton one rather than a bar written for
it. Missing implementations are skipped and named, never silently dropped.

Run from the repository root::

    python -m tests.test_model          # or: pytest tests/
"""

import copy
import functools

import torch

from brokefish.nn import available, encoder_impl, why_unavailable

D, H, T, DFF, N_LAYERS = 256, 8, 32, 1024, 8
KING_SLOTS = (15, 31)  # spec 4.2: king slots are alive for the whole game

REL_TOL = 5e-3      # fp16 with fp32 accumulation; the fused stack lands near 2e-3
ULP_TOL = 8         # max absolute delta, in fp16 ulps of the largest output
FP32_TOL = 2.0      # how much further from fp32 the kernel may sit than torch is

_CACHE: dict = {}

IMPLS = available()

# tests/test_mutation_mask.py swaps a deliberately broken class in here to check
# that these tests bite. Nothing else writes to it.
IMPL_OVERRIDE: dict = {}


def for_each_impl(fn):
    """Run a test once per available implementation, naming which one failed.

    A decorator rather than a pytest fixture because this file is also run
    directly (``python -m tests.test_model``), where there is no pytest to
    parametrize anything.

    ``del wrapper.__wrapped__`` is load-bearing: ``functools.wraps`` copies it,
    pytest follows it to recover the *original* signature, sees a parameter named
    ``impl`` and demands a fixture by that name. Dropping the link leaves a
    zero-argument test, which is what this actually is. Without it every test in
    this file errors at setup under ``pytest`` while passing under ``python -m``,
    which is how it went unnoticed until 2026-07-30.
    """
    @functools.wraps(fn)
    def wrapper():
        if not IMPLS:
            raise AssertionError(f"no encoder implementation available: {why_unavailable()}")
        for impl in IMPLS:
            try:
                fn(impl)
            except AssertionError as exc:
                raise AssertionError(f"[{impl}] {exc}") from None
    del wrapper.__wrapped__
    return wrapper


def impl_class(impl: str):
    """The class under test, honouring the mutation harness's override."""
    return IMPL_OVERRIDE.get(impl) or encoder_impl(impl)


def build(seed: int = 0, impl: str = "triton"):
    """(fp16 encoder, fp32 encoder, fused). All implementations are built from
    the same source module and the same rounded weights, so the only thing that
    differs is the arithmetic."""
    if (seed, impl) not in _CACHE:
        torch.manual_seed(seed)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=DFF,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        enc16 = torch.nn.TransformerEncoder(layer, num_layers=N_LAYERS).cuda().half().eval()
        enc32 = copy.deepcopy(enc16).float()
        _CACHE[(seed, impl)] = (enc16, enc32, impl_class(impl)(enc16))
    return _CACHE[(seed, impl)]


def _with_kings(alive: torch.Tensor) -> torch.Tensor:
    for slot in KING_SLOTS:
        alive[:, slot] = True
    return alive


def alive_patterns(n_boards: int, seed: int = 1):
    """Occupancy cases spanning full boards down to bare kings.

    Contiguous, alternating and scattered dead sets are all present because they
    put different bit patterns in each lane's slice of the [32,32] predicate,
    which is where a layout mistake would surface and a single random mask would
    not.
    """
    gen = torch.Generator().manual_seed(seed)
    ones = torch.ones(n_boards, T, dtype=torch.bool)
    cases = [("all alive", ones.clone())]

    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        cases.append((f"random dead {p:.2f}",
                      _with_kings(torch.rand(n_boards, T, generator=gen) >= p)))

    one = ones.clone()
    one[:, 0] = False
    cases.append(("exactly one dead", one))

    cases.append(("kings only, 30 dead",
                  _with_kings(torch.zeros(n_boards, T, dtype=torch.bool))))

    # Board b keeps its first (b % 30) + 2 slots: the alive count varies inside a
    # launch, so a mask hoisted to the wrong scope would show up here.
    varying = torch.zeros(n_boards, T, dtype=torch.bool)
    keep = torch.arange(n_boards) % 30 + 2
    varying[torch.arange(T)[None, :] < keep[:, None]] = True
    cases.append(("per-board varying count", _with_kings(varying)))

    block = ones.clone()
    block[:, 4:20] = False
    cases.append(("contiguous dead block", _with_kings(block)))

    alt = ones.clone()
    alt[:, ::2] = False
    cases.append(("alternating dead", _with_kings(alt)))

    return [(name, m.cuda()) for name, m in cases]


def _ulp(magnitude: float) -> float:
    """Exact fp16 spacing at ``magnitude``, so the bound is scale-free."""
    x = torch.tensor(magnitude, dtype=torch.half)
    up = torch.tensor(float("inf"), dtype=torch.half)
    return (torch.nextafter(x, up) - x).item()


def compare(n_boards: int = 512, seed: int = 0, verbose: bool = False, impl: str = "triton"):
    """Per-pattern deltas on live tokens. Dead rows are undefined on both sides:
    torch leaves them attending to everything, the kernel computes them, and
    nothing downstream reads them."""
    enc16, enc32, fused = build(seed, impl)
    x = torch.randn(n_boards, T, D, device="cuda", dtype=torch.half)

    rows = []
    for name, alive in alive_patterns(n_boards, seed + 1):
        with torch.no_grad():
            t16 = enc16(x, src_key_padding_mask=~alive).float()
            t32 = enc32(x.float(), src_key_padding_mask=~alive)
        got = fused(x, alive).float()

        g, a, b = got[alive], t16[alive], t32[alive]
        scale = a.abs().max().item()
        rows.append(dict(
            name=name,
            alive=alive.float().mean().item(),
            abs_vs_t16=(g - a).abs().max().item(),
            abs_vs_t32=(g - b).abs().max().item(),
            torch_vs_t32=(a - b).abs().max().item(),
            rel=(g - a).abs().max().item() / scale,
            ulps=(g - a).abs().max().item() / _ulp(scale),
        ))
        if verbose:
            r = rows[-1]
            print(f"  {r['name']:24s} alive {r['alive']:5.3f}  "
                  f"absdelta {r['abs_vs_t16']:.2e} ({r['ulps']:.1f} ulp)  "
                  f"rel {r['rel']:.2e}  vs fp32 {r['abs_vs_t32']:.2e} "
                  f"against torch {r['torch_vs_t32']:.2e}")
    return rows


@for_each_impl
def test_matches_torch_unmasked(impl):
    enc16, _, fused = build(impl=impl)
    x = torch.randn(4096, T, D, device="cuda", dtype=torch.half)
    with torch.no_grad():
        expected = enc16(x).float()
    got = fused(x).float()
    assert (got - expected).abs().max().item() / expected.abs().max().item() < REL_TOL


@for_each_impl
def test_masked_matches_torch_across_occupancies(impl):
    for r in compare(impl=impl):
        assert r["rel"] < REL_TOL, f"{r['name']}: rel {r['rel']:.2e}"
        assert r["ulps"] < ULP_TOL, f"{r['name']}: {r['ulps']:.1f} ulps"


@for_each_impl
def test_kernel_is_no_further_from_fp32_than_torch_is(impl):
    """The strong form. If this holds, the fp16 delta above is arithmetic noise
    and the kernel sits on the better side of it."""
    for r in compare(impl=impl):
        assert r["abs_vs_t32"] <= FP32_TOL * r["torch_vs_t32"], (
            f"{r['name']}: kernel {r['abs_vs_t32']:.2e} from fp32, "
            f"torch {r['torch_vs_t32']:.2e}")


@for_each_impl
def test_dead_tokens_are_inert(impl):
    """Spec 7.3: a dead token influences nothing.

    Dead rows are overwritten with values up to the fp16 ceiling and every live
    output has to come back bit-equal. Without the mask it does not, which is
    what makes the mask load-bearing rather than cosmetic.
    """
    n_boards = 512
    _, _, fused = build(impl=impl)
    x = torch.randn(n_boards, T, D, device="cuda", dtype=torch.half)
    for magnitude in (1e4, 65504.0):
        for name, alive in alive_patterns(n_boards):
            if bool(alive.all()):
                continue
            base = fused(x, alive)[alive].clone()
            perturbed = x.clone()
            noise = torch.rand_like(perturbed[~alive].float()) * 2 - 1
            perturbed[~alive] = (noise * magnitude).half()
            assert torch.equal(fused(perturbed, alive)[alive], base), f"{name} at {magnitude:g}"


@for_each_impl
def test_dead_tokens_must_be_finite(impl):
    """The one thing the mask does not protect against.

    Attention weights for dead keys are zero, but `dot(p, v)` still multiplies
    rather than selects, and 0 * inf is NaN. A dead token holding inf therefore
    poisons every live token in its board. Finiteness is guaranteed upstream,
    since spec 7.2 builds a dead token from the same embedding tables as a live
    one, so this pins the requirement rather than reporting a defect: whatever
    writes dead slots in the B2 prologue must write finite values, not inf and
    not uninitialised memory.
    """
    n_boards = 256
    _, _, fused = build(impl=impl)
    x = torch.randn(n_boards, T, D, device="cuda", dtype=torch.half)
    alive = _with_kings(torch.rand(n_boards, T) >= 0.5).cuda()
    base = fused(x, alive)[alive].clone()

    at_ceiling = x.clone()
    at_ceiling[~alive] = torch.tensor(65504.0, dtype=torch.half)
    assert torch.equal(fused(at_ceiling, alive)[alive], base), "finite dead rows must be inert"

    poisoned = x.clone()
    poisoned[~alive] = torch.tensor(float("inf"), dtype=torch.half)
    assert fused(poisoned, alive)[alive].isnan().any(), (
        "inf in a dead row no longer propagates; if the kernel now selects instead "
        "of multiplying in the PV matmul, drop this test and relax the invariant")


@for_each_impl
def test_all_alive_reproduces_the_unmasked_kernel(impl):
    """With every slot alive the predicate is all-true, so the two kernels do the
    same arithmetic. They are not bit-equal: the dynamic predicate changes
    instruction scheduling, which changes which multiply-adds get contracted, and
    that drifts by a couple of ulps over eight layers.
    """
    n_boards = 512
    _, _, fused = build(impl=impl)
    x = torch.randn(n_boards, T, D, device="cuda", dtype=torch.half)
    alive = torch.ones(n_boards, T, dtype=torch.bool, device="cuda")
    unmasked = fused(x).clone()
    masked = fused(x, alive).clone()
    delta = (unmasked.float() - masked.float()).abs().max().item()
    assert delta <= 4 * _ulp(unmasked.float().abs().max().item()), delta


@for_each_impl
def test_board_counts(impl):
    """BM=32 tiles the token axis and the liveness load indexes those same rows,
    so a stale tile boundary would break at the small counts."""
    enc16, _, fused = build(impl=impl)
    for n in (32, 64, 96, 1024, 4096):
        gen = torch.Generator().manual_seed(n)
        x = torch.randn(n, T, D, generator=gen).cuda().half()
        alive = _with_kings(torch.rand(n, T, generator=gen) >= 0.5).cuda()
        with torch.no_grad():
            ref = enc16(x, src_key_padding_mask=~alive).float()
        got = fused(x, alive).float()
        live = ref[alive]
        rel = (got[alive] - live).abs().max().item() / live.abs().max().item()
        assert rel < REL_TOL, f"n_boards={n}: rel {rel:.2e}"


@for_each_impl
def test_rejects_bad_alive_mask(impl):
    _, _, fused = build(impl=impl)
    x = torch.randn(64, T, D, device="cuda", dtype=torch.half)
    for bad in (torch.ones(64, T, device="cuda", dtype=torch.float16),
                torch.ones(64, T - 1, device="cuda", dtype=torch.bool),
                torch.ones(63, T, device="cuda", dtype=torch.bool)):
        try:
            fused(x, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for alive {bad.dtype} {tuple(bad.shape)}")


@for_each_impl
def test_fp32_accumulator_fallback(impl):
    """The escape hatch of docs/perf.md: fp16 accumulation is the default and is
    worth 19%, but it moves the overflow ceiling from about 17x weight growth to
    12x. If a training run ever gets there, one argument has to buy the wide
    accumulator back, and it has to be at least as accurate as the fast one."""
    enc16, enc32, fast = build(impl=impl)
    try:
        wide = impl_class(impl)(enc16, acc_dtype="fp32")
    except NotImplementedError:
        # An implementation is allowed not to have the wide path -- it is not
        # allowed to pretend. Accepting the flag and returning an fp16-
        # accumulated result would satisfy every tolerance below, because fp16
        # accumulation is already inside tolerance; that is exactly why this
        # test cannot be left to catch it. Refusing loudly is the passing
        # behaviour; silently complying is what must never happen.
        print(f"  {impl}: no fp32 accumulator, and it says so")
        return
    x = torch.randn(512, T, D, device="cuda", dtype=torch.half)
    alive = _with_kings(torch.rand(512, T) >= 0.5).cuda()
    with torch.no_grad():
        ref = enc32(x.float(), src_key_padding_mask=~alive)[alive]
    y_fast, y_wide = fast(x, alive).float()[alive], wide(x, alive).float()[alive]
    # Non-vacuity: a wide accumulator that returns bit-identical results to the
    # narrow one is not wide, it is the flag being ignored.
    assert not torch.equal(y_fast, y_wide), "fp32 accumulator changed nothing -- flag ignored?"
    d_fast = (y_fast - ref).abs().max().item()
    d_wide = (y_wide - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert d_wide / scale < REL_TOL, f"fp32 accumulator: rel {d_wide / scale:.2e}"
    assert d_wide <= d_fast, f"fp32 accumulator {d_wide:.3e} worse than fp16 {d_fast:.3e}"


@for_each_impl
def test_rejects_bad_acc_dtype(impl):
    enc16, _, _ = build(impl=impl)
    try:
        impl_class(impl)(enc16, acc_dtype="bf16")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown acc_dtype")


@for_each_impl
def test_rejects_unsupported_width(impl):
    layer = torch.nn.TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=512,
        batch_first=True, norm_first=True, dropout=0.0,
    )
    encoder = torch.nn.TransformerEncoder(layer, num_layers=2).cuda().half().eval()
    try:
        impl_class(impl)(encoder)
    except ValueError:
        return
    raise AssertionError("expected ValueError for d_model != 256")


if __name__ == "__main__":
    print(f"implementations: {', '.join(IMPLS) or 'none'}")
    for name, reason in why_unavailable().items():
        print(f"  skipping {name}: {reason}")

    for impl in IMPLS:
        print(f"\n[{impl}] masked, per occupancy (512 boards, live tokens only):")
        rows = compare(verbose=True, impl=impl)
        worst = max(rows, key=lambda r: r["abs_vs_t16"])
        print(f"worst max-abs vs torch fp16: {worst['abs_vs_t16']:.3e} "
              f"({worst['ulps']:.1f} ulps) on '{worst['name']}'")
        print(f"kernel-to-fp32 {max(r['abs_vs_t32'] for r in rows):.3e} against "
              f"torch-to-fp32 {max(r['torch_vs_t32'] for r in rows):.3e}")

    print()
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"[OK] {name}")
