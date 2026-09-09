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



# -- rule-derived piece inputs (2026-09-09) --------------------------------
#
# `BrokefishNet(rule_features=True)` adds two zero-initialised tables to the token
# embedding sum: `emb_dest`, summed over the set bits of the slot's own legality word,
# and `emb_attacked`, indexed by whether the live piece stands on a square the other
# colour controls. Torch only. Four properties carry it: off is the old network to
# the bit; a captured slot contributes nothing; slot i reads row i of the mask; and an
# old checkpoint loads into the flagged net with the tables at zero.

def _rule_positions(n: int = 64, plies: int = 40, seed: int = 21):
    from tests.boards import random_positions
    return random_positions(n, plies=plies, seed=seed)


def _rule_pair(seed: int = 5):
    """(net without the flag, net with it and the tables at zero), same weights."""
    from brokefish.nn.model import BrokefishNet
    torch.manual_seed(seed)
    base = BrokefishNet().cuda().eval()
    net = BrokefishNet(rule_features=True).cuda().eval()
    missing = net.load_state_dict(base.state_dict(), strict=False)
    assert set(missing.missing_keys) == {"emb_dest.weight", "emb_attacked.weight",
                                         "rule_features_mode"}, missing
    assert not missing.unexpected_keys
    return base, net


def test_rule_features_off_is_bit_identical_and_never_calls_movegen():
    """Off is the default, adds no key to the state_dict, ignores a passed mask and
    does not touch the environment; on at zero is the same forward to the bit."""
    from brokefish.env import torch_impl as env
    from brokefish.nn.model import BrokefishNet

    boards, control, rep = _rule_positions()
    mask, _ = env.movegen(boards, control)
    base, net = _rule_pair()
    assert not BrokefishNet().rule_features
    assert not any(k.startswith(("emb_dest", "emb_attacked", "rule_features"))
                   for k in base.state_dict())

    orig = env.movegen
    calls = []
    env.movegen = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        with torch.no_grad():
            off = base(boards, control, rep)
            off_with_mask = base(boards, control, rep, mask=mask)
            on = net(boards, control, rep, mask=mask)
            on_own_mask = net(boards, control, rep)
    finally:
        env.movegen = orig
    assert len(calls) == 1, "only the flagged net without a mask may call movegen"
    for name, a, b, c, d in zip(("policy", "promo", "value"), off, off_with_mask, on,
                                on_own_mask):
        assert torch.equal(a, b), f"{name}: the off net read the mask argument"
        assert torch.equal(a, c), f"{name}: zero tables moved the output"
        assert torch.equal(a, d), f"{name}: the net's own movegen differs"


def test_rule_features_captured_slot_contributes_nothing():
    """With random tables, the added vector is exactly zero on every dead slot, and a
    garbage mask row on a dead slot changes no output."""
    from brokefish.env import torch_impl as env
    from brokefish.nn.model import decode_boards

    boards, control, rep = _rule_positions(n=96, plies=60, seed=3)
    mask, _ = env.movegen(boards, control)
    base, net = _rule_pair()
    with torch.no_grad():
        net.emb_dest.weight.normal_(0.0, 0.5)
        net.emb_attacked.weight.normal_(0.0, 0.5)
        x_off, alive = base.embed(boards, control, rep)
        x_on, _ = net.embed(boards, control, rep, mask=mask)
    dead = ~alive
    assert dead.any(), "the batch has no captured slot, so the test is vacuous"
    assert (decode_boards(boards)[0] == 1).equal(dead)
    added = x_on - x_off
    assert torch.equal(added[dead], torch.zeros_like(added[dead]))
    assert (added[alive] != 0).any(), "live slots got nothing; the tables are unread"

    # A dead slot's word is zero in the movegen mask; feed it every bit instead.
    dirty = torch.where(dead, torch.full_like(mask, -1), mask)
    with torch.no_grad():
        clean_out = net(boards, control, rep, mask=mask)
        dirty_out = net(boards, control, rep, mask=dirty)
    for name, a, b in zip(("policy", "promo", "value"), clean_out, dirty_out):
        assert torch.equal(a, b), f"{name}: a dead slot's mask row reached the output"


def test_rule_features_slot_i_reads_mask_row_i():
    """Flipping bits in row i moves token i's embedding by exactly the rows of
    `emb_dest` that were flipped, and no other token."""
    from brokefish.env import torch_impl as env

    boards, control, rep = _rule_positions(n=32, plies=20, seed=8)
    mask, _ = env.movegen(boards, control)
    base, net = _rule_pair(seed=9)
    with torch.no_grad():
        net.emb_dest.weight.normal_(0.0, 0.5)
        net.emb_attacked.weight.normal_(0.0, 0.5)
        _, alive = base.embed(boards, control, rep)
    gen = torch.Generator().manual_seed(0)
    checked = 0
    for n in range(boards.shape[0]):
        live = alive[n].nonzero().flatten().tolist()
        i = live[int(torch.randint(len(live), (1,), generator=gen))]
        # bits 3 and 63: one ordinary square and the sign bit, which is h8
        sq = [3, 63] if (n % 2 == 0) else [int(torch.randint(64, (1,), generator=gen))]
        flip = mask[n].clone()
        for s in sq:
            flip[i] ^= (1 << s) if s < 63 else torch.iinfo(torch.int64).min
        with torch.no_grad():
            x0, _ = net.embed(boards[n:n + 1], control[n:n + 1], rep[n:n + 1],
                              mask=mask[n:n + 1])
            x1, _ = net.embed(boards[n:n + 1], control[n:n + 1], rep[n:n + 1],
                              mask=flip[None])
        delta = (x1 - x0)[0]
        others = torch.ones(T, dtype=torch.bool, device=delta.device)
        others[i] = False
        assert torch.equal(delta[others], torch.zeros_like(delta[others])), \
            f"board {n}: flipping slot {i}'s row moved another token"
        want = torch.zeros_like(delta[i])
        for s in sq:
            was_set = ((mask[n, i] >> s) & 1).item() == 1
            want = want + (-1.0 if was_set else 1.0) * net.emb_dest.weight[s]
        assert torch.allclose(delta[i], want, atol=1e-5, rtol=1e-5), \
            f"board {n}: slot {i}'s change is not the flipped rows of emb_dest"
        checked += 1
    assert checked == boards.shape[0]


def test_rule_features_attacked_bit_matches_python_chess():
    """The second input against the oracle: a live piece is flagged iff python-chess
    says its square is attacked by the other colour. Both colours, whoever moves."""
    import chess
    from brokefish.env import torch_impl as env
    from brokefish.env.interop import to_chess_board
    from brokefish.nn.model import decode_boards

    boards, control, rep = _rule_positions(n=128, plies=50, seed=13)
    mask, _ = env.movegen(boards, control)
    _, net = _rule_pair()
    with torch.no_grad():
        _, alive = net.embed_indices(boards, control, rep)
        _, attacked = net.rule_inputs(boards, control, alive, mask)
    captured, color, _, _, square = decode_boards(boards)
    ones = zeros = 0
    for n in range(boards.shape[0]):
        b = to_chess_board(boards[n].cpu(), control[n].cpu())
        for p in range(T):
            if captured[n, p]:
                assert attacked[n, p].item() == 0, f"board {n}: dead slot {p} flagged"
                continue
            mine_is_black = bool(color[n, p].item())
            want = b.is_attacked_by(chess.WHITE if mine_is_black else chess.BLACK,
                                    int(square[n, p].item()))
            got = bool(attacked[n, p].item())
            assert got == want, (f"board {n} slot {p} on {chess.square_name(int(square[n, p]))}: "
                                 f"got {got}, python-chess says {want}\n{b}")
            ones += want; zeros += not want
    assert ones > 0 and zeros > 0, "the batch does not exercise both rows"


def test_rule_features_checkpoint_round_trip_and_legacy_load():
    """A checkpoint without the tables loads into a flagged net with the tables at
    zero and the same forward; a flagged checkpoint rebuilds through `net_for_state`;
    the marker says so; a strict load into the wrong shape refuses."""
    import pytest
    from brokefish.nn.model import (BrokefishNet, net_for_state, rule_features_of,
                                    RULE_FEATURES_ID)

    boards, control, rep = _rule_positions(n=16, plies=30, seed=2)
    torch.manual_seed(1)
    old = BrokefishNet(n_value=3)
    legacy = {k: v.clone() for k, v in old.state_dict().items()}
    assert rule_features_of(legacy) is False
    assert net_for_state(legacy).rule_features is False

    new = BrokefishNet(n_value=3, rule_features=True)
    new.load_state_dict(legacy, strict=False)
    assert torch.equal(new.emb_dest.weight, torch.zeros_like(new.emb_dest.weight))
    assert torch.equal(new.emb_attacked.weight, torch.zeros_like(new.emb_attacked.weight))
    with torch.no_grad():
        a = old.cuda()(boards, control, rep)
        b = new.cuda()(boards, control, rep)
    for x, y in zip(a, b):
        assert torch.equal(x, y)

    with torch.no_grad():
        new.emb_dest.weight.normal_()
    state = {k: v.clone() for k, v in new.state_dict().items()}
    assert rule_features_of(state) is True
    assert int(state["rule_features_mode"]) == RULE_FEATURES_ID == 1, "on disk, cannot move"
    got = net_for_state(state)
    got.load_state_dict(state)                       # strict
    assert got.rule_features and got.n_value == 3
    with torch.no_grad():
        c = got.cuda()(boards, control, rep)
        d = new(boards, control, rep)
    assert torch.equal(got.emb_dest.weight, new.emb_dest.weight)
    for x, y in zip(c, d):
        assert torch.equal(x, y), "the round-tripped net is not the net that was saved"
    assert not torch.equal(c[0], b[0]), "a random emb_dest left the policy unmoved"
    with pytest.raises(RuntimeError):
        BrokefishNet(n_value=3).load_state_dict(state)          # unexpected keys
    with pytest.raises(ValueError, match="unknown rule_features_mode"):
        rule_features_of({"rule_features_mode": torch.tensor(7)})


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
