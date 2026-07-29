"""The B2 surface: boards in, policy/promo/value out. Torch is the oracle.

`tests/test_model.py` covers the encoder stack, taking activations in and giving
activations out, and it still does -- that path is unchanged and is the A/B
control against Triton. This file covers what B2 added around it: the embedding
gather of spec 7.2, the final LayerNorm, and the three heads of spec 7.4.

Two of these tests exist to pin decisions rather than to catch defects, and they
are the ones to read first if the contract ever looks arbitrary:
``test_policy_logits_are_not_masked`` and ``test_clock_and_rep_are_load_bearing``.

Run from the repository root::

    python -m tests.test_b2          # or: pytest tests/
"""

import functools

import torch

from brokefish.nn import available, encoder_impl, why_unavailable
from brokefish.nn.model import BrokefishNet, N_POLICY, N_PROMO, decode_boards
from tests.boards import random_positions

T = 32
REL_TOL = 5e-3          # same band the encoder stack is held to
VALUE_TOL = 8e-3        # one output through tanh, so an absolute bound

IMPLS = available()
_CACHE: dict = {}
_POS: dict = {}


def for_each_impl(fn):
    """Run a test once per available implementation, naming which one failed.

    A decorator rather than a pytest fixture because this file is also run
    directly, where there is no pytest to parametrize anything.

    ``del wrapper.__wrapped__`` is load-bearing: ``functools.wraps`` copies it,
    pytest follows it to recover the *original* signature, sees a parameter named
    ``impl`` and demands a fixture by that name. Dropping the link leaves a
    zero-argument test, which is what this actually is. Without it every test in
    the file errors at setup under ``pytest`` while passing under
    ``python -m``, and the same thing is true of ``tests/test_model.py`` today.
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


def build(impl: str, seed: int = 0):
    """(fp16 net, fused). Both implementations are built from the same net."""
    if (impl, seed) not in _CACHE:
        torch.manual_seed(seed)
        net = BrokefishNet().cuda().half().eval()
        _CACHE[(impl, seed)] = (net, encoder_impl(impl)(net))
    return _CACHE[(impl, seed)]


def positions(n: int = 256, plies: int = 16, seed: int = 1):
    if (n, plies, seed) not in _POS:
        _POS[(n, plies, seed)] = random_positions(n, plies=plies, seed=seed)
    return _POS[(n, plies, seed)]


def reference(net, boards, control, rep):
    with torch.no_grad():
        return net(boards, control, rep)


def _rel(got, want):
    scale = want.float().abs().max().item()
    return (got.float() - want.float()).abs().max().item() / max(scale, 1e-9)


# -- the gather -----------------------------------------------------------

@for_each_impl
def test_embeddings_are_bit_exact(impl):
    """Not "within tolerance" -- bit exact.

    The gather is five table lookups and four adds, the same arithmetic on both
    sides, so the only thing that can differ is the order of the adds. Spec 7.2's
    order is normative for exactly this reason: an approximate test here would
    pass with the adds regrouped and then the two implementations would drift on
    a quantity that has no rounding of its own to hide behind.
    """
    net, fused = build(impl)
    if not hasattr(fused, "forward_stage"):
        return                      # the Triton path runs the gather in torch itself
    for n, plies in ((256, 16), (64, 4), (512, 24)):
        boards, control, rep = positions(n, plies, seed=n)
        with torch.no_grad():
            want, _ = net.embed(boards, control, rep)
        got = fused.forward_stage(boards, control, rep, 9)
        assert torch.equal(got, want), (
            f"n={n}: max|diff| {(got.float() - want.float()).abs().max().item():.3e}")


@for_each_impl
def test_every_clock_and_rep_row_is_reachable(impl):
    """Self-play at 16 plies only ever sees clocks 1-9, so bit-exactness on it says
    nothing about the other 92 rows of a 101-entry table. This sweeps all of them,
    both signs of the control word, and all three repetition counts, on one fixed
    position -- the gather is a function of the word and the two scalars, so the
    position does not have to be reachable for the index arithmetic to be under test.
    """
    net, fused = build(impl)
    if not hasattr(fused, "forward_stage"):
        return
    boards, control, _ = positions(64, 4, seed=5)
    row = boards[:1]
    clocks = torch.arange(1, 102, device=row.device, dtype=torch.int16)
    for sign in (1, -1):
        for rp in (0, 1, 2):
            n = clocks.numel()
            b = row.expand(n, T).contiguous()
            c = (sign * clocks).contiguous()
            r = torch.full((n,), rp, dtype=torch.uint8, device=row.device)
            with torch.no_grad():
                want, _ = net.embed(b, c, r)
            got = fused.forward_stage(b, c, r, 9)
            assert torch.equal(got, want), f"sign={sign} rep={rp}"


@for_each_impl
def test_gather_covers_every_table_row(impl):
    """Non-vacuity for the test above: if the batch only ever hits a handful of
    rows, bit-exactness on it says very little. This asserts the batch actually
    exercises the index arithmetic that could be wrong."""
    boards, control, rep = positions()
    captured, color, special, ptype, square = decode_boards(boards)
    live = captured == 0
    assert len(set(square[live].tolist())) >= 48, "square coverage too thin"
    assert set(ptype[live].tolist()) == set(range(6)), "not all six piece types present"
    assert bool(special[live].any()), "no special bit set anywhere in the batch"
    # A slot in 0-7 holding a non-pawn: spec 2.1's promoted-piece case, which is
    # where "read the type from the word" earns its warning.
    pawn_slots = torch.cat([ptype[:, :8], ptype[:, 16:24]], dim=1)
    live_pawn_slots = torch.cat([live[:, :8], live[:, 16:24]], dim=1)
    assert bool((pawn_slots[live_pawn_slots] != 0).any()), "no promoted piece in a pawn slot"
    assert len(set(control.abs().tolist())) >= 8, "clock coverage too thin"
    assert set(rep.tolist()) == {0, 1, 2}, "repetition counts not spanned"


# -- the norm and the heads ----------------------------------------------

@for_each_impl
def test_final_norm_matches_torch(impl):
    net, fused = build(impl)
    if not hasattr(fused, "forward_stage"):
        return
    boards, control, rep = positions()
    with torch.no_grad():
        x, alive = net.embed(boards, control, rep)
        want = net.norm_f(net.encoder(x, src_key_padding_mask=~alive))
    got = fused.forward_stage(boards, control, rep, 8)
    assert _rel(got[alive], want[alive]) < REL_TOL, _rel(got[alive], want[alive])


@for_each_impl
def test_heads_match_torch(impl):
    net, fused = build(impl)
    boards, control, rep = positions()
    pol_w, pro_w, val_w = reference(net, boards, control, rep)
    pol, pro, val = fused.forward_full(boards, control, rep)
    assert pol.shape == (boards.shape[0], T, N_POLICY), pol.shape
    assert pro.shape == (boards.shape[0], T, N_PROMO), pro.shape
    assert val.shape == (boards.shape[0],) and val.dtype is torch.float, (val.shape, val.dtype)
    assert _rel(pol, pol_w) < REL_TOL, f"policy rel {_rel(pol, pol_w):.2e}"
    assert _rel(pro, pro_w) < REL_TOL, f"promo rel {_rel(pro, pro_w):.2e}"
    assert (val - val_w).abs().max().item() < VALUE_TOL, (val - val_w).abs().max().item()


@for_each_impl
def test_value_is_the_side_to_move_king(impl):
    """Spec 7.4's row select, checked against the norm output rather than against
    the reference's own select -- otherwise both sides could pick the same wrong
    row and agree. Also checks the tanh is applied and the mover's sign is what
    comes out, by flipping the side to move and watching the value follow the
    other king."""
    net, fused = build(impl)
    if not hasattr(fused, "forward_stage"):
        return
    boards, control, rep = positions()
    n = boards.shape[0]
    for flip in (False, True):
        ctl = -control if flip else control
        h = fused.forward_stage(boards, ctl, rep, 8)
        king = torch.where(ctl < 0, 31, 15).long()
        with torch.no_grad():
            want = torch.tanh(net.value(h[torch.arange(n, device=h.device), king]).float())
        got = fused.forward_full(boards, ctl, rep)[2]
        d = (got - want.squeeze(-1)).abs().max().item()
        assert d < VALUE_TOL, f"flip={flip}: max|diff| {d:.3e}"
    # And the two are genuinely different numbers, or the flip proved nothing.
    a = fused.forward_full(boards, control, rep)[2].clone()
    b = fused.forward_full(boards, -control, rep)[2].clone()
    assert (a - b).abs().max().item() > 1e-3, "flipping the side to move changed nothing"


@for_each_impl
def test_promo_is_per_token(impl):
    """A position can have two pawns on the seventh rank, so the promotion prior
    belongs to the token, not to the board. If the head were collapsed to one row
    per position every token would carry the same four logits."""
    net, fused = build(impl)
    boards, control, rep = positions()
    pro = fused.forward_full(boards, control, rep)[1]
    spread = (pro - pro[:, :1]).abs().max().item()
    assert spread > 1e-2, f"promo logits are constant across tokens (spread {spread:.2e})"


# -- decisions this file exists to pin ------------------------------------

@for_each_impl
def test_policy_logits_are_not_masked(impl):
    """Settled 2026-07-29: the encoder emits raw logits and the legality mask
    belongs to the search.

    The reasons are that the search needs a masked softmax anyway, so masking
    here means doing it twice or constraining what C1 can fuse; that the encoder
    stays a pure function of the position, which is what lets every test above
    run without a movegen fixture; and that it costs nothing either way. If this
    test starts failing because the mask moved back into the kernel, spec 7.4 has
    to move with it.
    """
    net, fused = build(impl)
    boards, control, rep = positions()
    from brokefish.env import torch_impl as env
    mask, _ = env.movegen(boards, control)
    legal = env.bitset_to_bool(mask)
    pol = fused.forward_full(boards, control, rep)[0]
    illegal = pol[~legal]
    assert illegal.isfinite().all(), "illegal logits are not finite -- something masked them"
    assert illegal.abs().max().item() > 1e-2, (
        "illegal logits are all ~0; if the kernel now masks, update spec 7.4")


@for_each_impl
def test_clock_and_rep_are_load_bearing(impl):
    """Spec 7.2 makes both required inputs. The same 32 words at clock 8 and at
    clock 98 have different outcomes, and a position seen twice stands one
    repetition from a draw, so a forward that ignores either is fitting noise.
    This is the test that would catch a table wired to a constant index."""
    net, fused = build(impl)
    boards, control, rep = positions()
    base = fused.forward_full(boards, control, rep)[0].clone()

    far = torch.where(control < 0, torch.full_like(control, -99),
                      torch.full_like(control, 99))
    assert (fused.forward_full(boards, far, rep)[0] - base).abs().max().item() > 1e-2, \
        "the halfmove clock does not reach the output"

    other = (rep.int() + 1).remainder(3).to(torch.uint8)
    assert (fused.forward_full(boards, control, other)[0] - base).abs().max().item() > 1e-2, \
        "the repetition count does not reach the output"


@for_each_impl
def test_dead_slots_are_inert(impl):
    """Spec 7.3, restated for B2: the alive mask now comes from bit 11 of the
    piece word instead of a separate array, so this checks the ballot.

    Garbage is written into the square and colour bits of every dead slot, with
    bit 11 left set. Those bits change the dead token's embedding -- the gather is
    branchless and reads them -- and every live output has to come back
    bit-equal. `type` is left alone because 6 and 7 are outside spec 2.1 and the
    torch reference is entitled to reject them.
    """
    net, fused = build(impl)
    boards, control, rep = positions()
    dead = ((boards >> 11) & 1) == 1
    assert bool(dead.any()), "no dead slots in the batch"
    alive = ~dead

    base = fused.forward_full(boards, control, rep)[0][alive].clone()
    gen = torch.Generator(device="cpu").manual_seed(7)
    noise = torch.randint(0, 64, boards.shape, generator=gen).to(boards.device).to(torch.int16)
    colour = torch.randint(0, 2, boards.shape, generator=gen).to(boards.device).to(torch.int16) << 10
    dirty = torch.where(dead, (boards & (1 << 11)) | noise | colour, boards)
    got = fused.forward_full(dirty, control, rep)[0][alive]
    assert torch.equal(got, base), "a dead slot's contents reached a live token"


# -- the plumbing ---------------------------------------------------------

def test_cuda_and_triton_agree():
    """The two implementations divide the work differently -- CUDA fuses the
    gather, the norm and the heads into the one launch, Triton runs them as torch
    ops around its kernel -- so agreeing here is a real cross-check of both."""
    if not {"cuda", "triton"} <= set(IMPLS):
        print(f"  only {IMPLS} available, skipping the cross-check")
        return
    boards, control, rep = positions()
    outs = []
    for impl in ("cuda", "triton"):
        net, fused = build(impl)
        outs.append([t.clone() for t in fused.forward_full(boards, control, rep)])
    for name, a, b in zip(("policy", "promo", "value"), *outs):
        assert _rel(a, b) < REL_TOL, f"{name}: rel {_rel(a, b):.2e}"


@for_each_impl
def test_board_counts(impl):
    net, fused = build(impl)
    for n in (32, 64, 96, 1024):
        boards, control, rep = random_positions(n, plies=8, seed=n)
        pol_w, _, val_w = reference(net, boards, control, rep)
        pol, _, val = fused.forward_full(boards, control, rep)
        assert _rel(pol, pol_w) < REL_TOL, f"n={n}: policy rel {_rel(pol, pol_w):.2e}"
        assert (val - val_w).abs().max().item() < VALUE_TOL, f"n={n}: value"


@for_each_impl
def test_rejects_bad_inputs(impl):
    net, fused = build(impl)
    if not hasattr(fused, "_check_inputs"):
        return
    boards, control, rep = positions(64, 4, seed=3)
    bad = [
        (boards.int(), control, rep),                 # wrong board dtype
        (boards[:, :31], control, rep),               # wrong slot count
        (boards, control.int(), rep),                 # wrong control dtype
        (boards, control[:32], rep),                  # control length
        (boards, control, rep.int()),                 # wrong rep dtype
    ]
    for args in bad:
        try:
            fused.forward_full(*args)
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"expected a rejection for {[getattr(a, 'shape', a) for a in args]}")


def test_backbone_only_construction_refuses_the_full_path():
    """Built from a bare nn.TransformerEncoder there are no tables and no heads,
    and saying so is the passing behaviour -- silently returning the backbone's
    activations dressed as logits is what must never happen."""
    for impl in IMPLS:
        torch.manual_seed(0)
        net = BrokefishNet().cuda().half().eval()
        backbone_only = encoder_impl(impl)(net.encoder)
        boards, control, rep = positions(64, 4, seed=3)
        try:
            backbone_only.forward_full(boards, control, rep)
        except ValueError:
            continue
        raise AssertionError(f"[{impl}] forward_full worked without the heads")


if __name__ == "__main__":
    print(f"implementations: {', '.join(IMPLS) or 'none'}")
    for name, reason in why_unavailable().items():
        print(f"  skipping {name}: {reason}")

    boards, control, rep = positions()
    print(f"\n{boards.shape[0]} positions, {int((((boards >> 11) & 1) == 0).sum())} live slots, "
          f"clock range {control.abs().min()}-{control.abs().max()}")
    for impl in IMPLS:
        net, fused = build(impl)
        pol_w, pro_w, val_w = reference(net, boards, control, rep)
        pol, pro, val = fused.forward_full(boards, control, rep)
        print(f"[{impl}] policy rel {_rel(pol, pol_w):.2e}   promo rel {_rel(pro, pro_w):.2e}   "
              f"value max|d| {(val - val_w).abs().max().item():.2e}   "
              f"value range [{val.min():.3f}, {val.max():.3f}]")

    print()
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"[OK] {name}")
