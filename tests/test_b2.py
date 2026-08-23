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

import pytest
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


# --------------------------------------------------------------------------
# The win/draw/loss value head, 2026-08-21. `--value-classes 3` puts three logits
# where one was. It costs nothing: the aux tile of the packed head matrix is padded
# from 5 columns to 32 and warp 2 already computes all of them, so the two extra
# columns come out of 27 that were zeros. What has to be pinned is that the kernel's
# epilogue and `nn/model.py` agree — a permuted class order would still produce a
# plausible value in [-1, 1] and nothing downstream could tell.

_WDL_CACHE: dict = {}


def build_wdl(impl: str, seed: int = 0):
    if (impl, seed) not in _WDL_CACHE:
        torch.manual_seed(seed)
        net = BrokefishNet(n_value=3).cuda().half().eval()
        _WDL_CACHE[(impl, seed)] = (net, encoder_impl(impl)(net))
    return _WDL_CACHE[(impl, seed)]


@for_each_impl
def test_the_wdl_epilogue_agrees_with_torch(impl):
    net, fused = build_wdl(impl)
    boards, control, rep = positions()
    with torch.no_grad():
        want = net(boards, control, rep)[2]
    got = fused.forward_full(boards, control, rep)[2]
    d = (got.float() - want.float()).abs().max().item()
    assert d < VALUE_TOL, f"max|diff| {d:.3e}"
    # And it is a live head, not a constant the tolerance would hide.
    assert got.std().item() > 0.05, f"the value collapsed to {got.mean():.4f}"


@for_each_impl
def test_the_wdl_class_order_is_loss_draw_win(impl):
    """The kernel reads the three columns in the same order the module writes them.

    Swapping the *loss* and *win* rows of `value.weight` negates `p(win) - p(loss)`
    exactly, whatever the position: `p0' = p2` and `p2' = p0`. So the kernel must
    come back with the negated value — and it must do so **without** consulting
    torch, which is what makes this a check of `VALUE_COL`'s layout rather than
    another comparison against the oracle.

    ⚠️ A permutation is the failure this exists for. Three real logits in the wrong
    order still produce a finite value in [-1, 1] and still train; nothing downstream
    of the head can see it, and neither can `test_the_wdl_epilogue_agrees_with_torch`
    if the module and the kernel are permuted the same way.
    """
    net, fused = build_wdl(impl)
    boards, control, rep = positions(n=64)
    base = fused.forward_full(boards, control, rep)[2].clone()
    assert base.abs().max().item() > 1e-2, "an all-draw head would pass anything"

    saved = net.value.weight.detach().clone()
    try:
        net.value.weight.data.copy_(saved[[2, 1, 0]])
        # Repacked from scratch: the head matrix is a constructor snapshot
        # (CLAUDE.md's third trap), so mutating the module is not enough.
        swapped = encoder_impl(impl)(net).forward_full(boards, control, rep)[2].clone()
    finally:
        net.value.weight.data.copy_(saved)
    d = (swapped + base).abs().max().item()
    assert d < VALUE_TOL, f"swapping loss and win did not negate the value: max {d:.3e}"

    # The draw row is the one that does not move the sign, only the magnitude.
    try:
        net.value.weight.data.copy_(saved[[0, 2, 1]])
        moved = encoder_impl(impl)(net).forward_full(boards, control, rep)[2].clone()
    finally:
        net.value.weight.data.copy_(saved)
    assert (moved - base).abs().max().item() > 1e-3, \
        "permuting the draw row changed nothing, so it is not being read"


@for_each_impl
def test_the_wdl_head_still_reads_the_side_to_move_king(impl):
    """Spec 7.4's row select is the same row select. Flipping the side to move has to
    move the value, or the three columns are being read off the wrong token."""
    net, fused = build_wdl(impl)
    boards, control, rep = positions()
    a = fused.forward_full(boards, control, rep)[2].clone()
    b = fused.forward_full(boards, -control, rep)[2].clone()
    assert (a - b).abs().max().item() > 1e-3, "flipping the side to move changed nothing"


def test_a_scalar_head_checkpoint_still_loads_after_the_option_exists():
    """⚠️ Retro-compatibility, and it is not decoration: `checkpoints/anchor.pt` is a
    scalar-head net and **every Elo scale in `docs/ledger/` anchors to it**. A
    checkpoint is a bare `state_dict` with no architecture in it, so the head's width
    is read back off `value.weight` — which is the only thing that lets an August
    checkpoint and a win/draw/loss one meet in one Bradley-Terry fit."""
    from brokefish.nn.model import n_value_of, net_for_state

    torch.manual_seed(4)
    old = BrokefishNet()                     # what every file on disk looks like
    state = {k: v.clone() for k, v in old.state_dict().items()}
    assert tuple(state["value.weight"].shape) == (1, 256)
    assert n_value_of(state) == 1

    got = net_for_state(state)
    got.load_state_dict(state)               # strict: a wrong width raises here
    boards, control, rep = positions(n=32)
    with torch.no_grad():
        a = old.cuda()(boards, control, rep)[2]
        b = got.cuda()(boards, control, rep)[2]
    assert torch.equal(a, b)

    new = BrokefishNet(n_value=3)
    ns = new.state_dict()
    assert n_value_of(ns) == 3
    net_for_state(ns).load_state_dict(ns)
    # And the two are not silently interchangeable.
    with pytest.raises(RuntimeError):
        BrokefishNet().load_state_dict(ns)


# --------------------------------------------------------------------------
# The pooled White/draw/Black value head, 2026-08-22. `--value-head pooled` replaces
# spec §7.4's row select of the side-to-move king with the masked mean of every live
# token, both colours, and predicts in White's frame instead of the mover's.
#
# It is free for the same reason the WDL head was: `norm_f` is upstream of the pool
# and the head is biasless, so `W @ mean(hn) == mean(W @ hn)`, and the aux columns for
# all 32 tokens are already in `scratch`. The epilogue averages instead of selecting.

_POOL_CACHE: dict = {}


def build_pooled(impl: str, n_value: int = 3, seed: int = 0):
    if (impl, n_value, seed) not in _POOL_CACHE:
        torch.manual_seed(seed)
        net = BrokefishNet(n_value=n_value, value_head="pooled").cuda().half().eval()
        _POOL_CACHE[(impl, n_value, seed)] = (net, encoder_impl(impl)(net))
    return _POOL_CACHE[(impl, n_value, seed)]


@for_each_impl
def test_the_pooled_epilogue_agrees_with_torch(impl):
    net, fused = build_pooled(impl)
    boards, control, rep = positions()
    with torch.no_grad():
        want = net(boards, control, rep)[2]
    got = fused.forward_full(boards, control, rep)[2]
    d = (got.float() - want.float()).abs().max().item()
    assert d < VALUE_TOL, f"max|diff| {d:.3e}"
    assert got.std().item() > 0.05, f"the value collapsed to {got.mean():.4f}"


@for_each_impl
def test_the_pool_skips_captured_slots(impl):
    """⚠️ A captured slot is `1 << 11` with colour, type and square wiped, so it
    decodes as **a live white pawn on a1** and its head output is a real vector that
    means nothing. Averaging it in would make the value track how many pieces have been
    taken, by an accident of the encoding.

    Driven by capturing a slot and demanding the kernel's value follow the mean over
    the *remaining* tokens, computed by hand from the normed stream.
    """
    net, fused = build_pooled(impl)
    if not hasattr(fused, "forward_stage"):
        return
    boards, control, rep = positions(n=64)
    boards = boards.clone()
    live0 = ((boards >> 11) & 1 == 0).sum(-1)
    # Capture a slot that is currently alive and is not a king (slots 15 and 31 are
    # guaranteed never captured by spec §2.5, and the pool would still be well defined,
    # but killing a king is not a position the engine can produce).
    for row in range(boards.shape[0]):
        for slot in range(32):
            if slot in (15, 31):
                continue
            if not ((int(boards[row, slot]) >> 11) & 1):
                boards[row, slot] = 1 << 11
                break
    live1 = ((boards >> 11) & 1 == 0).sum(-1)
    assert bool((live1 < live0).all()), "the fixture failed to capture anything"

    h = fused.forward_stage(boards, control, rep, 8)
    alive = ((boards >> 11) & 1) == 0
    with torch.no_grad():
        m = alive.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        raw = net.value(pooled).float()
        p = torch.softmax(raw, dim=-1)
        want = (p[:, 2] - p[:, 0]) * torch.where(control > 0, 1.0, -1.0)
    got = fused.forward_full(boards, control, rep)[2]
    d = (got.float() - want).abs().max().item()
    assert d < VALUE_TOL, f"max|diff| {d:.3e} — the pool is not masking dead slots"


@for_each_impl
def test_the_pooled_head_predicts_whites_frame(impl):
    """The head is absolute and the flip is deterministic: what leaves it must be
    `sign(control) * (p(White) - p(Black))`. Checked on the *torch* side, because it is
    the definition the kernel is then held to by the test above."""
    net, _ = build_pooled(impl)
    boards, control, rep = positions(n=128)
    # ⚠️ `random_openings` walks an **even** number of plies, so every position it
    # produces has White to move — the same fact `eval/match.py`'s lockstep rests on.
    # Half the batch is flipped to Black by negating the control word, which is the
    # null-move position and is all this invariant needs.
    control = control.clone()
    control[::2] = -control[::2]
    with torch.no_grad():
        _, _, v, raw = net(boards, control, rep, with_logits=True)
    p = torch.softmax(raw, dim=-1)
    want = (p[:, 2] - p[:, 0]) * torch.where(control > 0, 1.0, -1.0)
    assert torch.allclose(v, want, atol=1e-6)
    assert bool((control > 0).any()) and bool((control < 0).any())


def test_a_king_head_checkpoint_still_loads_after_the_pool_exists():
    """⚠️ `n_value` alone no longer identifies the head — `[3, 256]` is win/draw/loss
    from the king *or* White/draw/Black from the pool. The marker is a buffer that
    exists only in the pooled case, so every legacy `state_dict` still loads strict."""
    from brokefish.nn.model import net_for_state, value_head_of

    torch.manual_seed(5)
    for kw, want in ((dict(), "king"), (dict(n_value=3), "king"),
                     (dict(n_value=3, value_head="pooled"), "pooled")):
        net = BrokefishNet(**kw)
        state = {k: v.clone() for k, v in net.state_dict().items()}
        assert value_head_of(state) == want
        assert ("value_mode" in state) == (want == "pooled")
        got = net_for_state(state)
        got.load_state_dict(state)          # strict: a wrong shape raises here
        assert got.value_head == want and got.n_value == net.n_value

    # And the two [3, 256] heads are not silently interchangeable.
    pooled = BrokefishNet(n_value=3, value_head="pooled").state_dict()
    with pytest.raises(RuntimeError):
        BrokefishNet(n_value=3).load_state_dict(pooled)


# --------------------------------------------------------------------------
# `--value-head prenorm`, 2026-08-23: pool the RAW residual and apply `norm_f` to the
# pooled vector, instead of pooling vectors `norm_f` has already normalised.
#
# The defect it removes was measured, not assumed: on `t12h-wdb`, `|mean(LN(h))|`
# drifts 15.47 -> 10.86 across a 12 h run as the token cloud spreads, and
# `|value.weight|` grows 42 % chasing it. `LN(mean(h))` cannot drift.


def build_prenorm(impl: str, n_value: int = 3, seed: int = 0):
    key = ("pre", impl, n_value, seed)
    if key not in _POOL_CACHE:
        torch.manual_seed(seed)
        net = BrokefishNet(n_value=n_value, value_head="prenorm").cuda().half().eval()
        _POOL_CACHE[key] = (net, encoder_impl(impl)(net))
    return _POOL_CACHE[key]


@for_each_impl
def test_the_prenorm_epilogue_agrees_with_torch(impl):
    """⚠️ This one is **not** the free column-average the other two heads get.
    `LN(mean(h))` is not linear in the per-token value logits, so the kernel pools
    `bufA`, norms it and takes three dot products against an unpacked copy of the
    value rows. Nothing about that is shared with the torch path."""
    net, fused = build_prenorm(impl)
    boards, control, rep = positions()
    control = control.clone(); control[::2] = -control[::2]
    with torch.no_grad():
        want = net(boards, control, rep)[2]
    got = fused.forward_full(boards, control, rep)[2]
    d = (got.float() - want.float()).abs().max().item()
    assert d < VALUE_TOL, f"max|diff| {d:.3e}"
    assert got.std().item() > 0.05, f"the value collapsed to {got.mean():.4f}"


@for_each_impl
def test_the_prenorm_input_scale_does_not_move_with_the_live_count(impl):
    """The whole point. `mean(LN(h))` shrinks as the token cloud spreads and as pieces
    come off; `LN(mean(h))` cannot, because the norm sets the scale. Driven by
    capturing pieces and watching the two input norms."""
    net, _ = build_prenorm(impl)
    boards, control, rep = positions(n=64)
    norms = {"pre": [], "post": []}
    counts = []
    for kill in (0, 6, 12):
        b = boards.clone()
        for row in range(b.shape[0]):
            done = 0
            for slot in range(32):
                if done >= kill:
                    break
                if slot in (15, 31):
                    continue
                if not ((int(b[row, slot]) >> 11) & 1):
                    b[row, slot] = 1 << 11
                    done += 1
        alive = ((b >> 11) & 1) == 0
        with torch.no_grad():
            x, al = net.embed(b, control, rep)
            h = net.encoder(x, src_key_padding_mask=~al)
            m = alive.unsqueeze(-1).to(h.dtype)
            pre = net.norm_f((h * m).sum(1) / m.sum(1))
            post = (net.norm_f(h) * m).sum(1) / m.sum(1)
        counts.append(float(alive.float().sum(-1).mean()))
        norms["pre"].append(float(pre.float().norm(dim=-1).mean()))
        norms["post"].append(float(post.float().norm(dim=-1).mean()))
    assert counts[0] > counts[-1] + 3, f"the fixture did not remove pieces: {counts}"
    spread = lambda v: (max(v) - min(v)) / max(v)
    assert spread(norms["pre"]) < 0.01, (
        f"the prenorm input scale moved with the live count: {norms['pre']}")
    assert spread(norms["pre"]) < spread(norms["post"]), (
        f"prenorm is not more stable than post-norm pooling: "
        f"pre {norms['pre']} post {norms['post']}")


def test_every_value_head_round_trips_through_the_loader():
    """⚠️ Three heads now, and two of them are `[3, 256]`. The marker stores an **id**
    rather than a name so an old *pooled* checkpoint keeps resolving to `pooled` after
    `prenorm` was added."""
    from brokefish.nn.model import net_for_state, value_head_of, VALUE_MODE_ID

    for kw, want in ((dict(), "king"), (dict(n_value=3), "king"),
                     (dict(n_value=3, value_head="pooled"), "pooled"),
                     (dict(n_value=3, value_head="prenorm"), "prenorm"),
                     (dict(n_value=1, value_head="prenorm"), "prenorm")):
        net = BrokefishNet(**kw)
        state = {k: v.clone() for k, v in net.state_dict().items()}
        assert value_head_of(state) == want
        got = net_for_state(state)
        got.load_state_dict(state)
        assert got.value_head == want and got.n_value == net.n_value
    # a checkpoint written by a build that knows a mode this one does not must say so
    with pytest.raises(ValueError, match="unknown value_mode"):
        value_head_of({"value_mode": torch.tensor(99)})
    assert VALUE_MODE_ID["pooled"] == 1, "the pooled id is on disk and cannot move"
