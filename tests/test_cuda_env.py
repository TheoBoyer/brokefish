"""The CUDA engine against the PyTorch one, through the Python API.

`csrc/tests/` already validates the kernels bit for bit against flat binaries, so
what this file exists to catch is everything between the kernel and the caller:
dtype and stride assumptions, the optional arguments, the `-1` null move, the
empty batch, and the argument order in `engine.cu`'s six-tensor signatures. A
transposed pair of LUT pointers is invisible in the C++ and fails here.

It also runs perft through the Python API, which is the end-to-end proof that
`cuda_impl.step` and `cuda_impl.movegen` compose: `tperft.cu` proves the kernels
do, not that the binding hands them the right things.

    python -m pytest tests/test_cuda_env.py -q
"""

from __future__ import annotations

import pytest
import torch

from brokefish.env import torch_impl as ref
from brokefish.env.torch_impl import PAWN, bitset_to_bool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

cu = pytest.importorskip("brokefish.env.cuda_impl")

PLIES = [0, 1, 6, 24]
N = 256


@pytest.fixture(scope="module")
def batches():
    """Positions from random legal play at several depths, plus the start position.

    Shallow batches are nearly all opening moves and deep ones carry captures,
    promotions and dead slots, so both ends are needed; `plies=0` is 256 copies of
    the start position, which is the degenerate batch a bug in the index
    arithmetic would survive.
    """
    from tests.boards import random_positions

    out = []
    for plies in PLIES:
        boards, control, _ = random_positions(N, plies=plies, seed=plies, device="cuda")
        out.append((plies, boards, control))
    return out


def test_movegen_matches_reference(batches):
    for plies, boards, control in batches:
        mask, in_check = cu.movegen(boards, control)
        want_mask, want_check = ref.movegen(boards, control)
        assert torch.equal(mask, want_mask), f"mask differs at plies={plies}"
        assert torch.equal(in_check, want_check.bool()), f"in_check differs at plies={plies}"


def test_hash_position_matches_reference(batches):
    for plies, boards, control in batches:
        assert torch.equal(cu.hash_position(boards, control),
                           ref.hash_position(boards, control)), f"hash differs at plies={plies}"


def test_hash_inputs_match_reference(batches):
    """The two inputs the hash is built from, so a wrong hash localises."""
    for plies, boards, control in batches:
        _, rights, ep = cu.hash_inputs(boards, control)
        want_rights = (ref.castling_rights(boards).to(torch.int64)
                       << torch.arange(4, device=boards.device)).sum(-1).to(torch.uint8)
        assert torch.equal(rights, want_rights), f"castling rights differ at plies={plies}"
        assert torch.equal(ep.to(torch.int64), ref.legal_ep_file(boards, control)), \
            f"en passant file differs at plies={plies}"


def _first_legal(mask: torch.Tensor) -> torch.Tensor:
    """One legal move per position, or -1 where there is none."""
    flat = bitset_to_bool(mask).reshape(mask.shape[0], 2048)
    any_legal = flat.any(-1)
    return torch.where(any_legal, flat.to(torch.uint8).argmax(-1), torch.full_like(any_legal, -1,
                                                                                  dtype=torch.long))


@pytest.mark.parametrize("track_hash", [False, True])
def test_step_matches_reference(batches, track_hash):
    for plies, boards, control in batches:
        mask, _ = cu.movegen(boards, control)
        move = _first_legal(mask)
        h = ref.hash_position(boards, control) if track_hash else None

        got = cu.step(boards, control, move, hash=h)
        want = ref.step(boards, control, move, hash=h)
        assert torch.equal(got[0], want[0]), f"boards differ at plies={plies}"
        assert torch.equal(got[1], want[1]), f"control differs at plies={plies}"
        assert torch.equal(got[3], want[3].bool()), f"irreversible differs at plies={plies}"
        if track_hash:
            assert torch.equal(got[2], want[2]), f"hash differs at plies={plies}"
        else:
            assert got[2] is None, "an untracked step must return no hash"


def test_step_irreversible_survives_the_untracked_path(batches):
    """`irreversible` needs only the rights delta, so dropping the hash must not
    drop it. That is the whole reason the untracked path is not just `apply_move`."""
    _, boards, control = batches[-1]
    mask, _ = cu.movegen(boards, control)
    move = _first_legal(mask)
    h = ref.hash_position(boards, control)
    assert torch.equal(cu.step(boards, control, move)[3],
                       cu.step(boards, control, move, hash=h)[3])


def test_all_four_promotions(batches):
    """A promoting move is one mask bit and four edges (spec §3), and the promo
    field is the only thing that separates them."""
    # A white pawn on a7 with a black rook on b8: one promoting push and one
    # promoting capture, and the two kings well apart so the position is legal.
    boards, control = ref.from_fen("1r5k/P7/8/8/8/8/8/K7 w - - 0 1")
    boards, control = boards.cuda(), control.cuda()
    mask, _ = cu.movegen(boards, control)
    moves = bitset_to_bool(mask).reshape(1, 2048)[0].nonzero().flatten()
    slot_type = (boards[0, moves // 64] >> 6) & 0b111
    promo_moves = moves[(slot_type == PAWN) & ((moves % 64) // 8 == 7)]
    assert promo_moves.numel() == 2, \
        f"expected a promoting push and a promoting capture, got {promo_moves.tolist()}"

    seen = set()
    for mv in promo_moves.tolist():
        for promo in range(4):
            args = (boards, control, torch.tensor([mv], device="cuda"))
            kw = {"promo": torch.tensor([promo], dtype=torch.uint8, device="cuda")}
            got, want = cu.step(*args, **kw), ref.step(*args, **kw)
            assert torch.equal(got[0], want[0]), f"move {mv} promo {promo}"
            seen.add(int(((got[0][0, mv // 64]) >> 6) & 0b111))
    assert seen == {1, 2, 3, 4}, f"expected knight..queen, got types {sorted(seen)}"


def test_null_move_is_a_noop(batches):
    """spec §9: -1 leaves the position, the control word and the hash untouched,
    and is never irreversible."""
    _, boards, control = batches[-1]
    h = ref.hash_position(boards, control)
    move = torch.full((boards.shape[0],), -1, dtype=torch.long, device="cuda")
    nb, nc, nh, irr = cu.step(boards, control, move, hash=h)
    assert torch.equal(nb, boards)
    assert torch.equal(nc, control)
    assert torch.equal(nh, h)
    assert not bool(irr.any())


def test_mixed_batch_of_real_and_null_moves(batches):
    """spec §9 exists so a finished game can ride along in a batch, which only
    works if the null lanes do not disturb their neighbours."""
    _, boards, control = batches[-1]
    mask, _ = cu.movegen(boards, control)
    move = _first_legal(mask)
    move[::2] = -1
    h = ref.hash_position(boards, control)
    got, want = cu.step(boards, control, move, hash=h), ref.step(boards, control, move, hash=h)
    for i, name in enumerate(("boards", "control", "hash")):
        assert torch.equal(got[i], want[i]), name
    assert torch.equal(got[3], want[3].bool())


@pytest.mark.parametrize("with_history", [False, True])
def test_terminal_matches_reference(batches, with_history):
    for plies, boards, control in batches:
        mask, in_check = cu.movegen(boards, control)
        h = ring = length = None
        if with_history:
            h = ref.hash_position(boards, control)
            ring, length = ref.empty_history(boards.shape[0], device=boards.device)
            # Plant the current position twice, so the threefold branch fires
            # rather than being skipped: random play reaches one about never.
            ring[:, 0] = h
            ring[:, 1] = h
            length[:] = 2
        got = cu.terminal(mask, in_check, control, boards, hash=h, ring=ring, length=length)
        want = ref.terminal(mask, in_check.to(torch.bool), control, boards, hash=h, ring=ring,
                            length=length)
        assert torch.equal(got[0], want[0]), f"code differs at plies={plies}"
        assert torch.equal(got[1], want[1]), f"result differs at plies={plies}"
        if with_history:
            assert int((got[0] == ref.REPETITION).sum()) > 0, \
                "the planted history should have produced a threefold"


def test_terminal_overflow_position():
    """Four legal replies, every mask word 2**62, so the words sum to 2**64. The
    reference read that as an empty mask and reported checkmate; docs/env.md."""
    boards, control = ref.from_fen("1r1r2Rk/pp5p/4b2n/1P5P/n1Ppp3/R4P1p/1B1KP3/8 b - - 1 1")
    boards, control = boards.cuda(), control.cuda()
    mask, in_check = cu.movegen(boards, control)
    assert int(mask.sum()) == 0, "this position no longer exercises the overflow"
    assert bool(in_check[0])
    code, result = cu.terminal(mask, in_check, control, boards)
    assert int(code[0]) == ref.NONE and int(result[0]) == 0


def test_empty_batch():
    """Zero positions has to be a no-op and not a zero-block launch."""
    boards = torch.zeros((0, 32), dtype=torch.int16, device="cuda")
    control = torch.zeros((0,), dtype=torch.int16, device="cuda")
    mask, in_check = cu.movegen(boards, control)
    assert mask.shape == (0, 32) and in_check.shape == (0,)
    move = torch.zeros((0,), dtype=torch.long, device="cuda")
    nb, nc, nh, irr = cu.step(boards, control, move)
    assert nb.shape == (0, 32) and nc.shape == (0,) and irr.shape == (0,)
    assert cu.hash_position(boards, control).shape == (0,)
    assert cu.terminal(mask, in_check, control, boards)[0].shape == (0,)


def test_illegal_move_assertion(batches):
    """Passing `mask` turns on the check, at the price of a host synchronisation."""
    _, boards, control = batches[0]
    mask, _ = cu.movegen(boards, control)
    move = _first_legal(mask)
    cu.step(boards, control, move, mask=mask)  # legal, must not raise
    bad = move.clone()
    # Slot 20 is a black pawn and it is white to move at ply 0 of every game here.
    bad[0] = 20 * 64 + 0
    with pytest.raises(ValueError, match="illegal moves"):
        cu.step(boards, control, bad, mask=mask)


def test_rejects_cpu_tensors():
    boards, control = ref.initial_boards(2)
    with pytest.raises(ValueError, match="CUDA"):
        cu.movegen(boards, control)


# ---------------------------------------------------------------------------
# perft through the Python API
# ---------------------------------------------------------------------------

def _edges(boards: torch.Tensor, control: torch.Tensor):
    """(rows, move, promo) for every legal edge, promotions expanded to four."""
    mask, _ = cu.movegen(boards, control)
    b_idx, p_idx, s_idx = bitset_to_bool(mask).nonzero(as_tuple=True)
    is_promo = (((boards[b_idx, p_idx] >> 6) & 0b111) == PAWN) & (
        (s_idx >> 3 == 0) | (s_idx >> 3 == 7))
    reps = torch.where(is_promo, 4, 1)
    rows = b_idx.repeat_interleave(reps)
    move = (p_idx * 64 + s_idx).repeat_interleave(reps)
    starts = (reps.cumsum(0) - reps).repeat_interleave(reps)
    promo = torch.arange(rows.shape[0], device=rows.device) - starts
    return rows, move, promo.to(torch.uint8)


def _perft(boards: torch.Tensor, control: torch.Tensor, depth: int, chunk: int = 4096) -> int:
    if depth == 0:
        return boards.shape[0]
    total = 0
    for i in range(0, boards.shape[0], chunk):
        b, c = boards[i:i + chunk], control[i:i + chunk]
        rows, move, promo = _edges(b, c)
        if depth == 1:
            total += rows.shape[0]
            continue
        nb, nc, _, _ = cu.step(b[rows], c[rows], move, promo=promo)
        total += _perft(nb, nc, depth - 1)
    return total


@pytest.mark.parametrize("fen,counts", [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
], ids=["startpos", "kiwipete", "position5"])
def test_perft_through_the_binding(fen, counts):
    boards, control = ref.from_fen(fen)
    boards, control = boards.cuda(), control.cuda()
    for depth, want in enumerate(counts, start=1):
        assert _perft(boards, control, depth) == want, f"perft({depth}) on {fen}"
