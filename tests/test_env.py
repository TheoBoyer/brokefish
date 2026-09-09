"""Correctness of the environment.

Two independent checks, neither of which depends on the other:

* **perft** against published node counts. Needs nothing but the engine, so it
  runs anywhere and is what catches a botched translation.
* **differential fuzzing** against python-chess, on random positions drawn from
  real games. Needs ``pip install python-chess pyarrow`` and a parquet shard
  fetched by ``scripts/fetch_test_positions.py``; skipped otherwise.

Run from the repository root::

    python -m tests.test_env             # perft only, to depth 4
    pytest tests/test_env.py             # everything available
    pytest tests/test_env.py --slow      # and startpos perft(5), 4.9M nodes

Promotion is live as of A2: the mask carries one bit per (slot, target) and the
interop layer expands a last-rank pawn move into the four ``chess.Move`` objects
python-chess produces, so the comparison is set equality on full moves.
"""

import random
from typing import Optional, Tuple

import pytest
import torch

from brokefish.env import (bitset_to_bool, empty_history, from_fen, initial_boards,
                           insufficient_material, movegen, push_history,
                           repetition_count, step, terminal)
from brokefish.env.torch_impl import (CHECKMATE, FIFTY_MOVE, INSUFFICIENT, NONE,
                                      REPETITION, STALEMATE)
from brokefish.env.torch_impl import PAWN, castling_rights, hash_position, legal_ep_file

# Promotion is live: the mask carries one bit per (slot, target) and interop
# expands a last-rank pawn move into four chess.Move objects (spec §3).

# The standard perft suite, node counts as published and re-derived from
# python-chess. Position 5 has four promotions among its 44 moves at depth 1,
# which is what puts promotion inside a count rather than only inside the
# differential harness.
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4_FEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5_FEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6_FEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

PERFT_IDS = ["startpos", "kiwipete", "position3", "position4", "position5", "position6"]
PERFT_CASES = [
    (START_FEN, [20, 400, 8902, 197281]),
    (KIWIPETE_FEN, [48, 2039, 97862]),
    (POSITION_3_FEN, [14, 191, 2812, 43238]),
    (POSITION_4_FEN, [6, 264, 9467]),
    (POSITION_5_FEN, [44, 1486, 62379]),
    (POSITION_6_FEN, [46, 2079, 89890]),
]

# One ply deeper everywhere. Held out of the default run because the frontier is
# materialised, so startpos(5) alone is 4.9M positions, 30 s and a few hundred
# megabytes resident.
DEEP_PERFT_CASES = [
    (START_FEN, [20, 400, 8902, 197281, 4865609]),
    (KIWIPETE_FEN, [48, 2039, 97862, 4085603]),
    (POSITION_3_FEN, [14, 191, 2812, 43238, 674624]),
    (POSITION_4_FEN, [6, 264, 9467, 422333]),
    (POSITION_5_FEN, [44, 1486, 62379, 2103487]),
    (POSITION_6_FEN, [46, 2079, 89890, 3894594]),
]

CHUNK = 2048  # positions expanded at once; the brute-force second order is ~35x this


def _edges(b: torch.Tensor, c: torch.Tensor):
    """(rows, move, promo) for every legal edge of one chunk.

    Edges, not mask bits. Perft counts legal moves and a promotion is four of
    them, while the action space of [spec §3](../docs/spec.md) holds one bit for
    the pair (slot, target). The expansion rule is the one the search applies:
    the mover is a pawn and the target sits on rank 0 or 7.
    """
    legal = bitset_to_bool(movegen(b, c)[0])
    b_idx, p_idx, s_idx = legal.nonzero(as_tuple=True)

    is_promo = (((b[b_idx, p_idx] >> 6) & 0b111) == PAWN) & torch.isin(
        s_idx >> 3, torch.tensor([0, 7], device=s_idx.device))
    reps = torch.where(is_promo, 4, 1)
    rows = torch.repeat_interleave(b_idx, reps)
    move = torch.repeat_interleave(p_idx * 64 + s_idx, reps)
    # Rank within each group: 0 for a single child, 0..3 for a promotion.
    starts = torch.repeat_interleave(reps.cumsum(0) - reps, reps)
    return rows, move, torch.arange(rows.shape[0], device=rows.device) - starts


def expand(boards: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every position of the batch, played out one ply along every legal edge."""
    out_boards, out_control = [], []
    for i in range(0, boards.shape[0], CHUNK):
        b, c = boards[i:i + CHUNK], control[i:i + CHUNK]
        rows, move, promo = _edges(b, c)
        nb, nc, _, _ = step(b[rows], c[rows], move, promo=promo)
        out_boards.append(nb)
        out_control.append(nc)
    return torch.cat(out_boards), torch.cat(out_control)


def perft(boards: torch.Tensor, control: torch.Tensor, depth: int) -> int:
    """Leaf count at `depth`, without ever holding a whole level in memory.

    Depth-first over chunks, so the resident set is one expanded chunk per level
    rather than the frontier: startpos to depth 6 is 119M nodes, which would be
    7.6 GB of boards materialised and is about 30 MB this way. The last ply is
    counted rather than played, since perft asks how many moves exist and
    stepping them would only throw the results away.
    """
    if depth == 0:
        return boards.shape[0]
    total = 0
    for i in range(0, boards.shape[0], CHUNK):
        b, c = boards[i:i + CHUNK], control[i:i + CHUNK]
        rows, move, promo = _edges(b, c)
        if depth == 1:
            total += rows.shape[0]
            continue
        nb, nc, _, _ = step(b[rows], c[rows], move, promo=promo)
        total += perft(nb, nc, depth - 1)
    return total


def _assert_perft(fen: str, counts: list):
    boards, control = from_fen(fen)
    for depth, expected in enumerate(counts, start=1):
        got = perft(boards, control, depth)
        assert got == expected, f"perft({depth}) = {got}, expected {expected}\n  {fen}"


@pytest.mark.parametrize("fen,counts", PERFT_CASES, ids=PERFT_IDS)
def test_perft(fen: str, counts: list):
    _assert_perft(fen, counts)


@pytest.mark.slow
@pytest.mark.parametrize("fen,counts", DEEP_PERFT_CASES, ids=PERFT_IDS)
def test_perft_deep(fen: str, counts: list):
    _assert_perft(fen, counts)


@pytest.mark.slow
def test_perft_startpos_depth6():
    """The bar spec §10 sets. 119M nodes, so it runs on its own and takes minutes."""
    boards, control = from_fen(START_FEN)
    got = perft(boards, control, 6)
    assert got == 119060324, f"perft(6) = {got}, expected 119060324"


# --------------------------------------------------------------------------- #
# Differential fuzzing against python-chess
# --------------------------------------------------------------------------- #

try:
    import chess

    from brokefish.env import from_board, from_boards, play
    from brokefish.env.interop import list_legal_moves, move_to_args, to_chess_board
    from brokefish.env.positions import sample_board_batch, sample_shards
    HAVE_CHESS = True
except ImportError:
    HAVE_CHESS = False

def random_walk_boards(n: int, seed: int = 0, max_plies: int = 80):
    """Positions from random legal play, when no shard has been fetched.

    Weaker coverage than real games, since random play reaches endgames and
    stalemates far more often than openings, and it needs no download.
    """
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        out.append(chess.Board(board.fen()))
    return out


@pytest.fixture(scope="module")
def boards(n: int = 256):
    if not HAVE_CHESS:
        pytest.skip("python-chess not installed")
    try:
        return [chess.Board(b.fen()) for b in sample_board_batch(sample_shards(n))]
    except (FileNotFoundError, ImportError):
        return random_walk_boards(n)


@pytest.fixture
def plies() -> int:
    return 5


def assert_board_equal(ref: "chess.Board", got: "chess.Board"):
    assert ref.turn == got.turn
    assert ref.castling_rights == got.castling_rights
    assert ref.ep_square == got.ep_square
    assert ref.halfmove_clock == got.halfmove_clock
    assert ref.board_fen() == got.board_fen()


def assert_legal_moves_equal(ref: "chess.Board", words: torch.Tensor, mask: torch.Tensor):
    """Set equality on full `chess.Move` objects, promotion piece included."""
    got, want = set(list_legal_moves(words, mask)), set(ref.legal_moves)
    assert got == want, (
        f"{ref.fen()}\n  missing: {sorted(m.uci() for m in want - got)}"
        f"\n  spurious: {sorted(m.uci() for m in got - want)}")


def test_in_check_and_mate_vs_stalemate(boards, plies: int):
    """spec §4.1: an all-zero mask is checkmate when `in_check` is set and
    stalemate otherwise, and no other output separates the two."""
    rng = random.Random(2)
    seen_check = seen_terminal = 0
    for board in boards:
        board = chess.Board(board.fen())
        b, c = from_fen(board.fen())
        for _ in range(plies + 1):
            mask, in_check = movegen(b, c)
            assert bool(in_check[0]) == board.is_check(), board.fen()
            seen_check += bool(in_check[0])
            if bool((mask[0] == 0).all()):
                seen_terminal += 1
                assert bool(in_check[0]) == board.is_checkmate(), board.fen()
                assert (not bool(in_check[0])) == board.is_stalemate(), board.fen()
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
            b, c, _, _ = play(b, c, **move_to_args(move, b))
    assert seen_check > 0, "no position in the corpus was ever in check"


# Four legal replies, all of them capturing the checking rook on g8. Every mask
# word is 2**62, and 4 * 2**62 is 2**64, which is zero in int64: `terminal` used
# to read that as an empty mask and return checkmate with result -1. Two pieces
# able to reach h8 overflow the same way, which is the commoner shape. Found by
# the CUDA port on 2026-07-29.
OVERFLOW_FEN = "1r1r2Rk/pp5p/4b2n/1P5P/n1Ppp3/R4P1p/1B1KP3/8 b - - 1 1"


def test_terminal_survives_mask_overflow():
    b, c = from_fen(OVERFLOW_FEN)
    mask, in_check = movegen(b, c)
    assert int(mask.sum()) == 0, "this position no longer exercises the overflow"
    assert not bool((mask == 0).all()), "the mask is not actually empty"
    assert bool(in_check[0]), "black is in check here"
    code, result = terminal(mask, in_check, c, b)
    assert int(code[0]) == NONE, f"reported terminal code {int(code[0])}"
    assert int(result[0]) == 0


def test_zobrist_and_irreversible(boards, plies: int):
    """spec §6.1 and §4.2, against python-chess on three separate claims.

    The partition check is the one that matters. Grouping positions by our hash
    and by python-chess's own transposition key has to produce the same
    partition: a hash bucket holding two different keys is a collision, and a key
    split across two buckets means we call two identical positions different,
    which is what silently loses a threefold repetition.
    """
    rng = random.Random(3)
    ours_to_theirs, theirs_to_ours = {}, {}
    saw_ep = saw_irreversible = 0
    for board in boards:
        board = chess.Board(board.fen())
        b, c = from_fen(board.fen())
        h = hash_position(b, c)
        for _ in range(plies):
            assert int(h[0]) == int(hash_position(b, c)[0]), f"incremental drift: {board.fen()}"

            ep = int(legal_ep_file(b, c)[0])
            assert (ep >= 0) == board.has_legal_en_passant(), board.fen()
            if ep >= 0:
                saw_ep += 1
                assert ep == chess.square_file(board.ep_square), board.fen()

            rights = [bool(x) for x in castling_rights(b)[0]]
            assert rights == [bool(board.castling_rights & m) for m in
                              (chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8)], board.fen()

            key = board._transposition_key()
            ours_to_theirs.setdefault(int(h[0]), set()).add(key)
            theirs_to_ours.setdefault(key, set()).add(int(h[0]))

            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            # spec §6.2's definition, which is python-chess's `is_irreversible`
            # minus its `or has_legal_en_passant()` clause. That clause is a
            # tightening we deliberately do not adopt: a position with a legal ep
            # always follows a double push, which is a pawn move and so already
            # irreversible here, so the window already starts at it. Dropping the
            # clause can only leave the window longer, never shorter, and a
            # longer window cannot miss a repetition.
            want_irreversible = board.is_zeroing(move) or board._reduces_castling_rights(move)
            board.push(move)
            b, c, h, irreversible = step(b, c, hash=h, **move_to_args(move, b))
            assert bool(irreversible[0]) == want_irreversible, f"{board.fen()} after {move}"
            saw_irreversible += bool(irreversible[0])

    assert saw_ep > 0 and saw_irreversible > 0, "corpus never exercised ep or irreversibility"
    collisions = {h for h, k in ours_to_theirs.items() if len(k) > 1}
    splits = {k for k, h in theirs_to_ours.items() if len(h) > 1}
    assert not collisions, f"{len(collisions)} hash collisions"
    assert not splits, f"{len(splits)} identical positions given different hashes"


def _walk(board: "chess.Board", moves):
    """Play `moves` through both engines, maintaining the repetition ring.

    Yields (code, result, repetitions, board) before each move and once at the
    end, so a test can assert on every position of the line.
    """
    b, c = from_fen(board.fen())
    h = hash_position(b, c)
    ring, length = empty_history(1)
    for move in moves:
        mask, in_check = movegen(b, c)
        code, result = terminal(mask, in_check, c, b, h, ring, length)
        yield int(code[0]), int(result[0]), int(repetition_count(h, ring, length)[0]), board
        if move is None:
            return
        prev = h
        b, c, h, irreversible = step(b, c, hash=h, **move_to_args(move, b))
        ring, length = push_history(ring, length, prev, irreversible)
        board.push(move)
    mask, in_check = movegen(b, c)
    code, result = terminal(mask, in_check, c, b, h, ring, length)
    yield int(code[0]), int(result[0]), int(repetition_count(h, ring, length)[0]), board


def test_threefold_repetition():
    """Two knight round trips put the start position on the board three times."""
    scratch, shuffle = chess.Board(), []
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"):
        shuffle.append(scratch.parse_san(san))
        scratch.push(shuffle[-1])
    seen = []
    for code, result, reps, b in _walk(chess.Board(), shuffle):
        seen.append((code, reps, b.is_repetition(3)))
    codes = [s[0] for s in seen]
    assert codes[-1] == REPETITION, f"threefold not detected: {codes}"
    assert seen[-1][1] == 3, f"repetition count {seen[-1][1]}, expected 3"
    assert seen[-1][2], "python-chess disagrees that this is a threefold"
    # and it must not fire early: the start position is seen twice at index 4
    assert REPETITION not in codes[:-1], f"fired early: {codes}"
    assert seen[4][1] == 2, f"count at the second occurrence is {seen[4][1]}"


def test_terminal_against_python_chess(boards, plies: int):
    """spec §4.3, every code checked against python-chess's own predicate."""
    rng = random.Random(5)
    hits = {c: 0 for c in range(6)}
    for board in boards:
        board = chess.Board(board.fen())
        b, c = from_fen(board.fen())
        h = hash_position(b, c)
        ring, length = empty_history(1)
        for _ in range(plies + 1):
            mask, in_check = movegen(b, c)
            code, result = terminal(mask, in_check, c, b, h, ring, length)
            code, result = int(code[0]), int(result[0])
            hits[code] += 1

            assert (code == CHECKMATE) == board.is_checkmate(), board.fen()
            assert (code == STALEMATE) == board.is_stalemate(), board.fen()
            assert result == (-1 if code == CHECKMATE else 0), board.fen()
            # A draw code has to be justified, though not uniquely: several
            # conditions can hold at once and the first found wins.
            if code == FIFTY_MOVE:
                assert board.halfmove_clock >= 100, board.fen()
            if code == REPETITION:
                assert board.is_repetition(3), board.fen()
            if code == INSUFFICIENT:
                assert board.is_insufficient_material(), board.fen()
            # The direction that matters: never miss a finished game.
            if board.is_checkmate() or board.is_stalemate() \
                    or board.is_insufficient_material() or board.is_repetition(3):
                assert code != NONE, f"missed a terminal position: {board.fen()}"
            assert bool(insufficient_material(b)[0]) == board.is_insufficient_material(), board.fen()
            for n in (2, 3):
                assert (int(repetition_count(h, ring, length)[0]) >= n) \
                    == board.is_repetition(n), f"{board.fen()} at n={n}"

            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            prev = h
            b, c, h, irreversible = step(b, c, hash=h, **move_to_args(move, b))
            ring, length = push_history(ring, length, prev, irreversible)
            board.push(move)
    assert hits[NONE] > 0
    assert hits[CHECKMATE] + hits[STALEMATE] + hits[INSUFFICIENT] > 0, \
        f"corpus reached no terminal position at all: {hits}"


def test_null_move_is_a_noop():
    """spec §9: a finished game rides along in the batch untouched."""
    b, c = from_fen(chess.Board().fen())
    h = hash_position(b, c)
    b, c, h = torch.cat([b, b]), torch.cat([c, c]), torch.cat([h, h])
    real = torch.tensor([-1, 12 * 64 + 28])
    nb, nc, nh, irreversible = step(b, c, real, hash=h)
    assert torch.equal(nb[0], b[0]) and nc[0] == c[0] and nh[0] == h[0]
    assert not bool(irreversible[0])
    assert not torch.equal(nb[1], b[1]), "the live row of the batch did not move"


def test_from_fen_roundtrip(boards):
    for board in boards:
        b, c = from_fen(board.fen())
        assert_board_equal(board, to_chess_board(b[0], c[0]))


def test_from_board_roundtrip(boards):
    for board in boards:
        b, c = from_board(board)
        assert_board_equal(board, to_chess_board(b[0], c[0]))


def test_legal_moves(boards):
    for board in boards:
        b, c = from_fen(board.fen())
        assert_legal_moves_equal(board, b[0], movegen(b, c)[0][0])


# A third white rook: the surplus one lands in a pawn slot, and its castling right
# is decided by its square and the FEN, not by which slot it fell into. Only the
# queenside right is given, and the b1 rook blocks it.
THREE_ROOKS = "4k3/8/8/8/8/8/8/RR2K2R w Q - 0 1"


@pytest.mark.parametrize("importer", ["from_fen", "from_board"])
def test_a_third_rook_does_not_grant_a_castle(importer):
    """`docs/core-algorithm-review.md` §2: both importers left `special` clear on a
    rook stored in a pawn slot, and `movegen` -- which compares the whole word --
    then read the h1 rook as unmoved and generated e1g1, an illegal castle."""
    board = chess.Board(THREE_ROOKS)
    b, c = from_fen(board.fen()) if importer == "from_fen" else from_board(board)
    assert castling_rights(b)[0].tolist() == [False, True, False, False]
    assert_legal_moves_equal(board, b[0], movegen(b, c)[0][0])
    assert_board_equal(board, to_chess_board(b[0], c[0]))
    # And the two importers agree on every live word, `special` included. Captured
    # slots are left out: `from_fen` zeroes them and `from_board` keeps their pad,
    # which is a pre-existing difference in dead words and not a rights one.
    other, _ = from_board(board) if importer == "from_fen" else from_fen(board.fen())
    live = (b & (1 << 11)) == 0
    assert torch.equal(b[live], other[live]), (b.tolist(), other.tolist())


def test_state_transitions(boards, plies: int):
    rng = random.Random(0)
    for board in boards:
        board = chess.Board(board.fen())
        b, c = from_fen(board.fen())
        mask, _ = movegen(b, c)
        assert_legal_moves_equal(board, b[0], mask[0])
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            board.push(move)
            b, c, mask, _ = play(b, c, **move_to_args(move, b))
            assert_legal_moves_equal(board, b[0], mask[0])
            # The whole position, not only the move set: this is what catches the
            # halfmove clock, the castling rights and the en passant square, none
            # of which the legal-move comparison sees directly.
            assert_board_equal(board, to_chess_board(b[0], c[0]))


def test_batched_transitions(boards, plies: int):
    rng = random.Random(1)
    boards = [chess.Board(b.fen()) for b in boards]
    b, c = from_boards(boards)
    for _ in range(plies):
        # step() has no null move, so games that ended leave the batch.
        alive = [i for i, board in enumerate(boards) if board.legal_moves.count() > 0]
        if not alive:
            break
        boards = [boards[i] for i in alive]
        b, c = b[alive], c[alive]
        moves = []
        for board in boards:
            move = rng.choice(list(board.legal_moves))
            moves.append(move)
            board.push(move)
        b, c, mask, _ = play(b, c, **move_to_args(moves, b))
        for i, board in enumerate(boards):
            assert_legal_moves_equal(chess.Board(board.fen()), b[i], mask[i])


if __name__ == "__main__":
    for fen, counts in PERFT_CASES:
        boards, control = from_fen(fen)
        print(fen)
        for depth, expected in enumerate(counts, start=1):
            boards, control = expand(boards, control)
            got = boards.shape[0]
            print(f"  perft({depth}) = {got:>9}   expected {expected:>9}   "
                  f"{'ok' if got == expected else 'MISMATCH'}")
