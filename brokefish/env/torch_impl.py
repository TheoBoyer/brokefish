"""The chess environment in PyTorch: the reference implementation.

Ported from the ancestor engine in ``~/steakfish`` (class ``TCHESS``). It is the
oracle the CUDA movegen is written against, not a dependency of the training
loop, and python-chess is in turn the oracle for it.

Interface, following the engine contract in ``docs/spec.md`` §4, so that this
module and the CUDA one can be swapped without touching the caller::

    movegen(boards, control) -> mask, in_check
    step(boards, control, move, promo, hash) -> boards', control', hash', irreversible

with the state carried as plain tensors::

    boards   [N, 32] int16   one piece word per slot, spec §2.1
    control  [N]     int16   sign = side to move, magnitude = clock + 1, spec §2.2
    mask     [N, 32] int64   bit s of word p set iff the move p -> s is legal

Two conventions differ from the ancestor, both deliberate:

* the mask is returned as the ``int64`` bitset, one word per slot, with **1
  meaning legal**. ``TCHESS`` returned a ``[N, 32, 64]`` bool tensor with
  ``False`` meaning legal; that is 64x the traffic, and the policy head wants a
  device-side AND against the engine's own words.
* ``promo`` is the 2-bit field of spec §3 (``0:N 1:B 2:R 3:Q``), not a piece
  type code.

Spec §§2-6 are implemented in full as of A2: promotion through the `promo` field,
the fifty-move clock, the Zobrist hash and its incremental update, the
``irreversible`` flag, the repetition ring, ``in_check``, the terminal codes and
the null move. What is left is the CUDA implementation of the same contract (A1).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from . import luts

CAPTURED = 1 << 11
COLOR = 1 << 10
SPECIAL = 1 << 9
SQUARE = 0b111111

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
WHITE_KING_SLOT, BLACK_KING_SLOT = 15, 31

_COLS, _ROWS = "abcdefgh", "12345678"
_SQUARE_NAME_TO_ID = {
    f"{c}{r}": ri * 8 + ci
    for r, ri in zip(_ROWS, range(8))
    for c, ci in zip(_COLS, range(8))
}
_SQUARE_ID_TO_NAME = {v: k for k, v in _SQUARE_NAME_TO_ID.items()}
_TYPE_FROM_CHAR = {"p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN, "k": KING}
_SLOTS_FROM_CHAR = {"P": slice(0, 8), "N": slice(8, 10), "B": slice(10, 12),
                    "R": slice(12, 14), "Q": slice(14, 15), "K": slice(15, 16)}
_SCORE_FROM_SAN = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def decode(words: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Split piece words into (captured, color, special, type, square)."""
    words = words.to(torch.int16)
    return (
        ((words >> 11) & 1).bool(),
        ((words >> 10) & 1).bool(),
        ((words >> 9) & 1).bool(),
        (words >> 6) & 0b111,
        words & SQUARE,
    )


# --------------------------------------------------------------------------- #
# First order: what each piece attacks, before king safety
# --------------------------------------------------------------------------- #

def pawn_moves(boards: torch.Tensor, black_to_move: torch.Tensor, types: torch.Tensor,
               mine: torch.Tensor, captured: torch.Tensor, occupancy: torch.Tensor,
               opp_occupancy: torch.Tensor, mask: torch.Tensor,
               control_mode: bool = False) -> None:
    """Add pawn destinations to `mask`, in place.

    `control_mode` keeps the capture diagonals unconditional, which is what the
    second-order pass needs to build an attack map rather than a move list.
    """
    control_mode = int(control_mode)
    is_pawn = (types == PAWN) & mine & ~captured                      # [N,32]
    b_idx, _ = is_pawn.nonzero(as_tuple=True)
    values = boards[is_pawn]
    # Whether the pawn's own side is black, per selected pawn.
    black = black_to_move[..., None].expand(*black_to_move.shape, 32)[is_pawn]
    square = values & SQUARE
    row, col = square // 8, square % 8
    # Mirror black onto white's frame of reference, then shift back.
    base = square + -16 * black.long()

    forward = base + 8
    forward_free = (~(occupancy[b_idx] >> forward)) & 1
    mask[is_pawn] |= forward_free << forward

    double_forward = square + -32 * black.long() + 16
    on_start_row = (black & (row == 6)) | (~black & (row == 1))
    double_free = (~(occupancy[b_idx] >> double_forward)) & forward_free
    mask[is_pawn] |= (on_start_row & forward_free & double_free) << double_forward

    left, right = base + 7, base + 9
    mask[is_pawn] |= ((col > 0).long() << left) & ((control_mode << left) | opp_occupancy[b_idx])
    mask[is_pawn] |= ((col < 7).long() << right) & ((control_mode << right) | opp_occupancy[b_idx])

    # En passant: look for an enemy pawn word carrying the double-push flag on
    # an adjacent file of the fifth rank (from the mover's point of view).
    expected = (black_to_move[b_idx] * 0b001000011000
                + (~black_to_move[b_idx]) * 0b011000100000 + col)[..., None]
    can_catch = (~black_to_move[b_idx] + 3) == row
    mask[is_pawn] |= (can_catch & (col > 0) & (boards[b_idx] == (expected - 1)).any(-1)).long() << (base + 7)
    mask[is_pawn] |= (can_catch & (col < 7) & (boards[b_idx] == (expected + 1)).any(-1)).long() << (base + 9)
    # A push or capture onto the last rank is one bit here, which is the whole
    # design rather than a gap: spec §3 fixes the action space at 32 x 64 and
    # carries the promotion type beside the move. The caller turns that bit into
    # four edges, `step` applies the choice through `promo`, and
    # `interop.list_legal_moves` turns it into the four python-chess moves.


def slider_moves(boards: torch.Tensor, types: torch.Tensor, mine: torch.Tensor,
                 captured: torch.Tensor, occupancy: torch.Tensor, squares: torch.Tensor,
                 mask: torch.Tensor) -> None:
    """Cut the bishop/rook/queen rays at the first blocker, in place."""
    device = boards.device
    r8 = luts.get("range8", device)
    is_slider = ((types == BISHOP) | (types == ROOK) | (types == QUEEN)) & mine & ~captured
    values = boards[is_slider]
    slider_types, slider_squares = types[is_slider], squares[is_slider]
    b_idx, _ = is_slider.nonzero(as_tuple=True)
    slider_occupancy = occupancy[b_idx]

    # Which of the four directions this piece uses: bishops the two diagonals,
    # rooks the rank and file, queens all four.
    parts = torch.zeros(values.shape + (4,), dtype=torch.int64, device=device)
    parts_mask = ((slider_types - 1)[..., None]
                  >> (torch.arange(4, device=device) // 2) & 1).to(torch.bool)
    p_idx, dir_idx = torch.where(parts_mask)

    offsets = luts.get("occl_offsets", device)[dir_idx, slider_squares[p_idx].long()]
    valid = luts.get("occl_masks", device)[dir_idx, slider_squares[p_idx].long()]
    lines = (slider_occupancy[p_idx[:, None]] >> offsets) & valid
    pos_in_line = (slider_squares[p_idx][:, None] == offsets).int().argmax(-1)
    filled = luts.get("filled_lines", device)[pos_in_line, (lines << r8).sum(-1).long()]
    parts[parts_mask] = (((filled[:, None] >> r8) & valid) << offsets).sum(-1)
    mask[is_slider] &= parts.sum(-1)


def castling_moves(boards: torch.Tensor, black_to_move: torch.Tensor, special: torch.Tensor,
                   types: torch.Tensor, mine: torch.Tensor, captured: torch.Tensor,
                   occupancy: torch.Tensor, mask: torch.Tensor) -> None:
    """Add the two castling destinations for a king that still has the right.

    Only the rook's identity and the empty squares are checked here; the squares
    the king crosses are checked by the second-order pass.
    """
    can_castle = (types == KING) & mine & ~captured & ~special
    b_idx, s_idx = can_castle.nonzero(as_tuple=True)
    black = black_to_move[b_idx]
    row_offset = black * 56
    # An unmoved rook of our colour on the corner square, as a whole piece word.
    short_rook = ((~black) * 0b000011000111 + black * 0b010011111111)[..., None]
    long_rook = ((~black) * 0b000011000000 + black * 0b010011111000)[..., None]
    short_ok = (boards[b_idx] == short_rook).any(-1)
    long_ok = (boards[b_idx] == long_rook).any(-1)
    mask[b_idx, s_idx] |= (short_ok & (((occupancy[b_idx] >> (5 + row_offset)) & 0b11) == 0)).long() << (6 + row_offset)
    mask[b_idx, s_idx] |= (long_ok & (((occupancy[b_idx] >> (1 + row_offset)) & 0b111) == 0)).long() << (2 + row_offset)


def first_order_mask(boards: torch.Tensor, black_to_move: torch.Tensor,
                     control_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Pseudo-legal destinations, i.e. everything except king safety.

    Returns the ``[N,32] int64`` bitset (bit s of word p set iff p may go to s)
    and the decoded fields, which the second-order pass reuses.
    """
    boards = boards.to(torch.int16)
    captured = (boards >> 11).bool()
    mine = ((boards >> 10) & 1) == black_to_move[:, None]
    special = ((boards >> 9) & 1).bool()
    types = (boards >> 6) & 0b111
    squares = boards & SQUARE
    occupancy = ((~captured).long() << squares).sum(-1)
    own_occupancy = ((mine & ~captured).long() << squares).sum(-1)
    opp_occupancy = (((~mine) & ~captured).long() << squares).sum(-1)

    mask = luts.get("move_bitsets", boards.device)[(types * 64 + squares).long()] * mine * ~captured
    pawn_moves(boards, black_to_move, types, mine, captured, occupancy, opp_occupancy,
               mask, control_mode=control_mode)
    slider_moves(boards, types, mine, captured, occupancy, squares, mask)
    castling_moves(boards, black_to_move, special, types, mine, captured, occupancy, mask)
    if not control_mode:
        # Cannot land on one's own piece. In control mode those squares stay set:
        # a defended piece is still a square the king may not take.
        mask[mine] &= ~own_occupancy[mine.nonzero(as_tuple=True)[0]]
    return mask, {"captured": captured, "mine": mine, "types": types, "squares": squares}


# --------------------------------------------------------------------------- #
# The engine contract
# --------------------------------------------------------------------------- #

def bitset_to_bool(bitset: torch.Tensor) -> torch.Tensor:
    """[N,32] bitset -> [N,32,64] bool, True = legal. Debug and indexing helper."""
    return ((bitset[..., None] >> luts.get("range64", bitset.device)) & 1).bool()


def bool_to_bitset(legal: torch.Tensor) -> torch.Tensor:
    """[N,32,64] bool, True = legal -> [N,32] bitset."""
    return (legal.to(torch.int64) << luts.get("range64", legal.device)).sum(-1)


def _king_is_attacked(boards: torch.Tensor, defender_is_black: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(attack map of the other side, whether it covers `defender`'s king).

    Passing the *attacker's* colour as `black_to_move` makes `mine` the attacker's
    pieces, so `~mine` is the defender's and the king select needs no second pass.
    """
    bitset, cache = first_order_mask(boards, ~defender_is_black, control_mode=True)
    king = (cache["types"] == KING) & ~cache["mine"] & ~cache["captured"]
    king_square = (cache["squares"] & (king * SQUARE)).sum(-1)
    return bitset, ((bitset >> king_square[..., None]) & 1).any(-1)


def movegen(boards: torch.Tensor, control: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully legal moves and check status, per [spec §4.1](../../docs/spec.md).

    Returns ``mask [N,32] int64``, bit s of word p set iff p -> s is legal, and
    ``in_check [N] bool`` for the side to move. An all-zero mask is checkmate when
    ``in_check`` is set and stalemate otherwise, and nothing else separates them.

    Second order is brute force, as in the ancestor: every pseudo-legal move is
    played on a copy and the resulting position is asked whether our king stands
    on an attacked square. Roughly 35 replays per position; A1 keeps this
    algorithm and only moves it to CUDA. ``in_check`` costs one further pass over
    the *current* board, which is one replay against those 35.
    """
    black_to_move = control < 0
    _, in_check = _king_is_attacked(boards, black_to_move)
    fo_bitset, cache = first_order_mask(boards, black_to_move)
    legal = bitset_to_bool(fo_bitset)                        # [N,32,64], True = legal
    b_idx, p_idx, s_idx = legal.nonzero(as_tuple=True)

    so_boards, so_control, _, _ = step(boards[b_idx], control[b_idx], p_idx * 64 + s_idx)
    so_bitset, so_cache = first_order_mask(so_boards, so_control < 0, control_mode=True)

    # The side that just moved is the one that is not to move now.
    king = (so_cache["types"] == KING) & ~so_cache["mine"] & ~so_cache["captured"]
    king_square = (so_cache["squares"] & (king * SQUARE)).sum(-1)
    exposed = ((so_bitset >> king_square[..., None]) & 1).any(-1)
    legal[b_idx, p_idx, s_idx] = ~exposed

    # Castling additionally requires that the king neither starts, crosses nor
    # lands on an attacked square. The three squares are contiguous, and the
    # destination is included, so this overwrites the test above.
    dx = s_idx - cache["squares"][b_idx, p_idx]
    c_idx, = ((cache["types"][b_idx, p_idx] == KING) & (dx.abs() == 2)).nonzero(as_tuple=True)
    b_idx, p_idx, s_idx = b_idx[c_idx], p_idx[c_idx], s_idx[c_idx]
    crossed = 7 << (s_idx - 2 * (cache["squares"][b_idx, p_idx] < s_idx))
    legal[b_idx, p_idx, s_idx] = ~((so_bitset[c_idx] & crossed[..., None]) > 0).any(-1)
    return bool_to_bitset(legal), in_check


def castling_rights(boards: torch.Tensor) -> torch.Tensor:
    """``[N,4] bool`` in FEN order KQkq. A right holds while both the king and
    that corner's rook are alive and have never moved, which is what `special`
    records by its absence (spec §2.1)."""
    words = boards.to(torch.int32)
    captured = (words >> 11) & 1
    black = ((words >> 10) & 1).bool()
    intact = ((words >> 9) & 1 == 0) & (captured == 0)
    types = (words >> 6) & 0b111
    squares = words & SQUARE

    king_ok = ((types == KING) & intact & ~black).any(-1)
    black_king_ok = ((types == KING) & intact & black).any(-1)
    rook = (types == ROOK) & intact

    def corner(colour_is_black: bool, square: int) -> torch.Tensor:
        side = black if colour_is_black else ~black
        return (rook & side & (squares == square)).any(-1)

    return torch.stack([
        king_ok & corner(False, 7),          # K, h1
        king_ok & corner(False, 0),          # Q, a1
        black_king_ok & corner(True, 63),    # k, h8
        black_king_ok & corner(True, 56),    # q, a8
    ], dim=-1)


def legal_ep_file(boards: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """``[N] int64``, the file of a *legal* en passant capture, or -1.

    spec §6.1 hashes the file only when the capture is actually available,
    because FIDE defines repetition by the same en passant possibility and a
    dangling flag makes real repetitions invisible. "Actually" includes king
    safety: an ep capture that would expose the king does not count, which is a
    position python-chess reaches through `has_legal_en_passant()`.

    At most two pawns can ever capture a given ep target, so this replays at most
    two moves per position rather than the ~35 the full second order costs.
    """
    words = boards.to(torch.int32)
    black_to_move = control < 0
    live_pawn = (((words >> 6) & 0b111) == PAWN) & (((words >> 11) & 1) == 0)
    # The flagged pawn belongs to the side that just moved, so not to the mover.
    flagged = live_pawn & (((words >> 9) & 1) == 1) & (
        (((words >> 10) & 1) == 1) != black_to_move[:, None])

    out = torch.full((boards.shape[0],), -1, dtype=torch.int64, device=boards.device)
    rows = flagged.any(-1).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return out

    victim_sq = (words[rows] & SQUARE).gather(1, flagged[rows].int().argmax(-1, keepdim=True)).squeeze(1)
    victim_file = victim_sq % 8
    black = black_to_move[rows]
    # The capture lands behind the victim, on the square it skipped.
    target = victim_sq + torch.where(black, -8, 8)

    # Either neighbour may be the one that can capture, and both may be able to,
    # so legality is the OR over the two. The file recorded is the ep square's,
    # which is the victim's, never the capturer's: that is the file python-chess
    # puts in its transposition key and the one spec §6.1 means.
    legal = torch.zeros_like(rows, dtype=torch.bool)
    for delta in (-1, 1):
        capturer_file = victim_file + delta
        on_board = (capturer_file >= 0) & (capturer_file < 8)
        capturer_sq = victim_sq + delta
        is_capturer = (live_pawn[rows] & ((words[rows] & SQUARE) == capturer_sq[:, None])
                       & ((((words[rows] >> 10) & 1) == 1) == black[:, None]))
        cand = (on_board & is_capturer.any(-1)).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            continue
        slot = is_capturer[cand].int().argmax(-1)
        r = rows[cand]
        nb, nc, _, _ = step(boards[r], control[r], slot * 64 + target[cand])
        _, exposed = _king_is_attacked(nb, nc >= 0)  # defender is the side that moved
        legal[cand] |= ~exposed
    out[rows] = torch.where(legal, victim_file.long(), out[rows])
    return out


def hash_position(boards: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Full Zobrist hash of each position, ``[N] int64`` (spec §6.1).

    Keys are indexed by (colour, type, square) and never by slot: a promoted
    queen in a pawn slot and an original queen are the same position under FIDE
    and must hash the same. This is where the piece-list representation departs
    from a bitboard engine's Zobrist.
    """
    keys = luts.get("zobrist", boards.device)
    words = boards.to(torch.int32)
    live = ((words >> 11) & 1) == 0
    colour = (words >> 10) & 1
    types = (words >> 6) & 0b111
    squares = words & SQUARE

    idx = ((colour * 6 + types) * 64 + squares).long()
    contrib = torch.where(live, keys[luts.ZOBRIST_PIECE + idx], torch.zeros_like(idx))
    h = contrib[:, 0]
    for slot in range(1, boards.shape[1]):
        h = h ^ contrib[:, slot]

    h = h ^ torch.where(control < 0, keys[luts.ZOBRIST_SIDE], torch.zeros_like(h))
    rights = castling_rights(boards)
    for i in range(4):
        h = h ^ torch.where(rights[:, i], keys[luts.ZOBRIST_CASTLE + i], torch.zeros_like(h))
    ep = legal_ep_file(boards, control)
    h = h ^ torch.where(ep >= 0, keys[luts.ZOBRIST_EP + ep.clamp(min=0)], torch.zeros_like(h))
    return h


def _piece_keys(words: torch.Tensor) -> torch.Tensor:
    """``[N,32] int64``, each slot's Zobrist contribution, 0 when the slot is dead."""
    keys = luts.get("zobrist", words.device)
    idx = ((((words >> 10) & 1) * 6 + ((words >> 6) & 0b111)) * 64 + (words & SQUARE)).long()
    return torch.where(((words >> 11) & 1) == 0,
                       keys[luts.ZOBRIST_PIECE + idx], torch.zeros_like(idx))


def step(boards: torch.Tensor, control: torch.Tensor, move: torch.Tensor,
         promo: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
         in_place: bool = False, hash: Optional[torch.Tensor] = None
         ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Apply one move per position: ``(boards, control, hash, irreversible)``.

    ``move`` is ``slot * 64 + target``. ``promo`` is the spec §3 field
    (``0:N 1:B 2:R 3:Q``) and defaults to queen. Passing ``mask`` turns on the
    legality assertion. ``in_place`` is a hint, not a guarantee: always use the
    returned tensors.

    ``hash`` is optional and the returned hash is ``None`` when it is omitted.
    Tracking it costs a copy of the incoming board and an en passant legality
    test, which perft would pay 4.9M times over for a value it never reads. The
    search always passes one.

    ``irreversible`` is always returned, since it is nearly free and bounds the
    repetition window (spec §6.2). It is set on a capture, a pawn move, or a
    change of castling rights, which is one condition more than the fifty-move
    clock resets on: rights only ever decrease, so a rights change also makes
    every earlier position unreachable.
    """
    dtype = boards.dtype
    track = hash is not None
    orig = boards.to(torch.int32) if track else None
    work = boards.to(torch.int32)
    if work is boards and not in_place:
        work = work.clone()
    move = move.to(torch.int64)
    n = work.shape[0]
    b_idx = torch.arange(n, device=work.device)

    # spec §9: -1 is a no-op, which is how a finished or paused game rides along
    # in a batch. Its row is computed like any other and then restored, so every
    # kernel stays total and no lane takes an early return.
    is_null = move == -1
    move = torch.where(is_null, torch.zeros_like(move), move)
    if mask is not None:
        illegal = ((mask.view(n, 32)[b_idx, move // 64] >> (move % 64)) & 1) == 0
        if illegal.any():
            raise ValueError(f"illegal moves at batch indices {illegal.nonzero(as_tuple=True)[0].tolist()}")

    source, target = move // 64, move % 64
    values = work[b_idx, source]
    types = (values >> 6) & 0b111
    is_pawn = types == PAWN
    white_to_move = control > 0

    # Capture: a live piece standing on the target square, or, for a pawn moving
    # diagonally, the en passant victim, which stands beside it and not on it.
    squares = work & SQUARE
    on_target = (squares == target[..., None]) & ~(work >> 11).bool()
    on_target |= is_pawn[..., None] & (
        work == (((2 * white_to_move + 1) << 9) + target + 8 - white_to_move * 16)[..., None]
    )
    has_capture = on_target.any(-1)
    work[b_idx[has_capture], on_target.int().argmax(-1)[has_capture]] = CAPTURED

    work[b_idx, source] &= ~SQUARE & 0xFFF
    work[b_idx, source] += target

    is_promotion = is_pawn & (target // 8 == white_to_move * 7)
    if is_promotion.any():
        new_type = (QUEEN if promo is None
                    else (promo[is_promotion].to(torch.int32) + KNIGHT))
        work[b_idx[is_promotion], source[is_promotion]] += new_type << 6

    dx = target - (values & SQUARE)
    is_castling = (types == KING) & (dx.abs() == 2)
    if is_castling.any():
        c_idx = b_idx[is_castling]
        short = dx[is_castling] > 0
        rook_row = torch.where(white_to_move[c_idx], 0, 7)
        rook_col = torch.where(short, 7, 0)
        is_rook = (((work[c_idx] >> 6) & 0b111) == ROOK) & (
            squares[c_idx] == (rook_row * 8 + rook_col)[..., None])
        rook_slot = is_rook.int().argmax(-1)
        work[c_idx, rook_slot] &= ~SQUARE & 0xFFF
        work[c_idx, rook_slot] += (rook_row * 8 + torch.where(short, 5, 3)).int()

    # A king or rook that moves loses its castling right for good.
    moved_matters = (types == KING) | (types == ROOK)
    work[b_idx[moved_matters], source[moved_matters]] |= SPECIAL
    # The en passant flag lives for exactly one ply, so clear every pawn's before
    # setting the one that just double-pushed.
    work[((work >> 6) & 0b111) == PAWN] &= ~SPECIAL & 0xFFF
    work[b_idx[is_pawn], source[is_pawn]] |= (dx[is_pawn].abs() == 16).long() << 9

    # spec §2.2: the magnitude is the FIDE fifty-move counter, so it restarts at 1
    # on a capture and on a pawn move and otherwise counts up. It is not clamped
    # at 101: terminal() ends the game there, and clamping would silently disagree
    # with python-chess, which keeps counting.
    control = control.to(torch.int32)
    magnitude = torch.where(has_capture | is_pawn, 1, control.abs() + 1)
    new_control = (-torch.sign(control) * magnitude).to(torch.int16)
    new_boards = work.to(dtype)

    rights_before, rights_after = castling_rights(boards), castling_rights(work)
    rights_delta = rights_before != rights_after
    irreversible = has_capture | is_pawn | rights_delta.any(-1)

    new_hash = None
    if track:
        keys = luts.get("zobrist", work.device)
        zero = torch.zeros_like(hash)
        # Only slots whose (colour, type, square) moved contribute: a slot where
        # just `special` flipped xors its key out and straight back in.
        delta = _piece_keys(orig) ^ _piece_keys(work)
        new_hash = hash.clone()
        for slot in range(work.shape[1]):
            new_hash = new_hash ^ delta[:, slot]
        new_hash = new_hash ^ keys[luts.ZOBRIST_SIDE]  # the side to move always flips
        for i in range(4):
            new_hash = new_hash ^ torch.where(rights_delta[:, i],
                                              keys[luts.ZOBRIST_CASTLE + i], zero)
        ep_before = legal_ep_file(boards, control.to(torch.int16))
        ep_after = legal_ep_file(new_boards, new_control)
        for ep in (ep_before, ep_after):
            new_hash = new_hash ^ torch.where(
                ep >= 0, keys[luts.ZOBRIST_EP + ep.clamp(min=0)], zero)

    if is_null.any():
        new_boards = torch.where(is_null[:, None], boards, new_boards)
        new_control = torch.where(is_null, control.to(torch.int16), new_control)
        irreversible = irreversible & ~is_null
        if track:
            new_hash = torch.where(is_null, hash, new_hash)

    return new_boards, new_control, new_hash, irreversible


# --------------------------------------------------------------------------- #
# Repetition and termination
# --------------------------------------------------------------------------- #

MAX_HISTORY = 100  # spec §6.2: the window never exceeds 100 plies

NONE, CHECKMATE, STALEMATE, FIFTY_MOVE, REPETITION, INSUFFICIENT = range(6)


def empty_history(n: int, device: Optional[torch.device] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """A per-game repetition ring, ``[n, 100] int64`` plus ``[n] int64`` length."""
    return (torch.zeros((n, MAX_HISTORY), dtype=torch.int64, device=device),
            torch.zeros((n,), dtype=torch.int64, device=device))


def push_history(ring: torch.Tensor, length: torch.Tensor, hash: torch.Tensor,
                 irreversible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Record the position being left, then move on (spec §6.2).

    ``hash`` is the position *before* the move and ``irreversible`` describes the
    move itself. An irreversible move empties the ring, because no position at or
    before it can ever recur, which is also what bounds the ring at 100: the
    window is never longer than the fifty-move window, whose reset conditions are
    a subset of these.
    """
    ring, length = ring.clone(), length.clone()
    keep = ~irreversible
    slot = length.clamp(max=MAX_HISTORY - 1)
    rows = keep.nonzero(as_tuple=True)[0]
    ring[rows, slot[rows]] = hash[rows]
    return ring, torch.where(keep, (length + 1).clamp(max=MAX_HISTORY), 0)


def repetition_count(hash: torch.Tensor, ring: torch.Tensor,
                     length: torch.Tensor) -> torch.Tensor:
    """How many times this position has occurred, counting the present one.

    Feeds both the threefold test and ``emb_rep`` in spec §7.2.
    """
    live = torch.arange(ring.shape[1], device=ring.device)[None, :] < length[:, None]
    return 1 + ((ring == hash[:, None]) & live).sum(-1)


def insufficient_material(boards: torch.Tensor) -> torch.Tensor:
    """Neither side can deliver mate, ``[N] bool``.

    Amended on 2026-07-29 to agree with python-chess, which is the oracle: the
    four cases spec §4.3 originally listed miss K+2B against K with both bishops
    on one colour complex, and underpromotion to bishop puts that inside reach of
    self-play. The rule is no pawns, rooks or queens, and then either every
    bishop on a single colour complex with no knights, or exactly one knight and
    no bishops.
    """
    words = boards.to(torch.int32)
    live = ((words >> 11) & 1) == 0
    types = (words >> 6) & 0b111
    squares = words & SQUARE

    def count(t: int) -> torch.Tensor:
        return (live & (types == t)).sum(-1)

    if_mating_material = (count(PAWN) > 0) | (count(ROOK) > 0) | (count(QUEEN) > 0)
    knights, bishops = count(KNIGHT), count(BISHOP)

    is_bishop = live & (types == BISHOP)
    dark = ((squares % 8 + squares // 8) % 2 == 0) & is_bishop
    one_complex = (dark.sum(-1) == 0) | (dark.sum(-1) == bishops)

    return ~if_mating_material & (((knights == 0) & one_complex)
                                  | ((knights == 1) & (bishops == 0)))


def terminal(mask: torch.Tensor, in_check: torch.Tensor, control: torch.Tensor,
             boards: torch.Tensor, hash: Optional[torch.Tensor] = None,
             ring: Optional[torch.Tensor] = None, length: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(code [N] uint8, result [N] int8)`` per spec §4.3.

    Codes are 0 none, 1 checkmate, 2 stalemate, 3 fifty-move, 4 threefold
    repetition, 5 insufficient material, and the first that applies wins. A
    position can satisfy several draw conditions at once; they all give the same
    result, so only the reported code differs.

    ``result`` is from the side to move's point of view, so checkmate is -1 and
    every draw is 0. A value of +1 never occurs, because a position is never
    terminal in favour of the player about to move.

    Repetition is skipped when no history is supplied, which is what the
    fifty-move and material tests do not need.
    """
    # ⚠️ NOT `mask.sum(-1) == 0`. The mask words are int64 carrying uint64 bit
    # patterns, so summing them overflows: four pieces that can all reach g8 give
    # 4 * 2**62, which is exactly 2**64 and wraps to zero, and two that can reach
    # h8 do the same. That made `1r1r2Rk/pp5p/4b2n/1P5P/n1Ppp3/R4P1p/1B1KP3/8 b -
    # - 1 1` report checkmate with result -1 while having four legal replies, all
    # of them capturing the checking rook. Found by the CUDA port on 2026-07-29.
    no_moves = (mask == 0).all(-1)
    code = torch.zeros(mask.shape[0], dtype=torch.uint8, device=mask.device)
    code = torch.where(no_moves & ~in_check, torch.tensor(STALEMATE, dtype=torch.uint8,
                                                          device=code.device), code)
    code = torch.where(no_moves & in_check, torch.tensor(CHECKMATE, dtype=torch.uint8,
                                                         device=code.device), code)

    def draw(flag: torch.Tensor, value: int) -> None:
        nonlocal code
        code = torch.where((code == NONE) & flag,
                           torch.tensor(value, dtype=torch.uint8, device=code.device), code)

    draw(control.abs() >= 101, FIFTY_MOVE)
    if hash is not None and ring is not None and length is not None:
        draw(repetition_count(hash, ring, length) >= 3, REPETITION)
    draw(insufficient_material(boards), INSUFFICIENT)

    result = torch.where(code == CHECKMATE, -1, 0).to(torch.int8)
    return code, result


def play(boards: torch.Tensor, control: torch.Tensor, move: torch.Tensor,
         promo: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
         in_place: bool = False) -> Tuple[torch.Tensor, ...]:
    """`step` then `movegen`: (boards, control, mask, in_check)."""
    boards, control, _, _ = step(boards, control, move, promo=promo, mask=mask, in_place=in_place)
    return (boards, control) + movegen(boards, control)


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #

def initial_boards(n: int = 1, device: Optional[torch.device] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`n` copies of the starting position, white to move, clock 0."""
    boards = luts.get("start_board", device).unsqueeze(0).expand(n, -1).clone()
    control = torch.ones((n,), dtype=torch.int16, device=device)
    return boards, control


def empty_boards(n: int = 1, device: Optional[torch.device] = None,
                 black_to_move: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """`n` positions with every slot captured. Kings included, so illegal by the
    rules; this exists to be filled in by a constructor."""
    boards = (luts.get("start_board", device) | CAPTURED).unsqueeze(0).expand(n, -1).clone()
    control = torch.full((n,), -1 if black_to_move else 1, dtype=torch.int16, device=device)
    return boards, control


def from_board(c_board) -> Tuple[torch.Tensor, torch.Tensor]:
    """python-chess Board -> (boards [1,32], control [1]). Requires python-chess."""
    occupied = torch.tensor(c_board.occupied, dtype=torch.uint64).view(torch.int64)
    squares = ((occupied >> torch.arange(64, dtype=torch.long)) & 1).nonzero(as_tuple=True)[0]
    # occupied_co is indexed by colour and chess.WHITE is 1, so [1] is White.
    white = ((torch.tensor(c_board.occupied_co[1], dtype=torch.uint64).view(torch.int64) >> squares) & 1).bool()
    by_type = ((torch.tensor(
        [c_board.pawns, c_board.knights, c_board.bishops, c_board.rooks,
         c_board.queens, c_board.kings], dtype=torch.uint64).view(torch.int64)[:, None] >> squares) & 1).bool()

    if c_board.ep_square is not None:
        squares[squares == (c_board.ep_square + (-8 if c_board.turn else 8))] |= SPECIAL

    black = ~white
    w_pawns, b_pawns = squares[by_type[0] & white], squares[by_type[0] & black]
    w_knights, b_knights = squares[by_type[1] & white], squares[by_type[1] & black]
    w_bishops, b_bishops = squares[by_type[2] & white], squares[by_type[2] & black]
    w_rooks, b_rooks = squares[by_type[3] & white], squares[by_type[3] & black]
    w_queens, b_queens = squares[by_type[4] & white], squares[by_type[4] & black]
    w_king, b_king = squares[by_type[5] & white], squares[by_type[5] & black]

    # Material beyond the initial count is promoted material, and lives in the
    # pawn slots with its type bits overwritten (spec §2.1).
    w_pawns = torch.cat([w_pawns, w_knights[2:] | (KNIGHT << 6), w_bishops[2:] | (BISHOP << 6),
                         w_rooks[2:] | (ROOK << 6), w_queens[1:] | (QUEEN << 6)])
    b_pawns = torch.cat([b_pawns, b_knights[2:] | (KNIGHT << 6), b_bishops[2:] | (BISHOP << 6),
                         b_rooks[2:] | (ROOK << 6), b_queens[1:] | (QUEEN << 6)])

    pad = torch.nn.functional.pad
    board = torch.cat([
        pad(w_pawns, (0, 8 - len(w_pawns)), value=CAPTURED),
        pad(w_knights | (KNIGHT << 6), (0, 2 - len(w_knights)), value=CAPTURED),
        pad(w_bishops | (BISHOP << 6), (0, 2 - len(w_bishops)), value=CAPTURED),
        pad(w_rooks | (ROOK << 6), (0, 2 - len(w_rooks)), value=CAPTURED),
        pad(w_queens | (QUEEN << 6), (0, 1 - len(w_queens)), value=CAPTURED),
        pad(w_king | (KING << 6), (0, 1 - len(w_king)), value=CAPTURED),
        pad(b_pawns, (0, 8 - len(b_pawns)), value=CAPTURED),
        pad(b_knights | (KNIGHT << 6), (0, 2 - len(b_knights)), value=CAPTURED),
        pad(b_bishops | (BISHOP << 6), (0, 2 - len(b_bishops)), value=CAPTURED),
        pad(b_rooks | (ROOK << 6), (0, 2 - len(b_rooks)), value=CAPTURED),
        pad(b_queens | (QUEEN << 6), (0, 1 - len(b_queens)), value=CAPTURED),
        pad(b_king | (KING << 6), (0, 1 - len(b_king)), value=CAPTURED),
    ])[None]
    board[:, 16:32] |= COLOR

    # `special` records the loss of a right, so it is set when the right is gone.
    rights = torch.tensor(c_board.castling_rights, dtype=torch.uint64).view(torch.int64)
    if len(w_rooks) > 0:
        board[0, 12] |= ((~rights >> w_rooks[0]) & 1) << 9
    if len(w_rooks) > 1:
        board[0, 13] |= ((~rights >> w_rooks[1]) & 1) << 9
    board[0, 15] |= ((rights & 0b10000001) == 0) << 9
    if len(b_rooks) > 0:
        board[0, 28] |= ((~rights >> b_rooks[0]) & 1) << 9
    if len(b_rooks) > 1:
        board[0, 29] |= ((~rights >> b_rooks[1]) & 1) << 9
    board[0, 31] |= (((rights >> 56) & 0b10000001) == 0) << 9

    control = torch.tensor([(1 if c_board.turn else -1) * (c_board.halfmove_clock + 1)],
                           dtype=torch.int16)
    return board.to(torch.int16), control


def from_boards(c_boards: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """A list of python-chess Boards -> one batch."""
    pairs = [from_board(b) for b in c_boards]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


def from_fen(fen: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """FEN -> (boards [1,32], control [1]). No python-chess involved."""
    board, _ = empty_boards(1)
    board_fen, turn, castle, en_passant, halfmove, _fullmove = fen.split(" ")

    ep_square = None
    if en_passant != "-":
        ep_square = _SQUARE_NAME_TO_ID[en_passant] + (8 if turn == "b" else -8)

    row, col = 7, 0
    # char -> (next free index, slot offset, how many slots of that type exist)
    next_slot = {k: (0, o, n) for k, o, n in zip(
        "PNBRQKpnbrqk", (0, 8, 10, 12, 14, 15, 16, 24, 26, 28, 30, 31),
        (8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1))}
    for char in board_fen:
        if char in _ROWS:
            col += int(char)
        elif char == "/":
            row -= 1
            col = 0
        else:
            i, o, n = next_slot[char]
            if i >= n:  # promoted material goes into a pawn slot
                original = "p" if char.islower() else "P"
                i, o, n = next_slot[original]
                board[0, o + i] += _TYPE_FROM_CHAR[char.lower()] << 6
                char = original
            square = row * 8 + col
            board[0, o + i] &= ~(CAPTURED | SQUARE) & 0xFFF
            board[0, o + i] += square
            if char.lower() == "p" and square == ep_square:
                board[0, o + i] |= SPECIAL
            if char.lower() == "r":
                right = "qk"[int(square % 8 == 7)]
                right = right.upper() if char.isupper() else right
                if right not in castle:
                    board[0, o + i] |= SPECIAL
            if char.lower() == "k":
                rights = "KQ" if char.isupper() else "kq"
                if not (rights[0] in castle or rights[1] in castle):
                    board[0, o + i] |= SPECIAL
            next_slot[char] = (i + 1, o, n)
            col += 1

    control = torch.tensor([(-1 if turn == "b" else 1) * (int(halfmove) + 1)], dtype=torch.int16)
    return board, control


def parse_san(san: str, is_black: bool):
    """Minimal SAN parser: (type, target, file, rank, promotion, result)."""
    if san in _SCORE_FROM_SAN:
        return None, None, None, None, None, _SCORE_FROM_SAN[san]
    san = san.replace("x", "")
    piece_type = target = piece_col = piece_row = promote_to = None
    match san[0]:
        case "O":
            if san[:3] == "O-O":
                return ("K",) + (("g1", 4, 0, None, None) if not is_black else ("g8", 4, 7, None, None))
            return ("K",) + (("c1", 4, 0, None, None) if not is_black else ("c8", 4, 7, None, None))
        case "N" | "B" | "R" | "Q" | "K":
            piece_type, san = san[0:1], san[1:]
        case _:
            piece_type = "P"
    if san[0] in _COLS and san[1] in _COLS:
        piece_col, san = _COLS.index(san[0]), san[1:]
    elif san[0] in _ROWS and san[1] in _COLS:
        piece_row, san = _ROWS.index(san[0]), san[1:]
    elif san[0] in _COLS and san[1] in _ROWS and len(san) > 2 and san[2] in _COLS:
        piece_col, piece_row, san = _COLS.index(san[0]), _ROWS.index(san[1]), san[2:]
    target = san[:2]
    if len(san) > 2 and san[2] == "=":
        promote_to = san[3]
    return piece_type, target, piece_col, piece_row, promote_to, None


def from_pgn(pgn: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replay a PGN movetext through the engine and return the final position."""
    boards, control = initial_boards(1)
    mask, _ = movegen(boards, control)
    for i, san in enumerate(filter(lambda x: not x.endswith("."), pgn.split(" "))):
        is_black = bool(control[0] < 0)
        piece_type, target, piece_col, piece_row, _promote_to, result = parse_san(san, is_black)
        if result is not None:
            break
        target_id = _SQUARE_NAME_TO_ID[target]
        slots = _SLOTS_FROM_CHAR[piece_type]
        if is_black:
            slots = slice(slots.start + 16, slots.stop + 16)
        candidates = ((mask[..., slots] >> target_id) & 1).bool()
        squares = boards[..., slots] & SQUARE
        if piece_col is not None:
            candidates &= (squares % 8) == piece_col
        if piece_row is not None:
            candidates &= (squares // 8) == piece_row
        found = candidates.nonzero()
        assert found.shape[0] == 1, \
            f"{'ambiguous' if len(found) > 1 else 'invalid'} move {san} at ply {i}"
        move = (found[0, 1] + slots.start) * 64 + target_id
        boards, control, mask, _ = play(boards, control, move[None])
    return boards, control
