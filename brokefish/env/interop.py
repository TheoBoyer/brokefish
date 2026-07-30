"""Conversions between the piece-list representation and python-chess.

Test and debug scaffolding only: python-chess is the correctness oracle, and
nothing in the training loop may depend on this module.
"""

from typing import Dict, List, Optional, Union

import chess
import torch

from .torch_impl import KING, PAWN, ROOK, SQUARE, decode


def to_chess_board(words: torch.Tensor, control: torch.Tensor) -> chess.Board:
    """One position ([32] words, scalar control) -> a python-chess Board."""
    captured, color, special, types, squares = decode(words)
    board = chess.Board(None)
    for i in range(squares.shape[0]):
        if not captured[i]:
            board.set_piece_at(
                int(squares[i].item()),
                chess.Piece(chess.PIECE_TYPES[types[i].item()],
                            chess.BLACK if color[i].item() else chess.WHITE),
            )
    # `special` marks a lost right, so an unmoved king and rook mean the right holds.
    rights = ""
    intact = ~special & ~captured
    unmoved_king = (types == KING) & intact
    unmoved_rooks = (types == ROOK) & intact
    if unmoved_king.any() and unmoved_rooks.any():
        white_ok = (unmoved_king & ~color).any() & unmoved_rooks & ~color
        if (white_ok & (squares == 7)).any():
            rights += "K"
        if (white_ok & (squares == 0)).any():
            rights += "Q"
        black_ok = (unmoved_king & color).any() & unmoved_rooks & color
        if (black_ok & (squares == 63)).any():
            rights += "k"
        if (black_ok & (squares == 56)).any():
            rights += "q"
    board.set_castling_fen(rights if rights else "-")

    double_pushed = (types == PAWN) & special & ~captured
    if double_pushed.any():
        board.ep_square = (int(squares[double_pushed].item())
                           + (-8 if color[double_pushed].item() == 0 else 8))

    control = int(control.item() if torch.is_tensor(control) else control)
    board.turn = control > 0
    board.halfmove_clock = abs(control) - 1
    return board


def print_board(words: torch.Tensor, control: torch.Tensor) -> None:
    print(to_chess_board(words, control))


def print_bitmask(bitmask: int, white_perspective: bool = True) -> None:
    """An ASCII board of one 64-bit square set."""
    out = ""
    for rank, letter in enumerate("87654321" if white_perspective else "12345678"):
        out += letter + ". "
        for file, _ in enumerate("abcdefgh" if white_perspective else "hgfedcba"):
            square = rank * 8 + 7 - file
            if white_perspective:
                square = 63 - square
            out += "██" if (bitmask >> square) & 1 else "  "
        out += "\n"
    out += "   " + " ".join("a b c d e f g h".split() if white_perspective
                            else "h g f e d c b a".split())
    print(out)


def list_legal_moves(words: torch.Tensor, mask: torch.Tensor) -> List[chess.Move]:
    """One position -> its legal moves as python-chess Moves.

    The engine's action space is 32 x 64 and promotion lives outside it
    ([spec §3](../../docs/spec.md)), so a pawn move onto the last rank is one bit
    here and four `chess.Move` objects on the way out. The search expands the same
    bit into four tree edges, deriving it the same way: mover is a pawn, target is
    on rank 0 or rank 7. A pawn can never reach its own back rank, so the colour
    does not enter the test.
    """
    slots, targets = ((mask[..., None] >> torch.arange(64, device=mask.device)) & 1).bool().nonzero(as_tuple=True)
    moves = []
    for slot, target in zip(slots, targets):
        source, target = int(words[slot] & SQUARE), int(target)
        is_pawn = ((int(words[slot]) >> 6) & 0b111) == PAWN
        if is_pawn and target >> 3 in (0, 7):
            # spec §3 order 0:N 1:B 2:R 3:Q; python-chess types run 2:N .. 5:Q.
            moves += [chess.Move(source, target, promotion=p + 2) for p in range(4)]
        else:
            moves.append(chess.Move(source, target))
    return moves


def label_to_move(words: torch.Tensor, label: int) -> chess.Move:
    """One edge label ([spec §3](../../docs/spec.md), promotion field included)
    -> a python-chess Move.

    The single-label inverse of :func:`list_legal_moves`, which the search's edges
    need because a tree edge already carries the promotion choice that a legality
    mask leaves open. Source comes from the slot's own word rather than from the
    label, since the label names a slot and not a square.
    """
    move = int(label) & ((1 << 11) - 1)
    promo = (int(label) >> 11) & 0b11
    slot, target = divmod(move, 64)
    word = int(words[slot])
    source = word & SQUARE
    is_pawn = ((word >> 6) & 0b111) == PAWN
    if is_pawn and target >> 3 in (0, 7):
        # spec §3 order 0:N 1:B 2:R 3:Q; python-chess types run 2:N .. 5:Q.
        return chess.Move(source, target, promotion=promo + 2)
    return chess.Move(source, target)


def move_to_args(moves: Union[chess.Move, List[Optional[chess.Move]]],
                 words: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
    """python-chess Moves -> the `move` and `promo` tensors `step` takes.

    `words` is the batch of boards, one row per move. The source slot is found by
    matching the square *and* the captured bit, so a dead slot never matches.
    """
    if isinstance(moves, chess.Move):
        moves = [moves]
    packed = torch.tensor(
        [[m.from_square if m is not None else -1,
          m.to_square if m is not None else -1,
          # python-chess promotion is a piece type (2=N .. 5=Q); spec §3 is 0..3.
          m.promotion - 2 if m is not None and m.promotion is not None else -1]
         for m in moves],
        dtype=torch.long, device=words.device)
    source = ((words & (0b100000111111)) == packed[:, 0:1]).int().argmax(-1)
    return {
        "move": source * 64 + packed[:, 1],
        "promo": None if (packed[:, 2] == -1).all()
                 else torch.where(packed[:, 2] == -1, 3, packed[:, 2]),
    }
