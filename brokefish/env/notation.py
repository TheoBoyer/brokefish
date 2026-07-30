"""Position and game output: FEN, UCI, SAN and PGN.

The inverses of `from_fen`, `parse_san` and `from_pgn`, which live in
`torch_impl` and had no counterpart until now. Nothing in this repository could
emit a game, which blocks every layer of [Track D](../../docs/evals.md) that
talks to an external engine, and makes debugging the search harder than it needs
to be.

Host-side string I/O only. Nothing here is on the self-play path, so the loops
are plain Python over a batch: the tensor work (`movegen`, `play`) is batched and
the string assembly is not, because a PGN is a per-game object and 80 plies of
`str` concatenation is not what any of this costs.

    to_fen(boards, control, fullmove=1) -> list[str]
    to_uci(boards, move, promo=None)    -> list[str]
    to_san(boards, control, move, promo=None, mask=None) -> list[str]
    to_pgn(san_moves, headers=None, result="*", ...)     -> str

and `GameRecorder`, which is the shape a match harness wants: batched over games,
one PGN out per game.

⚠️ **The full-move number is not in the representation.** spec §2.2's control word
carries the side to move and the halfmove clock and nothing else, because the
network never sees a move number and the rules never consult one. So `to_fen`
takes it as an argument and defaults to 1, and a FEN round-tripped through the
engine is equal to the original in its first five fields only.

⚠️ **The en passant field follows the *legal* convention**, matching
`legal_ep_file` (spec §6.1) and python-chess's default `en_passant="legal"`: the
square appears only when the capture is actually available, king safety included.
The other convention — print it after any double push — makes real repetitions
invisible, which is the same argument §6.1 makes for the hash. Compare against
`chess.Board.fen()` and not against `fen(en_passant="fen")`.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch

from .torch_impl import (
    PAWN, KING, SQUARE,
    _SQUARE_ID_TO_NAME,
    castling_rights, decode, legal_ep_file, movegen, play,
)

__all__ = ["to_fen", "to_uci", "to_san", "to_pgn", "GameRecorder", "STARTPOS_FEN"]

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_PIECE_CHAR = "pnbrqk"          # spec §2.1 type order
_PROMO_CHAR = "nbrq"            # spec §3 promo order, 0:N 1:B 2:R 3:Q
_FILE_NAMES = "abcdefgh"
_RANK_NAMES = "12345678"

# The seven tag roster, in the order the PGN standard fixes.
_ROSTER = ("Event", "Site", "Date", "Round", "White", "Black", "Result")


def _batch(boards: torch.Tensor, control: torch.Tensor):
    return boards.reshape(-1, 32).cpu(), control.reshape(-1).cpu()


# --------------------------------------------------------------------------- #
# FEN
# --------------------------------------------------------------------------- #

def to_fen(boards: torch.Tensor, control: torch.Tensor,
           fullmove: Union[int, Sequence[int]] = 1) -> List[str]:
    """`(boards [N,32], control [N])` -> `N` FEN strings.

    The inverse of `from_fen` on the five fields the representation carries.
    `fullmove` is an int applied to every position or one int per position.
    """
    boards, control = _batch(boards, control)
    n = boards.shape[0]
    fulls = [fullmove] * n if isinstance(fullmove, int) else list(fullmove)
    if len(fulls) != n:
        raise ValueError(f"fullmove has {len(fulls)} entries for {n} positions")

    rights = castling_rights(boards)
    ep_file = legal_ep_file(boards, control)
    captured, color, _special, types, squares = decode(boards)

    out = []
    for i in range(n):
        grid: List[Optional[str]] = [None] * 64
        for s in range(32):
            if captured[i, s]:
                continue
            char = _PIECE_CHAR[int(types[i, s])]
            grid[int(squares[i, s])] = char if color[i, s] else char.upper()

        rows = []
        for rank in range(7, -1, -1):
            row, gap = "", 0
            for file in range(8):
                piece = grid[rank * 8 + file]
                if piece is None:
                    gap += 1
                    continue
                if gap:
                    row, gap = row + str(gap), 0
                row += piece
            rows.append(row + (str(gap) if gap else ""))

        white_to_move = bool(control[i] > 0)
        castle = "".join(c for c, ok in zip("KQkq", rights[i].tolist()) if ok) or "-"
        file = int(ep_file[i])
        # White to move means the flagged pawn is Black's, so the target is on
        # rank 6 (index 5); the mirror is rank 3 (index 2).
        ep = "-" if file < 0 else _SQUARE_ID_TO_NAME[(40 if white_to_move else 16) + file]
        out.append(" ".join(("/".join(rows), "w" if white_to_move else "b", castle, ep,
                             str(int(control[i].abs()) - 1), str(fulls[i]))))
    return out


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #

def _unpack(boards: torch.Tensor, move: torch.Tensor, promo: Optional[torch.Tensor]):
    """`(slot, target, source, type, is_null, promo)` as plain int64 tensors."""
    move = move.reshape(-1).cpu().long()
    is_null = move < 0
    safe = torch.where(is_null, torch.zeros_like(move), move)
    slot, target = safe // 64, safe % 64
    words = boards.gather(1, slot[:, None]).squeeze(1).long() & 0xFFF
    if promo is None:
        promo = torch.full_like(move, 3)          # spec §3 default: queen
    else:
        promo = promo.reshape(-1).cpu().long()
    return slot, target, words & SQUARE, (words >> 6) & 0b111, is_null, promo


def to_uci(boards: torch.Tensor, move: torch.Tensor,
           promo: Optional[torch.Tensor] = None) -> List[str]:
    """`(boards [N,32], move [N])` -> `N` UCI long-algebraic strings.

    `move` is `slot * 64 + target` (spec §3). Castling comes out as the king's
    own from-to, `e1g1`, because that is exactly how the move is encoded — the
    rook leg lives in `step` and never in the action space. A null move
    (`move == -1`) is UCI's `0000`.

    ⚠️ `promo` defaults to queen when omitted, matching `step` and
    `interop.move_to_args`. It is read only when the mover is a pawn landing on
    the back rank, so passing it for every move is harmless.
    """
    boards = boards.reshape(-1, 32).cpu()
    _slot, target, source, ptype, is_null, promo = _unpack(boards, move, promo)
    out = []
    for i in range(boards.shape[0]):
        if is_null[i]:
            out.append("0000")
            continue
        text = _SQUARE_ID_TO_NAME[int(source[i])] + _SQUARE_ID_TO_NAME[int(target[i])]
        if ptype[i] == PAWN and int(target[i]) >> 3 in (0, 7):
            text += _PROMO_CHAR[int(promo[i])]
        out.append(text)
    return out


def to_san(boards: torch.Tensor, control: torch.Tensor, move: torch.Tensor,
           promo: Optional[torch.Tensor] = None,
           mask: Optional[torch.Tensor] = None) -> List[str]:
    """`(boards, control, move)` -> `N` SAN strings, with check and mate suffixes.

    SAN is what a PGN's movetext is made of, so this is the piece D0 exists for.
    It costs one `movegen` for the disambiguation and one `play` for the suffix;
    pass `mask` if the caller already has the current one.

    Disambiguation follows the standard rule, which is subtler than "add the
    file": among the *other* legal moves of the same piece type to the same
    square, a candidate sharing our rank forces the file, a candidate sharing our
    file forces the rank, and anything else forces the file. Both can apply.
    """
    boards_c, control_c = _batch(boards, control)
    if mask is None:
        mask, _ = movegen(boards, control)
    mask = mask.reshape(-1, 32).cpu()
    slot, target, source, ptype, is_null, promo_t = _unpack(boards_c, move, promo)

    alive_captured, color, _special, types, squares = decode(boards_c)
    _b, _c, next_mask, next_check = play(boards, control,
                                         move.reshape(-1).to(boards.device), promo=promo)
    next_mask = next_mask.reshape(-1, 32).cpu()
    next_check = next_check.reshape(-1).cpu()
    gives_mate = next_check & (next_mask == 0).all(-1)

    out = []
    for i in range(boards_c.shape[0]):
        if is_null[i]:
            out.append("--")                       # PGN's null-move convention
            continue
        src, dst = int(source[i]), int(target[i])
        kind, me = int(ptype[i]), int(slot[i])
        src_file, src_rank, dst_file = src % 8, src // 8, dst % 8

        # A friendly piece on the target would make the move illegal, so any live
        # piece standing there is an enemy and the move is a capture.
        occupied = bool(((squares[i] == dst) & ~alive_captured[i]).any())
        if kind == KING and abs(dst_file - src_file) == 2:
            text = "O-O" if dst_file == 6 else "O-O-O"
        elif kind == PAWN:
            # A diagonal pawn move is a capture even onto an empty square: that
            # is en passant, whose victim does not stand on the target.
            capture = dst_file != src_file
            text = (_FILE_NAMES[src_file] + "x" if capture else "") + _SQUARE_ID_TO_NAME[dst]
            if dst >> 3 in (0, 7):
                text += "=" + _PROMO_CHAR[int(promo_t[i])].upper()
        else:
            others = [s for s in range(32)
                      if s != me and int(types[i, s]) == kind
                      and bool(color[i, s]) == bool(color[i, me])
                      and (int(mask[i, s]) >> dst) & 1]
            hint = ""
            if others:
                squares_o = [int(squares[i, s]) for s in others]
                need_file = any(sq // 8 == src_rank for sq in squares_o)
                need_rank = any(sq % 8 == src_file for sq in squares_o)
                if not need_rank and not need_file:
                    need_file = True
                hint = (_FILE_NAMES[src_file] if need_file else "") + \
                       (_RANK_NAMES[src_rank] if need_rank else "")
            text = _PIECE_CHAR[kind].upper() + hint + ("x" if occupied else "") \
                + _SQUARE_ID_TO_NAME[dst]

        out.append(text + ("#" if gives_mate[i] else "+" if next_check[i] else ""))
    return out


# --------------------------------------------------------------------------- #
# PGN
# --------------------------------------------------------------------------- #

def to_pgn(san_moves: Sequence[str], headers: Optional[Dict[str, str]] = None,
           result: str = "*", start_fen: Optional[str] = None,
           first_fullmove: int = 1, first_is_black: bool = False,
           width: int = 80) -> str:
    """SAN movetext plus tags -> one PGN game.

    `result` is one of `1-0`, `0-1`, `1/2-1/2`, `*`, and is written both as the
    `Result` tag and as the game terminator, which the standard requires to
    agree. A game that does not start from the initial position gets the
    `SetUp`/`FEN` tag pair, without which the movetext is unreplayable.
    """
    tags = {"Event": "?", "Site": "?", "Date": "????.??.??", "Round": "?",
            "White": "?", "Black": "?"}
    tags.update(headers or {})
    tags["Result"] = result
    if start_fen is not None and start_fen != STARTPOS_FEN:
        tags["SetUp"], tags["FEN"] = "1", start_fen

    ordered = [k for k in _ROSTER] + [k for k in tags if k not in _ROSTER]
    header = "".join(f'[{k} "{tags[k]}"]\n' for k in ordered if k in tags)

    tokens, number, black_to_move = [], first_fullmove, first_is_black
    if first_is_black and san_moves:
        tokens.append(f"{number}...")
    for san in san_moves:
        if not black_to_move:
            tokens.append(f"{number}.")
        tokens.append(san)
        if black_to_move:
            number += 1
        black_to_move = not black_to_move
    tokens.append(result)

    lines, line = [], ""
    for token in tokens:
        if line and len(line) + 1 + len(token) > width:
            lines.append(line)
            line = token
        else:
            line = f"{line} {token}" if line else token
    if line:
        lines.append(line)
    return header + "\n" + "\n".join(lines) + "\n"


class GameRecorder:
    """`N` games in flight, one PGN out per game.

    The shape the match harness of [`evals.md`](../../docs/evals.md) §10.1 wants:
    a batched `push`, since our side plays every game at once, and a per-game
    `pgn`, since that is what gets published.

    ⚠️ `active` exists because games finish at different plies. An inactive game
    is passed a null move, so its board does not advance and no SAN is recorded,
    which is what keeps a finished game from picking up junk while its neighbours
    play on.
    """

    def __init__(self, boards: torch.Tensor, control: torch.Tensor,
                 headers: Optional[Sequence[Optional[Dict[str, str]]]] = None,
                 fullmove: int = 1):
        self.boards, self.control = boards.clone(), control.clone()
        self.n = self.boards.reshape(-1, 32).shape[0]
        self.start_fen = to_fen(self.boards, self.control, fullmove)
        self.first_fullmove = fullmove
        self.first_is_black = [bool(c < 0) for c in self.control.reshape(-1).tolist()]
        self.headers = list(headers) if headers else [None] * self.n
        self.moves: List[List[str]] = [[] for _ in range(self.n)]

    def push(self, move: torch.Tensor, promo: Optional[torch.Tensor] = None,
             active: Optional[torch.Tensor] = None) -> None:
        """Record one ply per active game and advance those boards."""
        move = move.reshape(-1).to(self.control.device).long()
        if active is not None:
            move = torch.where(active.reshape(-1).to(move.device), move,
                               torch.full_like(move, -1))
        san = to_san(self.boards, self.control, move, promo=promo)
        for i, text in enumerate(san):
            if move[i] >= 0:
                self.moves[i].append(text)
        self.boards, self.control, _, _ = play(self.boards, self.control, move, promo=promo)

    def pgn(self, i: int, result: str = "*",
            headers: Optional[Dict[str, str]] = None) -> str:
        merged = dict(self.headers[i] or {})
        merged.update(headers or {})
        return to_pgn(self.moves[i], merged, result, self.start_fen[i],
                      self.first_fullmove, self.first_is_black[i])

    def all_pgns(self, results: Optional[Sequence[str]] = None) -> List[str]:
        results = results or ["*"] * self.n
        return [self.pgn(i, results[i]) for i in range(self.n)]
