"""Random legal positions, built from the rules and rejected by the rules.

`tests/boards.py` generates positions by random legal play from the start, which
is the right source for testing the *encoder*: it produces exactly the words the
rules produce, with a realistic spread of clocks, captures and dead slots.

It is the wrong source for the rule suites of `evals.md` §8.1, and that is a
measurement rather than an opinion. 170 924 positions of random play from the
start position (512 games, 400 plies) contain:

    mate in 1 available          2 150
    mate and stalemate available    85
    mate and threefold               3
    mate and insufficient            2
    mate and fifty-move              0
    underpromotion mates, queen does not   0

Random play spends its whole life in a full-board middlegame, and every motif
except plain mate-in-1 lives in a sparse endgame. So this module goes at the
endgame directly: scatter a couple of kings and a handful of pieces at random,
and let our own move generator throw out whatever is not a legal position.

⚠️ **This is a sampling distribution, not a chess opinion.** It biases *which*
positions get looked at — towards few pieces, where mates and stalemates are —
and it never biases what the right answer is, which stays a terminal code
computed by the rules. Nothing generated here enters training; §2's prohibition
is on evaluation output flowing backwards, and a position is an input.

Legality is decided by the engine, not by a hand-written checklist: the only
substantive test is that **the side not to move is not in check**, which is what
rules out both a capturable king and two adjacent kings in one line.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from brokefish.env import torch_impl as env

CAPTURED, COLOR, SPECIAL = env.CAPTURED, env.COLOR, env.SPECIAL
PAWN, ROOK, KING = env.PAWN, env.ROOK, env.KING
KING_SLOT_WHITE, KING_SLOT_BLACK = 15, 31


def _sample(n: int, gen: torch.Generator, device, max_extra: int,
            promotion_ready: bool = False, clock: int = 0,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One unfiltered draw: `n` candidate positions, most of them legal."""
    def rand(hi: int, shape) -> torch.Tensor:
        return torch.randint(0, hi, shape, generator=gen).to(device)

    slot = torch.arange(32, device=device)[None, :]
    ptype = rand(5, (n, 32))                       # P N B R Q, never a second king
    square = rand(64, (n, 32))
    # A pawn on the first or last rank is not a position the rules can produce.
    square = torch.where(ptype == PAWN, square % 48 + 8, square)

    n_white = rand(max_extra + 1, (n, 1))
    n_black = rand(max_extra + 1, (n, 1))
    alive = torch.where(slot < 16, slot < n_white, (slot - 16) < n_black)

    ptype = torch.where(slot % 16 == 15, torch.full_like(ptype, KING), ptype)
    square = torch.where(slot % 16 == 15, rand(64, (n, 32)), square)
    alive = alive | (slot % 16 == 15)

    color = (slot >= 16).long()
    # SPECIAL on a king or a rook means "has moved" (`from_fen` sets it when the
    # FEN withholds the right), so setting it here is what says "no castling
    # rights". On a pawn it would mean "just double-pushed", and leaving it clear
    # is what says "no en passant".
    special = ((ptype == KING) | (ptype == ROOK)).long()

    words = (color << 10) | (special << 9) | (ptype << 6) | square
    words = torch.where(alive, words, torch.full_like(words, CAPTURED))

    black_to_move = rand(2, (n,)).bool()
    if promotion_ready:
        # One pawn of the side to move, on the rank it promotes from. Without
        # this the underpromotion suite is unharvestable: a promotion motif needs
        # a pawn one square from the eighth rank, and 200 000 unconstrained draws
        # produced four usable positions.
        slot0 = torch.where(black_to_move, 16, 0)
        file = rand(8, (n,))
        pawn_sq = torch.where(black_to_move, 8 + file, 48 + file)
        pawn = (black_to_move.long() << 10) | (PAWN << 6) | pawn_sq
        words = words.scatter(1, slot0[:, None], pawn[:, None].to(words.dtype))

        # And the enemy king within a 5x5 box of the promotion square. A
        # promotion that mates needs the king in range of the promoted piece;
        # left free it lands there once in fifty, and the motif is already a
        # 10^-5 event without that factor on top.
        promo_sq = torch.where(black_to_move, pawn_sq - 8, pawn_sq + 8)
        kf = (promo_sq % 8 + rand(5, (n,)) - 2).clamp(0, 7)
        kr = (promo_sq // 8 + rand(5, (n,)) - 2).clamp(0, 7)
        eks = torch.where(black_to_move, KING_SLOT_WHITE, KING_SLOT_BLACK)
        eking = ((~black_to_move).long() << 10) | SPECIAL | (KING << 6) | (kr * 8 + kf)
        words = words.scatter(1, eks[:, None], eking[:, None].to(words.dtype))

    control = torch.where(black_to_move, -(clock + 1), clock + 1)
    return words.to(torch.int16), control.to(torch.int16)


def _is_legal(boards: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """``[N] bool``. Two squares occupied twice, or a king already capturable."""
    captured, _color, _special, _ptype, square = env.decode(boards)
    alive = captured == 0
    occ = torch.zeros((boards.shape[0], 64), dtype=torch.int16, device=boards.device)
    occ.scatter_add_(1, square.long(), alive.to(torch.int16))
    distinct = (occ <= 1).all(-1)

    # The engine decides this one. movegen under the flipped control word reports
    # whether the player who is *not* to move stands in check, which is illegal
    # and which also catches adjacent kings, since each attacks the other.
    _mask, other_in_check = env.movegen(boards, (-control).to(torch.int16))
    return distinct & ~other_in_check


def random_positions(n: int, seed: int = 0, device: str = "cuda",
                     max_extra: int = 6, max_rounds: int = 64,
                     min_legal_moves: int = 2, promotion_ready: bool = False,
                     clock: int = 0,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(boards [n,32] int16, control [n] int16)``, legal and non-terminal.

    Rejection sampling: draw, ask the engine, keep the survivors, redraw the
    rest. The acceptance rate is high enough (roughly two thirds) that the loop
    exits in a handful of rounds; `max_rounds` exists so a pathological
    `max_extra` cannot hang.

    ``min_legal_moves`` drops positions that are already terminal or forced —
    a suite item where the mover has one legal move measures nothing.

    ``promotion_ready`` puts one pawn of the side to move on its seventh rank.
    ``clock`` sets the halfmove clock, which is how the fifty-move suite is
    reached at all: at 99 a quiet move draws and a mate does not, and no amount
    of sampling reaches that state by accident.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    boards = torch.zeros((0, 32), dtype=torch.int16, device=device)
    control = torch.zeros((0,), dtype=torch.int16, device=device)

    for _ in range(max_rounds):
        need = n - int(boards.shape[0])
        if need <= 0:
            break
        # Overdraw: the round is a single kernel launch either way, and coming
        # up two positions short costs a whole extra round.
        b, c = _sample(max(need * 2, 256), gen, device, max_extra,
                       promotion_ready=promotion_ready, clock=clock)
        ok = _is_legal(b, c)
        if min_legal_moves > 0:
            mask, _ = env.movegen(b, c)
            ok &= env.bitset_to_bool(mask).reshape(b.shape[0], -1).sum(-1) >= min_legal_moves
        keep = ok.nonzero(as_tuple=True)[0][:need]
        boards = torch.cat([boards, b[keep]])
        control = torch.cat([control, c[keep]])

    if boards.shape[0] < n:
        raise RuntimeError(f"only {boards.shape[0]}/{n} legal positions in "
                           f"{max_rounds} rounds; max_extra={max_extra} is too high")
    return boards, control
