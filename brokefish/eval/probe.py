"""What happens after every legal reply, for a whole batch of positions.

This is the one primitive the rule suites of `evals.md` §8.1 are built on. Given
`N` positions it enumerates every legal `(move, promotion)` of every one of them,
applies it, and returns the terminal code of the resulting position. Mate in 1,
stalemate-in-1, the draw-by-repetition a move walks into and the promotion choice
that mates where the queen does not all fall out of that one table.

§8.1 calls it "running `movegen` twice", which is exactly the cost: the whole
scan is one `step` and one `movegen` over the flattened move list, roughly 30×
the batch. Nothing here is per-position Python.

Labels are the search's own edge encoding (`mcts.md` §6.4), `slot * 64 + target`
in the low 11 bits with the promotion type above it, so a suite's answer key can
be compared directly against `MoveRecord.played` with no translation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from brokefish.env import torch_impl as env

MOVE_BITS = 11
PROMO_SHIFT = MOVE_BITS
MOVE_MASK = (1 << MOVE_BITS) - 1

PAWN = env.PAWN


def promotion_targets(boards: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """``[N, 32, 64] bool``: the move ``slot -> square`` is a promotion.

    The type is read from the piece word, never from the slot index: a promoted
    queen keeps its pawn slot, so a slot in 0-7 does not imply a pawn
    (spec §2.1). Same rule as the search's `_promotion_targets`, and it has to
    stay the same rule, since the labels this module produces are compared
    against the search's.
    """
    words = boards.to(torch.int32)
    pawn = (((words >> 6) & 0b111) == PAWN) & (((words >> 11) & 1) == 0)
    last_rank = torch.where(control < 0, 0, 7).long()
    sq = torch.arange(64, device=boards.device)
    on_last = (sq[None, :] // 8) == last_rank[:, None]
    return pawn[:, :, None] & on_last[:, None, :]


def enumerate_moves(boards: torch.Tensor, control: torch.Tensor,
                    mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(game [M] int64, label [M] int64)``, every legal edge of every position.

    Flattened rather than padded, because the number of legal moves ranges from
    0 to 218 and padding to the worst case wastes 85 % of the work. The order is
    the canonical one of `mcts.md` §6.4 — ascending by slot, then target, then
    promotion type in spec §3's ``N B R Q`` order — so a suite's answer key and
    the search's edge list agree on index as well as on value.
    """
    if mask is None:
        mask, _ = env.movegen(boards, control)
    legal = env.bitset_to_bool(mask)                              # [N,32,64]
    is_promo = promotion_targets(boards, control) & legal
    cand = legal[..., None].expand(-1, -1, -1, 4).clone()
    cand[..., 1:] &= is_promo[..., None]                          # only promotions branch
    flat = cand.reshape(boards.shape[0], -1)                      # [N, 32*64*4]
    game, col = flat.nonzero(as_tuple=True)
    label = ((col // 4) | ((col % 4) << PROMO_SHIFT)).to(torch.int64)
    return game, label


def reply_codes(boards: torch.Tensor, control: torch.Tensor,
                hash: Optional[torch.Tensor] = None,
                ring: Optional[torch.Tensor] = None,
                length: Optional[torch.Tensor] = None,
                chunk: int = 1 << 15,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
    """``(game, label, code, next_hash, irreversible)`` — one row per legal reply.

    ``code`` is spec §4.3's terminal code of the position the reply produced.

    Codes: 0 none, 1 checkmate, 2 stalemate, 3 fifty-move, 4 threefold, 5
    insufficient material, first-applicable. They are from the point of view of
    the player who is to move *after* the reply, so code 1 means the mover just
    delivered mate.

    ⚠️ Without ``ring``/``length`` the repetition test is skipped, and code 4
    never appears. The threefold suite is therefore only harvestable from
    positions that carry their game history — which is why `harvest_suites`
    keeps the ring alongside the boards rather than regenerating positions from
    a FEN.

    Chunked because the ring is 100 ``int64`` per position and the flattened move
    list is ~30× the batch, so a 4096-position scan would otherwise materialise
    ~100 MB of history alone.
    """
    game, label = enumerate_moves(boards, control)
    if hash is None:
        hash = env.hash_position(boards, control)

    codes, hashes, irrevs = [], [], []
    for lo in range(0, int(game.numel()), chunk):
        g = game[lo:lo + chunk]
        lab = label[lo:lo + chunk]
        move = lab & MOVE_MASK
        promo = (lab >> PROMO_SHIFT) & 0b11
        h = hash[g]
        nb, nc, nh, irrev = env.step(boards[g], control[g], move, promo=promo, hash=h)
        if ring is not None and length is not None:
            # The ring records the position being *left*, and the move's own
            # irreversibility is what empties it (spec §6.2), so this push comes
            # before the terminal test of the position it produced.
            nring, nlen = env.push_history(ring[g], length[g], h, irrev)
        else:
            nring = nlen = None
        m, chk = env.movegen(nb, nc)
        code, _ = env.terminal(m, chk, nc, nb, nh, nring, nlen)
        codes.append(code)
        hashes.append(nh)
        irrevs.append(irrev)

    def cat(parts, dtype):
        return (torch.cat(parts) if parts
                else torch.zeros(0, dtype=dtype, device=boards.device))

    return (game, label, cat(codes, torch.uint8),
            cat(hashes, torch.int64), cat(irrevs, torch.bool))
