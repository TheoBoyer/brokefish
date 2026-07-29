"""Dump the differential test set the CUDA movegen is validated against.

Writes flat little-endian binaries, plus the per-stage snapshots that let a
kernel be validated one piece at a time (base bitsets, then pawns, then sliders,
then castling) rather than only at the end.

    boards.bin        [N,32]  uint16
    control.bin       [N]     int16   sign = side to move, magnitude = clock + 1
    masks.bin         [N,32]  uint64  fully legal, bit s set iff p -> s is legal
    in_check.bin      [N]     uint8   spec §4.1, what separates mate from stalemate
    hashes.bin        [N]     uint64  spec §6.1 Zobrist of the position
    terminal_code.bin [N]     uint8   spec §4.3, 0 none .. 5 insufficient material
    terminal_result.bin [N]   int8    -1 checkmate, 0 otherwise
    fo_masks.bin      [N,32]  uint64  first order only, before king safety
    fo_stage{1..4}.bin[N,32]  uint64  after base / pawns / sliders / castling
    fo_control.bin    [N,32]  uint64  first order in control mode, generated for
                                      the side NOT to move: the attack map that
                                      `in_check` and the second order both read
    lut_*.bin                         the tables, including the 781 Zobrist keys
    meta.txt                          N and M in ASCII, one per line

and the `step` cases, M of them, one per (position, move, promotion) triple:

    step_index.bin    [M]     int32   which position of boards.bin this applies to
    step_move.bin     [M]     int16   slot * 64 + target, or -1 for the null move
    step_promo.bin    [M]     uint8   spec §3 field, 0:N 1:B 2:R 3:Q
    step_boards.bin   [M,32]  uint16  the position after the move
    step_control.bin  [M]     int16   the control word after the move
    step_hash.bin     [M]     uint64  the incrementally updated Zobrist
    step_irrev.bin    [M]     uint8   spec §6.2, what bounds the repetition window

plus the two intermediates the hash is built from, because a wrong hash gives no
clue which of its three inputs was wrong:

    castle_rights.bin [N]     uint8   bits 0..3 = KQkq
    ep_file.bin       [N]     int8    file of a *legal* en passant capture, or -1

and the perft suite, so the CUDA side needs no FEN parser:

    perft_boards.bin  [6,32]  uint16  the six standard start positions
    perft_control.bin [6]     int16
    perft_cases.txt           one line per case: name, depth count, node counts

The step cases cover every legal move of every position, with promoting moves
expanded to all four choices (spec §3 carries the promotion beside the move, so
the mask holds one bit where the tree holds four edges), plus one null move per
position for spec §9. They carry no hash and no `irreversible`: those belong to
the second entry point, and perft needs neither.

⚠️ A dump is invalidated by any change to the control word or the hash. The
fifty-move reset landed on 2026-07-29, so every set written before that date
carries a ply counter where `control.bin` now holds a clock.

`terminal_code.bin` can never be 4: threefold needs the game that reached the
position, and a static set holds positions.

    python scripts/dump_cuda_testset.py [--out data/cuda_testset] [--n 10000]

The stage snapshots replicate the body of `first_order_mask` and have to be kept
in step with it. The set mixes hand-picked positions for the rules that are easy
to get wrong with random playouts for volume.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brokefish.env import (from_fen, hash_position, initial_boards, movegen, step,
                           terminal)
from brokefish.env.torch_impl import castling_rights, legal_ep_file
from brokefish.env import luts
from brokefish.env.torch_impl import (
    PAWN, SQUARE, bitset_to_bool, castling_moves, pawn_moves, first_order_mask,
    slider_moves,
)

SEED = 20260729
N_GAMES = 256
N_PLIES = 100

# Rules that random play reaches rarely or never, and the edges of the domain.
SPECIAL_FENS = [
    # Castling: both sides available, blocked, through check, rights lost
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "r3k2r/8/8/8/8/4r3/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/5N2/8/8/8/8/R3K2R b KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1",
    "rn2k2r/8/8/8/8/8/8/R3K1NR w KQkq - 0 1",
    # En passant: plain, two choices, and the pinned one that is illegal
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
    "rnbqkbnr/ppp1p1pp/8/8/3pPp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 4",
    "8/8/8/8/k2pP2Q/8/8/4K3 b - e3 0 1",
    "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
    # Promotions, by push and by capture
    "8/P7/8/8/8/8/p7/K6k w - - 0 1",
    "1n6/P7/8/8/8/8/p6K/1N5k b - - 0 1",
    # Check and pins
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqk1nr/pppp1ppp/8/4p3/1b6/3P4/PPP1PPPP/RNBQKBNR w KQkq - 2 3",
    "4k3/8/8/8/1b6/8/3P4/4K3 w - - 0 1",
    # Mate and stalemate: an all-zero mask is expected
    "rnb1kbnr/pppp1ppp/8/4p3/6PQ/5P2/PPPPP2P/RNB1KBNR b KQkq - 0 3",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "6k1/5R2/6K1/8/8/8/8/8 b - - 0 1",
    # Sparse endgames: many dead slots, which is many idle lanes
    "8/8/8/3k4/8/3K4/8/8 w - - 0 1",
    "8/2p5/8/8/8/8/6P1/K6k w - - 0 1",
    # The fifty-move boundary, spec §2.2: the magnitude is clock + 1 and the game
    # is drawn when it reaches 101. Random playouts never get near it, so the
    # branch was untested until these two landed. Material on the board, so the
    # code is fifty-move and not insufficient.
    "8/8/8/3k4/8/8/4Q3/4K3 w - - 100 1",
    "8/8/8/3k4/8/8/4Q3/4K3 w - - 99 1",
    # Insufficient material, the pair spec §4.3 was amended for on 2026-07-29:
    # two bishops on one colour complex cannot mate, two on opposite complexes can.
    "8/8/8/3k4/8/8/8/B1B1K3 w - - 0 1",
    "8/8/8/3k4/8/8/8/BB2K3 w - - 0 1",
    # Four legal replies, every mask word 2**62, so the words sum to 2**64 and
    # `terminal` used to call this checkmate. Pinned here so the CUDA path carries
    # the case too; docs/env.md has the story.
    "1r1r2Rk/pp5p/4b2n/1P5P/n1Ppp3/R4P1p/1B1KP3/8 b - - 1 1",
]


def snapshot_playouts():
    """Random games, snapshotting every position before its move is played."""
    g = torch.Generator().manual_seed(SEED)
    boards, control = initial_boards(N_GAMES)
    mask, _ = movegen(boards, control)
    all_b, all_c = [], []
    for _ in range(N_PLIES):
        all_b.append(boards.clone())
        all_c.append(control.clone())
        legal = mask.view(-1, 32, 1).bitwise_right_shift(
            torch.arange(64)).bitwise_and(1).view(-1, 2048).float()
        # step() has no null move, so finished games leave the batch rather than
        # being fed a -1.
        alive = legal.sum(-1) > 0
        if not alive.any():
            break
        boards, control, legal = boards[alive], control[alive], legal[alive]
        move = torch.multinomial(legal, 1, generator=g).squeeze(-1)
        boards, control, _, _ = step(boards, control, move)
        mask, _ = movegen(boards, control)
    return torch.cat(all_b), torch.cat(all_c)


def special_positions():
    pairs = [from_fen(fen) for fen in SPECIAL_FENS]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


def stage_snapshots(boards: torch.Tensor, control: torch.Tensor, out: Path):
    """Replicates the body of `first_order_mask`, dumping after each stage."""
    boards = boards.to(torch.int16)
    black_to_move = control < 0
    captured = (boards >> 11).bool()
    mine = ((boards >> 10) & 1) == black_to_move[:, None]
    special = ((boards >> 9) & 1).bool()
    types = (boards >> 6) & 0b111
    squares = boards & SQUARE
    occupancy = ((~captured).long() << squares).sum(-1)
    opp_occupancy = (((~mine) & ~captured).long() << squares).sum(-1)

    stage = luts.get("move_bitsets")[(types * 64 + squares).long()] * mine * ~captured
    stage.numpy().view(np.uint64).tofile(out / "fo_stage1.bin")
    pawn_moves(boards, black_to_move, types, mine, captured, occupancy, opp_occupancy, stage)
    stage.numpy().view(np.uint64).tofile(out / "fo_stage2.bin")
    slider_moves(boards, types, mine, captured, occupancy, squares, stage)
    stage.numpy().view(np.uint64).tofile(out / "fo_stage3.bin")
    castling_moves(boards, black_to_move, special, types, mine, captured, occupancy, stage)
    stage.numpy().view(np.uint64).tofile(out / "fo_stage4.bin")


# The standard perft suite, with the node counts spec §10 gates on. They are
# published values, not something this repository computed, which is what makes
# perft an oracle rather than a regression test. Dumped from `from_fen` so the
# CUDA side needs no FEN parser of its own.
PERFT_CASES = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     [20, 400, 8902, 197281, 4865609, 119060324]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862, 4085603]),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238, 674624]),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467, 422333]),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379, 2103487]),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890, 3894594]),
]


def dump_perft(out: Path) -> None:
    """The six perft start positions and their expected counts."""
    pairs = [from_fen(fen) for _, fen, _ in PERFT_CASES]
    torch.cat([p[0] for p in pairs]).numpy().astype(np.uint16).tofile(out / "perft_boards.bin")
    torch.cat([p[1] for p in pairs]).numpy().tofile(out / "perft_control.bin")
    lines = [f"{name} {len(counts)} " + " ".join(str(c) for c in counts)
             for name, _, counts in PERFT_CASES]
    (out / "perft_cases.txt").write_text("\n".join(lines) + "\n")


def step_cases(boards: torch.Tensor, control: torch.Tensor, mask: torch.Tensor):
    """Enumerate the (position, move, promo) triples the `step` kernel is tested on.

    Deterministic order, which is what lets the CUDA side index the dump without
    carrying a key: position, then slot, then target square, then promotion type,
    and the null moves last.
    """
    b_idx, p_idx, s_idx = bitset_to_bool(mask).nonzero(as_tuple=True)
    move = p_idx * 64 + s_idx

    # A promoting move is one mask bit and four tree edges, so it is expanded
    # here; every other move carries the queen default the reference uses.
    types = (boards[b_idx, p_idx] >> 6) & 0b111
    is_promo = (types == PAWN) & (s_idx // 8 == torch.where(control[b_idx] > 0, 7, 0))
    reps = torch.where(is_promo, 4, 1)
    starts = torch.cumsum(reps, 0) - reps
    promo = torch.full((int(reps.sum()),), 3, dtype=torch.uint8)
    for k in range(4):
        promo[starts[is_promo] + k] = k

    n = boards.shape[0]
    index = torch.cat([b_idx.repeat_interleave(reps), torch.arange(n)])
    move = torch.cat([move.repeat_interleave(reps), torch.full((n,), -1)])
    promo = torch.cat([promo, torch.full((n,), 3, dtype=torch.uint8)])
    return index.to(torch.int32), move.to(torch.int16), promo


def dump_step(boards: torch.Tensor, control: torch.Tensor, hashes: torch.Tensor,
              index: torch.Tensor, move: torch.Tensor, promo: torch.Tensor, out: Path,
              chunk: int = 50_000) -> None:
    """Run the reference `step` over the cases and write the results.

    The hash is tracked, so this exercises the whole of spec §4.2 rather than the
    mutation alone. Chunked because `step` holds a dozen [M,32] int32
    intermediates and M is upwards of 300k.
    """
    boards_out, control_out, hash_out, irrev_out = [], [], [], []
    for i in range(0, len(move), chunk):
        rows = index[i:i + chunk].long()
        b, c, h, irrev = step(boards[rows], control[rows], move[i:i + chunk].long(),
                              promo=promo[i:i + chunk], hash=hashes[rows])
        boards_out.append(b)
        control_out.append(c)
        hash_out.append(h)
        irrev_out.append(irrev)
    index.numpy().tofile(out / "step_index.bin")
    move.numpy().tofile(out / "step_move.bin")
    promo.numpy().tofile(out / "step_promo.bin")
    torch.cat(boards_out).numpy().astype(np.uint16).tofile(out / "step_boards.bin")
    torch.cat(control_out).numpy().tofile(out / "step_control.bin")
    torch.cat(hash_out).numpy().view(np.uint64).tofile(out / "step_hash.bin")
    torch.cat(irrev_out).numpy().astype(np.uint8).tofile(out / "step_irrev.bin")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "cuda_testset")
    p.add_argument("--n", type=int, default=10_000, help="target number of positions")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sb, sc = special_positions()
    pb, pc = snapshot_playouts()
    boards = torch.cat([sb, pb]).to(torch.int16)
    control = torch.cat([sc, pc]).to(torch.int16)

    # A finished game freezes, so the playouts duplicate positions.
    flat = np.concatenate([boards.numpy().view(np.uint8).reshape(len(boards), -1),
                           control.numpy().view(np.uint8).reshape(len(control), -1)], axis=1)
    _, idx = np.unique(flat, axis=0, return_index=True)
    idx = np.sort(idx)
    special_idx, playout_idx = idx[idx < len(sb)], idx[idx >= len(sb)]
    rng = np.random.default_rng(SEED)
    if len(special_idx) + len(playout_idx) > args.n:
        keep = rng.choice(len(playout_idx), args.n - len(special_idx), replace=False)
        playout_idx = np.sort(playout_idx[keep])
    idx = torch.from_numpy(np.concatenate([special_idx, playout_idx]))
    boards, control = boards[idx], control[idx]

    mask, in_check = movegen(boards, control)
    fo_mask, _ = first_order_mask(boards, control < 0)
    # The attacker is the side that is not to move, which is the frame
    # `_king_is_attacked` builds its map in.
    fo_control, _ = first_order_mask(boards, ~(control < 0), control_mode=True)
    hashes = hash_position(boards, control)
    # No per-game history here, so threefold cannot fire: a static test set holds
    # positions, and repetition is a property of the game that reached one.
    code, result = terminal(mask, in_check, control, boards)

    n = len(boards)
    boards.numpy().astype(np.uint16).tofile(out / "boards.bin")
    control.numpy().tofile(out / "control.bin")
    mask.numpy().view(np.uint64).tofile(out / "masks.bin")
    fo_mask.numpy().view(np.uint64).tofile(out / "fo_masks.bin")
    fo_control.numpy().view(np.uint64).tofile(out / "fo_control.bin")
    in_check.numpy().astype(np.uint8).tofile(out / "in_check.bin")
    hashes.numpy().view(np.uint64).tofile(out / "hashes.bin")
    code.numpy().tofile(out / "terminal_code.bin")
    result.numpy().tofile(out / "terminal_result.bin")
    stage_snapshots(boards, control, out)
    luts.dump(out)

    # The two hash inputs, so a wrong hash localises to one of them.
    rights = castling_rights(boards)
    packed = (rights.to(torch.int64) << torch.arange(4)).sum(-1).to(torch.uint8)
    packed.numpy().tofile(out / "castle_rights.bin")
    legal_ep_file(boards, control).to(torch.int8).numpy().tofile(out / "ep_file.bin")

    dump_perft(out)
    index, move, promo = step_cases(boards, control, mask)
    dump_step(boards, control, hashes, index, move, promo, out)
    (out / "meta.txt").write_text(f"{n}\n{len(move)}\n")

    n_moves = (mask != 0).sum().item()
    popcount = sum(int(bin(int(w) & 0xFFFFFFFFFFFFFFFF).count("1")) for w in mask.flatten())
    dead = ((boards >> 11) & 1).sum(-1).float().mean().item()
    is_pawn = ((boards >> 6) & 0b111) == 0
    ep = (((boards >> 9) & 1).bool() & is_pawn & ~((boards >> 11) & 1).bool()).any(-1).sum().item()
    print(f"N = {n} positions -> {out}")
    print(f"  legal moves: {popcount} total, {popcount / n:.1f} per position")
    print(f"  positions with no legal move (mate or stalemate): "
          f"{(mask == 0).all(-1).sum().item()}")
    print(f"  positions with a live en passant flag: {ep}")
    print(f"  captured slots per position: {dead:.1f}")
    print(f"  slots with at least one move: {n_moves}")
    print(f"M = {len(move)} step cases")
    print(f"  promotions, all four choices each: {(promo != 3).sum().item() // 3 * 4}")
    print(f"  null moves: {(move < 0).sum().item()}")
    print(f"  positions with a legal en passant: "
          f"{(legal_ep_file(boards, control) >= 0).sum().item()}")
    print(f"  castling rights present, by KQkq: {rights.sum(0).tolist()}")


if __name__ == "__main__":
    main()
