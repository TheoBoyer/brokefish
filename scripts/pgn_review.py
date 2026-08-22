"""Read a match PGN and say what actually happened in each game.

    python -m scripts.pgn_review --pgn logs/gate2-h2h-muon9int8.pgn --games 1,2,3

⚠️ **No engine is consulted and that is deliberate.** Every judgement here comes from
the rules alone -- `python-chess` legality, mate detection and piece values -- so the
review imports no chess opinion. Three things it can say without one:

  * **missed mate**: a mate in one was legal and something else was played;
  * **allowed mate**: the move played handed the opponent a mate in one;
  * **material swing**: a ply where the mover's material drops, i.e. something was
    left hanging or a capture was declined for nothing.

⚠️ Evaluation output. `evaluation.md`: it may never flow backwards into training or
checkpoint selection.
"""
from __future__ import annotations

import argparse
import re

import chess
import chess.pgn

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(b: chess.Board) -> int:
    """White minus black, in pawns."""
    return sum(VAL[p.piece_type] * (1 if p.color == chess.WHITE else -1)
               for p in b.piece_map().values())


def mate_in_one(b: chess.Board):
    for m in b.legal_moves:
        b.push(m)
        if b.is_checkmate():
            b.pop()
            return m
        b.pop()
    return None


def review(game, idx: int, swing: int):
    hs = game.headers
    w, bl = hs.get("White", "?"), hs.get("Black", "?")
    print(f"\n=== game {idx}: {hs.get('Result')}  {hs.get('PlyCount')} plies  "
          f"{hs.get('Termination')}")
    print(f"    White {w}\n    Black {bl}")
    print(f"    from  {hs.get('FEN')}")

    b = game.board()
    prev_mat = material(b)
    notes = []
    line = []
    for ply, mv in enumerate(game.mainline_moves(), 1):
        mover = "White" if b.turn == chess.WHITE else "Black"
        name = w if b.turn == chess.WHITE else bl
        who = name.split()[0]
        best = mate_in_one(b)
        san = b.san(mv)
        line.append(san)
        if best is not None and best != mv:
            notes.append(f"  ply {ply:3d} {mover:5s} ({who}) MISSED MATE IN 1: "
                         f"{b.san(best)} was legal, played {san}")
        b.push(mv)
        if not b.is_game_over():
            reply = mate_in_one(b)
            if reply is not None:
                notes.append(f"  ply {ply:3d} {mover:5s} ({who}) ALLOWED MATE IN 1: "
                             f"after {san}, {b.san(reply)} mates")
        mat = material(b)
        # A swing against the side that just moved: it gave material away.
        loss = (prev_mat - mat) if mover == "White" else (mat - prev_mat)
        if loss >= swing:
            notes.append(f"  ply {ply:3d} {mover:5s} ({who}) dropped {loss} pawns "
                         f"of material with {san}")
        prev_mat = mat

    print("    " + " ".join(line[:40]) + (" ..." if len(line) > 40 else ""))
    if b.is_checkmate():
        loser = "White" if b.turn == chess.WHITE else "Black"
        print(f"    -> CHECKMATE, {loser} is mated by {line[-1]}")
    elif b.is_stalemate():
        print("    -> STALEMATE")
    elif b.is_insufficient_material():
        print("    -> insufficient material")
    elif b.can_claim_threefold_repetition():
        print("    -> threefold repetition")
    elif b.can_claim_fifty_moves():
        print("    -> fifty-move rule")
    print(f"    final material (White - Black): {material(b):+d}")
    for n in notes:
        print(n)
    if not notes:
        print("    (no missed or allowed mate in one, no swing over the threshold)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", required=True)
    p.add_argument("--games", default="1,2,3", help="1-based indices, comma separated")
    p.add_argument("--swing", type=int, default=3, help="material-drop threshold, pawns")
    a = p.parse_args()

    want = {int(x) for x in a.games.split(",") if x.strip()}
    fh = open(a.pgn)
    i = 0
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        i += 1
        if i in want:
            review(g, i, a.swing)


if __name__ == "__main__":
    main()
