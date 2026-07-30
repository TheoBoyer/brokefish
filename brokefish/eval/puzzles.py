"""Layer 3's external half: the Lichess puzzle database, `evals.md` §8.2.

**Lichess's set, not chess.com's.** It is CC0, it is 6 014 381 puzzles in the June
2026 export, and — the actual reason — every puzzle carries a **Glicko-2 rating
and a rating deviation**, obtained by treating each solve attempt as a game
between the player and the puzzle. That gives a calibrated difficulty axis, so
the output of this layer is a solve-rate-versus-difficulty *curve* rather than
one opaque percentage. A single number would move when the puzzle mix moved and
say nothing about where the net's ceiling is.

⚠️ **Puzzle accuracy is not strength.** It is tactics-heavy and single-best-move,
and says nothing about positional play or endgame technique. Layer 3 answers
"why is this net bad"; it never answers "how strong is this net", and per §2 it
never selects a checkpoint. That prohibition is the one most likely to be
violated by accident here, because "keep the checkpoint with the best puzzle
score" looks exactly like good practice.

The file is not in the repository — it is a 300 MB zstd CSV and it is not ours.
Fetch it with:

    curl -L https://database.lichess.org/lichess_db_puzzle.csv.zst \\
        -o data/lichess_db_puzzle.csv.zst
    zstd -d data/lichess_db_puzzle.csv.zst

Format, one header line then:

    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

⚠️ **The FEN is the position before the opponent's blunder-punishing reply**, not
the position to solve. Lichess's convention is that the *first* move in `Moves`
is played for you and the solution starts at the second. Scoring the first move
is scoring the opponent, which passes tests and measures nothing.

⚠️ **This is the most expensive thing in `brokefish/eval/`, by a lot.** One puzzle
costs a whole search, so `puzzles x n` evaluations is the bill: 2 000 puzzles at
`n = 800` is 1.6M evals and about half a minute, and 20 000 is 16.4M and over five
minutes of GPU at 100 %. On a laptop whose GPU also drives the display that is not
a background job. Size it on the command line, deliberately.

⚠️ **A puzzle is a line, not a move.** We score the first solution move only, and
say so: multi-move scoring needs the opponent's replies played from `Moves`,
which is another layer of harness and is not what the curve is for. A first-move
score is an upper bound on line accuracy.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from brokefish.env import torch_impl as env
from .suites import _wilson

DEFAULT_PUZZLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "lichess_db_puzzle.csv")


@dataclass
class PuzzleSet:
    boards: torch.Tensor       # [N, 32] int16, after the set-up move
    control: torch.Tensor      # [N]     int16
    answer: torch.Tensor       # [N]     int16, the solution's edge label
    rating: torch.Tensor       # [N]     int32
    deviation: torch.Tensor    # [N]     int32
    puzzle_id: List[str]

    def __len__(self) -> int:
        return int(self.boards.shape[0])


def load_puzzles(path: Optional[str] = None, limit: Optional[int] = 20_000,
                 max_deviation: int = 100, min_rating: int = 400,
                 max_rating: int = 3200, seed: int = 0, device: str = "cuda",
                 ) -> PuzzleSet:
    """Read the CSV, play the set-up move, and keep the solution's first move.

    ``max_deviation`` is the filter that matters: the tail of rarely-attempted
    puzzles has a deviation of several hundred Elo, and binning by a rating that
    uncertain produces a curve whose x-axis is noise. Lichess's own convention is
    that a deviation under 100 means the rating has converged.

    Reservoir-free subsampling: the file is read in order and every row is
    accepted until ``limit``, so a `limit` well under the file size gives the
    puzzles Lichess happened to write first. That is fine for a curve binned by
    rating and would not be fine for anything that assumed a random sample —
    pass ``limit=None`` for the whole file.
    """
    path = path or DEFAULT_PUZZLE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is not in the repository (300 MB, not ours):\n"
            "  curl -L https://database.lichess.org/lichess_db_puzzle.csv.zst"
            " -o data/lichess_db_puzzle.csv.zst\n"
            "  zstd -d data/lichess_db_puzzle.csv.zst")

    fens, first_moves, answers, ratings, devs, ids = [], [], [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            dev = int(row["RatingDeviation"])
            rating = int(row["Rating"])
            if dev > max_deviation or not (min_rating <= rating <= max_rating):
                continue
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            fens.append(row["FEN"])
            first_moves.append(moves[0])
            answers.append(moves[1])
            ratings.append(rating)
            devs.append(dev)
            ids.append(row["PuzzleId"])
            if limit is not None and len(fens) >= limit:
                break

    boards = torch.cat([env.from_fen(f)[0] for f in fens]).to(device)
    control = torch.cat([env.from_fen(f)[1] for f in fens]).to(device)

    # Lichess's first move is the opponent's; the puzzle starts after it.
    setup = _labels_from_uci(boards, control, first_moves)
    boards, control, _m, _c = env.play(boards, control,
                                       (setup & 0x7FF).long(),
                                       promo=(setup.long() >> 11) & 0b11)
    answer = _labels_from_uci(boards, control, answers)

    return PuzzleSet(boards=boards, control=control, answer=answer,
                     rating=torch.tensor(ratings, dtype=torch.int32, device=device),
                     deviation=torch.tensor(devs, dtype=torch.int32, device=device),
                     puzzle_id=ids)


def _labels_from_uci(boards: torch.Tensor, control: torch.Tensor,
                     uci: List[str]) -> torch.Tensor:
    """UCI strings to the search's edge labels, `slot * 64 + target` plus promo.

    The slot has to be looked up rather than derived: UCI names the *square* a
    piece stands on and the representation is indexed by slot, and the map
    between them is the position. This is the one place in the package that goes
    from an external move notation back into ours.
    """
    files, ranks = "abcdefgh", "12345678"
    promo_char = "nbrq"
    device = boards.device
    captured, _color, _special, _type, square = env.decode(boards)

    src = torch.tensor([files.index(m[0]) + 8 * ranks.index(m[1]) for m in uci],
                       device=device)
    dst = torch.tensor([files.index(m[2]) + 8 * ranks.index(m[3]) for m in uci],
                       device=device)
    promo = torch.tensor([promo_char.index(m[4]) if len(m) > 4 else 0 for m in uci],
                         device=device)

    on_square = (square == src[:, None]) & (captured == 0)
    if not bool(on_square.any(-1).all()):
        bad = (~on_square.any(-1)).nonzero(as_tuple=True)[0][:5].tolist()
        raise ValueError(f"no piece on the source square of moves {bad}")
    slot = on_square.float().argmax(-1)
    return ((slot * 64 + dst) | (promo << 11)).to(torch.int16)


@torch.no_grad()
def score_puzzles(puzzles: PuzzleSet, net, n: int = 800, impl: Optional[str] = None,
                  search_impl: str = "torch", batch: int = 256, bin_width: int = 200,
                  seed: int = 0, device: str = "cuda") -> dict:
    """Solve rate against puzzle rating, in `bin_width`-Elo bins.

    Search configuration is the evaluation one of §3: no Dirichlet, greedy at the
    root. `n` defaults to the self-play budget rather than layer 0's, because
    this is the diagnostic that is allowed to be slow.
    """
    from .runner import make_search

    total = len(puzzles)
    correct = torch.zeros(total, dtype=torch.bool, device=device)
    for lo in range(0, total, batch):
        hi = min(lo + batch, total)
        b = hi - lo
        search = make_search(n, b, net, impl=impl, search_impl=search_impl,
                             seed=seed, device=device)
        search.reset(puzzles.boards[lo:hi], puzzles.control[lo:hi])
        played = search.self_play_move().played.to(torch.int16)
        correct[lo:hi] = played == puzzles.answer[lo:hi]

    bins = []
    r = puzzles.rating
    lo_edge = int(r.min()) // bin_width * bin_width
    hi_edge = int(r.max()) // bin_width * bin_width + bin_width
    for edge in range(lo_edge, hi_edge, bin_width):
        sel = (r >= edge) & (r < edge + bin_width)
        k, m = int(correct[sel].sum()), int(sel.sum())
        if m == 0:
            continue
        ci = _wilson(k, m)
        bins.append({"rating_lo": edge, "rating_hi": edge + bin_width,
                     "n": m, "solved": k, "rate": k / m, "ci95": list(ci)})

    k = int(correct.sum())
    return {"n_puzzles": total, "n_sims": n, "solve_rate": k / total if total else 0.0,
            "ci95": list(_wilson(k, total)), "bins": bins}


def uniform_baseline(puzzles: PuzzleSet) -> float:
    """What a net that picks uniformly among the legal moves would score.

    The null for the whole curve. Without it a solve rate is unreadable: 3 % looks
    catastrophic and is exactly what "no chess knowledge at all" produces, because
    a puzzle position has ~30 legal moves and one of them is the answer.
    """
    mask, _ = env.movegen(puzzles.boards, puzzles.control)
    n_legal = env.bitset_to_bool(mask).reshape(len(puzzles), -1).sum(-1).float()
    return float((1.0 / n_legal.clamp(min=1)).mean())


def main() -> None:
    import argparse
    import json

    import torch

    from brokefish.nn.model import BrokefishNet

    ap = argparse.ArgumentParser(description="the §8.2 solve-rate-versus-rating curve")
    ap.add_argument("checkpoint", nargs="?", help="a torch state_dict; random init if absent")
    ap.add_argument("--puzzles", default=DEFAULT_PUZZLE_PATH)
    # ⚠️ Deliberately small. The 4060 in this machine is also the display
    # adapter, and `limit=20000 --sims 800` is 16.4M evaluations -- five-plus
    # minutes of pinned GPU, which took the desktop down with it on 2026-07-30.
    # The defaults here are a ~30 s job; scale up only on a machine whose GPU is
    # not driving a screen, and say so on the command line rather than in a file.
    ap.add_argument("--limit", type=int, default=2_000)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bin-width", type=int, default=200)
    ap.add_argument("--max-deviation", type=int, default=100)
    ap.add_argument("--impl", default="cuda")
    ap.add_argument("--search-impl", default="cuda")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    net = BrokefishNet().to(args.device)
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    net.eval()

    puzzles = load_puzzles(path=args.puzzles, limit=args.limit,
                           max_deviation=args.max_deviation, device=args.device)
    null = uniform_baseline(puzzles)
    print(f"{len(puzzles)} puzzles, rating {int(puzzles.rating.min())}-"
          f"{int(puzzles.rating.max())}, n={args.sims}", flush=True)
    print(f"uniform-random baseline: {null:.4f}", flush=True)

    out = score_puzzles(puzzles, net, n=args.sims, impl=args.impl,
                        search_impl=args.search_impl, batch=args.batch,
                        bin_width=args.bin_width, device=args.device)
    out["uniform_baseline"] = null

    print(f"\n{'rating':>12}  {'n':>6}  {'solved':>6}  {'rate':>7}  95% CI", flush=True)
    for b in out["bins"]:
        print(f"{b['rating_lo']:>5}-{b['rating_hi']:<6} {b['n']:>6}  {b['solved']:>6}  "
              f"{b['rate']:>7.4f}  [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]", flush=True)
    print(f"\noverall {out['solve_rate']:.4f} "
          f"[{out['ci95'][0]:.4f}, {out['ci95'][1]:.4f}]  "
          f"vs uniform {null:.4f}", flush=True)

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
