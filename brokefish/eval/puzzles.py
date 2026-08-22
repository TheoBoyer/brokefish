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
    """A puzzle is a **line**, not a move.

    ⚠️ Until 2026-08-02 this held only the first solver move, and everything scored
    against it measured one move of a sequence averaging **2.32**. That is not a
    uniform loss: solver moves per puzzle rise from **1.47** in the 400-599 band to
    **3.66** at 2600-2799, so scoring move one discarded 32 % of an easy puzzle's
    difficulty and 73 % of a hard one's — which is precisely why the solve rate
    appeared to *rise* with rating. The whole line is kept now.

    ``step_*`` are the per-solver-move tensors, padded to the longest line: the
    position the solver faces at its `j`-th turn (having played every earlier move
    correctly), and the label it must produce. Storing the positions rather than
    replaying them keeps the scorer a loop over `j` with no game logic in it.
    """

    boards: torch.Tensor       # [N, 32] int16, after the set-up move
    control: torch.Tensor      # [N]     int16
    answer: torch.Tensor       # [N]     int16, the solution's *first* move
    rating: torch.Tensor       # [N]     int32
    deviation: torch.Tensor    # [N]     int32
    puzzle_id: List[str]
    step_boards: Optional[torch.Tensor] = None    # [N, S, 32] int16
    step_control: Optional[torch.Tensor] = None   # [N, S]     int16
    step_answer: Optional[torch.Tensor] = None    # [N, S]     int16
    step_len: Optional[torch.Tensor] = None       # [N]        int32, solver moves

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    @property
    def has_lines(self) -> bool:
        return self.step_len is not None


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
    lines: List[List[str]] = []
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
            lines.append(moves[1:])
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
    step = _walk_lines(boards, control, lines, device)

    return PuzzleSet(boards=boards, control=control, answer=answer,
                     rating=torch.tensor(ratings, dtype=torch.int32, device=device),
                     deviation=torch.tensor(devs, dtype=torch.int32, device=device),
                     puzzle_id=ids, **step)


def _walk_lines(boards: torch.Tensor, control: torch.Tensor,
                lines: List[List[str]], device) -> dict:
    """Play each solution line forward, recording what the solver faces each turn.

    ⚠️ **Labels are position-dependent**, because UCI names a square and the record
    names a *slot* — so the line cannot be decoded in one pass and has to be walked.
    The walk advances only the rows that still have moves left, which is what keeps a
    ten-move puzzle from corrupting a one-move one sharing the batch.

    The line alternates solver, opponent, solver, ... so solver turns are the even
    indices and the odd ones are replayed as given (teacher forcing: the solver is
    graded on each move having played every earlier one correctly, which is the only
    way a later move in the line is a well-posed question at all).
    """
    n = len(lines)
    n_solver = [(len(l) + 1) // 2 for l in lines]
    S = max(n_solver) if n else 0
    step_boards = torch.zeros((n, S, 32), dtype=torch.int16, device=device)
    step_control = torch.zeros((n, S), dtype=torch.int16, device=device)
    step_answer = torch.zeros((n, S), dtype=torch.int16, device=device)

    b, c = boards.clone(), control.clone()
    longest = max((len(l) for l in lines), default=0)
    for j in range(longest):
        idx = torch.tensor([i for i, l in enumerate(lines) if j < len(l)],
                           dtype=torch.long, device=device)
        if idx.numel() == 0:
            break
        sub_b, sub_c = b[idx], c[idx]
        lab = _labels_from_uci(sub_b, sub_c, [lines[i][j] for i in idx.tolist()])
        if j % 2 == 0:                      # a solver turn: record the question
            step_boards[idx, j // 2] = sub_b
            step_control[idx, j // 2] = sub_c
            step_answer[idx, j // 2] = lab
        nb, nc, _m, _ck = env.play(sub_b, sub_c, (lab & 0x7FF).long(),
                                   promo=(lab.long() >> 11) & 0b11)
        b[idx], c[idx] = nb, nc

    return {"step_boards": step_boards, "step_control": step_control,
            "step_answer": step_answer,
            "step_len": torch.tensor(n_solver, dtype=torch.int32, device=device)}


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


@torch.no_grad()
def score_puzzles_policy(puzzles: PuzzleSet, net, impl: Optional[str] = None,
                         batch: int = 2048, bin_width: int = 200,
                         ks: Tuple[int, ...] = (1, 3, 5),
                         device: str = "cuda") -> dict:
    """The same curve asked of the raw policy, with **no search at all**.

    One forward pass per position instead of `n` simulations, which turns the puzzle
    curve from the diagnostic that is allowed to be slow into one cheap enough to run
    beside a training run. `score_suite_policy` is the same idea on the rule suites,
    and the pairing is the same diagnostic: a policy that is right and a search that
    is wrong is a different failure from both being wrong.

    ⚠️ **This is what the search cannot do for it.** `state.md` records that a
    random-init network scores 0.147 *with* 800 simulations against a 0.051 uniform
    null, and that essentially all of it is mate-in-1 found by terminal nodes rather
    than by the network — MCTS proves mates with no evaluation function. Removing the
    search removes that free credit, so this number starts near the uniform baseline
    and every point above it is the policy's own.

    ⚠️ **Evaluation only** (`evaluation.md` §2). The puzzle set is human games with
    engine-verified solutions and is inside the tabula rasa boundary *only* because
    §2 puts evaluation outside it. Nothing that reads this may write to training —
    which is why the intended caller is an out-of-process watcher, not the loop.

    `rep` is zero everywhere: a puzzle is a position without a history, exactly as
    `score_puzzles` presents it to `Search.reset`.
    """
    from .suites import _evaluator, _policy_topk

    evaluate = _evaluator(net, impl)
    total = len(puzzles)
    kmax = max(ks)
    hit = {k: torch.zeros(total, dtype=torch.bool, device=device) for k in ks}
    for lo in range(0, total, batch):
        hi = min(lo + batch, total)
        item = PuzzleSet(boards=puzzles.boards[lo:hi].to(device),
                         control=puzzles.control[lo:hi].to(device),
                         answer=puzzles.answer[lo:hi], rating=puzzles.rating[lo:hi],
                         deviation=puzzles.deviation[lo:hi],
                         puzzle_id=puzzles.puzzle_id[lo:hi])
        rep = torch.zeros(hi - lo, dtype=torch.uint8, device=device)
        policy, promo, _v = evaluate(item.boards, item.control, rep)
        top = _policy_topk(item, policy, promo, k=kmax)          # [b, kmax]
        want = puzzles.answer[lo:hi].to(device)[:, None]
        for k in ks:
            hit[k][lo:hi] = (top[:, :k] == want).any(-1)
    correct = hit[1]

    bins = []
    r = puzzles.rating
    on_cpu = {k: hit[k].cpu() for k in ks}
    lo_edge = int(r.min()) // bin_width * bin_width
    hi_edge = int(r.max()) // bin_width * bin_width + bin_width
    for edge in range(lo_edge, hi_edge, bin_width):
        # `rating` may live on either device; the mask and the tensor it indexes
        # have to agree, and the bin loop is host-side arithmetic anyway.
        sel = ((r >= edge) & (r < edge + bin_width)).cpu()
        m = int(sel.sum())
        if m == 0:
            continue
        n1 = int(on_cpu[1][sel].sum()) if 1 in ks else 0
        row = {"rating_lo": edge, "rating_hi": edge + bin_width, "n": m,
               "solved": n1, "rate": n1 / m, "ci95": list(_wilson(n1, m))}
        # Per-band pass@k as well as the aggregate: the bands are where the shape
        # lives, and a policy that improves only on the easy end is a different
        # thing from one that improves everywhere.
        for k in ks:
            j = int(on_cpu[k][sel].sum())
            row[f"pass@{k}"] = j / m
            row[f"pass@{k}_ci95"] = list(_wilson(j, m))
        bins.append(row)

    n1 = int(correct.sum())
    out = {"n_puzzles": total, "n_sims": 0,
           "solve_rate": n1 / total if total else 0.0,
           "ci95": list(_wilson(n1, total)), "bins": bins}
    # pass@k: is the solution anywhere in the policy's top k? `solve_rate` stays
    # pass@1 so the field means what it has always meant.
    for k in ks:
        m = int(hit[k].sum())
        out[f"pass@{k}"] = m / total if total else 0.0
        out[f"pass@{k}_ci95"] = list(_wilson(m, total))
    return out


def _move_kinds(boards: torch.Tensor, control: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
    """`[N]` category per move: 0 quiet, 1 capture, 2 check, 3 mate.

    Mutually exclusive and ordered by force, so a mating capture counts once, as a
    mate. This is the decomposition that says *what kind* of move a network can and
    cannot find — the solve rate says only that something is missing.

    ⚠️ A capture is decided by the destination square holding an **enemy** piece
    before the move. Getting that comparison backwards silently reports 0 % captures
    everywhere, which is what it did on the first attempt (2026-08-02).
    """
    from .probe import MOVE_MASK, PROMO_SHIFT

    lab = labels.to(torch.int64)
    mv, pm = lab & MOVE_MASK, (lab >> PROMO_SHIFT) & 0b11
    dst = mv % 64
    captured, colour, _sp, _ty, square = env.decode(boards)
    alive = captured == 0
    mover_is_white = control > 0
    enemy_on_dst = torch.zeros(boards.shape[0], dtype=torch.bool, device=boards.device)
    for slot in range(32):
        # colour bit 1 is black, so the enemy is `colour == 1` exactly when the mover
        # is white -- hence the comparison against `mover_is_white`, not its negation.
        enemy_on_dst |= (alive[:, slot] & ((colour[:, slot] == 1) == mover_is_white)
                         & (square[:, slot] == dst))

    nb, nc, nm, in_check = env.play(boards.clone(), control.clone(), mv, promo=pm)
    code, _ = env.terminal(nm, in_check, nc, nb)
    kind = torch.zeros(boards.shape[0], dtype=torch.int8, device=boards.device)
    kind = torch.where(enemy_on_dst, torch.ones_like(kind), kind)
    kind = torch.where(in_check, torch.full_like(kind, 2), kind)
    kind = torch.where(code == env.CHECKMATE, torch.full_like(kind, 3), kind)
    return kind


KIND_NAMES = ("quiet", "capture", "check", "mate")


@torch.no_grad()
def score_puzzles_line(puzzles: PuzzleSet, net, impl: Optional[str] = None,
                       batch: int = 2048, bin_width: int = 200,
                       ks: Tuple[int, ...] = (1, 3, 5),
                       device: str = "cuda") -> dict:
    """The whole line, policy only. Two metrics, and they answer different questions.

    - **``solve_rate``** — the net's top move is correct at **every** solver turn.
      This is what "solved the puzzle" means on Lichess, and it is the only reading
      under which the puzzle's rating describes the task we set.
    - **``move_pass@k``** — the fraction of *solver turns*, pooled over all puzzles,
      whose answer is in the net's top `k`. Each turn is asked with every earlier
      move played correctly (teacher forcing), so a later move is a well-posed
      question rather than a consequence of an earlier miss.

    ⚠️ The two diverge hard and the gap is the point. A line of `L` moves needs `L`
    consecutive hits to solve, so at a per-move rate `p` the solve rate is near
    `p**L`, and `L` runs 1.47 in the 400-599 band to 3.66 at 2600-2799. Reporting
    only one of them hides either the compounding or the per-move skill.

    ⚠️ **Evaluation only** (`evaluation.md` §2), as with every puzzle metric here.
    """
    from .suites import _evaluator, _policy_topk

    if not puzzles.has_lines:
        raise ValueError("this PuzzleSet was built before line support; reload it")
    evaluate = _evaluator(net, impl)
    n, S = len(puzzles), puzzles.step_answer.shape[1]
    kmax = max(ks)
    lens = puzzles.step_len.to(device)
    # [N, S] per-turn hits, and [N] whether every turn of the line was hit at k=1.
    turn_hit = {k: torch.zeros((n, S), dtype=torch.bool, device=device) for k in ks}
    sol_kind = torch.zeros((n, S), dtype=torch.int8, device=device)
    net_kind = torch.zeros((n, S), dtype=torch.int8, device=device)

    for j in range(S):
        rows = (lens > j).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            break
        for lo in range(0, rows.numel(), batch):
            r = rows[lo:lo + batch]
            item = PuzzleSet(boards=puzzles.step_boards[r, j].to(device),
                             control=puzzles.step_control[r, j].to(device),
                             answer=puzzles.step_answer[r, j], rating=puzzles.rating[r],
                             deviation=puzzles.deviation[r], puzzle_id=[])
            rep = torch.zeros(r.numel(), dtype=torch.uint8, device=device)
            policy, promo, _v = evaluate(item.boards, item.control, rep)
            top = _policy_topk(item, policy, promo, k=kmax)
            want = puzzles.step_answer[r, j].to(device)[:, None]
            for k in ks:
                turn_hit[k][r, j] = (top[:, :k] == want).any(-1)
            # What kind of move was asked for, and what kind the net actually chose.
            sol_kind[r, j] = _move_kinds(item.boards, item.control, want[:, 0])
            net_kind[r, j] = _move_kinds(item.boards, item.control, top[:, 0])

    valid = torch.arange(S, device=device)[None, :] < lens[:, None]
    solved = (turn_hit[1] | ~valid).all(-1)
    n_turns = int(valid.sum())

    bins = []
    r_cpu = puzzles.rating.cpu()
    solved_cpu = solved.cpu()
    valid_cpu = valid.cpu()
    hit_cpu = {k: turn_hit[k].cpu() for k in ks}
    lo_edge = int(r_cpu.min()) // bin_width * bin_width
    hi_edge = int(r_cpu.max()) // bin_width * bin_width + bin_width
    for edge in range(lo_edge, hi_edge, bin_width):
        sel = ((r_cpu >= edge) & (r_cpu < edge + bin_width))
        m = int(sel.sum())
        if m == 0:
            continue
        t = int(valid_cpu[sel].sum())
        sv = int(solved_cpu[sel].sum())
        row = {"rating_lo": edge, "rating_hi": edge + bin_width, "n": m,
               "n_turns": t, "mean_line": t / m,
               "solved": sv, "solve_rate": sv / m, "ci95": list(_wilson(sv, m))}
        for k in ks:
            j = int((hit_cpu[k][sel] & valid_cpu[sel]).sum())
            row[f"move_pass@{k}"] = j / t if t else 0.0
            row[f"move_pass@{k}_ci95"] = list(_wilson(j, t))
        bins.append(row)

    sv = int(solved.sum())
    out = {"n_puzzles": n, "n_turns": n_turns, "n_sims": 0,
           "mean_line": n_turns / n if n else 0.0,
           "solve_rate": sv / n if n else 0.0, "ci95": list(_wilson(sv, n)),
           "bins": bins}
    for k in ks:
        j = int((turn_hit[k] & valid).sum())
        out[f"move_pass@{k}"] = j / n_turns if n_turns else 0.0
        out[f"move_pass@{k}_ci95"] = list(_wilson(j, n_turns))

    # ⚠️ The decomposition that explains the aggregate. A network that has learned
    # "take material" and nothing else scores respectably overall while missing every
    # mate, and only this split says so. `asked` is a property of the puzzle set and
    # is constant across checkpoints; `played` and `pass@1` are the network's.
    kinds = {}
    for idx, name in enumerate(KIND_NAMES):
        want_sel = (sol_kind == idx) & valid
        m = int(want_sel.sum())
        kinds[name] = {
            "asked": m / n_turns if n_turns else 0.0,
            "played": float(((net_kind == idx) & valid).sum()) / n_turns if n_turns else 0.0,
            "pass@1": int((turn_hit[1] & want_sel).sum()) / m if m else 0.0,
            "n_turns": m,
        }
        for k in ks:
            kinds[name][f"pass@{k}"] = (int((turn_hit[k] & want_sel).sum()) / m
                                        if m else 0.0)
    out["kinds"] = kinds
    return out


# ⚠️ `no_grad` on the function, not on the caller. `CLAUDE.md`'s fourth trap: an
# evaluation that forgets it builds an autograd graph over the fp32 master weights and
# OOMs the card -- here after ~200 puzzles, inside the FFN, asking for 128 MiB while
# holding 7.5 GiB. `score_puzzles_line` is safe only because `PuzzleProbe.run` happens to
# be decorated; these two carry their own guarantee.
@torch.no_grad()
def value_pick(evaluate, boards: torch.Tensor, control: torch.Tensor,
               chunk: int = 8192):
    """``argmax_m -v(child(m))`` per position: the value head used as logits over moves.

    Every legal move is played, every resulting position is evaluated, and the move
    chosen is the one that leaves the **opponent** worst off. That is the value head
    graded as a move-chooser, which is what a search at ``n >= 64`` actually uses it for.

    ⚠️ **The value head answers, and nothing else does.** No terminal collapse, no
    rules override, no search. An earlier version pinned terminal children to what the
    rules say — mate ``-1``, draw ``0`` — on the argument that `search.md` §6.3 discards
    those evaluations in play too. That is an argument about fidelity to the engine, and
    this is not a measurement of the engine. Measured 2026-08-19 over 20 000 puzzles, the
    pin was worth **+0.079 to +0.108 pass@1**, and it was worth *more the worse the head
    was* — so it did not merely inflate the score, it compressed the differences between
    checkpoints, which is the one thing a diagnostic must not do.

    The engine is still used to enumerate the legal moves and to play them, because there
    is no way to score a move without the position it leads to. It is not consulted about
    the *value* of anything.
    """
    # ⚠️ **`cuda_impl`, not this module's `env`.** `puzzles.py` binds `env` to
    # `torch_impl`, which is the *reference* engine: correct, and it materialises
    # intermediates per board. One puzzle expands to ~28 children, so a batch of 200
    # positions is 5 600 boards through movegen and the reference path OOMs an 8 GiB
    # card on that. The fused engine is the one that scales here.
    from ..env import cuda_impl as cenv
    from .probe import enumerate_moves

    game, label = enumerate_moves(boards, control)
    kids_b, kids_c, _, _ = cenv.play(boards[game], control[game],
                                     label & 0x7FF, promo=(label >> 11) & 0b11)

    v = torch.empty(kids_b.shape[0], dtype=torch.float32, device=boards.device)
    rep = torch.zeros(chunk, dtype=torch.uint8, device=boards.device)
    for lo in range(0, kids_b.shape[0], chunk):
        hi = min(lo + chunk, kids_b.shape[0])
        v[lo:hi] = evaluate(kids_b[lo:hi], kids_c[lo:hi],
                            rep[:hi - lo])[2].float().reshape(-1)
    return _argmax_move(-v, label, game, boards.shape[0])


def _argmax_move(score: torch.Tensor, label: torch.Tensor, game: torch.Tensor,
                 n: int):
    """Best-scoring move per position, ties broken by the lowest label.

    ⚠️ ``index_copy``/indexed assignment with **duplicate indices** has no defined
    ordering in torch, so "sort then let the last write win" is a race. Two
    ``scatter_reduce`` passes instead -- the best score, then the lowest label achieving
    it -- which is the lowest-index tie-break of `search.md` §6.4.
    """
    NONE = torch.iinfo(torch.int64).max
    best_score = torch.full((n,), -1e9, device=score.device).scatter_reduce(
        0, game, score, reduce="amax", include_self=True)
    cand = torch.where(score >= best_score[game], label, torch.full_like(label, NONE))
    best = torch.full((n,), NONE, dtype=torch.int64,
                      device=score.device).scatter_reduce(
        0, game, cand, reduce="amin", include_self=True)
    return torch.where(best == NONE, torch.full_like(best, -1), best), best_score


@torch.no_grad()
def score_puzzles_value(puzzles: PuzzleSet, net, impl: Optional[str] = None,
                        chunk: int = 8192, batch: int = 4096, bin_width: int = 200,
                        device: str = "cuda") -> dict:
    """The puzzle line scored through the **value** head, as a one-ply search.

    `score_puzzles_line` grades one head and one head only — the policy, at
    ``n_sims = 0``. This grades the other. Games at ``n >= 64`` are decided by the value
    head (Gumbel proposes 16 candidates from the prior and then *ranks them by
    completed-Q*), so a policy-only probe cannot see the thing that decides the game.

    ⚠️ **This exists because its absence hid a regression for three runs.** Measured
    2026-08-17: `t12h-muon9-int8` beat `t12h-gumbel` by +0.061 policy pass@1 and **lost to
    it 62-45-93 at n = 128**; on this metric it scores 0.414 against 0.486, which is the
    right way round. Neither run logged it, so nothing saw it until both had finished.

    ⚠️ ``impl`` selects the fused encoder. The packed weights are built inside
    `make_evaluator` on every call, so they cannot go stale the way `CLAUDE.md`'s third
    trap describes — but a fused score is **not comparable** with an unfused one, because
    the packing is fp16 (or int8) against fp32 master weights. Do not mix them in one
    series without measuring the offset.

    ⚠️ **Evaluation only** (`evaluation.md` §2).
    """
    from .suites import _evaluator

    if not puzzles.has_lines:
        raise ValueError("this PuzzleSet was built before line support; reload it")
    evaluate = _evaluator(net, impl)
    n, S = len(puzzles), puzzles.step_answer.shape[1]
    lens = puzzles.step_len.to(device)
    hit = torch.zeros((n, S), dtype=torch.bool, device=device)

    for j in range(S):
        rows = (lens > j).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            break
        for lo in range(0, rows.numel(), batch):
            r = rows[lo:lo + batch]
            best, _ = value_pick(evaluate, puzzles.step_boards[r, j].to(device),
                                 puzzles.step_control[r, j].to(device), chunk=chunk)
            hit[r, j] = best == puzzles.step_answer[r, j].to(device).to(torch.int64)

    valid = torch.arange(S, device=device)[None, :] < lens[:, None]
    solved = (hit | ~valid).all(-1)
    n_turns = int(valid.sum())

    bins = []
    r_cpu, hit_cpu = puzzles.rating.cpu(), hit.cpu()
    valid_cpu, solved_cpu = valid.cpu(), solved.cpu()
    lo_edge = int(r_cpu.min()) // bin_width * bin_width
    hi_edge = int(r_cpu.max()) // bin_width * bin_width + bin_width
    for edge in range(lo_edge, hi_edge, bin_width):
        sel = (r_cpu >= edge) & (r_cpu < edge + bin_width)
        m = int(sel.sum())
        if m == 0:
            continue
        t = int(valid_cpu[sel].sum())
        sv = int(solved_cpu[sel].sum())
        j = int((hit_cpu[sel] & valid_cpu[sel]).sum())
        bins.append({"rating_lo": edge, "rating_hi": edge + bin_width, "n": m,
                     "n_turns": t, "solved": sv, "solve_rate": sv / m,
                     "ci95": list(_wilson(sv, m)),
                     "value_pass@1": j / t if t else 0.0,
                     "value_pass@1_ci95": list(_wilson(j, t))})

    sv, j = int(solved.sum()), int((hit & valid).sum())
    return {"n_puzzles": n, "n_turns": n_turns,
            "value_pass@1": j / n_turns if n_turns else 0.0,
            "value_pass@1_ci95": list(_wilson(j, n_turns)),
            "solve_rate": sv / n if n else 0.0, "ci95": list(_wilson(sv, n)),
            "bins": bins}


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

    from brokefish.nn.model import BrokefishNet, net_for_state

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
        state = torch.load(args.checkpoint, map_location=args.device)
        net = net_for_state(state).to(args.device)
        net.load_state_dict(state)
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
