"""Where do our hung pieces come from: the budget, the prior, or the value head?

    uv run --no-project --python .venv/bin/python -m scripts.blunder_dissect
    uv run --no-project --python .venv/bin/python -m scripts.blunder_dissect \
        --match 24h=logs/h2h-t24hfull.pgn:runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt \
        --budgets 128,1024 --batch 64

`docs/journal/2026-08-19-the-league-does-not-transfer.md` counted, by hand, that our
networks lose material without compensation at 2.1x AlphaGateau's rate in the two
200-game matches. That count was ad hoc. This script is the committed version of the
detector, and it asks the next question of every blunder: with the same network,
does a bigger search avoid it (BUDGET), is the refutation invisible to the prior
(PRIOR), or does the search see the refutation and the value head not mind (VALUE)?

**The detector, rules only.** Piece values 1/3/3/5/9, `python-chess` for legality.
For every ply where our side moved: material before our move, then our move, then the
reply actually played, then our best recapture -- the capture on the reply's square
that maximises our material, if one exists (`--recapture any` widens that to every
capture). A drop of >= 3 pawns is a blunder. The rate is per 1000 plies of the
*whole* game, both sides' plies. ⚠️ Neither rule reproduces the journal's numbers
exactly; see `RECAPTURE` below for what each gives.

**The dissection, one search per budget.** Each blunder position is searched from its
FEN with the checkpoint that played it, under the match's own protocol
(`scripts/serve_brokefish.py`: `eval_config`, Gumbel m = 16, terminal collapse, no
noise, the fused CUDA encoder in fp16 -- the server never passed `int8`, whatever
the run trained with; `--int8` is here to test the other case). The tree is then
read directly: the root's prior, visits, Q and completed Q for the blunder edge, and
at the child the prior and visits of the *refutation*, which is the opponent reply
the material rule scores best (max loss for us after our best recapture). Every
value and Q in the CSV is in [-1, 1] **from our side's view**; `value_after` is the
raw network value of the position after the blunder, opponent to move, negated.

**The classes.** BUDGET: n = 1024 plays something else (`loss1024` says whether the
something else still hangs material by the rule). OTHER(forced): every legal move
loses >= 3 by the rule, i.e. the material was already lost and the detector's
three-ply window lands one ply late. PRIOR: n = 1024 still plays it and the
refutation's prior at the child is below 1/E or the refutation was never visited.
VALUE: the refutation was visited and the search still chose the move.

⚠️ The search runs **repetition-blind** from a FEN, exactly as the match did.

⚠️ `checkpoints/t12h-gumbel-004009.pt`, the checkpoint that played the 12 h match, no
longer exists on disk. The default falls back to the run's final checkpoint, step
4185, which is a different network by 176 steps; the `repro` column says whether the
n = 128 search still plays the recorded blunder, and the classification is only
about the network that was actually searched.

⚠️ Evaluation output (`evaluation.md`): nothing here may flow back into training or
into checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
THRESHOLD = 3
E_CAP = 96

DEFAULT_MATCHES = [
    "12h=logs/gate2-h2h.pgn:runs/t12h-gumbel/checkpoints/t12h-gumbel.pt",
    "24h=logs/h2h-t24hfull.pgn:runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt",
]


# -- the rule ------------------------------------------------------------- #

def material(b: chess.Board, us: chess.Color) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == us else -1)
               for p in b.piece_map().values())


#: Which captures count as "capturing back" (`--recapture`). ``square``: only a
#: capture on the square the reply just landed on, a recapture in the plain sense.
#: ``any``: every capture on the board. Measured 2026-09-09 on the two matches:
#: square gives 14.55 / 6.76 and 13.58 / 6.14 per 1000 plies (ours / AlphaGateau's),
#: any gives 9.36 / 4.60 and 8.98 / 3.97; the journal's 13.30 / 6.17 and 12.49 / 5.83
#: sit between them and neither reproduces them exactly. The ratio is 2.0-2.2x
#: under both rules, which is the claim that mattered.
RECAPTURE = "square"


def after_best_recapture(b: chess.Board, us: chess.Color,
                         square: Optional[int] = None) -> int:
    """`us` to move: our material after the capture that maximises it, or as is."""
    best = material(b, us)
    for m in b.legal_moves:
        if b.is_capture(m) and (RECAPTURE == "any" or m.to_square == square):
            b.push(m)
            best = max(best, material(b, us))
            b.pop()
    return best


def worst_reply(b: chess.Board, us: chess.Color, prefer: Optional[chess.Move] = None):
    """Opponent to move: the reply that leaves us worst off after our best recapture.

    Returns `(reply, our material after it)`. Ties go to `prefer` (the reply the game
    actually saw) so the refutation named is the one that was played when it is
    equally good by the rule.
    """
    best_move, best_val = None, None
    for r in b.legal_moves:
        b.push(r)
        v = (after_best_recapture(b, us, r.to_square) if not b.is_game_over()
             else material(b, us))
        b.pop()
        if best_val is None or v < best_val or (v == best_val and r == prefer):
            best_move, best_val = r, v
    return best_move, best_val


def move_loss(b: chess.Board, m: chess.Move, us: chess.Color) -> int:
    """How much `m` loses against the rule's worst reply. `us` to move at `b`."""
    before = material(b, us)
    b.push(m)
    if b.is_game_over():
        after = material(b, us)
    else:
        _, after = worst_reply(b, us)
    b.pop()
    return before - after


@dataclass
class Blunder:
    match: str
    game: int
    ply: int                      # 1-based ply of our move inside the game
    fen: str                      # before our move
    us: str
    move: str                     # the blunder, UCI
    reply: str                    # the reply played
    refutation: str               # the reply the rule scores best
    loss_played: int              # material lost after the played reply
    loss_rule: int                # material lost after the rule's refutation
    best_alt_loss: int            # loss of the best alternative by the same rule
    result: str
    plies_left: int
    probe: Dict[str, object] = field(default_factory=dict)


def scan_pgn(path: str, match: str, ours: str = "brokefish"):
    """Blunders and ply counts for both engines in one PGN."""
    ours_bl: List[Blunder] = []
    plies_total, games = 0, 0
    theirs = 0
    ours_games_hit = 0
    with open(path) as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            games += 1
            hs = game.headers
            white_ours = hs.get("White", "").startswith(ours)
            our_color = chess.WHITE if white_ours else chess.BLACK
            moves = list(game.mainline_moves())
            plies_total += len(moves)
            b = game.board()
            hit = False
            for i, mv in enumerate(moves):
                side = b.turn
                if i + 1 < len(moves):
                    before = material(b, side)
                    b.push(mv)
                    b.push(moves[i + 1])
                    after = after_best_recapture(b, side, moves[i + 1].to_square)
                    b.pop()
                    b.pop()
                    if before - after >= THRESHOLD:
                        if side == our_color:
                            hit = True
                            ours_bl.append(Blunder(
                                match=match, game=games, ply=i + 1, fen=b.fen(),
                                us="w" if side else "b", move=mv.uci(),
                                reply=moves[i + 1].uci(), refutation="",
                                loss_played=before - after, loss_rule=0,
                                best_alt_loss=0, result=hs.get("Result", "*"),
                                plies_left=len(moves) - i - 1))
                        else:
                            theirs += 1
                b.push(mv)
            ours_games_hit += hit
    return ours_bl, theirs, plies_total, games, ours_games_hit


def annotate_rule(bl: Blunder) -> None:
    """The refutation and the best alternative, both by the rule alone."""
    b = chess.Board(bl.fen)
    us = b.turn
    before = material(b, us)
    mv = chess.Move.from_uci(bl.move)
    b.push(mv)
    ref, after = worst_reply(b, us, prefer=chess.Move.from_uci(bl.reply))
    b.pop()
    bl.refutation, bl.loss_rule = ref.uci(), before - after
    best = None
    for m in b.legal_moves:
        if m == mv:
            continue
        loss = move_loss(b, m, us)
        best = loss if best is None else min(best, loss)
    bl.best_alt_loss = 99 if best is None else best


# -- the search ----------------------------------------------------------- #

class Prober:
    """One network, one search per budget, the tree read after each move."""

    def __init__(self, ckpt: str, budgets: List[int], batch: int,
                 device: str = "cuda", int8: bool = False):
        import torch
        from brokefish.env import cuda_impl as cenv
        from brokefish.eval.layer0 import load_net_state
        from brokefish.eval.runner import eval_config
        from brokefish.nn import encoder_impl
        from brokefish.nn.model import net_for_state
        from brokefish.search import search_impl

        self.torch, self.cenv, self.device, self.B = torch, cenv, device, batch
        state = load_net_state(ckpt, device="cpu")
        net = net_for_state(state)
        net.load_state_dict(state)
        self.net = net.to(device).eval()
        self.evaluate = encoder_impl("cuda")(self.net, int8=int8).forward_full
        self.searches = {}
        for n in budgets:
            cfg = eval_config(n, batch, E=E_CAP, gumbel=True, gumbel_m=16,
                              terminal_collapse=True)
            self.searches[n] = search_impl("cuda")(cfg, self.evaluate, env=cenv,
                                                   device=device, seed=0)
        self.prior_search = self.searches[min(budgets)]

    def _boards(self, fens):
        from brokefish.env import torch_impl as tenv
        fens = list(fens) + [fens[0]] * (self.B - len(fens))
        boards = self.torch.cat([tenv.from_fen(f)[0] for f in fens]).to(self.torch.int16)
        control = self.torch.cat([tenv.from_fen(f)[1] for f in fens]).to(self.torch.int16)
        return boards.to(self.device), control.to(self.device)

    def _edges(self, s, i: int, node: int) -> Dict[str, dict]:
        """uci -> (prior, N, Q in [-1, 1]) for node `node` of game `i`."""
        from brokefish.env import notation
        from brokefish.search.torch_impl import MOVE_BITS, PROMO_SHIFT
        k = int(s.node_nedges[i, node])
        if k == 0:
            return {}
        labels = s.edge_move[i, node, :k].to(self.torch.int64).cpu()
        board = s.node_board[i, node].cpu()[None].expand(k, -1)
        ucis = notation.to_uci(board, labels & ((1 << MOVE_BITS) - 1),
                               (labels >> PROMO_SHIFT) & 0b11)
        prior = s.edge_prior[i, node, :k].float().cpu()
        nvis = s.edge_N[i, node, :k].cpu()
        q = s.edge_Q[i, node, :k].cpu()
        return {u: {"p": float(prior[j]), "n": int(nvis[j]),
                    "q": 2.0 * float(q[j]) - 1.0, "e": j} for j, u in enumerate(ucis)}

    @staticmethod
    def _node_value(s, i: int, node: int) -> float:
        return 2.0 * float(s.node_value[i, node]) - 1.0

    def search(self, n: int, fens: List[str]):
        """Search every FEN at budget `n`; return per-position root and child tables."""
        from brokefish.env import notation
        from brokefish.search.torch_impl import MOVE_BITS, PROMO_SHIFT
        s = self.searches[n]
        boards, control = self._boards(fens)
        with self.torch.no_grad():
            s.reset(boards, control)
            rec = s.self_play_move(sims=n)
            played = rec.played.to(self.torch.int64).cpu()
            ucis = notation.to_uci(boards.cpu(), played & ((1 << MOVE_BITS) - 1),
                                   (played >> PROMO_SHIFT) & 0b11)
            root = self.torch.zeros_like(s._b)
            _, completed, _, _, _ = s._gumbel_completed(root)
            completed = completed.cpu()
            out = []
            for i in range(len(fens)):
                edges = self._edges(s, i, 0)
                for u, d in edges.items():
                    d["cq"] = 2.0 * float(completed[i, d["e"]]) - 1.0
                    child = int(s.edge_child[i, 0, d["e"]])
                    d["child"] = child
                out.append({"played": ucis[i], "root_value": self._node_value(s, i, 0),
                            "edges": edges, "search": s, "row": i})
        return out

    def child_table(self, s, i: int, child: int):
        """The child's edge table and raw value, if the tree expanded it."""
        if child < 0 or not (int(s.node_flags[i, child]) & 8):
            return None, None
        return self._edges(s, i, child), self._node_value(s, i, child)

    def raw(self, fens: List[str]):
        """Raw network prior over the legal moves and raw value, no search."""
        s = self.prior_search
        boards, control = self._boards(fens)
        with self.torch.no_grad():
            s.reset(boards, control)
            s.root_init(noise=False)
            return [(self._edges(s, i, 0), self._node_value(s, i, 0))
                    for i in range(len(fens))]


def classify(bl: Blunder, lo: int, hi: int) -> str:
    p = bl.probe
    if p[f"played{hi}"] != bl.move:
        return "BUDGET"
    if bl.best_alt_loss >= THRESHOLD:
        return "OTHER(forced)"
    if p["ref_prior_child"] < 1.0 / E_CAP:
        return "PRIOR"
    if p[f"ref_n{hi}"] == 0:
        return "PRIOR"
    return "VALUE"


def dissect(blunders: List[Blunder], ckpt: str, budgets: List[int], batch: int,
            device: str, int8: bool, log) -> None:
    lo, hi = min(budgets), max(budgets)
    prober = Prober(ckpt, budgets, batch, device=device, int8=int8)
    t0 = time.time()
    for start in range(0, len(blunders), batch):
        chunk = blunders[start:start + batch]
        fens = [bl.fen for bl in chunk]
        child_fens = []
        for bl in chunk:
            b = chess.Board(bl.fen)
            b.push(chess.Move.from_uci(bl.move))
            child_fens.append(b.fen())
        raw_child = prober.raw(child_fens)
        for n in budgets:
            tables = prober.search(n, fens)
            for bl, t in zip(chunk, tables):
                p = bl.probe
                p[f"played{n}"] = t["played"]
                p["root_value"] = t["root_value"]
                e = t["edges"].get(bl.move)
                p[f"prior_root"] = e["p"] if e else float("nan")
                p[f"n{n}"] = e["n"] if e else 0
                p[f"q{n}"] = e["q"] if e else float("nan")
                p[f"cq{n}"] = e["cq"] if e else float("nan")
                child_edges, child_v = (prober.child_table(t["search"], t["row"], e["child"])
                                        if e else (None, None))
                ref = child_edges.get(bl.refutation) if child_edges else None
                p[f"ref_n{n}"] = ref["n"] if ref else 0
                # `edge_Q` at the child is the opponent's view; flip to ours. An
                # unvisited edge holds a zero that means nothing, hence the nan.
                p[f"ref_q{n}"] = -ref["q"] if ref and ref["n"] > 0 else float("nan")
                p[f"child_expanded{n}"] = child_edges is not None
                # Played move's loss by the rule when it is not the blunder.
                if t["played"] != bl.move:
                    b = chess.Board(bl.fen)
                    p[f"loss{n}"] = move_loss(b, chess.Move.from_uci(t["played"]), b.turn)
                else:
                    p[f"loss{n}"] = bl.loss_rule
        for bl, (edges, v) in zip(chunk, raw_child):
            ref = edges.get(bl.refutation)
            bl.probe["ref_prior_child"] = ref["p"] if ref else 0.0
            bl.probe["ref_in_edges"] = ref is not None
            # `v` is the side to move's view, i.e. the opponent's; flip to ours.
            bl.probe["value_after"] = -v
            bl.probe["klass"] = classify(bl, lo, hi)
        print(f"  {min(start + batch, len(blunders))}/{len(blunders)} positions, "
              f"{time.time() - t0:.0f} s", file=log, flush=True)
    del prober
    prober_torch = sys.modules.get("torch")
    if prober_torch is not None:
        prober_torch.cuda.empty_cache()


# -- output --------------------------------------------------------------- #

COLUMNS = ["match", "game", "ply", "us", "fen", "move", "reply", "refutation",
           "loss_played", "loss_rule", "best_alt_loss", "result", "plies_left",
           "klass", "repro", "prior_root", "root_value", "value_after",
           "ref_prior_child", "ref_in_edges"]
PER_BUDGET = ["played{n}", "n{n}", "q{n}", "cq{n}", "child_expanded{n}",
              "ref_n{n}", "ref_q{n}", "loss{n}"]


def write_csv(path: str, blunders: List[Blunder], budgets: List[int]) -> None:
    cols = COLUMNS + [c.format(n=n) for n in budgets for c in PER_BUDGET]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for bl in blunders:
            row = {c: getattr(bl, c) for c in COLUMNS if hasattr(bl, c)}
            row.update(bl.probe)
            w.writerow([_fmt(row.get(c, "")) for c in cols])


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--match", action="append", default=None,
                    help="TAG=PGN:CKPT, repeatable; the checkpoint may be omitted to "
                         "count only")
    ap.add_argument("--budgets", default="128,1024")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--int8", action="store_true",
                    help="int8 fused encoder; the match server ran fp16")
    ap.add_argument("--limit", type=int, default=0,
                    help="dissect only the first N blunders per match (0 = all)")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--recapture", default="square", choices=("square", "any"),
                    help="what counts as capturing back; see RECAPTURE")
    ap.add_argument("--out", default="logs/blunder_dissect.csv")
    a = ap.parse_args()
    budgets = sorted(int(x) for x in a.budgets.split(","))
    global RECAPTURE
    RECAPTURE = a.recapture
    log = sys.stdout

    all_bl: List[Blunder] = []
    for spec in a.match or DEFAULT_MATCHES:
        tag, rest = spec.split("=", 1)
        pgn, _, ckpt = rest.partition(":")
        ours, theirs, plies, games, hit = scan_pgn(pgn, tag)
        per = 1000.0 / max(plies, 1)
        print(f"[{tag}] {pgn}: {games} games, {plies} plies; our blunders "
              f"{len(ours)} = {len(ours) * per:.2f} / 1000 plies, theirs {theirs} = "
              f"{theirs * per:.2f} / 1000 plies, games with >= 1 of ours "
              f"{hit}/{games}", file=log, flush=True)
        if a.count_only or not ckpt:
            continue
        if a.limit:
            ours = ours[:a.limit]
        t0 = time.time()
        for bl in ours:
            annotate_rule(bl)
        print(f"  rule annotation of {len(ours)} positions: {time.time() - t0:.0f} s",
              file=log, flush=True)
        print(f"  searching with {ckpt} at n = {budgets}, B = {a.batch}",
              file=log, flush=True)
        dissect(ours, ckpt, budgets, a.batch, a.device, a.int8, log)
        lo, hi = min(budgets), max(budgets)
        for bl in ours:
            bl.probe["repro"] = bl.probe[f"played{lo}"] == bl.move
        counts = Counter(bl.probe["klass"] for bl in ours)
        repro = sum(bl.probe["repro"] for bl in ours)
        print(f"  n = {lo} replays the recorded blunder in {repro}/{len(ours)}",
              file=log)
        budget_still = sum(1 for bl in ours if bl.probe["klass"] == "BUDGET"
                           and bl.probe[f"loss{hi}"] >= THRESHOLD)
        print(f"  classification: " + ", ".join(
            f"{k} {v}" for k, v in sorted(counts.items())), file=log)
        print(f"  of the BUDGET cases, n = {hi} still loses >= {THRESHOLD} by the rule "
              f"in {budget_still}", file=log)
        ref_differs = sum(bl.refutation != bl.reply for bl in ours)
        print(f"  the rule's refutation differs from the reply played in "
              f"{ref_differs}/{len(ours)}", file=log, flush=True)
        all_bl.extend(ours)

    if all_bl:
        write_csv(a.out, all_bl, budgets)
        print(f"wrote {a.out}: {len(all_bl)} rows", file=log)


if __name__ == "__main__":
    main()
