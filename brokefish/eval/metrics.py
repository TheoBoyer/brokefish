"""The layer-0 scalars of `evals.md` §4.

The job of layer 0 is to notice within minutes that a run has diverged. It is not
to measure strength, it never blocks, and per §2 nothing it returns may select a
checkpoint. Three quantities, each catching a failure the others miss:

- **value calibration** — the predicted value against the realised outcome. A
  value head that has stopped tracking outcomes is the earliest visible symptom
  of a diverging run, and it moves before Elo does.
- **policy entropy** — collapse to a delta and drift to uniform are different
  failures and both show here. Normalised by `log(legal moves)`, because the raw
  entropy of a 3-move position and a 40-move position are not comparable and a
  run whose game length changes would otherwise move this number for free.
- **draw rate and mean game length** — a self-play population that has collapsed
  onto one line shows up here first.

All three come out of one batch of self-play games, so `self_play_run` generates
them once and the rest is arithmetic.

⚠️ **`max |post-scale attention logit|` is deliberately not here**, though
`evals.md` §4 once listed it as a fourth scalar. It is an fp16 overflow watch on
the fused kernels, not a measurement of how good a net is: it has to reach inside
`net.encoder.layers` where everything else in this package treats the network as a
black box, it can only measure the torch oracle's logits rather than the
accumulator that actually overflows, and `CLAUDE.md` wants it logged *during
training* rather than once per checkpoint. It belongs to C2 — `docs/train.md` §11
carries it — and it was dropped from here on 2026-07-30.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from brokefish.env import torch_impl as env


# --------------------------------------------------------------------------- #
# One batch of self-play games, played out
# --------------------------------------------------------------------------- #

@dataclass
class SelfPlayRun:
    """Per-position rows and per-game rows from one batch played to the end."""

    value_pred: torch.Tensor       # [M] the net's value at the position, mover's view
    outcome: torch.Tensor          # [M] the realised result, mover's view, in {-1,0,1}
    visit_entropy: torch.Tensor    # [M] entropy of the root visit distribution, nats
    n_moves: torch.Tensor          # [M] legal moves at the position
    boards: torch.Tensor           # [M, 32] int16, the positions themselves
    control: torch.Tensor          # [M]     int16
    rep: torch.Tensor              # [M]     uint8
    game_plies: torch.Tensor       # [G] length of each finished game
    game_result: torch.Tensor      # [G] +1 white, 0 draw, -1 black
    game_code: torch.Tensor        # [G] spec §4.3 terminal code
    n_abandoned: int = 0           # collected games that passed `max_plies`
    n_in_flight: int = 0           # games a freed slot started and nobody waited for
    meta: Dict = field(default_factory=dict)


@torch.no_grad()
def self_play_run(net, games: int = 64, n_sims: int = 100, max_plies: int = 300,
                  batch: Optional[int] = None, impl: Optional[str] = None,
                  search_impl: str = "torch", seed: int = 0, device: str = "cuda",
                  max_iters: Optional[int] = None) -> SelfPlayRun:
    """Collect `games` finished games and keep what layer 0 needs.

    ⚠️ **This is the self-play protocol, not the evaluation one.** Dirichlet
    noise stays on and the first `tau_plies` moves are sampled, because the
    quantities being measured — draw rate, game length, value calibration — are
    properties of the distribution training actually sees. Turning exploration
    off here would produce a calibration curve for a policy that never generates
    a training example. `score_suite` is the other case and turns both off.

    ⚠️ **`games` is a count of games, not the batch size**, and the run collects
    the first `games` games *started*, never the first `games` to finish. That
    distinction is the whole reason this function is shaped the way it is.

    Collecting by finish order is much cheaper — the loop stops as soon as the
    quota is met instead of waiting for the longest game — and it was measured at
    29.6 s against 45.0 s for the same 64 games on the fast path, when the run still
    carried the attention scan
    (`logs/d1_layer0.log`). It is also **wrong**: short games finish first, so the
    sample is biased short, and mean game length and draw rate are two of the three
    things this run exists to measure. The observed mean fell from 125 plies to
    101 by switching, which is the bias, not an improvement. So the expensive
    version is the one that ships.

    ⚠️ **`batch < games` does not recover the difference, and that was measured
    rather than assumed.** The reasoning was that a small batch lets early
    finishers start the next game while the long ones run, shrinking the idle
    tail. It does — and `games=64, batch=16` came out at 45.8 s against 45.0 s,
    slightly *worse*, because a search over 16 boards is proportionally less
    efficient than one over 64 and gives back exactly what the tail saved. Both
    ideas for making layer 0 cheaper failed; the knobs that actually work are
    `games` and `n_sims`, which trade cost against resolution honestly.

    A game that passes `max_plies` is abandoned rather than scored: its outcome
    is unknown, and calling it a draw is exactly the bias the draw-rate metric
    exists to detect. `n_abandoned` counts those; `n_in_flight` counts the games
    a finished slot started and that this run does not wait for.
    """
    from .runner import evaluator, make_search

    B = batch or games
    evaluate = evaluator(net, impl)
    # `greedy=False`: this one keeps the *self-play* protocol, noise and all --
    # see the warning above.
    search = make_search(n_sims, B, net, impl=impl, search_impl=search_impl,
                         seed=seed, device=device, greedy=False)
    search.reset()

    rows: List[Dict[str, torch.Tensor]] = []
    game_id = torch.arange(B, device=device)
    next_id = B
    result: Dict[int, int] = {}
    plies_of: Dict[int, int] = {}
    code_of: Dict[int, int] = {}
    abandoned: set = set()
    # The run is over when every game in [0, games) has ended one way or the
    # other. Membership is decided at start time, which is what keeps the sample
    # unbiased; see the warning above.
    pending = set(range(games))
    # Each iteration is one move in every slot, so the bound is one wave of
    # maximum-length games per batch, with room to spare.
    cap = max_iters or (max_plies * (games // B + 2))

    for _ in range(cap):
        if not pending:
            break
        record = search.self_play_move()

        _p, _q, value = evaluate(record.board, record.control, record.rep)
        mask, _ = env.movegen(record.board, record.control)
        n_moves = env.bitset_to_bool(mask).reshape(B, -1).sum(-1)

        p = record.policy_prob.float()
        entropy = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(-1)

        rows.append({
            "gid": game_id.clone(),
            "value": value.float().clone(),
            "white": (record.control > 0).clone(),
            "entropy": entropy,
            "n_moves": n_moves.to(torch.int32),
            # 64 bytes a position: keeping every board of the run is ~1 MB, which
            # is cheap enough to be worth having for any later diagnostic that
            # wants the distribution the games actually visited rather than a
            # synthetic batch.
            "boards": record.board.clone(),
            "control": record.control.clone(),
            "rep": record.rep.clone(),
        })

        if bool(record.done.any()):
            # `record.result` is from the point of view of the player to move in
            # the position the move *produced*, so -1 there means the player who
            # just moved delivered mate. The control word after the move says who
            # that is.
            new_white = search.game_control > 0
            white_view = torch.where(new_white, record.result.to(torch.int8),
                                     (-record.result).to(torch.int8))
            _m, chk = env.movegen(search.game_board, search.game_control)
            code, _r = env.terminal(_m, chk, search.game_control, search.game_board,
                                    search.game_hash, search.game_ring, search.game_ring_len)
            done = record.done.nonzero(as_tuple=True)[0]
            for gid, w, pl, cd in zip(game_id[done].tolist(), white_view[done].tolist(),
                                      search.game_ply[done].tolist(), code[done].tolist()):
                result[gid], plies_of[gid], code_of[gid] = w, pl, cd
                pending.discard(gid)

        # A finished game may not be searched again (`mcts.md` §6.7, invariant 8),
        # and a game that will not end has to be let go of, so both restart here.
        too_long = search.game_ply >= max_plies
        gave_up = (too_long & ~record.done).nonzero(as_tuple=True)[0]
        for gid in game_id[gave_up].tolist():
            abandoned.add(gid)
            pending.discard(gid)
        restart = (record.done | too_long).nonzero(as_tuple=True)[0]
        if restart.numel():
            search.reset(rows=restart)
            game_id[restart] = torch.arange(next_id, next_id + int(restart.numel()),
                                            device=device)
            next_id += int(restart.numel())

    if not rows:
        raise RuntimeError("self_play_run produced no positions")

    gid = torch.cat([r["gid"] for r in rows])
    # -128 marks "not one of the collected games", which is why the table is
    # int16 and not int8: the sentinel has to sit outside {-1, 0, +1} without
    # colliding with a real result. Games with an id at or above `games` are
    # ones a freed slot started and that this run does not wait for, so they are
    # left at the sentinel rather than scored.
    table = torch.full((max(next_id, 1),), -128, dtype=torch.int16, device=device)
    order = sorted(g for g in result if g < games)
    if order:
        keys = torch.tensor(order, device=device)
        table[keys] = torch.tensor([result[g] for g in order],
                                   dtype=torch.int16, device=device)
    z_white = table[gid]
    scored = z_white != -128

    white = torch.cat([r["white"] for r in rows])[scored]
    z = z_white[scored].float()
    return SelfPlayRun(
        value_pred=torch.cat([r["value"] for r in rows])[scored],
        outcome=torch.where(white, z, -z),
        visit_entropy=torch.cat([r["entropy"] for r in rows])[scored],
        n_moves=torch.cat([r["n_moves"] for r in rows])[scored],
        boards=torch.cat([r["boards"] for r in rows])[scored],
        control=torch.cat([r["control"] for r in rows])[scored],
        rep=torch.cat([r["rep"] for r in rows])[scored],
        game_plies=torch.tensor([plies_of[g] for g in order], dtype=torch.int32,
                                device=device),
        game_result=torch.tensor([result[g] for g in order], dtype=torch.int8,
                                 device=device),
        game_code=torch.tensor([code_of[g] for g in order], dtype=torch.uint8,
                               device=device),
        n_abandoned=len(abandoned),
        n_in_flight=sum(1 for g in game_id.tolist() if g >= games or g in pending),
        meta={"games": len(order), "requested": games, "batch": B,
              "n_sims": n_sims, "max_plies": max_plies, "moves": len(rows),
              "extra_games_started": max(0, next_id - games)})


# --------------------------------------------------------------------------- #
# The scalars
# --------------------------------------------------------------------------- #

def value_calibration(run: SelfPlayRun, bins: int = 10) -> dict:
    """A reliability curve plus the two numbers that summarise it.

    The value head is `tanh`, so predictions live in [-1,1] and the bins are cut
    there rather than in the tree's [0,1]. Reported per bin: how many positions,
    the mean prediction, and the mean realised outcome. A calibrated head has the
    two equal.

    `ece` is the count-weighted mean gap between them; `brier` is the mean
    squared error, which `ece` cannot replace — a head that predicts the base
    rate everywhere is perfectly calibrated and useless, and only `brier` says so.
    """
    v, z = run.value_pred, run.outcome
    if v.numel() == 0:
        return {"n": 0, "ece": float("nan"), "brier": float("nan"), "bins": []}

    edges = torch.linspace(-1.0, 1.0, bins + 1, device=v.device)
    idx = torch.bucketize(v, edges[1:-1].contiguous())
    out, ece, total = [], 0.0, int(v.numel())
    for b in range(bins):
        sel = idx == b
        k = int(sel.sum())
        if k == 0:
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0,
                        "mean_pred": None, "mean_outcome": None})
            continue
        mp, mo = float(v[sel].mean()), float(z[sel].mean())
        ece += k / total * abs(mp - mo)
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k,
                    "mean_pred": mp, "mean_outcome": mo})

    return {"n": total, "ece": ece, "brier": float(((v - z) ** 2).mean()),
            "mean_pred": float(v.mean()), "mean_outcome": float(z.mean()),
            "bins": out}


def policy_entropy(run: SelfPlayRun) -> dict:
    """Entropy of the root visit distribution, raw and normalised.

    Normalised by `log(legal moves)`, so 1.0 is uniform over the legal moves and
    0.0 is a delta, independently of how many moves the position had. Positions
    with one legal move are dropped: their entropy is 0 by the rules and the
    normaliser is 0 too.
    """
    h, k = run.visit_entropy, run.n_moves
    sel = k > 1
    if int(sel.sum()) == 0:
        return {"n": 0, "entropy_nats": float("nan"), "entropy_normalised": float("nan")}
    norm = h[sel] / k[sel].float().log()
    return {"n": int(sel.sum()),
            "entropy_nats": float(h[sel].mean()),
            "entropy_normalised": float(norm.mean()),
            "entropy_normalised_p10": float(norm.quantile(0.10)),
            "entropy_normalised_p90": float(norm.quantile(0.90)),
            "mean_legal_moves": float(k.float().mean())}


def game_statistics(run: SelfPlayRun) -> dict:
    """Draw rate, mean game length and the breakdown by terminal code."""
    r, plies = run.game_result, run.game_plies
    n = int(r.numel())
    if n == 0:
        return {"n_games": 0, "n_abandoned": run.n_abandoned,
                "n_in_flight": run.n_in_flight}
    names = {1: "checkmate", 2: "stalemate", 3: "fifty_move",
             4: "threefold", 5: "insufficient"}
    by_code = {v: int((run.game_code == k).sum()) for k, v in names.items()}
    return {"n_games": n, "n_abandoned": run.n_abandoned,
            "n_in_flight": run.n_in_flight,
            "draw_rate": float((r == 0).float().mean()),
            "white_score": float(((r.float() + 1.0) / 2.0).mean()),
            "mean_plies": float(plies.float().mean()),
            "median_plies": float(plies.float().median()),
            "terminal_codes": by_code}
