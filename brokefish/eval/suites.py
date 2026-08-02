"""The rule-level suites of `evals.md` §8.1 — layer 3, and cheap enough for layer 0.

Every suite here has the same shape, and that shape is the whole design:

    a position, a set of moves that are **right by the rules**, and a set that
    are **wrong by the rules**, both computed by our own move generator.

"Right by the rules" is what keeps these inside the tabula rasa boundary. The
answer key is never a material count and never an evaluation — it is a spec §4.3
terminal code. *Mate beats stalemate* is a rule. *A knight is worth three pawns*
is an opinion, and no suite here needs one. That is what makes them runnable
during training and not only after it, and it is also why there is no
"best move" suite: naming the best move in a quiet position requires an opinion
we are not allowed to have.

The six suites, all one ply deep so the key is decidable without a search:

| suite | item | wrong |
|---|---|---|
| `mate_in_1` | a mate in 1 exists | — |
| `avoid_stalemate` | a mate and a stalemate both exist | stalemating |
| `avoid_fifty` | a mate exists at halfmove clock 99 | letting the clock run out |
| `avoid_threefold` | a mate exists and one reply repeats a position twice seen | repeating |
| `avoid_insufficient` | a mate exists and one reply strips the board to a dead draw | trading into it |
| `underpromotion` | **every** mating move is a promotion to something other than a queen | promoting to a queen |

Two of them cannot be sampled and are *constructed*, which is stated here rather
than buried:

⚠️ `avoid_fifty` sets the halfmove clock to 99 in the generator. Random play
reaches clock 99 with a mate on the board zero times in 170 924 positions.

⚠️ `avoid_threefold` **plants the ring**. It takes a harvested mate-in-1 position,
finds a reversible non-mating reply, and writes that reply's position hash into
the repetition ring twice, so playing it draws. The ring is exactly "positions
already seen", and two copies is what a real repetition looks like, but the
*path* that produced them is not replayed. What the suite measures — does the
search take a draw when mate is available — is unaffected; what it does not
prove is that such a game history exists. Nothing else in the repository plants
a ring, and `test_eval.py` checks the planted one really does read back as
`repetition_count == 3`.

Harvest rates, measured on 2026-07-30 and not guesses, per 50 000 sampled
positions (`positions.py` explains why sampling and not random play):

    mate_in_1            1 997        avoid_insufficient       96
    avoid_stalemate        582        underpromotion           12
    avoid_fifty          1 210        avoid_threefold   as many as mate_in_1

so the whole harvest is a couple of minutes and is meant to be cached to disk,
not regenerated per checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from brokefish.env import torch_impl as env
from .positions import random_positions
from .probe import MOVE_MASK, PROMO_SHIFT, promotion_targets, reply_codes

CHECKMATE, STALEMATE, FIFTY_MOVE, REPETITION, INSUFFICIENT = 1, 2, 3, 4, 5

SUITE_NAMES = ("mate_in_1", "avoid_stalemate", "avoid_fifty", "avoid_threefold",
               "avoid_insufficient", "underpromotion")


@dataclass
class Suite:
    """One suite. Every tensor is indexed by item; `good`/`bad` are padded with -1.

    Labels are the search's edge encoding (`mcts.md` §6.4), so scoring is a
    membership test against `MoveRecord.played` with no translation step.
    """

    name: str
    description: str
    boards: torch.Tensor       # [N, 32] int16
    control: torch.Tensor      # [N]     int16
    good: torch.Tensor         # [N, Kg] int16, -1 padded
    bad: torch.Tensor          # [N, Kb] int16, -1 padded
    ring: torch.Tensor         # [N, 100] int64
    ring_len: torch.Tensor     # [N]      int64

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    def to(self, device) -> "Suite":
        move = lambda t: t.to(device)
        return Suite(self.name, self.description, move(self.boards), move(self.control),
                     move(self.good), move(self.bad), move(self.ring), move(self.ring_len))

    def subset(self, idx: torch.Tensor) -> "Suite":
        return Suite(self.name, self.description, self.boards[idx], self.control[idx],
                     self.good[idx], self.bad[idx], self.ring[idx], self.ring_len[idx])


# --------------------------------------------------------------------------- #
# Grouping a flat (game, label) list into the padded per-item form
# --------------------------------------------------------------------------- #

def _group(game: torch.Tensor, label: torch.Tensor, keep: torch.Tensor,
           n: int) -> torch.Tensor:
    """``[n, K] int16`` of the kept labels, -1 padded.

    `enumerate_moves` emits rows sorted by game, and a boolean filter preserves
    that order, so the position of a label inside its group is its index minus
    the group's start. No sort, no loop.
    """
    g, lab = game[keep], label[keep]
    cnt = torch.bincount(g, minlength=n)
    k = max(int(cnt.max()) if cnt.numel() else 0, 1)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(g.numel(), device=g.device) - start[g]
    out = torch.full((n, k), -1, dtype=torch.int16, device=game.device)
    out[g, pos] = lab.to(torch.int16)
    return out


def _has(game: torch.Tensor, keep: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, dtype=torch.bool, device=game.device)
    out[game[keep]] = True
    return out


# --------------------------------------------------------------------------- #
# The builders. Each takes one scan and returns (selected items, good, bad).
# --------------------------------------------------------------------------- #

def _mate_and(code: torch.Tensor, game: torch.Tensor, label: torch.Tensor,
              n: int, draw_code: Optional[int]
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four "a mate exists, and so does a way to throw it away" suites."""
    mate = code == CHECKMATE
    sel = _has(game, mate, n)
    if draw_code is not None:
        sel &= _has(game, code == draw_code, n)
    good = _group(game, label, mate, n)
    bad = (_group(game, label, code == draw_code, n) if draw_code is not None
           else torch.full((n, 1), -1, dtype=torch.int16, device=game.device))
    return sel, good, bad


def _underpromotion(boards: torch.Tensor, control: torch.Tensor, code: torch.Tensor,
                    game: torch.Tensor, label: torch.Tensor, n: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Items where *every* mating move is a promotion to something but a queen.

    The stronger condition is deliberate. "A knight promotion mates" is not a
    test if a rook on the other side of the board also mates: the search can be
    right for a reason the suite is not asking about. Requiring that the
    underpromotions are the *only* mates makes a queen promotion — the move a net
    that has learned "promote to queen" will play — a definite failure.

    ⚠️ **The promotion field does not identify a promotion.** Spec §3's order is
    ``0:N 1:B 2:R 3:Q`` and a non-promotion move carries field 0, so `promo == 0`
    means "knight promotion *or* an ordinary move" and reading it as the former
    silently deletes the main motif. Promotion-ness comes from the board.
    """
    promo = (label >> PROMO_SHIFT) & 0b11
    is_promo = promotion_targets(boards, control).reshape(n, -1)[game, label & MOVE_MASK]
    mate = code == CHECKMATE
    under = is_promo & (promo != 3)

    sel = _has(game, mate & under, n) & ~_has(game, mate & ~under, n)
    good = _group(game, label, mate, n)
    # The wrong move is the queen promotion, which is what a net that has learned
    # "a promotion is a queen" will actually play.
    bad = _group(game, label, is_promo & (promo == 3), n)
    return sel, good, bad


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #

# (generator kwargs, draw code, description). Tuned by the yield measurements in
# the module docstring; `max_extra` is the number of non-king pieces per side.
_RECIPES: Dict[str, Tuple[dict, Optional[int], str]] = {
    "mate_in_1": (dict(max_extra=5), None,
                  "a mate in 1 is available"),
    "avoid_stalemate": (dict(max_extra=5), STALEMATE,
                        "a mate in 1 is available and so is a stalemate"),
    "avoid_fifty": (dict(max_extra=5, clock=99), FIFTY_MOVE,
                    "a mate in 1 is available at halfmove clock 99"),
    "avoid_insufficient": (dict(max_extra=3, promotion_ready=True), INSUFFICIENT,
                           "a mate in 1 is available and so is a dead-drawn material split"),
    "underpromotion": (dict(max_extra=5, promotion_ready=True), None,
                       "every mating move is a promotion to a knight, bishop or rook"),
}


def _harvest_one(name: str, target: int, seed: int, device: str, batch: int,
                 max_rounds: int) -> Suite:
    kwargs, draw_code, description = _RECIPES[name]
    parts: List[Suite] = []
    have = 0
    for r in range(max_rounds):
        if have >= target:
            break
        boards, control = random_positions(batch, seed=seed + r, device=device, **kwargs)
        game, label, code, _h, _i = reply_codes(boards, control)
        n = int(boards.shape[0])
        if name == "underpromotion":
            sel, good, bad = _underpromotion(boards, control, code, game, label, n)
        else:
            sel, good, bad = _mate_and(code, game, label, n, draw_code)
        idx = sel.nonzero(as_tuple=True)[0]
        if not idx.numel():
            continue
        ring, ring_len = env.empty_history(int(idx.numel()), device=boards.device)
        parts.append(Suite(name, description, boards[idx], control[idx],
                           good[idx], bad[idx], ring, ring_len))
        have += int(idx.numel())
    if not parts:
        raise RuntimeError(f"suite {name!r}: nothing harvested in {max_rounds} rounds")
    return _concat(parts).subset(torch.arange(min(have, target), device=device))


def _concat(parts: List[Suite]) -> Suite:
    kg = max(int(p.good.shape[1]) for p in parts)
    kb = max(int(p.bad.shape[1]) for p in parts)

    def pad(t: torch.Tensor, k: int) -> torch.Tensor:
        if t.shape[1] == k:
            return t
        fill = torch.full((t.shape[0], k - t.shape[1]), -1, dtype=t.dtype, device=t.device)
        return torch.cat([t, fill], dim=1)

    return Suite(
        parts[0].name, parts[0].description,
        torch.cat([p.boards for p in parts]), torch.cat([p.control for p in parts]),
        torch.cat([pad(p.good, kg) for p in parts]),
        torch.cat([pad(p.bad, kb) for p in parts]),
        torch.cat([p.ring for p in parts]), torch.cat([p.ring_len for p in parts]))


def _plant_threefold(base: Suite, device: str) -> Suite:
    """Turn mate-in-1 items into `avoid_threefold` by writing their ring.

    For each item, find a reply that is reversible and does not end the game, and
    write the hash of the position it reaches into the ring twice. Playing that
    reply then makes the position occur a third time, which is spec §4.3 code 4.

    ⚠️ The ring is planted, not replayed — see the module docstring. The item is
    dropped when the position has no reversible non-terminal reply at all, which
    is what makes this a filter rather than a relabelling.
    """
    hash_ = env.hash_position(base.boards, base.control)
    game, label, code, next_hash, irrev = reply_codes(
        base.boards, base.control, hash_)
    n = len(base)

    usable = (code == 0) & ~irrev
    # One reply per item, and specifically the first: a scatter with duplicate
    # indices has no defined winner in torch, so the choice is made by index
    # arithmetic instead. Rows come out of `enumerate_moves` sorted by game, so
    # each group starts at the exclusive cumulative count before it.
    g = game[usable]
    cnt = torch.bincount(g, minlength=n)
    first = torch.cumsum(cnt, 0) - cnt
    idx = (cnt > 0).nonzero(as_tuple=True)[0]
    pick_hash = next_hash[usable][first[idx]]
    pick_label = label[usable][first[idx]].to(torch.int16)
    out = base.subset(idx)
    ring, ring_len = env.empty_history(int(idx.numel()), device=device)
    ring[:, 0] = pick_hash
    ring[:, 1] = pick_hash
    ring_len[:] = 2
    return Suite("avoid_threefold",
                 "a mate in 1 is available and one reply repeats a position twice seen",
                 out.boards, out.control, out.good, pick_label[:, None],
                 ring, ring_len)


def harvest_suites(target: int = 200, seed: int = 20260730, device: str = "cuda",
                   batch: int = 20_000, max_rounds: int = 400,
                   names: Optional[Tuple[str, ...]] = None) -> Dict[str, Suite]:
    """Generate every suite. Minutes, not seconds — cache the result.

    ``target`` is per suite. At 200 items a 100 % score has a Wilson lower bound
    of 98 %, which is the resolution layer 0 needs; a suite is a tripwire, not a
    rating.
    """
    names = names or SUITE_NAMES
    out: Dict[str, Suite] = {}
    for i, name in enumerate(names):
        if name == "avoid_threefold":
            continue
        out[name] = _harvest_one(name, target, seed + 7919 * i, device, batch, max_rounds)
    if "avoid_threefold" in names:
        base = out.get("mate_in_1") or _harvest_one(
            "mate_in_1", target, seed + 104729, device, batch, max_rounds)
        out["avoid_threefold"] = _plant_threefold(base, device)
    return out


def save_suites(suites: Dict[str, Suite], path: str) -> None:
    torch.save({k: {f: getattr(v, f) for f in
                    ("name", "description", "boards", "control", "good", "bad",
                     "ring", "ring_len")}
                for k, v in suites.items()}, path)


def load_suites(path: str, device: str = "cuda") -> Dict[str, Suite]:
    raw = torch.load(path, map_location=device, weights_only=True)
    return {k: Suite(**v) for k, v in raw.items()}


DEFAULT_SUITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "suites.pt")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % interval on a proportion. Not the normal approximation: these counts
    sit at the ends of [0,1], where it produces bounds outside the interval."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _membership(played: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``[N] bool``: is each played label in its row of the -1-padded table."""
    return ((table == played[:, None]) & (table >= 0)).any(-1)


@torch.no_grad()
def score_suite(suite: Suite, net, n: int = 128, impl: Optional[str] = None,
                search_impl: str = "cuda", batch: int = 256, seed: int = 0,
                device: str = "cuda") -> dict:
    """Run the search on a suite and report the fraction it gets right.

    Evaluation configuration, per `evals.md` §3: **no Dirichlet noise and greedy
    move selection**. `tau_plies=0` is what makes `select_and_advance` take the
    argmax of the root visit counts instead of sampling from them; leaving it at
    the self-play default would make a suite score a sample from the policy
    rather than a measurement of it.
    """
    from .runner import make_search

    suite = suite.to(device)
    total = len(suite)
    right = wrong = 0
    for lo in range(0, total, batch):
        hi = min(lo + batch, total)
        item = suite.subset(torch.arange(lo, hi, device=device))
        b = hi - lo
        search = make_search(n, b, net, impl=impl, search_impl=search_impl,
                             seed=seed, device=device)
        search.reset(item.boards, item.control)
        # The suite carries its own history, and `reset` clears it. The
        # threefold suite is nothing but its ring, so this line is the suite.
        search.game_ring[:] = item.ring
        search.game_ring_len[:] = item.ring_len
        record = search.self_play_move()
        played = record.played.to(torch.int16)
        right += int(_membership(played, item.good).sum())
        wrong += int(_membership(played, item.bad).sum())

    lo_ci, hi_ci = _wilson(right, total)
    return {"suite": suite.name, "n_items": total, "n_sims": n,
            "accuracy": right / total if total else 0.0,
            "ci95": [lo_ci, hi_ci],
            "blunder_rate": wrong / total if total else 0.0}


@torch.no_grad()
def score_suite_policy(suite: Suite, net, impl: Optional[str] = None,
                       batch: int = 1024, device: str = "cuda") -> dict:
    """The same question asked of the raw policy, with no search at all.

    Worth having next to `score_suite`: the two together say whether a failure is
    the network's or the search's, and this one costs one forward pass. The
    comparison is the diagnostic — a policy that is right and a search that is
    wrong is a different bug from both being wrong.
    """
    suite = suite.to(device)
    evaluate = _evaluator(net, impl)
    total = len(suite)
    right = wrong = 0
    for lo in range(0, total, batch):
        hi = min(lo + batch, total)
        item = suite.subset(torch.arange(lo, hi, device=device))
        rep = (env.repetition_count(env.hash_position(item.boards, item.control),
                                    item.ring, item.ring_len) - 1
               ).clamp(0, 2).to(torch.uint8)
        policy, promo, _value = evaluate(item.boards, item.control, rep)
        played = _policy_argmax(item, policy, promo)
        right += int(_membership(played, item.good).sum())
        wrong += int(_membership(played, item.bad).sum())
    lo_ci, hi_ci = _wilson(right, total)
    return {"suite": suite.name, "n_items": total, "n_sims": 0,
            "accuracy": right / total if total else 0.0,
            "ci95": [lo_ci, hi_ci],
            "blunder_rate": wrong / total if total else 0.0}


def _policy_logits(item, policy: torch.Tensor, promo: torch.Tensor) -> torch.Tensor:
    """`[N, 32*64*4]` scores over every candidate edge, illegal ones at `-inf`.

    §6.4's arithmetic, minus the tree: the promotion log-softmax is added to the
    policy logit so one ranking settles the target square and the promotion type
    together. ``item`` needs only ``boards`` and ``control``, so a `PuzzleSet`
    works here as well as a `Suite`.
    """
    from .probe import promotion_targets
    mask, _ = env.movegen(item.boards, item.control)
    legal = env.bitset_to_bool(mask)
    is_promo = promotion_targets(item.boards, item.control) & legal
    cand = legal[..., None].expand(-1, -1, -1, 4).clone()
    cand[..., 1:] &= is_promo[..., None]

    lp = torch.log_softmax(promo.float(), dim=-1)
    logit = policy.float()[..., None].expand(-1, -1, -1, 4).clone()
    logit += torch.where(is_promo[..., None], lp[:, :, None, :], torch.zeros_like(logit))
    logit = torch.where(cand, logit, torch.full_like(logit, float("-inf")))
    return logit.reshape(logit.shape[0], -1)


def _cols_to_labels(col: torch.Tensor) -> torch.Tensor:
    """Flat candidate index to the search's own label encoding."""
    return ((col // 4) | ((col % 4) << PROMO_SHIFT)).to(torch.int16)


def _policy_topk(item, policy: torch.Tensor, promo: torch.Tensor,
                 k: int = 1) -> torch.Tensor:
    """`[N, k]` highest-prior legal edges, best first, as spec §3 labels.

    ⚠️ **A position can have fewer than `k` legal edges**, and `topk` then returns
    `-inf` slots whose flat index decodes to a real-looking label. Those are filled
    with the best edge instead, which is the only choice that cannot invent a
    membership: repeating a move already counted never adds a new hit.
    """
    logit = _policy_logits(item, policy, promo)
    k = min(k, logit.shape[-1])
    vals, cols = logit.topk(k, dim=-1)
    cols = torch.where(torch.isfinite(vals), cols, cols[:, :1].expand_as(cols))
    return _cols_to_labels(cols)


def _policy_argmax(item, policy: torch.Tensor, promo: torch.Tensor) -> torch.Tensor:
    """The single highest-prior legal edge. `_policy_topk` with `k = 1`."""
    return _policy_topk(item, policy, promo, k=1)[:, 0]


def _evaluator(net, impl: Optional[str]) -> Callable:
    from brokefish.search.torch_impl import make_evaluator
    return make_evaluator(net, impl)


def main() -> None:
    """Harvest and cache. Minutes; the result never changes, so do it once."""
    import argparse

    ap = argparse.ArgumentParser(description="harvest the rule suites of evals.md 8.1")
    ap.add_argument("--target", type=int, default=200, help="items per suite")
    ap.add_argument("--out", default=DEFAULT_SUITE_PATH)
    ap.add_argument("--batch", type=int, default=20_000, help="positions scanned per round")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = harvest_suites(target=args.target, seed=args.seed, device=args.device,
                         batch=args.batch, max_rounds=args.rounds)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_suites(out, args.out)
    for name in SUITE_NAMES:
        print(f"{name:22s} {len(out[name]):5d}  {out[name].description}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
