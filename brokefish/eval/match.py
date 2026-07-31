"""One match between two networks, and the openings it is played from.

`evaluation.md` §5.4. This is the primitive layer 2 is built out of: given two
evaluators and a set of starting positions, play every position twice — once with
each network as White — and return what happened.

Three decisions are load-bearing and each of them has a failure mode attached.

**Openings are self-generated, not a book.** `random_openings` walks a fixed
number of uniformly random *legal* plies from the start and rejects whatever is
already terminal. The reasons are in §5.4: the draws this league has to break come
from a near-uniform policy shuffling into threefold repetition, which is not the
mechanism a UHO book is built to break; random openings are naturally unbalanced,
which produces *more* decisive games, and the colour pairing below is what makes
that unbalance fair. It also keeps the tabula rasa boundary uncomplicated — the
positions come from the rules and from nothing else.

⚠️ **Evaluation is deterministic** (`runner.eval_config`: `eps = 0`,
`tau_plies = 0`), so the openings are the *only* source of diversity in the whole
league. Two engines replaying the same opening produce the same game every time.
A pairing of `G` games therefore needs `G/2` **distinct** openings, and
`random_openings` deduplicates by Zobrist hash so that "distinct" is a fact rather
than a hope.

**Colours are paired, and the unit of measurement is the pair.** Every opening is
played twice, with the colours swapped. Without that, a league on unbalanced
openings measures who drew the favourable side, not who plays better. The property
this buys is exact rather than statistical: a network played against *itself*
scores exactly 0.5 per game pair whatever the games do, which is what
`tests/test_league.py` uses as its end-to-end oracle.

**One network per batch, kept in lockstep.** The two networks alternate by ply, so
a batch that mixes both colour assignments needs two evaluations per position —
either both networks over the whole batch (2× the encoder) or a gather/scatter with
dynamic shapes. Neither is necessary: the two colour assignments are played as two
*separate* batches, and inside one batch every row is at the same ply, so at any
moment every row wants the same network. `_play_half` asserts that lockstep rather
than assuming it — see `_swap_evaluator`.

⚠️ **A finished game may not be searched again** (`search.md` §6.7, invariant 8),
so a finished row is restarted on a throwaway position and its later moves are
ignored. The throwaway is started with the **side to move matching the batch**,
because a row reset to the ordinary start position would be a ply out of phase with
everything else and would silently break the lockstep above. The cost is that the
batch does real work for games that have ended; with games running 40 to 250 plies
that tail is the known inefficiency of this harness and the reason `max_plies`
matters. It is not a correctness problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from brokefish.env import torch_impl as env

from .runner import eval_config, search_class

# spec §4.3, so a result table reads as words rather than as integers. Re-exported
# from `env` rather than restated: one copy of the mapping, in one place.
TERMINAL_NAMES = env.TERMINAL_NAMES


# --------------------------------------------------------------------------- #
# Openings
# --------------------------------------------------------------------------- #

def random_openings(count: int, plies: int = 8, seed: int = 0,
                    device: str | torch.device = "cuda",
                    max_rounds: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """`count` distinct, legal, non-terminal positions, `plies` from the start.

    Returns ``(boards [count, 32] int16, control [count] int16)``, ready to hand to
    ``Search.reset``.

    ``plies`` must be **even**, so every opening has White to move and a game pair
    is White-versus-Black in the ordinary sense. It is not a correctness
    requirement — the harness assigns colours explicitly — but an odd book would
    make every "White" in the results table the second player, which is the kind of
    detail that is discovered six weeks later in a plot.

    The random walk is driven by a **CPU** generator so that the same ``seed``
    produces the same book on any device: the league's openings have to be
    reproducible across machines or two runs of the curve are not comparable.

    Rejected: anything terminal after the walk (a game cannot start from a
    finished position), and anything with fewer than two legal moves (a position
    with one legal reply measures nothing). Duplicates are removed by Zobrist hash,
    which is what makes the "distinct openings" claim above true rather than
    probable.
    """
    if plies % 2:
        raise ValueError(f"plies must be even so White is to move, got {plies}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    keep_boards: List[torch.Tensor] = []
    keep_control: List[torch.Tensor] = []
    seen: set = set()
    have = 0

    for _ in range(max_rounds):
        if have >= count:
            break
        boards, control = env.initial_boards(count, device=device)
        mask, _ = env.movegen(boards, control)
        for _ply in range(plies):
            boards, control, mask = _random_legal_step(boards, control, mask, gen, device)

        mask, in_check = env.movegen(boards, control)
        code, _ = env.terminal(mask, in_check, control, boards)
        n_moves = env.bitset_to_bool(mask).reshape(boards.shape[0], -1).sum(-1)
        ok = (code == 0) & (n_moves >= 2)
        # The hash is over (position, side to move), which is exactly the identity
        # two openings have to differ in.
        hashes = env.hash_position(boards, control)

        for row, (good, h) in enumerate(zip(ok.tolist(), hashes.tolist())):
            if have >= count:
                break
            if not good or h in seen:
                continue
            seen.add(h)
            keep_boards.append(boards[row])
            keep_control.append(control[row])
            have += 1

    if have < count:
        raise RuntimeError(
            f"random_openings produced {have} of {count} distinct openings in "
            f"{max_rounds} rounds at plies={plies}; either the walk is too short to "
            f"give {count} distinct positions or something rejects everything")

    return torch.stack(keep_boards).contiguous(), torch.stack(keep_control).contiguous()


def _random_legal_step(boards: torch.Tensor, control: torch.Tensor, mask: torch.Tensor,
                       gen: torch.Generator, device) -> Tuple[torch.Tensor, ...]:
    """One uniformly random legal move per row. A terminal row stands still.

    The same construction as `tests/boards.py`: pick the `k`-th set bit of the
    legality bitset with a cumulative sum, so there is no host round trip and the
    distribution is uniform over legal moves rather than over the mask's words.
    """
    n = boards.shape[0]
    legal = env.bitset_to_bool(mask).reshape(n, -1)
    counts = legal.sum(dim=1)
    live = counts > 0
    r = torch.rand(n, generator=gen).to(device)
    pick = (r * counts.clamp(min=1).float()).long()
    cum = legal.cumsum(dim=1)
    move = (cum > pick[:, None]).float().argmax(dim=1)
    move = torch.where(live, move, torch.zeros_like(move))
    promo = torch.randint(0, 4, (n,), generator=gen).to(device)

    nb, nc, nm, _chk = env.play(boards, control, move.to(torch.int64), promo=promo)
    keep = live[:, None]
    return (torch.where(keep, nb, boards),
            torch.where(live, nc, control),
            torch.where(keep, nm, mask))


# --------------------------------------------------------------------------- #
# The result of a match
# --------------------------------------------------------------------------- #

@dataclass
class HalfResult:
    """One colour assignment of a match, one row per opening."""

    white_result: torch.Tensor   # [P] int8, +1 White won, 0 draw, -1 Black won
    finished: torch.Tensor       # [P] bool
    plies: torch.Tensor          # [P] int32
    code: torch.Tensor           # [P] uint8, spec §4.3


@dataclass
class MatchResult:
    """What one pairing did, from A's point of view.

    ``unfinished`` games are **dropped**, not scored. Calling a game that ran past
    `max_plies` a draw is the one adjudication this harness could make without an
    engine, and it is exactly the bias the draw rate is being measured for; the
    count is reported instead so a run where it is not ~0 is visible.
    """

    a_wins: int = 0
    draws: int = 0
    b_wins: int = 0
    unfinished: int = 0
    plies: List[int] = field(default_factory=list)
    codes: Dict[str, int] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.a_wins + self.draws + self.b_wins

    @property
    def a_score(self) -> float:
        """A's score per game, in [0, 1]. ``nan`` if nothing was scored."""
        n = self.games
        return float("nan") if n == 0 else (self.a_wins + 0.5 * self.draws) / n

    @property
    def draw_rate(self) -> float:
        n = self.games
        return float("nan") if n == 0 else self.draws / n

    @property
    def mean_plies(self) -> float:
        return float("nan") if not self.plies else sum(self.plies) / len(self.plies)

    def as_dict(self) -> dict:
        return {"a_wins": self.a_wins, "draws": self.draws, "b_wins": self.b_wins,
                "unfinished": self.unfinished, "games": self.games,
                "a_score": self.a_score, "draw_rate": self.draw_rate,
                "mean_plies": self.mean_plies, "codes": dict(self.codes)}


# --------------------------------------------------------------------------- #
# The match
# --------------------------------------------------------------------------- #

@torch.no_grad()
def play_match(eval_a: Callable, eval_b: Callable,
               openings: torch.Tensor, control: torch.Tensor,
               n_sims: int = 64, max_plies: int = 512,
               search_impl: str = "cuda", device: str | torch.device = "cuda",
               seed: int = 0) -> MatchResult:
    """Play every opening twice, colours swapped, and score it for A.

    ``eval_a`` and ``eval_b`` are evaluators in the sense of
    `search.make_evaluator`: ``(boards, control, rep) -> (policy, promo, value)``.
    Networks are not taken directly, because the league already owns packed
    encoders and because it makes the whole harness testable with stubs.

    ⚠️ **`no_grad` is not optional here.** `CLAUDE.md`: evaluation under autograd
    builds a graph across the run and takes the card out. It is on the function so
    a caller cannot forget it.
    """
    if openings.shape[0] != control.shape[0]:
        raise ValueError(f"openings {tuple(openings.shape)} and control "
                         f"{tuple(control.shape)} disagree on the number of positions")

    out = MatchResult()
    for a_is_white in (True, False):
        half = _play_half(eval_a, eval_b, a_is_white, openings, control,
                          n_sims=n_sims, max_plies=max_plies,
                          search_impl=search_impl, device=device, seed=seed)
        _accumulate(out, half, a_is_white)
    return out


def _accumulate(out: MatchResult, half: HalfResult, a_is_white: bool) -> None:
    """Fold one half into the running score. The only place the sign lives."""
    fin = half.finished
    out.unfinished += int((~fin).sum())
    if not bool(fin.any()):
        return
    white = half.white_result[fin].to(torch.int32)
    # A's result is White's result when A had White, and its negation otherwise.
    a = white if a_is_white else -white
    out.a_wins += int((a > 0).sum())
    out.draws += int((a == 0).sum())
    out.b_wins += int((a < 0).sum())
    out.plies.extend(half.plies[fin].tolist())
    for code, name in TERMINAL_NAMES.items():
        k = int((half.code[fin] == code).sum())
        if k:
            out.codes[name] = out.codes.get(name, 0) + k


@torch.no_grad()
def _play_half(eval_a: Callable, eval_b: Callable, a_is_white: bool,
               openings: torch.Tensor, control: torch.Tensor,
               n_sims: int, max_plies: int, search_impl: str,
               device: str | torch.device, seed: int) -> HalfResult:
    """One colour assignment, `P` games in one batch, played to the end."""
    P = int(openings.shape[0])
    device = torch.device(device)

    Search = search_class(search_impl)
    cfg = eval_config(n_sims, P)
    # The evaluator is replaced before every move; the constructor argument only
    # has to be callable, and `_swap_evaluator` sets the right one before the
    # first search runs.
    search = Search(cfg, eval_a, device=device, seed=seed)
    search.reset(openings.to(device).clone(), control.to(device).clone())

    white_result = torch.zeros((P,), dtype=torch.int8, device=device)
    finished = torch.zeros((P,), dtype=torch.bool, device=device)
    plies = torch.zeros((P,), dtype=torch.int32, device=device)
    code_of = torch.zeros((P,), dtype=torch.uint8, device=device)

    for _ in range(max_plies):
        if bool(finished.all()):
            break
        white_to_move = _swap_evaluator(search, eval_a, eval_b, a_is_white)
        record = search.self_play_move()

        newly = record.done & ~finished
        if bool(newly.any()):
            # `record.result` is from the point of view of the player to move in
            # the position the move *produced*, so -1 there is a win for the player
            # who just moved. The control word after the move says who that is.
            # Same construction as `metrics.self_play_run`.
            new_white = search.game_control > 0
            view = torch.where(new_white, record.result.to(torch.int8),
                               (-record.result).to(torch.int8))
            mask, in_check = env.movegen(search.game_board, search.game_control)
            code, _ = env.terminal(mask, in_check, search.game_control,
                                   search.game_board, search.game_hash,
                                   search.game_ring, search.game_ring_len)
            white_result = torch.where(newly, view, white_result)
            plies = torch.where(newly, search.game_ply, plies)
            code_of = torch.where(newly, code, code_of)
            finished = finished | newly

        # Invariant 8 again: anything that ended — a real game or a throwaway —
        # is restarted, with the side to move that keeps the batch in lockstep.
        dead = search.game_done.nonzero(as_tuple=True)[0]
        if dead.numel():
            boards, ctrl = env.initial_boards(int(dead.numel()), device=device)
            # `white_to_move` was the side to move *before* the move that just
            # happened, so the batch is now on the other one, and the throwaway has
            # to start there too. `initial_boards` gives White to move.
            if white_to_move:
                ctrl = -ctrl
            search.reset(boards, ctrl, rows=dead)

    return HalfResult(white_result=white_result, finished=finished,
                      plies=plies, code=code_of)


def _swap_evaluator(search, eval_a: Callable, eval_b: Callable, a_is_white: bool) -> bool:
    """Point the search at whichever network is to move, and prove it is one network.

    ⚠️ **The assertion is the design.** The reason this harness can run one network
    per batch instead of two is that every row of a batch is at the same ply, so
    every row wants the same network. That holds because all the openings have the
    same length and a finished row is restarted in phase — both of which are one
    edit away from being false. If it ever stops holding, half the positions in the
    batch are evaluated by the wrong network, the games stay legal, the results
    stay plausible, and the Elo curve is quietly meaningless. So it is checked
    every move rather than documented.

    Returns whether White is to move.
    """
    sign = torch.sign(search.game_control)
    if not bool((sign == sign[0]).all()):
        raise AssertionError(
            "evaluation.md §5.4: the batch is not in lockstep — some rows have White "
            "to move and some have Black, so no single network is the one to move. "
            "Either the openings have different lengths or a finished row was "
            "restarted out of phase.")
    white_to_move = bool(sign[0] > 0)
    search.evaluate = eval_a if (white_to_move == a_is_white) else eval_b
    return white_to_move
