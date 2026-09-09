"""D2, the league. `docs/reference/evaluation.md` §5.4.

Four oracles, because four different things can be wrong and none of them is
visible in a result table:

- **the rules** decide whether an opening is a legal, playable position, so
  `random_openings` is checked against `movegen`/`terminal` and not against a
  hand-written list;
- **symmetry** decides the colour pairing: a network played against *itself* scores
  exactly 0.5 per colour-swapped pair whatever the games do, and that is an
  identity rather than an average, so it catches a sign flip, a swapped
  assignment, and a mis-aggregation with one assertion and no statistics;
- **a rigged evaluator** decides the result convention, by forcing a known mate and
  checking who the harness says won;
- **a synthetic ladder** decides the fit: ratings are invented, games are simulated
  from them, and `fit_elo` has to recover the ratings it was never told.

Everything here runs on the CPU with stub evaluators. That is deliberate:
`tests/test_eval.py` learned on 2026-07-31 that a harness whose only real test is
`@pytest.mark.slow` ships broken, because the fast suite stays green while the path
anybody actually runs raises.
"""

from __future__ import annotations

import json
import math
import os
import random

import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.eval import curve as curve_mod
from brokefish.eval import league as league_mod
from brokefish.eval.elo import Edge, elo_half_width, fit_elo
from brokefish.eval.match import (MatchResult, TERMINAL_NAMES, _accumulate,
                                  _play_half, play_match, random_openings)

DEVICE = "cpu"

# ⚠️ Single-threaded on purpose. Every tensor in these tests is a handful of rows,
# so torch's intra-op threads are pure contention: the suite ran 3.5x slower with
# the default pool, and slower still beside a training run that is already using
# the cores. Measured 2026-07-31.
torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# Stub evaluators
# --------------------------------------------------------------------------- #

def flat_eval(boards, control, rep):
    """A network with no opinion: every logit zero, every value a draw.

    The search still plays a full, legal, deterministic game from this — the prior
    is uniform over legal moves and the tree does the rest — which is what makes it
    a usable stand-in for a real net at a thousandth of the cost.
    """
    n = boards.shape[0]
    dev = boards.device
    return (torch.zeros((n, 32, 64), dtype=torch.float32, device=dev),
            torch.zeros((n, 32, 4), dtype=torch.float32, device=dev),
            torch.zeros((n,), dtype=torch.float32, device=dev))


def hashed_eval(salt: int):
    """A deterministic, position-dependent stub. Two salts are two 'networks'.

    The logits are a cheap hash of the board words, so the evaluator is a pure
    function of the position — which is what the search assumes — while two salts
    disagree about essentially every position, so a match between them is a real
    game rather than a mirror.
    """
    def ev(boards, control, rep):
        n = boards.shape[0]
        dev = boards.device
        b = boards.to(torch.int64)
        key = (b * (2654435761 + salt) + salt * 40503) & 0xFFFF
        # [n, 32] -> [n, 32, 64] by mixing in the destination square index.
        sq = torch.arange(64, device=dev, dtype=torch.int64)
        mixed = ((key[:, :, None] * 2246822519 + sq[None, None, :] * 3266489917
                  + salt) & 0xFFFF).to(torch.float32)
        policy = mixed / 8192.0 - 4.0
        promo = (key.to(torch.float32) % 7.0)[:, :, None].expand(n, 32, 4) / 7.0
        value = torch.zeros((n,), dtype=torch.float32, device=dev)
        return policy.contiguous(), promo.contiguous(), value
    return ev


# --------------------------------------------------------------------------- #
# Openings
# --------------------------------------------------------------------------- #

class TestOpenings:

    def test_they_are_legal_playable_positions(self):
        boards, control = random_openings(12, plies=8, seed=1, device=DEVICE)
        assert boards.shape == (12, 32) and control.shape == (12,)
        mask, in_check = env.movegen(boards, control)
        code, _ = env.terminal(mask, in_check, control, boards)
        assert int((code != 0).sum()) == 0, "an opening may not be a finished game"
        n_moves = env.bitset_to_bool(mask).reshape(12, -1).sum(-1)
        assert int(n_moves.min()) >= 2, "a position with one legal reply measures nothing"

    def test_white_is_to_move(self):
        # An even number of plies from the start position. Not a correctness
        # requirement -- colours are assigned explicitly -- but an odd book would
        # make every "White" in the results table the second player.
        _boards, control = random_openings(8, plies=8, seed=2, device=DEVICE)
        assert bool((control > 0).all())

    def test_an_odd_book_is_refused(self):
        with pytest.raises(ValueError, match="even"):
            random_openings(4, plies=7, device=DEVICE)

    def test_they_are_distinct(self):
        boards, control = random_openings(48, plies=8, seed=3, device=DEVICE)
        h = env.hash_position(boards, control).tolist()
        assert len(set(h)) == 48, "evaluation is deterministic, so duplicate openings "\
                                  "are duplicate games and buy no information"

    def test_the_seed_reproduces_the_book(self):
        # The league's openings have to be the same on any machine or two runs of
        # the curve are not comparable; the walk is therefore driven by a CPU
        # generator rather than by the device's.
        a = random_openings(6, plies=8, seed=7, device=DEVICE)
        b = random_openings(6, plies=8, seed=7, device=DEVICE)
        c = random_openings(6, plies=8, seed=8, device=DEVICE)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert not torch.equal(a[0], c[0])

    def test_the_walk_actually_moves(self):
        boards, _ = random_openings(4, plies=8, seed=4, device=DEVICE)
        start, _ = env.initial_boards(4, device=DEVICE)
        assert not torch.equal(boards, start)


# --------------------------------------------------------------------------- #
# The match
# --------------------------------------------------------------------------- #

def _match(ev_a, ev_b, n_openings=2, n_sims=6, max_plies=120, seed=0, plies=8):
    openings, control = random_openings(n_openings, plies=plies, seed=seed, device=DEVICE)
    return play_match(ev_a, ev_b, openings, control, n_sims=n_sims,
                      max_plies=max_plies, search_impl="torch", device=DEVICE, seed=seed)


# A real game on the reference tree costs seconds even with a stub evaluator, and
# most of these assertions are about the *same* match. Module-scoped fixtures so the
# games are played once rather than once per assertion; the suite went from 6 min to
# under 2 that way, which is the difference between a test anybody runs and one
# nobody does.
@pytest.fixture(scope="module")
def self_match():
    ev = hashed_eval(11)
    return _match(ev, ev, n_openings=3)


@pytest.fixture(scope="module")
def duel():
    a, b = hashed_eval(1), hashed_eval(2)
    return _match(a, b), _match(b, a)


class TestColourPairing:

    def test_a_net_against_itself_scores_exactly_half(self, self_match):
        """The end-to-end oracle, and it is an identity rather than an average.

        Both halves of a self-match are literally the same game: same opening, same
        network on both sides, greedy selection. So whatever the game does, the pair
        contributes 1 point out of 2 -- one point to A when A had White and none
        when A had Black, or the reverse, or half and half on a draw.

        A sign flip, a swapped colour assignment or an aggregation that counts one
        half twice all break this exactly, with no sample size to argue about.
        """
        assert self_match.games > 0, "no game finished, so the oracle is vacuous"
        assert self_match.games % 2 == 0, "a finished game must come in pairs here"
        assert self_match.a_score == pytest.approx(0.5, abs=1e-12)

    def test_the_two_halves_are_the_same_game(self):
        """Stronger than the score identity: the games themselves must coincide."""
        ev = hashed_eval(5)
        openings, control = random_openings(2, plies=8, seed=0, device=DEVICE)
        kw = dict(n_sims=6, max_plies=60, search_impl="torch", device=DEVICE, seed=0)
        first = _play_half(ev, ev, True, openings, control, **kw)
        second = _play_half(ev, ev, False, openings, control, **kw)
        assert torch.equal(first.white_result, second.white_result)
        assert torch.equal(first.finished, second.finished)
        assert torch.equal(first.plies, second.plies)
        assert torch.equal(first.code, second.code)

    def test_a_match_is_reproducible(self):
        a, b = hashed_eval(1), hashed_eval(2)
        kw = dict(n_openings=1, n_sims=4, max_plies=40)
        assert _match(a, b, **kw).as_dict() == _match(a, b, **kw).as_dict()

    def test_swapping_the_two_engines_mirrors_the_score(self, duel):
        ab, ba = duel
        assert ab.games == ba.games
        assert ab.a_wins == ba.b_wins and ab.b_wins == ba.a_wins
        assert ab.a_score == pytest.approx(1.0 - ba.a_score)


class TestResultConvention:

    def test_the_winner_of_a_forced_mate_is_the_one_that_mated(self):
        """A rigged evaluator, so the sign is checked against a known outcome.

        `_play_half` maps `record.result` -- which is from the point of view of the
        player to move in the position the move *produced* -- onto White's point of
        view. That double negation is exactly the kind of thing that is wrong for
        six weeks, so here the mating move is found by brute force with the rules,
        the stub is spiked on it, and the harness has to agree that White won.
        """
        # A mate in one for White: Qh5xf7#.
        boards, control = env.from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1")
        mate = _find_mate_in_one(boards, control)
        if mate is None:
            pytest.skip("the fixture is not a mate in one under our own rules")
        slot, square = mate

        def spiked(b, c, r):
            n = b.shape[0]
            dev = b.device
            p = torch.zeros((n, 32, 64), dtype=torch.float32, device=dev)
            p[:, slot, square] = 40.0
            return (p, torch.zeros((n, 32, 4), dtype=torch.float32, device=dev),
                    torch.zeros((n,), dtype=torch.float32, device=dev))

        half = _play_half(spiked, spiked, True, boards, control, n_sims=8,
                          max_plies=4, search_impl="torch", device=DEVICE, seed=0)
        assert bool(half.finished[0]), "the rigged move should have ended the game"
        assert int(half.code[0]) == 1, "the terminal code should be checkmate"
        white_won = int(control[0]) > 0
        assert int(half.white_result[0]) == (1 if white_won else -1)


def _find_mate_in_one(boards, control):
    """`(slot, square)` of a move that mates, by the rules. None if there is none."""
    mask, _ = env.movegen(boards, control)
    legal = env.bitset_to_bool(mask).reshape(1, 32, 64)
    for slot in range(32):
        for square in range(64):
            if not bool(legal[0, slot, square]):
                continue
            move = torch.tensor([slot * 64 + square], dtype=torch.int64)
            nb, nc, nm, chk = env.play(boards.clone(), control.clone(), move,
                                       promo=torch.zeros(1, dtype=torch.int64))
            code, _ = env.terminal(nm, chk, nc, nb)
            if int(code[0]) == 1:
                return slot, square
    return None


class TestLockstep:

    def test_a_batch_out_of_phase_is_refused(self):
        """The assertion that lets one network serve a whole batch.

        If half the rows have White to move and half have Black, no single network
        is 'the one to move', and evaluating the batch with either produces legal
        games, plausible results and a meaningless Elo. So it raises.
        """
        boards, control = random_openings(4, plies=8, seed=0, device=DEVICE)
        control = control.clone()
        control[2] = -control[2]                 # one row a ply out of phase
        with pytest.raises(AssertionError, match="lockstep"):
            _play_half(flat_eval, flat_eval, True, boards, control, n_sims=4,
                       max_plies=2, search_impl="torch", device=DEVICE, seed=0)


class TestAccumulation:

    def _half(self):
        from brokefish.eval.match import HalfResult
        return HalfResult(
            white_result=torch.tensor([1, 0, -1, 0], dtype=torch.int8),
            finished=torch.tensor([True, True, True, False]),
            plies=torch.tensor([10, 20, 30, 0], dtype=torch.int32),
            code=torch.tensor([1, 4, 1, 0], dtype=torch.uint8))

    def test_unfinished_games_are_scored_as_draws(self):
        """Changed 2026-08-02, and the reason is a measurement.

        Dropping them looked conservative and was not: games run long *because*
        neither side can convert, so the dropped set is almost entirely draws and
        removing it inflates the Elo spread. The first `t24h-n256` league dropped
        **35 %**, rising with the strength gap. A draw is also what the fifty-move
        rule would eventually give, what a weak net holding a strong one earned, and
        what `loop.py:_apply_ply_cap` already does on the training side.
        """
        out = MatchResult()
        _accumulate(out, self._half(), a_is_white=True)
        assert (out.a_wins, out.draws, out.b_wins) == (1, 2, 1)
        assert out.unfinished == 1, "the adjudication rate stays visible"
        assert out.games == 4
        assert out.codes == {"checkmate": 2, "threefold": 1, "adjudicated": 1}

        mirror = MatchResult()
        _accumulate(mirror, self._half(), a_is_white=False)
        assert (mirror.a_wins, mirror.draws, mirror.b_wins) == (1, 2, 1)

    def test_dropping_is_still_reachable_for_an_A_B(self):
        out = MatchResult()
        _accumulate(out, self._half(), a_is_white=True, adjudicate=False)
        assert (out.a_wins, out.draws, out.b_wins, out.unfinished) == (1, 1, 1, 1)
        assert out.games == 3
        assert "adjudicated" not in out.codes

    def test_a_win_as_black_is_a_win(self):
        from brokefish.eval.match import HalfResult

        half = HalfResult(white_result=torch.tensor([-1], dtype=torch.int8),
                          finished=torch.tensor([True]),
                          plies=torch.tensor([9], dtype=torch.int32),
                          code=torch.tensor([1], dtype=torch.uint8))
        as_white, as_black = MatchResult(), MatchResult()
        _accumulate(as_white, half, a_is_white=True)
        _accumulate(as_black, half, a_is_white=False)
        assert (as_white.a_wins, as_white.b_wins) == (0, 1)
        assert (as_black.a_wins, as_black.b_wins) == (1, 0)


class TestMatchApi:

    def test_an_odd_game_count_is_refused_by_the_league(self):
        with pytest.raises(ValueError, match="colour-swapped pair"):
            league_mod.run_league([league_mod.PoolEntry("a", 0, None)], games=7)

    def test_mismatched_openings_and_control_are_refused(self):
        boards, control = random_openings(2, plies=8, seed=0, device=DEVICE)
        with pytest.raises(ValueError, match="disagree"):
            play_match(flat_eval, flat_eval, boards, control[:1], search_impl="torch",
                       device=DEVICE)


# --------------------------------------------------------------------------- #
# The calendar
# --------------------------------------------------------------------------- #

class TestCalendar:

    def test_the_offsets_are_sais(self):
        pairs = league_mod.sai_pairings(20, anchor_every=0)
        offsets = {j - i for i, j in pairs}
        assert offsets == set(league_mod.SAI_OFFSETS)

    def test_no_self_pairings_and_no_duplicates(self):
        pairs = league_mod.sai_pairings(20)
        assert all(i != j for i, j in pairs)
        assert len(set(pairs)) == len(pairs)

    def test_the_graph_is_connected(self):
        """A disconnected graph has no identified rating, and `fit_elo` says so."""
        n = 30
        pairs = league_mod.sai_pairings(n)
        seen, stack = {0}, [0]
        adj = {}
        for i, j in pairs:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        while stack:
            for nxt in adj.get(stack.pop(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        assert len(seen) == n

    def test_the_long_edges_exist(self):
        """Without them the graph is a path and the global fit is a chain again."""
        pairs = league_mod.sai_pairings(30, anchor_every=0)
        assert any(j - i == 12 for i, j in pairs)

    def test_the_anchor_reaches_the_far_end(self):
        pairs = league_mod.sai_pairings(60)
        assert (0, 59) in pairs, "a scale whose zero is only measured against the "\
                                 "start of the run is reached by a chain"

    def test_degree_is_around_thirteen(self):
        # SAI ran ~13 pairings per checkpoint; that is where §5.4's sizing comes from.
        pairs = league_mod.sai_pairings(40)
        deg = {}
        for i, j in pairs:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        middle = [deg[k] for k in range(13, 27)]
        assert all(11 <= d <= 14 for d in middle), deg


class TestSubsample:

    def test_it_keeps_the_ends(self):
        got = league_mod.subsample(list(range(100)), 5)
        assert got[0] == 0 and got[-1] == 99 and len(got) == 5

    def test_a_short_list_is_untouched(self):
        assert league_mod.subsample([1, 2, 3], 10) == [1, 2, 3]
        assert league_mod.subsample([1, 2, 3], None) == [1, 2, 3]

    def test_it_is_evenly_spaced_and_never_by_score(self):
        # §2: which checkpoints get rated is a cost decision; which one is best is
        # what evaluation may not act on. Spacing by index cannot smuggle one in.
        got = league_mod.subsample(list(range(101)), 11)
        assert got == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #

def _simulate(true_elo, pairs, games, draw_rate, rng):
    """Games generated from known ratings, with a symmetric draw band.

    ⚠️ **The draws are carved out of the score, not layered on top of it.** The
    expected score is `1/(1 + 10^(-Δ/400))` — `evaluation.md` §9's convention, and
    the one `fit_elo` inverts — and the draw band is then split symmetrically
    around it, so the mean score is unchanged. Drawing a fraction `d` of games
    *first* and deciding the rest by Δ, which is the obvious way to write this,
    silently generates a **different** model whose expected score is
    `0.5 + (1-d)(p - 0.5)`; a fitter that recovers Δ correctly then looks like it
    compresses the scale by `1 - d`. That cost half an hour on 2026-07-31.
    """
    edges = []
    for a, b in pairs:
        s = 1.0 / (1.0 + 10.0 ** (-(true_elo[a] - true_elo[b]) / 400.0))
        d = min(draw_rate, 2.0 * min(s, 1.0 - s) * 0.999)
        pw, pl = s - d / 2.0, 1.0 - s - d / 2.0
        w = dr = l = 0
        for _ in range(games):
            u = rng.random()
            if u < pw:
                w += 1
            elif u < pw + d:
                dr += 1
            else:
                l += 1
        edges.append(Edge(a=a, b=b, a_wins=w, draws=dr, b_wins=l))
    return edges


class TestEloFit:

    def test_the_anchor_is_pinned_at_zero(self):
        edges = [Edge("anchor", "x", 0, 0, 40)]
        fit = fit_elo(edges, anchor="anchor")
        assert fit.elo["anchor"] == 0.0
        assert fit.elo["x"] > 0.0

    def test_it_recovers_a_synthetic_ladder(self):
        """The oracle for the fit: ratings invented, games simulated, fit blind."""
        rng = random.Random(20260731)
        n = 12
        names = ["anchor"] + [f"cp{i}" for i in range(1, n)]
        true = {names[i]: 60.0 * i for i in range(n)}
        pairs = [(names[i], names[j]) for i, j in league_mod.sai_pairings(n)]
        edges = _simulate(true, pairs, games=400, draw_rate=0.3, rng=rng)

        fit = fit_elo(edges, anchor="anchor")
        assert fit.converged
        # Ordering is exact at this separation and this sample size.
        assert fit.ordered() == names
        for name in names:
            assert abs(fit.elo[name] - true[name]) < 45.0, (name, fit.elo[name], true[name])

    def test_the_interval_covers_the_truth(self):
        rng = random.Random(4)
        names = ["anchor"] + [f"cp{i}" for i in range(1, 9)]
        true = {names[i]: 80.0 * i for i in range(len(names))}
        pairs = [(names[i], names[j]) for i, j in league_mod.sai_pairings(len(names))]
        covered = 0
        trials = 12
        for t in range(trials):
            edges = _simulate(true, pairs, games=300, draw_rate=0.4, rng=rng)
            fit = fit_elo(edges, anchor="anchor")
            covered += sum(abs(fit.elo[n] - true[n]) <= fit.ci95(n)
                           for n in names if n != "anchor")
        total = trials * (len(names) - 1)
        assert covered / total > 0.80, f"{covered}/{total} inside the 95 % interval"

    def test_draws_shrink_the_dispersion(self):
        """`dispersion` should land near `1 - d`, which is what §9's √(1−d) is."""
        rng = random.Random(9)
        names = ["anchor", "a", "b", "c"]
        true = {n: 0.0 for n in names}
        pairs = [(names[i], names[j]) for i in range(4) for j in range(i + 1, 4)]
        for d in (0.0, 0.8):
            edges = _simulate(true, pairs, games=2000, draw_rate=d, rng=rng)
            fit = fit_elo(edges, anchor="anchor")
            assert abs(fit.dispersion - (1.0 - d)) < 0.12, (d, fit.dispersion)

    def test_an_all_draw_league_does_not_claim_zero_uncertainty(self):
        """Found by a real smoke league on 2026-07-31, which reported ±0 Elo.

        All draws means zero Pearson residual, so the estimated dispersion is
        exactly 0 and the corrected interval collapses to 0 — infinite precision
        from a sample that says nothing about who is better. The correction is only
        applied once there are decisive games to estimate it from.
        """
        edges = [Edge("anchor", "x", 0, 40, 0), Edge("x", "y", 0, 40, 0)]
        fit = fit_elo(edges, anchor="anchor")
        assert fit.dispersion == pytest.approx(0.0)
        assert not fit.dispersion_applied and fit.decisive == 0
        assert fit.se["x"] > 0.0 and fit.ci95("x") > 1.0
        assert fit.se["x"] == fit.se_raw["x"]

    def test_the_correction_switches_on_once_there_are_decisive_games(self):
        rng = random.Random(2)
        edges = _simulate({"anchor": 0.0, "x": 0.0}, [("anchor", "x")],
                          games=600, draw_rate=0.8, rng=rng)
        fit = fit_elo(edges, anchor="anchor")
        assert fit.dispersion_applied and fit.decisive >= 30
        assert fit.se["x"] < fit.se_raw["x"]

    def test_the_corrected_interval_is_the_tighter_one_when_draws_are_common(self):
        rng = random.Random(10)
        edges = _simulate({"anchor": 0.0, "x": 0.0}, [("anchor", "x")],
                          games=1000, draw_rate=0.8, rng=rng)
        fit = fit_elo(edges, anchor="anchor")
        assert fit.se["x"] < fit.se_raw["x"]
        # And it should be close to §9's closed form for one edge.
        closed = elo_half_width(1000, 0.8)
        assert 0.6 < fit.ci95("x") / closed < 1.6, (fit.ci95("x"), closed)

    def test_an_undefeated_player_still_has_a_finite_rating(self):
        """The failure the phantom prior exists for, and it *will* happen: the
        first real checkpoint against the random-init anchor is plausibly 100 %."""
        edges = [Edge("anchor", "x", 0, 0, 200)]
        fit = fit_elo(edges, anchor="anchor")
        assert math.isfinite(fit.elo["x"])
        assert fit.elo["x"] > 400.0

    def test_a_zero_prior_is_refused(self):
        with pytest.raises(ValueError, match="prior must be positive"):
            fit_elo([Edge("anchor", "x", 1, 0, 0)], anchor="anchor", prior=0.0)

    def test_a_disconnected_player_is_named_rather_than_silently_wrong(self):
        edges = [Edge("anchor", "a", 5, 5, 5), Edge("b", "c", 5, 5, 5)]
        # b and c are connected to each other but not to the anchor: their ratings
        # are identified only by the phantom, which is exactly what it is for.
        fit = fit_elo(edges, anchor="anchor")
        assert math.isfinite(fit.elo["b"]) and math.isfinite(fit.elo["c"])

    def test_a_self_pairing_is_refused(self):
        with pytest.raises(ValueError, match="cannot play itself"):
            fit_elo([Edge("x", "x", 1, 0, 0)], anchor="anchor")

    def test_predict_is_the_logistic_of_the_gap(self):
        fit = fit_elo([Edge("anchor", "x", 10, 20, 30)], anchor="anchor")
        assert fit.predict("x", "anchor") == pytest.approx(1.0 - fit.predict("anchor", "x"))
        assert fit.predict("x", "x") == pytest.approx(0.5)

    def test_a_real_sized_budget_ladder_league_converges(self):
        """The graph that broke `bt-mm/v1`: ~100 players, four runs interleaved by
        step, three budgets each, the untrained ladder, `random` at zero.

        Every joint league of this shape hit the 10 000-sweep cap until 2026-09-09,
        because the pool is tied to the pinned anchor only through `random`'s
        lopsided games and the phantom, so its softest mode is the whole pool
        sliding together and a coordinate iteration contracts it at ~0.998 per
        sweep. This asserts two things the cap never did: the fit stops on its own
        tolerance, and the point it stops at is the maximum of the likelihood, by
        the stationarity identity every free player satisfies there, checked
        without the optimiser's help.
        """
        rng = random.Random(20260909)
        runs = ["a", "b", "c", "d"]
        steps = [201 + 600 * i for i in range(8)]
        budgets = (16, 64, 256)
        pool = [league_mod.PoolEntry(name="random", step=0, path=None, run="", sims=0)]
        pool += [league_mod.PoolEntry(name=f"init:n{k}", step=0, path="init", run="",
                                      sims=k) for k in (1, 4, 16, 64)]
        entries = [league_mod.PoolEntry(name=f"{r}@{st}:n{k}", step=st, path=f"{r}{st}",
                                        run=r, sims=k)
                   for r in runs for st in steps for k in budgets]
        entries.sort(key=lambda e: (e.step, e.sims, e.run))
        pool += entries
        assert len(pool) == 101

        def truth(e):
            if e.path is None:
                return 0.0
            if e.run == "":
                return 40.0 * math.log2(e.sims + 1)
            speed = {"a": 1.0, "b": 1.1, "c": 0.9, "d": 1.05}[e.run]
            return (1500.0 * (1.0 - math.exp(-speed * e.step / 3000.0))
                    + 250.0 * math.log2(e.sims / 16.0))
        true = {e.name: truth(e) for e in pool}
        n = len(pool)
        pairs = sorted(set(league_mod.sai_pairings(n, anchor_every=4))
                       | set(league_mod.budget_ladder_pairs(pool))
                       | set(league_mod.endpoint_pairs(pool)))
        pairs = [(pool[i].name, pool[j].name) for i, j in pairs]
        edges = _simulate(true, pairs, games=36, draw_rate=0.7, rng=rng)

        fit = fit_elo(edges, anchor="random")
        assert fit.converged, fit.iterations
        assert fit.iterations < 40, fit.iterations

        # Stationarity of the Bradley-Terry likelihood with the phantom: each free
        # player's actual score equals its expected score under the fit.
        gamma = {k: 10.0 ** (v / 400.0) for k, v in fit.elo.items()}
        expected = {k: 0.0 for k in gamma}
        for e in edges:
            p = gamma[e.a] / (gamma[e.a] + gamma[e.b])
            expected[e.a] += e.games * p
            expected[e.b] += e.games * (1.0 - p)
        for k in gamma:
            if k == "random":
                continue
            expected[k] += gamma[k] / (gamma[k] + 1.0)      # prior = 1 vs Elo 0
            actual = fit.score[k] + 0.5
            assert abs(actual - expected[k]) < 1e-6, (k, actual, expected[k])

        # And the optimum is the truth. ⚠️ Checked at a small prior on purpose: at
        # `prior = 1` this graph's hundred phantoms, each pulling a 1000+ Elo player
        # toward zero with the full half-point a saturated game carries, outweigh
        # `random`'s ~200 informative games, and the whole pool sits hundreds of Elo
        # too low (measured 2026-09-09, on this league and on the saved ones). That
        # is a property of the prior, not of the optimiser, and it is not what this
        # test is about; the default is unchanged and the finding is reported.
        small = fit_elo(edges, anchor="random", prior=0.01)
        assert small.converged and small.iterations < 40, small.iterations
        misses = [k for k in gamma if k != "random"
                  and abs(small.elo[k] - true[k]) > 4.0 * small.se[k] + 5.0]
        assert len(misses) <= 2, misses

    def test_the_global_fit_beats_a_single_edge(self):
        """Why §5.2 wants one fit over the graph rather than a chain of matches."""
        rng = random.Random(3)
        names = ["anchor"] + [f"cp{i}" for i in range(1, 15)]
        true = {names[i]: 30.0 * i for i in range(len(names))}
        pairs = [(names[i], names[j]) for i, j in league_mod.sai_pairings(len(names))]
        edges = _simulate(true, pairs, games=36, draw_rate=0.8, rng=rng)
        fit = fit_elo(edges, anchor="anchor")
        one_edge = elo_half_width(36, 0.8)
        assert fit.ci95("cp7") < one_edge, (fit.ci95("cp7"), one_edge)


class TestSizing:

    def test_the_closed_form_matches_evaluation_md_table(self):
        # §9's table: 500 games at d = 0.4 resolves ±24 Elo.
        assert elo_half_width(500, 0.4) == pytest.approx(24.0, abs=1.0)
        assert elo_half_width(1000, 0.7) == pytest.approx(12.0, abs=1.0)

    def test_no_games_is_infinite(self):
        assert elo_half_width(0, 0.5) == float("inf")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def league():
    """One whole league on stub engines, played once for the class below."""
    # ⚠️ Index 0 is `random` at `sims = 0`: uniformly random legal play, no network,
    # which is the zero of the scale since 2026-08-08 (§5.1). It is a real player here
    # rather than a stub, because `random_move` is what the anchor actually is.
    pool = [league_mod.PoolEntry(league_mod.RANDOM_NAME, 0, None, sims=0)]
    pool += [league_mod.PoolEntry("init:n4", 0, "anchor.pt", sims=4)]
    pool += [league_mod.PoolEntry(f"r@{s}:n4", s, f"r-{s}.pt", run="r", sims=4)
             for s in (50, 100, 150)]
    # A distinct stub per player, so the games are real games. Keyed by name rather
    # than by `hash`, which is salted per process and would make this irreproducible.
    loader = lambda path: hashed_eval(sum(map(ord, path)) % 9973 + 1)  # noqa: E731
    cost = {"r": [(50, {"euros": 1.0, "training_seconds": 600.0}),
            (150, {"euros": 3.0, "training_seconds": 1800.0})]}
    report = league_mod.run_league(
        pool, games=2, n_sims=4, max_plies=40, anchor_every=1,
        search_impl="torch", device=DEVICE, seed=0, cost=cost,
        engine_loader=loader)
    return pool, report


class TestRunLeague:
    """The whole harness on stub engines: pool in, curve records out.

    `engine_loader` is injectable precisely so this can run without a checkpoint,
    without a card and in seconds. The point is the wiring — every league bug so
    far in this repository has been a rename or a mis-plumbed argument, not an
    arithmetic error, and those are invisible to unit tests on either side.
    """

    def test_it_produces_one_curve_point_per_player(self, league):
        pool, report = league
        assert len(report["curve"]) == len(pool)
        ids = [r["checkpoint_id"] for r in report["curve"]]
        assert ids == [p.name for p in pool]
        assert report["fit"]["anchor"] == league_mod.RANDOM_NAME
        assert report["curve"][0]["elo"] == 0.0, "the anchor is the zero of the scale"

    def test_every_game_is_scored_none_are_lost(self, league):
        """Stronger than it was before adjudication landed (2026-08-02).

        When unfinished games were dropped the invariant was
        `games + unfinished == asked`, i.e. some games vanished from the fit. Now
        every game is scored — `unfinished` counts how many of them were *decided
        by the ply cap* rather than by the rules, so it is a diagnostic rather than
        a leak.
        """
        _pool, report = league
        asked = report["config"]["games_per_pairing"]
        for m in report["matches"]:
            assert m["games"] == asked, "no game may be lost from the fit"
            assert m["unfinished"] <= m["draws"], "an adjudicated game is a draw"
        assert report["config"]["games_played"] == asked * len(report["matches"])

    def test_the_cost_axis_is_joined_by_this_writer(self, league):
        # §5.3: euros and Elo are written by the same writer or they get joined by
        # hand six weeks later.
        _pool, report = league
        by_id = {r["checkpoint_id"]: r for r in report["curve"]}
        assert by_id[league_mod.RANDOM_NAME]["euros_spent"] == 0.0
        assert by_id["r@100:n4"]["euros_spent"] == 1.0     # last record at or before 100
        assert by_id["r@150:n4"]["training_seconds"] == 1800.0

    def test_the_report_renders_as_a_curve(self, league):
        _pool, report = league
        text = curve_mod.format_curve(report)
        assert "r@150" in text and "self-anchored" in text
        assert curve_mod.cost_axis(report) == "euros_spent"
        assert curve_mod.slope_per_decade(report) is not None


# --------------------------------------------------------------------------- #
# The cost axis
# --------------------------------------------------------------------------- #

class TestCostAxis:

    def test_it_reads_both_log_schemas(self, tmp_path):
        """`train/loop.py` dropped its `phase/` wrapper on 2026-07-31.

        Reading only the new spelling would make the cost axis silently null for
        every log already on disk, and a null x-axis reads as "this run had no
        euros" rather than as a bug.
        """
        old = tmp_path / "old.jsonl"
        old.write_text(json.dumps({"step": 10, "phase/euros/training": 4.0,
                                   "phase/euros/training_seconds": 40.0}) + "\n")
        new = tmp_path / "new.jsonl"
        new.write_text(json.dumps({"step": 10, "euros/training": 4.0,
                                   "euros/training_seconds": 40.0}) + "\n")
        for path in (old, new):
            series = league_mod.training_series(str(path))
            assert league_mod.series_at(series, 10) == {"euros": 4.0,
                                                        "training_seconds": 40.0}

    def test_it_joins_by_the_last_step_at_or_before(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text("\n".join(json.dumps(
            {"step": s, "euros/training": s * 0.5,
             "euros/training_seconds": s * 10.0})
            for s in (10, 20, 30)) + "\n")
        series = league_mod.training_series(str(path))
        assert league_mod.series_at(series, 25)["euros"] == 10.0
        assert league_mod.series_at(series, 30)["training_seconds"] == 300.0
        assert league_mod.series_at(series, 5) == {}

    def test_the_anchor_costs_exactly_zero(self):
        assert league_mod.series_at([], 0) == {"euros": 0.0, "training_seconds": 0.0}

    def test_a_missing_log_is_empty_rather_than_fatal(self, tmp_path):
        assert league_mod.training_series(str(tmp_path / "nope.jsonl")) == []

    def test_a_truncated_line_is_skipped(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text('{"step": 1, "euros/training": 1.0}\n{"step": 2, "eur\n')
        assert len(league_mod.training_series(str(path))) == 1


class TestCurve:

    def _report(self, euros=True):
        rows = []
        for i, step in enumerate((0, 50, 100, 200)):
            rows.append({"checkpoint_id": "anchor" if i == 0 else f"r@{step}",
                         "step": step,
                         "euros_spent": float(step) * 0.1 if euros else None,
                         "training_seconds": float(step) * 12.0,
                         "games_played": 100, "elo": 100.0 * i,
                         "ci95": 20.0, "se": 10.2, "se_raw": 21.0,
                         "n_sims": 64, "draw_rate": 0.7,
                         "fit_version": "bt-mm/v1", "timestamp": 0.0})
        return {"config": {"pairings": 6, "games_played": 216, "n_sims": 64},
                "fit": {"anchor": "anchor"}, "curve": rows}

    def test_the_table_names_the_scale(self):
        text = curve_mod.format_curve(self._report())
        assert "self-anchored" in text
        assert "pinned" in text, "the anchor has no interval of its own"

    def test_the_slope_prefers_euros_and_falls_back_to_seconds(self):
        assert curve_mod.cost_axis(self._report(euros=True)) == "euros_spent"
        assert curve_mod.cost_axis(self._report(euros=False)) == "training_seconds"

    def test_a_flat_cost_axis_is_none_rather_than_a_number(self):
        report = self._report(euros=False)
        for r in report["curve"]:
            r["training_seconds"] = 0.0
        assert curve_mod.cost_axis(report) is None
        assert curve_mod.slope_per_decade(report) is None

    def test_the_slope_is_elo_per_decade(self):
        report = {"config": {}, "fit": {"anchor": "a"}, "curve": [
            {"checkpoint_id": "a", "step": 1, "euros_spent": 1.0, "elo": 0.0},
            {"checkpoint_id": "b", "step": 2, "euros_spent": 10.0, "elo": 500.0},
            {"checkpoint_id": "c", "step": 3, "euros_spent": 100.0, "elo": 1000.0}]}
        assert curve_mod.slope_per_decade(report) == pytest.approx(500.0)

    def test_the_csv_has_the_columns_of_5_3(self, tmp_path):
        out = tmp_path / "curve.csv"
        curve_mod.write_csv(self._report(), str(out))
        header = out.read_text().splitlines()[0].split(",")
        for column in ("checkpoint_id", "euros_spent", "elo", "ci95", "n_sims",
                       "draw_rate", "fit_version"):
            assert column in header


# --------------------------------------------------------------------------- #
# The boundary
# --------------------------------------------------------------------------- #

class TestAntiSelection:
    """`evaluation.md` §2, enforced rather than documented.

    The prohibition that will actually get violated is checkpoint selection --
    "keep the checkpoint with the best league rating" is distillation through a
    one-bit channel and it looks like good practice. The structural guarantee is
    that the training package cannot see the evaluation package at all, so the
    violation cannot be written without deleting a test.
    """

    def _train_sources(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "brokefish", "train")
        return [os.path.join(root, f) for f in sorted(os.listdir(root))
                if f.endswith(".py")]

    def test_the_probe_hands_the_training_process_no_score(self):
        """The guarantee, moved from the module graph to the dataflow (2026-08-02).

        The old rule was "train must not import eval", a proxy that also forced the
        puzzle measurement into a second process. The real rule is that no evaluation
        score may reach a training decision, and it is enforced directly now:
        `PuzzleProbe.run` writes to the logger and **returns None**, so there is no
        value inside the training process to threshold, compare, or select on.
        """
        import inspect
        from brokefish.eval.watch import PuzzleProbe

        # `from __future__ import annotations` stringifies it, so accept both forms.
        sig = inspect.signature(PuzzleProbe.run)
        assert sig.return_annotation in (None, "None", type(None)), \
            "run must be annotated -> None; the absence of a return value IS the guard"
        src = inspect.getsource(PuzzleProbe.run)
        assert "return None" not in src.replace("-> None", "")
        for line in src.splitlines():
            body = line.strip()
            assert not (body.startswith("return ") and body != "return"), \
                f"PuzzleProbe.run must return nothing, found: {body}"

    def test_the_loop_discards_the_probe_and_never_branches_on_it(self):
        """The call site may not bind the probe's output to a name, nor test it."""
        import re
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "brokefish", "train", "loop.py")).read()
        calls = [l.strip() for l in src.splitlines() if ".probe.run(" in l]
        assert calls, "the loop should call the probe"
        for c in calls:
            assert re.match(r"^(self\.)?probe\.run\(|^self\.probe\.run\(", c), \
                f"the probe's result must be discarded, not bound: {c}"
        # And nothing in the loop may read a puzzle metric back out.
        assert "puzzles/" not in src, \
            "the loop must not name a puzzle metric; it cannot act on what it cannot see"

    def test_training_still_imports_no_league_machinery(self):
        """Relaxed for the probe only. The league is still entirely out of reach."""
        offenders = []
        for path in self._train_sources():
            src = open(path).read()
            for needle in ("eval.league", "eval.elo", "eval.curve", "eval.match"):
                if needle in src:
                    offenders.append((os.path.basename(path), needle))
        assert not offenders, (
            f"{offenders} reach into the league. A training loop that reads an Elo "
            f"rating is selecting checkpoints on it (evaluation.md §2).")

    def test_training_never_reads_a_league_report(self):
        # Deliberately the *artefacts*, not the word "Elo": `train/sync.py` says
        # "the Elo curve is flat for a reason no diagnostic reports", which is the
        # comment that exists to prevent a bug rather than a violation. Naming the
        # things the league writes is unambiguous.
        offenders = []
        for path in self._train_sources():
            src = open(path).read()
            for needle in ("league-", "logs/league", "curve.json", "curve.csv",
                           "fit_elo", "EloFit", "play_match"):
                if needle in src:
                    offenders.append((os.path.basename(path), needle))
        assert not offenders, offenders

    def test_the_league_writes_only_where_it_is_allowed(self):
        """The league's write locations are derived, not typed, and pinned here.

        ⚠️ Since 2026-08-14 a league writes into `runs/<run>/`, which is the **same
        folder the training loop writes**. That is safe, and the reason is the sibling
        test above rather than the directory: no module under `brokefish/train/` so
        much as mentions `league-`, so nothing training reads can see a rating. The
        boundary evaluation.md §2 protects is about *what reads what*, not about which
        directory the bytes sit in -- and it never was, since both used to share
        `logs/`.

        Asserting the resolved paths rather than the flag's literal default also means
        this bites if `brokefish/paths.py` ever moves the layout underneath it.
        """
        from brokefish import paths

        parser = league_mod.build_parser()
        args = parser.parse_args(["--run", "x"])
        assert args.out is None            # -> runs/x/league-x.json
        assert args.checkpoints is None    # -> runs/x/checkpoints, per run
        assert paths.checkpoint_dir("x") == os.path.join("runs", "x", "checkpoints")
        assert paths.artifact("x", "league-x.json") == os.path.join(
            "runs", "x", "league-x.json")
        # A joint league lands in the first run named, never in a shared bucket.
        assert paths.joint_dir(["x", "y"]) == os.path.join("runs", "x")
        # ⚠️ The anchor is the one thing that must NOT be per-run: every Elo scale is
        # anchored to this single file, and a copy per run would let two leagues
        # anchor to two different networks without either saying so.
        assert args.anchor == os.path.join("checkpoints", "anchor.pt")


class TestTerminalNames:
    """spec §4.3 has one table; the code must have one copy of it.

    Three modules had grown private `{1: "checkmate", ...}` dicts by 2026-07-31 and a
    fourth was about to. That is how a code eventually gets two names in two places
    and a plot lies about what it is showing.
    """

    def test_the_names_match_the_spec_table(self):
        assert env.TERMINAL_NAMES == {
            0: "unfinished", 1: "checkmate", 2: "stalemate",
            3: "fifty_move", 4: "threefold", 5: "insufficient"}

    def test_the_indices_are_the_env_constants(self):
        for const, name in ((env.NONE, "unfinished"), (env.CHECKMATE, "checkmate"),
                            (env.STALEMATE, "stalemate"), (env.FIFTY_MOVE, "fifty_move"),
                            (env.REPETITION, "threefold"),
                            (env.INSUFFICIENT, "insufficient")):
            assert env.TERMINAL_NAMES[const] == name

    def test_match_and_search_share_the_one_copy(self):
        from brokefish.eval import match as match_mod
        from brokefish.search import torch_impl as search_mod
        assert match_mod.TERMINAL_NAMES is env.TERMINAL_NAMES
        assert search_mod._TERMINAL_NAMES_ORDERED == [
            env.TERMINAL_NAMES[i] for i in range(6)]

    def test_nobody_restates_the_mapping(self):
        # A literal `"checkmate"` next to a literal `"threefold"` in the same file is
        # a private copy of spec §4.3 unless that file is env/torch_impl.py.
        import glob
        offenders = []
        for path in glob.glob("brokefish/**/*.py", recursive=True):
            if path.endswith(os.path.join("env", "torch_impl.py")):
                continue
            src = open(path).read()
            if '"checkmate"' in src and '"threefold"' in src:
                offenders.append(path)
        assert not offenders, f"{offenders} restate spec §4.3; import env.TERMINAL_NAMES"

    def test_the_search_reports_codes_by_name(self):
        from brokefish.search.torch_impl import SearchStats
        snap = SearchStats(d_max=4, device="cpu").snapshot()
        for name in env.TERMINAL_NAMES.values():
            assert f"terminal_{name}" in snap, name
        assert "terminal_codes" not in snap, "the by-index list is gone"


# --------------------------------------------------------------------------- #
# §5.1 — the zero is uniformly random legal play
# --------------------------------------------------------------------------- #

class TestTheRandomAnchor:
    """The anchor rebased 2026-08-08. What it must be, and what it must not depend on."""

    def _search(self, B=32, n=8, seed=0):
        import torch
        from brokefish.nn.model import BrokefishNet
        from brokefish.search import Search, SearchConfig, make_evaluator
        torch.manual_seed(0)
        net = BrokefishNet().to(DEVICE).eval()
        s = Search(SearchConfig(n=n, B=B, E=96), evaluate=make_evaluator(net),
                   device=DEVICE, seed=seed)
        b, c = env.initial_boards(B, device=DEVICE)
        s.reset(b, c)
        return s

    def test_every_move_it_plays_is_legal(self):
        """Over a long random game, judged by `movegen` and not by the search."""
        import torch
        from brokefish.search.torch_impl import MOVE_BITS, PROMO_SHIFT
        s = self._search(B=32)
        illegal = 0
        with torch.no_grad():
            for _ in range(80):
                s.reset_finished()
                legal = env.bitset_to_bool(env.movegen(s.game_board, s.game_control)[0])
                promo_ok = s._promotion_targets(s.game_board, s.game_control) & legal
                lab = s.random_move().played.to(torch.int64)
                mv, pr = lab & ((1 << MOVE_BITS) - 1), (lab >> PROMO_SHIFT) & 0b11
                rows = torch.arange(32, device=DEVICE)
                illegal += int((~legal[rows, mv // 64, mv % 64]).sum())
                # A promotion *type* on a move that is not a promotion is illegal too.
                illegal += int(((pr > 0) & ~promo_ok[rows, mv // 64, mv % 64]).sum())
        assert illegal == 0

    def test_it_is_uniform_over_the_move_set(self):
        """χ² against the flat distribution on the opening position (20 legal moves).

        Uniformity is the whole definition, so it is tested as a distribution rather
        than as "it played several different moves".
        """
        import collections
        import torch
        s = self._search(B=8192, seed=1)
        with torch.no_grad():
            played = s.random_move().played.tolist()
        counts = collections.Counter(played)
        legal = int(env.bitset_to_bool(
            env.movegen(s.game_board[:1] * 0 + s.game_board[:1], s.game_control[:1])[0]).sum())
        assert len(counts) == 20 and legal == 20, (len(counts), legal)
        exp = 8192 / 20
        chi2 = sum((k - exp) ** 2 / exp for k in counts.values())
        # 19 df: the 99.9th percentile is 43.8. A biased sampler blows past it.
        assert chi2 < 43.8, f"chi2 = {chi2:.1f} on 19 df — not uniform"

    def test_it_never_touches_the_network(self):
        """The property the whole rebase rests on: independent of the *network*."""
        import torch
        from brokefish.search import Search, SearchConfig

        def refuse(*_a, **_k):
            raise AssertionError("the random player evaluated a position")

        s = Search(SearchConfig(n=8, B=8, E=96), evaluate=refuse, device=DEVICE, seed=0)
        b, c = env.initial_boards(8, device=DEVICE)
        s.reset(b, c)
        with torch.no_grad():
            for _ in range(30):
                s.reset_finished()
                s.random_move()

    def test_it_does_not_move_when_the_search_config_does(self):
        """⚠️ The failure that motivated the rebase.

        The old anchor was a network *plus a search*, so §6.1a's root terminal sweep
        made it stronger while its file was untouched. A random player built as a
        one-simulation search would inherit exactly that. This asserts the sequence of
        moves is identical under every search flag, from the same seed.
        """
        import torch
        from brokefish.nn.model import BrokefishNet
        from brokefish.search import Search, SearchConfig, make_evaluator

        def moves(**flags):
            torch.manual_seed(0)
            net = BrokefishNet().to(DEVICE).eval()
            s = Search(SearchConfig(n=8, B=16, E=96, **flags),
                       evaluate=make_evaluator(net), device=DEVICE, seed=3)
            b, c = env.initial_boards(16, device=DEVICE)
            s.reset(b, c)
            out = []
            with torch.no_grad():
                for _ in range(40):
                    s.reset_finished()
                    out.append(s.random_move().played.clone())
            return torch.stack(out)

        base = moves()
        for flags in ({"root_terminal_sweep": False}, {"terminal_collapse": True},
                      {"eps": 0.0}, {"pb_c_init": 99.0}, {"tau_plies": 0}):
            assert torch.equal(base, moves(**flags)), \
                f"the anchor moved when {flags} changed — it is not rules-only"

    def test_a_terminal_position_is_refused_rather_than_played(self):
        """`multinomial` on an all-zero row returns index 0 — an illegal 'move', silently."""
        import torch
        from brokefish.nn.model import BrokefishNet
        from brokefish.search import Search, SearchConfig, make_evaluator
        torch.manual_seed(0)
        net = BrokefishNet().to(DEVICE).eval()
        s = Search(SearchConfig(n=8, B=2, E=96), evaluate=make_evaluator(net),
                   device=DEVICE, seed=0, check_invariants=False)
        # Black is checkmated: no legal move at all.
        b, c = env.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        s.reset(b.to(DEVICE).expand(2, -1).contiguous(), c.to(DEVICE).expand(2).contiguous())
        with torch.no_grad():
            with pytest.raises(AssertionError, match="no legal move"):
                s.random_move()

    def test_the_record_it_emits_can_never_become_training_data(self):
        import torch
        s = self._search(B=8)
        with torch.no_grad():
            rec = s.random_move()
        assert int(rec.policy_len.sum()) == 0
        # train/loss.py refuses a zero-length policy, which is the second lock.
        from brokefish.train import loss as loss_mod
        assert "policy_len > 0" in open(loss_mod.__file__).read()


# --------------------------------------------------------------------------- #
# §5.1a — a player is a (network, budget) pair
# --------------------------------------------------------------------------- #

class TestTheBudgetAxis:

    def test_the_pool_is_ordered_by_strength_so_the_calendar_bridges_it(self, tmp_path):
        pool = _pool_with(tmp_path, steps=(100, 200, 300), sims=64,
                          ladder=(1, 4, 16, 64))
        names = [p.name for p in pool]
        assert names[0] == league_mod.RANDOM_NAME
        assert names[1:5] == ["init:n1", "init:n4", "init:n16", "init:n64"]
        assert [p.sims for p in pool[:5]] == [0, 1, 4, 16, 64]
        # The ladder rungs sit inside the small SAI offsets of the anchor, which is
        # what makes the zero estimable instead of hanging off a saturated edge.
        pairs = set(league_mod.sai_pairings(len(pool), anchor_every=4))
        assert {(0, 1), (0, 2), (0, 3)} <= pairs

    def test_budget_variants_of_one_checkpoint_always_play_each_other(self, tmp_path):
        pool = _pool_with(tmp_path, steps=(100, 200, 300), sims=64,
                          ladder=(64,), grid=(16, 256), grid_points=3)
        ladder = league_mod.budget_ladder_pairs(pool)
        by_name = {i: p.name for i, p in enumerate(pool)}
        got = {tuple(sorted((by_name[i], by_name[j]))) for i, j in ladder}
        for step in (100, 200, 300):
            for a, b in ((16, 64), (16, 256), (64, 256)):
                key = tuple(sorted((f"r@{step}:n{a}", f"r@{step}:n{b}")))
                assert key in got, f"{key} never plays, so the sims axis is unmeasured"

    def test_the_random_player_has_no_budget_edges(self, tmp_path):
        pool = _pool_with(tmp_path, steps=(100,), sims=64, ladder=(1, 64))
        for i, j in league_mod.budget_ladder_pairs(pool):
            assert pool[i].path is not None and pool[j].path is not None

    def test_a_pool_without_the_anchor_is_refused(self):
        """⚠️ `fit_elo` unions the anchor in, so it would invent a phantom at Elo 0
        that played no games and pin every rating to the prior. Silently."""
        with pytest.raises(ValueError, match="phantom"):
            league_mod.run_league(
                [league_mod.PoolEntry("a", 0, None), league_mod.PoolEntry("b", 1, None)],
                games=2)

    def test_the_anchor_span_stops_paying_for_saturated_edges(self):
        wide = set(league_mod.sai_pairings(60, anchor_every=4))
        capped = set(league_mod.sai_pairings(60, anchor_every=4, anchor_span=16))
        # `anchor_every=4` puts the anchor's spread on 1, 5, 9, ... — 41 is one of them.
        assert (0, 41) in wide and (0, 41) not in capped
        assert (0, 59) in capped, "one far edge is kept, as the saturation check"
        assert (0, 5) in capped and (0, 13) in capped, "the near edges must survive"

    def test_the_curve_row_carries_the_players_own_budget(self, league):
        _pool, report = league
        by_id = {r["checkpoint_id"]: r for r in report["curve"]}
        assert by_id[league_mod.RANDOM_NAME]["n_sims"] == 0
        assert by_id["r@100:n4"]["n_sims"] == 4

    def test_the_anchor_and_the_ladder_cost_exactly_zero_not_null(self, league):
        """⚠️ Regression: in a joint league `cost` is keyed by run and the anchor has
        none, so its x was written as null. A missing x reads as 'not measured'."""
        _pool, report = league
        row = {r["checkpoint_id"]: r for r in report["curve"]}[league_mod.RANDOM_NAME]
        assert row["euros_spent"] == 0.0 and row["training_seconds"] == 0.0


def _pool_with(tmp_path, steps, sims, ladder=(), grid=(), grid_points=0):
    """A pool over fake checkpoint files, so pool shape can be tested without a card."""
    for s in steps:
        (tmp_path / f"r-{s:06d}.pt").write_bytes(b"")
    (tmp_path / "anchor.pt").write_bytes(b"x")
    return league_mod.build_pool(
        ["r"], checkpoint_dir=str(tmp_path), anchor_path=str(tmp_path / "anchor.pt"),
        limit=None, sims=sims, ladder=ladder, grid=grid, grid_points=grid_points)


# --------------------------------------------------------------------------- #
# §5.1a — presenting three quantities without lying about two of them
# --------------------------------------------------------------------------- #

def _budget_report():
    """One run at three budgets, plus the untrained ladder.

    ⚠️ **The grid is on a *subset* of the checkpoints**, which is the real shape: the
    reference budget is rated at every checkpoint and the others at a handful. That
    asymmetry is exactly what makes a pooled fit wrong, and a fixture with the grid
    everywhere is symmetric enough that pooling accidentally gives the right answer.
    """
    def row(cid, step, sims, secs, elo, run=""):
        return {"checkpoint_id": cid, "run": run, "step": step, "n_sims": sims,
                "training_seconds": secs, "euros_spent": 0.0, "elo": elo,
                "ci95": 10.0, "games_played": 36, "draw_rate": 0.2}
    curve = [row("random", 0, 0, 0.0, 0.0)]
    curve += [row(f"init:n{k}", 0, k, 0.0, 40.0) for k in (1, 16, 64)]
    for step, secs, base in ((100, 1e3, 100.0), (1000, 1e4, 400.0), (10000, 1e5, 700.0)):
        budgets = ((16, -80.0), (64, 0.0), (256, 120.0)) if step > 100 else ((64, 0.0),)
        for k, bump in budgets:
            curve.append(row(f"r@{step}:n{k}", step, k, secs, base + bump, run="r"))
    return {"config": {"n_sims": 64, "budgets": [0, 1, 16, 64, 256]}, "curve": curve,
            "fit": {"anchor": "random"}}


class TestTheCurvePresentsBothAxes:

    def test_a_series_is_a_run_at_one_budget(self):
        """⚠️ Grouping by run alone mixes budgets: the x repeats and the line zigzags."""
        keys = {k for k, _ in curve_mod.group_by_series(_budget_report())}
        assert keys == {("r", 16), ("r", 64), ("r", 256)}
        for _k, rows in curve_mod.group_by_series(_budget_report()):
            xs = [r["training_seconds"] for r in rows]
            assert len(set(xs)) == len(xs), "a series must not repeat its x"

    def test_the_training_slope_is_never_fitted_across_budgets(self):
        """The number this protects: points at different budgets differ in *search*,
        so a least squares through them is not Elo per decade of training at all —
        and nothing about the result would look wrong."""
        report = _budget_report()
        per_series = {k: curve_mod.slope_per_decade(report, rows)
                      for k, rows in curve_mod.group_by_series(report)}
        # Each budget gained exactly 300 Elo over one decade of training, by construction.
        for k, slope in per_series.items():
            assert slope == pytest.approx(300.0), k
        # Pooling the run's nine points instead gives a different, meaningless number.
        pooled = curve_mod.slope_per_decade(
            report, [r for r in report["curve"] if r["run"] == "r"])
        assert abs(pooled - 300.0) > 1.0, \
            "the pooled fit happens to agree here; the fixture no longer bites"

    def test_the_search_axis_is_the_transpose_and_holds_the_net_fixed(self):
        levels = dict(curve_mod.by_training_level(_budget_report()))
        assert set(levels) == {("init", 0), ("r", 1000), ("r", 10000)}
        for (_run, _step), rows in levels.items():
            assert len({r["checkpoint_id"].split(":")[0] for r in rows}) == 1, \
                "a search-value series must be ONE network at several budgets"
            assert [r["n_sims"] for r in rows] == sorted(r["n_sims"] for r in rows)

    def test_elo_per_doubling_of_search(self):
        levels = dict(curve_mod.by_training_level(_budget_report()))
        # -80 at 16, 0 at 64, +120 at 256: 200 Elo over 4 doublings on the log2 fit.
        assert curve_mod.elo_per_doubling(levels[("r", 1000)]) == pytest.approx(50.0)
        assert curve_mod.elo_per_doubling(levels[("init", 0)]) == pytest.approx(0.0)

    def test_the_random_player_is_not_a_point_on_the_search_axis(self):
        """`random` has no network and no budget to vary; a 0 on a log2 axis is not
        a point, it is negative infinity."""
        for (_run, _step), rows in curve_mod.by_training_level(_budget_report()):
            assert all(r["n_sims"] > 0 for r in rows)

    def test_the_png_is_written_with_both_panels(self, tmp_path):
        path = curve_mod.plot(_budget_report(), str(tmp_path / "c.png"))
        if path is None:
            pytest.skip("matplotlib is not installed")
        assert os.path.getsize(path) > 5000


class TestRunEndpointsAlwaysMeet:
    """⚠️ Two runs of different lengths do not finish next to each other in a
    step-ordered calendar, so the pairing the joint league exists for can be missing."""

    def test_the_two_finals_play_at_every_shared_budget(self, tmp_path):
        for s in (100, 4000, 8000):
            (tmp_path / f"long-{s:06d}.pt").write_bytes(b"")
        for s in (100, 2000, 4000):
            (tmp_path / f"short-{s:06d}.pt").write_bytes(b"")
        (tmp_path / "anchor.pt").write_bytes(b"x")
        pool = league_mod.build_pool(
            ["long", "short"], checkpoint_dir=str(tmp_path),
            anchor_path=str(tmp_path / "anchor.pt"), limit=None, sims=64,
            ladder=(64,), grid=(256,), grid_points=3)
        idx = {p.name: i for i, p in enumerate(pool)}
        got = set(league_mod.endpoint_pairs(pool))
        for k in (64, 256):
            a, b = idx[f"long@8000:n{k}"], idx[f"short@4000:n{k}"]
            assert (min(a, b), max(a, b)) in got, f"the two finals never meet at n={k}"

    def test_the_ladder_and_the_anchor_are_not_endpoints(self, tmp_path):
        for s in (100, 400):
            (tmp_path / f"r-{s:06d}.pt").write_bytes(b"")
        (tmp_path / "anchor.pt").write_bytes(b"x")
        pool = league_mod.build_pool(
            ["r"], checkpoint_dir=str(tmp_path), anchor_path=str(tmp_path / "anchor.pt"),
            limit=None, sims=64, ladder=(1, 64))
        # One run has nobody to meet, and `random`/`init` belong to no run.
        assert league_mod.endpoint_pairs(pool) == []
