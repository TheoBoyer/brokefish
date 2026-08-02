"""Track D, layer 0 and layer 3: `brokefish/eval/`.

Two different questions get asked here and they need different oracles.

**Is the answer key right?** python-chess, the same oracle the engine uses. Every
suite claims a move is checkmate, or a stalemate, or a dead draw — all of which
python-chess can be asked directly, and all of which are wrong in a way no
internal consistency check would catch. A suite with a wrong key is worse than no
suite: it fails a correct net and it looks like a finding.

**Does the harness compute what it says?** Constructed cases with a known answer:
an evaluator rigged to play a specific move, a hand-built calibration set whose
ECE and Brier are arithmetic, an entropy whose normalisation is exactly 1.

⚠️ The threefold suite is the one python-chess cannot check, because its ring is
planted rather than replayed. What is checked instead is that the ring reads back
through our own `repetition_count` as 3, which is the property the suite depends
on.
"""

from __future__ import annotations

import math

import chess
import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.env.notation import to_fen, to_uci
from brokefish.eval import metrics, positions, probe, suites
from brokefish.nn.model import BrokefishNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the engine and the search are CUDA-shaped")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_board(boards, control, i: int) -> chess.Board:
    return chess.Board(to_fen(boards[i:i + 1], control[i:i + 1])[0])


def _to_move(boards, label: int) -> str:
    lab = torch.tensor([label], dtype=torch.int64, device=boards.device)
    return to_uci(boards, lab & probe.MOVE_MASK,
                  promo=(lab >> probe.PROMO_SHIFT) & 0b11)[0]


@pytest.fixture(scope="module")
def small_suites():
    return suites.harvest_suites(target=24, batch=6000, max_rounds=200, device=DEVICE)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return BrokefishNet().to(DEVICE).eval()


# --------------------------------------------------------------------------- #
# positions.py
# --------------------------------------------------------------------------- #

class TestRandomPositions:

    def test_python_chess_calls_them_valid(self):
        boards, control = positions.random_positions(64, seed=7, device=DEVICE)
        for i in range(64):
            board = _to_board(boards, control, i)
            status = board.status()
            # Our generator gives up castling rights and en passant, and does not
            # care how many pieces a side has, so the only clauses that may fire
            # are the ones a real game also permits. Anything else is a bug in
            # the generator, not a quirk of the FEN.
            assert status & ~chess.STATUS_TOO_MANY_KINGS == chess.STATUS_VALID, \
                f"{board.fen()} -> {status!r}"
            assert board.is_valid(), board.fen()

    def test_every_position_has_a_choice(self):
        boards, control = positions.random_positions(128, seed=8, device=DEVICE,
                                                     min_legal_moves=2)
        mask, _ = env.movegen(boards, control)
        assert int(env.bitset_to_bool(mask).reshape(128, -1).sum(-1).min()) >= 2

    def test_the_clock_knob_reaches_the_fifty_move_boundary(self):
        _b, control = positions.random_positions(16, seed=9, device=DEVICE, clock=99)
        assert torch.equal(control.abs(), torch.full_like(control.abs(), 100))

    def test_promotion_ready_puts_a_mover_pawn_on_the_seventh(self):
        boards, control = positions.random_positions(64, seed=10, device=DEVICE,
                                                     promotion_ready=True)
        for i in range(64):
            board = _to_board(boards, control, i)
            rank = 6 if board.turn == chess.WHITE else 1
            pawns = board.pieces(chess.PAWN, board.turn)
            assert any(chess.square_rank(s) == rank for s in pawns), board.fen()


# --------------------------------------------------------------------------- #
# probe.py
# --------------------------------------------------------------------------- #

class TestReplyCodes:

    def test_the_move_list_matches_python_chess(self):
        boards, control = positions.random_positions(64, seed=11, device=DEVICE)
        game, label = probe.enumerate_moves(boards, control)
        for i in range(64):
            ours = {_to_move(boards[i:i + 1], int(l))
                    for l in label[game == i]}
            theirs = {m.uci() for m in _to_board(boards, control, i).legal_moves}
            assert ours == theirs

    def test_the_terminal_code_matches_python_chess(self):
        # 512 and not 96: a stalemating reply is a ~1-in-2000 event under this
        # sampler, so a small batch passes this test without ever reaching the
        # branch it was written for.
        boards, control = positions.random_positions(512, seed=12, device=DEVICE)
        game, label, code, _h, _i = probe.reply_codes(boards, control)
        seen = {1: 0, 2: 0, 5: 0}
        for row in range(int(game.numel())):
            i = int(game[row])
            board = _to_board(boards, control, i)
            board.push_uci(_to_move(boards[i:i + 1], int(label[row])))
            c = int(code[row])
            if c == suites.CHECKMATE:
                assert board.is_checkmate(), board.fen()
            elif c == suites.STALEMATE:
                assert board.is_stalemate(), board.fen()
            elif c == suites.INSUFFICIENT:
                assert board.is_insufficient_material(), board.fen()
            elif c == 0:
                assert not board.is_game_over(claim_draw=False), board.fen()
            seen[c] = seen.get(c, 0) + 1
        # The scan is only worth what it covered; a vacuous pass is the failure
        # mode of a differential test over random positions.
        assert seen[1] > 0 and seen[2] > 0 and seen[5] > 0, seen

    def test_a_promotion_contributes_four_replies_and_a_quiet_move_one(self):
        boards, control = env.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        boards, control = boards.to(DEVICE), control.to(DEVICE)
        _g, label = probe.enumerate_moves(boards, control)
        promo_field = (label >> probe.PROMO_SHIFT) & 0b11
        assert int((promo_field > 0).sum()) == 3        # B, R, Q; N shares field 0
        assert int(label.numel()) == len(list(chess.Board(
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1").legal_moves))


# --------------------------------------------------------------------------- #
# The answer keys
# --------------------------------------------------------------------------- #

class TestSuiteKeys:

    @pytest.mark.parametrize("name", suites.SUITE_NAMES)
    def test_every_good_move_is_checkmate(self, small_suites, name):
        suite = small_suites[name]
        assert len(suite) > 0
        n_checked = 0
        for i in range(len(suite)):
            board = _to_board(suite.boards, suite.control, i)
            for label in suite.good[i].tolist():
                if label < 0:
                    continue
                after = board.copy()
                after.push_uci(_to_move(suite.boards[i:i + 1], label))
                assert after.is_checkmate(), f"{name} {board.fen()} {label}"
                n_checked += 1
        assert n_checked >= len(suite)

    def test_the_stalemate_key_stalemates(self, small_suites):
        suite = small_suites["avoid_stalemate"]
        for i in range(len(suite)):
            board = _to_board(suite.boards, suite.control, i)
            for label in suite.bad[i].tolist():
                if label < 0:
                    continue
                after = board.copy()
                after.push_uci(_to_move(suite.boards[i:i + 1], label))
                assert after.is_stalemate(), f"{board.fen()} {label}"

    def test_the_fifty_move_key_draws_by_the_clock(self, small_suites):
        suite = small_suites["avoid_fifty"]
        for i in range(len(suite)):
            board = _to_board(suite.boards, suite.control, i)
            assert board.halfmove_clock == 99
            for label in suite.bad[i].tolist():
                if label < 0:
                    continue
                after = board.copy()
                after.push_uci(_to_move(suite.boards[i:i + 1], label))
                assert after.is_fifty_moves(), f"{board.fen()} {label}"

    def test_the_insufficient_key_is_a_dead_position(self, small_suites):
        suite = small_suites["avoid_insufficient"]
        for i in range(len(suite)):
            board = _to_board(suite.boards, suite.control, i)
            for label in suite.bad[i].tolist():
                if label < 0:
                    continue
                after = board.copy()
                after.push_uci(_to_move(suite.boards[i:i + 1], label))
                assert after.is_insufficient_material(), f"{board.fen()} {label}"

    def test_underpromotion_means_the_queen_does_not_mate(self, small_suites):
        suite = small_suites["underpromotion"]
        for i in range(len(suite)):
            board = _to_board(suite.boards, suite.control, i)
            goods = [l for l in suite.good[i].tolist() if l >= 0]
            ucis = [_to_move(suite.boards[i:i + 1], l) for l in goods]
            # Every mating move is a promotion, and none of them is to a queen.
            assert all(len(u) == 5 and u[4] != "q" for u in ucis), (board.fen(), ucis)
            for label in suite.bad[i].tolist():
                if label < 0:
                    continue
                uci = _to_move(suite.boards[i:i + 1], label)
                assert uci.endswith("q"), uci
                after = board.copy()
                after.push_uci(uci)
                assert not after.is_checkmate(), f"{board.fen()} {uci}"

    def test_the_planted_ring_really_repeats(self, small_suites):
        """The one key python-chess cannot check: the ring is planted, not played."""
        suite = small_suites["avoid_threefold"]
        assert int(suite.ring_len.min()) == 2
        label = suite.bad[:, 0].to(torch.int64)
        hash_ = env.hash_position(suite.boards, suite.control)
        nb, nc, nh, irrev = env.step(suite.boards, suite.control,
                                     label & probe.MOVE_MASK,
                                     promo=(label >> probe.PROMO_SHIFT) & 0b11,
                                     hash=hash_)
        ring, length = env.push_history(suite.ring, suite.ring_len, hash_, irrev)
        assert not bool(irrev.any()), "the planted reply must be reversible"
        assert torch.equal(env.repetition_count(nh, ring, length),
                           torch.full_like(nh, 3))
        mask, chk = env.movegen(nb, nc)
        code, _r = env.terminal(mask, chk, nc, nb, nh, ring, length)
        assert torch.equal(code, torch.full_like(code, suites.REPETITION))

    def test_good_and_bad_never_overlap(self, small_suites):
        for name, suite in small_suites.items():
            for i in range(len(suite)):
                g = {l for l in suite.good[i].tolist() if l >= 0}
                b = {l for l in suite.bad[i].tolist() if l >= 0}
                assert not (g & b), f"{name} item {i}"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _rigged_evaluator(suite: suites.Suite, labels: torch.Tensor):
    """An evaluator whose policy puts all its mass on `labels[i]` for item `i`.

    This is how the scorer gets tested without needing a net that can play: a
    policy this sharp survives 128 simulations of PUCT, so `score_suite` returns
    exactly the fraction of `labels` that are in the answer key.
    """
    lookup = {}
    for i in range(len(suite)):
        key = (int(suite.boards[i].sum()), int(suite.control[i]))
        lookup.setdefault(key, []).append(int(labels[i]))

    def evaluate(boards, control, rep):
        n = boards.shape[0]
        policy = torch.zeros((n, 32, 64), dtype=torch.float16, device=boards.device)
        promo = torch.zeros((n, 32, 4), dtype=torch.float16, device=boards.device)
        value = torch.zeros((n,), dtype=torch.float16, device=boards.device)
        for i in range(n):
            key = (int(boards[i].sum()), int(control[i]))
            for label in lookup.get(key, []):
                slot, target = (label & probe.MOVE_MASK) // 64, (label & probe.MOVE_MASK) % 64
                policy[i, slot, target] = 20.0
                promo[i, slot, (label >> probe.PROMO_SHIFT) & 0b11] = 20.0
        return policy, promo, value
    return evaluate


class TestScoring:
    """⚠️ Every `score_suite` here passes `search_impl="torch"` deliberately.

    These test the *scorer*, and they do it with `_RiggedNet` — a callable standing
    in for a network so the right answer is known in advance. A rigged net has no
    `layers`, so no fused encoder can be built from it, and since 2026-07-31 the
    evaluation default is `search_impl="cuda"`, which pairs itself with the fused
    encoder (`runner.make_search`). The oracle tree is the correct path for a fake
    network; naming it keeps that a decision rather than a default these tests
    happen to inherit.
    """

    def test_the_scorer_reports_what_was_played(self, small_suites):
        suite = small_suites["avoid_stalemate"].subset(torch.arange(8, device=DEVICE))
        right = suites.score_suite(
            suite, _RiggedNet(_rigged_evaluator(suite, suite.good[:, 0])),
            n=16, device=DEVICE, batch=8, search_impl="torch")
        assert right["accuracy"] == 1.0 and right["blunder_rate"] == 0.0

        wrong = suites.score_suite(
            suite, _RiggedNet(_rigged_evaluator(suite, suite.bad[:, 0])),
            n=16, device=DEVICE, batch=8, search_impl="torch")
        assert wrong["accuracy"] == 0.0 and wrong["blunder_rate"] == 1.0

    def test_wilson_brackets_the_proportion(self):
        lo, hi = suites._wilson(200, 200)
        assert lo > 0.98 and hi == 1.0
        lo, hi = suites._wilson(0, 200)
        assert lo == 0.0 and hi < 0.02
        lo, hi = suites._wilson(100, 200)
        assert lo < 0.5 < hi

    def test_the_policy_scorer_agrees_with_the_search_on_a_sharp_policy(self, small_suites):
        suite = small_suites["mate_in_1"].subset(torch.arange(8, device=DEVICE))
        rigged = _RiggedNet(_rigged_evaluator(suite, suite.good[:, 0]))
        a = suites.score_suite(suite, rigged, n=16, device=DEVICE, batch=8,
                               search_impl="torch")
        b = suites.score_suite_policy(suite, rigged, device=DEVICE)
        assert a["accuracy"] == b["accuracy"] == 1.0


class _RiggedNet:
    """`make_evaluator(net, None)` calls `net(b, c, r)`, so a callable is a net."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, boards, control, rep):
        return self.fn(boards, control, rep)


# --------------------------------------------------------------------------- #
# metrics.py
# --------------------------------------------------------------------------- #

class TestMetrics:

    def test_calibration_is_exact_on_a_constructed_run(self):
        # A head that predicts +0.9 and always wins, and one that predicts -0.9
        # and always wins. The first is well calibrated, the second is not, and
        # ECE has to say so by exactly the amount constructed.
        v = torch.tensor([0.9] * 50 + [-0.9] * 50, device=DEVICE)
        z = torch.tensor([1.0] * 100, device=DEVICE)
        run = _fake_run(v, z)
        out = metrics.value_calibration(run, bins=10)
        assert out["n"] == 100
        assert math.isclose(out["ece"], 0.5 * 0.1 + 0.5 * 1.9, rel_tol=1e-5)
        assert math.isclose(out["brier"], 0.5 * 0.01 + 0.5 * 3.61, rel_tol=1e-5)

    def test_entropy_is_normalised_by_the_legal_move_count(self):
        # Uniform over k moves has entropy log k, so the normalised value is 1
        # whatever k is. That invariance is the whole reason for normalising.
        run = _fake_run(torch.zeros(3, device=DEVICE), torch.zeros(3, device=DEVICE))
        run.n_moves = torch.tensor([2, 20, 100], dtype=torch.int32, device=DEVICE)
        run.visit_entropy = run.n_moves.float().log()
        out = metrics.policy_entropy(run)
        assert math.isclose(out["entropy_normalised"], 1.0, rel_tol=1e-6)

    def test_a_single_legal_move_is_dropped_rather_than_dividing_by_zero(self):
        run = _fake_run(torch.zeros(2, device=DEVICE), torch.zeros(2, device=DEVICE))
        run.n_moves = torch.tensor([1, 4], dtype=torch.int32, device=DEVICE)
        run.visit_entropy = torch.tensor([0.0, math.log(4)], device=DEVICE)
        out = metrics.policy_entropy(run)
        assert out["n"] == 1 and math.isclose(out["entropy_normalised"], 1.0, rel_tol=1e-6)

    @pytest.mark.slow
    def test_a_self_play_run_produces_consistent_rows(self, net):
        run = metrics.self_play_run(net, games=8, n_sims=8, max_plies=60, device=DEVICE)
        n = int(run.value_pred.numel())
        for t in (run.outcome, run.visit_entropy, run.n_moves, run.control, run.rep):
            assert int(t.numel()) == n
        assert int(run.boards.shape[0]) == n
        assert set(run.outcome.unique().tolist()) <= {-1.0, 0.0, 1.0}
        stats = metrics.game_statistics(run)
        assert 0.0 <= stats["draw_rate"] <= 1.0
        # Every finished game reports the code that finished it, and a finished
        # game with code 0 would mean the loop stopped for a reason it invented.
        assert int((run.game_code == 0).sum()) == 0
        # 60 plies is far under the ~125 a random-init net takes, so this run
        # exercises the abandonment path rather than the collection one.
        assert stats["n_abandoned"] > 0
        assert int(run.game_plies.max()) <= 60

    @pytest.mark.slow
    def test_the_run_collects_the_number_of_games_it_was_asked_for(self, net):
        run = metrics.self_play_run(net, games=4, n_sims=4, max_plies=400, device=DEVICE)
        # `games` is a target, not a batch size: several slots can finish on the
        # same move, so the run collects at least what was asked for.
        assert metrics.game_statistics(run)["n_games"] >= 4
        assert run.meta["requested"] == 4 and run.meta["batch"] == 4


def _fake_run(v, z):
    n = int(v.numel())
    zeros = torch.zeros(n, dtype=torch.int32, device=v.device)
    return metrics.SelfPlayRun(
        value_pred=v, outcome=z, visit_entropy=torch.zeros(n, device=v.device),
        n_moves=zeros + 2,
        boards=torch.zeros((n, 32), dtype=torch.int16, device=v.device),
        control=torch.ones(n, dtype=torch.int16, device=v.device),
        rep=torch.zeros(n, dtype=torch.uint8, device=v.device),
        game_plies=zeros, game_result=torch.zeros(n, dtype=torch.int8, device=v.device),
        game_code=torch.ones(n, dtype=torch.uint8, device=v.device))


# --------------------------------------------------------------------------- #
# puzzles.py
# --------------------------------------------------------------------------- #

class TestPuzzles:

    def test_uci_round_trips_through_the_label_encoding(self):
        """`_labels_from_uci` is the only place an external notation comes back
        in, and it is the only place a slot has to be recovered from a square."""
        from brokefish.eval.puzzles import _labels_from_uci

        boards, control = positions.random_positions(48, seed=14, device=DEVICE)
        game, label = probe.enumerate_moves(boards, control)
        first = torch.tensor([int(label[(game == i).nonzero()[0]]) for i in range(48)],
                             device=DEVICE)
        ucis = [_to_move(boards[i:i + 1], int(first[i])) for i in range(48)]
        back = _labels_from_uci(boards, control, ucis)
        # A non-promotion UCI carries no promotion character, so it comes back
        # with field 0 -- which is what a non-promotion label holds anyway.
        assert torch.equal(back.to(torch.int64), first)

    def test_a_missing_file_says_where_to_get_it(self):
        from brokefish.eval.puzzles import load_puzzles
        with pytest.raises(FileNotFoundError, match="database.lichess.org"):
            load_puzzles(path="/nonexistent/puzzles.csv")

    def test_the_setup_move_is_played_and_the_solution_is_the_second(self, tmp_path):
        """The whole reader, on a CSV in Lichess's format built from our own rules.

        ⚠️ This proves the reader does what the format says, **not** that the
        format is what Lichess writes. The column names and the
        first-move-is-the-opponent's convention are the one part of `puzzles.py`
        with no oracle here, and they stay unverified until the real export is
        on disk. Everything downstream of parsing is covered.
        """
        from brokefish.eval.puzzles import load_puzzles

        path, expect = _write_puzzle_csv(tmp_path / "p.csv", n=64, seed=15)
        puzzles = load_puzzles(path=str(path), device=DEVICE, max_deviation=100)
        assert len(puzzles) == len(expect)

        for i, (fen_after, solution) in enumerate(expect):
            # The setup move was applied: the position the net is asked about is
            # the one *after* Lichess's first move, not the one in the FEN column.
            assert to_fen(puzzles.boards[i:i + 1], puzzles.control[i:i + 1],
                          fullmove=1)[0].rsplit(" ", 1)[0] == fen_after.rsplit(" ", 1)[0]
            # And the answer is the second move, in our label encoding. Compared
            # as UCI because a FEN round trip permutes slots, so the label itself
            # is not stable across the reader.
            assert _to_move(puzzles.boards[i:i + 1],
                            int(puzzles.answer[i])) == solution

    def test_the_deviation_and_rating_filters_bite(self, tmp_path):
        from brokefish.eval.puzzles import load_puzzles

        path, expect = _write_puzzle_csv(tmp_path / "p.csv", n=64, seed=16,
                                         deviation=lambda i: 40 if i % 2 else 400)
        kept = load_puzzles(path=str(path), device=DEVICE, max_deviation=100)
        # `_write_puzzle_csv` already drops the wide-deviation rows from `expect`.
        assert len(kept) == len(expect) == 32
        assert int(kept.deviation.max()) <= 100

        narrow = load_puzzles(path=str(path), device=DEVICE, max_deviation=100,
                              min_rating=1200, max_rating=1600)
        assert 0 < len(narrow) < len(kept)
        assert 1200 <= int(narrow.rating.min()) and int(narrow.rating.max()) <= 1600

    def test_the_curve_is_binned_by_rating_and_the_bins_partition(self, tmp_path):
        from brokefish.eval.puzzles import load_puzzles, score_puzzles

        path, _e = _write_puzzle_csv(tmp_path / "p.csv", n=48, seed=17)
        puzzles = load_puzzles(path=str(path), device=DEVICE)

        rigged = _RiggedNet(_rigged_evaluator(
            suites.Suite("p", "", puzzles.boards, puzzles.control,
                         puzzles.answer[:, None], puzzles.answer[:, None],
                         *env.empty_history(len(puzzles), device=DEVICE)),
            puzzles.answer))
        out = score_puzzles(puzzles, rigged, n=16, batch=48, device=DEVICE,
                            bin_width=200)
        assert out["solve_rate"] == 1.0
        assert out["n_puzzles"] == len(puzzles)
        # Every puzzle lands in exactly one bin, or the curve's x-axis is a lie.
        assert sum(b["n"] for b in out["bins"]) == len(puzzles)
        assert len(out["bins"]) > 1
        for b in out["bins"]:
            assert b["rating_lo"] <= b["rating_hi"] and b["rate"] == 1.0

    def test_the_policy_only_curve_agrees_with_the_searched_one_on_a_rigged_net(self, tmp_path):
        """`score_puzzles_policy` is the same question with the tree removed.

        A network that already prefers the solution needs no search to find it, so a
        rigged evaluator must score 1.0 both ways. That pins the label decoding and
        the promotion handling of the policy-only path against the searched one,
        which is where the two could silently drift apart.
        """
        from brokefish.eval.puzzles import (load_puzzles, score_puzzles,
                                            score_puzzles_policy)

        path, _e = _write_puzzle_csv(tmp_path / "p.csv", n=48, seed=17)
        puzzles = load_puzzles(path=str(path), device=DEVICE)
        rigged = _RiggedNet(_rigged_evaluator(
            suites.Suite("p", "", puzzles.boards, puzzles.control,
                         puzzles.answer[:, None], puzzles.answer[:, None],
                         *env.empty_history(len(puzzles), device=DEVICE)),
            puzzles.answer))

        searched = score_puzzles(puzzles, rigged, n=16, batch=48, device=DEVICE)
        policy = score_puzzles_policy(puzzles, rigged, batch=48, device=DEVICE)
        assert policy["solve_rate"] == searched["solve_rate"] == 1.0
        assert policy["n_sims"] == 0, "the point of this path is that no tree ran"
        assert policy["n_puzzles"] == len(puzzles)
        # Every puzzle in exactly one bin, same partition as the searched curve.
        assert sum(b["n"] for b in policy["bins"]) == len(puzzles)
        assert [b["rating_lo"] for b in policy["bins"]] == \
               [b["rating_lo"] for b in searched["bins"]]

    def test_the_policy_only_curve_scores_zero_when_the_net_plays_something_else(self, tmp_path):
        from brokefish.eval.puzzles import load_puzzles, score_puzzles_policy

        path, _e = _write_puzzle_csv(tmp_path / "p.csv", n=24, seed=18)
        puzzles = load_puzzles(path=str(path), device=DEVICE)
        other = _other_legal_move(puzzles.boards, puzzles.control, puzzles.answer)
        wrong = _RiggedNet(_rigged_evaluator(
            suites.Suite("p", "", puzzles.boards, puzzles.control,
                         other[:, None], other[:, None],
                         *env.empty_history(len(puzzles), device=DEVICE)),
            other))
        out = score_puzzles_policy(puzzles, wrong, batch=24, device=DEVICE)
        assert out["solve_rate"] == 0.0

    def test_a_net_that_plays_something_else_scores_zero(self, tmp_path):
        from brokefish.eval.puzzles import load_puzzles, score_puzzles

        path, _e = _write_puzzle_csv(tmp_path / "p.csv", n=24, seed=18)
        puzzles = load_puzzles(path=str(path), device=DEVICE)
        wrong = _other_legal_move(puzzles.boards, puzzles.control, puzzles.answer)
        rigged = _RiggedNet(_rigged_evaluator(
            suites.Suite("p", "", puzzles.boards, puzzles.control,
                         wrong[:, None], wrong[:, None],
                         *env.empty_history(len(puzzles), device=DEVICE)),
            wrong))
        out = score_puzzles(puzzles, rigged, n=16, batch=24, device=DEVICE)
        assert out["solve_rate"] == 0.0


def _write_puzzle_csv(path, n: int, seed: int, deviation=None):
    """A CSV in Lichess's column order, built from random legal play.

    Each row is a position, one legal move played for the solver (Lichess's
    convention: the first move in `Moves` is the opponent's), and one legal move
    of the resulting position as the "solution". The solution is an arbitrary
    legal move rather than a good one — this file tests the reader and the
    scorer, and neither of them has an opinion about which move is right.
    """
    import csv as _csv

    boards, control = positions.random_positions(n * 2, seed=seed, device=DEVICE)
    game, label = probe.enumerate_moves(boards, control)
    take = torch.tensor([int(label[(game == i).nonzero()[0]])
                         for i in range(boards.shape[0])], device=DEVICE)
    fens = to_fen(boards, control)
    setup = [_to_move(boards[i:i + 1], int(take[i])) for i in range(boards.shape[0])]

    after, after_c, _m, _c = env.play(boards, control, take & probe.MOVE_MASK,
                                      promo=(take >> probe.PROMO_SHIFT) & 0b11)
    g2, l2 = probe.enumerate_moves(after, after_c)

    rows, expect = [], []
    fens_after = to_fen(after, after_c)
    for i in range(after.shape[0]):
        if len(expect) >= n:
            break
        moves = (g2 == i).nonzero()
        # At least two, so a "wrong move" exists: with one legal reply a net that
        # is trying to be wrong still solves the puzzle, and a scorer test built
        # on such a row measures nothing.
        if moves.numel() < 2:
            continue
        sol = _to_move(after[i:i + 1], int(l2[moves[0]]))
        dev = deviation(len(expect)) if deviation else 40
        rows.append({"PuzzleId": f"p{len(expect):05d}", "FEN": fens[i],
                     "Moves": f"{setup[i]} {sol}",
                     "Rating": 800 + 50 * (len(expect) % 20), "RatingDeviation": dev,
                     "Popularity": 90, "NbPlays": 1000, "Themes": "mateIn1",
                     "GameUrl": "https://example.invalid", "OpeningTags": ""})
        expect.append((fens_after[i], sol))

    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if deviation:
        expect = [e for i, e in enumerate(expect) if deviation(i) <= 100]
    return path, expect


def _other_legal_move(boards, control, answer):
    """Any legal label that is not the answer, per position."""
    game, label = probe.enumerate_moves(boards, control)
    out = answer.clone()
    for i in range(boards.shape[0]):
        for lab in label[game == i].tolist():
            if lab != int(answer[i]):
                out[i] = lab
                break
    return out


class TestSearchImplPairing:
    """`search_impl` and `impl` are two axes and only one pairing is illegal.

    ⚠️ The kernel's `_expand` reads fp16 logits and refuses to cast an fp32 policy,
    because casting would change the numbers the fp32 reference computes and break
    `tests/test_search_cuda.py`'s tree-for-tree comparison. `impl=None` is the plain
    torch module, which emits fp32 — so a CUDA search with no fused encoder raises.

    This regression exists because flipping the evaluation default from `"torch"` to
    `"cuda"` (2026-07-31) broke exactly that pairing and the suite stayed green: the
    only tests that drive a real search through `self_play_run` are `@slow`, so a
    default `pytest` run skipped them and the failure surfaced in a layer-0 run
    instead. These are deliberately *not* slow.
    """

    def test_a_cuda_search_with_no_encoder_named_gets_the_fused_one(self, net):
        from brokefish.eval.runner import make_search

        s = make_search(n=8, B=2, net=net, impl=None, search_impl="cuda",
                        device=DEVICE)
        boards, control = env.initial_boards(2, device=DEVICE)
        s.reset(boards, control)
        with torch.no_grad():
            s.self_play_move()          # raises TypeError on fp32 logits

    def test_the_torch_search_still_accepts_the_plain_module(self, net):
        """The oracle path must keep working with no fused encoder at all."""
        from brokefish.eval.runner import make_search

        s = make_search(n=8, B=2, net=net, impl=None, search_impl="torch",
                        device=DEVICE)
        boards, control = env.initial_boards(2, device=DEVICE)
        s.reset(boards, control)
        with torch.no_grad():
            s.self_play_move()
