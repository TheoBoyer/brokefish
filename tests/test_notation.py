"""D0: FEN, UCI, SAN and PGN output, against python-chess as the oracle.

The same arrangement `docs/env.md` uses for the engine: python-chess is the
authority, our output has to match it string for string, and the positions come
from random legal play rather than from a hand-written list, so the rules that
matter are the ones that actually arise.

    python -m tests.test_notation
    pytest tests/test_notation.py

⚠️ **String equality, not "looks right".** SAN has three separate ways to be
subtly wrong — the disambiguation rule, the en-passant capture that lands on an
empty square, and the check/mate suffix — and every one of them produces a
plausible move that a human reader accepts. `chess.Board.san()` is the oracle
precisely because it is not us.

⚠️ The FEN comparison uses `chess.Board.fen()` with its default
`en_passant="legal"`. Our `to_fen` follows the same convention on purpose
(`notation.py`), so a mismatch here is a real bug and not a convention clash.
"""

import io
import random
import unittest

import chess
import chess.pgn
import torch

from brokefish import env
from brokefish.env import interop, notation
from brokefish.env.notation import GameRecorder, to_fen, to_pgn, to_san, to_uci

PLIES = 60
GAMES = 24
SEED = 20260730


def _playouts(games=GAMES, plies=PLIES, seed=SEED):
    """Random legal play, yielding `(boards, control, chess.Board)` per ply.

    One engine position and one python-chess position advanced in lockstep, so
    every comparison below is against the oracle's own view of the same game.
    """
    rng = random.Random(seed)
    for g in range(games):
        boards, control = env.initial_boards(1)
        ref = chess.Board()
        for _ in range(plies):
            mask, _ = env.movegen(boards, control)
            legal = interop.list_legal_moves(boards[0], mask[0])
            if not legal or ref.is_game_over(claim_draw=False):
                break
            yield boards, control, ref, legal, rng
            move = rng.choice(legal)
            args = interop.move_to_args(move, boards)
            boards, control, _, _ = env.play(boards, control, **args)
            ref.push(move)


class TestFen(unittest.TestCase):
    def test_startpos(self):
        boards, control = env.initial_boards(1)
        self.assertEqual(to_fen(boards, control)[0], notation.STARTPOS_FEN)

    def test_matches_python_chess_over_playouts(self):
        seen = 0
        for boards, control, ref, _legal, _rng in _playouts():
            self.assertEqual(to_fen(boards, control, ref.fullmove_number)[0], ref.fen())
            seen += 1
        self.assertGreater(seen, 500, "the playout did not produce enough positions")

    def test_round_trips_at_the_fen_level(self):
        """`to_fen(from_fen(f)) == f`, over positions from real play."""
        for boards, control, ref, _legal, _rng in _playouts(games=6):
            text = to_fen(boards, control, ref.fullmove_number)[0]
            back_b, back_c = env.from_fen(text)
            self.assertEqual(to_fen(back_b, back_c, ref.fullmove_number)[0], text)

    def test_the_round_trip_does_not_preserve_slots(self):
        """⚠️ FEN is lossy: it carries the position, not the piece list.

        `from_fen` assigns slots in FEN scan order (rank 8 down to rank 1), while
        the engine keeps a piece in the slot it started the game in. After
        1. Nh3 the two disagree — same position, permuted slots — so a FEN
        cannot be used to resume anything that indexes by slot.
        """
        boards, control = env.initial_boards(1)
        args = interop.move_to_args(chess.Move.from_uci("g1h3"), boards)
        boards, control, _, _ = env.play(boards, control, **args)
        back_b, back_c = env.from_fen(to_fen(boards, control)[0])
        self.assertEqual(to_fen(back_b, back_c)[0], to_fen(boards, control)[0])
        self.assertFalse(torch.equal(back_b, boards))
        self.assertEqual(sorted(boards[0].tolist()), sorted(back_b[0].tolist()))

    def test_en_passant_field_is_the_legal_convention(self):
        """A double push with no capturer available prints `-`, not the square."""
        boards, control = env.from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        args = interop.move_to_args(chess.Move.from_uci("a2a4"), boards)
        boards, control, _, _ = env.play(boards, control, **args)
        self.assertIn(" - 0 1", to_fen(boards, control)[0])

        # ...and prints the square when a black pawn is actually there to take.
        boards, control = env.from_fen("4k3/8/8/8/1p6/8/P7/4K3 w - - 0 1")
        args = interop.move_to_args(chess.Move.from_uci("a2a4"), boards)
        boards, control, _, _ = env.play(boards, control, **args)
        self.assertIn(" a3 ", to_fen(boards, control)[0])

    def test_promoted_piece_in_a_pawn_slot(self):
        """The type bits decide the letter, not the slot index."""
        boards, control = env.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        args = interop.move_to_args(chess.Move.from_uci("a7a8q"), boards)
        boards, control, _, _ = env.play(boards, control, **args)
        self.assertTrue(to_fen(boards, control)[0].startswith("Q3k3/"))

    def test_fullmove_argument(self):
        boards, control = env.initial_boards(3)
        self.assertTrue(to_fen(boards, control, 7)[0].endswith(" 0 7"))
        self.assertEqual([f.split()[-1] for f in to_fen(boards, control, [1, 2, 3])],
                         ["1", "2", "3"])
        with self.assertRaises(ValueError):
            to_fen(boards, control, [1, 2])

    def test_batched(self):
        boards, control = env.initial_boards(4)
        self.assertEqual(to_fen(boards, control), [notation.STARTPOS_FEN] * 4)


def _fanout(boards, control, legal):
    """One position and its `k` legal moves as a `k`-wide batch.

    `to_san` costs a `movegen` and a `play` per call, so asking it about a whole
    move list one move at a time is `k` launches where one will do. Tiling the
    board is what makes the exhaustive comparison below take seconds instead of
    minutes, and it exercises the batched path at the same time.
    """
    wide = boards.expand(len(legal), -1).contiguous()
    args = interop.move_to_args(legal, wide)
    return wide, control.expand(len(legal)).contiguous(), args


class TestUci(unittest.TestCase):
    def test_matches_python_chess_over_playouts(self):
        seen = 0
        for boards, control, _ref, legal, _rng in _playouts():
            wide, _wc, args = _fanout(boards, control, legal)
            self.assertEqual(to_uci(wide, **args), [m.uci() for m in legal])
            seen += len(legal)
        self.assertGreater(seen, 10_000)

    def test_castling_is_the_king_from_to(self):
        boards, control = env.from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        for uci in ("e1g1", "e1c1"):
            args = interop.move_to_args(chess.Move.from_uci(uci), boards)
            self.assertEqual(to_uci(boards, **args)[0], uci)

    def test_all_four_promotions(self):
        boards, control = env.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        got = [to_uci(boards, **interop.move_to_args(chess.Move.from_uci(f"a7a8{c}"),
                                                     boards))[0] for c in "nbrq"]
        self.assertEqual(got, ["a7a8n", "a7a8b", "a7a8r", "a7a8q"])

    def test_promo_defaults_to_queen(self):
        boards, control = env.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        move = interop.move_to_args(chess.Move.from_uci("a7a8n"), boards)["move"]
        self.assertEqual(to_uci(boards, move)[0], "a7a8q")

    def test_null_move(self):
        boards, control = env.initial_boards(1)
        self.assertEqual(to_uci(boards, torch.tensor([-1]))[0], "0000")


class TestSan(unittest.TestCase):
    def test_matches_python_chess_over_playouts(self):
        """Every legal move of every position, string for string."""
        seen = 0
        for boards, control, ref, legal, _rng in _playouts():
            wide, wide_c, args = _fanout(boards, control, legal)
            self.assertEqual(to_san(wide, wide_c, **args),
                             [ref.san(m) for m in legal], ref.fen())
            seen += len(legal)
        self.assertGreater(seen, 10_000)

    def test_disambiguation_by_file_rank_and_both(self):
        """The three branches of the rule, including the one needing both.

        ⚠️ A shared rank forces the *file* and a shared file forces the *rank*,
        which is the inversion that makes this rule easy to write backwards. The
        third case has a queen on our rank and another on our file at once, so
        neither hint alone resolves it — that is the case a wrong implementation
        still passes the first two on.
        """
        cases = [
            ("4k3/8/8/8/4K3/8/8/R6R w - - 0 1", "a1d1", "Rad1"),        # file
            ("R3k3/8/8/8/4K3/8/8/R7 w - - 0 1", "a1a4", "R1a4+"),       # rank
            ("4k3/8/8/8/Q7/8/6K1/Q2Q4 w - - 0 1", "a1d4", "Qa1d4+"),    # both
        ]
        for fen, uci, want in cases:
            boards, control = env.from_fen(fen)
            move = chess.Move.from_uci(uci)
            args = interop.move_to_args(move, boards)
            self.assertEqual(to_san(boards, control, **args)[0], want, fen)
            self.assertEqual(to_san(boards, control, **args)[0], chess.Board(fen).san(move))

    def test_en_passant_is_a_capture_onto_an_empty_square(self):
        boards, control = env.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        args = interop.move_to_args(chess.Move.from_uci("e5d6"), boards)
        self.assertEqual(to_san(boards, control, **args)[0], "exd6")

    def test_castling_notation(self):
        boards, control = env.from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        for uci, want in (("e1g1", "O-O"), ("e1c1", "O-O-O")):
            args = interop.move_to_args(chess.Move.from_uci(uci), boards)
            self.assertEqual(to_san(boards, control, **args)[0], want)

    def test_check_and_mate_suffixes(self):
        boards, control = env.from_fen("6k1/8/8/8/8/8/8/R5K1 w - - 0 1")
        args = interop.move_to_args(chess.Move.from_uci("a1a8"), boards)
        self.assertEqual(to_san(boards, control, **args)[0], "Ra8+")

        boards, control = env.from_fen("6k1/8/6K1/8/8/8/8/R7 w - - 0 1")
        args = interop.move_to_args(chess.Move.from_uci("a1a8"), boards)
        self.assertEqual(to_san(boards, control, **args)[0], "Ra8#")

    def test_promotion_with_capture_and_check(self):
        fen = "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"
        boards, control = env.from_fen(fen)
        ref = chess.Board(fen)
        move = chess.Move.from_uci("a7b8q")
        args = interop.move_to_args(move, boards)
        self.assertEqual(to_san(boards, control, **args)[0], ref.san(move))
        self.assertEqual(to_san(boards, control, **args)[0], "axb8=Q+")

    def test_null_move(self):
        boards, control = env.initial_boards(1)
        self.assertEqual(to_san(boards, control, torch.tensor([-1]))[0], "--")


class TestPgn(unittest.TestCase):
    def test_python_chess_replays_what_we_write(self):
        """The strongest test available: hand the PGN back to the oracle."""
        rng = random.Random(SEED + 1)
        boards, control = env.initial_boards(1)
        rec = GameRecorder(boards, control, [{"White": "brokefish", "Black": "brokefish"}])
        ref = chess.Board()
        for _ in range(PLIES):
            mask, _ = env.movegen(rec.boards, rec.control)
            legal = interop.list_legal_moves(rec.boards[0], mask[0])
            if not legal:
                break
            move = rng.choice(legal)
            ref.push(move)
            rec.push(**interop.move_to_args(move, rec.boards))

        game = chess.pgn.read_game(io.StringIO(rec.pgn(0, "1/2-1/2")))
        self.assertIsNotNone(game)
        replayed = chess.Board()
        for move in game.mainline_moves():
            replayed.push(move)
        self.assertEqual(replayed.fen(), ref.fen())
        self.assertEqual(game.headers["White"], "brokefish")
        self.assertEqual(game.headers["Result"], "1/2-1/2")

    def test_movetext_numbering(self):
        pgn = to_pgn(["e4", "e5", "Nf3"], result="1-0")
        self.assertIn("1. e4 e5 2. Nf3 1-0", pgn)

    def test_black_to_move_first(self):
        pgn = to_pgn(["e5", "Nf3"], result="*", first_fullmove=3, first_is_black=True)
        self.assertIn("3... e5 4. Nf3 *", pgn)

    def test_setup_tag_only_for_a_non_initial_position(self):
        self.assertNotIn("[FEN", to_pgn(["e4"], start_fen=notation.STARTPOS_FEN))
        fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
        self.assertIn(f'[FEN "{fen}"]', to_pgn(["Kf1"], start_fen=fen))
        self.assertIn('[SetUp "1"]', to_pgn(["Kf1"], start_fen=fen))

    def test_result_tag_and_terminator_agree(self):
        pgn = to_pgn(["e4"], result="0-1")
        self.assertIn('[Result "0-1"]', pgn)
        self.assertTrue(pgn.rstrip().endswith("0-1"))

    def test_seven_tag_roster_order(self):
        pgn = to_pgn([], headers={"Zulu": "z", "White": "w"})
        tags = [line.split('"')[0][1:].strip() for line in pgn.splitlines() if line.startswith("[")]
        self.assertEqual(tags[:7], ["Event", "Site", "Date", "Round", "White", "Black", "Result"])
        self.assertEqual(tags[-1], "Zulu")

    def test_lines_are_wrapped(self):
        pgn = to_pgn(["Nf3"] * 200, width=80)
        body = pgn.split("\n\n", 1)[1]
        self.assertTrue(all(len(line) <= 80 for line in body.splitlines()))
        self.assertGreater(len(body.splitlines()), 1)


class TestGameRecorder(unittest.TestCase):
    def test_batched_games_stay_independent(self):
        boards, control = env.initial_boards(2)
        rec = GameRecorder(boards, control)
        e4 = interop.move_to_args(chess.Move.from_uci("e2e4"), boards[0:1])["move"]
        d4 = interop.move_to_args(chess.Move.from_uci("d2d4"), boards[1:2])["move"]
        rec.push(torch.cat([e4, d4]))
        self.assertEqual(rec.moves[0], ["e4"])
        self.assertEqual(rec.moves[1], ["d4"])

    def test_inactive_game_records_nothing_and_does_not_advance(self):
        boards, control = env.initial_boards(2)
        rec = GameRecorder(boards, control)
        move = interop.move_to_args(chess.Move.from_uci("e2e4"), boards)["move"]
        before = rec.boards[1].clone()
        rec.push(torch.cat([move[0:1], move[0:1]]),
                 active=torch.tensor([True, False]))
        self.assertEqual(rec.moves[0], ["e4"])
        self.assertEqual(rec.moves[1], [])
        self.assertTrue(torch.equal(rec.boards[1], before))
        self.assertEqual(int(rec.control[1]), 1)

    def test_records_the_starting_position(self):
        fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 9"
        boards, control = env.from_fen(fen)
        rec = GameRecorder(boards, control, fullmove=9)
        self.assertEqual(rec.start_fen[0], fen)
        self.assertIn(f'[FEN "{fen}"]', rec.pgn(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
