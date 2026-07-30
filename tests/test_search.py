"""The MCTS v0 reference implementation, against `docs/mcts.md`.

There is no whole-search oracle the way `perft` was one for the move generator, so
the checks here come at it from five sides:

* the invariants of §5, which hold whatever the network says
* component oracles: §6.6 against an independent transcription of AlphaZero's
  `ucb_score`, and §6.1's noise against the moments of `Dir(alpha)`
* exact arithmetic on hand-built state, where the parity and the running mean can
  be equalities rather than tolerances
* forced descents, which pin one real ply against the engine without depending on
  the exploration schedule
* properties of chess, which no reimplementation of the algorithm can supply

⚠️ Several tests force the descent by zeroing priors rather than letting PUCT find
what they are about. That is deliberate: with first-play urgency at 0, whether the
search *reaches* a given node depends on the network's priors, so a test that waits
for it is measuring the exploration schedule. `test_finds_mate_in_one` is the one
that does wait, and it needs `n = 800` to pass.

    python -m tests.test_search [--slow]
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.nn.model import BrokefishNet
from brokefish.search import Search, SearchConfig, check_invariants, make_evaluator
from brokefish.search.torch_impl import IRREVERSIBLE, MOVE_BITS, TERMINAL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
PROMOTION = "8/P7/8/8/8/8/8/K6k w - - 0 1"
# Clock 99: every move takes it to 101, which spec §4.3 calls a draw.
FIFTY_MOVE = "4k3/8/8/8/8/8/8/4K2R w - - 99 1"


def make(n=32, B=2, E=64, seed=0, boards=None, control=None, **kw):
    """A search over `B` copies of one position, with an untrained network."""
    torch.manual_seed(seed)
    net = BrokefishNet().to(DEVICE).eval()
    cfg = SearchConfig(n=n, B=B, E=E, **kw)
    s = Search(cfg, evaluate=make_evaluator(net), device=DEVICE, seed=seed)
    if boards is None:
        boards, control = env.initial_boards(B, device=DEVICE)
    else:
        boards = boards.to(DEVICE).expand(B, -1).contiguous()
        control = control.to(DEVICE).expand(B).contiguous()
    s.reset(boards, control)
    return s


def from_fen(fen: str):
    b, c = env.from_fen(fen)
    return b.to(DEVICE), c.to(DEVICE)


# --------------------------------------------------------------------------- #
# The invariants of §5, which hold whatever the network says
# --------------------------------------------------------------------------- #

def test_invariants_hold_over_several_moves():
    s = make(n=48, B=4)
    with torch.no_grad():
        for _ in range(4):
            s.self_play_move()
            check_invariants(s)
            s.reset_finished()


def test_root_visits_sum_to_n():
    s = make(n=40, B=3)
    with torch.no_grad():
        s.self_play_move()
    ne = s.node_nedges[:, 0].long()
    valid = torch.arange(s.config.E, device=DEVICE)[None, :] < ne[:, None]
    total = torch.where(valid, s.edge_N.long()[:, 0], torch.zeros_like(valid, dtype=torch.long))
    assert total.sum(-1).tolist() == [40] * 3


def test_pool_is_exactly_filled():
    """`Nmax = n + 1` is the exact bound: the root plus one node per simulation."""
    s = make(n=25, B=2)
    with torch.no_grad():
        s.self_play_move()
    # Simulations that end on a stored terminal node create nothing, so the pool
    # is full only when none did. From the start position none can.
    assert s.node_count.tolist() == [26, 26]


def test_edge_q_stays_in_range():
    s = make(n=64, B=2)
    with torch.no_grad():
        s.self_play_move()
    assert float(s.edge_Q.min()) >= 0.0 and float(s.edge_Q.max()) <= 1.0


# --------------------------------------------------------------------------- #
# Properties of chess, which no reimplementation of the algorithm can supply
# --------------------------------------------------------------------------- #

def az_ucb_score(prior: float, child_n: int, child_q: float, parent_n: int,
                 base: float = 19652.0, init: float = 1.25) -> float:
    """AlphaZero's `ucb_score`, transcribed from the released pseudocode.

    Deliberately written out in scalar Python from the paper rather than factored
    out of `_select`, so that a mistake in one is not a mistake in both. `init` is
    added to the logarithm, not multiplied by it.
    """
    pb_c = math.log((parent_n + base + 1) / base) + init
    pb_c *= math.sqrt(parent_n) / (child_n + 1)
    return pb_c * prior + (child_q if child_n > 0 else 0.0)


def test_selection_matches_alphazeros_ucb_score():
    """§6.6 against an independent transcription, over random tree states.

    The strongest check in this file: it does not care what the network says or
    what the rules are, only whether the selection rule is the published one. A
    wrong `sqrt`, a multiplied `pb_c_init`, a parent count taken over invalid
    edges, or a tie broken the other way all fail here and nowhere else.
    """
    B, trials = 6, 200
    s = make(n=8, B=B)
    g = torch.Generator(device=DEVICE).manual_seed(3)
    bad = []
    for trial in range(trials):
        n_edges = torch.randint(1, s.config.E + 1, (B,), generator=g, device=DEVICE)
        s.node_nedges[:, 0] = n_edges.to(torch.uint8)
        s.edge_prior[:, 0] = torch.rand((B, s.config.E), generator=g, device=DEVICE).half()
        s.edge_N[:, 0] = torch.randint(0, 40, (B, s.config.E), generator=g,
                                       device=DEVICE).to(torch.int16)
        s.edge_Q[:, 0] = torch.rand((B, s.config.E), generator=g, device=DEVICE)
        got = s._select(torch.zeros(B, dtype=torch.long, device=DEVICE))
        for row in range(B):
            k = int(n_edges[row])
            parent = int(s.edge_N[row, 0, :k].sum())
            score = [az_ucb_score(float(s.edge_prior[row, 0, i]), int(s.edge_N[row, 0, i]),
                                  float(s.edge_Q[row, 0, i]), parent) for i in range(k)]
            # Ties by the lowest edge index, per §6.6.
            want = max(range(k), key=lambda i: (score[i], -i))
            if want != int(got[row]):
                bad.append((trial, row, want, int(got[row])))
    assert not bad, f"{len(bad)} of {trials * B} disagree, first {bad[:3]}"


def test_ties_go_to_the_lowest_edge_index():
    """§6.6 and §6.7's tie-break, which is a contract rather than a preference.

    ⚠️ `torch.argmax` returns the lowest index on both CPU and CUDA today, so
    replacing `_lowest_argmax` with it changes nothing measurable and
    `test_mutation_search.py` keeps that as its control. Torch does not promise it,
    and §12 compares trees with the kernel edge for edge, so the tie-break is
    pinned here to catch a change of *rule* rather than a change of backend.
    """
    s = make(n=8, B=1)
    s.node_nedges[0, 0] = 8
    s.edge_prior[0, 0, :8] = 0.1
    s.edge_N[0, 0, :8] = 5
    s.edge_Q[0, 0, :8] = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
                                      device=DEVICE)
    got = int(s._select(torch.zeros(1, dtype=torch.long, device=DEVICE))[0])
    assert got == 3, f"a three-way tie at 3, 4, 5 resolved to {got}"

    # And the same at the root's move choice, where the argmax is over visits.
    s = make(n=8, B=1, tau_plies=0)
    s.root_rep = torch.zeros(1, dtype=torch.uint8, device=DEVICE)
    s.node_nedges[0, 0] = 4
    s.node_flags[0, 0] = 0
    s.edge_N[0, 0, :4] = torch.tensor([1, 3, 3, 1], dtype=torch.int16, device=DEVICE)
    s.edge_move[0, 0, :4] = torch.tensor([10, 20, 30, 40], dtype=torch.int16, device=DEVICE)
    s.budget[0] = 8
    with torch.no_grad():
        rec = s.select_and_advance()
    assert int(rec.played[0]) == 20, "a tie in the visit counts took the wrong edge"


def test_backup_parity_at_every_depth():
    """§6.5's parity `(L - d) % 2`, at path lengths 1 to 5.

    One evaluation at the leaf has to update edges owned by players who alternate
    along the path, so every other level takes the complement. Inverting this
    produces a search that plays the worst available move, which §12 calls the
    most common bug in reimplementations.

    ⚠️ At `L = 2` the correct parity and the inverted `d % 2` agree on both levels,
    so a two-level test cannot tell them apart. At `L = 3` they are exact
    opposites, which is why this goes to 5.
    """
    q = 0.75                                     # exact in fp16
    for length in (1, 2, 3, 4, 5):
        s = make(n=8, B=1)
        s.node_nedges[0, :length] = 1
        s.node_flags[0, :length + 1] = 0
        s.path_node[0, :length] = torch.arange(length, dtype=torch.int16, device=DEVICE)
        s.path_edge[0, :length] = 0
        s.path_len[0] = length
        s.node_value[0, length] = q
        s.edge_N[0, :length] = 0
        s.edge_Q[0, :length] = 0.0
        s._backup(torch.tensor([length], device=DEVICE),
                  torch.ones(1, dtype=torch.bool, device=DEVICE))
        got = [float(s.edge_Q[0, d, 0]) for d in range(length)]
        want = [q if (length - d) % 2 == 0 else 1.0 - q for d in range(length)]
        assert got == want, f"L={length}: {got} != {want}"
        assert s.edge_N[0, :length, 0].tolist() == [1] * length


def test_running_mean_over_several_visits():
    s = make(n=4, B=1)
    s.node_nedges[0, :2] = 1
    s.node_flags[0, :2] = 0
    s.path_node[0, 0] = 0
    s.path_edge[0, 0] = 0
    s.path_len[0] = 1
    live = torch.ones(1, dtype=torch.bool, device=DEVICE)
    for v in (1.0, 0.5, 0.0):
        s.node_value[0, 1] = v
        s._backup(torch.tensor([1], device=DEVICE), live)
    # The edge into the leaf flips each time: 0, 0.5, 1 -> mean 0.5.
    assert s.edge_N[0, 0, 0].item() == 3
    assert abs(float(s.edge_Q[0, 0, 0]) - 0.5) < 1e-6


@pytest.mark.slow
def test_finds_mate_in_one():
    """The check §12 names, and a property of chess rather than of this code.

    At the v0 budget of `n = 800`, and with exploration noise off as at evaluation
    time. Both matter: with first-play urgency at 0 (§6.6) an unvisited edge is
    scored as a loss, so a mate whose prior is 0.017 under an untrained network is
    not reached until `pb_c * P * sqrt(N_v)` clears the visited edges' `Q`, which
    happens between `n = 400` and `n = 800` here. That is AlphaZero's behaviour
    and not a defect, and it is one concrete form of §14.2's low-`n` problem.
    """
    b, c = from_fen(MATE_IN_1)
    s = make(n=800, B=1, boards=b, control=c, tau_plies=0, eps=0.0)
    with torch.no_grad():
        record = s.self_play_move()
    assert bool(s.game_done[0]), "the mate was not played"
    assert int(record.result[0]) == -1
    code, _ = env.terminal(*env.movegen(s.game_board, s.game_control),
                           s.game_control, s.game_board)
    assert int(code[0]) == env.CHECKMATE

    best = int(s.edge_N[0, 0].argmax())
    child = int(s.edge_child[0, 0, best])
    assert int(s.node_flags[0, child] & TERMINAL) == env.CHECKMATE
    # The mate is a loss for the player to move at the child, so 0 in the [0,1]
    # convention, and the edge that reaches it is worth 1 to whoever played it.
    assert float(s.node_value[0, child]) == 0.0
    assert abs(float(s.edge_Q[0, 0, best]) - 1.0) < 1e-6
    assert int(s.node_nedges[0, child]) == 0, "invariant 4: a terminal node was expanded"


def test_terminal_children_are_never_expanded():
    """Invariant 4, over whatever terminals a real search happens to reach."""
    b, c = from_fen(MATE_IN_1)
    s = make(n=300, B=2, boards=b, control=c)
    with torch.no_grad():
        s.self_play_move()
    live = (torch.arange(s.config.n_max, device=DEVICE)[None, :] < s.node_count[:, None])
    term = live & ((s.node_flags & TERMINAL) != 0)
    assert int(term.sum()) > 0, "no terminal node was reached, so nothing was tested"
    assert int(s.node_nedges[term].sum()) == 0


# --------------------------------------------------------------------------- #
# §6.4, the edge enumeration
# --------------------------------------------------------------------------- #

def test_canonical_edge_order_and_promotion_expansion():
    b, c = from_fen(PROMOTION)
    s = make(n=1, B=1, boards=b, control=c)
    with torch.no_grad():
        s.root_init()

    ne = int(s.node_nedges[0, 0])
    labels = s.edge_move[0, 0, :ne].long().tolist()
    moves = [x & ((1 << MOVE_BITS) - 1) for x in labels]
    promos = [x >> MOVE_BITS for x in labels]

    # Ascending by slot then by square, with the four promotion types in spec §3's
    # N B R Q order inside one move.
    assert sorted(zip(moves, promos)) == list(zip(moves, promos))

    mask, _ = env.movegen(b, c)
    bits = int(env.bitset_to_bool(mask).sum())
    n_promo = sum(1 for m in set(moves) if moves.count(m) == 4)
    assert n_promo == 1, f"expected one promoting move, got {n_promo}"
    assert ne == bits + 3, f"{bits} mask bits should give {bits + 3} edges, got {ne}"

    prior = s.edge_prior[0, 0, :ne].float()
    assert abs(float(prior.sum()) - 1.0) < 2e-3
    assert float(s.edge_prior[0, 0, ne:].abs().max()) == 0.0


def test_priors_carry_the_promotion_factorisation():
    """The four edges of one promoting move split that move's mass by `promo`."""
    b, c = from_fen(PROMOTION)
    s = make(n=1, B=1, boards=b, control=c, eps=0.0)
    with torch.no_grad():
        policy, promo, _ = s.evaluate(b, c, torch.zeros(1, dtype=torch.uint8, device=DEVICE))
        s.root_init()
    ne = int(s.node_nedges[0, 0])
    labels = s.edge_move[0, 0, :ne].long()
    moves = labels & ((1 << MOVE_BITS) - 1)
    promos = labels >> MOVE_BITS
    group = (moves == moves[promos == 3][0]).nonzero(as_tuple=True)[0]
    assert group.numel() == 4
    share = s.edge_prior[0, 0, group].float()
    share = share / share.sum()
    want = torch.softmax(promo[0, int(moves[group[0]]) // 64].float(), -1)
    assert torch.allclose(share, want, atol=3e-3), f"{share.tolist()} vs {want.tolist()}"


def test_truncation_is_counted_with_its_prior_mass():
    """§4.3's bet, forced to fail by an `E` far below the branching factor."""
    s = make(n=8, B=2, E=4)
    with torch.no_grad():
        s.self_play_move()
    snap = s.stats.snapshot()
    assert snap["n_truncated"] > 0
    assert snap["max_edges"] >= 20, "the start position alone has 20 legal moves"
    assert 0.0 < snap["truncated_prior_mass"]
    assert int(s.node_nedges.max()) == 4
    assert snap["n_empty_mask_expansions"] == 0


def test_no_truncation_at_the_specified_cap():
    s = make(n=32, B=4, E=64)
    with torch.no_grad():
        for _ in range(3):
            s.self_play_move()
            s.reset_finished()
    assert s.stats.snapshot()["n_truncated"] == 0


# --------------------------------------------------------------------------- #
# §7, the in-tree repetition window
# --------------------------------------------------------------------------- #

def _window_count(s: Search, child_hash, child_irrev):
    return s._repetition_count(child_hash, child_irrev)


def test_rep_window_counts_the_ring_and_the_path():
    """§7: "the game so far" is the ring plus the path from the root to the parent."""
    s = make(n=8, B=1)
    h = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
    s.game_ring[0, :2] = torch.tensor([0xABCD, 0x1111], dtype=torch.int64, device=DEVICE)
    s.game_ring_len[0] = 2
    s.node_hash[0, :4] = torch.tensor([0x2222, 0xABCD, 0x3333, 0xABCD],
                                      dtype=torch.int64, device=DEVICE)
    s.node_flags[0, :4] = 0
    s.path_node[0, :3] = torch.tensor([0, 1, 2], dtype=torch.int16, device=DEVICE)
    s.path_len[0] = 3
    no = torch.zeros(1, dtype=torch.bool, device=DEVICE)
    # One in the ring, one on the path (node 1), plus the child itself.
    assert int(_window_count(s, h, no)[0]) == 3

    # An irreversible move into the child empties the window entirely.
    assert int(_window_count(s, h, ~no)[0]) == 1


def test_rep_window_stops_at_an_irreversible_move():
    """An irreversible move inside the tree hides the ring and everything above it."""
    s = make(n=8, B=1)
    h = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
    s.game_ring[0, 0] = 0xABCD
    s.game_ring_len[0] = 1
    s.node_hash[0, :4] = torch.tensor([0xABCD, 0x1111, 0xABCD, 0x4444],
                                      dtype=torch.int64, device=DEVICE)
    s.path_node[0, :3] = torch.tensor([0, 1, 2], dtype=torch.int16, device=DEVICE)
    s.path_len[0] = 3
    no = torch.zeros(1, dtype=torch.bool, device=DEVICE)

    s.node_flags[0, :4] = 0
    # Ring (1) + root (1) + node 2 (1) + itself.
    assert int(_window_count(s, h, no)[0]) == 4

    # Node 1's incoming move was irreversible: the ring and the root drop out.
    s.node_flags[0, 1] = IRREVERSIBLE
    assert int(_window_count(s, h, no)[0]) == 2

    # The root's own flag is the real game's business, and the ring already
    # accounts for it, so it must not cut the walk.
    s.node_flags[0, 1] = 0
    s.node_flags[0, 0] = IRREVERSIBLE
    assert int(_window_count(s, h, no)[0]) == 4


def test_threefold_inside_the_tree_ends_the_line():
    """A ring loaded with the root position turns a four-ply loop into a draw.

    Nothing forces the search down that loop, so this asserts on the counter
    rather than on a particular node: with the root at its second occurrence,
    any return to it inside the tree is the third and code 4 has to appear.
    """
    fen = "7k/8/8/8/8/8/8/R6K w - - 20 1"
    b, c = from_fen(fen)
    s = make(n=400, B=1, boards=b, control=c)
    s.game_ring[0, 0] = s.game_hash[0]
    s.game_ring_len[0] = 1
    with torch.no_grad():
        s.self_play_move()
    live = (torch.arange(s.config.n_max, device=DEVICE)[None, :] < s.node_count[:, None])
    codes = torch.bincount((s.node_flags[live] & TERMINAL).long(), minlength=6)
    assert int(codes[env.REPETITION]) > 0, f"no repetition reached, codes {codes.tolist()}"
    check_invariants(s)


# --------------------------------------------------------------------------- #
# §10 and §6.7
# --------------------------------------------------------------------------- #

def test_record_schema():
    s = make(n=32, B=3, E=64)
    with torch.no_grad():
        rec = s.self_play_move()
    K = s.config.E   # §10 stores the whole edge set, not only the visited edges
    assert rec.board.shape == (3, 32) and rec.board.dtype == torch.int16
    assert rec.policy_move.shape == (3, K) and rec.policy_prob.shape == (3, K)
    assert rec.rep.dtype == torch.uint8 and int(rec.rep.max()) <= 2
    for row in range(3):
        k = int(rec.policy_len[row])
        assert 0 < k <= K
        assert abs(float(rec.policy_prob[row, :k].float().sum()) - 1.0) < 2e-3
        assert float(rec.policy_prob[row, k:].abs().max()) == 0.0
        assert int(rec.played[row]) in rec.policy_move[row, :k].tolist()


def test_argmax_after_tau_and_sampling_before():
    """§6.7's temperature schedule, and that both branches stay inside the tree."""
    s = make(n=64, B=8, tau_plies=0, seed=3)
    with torch.no_grad():
        rec = s.self_play_move()
    best = s.edge_N[:, 0].argmax(-1)
    want = s.edge_move[s._b, 0, best]
    assert rec.played.tolist() == want.tolist()

    s = make(n=64, B=8, tau_plies=99, seed=3)
    with torch.no_grad():
        rec = s.self_play_move()
    visited = s.edge_N[:, 0] > 0
    played = (s.edge_move[:, 0] == rec.played[:, None]) & visited
    assert bool(played.any(-1).all()), "a move with no visits was played"


def test_the_played_move_advances_the_real_game():
    s = make(n=32, B=2, tau_plies=0)
    before = s.game_board.clone()
    with torch.no_grad():
        rec = s.self_play_move()
    label = rec.played.long()
    want, wc, _, _ = env.step(before, torch.ones(2, dtype=torch.int16, device=DEVICE),
                              label & ((1 << MOVE_BITS) - 1),
                              promo=(label >> MOVE_BITS) & 0b11)
    assert torch.equal(s.game_board, want) and torch.equal(s.game_control, wc)
    assert s.game_ply.tolist() == [1, 1]
    assert s.game_ring_len.tolist() == [1, 1] or s.game_ring_len.tolist() == [0, 0]


def test_every_child_terminal_spends_the_budget_on_stored_values():
    """§6.3's deliberate waste, and the fifty-move rule reaching into the tree.

    At clock 99 every move takes the clock to 101 and ends the game, so the whole
    tree is one ply deep: the first simulations create terminal children and the
    rest re-back-up values they already knew. A descent that creates no node still
    occupies its slot in the encoder batch, which is what §15.1 counts.
    """
    b, c = from_fen(FIFTY_MOVE)
    s = make(n=64, B=1, boards=b, control=c, tau_plies=0)
    with torch.no_grad():
        s.self_play_move()
    check_invariants(s)
    live = torch.arange(s.config.n_max, device=DEVICE) < s.node_count[0]
    created = int(s.node_count[0]) - 1
    assert int((s.node_flags[0][live][1:] & TERMINAL).eq(env.FIFTY_MOVE).all())
    assert int(s.node_nedges[0][live][1:].sum()) == 0
    snap = s.stats.snapshot()
    assert snap["n_terminal_children"] == created
    assert snap["n_terminal_descents"] == 64 - created
    assert bool(s.game_done[0]) and int(s.game_result[0]) == 0
    # PUCT reaches only a handful of the root's moves before the certain draws
    # stop it exploring, so the pool is nowhere near full. §15.1's pool-fill
    # shortfall is exactly this shape.
    assert 0 < created < int(s.node_nedges[0, 0])


def test_a_finished_game_is_not_searched():
    b, c = from_fen(FIFTY_MOVE)
    s = make(n=32, B=1, boards=b, control=c, tau_plies=0)
    with torch.no_grad():
        s.self_play_move()
    assert bool(s.game_done[0])
    try:
        with torch.no_grad():
            s.self_play_move()
    except AssertionError:
        pass
    else:
        raise AssertionError("invariant 8 was not enforced")
    rows = s.reset_finished()
    assert rows.tolist() == [0] and s.game_ply.tolist() == [0]
    with torch.no_grad():
        s.self_play_move()


# --------------------------------------------------------------------------- #
# §11's structural hook
# --------------------------------------------------------------------------- #

def test_per_game_budget():
    """The simulation loop reads `budget[b]`, which playout cap randomisation needs."""
    s = make(n=32, B=3)
    s.budget[0] = 5
    s.budget[1] = 12
    with torch.no_grad():
        s.self_play_move()
    assert s.node_count.tolist() == [6, 13, 33]
    check_invariants(s)


# --------------------------------------------------------------------------- #
# §15
# --------------------------------------------------------------------------- #

def test_stats_track_the_fixed_sizes():
    s = make(n=64, B=4)
    with torch.no_grad():
        s.self_play_move()
    snap = s.stats.snapshot()
    assert snap["simulations"] == 256 and snap["moves"] == 4
    assert 0 < snap["max_edges"] <= 64
    assert snap["max_depth"] >= snap["depth_p99"] >= snap["depth_p50"] >= 1
    assert snap["max_nodes_used"] == 65
    assert snap["n_empty_mask_expansions"] == 0
    assert 0.0 <= snap["search_disagrees_frac"] <= 1.0
    assert 0.0 < snap["mean_root_max_pi"] <= 1.0


# --------------------------------------------------------------------------- #
# One ply, exactly, over every root edge
# --------------------------------------------------------------------------- #

FORCED = {
    "white to move": MATE_IN_1,
    "black to move": "4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",   # mirrored and colour-swapped
    "stalemate available": "k7/2Q5/8/8/8/8/8/K7 w - - 0 1",      # Qb6 is stalemate
    "insufficient available": "8/8/8/4k3/8/3n4/8/4KB2 w - - 0 1",  # Bxd3 leaves K+B v K
}


def test_one_ply_is_exact_on_every_root_edge():
    """Every root edge visited exactly once, against arithmetic and the engine.

    This is the test the forced-mate check should have been. It does not depend on
    the exploration schedule at all: each row of the batch is forced down a
    different root edge, so `edge_Q` after one visit has to be exactly
    `1 - node_value(child)` for all of them, and a terminal child's stored value
    has to be exactly `(result + 1) / 2`.

    Row `j` is forced onto edge `j` by its prior. That works only from the second
    simulation on, because at `N_v = 0` every score is 0 and AlphaZero takes the
    first edge whatever the priors say, which is what the first block checks.
    """
    for name, fen in FORCED.items():
        b, c = from_fen(fen)
        s = make(n=8, B=64, boards=b, control=c, eps=0.0)
        with torch.no_grad():
            s.root_init()
            ne = int(s.node_nedges[0, 0])
            assert ne <= 64
            s.simulate(0)

            # Simulation 0: every row took edge 0, since sqrt(N_v) was 0.
            child0 = s.edge_child[:, 0, 0].long()
            assert bool((s.edge_N[:, 0, 0] == 1).all()), f"{name}: edge 0 was not taken"
            want0 = 1.0 - s.node_value[s._b, child0].float()
            assert torch.allclose(s.edge_Q[:, 0, 0], want0, atol=1e-7), name
            first = _check_child(s, name, torch.zeros(1, dtype=torch.long, device=DEVICE),
                                 torch.zeros(1, dtype=torch.long, device=DEVICE))

            rows = torch.arange(1, ne, device=DEVICE)
            s.edge_prior[:, 0, :ne] = 1e-4
            s.edge_prior[rows, 0, rows] = 1.0
            s.simulate(1)

        assert bool((s.edge_N[rows, 0, rows] == 1).all()), f"{name}: the force did not take"
        rest = _check_child(s, name, rows, rows)
        FORCED_SEEN[name] = [x + y for x, y in zip(first, rest)]


FORCED_SEEN: dict = {}


def _check_child(s, name, rows, edges):
    """For row `r` and root edge `e`: the parity, and §8's terminal mapping.

    The mapping is checked against the engine replaying the move rather than
    against what the search stored, so a wrong `result` conversion cannot agree
    with itself. Returns the histogram of terminal codes seen.
    """
    label = s.edge_move[rows, 0, edges].long()
    child = s.edge_child[rows, 0, edges].long()
    assert bool((child >= 0).all()), name

    got = s.edge_Q[rows, 0, edges]
    want = 1.0 - s.node_value[rows, child].float()
    assert torch.allclose(got, want, atol=1e-7), f"{name}: parity"

    nb, nc, _, _ = env.step(s.node_board[rows, 0], s.node_control[rows, 0],
                            label & ((1 << MOVE_BITS) - 1),
                            promo=(label >> MOVE_BITS) & 0b11)
    code, result = env.terminal(*env.movegen(nb, nc), nc, nb)
    stored = s.node_flags[rows, child] & TERMINAL
    assert stored.tolist() == code.tolist(), f"{name}: terminal codes disagree"
    term = code != 0
    if bool(term.any()):
        want_v = (result[term].float() + 1.0) / 2.0
        assert torch.allclose(s.node_value[rows[term], child[term]].float(),
                              want_v, atol=1e-7), f"{name}: terminal value"
        assert int(s.node_nedges[rows[term], child[term]].sum()) == 0
    return torch.bincount(code.long(), minlength=6).tolist()


def test_the_forced_sweep_covered_every_terminal_kind():
    """The sweep above is worth nothing if the codes it claims to test never appear."""
    if not FORCED_SEEN:
        test_one_ply_is_exact_on_every_root_edge()
    seen = [sum(v[i] for v in FORCED_SEEN.values()) for i in range(6)]
    for code, label in ((env.CHECKMATE, "checkmate"), (env.STALEMATE, "stalemate"),
                        (env.INSUFFICIENT, "insufficient material")):
        assert seen[code] > 0, f"no {label} among the forced children: {FORCED_SEEN}"


def test_a_revisited_terminal_keeps_its_value():
    """Sixty-three visits to one mate, all of which must back up the same 1.

    Invariant 7's visit accounting excludes terminal nodes, having no edges to
    sum, so nothing else covers the case where a simulation ends on a node it has
    already seen.
    """
    b, c = from_fen(MATE_IN_1)
    s = make(n=64, B=1, boards=b, control=c, eps=0.0)
    with torch.no_grad():
        s.root_init()
        ne = int(s.node_nedges[0, 0])
        mate = [i for i in range(ne)
                if int(s.edge_move[0, 0, i]) == 12 * 64 + 60][0]     # Re1-e8
        s.simulate(0)
        s.edge_prior[0, 0, :ne] = 1e-4
        s.edge_prior[0, 0, mate] = 1.0
        for k in range(1, 64):
            s.simulate(k)
    child = int(s.edge_child[0, 0, mate])
    assert int(s.node_flags[0, child] & TERMINAL) == env.CHECKMATE
    assert int(s.edge_N[0, 0, mate]) == 63
    assert float(s.edge_Q[0, 0, mate]) == 1.0
    assert float(s.node_value[0, child]) == 0.0
    assert int(s.node_count[0]) == 3, "a terminal node was allocated more than once"
    assert s.stats.snapshot()["n_terminal_descents"] == 62


def _force(s, node: int, edge: int, row: int = 0) -> None:
    """Make the descent take `edge` at `node`, whatever the network thinks.

    Zeroing every other edge's prior *and* its `Q` makes their score exactly 0
    (§6.6 gives an unvisited edge `Q = 0` too), while the target keeps a prior of
    1, so it wins by `pb_c` as long as the node has been visited once. It does not
    touch a visit count, so the accounting stays intact.
    """
    n_edges = int(s.node_nedges[row, node])
    s.edge_prior[row, node, :n_edges] = 0.0
    s.edge_Q[row, node, :n_edges] = 0.0
    s.edge_prior[row, node, edge] = 1.0


def test_a_terminal_two_plies_down_flips_once():
    """A real mate at path length 2, reached by a real descent.

    White's only non-losing tries are pawn moves; after Kg1 Black mates with Re1.
    The descent is forced down that line rather than left to the exploration
    schedule, which does not reach a depth-2 mate here even at `n = 1024`.

    The two levels must disagree, and a draw would prove nothing since 0.5 is its
    own complement: Black's move into the mate is worth 1 to Black, and White's
    move above it is worth 0 to White.
    """
    b, c = from_fen("4r1k1/5ppp/8/8/8/8/5PPP/7K w - - 0 1")
    s = make(n=8, B=1, boards=b, control=c, eps=0.0)
    KG1, RE1 = 15 * 64 + 6, 28 * 64 + 4
    with torch.no_grad():
        s.root_init()
        ne = int(s.node_nedges[0, 0])
        i = [k for k in range(ne) if int(s.edge_move[0, 0, k]) == KG1][0]
        s.simulate(0)                          # any edge, only to make `pb_c` nonzero

        _force(s, 0, i)
        s.simulate(1)                          # root -> Kg1, which creates the child
        child = int(s.edge_child[0, 0, i])
        assert int(s.node_control[0, child]) < 0, "the child is not Black to move"
        cn = int(s.node_nedges[0, child])
        j = [k for k in range(cn) if int(s.edge_move[0, child, k]) == RE1][0]

        # The child needs one visit before its priors can steer anything: at
        # `N_v = 0` every score is 0 and §6.6 takes edge 0 whatever the priors say.
        s.simulate(2)
        _force(s, child, j)
        # And the mate visit has to be the only one counted on the root edge, or
        # its mean mixes in the two earlier simulations' estimates.
        s.edge_N[0, 0, i] = 0
        s.edge_Q[0, 0, i] = 0.0
        s.simulate(3)                          # root -> Kg1 -> Re1, which is mate

    mate = int(s.edge_child[0, child, j])
    assert int(s.node_flags[0, mate] & TERMINAL) == env.CHECKMATE
    assert int(s.path_len[0]) == 2
    assert float(s.node_value[0, mate]) == 0.0
    assert float(s.edge_Q[0, child, j]) == 1.0, "Black's mating move must be worth 1"
    assert float(s.edge_Q[0, 0, i]) == 0.0, "White's move into a forced mate must be worth 0"


# --------------------------------------------------------------------------- #
# Both colours, and a batch that is not homogeneous (spec §9)
# --------------------------------------------------------------------------- #

def test_black_to_move_root():
    b, c = from_fen("4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    s = make(n=64, B=2, boards=b, control=c)
    assert int(s.game_control[0]) < 0
    with torch.no_grad():
        rec = s.self_play_move()
    check_invariants(s)
    mask, _ = env.movegen(b, c)
    label = rec.played.long()
    legal = env.bitset_to_bool(mask).reshape(1, 2048)
    for row in range(2):
        move = int(label[row]) & ((1 << MOVE_BITS) - 1)
        assert bool(legal[0, move]), "an illegal move was played for Black"


def test_mixed_side_and_mixed_position_batch():
    """spec §9: no part of the search may assume the batch is homogeneous."""
    from tests.boards import random_positions

    even, ec, _ = random_positions(16, plies=8, seed=5)
    odd, oc, _ = random_positions(16, plies=9, seed=6)
    boards = torch.cat([even, odd])
    control = torch.cat([ec, oc])
    black = int((control < 0).sum())
    assert 0 < black < 32, f"the batch is homogeneous: {black} of 32 black to move"

    s = make(n=32, B=32)
    s.reset(boards, control)
    with torch.no_grad():
        for _ in range(3):
            s.self_play_move()
            check_invariants(s)
            s.reset_finished()


# --------------------------------------------------------------------------- #
# A constant evaluator, where the whole tree is determined by the rules
# --------------------------------------------------------------------------- #

def constant_evaluator(value: float = 0.0):
    """Uniform priors and one value. The search becomes a function of the rules."""
    def evaluate(boards, control, rep):
        n = boards.shape[0]
        return (torch.zeros((n, 32, 64), device=boards.device),
                torch.zeros((n, 32, 4), device=boards.device),
                torch.full((n,), value, device=boards.device))
    return evaluate


def test_constant_evaluator_gives_one_value_everywhere():
    """Every backed-up value is 0.5, and 0.5 is its own complement, so exactly 0.5.

    Catches any stray arithmetic in the running mean or the parity that a
    tolerance-based comparison against a real network would absorb.
    """
    cfg = SearchConfig(n=64, B=4, E=64, eps=0.0)
    s = Search(cfg, evaluate=constant_evaluator(0.0), device=DEVICE, seed=0)
    s.reset(*env.initial_boards(4, device=DEVICE))
    with torch.no_grad():
        s.self_play_move()
    check_invariants(s)
    visited = s.edge_N > 0
    assert int(visited.sum()) > 0
    assert bool((s.edge_Q[visited] == 0.5).all()), \
        f"values drifted off 0.5: {s.edge_Q[visited].unique().tolist()[:5]}"
    live = torch.arange(cfg.n_max, device=DEVICE)[None, :] < s.node_count[:, None]
    assert bool((s.node_value[live].float() == 0.5).all())


def test_constant_evaluator_maps_the_value_range():
    """§3.5's `(v + 1) / 2`, at a value where the mistaken `(1 - v) / 2` differs.

    ⚠️ A constant evaluator at 0 cannot see this: `(0+1)/2` and `(1-0)/2` are both
    0.5, so the whole tree agrees under either convention. 0.5 is the smallest
    value that separates them, giving 0.75 against 0.25.
    """
    cfg = SearchConfig(n=32, B=2, E=64, eps=0.0)
    s = Search(cfg, evaluate=constant_evaluator(0.5), device=DEVICE, seed=0)
    s.reset(*env.initial_boards(2, device=DEVICE))
    with torch.no_grad():
        s.self_play_move()
    live = torch.arange(cfg.n_max, device=DEVICE)[None, :] < s.node_count[:, None]
    assert bool((s.node_value[live].float() == 0.75).all()), \
        f"stored {s.node_value[live].float().unique().tolist()[:4]}, want 0.75"
    # Every path here has odd length at least 1, and the edge into the leaf always
    # flips, so a depth-1 edge holds 0.25 rather than 0.75.
    root_edges = s.edge_N[:, 0] > 0
    assert bool((s.edge_Q[:, 0][root_edges] <= 0.75).all())
    assert float(s.edge_Q[s.edge_N > 0].min()) == 0.25


def test_constant_evaluator_is_deterministic_across_rows():
    """With no noise and one evaluator, identical positions must give identical trees.

    Any dependence on row index, any uninitialised read, and any tie broken by
    something other than the edge index shows up here as a difference between rows.
    """
    cfg = SearchConfig(n=48, B=8, E=64, eps=0.0)
    s = Search(cfg, evaluate=constant_evaluator(0.0), device=DEVICE, seed=0)
    s.reset(*env.initial_boards(8, device=DEVICE))
    with torch.no_grad():
        s.self_play_move()
    for name in ("node_board", "node_control", "node_hash", "node_flags", "node_nedges",
                 "node_parent", "node_pedge", "edge_move", "edge_N", "edge_Q",
                 "edge_child", "edge_prior"):
        t = getattr(s, name)
        assert bool((t == t[0]).all()), f"{name} differs between identical games"


# --------------------------------------------------------------------------- #
# §6.1's noise and §6.7's sampling, in isolation
# --------------------------------------------------------------------------- #

def test_dirichlet_marginals():
    """`Dir(alpha)`'s marginal is `Beta(alpha, (k-1) alpha)`, which pins both moments."""
    cfg = SearchConfig(n=4, B=8192, E=64, alpha=0.3)
    s = Search(cfg, evaluate=constant_evaluator(), device=DEVICE, seed=11)
    for k in (2, 20, 64):
        valid = (torch.arange(64, device=DEVICE)[None, :] < k).expand(8192, -1)
        eta = s.dirichlet(valid)
        a, total = cfg.alpha, cfg.alpha * k
        mean, var = a / total, a * (total - a) / (total ** 2 * (total + 1))
        x = eta[:, :k]
        assert torch.allclose(x.sum(-1), torch.ones(8192, device=DEVICE), atol=1e-5)
        if k < 64:
            assert float(eta[:, k:].abs().max()) == 0.0, "noise leaked past the valid edges"
        assert abs(float(x.mean()) - mean) < 0.02 * mean, f"k={k} mean"
        assert abs(float(x.var()) - var) < 0.06 * var, f"k={k} var"


def test_dirichlet_mixes_at_eps():
    """§6.1's mixture, with the noise replaced by something known."""
    class Fixed(Search):
        def dirichlet(self, valid):
            eta = torch.zeros(valid.shape, device=valid.device)
            eta[:, 0] = 1.0                       # all the mass on the first edge
            return eta

    torch.manual_seed(0)
    net = BrokefishNet().to(DEVICE).eval()
    cfg = SearchConfig(n=1, B=1, E=64, eps=0.25)
    plain = Search(cfg, evaluate=make_evaluator(net), device=DEVICE, seed=0)
    noisy = Fixed(cfg, evaluate=make_evaluator(net), device=DEVICE, seed=0)
    for s in (plain, noisy):
        s.reset(*env.initial_boards(1, device=DEVICE))
    plain.config = SearchConfig(n=1, B=1, E=64, eps=0.0)
    with torch.no_grad():
        plain.root_init()
        noisy.root_init()
    ne = int(plain.node_nedges[0, 0])
    p = plain.edge_prior[0, 0, :ne].float()
    got = noisy.edge_prior[0, 0, :ne].float()
    want = 0.75 * p
    want[0] += 0.25
    assert torch.allclose(got, want, atol=2e-3), f"{got[:4].tolist()} vs {want[:4].tolist()}"
    assert abs(float(got.sum()) - 1.0) < 2e-3


def test_temperature_one_samples_from_the_visit_distribution():
    """§6.7's sampling branch, with the visit counts hand-set so pi is known."""
    B = 8192
    cfg = SearchConfig(n=8, B=B, E=64, tau_plies=99)
    s = Search(cfg, evaluate=constant_evaluator(), device=DEVICE, seed=5,
               check_invariants=False)
    s.reset(*env.initial_boards(B, device=DEVICE))
    s.root_rep = torch.zeros(B, dtype=torch.uint8, device=DEVICE)
    s.node_nedges[:, 0] = 4
    s.node_flags[:, 0] = 0
    s.edge_N[:, 0, :4] = torch.tensor([4, 2, 1, 1], dtype=torch.int16, device=DEVICE)
    s.edge_move[:, 0, :4] = torch.tensor([10, 20, 30, 40], dtype=torch.int16, device=DEVICE)
    with torch.no_grad():
        rec = s.select_and_advance()
    counts = torch.bincount(rec.played.long(), minlength=41)[[10, 20, 30, 40]].float()
    got = (counts / B).tolist()
    want = [0.5, 0.25, 0.125, 0.125]
    assert int(counts.sum()) == B, "a move outside the root's edges was played"
    for g, w in zip(got, want):
        assert abs(g - w) < 0.02, f"{got} vs {want}"


# --------------------------------------------------------------------------- #
# §4.3's truncation, on its ordering rather than only its count
# --------------------------------------------------------------------------- #

def test_truncation_keeps_the_top_e_in_canonical_order():
    """§6.4 step 3: the E highest unnormalised priors, written in canonical order."""
    E = 8
    boards, control = env.initial_boards(1, device=DEVICE)
    s = make(n=1, B=1, E=E, boards=boards[0:1], control=control[0:1], eps=0.0)
    rep = torch.zeros(1, dtype=torch.uint8, device=DEVICE)
    with torch.no_grad():
        policy, promo, _ = s.evaluate(boards, control, rep)
        s.root_init()

    # The same candidate set and the same logits, rebuilt from the raw heads.
    mask, _ = env.movegen(boards, control)
    legal = env.bitset_to_bool(mask)[0]                       # [32,64]
    lp = torch.log_softmax(promo[0].float(), dim=-1)
    cands = []
    for p in range(32):
        for sq in range(64):
            if not bool(legal[p, sq]):
                continue
            base = float(policy[0, p, sq])
            promotes = bool(s._promotion_targets(boards, control)[0, p, sq])
            for k in (range(4) if promotes else (0,)):
                cands.append((p * 64 + sq, k, base + (float(lp[p, k]) if promotes else 0.0)))
    assert len(cands) > E, f"only {len(cands)} candidates, nothing to truncate"

    keep = sorted(cands, key=lambda t: (-t[2], t[0], t[1]))[:E]
    want = sorted((label | (k << MOVE_BITS)) for label, k, _ in keep)
    assert int(s.node_nedges[0, 0]) == E
    assert s.edge_move[0, 0, :E].long().tolist() == want
    assert s.stats.snapshot()["truncated_prior_mass"] > 0.0


def _is_slow(fn) -> bool:
    return any(m.name == "slow" for m in getattr(fn, "pytestmark", []))


def _tests(slow: bool):
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v) and (slow or not _is_slow(v))]


def main() -> int:
    slow = "--slow" in sys.argv
    tests = _tests(slow)
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures += 1
            print(f"  {name:52s} FAIL {type(exc).__name__}: {exc}")
        else:
            print(f"  {name:52s} OK")
    print(f"\n{len(tests) - failures}/{len(tests)} passed"
          + ("" if slow else "   (--slow adds the forced-mate check)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
