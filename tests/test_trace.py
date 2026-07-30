"""The search recorder, against `docs/debugger.md` §12.

Five checks, and they divide cleanly. Two are about the recorder not lying about
the search: a traced run has to be bit-identical to an untraced one (§8), and the
deltas it wrote have to replay back into the tree it recorded (§5). Two are about
the move translation layer, where python-chess is an external oracle: every edge
label has to name a move python-chess calls legal in that node's position, and
walking a node's own move sequence from the root has to arrive at its position.
The last is the schema.

Check 3 is the one carrying the weight. A debugger that mislabels moves is a
debugger that lies fluently, and every other check here would pass while it did.

    python -m tests.test_trace [--slow]
"""

from __future__ import annotations

import sys

import chess
import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.nn.model import BrokefishNet
from brokefish.search import Search, SearchConfig, make_evaluator
from brokefish.search.trace import (
    ENDINGS,
    FORMAT,
    TracingSearch,
    puct,
    read_trace,
    replay,
    write_trace,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# A position with promotions available, so the four-edges-per-pawn-move half of
# the translation layer is exercised rather than assumed.
PROMOTION = "8/P6k/8/8/8/8/6p1/K7 w - - 0 1"
# 218 legal moves at the root, well past `E = 64`, so truncation is on.
WIDE = "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1"


def net_and_eval(seed: int = 0):
    torch.manual_seed(seed)
    net = BrokefishNet().to(DEVICE).eval()
    return net, make_evaluator(net)


def traced(n=24, fen=None, seed=0, **kw):
    """One recorded move. Returns ``(search, trace)``."""
    _, ev = net_and_eval(seed)
    s = TracingSearch(SearchConfig(n=n, B=1, **kw), evaluate=ev, device=DEVICE, seed=seed)
    if fen is None:
        b, c = env.initial_boards(1, device=DEVICE)
    else:
        b, c = env.from_fen(fen)
        b, c = b.to(DEVICE).reshape(1, 32), c.to(DEVICE).reshape(1)
    s.reset(b, c)
    s.self_play_move()
    return s, s.trace()


def node_records(trace):
    out = {0: trace["root"]}
    for sim in trace["sims"]:
        if sim["created"] is not None:
            out[sim["created"]["id"]] = sim["created"]
    return out


# -- 1. identity, §8 --------------------------------------------------------- #

@pytest.mark.parametrize("fen", [None, PROMOTION])
def test_recording_does_not_change_the_search(fen):
    """A traced search and an untraced one from the same seed are the same tree."""
    s, _ = traced(fen=fen)
    _, ev = net_and_eval(0)
    plain = Search(SearchConfig(n=24, B=1), evaluate=ev, device=DEVICE, seed=0)
    plain.reset(_boards(fen), _control(fen))
    plain.self_play_move()

    for name in ("node_board", "node_control", "node_hash", "node_value",
                 "node_nedges", "node_flags", "node_parent", "node_pedge",
                 "edge_move", "edge_prior", "edge_child", "edge_N", "edge_Q",
                 "node_count", "game_board", "game_control", "game_hash",
                 "game_ply", "game_done", "game_result"):
        a, b = getattr(s, name), getattr(plain, name)
        assert torch.equal(a, b), f"{name} differs between traced and untraced"


def _boards(fen):
    if fen is None:
        return env.initial_boards(1, device=DEVICE)[0]
    return env.from_fen(fen)[0].to(DEVICE).reshape(1, 32)


def _control(fen):
    if fen is None:
        return env.initial_boards(1, device=DEVICE)[1]
    return env.from_fen(fen)[1].to(DEVICE).reshape(1)


# -- 2. replay, §5 ----------------------------------------------------------- #

@pytest.mark.parametrize("fen", [None, PROMOTION, WIDE])
def test_replay_reproduces_the_whole_tree(fen):
    """§5 over the file alone reproduces every node's `edge_N` and `edge_Q`."""
    s, trace = traced(fen=fen)
    state = replay(trace)
    count = int(s.node_count[0])
    assert len(state["nodes"]) == count, "replay built a different number of nodes"

    for i in range(count):
        ne = int(s.node_nedges[0, i])
        assert s.edge_N[0, i, :ne].tolist() == state["N"][i], f"node {i}: visits"
        want = s.edge_Q[0, i, :ne].tolist()
        for e, (a, b) in enumerate(zip(want, state["Q"][i])):
            # float32 running mean against float64: §3's precision caveat, and
            # the tolerance §12 asks for.
            assert abs(a - b) < 1e-6, f"node {i} edge {e}: Q {a} vs {b}"
        assert s.edge_child[0, i, :ne].tolist() == state["child"][i], f"node {i}: child"


def test_replay_prefix_matches_a_rerun():
    """The state after `k` simulations is the state a `k`-simulation search reaches."""
    _, trace = traced(n=24)
    short, _ = traced(n=8)
    state = replay(trace, k=8)
    for i in range(int(short.node_count[0])):
        ne = int(short.node_nedges[0, i])
        assert short.edge_N[0, i, :ne].tolist() == state["N"][i], f"node {i}"


def test_replay_agrees_with_the_recorded_tail():
    """§5's own check: the viewer's final state against `final_root`."""
    _, trace = traced()
    state = replay(trace)
    assert state["N"][0] == trace["final_root"]["N"]
    for a, b in zip(state["Q"][0], trace["final_root"]["Q"]):
        assert abs(a - b) < 1e-5


def test_derived_puct_ranks_the_edge_the_search_took():
    """§3's derive-do-not-store, checked where it is load-bearing.

    The score is never written to the file, so if deriving it from a replayed
    state were wrong the scrubber of §7.2 would show a search taking edges it did
    not take. Replayed to just before each simulation, the argmax of the derived
    score has to be the edge that simulation's path actually took at the root.
    """
    _, trace = traced(n=48)
    for k, sim in enumerate(trace["sims"]):
        state = replay(trace, k=k)
        scores = puct(trace, state, 0)
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        assert best == sim["path"][0][1], f"simulation {k}: root edge"


# -- 3. move round-trip, §12.3 ----------------------------------------------- #

@pytest.mark.parametrize("fen", [None, PROMOTION, WIDE])
def test_every_edge_is_a_legal_move_python_chess_agrees_with(fen):
    """The whole (slot, square) translation layer against the external oracle."""
    _, trace = traced(fen=fen)
    for node in node_records(trace).values():
        if "edges" not in node:
            continue
        board = chess.Board(node["fen"])
        legal = set(board.legal_moves)
        seen = set()
        for uci, san in zip(node["edges"]["uci"], node["edges"]["san"]):
            move = chess.Move.from_uci(uci)
            assert move in legal, f"node {node['id']}: {uci} illegal in {node['fen']}"
            assert board.san(move) == san, f"node {node['id']}: san {san} for {uci}"
            assert move not in seen, f"node {node['id']}: {uci} twice"
            seen.add(move)
        n_legal, kept = node["n_legal"], len(node["edges"]["move"])
        assert n_legal == len(legal), f"node {node['id']}: {n_legal} vs python-chess"
        assert kept == min(n_legal, trace["config"]["E"])


def test_truncation_is_recorded_where_it_happens():
    """§4.3 fires at `WIDE`, and the trace says so rather than silently dropping."""
    _, trace = traced(fen=WIDE)
    root = trace["root"]
    assert root["n_legal"] == 218
    assert len(root["edges"]["move"]) == trace["config"]["E"]
    assert 0.0 < root["truncated_mass"] < 1.0


# -- 4. position round-trip, §12.4 ------------------------------------------- #

@pytest.mark.parametrize("fen", [None, PROMOTION])
def test_walking_the_san_path_reaches_the_node(fen):
    """Applying a node's own move sequence from the root reproduces its position."""
    _, trace = traced(fen=fen)
    nodes = node_records(trace)
    for node in nodes.values():
        moves, cur = [], node
        while cur["parent"] >= 0:
            parent = nodes[cur["parent"]]
            moves.append(parent["edges"]["san"][cur["pedge"]])
            cur = parent
        board = chess.Board(trace["root"]["fen"])
        for san in reversed(moves):
            board.push_san(san)
        # The fullmove number is emitted as 1 everywhere (§6) and python-chess
        # counts it, so the comparison stops at the halfmove clock.
        assert board.fen().split()[:5] == node["fen"].split()[:5], f"node {node['id']}"
        assert len(moves) == node["depth"]


# -- 5. schema, §4 ----------------------------------------------------------- #

NODE_KEYS = {"id", "parent", "pedge", "depth", "board", "control", "fen", "hash",
             "rep", "rep_raw", "flags", "code", "value", "value_source"}


def check_node(node, root=False):
    assert NODE_KEYS <= set(node), f"node {node.get('id')} is missing {NODE_KEYS - set(node)}"
    assert len(node["board"]) == 32
    assert all(0 <= w < (1 << 16) for w in node["board"])
    assert node["hash"].startswith("0x") and len(node["hash"]) == 18
    assert 0 <= node["value"] <= 1.0
    assert node["value_source"] in ("net", "terminal")
    assert (node["value_source"] == "terminal") == (node["code"] != 0)
    assert 0 <= node["rep"] <= 2 and node["rep_raw"] >= 1
    assert (node["parent"] == -1) == root
    if node["code"]:
        assert "edges" not in node, "a terminal node has edges"
    else:
        e = node["edges"]
        k = len(e["move"])
        assert k and len(e["uci"]) == len(e["san"]) == len(e["prior"]) == k
        assert abs(sum(e["prior"]) - 1.0) < 5e-2, "priors are fp16 and should sum to 1"


def test_schema():
    _, trace = traced(fen=PROMOTION)
    assert trace["format"] == FORMAT
    assert trace["kind"] in ("search", "move")
    assert set(trace["config"]) == {"n", "B", "E", "pb_c_base", "pb_c_init",
                                    "alpha", "eps", "tau_plies", "seed"}
    assert trace["impl"]["search"] == "torch"
    check_node(trace["root"], root=True)
    assert len(trace["root"]["prior_pre_noise"]) == len(trace["root"]["edges"]["move"])
    assert len(trace["root"]["noise"]) == len(trace["root"]["edges"]["move"])

    ids = {0}
    for k, sim in enumerate(trace["sims"]):
        assert sim["s"] == k
        assert sim["ended"] in ENDINGS
        # §4.3's invariant: a descent that stopped on a stored terminal is the
        # only simulation that creates nothing.
        assert (sim["created"] is None) == (sim["ended"] == "terminal_stored")
        assert 1 <= len(sim["path"]) <= trace["config"]["n"]
        assert 0.0 <= sim["value"] <= 1.0
        if sim["created"] is not None:
            check_node(sim["created"])
            assert sim["created"]["id"] not in ids, "a node id was reused"
            ids.add(sim["created"]["id"])
            assert sim["leaf"] == sim["created"]["id"]
        else:
            assert sim["leaf"] in ids
        for v, e in sim["path"]:
            assert v in ids

    n = trace["final_root"]["N"]
    assert sum(n) == trace["config"]["n"]
    assert len(trace["pi"]) == len(n) and abs(sum(trace["pi"]) - 1.0) < 1e-9
    assert 0 <= trace["played"] < len(n)
    assert trace["temperature"] in ("sampled", "argmax")
    assert trace["stats"]["simulations"] == trace["config"]["n"]


def test_write_and_read_round_trip(tmp_path):
    _, trace = traced(n=8)
    path = write_trace(trace, str(tmp_path / "sub" / "t.json"))
    assert read_trace(path) == trace


def test_batched_recorder_is_refused():
    """§13: recording a row of a `B > 1` search is an extension, not a silent slice."""
    _, ev = net_and_eval()
    with pytest.raises(ValueError, match="one game"):
        TracingSearch(SearchConfig(n=8, B=4), evaluate=ev, device=DEVICE)


def test_trace_before_the_move_is_refused():
    _, ev = net_and_eval()
    s = TracingSearch(SearchConfig(n=8, B=1), evaluate=ev, device=DEVICE)
    s.reset()
    with pytest.raises(RuntimeError, match="finished move"):
        s.trace()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"] + sys.argv[1:]))
