"""Recording one search, to the format of ``docs/debugger.md`` §4.

``TracingSearch`` is the reference search of :mod:`brokefish.search.torch_impl`
with the per-simulation delta written out as it happens. It subclasses rather
than reimplements, and every override is the same shape: capture what the parent
returned, then return it unchanged. Nothing here draws a random number, branches
on what it observed, or writes a tree tensor, which is what §8's identity test
enforces and `tests/test_trace.py` checks against an untraced run from the same
seed.

Speed is explicitly not part of the contract (§8): the reference is not on the
self-play path, so the recorder syncs to the host every simulation and builds
python-chess boards inline.

Only the tree deltas are stored. Everything a replay can derive from them --
every PUCT score, every visit-weighted distribution -- is derived in the viewer
(§3), because at ``n = 800`` the scores alone are about 1.5M numbers that are a
function of data already in the file.

Usage, one traced move::

    search = TracingSearch(SearchConfig(n=64, B=1), evaluate=ev, device="cuda")
    search.reset(*env.initial_boards(1, device="cuda"))
    record = search.self_play_move()
    write_trace(search.trace(), "traces/0001.json")
"""

from __future__ import annotations

from dataclasses import replace

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

from brokefish.env.interop import label_to_move, to_chess_board
from brokefish.search.torch_impl import (
    FPU_DRAW,
    MOVE_BITS,
    Search,
    SearchConfig,
    SearchStats,
    TERMINAL,
)

# ``docs/debugger.md`` §4. Bumped whenever a field changes meaning; the viewer
# refuses a version it does not know rather than guessing.
FORMAT = 1

ENDINGS = ("expanded", "terminal_child", "terminal_stored")


class _RecordingStats(SearchStats):
    """``SearchStats`` that also hands ``on_expand``'s locals to the recorder.

    §4.2 wants ``n_legal`` before truncation and the prior mass §4.3 dropped, and
    both are locals of ``_expand``. They already leave it once, as arguments to
    the counter block, so the recorder listens there instead of recomputing them
    and risking a second, differently-wrong answer.
    """

    def __init__(self, d_max: int, device, owner: "TracingSearch") -> None:
        super().__init__(d_max, device)
        self._owner = owner

    def on_expand(self, count, n_edges, dropped, do) -> None:
        super().on_expand(count, n_edges, dropped, do)
        if bool(do[0]):
            self._owner._last_expand = (int(count[0]), float(dropped[0]))


class TracingSearch(Search):
    """A ``B = 1`` search that records itself.

    ``B = 1`` is a restriction of the recorder and not of the format: §13 lists
    tracing one row of a self-play batch as the extension, which costs a slice
    copy per simulation and no format change.
    """

    def __init__(self, config: SearchConfig, evaluate, *,
                 checkpoint: Optional[dict] = None, impl: Optional[dict] = None,
                 kind: str = "search", seed: int = 0, **kw) -> None:
        if config.B != 1:
            raise ValueError(f"the recorder traces one game; got B = {config.B}")
        # ⚠️ **§6.1a's root terminal sweep is forced off, and this is a known gap.**
        # `replay()` -- and its untested JavaScript twin in `static/trace.js` --
        # rebuilds each node's visit counts by accumulating the per-simulation
        # deltas from zero, so the visits the sweep seeds *before* the first
        # simulation are invisible to it and the replayed PUCT scores disagree with
        # what the search actually did. A viewer that silently misrepresents the
        # search is worse than one that refuses the case, so it refuses: tracing a
        # swept search needs `replay` to start from the root record's seeded `N`/`Q`
        # in both implementations, which is owed work (`debugger.md` §4).
        if config.root_terminal_sweep:
            config = replace(config, root_terminal_sweep=False)
        super().__init__(config, evaluate, seed=seed, **kw)
        self.stats = _RecordingStats(config.d_max, self.device, self)
        self.seed = seed
        self.kind = kind
        self.checkpoint = checkpoint
        self.impl = dict(impl) if impl else {}
        self.impl.setdefault("search", "torch")
        self._reset_trace()

    def _reset_trace(self) -> None:
        self.root_record: Optional[dict] = None
        self.sims: List[dict] = []
        self.tail: Optional[dict] = None
        self._last_expand: Optional[tuple] = None
        self._noise: Optional[List[float]] = None
        self._prior_pre_noise: Optional[List[float]] = None
        self._cur: Dict[str, Any] = {}

    # -- capture ------------------------------------------------------------ #

    def root_init(self) -> None:
        self._reset_trace()
        self._last_expand = None
        super().root_init()
        # `root_init` keeps only the clamped feature the network reads, and §4.2
        # wants the count itself, so it is asked for a second time. Over the ring
        # alone, since no tree exists yet (`mcts.md` §7).
        rep = self.env.repetition_count(self.game_hash, self.game_ring, self.game_ring_len)
        self.root_record = self._node_record(0, rep=int(self.root_rep[0]),
                                             rep_raw=int(rep[0]))
        if self._prior_pre_noise is not None:
            self.root_record["prior_pre_noise"] = self._prior_pre_noise
        if self._noise is not None:
            self.root_record["noise"] = self._noise

    def _add_exploration_noise(self) -> None:
        ne = int(self.node_nedges[0, 0])
        self._prior_pre_noise = self.edge_prior[0, 0, :ne].detach().float().tolist()
        super()._add_exploration_noise()

    def dirichlet(self, valid: torch.Tensor) -> torch.Tensor:
        eta = super().dirichlet(valid)
        self._noise = eta[0][valid[0]].tolist()
        return eta

    def simulate(self, s: int) -> None:
        self._cur = {"s": s}
        super().simulate(s)
        leaf = self._cur["leaf"]
        created = self._cur["created"]
        code = self._cur["code"]
        ended = ENDINGS[2] if created is None else (ENDINGS[1] if code else ENDINGS[0])
        self.sims.append({
            "s": s,
            "path": self._cur["path"],
            "leaf": leaf,
            "created": created,
            "value": float(self.node_value[0, leaf].detach()),
            "ended": ended,
        })

    def _descent(self, active):
        leaf, parent, edge = super()._descent(active)
        length = int(self.path_len[0])
        self._cur["path"] = [[int(self.path_node[0, d]), int(self.path_edge[0, d])]
                             for d in range(length)]
        return leaf, parent, edge

    def _repetition_count(self, child_hash, child_irrev):
        rep = super()._repetition_count(child_hash, child_irrev)
        self._cur["rep_raw"] = int(rep[0])
        return rep

    def _create_child(self, leaf, parent, edge, active):
        self._last_expand = None
        out = super()._create_child(leaf, parent, edge, active)
        node, rep, code, _mask, _expand = out
        self._cur["leaf"] = int(node[0])
        self._cur["code"] = int(code[0])
        self._cur["fresh"] = bool(parent[0] >= 0)
        self._cur["rep"] = int(rep[0])
        return out

    def _expand(self, node, mask, do, policy, promo, value) -> None:
        super()._expand(node, mask, do, policy, promo, value)
        if not self._cur:                      # the root, handled in `root_init`
            return
        if not self._cur["fresh"]:
            self._cur["created"] = None
            return
        self._cur["created"] = self._node_record(
            self._cur["leaf"], rep=self._cur["rep"], rep_raw=self._cur["rep_raw"])

    def select_and_advance(self):
        ply = int(self.game_ply[0])
        record = super().select_and_advance()
        ne = int(self.node_nedges[0, 0])
        n = self.edge_N[0, 0, :ne].tolist()
        total = float(sum(n)) or 1.0
        label = int(record.played[0])
        played = self.edge_move[0, 0, :ne].tolist().index(label)
        self.tail = {
            "final_root": {"N": n, "Q": self.edge_Q[0, 0, :ne].detach().tolist()},
            "pi": [x / total for x in n],
            "played": played,
            "temperature": "sampled" if ply < self.config.tau_plies else "argmax",
            "stats": self.stats.snapshot(),
        }
        return record

    # -- node records ------------------------------------------------------- #

    def _node_record(self, node: int, rep: int, rep_raw: Optional[int]) -> dict:
        """§4.2, read off the tree tensors after the node has been expanded."""
        node = int(node)
        words = self.node_board[0, node]
        control = self.node_control[0, node]
        flags = int(self.node_flags[0, node])
        code = flags & TERMINAL
        board = to_chess_board(words.cpu(), control.cpu())
        out = {
            "id": node,
            "parent": int(self.node_parent[0, node]),
            "pedge": int(self.node_pedge[0, node]),
            "depth": _depth(self, node),
            "board": [int(w) & 0xFFFF for w in words.tolist()],
            "control": int(control),
            "fen": board.fen(),
            "hash": f"0x{int(self.node_hash[0, node]) & 0xFFFFFFFFFFFFFFFF:016x}",
            "rep": int(rep),
            "rep_raw": int(rep_raw) if rep_raw is not None else None,
            "flags": flags,
            "code": code,
            "value": float(self.node_value[0, node].detach()),
            "value_source": "terminal" if code else "net",
        }
        ne = int(self.node_nedges[0, node])
        if ne:
            labels = self.edge_move[0, node, :ne].tolist()
            moves = [label_to_move(words, label) for label in labels]
            out["edges"] = {
                "move": labels,
                "uci": [m.uci() for m in moves],
                "san": [board.san(m) for m in moves],
                "prior": self.edge_prior[0, node, :ne].detach().float().tolist(),
            }
            count, dropped = self._last_expand or (ne, 0.0)
            out["n_legal"] = count
            out["truncated_mass"] = dropped
        return out

    # -- the document ------------------------------------------------------- #

    def trace(self) -> dict:
        """The §4 document for the move just searched.

        Valid after ``self_play_move`` (or after ``select_and_advance``); calling
        it mid-search raises, since a trace with no tail cannot be replayed
        against §5's final check.
        """
        if self.root_record is None or self.tail is None:
            raise RuntimeError("trace() needs a finished move: root_init, n "
                               "simulations, then select_and_advance")
        c = self.config
        doc = {
            "format": FORMAT,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": self.kind,
            "config": {
                "n": c.n, "B": c.B, "E": c.E,
                "pb_c_base": c.pb_c_base, "pb_c_init": c.pb_c_init,
                "alpha": c.alpha, "eps": c.eps, "tau_plies": c.tau_plies,
                "seed": self.seed,
            },
            "impl": self.impl,
            "root": self.root_record,
            "sims": self.sims,
        }
        if self.checkpoint is not None:
            doc["checkpoint"] = self.checkpoint
        doc.update(self.tail)
        return doc


def _depth(search: Search, node: int) -> int:
    """Distance to the root over ``node_parent``, which the tree does not store."""
    d = 0
    while node > 0:
        node = int(search.node_parent[0, node])
        d += 1
    return d


def checkpoint_info(path: Optional[str], net=None, weight_gen: int = 0) -> dict:
    """§4.1's checkpoint block. The hash is of the file, so a trace names its net."""
    out: Dict[str, Any] = {"path": path, "weight_gen": weight_gen}
    if path and os.path.exists(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    if net is not None:
        out["params"] = sum(p.numel() for p in net.parameters())
    return out


def write_trace(trace: dict, path: str) -> str:
    """Write one trace as JSON. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def read_trace(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"trace format {doc.get('format')!r}, this code reads {FORMAT}")
    return doc


# -- §5 ---------------------------------------------------------------------- #

def replay(trace: dict, k: Optional[int] = None) -> dict:
    """The tree state after the first ``k`` simulations (all of them by default).

    The Python twin of the viewer's replay, kept here because the test that says
    the format is sufficient has to reconstruct the tree from the file alone.
    Backwards is a replay from zero rather than an undo, since a running mean
    does not invert stably.

    Returns ``{"nodes", "N", "Q", "child"}``, all keyed by node id.
    """
    root = trace["root"]
    nodes = {0: root}
    width = lambda rec: len(rec["edges"]["move"]) if "edges" in rec else 0  # noqa: E731
    N = {0: [0] * width(root)}
    Q = {0: [0.0] * width(root)}
    child = {0: [-1] * width(root)}

    sims = trace["sims"] if k is None else trace["sims"][:k]
    for sim in sims:
        rec = sim["created"]
        if rec is not None:
            i = rec["id"]
            nodes[i] = rec
            N[i] = [0] * width(rec)
            Q[i] = [0.0] * width(rec)
            child[i] = [-1] * width(rec)
            child[rec["parent"]][rec["pedge"]] = i
        path = sim["path"]
        length = len(path)
        for d, (v, e) in enumerate(path):
            # `mcts.md` §6.5. Inverting this parity produces a viewer that shows
            # the search preferring its worst moves, and at L = 2 the wrong
            # parity agrees with the right one, so nothing shallow catches it.
            q = sim["value"] if (length - d) % 2 == 0 else 1.0 - sim["value"]
            N[v][e] += 1
            Q[v][e] += (q - Q[v][e]) / N[v][e]
    return {"nodes": nodes, "N": N, "Q": Q, "child": child}


def puct(trace: dict, state: dict, node: int) -> List[float]:
    """§6.6's score for every edge of ``node``, derived from a replayed state.

    Not stored in the trace and derived here for the same reason the viewer
    derives it (§3): it is a function of ``edge_N``, ``edge_Q``, ``edge_prior``
    and the node's own visit total, all of which the replay has.
    """
    c = trace["config"]
    n, q = state["N"][node], state["Q"][node]
    prior = state["nodes"][node]["edges"]["prior"]
    n_v = float(sum(n))
    import math
    pb_c = math.log((n_v + c["pb_c_base"] + 1.0) / c["pb_c_base"]) + c["pb_c_init"]
    pb_c *= math.sqrt(n_v)
    # First-play urgency. An unvisited edge takes `FPU_DRAW`, not 0: the tree
    # works in [0, 1], where 0 is a certain loss and would make every unexplored
    # move look lost. AGZ's literal 0 belongs to its [-1, 1] frame -- the same
    # slip the oracle carried. §6.6 and `torch_impl._select` agree on 0.5.
    return [pb_c * prior[i] / (1.0 + n[i]) + (q[i] if n[i] > 0 else FPU_DRAW)
            for i in range(len(n))]


__all__ = ["FORMAT", "ENDINGS", "TracingSearch", "checkpoint_info", "write_trace",
           "read_trace", "replay", "puct", "MOVE_BITS"]
