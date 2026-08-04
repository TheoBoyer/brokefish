"""MCTS v0 on CUDA, behind the reference's interface.

A subclass of :class:`brokefish.search.torch_impl.Search` that replaces the four
per-simulation steps with the kernels of ``csrc/search.cuh`` and inherits
everything that runs once per move. What is inherited is deliberate, and it is
the whole list:

* ``reset`` and ``reset_finished``, which touch game state and no tree,
* ``dirichlet`` and ``_add_exploration_noise``, §6.1's root noise. Marsaglia-Tsang
  with the ``alpha < 1`` boost on device needs a normal variate and therefore a
  Box-Muller of its own; drawing ``B x E`` numbers in torch instead is one launch
  out of 3202 per move and gives the reference's exact stream, which is what makes
  §12's tree-for-tree comparison possible without an ``eps = 0`` escape hatch.
  Device generation is a named seam and nothing depends on where the numbers
  come from.
* ``select_and_advance``, §6.7, once per move: fifteen torch launches and a
  ``multinomial``, no host synchronisation unless ``check_invariants`` is on.

Everything else is a kernel. ``root_init``, ``descent``, ``expand`` and ``backup``
each get one launch per simulation and share the tree tensors with the reference,
so §12's harness can run five steps of one implementation and one of the other.

⚠️ **The evaluator has to return fp16 logits.** ``expand`` reads
``policy_logits`` and ``promo`` as fp16, which is what both fused encoders emit;
``brokefish/nn/model.py`` keeps fp32 master weights and returns fp32, and casting
it here would silently change the numbers the reference computes in fp32. The
error names the fix rather than converting.

Usage, identical to the reference::

    from brokefish.search import search_impl
    Search = search_impl("cuda")
    s = Search(SearchConfig(n=64, B=8), evaluate=make_evaluator(net, "cuda"))
    s.reset(*env.initial_boards(8, device="cuda"))
    record = s.self_play_move()
"""

from __future__ import annotations

import functools
import struct
from typing import Callable, Dict, Optional

import torch

from brokefish.env import cuda_impl as _cuda_env
from brokefish.env import luts as _luts
from brokefish.nn._build import load_extension

from . import torch_impl as _ref
from .torch_impl import MoveRecord, SearchConfig, SearchStats, check_invariants  # noqa: F401

# The tree fields csrc/search.cu looks up by name. Kept here as one list so that
# a field added to §4.2 fails loudly on both sides instead of being dropped by
# whichever side forgot it.
TREE_FIELDS = (
    "node_board", "node_control", "node_hash", "node_value", "node_nedges",
    "node_flags", "node_parent", "node_pedge",
    # `edge_win` is §6.6a's collapse mask and is the one field that may be **empty**:
    # `torch_impl` allocates it only when `terminal_collapse` is on, and `make_tree`
    # reads an empty tensor as a null pointer, which is how the kernels see "off".
    # Same convention as the counter block, and it means the flag has exactly one
    # representation instead of a bool that can disagree with an allocation.
    "edge_move", "edge_prior", "edge_child", "edge_N", "edge_Q", "edge_win",
    "path_node", "path_edge", "path_len",
    "game_ring", "game_ring_len", "node_count", "budget",
    "leaf_board", "leaf_control", "leaf_rep", "leaf_node", "leaf_flags",
)

# csrc/search.cuh's leaf_flags.
NEEDS_EXPAND = 1 << 0
FRESH = 1 << 1


@functools.lru_cache(maxsize=1)
def _ext():
    return load_extension("brokefish_search", ["search.cu"])


# §15's two counter blocks and the one mapping between them.
#
# ⚠️ **`SearchStats` and this device block are two implementations of the same
# section**, and `tests/test_search_cuda.py::test_the_counters_match_the_reference_stats`
# proves they agree run for run. What was *not* handled until 2026-08-02 is the
# merge: `train/loop.py` did `snapshot() | device_counters()`, and because the two
# name the same quantity differently, the log carried the live device value *and*
# the torch field beside it -- which on this path is never written and reads a
# hard zero. `n_terminal_children = 0` next to `terminal_descent_frac = 0.0104`
# cost two wrong conclusions in one session ("terminals are not detected", "nothing
# is truncated"), both of them false negatives in the middle of a bug hunt.
#
# The mapping lives here rather than in the test that needed it, so there is one copy.
COUNTER_ALIASES: Dict[str, str] = {
    "simulations": "simulations",
    "max_edges": "max_edges",
    "truncated_nodes": "n_truncated",
    "truncated_mass": "truncated_prior_mass",
    "max_depth": "max_depth",
    "max_nodes": "max_nodes_used",
    "terminal_descents": "n_terminal_descents",
    "terminal_children": "n_terminal_children",
    "empty_mask_expansions": "n_empty_mask_expansions",
    "depth_p50": "depth_p50",
    "depth_p99": "depth_p99",
}


class Search(_ref.Search):
    """One batch of `B` games, each with its own tree, searched on device.

    ``collect_stats`` controls §15's counter block. With it off nothing in a
    simulation writes a counter, which is what the throughput benchmark measures;
    with it on the per-simulation counters are device atomics costing no host
    synchronisation, and the per-move ones stay in the inherited
    :class:`SearchStats`, which synchronises once per move.
    """

    def __init__(self, config: SearchConfig, evaluate: Callable, env=None,
                 device: str | torch.device = "cuda", seed: int = 0,
                 check_invariants: bool = True, collect_stats: bool = False) -> None:
        cap = int(_ext().edge_cap())
        if config.E != cap:
            raise ValueError(
                f"the kernel is compiled for E = {cap} and got E = {config.E}. E is a "
                f"compile-time constant because §6.6's scan gives each lane kE/32 = "
                f"{cap // 32} edges with no predicate on the tail; change kE in "
                f"csrc/search.cuh to move it")
        super().__init__(config, evaluate, env=env if env is not None else _cuda_env,
                         device=device, seed=seed, check_invariants=check_invariants)
        if self.device.type != "cuda":
            raise ValueError(f"cuda_impl needs a CUDA device, got {self.device}")

        B = config.B
        dev = self.device
        z = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731

        # The staging the descent writes and the encoder reads. The encoder is one
        # CTA per contiguous board and a simulation's leaf sits at a different node
        # index in every game, so the leaf is written out flat rather than gathered.
        self.leaf_board = z((B, 32), torch.int16)
        self.leaf_control = z((B,), torch.int16)
        self.leaf_rep = z((B,), torch.uint8)
        self.leaf_node = z((B,), torch.int16)
        self.leaf_flags = z((B,), torch.uint8)
        self.root_rep = z((B,), torch.uint8)

        self.collect_stats = collect_stats
        self._counter_bytes = int(_ext().counter_size())
        self._counters = z((self._counter_bytes,), torch.uint8)
        self._empty = z((0,), torch.uint8)
        self.depth_hist = z((config.d_max + 1,), torch.int64)

        self._tables = _cuda_env._tables(dev)
        self._tree: Dict[str, torch.Tensor] = {}
        self._refresh_tree()

    # -- the tree the kernels are handed ------------------------------------ #

    def _refresh_tree(self) -> None:
        """Rebind the name-to-tensor dict the kernels read.

        Not paranoia. ``select_and_advance`` calls ``env.push_history``, which is
        a pure function and returns a *new* ring rather than writing the old one,
        so ``self.game_ring`` is a different tensor after every move. A dict built
        once in the constructor then points at a ring that stopped being updated
        after move 0, and the symptom is a repetition count that is too low deep
        into a game, three moves and eight thousand simulations away from the
        cause. Rebuilding once per move costs nothing and removes the class.
        """
        self._tree = {name: getattr(self, name) for name in TREE_FIELDS}
        missing = [n for n, t in self._tree.items() if t is None]
        if missing:
            raise AssertionError(f"tree fields not allocated: {missing}")

    def _check_tree_is_current(self) -> None:
        """Every tree tensor is still the one the kernels were given."""
        moved = [n for n in TREE_FIELDS
                 if self._tree[n].data_ptr() != getattr(self, n).data_ptr()]
        if moved:
            raise AssertionError(
                f"these tree tensors were rebound since the last refresh: {moved}. "
                "The kernels hold the old pointers and are writing a tree nobody reads; "
                "call _refresh_tree() after whatever replaced them")

    # -- §15 ---------------------------------------------------------------- #

    def reset_counters(self) -> None:
        self._counters.zero_()
        self.depth_hist.zero_()
        self.stats.reset()

    def device_counters(self) -> Dict[str, float]:
        """The §15 block, decoded. One host synchronisation, by design once per phase."""
        names = list(_ext().counter_names())
        raw = self._counters.cpu().numpy().tobytes()
        values = struct.unpack(_ext().counter_format(), raw)
        out = dict(zip(names, values))
        sims = max(out["simulations"], 1)
        out["mean_depth"] = out["depth_sum"] / sims
        out["terminal_descent_frac"] = out["terminal_descents"] / sims
        hist = self.depth_hist
        total = int(hist.sum())
        if total:
            cum = hist.cumsum(0)
            out["depth_p50"] = int((cum >= 0.5 * total).to(torch.uint8).argmax())
            out["depth_p99"] = int((cum >= 0.99 * total).to(torch.uint8).argmax())
        return out

    def stats_snapshot(self) -> Dict[str, float]:
        """§15's block with **one** name per quantity, device values where they exist.

        ⚠️ Use this rather than `stats.snapshot() | device_counters()`. The torch
        `SearchStats` fields fed by `on_expand` and `on_simulation` are never written
        on this path -- those two callbacks live in the reference's `_expand` and
        `simulate`, and this class replaces both with kernels -- so they report zero
        rather than reporting nothing. Every one of them is aliased below and
        overwritten with the device counter that actually measured it; the fields
        `on_move` fills are shared Python and stay as they are.

        `device_counters()` keeps its raw device names: eight test sites and
        `bench/bench_search.py` read them, and they are the kernel's own vocabulary.
        """
        out = dict(self.stats.snapshot())
        raw = self.device_counters()
        for device_name, canonical in COUNTER_ALIASES.items():
            if device_name in raw:
                out[canonical] = raw[device_name]
        # Device-only quantities have no torch counterpart to alias onto.
        for extra in ("pool_overflow", "depth_overflow", "mean_depth", "depth_sum"):
            if extra in raw:
                out[extra] = raw[extra]
        out["terminal_descent_frac"] = raw.get("terminal_descent_frac",
                                               out.get("terminal_descent_frac", 0.0))
        return out

    @property
    def _ctr(self) -> torch.Tensor:
        return self._counters if self.collect_stats else self._empty

    # -- the encoder -------------------------------------------------------- #

    def _evaluate_staged(self):
        policy, promo, value = self.evaluate(self.leaf_board, self.leaf_control, self.leaf_rep)
        if policy.dtype is not torch.float16 or promo.dtype is not torch.float16:
            raise TypeError(
                f"expand reads fp16 logits and got policy {policy.dtype}, promo {promo.dtype}. "
                "Use a fused encoder (make_evaluator(net, 'cuda' | 'triton')); casting an "
                "fp32 policy here would change the numbers the reference computes in fp32")
        return policy.contiguous(), promo.contiguous(), value.float().contiguous()

    # -- §6.1 --------------------------------------------------------------- #

    def root_init(self) -> None:
        if self.check_invariants and bool(self.game_done.any()):
            raise AssertionError("invariant 8: a finished game was searched; "
                                 "call reset_finished() first")
        self._refresh_tree()
        _ext().root_init(self._tree, self.game_board, self.game_control, self.game_hash,
                         self.root_rep)
        policy, promo, value = self._evaluate_staged()
        # §6.1a's rules scan, before expansion. The codes it produces are what
        # `_seed_terminal_edges` reads afterwards, and what the `must_keep` channel
        # into `expand` uses so §4.3's truncation cannot drop a proven terminal.
        self._root_code = self._root_result = None
        if self.config.root_terminal_sweep:
            mask, _ = self.env.movegen(self.game_board, self.game_control)
            self._root_code, self._root_result = self._root_terminal_scan(mask)
        _ext().expand(self._tree, policy, promo, value, *self._tables, self._ctr)
        self._add_exploration_noise()
        # §6.1a. Inherited from the reference unchanged: it writes `edge_N` and
        # `edge_Q` in place, and those are the very tensors `self._tree` hands the
        # kernels, so the descent sees the seeded values without a kernel change.
        # That is also what keeps the two implementations identical here by
        # construction rather than by a second transcription.
        self._seed_terminal_edges()

    # -- §6.2 to §6.5 ------------------------------------------------------- #

    def simulate(self, s: int) -> None:
        """One simulation for every game whose budget is not exhausted.

        Four launches: the descent (which is also the step, the repetition scan,
        the terminal test and the allocation), the encoder, the expansion and the
        backup. §9.1 prices the fusion of the last two plus the next descent at
        800 launches a move, about 4 ms against a 51 s move, and v0 keeps them
        separate so each is checkable against the reference on its own.
        """
        c = self.config
        if self.check_invariants:
            self._check_tree_is_current()
        _ext().descent(self._tree, s, c.pb_c_base, c.pb_c_init, *self._tables, self._ctr)
        policy, promo, value = self._evaluate_staged()
        _ext().expand(self._tree, policy, promo, value, *self._tables, self._ctr)
        _ext().backup(self._tree)
        if self.collect_stats:
            # The depth histogram is §15.1's, and it is the one counter left in
            # torch: a per-simulation atomicAdd into one or two buckets from 4096
            # warps is contention the device block does not need, and bincount is
            # a launch with no synchronisation.
            self.depth_hist += torch.bincount(self.path_len.long(),
                                              minlength=c.d_max + 1)

    # -- the pieces §12's harness swaps individually ------------------------ #

    def descent_only(self, s: int) -> None:
        c = self.config
        _ext().descent(self._tree, s, c.pb_c_base, c.pb_c_init, *self._tables, self._ctr)

    def expand_only(self, policy, promo, value) -> None:
        _ext().expand(self._tree, policy.contiguous(), promo.contiguous(),
                      value.float().contiguous(), *self._tables, self._ctr)

    def backup_only(self) -> None:
        _ext().backup(self._tree)
