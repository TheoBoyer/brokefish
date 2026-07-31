"""MCTS v0 in PyTorch: the reference implementation of ``docs/mcts.md``.

The oracle the CUDA search will be written against, and the search C2 develops
its training loop on before the kernels exist. It is expected to be orders of
magnitude slower than the kernel; ``docs/mcts.md`` §12 puts it first anyway,
because A1 only landed once perft gave the move generator something external to
be wrong against.

Two things about the shape of the code, both deliberate.

**Batched over games, looped over depth and simulations.** §4.2's arrays are
tensors with a leading ``B`` axis and every step below is one torch op over the
whole batch. The Python loops are the two that the kernel also runs serially:
descent depth, and the ``n`` simulations. A per-game Python loop would be a
truer transcription of the pseudocode and would take a day per move at
``B = 4096``.

**Every departure from the kernel's structure is a departure in structure only,
never in values.** There are three, each marked where it happens and all three
listed in §12: the legality mask is carried from the terminal test into expansion
in a local variable rather than recomputed, since §6.2's warning describes a
kernel boundary this module does not have; ``edge_move`` and ``edge_N`` are
``int16`` and ``path_len`` ``int32``, because torch has no unsigned arithmetic
worth using; and the repetition count is computed here rather than passed through
the engine's ring, per §6.2 step 3. The tree that comes out is the tree §4.2
describes, node for node.

Both the environment and the network are injected, so the same code runs over
``env/torch_impl.py`` or ``env/cuda_impl.py`` and against ``nn/model.py`` or a
fused encoder. Running all four is a differential check on A1 and B2 as much as
on this.

Usage, one self-play move over a batch of games::

    search = Search(SearchConfig(n=64, B=8, E=64), evaluate=make_evaluator(net))
    search.reset(*env.initial_boards(8, device="cuda"))
    record = search.self_play_move()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from brokefish.env import torch_impl as _default_env
from brokefish.env.torch_impl import INSUFFICIENT, PAWN, REPETITION

# `node_flags` (§4.2). Bits 0-2 are the terminal code of spec §4.3, bit 3 says the
# node has been expanded, and bit 4 says the move that created it was
# irreversible, which is what bounds the in-tree repetition scan of §7.
TERMINAL = 0b111
EXPANDED = 1 << 3
IRREVERSIBLE = 1 << 4

# spec §3: a move is 11 bits and the promotion choice rides above it, so one
# `int16` carries a whole edge label.
MOVE_BITS = 11
PROMO_SHIFT = MOVE_BITS

# §6.6 first-play urgency: the value an unvisited edge is scored at.
#
# ⚠️ **0.5, not 0.** AGZ scores an untried move at `Q = 0`, and AGZ's values are in
# `[-1, 1]`, so that 0 is a *draw* -- a neutral prior on a move nobody has tried. The
# tree here works in `[0, 1]` (§3.5, deliberately, so `pb_c_init = 1.25` keeps the
# meaning it has in the published pseudocode), and under `q01 = (v + 1) / 2` a draw is
# **0.5**. Every constant living in Q-space has to ride the remap; §3.5 says exactly
# that and v0 applied it to the value and not to this.
#
# What the literal 0 cost, measured 2026-07-31 on run `c2-8h`: an untried move is
# scored as a certain loss, so against a value head near the draw value it needs
# `pb_c * prior > 0.5` to ever be tried -- `prior > 0.0145` at n = 800. Anything below
# that is unreachable *at any simulation budget*, so its prior can never be corrected
# upward and can only fall further. That is an absorbing state, and it turned a
# stagnating run into a monotonically collapsing one: 3 of 20 root moves visited at
# n = 64 and at n = 800 alike, decisive games 5.13 % -> 1.76 % over seven hours.
# With 0.5 the same network at n = 800 visits 20 of 20.
FPU_DRAW = 0.5

# spec §4.3's codes in index order, so `SearchStats` can report the histogram by
# name. Built from `env`'s `TERMINAL_NAMES` rather than restated, so there is exactly
# one place a code is given a name.
_TERMINAL_NAMES_ORDERED = [_default_env.TERMINAL_NAMES[i]
                           for i in range(len(_default_env.TERMINAL_NAMES))]


@dataclass
class SearchConfig:
    """§4.1. Defaults are the v0 values."""

    n: int = 800
    B: int = 4096
    E: int = 64
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    alpha: float = 0.3
    eps: float = 0.25
    tau_plies: int = 30

    @property
    def n_max(self) -> int:
        """Nodes per game. The root is made once and each simulation adds at most one."""
        return self.n + 1

    @property
    def d_max(self) -> int:
        """§4.1: the exact bound, not a cap. A path over `n + 1` nodes has `n` edges."""
        return self.n


@dataclass
class MoveRecord:
    """§10, minus the value, which C2 fills in when the game ends.

    One row per game. ``done`` and ``result`` are what let the caller close out
    the pending records of a game that just finished.

    ``root_value`` is §10's reserved field, written and trained on by nothing
    (`train.md` §4). It is the **search's** value of the root — the visit-weighted
    mean of the root edges' ``Q``, which is the improved estimate a KataGo-style
    bootstrapped target would mix into ``z``, not the raw network evaluation the
    root started from. Stored in ``[-1, 1]`` so it is directly comparable with the
    game outcome, while the tree itself works in ``[0, 1]`` (§3.5).
    """

    board: torch.Tensor        # [B, 32] int16
    control: torch.Tensor      # [B]     int16
    rep: torch.Tensor          # [B]     uint8, min(rep - 1, 2)
    policy_move: torch.Tensor  # [B, K]  int16, spec §3 label with the promo field
    policy_prob: torch.Tensor  # [B, K]  float16
    policy_len: torch.Tensor   # [B]     uint8
    played: torch.Tensor       # [B]     int16, the edge label actually played
    ply: torch.Tensor          # [B]     int32, the ply this record sits at
    root_value: torch.Tensor   # [B]     float32, §10's reserved field, see below
    weight_gen: int
    done: torch.Tensor         # [B]     bool
    result: torch.Tensor       # [B]     int8, from the new mover's point of view


def make_evaluator(net, impl: Optional[str] = None) -> Callable:
    """``(boards, control, rep) -> (policy_logits, promo, value)``.

    ``impl=None`` uses the module in ``brokefish/nn/model.py``, which is the
    oracle and runs on CPU. A name selects a fused implementation, whose
    ``forward_full`` is the same function three times faster and CUDA only.
    """
    if impl is None:
        return lambda b, c, r: net(b, c, r)
    from brokefish.nn import encoder_impl

    return encoder_impl(impl)(net).forward_full


def _lowest_argmax(x: torch.Tensor) -> torch.Tensor:
    """``argmax`` along the last axis with ties broken by the lowest index.

    ``torch.argmax`` does not promise which of several maxima it returns, and
    §6.6 and §6.7 both fix the tie-break, because the differential test of §12
    compares trees rather than distributions.
    """
    e = x.shape[-1]
    is_max = x == x.max(dim=-1, keepdim=True).values
    idx = torch.arange(e, device=x.device).expand_as(x)
    return torch.where(is_max, idx, torch.full_like(idx, e)).min(dim=-1).values


class Search:
    """One batch of `B` games, each with its own tree, sharing one network call.

    The instance owns both the trees (§4.2) and the games they are searching, so
    a caller drives it with :meth:`reset` and then :meth:`self_play_move`.
    """

    def __init__(self, config: SearchConfig, evaluate: Callable, env=None,
                 device: str | torch.device = "cuda", seed: int = 0,
                 check_invariants: bool = True) -> None:
        c = config
        if c.n_max > 32767:
            raise ValueError(f"n = {c.n} needs node indices wider than int16")
        if c.n > 32767:
            raise ValueError(f"n = {c.n} overflows the int16 visit counts of §4.2")
        if c.E > 255:
            raise ValueError(f"E = {c.E} overflows the uint8 edge indices of §4.2")

        self.config = c
        self.evaluate = evaluate
        self.env = env if env is not None else _default_env
        self.device = torch.device(device)
        self.check_invariants = check_invariants
        self.weight_gen = 0
        self._gen = torch.Generator(device=self.device).manual_seed(seed)

        B, N, E, D = c.B, c.n_max, c.E, c.d_max
        dev = self.device
        z = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731

        # §4.2. The integer widths are the document's, except where noted in the
        # module docstring; every one of these is indexed with `.long()` at use.
        self.node_board = z((B, N, 32), torch.int16)
        self.node_control = z((B, N), torch.int16)
        self.node_hash = z((B, N), torch.int64)
        self.node_value = z((B, N), torch.float16)
        self.node_nedges = z((B, N), torch.uint8)
        self.node_flags = z((B, N), torch.uint8)
        self.node_parent = z((B, N), torch.int16)
        self.node_pedge = z((B, N), torch.uint8)

        self.edge_move = z((B, N, E), torch.int16)
        self.edge_prior = z((B, N, E), torch.float16)
        self.edge_child = torch.full((B, N, E), -1, dtype=torch.int16, device=dev)
        self.edge_N = z((B, N, E), torch.int16)
        self.edge_Q = z((B, N, E), torch.float32)

        self.path_node = z((B, D), torch.int16)
        self.path_edge = z((B, D), torch.uint8)
        # §4.2 gives `path_len` as u8, which cannot hold a depth up to Dmax = 800.
        self.path_len = z((B,), torch.int32)

        self.game_ring, self.game_ring_len = self.env.empty_history(B, device=dev)
        self.game_ply = z((B,), torch.int32)
        self.node_count = z((B,), torch.int32)

        # The real position each game is at, which the tree searches from.
        self.game_board = z((B, 32), torch.int16)
        self.game_control = torch.ones((B,), dtype=torch.int16, device=dev)
        self.game_hash = z((B,), torch.int64)
        self.game_done = z((B,), torch.bool)
        self.game_result = z((B,), torch.int8)

        # §11's structural hook: the simulation loop reads a per-game budget
        # rather than the constant `n`, which is what playout cap randomisation
        # needs and what is awkward to retrofit. In v0 every entry is `n`.
        self.budget = torch.full((B,), c.n, dtype=torch.int32, device=dev)

        self._b = torch.arange(B, device=dev)
        self._e = torch.arange(E, device=dev)
        self._d = torch.arange(D, device=dev)
        self._sq = torch.arange(64, device=dev)
        self.stats = SearchStats(D, dev)

    # -- games -------------------------------------------------------------- #

    def reset(self, boards: Optional[torch.Tensor] = None,
              control: Optional[torch.Tensor] = None,
              rows: Optional[torch.Tensor] = None) -> None:
        """Start (or restart) games. ``rows`` restarts a subset, ``None`` all of them.

        A finished game is replaced at the start of the next move rather than
        compacted out of the batch, which is what keeps `B` constant (§6.7).
        """
        B = self.config.B
        if rows is None:
            rows = self._b
        if boards is None:
            boards, control = self.env.initial_boards(int(rows.numel()), device=self.device)
        self.game_board[rows] = boards.to(torch.int16)
        self.game_control[rows] = control.to(torch.int16)
        self.game_hash[rows] = self.env.hash_position(boards, control)
        self.game_ring[rows] = 0
        self.game_ring_len[rows] = 0
        self.game_ply[rows] = 0
        self.game_done[rows] = False
        self.game_result[rows] = 0
        assert self.game_board.shape[0] == B

    def reset_finished(self) -> torch.Tensor:
        """Restart every finished game from the initial position. Returns the rows."""
        rows = self.game_done.nonzero(as_tuple=True)[0]
        if rows.numel():
            self.reset(rows=rows)
        return rows

    # -- one move ----------------------------------------------------------- #

    def self_play_move(self) -> MoveRecord:
        """§6 end to end: `root_init`, `n` simulations, `select_and_advance`."""
        self.root_init()
        for s in range(self.config.n):
            self.simulate(s)
        return self.select_and_advance()

    def simulate(self, s: int) -> None:
        """One simulation for every game whose budget is not exhausted."""
        active = self.budget > s
        leaf, parent, edge = self._descent(active)
        leaf, rep, code, mask, expand = self._create_child(leaf, parent, edge, active)
        policy, promo, value = self.evaluate(
            self.node_board[self._b, leaf], self.node_control[self._b, leaf], rep)
        self._expand(leaf, mask, expand, policy, promo, value)
        self._backup(leaf, active)
        self.stats.on_simulation(self, active, parent >= 0, code)

    # -- §6.1 --------------------------------------------------------------- #

    def root_init(self) -> None:
        b = self._b
        if self.check_invariants and bool(self.game_done.any()):
            raise AssertionError("invariant 8: a finished game was searched; "
                                 "call reset_finished() first")

        self.node_count.fill_(1)
        self.node_board[:, 0] = self.game_board
        self.node_control[:, 0] = self.game_control
        self.node_hash[:, 0] = self.game_hash
        self.node_parent[:, 0] = -1
        self.node_pedge[:, 0] = 0
        self.node_flags[:, 0] = 0
        self.node_nedges[:, 0] = 0
        self._clear_edges(b, torch.zeros_like(b))

        # The root's repetition count is over the game ring alone: no tree exists
        # yet, and the ring holds every position before this one (§7).
        rep = self.env.repetition_count(self.game_hash, self.game_ring, self.game_ring_len)
        self.root_rep = (rep - 1).clamp(0, 2).to(torch.uint8)
        mask, _ = self.env.movegen(self.game_board, self.game_control)
        policy, promo, value = self.evaluate(self.game_board, self.game_control, self.root_rep)

        root = torch.zeros_like(b)
        self._expand(root, mask, torch.ones_like(self.game_done), policy, promo, value)
        self._add_exploration_noise()

    def _add_exploration_noise(self) -> None:
        """§6.1's Dirichlet mixture over the root's own edges."""
        c = self.config
        if c.eps <= 0.0:
            return
        ne = self.node_nedges[:, 0].long()
        valid = self._e[None, :] < ne[:, None]
        eta = self.dirichlet(valid)
        p = self.edge_prior[:, 0].float()
        self.edge_prior[:, 0] = torch.where(
            valid, (1.0 - c.eps) * p + c.eps * eta, torch.zeros_like(p)).to(torch.float16)

    def dirichlet(self, valid: torch.Tensor) -> torch.Tensor:
        """``Dir(alpha)`` over each row's valid entries, zero elsewhere. A §11 seam.

        The construction is §6.1's, because cuRAND has no device-side gamma
        generator and the kernel will have to build one the same way:
        ``eta_i = g_i / sum_j g_j`` with ``g ~ Gamma(alpha, 1)``, and
        ``Gamma(alpha < 1, 1)`` as ``Gamma(alpha + 1, 1) * U^(1/alpha)``.

        ⚠️ The *construction* matches and the *stream* does not. Torch draws one
        global sequence where a kernel gives each lane its own cuRAND state, so
        the two produce different noise from the same seed. §12's tree-for-tree
        comparison therefore has to drive both from the same numbers, which is
        what overriding this method is for.
        """
        g = self._gamma(valid.shape, self.config.alpha)
        g = torch.where(valid, g, torch.zeros_like(g))
        total = g.sum(-1, keepdim=True).clamp(min=torch.finfo(g.dtype).tiny)
        return g / total

    def _gamma(self, shape, alpha: float) -> torch.Tensor:
        """``Gamma(alpha, 1)``, Marsaglia-Tsang with the ``alpha < 1`` boost.

        Rejection sampling, vectorised by redrawing the whole tensor and keeping
        only the entries that have not been accepted yet. The acceptance rate is
        above 98 % for any alpha, so the loop runs two or three times.
        """
        boost = alpha < 1.0
        a = alpha + 1.0 if boost else alpha
        d = a - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        opts = dict(device=self.device, dtype=torch.float32)

        out = torch.zeros(shape, **opts)
        todo = torch.ones(shape, dtype=torch.bool, device=self.device)
        while bool(todo.any()):
            x = torch.randn(shape, generator=self._gen, **opts)
            v = (1.0 + c * x) ** 3
            u = torch.rand(shape, generator=self._gen, **opts)
            safe = torch.where(v > 0, v, torch.ones_like(v))
            ok = (v > 0) & (torch.log(u) < 0.5 * x * x + d - d * safe + d * torch.log(safe))
            out = torch.where(todo & ok, d * v, out)
            todo = todo & ~ok
        if boost:
            u = torch.rand(shape, generator=self._gen, **opts)
            out = out * u.clamp(min=torch.finfo(torch.float32).tiny) ** (1.0 / alpha)
        return out

    # -- §6.6 --------------------------------------------------------------- #

    def _select(self, v: torch.Tensor) -> torch.Tensor:
        """The PUCT argmax over node `v`'s edges, one row per game."""
        c = self.config
        b = self._b
        nvis = self.edge_N[b, v].to(torch.float32)
        prior = self.edge_prior[b, v].float()
        q = self.edge_Q[b, v]
        valid = self._e[None, :] < self.node_nedges[b, v].long()[:, None]

        n_v = torch.where(valid, nvis, torch.zeros_like(nvis)).sum(-1, keepdim=True)
        pb_c = torch.log((n_v + c.pb_c_base + 1.0) / c.pb_c_base) + c.pb_c_init
        pb_c = pb_c * n_v.sqrt() / (nvis + 1.0)
        # §6.6: an unvisited edge takes AGZ's first-play urgency, a *draw* from the
        # mover's point of view -- which is `FPU_DRAW`, not 0, because the tree works
        # in [0,1] (§3.5). See that constant for what carrying the literal 0 across
        # the remap did.
        score = pb_c * prior + torch.where(nvis > 0, q, torch.full_like(q, FPU_DRAW))
        score = torch.where(valid, score, torch.full_like(score, float("-inf")))
        return _lowest_argmax(score)

    # -- §6.2 --------------------------------------------------------------- #

    def _descent(self, active: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Walk from the root to a leaf, recording the path as it goes.

        Returns ``(leaf, parent, edge)``. ``parent >= 0`` marks a game that
        stopped on an unexpanded edge and needs a node allocated; the others
        stopped on a terminal node, which is already stored, and their ``leaf``
        is that node.
        """
        b = self._b
        self.path_len.zero_()
        v = torch.zeros_like(b)
        leaf = torch.zeros_like(b)
        parent = torch.full_like(b, -1)
        edge = torch.zeros_like(b)
        live = active.clone()
        d = 0

        while bool(live.any()):
            terminal = (self.node_flags[b, v] & TERMINAL) != 0
            if self.check_invariants and d == 0 and bool((terminal & live).any()):
                raise AssertionError("invariant 8: the root is terminal")
            leaf = torch.where(live & terminal, v, leaf)
            live = live & ~terminal
            if not bool(live.any()):
                break

            e = self._select(v)
            rows = live.nonzero(as_tuple=True)[0]
            self.path_node[rows, d] = v[rows].to(torch.int16)
            self.path_edge[rows, d] = e[rows].to(torch.uint8)
            self.path_len[rows] = d + 1

            child = self.edge_child[b, v, e].long()
            fresh = live & (child == -1)
            parent = torch.where(fresh, v, parent)
            edge = torch.where(fresh, e, edge)
            live = live & ~fresh
            v = torch.where(live, child, v)
            d += 1
            if d >= self.config.d_max and bool(live.any()):
                raise AssertionError(f"descent reached Dmax = {self.config.d_max}")

        return leaf, parent, edge

    def _create_child(self, leaf: torch.Tensor, parent: torch.Tensor, edge: torch.Tensor,
                      active: torch.Tensor):
        """Apply the chosen move, test it for termination, and allocate a node.

        Returns ``(leaf, rep, code, mask, expand)`` where ``rep`` is spec §7.2's
        clamped repetition feature for the leaf, ``mask`` its legality mask and
        ``expand`` the rows §6.4 has work for.
        """
        b = self._b
        fresh = parent >= 0
        src = torch.where(fresh, parent, leaf)
        label = self.edge_move[b, src, edge].to(torch.int64)
        # spec §9's null move for every game that is not creating a node: its row
        # is computed like any other and the result discarded, which is cheaper
        # than a compaction and is what keeps every launch shape static (§6.3).
        move = torch.where(fresh, label & ((1 << MOVE_BITS) - 1), torch.full_like(label, -1))
        promo = (label >> PROMO_SHIFT) & 0b11

        board, control, hash_, irrev = self.env.step(
            self.node_board[b, src], self.node_control[b, src], move,
            promo=promo, hash=self.node_hash[b, src])
        mask, in_check = self.env.movegen(board, control)

        rep = self._repetition_count(hash_, irrev)
        code, result = self.env.terminal(mask, in_check, control, board)
        # `terminal` is called without a ring because the count above is over the
        # ring *and* the path, which no engine-side signature can express: spec
        # §6.3's ring is [N, 100] and the tree half lives in the search. Code 4
        # is folded in at spec §4.3's priority, above insufficient material and
        # below everything else. Both are draws, so `result` does not move.
        third = rep >= 3
        code = torch.where(third & ((code == 0) | (code == INSUFFICIENT)),
                           torch.full_like(code, REPETITION), code)

        rows = fresh.nonzero(as_tuple=True)[0]
        c = self.node_count[b].long()
        if self.check_invariants and rows.numel() and bool((c[rows] >= self.config.n_max).any()):
            raise AssertionError("invariant 1: the node pool overflowed")
        c = torch.where(fresh, c, leaf)

        self.node_board[rows, c[rows]] = board[rows]
        self.node_control[rows, c[rows]] = control[rows]
        self.node_hash[rows, c[rows]] = hash_[rows]
        self.node_parent[rows, c[rows]] = parent[rows].to(torch.int16)
        self.node_pedge[rows, c[rows]] = edge[rows].to(torch.uint8)
        self.node_nedges[rows, c[rows]] = 0
        self.node_flags[rows, c[rows]] = (
            code[rows] | (irrev[rows].to(torch.uint8) * IRREVERSIBLE))
        self._clear_edges(rows, c[rows])
        # A terminal node carries its result in place of an evaluation, in the
        # [0,1] convention of §3.5: -1 becomes 0 and 0 becomes 0.5 (§8).
        terminal_rows = rows[code[rows] != 0]
        self.node_value[terminal_rows, c[terminal_rows]] = (
            (result[terminal_rows].float() + 1.0) / 2.0).to(torch.float16)
        self.edge_child[rows, parent[rows], edge[rows]] = c[rows].to(torch.int16)
        self.node_count[rows] += 1

        expand = fresh & (code == 0)
        return c, (rep - 1).clamp(0, 2).to(torch.uint8), code, mask, expand

    def _repetition_count(self, child_hash: torch.Tensor, child_irrev: torch.Tensor
                          ) -> torch.Tensor:
        """§7: how many times this position has occurred, counting itself.

        "The game so far" for a node inside the tree is the game ring plus every
        position from the root down to the node's parent, and the two halves are
        counted separately because no engine-side signature joins them: spec
        §6.3's ring is `[N, 100]` and the tree half is the search's own. The
        kernel splits it the same way, a warp-collective scan over the ring and a
        ballot over the path.
        """
        length = self.path_len.long()
        path = self.path_node.long().clamp(min=0)
        flags = self.node_flags.gather(1, path)
        hashes = self.node_hash.gather(1, path)

        on_path = self._d[None, :] < length[:, None]
        # Node 0's own irreversibility is the real game's business, since the ring
        # was already emptied by it (spec §6.2), so only levels 1 and below cut.
        irrev = ((flags & IRREVERSIBLE) != 0) & on_path & (self._d[None, :] >= 1)
        deepest = torch.where(irrev, self._d[None, :], torch.full_like(path, -1)).max(-1).values
        # An irreversible move into the child makes every earlier position
        # unreachable, which is the empty-window case.
        cut = torch.where(child_irrev, length, deepest.clamp(min=0))
        window = (self._d[None, :] >= cut[:, None]) & on_path
        on_tree = ((hashes == child_hash[:, None]) & window).sum(-1)

        keep_ring = (deepest < 0) & ~child_irrev
        ring_len = torch.where(keep_ring, self.game_ring_len, torch.zeros_like(self.game_ring_len))
        if self.check_invariants:
            # The window is bounded by the fifty-move window, whose reset
            # conditions are a subset of `irreversible`'s (spec §6.2), and the
            # clock keeps counting inside the tree, so ring plus path can never
            # outrun the ring's own 100 slots.
            span = ring_len + window.sum(-1)
            if bool((span > self.game_ring.shape[1]).any()):
                raise AssertionError("the repetition window outgrew the fifty-move window")
        return self.env.repetition_count(child_hash, self.game_ring, ring_len) + on_tree

    # -- §6.4 --------------------------------------------------------------- #

    def _clear_edges(self, rows: torch.Tensor, nodes: torch.Tensor) -> None:
        self.edge_move[rows, nodes] = 0
        self.edge_prior[rows, nodes] = 0
        self.edge_child[rows, nodes] = -1
        self.edge_N[rows, nodes] = 0
        self.edge_Q[rows, nodes] = 0

    def _promotion_targets(self, boards: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """``[B, 32, 64] bool``: the move ``p -> s`` promotes.

        The type is read from the piece word and never from the slot index: a
        promoted queen keeps its pawn slot, so a slot in 0-7 does not imply a
        pawn (spec §2.1).
        """
        words = boards.to(torch.int32)
        pawn = (((words >> 6) & 0b111) == PAWN) & (((words >> 11) & 1) == 0)
        last_rank = torch.where(control < 0, 0, 7).long()
        on_last = (self._sq[None, :] // 8) == last_rank[:, None]
        return pawn[:, :, None] & on_last[:, None, :]

    def _expand(self, node: torch.Tensor, mask: torch.Tensor, do: torch.Tensor,
                policy: torch.Tensor, promo: torch.Tensor, value: torch.Tensor) -> None:
        """Turn a legality mask into edges, in the canonical order of §6.4.

        The enumeration order is normative, since PUCT ties are broken by edge
        index and §12 compares trees rather than distributions: ascending by
        slot, then by target square, then by promotion type in spec §3's
        ``N B R Q`` order.
        """
        c = self.config
        b, E = self._b, self.config.E
        board = self.node_board[b, node]
        control = self.node_control[b, node]

        legal = self.env.bitset_to_bool(mask)                      # [B,32,64]
        is_promo = self._promotion_targets(board, control) & legal
        cand = legal[..., None].expand(-1, -1, -1, 4).clone()
        cand[..., 1:] &= is_promo[..., None]
        cand = cand & do[:, None, None, None]
        cand = cand.reshape(cand.shape[0], -1)                     # [B, 32*64*4]

        # §6.4: log space, so one softmax over the node's edges normalises both
        # halves of P(target | piece) * P(type | piece) at once.
        lp = torch.log_softmax(promo.float(), dim=-1)               # [B,32,4]
        logit = policy.float()[..., None].expand(-1, -1, -1, 4).clone()
        logit += torch.where(is_promo[..., None], lp[:, :, None, :], torch.zeros_like(logit))
        logit = logit.reshape(cand.shape)

        count = cand.sum(-1)
        dropped = torch.zeros_like(count, dtype=torch.float32)
        if E < int(count.max() if count.numel() else 0):
            # §4.3: keep the E highest priors, ties by canonical order, and the
            # kept edges stay in canonical order. Sorting ascending on the
            # negated logit makes the stable sort's tie-break the canonical one.
            key = torch.where(cand, -logit, torch.full_like(logit, float("inf")))
            order = key.argsort(dim=-1, stable=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(order.shape[1], device=self.device)
                          .expand_as(order))
            keep = cand & (rank < E)
            # §15.1 wants the mass discarded, not just the count, and the mass has
            # to be read off the distribution *before* truncation: the priors
            # written below are renormalised over the survivors, where the dropped
            # tail is zero by construction and says nothing about its size.
            before = torch.softmax(
                torch.where(cand, logit, torch.full_like(logit, float("-inf"))), dim=-1)
            dropped = torch.where(cand & ~keep, before, torch.zeros_like(before)).sum(-1)
        else:
            keep = cand

        n_edges = keep.sum(-1)
        if self.check_invariants and bool((do & (count == 0)).any()):
            raise AssertionError("invariant 5: expanding a node with no legal move")

        prior = torch.softmax(
            torch.where(keep, logit, torch.full_like(logit, float("-inf"))), dim=-1)
        slot = keep.cumsum(-1) - 1
        rows, cols = keep.nonzero(as_tuple=True)
        pos = slot[rows, cols]
        nodes = node[rows]
        field = cols // 4
        self.edge_move[rows, nodes, pos] = (
            field | ((cols % 4) << PROMO_SHIFT)).to(torch.int16)
        self.edge_prior[rows, nodes, pos] = prior[rows, cols].to(torch.float16)

        touched = do.nonzero(as_tuple=True)[0]
        self.node_nedges[touched, node[touched]] = n_edges[touched].to(torch.uint8)
        self.node_flags[touched, node[touched]] |= EXPANDED
        # §3.5: the value head is tanh in [-1,1] and the tree works in [0,1].
        self.node_value[touched, node[touched]] = (
            (value[touched].float() + 1.0) / 2.0).to(torch.float16)
        self.stats.on_expand(count, n_edges, dropped, do)

    # -- §6.5 --------------------------------------------------------------- #

    def _backup(self, leaf: torch.Tensor, active: torch.Tensor) -> None:
        """Attribute the leaf's value to every edge on the path, flipping by parity.

        ``edge_Q[v][e]`` is the value of move `e` seen by the player to move at
        `v`, because that is the player choosing among `v`'s edges in §6.6. A
        node at level `d` sits ``L - d`` plies before the leaf, so its mover is
        the leaf's mover exactly when ``L - d`` is even. The edge into the leaf
        always flips: whoever chose it is not the player to move there.
        """
        b = self._b
        length = self.path_len.long()
        q_leaf = self.node_value[b, leaf].float()
        for d in range(int(length.max()) if length.numel() else 0):
            rows = ((length > d) & active).nonzero(as_tuple=True)[0]
            if not rows.numel():
                continue
            v = self.path_node[rows, d].long()
            e = self.path_edge[rows, d].long()
            flip = ((length[rows] - d) % 2) == 1
            q = torch.where(flip, 1.0 - q_leaf[rows], q_leaf[rows])
            count = self.edge_N[rows, v, e].to(torch.float32) + 1.0
            self.edge_N[rows, v, e] = count.to(torch.int16)
            old = self.edge_Q[rows, v, e]
            self.edge_Q[rows, v, e] = old + (q - old) / count

    # -- §6.7 --------------------------------------------------------------- #

    def select_and_advance(self) -> MoveRecord:
        """Pick the move from the root visit counts, play it, and emit the record."""
        c = self.config
        b = self._b
        # `E`, not `min(E, n)`. §10 stores the root's **whole edge set**, not only the
        # edges a simulation happened to reach, so the width is bounded by the edge cap
        # rather than by the visit budget. Revised 2026-07-31; see below.
        K = c.E

        nvis = self.edge_N[:, 0].to(torch.float32)
        valid = self._e[None, :] < self.node_nedges[:, 0].long()[:, None]
        nvis = torch.where(valid, nvis, torch.zeros_like(nvis))
        total = nvis.sum(-1, keepdim=True)
        if self.check_invariants and not bool((total.squeeze(-1) == self.budget).all()):
            raise AssertionError("invariant 6: root visits do not sum to the budget")
        pi = nvis / total

        sampled = torch.multinomial(pi, 1, generator=self._gen).squeeze(-1)
        best = _lowest_argmax(pi)
        e = torch.where(self.game_ply < c.tau_plies, sampled, best)
        label = self.edge_move[b, 0, e].to(torch.int64)

        # ⚠️ **Every valid edge, not only the visited ones** (revised 2026-07-31,
        # `train.md` §3.5). An unvisited edge carries `pi = 0` and contributes nothing
        # to the training target, but its *label* is what tells the training path which
        # moves the softmax denominator runs over. Storing it costs nothing — the
        # arrays are already `E` wide and the tail was zero padding — and it is what
        # lets the loss stop recomputing `movegen` to rediscover a support the search
        # already knew. `policy_len` is therefore the root's edge count.
        keep = valid
        pos = keep.cumsum(-1) - 1
        rows, cols = keep.nonzero(as_tuple=True)
        policy_move = torch.zeros((c.B, K), dtype=torch.int16, device=self.device)
        policy_prob = torch.zeros((c.B, K), dtype=torch.float16, device=self.device)
        policy_move[rows, pos[rows, cols]] = self.edge_move[rows, 0, cols]
        policy_prob[rows, pos[rows, cols]] = pi[rows, cols].to(torch.float16)

        # §10's reserved field. `pi` is zero outside the root's own edges and
        # `_clear_edges` zeroed `edge_Q` there, so this is the visit-weighted mean
        # over exactly the valid edges, in [0,1], mapped to [-1,1] to match `z`.
        root_value = (2.0 * (pi * self.edge_Q[:, 0]).sum(-1) - 1.0).float()

        record = MoveRecord(
            board=self.game_board.clone(), control=self.game_control.clone(),
            rep=self.root_rep.clone(), policy_move=policy_move, policy_prob=policy_prob,
            policy_len=keep.sum(-1).to(torch.uint8), played=label.to(torch.int16),
            ply=self.game_ply.clone(), root_value=root_value,
            weight_gen=self.weight_gen,
            done=torch.zeros_like(self.game_done), result=torch.zeros_like(self.game_result))

        move = label & ((1 << MOVE_BITS) - 1)
        promo = (label >> PROMO_SHIFT) & 0b11
        board, control, hash_, irrev = self.env.step(
            self.game_board, self.game_control, move, promo=promo, hash=self.game_hash)
        # The ring records the position being left, and the move's own
        # irreversibility is what empties it (spec §6.2), so this comes before
        # the terminal test of the new position.
        self.game_ring, self.game_ring_len = self.env.push_history(
            self.game_ring, self.game_ring_len, self.game_hash, irrev)
        mask, in_check = self.env.movegen(board, control)
        code, result = self.env.terminal(mask, in_check, control, board, hash_,
                                         self.game_ring, self.game_ring_len)

        self.game_board, self.game_control, self.game_hash = board, control, hash_
        self.game_ply += 1
        self.game_done = code != 0
        self.game_result = result
        record.done = self.game_done.clone()
        record.result = result.clone()

        self.stats.on_move(self, pi, code, e)
        return record


class SearchStats:
    """§15's counter block, accumulated over a self-play phase.

    Every fixed size in §4 is a bet on a distribution that the policy moves, so
    the counters that watch those bets are part of the search rather than a
    debugging afterthought. Nothing here costs a host synchronisation in the
    kernel; here it is torch reductions, since the reference already syncs.
    """

    def __init__(self, d_max: int, device) -> None:
        self.device = device
        self.d_max = d_max
        self.reset()

    def reset(self) -> None:
        dev = self.device
        self.simulations = 0
        self.moves = 0
        self.max_edges = 0
        self.n_truncated = 0
        self.truncated_mass = 0.0
        self.max_depth = 0
        self.depth_hist = torch.zeros(self.d_max + 1, dtype=torch.int64, device=dev)
        self.max_nodes_used = 0
        self.pool_fill = 0
        self.n_terminal_descents = 0
        self.n_terminal_children = 0
        self.n_empty_mask_expansions = 0
        self.root_max_pi = 0.0
        self.root_entropy = 0.0
        self.root_covered = 0
        self.search_disagrees = 0
        self.saturated_value = 0
        self.value_samples = 0
        self.terminal_codes = torch.zeros(6, dtype=torch.int64, device=dev)
        self.game_lengths = []

    # -- collection ------------------------------------------------------- #

    def on_expand(self, count, n_edges, dropped, do) -> None:
        rows = do.nonzero(as_tuple=True)[0]
        if not rows.numel():
            return
        self.max_edges = max(self.max_edges, int(count[rows].max()))
        cut = do & (n_edges < count)
        if bool(cut.any()):
            self.n_truncated += int(cut.sum())
            # The mass matters more than the count: dropping a tail worth 0.1 %
            # of the prior is harmless and dropping one worth 20 % is not (§15.1).
            self.truncated_mass += float(dropped[cut].sum())
        self.n_empty_mask_expansions += int((do & (count == 0)).sum())

    def on_simulation(self, search: "Search", active, fresh, code) -> None:
        self.simulations += int(active.sum())
        depth = search.path_len[active].long()
        if depth.numel():
            self.max_depth = max(self.max_depth, int(depth.max()))
            self.depth_hist += torch.bincount(depth, minlength=self.d_max + 1)
        # A descent that created no node ended on a stored terminal, and its slot
        # in the encoder batch was spent on a value it already knew (§6.3).
        self.n_terminal_descents += int((active & ~fresh).sum())
        self.n_terminal_children += int((active & fresh & (code != 0)).sum())

    def on_move(self, search: "Search", pi, code, chosen) -> None:
        self.moves += int(pi.shape[0])
        self.max_nodes_used = max(self.max_nodes_used, int(search.node_count.max()))
        self.pool_fill += int(search.node_count.sum())
        self.root_max_pi += float(pi.max(-1).values.sum())
        p = pi.clamp(min=torch.finfo(pi.dtype).tiny)
        self.root_entropy += float(-(pi * p.log()).sum())
        self.root_covered += int((pi > 0).sum())
        # Search that never disagrees with the policy is search that is not
        # earning its cost, and this is the cheapest signal that `n` is too small
        # or that `cpuct` is wrong (§15.2).
        prior = search.edge_prior[:, 0].float()
        self.search_disagrees += int((_lowest_argmax(pi) != _lowest_argmax(prior)).sum())
        v = search.node_value[:, 0].float()
        self.saturated_value += int(((2.0 * v - 1.0).abs() > 0.99).sum())
        self.value_samples += int(v.numel())
        self.terminal_codes += torch.bincount(code.long(), minlength=6)
        ended = search.game_done.nonzero(as_tuple=True)[0]
        if ended.numel():
            self.game_lengths.extend(search.game_ply[ended].tolist())

    # -- reporting -------------------------------------------------------- #

    def _depth_quantile(self, q: float) -> int:
        total = int(self.depth_hist.sum())
        if total == 0:
            return 0
        cum = self.depth_hist.cumsum(0)
        return int((cum >= q * total).to(torch.uint8).argmax())

    def snapshot(self) -> dict:
        moves = max(self.moves, 1)
        sims = max(self.simulations, 1)
        lengths = self.game_lengths
        return {
            "simulations": self.simulations,
            "moves": self.moves,
            # §15.1, the fixed sizes
            "max_edges": self.max_edges,
            "n_truncated": self.n_truncated,
            "truncated_prior_mass": self.truncated_mass,
            "max_depth": self.max_depth,
            "depth_p50": self._depth_quantile(0.5),
            "depth_p99": self._depth_quantile(0.99),
            "max_nodes_used": self.max_nodes_used,
            "mean_pool_fill": self.pool_fill / moves,
            "n_terminal_descents": self.n_terminal_descents,
            "n_terminal_children": self.n_terminal_children,
            "n_empty_mask_expansions": self.n_empty_mask_expansions,
            # §15.2, whether the search is doing anything
            "mean_root_max_pi": self.root_max_pi / moves,
            "mean_root_entropy": self.root_entropy / moves,
            "mean_root_edges_visited": self.root_covered / moves,
            "search_disagrees_frac": self.search_disagrees / moves,
            # §15.3, numerical and rules health
            "saturated_value_frac": self.saturated_value / max(self.value_samples, 1),
            # By name, not by index. `terminal_codes/4` in a dashboard means nothing
            # to the person reading it, and the mapping is spec §4.3 — `env`'s
            # `TERMINAL_NAMES` is the one copy of it.
            **{f"terminal_{name}": int(n)
               for name, n in zip(_TERMINAL_NAMES_ORDERED, self.terminal_codes.tolist())},
            "mean_game_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "games_finished": len(lengths),
            "terminal_descent_frac": self.n_terminal_descents / sims,
        }


def check_invariants(search: Search) -> None:
    """§5, asserted over the whole batch. The reference runs this in its tests."""
    c = search.config
    b = search._b
    if not bool((search.node_count <= c.n_max).all()):
        raise AssertionError("1: node_count exceeded Nmax")
    if not bool((search.node_parent[:, 0] == -1).all()):
        raise AssertionError("2: node 0 is not the root")

    live = (torch.arange(c.n_max, device=search.device)[None, :]
            < search.node_count[:, None])
    code = search.node_flags & TERMINAL
    nedges = search.node_nedges.long()

    child = search.edge_child.long()
    has = child >= 0
    rows, nodes, edges = has.nonzero(as_tuple=True)
    if rows.numel():
        kid = child[rows, nodes, edges]
        if not bool((search.node_parent[rows, kid].long() == nodes).all()):
            raise AssertionError("3: node_parent disagrees with edge_child")
        if not bool((search.node_pedge[rows, kid].long() == edges).all()):
            raise AssertionError("3: node_pedge disagrees with edge_child")

    if bool((live & (code != 0) & (nedges != 0)).any()):
        raise AssertionError("4: a terminal node has edges")
    if bool((live & (code == 0) & (nedges == 0)).any()):
        raise AssertionError("5: a non-terminal node has no edges")

    valid = (torch.arange(c.E, device=search.device)[None, None, :] < nedges[:, :, None])
    visits = torch.where(valid, search.edge_N.long(), torch.zeros_like(nedges)[:, :, None])
    if not bool((visits[:, 0].sum(-1) == search.budget).all()):
        raise AssertionError("6: root visits do not sum to the budget")
    if bool(((search.edge_N == 0) & (search.edge_child != -1)).any()):
        raise AssertionError("7: an unvisited edge has a child")

    node_visits = visits.sum(-1)
    inner = live & (code == 0)
    inner[:, 0] = False
    rows, nodes = inner.nonzero(as_tuple=True)
    if rows.numel():
        parent = search.node_parent[rows, nodes].long()
        pedge = search.node_pedge[rows, nodes].long()
        own = node_visits[rows, nodes]
        through = search.edge_N[rows, parent, pedge].long()
        # A node's own visit count is one behind the edge that reaches it: the
        # edge counts the simulation that created the node, which had no
        # statistics to walk through it yet.
        if not bool(((through - own) == 1).all()):
            raise AssertionError("7: edge visits disagree with the subtree's")

    if bool((search.game_done & (search.game_ply == 0)).any()):
        raise AssertionError("8: a game is done before it started")
    q = search.edge_Q
    if bool(((q < 0.0) | (q > 1.0)).any()):
        raise AssertionError("9: edge_Q left [0,1]")
