"""An independent AlphaZero MCTS, written from the papers, as an oracle for `docs/mcts.md`.

`perft` gave the move generator something outside this repository to be wrong
against. The search had nothing, so `brokefish/search/torch_impl.py` was checked by
construction and by parts. This is the missing piece: a second implementation that
shares no code with it and is deliberately built the other way round, so that
agreement between the two is evidence rather than tautology.

Provenance. Every rule below is taken from one of two documents and the source is
named at each step:

* **AGZ** = Silver et al., *Mastering the game of Go without human knowledge*,
  Nature 550:354-359 (2017), Methods, "Search Algorithm". This is the authority for
  the search itself, because AlphaZero's own paper says "unless otherwise
  specified, the training and search algorithm and parameters are identical to
  AlphaGo Zero".
* **AZ** = Silver et al., *Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm*, arXiv:1712.01815 (2017), and its Science
  version. The authority for the parameters that differ for chess.

⚠️ The `pseudocode.py` released with the Science paper is **not** used as an
authority here, and three of its details are not followed. `docs/mcts.md` §13
records why. It stores values per node and reads them un-negated, it uses
`parent.visit_count` where AGZ's formula uses the sum over the node's edges, and it
mixes un-normalised `numpy.random.gamma` samples where AGZ specifies `Dir(alpha)`.

Four things are deliberately the opposite of the reference implementation, so that
a shared mistake cannot hide:

1. **Values are stored per node**, in that node's own mover's frame, and a parent
   reads `1 - child.value()`. The reference stores per edge and flips at write
   time. `docs/mcts.md` §6.5 says the two are equivalent; this is what checks it.
2. **The tree is dicts and objects**, one game at a time, with the path held as a
   Python list of node references rather than an index array.
3. **The repetition window is walked up the parent links**, not read from a path
   array.
4. **Node indices are assigned by a counter**, and the comparison asserts that node
   `i` of the oracle is node `i` of the reference, which pins the allocation order
   as well as the contents.

The rules come from `brokefish.env`, the same engine the reference uses. That is
deliberate: `perft` already validates the engine to depth 6, and what is under test
here is the search.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from brokefish.env import torch_impl as _default_env
from brokefish.env.torch_impl import INSUFFICIENT, PAWN, REPETITION, SQUARE

M64 = (1 << 64) - 1
PHI = 0x9E3779B97F4A7C15
MOVE_BITS = 11


# --------------------------------------------------------------------------- #
# A deterministic evaluator both implementations can share, bit for bit
# --------------------------------------------------------------------------- #
#
# A differential test needs both sides to see identical numbers. The real network
# cannot supply that: it is called on a batch of B in the reference and on one
# position in the oracle, and an fp32 reduction is not guaranteed to give the same
# last bit at two batch sizes. So the evaluator is integer arithmetic, and every
# float it produces is a small integer over a power of two, which is exact in fp32
# and in fp64 alike.
#
# splitmix64's finaliser, the same one `brokefish/env/luts.py` uses for the Zobrist
# keys, chosen for the same reason: ten lines of shift-multiply-xor that give
# identical output in Python and in torch.


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def _signed(x: int) -> int:
    """The same 64 bits as an int64, since torch has no usable uint64 arithmetic."""
    return x - (1 << 64) if x >> 63 else x


def _lshr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical shift right on int64. torch's `>>` is arithmetic and would sign-extend."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _signed_mul(k: torch.Tensor) -> torch.Tensor:
    """`PHI * k mod 2**64` as int64. int64 multiplication wraps, which is what we want."""
    return k * _signed(PHI)


def _mix64_t(z: torch.Tensor) -> torch.Tensor:
    z = (z ^ _lshr(z, 30)) * _signed(0xBF58476D1CE4E5B9)
    z = (z ^ _lshr(z, 27)) * _signed(0x94D049BB133111EB)
    return z ^ _lshr(z, 31)


def _seed(words: Sequence[int], control: int, rep: int) -> int:
    """One 64-bit seed per (position, rep). XOR-folded, so slot order cannot matter."""
    acc = mix64((PHI * 41 + (control & 0xFFFF)) & M64) ^ mix64((PHI * 42 + rep) & M64)
    for i, w in enumerate(words):
        acc ^= mix64((PHI * (i + 1) + (w & 0xFFFF)) & M64)
    return mix64(acc)


def _entry(seed: int, index: int) -> float:
    """A float in [0,1) from 24 bits, so it is exact in fp32 and fp64 alike."""
    return float(mix64((seed + PHI * (index + 1)) & M64) >> 40) / 16777216.0


class ScalarEval:
    """The evaluator for one position, computed lazily. Logits in [-4,4), value in [-1,1)."""

    __slots__ = ("seed", "value")

    def __init__(self, words: Sequence[int], control: int, rep: int) -> None:
        self.seed = _seed(words, control, rep)
        self.value = (_entry(self.seed, 4096) - 0.5) * 2.0

    def policy(self, slot: int, square: int) -> float:
        return (_entry(self.seed, slot * 64 + square) - 0.5) * 8.0

    def promo(self, slot: int, kind: int) -> float:
        return (_entry(self.seed, 2048 + slot * 4 + kind) - 0.5) * 8.0


def batched_eval(boards: torch.Tensor, control: torch.Tensor, rep: torch.Tensor):
    """The same function over a batch, in the shape `Search` expects for `evaluate`."""
    dev = boards.device
    w = boards.to(torch.int64) & 0xFFFF
    ones = torch.ones_like(control, dtype=torch.int64)
    acc = (_mix64_t(ones * _signed(PHI * 41 & M64) + (control.to(torch.int64) & 0xFFFF))
           ^ _mix64_t(ones * _signed(PHI * 42 & M64) + rep.to(torch.int64)))
    slots = torch.arange(w.shape[1], device=dev, dtype=torch.int64)
    per_slot = _mix64_t(_signed_mul(slots + 1) + w)
    seed = _mix64_t(acc ^ _fold_xor(per_slot))[:, None]

    def entries(index: torch.Tensor) -> torch.Tensor:
        u = _lshr(_mix64_t(seed + _signed_mul(index.reshape(1, -1) + 1)), 40)
        return u.to(torch.float32) / 16777216.0

    n = boards.shape[0]
    policy = (entries(torch.arange(2048, device=dev)) - 0.5) * 8.0
    promo = (entries(torch.arange(2048, 2048 + 128, device=dev)) - 0.5) * 8.0
    value = (entries(torch.tensor([4096], device=dev)) - 0.5) * 2.0
    return policy.view(n, 32, 64), promo.view(n, 32, 4), value.view(n)


def _fold_xor(x: torch.Tensor) -> torch.Tensor:
    """XOR along the last axis. torch has no `xor` reduction, and a loop over 32 is fine."""
    out = x[:, 0]
    for i in range(1, x.shape[1]):
        out = out ^ x[:, i]
    return out


def _f16(x: float) -> float:
    """§4.2 stores priors and node values as fp16, so the oracle does too."""
    return float(np.float16(x))


def ulp16(x: float) -> float:
    """The gap between adjacent fp16 values at `x`.

    Both implementations compute the same real prior and round it to fp16, but by
    different routes: torch softmaxes over the 8192-entry candidate axis in fp32, the
    oracle over the node's thirty-odd edges in fp64. When the real value sits near a
    rounding boundary the two can land on either side of it, which is a one-ULP
    disagreement and not an algorithmic one. Measured at 64 simulations it never
    happens; at 512 it happens twice in about 15,000 edges.
    """
    x = abs(float(np.float16(x)))
    return float(np.spacing(np.float16(x if x else 1e-8)))


# --------------------------------------------------------------------------- #
# The tree
# --------------------------------------------------------------------------- #

class AZNode:
    """A node, with AGZ's per-edge statistics held as parallel lists.

    ⚠️ `value_sum` is the sum of backed-up values **in this node's own mover's
    frame**, which is the released pseudocode's convention and the opposite of the
    reference's per-edge storage. A parent therefore reads `1 - child.value()`.
    """

    __slots__ = ("index", "board", "control", "hash", "rep", "code", "result",
                 "value", "irreversible", "parent", "pedge",
                 "moves", "priors", "children", "visits", "value_sum")

    def __init__(self, index: int, board: List[int], control: int, hash_: int,
                 rep: int, code: int, result: int, irreversible: bool,
                 parent: Optional["AZNode"], pedge: int) -> None:
        self.index = index
        self.board = board
        self.control = control
        self.hash = hash_
        self.rep = rep
        self.code = code
        self.result = result
        self.irreversible = irreversible
        self.parent = parent
        self.pedge = pedge
        self.value = 0.0
        self.moves: List[Tuple[int, int]] = []
        self.priors: List[float] = []
        self.children: List[Optional["AZNode"]] = []
        self.visits = 0
        self.value_sum = 0.0

    def q(self) -> float:
        """AGZ: `Q = W / N`. Zero for an unvisited edge, which is AGZ's `Q(s_L,a) = 0`."""
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class AZSearch:
    """One game's search. Construct, then call `run` once per move.

    `n`, `E`, `pb_c_base`, `pb_c_init`, `alpha`, `eps` and `tau_plies` have the
    meanings of `docs/mcts.md` §4.1. `noise` overrides the Dirichlet draw, which is
    how the differential test drives both implementations from the same numbers.
    """

    def __init__(self, board: torch.Tensor, control: torch.Tensor,
                 ring: Sequence[int] = (), n: int = 64, E: int = 64,
                 pb_c_base: float = 19652.0, pb_c_init: float = 1.25,
                 eps: float = 0.0, noise: Optional[Sequence[float]] = None,
                 env=None) -> None:
        self.env = env if env is not None else _default_env
        self.n, self.E = n, E
        self.pb_c_base, self.pb_c_init = pb_c_base, pb_c_init
        self.eps, self.noise = eps, noise
        self.ring = list(ring)
        self.count = 0
        self.margins: List[float] = []
        self.exact_ties = 0

        self.device = board.device
        self.board_t = board.reshape(1, 32).to(torch.int16)
        self.control_t = control.reshape(1).to(torch.int16)
        hash_t = self.env.hash_position(self.board_t, self.control_t)
        rep = 1 + sum(1 for h in self.ring if h == int(hash_t[0]))
        mask, in_check = self.env.movegen(self.board_t, self.control_t)
        code, result = self.env.terminal(mask, in_check, self.control_t, self.board_t)
        if rep >= 3 and int(code[0]) in (0, INSUFFICIENT):
            code, result = torch.tensor([REPETITION]), torch.tensor([0])
        if int(code[0]) != 0:
            raise AssertionError("invariant 8: the root is terminal")

        self.root = self._node(self.board_t, self.control_t, hash_t, rep,
                               int(code[0]), int(result[0]), False, None, 0)
        self._expand(self.root, mask)
        self._add_noise(self.root)

    # -- allocation ------------------------------------------------------- #

    def _node(self, board, control, hash_, rep, code, result, irrev, parent, pedge):
        node = AZNode(self.count, board.reshape(-1).tolist(), int(control[0]),
                      int(hash_[0]), rep, code, result, irrev, parent, pedge)
        self.count += 1
        return node

    # -- AGZ, "Expand and evaluate" --------------------------------------- #

    def _is_promotion(self, board: List[int], control: int, slot: int, square: int) -> bool:
        """A live pawn moving onto the mover's last rank (spec §3).

        The type comes from the piece word and never from the slot index: a promoted
        queen keeps its pawn slot (spec §2.1).
        """
        word = board[slot] & 0xFFFF
        if (word >> 11) & 1 or ((word >> 6) & 0b111) != PAWN:
            return False
        return square // 8 == (0 if control < 0 else 7)

    def _expand(self, node: AZNode, mask: torch.Tensor) -> None:
        """AGZ: each edge is initialised to `{N=0, W=0, Q=0, P=p_a}`.

        The enumeration order is `docs/mcts.md` §6.4's: ascending by slot, then by
        target square, then by promotion type in spec §3's `N B R Q` order.
        """
        ev = ScalarEval(node.board, node.control, min(node.rep - 1, 2))
        node.value = _f16((ev.value + 1.0) / 2.0)

        words = mask.reshape(32).tolist()
        cands: List[Tuple[int, int, float]] = []
        for slot in range(32):
            word = words[slot] & M64      # int64 carrying a uint64 pattern
            for square in range(64):
                if not (word >> square) & 1:
                    continue
                base = ev.policy(slot, square)
                if self._is_promotion(node.board, node.control, slot, square):
                    # §6.4: log space, so one softmax normalises both halves of
                    # P(target | piece) * P(type | piece) at once.
                    logits = [ev.promo(slot, k) for k in range(4)]
                    top = max(logits)
                    total = math.log(sum(math.exp(x - top) for x in logits)) + top
                    for k in range(4):
                        cands.append((slot * 64 + square, k, base + logits[k] - total))
                else:
                    cands.append((slot * 64 + square, 0, base))

        if not cands:
            raise AssertionError("invariant 5: expanding a node with no legal move")
        if len(cands) > self.E:
            # §6.4 step 3: the E highest unnormalised priors, ties by canonical
            # order, survivors written in canonical order.
            best = sorted(range(len(cands)), key=lambda j: (-cands[j][2], j))[:self.E]
            cands = [cands[j] for j in sorted(best)]

        top = max(c[2] for c in cands)
        exps = [math.exp(c[2] - top) for c in cands]
        total = sum(exps)
        node.moves = [(c[0], c[1]) for c in cands]
        node.priors = [_f16(e / total) for e in exps]
        node.children = [None] * len(cands)

    def _add_noise(self, node: AZNode) -> None:
        """AGZ: `P(s,a) = (1 - eps) p_a + eps eta_a`, `eta ~ Dir(alpha)`, `eps = 0.25`.

        ⚠️ `eta` is a **normalised** Dirichlet sample. The released pseudocode mixes
        raw `numpy.random.gamma` draws, which for about 31 legal moves sum to
        roughly `31 * 0.3 = 9.3` rather than 1 and inflate the exploration term
        several-fold. AGZ's `Dir(alpha)` is unambiguous and is what this follows.
        """
        if self.eps <= 0.0:
            return
        if self.noise is None:
            raise ValueError("pass `noise` explicitly: the oracle draws none of its own")
        eta = list(self.noise)[:len(node.priors)]
        if abs(sum(eta) - 1.0) > 1e-5:
            raise ValueError(f"noise is not normalised, sums to {sum(eta)}")
        node.priors = [_f16((1.0 - self.eps) * p + self.eps * e)
                       for p, e in zip(node.priors, eta)]

    # -- AGZ, "Select" ---------------------------------------------------- #

    def _select(self, node: AZNode) -> int:
        """`a = argmax(Q(s,a) + U(s,a))` with AGZ's

            U(s,a) = c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))

        ⚠️ `sum_b N(s,b)` is the sum over **this node's own edges**, which for an
        interior node is one less than the number of times the node has been visited
        (the simulation that created it took no edge here). The released pseudocode
        uses `parent.visit_count` instead, which is that number, and therefore
        differs from AGZ's formula at every node below the root.

        `c_puct` is "a constant determining the level of exploration" in AGZ and its
        value is not published. The logarithmic form below comes from the released
        pseudocode; over `n = 800` simulations it moves between 1.25 and 1.29.
        """
        visits = [0 if c is None else c.visits for c in node.children]
        total = sum(visits)
        c_puct = math.log((total + self.pb_c_base + 1.0) / self.pb_c_base) + self.pb_c_init
        common = c_puct * math.sqrt(total)

        best, best_score, runner = -1, -math.inf, -math.inf
        for i, child in enumerate(node.children):
            u = common * node.priors[i] / (visits[i] + 1.0)
            # The read-time flip: `child.q()` is in the child's mover's frame and the
            # player choosing here is the other one. An unvisited edge scores 0,
            # which is AGZ's `Q(s_L,a) = 0` initialisation.
            q = 0.0 if child is None or child.visits == 0 else 1.0 - child.q()
            score = u + q
            if score > best_score:
                runner, best_score, best = best_score, score, i
            elif score > runner:
                runner = score
        if runner > -math.inf:
            margin = best_score - runner
            if margin == 0.0:
                self.exact_ties += 1
            else:
                self.margins.append(margin)
        return best

    # -- §7, the repetition window ---------------------------------------- #

    def _repetition(self, parent: AZNode, child_hash: int, child_irrev: bool) -> int:
        """How many times this position has occurred, counting itself.

        Walks up the parent links rather than reading a path array, which is the
        point: `docs/mcts.md` §7's rule is implemented twice, two different ways.
        """
        if child_irrev:
            return 1
        count = 1
        use_ring = True
        node: Optional[AZNode] = parent
        while node is not None:
            if node.hash == child_hash:
                count += 1
            # The root's own flag belongs to the real game and the ring already
            # accounts for it (spec §6.2), so only nodes below the root cut.
            if node.parent is not None and node.irreversible:
                use_ring = False
                break
            node = node.parent
        if use_ring:
            count += sum(1 for h in self.ring if h == child_hash)
        return count

    # -- one simulation --------------------------------------------------- #

    def _child(self, node: AZNode, edge: int) -> AZNode:
        move, promo = node.moves[edge]
        board = torch.tensor([node.board], dtype=torch.int16, device=self.device)
        control = torch.tensor([node.control], dtype=torch.int16, device=self.device)
        hash_ = torch.tensor([node.hash], dtype=torch.int64, device=self.device)
        nb, nc, nh, irrev = self.env.step(
            board, control, torch.tensor([move], device=self.device),
            promo=torch.tensor([promo], device=self.device), hash=hash_)
        mask, in_check = self.env.movegen(nb, nc)
        rep = self._repetition(node, int(nh[0]), bool(irrev[0]))
        code, result = self.env.terminal(mask, in_check, nc, nb)
        code, result = int(code[0]), int(result[0])
        if rep >= 3 and code in (0, INSUFFICIENT):
            code, result = REPETITION, 0

        child = self._node(nb, nc, nh, rep, code, result, bool(irrev[0]), node, edge)
        if code != 0:
            # §8: a terminal node carries its result in the [0,1] convention and is
            # never expanded, so the network is never called on it.
            child.value = _f16((result + 1.0) / 2.0)
        else:
            self._expand(child, mask)
        return child

    def _simulate(self) -> None:
        path = [self.root]
        node = self.root
        while node.code == 0:
            edge = self._select(node)
            child = node.children[edge]
            if child is None:
                child = self._child(node, edge)
                node.children[edge] = child
                path.append(child)
                break
            path.append(child)
            node = child

        # AGZ, "Backup": N += 1 and W += v for every step on the path. The value is
        # re-expressed in each node's own mover's frame, which alternates.
        leaf = path[-1].value
        last = len(path) - 1
        for k, step in enumerate(path):
            step.visits += 1
            step.value_sum += leaf if (last - k) % 2 == 0 else 1.0 - leaf

    def run(self) -> AZNode:
        for _ in range(self.n):
            self._simulate()
        return self.root

    # -- §6.7 ------------------------------------------------------------- #

    def visit_distribution(self) -> List[float]:
        """AGZ's `pi(a|s0) = N(s0,a)^(1/tau) / sum_b N(s0,b)^(1/tau)` at `tau = 1`."""
        visits = [0 if c is None else c.visits for c in self.root.children]
        total = float(sum(visits))
        return [v / total for v in visits]

    def min_margin(self) -> float:
        return min(self.margins) if self.margins else math.inf


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #

def walk(root: AZNode) -> List[AZNode]:
    """Every node, indexed by its allocation order."""
    out: List[AZNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(c for c in node.children if c is not None)
    out.sort(key=lambda n: n.index)
    return out


def deltas(search, row: int, oracle: AZSearch) -> Tuple[float, float, float]:
    """The largest `(prior, Q, value)` disagreement, as magnitudes rather than verdicts.

    What the differential test needs is not a tolerance but a *bound*: a selection can
    only have been flipped by arithmetic if two scores were closer together than the
    error in a score. Priors come out bit-identical, so that error is the error in `Q`,
    and measuring it here lets the guard in `test_oracle.py` calibrate itself instead
    of trusting a constant.
    """
    prior = q = value = 0.0
    for node in walk(oracle.root)[: int(search.node_count[row])]:
        i = node.index
        value = max(value, abs(float(search.node_value[row, i]) - node.value))
        for e in range(len(node.moves)):
            prior = max(prior, abs(float(search.edge_prior[row, i, e]) - node.priors[e]))
            child = node.children[e]
            want = 0.0 if child is None or child.visits == 0 else 1.0 - child.q()
            q = max(q, abs(float(search.edge_Q[row, i, e]) - want))
    return prior, q, value


def differences(search, row: int, oracle: AZSearch, atol_q: float = 1e-6,
                atol_prior: float = 0.0, prior_ulps: float = 1.0) -> List[str]:
    """Every place the reference's row `row` disagrees with the oracle's tree.

    Node `i` of one must be node `i` of the other, which pins the allocation order
    as well as the contents.

    The tolerances are measured rather than guessed. Priors and node values come out
    **bit-identical**, because the evaluator is exact in fp32 and both sides round to
    fp16 as §4.2 requires, so `atol_prior` defaults to zero and a regression there is
    worth hearing about. `Q` is the reference's fp32 running mean against the
    oracle's fp64 sum-then-divide, measured at most 1e-7 apart over 64 simulations on
    seven positions.
    """
    from brokefish.search.torch_impl import IRREVERSIBLE, TERMINAL

    diffs: List[str] = []
    nodes = walk(oracle.root)
    got_count = int(search.node_count[row])
    if got_count != len(nodes):
        diffs.append(f"node count: reference {got_count}, oracle {len(nodes)}")

    for node in nodes[:got_count]:
        i, at = node.index, f"node {node.index}"
        if search.node_board[row, i].tolist() != node.board:
            diffs.append(f"{at}: board differs")
            continue
        if int(search.node_control[row, i]) != node.control:
            diffs.append(f"{at}: control {int(search.node_control[row, i])} vs {node.control}")
        if int(search.node_hash[row, i]) != node.hash:
            diffs.append(f"{at}: hash differs")
        if int(search.node_flags[row, i] & TERMINAL) != node.code:
            diffs.append(f"{at}: terminal code "
                         f"{int(search.node_flags[row, i] & TERMINAL)} vs {node.code}")
        if bool(search.node_flags[row, i] & IRREVERSIBLE) != node.irreversible:
            diffs.append(f"{at}: irreversible flag differs")
        if float(search.node_value[row, i]) != node.value:
            diffs.append(f"{at}: value {float(search.node_value[row, i])} vs {node.value}")
        if int(search.node_nedges[row, i]) != len(node.moves):
            diffs.append(f"{at}: nedges {int(search.node_nedges[row, i])} vs {len(node.moves)}")
            continue
        if node.parent is not None:
            if int(search.node_parent[row, i]) != node.parent.index:
                diffs.append(f"{at}: parent differs")
            if int(search.node_pedge[row, i]) != node.pedge:
                diffs.append(f"{at}: parent edge differs")

        for e, (move, promo) in enumerate(node.moves):
            label = int(search.edge_move[row, i, e])
            if label != (move | (promo << MOVE_BITS)):
                diffs.append(f"{at} edge {e}: move {label} vs {move}|{promo}<<11")
                continue
            child = node.children[e]
            want_n = 0 if child is None else child.visits
            got_n = int(search.edge_N[row, i, e])
            if got_n != want_n:
                diffs.append(f"{at} edge {e}: N {got_n} vs {want_n}")
            # The parity, read the other way round: per-edge Q at the parent has to
            # equal one minus the per-node mean at the child.
            want_q = 0.0 if child is None or child.visits == 0 else 1.0 - child.q()
            got_q = float(search.edge_Q[row, i, e])
            if abs(got_q - want_q) > atol_q:
                diffs.append(f"{at} edge {e}: Q {got_q:.6f} vs {want_q:.6f}")
            got_p = float(search.edge_prior[row, i, e])
            if abs(got_p - node.priors[e]) > atol_prior + prior_ulps * ulp16(node.priors[e]):
                diffs.append(f"{at} edge {e}: prior {got_p:.6f} vs {node.priors[e]:.6f}")
            want_c = -1 if child is None else child.index
            if int(search.edge_child[row, i, e]) != want_c:
                diffs.append(f"{at} edge {e}: child "
                             f"{int(search.edge_child[row, i, e])} vs {want_c}")
        if len(diffs) > 40:
            diffs.append("... truncated")
            break
    return diffs
