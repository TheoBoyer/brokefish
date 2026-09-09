"""The CUDA search against the reference, tree for tree.

`docs/mcts.md` §12 makes the PyTorch reference the oracle for the kernels, so
every test here runs one batch of positions through both implementations from the
same seed and compares §4.2's arrays after **every simulation**. A divergence is
then attributable to the simulation that caused it rather than to the four hundred
that followed.

What agreement is worth, measured rather than asserted. `edge_Q`, `node_value` and
every integer field come out bit-identical. `edge_prior` does not always: the
kernel's softmax denominator is a warp reduction over at most 64 surviving
candidates and the reference's is a torch reduction over the 8192-wide masked row,
so the two agree to about one fp16 ULP and differ on roughly one edge in a
hundred. That difference is small but it is *not* provably too small to change a
selection, and :func:`_probe` measures both sides of that: the closest call any
selection came, and the largest score perturbation the prior rounding could have
caused. The numbers are printed rather than hidden, because for the hash evaluator
the second is larger than the first and the agreement below is therefore empirical.

:func:`test_flat_priors_agree_bit_for_bit` is the test with a hard guarantee.
Constant logits make every prior an exact `1/k` on both sides, so nothing is within
tolerance of anything, and every PUCT score in a fresh node ties, which is the
hardest available exercise of §6.6's lowest-index tie-break.

    python -m tests.test_search_cuda [--slow]
"""

from __future__ import annotations

import sys

import pytest
import torch

from brokefish.env import cuda_impl as env
from brokefish.search import SearchConfig, search_impl
from brokefish.search.cuda_impl import _ext, TREE_FIELDS
from brokefish.search.torch_impl import FPU_DRAW, TERMINAL, Search as RefSearch
from tests.boards import random_positions
from tests.oracle import batched_eval

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the CUDA search needs a GPU")

# Clock 99: every move takes it to 101, which spec §4.3 calls a draw, so the
# terminal code lands one ply into the tree rather than needing to be searched for.
FIFTY_MOVE = "4k3/8/8/8/8/8/8/4K2R w - - 99 1"
# The classic maximum-mobility position: 218 legal moves for White and no pawns, so
# every candidate's logit is a policy entry read identically on both sides and
# §4.3's truncation has to agree exactly rather than to a tolerance.
MAX_MOBILITY = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"
# White mates in one with Re1-e8, and it is the only mate, so §6.6a collapses the
# root onto a single edge and the two implementations have to agree on which.
MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
# Pawnless, so no promotion can ever appear and every candidate's logit under
# `flat_eval` is exactly zero. See `test_flat_priors_agree_bit_for_bit`.
PAWNLESS = ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
            "3qk3/8/8/8/8/8/8/3QK2R w K - 0 1",
            "4k1r1/8/8/8/5N2/8/8/4K1NR w K - 0 1",
            "2b1k3/8/8/8/8/8/8/2B1K1NR w K - 0 1")

# Fields where the two implementations must agree exactly, not to a tolerance.
EXACT = ("node_board", "node_control", "node_hash", "node_nedges", "node_flags",
         "node_parent", "node_pedge", "edge_move", "edge_child", "edge_N",
         "path_node", "path_edge", "path_len", "node_count")


# --------------------------------------------------------------------------- #
# The two searches, and the comparison
# --------------------------------------------------------------------------- #

def hash_eval(boards, control, rep):
    """`tests.oracle`'s integer evaluator, in fp16 as `expand` reads it.

    Both implementations get the same tensor, so the fp16 rounding here is shared
    and is not a source of disagreement. The reference converts it back to fp32
    exactly, which is what the kernel's `__half2float` also does.
    """
    policy, promo, value = batched_eval(boards, control, rep)
    return policy.half(), promo.half(), value.float()


def flat_eval(boards, control, rep):
    """Constant logits, so every prior is an exact `1/k` on both sides.

    The value still varies per position, so the tree is not degenerate; only the
    policy is flat. Positions with a promotion are excluded by the caller, since
    a promotion edge's logit picks up `log_softmax` of four equal numbers, which
    is `-log 4` and not zero.
    """
    _, _, value = batched_eval(boards, control, rep)
    n = boards.shape[0]
    zeros = torch.zeros(n, 32, 64, device=boards.device, dtype=torch.float16)
    return zeros, zeros[:, :, :4].contiguous(), value.float()


class Recording(RefSearch):
    """The reference, keeping the four values it hands the encoder.

    §6.3's staging is the one thing the two implementations compute in genuinely
    different shapes: the reference gathers `node_board[b, leaf]` and the kernel
    writes the leaf out flat as it creates it. Comparing the gather against the
    write is what pins `leaf_rep`, and a wrong `rep` is invisible in the tree until
    it changes the evaluator's output several simulations later.
    """

    def simulate(self, s):
        active = self.budget > s
        leaf, parent, edge = self._descent(active)
        leaf, rep, code, mask, expand = self._create_child(leaf, parent, edge, active)
        self.staged = dict(node=leaf.clone(), rep=rep.clone(),
                           board=self.node_board[self._b, leaf].clone(),
                           control=self.node_control[self._b, leaf].clone())
        policy, promo, value = self.evaluate(
            self.node_board[self._b, leaf], self.node_control[self._b, leaf], rep)
        self._expand(leaf, mask, expand, policy, promo, value)
        self._backup(leaf, active)
        self.stats.on_simulation(self, active, parent >= 0, code)


class Probe:
    """Wraps the reference's `_select` to measure whether agreement could have failed.

    Two numbers per run. `margin` is the smallest gap between the best and the
    second-best PUCT score at any selection, which is how close the search came to
    a different move. `perturbation` is the largest difference between the score
    the reference would compute and the score the *kernel's own tree* implies at
    the same node, which is what the fp16 prior rounding is worth. Agreement is
    guaranteed only where the first exceeds the second.
    """

    def __init__(self, ref, cu):
        self.ref, self.cu, self.inner = ref, cu, ref._select
        self.margin = float("inf")
        self.perturbation = 0.0
        self.selections = 0
        self.ties = 0

    @staticmethod
    def _scores(s, v):
        """The reference's own score function, applied to whichever tree is passed.

        ⚠️ It **calls** `_puct_scores` rather than restating it. This used to be a
        transcription, and the constant in it had to be kept in step by hand -- which
        is the shape of the bug that produced four copies of the FPU slip. `cuda_impl`
        inherits the method, so `_scores(cu, v)` scores the kernel's tree with the
        reference's arithmetic, which is exactly what the perturbation below means.
        """
        return s._puct_scores(v)

    def __call__(self, v, depth: int = 0):
        # `depth` reaches `_select` because §11's two Gumbel rules differ by it; the
        # PUCT probe ignores it and passes it straight through.
        a = self._scores(self.ref, v)
        b = self._scores(self.cu, v)
        finite = torch.isfinite(a) & torch.isfinite(b)
        if bool(finite.any()):
            self.perturbation = max(self.perturbation,
                                    float((a - b)[finite].abs().max()))
        top = a.topk(min(2, a.shape[1]), dim=-1).values
        if top.shape[1] == 2:
            gap = top[:, 0] - top[:, 1]
            gap = gap[torch.isfinite(gap)]
            # An exact tie is not a rounding risk and is excluded from the margin.
            # At a freshly created node `sqrt(N_v)` is 0, so every score is
            # `0 * prior + 0`, exactly zero whatever the prior rounded to, and the
            # tie-break decides it identically on both sides by construction.
            self.ties += int((gap == 0).sum())
            gap = gap[gap > 0]
            if gap.numel():
                self.margin = min(self.margin, float(gap.min()))
        self.selections += int(v.numel())
        return self.inner(v, depth)


def _pair(boards, control, n, eps=0.0, seed=11, stats=False, tau_plies=30,
          collapse=False):
    """A reference and a kernel search over the same games, from the same seed.

    The Dirichlet noise is not injected. Both classes inherit §6.1's torch
    implementation and make the same sequence of generator calls, so one seed
    gives one stream and §12's comparison runs at the real `eps` rather than
    having to switch the noise off.
    """
    B = boards.shape[0]
    # E follows the kernel's compile-time `kE`; pinning a literal here would make
    # every CUDA test fail the moment the cap moves, which is noise rather than a
    # signal -- `cuda_impl` already refuses a mismatch on construction.
    cfg = SearchConfig(n=n, B=B, E=int(_ext().edge_cap()), eps=eps, tau_plies=tau_plies,
                       terminal_collapse=collapse)
    ref = Recording(cfg, hash_eval, env=env, device="cuda", seed=seed)
    cu = search_impl("cuda")(cfg, hash_eval, env=env, device="cuda", seed=seed,
                             collect_stats=stats)
    ref.reset(boards.clone(), control.clone())
    cu.reset(boards.clone(), control.clone())
    probe = Probe(ref, cu)
    ref._select = probe
    return ref, cu, probe


def _ulp16(x: torch.Tensor) -> torch.Tensor:
    """The gap between adjacent fp16 values at each entry, elementwise on device.

    `tests.oracle.ulp16` is the scalar version and says why this tolerance exists.
    It has to be a tensor op here: the comparison runs after every one of up to 800
    simulations over an array of 3.3M priors, and `np.vectorize` over that is a
    Python loop that turns a two-minute test into an overnight one.
    """
    a = x.abs()
    # Below 2^-14 fp16 is subnormal and the spacing is a flat 2^-24; zero lands
    # here too, which is what we want.
    exponent = torch.floor(torch.log2(a.clamp(min=6.103515625e-5)))
    return torch.where(a < 6.103515625e-5,
                       torch.full_like(a, 5.9604644775390625e-8),
                       torch.exp2(exponent - 10.0))


def _differences(ref, cu, prior_ulps=1.0, atol_q=0.0):
    """Everywhere the two trees disagree, over the allocated part of the pool."""
    live = int(max(ref.node_count.max(), cu.node_count.max()))
    out = []
    for f in EXACT:
        x, y = getattr(ref, f), getattr(cu, f)
        if x.dim() >= 2 and x.shape[1] == ref.config.n_max:
            x, y = x[:, :live], y[:, :live]
        if not bool((x == y).all()):
            where = (x != y).nonzero()[:3].tolist()
            out.append(f"{f}: {int((x != y).sum())} entries differ, first at {where}")

    # §6.6a's mask, when it is allocated. Comparing only its *effect* would let the
    # two implementations disagree about which edges are winning and stay green for
    # as long as the disagreement never reached a node the descent revisited.
    if ref.edge_win.numel():
        x, y = ref.edge_win[:, :live], cu.edge_win[:, :live]
        if not bool((x == y).all()):
            out.append(f"edge_win: {int((x != y).sum())} entries differ, first at "
                       f"{(x != y).nonzero()[:3].tolist()}")

    p, q = ref.edge_prior[:, :live].float(), cu.edge_prior[:, :live].float()
    if prior_ulps == 0.0:
        if not bool((p == q).all()):
            out.append(f"edge_prior: {int((p != q).sum())} differ and none may")
    else:
        tol = prior_ulps * _ulp16(p)
        if ref.config.eps > 0.0:
            # The root's priors are quantised to fp16 **twice**: the kernel writes
            # them, then §6.1's noise reads them back, mixes `0.75 p + 0.25 eta` in
            # fp32 and writes fp16 again. A one-ULP difference in `p` survives the
            # 0.75 and picks up the output's own half-ULP rounding, so the two can
            # land two fp16 steps apart. Interior edges go through fp16 once and
            # stay inside one. Measured: exactly 2.00 ULP, at the root, on about one
            # edge in 10^5 with the noise on.
            tol[:, 0] *= 2.0
        over = (p - q).abs() > tol
        if bool(over.any()):
            worst = ((p - q).abs() / _ulp16(p)).max()
            out.append(f"edge_prior: {int(over.sum())} beyond tolerance, worst "
                       f"{float((p - q).abs().max()):.3g} = {float(worst):.2f} fp16 ULP")

    dq = float((ref.edge_Q[:, :live] - cu.edge_Q[:, :live]).abs().max())
    if dq > atol_q:
        out.append(f"edge_Q: max|delta| {dq:.3g}")
    dv = float((ref.node_value[:, :live].float()
                - cu.node_value[:, :live].float()).abs().max())
    if dv > 0.0:
        out.append(f"node_value: max|delta| {dv:.3g}")

    for name, want in (("node", ref.staged["node"].to(torch.int16)),
                       ("rep", ref.staged["rep"]), ("board", ref.staged["board"]),
                       ("control", ref.staged["control"])):
        got = getattr(cu, f"leaf_{name}")
        if not bool((want == got).all()):
            rows = (want != got).nonzero()[:3].tolist()
            out.append(f"leaf_{name}: differs at {rows}")
    return out


def _run(ref, cu, moves=1, prior_ulps=1.0, label=""):
    """Both searches, compared after every simulation and after every move."""
    n = ref.config.n
    for m in range(moves):
        ref.root_init()
        cu.root_init()
        for s in range(n):
            ref.simulate(s)
            cu.simulate(s)
            diffs = _differences(ref, cu, prior_ulps=prior_ulps)
            assert not diffs, (f"{label}move {m}, simulation {s}:\n  "
                               + "\n  ".join(diffs[:6]))
        r1, r2 = ref.select_and_advance(), cu.select_and_advance()
        for f in ("board", "control", "rep", "policy_move", "policy_prob", "policy_len",
                  "played", "ply", "root_value", "done", "result"):
            x, y = getattr(r1, f), getattr(r2, f)
            same = ((x.float() - y.float()).abs().max() == 0 if x.dtype.is_floating_point
                    else bool((x == y).all()))
            assert same, f"{label}move {m}: the training record's {f} differs"
        for f in ("game_board", "game_control", "game_hash", "game_ring",
                  "game_ring_len", "game_ply", "game_done", "game_result"):
            assert bool((getattr(ref, f) == getattr(cu, f)).all()), \
                f"{label}move {m}: {f} differs after the move"
        if bool(ref.game_done.any()):
            ref.reset_finished()
            cu.reset_finished()


def _report(probe, label):
    guaranteed = probe.margin > probe.perturbation
    print(f"    {label}: {probe.selections} selections ({probe.ties} exact ties), "
          f"closest non-tie {probe.margin:.3g}, prior rounding worth "
          f"{probe.perturbation:.3g} ({'guaranteed' if guaranteed else 'empirical'})")


def _live(n, plies, seed):
    """Random positions with the terminals dropped: invariant 8 forbids searching one."""
    boards, control, _ = random_positions(n, plies=plies, seed=seed, device="cuda")
    mask, in_check = env.movegen(boards, control)
    code, _ = env.terminal(mask, in_check, control, boards)
    keep = (code == 0).nonzero(as_tuple=True)[0]
    return boards[keep].contiguous(), control[keep].contiguous()


def _from_fen(fen, copies=2):
    b, c = env.from_fen(fen)
    return (b.to("cuda").expand(copies, -1).contiguous(),
            c.to("cuda").expand(copies).contiguous())


# --------------------------------------------------------------------------- #
# The guaranteed comparison
# --------------------------------------------------------------------------- #

def test_flat_priors_agree_bit_for_bit():
    """Constant logits, so no prior is within a tolerance of anything.

    Every prior is an exact `1/k`: the kernel's denominator is a sum of `k` ones
    and so is the reference's, and an integer sum in fp32 does not care about
    reduction order. That removes the only source of disagreement between the two,
    so this comparison is exact and not approximate. It is also the hardest test of
    §6.6's tie-break, since equal priors make every score at a fresh node tie.

    Pawnless positions, because a promotion edge picks up ``log_softmax`` of four
    equal numbers, which is ``-log 4`` and not zero, and the denominator stops
    being an integer. That is not hypothetical: from the start position this
    evaluator walks 45 plies deep and promotes.

    The depth is the other reason this test earns its place. With every prior
    equal and first-play urgency at 0, an unvisited edge scores 0 and a visited one
    with `Q > 0` beats it, so the search dives rather than spreads. It is the only
    test here that pushes a path past 32 levels, which is where §7's repetition
    scan and §6.5's backup need more than one iteration per lane.
    """
    pairs = [env.from_fen(f) for f in PAWNLESS]
    boards = torch.cat([b for b, _ in pairs]).to("cuda")
    control = torch.cat([c for _, c in pairs]).to("cuda")
    ref, cu, probe = _pair(boards, control, n=256, eps=0.0, stats=True)
    ref.evaluate = cu.evaluate = flat_eval
    _run(ref, cu, moves=1, prior_ulps=0.0, label="flat ")

    live = int(cu.node_count.max())
    words = cu.node_board[:, :live].long() & 0xFFFF
    is_pawn = (((words >> 11) & 1) == 0) & (((words >> 6) & 0b111) == 0)
    # An untouched pool slot is all zeros, which decodes as a live white pawn on
    # a1, so the check has to stop at each game's own node count and not at the
    # batch's maximum.
    allocated = (torch.arange(live, device="cuda")[None, :]
                 < cu.node_count[:, None].long())
    is_pawn &= allocated[:, :, None]
    assert not bool(is_pawn.any()), (
        "a pawn appeared, so a promotion edge can exist, its logit is no longer "
        "zero and this test has stopped being exact")
    # With equal priors every score at an unvisited node ties, so the tie-break
    # alone decides the first descent below every node the search creates.
    assert probe.ties > 0, "no selection tied, so the tie-break was not exercised"
    depth = cu.device_counters()["max_depth"]
    _report(probe, f"flat (max depth {depth})")


def spiked_eval(boards, control, rep):
    """A prior spiked on one move per position, so the search dives instead of spreads.

    ⚠️ **This used to be `flat_eval`, and the reason it cannot be is the §6.6 fix.**
    The old version relied on first-play urgency being `0`: an untried edge scored a
    certain loss, so any visited edge with `Q > 0` beat it and the search walked down
    a single line. That constant was the bug (`search.md` §6.6), and under the correct
    `FPU = 0.5` a flat prior makes the search expand every sibling before going deeper
    — max depth 3 instead of 33, which is the *right* behaviour and leaves §7's
    two-iteration scan and §6.5's backup loop untested.

    A prior concentrated on edge 0 restores the dive on its merits: the top edge
    scores `pb_c * ~1 + Q` against a sibling's `pb_c * ~0 + 0.5`, and with `Q` near
    the draw value the `pb_c` term keeps it ahead at every visit count.
    """
    _, _, value = batched_eval(boards, control, rep)
    n = boards.shape[0]
    # ⚠️ The spike has to land on a *legal* edge, so it is placed by movegen rather
    # than at a fixed index: slot 0 is the a1 rook, which has no legal move from the
    # start position, and spiking it leaves the prior flat and the tree shallow.
    mask = env.movegen(boards, control)[0]                       # [n, 32] bitboards
    bits = ((mask.unsqueeze(-1) >> torch.arange(64, device=boards.device)) & 1).bool()
    flat = bits.reshape(n, -1).float()
    first = torch.argmax(flat, dim=-1)                           # lowest legal (slot, sq)
    logits = torch.zeros(n, 32 * 64, device=boards.device, dtype=torch.float16)
    logits[torch.arange(n, device=boards.device), first] = 16.0
    promo = torch.zeros(n, 32, 4, device=boards.device, dtype=torch.float16)
    return logits.reshape(n, 32, 64), promo, value.float()


def test_agrees_on_a_path_longer_than_a_warp():
    """A path past 32 levels, where §7's scan and §6.5's backup loop twice per lane.

    From the start position, which is the deepest thing in this file. The pawns that
    reach the eighth rank on the way are why this cannot also be the bit-exact test.
    """
    boards, control = env.initial_boards(8, device="cuda")
    # n = 320, not 192: under the corrected §6.6 the dive costs one simulation per
    # ply, and 192 reaches depth 29 -- just under the 32 this test exists to cross.
    ref, cu, probe = _pair(boards, control, n=320, eps=0.0, stats=True)
    ref.evaluate = cu.evaluate = spiked_eval
    _run(ref, cu, label="deep path ")
    depth = cu.device_counters()["max_depth"]
    assert depth > 32, f"the deepest path was {depth}, so no lane looped twice"
    _report(probe, f"deep path (max depth {depth})")


# --------------------------------------------------------------------------- #
# The comparison under a real policy
# --------------------------------------------------------------------------- #

def test_agrees_from_the_start_position():
    boards, control = env.initial_boards(8, device="cuda")
    ref, cu, probe = _pair(boards, control, n=192, eps=0.0)
    _run(ref, cu, label="startpos ")
    _report(probe, "startpos")


def test_agrees_with_root_noise():
    """§6.1's Dirichlet, drawn from one torch stream and consumed by both."""
    boards, control = env.initial_boards(8, device="cuda")
    ref, cu, probe = _pair(boards, control, n=192, eps=0.25)
    _run(ref, cu, label="noise ")
    assert float(cu.edge_prior[:, 0].float().max()) > 0.0
    _report(probe, "noise")


def test_agrees_across_moves_and_through_the_temperature_switch():
    """Three moves, including §6.7's sampling and its argmax branch.

    `tau_plies = 1` puts move 0 on the multinomial and moves 1 and 2 on the
    argmax, so both halves of the temperature schedule are compared, and the
    training record and the game ring are compared with them.
    """
    boards, control = env.initial_boards(8, device="cuda")
    ref, cu, probe = _pair(boards, control, n=96, eps=0.25, tau_plies=1)
    _run(ref, cu, moves=3, label="moves ")
    assert int(ref.game_ply.max()) == 3
    _report(probe, "moves")


def test_agrees_on_random_positions():
    """Promotions, checks and captures, which the start position does not reach."""
    boards, control = _live(24, plies=60, seed=3)
    ref, cu, probe = _pair(boards, control, n=96, eps=0.25)
    _run(ref, cu, moves=3, label="random ")
    live = int(cu.node_count.max())
    promos = int(((cu.edge_move[:, :live].long() >> 11) != 0).sum())
    assert promos > 0, "no promotion edge, so §6.4's four-per-move fan-out is untested"
    _report(probe, f"random p60 ({promos} promotion edges)")


def test_agrees_deep_in_the_endgame():
    """Low material, which is the only place the terminal codes appear on their own.

    300 random plies leaves mostly bare kings and pawn endings, and the search then
    reaches checkmate, repetition and insufficient material inside the tree. It is
    also the only test here that exercises §8's other half in bulk, a descent that
    ends on an already-stored terminal and creates no node at all.
    """
    boards, control = _live(24, plies=300, seed=9)
    ref, cu, probe = _pair(boards, control, n=192, eps=0.25, stats=True)
    seen = set()
    for _ in range(3):
        _run(ref, cu, label="endgame ")
        live = int(cu.node_count.max())
        seen |= {int(c) for c in (cu.node_flags[:, :live].long() & TERMINAL).unique()}
    counters = cu.device_counters()
    assert 1 in seen, f"no checkmate anywhere in the trees, only codes {sorted(seen)}"
    assert counters["terminal_children"] > 0 and counters["terminal_descents"] > 0, \
        "no terminal was created and none was revisited, so §8 went untested"
    _report(probe, f"endgame p300 (terminal codes {sorted(seen)})")


# --------------------------------------------------------------------------- #
# The mechanisms that only fire on constructed positions
# --------------------------------------------------------------------------- #

def test_agrees_on_the_fifty_move_boundary():
    """Every child is drawn at clock 101, so the first ply of the tree is terminal."""
    boards, control = _from_fen(FIFTY_MOVE, copies=4)
    ref, cu, probe = _pair(boards, control, n=96, eps=0.0)
    _run(ref, cu, label="fifty-move ")
    live = int(cu.node_count.max())
    codes = cu.node_flags[:, 1:live].long() & TERMINAL
    assert bool((codes == 3).any()), "no node was drawn by the fifty-move rule"
    _report(probe, "fifty-move")


def test_agrees_on_a_threefold_inside_the_tree():
    """§7's ring half: a child whose hash is already twice in the game ring.

    The move chosen for the trap has to be reversible, because an irreversible one
    empties the window and the ring stops counting, which is the case §7 gets right
    and a naive count does not.
    """
    boards, control = _live(16, plies=200, seed=17)
    B = boards.shape[0]
    mask, _ = env.movegen(boards, control)
    legal = env.bitset_to_bool(mask).reshape(B, 32 * 64)

    ring = torch.zeros(B, 100, dtype=torch.int64, device="cuda")
    length = torch.zeros(B, dtype=torch.int64, device="cuda")
    hashes = env.hash_position(boards, control)
    for row in range(B):
        for move in legal[row].nonzero(as_tuple=True)[0].tolist():
            child, ctrl, h, irrev = env.step(
                boards[row:row + 1], control[row:row + 1],
                torch.tensor([move], device="cuda"), hash=hashes[row:row + 1])
            if bool(irrev[0]):
                continue
            ring[row, 0] = ring[row, 1] = h[0]
            length[row] = 2
            break

    keep = (length > 0).nonzero(as_tuple=True)[0]
    assert keep.numel() >= 4, "not enough positions with a reversible move"
    boards, control = boards[keep].contiguous(), control[keep].contiguous()
    ref, cu, probe = _pair(boards, control, n=64, eps=0.0)
    ref.game_ring, ref.game_ring_len = ring[keep].contiguous(), length[keep].contiguous()
    cu.game_ring, cu.game_ring_len = ring[keep].clone(), length[keep].clone()
    cu._refresh_tree()
    _run(ref, cu, label="threefold ")

    live = int(cu.node_count.max())
    codes = cu.node_flags[:, :live].long() & TERMINAL
    assert bool((codes == 4).any()), "no node was drawn by repetition, §7 went untested"
    _report(probe, "threefold")


def test_agrees_under_truncation():
    """§4.3: more than E candidates, so the kernel's radix select has to agree.

    The maximum-mobility position has no pawns, so every candidate's logit is a
    policy entry both implementations read identically and the threshold is exact
    rather than within a ULP. What is being compared is the selection of the E
    largest and the canonical order they are written in.
    """
    boards, control = _from_fen(MAX_MOBILITY, copies=4)
    mask, _ = env.movegen(boards, control)
    n_moves = int(env.bitset_to_bool(mask)[0].sum())
    cap = int(_ext().edge_cap())
    assert n_moves > cap, f"only {n_moves} legal moves, nothing to truncate at E = {cap}"

    ref, cu, probe = _pair(boards, control, n=96, eps=0.0, stats=True)
    _run(ref, cu, prior_ulps=0.0, label="truncation ")
    assert int(cu.node_nedges[0, 0]) == cap
    counters = cu.device_counters()
    assert counters["truncated_nodes"] > 0
    assert counters["max_edges"] >= n_moves
    assert counters["truncated_mass"] > 0.0
    _report(probe, f"truncation ({n_moves} legal moves at the root)")


# --------------------------------------------------------------------------- #
# The counters, the invariants and the guards
# --------------------------------------------------------------------------- #

def test_the_counters_match_the_reference_stats():
    """§15's device block against the reference's torch one, over the same run."""
    boards, control = _live(24, plies=140, seed=5)
    ref, cu, probe = _pair(boards, control, n=96, eps=0.25, stats=True)
    _run(ref, cu, moves=2, label="counters ")
    got = cu.device_counters()
    want = ref.stats.snapshot()
    # The mapping now lives in `cuda_impl` -- `train/loop.py` needs it too, and a
    # second copy is how the two blocks drift apart.
    from brokefish.search.cuda_impl import COUNTER_ALIASES
    for mine, theirs in COUNTER_ALIASES.items():
        assert got[mine] == want[theirs], \
            f"counter {mine} = {got[mine]}, reference {theirs} = {want[theirs]}"
    # And the merged view must carry no zero where the kernel measured something.
    merged = cu.stats_snapshot()
    for _device_name, canonical in COUNTER_ALIASES.items():
        assert merged[canonical] == want[canonical], (
            f"stats_snapshot left {canonical} at the unwritten torch value "
            f"{merged[canonical]} instead of the kernel's {want[canonical]}")
    assert got["pool_overflow"] == 0 and got["depth_overflow"] == 0


def test_the_invariants_of_section_5_hold_on_the_kernels_tree():
    from brokefish.search import check_invariants
    boards, control = _live(16, plies=80, seed=21)
    ref, cu, _ = _pair(boards, control, n=96, eps=0.25)
    _run(ref, cu, moves=2, label="invariants ")
    check_invariants(cu)


def test_a_rebound_tree_tensor_is_caught():
    """The bug this guard exists for: `push_history` returns a new ring.

    A dict of tensors built once and reused points at the ring that stopped being
    updated after move 0, and the symptom appears three moves and eight thousand
    simulations later as a repetition count that is too low.
    """
    boards, control = env.initial_boards(4, device="cuda")
    cu = search_impl("cuda")(SearchConfig(n=8, B=4), hash_eval, env=env, seed=0)
    cu.reset(boards, control)
    cu.root_init()
    cu.game_ring = cu.game_ring.clone()
    with pytest.raises(AssertionError, match="rebound"):
        cu.simulate(0)


def test_an_fp32_evaluator_is_refused():
    boards, control = env.initial_boards(4, device="cuda")

    def fp32(b, c, r):
        p, q, v = batched_eval(b, c, r)
        return p, q, v

    cu = search_impl("cuda")(SearchConfig(n=4, B=4), fp32, env=env, seed=0)
    cu.reset(boards, control)
    with pytest.raises(TypeError, match="fp16"):
        cu.root_init()


def test_a_different_edge_cap_is_refused():
    with pytest.raises(ValueError, match="compile-time"):
        search_impl("cuda")(SearchConfig(n=4, B=4, E=32), hash_eval, env=env)


def test_the_comparison_is_not_vacuous():
    """Every field the comparison covers, perturbed one at a time and caught.

    Without this the suite reads as thorough and could be comparing nothing: a
    field left out of `EXACT`, a tolerance wide enough to swallow anything, or a
    staged buffer nobody looks at.
    """
    boards, control = env.initial_boards(4, device="cuda")
    ref, cu, _ = _pair(boards, control, n=16, eps=0.0)
    ref.root_init()
    cu.root_init()
    for s in range(16):
        ref.simulate(s)
        cu.simulate(s)
    assert not _differences(ref, cu), "the run itself has to agree first"

    fields = list(EXACT) + ["edge_prior", "edge_Q", "node_value",
                            "leaf_node", "leaf_rep", "leaf_board", "leaf_control"]
    missed = []
    for f in fields:
        t = getattr(cu, f)
        saved = t.clone()
        flat = t.reshape(-1)
        # `+1` on the first entry moves node 0 or edge 0, which every field's
        # comparison reaches; a random index could land in the unallocated pool.
        flat[0] = flat[0] + (1 if t.dtype != torch.bool else True)
        if not _differences(ref, cu):
            missed.append(f)
        t.copy_(saved)
    assert not missed, f"perturbing these fields changed nothing: {missed}"
    assert not _differences(ref, cu), "the restore did not restore"


def test_every_tree_field_reaches_the_kernel():
    """The dict `csrc/search.cu` looks names up in covers every array of §4.2."""
    boards, control = env.initial_boards(2, device="cuda")
    cu = search_impl("cuda")(SearchConfig(n=4, B=2), hash_eval, env=env, seed=0)
    cu.reset(boards, control)
    for name in TREE_FIELDS:
        assert cu._tree[name].data_ptr() == getattr(cu, name).data_ptr()
    del cu._tree["edge_Q"]
    # `backup_only` is the entry point that does not refresh the dict first, so
    # the missing name reaches the binding's own lookup.
    with pytest.raises(Exception, match="edge_Q"):
        cu.backup_only()


@pytest.mark.slow
def test_agrees_at_a_self_play_batch():
    """`B = 1024`, two moves, which is the shape the gate is measured at.

    Everything else in this file runs at `B <= 64`, because the reference costs a
    torch launch per descent level and the comparison a synchronisation per
    simulation. That leaves the whole batch axis untested, and the batch axis is
    where an index that fits in 32 bits at `B = 8` stops fitting, where a grid is
    sized wrong, and where a race between blocks would first appear. It found the
    one thing in this file that a small batch never showed: a root prior two fp16
    steps apart, which is the double quantisation `_differences` now allows for.
    """
    boards, control = _live(256, plies=30, seed=41)
    idx = torch.arange(1024, device="cuda") % boards.shape[0]
    ref, cu, probe = _pair(boards[idx].contiguous(), control[idx].contiguous(),
                           n=48, eps=0.25, stats=True)
    _run(ref, cu, moves=2, label="B=1024 ")
    # The counters accumulate over the phase, so two moves at n = 48 over 1024
    # games is what a clean run has to add up to, exactly.
    counters = cu.device_counters()
    assert counters["simulations"] == 2 * 48 * 1024, counters["simulations"]
    assert counters["pool_overflow"] == 0 and counters["depth_overflow"] == 0
    _report(probe, "B=1024")


@pytest.mark.slow
def test_agrees_at_the_full_budget():
    """n = 800 and B = 64, the shape self-play runs at, one move.

    The shallow runs above cannot reach the depths where `N_v` is large, the
    logarithm in §6.6 has moved off `pb_c_init`, and the repetition window has a
    path long enough to need more than one lane's worth of scanning.
    """
    boards, control = _live(64, plies=40, seed=31)
    ref, cu, probe = _pair(boards, control, n=800, eps=0.25, stats=True)
    _run(ref, cu, label="full budget ")
    counters = cu.device_counters()
    # ⚠️ The bound is calibrated to the search's first-play urgency. On these 64
    # positions the reference reaches depth 13 with an untried edge scored as a loss
    # (`FPU_DRAW = 0`, the search this test was written on, 2026-07-30) and depth 8
    # with it scored as a draw (`FPU_DRAW = 0.5`, since 2026-08-02: the tree is wider
    # and shallower, 63 leaves at depth 7 and 2 at depth 8). The `> 8` this used to
    # assert was the old search's number, and this slow test had not run since.
    # Measured 2026-09-09, `docs/journal/2026-09-09-core-algorithm-review-fixes.md`.
    assert counters["max_depth"] >= 7, \
        f"the deepest path was {counters['max_depth']}, so this was not a deep run"
    _report(probe, f"n=800 (max depth {counters['max_depth']})")


# --------------------------------------------------------------------------- #

def _is_slow(fn) -> bool:
    return any(m.name == "slow" for m in getattr(fn, "pytestmark", []))


def _tests(slow: bool):
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v) and (slow or not _is_slow(v))]


def main() -> int:
    if not CUDA:
        print("no GPU; the CUDA search cannot run")
        return 0
    slow = "--slow" in sys.argv
    tests = _tests(slow)
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures += 1
            print(f"  {name:56s} FAIL {type(exc).__name__}: {exc}")
        else:
            print(f"  {name:56s} OK")
    print(f"\n{len(tests) - failures}/{len(tests)} passed"
          + ("" if slow else "   (--slow adds the n = 800 run)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# §6.6a, the terminal collapse
# --------------------------------------------------------------------------- #

def test_agrees_with_the_collapse_on_a_won_root():
    """The root is proved won, so every simulation is collapsed from the first.

    This is the branch the plain suite never enters: with the flag off `edge_win` is
    a null pointer in the kernel and an empty tensor in the reference, so nothing
    above exercises the ballot, the `-N` score or the seeded mask.
    """
    boards, control = _from_fen(MATE_IN_1, copies=2)
    ref, cu, probe = _pair(boards, control, n=96, eps=0.0, collapse=True)
    _run(ref, cu, moves=1, label="collapse, won root: ")
    assert int(ref.edge_win[0, 0].sum()) == 1, "the sweep proved no win on a mate in one"


def test_agrees_with_the_collapse_on_random_positions():
    """Most roots have no win, so this is the pass-through branch plus the interior.

    Several moves, so nodes are created below the root and `_create_child`'s half of
    the mask -- the half where the parity is written a second time -- is compared
    against the kernel's rather than only against the rules.
    """
    boards, control = _live(24, plies=60, seed=5)
    ref, cu, probe = _pair(boards, control, n=96, eps=0.25, collapse=True)
    _run(ref, cu, moves=3, label="collapse, random: ")
    _report(probe, "collapse, random")


def test_agrees_with_the_collapse_deep_in_the_endgame():
    """Where mates are dense, so the interior branch actually fires."""
    boards, control = _live(24, plies=160, seed=9)
    ref, cu, probe = _pair(boards, control, n=128, eps=0.25, collapse=True, stats=True)
    _run(ref, cu, moves=3, label="collapse, endgame: ")
    live = int(max(ref.node_count.max(), cu.node_count.max()))
    marked = int(ref.edge_win[:, :live].sum())
    print(f"    collapse, endgame: {marked} winning edges proved across the batch")


# --------------------------------------------------------------------------- #
# §11, Gumbel: the comparison that cannot be bit-exact, and what replaces it
# --------------------------------------------------------------------------- #

def _gumbel_pair(boards, control, n, seed=11, collapse=False, interior=True):
    """A reference and a kernel Gumbel search over the same games and the same noise.

    ⚠️ The Gumbel draw is **copied**, not reseeded. Both classes inherit
    `_draw_root_gumbel` and would consume the same generator stream only as long as
    the two `root_init` paths make the same number of calls — which is exactly the
    coupling `dirichlet`'s docstring warns is fragile. Copying makes the noise a
    shared input rather than a coincidence, so what this measures is the selection
    arithmetic and nothing else.
    """
    B = boards.shape[0]
    cfg = SearchConfig(n=n, B=B, E=int(_ext().edge_cap()), gumbel=True,
                       terminal_collapse=collapse, gumbel_interior=interior)
    # `Recording`, not the plain reference: `_differences` compares §6.3's staging
    # too, and only this subclass keeps what the reference handed the encoder.
    ref = Recording(cfg, hash_eval, env=env, device="cuda", seed=seed)
    cu = search_impl("cuda")(cfg, hash_eval, env=env, device="cuda", seed=seed)
    ref.reset(boards.clone(), control.clone())
    cu.reset(boards.clone(), control.clone())
    return ref, cu


def _run_gumbel(ref, cu, moves=1, label=""):
    """Both searches, simulation by simulation, reporting where they first part.

    ⚠️ **Bit-exactness is not available here and the kernel says why.** PUCT's only
    reduction is a sum of non-negative integers, exact in any order; Gumbel needs
    float sums — `sum_probs`, the softmax denominator — whose warp-shuffle tree
    differs from torch's, so the two differ in the last ulp. What makes that
    survivable is that PUCT's *ties are the common case* (every score at a fresh
    node is exactly 0, and the tie-break alone decides the first descent) while
    Gumbel's are measure-zero: continuous noise and distinct logits.

    So this asserts the trees are identical, and on a divergence prints the
    simulation it happened at rather than only that it did — a divergence at
    simulation 3 is a wrong formula, one at simulation 500 is an ulp.
    """
    n = ref.config.n
    for m in range(moves):
        ref.root_init()
        cu.root_init()
        cu.root_gumbel.copy_(ref.root_gumbel)
        for s in range(n):
            ref.simulate(s)
            cu.simulate(s)
            diffs = _differences(ref, cu)
            assert not diffs, (f"{label}diverged at move {m}, simulation {s} of {n}:"
                               f"\n  " + "\n  ".join(diffs[:6]))
        r1, r2 = ref.select_and_advance(), cu.select_and_advance()
        for f in ("board", "control", "rep", "policy_move", "policy_len", "played",
                  "ply", "done", "result"):
            x, y = getattr(r1, f), getattr(r2, f)
            assert bool((x == y).all()), f"{label}move {m}: the record's {f} differs"
        # ⚠️ The target and `root_value` are computed by the *inherited torch*
        # `select_and_advance` on each side's own tree, so they are equal exactly
        # when the trees are — a tolerance here would hide a tree difference the
        # loop above is meant to catch. fp16 storage is the only slack allowed.
        assert float((r1.policy_prob.float() - r2.policy_prob.float()).abs().max()) == 0.0
        assert float((r1.root_value - r2.root_value).abs().max()) == 0.0
        for f in ("game_board", "game_control", "game_hash", "game_ring",
                  "game_ring_len", "game_ply", "game_done", "game_result"):
            assert bool((getattr(ref, f) == getattr(cu, f)).all()), \
                f"{label}move {m}: {f} differs after the move"
        if bool(ref.game_done.any()):
            ref.reset_finished()
            cu.reset_finished()
    print(f"    {label}{moves} move(s) x {n} simulations: trees identical")


def test_gumbel_agrees_from_the_start_position():
    boards, control = env.initial_boards(8, device="cuda")
    ref, cu = _gumbel_pair(boards, control, n=192)
    _run_gumbel(ref, cu, label="gumbel startpos: ")


def test_gumbel_agrees_across_moves_on_random_positions():
    """Several moves and wide edge sets, which is where the halving phases differ.

    `m = min(16, nedges)` varies across the batch here, so every row is on a
    different schedule row of the table — the case a single-position test cannot
    reach, and the one an off-by-one in the schedule index would break.
    """
    boards, control = _live(24, plies=60, seed=5)
    ref, cu = _gumbel_pair(boards, control, n=96)
    _run_gumbel(ref, cu, moves=3, label="gumbel random: ")


def test_gumbel_agrees_with_the_root_rule_alone():
    """`gumbel_interior = False`: §6.6 below the root, Gumbel at it.

    The two rules are dispatched by depth in one kernel branch, so this is what
    says the branch is on `d == 0` and not on something correlated with it.
    """
    boards, control = _live(16, plies=40, seed=3)
    ref, cu = _gumbel_pair(boards, control, n=96, interior=False)
    _run_gumbel(ref, cu, moves=2, label="gumbel root-only: ")


def test_gumbel_agrees_with_the_collapse_deep_in_the_endgame():
    """§6.6a overriding both Gumbel rules, where mates are dense enough to fire."""
    boards, control = _live(24, plies=160, seed=9)
    ref, cu = _gumbel_pair(boards, control, n=128, collapse=True)
    _run_gumbel(ref, cu, moves=3, label="gumbel collapse: ")
    live = int(max(ref.node_count.max(), cu.node_count.max()))
    print(f"    gumbel collapse: {int(ref.edge_win[:, :live].sum())} winning edges proved")


def test_gumbel_agrees_with_the_collapse_on_a_won_root():
    """The collapse overriding the halving from the first simulation.

    ⚠️ Added because `test_gumbel_agrees_with_the_collapse_deep_in_the_endgame`
    proves **0** winning edges on positions where the PUCT run proves 2 — the two
    rules descend differently, so a test that only asserts agreement can leave the
    collapse branch of `gumbel_argmax` entirely dead and still pass. Here the root
    itself is proved won by §6.1a's sweep, so the ballot, the `-N` score and the
    seeded mask are all on the path, and the count below says so rather than
    assuming it.
    """
    boards, control = _from_fen(MATE_IN_1, copies=2)
    ref, cu = _gumbel_pair(boards, control, n=96, collapse=True)
    _run_gumbel(ref, cu, moves=1, label="gumbel won root: ")
    assert int(ref.edge_win[0, 0].sum()) == 1, "the sweep proved no win on a mate in one"
    # ⚠️ And no *visit* was seeded: the halving schedule indexes by `sum(N)`, so a
    # root starting at N = 1 would be out of phase for the whole move.
    assert int(ref.seeded.sum()) == 0
    assert int(cu.edge_N[0, 0].sum()) == 96
