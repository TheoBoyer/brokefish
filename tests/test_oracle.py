"""The reference search against the independent oracle of `tests/oracle.py`.

This is the check `docs/mcts.md` §12 said was missing: everything else in
`test_search.py` checks the search by construction or by parts, and none of it
looks at the trajectory, which is where a composition bug would live. Here two
implementations that share no search code run the same `n` simulations from the
same position on the same numbers, and every node and every edge of the two trees
has to agree.

The oracle is built the opposite way round on purpose (`tests/oracle.py` lists the
four ways), so an agreement is evidence rather than a tautology. In particular it
stores values per node and flips at read time where the reference stores per edge
and flips at write time, which is what makes the comparison a real check on the
parity of §6.5.

    python -m tests.test_oracle
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.search import Search, SearchConfig, check_invariants
from tests.oracle import AZSearch, ScalarEval, batched_eval, deltas, differences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# One per behaviour the search has to get right, so a divergence points somewhere.
POSITIONS = {
    "startpos": None,
    "black to move": "4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
    "mate available": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
    "every child terminal": "4k3/8/8/8/8/8/8/4K2R w - - 99 1",
    "promotion available": "8/P7/8/8/8/8/8/K6k w - - 0 1",
    "endgame, few moves": "7k/8/8/8/8/8/8/R6K w - - 20 1",
    "midgame, many moves": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
}
# Two trees agreeing means nothing unless a *disagreement* would also have been
# meaningful, and it would not be if the two implementations' scores were closer
# together than their arithmetic error. So every run measures both: the largest `Q`
# disagreement (priors and node values come out bit-identical, so `Q` is the only
# term that can move a score) and the smallest gap between the best and second-best
# score at any selection. The gap has to clear the error by this factor.
MARGIN_FACTOR = 20.0


def _boards():
    out = []
    for fen in POSITIONS.values():
        if fen is None:
            b, c = env.initial_boards(1, device=DEVICE)
        else:
            b, c = env.from_fen(fen)
            b, c = b.to(DEVICE), c.to(DEVICE)
        out.append((b, c))
    return out


def _run_both(n=64, E=64, eps=0.0, noise=None, rings=None):
    """The reference over the whole batch at once, the oracle one game at a time."""
    pairs = _boards()
    B = len(pairs)
    boards = torch.cat([b for b, _ in pairs])
    control = torch.cat([c for _, c in pairs])

    # ⚠️ §6.1a's root terminal sweep is OFF here, and must stay off. This suite's
    # whole value is that `tests/oracle.py` is an *independent transcription of
    # AGZ* -- the strongest evidence in the project that our search is a correct
    # reproduction (`journal/2026-07-30-fidelity.md`). The sweep is a documented
    # deviation with no AlphaZero analogue, so comparing a swept search against
    # the oracle would only ever prove that the deviation exists.
    cfg = SearchConfig(n=n, B=B, E=E, eps=eps, tau_plies=0,
                       root_terminal_sweep=False)
    search = Search(cfg, evaluate=batched_eval, device=DEVICE, seed=0)
    if noise is not None:
        search.dirichlet = lambda valid: torch.where(
            valid, torch.tensor(noise, device=DEVICE)[: valid.shape[1]].expand_as(valid),
            torch.zeros(valid.shape, device=DEVICE))
    search.reset(boards, control)
    if rings:
        for row, ring in rings.items():
            for i, h in enumerate(ring):
                search.game_ring[row, i] = h
            search.game_ring_len[row] = len(ring)
    with torch.no_grad():
        search.root_init()
        for s in range(n):
            search.simulate(s)

    oracles = []
    for row, (b, c) in enumerate(pairs):
        ring = (rings or {}).get(row, ())
        oracle = AZSearch(b, c, ring=ring, n=n, E=E, eps=eps, noise=noise)
        oracle.run()
        oracles.append(oracle)
    return search, oracles


def _assert_agree(search, oracles, atol_prior=0.0, prior_ulps=1.0, strict_margin=True):
    """Every tree compared, and every comparison shown to have been able to fail.

    `strict_margin` is the guard that makes an agreement attributable: no selection
    may have been closer than the arithmetic error, or a divergence could have come
    from rounding rather than from the algorithm. It holds comfortably at 64
    simulations and stops holding around 512, where the deep test reports the count
    of sub-floor selections instead of demanding zero.
    """
    failures = []
    for row, (name, oracle) in enumerate(zip(POSITIONS, oracles)):
        margin = oracle.min_margin()
        _, q_delta, _ = deltas(search, row, oracle)
        floor = MARGIN_FACTOR * max(q_delta, 1e-12)
        if strict_margin and margin <= floor:
            failures.append(f"{name}: the closest selection was {margin:.2e} apart while "
                            f"Q disagrees by up to {q_delta:.2e}, so a divergence here "
                            f"would not be attributable to the algorithm")
        diffs = differences(search, row, oracle, atol_prior=atol_prior,
                            prior_ulps=prior_ulps)
        if diffs:
            failures.append(f"{name} ({int(search.node_count[row])} nodes):\n    "
                            + "\n    ".join(diffs[:8]))
    assert not failures, "\n  " + "\n  ".join(failures)


# --------------------------------------------------------------------------- #
# The shared evaluator has to be identical on both paths or nothing below means
# anything
# --------------------------------------------------------------------------- #

def test_the_two_evaluators_are_bit_identical():
    """Every one of the 2177 numbers, over positions the rules actually produce."""
    from tests.boards import random_positions

    boards, control, rep = random_positions(6, plies=9, seed=2, device=DEVICE)
    policy, promo, value = batched_eval(boards, control, rep)
    for row in range(boards.shape[0]):
        ev = ScalarEval(boards[row].tolist(), int(control[row]), int(rep[row]))
        assert float(value[row]) == ev.value
        want_p = [ev.policy(p, s) for p in range(32) for s in range(64)]
        assert policy[row].reshape(-1).tolist() == want_p
        want_q = [ev.promo(p, k) for p in range(32) for k in range(4)]
        assert promo[row].reshape(-1).tolist() == want_q


def test_the_evaluator_separates_positions():
    """A constant evaluator would make the comparison trivial. This one is not."""
    from tests.boards import random_positions

    boards, control, rep = random_positions(64, plies=11, seed=3, device=DEVICE)
    _, _, value = batched_eval(boards, control, rep)
    assert len(set(value.tolist())) > 50, "the evaluator is nearly constant"
    # And it must see `rep`, or the in-tree repetition count is never tested. Every
    # row's `rep` changes here, so every row's value has to.
    _, _, other = batched_eval(boards, control, rep + 1)
    assert int((value != other).sum()) == 64, "the evaluator ignores rep"


# --------------------------------------------------------------------------- #
# The differential test
# --------------------------------------------------------------------------- #

def test_oracle_agrees_tree_for_tree():
    """Seven positions, 64 simulations each, every node and every edge compared."""
    search, oracles = _run_both(n=64)
    _assert_agree(search, oracles)
    check_invariants(search)
    total = int(search.node_count.sum())
    assert total > 7 * 20, f"only {total} nodes across the batch, too little to mean much"


def test_oracle_agrees_on_the_ring_half_of_the_window():
    """§7's ring half, reaching into the tree.

    Row 5's first legal move is a reversible rook move; its result is loaded into the
    ring twice, so the node that move creates is a third occurrence and has to carry
    code 4 the moment it appears. That makes the case deterministic instead of waiting
    for the search to walk into a repetition.
    """
    from tests.oracle import walk

    b, c = _boards()[5]
    mask, _ = env.movegen(b, c)
    move = int(env.bitset_to_bool(mask).reshape(-1).nonzero()[0])
    _, _, child_hash, irreversible = env.step(
        b, c, torch.tensor([move], device=DEVICE), hash=env.hash_position(b, c))
    assert not bool(irreversible[0]), "the move has to be reversible or the ring is cleared"

    search, oracles = _run_both(n=64, rings={5: [int(child_hash[0])] * 2})
    _assert_agree(search, oracles)
    # More than one node can carry it: two different rook moves can transpose into
    # the same position, and each is its own node in a tree.
    hits = [x.rep for x in walk(oracles[5].root) if x.code == env.REPETITION]
    assert hits and all(r == 3 for r in hits), f"expected threefolds at rep 3, got {hits}"


def test_oracle_agrees_on_the_in_tree_half_of_the_window():
    """§7's tree half, with an empty ring, so the count can only come from the path.

    The endgame position returns to an earlier tree position after four plies, which
    the search reaches around 190 simulations. The evaluator reads `rep`, so a wrong
    in-tree count changes that node's logits and the two trees diverge below it.
    """
    from tests.oracle import walk

    search, oracles = _run_both(n=192)
    _assert_agree(search, oracles)
    recurring = [x.rep for x in walk(oracles[5].root) if x.rep > 1]
    assert recurring, "no position recurred inside any tree, so the tree half is untested"
    assert int(search.game_ring_len[5]) == 0, "the ring must be empty for this to mean anything"


def test_oracle_agrees_under_truncation():
    """`E` below the branching factor, so §6.4 step 3's ordering is compared too."""
    search, oracles = _run_both(n=64, E=6)
    _assert_agree(search, oracles)
    assert int(search.node_nedges.max()) == 6
    assert search.stats.snapshot()["n_truncated"] > 0


def test_oracle_agrees_with_root_noise():
    """§6.1's mixture, with the same `eta` handed to both implementations.

    The Dirichlet *stream* is not reproducible across the two (`docs/mcts.md` §12),
    so the draw is replaced by a fixed vector on both sides. What is compared is the
    mixing and everything downstream of it.
    """
    eta = [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02] + [0.0] * 57
    assert abs(sum(eta) - 1.0) < 1e-12
    search, oracles = _run_both(n=64, eps=0.25, noise=eta)
    # The mixture adds a second rounding on top of the softmax's, so allow one more
    # fp16 step of disagreement in the priors than the noiseless runs need.
    _assert_agree(search, oracles, prior_ulps=2.0)


@pytest.mark.slow
def test_oracle_agrees_deeper():
    """The same at 512 simulations, where trees pass depth 10.

    ⚠️ This is a weaker check than the 64-simulation one and the reason is worth
    knowing. With eight times the selections, the smallest gap between the best and
    second-best score falls to about 1e-7, which is the size of the disagreement in
    `Q` between an fp32 running mean and an fp64 mean. Past that point the two
    implementations *could* diverge from rounding alone, so agreement stops being
    attributable and the guard is reported rather than asserted. The trees still have
    to match, which is the part that carries information.
    """
    search, oracles = _run_both(n=512)
    _assert_agree(search, oracles, strict_margin=False)
    inside = []
    for row, oracle in enumerate(oracles):
        _, q_delta, _ = deltas(search, row, oracle)
        floor = MARGIN_FACTOR * max(q_delta, 1e-12)
        inside.append(sum(1 for m in oracle.margins if m <= floor))
    total = sum(len(o.margins) for o in oracles)
    print(f"    {sum(inside)} of {total} selections inside the precision floor: {inside}")
    assert sum(inside) < 0.001 * total, (
        f"{sum(inside)} of {total} selections could have been decided by rounding, "
        "which is too many for the agreement above to mean much")
    assert search.stats.snapshot()["max_depth"] >= 10


# --------------------------------------------------------------------------- #
# The harness itself
# --------------------------------------------------------------------------- #

def test_the_comparison_is_not_vacuous():
    """A comparison that cannot fail proves nothing, so make it fail on purpose.

    Every field the comparison claims to check is perturbed in turn, one at a time,
    and each perturbation has to be reported.
    """
    search, oracles = _run_both(n=32)
    _assert_agree(search, oracles)

    fields = ["node_board", "node_control", "node_hash", "node_flags", "node_value",
              "node_nedges", "node_parent", "node_pedge",
              "edge_move", "edge_prior", "edge_N", "edge_Q", "edge_child"]
    missed = []
    for name in fields:
        tensor = getattr(search, name)
        saved = tensor[0].clone()
        index = (0, 1, 0) if tensor.dim() == 3 else ((0, 1) if tensor.dim() == 2 else (0,))
        if name == "node_board":
            index = (0, 1, 5)
        tensor[index] = saved.new_tensor(0) if float(tensor[index]) != 0 else \
            saved.new_tensor(7)
        if not differences(search, 0, oracles[0]):
            missed.append(name)
        tensor[0] = saved
    assert not missed, f"the comparison ignores {missed}"

    # And the node count, which is checked separately from the fields.
    saved = int(search.node_count[0])
    search.node_count[0] = saved - 1
    assert differences(search, 0, oracles[0]), "the comparison ignores the node count"
    search.node_count[0] = saved
    assert not differences(search, 0, oracles[0])


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
            print(f"  {name:44s} FAIL {type(exc).__name__}: {exc}")
        else:
            print(f"  {name:44s} OK")
    print(f"\n{len(tests) - failures}/{len(tests)} passed"
          + ("" if slow else "   (--slow adds the 512-simulation run)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_the_two_fpu_constants_agree():
    """The oracle restates `FPU_DRAW` rather than importing it — so pin them here.

    ⚠️ `tests/oracle.py` is only worth having because it is an *independent*
    transcription: if it imported the reference's constants the comparison would
    agree by construction. The cost of that independence is that a change to one
    can silently walk away from the other, which is exactly what happened when the
    reference moved to 0.5 and this file stayed at AGZ's literal 0 — the suite went
    red and stayed red, and `journal/2026-07-30-fidelity.md` went on citing it as
    ~99 % evidence that the reference implements `search.md`. This test is the
    cheap thing that would have caught it.
    """
    from brokefish.search.torch_impl import FPU_DRAW as reference
    from tests.oracle import FPU_DRAW as transcription

    assert transcription == reference == 0.5, (
        "the reference and the AGZ transcription must score an untried move the "
        "same way; 0.5 is a draw in the [0,1] frame both of them use (§3.5)")
