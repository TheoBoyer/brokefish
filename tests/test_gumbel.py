"""Gumbel MuZero, against the paper and against `mctx`.

`search.md` §11's three seams at once, so there is no single oracle. The checks
come at it from five sides, in rough order of how much they would catch:

* **the Gumbel-top-k theorem**, which is exact — at `n = 1` the move played must
  be distributed *exactly* as `softmax(prior)`. This is the one test that catches
  a wrong noise scale, a missing `logits` term, or noise drawn per simulation
  instead of per move, and it needs no network and no chess.
* **the budget**, which sequential halving must spend to the visit — the failure
  mode the playout-cap bug of 2026-08-07 had, where every counter said it worked.
* **the halving schedule** against `mctx`'s published behaviour, at the level of
  the sequence rather than of the code.
* **policy improvement**, the paper's actual claim, on a bandit with known Q.
* **the invariants of §5**, which hold whatever the selection rule is, plus the
  properties of chess that no reimplementation of the algorithm can supply.

    python -m tests.test_gumbel
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.nn.model import BrokefishNet
from brokefish.search import Search, SearchConfig, check_invariants, make_evaluator
from brokefish.search import gumbel as G

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"

@pytest.fixture(autouse=True)
def _no_grad():
    """⚠️ `CLAUDE.md`'s fourth cross-cutting trap, and it bites here.

    The sampling tests want thousands of boards, and without this the reference
    network builds an autograd graph across every simulation of every round and
    OOMs the card at B = 512. Self-play and evaluation are `no_grad` for exactly
    this reason.

    ⚠️ A fixture and not a module-level `torch.set_grad_enabled(False)`, which is
    what this was first — that is process-global, survives the file, and made
    `test_muon.py::test_a_few_steps_actually_reduce_a_loss` fail from three files
    away with no gradients to step on. It only shows up in a whole-suite run.
    """
    with torch.no_grad():
        yield


def make(n=32, B=2, E=64, seed=0, boards=None, control=None, **kw):
    torch.manual_seed(seed)
    net = BrokefishNet().to(DEVICE).eval()
    cfg = SearchConfig(n=n, B=B, E=E, gumbel=True, **kw)
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


# -- 1. the theorem -------------------------------------------------------- #

def test_the_top_k_trick_samples_the_prior_exactly():
    """`argmax(g + logits)` is a draw from `softmax(logits)`. Exactly, not nearly.

    The whole construction rests on this identity, and it is the only place a
    wrong `gumbel_scale`, a dropped `logits` term or noise redrawn per simulation
    would show up as a *distributional* error rather than as a slightly worse
    search. Chi-square against the analytic prior, no network and no tree.
    """
    torch.manual_seed(0)
    K, N = 6, 200_000
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, -3.0], device=DEVICE)
    want = torch.softmax(logits, -1)

    u = torch.rand((N, K), device=DEVICE).clamp(min=1e-30, max=1 - 2 ** -24)
    g = -torch.log(-torch.log(u))
    pick = (g + logits).argmax(-1)
    got = torch.bincount(pick, minlength=K).float() / N

    chi2 = float((((got - want) ** 2) / want).sum() * N)
    # 5 degrees of freedom, upper 0.1 % point is 20.5.
    assert chi2 < 20.5, f"chi2 = {chi2:.1f}, got {got.tolist()} want {want.tolist()}"


def test_at_one_simulation_the_played_move_is_a_sample_from_the_prior():
    """The same identity, end to end through the search on a real position.

    At `n = 1` sequential halving has nothing to allocate, so Gumbel degenerates
    to sampling the prior — which is already a policy improvement over the greedy
    move, and is the paper's `n = 1` corner. Measured against the network's *own*
    root prior, so this also pins that `edge_logits` inverts `_expand`'s softmax.
    """
    # B x rounds rather than one big batch: the reference network is plain torch
    # attention and 4096 boards of it does not fit in 8 GB beside the tree.
    B, ROUNDS = 512, 16
    s = make(n=1, B=B, E=64, seed=3)
    boards, control = env.initial_boards(B, device=DEVICE)
    s.root_init(noise=True)
    prior = s.edge_prior[:, 0].float()
    ne = int(s.node_nedges[0, 0])
    assert bool((s.node_nedges[:, 0] == ne).all()), "one position, one edge count"
    labels = s.edge_move[0, 0, :ne].to(torch.int64)

    counts = torch.zeros(ne, device=DEVICE)
    for _ in range(ROUNDS):
        s.reset(boards, control)
        played = s.self_play_move(sims=1).played.to(torch.int64)
        idx = (played[:, None] == labels[None, :]).float().argmax(-1)
        counts += torch.bincount(idx, minlength=ne).float()
    B = B * ROUNDS
    got = counts / B

    want = prior[0, :ne]
    keep = want > 3.0 / B          # cells the sample can say anything about
    chi2 = float(((got[keep] - want[keep]) ** 2 / want[keep]).sum() * B)
    dof = int(keep.sum()) - 1
    # 3-sigma on chi2 with `dof` degrees of freedom.
    assert chi2 < dof + 3.0 * math.sqrt(2.0 * dof), f"chi2 = {chi2:.1f}, dof = {dof}"


def test_a_zero_scale_is_deterministic_and_a_nonzero_one_is_not():
    """`gumbel_scale` is the evaluation knob, so it has to actually switch.

    ⚠️ This is the property `eval/match.py` depends on: evaluation takes all of
    its diversity from the random openings and none from the search, so a league
    run at scale 1 would be rating the noise.
    """
    # ⚠️ **One network across every arm**, built once. Varying the search seed
    # through `make` would also reseed the network, and then scale 0 would look
    # non-deterministic for a reason that has nothing to do with Gumbel.
    torch.manual_seed(0)
    net = BrokefishNet().to(DEVICE).eval()
    evaluate = make_evaluator(net)
    boards, control = env.initial_boards(8, device=DEVICE)

    for scale, want_same in ((0.0, True), (1.0, False)):
        picks = []
        for seed in (1, 2, 3):
            cfg = SearchConfig(n=16, B=8, E=64, gumbel=True, gumbel_scale=scale)
            s = Search(cfg, evaluate=evaluate, device=DEVICE, seed=seed)
            s.reset(boards, control)
            picks.append(s.self_play_move().played.clone())
        same = all(bool((p == picks[0]).all()) for p in picks[1:])
        assert same == want_same, f"scale {scale}: identical across seeds = {same}"


# -- 2. the budget --------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 17, 64, 100])
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 16])
def test_the_schedule_spends_exactly_the_budget(n, m):
    """Every simulation goes somewhere, and no visit count is ever skipped.

    The schedule is a table of *required visit counts*; if the count required at
    step `s` is held by no surviving edge, that simulation has no legal target and
    the budget is silently misspent. Replaying the schedule against a running
    visit vector is the direct test of that.
    """
    seq = G.considered_visit_sequence(m, n)
    assert len(seq) == n
    visits = [0] * m
    for s, want in enumerate(seq):
        matching = [i for i in range(m) if visits[i] == want]
        assert matching, f"m={m} n={n} step {s} wants N={want}, held by nobody"
        visits[matching[0]] += 1
    assert sum(visits) == n


@pytest.mark.parametrize("n", [4, 16, 64])
def test_the_root_visits_sum_to_the_budget(n):
    """§5 invariant 6, which the seeded terminal edges no longer perturb.

    ⚠️ Under Gumbel `_seed_terminal_edges` writes `edge_win` and *not* `edge_N`:
    an edge starting a move at `N = 1` while its rivals start at 0 is out of phase
    with the halving table for the whole move. `seeded` must therefore be zero.
    """
    s = make(n=n, B=4, E=64, seed=1, terminal_collapse=True)
    s.self_play_move()
    assert int(s.seeded.sum()) == 0
    ne = s.node_nedges[:, 0].long()
    valid = torch.arange(s.config.E, device=DEVICE)[None, :] < ne[:, None]
    total = torch.where(valid, s.edge_N[:, 0], torch.zeros_like(s.edge_N[:, 0])).sum(-1)
    assert bool((total == n).all()), total.tolist()


def test_the_halving_actually_halves():
    """The final phase concentrates the budget, which is the point of the method.

    With `m = 16` at `n = 64`, plain round-robin would give 4 visits to each of 16
    edges. Sequential halving must instead leave a small set carrying most of the
    visits — otherwise it is an expensive way to be uniform.
    """
    seq = G.considered_visit_sequence(16, 64)
    visits = [0] * 16
    for want in seq:
        visits[next(i for i in range(16) if visits[i] == want)] += 1
    visits.sort(reverse=True)
    assert visits[0] >= 4 * visits[8], f"top {visits[0]}, median {visits[8]}"
    assert sum(visits) == 64


# -- 3. the Q transform ---------------------------------------------------- #

def test_the_completion_value_is_the_papers_v_mix():
    """`_compute_mixed_value` on hand arithmetic, not on a tolerance.

    With one visited edge the formula collapses to `(v + N q) / (N + 1)`, the
    prior weights cancelling — which is the corner where a misplaced `sum_probs`
    would not show up in a random test.
    """
    q = torch.tensor([[0.8, 0.0, 0.0]], device=DEVICE)
    nvis = torch.tensor([[3.0, 0.0, 0.0]], device=DEVICE)
    valid = torch.ones((1, 3), dtype=torch.bool, device=DEVICE)
    prior = torch.tensor([[0.2, 0.5, 0.3]], device=DEVICE)
    value = torch.tensor([0.4], device=DEVICE)

    _, completed = G.completed_q(q, nvis, valid, value, prior)
    v_mix = (0.4 + 3.0 * 0.8) / 4.0
    assert abs(float(completed[0, 0]) - 0.8) < 1e-6
    assert abs(float(completed[0, 1]) - v_mix) < 1e-6
    assert abs(float(completed[0, 2]) - v_mix) < 1e-6


def test_with_no_visits_the_completion_is_the_raw_value():
    """Before the first simulation `pi'` must be the prior, exactly.

    `sum_visits = 0` makes `v_mix = v`, every completed Q equal, the min-max
    rescale degenerate, and `sigma` therefore constant — so `softmax(logits +
    sigma)` is `softmax(logits)`. If it is not, the search starts from a
    distribution the network never produced.
    """
    s = make(n=8, B=2, E=64, seed=5)
    s.root_init(noise=True)
    root = torch.zeros(2, dtype=torch.long, device=DEVICE)
    sigma, completed, logits, nvis, valid = s._gumbel_completed(root)
    assert float(nvis.sum()) == 0.0
    pi = G.improved_policy(logits, sigma, valid)
    prior = s.edge_prior[:, 0].float()
    assert float((pi - prior).abs().max()) < 2e-3, float((pi - prior).abs().max())


def test_an_unreachable_prior_does_not_become_an_impossible_move():
    """fp16 `edge_prior` underflows to a hard zero, and `log(0)` is `-inf`.

    A `-inf` logit on a *legal* move is the absorbing state that cost this project
    a seven-hour run (`FPU_DRAW`'s comment): the move can never be sampled, so its
    prior can never be corrected upward. The clamp in `edge_logits` is what stops
    it, and this is the test that it is there.
    """
    prior = torch.tensor([[0.5, 0.5, 0.0]], device=DEVICE, dtype=torch.float16).float()
    valid = torch.ones((1, 3), dtype=torch.bool, device=DEVICE)
    lg = G.edge_logits(prior, valid)
    assert torch.isfinite(lg).all(), lg.tolist()
    assert float(lg[0, 2]) < float(lg[0, 0])


# -- 4. the claim ---------------------------------------------------------- #

def test_the_search_improves_on_the_prior():
    """The paper's actual guarantee, on a bandit where the answer is known.

    Given a prior that puts most of its mass on a bad arm and a Q that says so,
    the move played must beat a draw from the prior in expected Q. This holds for
    *any* budget in the paper; measured here at `n = 4`, which is far below where
    a visit-count target means anything at all.
    """
    torch.manual_seed(0)
    B, K, n = 20_000, 8, 4
    logits = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], device=DEVICE)
    q = torch.tensor([0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9], device=DEVICE)
    prior = torch.softmax(logits, -1)
    valid = torch.ones((B, K), dtype=torch.bool, device=DEVICE)

    nvis = torch.zeros((B, K), device=DEVICE)
    u = torch.rand((B, K), device=DEVICE).clamp(min=1e-30, max=1 - 2 ** -24)
    gum = -torch.log(-torch.log(u))
    table = G.visit_table(K, n, DEVICE)
    lg = logits.expand(B, K).contiguous()
    for s in range(n):
        sigma, _ = G.completed_q(q.expand(B, K), nvis, valid,
                                 torch.full((B,), 0.5, device=DEVICE),
                                 prior.expand(B, K), )
        considered = table[K, s].to(nvis.dtype).expand(B, 1)
        pick = G.root_scores(gum, lg, sigma, nvis, considered, valid).argmax(-1)
        nvis[torch.arange(B, device=DEVICE), pick] += 1.0

    sigma, _ = G.completed_q(q.expand(B, K), nvis, valid,
                             torch.full((B,), 0.5, device=DEVICE), prior.expand(B, K))
    considered = nvis.max(-1, keepdim=True).values
    played = G.root_scores(gum, lg, sigma, nvis, considered, valid).argmax(-1)

    got = float(q[played].mean())
    base = float((prior * q).sum())
    assert got > base + 0.05, f"played Q {got:.3f} vs prior Q {base:.3f}"


def test_the_target_is_dense_where_the_visits_are_not():
    """§6.7 under Gumbel: `pi` stops being the visit histogram.

    At `n = 16` over a full opening move list, most edges get no visit at all, and
    `N / n` would hand them a flat zero — the same absorbing state §6.1a exists to
    break, one level up. The completed-Q target has to put mass on every legal
    move, and it has to still be a distribution.
    """
    s = make(n=16, B=4, E=64, seed=2)
    rec = s.self_play_move()
    ne = rec.policy_len.long()
    probs = rec.policy_prob.float()
    idx = torch.arange(probs.shape[1], device=DEVICE)[None, :]
    valid = idx < ne[:, None]

    mass = torch.where(valid, probs, torch.zeros_like(probs)).sum(-1)
    assert float((mass - 1.0).abs().max()) < 3e-3, mass.tolist()
    assert not bool((torch.where(valid, probs, torch.ones_like(probs)) <= 0).any())

    visited = torch.where(valid, s.edge_N[:, 0] > 0, torch.zeros_like(valid))
    assert int(visited.sum()) < int(valid.sum()), "no unvisited edge, test says nothing"


# -- 5. the search is still a search --------------------------------------- #

def test_the_invariants_hold_under_gumbel():
    """§5, which is about the tree and not about the selection rule."""
    s = make(n=32, B=4, E=64, seed=7)
    for _ in range(3):
        s.reset_finished()
        s.self_play_move()
        check_invariants(s)


def test_it_still_finds_mate_in_one():
    """A property of chess, and the one §6.1a and §6.6a were built for.

    Run with the collapse on, which is how a Gumbel run would be launched: the
    depth-1 sweep proves the mate before any simulation, the collapse sends the
    budget onto it, and `select_and_advance` makes the target a point mass. That
    path has to survive the target no longer being `N / n`.
    """
    b, c = from_fen(MATE_IN_1)
    s = make(n=16, B=4, E=64, seed=0, boards=b, control=c, terminal_collapse=True)
    rec = s.self_play_move()
    assert bool(rec.done.all()), "mate in one was not played"
    # §10: `result` is from the *new* mover's point of view, so the side that was
    # just mated reports -1. `test_search.test_finds_mate_in_one` needs n = 800 for
    # the same position under PUCT with the sweep and the collapse off; here it is
    # 16, which is the collapse doing its job and not Gumbel doing it.
    assert bool((rec.result == -1).all()), rec.result.tolist()
    top = rec.policy_prob.float().max(-1).values
    assert float(top.min()) > 0.99, f"target is not a point mass: {top.tolist()}"


def test_a_won_root_reports_a_won_value():
    """`root_value` under the collapse, which no seeded `edge_Q` carries any more.

    ⚠️ The trap this pins: under PUCT the value came from the seeded `edge_Q = 1.0`
    on the mating edge. Gumbel does not seed, so the completion would have given
    the root `v_mix` — the untrained network's own guess — on a position that is
    proved won.
    """
    b, c = from_fen(MATE_IN_1)
    s = make(n=16, B=4, E=64, seed=0, boards=b, control=c, terminal_collapse=True)
    rec = s.self_play_move()
    assert float(rec.root_value.min()) > 0.999, rec.root_value.tolist()


def test_the_interior_rule_can_be_turned_off_alone():
    """`gumbel_interior = False` is root-only Gumbel, which must still be a search."""
    s = make(n=32, B=4, E=64, seed=4, gumbel_interior=False)
    s.self_play_move()
    check_invariants(s)


def test_the_cuda_search_refuses_rather_than_running_puct():
    """⚠️ The failure this would otherwise be: a PUCT tree read as a Gumbel one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from brokefish.search import search_impl
    torch.manual_seed(0)
    net = BrokefishNet().to("cuda").eval()
    with pytest.raises(NotImplementedError, match="descent_kernel"):
        search_impl("cuda")(SearchConfig(n=8, B=2, E=96, gumbel=True),
                            evaluate=make_evaluator(net), device="cuda")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
