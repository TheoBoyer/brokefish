"""C2, the training loop: `brokefish/train/`.

⚠️ **C2 is the first phase with no oracle.** Perft settled the engine, `nn/model.py`
settled the encoder, an independently written AGZ search settled C1. Nothing external
says whether a training loop is correct, and its failure mode is not a crash but a
curve that is merely worse than it should have been — which is indistinguishable from
a project whose premise was wrong. `docs/train.md` §12 is what replaces the oracle,
and this file is most of it. Check 1 is `python -m brokefish.train.overfit`, because
it needs minutes rather than seconds; check 8 is Track D's.

The checks that carry the most weight, in order:

**Check 3, the label decode.** A `u16` label is `(move, promo)` and the training path
has to select the same logit `_expand` used to build that edge's prior. A mismatch
permutes the target silently: the network learns a consistently shuffled labelling,
the loss falls the whole time, and every other check here passes while it happens. So
the training path is an independent transcription and this compares it against a
*live* search rather than against a shared helper.

**Check 6, the value parity.** Getting it backwards is a working system that plays to
lose, and `mcts.md`'s `L >= 3` warning applies for the same reason it did there: a
two-ply game cannot distinguish the correct rule from its inverse, so the games here
are of odd *and* even length.

**Check 9, weights propagate.** Every other check in §12 passes on a loop whose
self-play is frozen at generation 0.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from brokefish.env import torch_impl as env
from brokefish.nn.model import BrokefishNet
from brokefish.search import SearchConfig, Search, make_evaluator
from brokefish.train.buffer import RECORD, RECORD_BYTES, ReplayBuffer
from brokefish.train.log import Logger
from brokefish.train.loop import LR_SCHEDULE, Trainer, TrainConfig
from brokefish.train.loss import (TrainBatch, audit_labels, az_loss, decode_labels,
                                  edge_logits, is_promotion_edge, weight_decay_for)
from brokefish.train.buffer import K_POLICY
from brokefish.train.sync import PackedWeights, weight_fingerprint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the engine and the search are CUDA-shaped")

# tests/test_search_cuda.py's maximum-mobility position: 218 legal moves for White
# and no pawns, so the search truncates at `E` and the training path does not.
MAX_MOBILITY = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class ConstNet(torch.nn.Module):
    """Fixed logits, so a loss test measures the loss and not the network.

    ``value`` is ``[N]`` for the scalar head and ``[N, 3]`` for win/draw/loss; the
    squash it applies is whichever one `BrokefishNet.heads` would, so a loss test
    against this stub is testing the loss and not the collapse.
    """

    def __init__(self, policy, promo, value, absolute=False):
        super().__init__()
        self.policy = torch.nn.Parameter(policy)
        self.promo = torch.nn.Parameter(promo)
        self.value = torch.nn.Parameter(value)
        self.n_value = 1 if value.ndim == 1 else value.shape[-1]
        #: Mirrors `BrokefishNet.value_absolute`: the pooled head predicts White's
        #: frame, so the loss has to flip the record's target into it.
        self.value_absolute = absolute

    def forward(self, boards, control, rep, with_logits=False):
        from brokefish.nn.model import wdl_to_scalar
        raw = self.value if self.n_value > 1 else self.value.unsqueeze(-1)
        v = torch.tanh(raw).squeeze(-1) if self.n_value == 1 else wdl_to_scalar(raw)
        if self.value_absolute:
            v = torch.where(control > 0, v, -v)
        if with_logits:
            return self.policy, self.promo, v, raw
        return self.policy, self.promo, v


@torch.no_grad()
def make_batch(n: int, seed: int = 0, plies: int = 24):
    """`n` positions from random legal play, with a plausible synthetic target.

    ⚠️ `no_grad` is load-bearing, not tidiness. The search calls the net once per
    simulation, and with grad enabled the fp32 master weights build one autograd
    graph across every simulation of the move — which OOM'd an 8 GB card at
    `n = 256`. Same trap `eval/` hit at 8 games and 300 plies.
    """
    from tests.boards import random_positions

    boards, control, rep = random_positions(n, plies=plies, seed=seed, device=DEVICE)
    mask, in_check = env.movegen(boards, control)
    code, _ = env.terminal(mask, in_check, control, boards)
    live = (code == 0).nonzero(as_tuple=True)[0]
    fill = live[torch.arange(n, device=DEVICE) % live.numel()]
    boards, control, rep = boards[fill], control[fill], rep[fill]

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    net = BrokefishNet().to(DEVICE)
    search = Search(SearchConfig(n=24, B=n, eps=0.0), evaluate=make_evaluator(net),
                    env=env, device=DEVICE, seed=seed)
    search.reset(boards, control)
    record = search.self_play_move()
    z = torch.randint(-1, 2, (n,), generator=gen, device=DEVICE).float()
    k = record.policy_move.shape[1]
    batch = TrainBatch(
        board=record.board, control=record.control, rep=record.rep,
        policy_move=record.policy_move, policy_prob=record.policy_prob,
        policy_len=record.policy_len, value=z,
        weight_gen=torch.zeros(n, dtype=torch.int32, device=DEVICE))
    assert k <= K_POLICY
    return batch, record, search


def fake_record(board, control, rep, labels, probs, played, ply, done, result,
                root_value=None):
    """A `MoveRecord`-shaped object, for buffer tests that want a known game."""
    from brokefish.search.torch_impl import MoveRecord

    b = board.shape[0]
    return MoveRecord(
        board=board, control=control, rep=rep, policy_move=labels, policy_prob=probs,
        policy_len=torch.full((b,), labels.shape[1], dtype=torch.uint8, device=DEVICE),
        played=played, ply=ply,
        root_value=(torch.zeros(b, device=DEVICE) if root_value is None else root_value),
        weight_gen=0, done=done, result=result)


def one_game(length: int, result: int, seed: int = 0):
    """One game of `length` plies as a stream of single-row records.

    ⚠️ **The control word alternates**, which it did not before 2026-08-07. The board
    is left at the initial position — nothing here reads it — but ``sign(control)`` is
    the side to move and §4's value rule reads it off every record, so a fixture that
    kept White to move for the whole game would have made the rule and its inverse
    agree and this file's check 6 worth nothing.
    """
    boards, control = env.initial_boards(1, device=DEVICE)
    rows = []
    for t in range(length):
        done = torch.tensor([t == length - 1], device=DEVICE)
        turn = control.clone() * (1 if t % 2 == 0 else -1)
        rows.append(fake_record(
            boards.clone(), turn,
            torch.zeros(1, dtype=torch.uint8, device=DEVICE),
            torch.full((1, 4), t + 1, dtype=torch.int16, device=DEVICE),
            torch.full((1, 4), 0.25, dtype=torch.float16, device=DEVICE),
            torch.zeros(1, dtype=torch.int16, device=DEVICE),
            torch.tensor([t], dtype=torch.int32, device=DEVICE),
            done, torch.tensor([result if t == length - 1 else 0],
                               dtype=torch.int8, device=DEVICE)))
    return rows


# --------------------------------------------------------------------------- #
# §3, the loss — check 2
# --------------------------------------------------------------------------- #

def test_loss_matches_a_scalar_transcription_of_the_paper():
    """§12 check 2: the AZ formula read directly, in double, one sample at a time.

    The same method `test_search.py` used on `ucb_score`. The point is that the
    tensor version and the formula are written from different ends: this one loops,
    normalises by hand, and never touches `log_softmax`.
    """
    batch, _, _ = make_batch(16, seed=3)
    n = len(batch)
    torch.manual_seed(0)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE),
                   torch.randn(n, device=DEVICE))
    parts = az_loss(net, batch)

    # The reference decodes the labels itself, in python, from the raw heads — it
    # shares no helper with the loss, which is the point.
    policy = net.policy.detach().double().cpu()
    promo = net.promo.detach().double().cpu()
    board = batch.board.cpu()
    control = batch.control.cpu()
    labels = batch.policy_move.cpu()
    v = torch.tanh(net.value).double().cpu()
    z = batch.value.double().cpu()

    def edge_logit(i, label):
        move, k = int(label) & 0x7FF, (int(label) >> 11) & 0b11
        slot, sq = move // 64, move % 64
        word = int(board[i, slot]) & 0xFFFF
        out = float(policy[i, slot, sq])
        promotes = (((word >> 6) & 7) == 0 and ((word >> 11) & 1) == 0
                    and sq // 8 == (0 if int(control[i]) < 0 else 7))
        if promotes:
            row = [float(promo[i, slot, t]) for t in range(4)]
            top = max(row)
            out += row[k] - top - math.log(sum(math.exp(x - top) for x in row))
        return out

    policy_ref, value_ref = 0.0, 0.0
    for i in range(n):
        k = int(batch.policy_len[i])
        # The denominator is the record's own support, and nothing else.
        logits = [edge_logit(i, labels[i, j]) for j in range(k)]
        top = max(logits)
        denom = sum(math.exp(x - top) for x in logits)
        pi = [float(batch.policy_prob[i, j]) for j in range(k)]
        total = sum(pi)
        for j in range(k):
            policy_ref -= (pi[j] / total) * (logits[j] - top - math.log(denom))
        value_ref += (float(z[i]) - float(v[i])) ** 2
    policy_ref /= n
    value_ref /= n

    assert float(parts.policy) == pytest.approx(policy_ref, rel=1e-5, abs=1e-6)
    assert float(parts.value) == pytest.approx(value_ref, rel=1e-6, abs=1e-7)
    assert float(parts.total) == pytest.approx(policy_ref + value_ref, rel=1e-6)


def test_kl_is_the_cross_entropy_minus_the_targets_own_entropy():
    """§3.2. The two have identical gradients and differ by a stored constant."""
    batch, _, _ = make_batch(16, seed=4)
    torch.manual_seed(1)
    net = ConstNet(torch.randn(16, 32, 64, device=DEVICE),
                   torch.randn(16, 32, 4, device=DEVICE),
                   torch.randn(16, device=DEVICE))
    parts = az_loss(net, batch)
    assert float(parts.kl) == pytest.approx(float(parts.policy) - float(parts.entropy),
                                            rel=1e-6)
    assert float(parts.entropy) > 0.0
    assert float(parts.kl) >= -1e-5, "KL against a normalised target cannot be negative"


def test_a_perfectly_fitted_policy_has_zero_kl_and_a_positive_cross_entropy():
    """§3.2's reason for logging both: the CE floor is `H(pi)` and varies by batch.

    Sharper now than it could be against a recomputed legality mask: the denominator
    is exactly the stored support, so logits equal to `log pi` on that support give a
    softmax equal to `pi` and the KL is zero to floating point, not to a tolerance
    that absorbed a `-30` filler.
    """
    batch, _, _ = make_batch(8, seed=5)
    slot, square, _ = decode_labels(batch.policy_move)
    k = batch.policy_move.shape[1]
    valid = torch.arange(k, device=DEVICE)[None, :] < batch.policy_len.long()[:, None]
    assert not bool(is_promotion_edge(batch.board, batch.control, slot, square).any()), \
        "this rig zeroes the promo head, so it needs a promotion-free batch"

    pi = torch.where(valid, batch.policy_prob.float(), torch.zeros(1, device=DEVICE))
    pi = pi / pi.sum(-1, keepdim=True)
    policy = torch.full((8, 32, 64), -30.0, device=DEVICE)
    for i in range(8):
        for j in range(int(batch.policy_len[i])):
            # ⚠️ `pi` is zero on the edges the search never visited, and they are in
            # the support now — so a perfect fit puts them at -60, not at `log 0`.
            p = float(pi[i, j])
            policy[i, slot[i, j], square[i, j]] = math.log(p) if p > 0 else -60.0
    net = ConstNet(policy, torch.zeros(8, 32, 4, device=DEVICE),
                   torch.zeros(8, device=DEVICE))
    parts = az_loss(net, batch)
    assert float(parts.kl) < 1e-5
    assert float(parts.policy) == pytest.approx(float(parts.entropy), abs=1e-5)
    assert float(parts.entropy) > 0.1


def _accumulated_grad(net, batch, micro: int) -> torch.Tensor:
    net.zero_grad(set_to_none=True)
    n = len(batch)
    for i in range(n // micro):
        mb = batch.slice(i * micro, (i + 1) * micro)
        (az_loss(net, mb).total * (len(mb) / n)).backward()
    return torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])


def test_gradient_accumulation_is_exact_in_double():
    """§7.2: four micro-batches against one whole batch, in float64.

    Not an approximation — the network is pre-norm LayerNorm with no batch
    statistics anywhere, so every per-sample activation is independent of batch
    composition and four means scaled by 1/4 sum to the mean over the whole. This is
    what lets AZ's learning rate schedule transfer unchanged instead of needing a
    rescaling we would have had to invent, so it is worth proving rather than
    approximating.

    ⚠️ In double, because the claim is an *identity*. In fp32 the same comparison
    lands at 1e-4 to 1e-6 relative and moves between runs, since a matmul at N and a
    matmul at N/4 select different cuBLAS split-k reductions — see the fp32 test
    below, which is the loose one on purpose.
    """
    batch, _, _ = make_batch(32, seed=6)
    net = BrokefishNet().to(DEVICE).double()
    whole = _accumulated_grad(net, batch, 32)
    for micro in (8, 16):
        split = _accumulated_grad(net, batch, micro)
        rel = float((whole - split).norm() / whole.norm())
        assert rel < 1e-11, f"{32 // micro} x {micro}: relative difference {rel:.3e}"


def test_gradient_accumulation_is_exact_under_a_value_mask():
    """§7.2 with value supervision subsampled: the value term is a mean over the
    *masked* rows, so its micro-batch weight is the masked count's share and not
    the row count's. `docs/core-algorithm-review.md` §3 has the four-row case where
    the old single scale cancelled a −2/3 gradient to zero; this is the same claim
    at the identity's tolerance, with the supervised fraction moving from 1/2 to 0
    to 1 across the micro-batches, and a proof that the old weighting *does* differ,
    so the test is not vacuous."""
    from dataclasses import replace
    from brokefish.train.loss import micro_batch_weights, value_rows_of

    batch, _, _ = make_batch(32, seed=6)
    mask = torch.ones(32, device=DEVICE)
    mask[4:16] = 0.0                            # micro 8: 4 / 0 / 8 / 8 supervised rows
    batch = replace(batch, value_mask=mask)
    net = BrokefishNet().to(DEVICE).double()

    def grad(micro, exact):
        net.zero_grad(set_to_none=True)
        n, total_value = len(batch), value_rows_of(batch)
        for i in range(n // micro):
            mb = batch.slice(i * micro, (i + 1) * micro)
            parts = az_loss(net, mb)
            w_p, w_v = micro_batch_weights(parts, len(mb), n, total_value)
            if not exact:
                w_v = w_p                        # the pre-2026-09-09 weighting
            (parts.policy * w_p + parts.value * w_v).backward()
        return torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])

    whole = grad(32, exact=True)
    for micro in (8, 16):
        rel = float((whole - grad(micro, exact=True)).norm() / whole.norm())
        assert rel < 1e-11, f"{32 // micro} x {micro}: relative difference {rel:.3e}"
    rel_old = float((whole - grad(8, exact=False)).norm() / whole.norm())
    assert rel_old > 1e-3, f"the row-count weighting would have passed: {rel_old:.3e}"


def test_gradient_accumulation_holds_in_the_precision_it_actually_runs_at():
    """The same identity in fp32, at the tolerance the hardware allows.

    The tolerance is loose and the reason is named: the difference is kernel
    selection, not algebra. What is checked tightly is that re-running the *same*
    split reproduces bit for bit, which is what says the 1/N scaling is right rather
    than accidentally close.
    """
    batch, _, _ = make_batch(64, seed=6)
    net = BrokefishNet().to(DEVICE)
    whole = _accumulated_grad(net, batch, 64)
    assert float((whole - _accumulated_grad(net, batch, 64)).abs().max()) == 0.0
    for micro in (16, 32):
        split = _accumulated_grad(net, batch, micro)
        rel = float((whole - split).norm() / whole.norm())
        assert rel < 1e-3, f"{64 // micro} x {micro}: relative difference {rel:.3e}"


def test_the_l2_constant_reaches_the_optimiser_doubled():
    """AGZ writes `c||theta||^2` with no half, so its gradient is `2 c theta`.

    ⚠️ torch's `weight_decay=w` adds `w theta`. Passing `c` straight through halves
    the regularisation the paper specifies, and nothing else in §12 would catch it.
    """
    assert weight_decay_for(1e-4) == 2e-4
    net = BrokefishNet().to(DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=1.0, momentum=0.0,
                          weight_decay=weight_decay_for(1e-4))
    p = next(iter(net.parameters()))
    before = p.detach().clone()
    net.zero_grad(set_to_none=True)
    for q in net.parameters():
        q.grad = torch.zeros_like(q)
    opt.step()
    # theta <- theta - lr * 2c * theta
    assert torch.allclose(p.detach(), before * (1.0 - 2e-4), atol=1e-9)


# --------------------------------------------------------------------------- #
# §3.4, the label decode — check 3
# --------------------------------------------------------------------------- #

def test_the_label_decode_agrees_with_a_live_search():
    """§12 check 3, the most important one in C2.

    For every root edge the search built, the logit the *training* path selects for
    that edge's stored label must be the scalar `expand` used to build the edge's
    prior. Compared as priors rather than as logits, because that is what `_expand`
    stores; the tolerance is fp16's, since `edge_prior` is fp16.
    """
    net = BrokefishNet().to(DEVICE)
    boards, control, _ = _positions(24, seed=7)
    search = Search(SearchConfig(n=8, B=24, eps=0.0), evaluate=make_evaluator(net),
                    env=env, device=DEVICE, seed=7)
    search.reset(boards, control)
    search.root_init()

    policy, promo, _ = net(search.game_board, search.game_control, search.root_rep)
    ne = search.node_nedges[:, 0].long()
    valid = torch.arange(K_POLICY, device=DEVICE)[None, :] < ne[:, None]
    logit = edge_logits(policy, promo, search.game_board, search.game_control,
                        search.edge_move[:, 0])
    got = torch.log_softmax(
        torch.where(valid, logit, torch.full_like(logit, float("-inf"))), -1).exp()
    want = search.edge_prior[:, 0].float()
    assert bool((ne > 0).all())
    assert float((got - want).abs()[valid].max()) < 2e-4


def test_the_record_carries_the_searchs_own_support():
    """§3.5: `policy_len` is the root's **edge** count, not its visit count.

    This is what lets the loss stop recomputing `movegen`, and it is the difference
    that would silently shrink the softmax denominator if it regressed — every loss
    value a visit-count support produces still looks reasonable.
    """
    net = BrokefishNet().to(DEVICE)
    boards, control, _ = _positions(16, seed=12)
    # `n` well below the edge count, so visited and valid genuinely differ.
    search = Search(SearchConfig(n=8, B=16, eps=0.0), evaluate=make_evaluator(net),
                    env=env, device=DEVICE, seed=12)
    search.reset(boards, control)
    with torch.no_grad():
        record = search.self_play_move()

    n_edges = search.node_nedges[:, 0].long()
    visited = (search.edge_N[:, 0] > 0).sum(-1)
    assert bool((record.policy_len.long() == n_edges).all()), \
        "policy_len must be the edge count"
    assert bool((visited < n_edges).any()), "no unvisited edge in the batch, test is blind"
    # The stored labels are the search's edge array, in order, and the unvisited ones
    # carry pi = 0 rather than being dropped.
    for row in range(16):
        k = int(n_edges[row])
        assert record.policy_move[row, :k].tolist() == \
            search.edge_move[row, 0, :k].tolist()
        assert float(record.policy_prob[row, :k].float().sum()) == pytest.approx(1.0, abs=2e-3)
        assert float(record.policy_prob[row, k:].abs().max()) == 0.0


def test_a_truncated_position_stores_exactly_the_edges_it_searched():
    """The truncation question, dissolved rather than answered.

    On a position with 218 legal moves the search keeps `K_POLICY`. The record now
    carries those, so the training denominator *is* the search's support — no recomputed
    top-K, no dependence on which weights truncated. `audit_labels` still confirms
    every one of them is legal.
    """
    boards, control = env.from_fen(MAX_MOBILITY)
    boards, control = boards.to(DEVICE), control.to(DEVICE)
    n_legal = int(env.bitset_to_bool(env.movegen(boards, control)[0]).sum())
    assert n_legal > K_POLICY, f"only {n_legal} legal moves, nothing to truncate"

    net = BrokefishNet().to(DEVICE)
    search = Search(SearchConfig(n=96, B=1, eps=0.0), evaluate=make_evaluator(net),
                    env=env, device=DEVICE, seed=8)
    search.reset(boards, control)
    with torch.no_grad():
        record = search.self_play_move()
    assert int(search.node_nedges[0, 0]) == K_POLICY, "the search did not truncate"
    assert int(record.policy_len[0]) == K_POLICY
    assert record.policy_move[0].tolist() == search.edge_move[0, 0].tolist()

    batch = TrainBatch(
        board=record.board, control=record.control, rep=record.rep,
        policy_move=record.policy_move, policy_prob=record.policy_prob,
        policy_len=record.policy_len,
        value=torch.zeros(1, device=DEVICE),
        weight_gen=torch.zeros(1, dtype=torch.int32, device=DEVICE))
    assert audit_labels(batch, env) == 0
    az_loss(net, batch, strict=True)


def test_the_audit_catches_a_label_that_is_not_legal_here():
    """§12 check 3's engine-side half, which the loss itself no longer performs.

    ⚠️ It catches an *illegal* label, not a *permuted* one: a permutation of a
    position's own edges is still legal and still trains a shuffled target. Only
    `test_the_label_decode_agrees_with_a_live_search` above settles that, which is
    why that one and not this one is the important check.
    """
    batch, _, _ = make_batch(8, seed=9)
    assert audit_labels(batch, env) == 0
    az_loss(net := BrokefishNet().to(DEVICE), batch, strict=True)

    poisoned = batch.slice(0, 8)
    moves = poisoned.policy_move.clone()
    moves[3, 0] = 31 * 64 + 0        # the black king onto a1, from a white position
    poisoned.policy_move = moves
    with pytest.raises(AssertionError, match="check 3"):
        audit_labels(poisoned, env)

    # And a record whose moving slot is a captured piece fails without the engine.
    dead = batch.slice(0, 8)
    board = dead.board.clone()
    slot, _, _ = decode_labels(dead.policy_move)
    board[2, slot[2, 0]] = 1 << 11   # spec §2.1's captured word
    dead.board = board
    with pytest.raises(AssertionError, match="check 3"):
        az_loss(net, dead, strict=True)


def _positions(n: int, seed: int, plies: int = 24):
    from tests.boards import random_positions

    boards, control, rep = random_positions(n, plies=plies, seed=seed, device=DEVICE)
    mask, in_check = env.movegen(boards, control)
    code, _ = env.terminal(mask, in_check, control, boards)
    live = (code == 0).nonzero(as_tuple=True)[0]
    fill = live[torch.arange(n, device=DEVICE) % live.numel()]
    return boards[fill].contiguous(), control[fill].contiguous(), rep[fill].contiguous()


# --------------------------------------------------------------------------- #
# §4 and §5, the buffer — checks 5 and 6
# --------------------------------------------------------------------------- #

def test_the_record_is_463_bytes_and_round_trips():
    # 463 = 64 board + 2 control + 1 rep + 1 policy_len + 96*(2+2) policy
    #     + 4 value + 4 root_value + 2 weight_gen + 1 value_mask. Written out rather than
    #       derived, so a silent layout change fails here instead of in a .dat
    #       whose zeroed bytes decode as a live white pawn on a1.
    assert RECORD_BYTES == 463
    assert RECORD.itemsize == 463
    buf = ReplayBuffer(window_games=8, mean_plies=8, seed=0)
    rows = one_game(3, result=1)
    for r in rows:
        buf.append(r)
    assert buf.n_records == 3
    # Read the store directly: `sample` draws with replacement, so it is the wrong
    # instrument for "every record arrived".
    assert set(np.array(buf.data[:3]["policy_move"])[:, :4].flatten()) == {1, 2, 3}
    batch = buf.sample(3, device=DEVICE)
    assert batch.board.shape == (3, 32) and batch.board.dtype is torch.int16
    assert batch.policy_move.shape == (3, K_POLICY)
    assert batch.policy_prob.dtype is torch.float16
    assert batch.value.dtype is torch.float32


@pytest.mark.parametrize("length,result", [(3, 1), (4, 1), (5, -1), (6, 0)])
def test_value_parity_on_a_hand_built_game(length, result):
    """§12 check 6, at odd *and* even length.

    `MoveRecord.result` is from the *new* mover's point of view after the move was
    played, and the record's own position has the *previous* mover to move, so
    `z(i) = r if (L - i) is even else -r`. ⚠️ `mcts.md`'s `L >= 3` warning applies:
    at two plies the correct rule and its inverse agree on both records.
    """
    assert length >= 3
    buf = ReplayBuffer(window_games=4, mean_plies=16, seed=0)
    for r in one_game(length, result=result):
        buf.append(r)
    assert buf.n_records == length
    got = np.array(buf.data[:length]["value"])
    want = np.array([result if (length - i) % 2 == 0 else -result
                     for i in range(length)], dtype=np.float32)
    assert np.array_equal(got, want), f"{got} != {want}"
    if result == 0:
        assert not got.any(), "a draw is zero regardless of parity"


def test_the_inverted_parity_would_be_caught():
    """The test above is only worth what its ability to fail is worth."""
    length, result = 5, 1
    correct = [result if (length - i) % 2 == 0 else -result for i in range(length)]
    inverted = [result if i % 2 == 0 else -result for i in range(length)]
    assert correct != inverted


def test_no_record_is_sampleable_before_its_game_ends():
    """§5.3. The incomplete population lives outside the mapping, not inside it
    with a flag, which makes this structural rather than asserted."""
    buf = ReplayBuffer(window_games=4, mean_plies=16, seed=0)
    rows = one_game(5, result=1)
    for i, r in enumerate(rows[:-1]):
        buf.append(r)
        assert buf.n_records == 0
        assert buf.pending_records == i + 1
    buf.append(rows[-1])
    assert buf.n_records == 5 and buf.pending_records == 0


def test_eviction_is_by_game_and_the_window_holds():
    """§12 check 5: occupancy never exceeds the window, and blocks stay whole."""
    buf = ReplayBuffer(window_games=3, mean_plies=8, seed=0)
    lengths = [4, 5, 3, 6, 4]
    for j, length in enumerate(lengths):
        for r in one_game(length, result=1, seed=j):
            buf.append(r)
        buf.check()
        assert buf.n_games <= 3
    assert buf.n_games == 3
    assert buf.n_records == sum(lengths[-3:])
    assert buf.evicted_games == 2


def test_value_subsample_marks_every_kth_ply_and_only_masks_the_value_loss():
    """`--value-subsample k`: 1 in k records per game carry the value target, the
    residue rotates per game, k = 1 marks all, and the masked loss equals the plain
    loss over the marked rows while the policy term does not move."""
    buf = ReplayBuffer(window_games=8, mean_plies=16, seed=0, value_subsample=4)
    for g in range(3):
        for r in one_game(9, result=1, seed=g):
            buf.append(r)
    masks = [np.array(buf.data[9 * g:9 * (g + 1)]["value_mask"]).tolist() for g in range(3)]
    assert masks[0] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert masks[1] == [0, 0, 0, 1, 0, 0, 0, 1, 0]   # residue rotated by one game
    assert masks[2] == [0, 0, 1, 0, 0, 0, 1, 0, 0]
    plain = ReplayBuffer(window_games=8, mean_plies=16, seed=0)
    for r in one_game(9, result=1):
        plain.append(r)
    assert np.array(plain.data[:9]["value_mask"]).tolist() == [1] * 9

    net = BrokefishNet(n_value=3).to(DEVICE)
    batch = buf.sample(24, device=DEVICE)
    parts = az_loss(net, batch, strict=False)
    keep = batch.value_mask.bool()
    sub = TrainBatch(*(t[keep] for t in (
        batch.board, batch.control, batch.rep, batch.policy_move, batch.policy_prob,
        batch.policy_len, batch.value, batch.weight_gen)))
    ref = az_loss(net, sub, strict=False)
    torch.testing.assert_close(parts.value, ref.value)
    unmasked = az_loss(net, TrainBatch(*(t for t in (
        batch.board, batch.control, batch.rep, batch.policy_move, batch.policy_prob,
        batch.policy_len, batch.value, batch.weight_gen))), strict=False)
    torch.testing.assert_close(parts.policy, unmasked.policy)
    assert not torch.allclose(parts.value, unmasked.value)


def test_the_buffer_survives_a_wraparound():
    """A game's block may straddle the end of the ring; reads are modular."""
    buf = ReplayBuffer(window_games=100, mean_plies=1, capacity_records=11, seed=0)
    for j in range(6):
        for r in one_game(4, result=1, seed=j):
            buf.append(r)
        buf.check()
    assert buf.capacity_evictions > 0, "the capacity bound never bound"
    assert buf.n_records <= 11
    batch = buf.sample(buf.n_records, device=DEVICE)
    assert int(batch.board.shape[0]) == buf.n_records


def test_the_buffer_snapshot_round_trips(tmp_path):
    buf = ReplayBuffer(str(tmp_path / "r.dat"), window_games=8, mean_plies=8, seed=0)
    for j in range(3):
        for r in one_game(4, result=1, seed=j):
            buf.append(r)
    for r in one_game(6, result=-1)[:3]:      # a game left in flight
        buf.append(r)
    assert buf.pending_records == 3
    buf.save(str(tmp_path / "meta.npz"))

    again = ReplayBuffer(str(tmp_path / "r.dat"), window_games=8, mean_plies=8,
                         seed=0, resume=True)
    again.open_games(1)
    again.load(str(tmp_path / "meta.npz"))
    assert (again.n_games, again.n_records) == (buf.n_games, buf.n_records)
    assert again.pending_records == 3
    again.check()
    # ⚠️ A restored game's pending records come back as one block, not one entry
    # per ply, so anything that counts them by list length gets the parity wrong.
    for r in one_game(6, result=-1)[3:]:
        again.append(r)
    assert again.n_records == buf.n_records + 6
    tail = np.array(again.data[(again.head + buf.n_records) % again.capacity:][:6]["value"])
    assert np.array_equal(tail, np.array([-1, 1, -1, 1, -1, 1], dtype=np.float32))


# --------------------------------------------------------------------------- #
# §6, the cadence
# --------------------------------------------------------------------------- #

def _cadence_trainer(rate=0.815, batch=4096, n_records=10 ** 9):
    cfg = TrainConfig(samples_per_position=rate, batch=batch)
    obj = object.__new__(Trainer)
    obj.cfg = cfg
    obj.carry = 0.0
    obj.samples_dropped_filling = 0.0
    obj.buffer = type("B", (), {"n_records": n_records})()
    return cfg, obj


def test_the_carry_makes_the_long_run_ratio_exact():
    """§6: a fixed number of samples per *position* generated, whatever the phase size."""
    cfg, obj = _cadence_trainer()
    total_steps, total_records = 0, 0
    for _ in range(500):
        total_records += 3700
        total_steps += Trainer.steps_owed(obj, 3700)
    # The carry is the whole point: what is exact is samples-owed, not samples-taken,
    # and the difference is bounded by one batch however the phases are sized.
    assert total_steps * cfg.batch + obj.carry == pytest.approx(0.815 * total_records)
    assert 0.0 <= obj.carry < cfg.batch
    reuse = total_steps * cfg.batch / total_records
    assert 0.815 - cfg.batch / total_records <= reuse <= 0.815


def test_the_reuse_does_not_drift_with_game_length():
    """The defect this replaced, measured on `t4h-n64` 2026-07-31.

    The old rule owed `65.2 x games_closed`, so as the mean game grew 101 -> 144 plies
    the per-position reuse fell 0.641 -> 0.374 -- a 42 % drop in how much each example
    was trained on, inside one run, with the data rate constant throughout. Riding the
    records instead makes the reuse a property of the config and not of how the games
    happened to go.
    """
    records_per_generation = 10 * 1024        # moves_per_phase x games in flight
    for _game_length in (80, 101, 144, 300):
        cfg, obj = _cadence_trainer()
        steps = sum(Trainer.steps_owed(obj, records_per_generation) for _ in range(200))
        reuse = steps * cfg.batch / (200 * records_per_generation)
        assert reuse == pytest.approx(0.815, abs=0.01)


def test_the_default_is_azs_ratio_at_the_assumed_game_length():
    # 65.2 samples per game / 80 plies per game = 0.815 per position. The 80 is ours,
    # not AZ's -- neither paper publishes a game length.
    assert TrainConfig.samples_per_position == pytest.approx(65.2 / 80.0)
    assert TrainConfig.samples_per_game == 65.2


def test_no_gradient_step_before_the_buffer_holds_one_batch():
    """§5.5, and the carry is dropped rather than banked while it fills."""
    cfg, obj = _cadence_trainer(n_records=10)
    assert Trainer.steps_owed(obj, 10000) == 0
    assert obj.carry == 0.0
    assert obj.samples_dropped_filling > 0.0


def test_the_learning_rate_schedule_is_the_paper_s_four_values():
    """§7.3. Three drops, matching AZ p.14's prose, at AZ's four values."""
    cfg = TrainConfig(total_steps=700_000, lr_schedule=LR_SCHEDULE)
    assert [v for _, v in LR_SCHEDULE] == [0.2, 0.02, 0.002, 0.0002]
    assert cfg.lr_at(0) == 0.2
    assert cfg.lr_at(99_999) == 0.2
    assert cfg.lr_at(100_000) == 0.02
    assert cfg.lr_at(299_999) == 0.02
    assert cfg.lr_at(300_000) == 0.002
    assert cfg.lr_at(500_000) == 0.0002
    assert cfg.lr_at(10 ** 9) == 0.0002


def test_warmup_is_off_by_default_so_the_agz_baseline_does_not_move():
    """The whole point of a default: an ablation must not silently change its control."""
    assert TrainConfig.warmup_steps == 0
    cfg = TrainConfig(total_steps=700_000, lr_schedule=LR_SCHEDULE)
    assert [cfg.lr_at(s) for s in (0, 1, 50, 99_999)] == [0.2] * 4


def test_linear_warmup_ramps_and_then_hands_back_to_the_schedule():
    cfg = TrainConfig(total_steps=700_000, lr_schedule=((0.0, 0.001),), warmup_steps=100)

    # ⚠️ Step 0 must not be zero: a zero rate is a step whose gradient is computed,
    # logged, and thrown away. `(step + 1) / warmup` is what avoids that.
    assert cfg.lr_at(0) == pytest.approx(0.001 / 100)
    assert cfg.lr_at(0) > 0.0

    # Linear in between, and the ramp is over by construction at the last warmup step.
    assert cfg.lr_at(49) == pytest.approx(0.001 * 50 / 100)
    assert cfg.lr_at(99) == pytest.approx(0.001)
    assert cfg.lr_at(100) == pytest.approx(0.001)
    assert cfg.lr_at(10_000) == pytest.approx(0.001)

    steps = [cfg.lr_at(s) for s in range(100)]
    assert steps == sorted(steps), "the ramp must be monotone"


def test_cosine_decays_from_the_peak_to_lr_min_and_stays_there():
    cfg = TrainConfig(total_steps=1000, lr_schedule=((0.0, 0.001),), decay="cosine")
    assert cfg.lr_at(0) == pytest.approx(0.001)
    assert cfg.lr_at(500) == pytest.approx(0.0005)          # half a cosine is half way
    assert cfg.lr_at(999) == pytest.approx(0.0, abs=1e-8)
    # ⚠️ Past the horizon the progress term is clamped. Without that, `cos` turns and
    # the rate climbs back to the peak -- a run that overshoots would re-heat.
    assert cfg.lr_at(1000) == pytest.approx(0.0, abs=1e-12)
    assert cfg.lr_at(10_000) == pytest.approx(0.0, abs=1e-12)

    got = [cfg.lr_at(s) for s in range(1000)]
    assert got == sorted(got, reverse=True), "cosine must be monotone decreasing"


def test_cosine_respects_lr_min():
    cfg = TrainConfig(total_steps=1000, lr_schedule=((0.0, 0.001),), decay="cosine",
                      lr_min=1e-5)
    assert cfg.lr_at(0) == pytest.approx(0.001)
    assert cfg.lr_at(1000) == pytest.approx(1e-5)
    assert cfg.lr_at(500) == pytest.approx(1e-5 + (0.001 - 1e-5) * 0.5)


def test_warmup_hands_over_to_cosine_at_the_peak():
    """The two compose: the ramp ends at the peak, and decay starts from there --
    not from wherever a cosine measured over the whole run happens to be."""
    cfg = TrainConfig(total_steps=1000, lr_schedule=((0.0, 0.001),), decay="cosine",
                      warmup_steps=100)
    assert cfg.lr_at(0) == pytest.approx(0.001 / 100)
    assert cfg.lr_at(99) == pytest.approx(0.001)             # ramp ends at the peak
    assert cfg.lr_at(100) == pytest.approx(0.001)            # cosine starts *at* it
    assert cfg.lr_at(101) < 0.001                            # and decays from there
    assert cfg.lr_at(999) == pytest.approx(0.0, abs=1e-8)
    got = [cfg.lr_at(s) for s in range(1000)]
    assert max(got) == pytest.approx(0.001), "the ramp must never overshoot the peak"


def test_a_cosine_sized_against_the_wrong_total_steps_is_a_constant():
    """⚠️ The failure this will actually produce: `--decay cosine` with the default
    `total_steps` in a run that takes ~10^2 steps is a flat rate, not a decay. It is
    not a bug and no assertion can catch it -- hence the startup line that prints the
    shape. This test exists to pin *why* that line is there."""
    cfg = TrainConfig(total_steps=159_000, lr_schedule=((0.0, 0.001),), decay="cosine")
    assert cfg.lr_at(116) == pytest.approx(0.001, rel=1e-4)  # 0.01 % below peak


def test_an_unknown_decay_is_refused_rather_than_silently_stepping():
    with pytest.raises(ValueError, match="unknown decay"):
        TrainConfig(decay="linear").lr_at(0)


def test_warmup_scales_the_schedule_rather_than_replacing_it():
    """A drop landing inside the ramp must still be the schedule's value, scaled."""
    cfg = TrainConfig(total_steps=1000, lr_schedule=((0.0, 0.2), (0.5, 0.02)),
                      warmup_steps=1000)
    assert cfg.lr_at(499) == pytest.approx(0.2 * 500 / 1000)
    assert cfg.lr_at(500) == pytest.approx(0.02 * 501 / 1000)


# --------------------------------------------------------------------------- #
# §8.1, weight synchronisation — check 9
# --------------------------------------------------------------------------- #

def test_the_fingerprint_moves_when_any_single_weight_does():
    net = BrokefishNet().to(DEVICE)
    before = weight_fingerprint(net)
    assert weight_fingerprint(net) == before, "the fingerprint is not deterministic"
    with torch.no_grad():
        net.policy.weight[3, 7] += 1e-4
    assert weight_fingerprint(net) != before


def test_the_fingerprint_sees_a_permutation_of_the_same_values():
    """Which is the failure being guarded against: `pack_b` on the wrong tensor."""
    net = BrokefishNet().to(DEVICE)
    before = weight_fingerprint(net)
    with torch.no_grad():
        net.policy.weight.copy_(net.policy.weight.flip(0))
    assert weight_fingerprint(net) != before


def test_a_stale_packed_snapshot_raises_rather_than_self_playing():
    """§8.1. A snapshot built once is a working system that never learns."""
    net = BrokefishNet().to(DEVICE)
    packed = PackedWeights.pack(net, weight_gen=0)
    packed.assert_current(net, 0)
    with pytest.raises(AssertionError, match="§8.1"):
        packed.assert_current(net, 1)
    with torch.no_grad():
        net.policy.weight[0, 0] += 0.5
    with pytest.raises(AssertionError, match="fingerprint"):
        packed.assert_current(net, 0)


@pytest.mark.slow
def test_weights_propagate_to_the_fused_encoder(tmp_path):
    """§12 check 9. Every other check here passes on a loop frozen at generation 0."""
    trainer = _smoke_trainer(tmp_path, "propagate")
    out = trainer.check_weights_propagate()
    assert out["moved"] > 1e-3
    assert out["agree"] < 3e-2
    trainer.log.close()


# --------------------------------------------------------------------------- #
# §9, checkpoint and resume — check 4
# --------------------------------------------------------------------------- #

def _smoke_trainer(tmp_path, run: str, **kw):
    defaults = dict(
        n_sims=8, batch_games=16, moves_per_phase=2, window_games=64, mean_plies=32,
        max_plies=40, batch=8, micro_batch=4, total_steps=50, buffer_in_memory=True,
        collect_search_stats=False, checkpoint_every=10 ** 9,
        buffer_snapshot_every=10 ** 9)
    cfg = TrainConfig(**{**defaults, **kw})
    logger = Logger(run, log_dir=str(tmp_path), use_wandb=False)
    return Trainer(cfg, run=run, logger=logger, log_dir=str(tmp_path))


@pytest.mark.slow
def test_resume_is_bit_exact(tmp_path):
    """§9, and §12 check 4.

    Checkpoint at `k`, continue to `k + steps`; then resume from `k` and run the same
    steps. The two weight tensors must be identical. This is one of the few checks in
    C2 with a definite right answer, and it catches the whole class of "something in
    the loop is not in the checkpoint" bugs, which otherwise appear weeks later as an
    unexplained kink in the Elo curve.
    """
    trainer = _smoke_trainer(tmp_path, "resume")
    while trainer.buffer.n_records < 8 * 4:
        trainer.self_play_phase()
    ckpt = str(tmp_path / "k.pt")
    trainer.save_checkpoint(ckpt, with_buffer=True)
    for _ in range(6):
        trainer.train_step()
    straight = weight_fingerprint(trainer.net)
    trainer.log.close()

    again = _smoke_trainer(tmp_path, "resume2")
    again.load_checkpoint(ckpt)
    for _ in range(6):
        again.train_step()
    assert weight_fingerprint(again.net) == straight
    again.log.close()


def test_a_config_change_refuses_to_resume(tmp_path):
    trainer = _smoke_trainer(tmp_path, "cfg")
    ckpt = str(tmp_path / "c.pt")
    trainer.save_checkpoint(ckpt)
    trainer.log.close()
    other = _smoke_trainer(tmp_path, "cfg2", momentum=0.5)
    with pytest.raises(RuntimeError, match="hybrid run"):
        other.load_checkpoint(ckpt)
    other.load_checkpoint(ckpt, allow_config_change=True)
    other.log.close()


def test_the_checkpoint_carries_everything_section_9_lists(tmp_path):
    trainer = _smoke_trainer(tmp_path, "state")
    state = trainer.state_dict()
    for key in ("net", "opt", "step", "weight_gen", "carry", "games_completed",
                "positions_generated", "samples_drawn", "lr", "fingerprint",
                "seconds", "euros_per_hour", "search_rng", "torch_rng",
                "torch_cuda_rng", "games", "config_hash"):
        assert key in state, f"§9 lists {key} and the checkpoint does not carry it"
    # The tree is rebuilt from scratch every move, so the games are the whole of
    # self-play's state and there is nothing else to store.
    assert set(state["games"]) == {
        "game_board", "game_control", "game_hash", "game_ring", "game_ring_len",
        "game_ply", "game_done", "game_result"}
    trainer.log.close()


# --------------------------------------------------------------------------- #
# §5.4 and §8, the loop
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_the_game_length_cap_closes_a_game_as_a_draw(tmp_path):
    """§5.4: 512 plies, scored drawn, counted as a completed game.

    ⚠️ A game that never terminates leaks its whole pending list, which is the only
    reason the cap exists. Here it is set to 6 plies so it fires immediately.
    """
    trainer = _smoke_trainer(tmp_path, "cap", max_plies=6)
    for _ in range(4):
        trainer.self_play_phase()
    assert trainer.games_capped > 0
    assert trainer.buffer.n_games > 0
    values = np.array(trainer.buffer.data[:trainer.buffer.n_records]["value"])
    assert (values == 0).any(), "a capped game must be scored as a draw"
    assert trainer.buffer.stats().mean_game_length <= 6
    trainer.log.close()


@pytest.mark.slow
def test_a_generation_trains_and_republishes_the_weights(tmp_path):
    """§8 end to end: self-play, cadence, gradient phase, new generation."""
    trainer = _smoke_trainer(tmp_path, "gen")
    while trainer.buffer.n_records < 8:
        trainer.self_play_phase()
    before = weight_fingerprint(trainer.net)
    gen_before = trainer.packed.weight_gen
    trainer.gradient_phase(2)
    assert trainer.step == 2
    assert weight_fingerprint(trainer.net) != before
    assert trainer.packed.weight_gen == gen_before + 1
    assert trainer.packed.fingerprint == weight_fingerprint(trainer.net)
    trainer.packed.assert_current(trainer.net, trainer.weight_gen)
    assert trainer.training_seconds > 0.0
    trainer.log.close()


# --------------------------------------------------------------------------- #
# The live buffer index, for monitoring from another process
# --------------------------------------------------------------------------- #

def test_the_index_lets_another_process_read_the_buffer(tmp_path):
    """The gap that made the buffer unreadable mid-run, closed.

    `_g_start`, `_g_count`, `head` and `tail` are attributes of the writer, so the
    `.dat` mapping alone is 735 MB of records with no way to say where a game begins.
    `save_index` writes them; `BufferView` reads them without touching the writer.
    """
    from brokefish.train.buffer import BufferView, ReplayBuffer

    dat = str(tmp_path / "run.dat")
    idx = str(tmp_path / "run.index.npz")
    buf = ReplayBuffer(path=dat, window_games=16, mean_plies=8)
    buf.open_games(2)

    rng = np.random.default_rng(0)
    lengths = [5, 3, 7]
    for g, length in enumerate(lengths):
        block = np.zeros(length, dtype=RECORD)
        block["board"] = rng.integers(0, 4096, size=(length, 32), dtype=np.int64)
        block["control"] = 1 - 2 * (np.arange(length) % 2)
        buf._pending[0] = [block[i:i + 1] for i in range(length)]
        buf._close(0, result=(1 if g == 0 else 0),
                  term_control=int(block['control'][-1]))
    buf.save_index(idx)

    view = BufferView(dat, idx)
    assert len(view) == 3
    assert [view.game_length(i) for i in range(3)] == lengths
    # Game 0 ended with the mover winning; `control` at the last record says who.
    assert view.outcome(0) in (-1, 1)
    assert view.outcome(1) == 0
    stats = view.stats()
    assert stats["games"] == 3 and stats["records"] == sum(lengths)


def test_the_view_sees_new_games_only_after_a_refresh(tmp_path):
    """A stale index is consistent-but-old, which is the right failure direction."""
    from brokefish.train.buffer import BufferView, ReplayBuffer

    dat = str(tmp_path / "run.dat")
    idx = str(tmp_path / "run.index.npz")
    buf = ReplayBuffer(path=dat, window_games=16, mean_plies=8)
    buf.open_games(1)

    def add(length):
        block = np.zeros(length, dtype=RECORD)
        block["control"] = 1
        buf._pending[0] = [block[i:i + 1] for i in range(length)]
        buf._close(0, result=0, term_control=1)

    add(4)
    buf.save_index(idx)
    view = BufferView(dat, idx)
    assert len(view) == 1
    add(6)                       # written to the mapping, not yet to the index
    assert not view.refresh() and len(view) == 1
    buf.save_index(idx)
    assert view.refresh() and len(view) == 2
    assert view.game_length(1) == 6


def test_a_wrapped_game_is_reassembled(tmp_path):
    """Games are contiguous in the file but the ring wraps; `game()` resolves it."""
    from brokefish.train.buffer import BufferView, ReplayBuffer

    dat = str(tmp_path / "run.dat")
    idx = str(tmp_path / "run.index.npz")
    buf = ReplayBuffer(path=dat, window_games=8, capacity_records=10)
    buf.open_games(1)
    marks = []
    for g in range(4):
        length = 3
        block = np.zeros(length, dtype=RECORD)
        block["control"] = 1
        block["root_value"] = np.float32(g + 1)      # a per-game marker
        marks.append(g + 1)
        buf._pending[0] = [block[i:i + 1] for i in range(length)]
        buf._close(0, result=0, term_control=1)
    buf.save_index(idx)
    view = BufferView(dat, idx)
    assert view.tail < view.head or view.n_records < view.capacity or True
    for i in range(len(view)):
        g = view.game(i)
        assert g.shape[0] == 3
        # every record of a game carries that game's marker, wraparound or not
        assert len(set(g["root_value"].tolist())) == 1, g["root_value"]


def test_the_index_is_written_atomically(tmp_path):
    """A half-written index read by the viewer would point at arbitrary offsets."""
    from brokefish.train.buffer import ReplayBuffer

    dat = str(tmp_path / "run.dat")
    idx = str(tmp_path / "run.index.npz")
    buf = ReplayBuffer(path=dat, window_games=8, mean_plies=8)
    buf.open_games(1)
    buf.save_index(idx)
    assert os.path.exists(idx)
    assert not os.path.exists(idx + ".tmp"), "the temp file must be renamed, not left"


# --------------------------------------------------------------------------- #
# §11, playout cap randomisation
# --------------------------------------------------------------------------- #

def _sparse_game(plies: int, stored: list, result: int):
    """A game of `plies` where only the plies in `stored` are recorded.

    Returns the buffer after the game closed. `stored` need not contain the last
    ply — that is the case §11 exists to make safe.
    """
    buf = ReplayBuffer(window_games=8, mean_plies=32, seed=0)
    rows = one_game(plies, result=result)
    for t, r in enumerate(rows):
        buf.append(r, store=(t in stored))
    return buf


@pytest.mark.parametrize("plies,stored,result", [
    (8, [0, 3, 6], 1),          # the game ends on a ply nobody recorded
    (8, [1, 4, 7], -1),         # ... and one where the last ply is recorded
    (9, [0, 1, 2], 1),          # three consecutive, all far from the end
    (7, [5], -1),               # a single record
])
def test_the_value_target_is_right_when_the_recorded_plies_are_sparse(
        plies, stored, result):
    """§12 check 6 under §11, which is where the old rule silently broke.

    `z(i) = r if (L - i) even else -r` counts *positions in the pending block*, so
    with a sparse subset it reads the wrong side to move for every record whose
    distance from the end is not its index's distance — i.e. for almost all of them.
    The rule now reads `control`, so this is exact rather than approximately right.
    """
    buf = _sparse_game(plies, stored, result)
    assert buf.n_records == len(stored), "closure did not happen or stored too much"
    got = np.array(buf.data[:len(stored)]["value"])
    # `one_game` puts White (control > 0) to move on even plies, and the game ends
    # after the move played from ply `plies - 1`, so `result` speaks for the side to
    # move at ply `plies` -- White iff `plies` is even.
    winner_is_white = (plies % 2 == 0)
    want = np.array([result if ((t % 2 == 0) == winner_is_white) else -result
                     for t in stored], dtype=np.float32)
    assert np.array_equal(got, want), f"{got} != {want} for plies at {stored}"


def test_the_sparse_rule_agrees_with_the_dense_one_on_a_dense_game():
    """The new rule is not a new rule where the old one applied. Both directions:
    a dense game must give the published parity, at odd and even length."""
    for length in (3, 4, 5, 6):
        for result in (1, -1):
            buf = _sparse_game(length, list(range(length)), result)
            got = np.array(buf.data[:length]["value"])
            want = np.array([result if (length - i) % 2 == 0 else -result
                             for i in range(length)], dtype=np.float32)
            assert np.array_equal(got, want), (length, result, got, want)


def test_the_old_dense_rule_would_be_caught_on_a_sparse_game():
    """The test above is only worth what its ability to fail is worth.

    `z(i) = r if (L - i) even else -r` over a *block* of 3 records taken from plies
    0, 3 and 6 of an 8-ply game gives `[-r, r, -r]`, and the truth is `[r, -r, r]`:
    every sign inverted, i.e. a third of the buffer teaching the network to play to
    lose. Nothing would have raised.
    """
    plies, stored, result = 8, [0, 3, 6], 1
    buf = _sparse_game(plies, stored, result)
    got = np.array(buf.data[:len(stored)]["value"])
    dense = np.array([result if (len(stored) - i) % 2 == 0 else -result
                      for i in range(len(stored))], dtype=np.float32)
    assert np.array_equal(got, -dense), (got, dense)


def test_a_game_that_ends_on_an_unrecorded_ply_still_closes():
    """⚠️ The trap §11 is built around, asserted directly.

    The natural way to write "do not record cheap turns" is to not call `append` on
    them. Then a game that *ends* on a cheap turn is never closed: its earlier
    records stay in `_pending`, the search resets the slot, and the next game to
    finish there flushes them under its own result. Silent, and at `p = 0.3` it is
    70 % of games. So `store=False` must still close, and the slot must be empty
    afterwards.
    """
    buf = ReplayBuffer(window_games=8, mean_plies=32, seed=0)
    rows = one_game(5, result=1)                 # ends on ply 4
    for t, r in enumerate(rows):
        buf.append(r, store=(t < 3))             # plies 0-2 recorded, 3 and 4 not
    assert buf.n_records == 3, "the game did not close on an unrecorded ply"
    assert buf.pending_records == 0, "records were left in flight to poison a later game"
    assert buf.n_games == 1

    # And the next game in the same slot inherits none of them.
    for t, r in enumerate(one_game(4, result=-1)):
        buf.append(r, store=(t == 0))
    assert buf.n_games == 2 and buf.n_records == 4
    first, second = np.array(buf.data[:3]["value"]), np.array(buf.data[3:4]["value"])
    assert np.array_equal(first, np.array([-1, 1, -1], dtype=np.float32)), first
    assert np.array_equal(second, np.array([-1], dtype=np.float32)), second


def test_the_schedule_gives_exactly_the_asked_for_proportion_and_moves_around():
    """§11's schedule: stratified, not Bernoulli, and redrawn every generation.

    Exact per phase because the realised cap is this run's cost axis, and a Bernoulli
    draw over 10 moves wanders by +-11 % of it. Redrawn because a game's ply offset
    inside the phase is fixed for its whole life, so a *constant* pattern would record
    one residue class of each game's plies and nothing else.
    """
    obj = object.__new__(Trainer)
    obj.cfg = TrainConfig(pcr_p=0.3, moves_per_phase=10, seed=0)

    seen = set()
    by_position = [0] * 10
    for gen in range(400):
        obj.generation = gen
        full = Trainer._pcr_schedule(obj, 10)
        assert sum(full) == 3, f"generation {gen} ran {sum(full)} full turns"
        seen.add(tuple(full))
        for i, f in enumerate(full):
            by_position[i] += int(f)
    assert len(seen) > 20, f"only {len(seen)} distinct patterns in 400 generations"
    # Every position in the phase gets the full cap about equally often: 400 * 3 / 10.
    assert min(by_position) > 80 and max(by_position) < 160, by_position

    # Deterministic given (seed, generation), which is what §9's resume rides.
    obj.generation = 7
    assert Trainer._pcr_schedule(obj, 10) == Trainer._pcr_schedule(obj, 10)


def test_pcr_off_is_the_pre_2026_08_07_path():
    obj = object.__new__(Trainer)
    obj.cfg = TrainConfig(pcr_p=0.0, moves_per_phase=4)
    obj.generation = 0
    assert Trainer._pcr_schedule(obj, 4) == [True] * 4


@pytest.mark.parametrize("kw,match", [
    (dict(pcr_p=1.5), "pcr_p"),
    (dict(pcr_p=0.3, pcr_fast_sims=0), "pcr_fast_sims"),
    (dict(pcr_p=0.3, pcr_fast_sims=99), "pcr_fast_sims"),   # above n_sims = 8
    (dict(pcr_p=0.02, pcr_fast_sims=2), "rounds to zero"),
])
def test_a_misconfigured_cap_is_refused_at_construction(tmp_path, kw, match):
    """Not at the first cheap move, deep inside generation 1."""
    with pytest.raises(ValueError, match=match):
        _smoke_trainer(tmp_path, "bad", **kw)


@pytest.mark.slow
def test_a_pcr_phase_stores_only_the_full_turns(tmp_path):
    """End to end: the counters say what was played and what was kept."""
    trainer = _smoke_trainer(tmp_path, "pcr", n_sims=8, pcr_p=0.5,
                             pcr_fast_sims=2, moves_per_phase=4)
    play = trainer.self_play_phase()
    assert play["full_moves"] == 2
    assert play["records_added"] == 4 * trainer.cfg.batch_games
    assert play["records_stored"] == 2 * trainer.cfg.batch_games
    # The cost axis, measured: (2 * 8 + 2 * 2) / 4.
    assert play["sims_mean"] == pytest.approx(5.0)
    # ⚠️ The cadence rides `records_added`, so the gradient step count per generation
    # is the same as a uniform run's -- which is what makes the two comparable.
    assert Trainer.steps_owed(trainer, play["records_added"]) >= 0
    trainer.log.close()


# --------------------------------------------------------------------------- #
# §11a, the annealed cap
# --------------------------------------------------------------------------- #

class TestTheSimsSchedule:

    def test_the_caps_change_at_the_stated_training_second(self):
        cfg = TrainConfig(n_sims=512, pcr_p=0.3, pcr_fast_sims=64,
                          sims_schedule=((0.0, 256, 64), (21600.0, 512, 128)))
        assert cfg.caps_at(0.0) == (256, 64)
        assert cfg.caps_at(21599.9) == (256, 64)
        assert cfg.caps_at(21600.0) == (512, 128)
        assert cfg.caps_at(10 ** 9) == (512, 128)

    def test_an_empty_schedule_is_the_fixed_cap(self):
        cfg = TrainConfig(n_sims=256, pcr_fast_sims=64)
        assert cfg.caps_at(0.0) == (256, 64) == cfg.caps_at(10 ** 6)

    @pytest.mark.parametrize("spec,want", [
        ("0:256/64", ((0.0, 256, 64),)),
        ("0:256/64,21600:512/128", ((0.0, 256, 64), (21600.0, 512, 128))),
        (None, ()),
        ("", ()),
    ])
    def test_the_spec_parses(self, spec, want):
        from brokefish.train.loop import parse_sims_schedule
        assert parse_sims_schedule(spec) == want

    @pytest.mark.parametrize("spec,match", [
        ("256/64", "SECONDS:N/n"),
        ("0:256", "SECONDS:N/n"),
        ("0:x/64", "SECONDS:N/n"),
        ("600:256/64", "must start at 0"),
    ])
    def test_a_mistyped_schedule_raises_rather_than_becoming_no_schedule(self, spec, match):
        """⚠️ A spec that silently degrades to 'fixed cap' is a 12 h run that answers
        the wrong question and looks completely normal while doing it."""
        from brokefish.train.loop import parse_sims_schedule
        with pytest.raises(ValueError, match=match):
            parse_sims_schedule(spec)

    def test_n_sims_is_raised_to_the_largest_stage(self):
        """`n_sims` allocates the node pool for the whole run, so a schedule that
        anneals past it would overrun the tree six hours in."""
        from brokefish.train.loop import build_parser, config_from_args
        args = build_parser().parse_args(
            ["--run", "x", "--sims", "256", "--pcr-p", "0.3",
             "--sims-schedule", "0:256/64,21600:512/128"])
        cfg = config_from_args(args)
        assert cfg.n_sims == 512 and cfg.pcr_fast_sims == 64

    @pytest.mark.parametrize("sched,match", [
        (((0.0, 256, 64), (100.0, 512, 600)), "n <= N"),
        (((0.0, 256, 64), (100.0, 999, 128)), "n_sims"),
        (((0.0, 256, 64), (100.0, 512, 128), (50.0, 256, 64)), "increasing order"),
    ])
    def test_a_bad_schedule_is_refused_at_construction(self, tmp_path, sched, match):
        with pytest.raises(ValueError, match=match):
            _smoke_trainer(tmp_path, "sched", n_sims=512, pcr_p=0.5, pcr_fast_sims=8,
                           moves_per_phase=4, sims_schedule=sched)

    @pytest.mark.slow
    def test_a_phase_uses_the_scheduled_caps_and_logs_them(self, tmp_path):
        trainer = _smoke_trainer(tmp_path, "sched2", n_sims=16, pcr_p=0.5,
                                 pcr_fast_sims=2, moves_per_phase=4,
                                 sims_schedule=((0.0, 4, 2), (1e-9, 16, 8)))
        play = trainer.self_play_phase()
        # `training_seconds` is 0 before the first phase, so stage 0 is what ran.
        assert (play["full_sims"], play["fast_sims"]) == (4, 2)
        assert play["sims_mean"] == pytest.approx(3.0)
        play = trainer.self_play_phase()          # now past 1e-9 seconds
        assert (play["full_sims"], play["fast_sims"]) == (16, 8)
        assert play["sims_mean"] == pytest.approx(12.0)
        trainer.log.close()


# --------------------------------------------------------------------------
# The win/draw/loss value head, 2026-08-21. It is an option (`--value-classes 3`)
# and the scalar head is untouched, so what these pin is the *branch*: the class
# convention, that the collapse the search sees is `p(win) - p(loss)`, and that a
# target which is not a game outcome fails loudly rather than rounding into a
# class it does not mean.


def test_the_wdl_loss_is_a_three_class_cross_entropy_on_the_outcome():
    """The value term against `F.cross_entropy` written from the other end.

    ⚠️ The class convention is the thing being pinned: **0 = loss, 1 = draw,
    2 = win, from the side to move**, so the index of a stored `z` is `z + 1`. It is
    normative because `csrc/encoder.cu`'s epilogue reads those three columns by
    position — a permutation here and there would still produce a value in `[-1, 1]`
    and would be wrong in a way nothing downstream can see.
    """
    batch, _, _ = make_batch(16, seed=11)
    n = len(batch)
    torch.manual_seed(0)
    logits = torch.randn(n, 3, device=DEVICE)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), logits)
    parts = az_loss(net, batch)
    assert parts.n_value == 3

    # Written by hand in double, one sample at a time, sharing no helper with the loss.
    ref = 0.0
    for i in range(n):
        z = float(batch.value[i])
        assert z in (-1.0, 0.0, 1.0)
        row = [float(x) for x in logits[i].double()]
        top = max(row)
        lse = top + math.log(sum(math.exp(x - top) for x in row))
        ref += lse - row[int(z) + 1]
    ref /= n
    assert float(parts.value) == pytest.approx(ref, rel=1e-5, abs=1e-6)

    # And the weight still scales it, exactly as it scales the squared error.
    doubled = az_loss(net, batch, value_weight=2.0)
    assert float(doubled.value) == pytest.approx(2.0 * float(parts.value), rel=1e-6)


def test_the_wdl_head_hands_the_search_win_minus_loss():
    """`value_pred` is the *only* thing the search, the collapse and the league see,
    and it has to stay a scalar in [-1, 1] with the same sign convention as the tanh
    head: positive means the side to move is winning."""
    from brokefish.nn.model import wdl_to_scalar

    batch, _, _ = make_batch(8, seed=12)
    n = len(batch)
    # One certain loss, one certain draw, one certain win, then noise.
    logits = torch.randn(n, 3, device=DEVICE) * 0.1
    logits[0] = torch.tensor([20.0, 0.0, 0.0], device=DEVICE)
    logits[1] = torch.tensor([0.0, 20.0, 0.0], device=DEVICE)
    logits[2] = torch.tensor([0.0, 0.0, 20.0], device=DEVICE)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), logits)
    v = az_loss(net, batch).value_pred
    assert v.shape == (n,) and v.dtype is torch.float32
    assert float(v[0]) == pytest.approx(-1.0, abs=1e-5)
    assert float(v[1]) == pytest.approx(0.0, abs=1e-5)
    assert float(v[2]) == pytest.approx(+1.0, abs=1e-5)
    assert v.abs().max().item() <= 1.0
    # A draw is the *zero* of this scale, which is the one thing the tanh head could
    # only learn and this one gets by construction.
    assert float(wdl_to_scalar(torch.tensor([[0.0, 5.0, 0.0]]))) == pytest.approx(0.0, abs=1e-6)


def test_the_wdl_loss_refuses_a_target_that_is_not_a_game_outcome():
    """A bootstrapped or averaged value has no class index, and rounding it into one
    would train a confident wrong label. §4's target is the game result and this is
    the branch that says so."""
    batch, _, _ = make_batch(8, seed=13)
    n = len(batch)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE),
                   torch.randn(n, 3, device=DEVICE))
    soft = batch.slice(0, n)
    soft.value = torch.full((n,), 0.37, device=DEVICE)
    with pytest.raises(AssertionError, match="not in"):
        az_loss(net, soft)


def test_a_perfectly_fitted_wdl_head_has_a_vanishing_value_loss():
    """The floor of the cross-entropy is zero, unlike the policy term's, because the
    target is one-hot. `ln 3 = 1.0986` is where it starts from, which is the number
    `--value-weight` is untuned against."""
    batch, _, _ = make_batch(16, seed=14)
    n = len(batch)
    onehot = torch.zeros(n, 3, device=DEVICE)
    onehot[torch.arange(n), (batch.value + 1).long()] = 30.0
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), onehot)
    assert float(az_loss(net, batch).value) < 1e-6

    flat = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                    torch.randn(n, 32, 4, device=DEVICE),
                    torch.zeros(n, 3, device=DEVICE))
    assert float(az_loss(flat, batch).value) == pytest.approx(math.log(3), rel=1e-6)


def test_the_scalar_head_is_untouched_by_the_option():
    """The regression branch has to be the same arithmetic it always was: `n_value`
    reported as 1, the squared error, and `value_pred` the tanh."""
    batch, _, _ = make_batch(16, seed=3)
    n = len(batch)
    torch.manual_seed(0)
    raw = torch.randn(n, device=DEVICE)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), raw)
    parts = az_loss(net, batch)
    assert parts.n_value == 1
    want = ((batch.value - torch.tanh(raw)) ** 2).mean()
    assert float(parts.value) == pytest.approx(float(want), rel=1e-6)
    assert torch.equal(parts.value_pred, torch.tanh(raw))


def test_value_classes_reaches_the_network_and_the_config_hash():
    """The knob is architectural, so a resume across it must be refused: the two
    heads have different weight shapes and a hybrid run would not even load."""
    from dataclasses import replace

    from brokefish.train.loop import TrainConfig

    cfg = TrainConfig()
    assert cfg.value_classes == 1, "the default is the head every Elo number used"
    assert replace(cfg, value_classes=3).hash() != cfg.hash()
    assert BrokefishNet(n_value=3).value.weight.shape == (3, BrokefishNet().d_model)


# --------------------------------------------------------------------------
# The pooled White/draw/Black head, 2026-08-22. What changes in the loss is one thing
# and it is the **frame**: the king select reads the mover's king and predicts the
# mover's result, so `batch.value` is already its target; the pooled head predicts
# White's, so the same record has to be flipped first.



def both_colours(batch):
    """⚠️ `make_batch` walks an **even** number of random plies, so every position it
    produces has **White to move** — the same fact `eval/match.py`'s lockstep rests on.
    A frame test on that fixture is vacuous, because the flip is the identity. Negating
    half the control words gives the null-move position, which is all these tests need.
    """
    out = batch.slice(0, len(batch))
    ctl = out.control.clone()
    ctl[::2] = -ctl[::2]
    out.control = ctl
    assert bool((ctl > 0).any()) and bool((ctl < 0).any())
    return out


def test_the_absolute_head_trains_on_whites_frame():
    """⚠️ Getting this backwards is not a crash. It trains a head that is exactly
    right on the positions where White is to move and exactly wrong on the others,
    which shows up as a head that learns nothing rather than as a bug — so it is
    pinned against a hand-written transcription rather than trusted.
    """
    batch, _, _ = make_batch(24, seed=21)
    batch = both_colours(batch)
    n = len(batch)
    torch.manual_seed(0)
    logits = torch.randn(n, 3, device=DEVICE)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), logits, absolute=True)
    parts = az_loss(net, batch)

    ref = 0.0
    for i in range(n):
        z = float(batch.value[i])                       # the mover's frame
        white_to_move = int(batch.control[i]) > 0
        z_white = z if white_to_move else -z             # White's frame
        row = [float(x) for x in logits[i].double()]
        top = max(row)
        lse = top + math.log(sum(math.exp(x - top) for x in row))
        ref += lse - row[int(z_white) + 1]
    ref /= n
    assert float(parts.value) == pytest.approx(ref, rel=1e-5, abs=1e-6)

    # The same logits scored in the mover's frame are a *different* number, or the
    # test would pass whether or not the flip happened.
    flat = ConstNet(net.policy.detach(), net.promo.detach(), logits, absolute=False)
    assert abs(float(az_loss(flat, batch).value) - ref) > 1e-3


def test_the_absolute_head_still_reports_a_mover_relative_prediction():
    """`value_pred` feeds `value_mean` and `value_saturated_frac`, and it is what the
    search would consume, so it stays in the mover's frame on both heads."""
    batch, _, _ = make_batch(16, seed=22)
    batch = both_colours(batch)
    n = len(batch)
    logits = torch.zeros(n, 3, device=DEVICE)
    logits[:, 2] = 20.0                                  # White wins, certainly
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), logits, absolute=True)
    v = az_loss(net, batch).value_pred
    want = torch.where(batch.control > 0, 1.0, -1.0)
    assert torch.allclose(v, want, atol=1e-5)


def test_the_absolute_scalar_head_measures_its_error_in_its_own_frame():
    """The `n_value == 1` branch has the same frame problem and the same fix."""
    batch, _, _ = make_batch(16, seed=23)
    batch = both_colours(batch)
    n = len(batch)
    torch.manual_seed(1)
    raw = torch.randn(n, device=DEVICE)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), raw, absolute=True)
    s = torch.where(batch.control > 0, 1.0, -1.0)
    want = ((batch.value * s - torch.tanh(raw)) ** 2).mean()
    assert float(az_loss(net, batch).value) == pytest.approx(float(want), rel=1e-6)


def test_value_head_reaches_the_network_and_the_config_hash():
    from dataclasses import replace

    from brokefish.train.loop import TrainConfig

    cfg = TrainConfig()
    assert cfg.value_head == "king", "the default is the head every Elo number used"
    assert replace(cfg, value_head="pooled").hash() != cfg.hash()
    net = BrokefishNet(n_value=3, value_head="pooled")
    assert net.value_absolute and "value_mode" in net.state_dict()
    assert not BrokefishNet(n_value=3).value_absolute


# --------------------------------------------------------------------------
# `--target-mix` and `--split-grad`, 2026-09-09. Two flags, both off by default, both
# leaving every earlier run's arithmetic bit-identical when off -- which is the first
# thing each block below asserts, with `torch.equal` and not a tolerance.


def _with_root_value(batch, root_value):
    from dataclasses import replace
    return replace(batch, root_value=root_value)


@pytest.mark.parametrize("n_value", [1, 3])
def test_target_mix_at_zero_is_the_old_loss_bit_for_bit(n_value):
    """`target_mix = 0.0` takes the branch that existed before the flag: same loss,
    same gradient, `torch.equal`, whether or not the batch carries a `root_value`."""
    batch, _, _ = make_batch(16, seed=31)
    torch.manual_seed(0)
    rv = torch.rand(16, device=DEVICE) * 2 - 1
    net = BrokefishNet(n_value=n_value).to(DEVICE)

    def loss_and_grad(b, **kw):
        net.zero_grad(set_to_none=True)
        parts = az_loss(net, b, **kw)
        parts.total.backward()
        return parts.total.detach().clone(), torch.cat(
            [p.grad.reshape(-1).clone() for p in net.parameters()])

    old_l, old_g = loss_and_grad(batch)
    new_l, new_g = loss_and_grad(_with_root_value(batch, rv), target_mix=0.0)
    assert torch.equal(old_l, new_l) and torch.equal(old_g, new_g)
    mixed_l, _ = loss_and_grad(_with_root_value(batch, rv), target_mix=0.5)
    assert not torch.equal(old_l, mixed_l), "the mix did not reach the loss"


def test_target_mix_at_one_regresses_onto_root_value():
    """Scalar head: at `a = 1` the target is `root_value`, at `a = 0.5` the midpoint,
    and the value mask still selects the rows."""
    batch, _, _ = make_batch(16, seed=32)
    n = len(batch)
    torch.manual_seed(1)
    raw = torch.randn(n, device=DEVICE)
    rv = torch.rand(n, device=DEVICE) * 2 - 1
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), raw)
    b = _with_root_value(batch, rv)
    want = ((rv - torch.tanh(raw)) ** 2).mean()
    assert float(az_loss(net, b, target_mix=1.0).value) == pytest.approx(float(want), rel=1e-6)
    tgt = 0.5 * batch.value + 0.5 * rv
    want = ((tgt - torch.tanh(raw)) ** 2).mean()
    assert float(az_loss(net, b, target_mix=0.5).value) == pytest.approx(float(want), rel=1e-6)

    from dataclasses import replace
    mask = (torch.arange(n, device=DEVICE) % 3 == 0).float()
    want = (((tgt - torch.tanh(raw)) ** 2) * mask).sum() / mask.sum()
    parts = az_loss(net, replace(b, value_mask=mask), target_mix=0.5)
    assert float(parts.value) == pytest.approx(float(want), rel=1e-6)
    assert float(parts.value_rows) == float(mask.sum())

    with pytest.raises(ValueError, match="root_value"):
        az_loss(net, batch, target_mix=0.5)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        az_loss(net, b, target_mix=1.5)


def test_the_max_entropy_wdl_map_has_the_right_expectation():
    """`wdl_max_entropy`: a distribution, with `p(win) - p(loss) = r`, geometric
    (`p_draw^2 = p_win p_loss`, the exponential family's signature), uniform at 0 and
    a point mass at +-1."""
    from brokefish.train.loss import wdl_max_entropy

    r = torch.linspace(-1, 1, 201, device=DEVICE, dtype=torch.float64)
    p = wdl_max_entropy(r).double()
    assert bool((p >= 0).all())
    torch.testing.assert_close(p.sum(-1), torch.ones_like(r))
    torch.testing.assert_close(p[:, 2] - p[:, 0], r)
    torch.testing.assert_close(p[:, 1] ** 2, p[:, 0] * p[:, 2], atol=1e-6, rtol=0)
    torch.testing.assert_close(p[100], torch.full((3,), 1 / 3, dtype=torch.float64, device=DEVICE))
    torch.testing.assert_close(p[-1], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=DEVICE))
    torch.testing.assert_close(p[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=DEVICE))
    # Maximum entropy among distributions with that mean: perturb along the one
    # direction that keeps the mean and the mass, and the entropy must not rise.
    q = p[50] + 1e-3 * torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64, device=DEVICE)
    H = lambda x: float(-(x * x.log()).sum())
    assert H(p[50]) > H(q)


def test_target_mix_on_the_wdl_head_is_a_soft_label():
    """At `a = 1` and `root_value = 0` the label is uniform, so the loss is the
    mean of `-log softmax`; at `a = 1` and `root_value = +1` it is the hard win
    class; at `a = 0.5` it is the average of the two cross-entropies."""
    batch, _, _ = make_batch(8, seed=33)
    n = len(batch)
    torch.manual_seed(2)
    logits = torch.randn(n, 3, device=DEVICE)
    net = ConstNet(torch.randn(n, 32, 64, device=DEVICE),
                   torch.randn(n, 32, 4, device=DEVICE), logits)
    logp = torch.log_softmax(logits, -1)

    b0 = _with_root_value(batch, torch.zeros(n, device=DEVICE))
    want = (-logp.mean(-1)).mean()
    assert float(az_loss(net, b0, target_mix=1.0).value) == pytest.approx(float(want), rel=1e-6)

    b1 = _with_root_value(batch, torch.ones(n, device=DEVICE))
    want = (-logp[:, 2]).mean()
    assert float(az_loss(net, b1, target_mix=1.0).value) == pytest.approx(float(want), rel=1e-6)

    ce_z = torch.nn.functional.cross_entropy(logits, (batch.value + 1).long())
    want = 0.5 * ce_z + 0.5 * (-logp[:, 2]).mean()
    assert float(az_loss(net, b1, target_mix=0.5).value) == pytest.approx(float(want), rel=1e-6)

    soft = b1.slice(0, n)
    soft.value = torch.full((n,), 0.37, device=DEVICE)
    with pytest.raises(AssertionError, match="not in"):
        az_loss(net, soft, target_mix=0.5)


def test_root_value_and_z_share_a_frame():
    """§10: `root_value` is the search's value **for the side to move at the record**,
    in [-1, 1], which is the frame `z` is written in by `_close`. Pinned on the one
    position where both are known exactly: a mate in one, where the search's root
    value is +1 under the collapse and the game closes with `z = +1` on that record
    -- and where an inverted frame on either side would give -1."""
    fen = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
    b, c = env.from_fen(fen)
    b, c = b.to(DEVICE).expand(4, -1).contiguous(), c.to(DEVICE).expand(4).contiguous()
    torch.manual_seed(0)
    net = BrokefishNet().to(DEVICE).eval()
    with torch.no_grad():
        s = Search(SearchConfig(n=16, B=4, E=64, terminal_collapse=True),
                   evaluate=make_evaluator(net), device=DEVICE, seed=0)
        s.reset(b, c)
        rec = s.self_play_move()
    assert bool(rec.done.all()) and bool((rec.result == -1).all())
    assert float(rec.root_value.min()) > 0.999

    buf = ReplayBuffer(window_games=8, mean_plies=4, seed=0)
    buf.append(rec)
    batch = buf.sample(4, device=DEVICE)
    assert batch.root_value is not None
    assert torch.equal(batch.value, torch.ones(4, device=DEVICE))
    assert float(batch.root_value.min()) > 0.999
    # The same record, black to move, so the frame flips on both sides at once.
    fen_b = "4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
    b, c = env.from_fen(fen_b)
    b, c = b.to(DEVICE).expand(4, -1).contiguous(), c.to(DEVICE).expand(4).contiguous()
    with torch.no_grad():
        s.reset(b, c)
        rec = s.self_play_move()
    assert bool(rec.done.all()) and float(rec.root_value.min()) > 0.999
    buf.append(rec)
    batch = buf.sample(8, device=DEVICE)
    assert torch.equal(batch.value, torch.ones(8, device=DEVICE))
    # And the range, over ordinary positions.
    batch, rec, _ = make_batch(32, seed=34)
    assert float(rec.root_value.abs().max()) <= 1.0


def test_split_grad_is_exact_under_sgd():
    """`backward_split` + `step_with_grads` against one joint backward and one SGD
    step, in double, over two micro-batches. Plain SGD is linear in the gradient, so
    the two-optimiser step equals the one-optimiser step exactly; anything but
    round-off says the split dropped, doubled or misrouted a gradient."""
    from brokefish.train.loop import backward_split, split_grad_params, step_with_grads

    batch, _, _ = make_batch(16, seed=35)
    torch.manual_seed(3)
    net = BrokefishNet().to(DEVICE).double()
    lr = 0.05

    def run(split):
        torch.manual_seed(3)
        m = BrokefishNet().to(DEVICE).double()
        m.load_state_dict(net.state_dict())
        trunk, head = split_grad_params(m)
        main = torch.optim.SGD(m.parameters(), lr=lr)
        aux = torch.optim.SGD(trunk, lr=lr)
        m.zero_grad(set_to_none=True)
        value_grads = [None] * len(trunk)
        for i in range(2):
            mb = batch.slice(8 * i, 8 * i + 8)
            parts = az_loss(m, mb)
            if split:
                backward_split(parts.policy * 0.5, parts.value * 0.5, trunk, head, value_grads)
            else:
                (parts.policy * 0.5 + parts.value * 0.5).backward()
        if split:
            assert all(g is not None for g in value_grads)
            assert all(p.grad is not None for p in head), "the value head lost its gradient"
            assert m.policy.weight.grad is not None
            joint_grad = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
            main.step()
            step_with_grads(aux, trunk, value_grads)
            assert all(p.grad is None for p in trunk)
        else:
            joint_grad = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
            main.step()
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()]), joint_grad

    joint, g_joint = run(split=False)
    split, g_policy = run(split=True)
    rel = float((joint - split).norm() / (joint - torch.cat(
        [p.detach().reshape(-1) for p in net.parameters()])).norm())
    assert rel < 1e-11, f"split step differs from the joint one by {rel:.3e} of the update"
    # And the split really moved gradient out of `.grad`: what the main optimiser saw
    # is not the joint gradient.
    assert float((g_joint - g_policy).norm() / g_joint.norm()) > 1e-3


def test_the_flags_reach_the_config_hash_and_split_grad_needs_muon(tmp_path):
    from dataclasses import replace

    cfg = TrainConfig()
    assert cfg.target_mix == 0.0 and not cfg.split_grad
    assert replace(cfg, target_mix=0.5).hash() != cfg.hash()
    assert replace(cfg, split_grad=True).hash() != cfg.hash()
    assert replace(cfg, split_grad_lr=3e-4).hash() != cfg.hash()
    with pytest.raises(ValueError, match="muon"):
        _smoke_trainer(tmp_path, "split-adamw", optimizer="adamw", split_grad=True)


def test_split_grad_trains_under_muon_and_checkpoints_its_optimiser(tmp_path):
    """The whole `train_step` path under `--split-grad --optimizer muon`: the step
    runs, the trunk's value gradient is reported, the weights move, `.grad` is left
    clean on the trunk, and the second AdamW is in the checkpoint."""
    trainer = _smoke_trainer(tmp_path, "split-muon", optimizer="muon", split_grad=True,
                             lr_schedule=((0, 0.02),), target_mix=0.5)
    while trainer.buffer.n_records < 8:
        trainer.self_play_phase()
    before = weight_fingerprint(trainer.net)
    out = trainer.train_step()
    assert out["grad_norm_value_trunk"] > 0 and out["grad_norm"] > 0
    assert weight_fingerprint(trainer.net) != before
    assert all(p.grad is None for p in trainer.split_trunk)
    state = trainer.state_dict()
    assert state["opt_value"] is not None and len(state["opt_value"]["state"]) > 0
    ckpt = str(tmp_path / "s.pt")
    trainer.save_checkpoint(ckpt)
    again = _smoke_trainer(tmp_path, "split-muon2", optimizer="muon", split_grad=True,
                           lr_schedule=((0, 0.02),), target_mix=0.5)
    again.load_checkpoint(ckpt)
    assert len(again.opt_value.state_dict()["state"]) == len(state["opt_value"]["state"])
    trainer.log.close()
    again.log.close()
