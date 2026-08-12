"""The AlphaZero loss of ``docs/train.md`` §3, and the label decode it turns on.

AZ eq. (1), unchanged::

    l = (z - v)^2 - pi^T log p + c ||theta||^2

with the cross-entropy and the mean-squared error weighted equally (AGZ Methods,
Optimisation) and the L2 term left to the optimiser (§7.1, and :func:`l2_penalty`
here only so it can be *logged*).

Three things in this module are load-bearing and are the reason it is not four
lines of ``F.cross_entropy``.

**The softmax denominator is the root's edge set, and the record carries it** (§3.5,
revised 2026-07-31). The training path recomputes nothing about the position: the
search already knew exactly which moves it normalised over, and since ``mcts.md`` §10's
policy arrays are ``E = K_POLICY`` wide whether or not the entries are used, storing every
edge instead of only the visited ones costs **zero extra bytes**. What was a
``[N, 8192]`` masked softmax over a recomputed legality mask is now an ``[N, K_POLICY]``
gather, and ``env`` no longer appears in this module's hot path at all.

⚠️ **The unvisited edges are the point.** They carry ``pi = 0`` and contribute nothing
to ``-sum pi log p`` directly, and everything through the denominator. Restricting the
softmax to the ``N(a) > 0`` moves optimises a different objective, and every loss value
it produces looks reasonable — which is why the record stores the edge count and not
the visit count.

**The label decode is an independent transcription** of ``search/torch_impl.py``'s
``_expand``, not a shared helper. A mismatch between the two permutes the target
silently: the network learns a consistently shuffled labelling, the loss falls the
whole time, and nothing else in ``train.md`` §12 notices. Two transcriptions plus a
differential test against a live search is the same discipline ``test_search.py``
used on ``ucb_score``, and it is check 3 — the most important check in C2.

**The ``E`` truncation stops being a problem rather than being solved.** An
earlier draft recomputed the legality mask and re-derived the top-``E`` support, which
cannot be done correctly — the search truncated with the *generating* weights and the
training path only has the *current* ones, and the two disagreed within eight
generations of a smoke run, on ``r6r/1ppq1kpp/5n2/2n1pbP1/7P/1P1P1N2/PpP1PK2/2R2Q1R
b - - 1 1``, where the search kept ``b2b1=N`` and a recomputed top-``E`` dropped it.
Reading the support out of the record makes the question disappear: the stored edges
*are* the support, exactly, by construction, whichever weights chose them.

⚠️ The residual difference from AZ is that on a position with more legal moves than the
cap the denominator is the search's support rather than every legal move, which is what AZ
Methods renormalises over. Softmax is consistent under restriction, so training the
distribution the search actually consumes is coherent; it is recorded in
[`fidelity.md`](fidelity.md) §4.2(i) as ours and not the paper's.

**One thing the label alone cannot tell you, and the board can.** ``promo == 0`` does
not mean "not a promotion" — spec §3 numbers the types ``0:N 1:B 2:R 3:Q`` and a quiet
move also carries field 0, so testing ``promo == 0`` silently deletes the whole
underpromotion motif. Whether the promotion term belongs in an edge's logit is decided
from the *piece word and the target rank*, which the record's board carries, and never
from the label. :func:`is_promotion_edge` is that decision and it needs no movegen.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

# spec §3: a move is 11 bits and the promotion choice rides above it, so one
# int16 carries a whole edge label. Same constants as the search, restated rather
# than imported, because this file exists to be a second opinion about them.
MOVE_BITS = 11
PROMO_SHIFT = MOVE_BITS
MOVE_MASK = (1 << MOVE_BITS) - 1

N_SLOTS, N_SQUARES, N_PROMO = 32, 64, 4

PAWN = 0   # spec §2.1's type field; env.torch_impl.PAWN, restated for the same reason


@dataclass
class TrainBatch:
    """One sampled minibatch, on device. The record of §5.1 minus what it does not need."""

    board: torch.Tensor        # [N, 32] int16
    control: torch.Tensor      # [N]     int16
    rep: torch.Tensor          # [N]     uint8
    policy_move: torch.Tensor  # [N, K_POLICY] int16, spec §3 labels
    policy_prob: torch.Tensor  # [N, K_POLICY] float16, pi = N(a)/n
    policy_len: torch.Tensor   # [N]     uint8
    value: torch.Tensor        # [N]     float32, z in {-1, 0, +1}
    weight_gen: torch.Tensor   # [N]     int32, for the staleness counter of §11

    def __len__(self) -> int:
        return int(self.board.shape[0])

    def slice(self, lo: int, hi: int) -> "TrainBatch":
        return TrainBatch(*(t[lo:hi] for t in (
            self.board, self.control, self.rep, self.policy_move,
            self.policy_prob, self.policy_len, self.value, self.weight_gen)))


@dataclass
class LossParts:
    """Undetached tensors. ``total`` is what to call ``backward`` on; the rest are
    for §11 and are accumulated by the caller so one synchronisation covers a step."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor      # H(pi), the target's own; CE's floor
    kl: torch.Tensor           # CE - H(pi), which goes to zero at a perfect fit
    value_pred: torch.Tensor   # [N], for the calibration scalars


def decode_labels(policy_move: torch.Tensor):
    """spec §3 labels ``[N, K] int16`` to ``(slot, square, promo)``, each ``[N, K]``."""
    lab = policy_move.to(torch.int64)
    move = lab & MOVE_MASK
    return move // N_SQUARES, move % N_SQUARES, (lab >> PROMO_SHIFT) & 0b11


def is_promotion_edge(boards: torch.Tensor, control: torch.Tensor,
                      slot: torch.Tensor, square: torch.Tensor) -> torch.Tensor:
    """``[N, K] bool``: does edge ``slot -> square`` promote, given the position?

    ⚠️ **This cannot be read off the label** and that is the whole reason the
    function exists. spec §3 numbers the promotion types ``0:N 1:B 2:R 3:Q`` and a
    quiet move carries field 0 too, so ``promo == 0`` is ambiguous between "knight
    promotion" and "not a promotion". The answer comes from the piece word and the
    target rank, and it decides whether the ``log_softmax(promo)`` term belongs in
    this edge's logit — ``_expand`` makes the same decision the same way.

    The type is read from the piece word, never from the slot index: a promoted
    queen keeps its pawn slot, so a slot in 0-7 does not imply a pawn (spec §2.1).
    """
    w = boards.gather(1, slot).to(torch.int32)
    pawn = (((w >> 6) & 0b111) == PAWN) & (((w >> 11) & 1) == 0)
    last_rank = torch.where(control < 0, 0, 7).long()
    return pawn & ((square // 8) == last_rank[:, None])


def edge_logits(policy_logits: torch.Tensor, promo_logits: torch.Tensor,
                boards: torch.Tensor, control: torch.Tensor,
                policy_move: torch.Tensor) -> torch.Tensor:
    """``[N, K] fp32``: the logit of each stored edge, in ``_expand``'s own arithmetic.

    §6.4's log-space trick: a promotion edge's logit is the target square's logit
    plus ``log_softmax(promo)[k]``, so one softmax over the edges normalises both
    ``P(target | piece)`` and ``P(type | piece)`` at once.

    A gather over ``K <= K_POLICY`` edges rather than a mask over all 8192 labels, because
    the record carries the support (§3.5). At ``N = 1024`` that is a ``[1024, 96]``
    tensor where the masked form built two ``[1024, 8192]`` ones.
    """
    n = boards.shape[0]
    slot, square, promo = decode_labels(policy_move)

    base = policy_logits.float().reshape(n, -1).gather(1, slot * N_SQUARES + square)
    lp = torch.log_softmax(promo_logits.float(), dim=-1).reshape(n, -1)
    lp_at = lp.gather(1, slot * N_PROMO + promo)
    promotes = is_promotion_edge(boards, control, slot, square)
    return base + torch.where(promotes, lp_at, torch.zeros_like(lp_at))


def az_loss(net, batch: TrainBatch, strict: bool = True,
            value_weight: float = 1.0) -> LossParts:
    """AZ eq. (1) without the L2 term, for one (micro-)batch.

    ``value_weight`` scales the squared-error term against the policy
    cross-entropy. **1.0 is AlphaZero's and this project's**, and is the default, so
    nothing measured before it moves.

    ⚠️ It exists because **AlphaGateau's code and its own paper disagree**. Their
    eq. (10) is ``-pi^T log(pi~) + (v - v~)^2``, unweighted; their `train.py:275`
    calls ``optax.l2_loss``, which is ``0.5 * (x - y)^2``. So the run that produced
    their published numbers weighted value at **half** the policy, and reproducing
    the paper's formula would not reproduce the experiment. 0.5 is the faithful
    value.

    ⚠️ **No ``env``.** The record carries its own support (§3.5), so the loss touches
    no engine, no legality mask and no move generator. Whether that is *correct* is
    what :func:`audit_labels` checks, and it is a periodic audit rather than a step
    cost.

    ``strict`` costs one host synchronisation and asserts the invariants that survive
    without the engine: a stored edge names a live piece, and ``pi`` is a distribution
    over the stored support. Neither can catch a label decode that drifted apart from
    ``_expand``'s — for that there is :func:`audit_labels` and, definitively, §12
    check 3's comparison against a live search.
    """
    policy_logits, promo_logits, value_pred = net(batch.board, batch.control, batch.rep)

    # The loss is computed outside autocast on purpose (§8.2): it is a reduction
    # over <= K_POLICY logits and one position, so bf16 saves nothing measurable and
    # costs precision on the exact quantity being optimised.
    with torch.autocast(device_type=batch.board.device.type, enabled=False):
        k = batch.policy_move.shape[1]
        valid = (torch.arange(k, device=batch.board.device)[None, :]
                 < batch.policy_len.long()[:, None])
        logit = edge_logits(policy_logits, promo_logits, batch.board,
                            batch.control, batch.policy_move)
        # The denominator is the stored edge set and nothing else. Padding entries
        # are excluded rather than left at whatever a zero label decodes to — which
        # is `slot 0 -> a1`, a real logit, and would quietly join every denominator.
        logp = torch.log_softmax(
            torch.where(valid, logit, torch.full_like(logit, float("-inf"))), dim=-1)

        if strict:
            _check(batch, valid, logit)

        # pi is stored fp16 and renormalised here rather than used raw: N(a)/n sums
        # to one exactly in fp32 and to 1 +- 1e-3 after the cast, which would put a
        # floating floor under the KL and under §12 check 1's target of zero.
        pi = torch.where(valid, batch.policy_prob.float(), torch.zeros_like(logit))
        pi = pi / pi.sum(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
        logp = torch.where(valid, logp, torch.zeros_like(logp))

        policy_loss = -(pi * logp).sum(-1).mean()
        entropy = -(pi * pi.clamp(min=torch.finfo(torch.float32).tiny).log()).sum(-1).mean()
        v = value_pred.float()
        value_loss = value_weight * ((batch.value - v) ** 2).mean()

    return LossParts(total=policy_loss + value_loss, policy=policy_loss,
                     value=value_loss, entropy=entropy, kl=policy_loss - entropy,
                     value_pred=v.detach())


def _check(batch: TrainBatch, valid: torch.Tensor, logit: torch.Tensor) -> None:
    """What can be asserted about a record without consulting the engine."""
    if not bool((batch.policy_len > 0).all()):
        raise AssertionError(
            "a record has policy_len = 0: a non-terminal position was searched and "
            "produced no edges, or an unfilled buffer slot was sampled (an all-zero "
            "record decodes as a live white pawn on a1, not as an error)")
    slot, _, _ = decode_labels(batch.policy_move)
    dead = valid & (((batch.board.gather(1, slot).to(torch.int32) >> 11) & 1) == 1)
    if bool(dead.any()):
        row = int(dead.any(-1).nonzero()[0])
        raise AssertionError(
            f"train.md §12 check 3: {int(dead.sum())} stored edges move a captured "
            f"piece. First offending row {row}. Either the label decode here and in "
            f"_expand have drifted apart, or the record's board is not the position "
            f"the search ran on.")
    mass = torch.where(valid, batch.policy_prob.float(), torch.zeros_like(logit)).sum(-1)
    if not bool(((mass - 1.0).abs() < 5e-2).all()):
        raise AssertionError(
            f"pi does not sum to 1 over the stored support: worst row is "
            f"{float((mass - 1.0).abs().max()) + 1.0:.4f}. The visit counts are "
            f"normalised by the search, so this is a record that was written or read "
            f"wrong, not a rounding question.")


@torch.no_grad()
def audit_labels(batch: TrainBatch, env) -> int:
    """§12 check 3 as a periodic audit: is every stored edge actually legal?

    The one thing the loss gave up by reading its support out of the record is the
    cross-check against the engine, so it comes back here — at a movegen per sample,
    which is 0.13 % of a step and therefore affordable every hundredth one rather
    than never. Returns the number of offending edges and raises if there are any.

    ⚠️ It checks *legality*, not the *order* — a permutation of the edge array within
    a position is legal and would still be a shuffled target. Only the comparison
    against a live search settles that, and that lives in ``tests/test_train.py``.
    """
    mask, _ = env.movegen(batch.board, batch.control)
    legal = env.bitset_to_bool(mask).reshape(len(batch), -1)
    slot, square, _ = decode_labels(batch.policy_move)
    k = batch.policy_move.shape[1]
    valid = (torch.arange(k, device=batch.board.device)[None, :]
             < batch.policy_len.long()[:, None])
    ok = legal.gather(1, slot * N_SQUARES + square)
    bad = valid & ~ok
    n = int(bad.sum())
    if n:
        row = int(bad.any(-1).nonzero()[0])
        raise AssertionError(
            f"train.md §12 check 3: {n} stored edges are illegal in the recomputed "
            f"position. First offending row {row}, labels "
            f"{batch.policy_move[row][bad[row]].tolist()[:8]}. The record's board is "
            f"not the position the search generated its edges from.")
    return n


def l2_penalty(net, c: float) -> torch.Tensor:
    """``c ||theta||^2``, for logging only.

    ⚠️ It is **not** added to the loss. §7.1: the term is applied once per
    optimiser step through ``weight_decay``, not once per micro-batch, or its
    effective strength becomes a function of the accumulation count.
    """
    with torch.no_grad():
        total = sum((p.float() ** 2).sum() for p in net.parameters())
    return c * total


def weight_decay_for(c: float) -> float:
    """AGZ's ``c`` as PyTorch's ``weight_decay``, which is **twice** it.

    ⚠️ AGZ writes the penalty as ``c ||theta||^2`` with no factor of a half, so its
    gradient is ``2 c theta``. ``torch.optim.SGD(weight_decay=w)`` adds ``w theta``.
    Passing ``c`` straight through halves the regularisation the paper specifies —
    a silent factor of two that no test in §12 would catch, since it changes
    nothing except a curve months later.
    """
    return 2.0 * c
