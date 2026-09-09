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
from typing import Optional

import torch
import torch.nn.functional as F

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
    #: [N] float32 in {0, 1}: which rows the value loss is taken over. ``None`` is
    #: every row (the buffer writes it; hand-built batches may omit it).
    value_mask: Optional[torch.Tensor] = None
    #: [N] float32 in [-1, 1]: the search's own value of the root, search.md §10's
    #: `root_value`, **in the same frame as `value`** -- the side to move at the
    #: record (`test_train.py::test_root_value_and_z_share_a_frame`). ``None`` when
    #: the batch was hand-built without it; only read under ``target_mix > 0``.
    root_value: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.board.shape[0])

    def slice(self, lo: int, hi: int) -> "TrainBatch":
        m = None if self.value_mask is None else self.value_mask[lo:hi]
        r = None if self.root_value is None else self.root_value[lo:hi]
        return TrainBatch(*(t[lo:hi] for t in (
            self.board, self.control, self.rep, self.policy_move,
            self.policy_prob, self.policy_len, self.value, self.weight_gen)),
            value_mask=m, root_value=r)


@dataclass
class LossParts:
    """Undetached tensors. ``total`` is what to call ``backward`` on; the rest are
    for §11 and are accumulated by the caller so one synchronisation covers a step."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor      # H(pi), the target's own; CE's floor
    kl: torch.Tensor           # CE - H(pi), which goes to zero at a perfect fit
    value_pred: torch.Tensor   # [N] in [-1, 1], for the calibration scalars
    #: How many rows the value head that produced this has: 1 = scalar tanh + MSE,
    #: 3 = win/draw/loss + cross-entropy. ⚠️ `value` is **not comparable across the
    #: two** -- an MSE against z in {-1, 0, 1} starts near 1.0 and a 3-class CE starts
    #: at ln 3 = 1.0986, and they are different quantities that happen to look alike.
    n_value: int = 1
    #: RMS of the raw value logits. ⚠️ It is here because the **pooled** head's input
    #: is not unit-scale: `norm_f` normalises each token, but the mean of several
    #: normed vectors is not normed, and its magnitude moves with how aligned the
    #: tokens are and with **how many pieces are alive**. So the head's input shrinks
    #: and drifts across a game in a way a single king token never did. This is the
    #: cheap proxy for that drift; it is not the pooled vector's own norm.
    value_logit_rms: Optional[torch.Tensor] = None
    #: 0-d float: how many rows of this (micro-)batch the value term averaged over
    #: -- the mask's sum, or N without a mask. `value` is the mean over these rows,
    #: so accumulating it across micro-batches weights it by this and **not** by the
    #: row count; see `micro_batch_weights`.
    value_rows: Optional[torch.Tensor] = None


def value_rows_of(batch: TrainBatch) -> torch.Tensor:
    """How many rows of ``batch`` the value loss is taken over, as a 0-d tensor."""
    if batch.value_mask is None:
        return torch.tensor(float(len(batch)), device=batch.board.device)
    return batch.value_mask.float().sum()


def micro_batch_weights(parts: LossParts, n_rows: int, total_rows: int,
                        total_value_rows: torch.Tensor):
    """The two scalars that make gradient accumulation exact (training.md §7.2).

    Returns ``(w_policy, w_value)``. ``parts.policy`` (and ``entropy``, ``kl``) is a
    mean over the micro-batch's rows, so ``n_rows / total_rows`` of it is its share
    of the mean over the whole batch. ``parts.value`` is a mean over the
    **value-supervised** rows only, so its share is ``value_rows / total_value_rows``.
    ⚠️ Scaling both by the row count -- what `train_step` did until 2026-09-09 --
    is exact for the policy and wrong for the value whenever the supervised
    fraction differs between micro-batches: a micro-batch holding one supervised row
    got the same weight as one holding two, and `docs/core-algorithm-review.md` §3
    shows the value gradient cancelling to zero on four rows where the whole-batch
    gradient is -2/3. A whole batch with no supervised row gives ``w_value = 0``
    everywhere, which is the only weight that batch's value term can have.
    """
    w_policy = n_rows / total_rows
    rows = parts.value_rows
    if rows is None:
        rows = torch.tensor(float(n_rows), device=total_value_rows.device)
    w_value = rows / total_value_rows.clamp(min=1.0)
    return w_policy, w_value


def wdl_max_entropy(r: torch.Tensor) -> torch.Tensor:
    """``[N] in [-1, 1]`` to ``[N, 3]``: the maximum-entropy win/draw/loss distribution
    whose expectation ``p(win) - p(loss)`` is ``r``. Columns are (loss, draw, win),
    the class order `az_loss` trains the WDL head in.

    The record stores the search's root value as one scalar and not as three
    probabilities, so a soft WDL label built from it has to pick *some* distribution
    with that expectation. Maximum entropy is the one that adds no information the
    scalar does not carry. It is well defined and unique: on the support ``{-1, 0, +1}``
    the max-entropy distribution with a fixed mean is the exponential family
    ``p_k ∝ exp(lambda k)``, which has ``p_draw^2 = p_win p_loss``. With
    ``s = p_win + p_loss`` that is ``(s^2 - r^2) / 4 = (1 - s)^2``, whose root in
    ``[|r|, 1]`` is

        s = (4 - sqrt(4 - 3 r^2)) / 3

    so ``p_draw = 1 - s``, ``p_win = (s + r) / 2``, ``p_loss = (s - r) / 2``. No
    solve for ``lambda``, no clamp: ``r = 0`` gives the uniform third, ``r = +-1``
    the point mass, and the radicand is at least 1 on the whole range.

    ⚠️ What it is not: the search's actual draw probability. A root the search rates
    at 0 is given a draw mass of 1/3, whether it is a dead draw or a wild position
    the tree cannot decide. The scalar cannot tell the two apart, and storing the
    three probabilities is a record-format change this function is here to avoid.
    """
    r = r.float()
    s = (4.0 - torch.sqrt(4.0 - 3.0 * r * r)) / 3.0
    return torch.stack([(s - r) / 2.0, 1.0 - s, (s + r) / 2.0], dim=-1)


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
            value_weight: float = 1.0, target_mix: float = 0.0) -> LossParts:
    """AZ eq. (1) without the L2 term, for one (micro-)batch.

    ``target_mix`` is ``--target-mix``: the value target becomes
    ``(1 - a) * z + a * root_value``, the outcome mixed with the search's own value
    of the root (search.md §10, `journal/2026-08-24-the-value-head-is-a-calibration-
    failure.md`). ``0.0`` is every run on the ledger and takes the branch that
    existed before the flag, so it is bit-identical rather than merely equal, and
    reads ``batch.root_value`` not at all. On the scalar head the mix is a plain
    regression target. On the WDL head it is a soft label,
    ``(1 - a) * onehot(z) + a * q(root_value)``, with ``q`` the maximum-entropy
    distribution of :func:`wdl_max_entropy`, trained with the soft-label
    cross-entropy ``-sum_k p_k log softmax_k``.

    ⚠️ **The value term has two forms and the network picks which.** At
    ``net.n_value == 1`` it is AZ's ``(z - v)^2`` on the scalar tanh head, unchanged
    and bit-identical to every run on the ledger. At ``net.n_value == 3`` it is a
    3-class cross-entropy on a win/draw/loss classifier, which is what KataGo and
    Leela train, with the stored outcome ``z in {-1, 0, +1}`` as the class index
    ``z + 1``. Nothing else in the loss moves, and nothing outside it moves at all:
    the head still hands the search a scalar (`nn/model.py:wdl_to_scalar`).

    ⚠️ **``value_weight`` does not mean the same thing on the two branches.** The MSE
    on ``z in {-1, 0, 1}`` starts near 1.0 and falls to ~0.3; the CE starts at
    ``ln 3 = 1.0986``. 1.0 is AlphaZero's number for the squared error and is *not*
    a calibrated number for the cross-entropy -- it is the untuned default, and that
    is a thing this run will be measuring rather than a thing it has settled.

    ``value_weight`` scales the value term against the policy
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
    policy_logits, promo_logits, value_pred, value_logits = net(
        batch.board, batch.control, batch.rep, with_logits=True)
    n_value = value_logits.shape[-1]
    # ⚠️ **Which frame the head predicts in decides what the target is.** The king
    # select reads the *mover's* king and predicts the mover's result, so `batch.value`
    # is already in its frame. The pooled head reads every token symmetrically and
    # predicts White's, so the same record has to be flipped into White's frame before
    # it can be a target. Getting this backwards trains a head that is exactly wrong on
    # half the positions and right on the other half, which looks like a head that
    # learns nothing rather than like a bug.
    absolute = bool(getattr(net, "value_absolute", False))

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
        # `+1` where White is to move, `-1` where Black is (spec §2.4: the control
        # word's magnitude is the clock, so it is never 0 and the sign never vanishes).
        flip = torch.where(batch.control > 0, 1.0, -1.0) if absolute else None
        # The value term is a mean over the *masked* rows: with every row marked this
        # is the plain mean and the loss is unchanged; with 1 in k marked, the term
        # keeps its magnitude and the unmarked rows contribute no value gradient at
        # all. Per micro-batch, so `train_step` weights it by `value_rows` and not by
        # the row count -- `micro_batch_weights` -- or the accumulated gradient is not
        # the whole batch's (`docs/core-algorithm-review.md` §3).
        mask = batch.value_mask
        if mask is None:
            mask = torch.ones_like(v)
        mask = mask.float()
        value_rows = mask.sum()
        m_sum = value_rows.clamp(min=1.0)
        if not 0.0 <= target_mix <= 1.0:
            raise ValueError(f"target_mix must be in [0, 1], got {target_mix}")
        if target_mix > 0.0 and batch.root_value is None:
            raise ValueError(
                "target_mix > 0 needs `batch.root_value`, and this batch carries none. "
                "`ReplayBuffer.sample` writes it; a hand-built TrainBatch has to")
        # Both `z` and `root_value` are in the mover's frame and in [-1, 1] (§10,
        # pinned by `test_root_value_and_z_share_a_frame`), so the mix is a convex
        # combination in one frame, and flipping it into White's for the pooled head
        # is the same `flip` on both. The branch keeps `target_mix = 0` on the exact
        # arithmetic of every earlier run: `1.0 * z + 0.0 * r` is `z` in fp32 too,
        # but only while `r` is finite, and a bit-identical claim should not rest on that.
        if target_mix > 0.0:
            z_mix = (1.0 - target_mix) * batch.value + target_mix * batch.root_value.float()
        else:
            z_mix = batch.value
        if n_value == 1:
            # `v * flip` un-does `heads`' flip, i.e. it is the head's own output again.
            tgt = z_mix if flip is None else z_mix * flip
            pred = v if flip is None else v * flip
            value_loss = value_weight * (((tgt - pred) ** 2) * mask).sum() / m_sum
        elif target_mix > 0.0:
            # Soft label: the outcome's one-hot mixed with the max-entropy WDL
            # distribution that has the search's root value as its expectation. `z`
            # itself still has to be a game outcome, the same check as below.
            z = batch.value if flip is None else batch.value * flip
            r = batch.root_value.float() if flip is None else batch.root_value.float() * flip
            cls = (z + 1.0).round()
            if not bool(((cls - (z + 1.0)).abs() < 1e-6).all()):
                raise AssertionError(
                    "a value target is not in {-1, 0, +1}, so it has no class index "
                    "to mix the search's value into")
            onehot = F.one_hot(cls.long(), 3).float()
            p = (1.0 - target_mix) * onehot + target_mix * wdl_max_entropy(r)
            logp = torch.log_softmax(value_logits.float(), dim=-1)
            value_loss = value_weight * (-(p * logp).sum(-1) * mask).sum() / m_sum
        else:
            # ⚠️ `batch.value` is exactly {-1, 0, +1} by construction -- `buffer.py:253`
            # writes `-result * s_rec * s_end` and every factor is a sign -- so `+1` is
            # an exact class index and not a rounding. `round()` is here so that a
            # future bootstrapped target fails the check below rather than silently
            # truncating toward a class it does not mean.
            z = batch.value if flip is None else batch.value * flip
            cls = (z + 1.0).round()
            if not bool(((cls - (z + 1.0)).abs() < 1e-6).all()):
                raise AssertionError(
                    "a value target is not in {-1, 0, +1}, so it has no class index. "
                    "The win/draw/loss head is trained on the game outcome (§4); a "
                    "bootstrapped or averaged target needs a soft-label loss, not this "
                    "one")
            value_loss = value_weight * (F.cross_entropy(
                value_logits.float(), cls.long(), reduction="none") * mask).sum() / m_sum

    return LossParts(total=policy_loss + value_loss, policy=policy_loss,
                     value=value_loss, entropy=entropy, kl=policy_loss - entropy,
                     value_pred=v.detach(), n_value=n_value,
                     value_logit_rms=value_logits.detach().float().pow(2).mean().sqrt(),
                     value_rows=value_rows.detach())


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
