"""The network of spec section 7, end to end: boards in, logits out.

This module is the oracle. It is plain PyTorch, it is never used in the hot path,
and every fused implementation is checked against it. The fused kernels own the
same three stages:

    embed(boards, control, rep) -> [N, 32, 256]      spec 7.2
    the pre-norm encoder stack                       spec 7.1
    norm_f, then the three heads                     spec 7.4

Two things here are not in the original spec text and were settled on 2026-07-29.

**There is a final LayerNorm.** The stack is pre-norm, so its output is a raw
residual stream whose scale grows with depth, and it is fp16. Feeding that
straight to a linear head is the classic pre-norm mistake, and it also loses the
kernel a padded shared-memory buffer to read from. `norm_f` fixes both.

**Nothing is masked here.** The policy output is raw logits. The legality mask
lives with the search, which needs a masked softmax anyway; applying it here
would mean either doing it twice or constraining what the search can fuse. Spec
7.4's "device-side AND, no host round-trip" is satisfied wherever that op runs,
as long as it runs on the device.

The heads are biasless. `norm_f` has affine, so `beta` is a learned 256-vector
every head sees and each head's effective bias is `W @ beta` -- any vector in the
head's output space, since the heads are all wider in than out. What is given up
is independence between the three heads' biases, not expressiveness.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

D_MODEL, N_LAYERS, N_HEADS, D_FF, T = 256, 8, 8, 1024, 32

# Embedding table sizes, spec 7.2. The flattening of the two-dimensional tables
# is normative because the kernel computes the same index arithmetic:
#   type_special = type * 2 + special      (6 types x 2)
#   color_turn   = color * 2 + stm         (2 colors x 2)
N_SQUARE, N_TYPE_SPECIAL, N_COLOR_TURN, N_CLOCK, N_REP = 64, 12, 4, 101, 3

KING_SLOT_WHITE, KING_SLOT_BLACK = 15, 31
N_POLICY, N_PROMO = 64, 4

#: How many rows the value head has. ``1`` is the scalar ``tanh`` regression head of
#: spec §7.4 and of every number on the ledger before 2026-08-21; ``3`` is the
#: win/draw/loss classifier (KataGo's and Leela's shape).
#:
#: ⚠️ **The class order is normative and the CUDA epilogue depends on it**: row 0 is
#: *loss*, row 1 is *draw*, row 2 is *win*, all from the side to move, so the class
#: index of a stored outcome ``z in {-1, 0, +1}`` is ``z + 1``.
N_VALUE_SCALAR, N_VALUE_WDL = 1, 3
VALUE_CLASSES = (N_VALUE_SCALAR, N_VALUE_WDL)

#: Where the value head reads from, and therefore which frame it predicts in.
#:
#: ``"king"`` is spec §7.4's original: a **row select** of the side-to-move king's
#: token, slot 15 or 31, predicting from the **mover's** point of view.
#:
#: ``"pooled"`` is the masked mean of *every live token, both colours*, predicting in
#: an **absolute** frame -- White / draw / Black -- which the collapse then flips into
#: the mover's by the sign of the control word.
#:
#: ⚠️ The hypothesis it exists to test (2026-08-22): the policy head is applied to all
#: 32 tokens, so every token takes policy gradient directly, while the king select
#: sends value gradient into *one* token and everything else is reached only through
#: that token's attention. A bottleneck, not a disconnection -- 8 non-causal layers do
#: carry it -- but a bottleneck the pooled head removes.
VALUE_HEADS = ("king", "pooled")


def wdl_to_scalar(logits: torch.Tensor) -> torch.Tensor:
    """``[N, 3]`` logits to the ``[N]`` scalar in ``[-1, 1]`` every consumer expects.

    ``p(win) - p(loss)``, which is the expected result under a 1/0.5/0 scoring with
    the draw mass dropping out. It is what keeps this an *option* rather than a
    rewrite: the search (`search.cuh:1076`), the terminal collapse, the value-head
    puzzle probe and the Elo pipeline all consume a single fp32 in ``[-1, 1]`` and
    none of them can tell which head produced it.

    ⚠️ It also throws information away on purpose. A draw-aware search — KataGo's
    utility with a separate draw term — is a **different** knob and would move the
    search as well as the head; one variable at a time.
    """
    p = torch.softmax(logits, dim=-1)
    return p[..., 2] - p[..., 0]


def n_value_of(state: dict) -> int:
    """How many value rows a ``state_dict`` was trained with.

    ⚠️ **This is the whole of the retro-compatibility story.** Checkpoints are bare
    ``state_dict``s with the architecture hardcoded in the constructor
    (`eval/league.py:157`), so nothing in a file says which head it has except the
    shape of ``value.weight`` — and every checkpoint on the ledger, `anchor.pt`
    included, carries ``[1, 256]``. Reading the width back out is what lets an old
    checkpoint and a new one sit in the same Bradley-Terry fit.
    """
    w = state.get("value.weight")
    if w is None:
        raise KeyError(
            "no `value.weight` in this state_dict, so its value head cannot be sized. "
            "Pass the network's weights, not a training checkpoint (see "
            "`brokefish.eval.layer0.load_net_state`)")
    return int(w.shape[0])


def value_head_of(state: dict) -> str:
    """``"pooled"`` if the state_dict carries the marker, ``"king"`` otherwise.

    ⚠️ Absence is the legacy answer, not an error: every checkpoint written before
    2026-08-22 predates the marker and is a king select.
    """
    return "pooled" if "value_mode" in state else "king"


def net_for_state(state: dict, **kw) -> "BrokefishNet":
    """An empty net shaped to hold ``state``. Load into it; do not skip the load."""
    return BrokefishNet(n_value=n_value_of(state),
                        value_head=value_head_of(state), **kw)


def decode_boards(boards: torch.Tensor):
    """Unpack the 12-bit piece word of spec 2.1.

    ``captured`` has to be tested before anything else is believed: a dead slot
    is ``1 << 11`` exactly, with colour, type and square all wiped, so its other
    fields decode to "white pawn on a1" rather than to anything meaningful.
    """
    w = boards.long()
    return (
        (w >> 11) & 1,   # captured
        (w >> 10) & 1,   # color
        (w >> 9) & 1,    # special
        (w >> 6) & 7,    # type
        w & 63,          # square
    )


class BrokefishNet(nn.Module):
    """Piece-token transformer, 32 tokens for 32 slots. Spec section 7.

    Built in fp32 -- these are the master weights. ``.half()`` gives the
    inference copy the fused implementations are constructed from and compared
    against, exactly as before B2.
    """

    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_ff: int = D_FF, eps: float = 1e-5,
                 n_value: int = N_VALUE_SCALAR, value_head: str = "king"):
        super().__init__()
        if n_value not in VALUE_CLASSES:
            raise ValueError(f"n_value must be one of {VALUE_CLASSES} "
                             f"(1 = the scalar tanh head, 3 = win/draw/loss), "
                             f"got {n_value}")
        if value_head not in VALUE_HEADS:
            raise ValueError(f"value_head must be one of {VALUE_HEADS}, got "
                             f"{value_head!r}")
        self.d_model, self.n_layers, self.n_value = d_model, n_layers, n_value
        self.value_head = value_head

        self.emb_square = nn.Embedding(N_SQUARE, d_model)
        self.emb_type_special = nn.Embedding(N_TYPE_SPECIAL, d_model)
        self.emb_color_turn = nn.Embedding(N_COLOR_TURN, d_model)
        self.emb_clock = nn.Embedding(N_CLOCK, d_model)
        self.emb_rep = nn.Embedding(N_REP, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True, dropout=0.0, layer_norm_eps=eps,
        )
        # norm=None: the final LayerNorm is ours, and it lives outside the stack
        # because the fused kernels treat it as part of the head epilogue rather
        # than as a ninth layer.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=None,
                                             enable_nested_tensor=False)
        self.norm_f = nn.LayerNorm(d_model, eps=eps)

        self.policy = nn.Linear(d_model, N_POLICY, bias=False)
        self.promo = nn.Linear(d_model, N_PROMO, bias=False)
        self.value = nn.Linear(d_model, n_value, bias=False)

        # muP-ish: one scale, 1/sqrt(d), for every table and every readout. Real
        # muP treats embeddings and readouts differently and that is a training
        # decision (C2), not a forward-pass one; the forward barely cares, since
        # the first thing downstream of the sum is a LayerNorm.
        # ⚠️ **Registered only in the pooled case, and that is deliberate.**
        # `n_value` alone no longer identifies the head -- a `[3, 256]` weight is
        # win/draw/loss-from-the-king *or* White/draw/Black-from-the-pool. A marker in
        # the state_dict is the only thing that can tell them apart, and adding it
        # unconditionally would give every legacy checkpoint a missing key under
        # `strict=True`. Present means pooled; absent means king, which is what every
        # file written before 2026-08-22 is.
        if value_head == "pooled":
            self.register_buffer("value_mode", torch.tensor(1, dtype=torch.int64))

        std = d_model ** -0.5
        for mod in (self.emb_square, self.emb_type_special, self.emb_color_turn,
                    self.emb_clock, self.emb_rep,
                    self.policy, self.promo, self.value):
            nn.init.normal_(mod.weight, mean=0.0, std=std)

    # -- stage 1 -----------------------------------------------------------

    def embed(self, boards: torch.Tensor, control: torch.Tensor, rep: torch.Tensor):
        """``[N, 32] uint16`` boards to ``[N, 32, d]`` tokens, plus the live mask.

        The summation order is normative -- square, type_special, color_turn,
        clock, rep, left-associative -- because in fp16 a different order is a
        different number and the CUDA prologue is checked bit-for-bit against
        this.

        Dead slots are gathered, not branched around: their word decodes to a
        valid set of indices, so the five lookups happen anyway and produce a
        finite vector that the attention mask then makes irrelevant. That is
        spec 7.3's bargain, and it is what keeps the prologue divergence-free.
        """
        captured, color, special, ptype, square = decode_boards(boards)
        stm = (control < 0).long()                          # 1 = black to move
        clock = (control.abs().long() - 1).clamp_(0, N_CLOCK - 1)
        rep_i = rep.long().clamp_(0, N_REP - 1)

        # The clock and repetition tables are indexed per board, so their sum is
        # one vector for all 32 tokens. Adding them to each other first is what
        # the kernel does -- four half2 adds per board instead of eight per token
        # -- and in fp16 the grouping is part of the answer, not an optimisation
        # detail. Hence the order: (square + type_special + color_turn) + (clock + rep).
        per_board = self.emb_clock(clock) + self.emb_rep(rep_i)
        x = self.emb_square(square)
        x = x + self.emb_type_special(ptype * 2 + special)
        x = x + self.emb_color_turn(color * 2 + stm[:, None])
        x = x + per_board[:, None, :]
        return x, captured == 0

    # -- stage 3 -----------------------------------------------------------

    @property
    def value_absolute(self) -> bool:
        """Whether the head predicts in White's frame rather than the mover's.

        The loss needs this to put its target in the same frame; nothing else does,
        because :meth:`heads` flips before returning.
        """
        return self.value_head == "pooled"

    def heads(self, h: torch.Tensor, control: torch.Tensor,
              alive: Optional[torch.Tensor] = None,
              with_logits: bool = False):
        """``[N, 32, d]`` normed tokens to (policy_logits, promo, value).

        ``h`` must already be through ``norm_f``. ``policy_logits`` is raw --
        illegal moves are the search's business.

        The value comes from the side-to-move king's token, slots 15 and 31,
        which spec 2.5 guarantees are never captured. That is a row select with
        no reduction, and it arrives from the mover's point of view for free.

        ⚠️ **Two heads live behind that row select and only one is the default.**
        ``n_value == 1`` is spec §7.4's scalar: one logit through ``tanh``, trained
        against ``z`` with a squared error. ``n_value == 3`` is a win/draw/loss
        classifier trained with a cross-entropy, collapsed here by
        :func:`wdl_to_scalar`. Both return the *same* ``[N]`` fp32 in ``[-1, 1]``, so
        the choice is invisible to the search, to the collapse and to the league --
        it is visible only to the loss and to the packed head matrix.
        ``promo`` stays per-token because a position can have more than one pawn
        on the seventh rank, and the head is indexed by the *moving slot* of the
        move -- a promoted queen keeps its pawn slot, so a slot in 0-7 does not
        imply a pawn.
        """
        if self.value_head == "pooled":
            if alive is None:
                raise ValueError(
                    "the pooled value head needs the live-slot mask: a captured slot "
                    "decodes as a live white pawn on a1 (CLAUDE.md), so an unmasked "
                    "mean would average 32 real vectors of which only some mean "
                    "anything, and the dead ones would track material lost")
            # ⚠️ **`norm_f` is upstream of this pool**, so the vectors being averaged
            # are already per-token normed and the only thing after the pool is a
            # biasless linear. `W @ mean(hn) == mean(W @ hn)` exactly, which is why
            # the CUDA epilogue may average the *logits* of the 32 tokens it already
            # computes instead of re-doing a GEMM on a pooled vector. In fp16 the two
            # orders are different numbers, which is what `VALUE_TOL` is for.
            m = alive.unsqueeze(-1).to(h.dtype)
            hv = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            king = torch.where(control < 0, KING_SLOT_BLACK, KING_SLOT_WHITE).long()
            hv = h[torch.arange(h.shape[0], device=h.device), king]
        # ⚠️ The scalar branch is written so that `n_value == 1` is the *same*
        # sequence of ops it always was -- one `.float()`, one `tanh`, one squeeze --
        # because every Elo number on the ledger was produced by it and a reordering
        # in fp16 is a different number (`docs/ledger/perf.md`).
        raw = self.value(hv).float()
        value = torch.tanh(raw).squeeze(-1) if self.n_value == 1 else wdl_to_scalar(raw)
        # The pooled head has no notion of whose turn it is baked into *which token it
        # read*, so it predicts White's frame; the mover's is one deterministic flip
        # away and nothing downstream has to learn it. `control > 0` is White to move
        # (spec §2.4: the magnitude is the clock, never 0).
        if self.value_absolute:
            value = torch.where(control > 0, value, -value)
        if with_logits:
            return self.policy(h), self.promo(h), value, raw
        return self.policy(h), self.promo(h), value

    # -- the whole thing ---------------------------------------------------

    def forward(self, boards: torch.Tensor, control: torch.Tensor, rep: torch.Tensor,
                with_logits: bool = False):
        """The 3-tuple ``(policy, promo, value)`` -- ``value`` is ``[N]`` fp32 in
        ``[-1, 1]`` whatever the head is, which is why nothing downstream changed.

        ``with_logits`` appends the raw ``[N, n_value]`` head output as a fourth
        element. Only the loss wants it, and only because a classifier is trained on
        logits while the search is fed a scalar.
        """
        x, alive = self.embed(boards, control, rep)
        h = self.encoder(x, src_key_padding_mask=~alive)
        return self.heads(self.norm_f(h), control, alive=alive, with_logits=with_logits)


def full_forward_via_backbone(net: BrokefishNet, backbone, boards, control, rep,
                              with_logits: bool = False):
    """The full model with only the encoder stack fused.

    This is how the Triton implementation gets a complete forward: the gather,
    the final norm and the three heads are 0.3 % of the FLOPs, so running them
    as ordinary torch ops around the fused stack costs under 1 % of its time and
    keeps it a real A/B control against the CUDA path, which fuses all three.
    Writing a Triton prologue and epilogue for symmetry would produce no number
    we do not already have.
    """
    x, alive = net.embed(boards, control, rep)
    h = backbone(x, alive)
    return net.heads(net.norm_f(h), control, alive=alive, with_logits=with_logits)
