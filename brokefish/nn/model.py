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
                 n_heads: int = N_HEADS, d_ff: int = D_FF, eps: float = 1e-5):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers

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
        self.value = nn.Linear(d_model, 1, bias=False)

        # muP-ish: one scale, 1/sqrt(d), for every table and every readout. Real
        # muP treats embeddings and readouts differently and that is a training
        # decision (C2), not a forward-pass one; the forward barely cares, since
        # the first thing downstream of the sum is a LayerNorm.
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

    def heads(self, h: torch.Tensor, control: torch.Tensor):
        """``[N, 32, d]`` normed tokens to (policy_logits, promo, value).

        ``h`` must already be through ``norm_f``. ``policy_logits`` is raw --
        illegal moves are the search's business.

        The value comes from the side-to-move king's token, slots 15 and 31,
        which spec 2.5 guarantees are never captured. That is a row select with
        no reduction, and it arrives from the mover's point of view for free.
        ``promo`` stays per-token because a position can have more than one pawn
        on the seventh rank, and the head is indexed by the *moving slot* of the
        move -- a promoted queen keeps its pawn slot, so a slot in 0-7 does not
        imply a pawn.
        """
        king = torch.where(control < 0, KING_SLOT_BLACK, KING_SLOT_WHITE).long()
        hk = h[torch.arange(h.shape[0], device=h.device), king]
        value = torch.tanh(self.value(hk).float()).squeeze(-1)
        return self.policy(h), self.promo(h), value

    # -- the whole thing ---------------------------------------------------

    def forward(self, boards: torch.Tensor, control: torch.Tensor, rep: torch.Tensor):
        x, alive = self.embed(boards, control, rep)
        h = self.encoder(x, src_key_padding_mask=~alive)
        return self.heads(self.norm_f(h), control)


def full_forward_via_backbone(net: BrokefishNet, backbone, boards, control, rep):
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
    return net.heads(net.norm_f(h), control)
