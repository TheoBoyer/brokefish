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
from torch.nn import functional as F

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
#: ``"prenorm"`` pools the **raw residual stream** and applies `norm_f` to the pooled
#: vector, instead of pooling vectors `norm_f` has already normalised. Two things
#: follow, and they are the reason it exists (2026-08-23):
#:
#: ⚠️ **The head's input scale is fixed by construction.** Measured on `t12h-wdb`,
#: `|mean(LN(h))|` drifts **15.47 -> 10.86** across a 12 h run as the token cloud
#: spreads, and `|value.weight|` grows **+42 %** chasing it while `value_saturated_frac`
#: reaches 0.111. `LN(mean(h))` cannot drift: LayerNorm sets the scale.
#:
#: ⚠️ **It restores a learned per-token weighting.** Pooling *after* `norm_f` gives
#: every live slot exactly equal weight; pooling before it lets a token with a larger
#: residual count for more. That is the thing AlphaZero's per-square value head has and
#: an unweighted mean does not.
#:
#: ⚠️ It is **not** free in the kernel, unlike the other two: `LN(mean(h))` is not a
#: linear function of the per-token value logits, so the epilogue cannot just average
#: columns it already has.
VALUE_HEADS = ("king", "pooled", "prenorm")
#: What the `value_mode` buffer holds. Absent means `"king"`, which is every checkpoint
#: written before 2026-08-22.
VALUE_MODE_ID = {"pooled": 1, "prenorm": 2}

#: **Per-layer re-injection of the model's own inputs** (2026-08-28). Each block's
#: LayerNorm input becomes ``h + sum_k c[site, k] * E_k``, where the five ``E_k`` are the
#: embedding tables of spec 7.2 -- square, type_special, color_turn, clock, rep -- and
#: ``c`` is a learned scalar per (site, source). **The residual stream is untouched**:
#: the mix enters the LN's argument and nothing else, so ``h`` still carries exactly
#: ``h + attn(.) + ffn(.)``.
#:
#: ``"none"`` is every network before this date and runs the *same* ``self.encoder(x)``
#: call it always did. ``"ln1"`` injects at the attention norm of each block, 8 sites.
#: ``"both"`` injects at the FFN norm as well, 16 sites, and costs twice the gather.
#:
#: ⚠️ **The residual's own coefficient is fixed at 1 and that loses nothing.** The mix
#: is the first thing a LayerNorm sees and LN is scale-invariant, so only the *ratio*
#: between the residual and the injected term is observable; a learned coefficient on
#: ``h`` would be a degenerate direction for the optimiser to wander along.
#:
#: ⚠️ **Zero-initialised, so an untrained ``reinject`` net is the old network exactly**
#: -- not approximately. The kernel is templated on the mode rather than branching on
#: it, so ``"none"`` is also byte-identical machine code.
REINJECT_MODES = ("none", "ln1", "both")
REINJECT_MODE_ID = {"ln1": 1, "both": 2}
#: How many embedding tables feed a site, in the normative order of `EmbOff` in
#: `csrc/encoder.cu:199` and of `cuda_impl.PackedWeights.EMB_TABLES`: square,
#: type_special, color_turn, clock, rep. The kernel indexes `reinject_c` with the same
#: integers, so this order is part of the file format.
N_SOURCES = 5


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
    """Which value head a ``state_dict`` was written with.

    ⚠️ Absence of the marker is the legacy answer, not an error: every checkpoint
    written before 2026-08-22 predates it and is a king select. The id is stored rather
    than the name so an old *pooled* checkpoint keeps resolving to `"pooled"` after
    `"prenorm"` was added.
    """
    if "value_mode" not in state:
        return "king"
    got = int(state["value_mode"])
    for name, i in VALUE_MODE_ID.items():
        if i == got:
            return name
    raise ValueError(f"unknown value_mode {got} in this state_dict; this build knows "
                     f"{VALUE_MODE_ID}")


def reinject_of(state: dict) -> str:
    """Which per-layer input re-injection a ``state_dict`` was written with.

    ⚠️ Absence of the marker is the legacy answer, exactly as for ``value_mode``: every
    checkpoint written before 2026-08-28 predates the feature and injects nothing. The
    id is stored rather than the mode name so the meaning of an old file cannot move
    when a mode is added.
    """
    if "reinject_mode" not in state:
        return "none"
    got = int(state["reinject_mode"])
    for name, i in REINJECT_MODE_ID.items():
        if i == got:
            return name
    raise ValueError(f"unknown reinject_mode {got} in this state_dict; this build "
                     f"knows {REINJECT_MODE_ID}")


def net_for_state(state: dict, **kw) -> "BrokefishNet":
    """An empty net shaped to hold ``state``. Load into it; do not skip the load."""
    return BrokefishNet(n_value=n_value_of(state),
                        value_head=value_head_of(state),
                        reinject=reinject_of(state), **kw)


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
                 n_value: int = N_VALUE_SCALAR, value_head: str = "king",
                 reinject: str = "none"):
        super().__init__()
        if n_value not in VALUE_CLASSES:
            raise ValueError(f"n_value must be one of {VALUE_CLASSES} "
                             f"(1 = the scalar tanh head, 3 = win/draw/loss), "
                             f"got {n_value}")
        if value_head not in VALUE_HEADS:
            raise ValueError(f"value_head must be one of {VALUE_HEADS}, got "
                             f"{value_head!r}")
        if reinject not in REINJECT_MODES:
            raise ValueError(f"reinject must be one of {REINJECT_MODES}, got "
                             f"{reinject!r}")
        self.d_model, self.n_layers, self.n_value = d_model, n_layers, n_value
        self.value_head = value_head
        self.reinject = reinject
        self.n_sites = 0 if reinject == "none" else n_layers * (2 if reinject == "both" else 1)

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
        if value_head in VALUE_MODE_ID:
            self.register_buffer("value_mode",
                                 torch.tensor(VALUE_MODE_ID[value_head], dtype=torch.int64))

        # ⚠️ **Zeros, and that is the whole retro-compatibility story here.** A net
        # built with `reinject` and never trained is the old network to the last bit,
        # so an A/B that changes only this flag starts from a shared point rather than
        # from a re-randomised one -- and a `t12h` checkpoint can be loaded into it,
        # since the only new tensors are these and they have no old counterpart to
        # conflict with. The marker rides beside them for the same reason `value_mode`
        # does: nothing else in a bare `state_dict` says what shape the net had.
        if reinject in REINJECT_MODE_ID:
            self.reinject_c = nn.Parameter(torch.zeros(self.n_sites, N_SOURCES))
            self.register_buffer("reinject_mode",
                                 torch.tensor(REINJECT_MODE_ID[reinject], dtype=torch.int64))

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
        idx, alive = self.embed_indices(boards, control, rep)

        # The clock and repetition tables are indexed per board, so their sum is
        # one vector for all 32 tokens. Adding them to each other first is what
        # the kernel does -- four half2 adds per board instead of eight per token
        # -- and in fp16 the grouping is part of the answer, not an optimisation
        # detail. Hence the order: (square + type_special + color_turn) + (clock + rep).
        per_board = self.emb_clock(idx[3]) + self.emb_rep(idx[4])
        x = self.emb_square(idx[0])
        x = x + self.emb_type_special(idx[1])
        x = x + self.emb_color_turn(idx[2])
        x = x + per_board[:, None, :]
        return x, alive

    def embed_indices(self, boards: torch.Tensor, control: torch.Tensor,
                      rep: torch.Tensor):
        """The five table indices of spec 7.2, plus the live-slot mask.

        Split out of :meth:`embed` because per-layer re-injection needs the *indices*
        rather than their summed lookup: it gathers the same five rows again at every
        site with different coefficients. The order is normative and shared with
        `EmbOff` in the kernel -- square, type_special, color_turn, clock, rep -- and
        the last two are ``[N]``, one row per board, where the first three are
        ``[N, 32]``.
        """
        captured, color, special, ptype, square = decode_boards(boards)
        stm = (control < 0).long()                          # 1 = black to move
        idx = (square,
               ptype * 2 + special,
               color * 2 + stm[:, None],
               (control.abs().long() - 1).clamp_(0, N_CLOCK - 1),
               rep.long().clamp_(0, N_REP - 1))
        return idx, captured == 0

    # -- stage 2 -----------------------------------------------------------

    def mix(self, idx, c: torch.Tensor) -> torch.Tensor:
        """``sum_k c[k] * E_k`` for one site: ``[N, 32, d]``.

        ⚠️ **The coefficient scales the table, not the gathered result.** Both give the
        same number; only this one multiplies a ``[rows, d]`` tensor instead of an
        ``[N, 32, d]`` one, and -- the part that matters -- it leaves autograd saving
        the *indices* for the gather's backward rather than five ``[N, 32, d]``
        activations per site. At 8 sites and a 1024-row batch that is the difference
        between ~0 and 1.3 GB of graph.

        The summation order mirrors :meth:`embed`'s for the same reason it is normative
        there: (square + type_special + color_turn) + (clock + rep), so the kernel's
        grouping and this one differ only where fp16 rounds.
        """
        m = F.embedding(idx[0], c[0] * self.emb_square.weight)
        m = m + F.embedding(idx[1], c[1] * self.emb_type_special.weight)
        m = m + F.embedding(idx[2], c[2] * self.emb_color_turn.weight)
        per_board = (F.embedding(idx[3], c[3] * self.emb_clock.weight)
                     + F.embedding(idx[4], c[4] * self.emb_rep.weight))
        return m + per_board[:, None, :]

    def encode(self, x: torch.Tensor, alive: torch.Tensor, idx=None) -> torch.Tensor:
        """The eight blocks. ``idx`` is required when ``reinject`` is on.

        ⚠️ **The ``"none"`` path is the original call, untouched.** It goes through
        `nn.TransformerEncoder`, which owns a fused fast path this hand-rolled loop does
        not reproduce bit-for-bit; keeping the default on it means no existing number
        moves because this feature exists.

        The re-injecting path calls each layer's own ``_sa_block`` / ``_ff_block``
        rather than reimplementing them, so the parameters, their names and the
        arithmetic are the layer's -- the only edit is what goes into ``norm1`` and
        ``norm2``.
        """
        if self.reinject == "none":
            return self.encoder(x, src_key_padding_mask=~alive)
        if idx is None:
            raise ValueError("reinject needs the embedding indices; call "
                             "`embed_indices` and pass them, or use `forward`")
        pad = ~alive
        both = self.reinject == "both"
        for i, lay in enumerate(self.encoder.layers):
            site = i * 2 if both else i
            x = x + lay._sa_block(lay.norm1(x + self.mix(idx, self.reinject_c[site])),
                                  None, pad, is_causal=False)
            n2 = lay.norm2(x + self.mix(idx, self.reinject_c[site + 1])) if both \
                else lay.norm2(x)
            x = x + lay._ff_block(n2)
        return x

    # -- stage 3 -----------------------------------------------------------

    @property
    def value_absolute(self) -> bool:
        """Whether the head predicts in White's frame rather than the mover's.

        The loss needs this to put its target in the same frame; nothing else does,
        because :meth:`heads` flips before returning.
        """
        return self.value_head in ("pooled", "prenorm")

    def heads(self, h: torch.Tensor, control: torch.Tensor,
              alive: Optional[torch.Tensor] = None,
              with_logits: bool = False,
              h_raw: Optional[torch.Tensor] = None):
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
        if self.value_head == "prenorm":
            if alive is None or h_raw is None:
                raise ValueError(
                    "the prenorm value head needs the live-slot mask and the "
                    "*pre-norm* residual stream: it pools before `norm_f`, which is "
                    "the whole point -- the pooled vector's scale is then fixed by the "
                    "norm instead of drifting with how spread the token cloud is")
            m = alive.unsqueeze(-1).to(h_raw.dtype)
            hv = self.norm_f((h_raw * m).sum(1) / m.sum(1).clamp(min=1))
        elif self.value_head == "pooled":
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
                with_logits: bool = False):  # noqa: D401
        """The 3-tuple ``(policy, promo, value)`` -- ``value`` is ``[N]`` fp32 in
        ``[-1, 1]`` whatever the head is, which is why nothing downstream changed.

        ``with_logits`` appends the raw ``[N, n_value]`` head output as a fourth
        element. Only the loss wants it, and only because a classifier is trained on
        logits while the search is fed a scalar.
        """
        idx, alive = self.embed_indices(boards, control, rep)
        per_board = self.emb_clock(idx[3]) + self.emb_rep(idx[4])
        x = self.emb_square(idx[0])
        x = x + self.emb_type_special(idx[1])
        x = x + self.emb_color_turn(idx[2])
        x = x + per_board[:, None, :]
        h = self.encode(x, alive, idx)
        return self.heads(self.norm_f(h), control, alive=alive,
                          with_logits=with_logits, h_raw=h)


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
    if net.reinject != "none":
        raise NotImplementedError(
            "the Triton backbone does not implement per-layer input re-injection. "
            "It is a stack-level change -- the mix enters every block's LayerNorm -- "
            "so a fused backbone that only sees `x` cannot express it. Use "
            "`--impl cuda` or `--impl torch`")
    x, alive = net.embed(boards, control, rep)
    h = backbone(x, alive)
    return net.heads(net.norm_f(h), control, alive=alive, with_logits=with_logits,
                     h_raw=h)
