"""Single-kernel forward pass for the Brokefish policy/value backbone, in Triton.

The reference implementation of the encoder contract: whatever
``brokefish/nn/cuda_impl.py`` computes, it computes the same thing to within
fp16 noise, and `tests/test_model.py` runs the same bar against both.

The network is a pre-norm transformer encoder over 32 piece tokens
(``d_model=256``, 8 layers, 8 heads, ``d_ff=1024``, ReLU, fp16). Boards are
independent, so the whole stack runs with no grid-wide synchronization: one CTA
owns ``BM`` token rows and carries them through every layer inside a single
kernel launch.

Why one kernel. At T=32 tokens the per-op tensor-core work is tiny, so a
layer-by-layer PyTorch forward is dominated by kernel boundaries and by
activation round-trips to HBM. Fusing the eight layers into one launch drops
DRAM traffic to 1.5% of the roofline and buys 2.12x over torch eager
(42.6k evals/s on an RTX 4060 Laptop, B=16384, fp16, validated to 2.6e-3 against
torch). The dead-token mask that spec 7.3 makes mandatory costs 1.0% of that,
measured interleaved over 12 rounds; fp16 accumulation is worth +19%.

The split alternative is refuted rather than untried: cuBLAS on the four real
per-layer shapes, at the batch size where it is at its best, costs 406.8 ms for
eight layers before attention, LayerNorm, or any activation traffic.

What limits it now. Warps are generalist and sequential, and the CTA barriers
between phases align every warp on the same instruction type at the same
instant: one pipe saturates while the others idle, in rotation (No-Eligible
89.9%, math-pipe stalls 47.9%). The PTX shows 90 ``bar.sync`` per layer, because
every weight tile travels global -> registers -> SMEM -> ldmatrix -> mma and any
transit through shared memory costs a CTA barrier. Removing those barriers needs
``cp.async`` with per-tile mbarriers and warp specialization, which Triton cannot
express -- hence the planned CUDA C++ replacement. See ``docs/perf.md`` for the
full ledger, including the levers already measured to be worthless.

Shape specialization. Triton has no lists and no dynamic register indexing, so
the ``D/BC = 4`` column tiles are unrolled by hand into ``x0..x3`` / ``a0..a3``.
That pins ``D == 4 * BC``; :class:`FusedEncoder` checks it at construction time.
Everything else (layer count, d_ff, head count, sequence length) is generic.

The accumulator. ``acc_dtype`` selects the mma variant: fp16 accumulation issues
at 35.5 TFLOPS against 18.0 for fp32 on this card, because fp32 accumulate is
half rate on GeForce Ada, and it measures +19% end to end. It is the default. The
cost is 1.3-1.5e-3 relative error, the same band as torch's own fp16 forward, and
an overflow ceiling: see ``docs/perf.md`` for both. ``acc_dtype="fp32"`` restores
the slower, wider path in one argument.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["FusedEncoder"]

# Column tile width, in elements. 64 keeps a weight tile at [64,64] fp16 = 8 KB,
# which fits 2-3 CTAs per SM; 128 halves the weight-load tax but drops to 1 CTA
# and measures slower (see docs/perf.md).
BC = 64


@triton.jit
def _encoder_fwd(
    y_ptr, qkv_scratch, attn_scratch, alive_ptr,
    w_qkv, b_qkv, w_o, b_o, w_ff1, b_ff1, w_ff2, b_ff2,
    ln1_w, ln1_b, ln2_w, ln2_b,
    eps, n_layers,
    BM: tl.constexpr, D: tl.constexpr, DFF: tl.constexpr,
    T: tl.constexpr, DH: tl.constexpr, H: tl.constexpr,
    BC: tl.constexpr, HAS_MASK: tl.constexpr, ACC: tl.constexpr,
):
    """Full pre-norm encoder stack, in place on ``y_ptr`` ([n_tokens, D] fp16).

    One program owns ``BM`` consecutive token rows for the whole stack. Each
    layer runs in three phases separated by CTA barriers, because attention
    remaps which thread holds which data:

    1. LayerNorm + QKV projection  -> ``qkv_scratch``
    2. per-head attention          -> ``attn_scratch``
    3. output projection, residual, LayerNorm, feed-forward, residual -> ``y_ptr``

    The scratch buffers live in global memory rather than SMEM: at BM=32 they
    total 28.7 KB per CTA, and keeping them in SMEM would not lift the occupancy
    limit, which is set by registers (255/thread), not by shared memory.

    ``HAS_MASK`` switches on the dead-token mask required by spec 7.3.
    ``alive_ptr`` is then an [n_tokens] int8 vector, nonzero where the slot holds
    a live piece. Rows of dead tokens are computed and written, and hold garbage;
    nothing reads them, since they are absent from every key set, their policy
    logits die against the engine mask, and the value comes from a king token.

    One invariant the mask does not enforce: dead tokens have to be finite. Their
    attention weight is zero, but the PV matmul multiplies rather than selects,
    and 0 * inf is NaN, so an infinite dead token poisons its whole board. Spec
    7.2 builds dead tokens from the same embedding tables as live ones, so this
    holds by construction; it constrains whatever writes dead slots.

    ``ACC`` is the mma accumulator type. Every accumulator here contracts a
    LayerNorm output against a weight tile, and LayerNorm is scale-invariant in
    the residual stream, so the residual growing through training does not reach
    them: at initialisation they peak between 1.1 and 3.6 against fp16's 65504.
    The exception is the attention score, which contracts Q against K and grows
    as the fourth power of any weight growth. The host folds 1/sqrt(DH) into
    W_q for exactly that reason, so this kernel accumulates post-scale logits
    (peak 3.1, headroom 21000x) rather than pre-scale ones (peak 17.4).
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, BC)
    head_col = tl.arange(0, DH)
    board = tl.arange(0, BM) // T  # which board each row belongs to

    # Attention keys and queries are both `rows`, so one [BM] load covers both
    # axes. Liveness is a property of the position rather than of the layer, so
    # the predicate tile is built once and reused by all 64 (layer, head) pairs.
    if HAS_MASK:
        alive = tl.load(alive_ptr + rows) != 0
        keep = (board[:, None] == board[None, :]) & alive[None, :]
    else:
        keep = board[:, None] == board[None, :]

    for layer in range(0, n_layers):
        # -- phase 1: LayerNorm, then the fused QKV projection ----------------
        x0 = tl.load(y_ptr + rows[:, None] * D + 0 * BC + col[None, :]).to(tl.float32)
        x1 = tl.load(y_ptr + rows[:, None] * D + 1 * BC + col[None, :]).to(tl.float32)
        x2 = tl.load(y_ptr + rows[:, None] * D + 2 * BC + col[None, :]).to(tl.float32)
        x3 = tl.load(y_ptr + rows[:, None] * D + 3 * BC + col[None, :]).to(tl.float32)

        mean = (tl.sum(x0, 1) + tl.sum(x1, 1) + tl.sum(x2, 1) + tl.sum(x3, 1)) / D
        x0 -= mean[:, None]
        x1 -= mean[:, None]
        x2 -= mean[:, None]
        x3 -= mean[:, None]
        # Two-pass variance on centred values. Var = E[x^2] - mean^2 would make
        # the two reductions independent, but it measures as noise: the
        # dependency is not on the critical path (docs/perf.md).
        var = (tl.sum(x0 * x0, 1) + tl.sum(x1 * x1, 1)
               + tl.sum(x2 * x2, 1) + tl.sum(x3 * x3, 1))
        rstd = 1.0 / tl.sqrt(var / D + eps)

        n0 = (x0 * rstd[:, None] * tl.load(ln1_w + layer * D + 0 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln1_b + layer * D + 0 * BC + col).to(tl.float32)[None, :]).to(tl.float16)
        n1 = (x1 * rstd[:, None] * tl.load(ln1_w + layer * D + 1 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln1_b + layer * D + 1 * BC + col).to(tl.float32)[None, :]).to(tl.float16)
        n2 = (x2 * rstd[:, None] * tl.load(ln1_w + layer * D + 2 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln1_b + layer * D + 2 * BC + col).to(tl.float32)[None, :]).to(tl.float16)
        n3 = (x3 * rstd[:, None] * tl.load(ln1_w + layer * D + 3 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln1_b + layer * D + 3 * BC + col).to(tl.float32)[None, :]).to(tl.float16)

        for j in range(3 * D // BC):  # one output tile of [Q|K|V] per iteration
            # The accumulator is threaded through tl.dot rather than summed
            # afterwards: that is what makes Triton emit an mma whose D operand
            # is ACC. Adding fp32 results of fp32-accumulate mma would compile to
            # the slow instruction whatever ACC says.
            acc = tl.zeros((BM, BC), dtype=ACC)
            w = tl.load(w_qkv + layer * D * 3 * D + (0 * BC + col)[:, None] * (3 * D) + j * BC + col[None, :])
            acc = tl.dot(n0, w, acc, out_dtype=ACC)
            w = tl.load(w_qkv + layer * D * 3 * D + (1 * BC + col)[:, None] * (3 * D) + j * BC + col[None, :])
            acc = tl.dot(n1, w, acc, out_dtype=ACC)
            w = tl.load(w_qkv + layer * D * 3 * D + (2 * BC + col)[:, None] * (3 * D) + j * BC + col[None, :])
            acc = tl.dot(n2, w, acc, out_dtype=ACC)
            w = tl.load(w_qkv + layer * D * 3 * D + (3 * BC + col)[:, None] * (3 * D) + j * BC + col[None, :])
            acc = tl.dot(n3, w, acc, out_dtype=ACC)
            out = acc.to(tl.float32) + tl.load(b_qkv + layer * 3 * D + j * BC + col).to(tl.float32)[None, :]
            tl.store(qkv_scratch + rows[:, None] * (3 * D) + j * BC + col[None, :], out.to(tl.float16))
        tl.debug_barrier()

        # -- phase 2: attention, materialized (T=32 makes tiling pointless) ---
        for h in range(0, H):
            q = tl.load(qkv_scratch + rows[:, None] * (3 * D) + h * DH + head_col[None, :])
            k = tl.load(qkv_scratch + rows[:, None] * (3 * D) + D + h * DH + head_col[None, :])
            # `keep` carries two conditions. The block-diagonal one, keeping rows
            # of different boards apart, folds away when BM == T (one board per
            # program, the default) and leaves no -inf in the PTX. The liveness
            # one is dynamic and survives; it measures at 1.0% (docs/perf.md).
            # No 1/sqrt(DH) here: the host folded it into W_q, which keeps this
            # accumulator on the post-scale logits.
            s = tl.where(keep, tl.dot(q, tl.trans(k), out_dtype=ACC).to(tl.float32), -float("inf"))
            p = tl.exp(s - tl.max(s, 1)[:, None])
            # Normalize before the PV matmul: T=32 < DH=64, so dividing here
            # touches fewer elements than dividing the output would.
            p = p / tl.sum(p, 1)[:, None]
            v = tl.load(qkv_scratch + rows[:, None] * (3 * D) + 2 * D + h * DH + head_col[None, :])
            tl.store(attn_scratch + rows[:, None] * D + h * DH + head_col[None, :],
                     tl.dot(p.to(tl.float16), v, out_dtype=ACC).to(tl.float16))
        tl.debug_barrier()

        # -- phase 3: output proj + residual, LayerNorm, FFN + residual -------
        o0 = tl.load(attn_scratch + rows[:, None] * D + 0 * BC + col[None, :])
        o1 = tl.load(attn_scratch + rows[:, None] * D + 1 * BC + col[None, :])
        o2 = tl.load(attn_scratch + rows[:, None] * D + 2 * BC + col[None, :])
        o3 = tl.load(attn_scratch + rows[:, None] * D + 3 * BC + col[None, :])

        # The projection accumulates on its own, and the residual and bias join
        # in fp32 afterwards. Starting from the residual would force an fp32
        # accumulator whatever ACC says. Measured cost of the restructure on the
        # fp32 path: 0.3%, below this machine's drift.
        p0 = tl.zeros((BM, BC), dtype=ACC)
        p1 = tl.zeros((BM, BC), dtype=ACC)
        p2 = tl.zeros((BM, BC), dtype=ACC)
        p3 = tl.zeros((BM, BC), dtype=ACC)
        for i in range(D // BC):  # accumulate over the K tiles of W_o
            # Manual select: Triton cannot index a list of register tiles.
            oi = o0
            if i == 1:
                oi = o1
            if i == 2:
                oi = o2
            if i == 3:
                oi = o3
            w = tl.load(w_o + layer * D * D + (i * BC + col)[:, None] * D + 0 * BC + col[None, :])
            p0 = tl.dot(oi, w, p0, out_dtype=ACC)
            w = tl.load(w_o + layer * D * D + (i * BC + col)[:, None] * D + 1 * BC + col[None, :])
            p1 = tl.dot(oi, w, p1, out_dtype=ACC)
            w = tl.load(w_o + layer * D * D + (i * BC + col)[:, None] * D + 2 * BC + col[None, :])
            p2 = tl.dot(oi, w, p2, out_dtype=ACC)
            w = tl.load(w_o + layer * D * D + (i * BC + col)[:, None] * D + 3 * BC + col[None, :])
            p3 = tl.dot(oi, w, p3, out_dtype=ACC)

        # Re-reading y_ptr for the residual is deliberate: keeping both the raw
        # and the centred tiles alive costs more registers than the reload
        # costs latency, and registers are what caps occupancy here.
        y0 = (tl.load(y_ptr + rows[:, None] * D + 0 * BC + col[None, :]).to(tl.float32)
              + tl.load(b_o + layer * D + 0 * BC + col).to(tl.float32)[None, :] + p0.to(tl.float32))
        y1 = (tl.load(y_ptr + rows[:, None] * D + 1 * BC + col[None, :]).to(tl.float32)
              + tl.load(b_o + layer * D + 1 * BC + col).to(tl.float32)[None, :] + p1.to(tl.float32))
        y2 = (tl.load(y_ptr + rows[:, None] * D + 2 * BC + col[None, :]).to(tl.float32)
              + tl.load(b_o + layer * D + 2 * BC + col).to(tl.float32)[None, :] + p2.to(tl.float32))
        y3 = (tl.load(y_ptr + rows[:, None] * D + 3 * BC + col[None, :]).to(tl.float32)
              + tl.load(b_o + layer * D + 3 * BC + col).to(tl.float32)[None, :] + p3.to(tl.float32))

        mean2 = (tl.sum(y0, 1) + tl.sum(y1, 1) + tl.sum(y2, 1) + tl.sum(y3, 1)) / D
        c0 = y0 - mean2[:, None]
        c1 = y1 - mean2[:, None]
        c2 = y2 - mean2[:, None]
        c3 = y3 - mean2[:, None]
        var2 = (tl.sum(c0 * c0, 1) + tl.sum(c1 * c1, 1)
                + tl.sum(c2 * c2, 1) + tl.sum(c3 * c3, 1))
        rstd2 = 1.0 / tl.sqrt(var2 / D + eps)

        m0 = (c0 * rstd2[:, None] * tl.load(ln2_w + layer * D + 0 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln2_b + layer * D + 0 * BC + col).to(tl.float32)[None, :]).to(tl.float16)
        m1 = (c1 * rstd2[:, None] * tl.load(ln2_w + layer * D + 1 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln2_b + layer * D + 1 * BC + col).to(tl.float32)[None, :]).to(tl.float16)
        m2 = (c2 * rstd2[:, None] * tl.load(ln2_w + layer * D + 2 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln2_b + layer * D + 2 * BC + col).to(tl.float32)[None, :]).to(tl.float16)
        m3 = (c3 * rstd2[:, None] * tl.load(ln2_w + layer * D + 3 * BC + col).to(tl.float32)[None, :]
              + tl.load(ln2_b + layer * D + 3 * BC + col).to(tl.float32)[None, :]).to(tl.float16)

        a0 = tl.zeros((BM, BC), dtype=ACC)
        a1 = tl.zeros((BM, BC), dtype=ACC)
        a2 = tl.zeros((BM, BC), dtype=ACC)
        a3 = tl.zeros((BM, BC), dtype=ACC)
        # Hidden tiles are consumed as they are produced: the [BM, DFF]
        # intermediate is never materialized. a0..a3 therefore carry the whole
        # K=1024 contraction in one accumulator, which is the longest one in the
        # kernel; measured peak at initialisation is 1.2 against fp16's 65504.
        for f in range(DFF // BC):
            hid = tl.zeros((BM, BC), dtype=ACC)
            w = tl.load(w_ff1 + layer * D * DFF + (0 * BC + col)[:, None] * DFF + f * BC + col[None, :])
            hid = tl.dot(m0, w, hid, out_dtype=ACC)
            w = tl.load(w_ff1 + layer * D * DFF + (1 * BC + col)[:, None] * DFF + f * BC + col[None, :])
            hid = tl.dot(m1, w, hid, out_dtype=ACC)
            w = tl.load(w_ff1 + layer * D * DFF + (2 * BC + col)[:, None] * DFF + f * BC + col[None, :])
            hid = tl.dot(m2, w, hid, out_dtype=ACC)
            w = tl.load(w_ff1 + layer * D * DFF + (3 * BC + col)[:, None] * DFF + f * BC + col[None, :])
            hid = tl.dot(m3, w, hid, out_dtype=ACC)
            biased = hid.to(tl.float32) + tl.load(b_ff1 + layer * DFF + f * BC + col).to(tl.float32)[None, :]
            act = tl.maximum(biased, 0.0).to(tl.float16)

            w = tl.load(w_ff2 + layer * DFF * D + (f * BC + col)[:, None] * D + 0 * BC + col[None, :])
            a0 = tl.dot(act, w, a0, out_dtype=ACC)
            w = tl.load(w_ff2 + layer * DFF * D + (f * BC + col)[:, None] * D + 1 * BC + col[None, :])
            a1 = tl.dot(act, w, a1, out_dtype=ACC)
            w = tl.load(w_ff2 + layer * DFF * D + (f * BC + col)[:, None] * D + 2 * BC + col[None, :])
            a2 = tl.dot(act, w, a2, out_dtype=ACC)
            w = tl.load(w_ff2 + layer * DFF * D + (f * BC + col)[:, None] * D + 3 * BC + col[None, :])
            a3 = tl.dot(act, w, a3, out_dtype=ACC)

        z0 = a0.to(tl.float32) + tl.load(b_ff2 + layer * D + 0 * BC + col).to(tl.float32)[None, :] + y0
        z1 = a1.to(tl.float32) + tl.load(b_ff2 + layer * D + 1 * BC + col).to(tl.float32)[None, :] + y1
        z2 = a2.to(tl.float32) + tl.load(b_ff2 + layer * D + 2 * BC + col).to(tl.float32)[None, :] + y2
        z3 = a3.to(tl.float32) + tl.load(b_ff2 + layer * D + 3 * BC + col).to(tl.float32)[None, :] + y3
        tl.store(y_ptr + rows[:, None] * D + 0 * BC + col[None, :], z0.to(tl.float16))
        tl.store(y_ptr + rows[:, None] * D + 1 * BC + col[None, :], z1.to(tl.float16))
        tl.store(y_ptr + rows[:, None] * D + 2 * BC + col[None, :], z2.to(tl.float16))
        tl.store(y_ptr + rows[:, None] * D + 3 * BC + col[None, :], z3.to(tl.float16))
        tl.debug_barrier()


class FusedEncoder:
    """Inference-only fused forward for a pre-norm ``nn.TransformerEncoder``.

    Stacks the layer weights once at construction, then runs the whole stack in
    a single kernel launch. The module holds its own scratch and output buffers
    and reuses them across calls, so :meth:`forward` allocates nothing in steady
    state -- which is what a batched MCTS loop needs.

    The source encoder must be ``batch_first=True``, ``norm_first=True``, ReLU,
    ``dropout=0``, fp16, on CUDA, and must have no final ``norm``. The network
    *does* have a final LayerNorm (spec 7.4) -- it lives on
    :class:`~brokefish.nn.model.BrokefishNet` as ``norm_f``, outside the stack,
    because both fused implementations treat it as the first step of the head
    epilogue rather than as a ninth layer.

    Example::

        # the backbone alone, which is the A/B control
        enc = torch.nn.TransformerEncoder(layer, num_layers=8).cuda().half().eval()
        model = FusedEncoder(enc)
        out = model(x, alive)   # x: [n_boards, 32, 256] fp16 cuda
                                # alive: [n_boards, 32] bool cuda

        # the whole network: boards in, logits out
        net = BrokefishNet().cuda().half().eval()
        policy_logits, promo, value = FusedEncoder(net).forward_full(boards, control, rep)
    """

    # Winner of the joint 126-configuration sweep (2026-07-29, RTX 4060 Laptop).
    # These are not independent: maxnreg=168 -- the threshold that lets a third
    # CTA fit per SM -- only wins jointly with BM=32, and never shows up in a
    # one-parameter-at-a-time sweep. Anything below 168 spills.
    BM = 32
    NUM_WARPS = 4
    NUM_STAGES = 1
    MAXNREG = 168

    # fp16 accumulation halves the register footprint of every accumulator, so
    # a lower maxnreg looks tempting. It measures 9% slower (40.2k against
    # 43.7k): 168 is the third-CTA threshold and dropping below it spills.
    ACC_DTYPES = {"fp16": tl.float16, "fp32": tl.float32}

    def __init__(self, source, acc_dtype: str = "fp16"):
        # B2: a BrokefishNet gives the whole model, a bare nn.TransformerEncoder
        # gives the backbone alone. The gather, the final norm and the heads are
        # 0.3 % of the FLOPs, so this implementation runs them as ordinary torch
        # ops around the kernel rather than inside it -- see
        # model.full_forward_via_backbone for why that keeps the A/B honest.
        from brokefish.nn.model import BrokefishNet

        self.net = source if isinstance(source, BrokefishNet) else None
        encoder = self.net.encoder if self.net is not None else source
        layers = encoder.layers
        first = layers[0]

        if acc_dtype not in self.ACC_DTYPES:
            raise ValueError(f"acc_dtype must be one of {sorted(self.ACC_DTYPES)}, got {acc_dtype!r}")
        self.acc_dtype = acc_dtype
        self.acc = self.ACC_DTYPES[acc_dtype]

        self.n_layers = len(layers)
        self.d_model = first.self_attn.embed_dim
        self.n_heads = first.self_attn.num_heads
        self.d_ff = first.linear1.out_features
        self.d_head = self.d_model // self.n_heads
        self.eps = first.norm1.eps

        if self.d_model != 4 * BC:
            raise ValueError(
                f"kernel is unrolled for d_model == 4 * {BC} = {4 * BC}, got {self.d_model}. "
                "Widening it means adding register tiles by hand (Triton has no "
                "dynamic register indexing)."
            )
        if self.d_ff % BC or self.d_model % self.n_heads:
            raise ValueError(f"d_ff must be a multiple of {BC} and d_model of n_heads")

        # Weights are transposed once here so the kernel reads [in, out] tiles
        # with the contraction on the row axis, matching the mma operand order.
        # .half() so a BrokefishNet held in fp32 master weights can be handed
        # over directly: the fused path is fp16 by definition.
        stack = lambda pick: torch.stack([pick(x) for x in layers]).contiguous().half()
        self.w_qkv = stack(lambda x: x.self_attn.in_proj_weight.detach().t())
        self.b_qkv = stack(lambda x: x.self_attn.in_proj_bias.detach())
        self.w_o = stack(lambda x: x.self_attn.out_proj.weight.detach().t())
        self.b_o = stack(lambda x: x.self_attn.out_proj.bias.detach())
        self.w_ff1 = stack(lambda x: x.linear1.weight.detach().t())
        self.b_ff1 = stack(lambda x: x.linear1.bias.detach())
        self.w_ff2 = stack(lambda x: x.linear2.weight.detach().t())
        self.b_ff2 = stack(lambda x: x.linear2.bias.detach())
        self.ln1_w = stack(lambda x: x.norm1.weight.detach())
        self.ln1_b = stack(lambda x: x.norm1.bias.detach())
        self.ln2_w = stack(lambda x: x.norm2.weight.detach())
        self.ln2_b = stack(lambda x: x.norm2.bias.detach())

        # Fold 1/sqrt(d_head) into the Q half of the QKV projection, exactly as
        # scaling Q would. Q * K^T is the one accumulator that can overflow
        # fp16: it grows as the fourth power of any weight growth, where every
        # other accumulator grows as the square or cube, and it overflows at
        # about 8x weight growth if the scale is applied after the contraction.
        # Folding it moves that to 12x and costs one multiply at construction.
        scale = 1.0 / math.sqrt(self.d_head)
        self.w_qkv[:, :, :self.d_model] *= scale
        self.b_qkv[:, :self.d_model] *= scale

        self._buffered_tokens = 0

    def _ensure_buffers(self, n_tokens: int) -> None:
        if self._buffered_tokens != n_tokens:
            opts = dict(device="cuda", dtype=torch.half)
            self.qkv_scratch = torch.empty(n_tokens, 3 * self.d_model, **opts)
            self.attn_scratch = torch.empty(n_tokens, self.d_model, **opts)
            self.state = torch.empty(n_tokens, self.d_model, **opts)
            self._buffered_tokens = n_tokens

    @staticmethod
    def _alive_arg(alive: torch.Tensor, n_boards: int, seq_len: int) -> torch.Tensor:
        if alive.shape != (n_boards, seq_len):
            raise ValueError(f"alive must be [{n_boards}, {seq_len}], got {tuple(alive.shape)}")
        if alive.dtype is torch.bool:
            alive = alive.view(torch.int8)  # free reinterpret, both are one byte
        elif alive.dtype is not torch.int8:
            raise ValueError(f"alive must be bool or int8, got {alive.dtype}")
        return alive.contiguous().view(-1)

    def forward(
        self,
        x: torch.Tensor,
        alive: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the stack on ``x`` of shape ``[n_boards, seq_len, d_model]``.

        ``alive`` is the dead-token mask of spec 7.3, a ``[n_boards, seq_len]``
        bool or int8 tensor that is true where the slot holds a live piece. Dead
        slots are dropped from every key set, so they influence nothing. Their
        own output rows are computed and are meaningless; the policy head reads
        them into logits that the engine mask kills, and the value head reads a
        king token, which is always alive.

        Passing ``None`` runs unmasked, which is the pre-spec behaviour and is
        kept for the A/B against the 37.2k baseline. It compiles to a separate
        kernel, so neither path pays for the other.

        Returns a view on an internal buffer unless ``out`` is given; copy it if
        you need the values to survive the next call.
        """
        n_boards, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")
        n_tokens = n_boards * seq_len
        if n_tokens % self.BM:
            raise ValueError(f"n_boards * seq_len must be a multiple of {self.BM}")

        has_mask = alive is not None
        # Triton needs a pointer either way; x is a stand-in the kernel never
        # dereferences when HAS_MASK is false.
        alive_arg = self._alive_arg(alive, n_boards, seq_len) if has_mask else x

        self._ensure_buffers(n_tokens)
        y = self.state if out is None else out
        y.copy_(x.view(n_tokens, d_model))

        _encoder_fwd[(n_tokens // self.BM,)](
            y, self.qkv_scratch, self.attn_scratch, alive_arg,
            self.w_qkv, self.b_qkv, self.w_o, self.b_o,
            self.w_ff1, self.b_ff1, self.w_ff2, self.b_ff2,
            self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b,
            self.eps, self.n_layers,
            BM=self.BM, D=self.d_model, DFF=self.d_ff, T=seq_len,
            DH=self.d_head, H=self.n_heads, BC=BC, HAS_MASK=has_mask, ACC=self.acc,
            num_warps=self.NUM_WARPS, num_stages=self.NUM_STAGES,
            maxnreg=self.MAXNREG,
        )
        return y.view(n_boards, seq_len, d_model)

    __call__ = forward

    def forward_full(self, boards, control, rep):
        """Boards to (policy_logits, promo, value), with only the stack fused.

        Same outputs as the CUDA implementation's ``forward_full``, so the two
        can be compared directly; the difference is that here the prologue and
        epilogue are torch ops, which is a deliberate choice and not a gap.
        """
        from brokefish.nn.model import full_forward_via_backbone

        if self.net is None:
            raise ValueError(
                "forward_full needs the whole model: build this from a BrokefishNet, "
                "not from a bare nn.TransformerEncoder")
        return full_forward_via_backbone(self.net, self, boards, control, rep)
