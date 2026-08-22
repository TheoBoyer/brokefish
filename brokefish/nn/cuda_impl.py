"""CUDA C++ implementation of the encoder contract.

Same interface as :mod:`brokefish.nn.triton_impl`, so `tests/test_model.py` and
`bench/bench_model.py` pick it up by name and duel the two in one process.

What the host owns: the weight layout. The kernel wants one contiguous fp16
buffer, per layer, in the order `csrc/encoder.cu` declares in ``struct Off``,
with the matrices in torch's own ``[out, in]`` orientation — that orientation is
what makes an mma B fragment a plain ``ldmatrix`` with no ``.trans``.

One arithmetic change happens here rather than in the kernel: 1/sqrt(d_head) is
folded into the Q half of the QKV projection. Q·Kᵀ is the only accumulator in
the network that can overflow fp16, it grows as the fourth power of weight
growth, and folding the scale in front of the contraction rather than after it
buys a factor of 5.7 of headroom for one multiply at construction. Same reason,
same code, as in the Triton implementation.

**B2.** Constructed from a :class:`brokefish.nn.model.BrokefishNet` this also
packs the five embedding tables and the three heads, and :meth:`forward_full`
runs boards to logits in the one launch. Constructed from a bare
``nn.TransformerEncoder`` it is the pre-B2 backbone and nothing else, which is
what the A/B against Triton and ``tests/test_model.py`` still use.
"""

from __future__ import annotations

import torch

from brokefish.nn import _build
from brokefish.nn.model import BrokefishNet, N_POLICY, N_PROMO


def pack_b(w: torch.Tensor) -> torch.Tensor:
    """Permute a weight matrix [N, K] into mma B-fragment order.

    The kernel reads its B operands straight from global memory into registers
    with one coalesced 128-bit load per lane -- no shared memory, no ldmatrix,
    no barrier. That only works if the bytes are already sitting in the order
    the lanes want them, which is what this does.

    A lane L of an m16n8k16 fragment (g = L//4, t = L%4) holds b[0] =
    W[n0+g][k0+2t], W[n0+g][k0+2t+1] and b[1] the same 8 columns further along.
    Two k-steps fill a uint4, so the result is indexed

        packed[n8][k32][lane][j]     j = k16*4 + half8*2 + pair

    The device-side contract is documented at `gemm_direct` in csrc/encoder.cu.
    These two must change together; a mismatch is silent and produces plausible
    numbers, so `csrc/tests/tdirect.cu` pins the layout independently.
    """
    n, k = w.shape
    if n % 8 or k % 32:
        raise ValueError(f"packed weights need N%8==0 and K%32==0, got [{n}, {k}]")
    # col = k32*32 + k16*16 + half8*8 + t*2 + pair ; row = n8*8 + g
    v = w.reshape(n // 8, 8, k // 32, 2, 2, 4, 2)
    #             n8      g  k32      k16 h8 t  pair
    return v.permute(0, 2, 1, 5, 3, 4, 6).reshape(-1)

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = _build.load_extension("brokefish_encoder", ["encoder.cu"])
    return _EXT


def _fold_norm(w: torch.Tensor, b: torch.Tensor, norm):
    """Fold a LayerNorm's affine into the GEMM that consumes it (§CODA).

    `LN(x) = gamma . x_hat + beta`, so

        LN(x) W^T + b = x_hat (W . gamma)^T + (b + W beta)

    with `x_hat = (x - mu) / sigma`. Both are exact and both are free at run time; the
    kernel's `layernorm<false>` then emits `x_hat` and stops there. Measured worth:
    `encoder_kernel<true>` goes from 4056 SASS instructions to 3888.

    ⚠️ **The bias uses `W` before gamma is folded in.** `beta` is added *after* the
    elementwise gamma, so it multiplies the original matrix; doing it the other way
    round scales beta by gamma twice and is wrong by an amount that looks like a
    quantisation artefact. `tests/test_model.py` compares against torch and would
    catch it, which is the only reason it is safe to say so rather than prove it here.

    Returns fp32; the caller casts once, at the end.
    """
    g = norm.weight.detach().float()
    e = norm.bias.detach().float()
    return w * g[None, :], b + w @ e


#: Which matmuls `--int8` quantises, by kernel mode. Ordered by how much of the
#: encoder they cover; the number is `csrc/encoder.cu`'s `quant` argument.
SCHEMES = {"ffn": 2, "ffn+qkv": 4, "all": 3}

#: **What `--int8` means.** One name, decided once, measured before it was chosen --
#: there is deliberately no way to spell a scheme in any CLI, because a switch whose
#: settings differ by 2.8 % of the clock and a decimal place of rounding error is not
#: a decision to hand to whoever is launching a run.
#:
#: `"all"` is the FFN's two matmuls plus packed QKV plus out_proj, measured
#: 2026-08-16 over 8 checkpoints x 775 positions:
#:
#:     scheme     worst max|dp|   encoder    real MCTS
#:     ffn          1.20e-2       1.379x     78 890/s
#:     ffn+qkv      2.14e-2       1.525x     ~90 100/s
#:     all          2.38e-2       1.567x     92 639/s
#:
#: ⚠️ The worst column **excludes `t24h-muon-008215`**, on Theo's call 2026-08-16: it
#: is the endpoint of the un-decayed muon run whose weight norm reached 3.8x every
#: other network's, and it is the worst checkpoint for every format including e4m3
#: (7.72e-2). Including it the three numbers are 1.84e-2 / 3.02e-2 / 5.08e-2, and the
#: choice between the last two would go the other way. That exclusion is the whole
#: argument, so it is recorded here and not only in the journal.
SCHEME = "all"


class FusedEncoder:
    """Inference-only fused forward, in CUDA C++. See the Triton twin for the
    contract; the differences are internal."""

    D = 256
    DFF = 1024
    T = 32
    H = 8
    ACC_DTYPES = ("fp16", "fp32")

    # The head matrix the kernel wants: policy in rows 0-63, promo in 64-67,
    # value in 68 (or 68-70 for win/draw/loss), zeros to 96 so the aux tile is a
    # whole four-n-tile group. ⚠️ `VALUE_ROW` and the class order are pinned to
    # `csrc/encoder.cu`'s `VALUE_COL`; the kernel reads these rows by index and a
    # permutation here would be silent -- three real logits in the wrong order give a
    # plausible value.
    NHEAD = 96
    VALUE_ROW = N_POLICY + N_PROMO
    # ⚠️ A class attribute and not only an instance one: `_pack_tail` is the only
    # writer and it runs only when this is built from a whole `BrokefishNet`. The
    # backbone-only construction never has a value head at all, and an attribute that
    # exists in one arm and not the other is the shape of CLAUDE.md's third trap.
    n_value = 1
    #: 1 when the value head is the masked mean over live tokens (and therefore
    #: predicts White's frame), 0 for spec §7.4's king row select. Same lifetime and
    #: same reason as `n_value`.
    value_pooled = 0
    EMB_TABLES = ("emb_square", "emb_type_special", "emb_color_turn",
                  "emb_clock", "emb_rep")

    def __init__(self, source, acc_dtype: str = "fp16", fp8: bool = False,
                 int8: bool = False, two_boards: bool | None = None,
                 int8_scheme: str = SCHEME):
        net = source if isinstance(source, BrokefishNet) else None
        encoder = net.encoder if net is not None else source
        layers = encoder.layers
        first = layers[0]

        if acc_dtype not in self.ACC_DTYPES:
            raise ValueError(f"acc_dtype must be one of {self.ACC_DTYPES}, got {acc_dtype!r}")
        if acc_dtype == "fp32":
            # The kernel only emits mma...f16.f16.f16.f16. Accepting the flag and
            # returning an fp16-accumulated result would pass every tolerance
            # test in the suite while silently defeating the one thing the flag
            # exists for, so it refuses instead. Triton has the working fp32 path.
            raise NotImplementedError(
                "the CUDA kernel accumulates in fp16 only; use encoder_impl('triton') "
                "with acc_dtype='fp32' for the wide accumulator")
        self.acc_dtype = acc_dtype

        self.n_layers = len(layers)
        self.d_model = first.self_attn.embed_dim
        self.n_heads = first.self_attn.num_heads
        self.d_ff = first.linear1.out_features
        self.d_head = self.d_model // self.n_heads
        self.eps = first.norm1.eps

        if (self.d_model, self.d_ff, self.n_heads) != (self.D, self.DFF, self.H):
            raise ValueError(
                f"kernel is compiled for d_model={self.D}, d_ff={self.DFF}, heads={self.H}; "
                f"got {self.d_model}, {self.d_ff}, {self.n_heads}. Widening it means "
                "editing the constants in csrc/encoder.cu and re-tuning."
            )

        scale = self.d_head ** -0.5
        packed, ff1_folded, qkv_folded = [], [], []
        for lay in layers:
            w_qkv = lay.self_attn.in_proj_weight.detach().float().clone()
            b_qkv = lay.self_attn.in_proj_bias.detach().float().clone()
            w_qkv[: self.D] *= scale
            b_qkv[: self.D] *= scale
            w_ff1 = lay.linear1.weight.detach().float().clone()
            b_ff1 = lay.linear1.bias.detach().float().clone()
            w_qkv, b_qkv = _fold_norm(w_qkv, b_qkv, lay.norm1)
            w_ff1, b_ff1 = _fold_norm(w_ff1, b_ff1, lay.norm2)
            ff1_folded.append(w_ff1)
            # ⚠️ Post-`1/sqrt(d_head)` **and** post-fold, i.e. exactly the matrix the
            # kernel's fp16 path multiplies by. Both rescalings move a 128-output
            # block's amax, so quantising `in_proj_weight` itself would be a different
            # network rather than a different rounding.
            qkv_folded.append(w_qkv)
            # ⚠️ Ones and zeros, not the real affine. The kernel's `layernorm<false>`
            # does not read these, and writing the true values back would leave a slab
            # that is wrong for anything that does. Writing the identity means a path
            # which still applies the affine stays *correct*, only slower.
            one = torch.ones(self.D, dtype=torch.float, device=w_qkv.device)
            zero = torch.zeros(self.D, dtype=torch.float, device=w_qkv.device)
            # Matrices go through pack_b; vectors (LayerNorm affine, biases)
            # stay in their natural order -- the kernel indexes those by column.
            packed += [
                one, zero,
                pack_b(w_qkv), b_qkv,
                pack_b(lay.self_attn.out_proj.weight.detach()),
                lay.self_attn.out_proj.bias.detach(),
                one, zero,
                pack_b(w_ff1), b_ff1,
                pack_b(lay.linear2.weight.detach()), lay.linear2.bias.detach(),
            ]
        self.weights = torch.cat([t.reshape(-1).half() for t in packed]).contiguous().cuda()

        # §fp8. Built only when asked, because it is 512 KB per layer of extra device
        # memory and every measurement before 2026-08-04 was taken without it.
        # `docs/journal/2026-08-04-fp8-encoder.md`: the FFN's two matmuls in e4m3,
        # 1.70x on the tile, 0.66 % max prior-space error measured in emulation.
        #
        # §int8. The same two matmuls, the same slab layout, the same packed B order --
        # `docs/journal/2026-08-14-int8-kernel-spec.md`. int8 issues at exactly the e4m3
        # rate (72.2 TFLOPS, both) and measures 2.3x lower flip risk, because our tiles
        # span ~2 binades of e4m3's 18 and an exponent buys nothing on data with no
        # outliers.
        #
        # ⚠️ The two slabs are byte-identical in shape and size, so nothing downstream
        # can tell them apart. `quant` is passed explicitly to the kernel for exactly
        # that reason: int8 bytes read as e4m3 stay in range and produce plausible
        # logits, which is a wrong answer that no assertion would catch.
        self.fp8 = bool(fp8)
        self.int8 = bool(int8)
        if self.fp8 and self.int8:
            raise ValueError("fp8 and int8 quantise the same two matmuls; pick one")
        # ⚠️ **`int8_scheme` is not a decision anybody should be making.** `--int8`
        # selects `SCHEME` and that is the whole interface; this argument exists so a
        # bench or a test can hold two schemes side by side in one process, which is
        # the only way an interleaved A/B on this card is valid. See `SCHEME`.
        if int8_scheme not in SCHEMES:
            raise ValueError(f"int8_scheme must be one of {sorted(SCHEMES)}, "
                             f"got {int8_scheme!r}")
        self.int8_scheme = int8_scheme
        self.quant = SCHEMES[int8_scheme] if self.int8 else (1 if self.fp8 else 0)
        # ⚠️ Two boards per CTA is an *occupancy* change, not a numerics one: the
        # logits come out **bit-identical**, because a board's arithmetic cannot depend
        # on who it shares a CTA with. What it buys is half the L2->SM weight traffic
        # per board -- measured 1.18x inside the real MCTS.
        #
        # It therefore defaults **on** whenever int8 is on, and there is deliberately no
        # flag for it above this layer: a switch whose two settings produce identical
        # numbers is not a decision anybody should have to make, and offering it invites
        # a run configured the slow way for no reason. `two_boards=False` stays
        # reachable so `tests/test_quant.py` can prove the two agree bit for bit.
        self.two_boards = bool(self.int8) if two_boards is None else bool(two_boards)
        if self.two_boards and self.fp8:
            raise ValueError("two_boards is built for int8 and fp16; int8 dominates fp8")
        if self.quant >= 3 and not self.two_boards:
            raise ValueError(f"the {self.int8_scheme!r} scheme is instantiated at two "
                             f"boards per CTA only")
        self._wq8 = torch.empty(0, dtype=torch.uint8, device="cuda")
        self._sq8 = torch.empty(0, dtype=torch.float, device="cuda")
        if self.fp8:
            self._pack_fp8(layers, ff1_folded)
        elif self.int8:
            self._pack_int8(layers, ff1_folded, qkv_folded)

        self._empty = torch.empty(0, dtype=torch.int8, device="cuda")
        self._empty_h = torch.empty(0, dtype=torch.half, device="cuda")
        self._empty_f = torch.empty(0, dtype=torch.float, device="cuda")
        self._buffered = 0
        self._buffered_full = 0
        self.debug_stage = 0

        self.net = net
        if net is not None:
            self._pack_tail(net)

    @property
    def label(self) -> str:
        """What this instance actually runs, for a bench header or a run log.

        Derived from `quant`, not from the constructor arguments, so a default that
        moves (`two_boards`, `SCHEME`) cannot leave a log describing the old one.
        """
        base = {0: "fp16", 1: "e4m3 FFN", 2: "int8 FFN", 3: "int8 FFN+QKV+out_proj",
                4: "int8 FFN+QKV"}[self.quant]
        return base + (", 2 boards/CTA" if self.two_boards else ", 1 board/CTA")

    def _pack_fp8(self, layers, ff1_folded) -> None:
        """The FFN weights in e4m3, in `Fp8Off`/`Fp8SOff` order.

        ⚠️ The order here is `csrc/encoder.cu`'s `Fp8Off`: `w_ff1` then `w_ff2`, one
        slab per layer, and the scales likewise. A permutation is silent -- every byte
        is in range and the numbers stay plausible -- which is why
        `csrc/tests/tfp8.cu` packs independently in C++ and
        `tests/test_quant.py::test_the_packed_layout_decodes_back` re-derives the index
        rather than re-running `pack_b_fp8`.
        """
        from brokefish.nn.quant import Q_MAX, pack_b_fp8, quantise_weights_bytes

        blobs, scales = [], []
        for lay, w_ff1 in zip(layers, ff1_folded):
            # ⚠️ `w_ff1` is the **folded** matrix, `linear1.weight * norm2.gamma`, not
            # the module's. Quantising the unfolded one would be a different network:
            # the kernel's LayerNorm no longer applies gamma, so it has to be in here.
            # It is also a numerical change -- gamma rescales the input axis and so
            # moves each 128-output block's amax -- which is why the prior-space
            # comparison in nn/validate.py is the gate on this and not the tests.
            for w in (w_ff1, lay.linear2.weight.detach()):
                # ⚠️ `Q_MAX` explicitly, never the default. The activations are
                # quantised to `fp8::kQMax` inside the kernel and the two have to be
                # the same number or the fp16 accumulator overflows.
                #
                # ⚠️ `tile_k = K`: one scale per 128 output columns for the **whole**
                # reduction, because `gemm_fp8_row` runs one fp16 accumulator over the
                # whole depth and can only descale once. A per-128-k scale here would
                # be silently ignored except for its first column, which is a wrong
                # answer that stays finite and plausible.
                qb, sc = quantise_weights_bytes(w.float(), Q_MAX, tile_k=w.shape[1])
                blobs.append(pack_b_fp8(qb).reshape(-1))
                scales.append(sc.reshape(-1))
        self._wq8 = torch.cat(blobs).contiguous().cuda()
        self._sq8 = torch.cat(scales).float().contiguous().cuda()

    def _pack_int8(self, layers, ff1_folded, qkv_folded) -> None:
        """The quantised weights in int8, in `Fp8OffT<QATT>`/`Fp8SOffT<QATT>` order.

        ⚠️ **`pack_b_fp8` is reused verbatim and that is correct, not lazy.** The
        m16n8k32 B-fragment map is a property of the element *width*: eight bytes per
        lane, laid out the same whether the bytes are e4m3 or s8. Reusing it is also
        what keeps `csrc/tests/tfp8.cu`'s independent C++ packing and
        `test_the_packed_layout_decodes_back` load-bearing for this path too.

        ⚠️ `w_ff1` is the **folded** matrix, `linear1.weight * norm2.gamma`, for the
        same reason the e4m3 packer takes it: the kernel's LayerNorm no longer applies
        gamma, and folding moves each 128-output block's amax, so quantising the
        unfolded matrix would be a different network rather than a different rounding.
        """
        from brokefish.nn.quant import pack_b_fp8, quantise_weights_int8_bytes

        blobs, scales = [], []
        for lay, w_ff1, w_qkv in zip(layers, ff1_folded, qkv_folded):
            # ⚠️ Order is `Fp8OffT`'s and a permutation is silent -- every byte stays in
            # range. The two attention matrices are appended *after* the FFN's so that
            # `w_ff1`/`w_ff2` keep their offsets and only the stride moves.
            mats = [w_ff1, lay.linear2.weight.detach()]
            if self.quant >= 3:
                mats += [w_qkv, lay.self_attn.out_proj.weight.detach()]
            for w in mats:
                # No `q_max` argument, and none exists: int8 has no accumulator squeeze
                # to co-ordinate across two languages. `Q_MAX`'s cross-language pin was
                # a day's debugging; this path cannot have that bug.
                qb, sc = quantise_weights_int8_bytes(w.float())
                blobs.append(pack_b_fp8(qb).reshape(-1))
                scales.append(sc.reshape(-1))
        self._wq8 = torch.cat(blobs).contiguous().cuda()
        self._sq8 = torch.cat(scales).float().contiguous().cuda()

    # -- B2 weight slabs ---------------------------------------------------

    def _pack_tail(self, net: BrokefishNet) -> None:
        """The embedding tables and the head matrix, in the layouts the kernel reads.

        The tables stay row-major: they are gathers, not mma operands, so a lane
        wants eight contiguous columns of one row and that is what row-major
        already gives. Only the head matrix goes through ``pack_b``.
        """
        self.emb = torch.cat([
            getattr(net, name).weight.detach().reshape(-1).half()
            for name in self.EMB_TABLES
        ]).contiguous().cuda()

        head = torch.zeros(self.NHEAD, self.D, dtype=torch.half, device="cuda")
        head[:N_POLICY] = net.policy.weight.detach().half()
        head[N_POLICY:N_POLICY + N_PROMO] = net.promo.weight.detach().half()
        # 1 row for the scalar tanh head, 3 for win/draw/loss. The rest of the aux
        # tile stays zero, as it already was for 27 of its 32 columns.
        self.n_value = int(net.value.weight.shape[0])
        self.value_pooled = int(getattr(net, "value_head", "king") == "pooled")
        head[self.VALUE_ROW:self.VALUE_ROW + self.n_value] = net.value.weight.detach().half()
        self.tail = torch.cat([
            net.norm_f.weight.detach().reshape(-1).half().cuda(),
            net.norm_f.bias.detach().reshape(-1).half().cuda(),
            pack_b(head),
        ]).contiguous().cuda()

    def _require_net(self, what: str) -> BrokefishNet:
        if self.net is None:
            raise ValueError(
                f"{what} needs the whole model: build this from a BrokefishNet, not "
                "from a bare nn.TransformerEncoder")
        return self.net

    # -- the pre-B2 backbone path, unchanged -------------------------------

    @staticmethod
    def _alive_arg(alive, n_boards, seq_len):
        if alive.shape != (n_boards, seq_len):
            raise ValueError(f"alive must be [{n_boards}, {seq_len}], got {tuple(alive.shape)}")
        if alive.dtype is torch.bool:
            alive = alive.view(torch.int8)
        elif alive.dtype is not torch.int8:
            raise ValueError(f"alive must be bool or int8, got {alive.dtype}")
        return alive.contiguous()

    def forward(self, x, alive=None, out=None):
        n_boards, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")
        alive_arg = self._alive_arg(alive, n_boards, seq_len) if alive is not None else self._empty

        if out is None:
            if self._buffered != n_boards:
                self.state = torch.empty(n_boards, seq_len, d_model, device="cuda",
                                         dtype=torch.half)
                self._buffered = n_boards
            out = self.state
        out.copy_(x)
        _ext().encoder_forward(out, self.weights, alive_arg, self.n_layers, self.eps, self.debug_stage)
        return out

    __call__ = forward

    # -- the B2 path: boards in, logits out -------------------------------

    def _ensure_full_buffers(self, n: int) -> None:
        if self._buffered_full != n:
            self.policy_out = torch.empty(n, self.T, N_POLICY, device="cuda", dtype=torch.half)
            self.promo_out = torch.empty(n, self.T, N_PROMO, device="cuda", dtype=torch.half)
            self.value_out = torch.empty(n, device="cuda", dtype=torch.float)
            self._buffered_full = n

    @staticmethod
    def _check_inputs(boards, control, rep):
        n = boards.shape[0]
        if boards.dim() != 2 or boards.shape[1] != FusedEncoder.T:
            raise ValueError(f"boards must be [n, 32], got {tuple(boards.shape)}")
        if boards.dtype is not torch.int16:
            raise ValueError(f"boards must be int16, got {boards.dtype}")
        if control.shape != (n,) or control.dtype is not torch.int16:
            raise ValueError(f"control must be [{n}] int16, got {tuple(control.shape)} {control.dtype}")
        if rep.shape != (n,) or rep.dtype is not torch.uint8:
            raise ValueError(f"rep must be [{n}] uint8, got {tuple(rep.shape)} {rep.dtype}")
        return n

    def forward_full(self, boards, control, rep):
        """Boards to (policy_logits, promo, value) in one launch.

        ``policy_logits`` is raw: the legality mask is the search's, applied
        wherever the masked softmax lives. Returns views on internal buffers,
        which the next call overwrites.
        """
        self._require_net("forward_full")
        n = self._check_inputs(boards, control, rep)
        self._ensure_full_buffers(n)
        _ext().model_forward(
            boards.contiguous(), control.contiguous(), rep.contiguous(),
            self.weights, self.emb, self.tail,
            self.policy_out, self.promo_out, self.value_out, self._empty_h,
            self.n_layers, self.eps, 0, self._wq8, self._sq8, self.quant,
            int(self.two_boards), self.n_value, self.value_pooled)
        return self.policy_out, self.promo_out, self.value_out

    def forward_stage(self, boards, control, rep, stage: int):
        """Activations at an intermediate stage, for testing the prologue and epilogue.

        Stage 9 is the embedding gather alone -- no layer runs -- and stage 8 is
        the output of ``norm_f``. The stages in between are the pre-B2 ones and
        mean the same thing they always did, now fed from boards.
        """
        self._require_net("forward_stage")
        n = self._check_inputs(boards, control, rep)
        if self._buffered != n:
            self.state = torch.empty(n, self.T, self.D, device="cuda", dtype=torch.half)
            self._buffered = n
        _ext().model_forward(
            boards.contiguous(), control.contiguous(), rep.contiguous(),
            self.weights, self.emb, self.tail,
            self._empty_h, self._empty_h, self._empty_f, self.state,
            self.n_layers, self.eps, stage, self._wq8, self._sq8, self.quant,
            int(self.two_boards))
        return self.state
