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


class FusedEncoder:
    """Inference-only fused forward, in CUDA C++. See the Triton twin for the
    contract; the differences are internal."""

    D = 256
    DFF = 1024
    T = 32
    H = 8
    ACC_DTYPES = ("fp16", "fp32")

    # The head matrix the kernel wants: policy in rows 0-63, promo in 64-67,
    # value in 68, zeros to 96 so the aux tile is a whole four-n-tile group.
    NHEAD = 96
    EMB_TABLES = ("emb_square", "emb_type_special", "emb_color_turn",
                  "emb_clock", "emb_rep")

    def __init__(self, source, acc_dtype: str = "fp16", fp8: bool = False):
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
        packed = []
        for lay in layers:
            w_qkv = lay.self_attn.in_proj_weight.detach().clone()
            b_qkv = lay.self_attn.in_proj_bias.detach().clone()
            w_qkv[: self.D] *= scale
            b_qkv[: self.D] *= scale
            # Matrices go through pack_b; vectors (LayerNorm affine, biases)
            # stay in their natural order -- the kernel indexes those by column.
            packed += [
                lay.norm1.weight.detach(), lay.norm1.bias.detach(),
                pack_b(w_qkv), b_qkv,
                pack_b(lay.self_attn.out_proj.weight.detach()),
                lay.self_attn.out_proj.bias.detach(),
                lay.norm2.weight.detach(), lay.norm2.bias.detach(),
                pack_b(lay.linear1.weight.detach()), lay.linear1.bias.detach(),
                pack_b(lay.linear2.weight.detach()), lay.linear2.bias.detach(),
            ]
        self.weights = torch.cat([t.reshape(-1).half() for t in packed]).contiguous().cuda()

        # §fp8. Built only when asked, because it is 512 KB per layer of extra device
        # memory and every measurement before 2026-08-04 was taken without it.
        # `docs/journal/2026-08-04-fp8-encoder.md`: the FFN's two matmuls in e4m3,
        # 1.70x on the tile, 0.66 % max prior-space error measured in emulation.
        self.fp8 = bool(fp8)
        self._wq8 = torch.empty(0, dtype=torch.uint8, device="cuda")
        self._sq8 = torch.empty(0, dtype=torch.float, device="cuda")
        if self.fp8:
            self._pack_fp8(layers)

        self._empty = torch.empty(0, dtype=torch.int8, device="cuda")
        self._empty_h = torch.empty(0, dtype=torch.half, device="cuda")
        self._empty_f = torch.empty(0, dtype=torch.float, device="cuda")
        self._buffered = 0
        self._buffered_full = 0
        self.debug_stage = 0

        self.net = net
        if net is not None:
            self._pack_tail(net)

    def _pack_fp8(self, layers) -> None:
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
        for lay in layers:
            for w in (lay.linear1.weight.detach(), lay.linear2.weight.detach()):
                # ⚠️ `Q_MAX` explicitly, never the default. The activations are
                # quantised to `fp8::kQMax` inside the kernel and the two have to be
                # the same number or the fp16 accumulator overflows.
                qb, sc = quantise_weights_bytes(w.float(), Q_MAX)
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
        head[N_POLICY + N_PROMO] = net.value.weight.detach().half()[0]
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
            self.n_layers, self.eps, 0, self._wq8, self._sq8)
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
            self.n_layers, self.eps, stage, self._wq8, self._sq8)
        return self.state
