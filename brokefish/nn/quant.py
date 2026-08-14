"""e4m3 quantisation for the encoder GEMMs: the scheme, and an emulation of it.

`docs/journal/2026-08-04-fp8-encoder.md` is why. In one line: `mma.m16n8k32` with an
**fp16 accumulator** runs at 72.0 TFLOPS on this card against the 35.6 the shipped
encoder is ceilinged by, so fp8 is a 2× lever — if the accuracy survives 3 mantissa
bits. This module answers that in torch before any kernel is written.

**The scheme is defined here and nowhere else.** When the kernel exists it will be
checked against these functions, not against a second transcription of them. Four
copies of one expression is how this repository lost three days to first-play urgency.

## What is quantised

The four weight matmuls of each `TransformerEncoderLayer` -- packed QKV, attention
output, and the two feed-forward projections -- which are 98 % of the FLOPs.
LayerNorm, the softmax, `QK^T`, `AV`, the five embedding tables and the three heads
stay in fp16, which is DeepSeek-V3's own exclusion list and costs nothing: attention's
own matmuls are 2 % of the network and the heads are 0.3 %.

## The two granularities, and why they are not one

Per-tensor scaling is what everyone tries first and DeepSeek-V3 reports it
destabilised training. This uses their granularity:

* activations, **1 x 128 along the contracted axis** -- one scale per row per k-tile;
* weights, **128 x 128 blocks**.

The shapes divide exactly: `d_model = 256`, `3d = 768`, `d_ff = 1024`.

The granularity is chosen by what a kernel can actually do. `out[m,n] = sum_k A W` is
computed k-tile by k-tile; inside one tile both operands carry a single scale, so the
tile's product sum can be descaled **once**, with one multiply, and added to a wider
running total. Any finer and the descale lands inside the mma loop.

## ⚠️ The accumulator is the whole problem

The 2× comes from accumulating in **fp16**, which is not what DeepSeek does -- they
accumulate in fp32 precisely because Hopper's fp8 path is effectively 14-bit. Going
the other way has a hard constraint nobody writes down:

    an mma accumulates 32 products per instruction. With operands scaled to e4m3's
    full +-448, one such partial sum reaches 32 * 448^2 = 6.4e6, and fp16 stops at
    65504.

So **quantising to the full e4m3 range overflows the accumulator**, and the overflow
is silent-ish: fp16 has no saturation, the partial becomes `inf`, and the logits come
out NaN. The knob is :attr:`QuantConfig.q_max` -- the value the tile maximum is mapped
to instead of 448. Smaller is safer in the accumulator and worse at the bottom of the
tile, because e4m3's smallest normal is 2^-6, so anything below `q_max / 64` of the
tile max goes subnormal and starts losing mantissa bits. That trade is a measurement,
not a guess, and :func:`sweep_q_max` is how it is taken.

Typical magnitudes are well under the maximum and signs cancel, so the practical
bound is nearer `sqrt(32) * (q_max/4)^2`; at `q_max = 448` that is ~7e4, i.e. exactly
at the fp16 edge. `q_max = 448` is therefore expected to be marginal rather than
comfortably wrong, which is why the emulation models fp16 rounding after every
32-element chunk and lets the overflow happen instead of asserting it away.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F

# IEEE-ish e4m3 as NVIDIA defines it and as `mma...e4m3` consumes it: no infinities,
# one NaN encoding, maximum 448. `torch.float8_e4m3fn` is that format; `e4m3fnuz` is
# AMD's and has a different bias, so the `fn` suffix is load-bearing.
E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
# The smallest *normal* e4m3. Below this the mantissa starts disappearing, which is
# the cost of a small `q_max`.
E4M3_MIN_NORMAL = 2.0 ** -6

# The k-tile of the activation scale and the square block of the weight scale, from
# DeepSeek-V3. `MMA_K` is `mma.sync.m16n8k32`'s k, i.e. how many products land in the
# accumulator between two fp16 roundings.
TILE_K = 128
BLOCK_N = 128
MMA_K = 32

#: ⚠️ **Must equal `fp8::kRowK` in csrc/fp8_gemm.cuh.** The k-depth one activation
#: scale covers, and therefore the depth of the kernel's single fp16 accumulator. The
#: shipping scheme scales per *row* rather than per 128-tile, because the per-tile form
#: needs an `float acc[2][4][4]` running total on top of the mma fragment and the
#: kernel has exactly 128 registers -- 65536 / (2 CTAs * 256 threads). Measured:
#: per-tile spilled 960 B and ran 1.154x; giving it the registers instead
#: (`__launch_bounds__(THREADS, 1)`, 200 registers, no spill) ran **0.978x**.
ROW_K = 256

#: ⚠️ **Must equal `fp8::kQMax` in csrc/fp8_gemm.cuh.** This is the value a row's
#: maximum is mapped to instead of e4m3's 448, and it is what keeps the fp16
#: accumulator in range: `ROW_K` products of two `q_max`-bounded operands is
#: `ROW_K * q_max^2`, which at 8 is 16384 against fp16's 65504 and at 448 is 5.1e7.
#:
#: It cost a day. `quantise_weights_bytes` defaulted to `E4M3_MAX` while the kernel
#: quantised activations at 16, so weight * activation reached 7168, one mma summed to
#: ~2.3e5, the accumulator went infinite, the ReLU passed the infinity through, and the
#: *next* tile's amax was infinite -- making `inv` zero and `inf * 0` a NaN. 116 boards
#: in 128 came out NaN and every isolated component tested clean.
#: `tests/test_quant.py::test_the_python_and_cuda_q_max_agree` now pins the two.
Q_MAX = 8.0

assert ROW_K * Q_MAX * Q_MAX <= 65504.0 / 3.0, "fp16 accumulator margin"


@dataclass(frozen=True)
class QuantConfig:
    """Which parts of the scheme are on. Every one of them defaults to *off*.

    Separately switchable because the question is not "is fp8 accurate enough" but
    "which part of it costs what": weights, activations and the accumulator fail for
    different reasons and the fixes are different.
    """

    weights: bool = False
    activations: bool = False
    #: Round the accumulator to fp16 every `MMA_K` products, as the hardware does.
    acc16: bool = False
    #: What a tile's maximum is mapped to. 448 uses all of e4m3 and overflows an fp16
    #: accumulator; see the module docstring.
    q_max: float = 448.0
    #: The activation scale's k-tile and the weight scale's block. DeepSeek-V3 uses
    #: 128 for a 7168-wide model; ours is 256 wide, so 128 spans *half* the reduction
    #: of `in_proj`/`out_proj`/`linear1` and an eighth of `linear2`'s. Smaller is more
    #: accurate and costs one descale per tile, so it trades directly against speed --
    #: which is why both axes are swept rather than either being assumed.
    tile_k: int = TILE_K
    block_n: int = BLOCK_N

    @property
    def any(self) -> bool:
        return self.weights or self.activations or self.acc16

    def label(self) -> str:
        bits = [n for n, on in (("W", self.weights), ("A", self.activations),
                                ("acc16", self.acc16)) if on]
        tail = f"@{self.q_max:g}/k{self.tile_k}" if self.any else ""
        return "+".join(bits or ["off"]) + tail

    def __post_init__(self) -> None:
        if self.tile_k % MMA_K:
            raise ValueError(
                f"tile_k = {self.tile_k} is not a multiple of the mma's k = {MMA_K}. "
                f"The descale happens between mma instructions, so a tile that ends "
                f"mid-instruction is not expressible in the kernel this emulates")


def _to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to e4m3 and back to fp32.

    ⚠️ The clamp is not defensive padding. `scale = amax / q_max` makes the largest
    entry land on `q_max` in exact arithmetic, but the division rounds, so it can come
    out a hair above -- and e4m3 has **no infinity**, so a value past 448 becomes NaN
    rather than saturating. One NaN in one tile poisons the whole row's logits.
    """
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).to(torch.float32)


def quantise_activations(x: torch.Tensor, q_max: float = E4M3_MAX,
                         tile_k: int = TILE_K):
    """`[M, K]` -> `(q [M, K], scale [M, K // tile_k])`, one scale per row per k-tile.

    The scale is computed **online**, from this tensor. DeepSeek-V3 uses delayed
    scaling with a history for training stability across steps; at inference there is
    no history to keep and the tile is in registers when its maximum is needed, so
    online is both simpler and strictly more accurate.
    """
    M, K = x.shape
    if K % tile_k:
        raise ValueError(f"K = {K} is not a multiple of the {tile_k}-element tile")
    tiles = x.view(M, K // tile_k, tile_k)
    amax = tiles.abs().amax(dim=-1, keepdim=True)
    # An all-zero tile has no scale; 1.0 leaves it exactly zero after the round trip.
    scale = torch.where(amax > 0, amax / q_max, torch.ones_like(amax))
    return _to_e4m3(tiles / scale).view(M, K), scale.squeeze(-1)


def quantise_weights(w: torch.Tensor, q_max: float = E4M3_MAX,
                     tile_k: int = TILE_K, block_n: int = BLOCK_N):
    """`[N, K]` -> `(q [N, K], scale [N // block_n, K // tile_k])`, blockwise.

    Weights are static at inference, so both the quantised values and the scales are
    computed once and stored; none of this is on the hot path.
    """
    N, K = w.shape
    if N % block_n or K % tile_k:
        raise ValueError(f"weight {tuple(w.shape)} does not tile into "
                         f"{block_n}x{tile_k} blocks")
    blocks = w.view(N // block_n, block_n, K // tile_k, tile_k)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.where(amax > 0, amax / q_max, torch.ones_like(amax))
    return _to_e4m3(blocks / scale).view(N, K), scale.reshape(N // block_n, K // tile_k)


def quantise_weights_bytes(w: torch.Tensor, q_max: float = Q_MAX,
                           tile_k: int = TILE_K, block_n: int = BLOCK_N):
    """`[N, K]` -> `(bytes [N, K] uint8, scale [N // block_n, K // tile_k] fp32)`.

    The same scheme as :func:`quantise_weights`, returning the **stored e4m3 bytes**
    rather than their float values, because that is what the kernel reads. Weights
    are static at inference, so this runs once at `FusedEncoder` construction.
    """
    N, K = w.shape
    if N % block_n or K % tile_k:
        raise ValueError(f"weight {tuple(w.shape)} does not tile into "
                         f"{block_n}x{tile_k} blocks")
    blocks = w.float().reshape(N // block_n, block_n, K // tile_k, tile_k)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.where(amax > 0, amax / q_max, torch.ones_like(amax))
    q = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).reshape(N, K)
    return q.view(torch.uint8), scale.reshape(N // block_n, K // tile_k).contiguous()


# --------------------------------------------------------------------------- #
# int8, the same scheme in a different format
# --------------------------------------------------------------------------- #

#: ⚠️ **Must equal `int8q::kQMaxS` / `kQMaxU` in csrc/int8_gemm.cuh.** Unlike
#: :data:`Q_MAX` these are not a tuning choice and there is nothing to sweep: they are
#: the format's own limits. The e4m3 path needs `q_max = 8` because an fp16 accumulator
#: sums `ROW_K = 256` products; int8's accumulator is `s32` and the worst partial is
#: `127 * 127 * 256 = 4.13e6` against 2.147e9, a 520x margin. There is no saturation to
#: get wrong, no infinity and no NaN encoding.
Q_MAX_S8 = 127.0
#: The post-ReLU FFN hidden is provably non-negative, so its sign bit is dead weight
#: and `mma...u8.s8.s32` exists. 256 levels instead of 127. A float cannot make this
#: trade, which is why there is no e4m3 analogue.
Q_MAX_U8 = 255.0

assert Q_MAX_S8 * Q_MAX_U8 * ROW_K < 2 ** 31 - 1, "s32 accumulator margin"


def quantise_activations_int8(x: torch.Tensor, unsigned: bool = False):
    """`[M, K]` -> `(q [M, K] float, scale [M, 1])`, one scale per row.

    Symmetric, round to nearest, clamped. ⚠️ `-128` is representable but unreachable:
    with `scale = amax/127` the largest magnitude maps to exactly 127, so the clamp is
    the same defensive line `_to_e4m3`'s is and never actually fires. The kernel's
    `cvt.rni.sat.s8.f32` saturates identically.
    """
    if unsigned:
        amax = x.clamp(min=0).amax(dim=-1, keepdim=True)
        s = torch.where(amax > 0, amax / Q_MAX_U8, torch.ones_like(amax))
        return torch.round(x.clamp(min=0) / s).clamp(0, Q_MAX_U8), s
    amax = x.abs().amax(dim=-1, keepdim=True)
    s = torch.where(amax > 0, amax / Q_MAX_S8, torch.ones_like(amax))
    return torch.round(x / s).clamp(-Q_MAX_S8, Q_MAX_S8), s


def quantise_weights_int8_bytes(w: torch.Tensor, block_n: int = BLOCK_N):
    """`[N, K]` -> `(bytes [N, K] uint8, scale [N // block_n] fp32)`.

    One scale per `block_n` output channels over the **whole** reduction, because
    `gemm_s8_row` reads `w_scale[n8 / 16]` with no k index at all — exactly the
    constraint :func:`quantise_weights_bytes` documents for e4m3, and for the same
    reason: one accumulator spans the whole depth and can only be descaled once.
    """
    N, K = w.shape
    if N % block_n:
        raise ValueError(f"weight {tuple(w.shape)} does not tile into {block_n} rows")
    b = w.float().reshape(N // block_n, block_n, K)
    amax = b.abs().amax(dim=(1, 2), keepdim=True)
    scale = torch.where(amax > 0, amax / Q_MAX_S8, torch.ones_like(amax))
    q = torch.round(b / scale).clamp(-Q_MAX_S8, Q_MAX_S8).to(torch.int8).reshape(N, K)
    return q.view(torch.uint8), scale.reshape(N // block_n).contiguous()


def int8_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor],
                unsigned: bool = False, row_k: int = ROW_K) -> torch.Tensor:
    """`F.linear` as `gemm_s8_row` computes it: exact s32 accumulation, one descale.

    ⚠️ The accumulation is done in float64 rather than float32. int32 is *exact* on the
    device and fp32 is not — `127 * 127 * 256 = 4.13e6` needs 23 bits and fp32 has 24,
    so fp32 would be exact here by one bit and would stop being so the moment anybody
    widened `row_k`. Modelling the hardware's exactness with a type that is only
    accidentally exact is how an emulation stops predicting its kernel.
    """
    shape = x.shape
    a = x.reshape(-1, shape[-1]).float()
    M, K = a.shape
    N = w.shape[0]
    wq, ws = quantise_weights_int8_bytes(w.float())
    wq = wq.view(torch.int8).double()

    out = torch.zeros((M, N), dtype=torch.float64, device=a.device)
    for lo in range(0, K, row_k):
        tq, sa = quantise_activations_int8(a[:, lo:lo + row_k], unsigned)
        out += (tq.double() @ wq[:, lo:lo + row_k].T) * sa.double()
    out = out.float() * ws.repeat_interleave(BLOCK_N)[None, :]
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*shape[:-1], N).to(x.dtype)


def pack_b_fp8(qbytes: torch.Tensor) -> torch.Tensor:
    """Permute e4m3 weight bytes `[N, K]` into `mma.m16n8k32` B-fragment order.

    The fp8 twin of `cuda_impl.pack_b`, and the same contract: the kernel reads its B
    operand straight from global memory into registers with one 64-bit load per lane,
    which only works if the bytes already sit in the order the lanes want.

    Lane `L` (g = L // 4, t = L % 4) of an m16n8k32 fragment holds

        b[0] = W[n0 + g][k0 + 4t   .. 4t+3 ]
        b[1] = W[n0 + g][k0 + 4t+16 .. 4t+19]

    so with `col = k32*32 + half16*16 + t*4 + j` and `row = n8*8 + g` the result is
    indexed `packed[n8][k32][g][t][half16][j]` -- 8 bytes per lane against the fp16
    path's 16.

    ⚠️ This and `gemm_fp8` in csrc/fp8_gemm.cuh must change together, and a mismatch
    is **silent**: every address stays in range and the numbers stay plausible. That
    is why `csrc/tests/tfp8.cu` packs independently in C++ and
    `tests/test_quant.py::test_the_packed_layout_decodes_back` re-derives the index
    here rather than re-running this permutation.
    """
    n, k = qbytes.shape
    if n % 8 or k % 32:
        raise ValueError(f"packed fp8 weights need N%8==0 and K%32==0, got [{n}, {k}]")
    v = qbytes.reshape(n // 8, 8, k // 32, 2, 4, 4)
    #                  n8      g  k32     h16 t  j
    return v.permute(0, 2, 1, 4, 3, 5).reshape(-1).contiguous()


def _tile_product(a: torch.Tensor, b: torch.Tensor, acc16: bool) -> torch.Tensor:
    """`a [M, TILE_K] @ b [N, TILE_K].T` the way the hardware would.

    With ``acc16``, the accumulator is rounded to fp16 after each `MMA_K` products,
    which is what `mma.sync.m16n8k32.f16...f16` does between instructions. The k = 32
    dot product inside one instruction is treated as exact: the hardware's internal
    reduction tree is wider than fp16 and is not documented, so modelling it as exact
    is the assumption that *understates* the error rather than inventing one.

    Overflow is deliberately not guarded. An fp16 partial past 65504 becomes `inf`
    here exactly as it would on the device, and `validate.py` counts non-finite
    outputs -- which is the intended way to discover that `q_max` is too large.
    """
    if not acc16:
        return a @ b.T
    acc = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float16, device=a.device)
    for lo in range(0, a.shape[1], MMA_K):
        chunk = a[:, lo:lo + MMA_K] @ b[:, lo:lo + MMA_K].T
        acc = (acc.float() + chunk).half()
    return acc.float()


def fp8_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor],
               cfg: QuantConfig) -> torch.Tensor:
    """`F.linear` with the scheme above applied to the GEMM.

    The k-tile loop is the kernel's loop, written out: quantise, multiply inside the
    tile, descale once with `sa * sw`, promote, accumulate in fp32 across tiles. What
    a real kernel adds on top is only that `sa` is computed in registers rather than
    ahead of time.
    """
    shape = x.shape
    a = x.reshape(-1, shape[-1]).float()
    wf = w.float()
    M, K = a.shape
    N = wf.shape[0]

    if cfg.activations:
        aq, sa = quantise_activations(a, cfg.q_max, cfg.tile_k)
    else:
        aq, sa = a, None
    if cfg.weights:
        wq, sw = quantise_weights(wf, cfg.q_max, cfg.tile_k, cfg.block_n)
    else:
        wq, sw = wf, None

    out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    for t, lo in enumerate(range(0, K, cfg.tile_k)):
        part = _tile_product(aq[:, lo:lo + cfg.tile_k], wq[:, lo:lo + cfg.tile_k],
                             cfg.acc16)
        if sa is not None:
            part = part * sa[:, t:t + 1]                     # [M,1] broadcast over N
        if sw is not None:
            # One scale per `block_n` output channels, expanded over the block.
            part = part * sw[:, t].repeat_interleave(cfg.block_n)[None, :]
        out += part

    if bias is not None:
        out = out + bias.float()
    return out.reshape(*shape[:-1], N).to(x.dtype)


#: The four matmuls of a layer, by FLOP share of the encoder body:
#: `qkv` 6d^2, `out` 2d^2, `ffn1` 8d^2, `ffn2` 8d^2 per token, of 24d^2 total.
MATMULS = ("qkv", "out", "ffn1", "ffn2")
FLOP_SHARE = {"qkv": 6 / 24, "out": 2 / 24, "ffn1": 8 / 24, "ffn2": 8 / 24}
FFN = ("ffn1", "ffn2")


def body_weights(net, which: Optional[Sequence[str]] = None) -> List[torch.Tensor]:
    """The selected matmul weights of every encoder layer, and nothing else.

    Explicitly **not** the heads, `norm_f`, the LayerNorms or the embeddings: those are
    DeepSeek-V3's exclusion list, and between them they are under 1 % of the FLOPs, so
    quantising them buys nothing and risks the parts of the network with the widest
    dynamic range.
    """
    which = tuple(which) if which is not None else MATMULS
    unknown = [w for w in which if w not in MATMULS]
    if unknown:
        raise ValueError(f"unknown matmul {unknown}, expected some of {MATMULS}")
    out = []
    for layer in net.encoder.layers:
        by_name = {"qkv": layer.self_attn.in_proj_weight,
                   "out": layer.self_attn.out_proj.weight,
                   "ffn1": layer.linear1.weight, "ffn2": layer.linear2.weight}
        out += [by_name[w] for w in which]
    return out


@contextlib.contextmanager
def fake_fp8(net, cfg: QuantConfig, expect_calls: Optional[int] = None,
             which: Optional[Sequence[str]] = None) -> Iterator[list]:
    """Route this net's encoder-body matmuls through :func:`fp8_linear`.

    ⚠️ **The fast path has to be off, and this is the detail the whole measurement
    turns on.** `nn.TransformerEncoderLayer.forward` dispatches to
    `torch._transformer_encoder_layer_fwd` when it can, which never calls `F.linear`.
    Measured on this model: 3 calls per forward instead of 35 -- the three heads --
    so a patch installed without disabling it emulates **nothing at all** and reports
    that fp8 is free. Disabling costs 1.5e-6 on fp32 logits of scale 3.3, which is
    reordering and not a change of function.

    Yields the call-count list so a caller can assert on it; `expect_calls` asserts on
    exit, which is the cheaper habit.
    """
    targets = {id(w) for w in body_weights(net, which)}
    if not targets:
        raise ValueError("no encoder-body weights found; is this a BrokefishNet?")
    calls: list = [0]
    original = F.linear
    was_enabled = torch.backends.mha.get_fastpath_enabled()

    def patched(inp, weight, bias=None):
        if id(weight) not in targets:
            return original(inp, weight, bias)
        calls[0] += 1
        return fp8_linear(inp, weight, bias, cfg)

    torch.backends.mha.set_fastpath_enabled(False)
    F.linear = patched
    try:
        yield calls
    finally:
        F.linear = original
        torch.backends.mha.set_fastpath_enabled(was_enabled)
    if expect_calls is not None and calls[0] != expect_calls:
        raise AssertionError(
            f"the quantised path ran {calls[0]} times, expected {expect_calls}. "
            f"A count of 0 means the patch never fired; anything else means the "
            f"model's shape changed and this scheme no longer covers what it claims")


# --------------------------------------------------------------------------- #
# Stage 1: measure the scheme before writing any kernel
# --------------------------------------------------------------------------- #

#: Four target matmuls per layer, eight layers. `fake_fp8` asserts on this, because a
#: count of zero is what a silently-bypassed patch looks like.
BODY_MATMULS = 32


def emulated_forward(net32, cfg: QuantConfig, which: Optional[Sequence[str]] = None):
    """A `(boards, control, rep) -> (policy, promo, value)` running under the scheme."""
    n_layers = len(net32.encoder.layers)
    expect = n_layers * len(which if which is not None else MATMULS)

    def forward(boards, control, rep):
        with fake_fp8(net32, cfg, expect_calls=expect, which=which):
            return net32(boards, control, rep)
    return forward


def main() -> int:
    import argparse
    import copy

    from brokefish.nn.validate import (drop_terminal, load_net, positions_adversarial,
                                       positions_from_buffer, positions_random, validate)

    p = argparse.ArgumentParser(description="stage 1/2: what e4m3 costs, before a kernel")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--buffer", default=None)
    p.add_argument("--buffer-dir", default="data/replay")
    p.add_argument("--positions", type=int, default=384)
    p.add_argument("--q-max", type=float, nargs="*", default=[16.0])
    p.add_argument("--tile-k", type=int, nargs="*", default=[128, 64, 32])
    p.add_argument("--targets", nargs="*", default=["ffn", "all"],
                   help="'ffn', 'all', or an explicit matmul name")
    a = p.parse_args()

    parts = []
    if a.buffer:
        parts.append(positions_from_buffer(a.buffer, a.positions, a.buffer_dir))
    else:
        parts.append(positions_random(a.positions, plies=60, seed=0))
    parts.append(positions_adversarial())
    boards = torch.cat([x[0] for x in parts])
    control = torch.cat([x[1] for x in parts])
    rep = torch.cat([x[2] for x in parts])
    boards, control, rep = drop_terminal(boards, control, rep)

    net16, source = load_net(a.checkpoint)
    net32 = copy.deepcopy(net16).float()

    def resolve(name):
        return {"ffn": FFN, "all": MATMULS}.get(name, (name,))

    # `off` runs the whole tiled path with nothing quantised: the self-check that every
    # number under it is a cost of e4m3 and not a bug in the tiling.
    ladder = [("off", QuantConfig(), MATMULS)]
    for tgt in a.targets:
        which = resolve(tgt)
        share = sum(FLOP_SHARE[m] for m in which)
        for q in a.q_max:
            for k in a.tile_k:
                cfg = QuantConfig(weights=True, activations=True, acc16=True,
                                  q_max=q, tile_k=k, block_n=min(128, k * 4))
                ladder.append((f"{tgt}/k{k}@{q:g}", cfg, which))
        print(f"  {tgt:>4}: {len(which)} matmuls, {100 * share:.1f} % of body FLOPs, "
              f"ideal speedup {1 / (1 - share + share / 1.83):.2f}x at 1.83x arithmetic")

    names = [n for n, _, _ in ladder]
    forwards = {n: emulated_forward(net32, c, w) for n, c, w in ladder}
    print(f"\ncheckpoint {source}\npositions  {boards.shape[0]} non-terminal\n")
    deltas = validate(None, boards, control, rep, names, verbose=False,
                      forwards=forwards, net=net16)

    print(f"  {'variant':<16} {'max|dp|':>10} {'p95|dp|':>10} {'top1 moved':>11} "
          f"{'flip':>8} {'non-finite':>11}")
    for n in names:
        d = deltas[n]
        print(f"  {n:<16} {d.max_abs:10.3e} {d.p95_abs:10.3e} {d.top1_disagree:10.3%} "
              f"{d.flip_risk:7.2%} {d.nonfinite:11d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
