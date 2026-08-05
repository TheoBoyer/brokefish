"""`brokefish/nn/quant.py`: the e4m3 scheme, before any kernel is written against it.

⚠️ **This module is a specification, not a utility.** The fp8 kernel will be checked
against these functions rather than against a second transcription of them, so an
error here would be copied into the kernel and then confirmed by the comparison. Every
check below is either an exact identity or a bound derived from the format, never a
tolerance chosen to make it pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from brokefish.nn.quant import (BODY_MATMULS, E4M3_MAX, FFN, MATMULS, MMA_K, QuantConfig,
                                body_weights, fake_fp8, fp8_linear, quantise_activations,
                                quantise_weights)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the emulation runs on the GPU")


def _net(seed: int = 0):
    from brokefish.nn.model import BrokefishNet

    torch.manual_seed(seed)
    return BrokefishNet().cuda().eval()


def _positions(n: int = 24):
    from tests.boards import random_positions

    b, c, _ = random_positions(n, plies=30, seed=1, device="cuda")
    return b, c, torch.zeros(b.shape[0], dtype=torch.uint8, device="cuda")


# -- the quantisers ---------------------------------------------------------- #

@pytest.mark.parametrize("q_max", [448.0, 16.0])
@pytest.mark.parametrize("tile_k", [32, 128])
def test_activation_tiles_use_the_whole_range_and_never_nan(q_max, tile_k):
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda") * torch.logspace(-3, 2, 256, device="cuda")
    q, scale = quantise_activations(x, q_max, tile_k)
    assert scale.shape == (64, 256 // tile_k)
    assert torch.isfinite(q).all(), "e4m3 has no infinity: an overflow becomes NaN"
    # The tile maximum has to land on `q_max`, or the format is not being used and the
    # quantisation is throwing away mantissa for nothing.
    tiles = q.view(64, 256 // tile_k, tile_k).abs().amax(-1)
    live = tiles > 0
    assert float((tiles[live] - q_max).abs().max()) <= q_max * 0.13, tiles[live].max()
    # e4m3 keeps 3 mantissa bits, so a value is within 2^-4 of its neighbour's midpoint.
    rel = ((q * scale.repeat_interleave(tile_k, 1) - x).abs()
           / x.abs().clamp(min=1e-30))
    big = x.abs() > x.abs().amax() * 1e-3
    assert float(rel[big].max()) < 0.07, float(rel[big].max())


def test_an_all_zero_tile_stays_zero():
    x = torch.zeros(4, 128, device="cuda")
    q, scale = quantise_activations(x, 16.0, 128)
    assert float(q.abs().max()) == 0.0 and float(scale.min()) > 0


def test_weight_blocks_are_two_dimensional():
    """A 128x128 block scale is one number for 128 *output channels* as well as 128
    inputs. Getting the reshape wrong gives a scale that is right on average and wrong
    per row, which is invisible in a mean and fatal in a max."""
    torch.manual_seed(0)
    w = torch.randn(256, 512, device="cuda")
    w[0, :] *= 100.0                       # one loud row inside block (0, *)
    q, scale = quantise_weights(w, 448.0, 128, 128)
    assert scale.shape == (2, 4)
    back = q * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.isfinite(back).all()
    # The loud row dominates its own blocks and must not disturb the other block row.
    assert float(scale[0, 0]) > 10 * float(scale[1, 0])


@pytest.mark.parametrize("bad", [(200, 256), (256, 200)])
def test_a_shape_that_does_not_tile_is_refused(bad):
    with pytest.raises(ValueError):
        quantise_weights(torch.zeros(*bad, device="cuda"), 448.0, 128, 128)


def test_a_tile_that_ends_mid_instruction_is_refused():
    with pytest.raises(ValueError, match="mma"):
        QuantConfig(tile_k=MMA_K + 1)


# -- the linear -------------------------------------------------------------- #

def test_the_tiled_path_with_nothing_on_is_the_ordinary_linear():
    """The self-check the whole ladder rests on: split into k-tiles, descale, promote,
    accumulate -- with no quantisation this must reproduce `F.linear`. Anything else
    means every fp8 number measured through it is a tiling bug wearing a costume."""
    torch.manual_seed(0)
    x = torch.randn(97, 256, device="cuda")
    w = torch.randn(384, 256, device="cuda")
    b = torch.randn(384, device="cuda")
    want = F.linear(x, w, b)
    for tile_k in (32, 64, 128):
        got = fp8_linear(x, w, b, QuantConfig(tile_k=tile_k))
        rel = float((got - want).abs().max() / want.abs().max())
        assert rel < 1e-6, f"tile_k={tile_k}: {rel:.2e}"


def test_the_fp16_accumulator_overflows_at_full_range():
    """Pinned because it is the constraint that sets `q_max`, and because it is silent:
    fp16 has no saturation, so the partial sum becomes `inf` and the logits go NaN
    rather than merely losing precision. Measured on the real network: `q_max = 448`
    produced 868,623 non-finite outputs and `q_max = 64` still produced 121,912."""
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")
    hot = QuantConfig(weights=True, activations=True, acc16=True, q_max=448.0)
    cold = QuantConfig(weights=True, activations=True, acc16=True, q_max=16.0)
    assert not torch.isfinite(fp8_linear(x, w, None, hot)).all()
    assert torch.isfinite(fp8_linear(x, w, None, cold)).all()


# -- the patch, which is where the measurement can silently become vacuous ---- #

def test_the_patch_fires_on_every_body_matmul():
    net = _net()
    b, c, r = _positions()
    with torch.no_grad(), fake_fp8(net, QuantConfig(), expect_calls=BODY_MATMULS) as n:
        net(b, c, r)
    assert n[0] == BODY_MATMULS == 32


def test_the_patch_disables_the_fast_path_and_restores_it():
    """⚠️ Without this, the whole emulation is vacuous. `TransformerEncoderLayer`
    dispatches to `torch._transformer_encoder_layer_fwd`, which never calls
    `F.linear`: measured, 3 calls per forward instead of 35, and all three are the
    heads. A patch installed with the fast path left on emulates nothing and reports
    that fp8 costs nothing."""
    was = torch.backends.mha.get_fastpath_enabled()
    net = _net()
    with fake_fp8(net, QuantConfig()):
        assert torch.backends.mha.get_fastpath_enabled() is False
    assert torch.backends.mha.get_fastpath_enabled() is was

    # And the count it protects: with the fast path on, F.linear sees only the heads.
    b, c, r = _positions(8)
    calls = [0]
    original = F.linear
    F.linear = lambda i, w, bi=None: (calls.__setitem__(0, calls[0] + 1),
                                      original(i, w, bi))[1]
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        with torch.no_grad():
            net(b, c, r)
    finally:
        F.linear = original
        torch.backends.mha.set_fastpath_enabled(was)
    assert calls[0] == 3, f"expected the 3 heads only, got {calls[0]}"


def test_a_wrong_call_count_is_an_error():
    net = _net()
    b, c, r = _positions(4)
    with pytest.raises(AssertionError, match="quantised path ran"):
        with torch.no_grad(), fake_fp8(net, QuantConfig(), expect_calls=999):
            net(b, c, r)


@pytest.mark.parametrize("which,count", [(FFN, 16), (("qkv",), 8), (MATMULS, 32)])
def test_selecting_a_subset_selects_exactly_that(which, count):
    net = _net()
    b, c, r = _positions(4)
    assert len(body_weights(net, which)) == count
    with torch.no_grad(), fake_fp8(net, QuantConfig(), expect_calls=count,
                                   which=which) as n:
        net(b, c, r)
    assert n[0] == count


def test_an_unknown_matmul_name_is_refused():
    with pytest.raises(ValueError, match="unknown matmul"):
        body_weights(_net(), ("ffn3",))


# -- the packed weight layout ------------------------------------------------ #

def test_the_packed_layout_decodes_back():
    """⚠️ The index is **re-derived here**, not by calling `pack_b_fp8` again.

    A test that reproduces the permutation confirms only that `permute` is
    deterministic. This one walks (n8, k32, lane, half16, j) and asserts the byte
    found there is the one the fragment definition says belongs there, which is the
    same discipline `csrc/tests/tdirect.cu` applies to the fp16 packer.
    """
    from brokefish.nn.quant import pack_b_fp8, quantise_weights_bytes

    torch.manual_seed(0)
    N, K = 128, 256
    w = torch.randn(N, K, device="cuda") * 4.0
    qb, scale = quantise_weights_bytes(w, 16.0, 128, 128)
    assert qb.shape == (N, K) and qb.dtype == torch.uint8
    assert scale.shape == (N // 128, K // 128)

    packed = pack_b_fp8(qb).cpu()
    ref = qb.cpu()
    nk32 = K // 32
    for n8 in range(0, N // 8, 5):                 # a stride, not every one: 2 s vs 40
        for k32 in range(nk32):
            for lane in range(32):
                g, t = lane // 4, lane % 4
                for half16 in range(2):
                    for j in range(4):
                        idx = ((((n8 * nk32 + k32) * 32 + lane) * 2 + half16) * 4 + j)
                        col = k32 * 32 + half16 * 16 + t * 4 + j
                        assert int(packed[idx]) == int(ref[n8 * 8 + g, col]), (
                            n8, k32, lane, half16, j)


def test_the_packed_weights_are_half_the_bytes():
    """The second-order win: the FFN's 8.4 MB of fp16 weights become 4.2 MB."""
    from brokefish.nn.quant import pack_b_fp8, quantise_weights_bytes

    w = torch.randn(1024, 256, device="cuda")
    qb, _ = quantise_weights_bytes(w, 16.0)
    assert pack_b_fp8(qb).numel() == w.numel()
    assert pack_b_fp8(qb).element_size() * w.numel() == w.numel()   # 1 byte each


def test_the_python_and_cuda_q_max_agree():
    """⚠️ Two languages, one constant. `quantise_weights_bytes` defaulted to e4m3's 448
    while the kernel quantised activations at 16; the product then reached 7168, one
    mma summed past fp16's 65504, and the infinity became a NaN two tiles later. Every
    component tested clean in isolation and 116 boards of 128 came out NaN."""
    from brokefish.nn._build import load_extension
    from brokefish.nn.quant import Q_MAX

    ext = load_extension("brokefish_encoder", ["encoder.cu"])
    assert float(ext.fp8_q_max()) == Q_MAX
    # And the bound it exists to satisfy: one mma sums 32 products of two operands.
    assert MMA_K * Q_MAX * Q_MAX < 65504.0
