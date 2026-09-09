"""`brokefish/nn/quant.py`: the e4m3 scheme, before any kernel is written against it.

⚠️ **This module is a specification, not a utility.** The fp8 kernel will be checked
against these functions rather than against a second transcription of them, so an
error here would be copied into the kernel and then confirmed by the comparison. Every
check below is either an exact identity or a bound derived from the format, never a
tolerance chosen to make it pass.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from brokefish.nn.cuda_impl import FusedEncoder
from brokefish.nn.model import BrokefishNet
from tests.boards import random_positions

from brokefish.nn.quant import (BODY_MATMULS, E4M3_MAX, FFN, MATMULS, MMA_K, QuantConfig,
                                body_weights, fake_fp8, fp8_linear, quantise_activations,
                                quantise_weights)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the emulation runs on the GPU")


def _net(seed: int = 0):
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
    from brokefish.nn.quant import Q_MAX, ROW_K

    ext = load_extension("brokefish_encoder", ["encoder.cu"])
    assert float(ext.fp8_q_max()) == Q_MAX
    # ⚠️ The bound it exists to satisfy, and it is over **the whole row**, not one mma.
    # `gemm_fp8_row` runs a single fp16 accumulator for all `ROW_K` products so that it
    # needs no fp32 running total and no per-tile descale -- which is what took the
    # spill from 960 B to 200 B and the kernel from 1.154x to 1.214x. The price is that
    # the accumulator bound is `ROW_K` times bigger than the per-mma one.
    assert ROW_K * Q_MAX * Q_MAX <= 65504.0 / 3.0, "fp16 accumulator margin"
    assert ROW_K % MMA_K == 0


def test_the_python_and_cuda_int8_maxima_agree():
    """The int8 twin of the pin above -- and it exists even though the failure it
    guards against is unreachable here.

    ⚠️ int8 cannot reproduce the e4m3 disaster: the worst s32 partial is
    `127 * 127 * 256 = 4.13e6` against 2.147e9, there is no infinity, no NaN encoding
    and no saturating convert to get wrong. The constants are pinned anyway because
    they are still two transcriptions of one number, and a mismatch would show up as a
    quiet 2x scale error on every FFN output -- finite, plausible, and invisible to
    everything except a prior-space comparison."""
    from brokefish.nn._build import load_extension
    from brokefish.nn.quant import Q_MAX_S8, Q_MAX_U8, ROW_K

    ext = load_extension("brokefish_encoder", ["encoder.cu"])
    s8, u8 = ext.int8_q_max()
    assert float(s8) == Q_MAX_S8
    assert float(u8) == Q_MAX_U8
    assert Q_MAX_S8 * Q_MAX_U8 * ROW_K < 2 ** 31 - 1, "s32 accumulator margin"


def test_int8_beats_e4m3_in_prior_space_on_a_trained_checkpoint():
    """The claim the int8 path exists for, as a test rather than a journal line.

    Not a tolerance: a **comparison**. int8 and e4m3 quantise the same two matmuls at
    the same granularity, so if int8 is ever not the more accurate of the two on a
    trained network, the scheme has changed under us and the reason to prefer it is
    gone. Random init would not bite -- 2026-08-14 measured the gap growing 15-24x over
    a training run, so an untrained net is where the two formats look most alike."""
    import os
    import torch

    ck = "checkpoints/t12h-gumbel-004009.pt"
    if not os.path.exists(ck):
        pytest.skip(f"{ck} is not in this checkout")
    from brokefish.nn.validate import (drop_terminal, load_net, positions_adversarial,
                                       positions_random, validate)

    net16, _ = load_net(ck)
    b, c, r = positions_random(256, plies=60, seed=0)
    ab, ac, ar = positions_adversarial()
    B, C, R = drop_terminal(torch.cat([b, ab]), torch.cat([c, ac]), torch.cat([r, ar]))
    fw = {"fp8": FusedEncoder(copy.deepcopy(net16), fp8=True).forward_full,
          "int8": FusedEncoder(copy.deepcopy(net16), int8=True).forward_full}
    d = validate(None, B, C, R, list(fw), verbose=False, net=net16, forwards=fw)
    assert d["int8"].max_abs < d["fp8"].max_abs
    assert d["int8"].flip_risk <= d["fp8"].flip_risk
    assert d["int8"].nonfinite == 0


def test_fp8_and_int8_are_mutually_exclusive():
    """Both write the same slab through the same packer, so the second would silently
    win. `quant` is an explicit mode in the kernel for the same reason."""
    net = BrokefishNet().cuda().half().eval()
    with pytest.raises(ValueError, match="pick one"):
        FusedEncoder(net, fp8=True, int8=True)


def test_int8_is_bit_deterministic():
    """Two calls on the same input must produce the same bytes.

    Self-play replays and the differential search harness both assume the evaluator is
    a function. A quantiser that computed its scale from anything batch- or
    launch-dependent would break that silently, and the search would only show it as
    an occasional unreproducible tree."""
    net = BrokefishNet().cuda().half().eval()
    enc = FusedEncoder(net, int8=True)
    b, c, r = random_positions(96, plies=20, seed=3)
    with torch.no_grad():
        p1, q1, v1 = (t.clone() for t in enc.forward_full(b, c, r))
        p2, q2, v2 = enc.forward_full(b, c, r)
    assert torch.equal(p1, p2) and torch.equal(q1, q2) and torch.equal(v1, v2)


@pytest.mark.parametrize("n", [1, 3, 31, 33, 127, 257])
def test_int8_is_independent_of_batch_composition(n):
    """One board per CTA means a board's logits cannot depend on who it shares a launch
    with. ⚠️ This is the invariant the **two-boards-per-CTA** rework will put under
    real pressure -- it makes two boards share a shared-memory allocation and a weight
    stream -- so it is pinned now, while it is still trivially true, rather than after
    the change that could break it."""
    net = BrokefishNet().cuda().half().eval()
    enc = FusedEncoder(net, int8=True)
    b, c, r = random_positions(257, plies=20, seed=4)
    with torch.no_grad():
        full = enc.forward_full(b, c, r)[0].clone()
        part = enc.forward_full(b[:n], c[:n], r[:n])[0].clone()
    assert torch.isfinite(part).all()
    assert torch.equal(part, full[:n]), "a board's logits moved when the batch changed"


def test_int8_agrees_with_the_torch_model_within_the_quantised_bar():
    """The §12 check-9 bar, which the training loop applies to any quantised path.

    8 % of the logit maximum: measured at initialisation, fp16 disagrees by 0.19 % and
    e4m3 by 3.28 %. int8 has to clear the same bar or `--int8` would trip the loop's
    own consistency check on its first generation."""
    net = BrokefishNet().cuda().half().eval()
    enc = FusedEncoder(copy.deepcopy(net), int8=True)
    b, c, r = random_positions(128, plies=20, seed=5)
    with torch.no_grad():
        want = net(b, c, r)[0].float()
        got = enc.forward_full(b, c, r)[0].float()
    scale = want.abs().max().item()
    agree = (got - want).abs().max().item()
    assert agree <= 0.08 * scale, f"{agree:.4g} over {0.08 * scale:.4g}"


def test_int8_runs_the_debug_stages():
    """`forward_stage` threads the same `quant` argument. A mode that only worked on
    the full path would leave every intermediate-buffer test running fp16 while
    claiming to test int8."""
    net = BrokefishNet().cuda().half().eval()
    enc = FusedEncoder(net, int8=True)
    b, c, r = random_positions(16, plies=20, seed=6)
    with torch.no_grad():
        for stage in (9, 1, 2, 3, 8):
            out = enc.forward_stage(b, c, r, stage)
            assert torch.isfinite(out).all(), f"stage {stage} is not finite"


def test_int8_inside_a_real_search():
    """A tree, not a forward pass. ⚠️ A prior that came out zero is unreachable at
    every simulation budget (`search.md` §6.6), and a NaN in one board's logits
    poisons a whole batch's selection -- neither shows up as an exception."""
    from brokefish.search.torch_impl import Search, SearchConfig
    from tests.boards import random_positions as rp

    net = BrokefishNet().cuda().half().eval()
    enc = FusedEncoder(net, int8=True)
    boards, control, _ = rp(32, plies=20, seed=7, device="cuda")
    cfg = SearchConfig(n=64, B=32, eps=0.0)
    s = Search(cfg, enc.forward_full, device="cuda", check_invariants=True)
    s.reset(boards.clone(), control.clone())
    s.root_init(noise=False)
    for i in range(cfg.n):
        s.simulate(i)
    pri = s.edge_prior[:, 0].float()
    valid = torch.arange(pri.shape[1], device=pri.device)[None, :] < s.node_nedges[:, 0:1]
    assert torch.isfinite(pri).all()
    assert (pri[valid] > 0).all(), "a legal edge got a zero prior"
    assert int(s.edge_N[:, 0].sum(-1).min()) > 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 31, 32, 33, 127, 128, 257])
def test_two_boards_per_cta_is_bit_identical_to_one(n):
    """Two boards per CTA is an **occupancy** change, so the logits must not move.

    ⚠️ This is the strongest statement available about the rework and it is stronger
    than a tolerance: a board's arithmetic cannot depend on who it shares a CTA with,
    so anything other than bit-equality means the two boards are interfering — through
    the shared-memory split, the per-board attention slice, or the odd-batch tail
    guard. The odd sizes are the guard: at n = 3 the last CTA holds one real board and
    one duplicate, whose outputs must be computed and never written."""
    net = BrokefishNet().cuda().half().eval()
    # `"ffn"` by name: the default scheme, `"all"`, is instantiated at two boards per
    # CTA only, so it cannot be the one-board side of this comparison. The claim is
    # about the CTA split, not the scheme, and `"ffn"` exists on both sides.
    one = FusedEncoder(copy.deepcopy(net), int8=True, two_boards=False, int8_scheme="ffn")
    two = FusedEncoder(copy.deepcopy(net), int8=True, int8_scheme="ffn")
    b, c, r = random_positions(n, plies=20, seed=n)
    with torch.no_grad():
        p1, q1, v1 = (t.clone() for t in one.forward_full(b, c, r))
        p2, q2, v2 = two.forward_full(b, c, r)
    assert torch.isfinite(p2).all() and torch.isfinite(v2).all()
    assert torch.equal(p1, p2), "policy logits moved"
    assert torch.equal(q1, q2), "promo logits moved"
    assert torch.equal(v1, v2), "value moved"


def test_two_boards_rejects_fp8():
    """Not instantiated: int8 dominates e4m3 on both axes, so a two-board e4m3 kernel
    is a configuration nobody would run, paid for in compile time on every build."""
    net = BrokefishNet().cuda().half().eval()
    with pytest.raises(ValueError, match="int8 dominates"):
        FusedEncoder(net, fp8=True, two_boards=True)


def test_two_boards_runs_the_debug_stages():
    net = BrokefishNet().cuda().half().eval()
    one = FusedEncoder(copy.deepcopy(net), int8=True, two_boards=False, int8_scheme="ffn")
    two = FusedEncoder(copy.deepcopy(net), int8=True, int8_scheme="ffn")
    b, c, r = random_positions(33, plies=20, seed=11)
    with torch.no_grad():
        for stage in (9, 1, 2, 3, 8):
            a = one.forward_stage(b, c, r, stage).clone()
            z = two.forward_stage(b, c, r, stage)
            assert torch.isfinite(z).all(), f"stage {stage} not finite"
            assert torch.equal(a, z), f"stage {stage} moved"


def test_the_shipping_weight_scale_covers_the_whole_reduction():
    """⚠️ `_pack_fp8` must pass `tile_k = K`, and getting it wrong is silent.

    `gemm_fp8_row` descales once, reading `w_scale[n8 / 16]` with no k index at all. A
    packer that still emitted a scale per 128 k-elements would hand it an array whose
    first `N / 128` entries are the k-tile-0 scales, so every output would be scaled by
    the wrong block's constant -- in range, finite, and wrong by a factor that looks
    like quantisation error.
    """
    from brokefish.nn.quant import Q_MAX, ROW_K, quantise_weights_bytes

    w = torch.randn(1024, ROW_K, device="cuda")
    _, sc = quantise_weights_bytes(w, Q_MAX, tile_k=w.shape[1])
    assert sc.shape == (1024 // 128, 1), sc.shape
    # And the per-tile form it replaced, which must now be rejected by shape at the
    # call site rather than accepted and half-read.
    _, wrong = quantise_weights_bytes(w, Q_MAX)
    assert wrong.shape == (1024 // 128, ROW_K // 128) != sc.shape
