"""`brokefish/train/muon.py`, against `torch.optim.Muon` as the oracle.

The file is deliberately split in two. Everything on the unchunked path has an oracle
in torch 2.13 and is asserted **bit-exact** against it -- if those pass, the momentum,
the Nesterov mix, the Newton-Schulz iteration, the epsilon placement, the
learning-rate adjustment, the decay ordering and the bf16 rounding are all correct and
none of them needs to be reasoned about again. What is left is the chunking, which
torch does not do, and that is what the rest of the file is about.

⚠️ Two of these checks exist because the failure they catch is silent. `test_out_proj
_chunks_columns_not_rows` catches an optimiser that runs and converges and is a
different algorithm; `test_rms_is_invariant_to_chunking` catches a `head_group` sweep
that is secretly also a learning-rate sweep.
"""

from __future__ import annotations

import math

import pytest
import torch

from brokefish.train.muon import (Muon, adjust_ratio, chunk_spec, is_muon_param,
                                  muon_param_groups, newton_schulz, NS_COEFFS, NS_STEPS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _net():
    from brokefish.nn.model import BrokefishNet
    torch.manual_seed(0)
    return BrokefishNet().to(DEVICE)


def _pair(shape, seed=0):
    """The same parameter twice, so two optimisers can be driven from one gradient."""
    torch.manual_seed(seed)
    p = torch.randn(*shape, device=DEVICE)
    a, b = p.clone().requires_grad_(), p.clone().requires_grad_()
    return a, b


# -- the oracle: unchunked, we must BE torch --------------------------------- #

SHAPES = [(768, 256), (256, 256), (1024, 256), (256, 1024), (64, 256), (256, 32)]


@pytest.mark.parametrize("shape", SHAPES)
def test_unchunked_muon_is_bit_exact_against_torch(shape):
    """Ten steps, fresh gradients each step, no tolerance at all.

    Ten and not one: a single step cannot tell a wrong momentum coefficient from a
    right one, because the buffer starts at zero and the first update is dominated by
    the gradient whichever convention is used.
    """
    ours, theirs = _pair(shape)
    lr, wd, mom = 0.02, 0.01, 0.95
    mine = Muon([{"params": [ours], "use_muon": True, "chunks": [(0, 1)], "lr": lr,
                  "weight_decay": wd, "momentum": mom, "nesterov": True}])
    ref = torch.optim.Muon([theirs], lr=lr, weight_decay=wd, momentum=mom,
                           nesterov=True)
    torch.manual_seed(7)
    for _ in range(10):
        g = torch.randn(*shape, device=DEVICE)
        ours.grad, theirs.grad = g.clone(), g.clone()
        mine.step()
        ref.step()
    assert torch.equal(ours, theirs), (ours - theirs).abs().max().item()


@pytest.mark.parametrize("shape", [(64, 256), (256,), (101, 256)])
def test_aux_adamw_is_bit_exact_against_torch(shape):
    """The 91,904-parameter residue must be the *same* optimiser as the control's.

    ⚠️ `foreach=False`, and the difference is worth knowing rather than hiding.
    Against torch's default multi-tensor path this is bit-exact **except for one fp32
    ulp** (1.19e-7), which is its fusion order and not an algorithm difference -- the
    single-tensor path matches exactly, and the single-tensor path is what torch's
    documented pseudocode describes. The control `t7h-fp8` ran the `foreach` path, so
    the two arms' auxiliary groups differ by that ulp on 1.4 % of the parameters. That
    is not a confound; it is recorded so nobody rediscovers it as one.
    """
    ours, theirs = _pair(shape)
    lr, wd, betas, eps = 1e-3, 0.01, (0.9, 0.95), 1e-8
    mine = Muon([{"params": [ours], "use_muon": False, "lr": lr, "weight_decay": wd,
                  "betas": betas, "eps": eps}])
    ref = torch.optim.AdamW([theirs], lr=lr, weight_decay=wd, betas=betas, eps=eps,
                            foreach=False)
    torch.manual_seed(7)
    for _ in range(10):
        g = torch.randn(*shape, device=DEVICE)
        ours.grad, theirs.grad = g.clone(), g.clone()
        mine.step()
        ref.step()
    assert torch.equal(ours, theirs), (ours - theirs).abs().max().item()


@pytest.mark.parametrize("shape", SHAPES + [(4, 256), (1, 256), (256, 1)])
def test_adjust_ratio_matches_torch_unchunked(shape):
    """Our generalisation has to *be* torch's formula when there is one chunk."""
    from torch.optim._muon import _adjust_lr
    theirs = _adjust_lr(1.0, None, torch.Size(shape))
    assert adjust_ratio(shape[0], shape[1], 0, 1) == pytest.approx(theirs, rel=1e-12)
    assert adjust_ratio(shape[0], shape[1], 1, 1) == pytest.approx(theirs, rel=1e-12)


def _conditioned(cond: float, n: int = 256):
    """A matrix with a prescribed condition number and random singular vectors."""
    u, _ = torch.linalg.qr(torch.randn(n, n, device=DEVICE))
    v, _ = torch.linalg.qr(torch.randn(n, n, device=DEVICE))
    s = torch.logspace(-math.log10(cond), 0, n, device=DEVICE)
    return u @ torch.diag(s) @ v.T


@pytest.mark.parametrize("cond", [10.0, 100.0])
def test_newton_schulz_flattens_the_spectrum_into_torchs_stated_band(cond):
    """torch's docstring promises `S'_ii ~ Uniform(0.5, 1.5)`, not ones.

    ⚠️ **That promise holds for well-conditioned inputs and not in general**, which is
    worth knowing before reading anything into a Muon update. Measured here: a
    condition number of 10 or 100 collapses to under 2 and lands inside the band, but
    at 1000 the smallest singular value only reaches 0.095 in five steps, and a square
    Gaussian -- whose Marchenko-Pastur spectrum has mass at zero -- reaches 0.082. The
    iteration is tuned for slope at zero, not for reach.
    """
    torch.manual_seed(0)
    sv = torch.linalg.svdvals(newton_schulz(_conditioned(cond)).float())
    assert sv.min() > 0.5 and sv.max() < 1.5, (sv.min().item(), sv.max().item())
    assert sv.max() / sv.min() < 2.0


def test_newton_schulz_does_not_reach_the_bottom_of_a_hard_spectrum():
    """The negative half of the above, asserted so it stays a known fact, not a surprise."""
    torch.manual_seed(0)
    sv = torch.linalg.svdvals(newton_schulz(_conditioned(1000.0)).float())
    assert sv.min() < 0.2 and sv.max() > 1.0


# -- the chunking, which has no oracle --------------------------------------- #

def test_chunk_spec_ladder():
    assert chunk_spec("...in_proj_weight", 8, True, 8) == (0, 3)
    assert chunk_spec("...in_proj_weight", 8, True, 1) == (0, 24)
    assert chunk_spec("...in_proj_weight", 8, True, 2) == (0, 12)
    assert chunk_spec("...in_proj_weight", 8, False, 8) == (0, 1)
    assert chunk_spec("...self_attn.out_proj.weight", 8, True, 8) == (1, 1)
    assert chunk_spec("...self_attn.out_proj.weight", 8, True, 1) == (1, 8)
    assert chunk_spec("...linear1.weight", 8, True, 1) == (0, 1)
    with pytest.raises(ValueError):
        chunk_spec("...in_proj_weight", 8, False, 1)   # straddles Q/K/V
    with pytest.raises(ValueError):
        chunk_spec("...in_proj_weight", 8, True, 3)    # 3 does not divide 8


def test_qkv_chunking_is_exactly_three_independent_orthogonalisations():
    """Not "close to" -- the concatenation must equal the three separate results."""
    torch.manual_seed(0)
    g = torch.randn(768, 256, device=DEVICE)
    opt = Muon([{"params": [torch.zeros(1, device=DEVICE)], "use_muon": False}])
    got = opt._orthogonalise(g, dim=0, n_chunks=3)
    want = torch.cat([newton_schulz(g[i * 256:(i + 1) * 256]) for i in range(3)], dim=0)
    assert torch.equal(got, want)


def test_chunking_actually_changes_the_update():
    """If it did not, every claim in the survey would be about nothing."""
    torch.manual_seed(0)
    g = torch.randn(768, 256, device=DEVICE)
    opt = Muon([{"params": [torch.zeros(1, device=DEVICE)], "use_muon": False}])
    fused = opt._orthogonalise(g, 0, 1).float()
    split = opt._orthogonalise(g, 0, 3).float()
    # Compare directions, since `adjust_ratio` handles the sizes separately.
    cos = torch.nn.functional.cosine_similarity(
        fused.flatten() * math.sqrt(3), split.flatten(), dim=0)
    assert cos < 0.95, f"chunked and fused updates are nearly identical: cos = {cos}"


def test_out_proj_chunks_columns_not_rows():
    """⚠️ The silent-wrongness test.

    `out_proj` carries its head structure on the **input** dimension. Build a gradient
    where head `h`'s columns are scaled by `10^h`, so the fused orthogonalisation is
    dominated by head 7 and the small heads get under-normalised -- which is exactly
    Kimi K3's stated motivation. Chunking on `dim=1` must fix it; chunking on `dim=0`
    must not, because rows are not heads.
    """
    torch.manual_seed(0)
    g = torch.randn(256, 256, device=DEVICE)
    scales = torch.repeat_interleave(torch.logspace(0, 7, 8, device=DEVICE), 32)
    g = g * scales                                    # column h-block scaled by 10^h

    def head_energy(update):
        blocks = update.float().chunk(8, dim=1)
        return torch.tensor([b.norm() for b in blocks])

    opt = Muon([{"params": [torch.zeros(1, device=DEVICE)], "use_muon": False}])
    fused = head_energy(opt._orthogonalise(g, 1, 1))
    right = head_energy(opt._orthogonalise(g, 1, 8))   # correct axis
    wrong = head_energy(opt._orthogonalise(g, 0, 8))   # rows, the bug

    # Correct axis: every head gets the same update energy, whatever its gradient scale.
    assert right.max() / right.min() < 1.05, right
    # Fused: the loud head dominates and the quiet ones are starved.
    assert fused.max() / fused.min() > 3.0, fused
    # Wrong axis: row-chunking does nothing for head imbalance, so it looks like fused.
    assert wrong.max() / wrong.min() > 3.0, wrong


CHUNKINGS = [(0, 1, (768, 256)), (0, 3, (768, 256)), (0, 24, (768, 256)),
             (1, 1, (256, 256)), (1, 8, (256, 256)),
             (0, 1, (1024, 256)), (0, 1, (256, 1024))]


@pytest.mark.parametrize("dim,n_chunks,shape", CHUNKINGS)
def test_adjust_ratio_puts_the_update_rms_at_one_over_sqrt_fanin(dim, n_chunks, shape):
    """⚠️ The learning-rate-transfer test, and the reason `adjust_ratio` exists.

    The whole purpose of the `sqrt(max(1, d_out/d_in))` factor -- torch's own docstring
    says so -- is a *consistent update RMS across shapes*. Extended to chunkings, the
    RMS must come out at `1 / sqrt(fan_in)` every time, so that sweeping `head_group`
    sweeps geometry and not step size.

    Newton-Schulz is deliberately absent here and replaced by an exact semi-orthogonal
    factor from QR. It has to be: the iteration's *convergence* depends on the block's
    aspect ratio (see the test below), so running it would confound a test of the
    arithmetic with a property of the iteration. This asserts the formula; the next
    test measures the iteration's residue.
    """
    torch.manual_seed(0)
    r, c = (shape[0] // n_chunks, shape[1]) if dim == 0 else (shape[0], shape[1] // n_chunks)
    blocks = []
    for _ in range(n_chunks):
        q, _ = torch.linalg.qr(torch.randn(max(r, c), min(r, c), device=DEVICE))
        blocks.append(q.T if r < c else q)             # exactly semi-orthogonal
    u = torch.cat(blocks, dim=dim) * adjust_ratio(shape[0], shape[1], dim, n_chunks)
    assert u.square().mean().sqrt().item() == pytest.approx(
        1.0 / math.sqrt(shape[1]), rel=1e-4)


@pytest.mark.parametrize("dim,n_chunks,shape", CHUNKINGS)
def test_newton_schulz_leaves_a_known_aspect_ratio_dependent_shortfall(dim, n_chunks, shape):
    """⚠️ Measured, and recorded because it is a real ~10 % effect that is NOT a bug here.

    With the real iteration in the loop the achieved RMS is `0.85 - 0.96` of target,
    and where it lands depends on the *block's* aspect ratio, because a square Gaussian
    block has Marchenko-Pastur mass near zero that five steps do not lift while a very
    rectangular one has a concentrated spectrum that they do.

    Measured 2026-08-07, `x sqrt(fan_in)`: (768,256) unchunked 0.956, split 3-ways
    0.904, per-head 24-ways 0.864; (256,256) unchunked 0.904, per-head 8-ways 0.861;
    (1024,256) 0.957; (256,1024) 0.956; (64,256) 0.852.

    So **the QKV split costs about 5 % of effective step size** on `in_proj` and
    per-head another 4 %. That is inherent to Newton-Schulz and is present in every
    implementation that splits QKV, torch's and Jordan's included -- it is not
    something `adjust_ratio` can or should correct, since correcting it would mean
    rescaling by a quantity that depends on the gradient's spectrum. It is recorded
    so that a `head_group` sweep is read with it in mind.
    """
    torch.manual_seed(0)
    g = torch.randn(*shape, device=DEVICE)
    opt = Muon([{"params": [torch.zeros(1, device=DEVICE)], "use_muon": False}])
    u = opt._orthogonalise(g, dim, n_chunks).float()
    u = u * adjust_ratio(shape[0], shape[1], dim, n_chunks)
    got = u.square().mean().sqrt().item() * math.sqrt(shape[1])
    assert 0.84 < got < 0.97, got


# -- the wiring -------------------------------------------------------------- #

def test_grouping_is_the_98_6_percent_the_survey_measured():
    net = _net()
    groups = muon_param_groups(net, lr=0.02, aux_lr=1e-3, wd=0.01)
    counts = [sum(p.numel() for p in g["params"]) for g in groups]
    assert counts[0] == 6_291_456           # 8 layers x (768+256+1024+256) x 256
    assert counts[1] + counts[2] == 91_904
    assert sum(counts) == sum(p.numel() for p in net.parameters()) == 6_383_360
    # Every Muon tensor is 2-D and none of them is an embedding or a head.
    for g_ in groups[0]["params"]:
        assert g_.ndim == 2
    assert not any(is_muon_param(n, p) for n, p in net.named_parameters()
                   if "emb_" in n or n.startswith(("policy", "promo", "value")))


def test_chunk_counts_across_the_ladder():
    net = _net()
    for g, want_in, want_out in [(8, 3, 1), (4, 6, 2), (2, 12, 4), (1, 24, 8)]:
        groups = muon_param_groups(net, lr=0.02, aux_lr=1e-3, wd=0.01, head_group=g)
        specs = groups[0]["chunks"]
        # 4 tensors per layer, in the order named_parameters yields them.
        assert specs[0] == (0, want_in) and specs[1] == (1, want_out)
        assert specs[2] == (0, 1) and specs[3] == (0, 1)      # linear1, linear2


def test_state_dict_round_trips_including_the_chunk_metadata():
    net = _net()
    opt = Muon(muon_param_groups(net, lr=0.02, aux_lr=1e-3, wd=0.01))
    for p in net.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    saved = opt.state_dict()

    net2 = _net()
    opt2 = Muon(muon_param_groups(net2, lr=0.02, aux_lr=1e-3, wd=0.01))
    opt2.load_state_dict(saved)
    assert opt2.param_groups[0]["chunks"] == opt.param_groups[0]["chunks"]
    a = opt.state[opt.param_groups[0]["params"][0]]["momentum_buffer"]
    b = opt2.state[opt2.param_groups[0]["params"][0]]["momentum_buffer"]
    assert torch.equal(a, b)


def test_normuon_preserves_the_frobenius_norm_it_redistributes():
    torch.manual_seed(0)
    opt = Muon([{"params": [torch.zeros(1, device=DEVICE)], "use_muon": False}],
               normuon=True)
    o = torch.randn(256, 256, device=DEVICE) * torch.logspace(
        0, 3, 256, device=DEVICE).unsqueeze(1)
    state = {}
    out = opt._normuon(state, o)
    assert out.norm().item() == pytest.approx(o.norm().item(), rel=1e-3)
    # And it did something: the row-norm spread must shrink.
    before = o.norm(dim=1)
    after = out.norm(dim=1)
    assert (after.max() / after.min()) < (before.max() / before.min())


def test_a_few_steps_actually_reduce_a_loss():
    """The cheapest end-to-end sanity check: it has to descend."""
    torch.manual_seed(0)
    net = _net()
    groups = muon_param_groups(net, lr=0.02, aux_lr=1e-3, wd=0.0)
    opt = Muon(groups)
    x = torch.randn(64, 32, 256, device=DEVICE)
    target = torch.randn(64, 32, 256, device=DEVICE)
    losses = []
    for _ in range(20):
        out = net.encoder(x)
        loss = (out - target).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9, losses[::5]
