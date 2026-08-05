"""`brokefish/nn/validate.py` itself: does the QA suite detect a broken encoder?

⚠️ **This file exists because a validator nobody has watched fail is an assertion,
not a measurement.** `tests/test_mutation_mask.py` makes the same argument for the
encoder tests. Every check here either corrupts an encoder in a known way and
demands that the suite say so, or feeds it a clean one and demands silence.

The first thing this suite found was in its own harness: `cuda.forward_full` returns
views into buffers it reuses across calls, so a chunked caller that concatenates gets
the last chunk repeated. That read as a 950x kernel error before it was understood.
`test_aliasing_is_detected` is that lesson, pinned.
"""

from __future__ import annotations

import pytest
import torch

from brokefish.nn import available, why_unavailable
from brokefish.nn.validate import (Delta, aliases_across_calls,
                                   positions_adversarial, positions_random, report,
                                   run_priors, validate)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the fused encoders need a GPU")

IMPLS = available()
N_POS = 96


def _positions(adversarial: bool = True):
    b, c, r = positions_random(N_POS, plies=30, seed=3)
    if not adversarial:
        return b, c, r
    ab, ac, ar = positions_adversarial()
    return torch.cat([b, ab]), torch.cat([c, ac]), torch.cat([r, ar])


def _net(seed: int = 0):
    """⚠️ Seeded. `BrokefishNet()` draws from the global generator, and an unseeded
    net made `test_a_clean_encoder_passes` depend on whether that draw happened to
    tie the 218-move position's priors closely enough for §4.3 to truncate it
    differently in fp16 than in fp32 -- a one-in-a-few-runs failure with no visible
    cause."""
    from brokefish.nn.model import BrokefishNet

    torch.manual_seed(seed)
    return BrokefishNet().cuda().half().eval()


def _delta(**kw) -> Delta:
    base = dict(name="x", max_abs=0.0, p95_abs=0.0, max_rel=0.0, max_kl=0.0,
                mean_kl=0.0, top1_disagree=0.0, top3_miss=0.0, flip_risk=0.0,
                flushed=0, edge_set_differs=0, max_abs_logit=1.0, nonfinite=0,
                max_abs_value=0.0)
    base.update(kw)
    return Delta(**base)


# -- the harness's own trap -------------------------------------------------- #

def test_aliasing_is_detected():
    """A forward that reuses its output buffer must be reported, not silently used."""
    buf = torch.zeros(64, 32, 64, device="cuda", dtype=torch.half)

    def reusing(b, c, r):
        n = b.shape[0]
        buf[:n] = torch.randn(n, 32, 64, device="cuda", dtype=torch.half)
        return buf[:n], buf[:n, :, :4], torch.zeros(n, device="cuda")

    def fresh(b, c, r):
        n = b.shape[0]
        return (torch.zeros(n, 32, 64, device="cuda", dtype=torch.half),
                torch.zeros(n, 32, 4, device="cuda", dtype=torch.half),
                torch.zeros(n, device="cuda"))

    b, c, r = _positions()
    assert aliases_across_calls(reusing, b, c, r) is True
    assert aliases_across_calls(fresh, b, c, r) is False


@pytest.mark.skipif("cuda" not in IMPLS, reason=f"no CUDA encoder: {why_unavailable()}")
def test_the_cuda_encoder_still_reuses_its_buffers():
    """The contract this suite discovered, pinned so a change to it is deliberate.

    Not a defect: the search hands the output straight to `expand` in the same step,
    and re-allocating on a path that runs 800 times a move would cost. It is pinned
    because `triton` does *not* do it, so code written against one silently breaks on
    the other, and because nothing else in the repository states it.
    """
    from brokefish.nn import encoder_impl
    from brokefish.nn.model import BrokefishNet

    net = _net()
    b, c, r = _positions()
    with torch.no_grad():
        assert aliases_across_calls(encoder_impl("cuda")(net).forward_full, b, c, r)
        assert not aliases_across_calls(encoder_impl("triton")(net).forward_full, b, c, r)


# -- does it bite? ----------------------------------------------------------- #

@pytest.mark.skipif(not IMPLS, reason="no encoder available")
def test_a_clean_encoder_passes():
    b, c, r = _positions(adversarial=False)
    deltas = validate(None, b, c, r, IMPLS, verbose=False, net=_net())
    assert not report(deltas, IMPLS, max_abs_dp=0.01)
    for name in IMPLS:
        assert deltas[name].nonfinite == 0
        assert deltas[name].edge_set_differs == 0


@pytest.mark.skipif(not IMPLS, reason="no encoder available")
@pytest.mark.parametrize("kind", ["scale", "one-legal-move"])
@pytest.mark.parametrize("size", [0.01, 0.1])
def test_a_perturbed_encoder_is_caught(kind, size):
    """A wrong encoder must be flagged at a gate a right one passes.

    The gate is calibrated from the clean encoder in the same run rather than
    written down, so this asks the only question that matters operationally --
    *can the suite separate them* -- instead of asserting a magic number that drifts
    with the hardware.

    Two shapes, because they fail differently. ``scale`` multiplies every logit,
    which is what a wrong normalisation or attention scale looks like and which no
    softmax can cancel. ``one-legal-move`` nudges a checkerboard of the *legal*
    entries only.

    ⚠️ The perturbation has to reach a **legal** move. The first version of this test
    bumped `policy[:, 0, 0]` -- slot 0 to a1 -- and measured nothing at any size,
    because slot 0 is a white pawn and that move is never legal, so §6.4's mask drops
    the logit before the softmax ever sees it. A test that perturbs an unreachable
    input is a test that always passes.

    At ``size = 0.01`` the induced logit shift is about half the old logit-space bar
    (`REL_TOL = 5e-3` on `max|logit| ~ 3.7` is 0.019), so this is squarely inside the
    band the previous suite would have waved through.
    """
    from brokefish.env import torch_impl as _env
    from brokefish.nn import encoder_impl
    from brokefish.nn.model import BrokefishNet

    net = _net()
    impl = IMPLS[0]
    inner = encoder_impl(impl)(net).forward_full
    # ⚠️ Without the 218-move position. §4.3's truncation genuinely ties there on an
    # untrained policy and the two precisions keep different edge sets, which is a
    # real disagreement and a hard failure -- but it is not the numerical question
    # this test is calibrating. `test_truncation_can_tie_on_a_near_uniform_policy`
    # owns that one.
    b, c, r = _positions(adversarial=False)

    def broken(bo, co, re):
        p, q, v = inner(bo, co, re)
        p = p.clone()
        if kind == "scale":
            p = p * (1.0 + size)
        else:
            mask, _ = _env.movegen(bo, co)
            legal = _env.bitset_to_bool(mask)
            checker = ((torch.arange(32, device=p.device)[None, :, None]
                        + torch.arange(64, device=p.device)[None, None, :]) % 2 == 0)
            p = p + size * (legal & checker).to(p.dtype)
        return p, q.clone(), v.clone()

    clean = validate(None, b, c, r, [impl], verbose=False, net=net)[impl]
    dirty = validate(None, b, c, r, ["broken"], verbose=False,
                     forwards={"broken": broken}, net=net)["broken"]

    gate = 4.0 * clean.max_abs
    assert not report({impl: clean, "torch16": clean}, [impl], max_abs_dp=gate), \
        "the clean encoder does not pass its own gate; the calibration is wrong"
    assert dirty.max_abs > gate, (
        f"{kind} at {size} moved the prior by {dirty.max_abs:.2e}, inside the clean "
        f"encoder's own {clean.max_abs:.2e}; the suite cannot see this error")
    assert report({"broken": dirty, "torch16": clean}, ["broken"], max_abs_dp=gate), \
        "the delta was large enough but `report` did not fail"


@pytest.mark.skipif(not IMPLS, reason="no encoder available")
def test_a_nonfinite_output_is_caught():
    from brokefish.nn import encoder_impl

    net = _net()
    inner = encoder_impl(IMPLS[0])(net).forward_full

    def broken(bo, co, re):
        p, q, v = inner(bo, co, re)
        p = p.clone()
        p[0, 0, 0] = float("inf")
        return p, q.clone(), v.clone()

    b, c, r = _positions()
    deltas = validate(None, b, c, r, ["inf"], verbose=False, forwards={"inf": broken},
                      net=net)
    assert deltas["inf"].nonfinite > 0
    assert any("non-finite" in m for m in report(deltas, ["inf"], max_abs_dp=None))


# -- the gates --------------------------------------------------------------- #

def test_report_fails_on_each_gate():
    ok = _delta(name="torch16", max_abs=1e-3, p95_abs=1e-4)
    assert not report({"torch16": ok, "k": _delta(name="k", max_abs=1e-3, p95_abs=1e-4)},
                      ["k"], max_abs_dp=0.01)
    for kw, needle in ((dict(nonfinite=3), "non-finite"),
                       (dict(edge_set_differs=2), "truncation"),
                       (dict(max_abs=1.0, p95_abs=1e-4), "over"),
                       (dict(p95_abs=1.0), "torch fp16")):
        bad = report({"torch16": ok, "k": _delta(name="k", **kw)}, ["k"], max_abs_dp=0.01)
        assert any(needle in m for m in bad), f"{kw} was not caught: {bad}"


# -- the prior really is the search's -------------------------------------- #

@pytest.mark.skipif(not IMPLS, reason="no encoder available")
def test_the_prior_is_a_distribution_over_the_legal_edges():
    """What `run_priors` returns has to be what PUCT reads, or nothing above means
    anything: normalised over the node's own edges and exactly zero outside them."""
    net = _net()
    b, c, r = _positions()
    with torch.no_grad():
        p, q, v = net(b, c, r)
    prior, move, nedges = run_priors(b, c, p, q, v, e_cap=96)
    idx = torch.arange(96, device=prior.device)
    valid = idx[None, :] < nedges[:, None]
    assert bool((nedges > 0).all()), "a non-terminal position with no edges"
    assert float(prior[~valid].abs().max()) == 0.0, "mass outside the edge set"
    total = prior.sum(-1)
    # fp16 storage, so the sum is exact only to the resolution the array carries.
    assert float((total - 1.0).abs().max()) < 5e-3, float((total - 1.0).abs().max())


@pytest.mark.skipif(not IMPLS, reason="no encoder available")
def test_truncation_can_tie_on_a_near_uniform_policy():
    """§4.3 keeps the `E` highest priors, and on an untrained policy over 218 legal
    moves the 96th and 97th are close enough that fp16 and fp32 order them
    differently. The suite must **say so** -- a different edge set is a different
    move list, which is a worse failure than a shifted probability and is why
    `report` treats it as a hard stop rather than folding it into a delta.

    On the trained checkpoint this was measured on it is zero for every variant;
    this is the untrained regime, and it is pinned so that a future change which
    makes truncation *silently* precision-dependent has to argue with a test.
    """
    boards, control, rep = positions_adversarial()
    deltas = validate(None, boards, control, rep, IMPLS, verbose=False, net=_net())
    assert any(deltas[i].edge_set_differs for i in IMPLS), \
        "the 218-move position no longer ties; pick a wider one or drop this test"
    assert any("truncation" in m for m in report(deltas, IMPLS, max_abs_dp=None))
