"""Which search runs an evaluation, and the config that evaluation uses.

Two lines of policy that three modules would otherwise each get slightly wrong.

**The search implementation is a choice, not a constant.** `search/torch_impl` is
the reference and it is what the tests compare against; `search/cuda_impl` is the
same tree three orders of magnitude faster and is what layer 0 has to use to cost
seconds rather than an hour. They are interchangeable by construction — C1's whole
point — so the evaluation harness takes the name and nothing else changes.

⚠️ `cuda_impl` requires an evaluator that returns **fp16** logits, so
``search_impl="cuda"`` needs a fused ``impl`` too. `nn/model.py` keeps fp32 master
weights and returns fp32, and the CUDA search raises rather than casting, because
casting in the driver would silently change the numbers the reference computes.

**Evaluation is greedy and noiseless** (`evals.md` §3): `eps=0` removes the
Dirichlet root noise and `tau_plies=0` makes `select_and_advance` take the argmax
of the root visit counts instead of sampling from them. Leaving the self-play
defaults on turns every suite score into a sample from the policy rather than a
measurement of it, and it is exactly the kind of mistake that shows up as a
mysteriously noisy metric six weeks later.
"""

from __future__ import annotations

from typing import Callable, Optional


def search_class(search_impl: str = "torch"):
    if search_impl == "torch":
        from brokefish.search.torch_impl import Search
        return Search
    if search_impl == "cuda":
        from brokefish.search.cuda_impl import Search
        return Search
    raise ValueError(f"search_impl must be 'torch' or 'cuda', got {search_impl!r}")


def eval_config(n: int, B: int, **kw):
    """A `SearchConfig` in the evaluation protocol of `evals.md` §3.

    ⚠️ ``gumbel_scale = 0`` belongs with ``eps = 0`` and ``tau_plies = 0``, and for
    the same reason. Gumbel replaces both of those: at scale 1 every move is a
    sample forever, where v0 goes argmax past ply 30. `match.py`'s header states
    the protocol this breaks — evaluation takes all of its diversity from the
    random openings, and a pairing of `G` games needs `G/2` distinct ones because
    two engines replaying an opening produce the same game every time. Left at 1,
    a league would rate the noise, and a Gumbel run measured against a PUCT control
    would be measured on a different protocol from it.
    """
    from brokefish.search.torch_impl import SearchConfig
    kw.setdefault("gumbel_scale", 0.0)
    return SearchConfig(n=n, B=B, eps=0.0, tau_plies=0, **kw)


def make_search(n: int, B: int, net, impl: Optional[str] = None,
                search_impl: str = "cuda", seed: int = 0, device: str = "cuda",
                greedy: bool = True, **kw):
    """The whole "run a search over these positions" preamble, in one call.

    ⚠️ **``impl`` is the encoder, ``search_impl`` is the tree.** Two axes, easy to
    confuse, and the confusion had teeth: this defaulted to ``"torch"`` until
    2026-07-31, so every evaluation in the repository silently drove the *reference*
    search. The reference exists to be the oracle the kernel is checked against
    (`tests/test_search_cuda.py` holds them together tree for tree), not to be what a
    measurement runs on — and it is roughly an order of magnitude slower, which put
    `state.md`'s 44.4 s layer-0 cost against a prediction that assumed the kernel.
    Pass ``"torch"`` deliberately when the oracle *is* the point.

    ⚠️ **The two axes are coupled in one direction**: the kernel's ``_expand`` reads
    fp16 logits and *refuses* to cast an fp32 policy, because casting would change
    the numbers the reference computes in fp32 and silently break the tree-for-tree
    comparison. ``impl=None`` is the plain torch module, which emits fp32. So a CUDA
    search with ``impl=None`` raises — which is what the old ``"torch"`` default was
    quietly protecting, and what flipping it exposed. The pairing is resolved here
    rather than left to every caller: a CUDA search with no encoder named gets the
    fused CUDA encoder. Name ``impl`` explicitly to override.
    """
    if search_impl == "cuda" and impl is None:
        impl = "cuda"
    from brokefish.search.torch_impl import SearchConfig, make_evaluator

    cfg = (eval_config(n, B, **kw) if greedy else SearchConfig(n=n, B=B, **kw))
    return search_class(search_impl)(cfg, make_evaluator(net, impl),
                                     device=device, seed=seed)


def evaluator(net, impl: Optional[str] = None) -> Callable:
    from brokefish.search.torch_impl import make_evaluator
    return make_evaluator(net, impl)
