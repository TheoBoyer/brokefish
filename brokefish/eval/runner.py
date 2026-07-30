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
    """A `SearchConfig` in the evaluation protocol of `evals.md` §3."""
    from brokefish.search.torch_impl import SearchConfig
    return SearchConfig(n=n, B=B, eps=0.0, tau_plies=0, **kw)


def make_search(n: int, B: int, net, impl: Optional[str] = None,
                search_impl: str = "torch", seed: int = 0, device: str = "cuda",
                greedy: bool = True, **kw):
    """The whole "run a search over these positions" preamble, in one call."""
    from brokefish.search.torch_impl import SearchConfig, make_evaluator

    cfg = (eval_config(n, B, **kw) if greedy else SearchConfig(n=n, B=B, **kw))
    return search_class(search_impl)(cfg, make_evaluator(net, impl),
                                     device=device, seed=seed)


def evaluator(net, impl: Optional[str] = None) -> Callable:
    from brokefish.search.torch_impl import make_evaluator
    return make_evaluator(net, impl)
