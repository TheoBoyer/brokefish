"""Gumbel MuZero: root sampling, sequential halving, and the completed-Q target.

`search.md` §11's `select_root`, `select_interior` and `policy_target` seams at
once — Danihelka, Guez, Schrittwieser & Silver, *Policy improvement by planning
with Gumbel*, ICLR 2022 (`openreview.net/forum?id=bERaNdoegnO`).

Everything here is pure arithmetic on `[B, E]` tensors and holds no tree state,
so each piece is testable against the paper without a search.
:class:`~brokefish.search.torch_impl.Search` owns the state and calls in.

## What the four pieces are

1. **Root sampling.** Draw `g(a) ~ Gumbel(0)` once per move and keep it for the
   whole search. `argtop_m(g + logits)` is *exactly* a sample of `m` distinct
   actions without replacement from `softmax(logits)` — the Gumbel-top-k
   theorem, exact and not an approximation. This is why there is no Dirichlet
   under Gumbel: the noise is the sampling, not a perturbation of it.
2. **Sequential halving** allocates the budget over those `m`, in
   `ceil(log2 m)` phases. Implemented as :func:`visit_table`, below.
3. **The training target** is `softmax(logits + sigma(completedQ))` over the
   *whole* edge set, not `N / n`. Unvisited edges are completed with `v_mix`
   rather than left at zero, which is the entire point: at `n = 128` over ~42
   root edges, `N / n` is a 3-visits-per-edge histogram and is not a policy
   improvement operator at all.
4. **Interior selection** is `argmax(pi' - N / (1 + sum N))`, deterministic, with
   no `pb_c` and no first-play-urgency constant — an unvisited edge is scored at
   `v_mix`, not at a fixed number. Given what a literal `0` in that position cost
   this project (`FPU_DRAW`'s comment, and the 2026-07-31 collapse), removing the
   constant entirely is not a small thing.

## ⚠️ Transcribed, not derived

The visit table and the constants come from DeepMind's `mctx`
(`_src/seq_halving.py`, `_src/qtransforms.py`, `_src/action_selection.py`), which
is what AlphaGateau and every published Gumbel result actually ran. The paper
gives `c_visit = 50` and `c_scale = 1.0` for Q in `[-1, 1]`; `mctx` ships
`maxvisit_init = 50.0` and `value_scale = 0.1` applied *after* a min-max rescale
of the completed Q values into `[0, 1]`, and the defaults here are `mctx`'s. An
off-by-one in :func:`considered_visit_sequence` produces a tree that looks
healthy in every counter and spends its budget wrong — the same failure shape as
the playout-cap bug of 2026-08-07 — so it is copied line for line.

## ⚠️ One deliberate deviation from `mctx`

`mctx` min-max rescales over the *whole* fixed action space, illegal actions
included; they carry the completion value and so can widen the range. Chess here
has a ragged edge list under a cap of `E`, and the tail is padding rather than
illegal moves, so :func:`completed_q` rescales over the valid prefix alone. It
is the more defensible choice and it is a real difference, which matters if
`mctx` is ever used as the oracle of §12.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

# `mctx`'s `qtransform_completed_by_mix_value` defaults. See the module note.
C_VISIT = 50.0
C_SCALE = 0.1
# `score_considered`'s floor. It keeps a row from being all `-inf` before the
# visit-count penalty is applied, so the argmax always has something to return.
LOW_LOGIT = -1e9
_EPS = 1e-8


# -- sequential halving ---------------------------------------------------- #

def considered_visit_sequence(m: int, n: int) -> List[int]:
    """`mctx.seq_halving.get_sequence_of_considered_visits`, transcribed.

    Returns, for each simulation index `s < n`, **the visit count that the edge
    to be visited at `s` currently holds**. That indirection is the trick: the
    halving needs no explicit alive set, because an edge dropped at a phase
    boundary never again holds the required count, so the visit-count match
    *is* the survivor test.
    """
    if m <= 1:
        return list(range(n))
    log2max = int(math.ceil(math.log2(m)))
    sequence: List[int] = []
    visits = [0] * m
    num_considered = m
    while len(sequence) < n:
        extra = max(1, int(n / (log2max * num_considered)))
        for _ in range(extra):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return sequence[:n]


def visit_table(m_max: int, n: int, device) -> torch.Tensor:
    """`[m_max + 1, n]` int32, row `m` being the schedule for `m` considered edges.

    Row 0 is never indexed — a node with no edge is not searched (invariant 5) —
    and holds row 1's schedule so that a bad index degrades to round-robin on one
    edge rather than to garbage.
    """
    rows = [considered_visit_sequence(m, n) for m in range(m_max + 1)]
    return torch.tensor(rows, dtype=torch.int32, device=device)


# -- the Q transform ------------------------------------------------------- #

def completed_q(q01: torch.Tensor, nvis: torch.Tensor, valid: torch.Tensor,
                node_value01: torch.Tensor, prior: torch.Tensor,
                c_visit: float = C_VISIT, c_scale: float = C_SCALE
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`mctx.qtransforms.qtransform_completed_by_mix_value`.

    Returns ``(sigma, completed)``: the scaled quantity that is added to the
    logits, and the raw completed Q in `[0, 1]` that `root_value` is formed from.

    Everything is in the tree's `[0, 1]` convention (§3.5) — ``q01`` is
    ``edge_Q``, ``node_value01`` is ``node_value``. The min-max rescale means the
    convention drops out of ``sigma``, but ``completed`` keeps it.

    ``v_mix`` is the paper's completion value: the node's own network estimate
    blended with the prior-weighted mean of the Q values that *were* visited,
    weighted by how much search has happened. With no visits it is the raw value;
    with many it is the search's own answer.
    """
    visited = valid & (nvis > 0)
    zero = torch.zeros_like(q01)

    sum_visits = torch.where(valid, nvis, zero).sum(-1)
    sum_probs = torch.where(visited, prior, zero).sum(-1)
    # The `where` on the denominator is `mctx`'s: the numerator is already zero
    # wherever `sum_probs` is, so this only keeps 0/0 out of the graph.
    denom = torch.where(sum_probs > 0, sum_probs, torch.ones_like(sum_probs))
    weighted_q = torch.where(visited, prior * q01 / denom[:, None], zero).sum(-1)
    v_mix = (node_value01 + sum_visits * weighted_q) / (sum_visits + 1.0)

    completed = torch.where(visited, q01, v_mix[:, None])

    # ⚠️ Over the valid prefix only — see the module docstring's deviation note.
    big = torch.full_like(completed, float("inf"))
    lo = torch.where(valid, completed, big).min(-1, keepdim=True).values
    hi = torch.where(valid, completed, -big).max(-1, keepdim=True).values
    scaled = (completed - lo) / (hi - lo).clamp(min=_EPS)

    visit_scale = c_visit + torch.where(valid, nvis, zero).max(-1, keepdim=True).values
    sigma = visit_scale * c_scale * scaled
    return torch.where(valid, sigma, zero), torch.where(valid, completed, zero)


def edge_logits(prior: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The node's logits, recovered from `edge_prior`, `-inf` off the edge set.

    `_expand` writes a softmax renormalised over the kept edges, so `log(prior)`
    is the logits up to an additive constant — and every use of this is inside a
    softmax or an argmax, both shift-invariant. The clamp is load-bearing:
    `edge_prior` is fp16 and underflows to a hard zero below ~6e-8, which would
    put `-inf` on a *legal* move and make it unreachable at every budget.
    """
    p = prior.float().clamp(min=torch.finfo(torch.float16).tiny)
    lg = torch.log(p)
    return torch.where(valid, lg, torch.full_like(lg, float("-inf")))


# -- the three selection rules --------------------------------------------- #

def root_scores(gumbel: torch.Tensor, logits: torch.Tensor, sigma: torch.Tensor,
                nvis: torch.Tensor, considered_visit: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    """`mctx.seq_halving.score_considered`, for the argmax of one simulation.

    ``considered_visit`` is `[B, 1]` from :func:`visit_table`. Only edges holding
    exactly that visit count are eligible, which is both the round-robin *and*
    the halving.
    """
    ninf = torch.full_like(sigma, float("-inf"))
    shifted = logits - torch.where(valid, logits, ninf).max(-1, keepdim=True).values
    s = torch.maximum(gumbel + shifted + sigma, torch.full_like(sigma, LOW_LOGIT))
    return torch.where(valid & (nvis == considered_visit), s, ninf)


def interior_scores(logits: torch.Tensor, sigma: torch.Tensor, nvis: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """`mctx.action_selection._prepare_argmax_input`: `pi' - N / (1 + sum N)`.

    Argmaxing this repeatedly, with `N` updated each time, drives the visit
    frequencies towards `pi'` — paper §5, "Planning at non-root nodes". No
    Gumbel noise below the root: the improvement comes from the *deterministic*
    visit matching, and noise there would only add variance to the estimate the
    root is trying to read.
    """
    probs = improved_policy(logits, sigma, valid)
    total = torch.where(valid, nvis, torch.zeros_like(nvis)).sum(-1, keepdim=True)
    return torch.where(valid, probs - nvis / (1.0 + total),
                       torch.full_like(probs, float("-inf")))


def improved_policy(logits: torch.Tensor, sigma: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """`softmax(logits + sigma(completedQ))` over the edge set — §6.7's target.

    Dense over every valid edge, including the ones no simulation reached, which
    is what `N / n` cannot be. §10 already stores the root's whole edge set with
    `pi = 0` on the unvisited ones (revised 2026-07-31, for the training
    *denominator*), so the record schema does not change and the buffer does not
    migrate — those entries simply stop being zero.
    """
    ninf = torch.full_like(sigma, float("-inf"))
    return torch.softmax(torch.where(valid, logits + sigma, ninf), dim=-1)
