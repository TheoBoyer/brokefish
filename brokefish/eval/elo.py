"""One global Bradley-Terry fit over the whole graph of games. `evaluation.md` §5.2.

⚠️ **Not a chain.** Rating checkpoint `k` against `k−1` and summing the deltas
inflates the scale — pairwise error accumulates along the path and non-transitivity
biases every link upward (SAI §3.5). So this module never sees a sequence: it takes
the *set* of edges that were played and fits every rating at once, which is
KataGo's "global Bayesian maximum-likelihood Elo based on all game results so far".

The model is plain Bradley-Terry on scores in {0, 0.5, 1}, fitted by
minorization-maximization (the Zermelo iteration), with two departures from the
textbook, both of which matter in practice:

**The anchor is pinned, not fitted.** `evaluation.md` §5.1's frozen random-init
network is held at `gamma = 1`, i.e. Elo 0, and never updated. That is what makes
the scale mean something across runs and what makes "the anchor drifted" a
detectable event rather than an invisible one.

**A phantom opponent regularizes.** A player who won every game has an infinite
maximum-likelihood rating, and one will: the first real checkpoint against the
random-init anchor is a plausible 100 %. So every player also plays `prior` drawn
games against a phantom fixed at Elo 0. This is the standard BayesElo/Ordo prior in
its simplest form, it keeps every estimate finite, and at `prior = 1` against the
several hundred real games a league player has it moves a rating by well under an
Elo point. It is a *shrinkage toward zero*, so a rating this reports is if anything
slightly conservative.

**On the interval.** The Fisher information of Bradley-Terry is a weighted graph
Laplacian, and inverting it gives the covariance of the ratings — the part that
makes a global fit worth having, since a checkpoint borrows precision from every
path through the graph and not only from its own edges. But BT with half-points is
*misspecified*: it models a game's score as having variance `p(1−p)`, while a match
with draw fraction `d` at even strength has variance `(1−d)/4`, which is smaller.
So the raw Laplacian interval is too wide, by roughly `1/√(1−d)` — at `d = 0.8`
that is a factor of 2.2, which is not a rounding error. The fix is the standard
quasi-likelihood one: estimate the dispersion `phi` from the Pearson residuals and
scale the covariance by it. `dispersion` is reported so the correction is visible
rather than baked in silently, and `EloFit.se_raw` keeps the uncorrected number.

⚠️ Nothing here may select a checkpoint (`evaluation.md` §2). It measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 400 / ln(10): the Elo scale's conversion from natural log-odds.
ELO_PER_LOGIT = 400.0 / math.log(10.0)

FIT_VERSION = "bt-mm/v1"


@dataclass(frozen=True)
class Edge:
    """The aggregate of every game played between two players.

    ``a_wins + draws + b_wins`` is the number of games; the order of `a` and `b`
    carries no meaning beyond which count is which.
    """

    a: str
    b: str
    a_wins: int
    draws: int
    b_wins: int

    @property
    def games(self) -> int:
        return self.a_wins + self.draws + self.b_wins

    @property
    def a_score(self) -> float:
        return self.a_wins + 0.5 * self.draws


@dataclass
class EloFit:
    """Ratings on the anchor's scale, with everything needed to audit them."""

    elo: Dict[str, float]
    se: Dict[str, float]              # quasi-likelihood corrected, see the module docstring
    se_raw: Dict[str, float]          # the plain Bradley-Terry Laplacian interval
    games: Dict[str, int]
    score: Dict[str, float]
    anchor: str
    dispersion: float
    dispersion_applied: bool          # false when there were too few decisive games
    decisive: int
    iterations: int
    converged: bool
    fit_version: str = FIT_VERSION

    def ci95(self, name: str) -> float:
        return 1.96 * self.se[name]

    def predict(self, a: str, b: str) -> float:
        """P(a scores) under the fit. What variance-proportional pairing needs."""
        d = (self.elo[a] - self.elo[b]) / 400.0
        return 1.0 / (1.0 + 10.0 ** (-d))

    def ordered(self) -> List[str]:
        return sorted(self.elo, key=lambda k: self.elo[k])


def fit_elo(edges: Sequence[Edge], anchor: str, prior: float = 1.0,
            max_iters: int = 10_000, tol: float = 1e-11,
            min_decisive: int = 30) -> EloFit:
    """Fit every rating at once. `anchor` is pinned at Elo 0.

    ``prior`` is the number of drawn games each player is given against a phantom
    at Elo 0. It must be positive: at `prior = 0` an undefeated player diverges and
    the iteration walks off to infinity instead of failing.

    ⚠️ ``min_decisive`` is the guard on the dispersion correction, and it exists
    because a real league produced ``±0 Elo`` on 2026-07-31. A league whose games
    are *all* draws has zero Pearson residual, so the estimated dispersion is
    exactly 0 and the corrected interval is exactly 0 — which reads as infinite
    precision and means the opposite. The quasi-likelihood correction is only
    believable when there are decisive games to estimate it from, so below this
    many the uncorrected (conservative, too wide) interval is reported instead and
    ``dispersion_applied`` says so.
    """
    if prior <= 0.0:
        raise ValueError(f"prior must be positive or an undefeated player has no "
                         f"finite rating, got {prior}")

    names = sorted({n for e in edges for n in (e.a, e.b)} | {anchor})
    if anchor not in names:
        raise ValueError(f"anchor {anchor!r} is not a player in this league")
    index = {n: i for i, n in enumerate(names)}
    k = len(names)

    # score[i] is i's total points; opp[i] is the list of (j, games) it played.
    score = [0.0] * k
    games = [0] * k
    opp: List[Dict[int, int]] = [dict() for _ in range(k)]
    for e in edges:
        if e.a == e.b:
            raise ValueError(f"a player cannot play itself: {e.a!r}")
        if e.games == 0:
            continue
        i, j = index[e.a], index[e.b]
        score[i] += e.a_score
        score[j] += e.games - e.a_score
        games[i] += e.games
        games[j] += e.games
        opp[i][j] = opp[i].get(j, 0) + e.games
        opp[j][i] = opp[j].get(i, 0) + e.games

    a_idx = index[anchor]
    gamma = [1.0] * k

    # -- the MM iteration ---------------------------------------------------- #
    #
    # Zermelo's update, which for Bradley-Terry is a proper minorization step and
    # therefore monotone in the likelihood: no line search and no learning rate.
    #
    #     gamma_i <- w_i / ( sum_j n_ij/(gamma_i + gamma_j)  +  prior/(gamma_i + 1) )
    #
    # `w_i` is i's total score including the phantom's `prior/2`, and the trailing
    # term is the phantom's contribution to the denominator, at gamma = 1.
    converged, used = False, 0
    for used in range(1, max_iters + 1):
        delta = 0.0
        for i in range(k):
            if i == a_idx:
                continue                    # §5.1: the anchor defines the zero
            denom = prior / (gamma[i] + 1.0)
            for j, n in opp[i].items():
                denom += n / (gamma[i] + gamma[j])
            if denom <= 0.0:
                continue
            new = (score[i] + 0.5 * prior) / denom
            delta = max(delta, abs(math.log(new) - math.log(gamma[i])))
            gamma[i] = new
        if delta < tol:
            converged = True
            break

    elo = {n: ELO_PER_LOGIT * math.log(gamma[index[n]]) for n in names}

    # -- dispersion and covariance ------------------------------------------- #
    dispersion = _dispersion(edges, index, gamma, prior, k)
    decisive = sum(e.a_wins + e.b_wins for e in edges)
    applied = decisive >= min_decisive
    scale = math.sqrt(dispersion) if applied else 1.0
    se_raw = _standard_errors(opp, gamma, prior, a_idx, k)
    se = {n: se_raw[index[n]] * scale for n in names}

    return EloFit(
        elo=elo,
        se=se,
        se_raw={n: se_raw[index[n]] for n in names},
        games={n: games[index[n]] for n in names},
        score={n: score[index[n]] for n in names},
        anchor=anchor,
        dispersion=dispersion,
        dispersion_applied=applied,
        decisive=decisive,
        iterations=used,
        converged=converged,
    )


def _dispersion(edges: Sequence[Edge], index: Dict[str, int], gamma: List[float],
                prior: float, k: int) -> float:
    """Pearson dispersion of the fitted model. Below 1 when draws are frequent.

    ``sum (s - p)^2 / (p(1-p))`` over games, divided by the residual degrees of
    freedom. A binomial model that fits has this near 1; a league full of draws has
    it near `1 - d`, and that is the factor the interval is too wide by.
    """
    chi2, n_games = 0.0, 0
    for e in edges:
        if e.games == 0:
            continue
        gi, gj = gamma[index[e.a]], gamma[index[e.b]]
        p = gi / (gi + gj)
        var = p * (1.0 - p)
        if var <= 0.0:
            continue
        chi2 += (e.a_wins * (1.0 - p) ** 2
                 + e.draws * (0.5 - p) ** 2
                 + e.b_wins * p ** 2) / var
        n_games += e.games
    # k - 1 free parameters: every player but the pinned anchor.
    dof = n_games - (k - 1)
    if dof <= 0:
        return 1.0
    return chi2 / dof


def _standard_errors(opp: List[Dict[int, int]], gamma: List[float], prior: float,
                     a_idx: int, k: int) -> List[float]:
    """Elo standard errors from the inverse Fisher information.

    In natural log-gamma units the information matrix is the graph Laplacian with
    edge weights ``n_ij p_ij (1 - p_ij)``. The anchor's row and column are removed
    rather than pseudo-inverted, because the anchor is genuinely fixed rather than
    merely unidentified; the phantom adds `prior * p(1-p)` to every diagonal, which
    is what keeps the remaining block non-singular even for a player whose only
    opponent is the anchor.
    """
    free = [i for i in range(k) if i != a_idx]
    pos = {i: t for t, i in enumerate(free)}
    m = len(free)
    if m == 0:
        return [0.0] * k

    info = [[0.0] * m for _ in range(m)]
    for i in free:
        ti = pos[i]
        gi = gamma[i]
        w_phantom = prior * (gi / (gi + 1.0)) * (1.0 / (gi + 1.0))
        info[ti][ti] += w_phantom
        for j, n in opp[i].items():
            gj = gamma[j]
            p = gi / (gi + gj)
            w = n * p * (1.0 - p)
            info[ti][ti] += w
            if j != a_idx:
                info[ti][pos[j]] -= w

    cov = _invert(info)
    out = [0.0] * k
    for i in free:
        v = cov[pos[i]][pos[i]]
        out[i] = ELO_PER_LOGIT * math.sqrt(v) if v > 0.0 else float("nan")
    out[a_idx] = 0.0     # pinned by construction, so it has no interval of its own
    return out


def _invert(a: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan with partial pivoting. `k` is tens, so this is not the cost."""
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ValueError(
                "the information matrix is singular: some player is disconnected from "
                "the rest of the league graph, so its rating is not identified. Add a "
                "pairing against a rated opponent.")
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        m[col] = [x / d for x in m[col]]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f:
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [row[n:] for row in m]


# --------------------------------------------------------------------------- #
# Sizing, so nobody re-derives it in a notebook
# --------------------------------------------------------------------------- #

def elo_half_width(games: int, draw_rate: float) -> float:
    """`evaluation.md` §9's closed form: the 95 % half-width of one match, in Elo.

    ``347 * sqrt(1 - d) / sqrt(N)``, times 1.96. Arithmetic, not a measurement, and
    it describes a *single edge* — the global fit does better than this, because a
    checkpoint's rating is informed by every path through the graph. Use it to size
    a pairing; use `EloFit.se` to report one.
    """
    if games <= 0:
        return float("inf")
    return 1.96 * 347.0 * math.sqrt(max(0.0, 1.0 - draw_rate)) / math.sqrt(games)
