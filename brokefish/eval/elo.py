"""One global Bradley-Terry fit over the whole graph of games. `evaluation.md` §5.2.

⚠️ **Not a chain.** Rating checkpoint `k` against `k−1` and summing the deltas
inflates the scale — pairwise error accumulates along the path and non-transitivity
biases every link upward (SAI §3.5). So this module never sees a sequence: it takes
the *set* of edges that were played and fits every rating at once, which is
KataGo's "global Bayesian maximum-likelihood Elo based on all game results so far".

The model is plain Bradley-Terry on scores in {0, 0.5, 1}, fitted to its exact
maximum likelihood by Newton's method with a backtracking line search, warm-started
by a few minorization-maximization (Zermelo) steps, with two departures from the
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
⚠️ **"Well under an Elo point" is true per player and false for the pool**
(measured 2026-09-09, while fixing the convergence). A phantom game against a player
a thousand Elo above zero is a saturated game: it pulls with its full half-point
whatever the gap, and a hundred of them act together on the pool's softest mode,
its distance to the anchor, against `random`'s ~200 informative games. On the
101-player `t24h-reinject-lr6` joint league the final checkpoints sit **~640 Elo
lower at `prior = 1` than at `prior = 0.01`** (1831 vs 2469 for
`t24h-adamw-int8@10218:n256`); a 56-player league moves ~200. The pull depends on
the pool size, so the *level* of two leagues of different sizes is not comparable
even with the same anchor; differences between neighbouring players move by well
under a point. The default is not changed here; that is a decision about what a
rating means, not about the optimiser.

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

**Why Newton and not the Zermelo iteration alone** (2026-09-09). `bt-mm/v1` ran
Zermelo's update to a tolerance of 1e-11 in log-gamma and every joint league of more
than ~60 players hit the 10 000-iteration cap with `converged: False`. The scheme is
sound and monotone; it is *slow* on exactly this graph. The pinned anchor is tied to
the rest of the pool only through `random`'s few hundred games, played mostly at
lopsided scores where a game carries `p(1-p) ≈ 0` information, and through the
phantom's `prior` at `p(1-p) ≈ e^(-θ)` for a 1500-Elo player, i.e. nothing. So the
Hessian's softest direction is *the whole pool sliding together against the anchor*
(smallest eigenvalue 0.16-0.19 against 130-140 on the stiffest), and a Jacobi-type
coordinate iteration contracts that mode at the spectral radius of its iteration
matrix — measured 0.9974-0.9979 per sweep, so 1e-2 → 1e-11 takes ~10 000 sweeps.
Newton solves that mode in one step. The optimum is the same, and the cost of the
cap was measured before the change: ratings moved by at most 0.16 Elo, uniformly
across the pool, and every pairwise difference and every standard error by under
0.002 Elo, on every joint league saved under `runs/` as of 2026-09-09.

⚠️ Nothing here may select a checkpoint (`evaluation.md` §2). It measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 400 / ln(10): the Elo scale's conversion from natural log-odds.
ELO_PER_LOGIT = 400.0 / math.log(10.0)

# v1 was the Zermelo iteration alone and hit its cap on every league above ~60
# players; v2 reaches the same optimum. Ratings from the two are on the same scale.
FIT_VERSION = "bt-newton/v2"


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
            max_iters: int = 200, tol: float = 1e-9,
            min_decisive: int = 30) -> EloFit:
    """Fit every rating at once. `anchor` is pinned at Elo 0.

    ``prior`` is the number of drawn games each player is given against a phantom
    at Elo 0. It must be positive: at `prior = 0` an undefeated player diverges and
    the iteration walks off to infinity instead of failing.

    ``tol`` is on the Newton step in natural log-gamma units: 1e-9 is 1.7e-7 Elo,
    four orders below anything reported and two above the floor float64 leaves on
    the gradient of a 20 000-game league (v1's 1e-11 sat on that floor).
    ``max_iters`` counts warm-start sweeps and Newton steps together; a real league
    converges in 15-20, the cap is a guard against a pathological graph, and
    `converged` says whether it bit.

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
    free = [i for i in range(k) if i != a_idx]
    theta = [0.0] * k                       # log gamma; the anchor stays at 0

    # -- the warm start: a few Zermelo sweeps ------------------------------- #
    #
    #     gamma_i <- w_i / ( sum_j n_ij/(gamma_i + gamma_j)  +  prior/(gamma_i + 1) )
    #
    # A proper minorization step, so monotone in the likelihood with no step size to
    # choose, and it takes the ratings from 0 to within a few Elo of the optimum in
    # ten sweeps. What it cannot do is finish: see the module docstring.
    used = 0
    for _ in range(WARM_SWEEPS):
        if used >= max_iters:
            break
        used += 1
        delta = 0.0
        for i in free:
            gi = math.exp(theta[i])
            denom = prior / (gi + 1.0)
            for j, n in opp[i].items():
                denom += n / (gi + math.exp(theta[j]))
            new = math.log((score[i] + 0.5 * prior) / denom)
            delta = max(delta, abs(new - theta[i]))
            theta[i] = new
        if delta < WARM_TOL:
            break

    # -- Newton with a backtracking line search ----------------------------- #
    #
    # The log-likelihood is strictly concave in theta once the phantom is in (the
    # Hessian is a weighted graph Laplacian plus a positive diagonal), so damped
    # Newton converges from anywhere and quadratically near the optimum. The step
    # solves the same matrix `_standard_errors` inverts, so the covariance is exact
    # at the point the ratings are reported at, which the cap never guaranteed.
    converged = False
    ll = _loglik(theta, score, opp, prior, free)
    while used < max_iters:
        used += 1
        grad, hess = _gradient_and_hessian(theta, score, opp, prior, free)
        step = _solve(hess, grad)
        # The Newton decrement: how much the quadratic model expects to gain.
        decrement = sum(g * d for g, d in zip(grad, step))
        if max(abs(d) for d in step) < tol or decrement < tol * tol:
            converged = True
            break
        t = 1.0
        while True:
            trial = theta[:]
            for pos_i, i in enumerate(free):
                trial[i] = theta[i] + t * step[pos_i]
            ll_trial = _loglik(trial, score, opp, prior, free)
            # Armijo, with a floor at float64's resolution on `ll`: near the
            # optimum the predicted gain is ~1e-16 nats on a sum of ~1e4, which
            # rounding cannot see, and a strict test then rejects the exact step
            # and stalls at ~1e-5 Elo from the optimum (measured 2026-09-09).
            if ll_trial >= ll + 1e-4 * t * decrement - 1e-11 * (1.0 + abs(ll)):
                theta, ll = trial, ll_trial
                break
            t *= 0.5
            if t < 1e-10:
                break
        if t < 1e-10:
            # No ascent left at floating-point resolution. That is the optimum
            # unless the step was still macroscopic, in which case say so.
            converged = max(abs(d) for d in step) < 1e-8
            break

    gamma = [math.exp(x) for x in theta]

    elo = {n: ELO_PER_LOGIT * theta[index[n]] for n in names}

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


# Warm start: enough Zermelo sweeps to leave the region where Newton would need to
# backtrack, few enough that they are not the cost. Measured: ten sweeps on a real
# 100-player league land within ~15 Elo of the optimum, Newton then needs 4-6 steps.
WARM_SWEEPS = 10
WARM_TOL = 1e-3


def _softplus(x: float) -> float:
    """log(1 + e^x) without overflow."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _loglik(theta: List[float], score: List[float], opp: List[Dict[int, int]],
            prior: float, free: List[int]) -> float:
    """Bradley-Terry log-likelihood with the phantom, up to a constant.

    ``sum_i theta_i * (score_i + prior/2) - sum_{i<j} n_ij log(e^theta_i + e^theta_j)
    - prior * sum_i log(e^theta_i + 1)``; the anchor's terms are constants and the
    pair sum is visited once per unordered pair.
    """
    free_set = set(free)
    ll = 0.0
    for i in free:
        ll += theta[i] * (score[i] + 0.5 * prior)
        ll -= prior * (max(theta[i], 0.0) + _softplus(-abs(theta[i])))
        for j, n in opp[i].items():
            # Each unordered pair once. A pair with the anchor is only ever seen
            # from its free end, so it is counted whatever the anchor's index.
            if j < i and j in free_set:
                continue
            ll -= n * (max(theta[i], theta[j]) + _softplus(-abs(theta[i] - theta[j])))
    return ll


def _gradient_and_hessian(theta: List[float], score: List[float],
                          opp: List[Dict[int, int]], prior: float,
                          free: List[int]) -> Tuple[List[float], List[List[float]]]:
    """Gradient and *negated* Hessian of `_loglik` over the free players.

    The negated Hessian is the Fisher information: the weighted graph Laplacian
    with edge weights ``n_ij p_ij (1 - p_ij)`` plus the phantom's ``prior p(1-p)``
    on the diagonal, exactly the matrix `_standard_errors` inverts.
    """
    pos = {i: t for t, i in enumerate(free)}
    m = len(free)
    grad = [0.0] * m
    hess = [[0.0] * m for _ in range(m)]
    for i in free:
        ti = pos[i]
        p0 = 1.0 / (1.0 + math.exp(-theta[i]))           # vs the phantom at 0
        g = score[i] + 0.5 * prior - prior * p0
        hess[ti][ti] += prior * p0 * (1.0 - p0)
        for j, n in opp[i].items():
            p = 1.0 / (1.0 + math.exp(theta[j] - theta[i]))
            g -= n * p
            w = n * p * (1.0 - p)
            hess[ti][ti] += w
            if j in pos:
                hess[ti][pos[j]] -= w
        grad[ti] = g
    return grad, hess


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve ``a x = b`` for symmetric positive-definite `a` by Cholesky.

    The information matrix is SPD whenever the phantom is in (`prior > 0`), so this
    cannot hit a zero pivot; if it does, the graph is disconnected and the message
    names it, as `_invert` does.
    """
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][t] * L[j][t] for t in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError(
                        "the information matrix is singular: some player is "
                        "disconnected from the rest of the league graph, so its "
                        "rating is not identified. Add a pairing against a rated "
                        "opponent.")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][t] * y[t] for t in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[t][i] * x[t] for t in range(i + 1, n))) / L[i][i]
    return x


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
