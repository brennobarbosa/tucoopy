"""
# Shapley value and semivalues.

This module implements the Shapley value, weighted Shapley variants, Monte Carlo approximation,
and a generic semivalue helper.
"""

from __future__ import annotations

from math import factorial
from random import Random
from typing import Sequence, Literal

from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..transforms._utils import to_dense_values
from ..transforms.harsanyi import harsanyi_dividends
from ..transforms.mobius import mobius_transform


def semivalue(
    game: GameProtocol, *, weights_by_k: Sequence[float], normalize: bool = False
) -> list[float]:
    """
    Compute a **semivalue** for a TU cooperative game.

    A semivalue is defined by weights $p_k$ for coalition sizes $k=0,\\ldots,n-1$:

    $$
    \\phi_i =
    \\sum_{S \\subseteq N \\setminus \\{i\\}}
    p_{|S|}
    \\big( v(S \\cup \\{i\\}) - v(S) \\big).
    $$

    The standard semivalue normalization is:

    $$
    \\sum_{k=0}^{n-1} \\binom{n-1}{k} \\, p_k = 1.
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.
    weights_by_k : Sequence[float]
        Sequence of length `n_players` where entry k is $p_k$.
    normalize : bool, default=False
        If True, rescale $p_k$ to satisfy the semivalue normalization.

    Returns
    -------
    list[float]
        Allocation vector of length `n_players`.

    Raises
    ------
    InvalidParameterError
        If `weights_by_k` has the wrong length, or normalization is requested
        with all-zero weights.

    Notes
    -----
    - The Shapley value is a special case of a semivalue with
      $p_k = \\frac{k!(n-k-1)!}{n!}$.
    - Complexity is $O(n 2^n)$ due to coalition enumeration.

    Examples
    --------
    >>> # Example: choose p_k proportional to 1/(n * C(n-1,k)) and normalize
    >>> phi = semivalue(g, weights_by_k=[1.0]*g.n_players, normalize=True)
    >>> len(phi) == g.n_players
    True
    """
    n = game.n_players
    if len(weights_by_k) != n:
        raise InvalidParameterError("weights_by_k must have length n_players (k=0..n-1)")

    p = [float(v) for v in weights_by_k]

    # Normalize if requested (binomial normalization).
    if normalize:
        # compute sum_{k} C(n-1,k) p_k
        denom = 0.0
        for k in range(n):
            denom += float(
                factorial(n - 1) / (factorial(k) * factorial(n - 1 - k))
            ) * float(p[k])
        if denom == 0.0:
            raise InvalidParameterError("cannot normalize semivalue: weights sum to 0")
        p = [float(v) / float(denom) for v in p]

    phi = [0.0] * n
    for i in range(n):
        bit_i = 1 << i
        for S in range(1 << n):
            if S & bit_i:
                continue
            k = int(S).bit_count()
            phi[i] += float(p[k]) * float(game.value(S | bit_i) - game.value(S))

    return [float(v) for v in phi]


def weighted_shapley_value(game: GameProtocol, *, weights: Sequence[float]) -> list[float]:
    """
    Compute the **weighted Shapley value** (Shapley value with weighted symmetry).

    Using Harsanyi dividends $d(S)$, the weighted Shapley value is:

    $$
    \\phi_i =
    \\sum_{S \\ni i} d(S) \\frac{w_i}{\\sum_{j \\in S} w_j}.
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.
    weights : Sequence[float]
        Positive player weights (length `n_players`).

    Returns
    -------
    list[float]
        Allocation vector of length `n_players`.

    Raises
    ------
    InvalidParameterError
        If the weights are non-positive or have the wrong length.

    Notes
    -----
    - This uses Harsanyi dividends internally, so the overall cost is dominated
      by computing the Möbius transform (typically $O(n2^n)$).
    - Setting all weights equal reduces to the standard Shapley value.

    Examples
    --------
    >>> phi_w = weighted_shapley_value(g, weights=[2.0, 1.0, 1.0])
    >>> len(phi_w) == g.n_players
    True
    """
    n = game.n_players
    if len(weights) != n:
        raise InvalidParameterError("weights must have length n_players")
    w = [float(v) for v in weights]
    if any(v <= 0.0 for v in w):
        raise InvalidParameterError("weights must be positive")

    d = harsanyi_dividends(game)
    out = [0.0] * n

    for S, div in d.items():
        if S == 0:
            continue
        denom = 0.0
        for i in range(n):
            if S & (1 << i):
                denom += w[i]
        if denom <= 0.0:
            raise InvalidParameterError("invalid weights: sum weights in coalition is zero")
        for i in range(n):
            if S & (1 << i):
                out[i] += float(div) * (w[i] / denom)

    return [float(v) for v in out]


def shapley_value(game: GameProtocol) -> list[float]:
    """
    Compute the **Shapley value** for a TU cooperative game.

    The Shapley value assigns to each player their expected marginal
    contribution under a uniformly random permutation of players.

    Formally:

    $$
    \\phi_i =
    \\sum_{S \\subseteq N \\setminus \\{i\\}}
        \\frac{|S|! (n-|S|-1)!}{n!}
        \\bigl( v(S \\cup \\{i\\}) - v(S) \\bigr).
    $$

    Parameters
    ----------
    game : Game
        TU cooperative game.

    Returns
    -------
    list[float]
        Shapley value allocation (length `n_players`).

    Notes
    -----
    - Complexity is $O(n 2^n)$.
    - Exact and deterministic; suitable for small to medium `n`.

    Examples
    --------
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
    >>> shapley_value(g)
    [1.0, 1.0, 1.0]
    """
    n = game.n_players
    n_fact = factorial(n)
    phi = [0.0 for _ in range(n)]

    for i in range(n):
        bit_i = 1 << i
        for S in range(1 << n):
            if S & bit_i:
                continue
            s_size = S.bit_count()
            w = factorial(s_size) * factorial(n - s_size - 1) / n_fact
            phi[i] += w * (game.value(S | bit_i) - game.value(S))

    return phi


def shapley_value_fast(
    game: GameProtocol,
    *,
    backend: Literal["auto", "numpy", "python"] = "auto",
) -> list[float]:
    """
    Fast exact Shapley value via Harsanyi dividends (Möbius transform).

    Using Harsanyi dividends d(S) (unanimity coordinates), the Shapley value admits
    the closed form:

    $$
    \\phi_i(v) = \\sum_{S \\ni i} \\frac{d(S)}{|S|}.
    $$

    This implementation computes dividends via a fast Möbius transform and then
    aggregates contributions across coalitions.

    Parameters
    ----------
    game : Game
        TU game.
    backend : {"auto", "numpy", "python"}, default="auto"
        Backend for the Möbius transform:
        - "auto": try NumPy, fall back to Python
        - "numpy": force NumPy (raises ImportError if unavailable)
        - "python": pure-Python

    Returns
    -------
    list[float]
        The Shapley value allocation (length `n_players`).

    Notes
    -----
    - Complexity is $O(n 2^n)$, like the classic formula, but with substantially
      better constants when the Möbius transform uses NumPy.
    - This is an exact method (not Monte Carlo).
    - Conceptually, it decomposes the game into unanimity games and distributes
      each dividend equally among players in the coalition.

    Examples
    --------
    >>> phi = shapley_value_fast(g, backend="auto")
    >>> len(phi) == g.n_players
    True
    """
    n = game.n_players

    # Dense v over all masks
    values = to_dense_values(game)  # length 2^n, values[mask] = v(mask)
    dividends = mobius_transform(values, n_players=n, backend=backend)  # d[mask]

    phi = [0.0] * n
    for S in range(1, 1 << n):
        dS = float(dividends[S])
        if dS == 0.0:
            continue
        k = int(S).bit_count()
        share = dS / float(k)
        # distribute among members of S
        for i in range(n):
            if S & (1 << i):
                phi[i] += share

    return [float(v) for v in phi]


def shapley_value_sample(
    game: GameProtocol,
    *,
    n_samples: int,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Monte Carlo estimate of the **Shapley value** using random permutations.

    For each sampled permutation, the marginal contribution vector is computed.
    The estimator returns the sample mean and an estimated per-player
    standard error.

    Parameters
    ----------
    game : Game
        TU cooperative game.
    n_samples : int
        Number of random permutations to sample.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    (phi_hat, stderr)
        Estimated Shapley value and per-player standard error.

    Notes
    -----
    - Complexity is $O(n \\cdot n_{samples})$.
    - Suitable for large `n` when exact computation is infeasible.

    Examples
    --------
    >>> phi_hat, stderr = shapley_value_sample(g, n_samples=1000)
    """
    n = game.n_players
    if n_samples < 1:
        raise InvalidParameterError("n_samples must be >= 1")

    rng = Random(seed)
    players = list(range(n))

    mean = [0.0] * n
    m2 = [0.0] * n

    for k in range(1, int(n_samples) + 1):
        rng.shuffle(players)
        mask = 0
        prev_v = float(game.value(0))
        contrib = [0.0] * n

        for idx in range(n):
            i = int(players[idx])
            mask |= 1 << i
            vS = float(game.value(mask))
            d = vS - prev_v
            contrib[i] = d
            prev_v = vS

        # Welford update
        for i in range(n):
            x = float(contrib[i])
            delta = x - mean[i]
            mean[i] += delta / float(k)
            delta2 = x - mean[i]
            m2[i] += delta * delta2

    if n_samples == 1:
        return mean, [0.0 for _ in range(n)]

    # stderr = sqrt( sample_variance / n_samples )
    stderr = []
    for i in range(n):
        var = m2[i] / float(n_samples - 1)
        stderr.append((var / float(n_samples)) ** 0.5)
    return mean, stderr


def shapley_value_sample_stratified(
    game: GameProtocol,
    *,
    samples_per_k: int,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Stratified Monte Carlo estimate of the Shapley value by coalition size.

    This estimator samples random coalitions $S$ of each size $k = 0 \\ldots n-1$ from
    $N \\setminus \\{i\\}$ and averages the marginal contributions:

    $$\\Delta_i(S) = v(S \\cup \\{i\\}) - v(S)$$

    weighted by the Shapley coefficient:

    $$w_k = \\frac{k!(n-k-1)!}{n!}.$$

    Compared to permutation sampling, this can reduce variance in games where
    marginal contributions depend strongly on coalition size.

    Parameters
    ----------
    game : Game
        TU game.
    samples_per_k : int
        Number of sampled coalitions per coalition size k, for each player.
        Must be >= 1.
    seed : int | None, optional
        RNG seed.

    Returns
    -------
    (phi_hat, stderr)
        `phi_hat` is the estimated Shapley value and `stderr` is an estimated
        per-player standard error (based on sample variance across draws).

    Notes
    -----
    - Total samples scale as O(n^2 * samples_per_k) because we sample for each player
      and each k.
    - This is still much cheaper than O(n 2^n) when n is moderate/large.
    - stderr here is an empirical standard error from the stratified samples.

    Examples
    --------
    >>> phi_hat, se = shapley_value_sample_stratified(g, samples_per_k=200, seed=0)
    >>> len(phi_hat) == g.n_players
    True
    """
    n = game.n_players
    if samples_per_k < 1:
        raise InvalidParameterError("samples_per_k must be >= 1")

    rng = Random(seed)
    n_fact = factorial(n)
    fact = [factorial(k) for k in range(n + 1)]
    w_by_k = [(fact[k] * fact[n - k - 1]) / n_fact for k in range(n)]

    mean = [0.0] * n
    m2 = [0.0] * n
    draws = [0] * n  # same for all, but keep per player for clarity

    # Pre-build list of players for sampling
    all_players = list(range(n))

    for i in range(n):
        others = [p for p in all_players if p != i]
        bit_i = 1 << i

        # For each size k, sample subsets uniformly among those of size k
        for k in range(0, n):
            if k > len(others):
                continue
            wk = float(w_by_k[k])

            # Sample `samples_per_k` subsets of size k
            for _ in range(int(samples_per_k)):
                # random subset of size k from others
                rng.shuffle(others)
                chosen = others[:k]
                S = 0
                for p in chosen:
                    S |= 1 << p

                delta = float(game.value(S | bit_i) - game.value(S))
                x = wk * delta

                draws[i] += 1
                t = float(draws[i])

                # Welford update
                d = x - mean[i]
                mean[i] += d / t
                d2 = x - mean[i]
                m2[i] += d * d2

    # stderr per player
    stderr = []
    for i in range(n):
        if draws[i] <= 1:
            stderr.append(0.0)
            continue
        var = m2[i] / float(draws[i] - 1)
        stderr.append((var / float(draws[i])) ** 0.5)

    return [float(v) for v in mean], [float(v) for v in stderr]


__all__ = [
    "shapley_value",
    "shapley_value_fast",
    "shapley_value_sample",
    "shapley_value_sample_stratified",
    "semivalue",
    "weighted_shapley_value",
]
