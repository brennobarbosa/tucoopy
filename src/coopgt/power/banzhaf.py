"""
# Banzhaf power index.

This module provides the Banzhaf index for simple (0–1) games, as well as an
efficient dynamic-programming variant for integer weighted voting games.
"""

from __future__ import annotations

from typing import Sequence

from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..properties.simple_games import validate_simple_game
from ..solutions.banzhaf import banzhaf_value, normalized_banzhaf_value
from ._simple_utils import normalize, validate_int_weights


def banzhaf_index(game: GameProtocol, *, normalized: bool = True) -> list[float]:
    """
    Compute the **Banzhaf power index** for a simple game.

    In a simple game ($v(S) \\in \\{0,1\\}$), a player $i$ is **critical**
    (or a *swing player*) in a coalition $S$ if:

    $$
    v(S) = 1 \\quad \\text{and} \\quad v(S \\setminus \\{i\\}) = 0.
    $$

    The (raw) Banzhaf value of player $i$ is the number of coalitions
    in which $i$ is critical. The normalized Banzhaf index divides these
    counts by the total across all players so that the index sums to 1.

    Parameters
    ----------
    game : GameProtocol
        A simple game.
    normalized : bool, default=True
        If True, return the normalized Banzhaf index.
        If False, return the raw Banzhaf value (number of swings scaled by
        $2^{-(n-1)}$).

    Returns
    -------
    list[float]
        Banzhaf index for each player.

    Raises
    ------
    InvalidParameterError
        If the game is not a valid simple game.

    Notes
    -----
    - The Banzhaf index measures **criticality** across all coalitions,
      without weighting by coalition size or permutations.
    - It differs from the Shapley–Shubik index, which is based on pivotality
      in permutations.

    Examples
    --------
    >>> bi = banzhaf_index(g)
    >>> len(bi) == g.n_players
    True
    """
    validate_simple_game(game)
    return normalized_banzhaf_value(game) if normalized else banzhaf_value(game)


def banzhaf_index_weighted_voting(
    weights: Sequence[int],
    quota: int,
    *,
    normalized: bool = True,
) -> list[float]:
    """
    Compute the **Banzhaf index** for an integer weighted voting game
    using dynamic programming (without enumerating all $2^n$ coalitions).

    A weighted voting game is defined by weights $w_1,\\ldots,w_n$ and
    a quota $q$. A coalition $S$ is winning if:

    $$
    \\sum_{i \\in S} w_i \\ge q.
    $$

    A player $i$ is critical in a coalition if removing $i$ changes
    the coalition from winning to losing. This implementation counts
    how many subsets of the other players have total weight in the
    pivotal interval $[q-w_i,\\, q-1]$.

    Parameters
    ----------
    weights : Sequence[int]
        Integer player weights.
    quota : int
        Decision quota.
    normalized : bool, default=True
        If True, normalize the index to sum to 1.

    Returns
    -------
    list[float]
        Banzhaf index for each player.

    Notes
    -----
    - Complexity is pseudo-polynomial in the quota/weight scale.
    - Much faster than full coalition enumeration for moderate weights.
    - If $q = 0$ or $q > \\sum_{i=1}^n w_i$, no player is critical
      and all indices are zero.

    Examples
    --------
    >>> banzhaf_index_weighted_voting([2, 1, 1], quota=3)
    [0.5, 0.25, 0.25]
    """
    w, q = validate_int_weights(weights, quota)
    n = len(w)
    if n < 1:
        raise InvalidParameterError("need at least 1 player")
    if q == 0 or q > sum(w):
        return [0.0 for _ in range(n)]

    out = [0.0] * n
    for i in range(n):
        wi = w[i]
        cap = max(0, q - 1)
        max_w = min(sum(w) - wi, cap)
        dp = [0] * (max_w + 1)
        dp[0] = 1
        for j in range(n):
            if j == i:
                continue
            wj = w[j]
            if wj <= 0:
                for s in range(max_w, -1, -1):
                    dp[s] += dp[s]
                continue
            if wj > max_w:
                continue
            for s in range(max_w - wj, -1, -1):
                dp[s + wj] += dp[s]

        lo = max(0, q - wi)
        hi = min(q - 1, max_w)
        swings = 0
        if lo <= hi:
            for s in range(lo, hi + 1):
                swings += dp[s]
        out[i] = float(swings) / float(2 ** (n - 1))

    if not normalized:
        return out
    return normalize(out)


__all__ = ["banzhaf_index", "banzhaf_index_weighted_voting"]
