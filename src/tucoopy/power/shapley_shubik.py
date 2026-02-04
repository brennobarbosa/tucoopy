"""
# Shapley–Shubik power index.

For simple games, the Shapley–Shubik index coincides with the Shapley value.
This module exposes both a game-level wrapper and a dynamic-programming variant
for integer weighted voting games.
"""

from __future__ import annotations

from math import factorial
from typing import Sequence

from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..properties.simple_games import validate_simple_game
from ..solutions.shapley import shapley_value
from ._simple_utils import validate_int_weights


def shapley_shubik_index(game: GameProtocol) -> list[float]:
    """
    Compute the **Shapley–Shubik power index** for a simple (0–1) game.

    In a simple game, coalitions are either losing or winning:
    $v(S) \\in \\{0,1\\}$. The Shapley–Shubik index of a player is the probability
    (under a uniformly random permutation/order of players) that the player is
    **pivotal**, i.e. the player whose entry into the coalition turns it from
    losing to winning.

    Equivalently, for simple games the Shapley–Shubik index coincides with the
    **Shapley value** of the game:

    $$
    \\text{SSI}(v) = \\phi(v).
    $$

    Parameters
    ----------
    game : Game
        TU game expected to be a *simple game* (0–1 valued).

    Returns
    -------
    list[float]
        Shapley–Shubik index for each player (length `n_players`).

    Raises
    ------
    InvalidParameterError
        If `game` is not a valid simple game (checked by
        `tucoopy.properties.simple_games.validate_simple_game`).

    Notes
    -----
    This function is a thin wrapper around `tucoopy.solutions.shapley.shapley_value`
    after validating that the input is a simple game.

    Examples
    --------
    >>> validate_simple_game(g)
    >>> ssi = shapley_shubik_index(g)
    >>> len(ssi) == g.n_players
    True
    """
    validate_simple_game(game)
    return shapley_value(game)


def shapley_shubik_index_weighted_voting(weights: Sequence[int], quota: int) -> list[float]:
    """
    Compute the **Shapley–Shubik power index** for an integer weighted voting game.

    A weighted voting game is defined by nonnegative integer weights
    $w_1,\\ldots,w_n$ and a quota $q$. A coalition $S$ is winning if:

    $$
    \\sum_{i \\in S} w_i \\ge q.
    $$

    The Shapley–Shubik index of player $i$ is the probability (under a uniformly
    random permutation of players) that player $i$ is **pivotal**, meaning that
    the total weight of players before $i$ in the permutation is strictly below
    the quota, but reaches/exceeds the quota when $i$ is added.

    This implementation uses dynamic programming to count, for each player $i$,
    how many coalitions of size $k$ have total weight in the pivotal interval
    $[q-w_i,\\, q-1]$. Those counts are then weighted by the standard Shapley
    combinatorial coefficients:

    $$
    \\frac{k!\\,(n-k-1)!}{n!}.
    $$

    Parameters
    ----------
    weights : Sequence[int]
        Integer weights for the players.
    quota : int
        Decision quota (integer).

    Returns
    -------
    list[float]
        Shapley–Shubik indices for each player (length `len(weights)`).

    Raises
    ------
    InvalidParameterError
        If inputs are invalid (e.g., negative quota, invalid weights, etc.).

    Notes
    -----
    - This routine is typically much faster than computing the Shapley value by
      enumerating all $2^n$ coalitions when the weights and quota are moderate.
    - Complexity is pseudo-polynomial in the quota/weight scale: it depends on
      the maximum tracked weight sum (effectively up to `quota-1` after validation).
    - Corner cases:
        * If $ q = 0$, everyone is never pivotal (returns all zeros).
        * If $quota > \\sum_{i=1}^n w_i$, no coalition can win (returns all zeros).

    Examples
    --------
    Majority game with weights [2,1,1] and quota 3:

    >>> shapley_shubik_index_weighted_voting([2, 1, 1], quota=3)
    [0.666..., 0.166..., 0.166...]
    """
    w, q = validate_int_weights(weights, quota)
    n = len(w)
    if n < 1:
        raise InvalidParameterError("need at least 1 player")
    if q == 0 or q > sum(w):
        return [0.0 for _ in range(n)]

    n_fact = factorial(n)
    fact = [factorial(k) for k in range(n + 1)]
    out = [0.0] * n

    for i in range(n):
        wi = w[i]
        cap = max(0, q - 1)
        max_w = min(sum(w) - wi, cap)
        dp = [[0] * (max_w + 1) for _ in range(n)]
        dp[0][0] = 1
        for j in range(n):
            if j == i:
                continue
            wj = w[j]
            if wj <= 0:
                for k in range(n - 2, -1, -1):
                    row = dp[k]
                    row2 = dp[k + 1]
                    for s in range(max_w, -1, -1):
                        if row[s]:
                            row2[s] += row[s]
                continue
            if wj > max_w:
                continue
            for k in range(n - 2, -1, -1):
                row = dp[k]
                row2 = dp[k + 1]
                for s in range(max_w - wj, -1, -1):
                    if row[s]:
                        row2[s + wj] += row[s]

        lo = max(0, q - wi)
        hi = min(q - 1, max_w)
        if lo > hi:
            continue

        acc = 0.0
        for k in range(0, n):
            weight_coeff = (fact[k] * fact[n - k - 1]) / n_fact
            cnt = 0
            for s in range(lo, hi + 1):
                cnt += dp[k][s]
            if cnt:
                acc += weight_coeff * float(cnt)
        out[i] = acc

    return out


__all__ = ["shapley_shubik_index", "shapley_shubik_index_weighted_voting"]
