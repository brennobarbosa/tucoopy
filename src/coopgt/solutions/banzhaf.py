"""
# Banzhaf values for TU games.

This module provides the (raw) Banzhaf value and common variants as semivalues.
"""

from __future__ import annotations

from .shapley import semivalue
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError


def banzhaf_value(game: GameProtocol) -> list[float]:
    """
    Compute the **(raw) Banzhaf value** for a TU game.

    The Banzhaf value measures how often a player is *critical* across all
    coalitions, treating all coalitions of the other players as equally likely.

    For player $i$:

    $$
    \\beta_i =
    \\frac{1}{2^{n-1}}
    \\sum_{S \\subseteq N \\setminus \\{i\\}}
    \\big( v(S \\cup \\{i\\}) - v(S) \\big).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Banzhaf value vector of length `n_players`.

    Notes
    -----
    - This is the **raw** Banzhaf value (not normalized across players).
    - It is a semivalue corresponding to uniform weights over coalition sizes.
    - Often used in voting power analysis and simple games.

    Examples
    --------
    >>> beta = banzhaf_value(g)
    >>> len(beta) == g.n_players
    True
    """
    n = game.n_players
    denom = 2 ** (n - 1)
    beta = [0.0 for _ in range(n)]

    for i in range(n):
        bit_i = 1 << i
        total = 0.0
        for S in range(1 << n):
            if S & bit_i:
                continue
            total += game.value(S | bit_i) - game.value(S)
        beta[i] = total / denom

    return beta


def normalized_banzhaf_value(game: GameProtocol) -> list[float]:
    """
    Compute the **normalized Banzhaf value** for a TU game.

    The raw Banzhaf values are rescaled so that the resulting allocation
    sums to the value of the grand coalition:

    $$
    \\hat{\\beta}_i =
    \\frac{\\beta_i}{\\sum_j \\beta_j} \\, v(N).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Normalized Banzhaf allocation.

    Notes
    -----
    - If all raw Banzhaf values are zero, the function returns a zero vector.
    - This normalization makes the Banzhaf value comparable to other
      allocation rules such as the Shapley value.

    Examples
    --------
    >>> nb = normalized_banzhaf_value(g)
    >>> sum(nb) == g.value(g.grand_coalition)
    True
    """
    beta = banzhaf_value(game)
    s = sum(beta)
    vN = float(game.value(game.grand_coalition))
    if s == 0:
        return [0.0 for _ in beta]
    return [b / s * vN for b in beta]


def weighted_banzhaf_value(game: GameProtocol, *, p: float = 0.5) -> list[float]:
    """
    Weighted Banzhaf value (p-binomial semivalue).

    Interprets the marginal contribution as an expectation where each other player
    joins independently with probability p.

    For p=0.5 this equals the (raw) Banzhaf value.

    Parameters
    ----------
    game
        TU game.
    p
        Inclusion probability in [0, 1].

    Returns
    -------
    list[float]
        Allocation vector of length n_players.

    Raises
    ------
    InvalidParameterError
        If p is outside [0, 1].
    """
    if not (0.0 <= float(p) <= 1.0):
        raise InvalidParameterError("p must be in [0,1]")
    n = game.n_players
    weights_by_k = [float(p) ** float(k) * (1.0 - float(p)) ** float(n - 1 - k) for k in range(n)]
    return semivalue(game, weights_by_k=weights_by_k, normalize=False)


__all__ = [
    "banzhaf_value",
    "normalized_banzhaf_value",
    "weighted_banzhaf_value",
]
