"""
# Gately point.

The Gately point is an imputation based on a proportionality condition involving individual rationality
and utopia payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.types import GameProtocol
from ..base.exceptions import InvalidGameError, InvalidParameterError


@dataclass(frozen=True)
class GatelyResult:
    """
    Result of the Gately point computation.

    Attributes
    ----------
    x
        Allocation vector (length `n_players`).
    d
        Common propensity-to-disrupt value.
    """

    x: list[float]
    d: float


def gately_point(game: GameProtocol, *, tol: float = 1e-12) -> GatelyResult:
    """
    Compute the **Gately point** for an essential TU cooperative game.

    The Gately point is a single-valued imputation obtained by equalizing
    the **propensity to disrupt** across players. For an imputation $x$,
    the propensity to disrupt of player $i$ is defined as:

    $$
    d_i(x) =
    \\frac{v(N) - x_i}{x_i - v(\\{i\\})}.
    $$

    The Gately point is the imputation $x$ for which all players have the
    same propensity:

    $$
    d_1(x) = d_2(x) = \\cdots = d_n(x).
    $$

    For essential games (where $v(N) > \\sum_i v(\\{i\\})$), this condition
    yields a closed-form solution.

    Parameters
    ----------
    game : GameProtocol
        TU cooperative game.
    tol : float, default=1e-12
        Numerical tolerance for detecting non-essential games.

    Returns
    -------
    GatelyResult
        Allocation and common propensity-to-disrupt value.

    Raises
    ------
    InvalidParameterError
        If:
        - `n_players < 2`, or
    InvalidGameError
        If:
        - the game is non-essential, i.e.
          $v(N) \\le \\sum_i v(\\{i\\})$.

    Notes
    -----
    - The Gately point lies in the imputation set.
    - It balances the incentives of players to leave the grand coalition.
    - The formula used here is valid precisely when the game is essential.
    - The result is numerically adjusted to ensure efficiency
      (sum of allocations equals $v(N)$).

    Examples
    --------
    >>> x = gately_point(g)
    >>> sum(x) == g.value(g.grand_coalition)
    True
    """
    n = game.n_players
    if n < 2:
        raise InvalidParameterError("gately_point requires n_players >= 2")

    vN = float(game.value(game.grand_coalition))
    v1 = [float(game.value(1 << i)) for i in range(n)]
    sum_v1 = float(sum(v1))

    denom = vN - sum_v1
    if denom <= tol:
        raise InvalidGameError("gately_point undefined for non-essential games (need v(N) > sum v({i}))")

    # Common propensity d.
    d = float((n - 1) * vN / denom)
    x = [float((vN + d * v1[i]) / (1.0 + d)) for i in range(n)]

    # Snap efficiency (numerical).
    err = sum(x) - vN
    if abs(err) > 1e-9:
        x[0] -= err

    return GatelyResult(x=[float(v) for v in x], d=float(d))


__all__ = ["GatelyResult", "gately_point"]
