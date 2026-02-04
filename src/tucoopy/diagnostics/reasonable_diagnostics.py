"""
# Reasonable set diagnostics.

The **reasonable set** bounds each player's payoff between individual
rationality and their utopia payoff, while requiring efficiency:

$$R(v) = \\{ x : \\sum_{i=1}^n x_i = v(N), \\, v(\\{i\\}) \\leq x_i \\leq M_i \\}$$

where $M$ is the utopia payoff vector.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.diagnostics.reasonable_diagnostics import reasonable_set_diagnostics
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> reasonable_set_diagnostics(g, [0.5, 0.5]).in_set
True
"""

from __future__ import annotations

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..solutions.tau import utopia_payoff
from .bounds import BoundViolation, BoxBoundSetDiagnostics


def reasonable_set_diagnostics(
    game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL
) -> BoxBoundSetDiagnostics:
    """
    Membership diagnostics for the reasonable set.

    $$R(v) = \\{ x : \\sum_{i=1}^n x_i = v(N), \\, v(\\{i\\}) \\leq x_i \\leq M_i \\}.$$

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector of length ``game.n_players``.
    tol
        Numerical tolerance used for the efficiency check and bound comparisons.

    Returns
    -------
    BoxBoundSetDiagnostics
        Diagnostics object including the bounds and any bound violations.

    Raises
    ------
    InvalidParameterError
        If ``x`` does not have length ``game.n_players``.

    See also
    --------
    tucoopy.geometry.reasonable_set.ReasonableSet
        The set-valued wrapper around these constraints.

    Examples
    --------
    A minimal 3-player example (only the grand coalition has value 1):

    >>> from tucoopy import Game
    >>> from tucoopy.diagnostics.reasonable_diagnostics import reasonable_set_diagnostics
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0:0, 1:0, 2:0, 4:0,
    ...     3:0, 5:0, 6:0,
    ...     7:1,
    ... })
    >>> d = reasonable_set_diagnostics(g, [1/3, 1/3, 1/3])
    >>> d.in_set
    True
    """
    n = game.n_players
    if len(x) != n:
        raise InvalidParameterError("x must have length n_players")

    vN = float(game.value(game.grand_coalition))
    sum_x = float(sum(x))
    efficient = abs(sum_x - vN) <= tol

    lower = [float(game.value(1 << i)) for i in range(n)]
    upper = [float(v) for v in utopia_payoff(game)]

    violations: list[BoundViolation] = []
    for i in range(n):
        xi = float(x[i])
        if xi + tol < float(lower[i]):
            violations.append(
                BoundViolation(player=i, kind="lower", bound=float(lower[i]), value=xi)
            )
        if xi - tol > float(upper[i]):
            violations.append(
                BoundViolation(player=i, kind="upper", bound=float(upper[i]), value=xi)
            )

    in_set = efficient and not violations
    return BoxBoundSetDiagnostics(
        n_players=n,
        vN=vN,
        sum_x=sum_x,
        efficient=efficient,
        in_set=in_set,
        lower_bounds=lower,
        upper_bounds=upper,
        violations=violations,
    )


__all__ = [
    "reasonable_set_diagnostics",
]
