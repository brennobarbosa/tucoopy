"""
# Imputation and pre-imputation diagnostics.

This module provides checks for whether a payoff vector is:

- efficient (pre-imputation), and
- individually rational (imputation).

It is used both directly and as a building block for higher-level set
diagnostics.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.diagnostics.imputation_diagnostics import imputation_diagnostics
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> imputation_diagnostics(g, [0.5, 0.5]).in_set
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from .bounds import BoundViolation


@dataclass(frozen=True)
class ImputationDiagnostics:
    """
    Diagnostics for membership in the imputation set.

    Examples
    --------
    >>> from tucoopy.diagnostics.bounds import BoundViolation
    >>> d = ImputationDiagnostics(
    ...     n_players=2,
    ...     vN=1.0,
    ...     sum_x=1.0,
    ...     efficient=True,
    ...     in_set=True,
    ...     lower_bounds=[0.0, 0.0],
    ...     violations=[],
    ... )
    >>> d.in_set
    True
    """

    n_players: int
    vN: float
    sum_x: float
    efficient: bool
    in_set: bool
    lower_bounds: list[float]
    violations: list[BoundViolation]

    def to_dict(self) -> dict[str, object]:
        """
        Convert diagnostics to a JSON-serializable dictionary.

        Examples
        --------
        >>> d = ImputationDiagnostics(
        ...     n_players=2,
        ...     vN=1.0,
        ...     sum_x=1.0,
        ...     efficient=True,
        ...     in_set=True,
        ...     lower_bounds=[0.0, 0.0],
        ...     violations=[],
        ... )
        >>> d.to_dict()["sum_x"]
        1.0
        """
        return asdict(self)


def imputation_diagnostics(
    game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL
) -> ImputationDiagnostics:
    """
    Check whether $x$ is an imputation (efficient + individually rational).

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector of length ``game.n_players``.
    tol
        Numerical tolerance used for the efficiency check and the individual
        rationality comparisons.

    Returns
    -------
    ImputationDiagnostics
        Diagnostics object including ``in_set`` and any lower-bound violations.

    Raises
    ------
    InvalidParameterError
        If ``x`` does not have length ``game.n_players``.

    Examples
    --------
    A minimal 3-player example (only the grand coalition has value 1):

    >>> from tucoopy import Game
    >>> from tucoopy.diagnostics.imputation_diagnostics import imputation_diagnostics
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0:0, 1:0, 2:0, 4:0,
    ...     3:0, 5:0, 6:0,
    ...     7:1,
    ... })
    >>> d = imputation_diagnostics(g, [1/3, 1/3, 1/3])
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
    violations: list[BoundViolation] = []
    for i in range(n):
        if float(x[i]) + tol < lower[i]:
            violations.append(BoundViolation(player=i, kind="lower", bound=float(lower[i]), value=float(x[i])))

    in_set = efficient and not violations
    return ImputationDiagnostics(
        n_players=n,
        vN=vN,
        sum_x=sum_x,
        efficient=efficient,
        in_set=in_set,
        lower_bounds=lower,
        violations=violations,
    )
