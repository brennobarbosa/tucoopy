"""
# Allocation-level diagnostics.

This module contains small checks for payoff vectors $x$ such as:

- efficiency ($\\sum_{i=1}^n x_i = v(N)$),
- individual rationality ($x_i \\geq v(\\{i\\})$),
- and convenience wrappers used by core-family diagnostics.

The functions here are lightweight and intended to be stable and
serialization-friendly.

Examples
--------
>>> from tucoop import Game
>>> from tucoop.diagnostics.allocation_diagnostics import is_imputation
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
>>> is_imputation(g, [0.5, 0.5])
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from .core_diagnostics import CoreDiagnostics, core_diagnostics


def is_efficient(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
    """
    Check efficiency: $\\sum_{i=1}^n x_i = v(N)$ within tolerance.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector.
    tol
        Numerical tolerance.

    Returns
    -------
    bool
        True if efficient.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.allocation_diagnostics import is_efficient
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> is_efficient(g, [0.2, 0.8])
    True
    """
    return abs(sum(float(v) for v in x) - float(game.value(game.grand_coalition))) <= float(tol)


def is_individually_rational(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
    """
    Check individual rationality: $x_i \\geq v(\\{i\\})$ for all $i$ (within tolerance).

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.allocation_diagnostics import is_individually_rational
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0.2, 2: 0.0, 3: 1.0})
    >>> is_individually_rational(g, [0.2, 0.8])
    True
    >>> is_individually_rational(g, [0.1, 0.9])
    False
    """
    n = game.n_players
    if len(x) != n:
        return False
    for i in range(n):
        if float(x[i]) + float(tol) < float(game.value(1 << i)):
            return False
    return True


def is_imputation(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
    """
    Check imputation membership: efficiency + individual rationality.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.allocation_diagnostics import is_imputation
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0.2, 2: 0.0, 3: 1.0})
    >>> is_imputation(g, [0.2, 0.8])
    True
    >>> is_imputation(g, [0.2, 0.9])  # not efficient
    False
    """
    return is_efficient(game, x, tol=tol) and is_individually_rational(game, x, tol=tol)


@dataclass(frozen=True)
class AllocationChecks:
    """
    Small set of boolean checks + core explanation for a candidate allocation.

    This is intended for UI/debug usage and for JSON export.

    Parameters
    ----------
    efficient : bool
        Whether the allocation is efficient (sum equals grand coalition value).
    imputation : bool
        Whether the allocation is an imputation (efficient and individually rational).
    core : CoreDiagnostics
        Diagnostics for core membership.

    Examples
    --------
    >>> from tucoop.base.game import Game
    >>> from tucoop.diagnostics.allocation_diagnostics import check_allocation
    >>> g = Game([0, 0, 0, 1, 1, 1, 2])
    >>> x = [1, 1]
    >>> result = check_allocation(g, x)
    >>> result.efficient
    True
    >>> result.imputation
    True
    >>> isinstance(result.core, object)
    True
    """

    efficient: bool
    imputation: bool
    core: CoreDiagnostics

    def to_dict(self) -> dict[str, object]:
        """
        Convert the checks to a dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation of the checks.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.diagnostics.allocation_diagnostics import check_allocation
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> d = check_allocation(g, [0.5, 0.5])
        >>> d.to_dict()["efficient"]
        True
        """
        return asdict(self)


def check_allocation(
    game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, core_top_k: int = 8
) -> AllocationChecks:
    """
    Check common conditions for an allocation $x$:
    - efficiency
    - imputation membership
    - core membership diagnostics

    Parameters
    ----------
    game
        TU game (game-like object).
    x : list of float
        The candidate allocation vector.
    tol : float, optional
        Tolerance for numerical checks (default 1e-9).
    core_top_k : int, optional
        Number of top core violations to include in diagnostics (default 8).

    Returns
    -------
    AllocationChecks
        Object containing boolean checks and core diagnostics.

    Examples
    --------
    >>> from tucoop.base.game import Game
    >>> from tucoop.diagnostics.allocation_diagnostics import check_allocation
    >>> g = Game([0, 0, 0, 1, 1, 1, 2])
    >>> x = [1, 1]
    >>> result = check_allocation(g, x)
    >>> result.efficient
    True
    >>> result.imputation
    True
    """
    eff = bool(is_efficient(game, x, tol=tol))
    imp = bool(is_imputation(game, x, tol=tol))
    cd = core_diagnostics(game, x, tol=tol, top_k=core_top_k)
    return AllocationChecks(efficient=eff, imputation=imp, core=cd)


__all__ = [
    "is_efficient",
    "is_individually_rational",
    "is_imputation",
    "AllocationChecks",
    "check_allocation",
]
