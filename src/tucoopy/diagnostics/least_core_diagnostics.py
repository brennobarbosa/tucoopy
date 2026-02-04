"""
# Least-core diagnostics.

The **least-core** is the epsilon-core with the smallest feasible epsilon value
$\\epsilon*$.

This module provides `least_core_diagnostics`, which:

1. Computes $\\epsilon*$ via an LP (requires an LP backend),
2. Reuses `tucoopy.diagnostics.epsilon_core_diagnostics.epsilon_core_diagnostics`
   to assess membership of a specific allocation $x$ in the epsilon-core at
   $\\epsilon = \\epsilon*$.

Notes
-----
If the LP backend is unavailable, diagnostics are returned with
``available=False`` and a human-readable ``reason``.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.diagnostics.least_core_diagnostics import least_core_diagnostics
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> d = least_core_diagnostics(g, [0.5, 0.5])  # doctest: +SKIP
>>> d.available  # doctest: +SKIP
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from .epsilon_core_diagnostics import EpsilonCoreDiagnostics, epsilon_core_diagnostics


@dataclass(frozen=True)
class LeastCoreDiagnostics:
    """
    Diagnostics for membership in the least-core ($\\epsilon*$).

    If the LP backend is unavailable, `available=False` and `reason` is set.

    Attributes
    ----------
    available
        Whether the least-core computation was available (LP backend present).
    reason
        Explanation when ``available`` is False.
    epsilon_star
        The least-core value ``epsilon*`` when available.
    epsilon_core
        Epsilon-core diagnostics at ``epsilon = epsilon*`` when available.

    Examples
    --------
    >>> d = LeastCoreDiagnostics(available=False, reason="no backend", epsilon_star=None, epsilon_core=None)
    >>> d.available
    False
    """

    available: bool
    reason: str | None
    epsilon_star: float | None
    epsilon_core: EpsilonCoreDiagnostics | None

    def to_dict(self) -> dict[str, object]:
        """
        Convert diagnostics to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation of the diagnostics dataclass.

        Examples
        --------
        >>> d = LeastCoreDiagnostics(available=False, reason="no backend", epsilon_star=None, epsilon_core=None)
        >>> d.to_dict()["available"]
        False
        """
        return asdict(self)


def least_core_diagnostics(
    game: GameProtocol,
    x: list[float],
    *,
    tol: float = DEFAULT_GEOMETRY_TOL,
    top_k: int = 8,
) -> LeastCoreDiagnostics:
    """
    Compute least-core membership diagnostics for an allocation $x$.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector of length ``game.n_players``.
    tol
        Numerical tolerance passed to the least-core LP routine and to the
        epsilon-core diagnostic.
    top_k
        Maximum number of violating coalitions to include in the epsilon-core
        sub-diagnostic.

    Returns
    -------
    LeastCoreDiagnostics
        Diagnostics including the computed ``epsilon_star`` (when available)
        and the epsilon-core diagnostics at that epsilon.

    Notes
    -----
    This function catches exceptions from the least-core LP routine and returns
    them as ``reason``. This is intentional to keep diagnostics "UI-friendly".

    Examples
    --------
    Minimal example (requires an LP backend; skip if not available):

    >>> from tucoopy import Game
    >>> from tucoopy.diagnostics.least_core_diagnostics import least_core_diagnostics
    >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
    >>> d = least_core_diagnostics(g, [0.5, 0.5])  # doctest: +SKIP
    >>> d.available  # doctest: +SKIP
    True
    """
    try:
        from ..solutions.least_core import least_core_epsilon_star

        eps_star = float(least_core_epsilon_star(game, tol=float(tol)))
    except Exception as e:  # keep this diagnostic-friendly
        return LeastCoreDiagnostics(
            available=False,
            reason=str(e),
            epsilon_star=None,
            epsilon_core=None,
        )

    d = epsilon_core_diagnostics(game, x, epsilon=eps_star, tol=tol, top_k=top_k)
    return LeastCoreDiagnostics(
        available=True,
        reason=None,
        epsilon_star=eps_star,
        epsilon_core=d,
    )


__all__ = [
    "LeastCoreDiagnostics",
    "least_core_diagnostics",
]
