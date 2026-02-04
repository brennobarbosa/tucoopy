"""
# Reasonable set (polyhedral superset of the core).

This module provides :class:`ReasonableSet`, which is the efficient box defined
by per-player lower bounds (individual rationality) and upper bounds (utopia
payoffs).

The reasonable set is polyhedral and is often used as a computationally friendly
outer approximation of the core.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.geometry import ReasonableSet
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> ReasonableSet(g).contains([0.5, 0.5])
True
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import DEFAULT_GEOMETRY_MAX_DIM, DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..solutions.tau import utopia_payoff
from .polyhedron import PolyhedralSet


@dataclass(frozen=True)
class ReasonableSet:
    """
    Reasonable set as a polyhedral set.

    Background
    ----------
    For a TU cooperative game, define for each player the **utopia payoff**

    $$
    M_i = v(N) - v(N \\setminus \\{i\\}),
    $$

    which represents the maximum amount player $i$ could conceivably receive
    without making the remaining players worse off than standing alone.

    The **reasonable set** is defined as

    $$
    R = \\left\\{ x \\in \\mathbb{R}^n :
        \\sum_i x_i = v(N), \\quad
        v(\\{i\\}) \\le x_i \\le M_i \\ \\text{for all } i
    \\right\\}.
    $$

    It is the set of allocations that are:

    - **efficient** (use the whole worth of the grand coalition),
    - **individually rational** ($x_i \\ge v(\\{i\\})$),
    - **utopia-bounded** ($x_i \\le M_i$).

    Geometrically, this is the intersection of the imputation simplex with the
    hyper-rectangle defined by the utopia payoffs.

    Parameters
    ----------
    game
        TU game.

    Attributes
    ----------
    poly
        Underlying `PolyhedralSet` representing the reasonable set in H-representation.

    Notes
    -----
    - The reasonable set always contains the **tau-value** (when defined).
    - It is typically a superset of the core when the core is non-empty.
    - For small n, `vertices()` can be used for visualization.

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.geometry import ReasonableSet
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
    >>> R = ReasonableSet(g)
    >>> R.contains([1.0, 1.0, 1.0])
    True
    """

    game: GameProtocol

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation of the reasonable set.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).poly.n_vars
        2
        """
        n = self.game.n_players
        vN = float(self.game.value(self.game.grand_coalition))
        lower = [float(self.game.value(1 << i)) for i in range(n)]
        upper = [float(v) for v in utopia_payoff(self.game)]
        return PolyhedralSet.from_hrep(
            A_eq=[[1.0] * n],
            b_eq=[vN],
            bounds=[(lower[i], upper[i]) for i in range(n)],
        )

    def contains(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Check if x belongs to the reasonable set (within tolerance).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).contains([0.5, 0.5])
        True
        """
        return self.poly.contains(x, tol=tol)

    def check(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL):
        """
        Return reasonable-set membership diagnostics for ``x``.

        Notes
        -----
        This delegates to :func:`tucoopy.diagnostics.reasonable_diagnostics.reasonable_set_diagnostics`.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).check([0.5, 0.5]).in_set
        True
        """
        from ..diagnostics.reasonable_diagnostics import reasonable_set_diagnostics

        return reasonable_set_diagnostics(self.game, x, tol=tol)

    def explain(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> list[str]:
        """
        Return a short human-readable explanation of reasonable-set membership.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).explain([0.5, 0.5])
        ['In the reasonable set.']
        """
        d = self.check(x, tol=tol)
        lines: list[str] = []
        if not d.efficient:
            lines.append(f"Not efficient: sum(x)={d.sum_x:.6g} but v(N)={d.vN:.6g}.")
        for v in d.violations[:1]:
            lines.append(f"Violates bound: x[{v.player}]={v.value:.6g} outside [{d.lower_bounds[v.player]:.6g}, {d.upper_bounds[v.player]:.6g}].")
        if not lines and d.in_set:
            lines.append("In the reasonable set.")
        return lines

    def is_empty(self) -> bool:
        """
        Check if the reasonable set is empty.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Attempt to sample one point from the reasonable set.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Chebyshev center of the reasonable set.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points (small dimension).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).extreme_points(max_dim=2)
        [[0.0, 1.0], [1.0, 0.0]]
        """
        return self.poly.extreme_points(tol=tol, max_dim=max_dim)

    def project(
        self,
        dims: tuple[int, ...] | list[int],
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        """
        Project the reasonable set to selected coordinates.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).project((0,), max_dim=2)
        [[0.0], [1.0]]
        """
        return self.poly.project(dims, tol=tol, max_dim=max_dim)

    def vertices(
        self,
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_players: int = DEFAULT_GEOMETRY_MAX_PLAYERS,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        """
        Vertices of the reasonable set (small dimension).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ReasonableSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ReasonableSet(g).vertices(max_dim=2)
        [[0.0, 1.0], [1.0, 0.0]]
        """
        _ = max_players  # part of the shared vertices(...) signature; not used here
        return self.extreme_points(tol=tol, max_dim=max_dim)


__all__ = ["ReasonableSet"]
