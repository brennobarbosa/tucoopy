"""
# Core cover (polyhedral superset of the core).

This module provides :class:`CoreCover`, a set-valued object with a polyhedral
representation (H-representation) accessible via the :attr:`CoreCover.poly`
property.

The core cover is a classical polyhedral superset of the core defined using the
**minimal rights** vector and the **utopia payoff** vector.

See Also
--------
tucoop.solutions.tau.minimal_rights
tucoop.solutions.tau.utopia_payoff
tucoop.geometry.PolyhedralSet

Examples
--------
>>> from tucoop import Game
>>> from tucoop.geometry import CoreCover
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> CoreCover(g).contains([0.5, 0.5])
True
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import DEFAULT_GEOMETRY_MAX_DIM, DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..solutions.tau import minimal_rights, utopia_payoff
from .polyhedron import PolyhedralSet


@dataclass(frozen=True)
class CoreCover:
    """
    Core cover polytope.

    Definition
    ----------
    Let:

    $$
    M_i = \\text{ utopia payoff of player } i \\\\
    m_i = \\text{ minimal rights of player } i
    $$

    The **core cover** is the polytope:

    $$
    CC(v) = \\left\\{ x \\in \\mathbb{R}^n \\, : \\, \\sum_{i=1}^n x_i = v(N), m_i \\leq x_i \\leq M_i \\text{ for all } i \\right\\}
    $$

    Interpretation
    --------------
    - The utopia payoff $M_i$ represents the maximum amount player $i$ could
      hope to obtain without making any coalition worse off.
    - The minimal rights $m_i$ represent the minimal amount player $i$ can
      guarantee for themselves considering all coalitional possibilities.
    - The core cover is therefore a **box-constrained efficiency slice**,
      often much easier to compute and visualize than the core.

    The core is always contained in the core cover:

    $$
    \\text{Core}(v) \\subseteq \\text{CoreCover}(v)
    $$

    and for many classes of games, the core cover provides a very tight
    outer approximation of the core.

    Notes
    -----
    - This object is purely geometric and built as a `PolyhedralSet`.
    - All geometric operations (sampling, Chebyshev center, projection,
      extreme points) are delegated to the underlying polyhedron.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.geometry import CoreCover
    >>> g = Game.from_value_function(3, lambda S: float(len(S)))
    >>> cc = CoreCover(g)
    >>> cc.is_empty()
    False
    >>> cc.sample_point()
    [1.0, 1.0, 1.0]
    """

    game: GameProtocol

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation of the core cover.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).poly.n_vars
        2
        """
        vN = float(self.game.value(self.game.grand_coalition))
        M = utopia_payoff(self.game)
        m = minimal_rights(self.game, M=M)
        lower = [float(v) for v in m]
        upper = [float(v) for v in M]
        return PolyhedralSet.from_hrep(
            A_eq=[[1.0] * self.game.n_players],
            b_eq=[vN],
            bounds=[(lower[i], upper[i]) for i in range(self.game.n_players)],
        )

    def contains(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Check if $x$ belongs to the core cover.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).contains([0.5, 0.5])
        True
        """
        return self.poly.contains(x, tol=tol)

    def check(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL):
        """
        Return core-cover membership diagnostics for $x$.

        Notes
        -----
        This delegates to `tucoop.diagnostics.core_cover_diagnostics.core_cover_diagnostics`.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).check([0.5, 0.5]).in_set
        True
        """
        from ..diagnostics.core_cover_diagnostics import core_cover_diagnostics

        return core_cover_diagnostics(self.game, x, tol=tol)

    def explain(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> list[str]:
        """
        Return a short human-readable explanation of core-cover membership.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).explain([0.5, 0.5])
        ['In the core cover.']
        """
        d = self.check(x, tol=tol)
        lines: list[str] = []
        if not d.efficient:
            lines.append(f"Not efficient: sum(x)={d.sum_x:.6g} but v(N)={d.vN:.6g}.")
        for v in d.violations[:1]:
            lines.append(f"Violates bound: x[{v.player}]={v.value:.6g} outside [{d.lower_bounds[v.player]:.6g}, {d.upper_bounds[v.player]:.6g}].")
        if not lines and d.in_set:
            lines.append("In the core cover.")
        return lines

    def is_empty(self) -> bool:
        """
        Check if the core cover is empty.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Sample a feasible point from the core cover.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Compute the Chebyshev center of the core cover.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points of the core cover (small dimension).

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).extreme_points(max_dim=2)
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
        Project the core cover onto selected dimensions.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).project((0,), max_dim=2)
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
        Vertices of the core cover (small dimension).

        Notes
        -----
        This delegates to `extreme_points` for consistency with other
        set-valued objects.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import CoreCover
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> CoreCover(g).vertices(max_dim=2)
        [[0.0, 1.0], [1.0, 0.0]]
        """
        _ = max_players  # part of the shared vertices(...) signature; not used here
        return self.extreme_points(tol=tol, max_dim=max_dim)


__all__ = ["CoreCover"]
