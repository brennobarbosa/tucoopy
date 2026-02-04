"""
# Core as a polyhedral set and a set-valued wrapper.

This module provides:

- `core_polyhedron`: build the core in H-representation,
- `Core`: a thin wrapper exposing a `Core.poly` and delegating common
  operations to it (membership, sampling, vertex enumeration for small $n$).

Notes
-----
The core is described by one equality (efficiency) and one inequality per
non-empty proper coalition. For large $n$, enumerating all inequalities is
expensive; most geometry helpers in this package assume "small $n$" workflows.

Examples
--------
>>> from tucoop import Game
>>> from tucoop.geometry import Core
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> Core(g).contains([0.5, 0.5])
True
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import DEFAULT_GEOMETRY_MAX_DIM, DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..base.exceptions import NotSupportedError
from ..base.coalition import all_coalitions
from .polyhedron import PolyhedralSet


def core_polyhedron(game: GameProtocol, *, restrict_to_imputation: bool = False) -> PolyhedralSet:
    """
    Core as a polyhedral set in H-representation.

    Constraints
    ----------
    - Efficiency:
        
    $$\\sum_{i=1}^n x_i = v(N)$$

    - Coalitional rationality (core constraints):
        
    $$x(S) \\geq v(S) \\text{ for all non-empty proper coalitions } S$$

    If ``restrict_to_imputation=True``, also enforce individual rationality
    $x_i \\geq v(\\{i\\})$ via bounds.

    Parameters
    ----------
    game
        TU game.
    restrict_to_imputation
        If True, add bounds ``x_i >= v({i})``. This can be useful for numerical
        stability when sampling/centers, but note the core is already a subset of
        the imputation set for TU games.

    Returns
    -------
    PolyhedralSet
        A polyhedral set in H-representation.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.geometry import core_polyhedron
    >>>
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
    >>> poly = core_polyhedron(g)
    >>> poly.contains([1.0, 1.0, 1.0])
    True
    >>>
    >>> # Same object but with explicit individual-rationality bounds:
    >>> poly2 = core_polyhedron(g, restrict_to_imputation=True)
    >>> poly2.contains([1.0, 1.0, 1.0])
    True
    """
    n = game.n_players
    N = game.grand_coalition

    A_eq = [[1.0] * n]
    b_eq = [float(game.value(N))]

    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    for S in all_coalitions(n):
        if S == 0 or S == N:
            continue
        row = [0.0] * n
        for i in range(n):
            if S & (1 << i):
                row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-float(game.value(S)))

    bounds: list[tuple[float | None, float | None]]
    if restrict_to_imputation:
        bounds = [(float(game.value(1 << i)), None) for i in range(n)]
    else:
        bounds = [(None, None) for _ in range(n)]

    return PolyhedralSet.from_hrep(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)


@dataclass(frozen=True)
class Core:
    """
    Core as a set-valued object (polyhedron).

    This is a thin wrapper around `core_polyhedron` providing convenience
    methods such as membership tests, sampling, and (small-n) vertex enumeration.

    Parameters
    ----------
    game
        TU game.
    restrict_to_imputation
        If True, use individual-rationality bounds ``x_i >= v({i})`` when building
        the underlying polyhedron.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.geometry import Core
    >>>
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
    >>> C = Core(g)
    >>> C.contains([1.0, 1.0, 1.0])
    True
    >>> C.vertices()
    [[1.0, 1.0, 1.0]]
    """
    game: GameProtocol
    restrict_to_imputation: bool = False

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation of the core.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> C = Core(g)
        >>> C.poly.n_vars
        2
        """
        return core_polyhedron(self.game, restrict_to_imputation=self.restrict_to_imputation)

    def vertices(
        self,
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_players: int = DEFAULT_GEOMETRY_MAX_PLAYERS,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        """
        Enumerate core vertices (small $n$).

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        >>> Core(g).vertices()
        [[1.0, 1.0, 1.0]]
        """
        n = self.game.n_players
        if n > max_players:
            raise NotSupportedError(f"core vertices enumeration is intended for small n (got n={n})")

        return self.poly.extreme_points(tol=tol, max_dim=min(n, int(max_dim)))

    def contains(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Membership test: check whether $x$ lies in the core.

        Parameters
        ----------
        x
            Allocation vector.
        tol
            Numerical tolerance.

        Returns
        -------
        bool
            True if ``x`` satisfies efficiency and all core inequalities.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        >>> C = Core(g)
        >>> C.contains([1.0, 1.0, 1.0])
        True
        >>> C.contains([2.0, 0.0, 0.0])  # violates some coalitional constraints
        False
        """
        return self.poly.contains(x, tol=tol)

    def check(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, top_k: int = 8):
        """
        Return core-membership diagnostics for $x$.

        Notes
        -----
        This method delegates to `tucoop.diagnostics.core_diagnostics.core_diagnostics`.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> Core(g).check([0.5, 0.5]).in_core
        True
        """
        from ..diagnostics.core_diagnostics import core_diagnostics

        return core_diagnostics(self.game, x, tol=tol, top_k=top_k)

    def explain(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, top_k: int = 3) -> list[str]:
        """
        Return a short human-readable explanation of core membership.

        This delegates to :func:`tucoop.diagnostics.core_diagnostics.explain_core_membership`.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> Core(g).explain([0.5, 0.5])[0].startswith("In the core")
        True
        """
        from ..diagnostics.core_diagnostics import explain_core_membership

        return explain_core_membership(self.game, x, tol=tol, top_k=top_k)

    def is_empty(self) -> bool:
        """
        Check whether the core is empty.

        Returns
        -------
        bool
            True if the underlying polyhedron is infeasible.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> empty = Game.from_coalitions(
        ...     n_players=3,
        ...     values={
        ...         (): 0.0,
        ...         (0,): 1.0, (1,): 1.0, (2,): 1.0,
        ...         (0, 1): 2.0, (0, 2): 2.0, (1, 2): 2.0,
        ...         (0, 1, 2): 2.0,
        ...     },
        ... )
        >>> Core(empty).is_empty()
        True
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Return an arbitrary feasible point in the core, if any.

        Returns
        -------
        list[float] | None
            A feasible core allocation, or None if the core is empty.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        >>> x = Core(g).sample_point()
        >>> x is not None
        True
        >>> Core(g).contains(x)  # doctest: +ELLIPSIS
        True
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Chebyshev center of the core polyhedron (if non-empty).

        Returns
        -------
        (x, r) | None
            ``x`` is the center and ``r`` is the radius of the largest Euclidean
            ball contained in the core (as computed by the underlying backend).
            Returns None if the core is empty.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        >>> cc = Core(g).chebyshev_center()
        >>> cc is not None
        True
        >>> x, r = cc  # doctest: +ELLIPSIS
        >>> Core(g).contains(x)  # center is feasible
        True
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self,
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        """
        Enumerate extreme points of the core polyhedron via the underlying
        polyhedral representation.

        This is a more general (but potentially more expensive) alternative to
        `vertices`, delegated to the polyhedral backend.

        Parameters
        ----------
        tol
            Numerical tolerance for feasibility and de-duplication.
        max_dim
            Safety limit for the backend enumeration routine.

        Returns
        -------
        list of list of float
            Extreme points of the core, or an empty list if the core is empty.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        >>> C = Core(g)
        >>> eps = C.extreme_points()
        >>> len(eps) > 0
        True
        >>> all(C.contains(x) for x in eps)
        True
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
        Project the core polyhedron onto a subset of coordinates.

        This is useful for visualization in low dimensions (e.g. 2D/3D plots of
        the core by selecting two or three players).

        Parameters
        ----------
        dims
            Indices of players (coordinates) to keep in the projection.
        tol
            Numerical tolerance for the backend projection routine.
        max_dim
            Safety limit for the backend routine.

        Returns
        -------
        list of list of float
            Vertices of the projected polytope.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import Core
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        >>> C = Core(g)
        >>> # Project core onto players 0 and 1
        >>> proj = C.project((0, 1))
        >>> len(proj) > 0
        True
        """
        return self.poly.project(dims, tol=tol, max_dim=max_dim)


__all__ = [
    "core_polyhedron",
    "Core",
]
