"""
# Least-core as a set-valued object.

The least-core is the epsilon-core with the smallest feasible ``epsilon*``. This
module defines `LeastCore` as a thin wrapper that:

- computes/stores the optimal ``epsilon*`` (when an LP backend is available),
- exposes a corresponding `tucoop.geometry.EpsilonCore` / polyhedron, and
- provides convenience methods for membership and small-n geometry operations.

Examples
--------
>>> from tucoop import Game
>>> from tucoop.geometry import LeastCore
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> lc = LeastCore(g)
>>> lc.contains([0.5, 0.5])  # doctest: +SKIP
True
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import DEFAULT_GEOMETRY_MAX_DIM, DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from .epsilon_core_set import EpsilonCore
from .polyhedron import PolyhedralSet


@dataclass
class LeastCore:
    """
    Least-core as a **set-valued** object.

    Background
    ----------
    For a TU game, the (standard) core is the set of efficient allocations that
    satisfy all coalitional constraints:

    $$
    \\sum_i x_i = v(N), \\qquad x(S) \\ge v(S) \\; \\text{for all } \\varnothing \\ne S \\subsetneq N.
    $$

    When the core is empty, a common relaxation is the **epsilon-core**:

    $$
    \\sum_i x_i = v(N), \\qquad x(S) \\ge v(S) - \\varepsilon \\; \\text{for all } \\varnothing \\ne S \\subsetneq N.
    $$

    The **least-core** is the epsilon-core at the *smallest* relaxation level
    $\\varepsilon^*$ for which the epsilon-core becomes non-empty:

    $$
    \\varepsilon^* = \\min\\{\\varepsilon \\ge 0 : \\text{epsilon-core}(\\varepsilon) \\ne \\varnothing\\}.
    $$

    This class is a thin wrapper around ``EpsilonCore(game, epsilon*)``, computed
    lazily on first access.

    Lazy evaluation / dependencies
    ------------------------------
    The value ``epsilon`` is computed on demand via the LP-based solver
    ``tucoop.solutions.least_core.least_core`` (which requires SciPy/HiGHS at runtime,
    typically via ``pip install "tucoop[lp]"``). Once computed, it is cached in
    ``_epsilon``.

    Parameters
    ----------
    game
        TU game.
    restrict_to_imputation
        If True, additionally enforce individual rationality ``x_i >= v({i})``
        (handled as bounds) when forming the underlying polyhedron.
        This selects the least-core *within* the imputation set.
    tol
        Numerical tolerance forwarded to the LP solver and set-membership checks.
    _epsilon
        Optional cached value for ``epsilon`` (internal). If provided, the LP solve
        is skipped.

    Attributes
    ----------
    epsilon
        The least-core value ``epsilon*`` (computed lazily and cached).
    epsilon_core
        The epsilon-core object at ``epsilon*``.
    poly
        The underlying polyhedral representation (``PolyhedralSet``) of the least-core.

    Notes
    -----
    - ``LeastCore`` is set-valued. If you need a *single-valued selector*, use
      a routine like ``least_core_point`` (e.g., Chebyshev center) on top of this set.
    - The method ``vertices`` enumerates vertices using a brute-force routine intended
      for small n (visualization).

    Examples
    --------
    Minimal example (may have empty core; least-core always exists for TU games
    under this LP formulation):

    >>> from tucoop import Game
    >>> from tucoop.geometry import LeastCore
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
    >>> lc = LeastCore(g)
    >>> eps = lc.epsilon
    >>> eps
    0.0
    >>> lc.sample_point()  # one least-core point (here also core point)
    [1.0, 1.0, 1.0]
    """

    game: GameProtocol
    restrict_to_imputation: bool = False
    tol: float = DEFAULT_GEOMETRY_TOL
    _epsilon: float | None = None

    @property
    def epsilon(self) -> float:
        """
        Least-core value ``epsilon*`` (computed lazily).

        Notes
        -----
        This requires an LP backend (typically SciPy/HiGHS via ``tucoop[lp]``).

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).epsilon  # doctest: +SKIP
        0.0
        """
        if self._epsilon is None:
            from ..solutions.least_core import least_core_epsilon_star

            self._epsilon = float(least_core_epsilon_star(self.game, tol=float(self.tol)))
        return float(self._epsilon)

    @property
    def epsilon_core(self) -> EpsilonCore:
        """
        Epsilon-core object at ``epsilon*``.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> lc = LeastCore(g)
        >>> lc.epsilon_core.epsilon  # doctest: +SKIP
        0.0
        """
        return EpsilonCore(self.game, self.epsilon, restrict_to_imputation=self.restrict_to_imputation)

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation of the least-core.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).poly.n_vars  # doctest: +SKIP
        2
        """
        return self.epsilon_core.poly

    def contains(self, x: list[float], *, tol: float | None = None) -> bool:
        """
        Check whether x lies in the least-core (within tolerance).

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).contains([0.5, 0.5])  # doctest: +SKIP
        True
        """
        t = float(self.tol if tol is None else tol)
        return self.epsilon_core.contains(x, tol=t)

    def check(self, x: list[float], *, tol: float | None = None, top_k: int = 8):
        """
        Return least-core membership diagnostics for $x$.

        Notes
        -----
        This delegates to `tucoop.diagnostics.least_core_diagnostics.least_core_diagnostics`.
        If the LP backend is unavailable, the returned diagnostics will have
        ``available=False``.
        """
        from ..diagnostics.least_core_diagnostics import least_core_diagnostics

        t = float(self.tol if tol is None else tol)
        return least_core_diagnostics(self.game, x, tol=t, top_k=top_k)

    def explain(self, x: list[float], *, tol: float | None = None, top_k: int = 3) -> list[str]:
        """
        Return a short human-readable explanation of least-core membership.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).explain([0.5, 0.5])[0]  # doctest: +SKIP
        'In the least-core (epsilon*=0).'
        """
        d = self.check(x, tol=tol, top_k=top_k)
        if not d.available:
            return [f"Least-core diagnostics unavailable: {d.reason}"]
        if d.epsilon_star is None or d.epsilon_core is None:
            return ["Least-core diagnostics unavailable (missing epsilon*)."]
        if d.epsilon_core.in_set:
            return [f"In the least-core (epsilon*={d.epsilon_star:.6g})."]
        return [f"Not in the least-core (epsilon*={d.epsilon_star:.6g}, max_excess={d.epsilon_core.max_excess:.6g})."]

    def is_empty(self) -> bool:
        """
        Return True if the least-core polytope is empty.

        Notes
        -----
        This delegates to the underlying epsilon-core polyhedron. Requires an LP backend.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Attempt to sample one point from the least-core.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Compute the Chebyshev center of the least-core polytope.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points of the least-core (small dimension).

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).extreme_points(max_dim=2)  # doctest: +SKIP
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
        Project the least-core to selected coordinates.

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).project((0,), max_dim=2)  # doctest: +SKIP
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
        # Use the epsilon-core vertex enumerator (brute-force, small n) for consistency.
        """
        Enumerate vertices of the least-core polytope (small dimension).

        Examples
        --------
        >>> from tucoop import Game
        >>> from tucoop.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).vertices(max_dim=2)  # doctest: +SKIP
        [[0.0, 1.0], [1.0, 0.0]]
        """
        return self.epsilon_core.vertices(tol=tol, max_players=max_players, max_dim=max_dim)


__all__ = ["LeastCore"]
