"""
# Epsilon-core as a polyhedral set and a set-valued wrapper.

This module defines the epsilon-core constraints

$x(S) \\geq v(S) - \\epsilon$ for all non-empty proper coalitions $S$,

plus efficiency. It provides both:

- polyhedral constructors (H-representation), and
- `EpsilonCore`, a convenience wrapper exposing `EpsilonCore.poly`.

Notes
-----
The epsilon-core is a key building block for the least-core and for LP-based
explanations (e.g. "most violated coalition").

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.geometry import EpsilonCore
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> EpsilonCore(g, epsilon=0.0).contains([0.5, 0.5])
True
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import DEFAULT_GEOMETRY_MAX_DIM, DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..base.exceptions import NotSupportedError
from ..base.coalition import all_coalitions
from .polyhedron import PolyhedralSet


def _row_for_coalition(mask: int, n: int) -> list[float]:
    """
    Build the inequality row for a coalition constraint in H-representation.

    We represent the epsilon-core constraints as linear inequalities of the form:

    $$
    A_{ub} x \\leq b_{ub}
    $$

    For a coalition S, the coalitional rationality constraint is:

    $$
    x(S) \\geq v(S) - \\epsilon
    $$

    where $x(S) = \\sum_{i \\in S} x_i$.

    In inequality form:

    $$
    -x(S) \\leq -(v(S) - \\epsilon) = -v(S) + \\epsilon
    $$

    This helper returns the row vector for $-x(S)$, i.e. it contains -1.0 for players
    in $S$ and 0.0 otherwise.

    Parameters
    ----------
    mask
        Coalition bitmask S.
    n
        Number of players.

    Returns
    -------
    list[float]
        Row vector of length n, encoding -x(S).

    Examples
    --------
    Coalition ``S={0,2}`` in a 3-player game:

    >>> _row_for_coalition(0b101, 3)
    [-1.0, 0.0, -1.0]
    """
    row = [0.0] * n
    for i in range(n):
        if mask & (1 << i):
            row[i] = -1.0
    return row


def _epsilon_core_constraints(
    game: GameProtocol,
    epsilon: float,
    *,
    restrict_to_imputation: bool = False,
) -> tuple[
    list[list[float]],
    list[float],
    list[list[float]],
    list[float],
    list[tuple[float | None, float | None]],
]:
    """
    Build the H-representation of the epsilon-core as linear constraints.

    Definition
    ----------
    The **epsilon-core** of a TU game ($v$) is the set of allocations $x$ such that:

    - Efficiency:

        $$\\sum_i x_i = v(N)$$

    - Relaxed coalitional rationality:

    $$
    x(S) \\geq v(S) - \\epsilon \\text{ for all non-empty proper coalitions} S
    $$

    where $x(S) = \\sum_{i \\in S} x_i$ and $\\epsilon \\geq 0$.

    If ``restrict_to_imputation=True``, we also enforce individual rationality:

    $$
    x_i \\geq v(\\{i\\}) \\text{ for all } i,
    $$

    implemented via variable bounds (cleaner than adding separate inequalities).

    Parameters
    ----------
    game
        TU game.
    epsilon
        Relaxation parameter (epsilon). Larger epsilon makes the set larger.
    restrict_to_imputation
        If True, intersect the epsilon-core with individual rationality bounds.

    Returns
    -------
    (A_ub, b_ub, A_eq, b_eq, bounds)
        Constraints suitable for ``PolyhedralSet.from_hrep(...)``:

        - A_ub, b_ub: inequalities A_ub x <= b_ub
        - A_eq, b_eq: equalities  A_eq x  = b_eq
        - bounds: per-coordinate bounds (lo, hi)

    Notes
    -----
    - Coalitional constraints are built for all non-empty proper coalitions
      $S \\subset N, S \\neq \\vanothing.
    - The representation uses only one equality (efficiency).

    Examples
    --------
    Build constraints for a small 2-player game:

    >>> from tucoopy import Game
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
    >>> A_ub, b_ub, A_eq, b_eq, bounds = _epsilon_core_constraints(g, epsilon=0.0)
    >>> (len(A_eq), len(b_eq), len(bounds))
    (1, 1, 2)
    """
    n = game.n_players
    N = game.grand_coalition
    vN = float(game.value(N))
    eps = float(epsilon)

    # Efficiency: sum_i x_i = v(N)
    A_eq = [[1.0] * n]
    b_eq = [vN]

    # Coalitional constraints: -x(S) <= -v(S) + eps
    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    for S in all_coalitions(n):
        if S == 0 or S == N:
            continue
        A_ub.append(_row_for_coalition(S, n))
        b_ub.append(-float(game.value(S)) + eps)

    # Optional individual rationality: x_i >= v({i})
    bounds: list[tuple[float | None, float | None]]
    if restrict_to_imputation:
        bounds = [(float(game.value(1 << i)), None) for i in range(n)]
    else:
        bounds = [(None, None) for _ in range(n)]

    return A_ub, b_ub, A_eq, b_eq, bounds


def epsilon_core_polyhedron(
    game: GameProtocol,
    epsilon: float,
    *,
    restrict_to_imputation: bool = False,
) -> PolyhedralSet:
    """
    Build the epsilon-core as a `PolyhedralSet` (H-representation).

    Parameters
    ----------
    game
        TU game.
    epsilon
        Relaxation parameter epsilon.
    restrict_to_imputation
        If True, also enforce individual rationality bounds x_i >= v({i}).

    Returns
    -------
    PolyhedralSet
        The epsilon-core polyhedron represented as linear equalities/inequalities.

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.geometry.epsilon_core_set import epsilon_core_polyhedron
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
    >>> poly = epsilon_core_polyhedron(g, epsilon=0.0)
    >>> poly.contains([1.0, 1.0, 1.0])
    True
    """
    A_ub, b_ub, A_eq, b_eq, bounds = _epsilon_core_constraints(
        game,
        epsilon,
        restrict_to_imputation=restrict_to_imputation,
    )
    return PolyhedralSet.from_hrep(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)


@dataclass(frozen=True)
class EpsilonCore:
    """
    Epsilon-core polytope as a set-valued object.

    This is a thin wrapper around ``epsilon_core_polyhedron(...)`` that exposes
    convenience methods like containment, sampling, Chebyshev center, projections,
    and (small-$n$) brute-force vertex enumeration.

    Attributes
    ----------
    game : Game
        TU game.
    epsilon : float
        Relaxation parameter epsilon.
    restrict_to_imputation : bool
        If True, intersect with individual rationality bounds x_i >= v({i}).

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.geometry import EpsilonCore
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> C = EpsilonCore(g, epsilon=0.0)
    >>> C.contains([0.5, 0.5])
    True
    """

    game: GameProtocol
    epsilon: float
    restrict_to_imputation: bool = False

    @property
    def poly(self) -> PolyhedralSet:
        """
        The underlying polyhedral representation (H-rep).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).poly.n_vars
        2
        """
        return epsilon_core_polyhedron(
            self.game,
            self.epsilon,
            restrict_to_imputation=self.restrict_to_imputation,
        )

    def contains(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Check whether x lies in the epsilon-core (within tolerance).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).contains([0.5, 0.5])
        True
        """
        return self.poly.contains(x, tol=tol)

    def check(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, top_k: int = 8):
        """
        Return epsilon-core membership diagnostics for ``x``.

        Notes
        -----
        This delegates to :func:`tucoopy.diagnostics.epsilon_core_diagnostics.epsilon_core_diagnostics`.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).check([0.5, 0.5]).in_set
        True
        """
        from ..diagnostics.epsilon_core_diagnostics import epsilon_core_diagnostics

        return epsilon_core_diagnostics(self.game, x, epsilon=float(self.epsilon), tol=tol, top_k=top_k)

    def explain(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, top_k: int = 3) -> list[str]:
        """
        Return a short human-readable explanation of epsilon-core membership.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).explain([0.5, 0.5])[0].startswith("In the epsilon-core")
        True
        """
        d = self.check(x, tol=tol, top_k=top_k)
        lines: list[str] = []
        if not d.efficient:
            lines.append(f"Not efficient: sum(x)={d.sum_x:.6g} but v(N)={d.vN:.6g}.")
        if d.in_set:
            lines.append(f"In the epsilon-core (epsilon={d.epsilon:.6g}, max_excess={d.max_excess:.6g}).")
            return lines
        lines.append(f"Not in the epsilon-core (epsilon={d.epsilon:.6g}, max_excess={d.max_excess:.6g}).")
        if d.violations:
            v0 = d.violations[0]
            lines.append(
                "Most violated coalition: "
                f"S={v0.coalition_mask} players={v0.players} excess={v0.excess:.6g} (v(S)={v0.vS:.6g}, x(S)={v0.xS:.6g})."
            )
        return lines

    def is_empty(self) -> bool:
        """
        Return True if the epsilon-core polytope is empty.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Attempt to find any feasible point in the epsilon-core.

        Returns None if the set is empty or if the backend fails to find a point.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Compute a Chebyshev center of the epsilon-core (if available).

        Returns
        -------
        (x, r) or None
            x is the center point and r is the radius of the largest inscribed ball.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points of the epsilon-core (backend-dependent).

        Notes
        -----
        This method delegates to ``PolyhedralSet.extreme_points`` and may be more
        robust to degeneracy than the brute-force ``vertices`` method, depending
        on the backend implementation.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).extreme_points(max_dim=2)
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
        Project the epsilon-core onto selected coordinates.

        Parameters
        ----------
        dims
            Coordinates to keep (e.g., (0,1) for a 2D projection).
        tol
            Numerical tolerance.
        max_dim
            Safety cap for projection dimension in the backend.

        Returns
        -------
        list[list[float]]
            Extreme points of the projected polytope (backend-dependent).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).project((0,), max_dim=2)
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
        Enumerate vertices of the epsilon-core polytope (small $n$).

        This is intended mainly for visualization and delegates to
        ``PolyhedralSet.extreme_points`` via `poly`.

        Parameters
        ----------
        tol
            Numerical tolerance used for feasibility and de-duplication.
        max_players
            Safety cap. This method is exponential in n and intended for small n.

        Returns
        -------
        list[list[float]]
            List of epsilon-core vertices, or an empty list if the epsilon-core is empty.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import EpsilonCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> EpsilonCore(g, epsilon=0.0).vertices(max_dim=2)
        [[0.0, 1.0], [1.0, 0.0]]
        """
        n = self.game.n_players
        if n > max_players:
            raise NotSupportedError(f"epsilon-core vertices enumeration is intended for small n (got n={n})")
        return self.poly.extreme_points(tol=tol, max_dim=min(n, int(max_dim)))


@dataclass(frozen=True)
class LeastCorePolytope:
    """
    Container for the least-core polytope (small-n visualization output).

    Attributes
    ----------
    epsilon : float
        Least-core epsilon (minimum relaxation that makes the epsilon-core non-empty).
    vertices : list[list[float]]
        Vertices of the least-core polytope (computed via brute-force enumeration).

    Examples
    --------
    >>> lc = LeastCorePolytope(epsilon=0.0, vertices=[[1.0, 1.0, 1.0]])
    >>> lc.epsilon
    0.0
    """

    epsilon: float
    vertices: list[list[float]]


def least_core_polytope(
    game: GameProtocol,
    *,
    restrict_to_imputation: bool = False,
    tol: float = DEFAULT_GEOMETRY_TOL,
    max_players: int = DEFAULT_GEOMETRY_MAX_PLAYERS,
) -> LeastCorePolytope:
    """
    Compute the least-core epsilon and (small-$n$) vertices of the least-core set.

    The **least-core** is the epsilon-core with the smallest epsilon for which
    the epsilon-core is non-empty. This function:

    1. computes least-core epsilon via an LP (SciPy/HiGHS),
    2. enumerates vertices of the resulting epsilon-core (small $n$, brute force).

    Parameters
    ----------
    game
        TU game.
    restrict_to_imputation
        If True, also enforce individual rationality bounds $x_i \\geq v(\\{i\\})$.
    tol
        Numerical tolerance passed to the LP solver and vertex enumeration.
    max_players
        Safety cap for brute-force vertex enumeration.

    Returns
    -------
    LeastCorePolytope
        Object containing (epsilon, vertices).

    Requires
    --------
    SciPy at runtime (install with: ``pip install "tucoopy[lp]"``).

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.geometry.epsilon_core_set import least_core_polytope
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
    >>> lc = least_core_polytope(g)
    >>> lc.epsilon
    0.0
    >>> lc.vertices
    [[1.0, 1.0, 1.0]]
    """
    from .least_core_set import least_core  # local import to keep base lightweight

    lc = least_core(game, tol=tol)
    verts = EpsilonCore(game, lc.epsilon, restrict_to_imputation=restrict_to_imputation).vertices(
        tol=tol, max_players=max_players
    )
    return LeastCorePolytope(epsilon=float(lc.epsilon), vertices=verts)
