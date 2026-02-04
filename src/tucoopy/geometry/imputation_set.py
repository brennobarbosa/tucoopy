"""
# Imputation and pre-imputation sets.

This module provides polyhedral representations of:

- the **pre-imputation set** (efficiency only), and
- the **imputation set** (efficiency + individual rationality).

The imputation set is an intersection of an affine hyperplane with box
constraints, and is therefore polyhedral.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.geometry import ImputationSet
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> ImputationSet(g).contains([0.5, 0.5])
True
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import (
    DEFAULT_GEOMETRY_MAX_DIM,
    DEFAULT_GEOMETRY_MAX_PLAYERS,
    DEFAULT_GEOMETRY_TOL,
    DEFAULT_IMPUTATION_SAMPLE_TOL,
)
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError, NotSupportedError
from .polyhedron import PolyhedralSet


def imputation_lower_bounds(game: GameProtocol) -> list[float]:
    """
    Compute the individual-rationality lower bounds for the imputation set.

    In a TU cooperative game with characteristic function $v$,
    the imputation set requires *individual rationality*:

    $$
        x_i \\ge v(\\{i\\}), \\quad i=1,\\dots,n.
    $$

    Define the lower bounds

    $$
        l_i := v(\\{i\\}).
    $$

    This function returns the vector $l = (l_1,\\dots,l_n)$.

    Parameters
    ----------
    game
        TU game.

    Returns
    -------
    list[float]
        The lower bounds ``[l_0, ..., l_{n-1}]`` with ``l_i = v({i})``.

    Examples
    --------
    Minimal runnable example using a tiny dummy game (3 players)::

        >>> class _G:
        ...     n_players = 3
        ...     grand_coalition = (1 << 3) - 1
        ...     def value(self, S: int) -> float:
        ...         # singleton values: v({0})=1, v({1})=2, v({2})=0.5
        ...         return {1: 1.0, 2: 2.0, 4: 0.5}.get(S, 0.0)
        ...
        >>> imputation_lower_bounds(_G())
        [1.0, 2.0, 0.5]

    Notes
    -----
    - Coalitions are encoded as bitmasks: ``{i}`` is ``1 << i``.
    - The returned bounds are floats.
    """
    return [float(game.value(1 << i)) for i in range(game.n_players)]


def is_in_imputation_set(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
    """
    Test whether an allocation belongs to the imputation set.

    The **imputation set** for a TU game $v$ is the set of allocations
    that are:

    - **Efficient** (feasible for the grand coalition):

    $$
          \\sum_{i=1}^n x_i = v(N)
    $$

    - **Individually rational**:

    $$
          x_i \\ge v(\\{i\\}), \\quad i=1,\\dots,n.
    $$

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector (length ``n_players``).
    tol
        Numerical tolerance used in comparisons. Efficiency is checked with
        ``abs(sum(x) - v(N)) <= tol`` and individual rationality with
        ``x_i + tol >= v({i})``.

    Returns
    -------
    bool
        ``True`` if ``x`` is efficient and individually rational; otherwise ``False``.

    Examples
    --------
    A 2-player dummy game where $v(\\{0\\})=1$, $v(\\{1\\})=2$, $v(N)=5$::

        >>> class _G:
        ...     n_players = 2
        ...     grand_coalition = 3
        ...     def value(self, S: int) -> float:
        ...         return {1: 1.0, 2: 2.0, 3: 5.0}.get(S, 0.0)
        ...
        >>> is_in_imputation_set(_G(), [1.0, 4.0])
        True
        >>> is_in_imputation_set(_G(), [0.9, 4.1])  # violates x0 >= 1
        False
        >>> is_in_imputation_set(_G(), [1.0, 3.9])  # not efficient (sum != 5)
        False

    Notes
    -----
    This is a lightweight membership check. For richer diagnostics, use
    `ImputationSet.check` / `ImputationSet.explain`.
    """
    if len(x) != game.n_players:
        return False
    vN = float(game.value(game.grand_coalition))
    if abs(sum(x) - vN) > tol:
        return False
    l = imputation_lower_bounds(game)
    for i in range(game.n_players):
        if float(x[i]) + tol < l[i]:
            return False
    return True


def _project_to_simplex(y: list[float], r: float) -> list[float]:
    """
    Project a vector onto the scaled probability simplex.

    This computes the Euclidean projection of $y$ onto the simplex

    $$
        \\Delta(r) := \\{ z \\in \\mathbb{R}^n : z_i \\ge 0, \\ \\sum_{i=1}^n z_i = r \\},
    $$

    i.e. it returns

    $$
        \\operatorname{argmin}_{z \\in \\Delta(r)} \\ \\|z - y\\|_2^2.
    $$

    Parameters
    ----------
    y
        Input vector in $\\mathbb{R}^n`.
    r
        Target sum (must be ``>= 0`` for a non-empty simplex). If ``r == 0``,
        the projection is the all-zeros vector.

    Returns
    -------
    list[float]
        The projected vector ``z``.

    Examples
    --------
    Project onto the standard simplex (sum = 1)::

        >>> z = _project_to_simplex([0.2, -0.3, 1.7], 1.0)
        >>> round(sum(z), 10)
        1.0
        >>> all(v >= -1e-12 for v in z)
        True

    If ``r = 0``, the only feasible point is zero::

        >>> _project_to_simplex([1.0, 2.0, 3.0], 0.0)
        [0.0, 0.0, 0.0]

    Notes
    -----
    - Implementation follows the classic sort-and-threshold method.
    - Time complexity: $O(n \\log n)$ due to sorting.
    - This is an internal helper used by `project_to_imputation`.
    """
    n = len(y)
    if n == 0:
        return []
    if r == 0.0:
        return [0.0] * n

    u = sorted((float(v) for v in y), reverse=True)
    csum = 0.0
    rho = -1
    theta = 0.0
    for j in range(n):
        csum += u[j]
        t = (csum - r) / float(j + 1)
        if u[j] - t > 0:
            rho = j
            theta = t
    if rho == -1:
        return [r / float(n)] * n

    return [max(float(v) - theta, 0.0) for v in y]


@dataclass(frozen=True)
class ImputationProjection:
    """
    Output of `project_to_imputation`.

    Attributes
    ----------
    x
        Projected allocation (same dimension as the input vector).
    feasible
        ``False`` if the imputation set is empty, i.e.

        $$
            \\sum_{i=1}^n v(\\{i\\}) > v(N).
    $$

        In this case, ``x`` is returned unchanged (a copy of the input).

    Examples
    --------
    >>> p = ImputationProjection(x=[0.5, 0.5], feasible=True)
    >>> p.feasible
    True
    """
    x: list[float]
    feasible: bool


def preimputation_polyhedron(game: GameProtocol) -> PolyhedralSet:
    """
    Return the pre-imputation set as a polyhedral (affine) set.

    The **pre-imputation set** consists of allocations that satisfy only
    **efficiency**:

    $$
        \\sum_{i=1}^n x_i = v(N),
    $$

    with no lower bounds $x_i \\ge v(\\{i\\})$.

    Parameters
    ----------
    game
        TU game.

    Returns
    -------
    PolyhedralSet
        H-representation of the affine set ``sum(x)=v(N)`` with free bounds.

    Examples
    --------
    For a 3-player game with ``v(N)=10`` , the set is the plane ``x0+x1+x_2=10``::

        >>> class _G:
        ...     n_players = 3
        ...     grand_coalition = 7
        ...     def value(self, S: int) -> float:
        ...         return 10.0 if S == 7 else 0.0
        ...
        >>> poly = preimputation_polyhedron(_G())
        >>> poly.contains([3.0, 3.0, 4.0])
        True
        >>> poly.contains([3.0, 3.0, 3.9])
        False
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    return PolyhedralSet.from_hrep(A_eq=[[1.0] * n], b_eq=[vN], bounds=[(None, None) for _ in range(n)])


def imputation_polyhedron(game: GameProtocol) -> PolyhedralSet:
    """
    Return the imputation set as a polyhedral set (H-representation).

    The **imputation set** is the intersection of the efficiency hyperplane
    with the individual-rationality halfspaces:

    $$
        \\left\\{ x : \\sum_{i=1}^n x_i = v(N),\\ x_i \\ge v(\\{i\\}) \\right\\}.
    $$

    Parameters
    ----------
    game
        TU game.

    Returns
    -------
    PolyhedralSet
        H-representation using a single equality constraint and per-player
        lower bounds.

    Examples
    --------
    A 2-player game with ``v(N)=5`` and singleton values ``(1,2)``::

        >>> class _G:
        ...     n_players = 2
        ...     grand_coalition = 3
        ...     def value(self, S: int) -> float:
        ...         return {1: 1.0, 2: 2.0, 3: 5.0}.get(S, 0.0)
        ...
        >>> poly = imputation_polyhedron(_G())
        >>> poly.contains([1.0, 4.0])
        True
        >>> poly.contains([0.5, 4.5])  # violates x0 >= 1
        False
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    bounds = [(float(game.value(1 << i)), None) for i in range(n)]
    return PolyhedralSet.from_hrep(A_eq=[[1.0] * n], b_eq=[vN], bounds=bounds)


@dataclass(frozen=True)
class PreImputationSet:
    """
    Pre-imputation set wrapper.

    This is a convenience wrapper around `preimputation_polyhedron`
    exposing common geometric operations.

    Parameters
    ----------
    game
        TU game.

    Examples
    --------
    ::

        >>> class _G:
        ...     n_players = 3
        ...     grand_coalition = 7
        ...     def value(self, S: int) -> float:
        ...         return 10.0 if S == 7 else 0.0
        ...
        >>> P = PreImputationSet(_G())
        >>> P.contains([1.0, 2.0, 7.0])
        True
    """

    game: GameProtocol

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation (efficiency only).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).poly.n_vars
        2
        """
        return preimputation_polyhedron(self.game)

    def contains(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Test membership in the pre-imputation set.

        Parameters
        ----------
        x
            Allocation vector.
        tol
            Numerical tolerance.

        Returns
        -------
        bool
            ``True`` if ``sum(x) = v(N)`` within tolerance.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).contains([0.25, 0.75])
        True
        """
        return self.poly.contains(x, tol=tol)

    def check(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL):
        """
        Run allocation diagnostics for $x$.

        This is a convenience hook that delegates to
        `tucoopy.diagnostics.checks.check_allocation`.

        Parameters
        ----------
        x
            Allocation vector.
        tol
            Numerical tolerance.

        Returns
        -------
        Any
            The diagnostics object returned by ``check_allocation``.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).check([0.5, 0.5]).efficient
        True
        """
        from ..diagnostics.allocation_diagnostics import check_allocation

        return check_allocation(self.game, x, tol=tol)

    def is_empty(self) -> bool:
        """
        Return whether the pre-imputation set is empty.

        Notes
        -----
        For standard TU games, $\\sum_{i=1}^n x_i = v(N)$$ with free bounds is non-empty.
        Emptiness can occur only if the underlying polyhedral machinery
        considers the constraint system infeasible (e.g. NaNs).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Try to sample one point from the set.

        Returns
        -------
        list[float] | None
            A feasible point, or ``None`` if sampling fails.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Compute the Chebyshev center (if defined by the backend).

        Returns
        -------
        (center, radius) or None

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points (when the set is bounded in the projected space).

        Parameters
        ----------
        tol
            Numerical tolerance.
        max_dim
            Maximum dimension for vertex enumeration (backend-dependent safeguard).

        Returns
        -------
        list[list[float]]
            List of vertices/extreme points.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).extreme_points(max_dim=2)
        []
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
        Project the set onto a subset of coordinates.

        Parameters
        ----------
        dims
            Indices of coordinates to keep (e.g. ``(0, 1)``).
        tol
            Numerical tolerance.
        max_dim
            Maximum dimension for vertex enumeration.

        Returns
        -------
        list[list[float]]
            Vertices of the projected polytope (when representable).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import PreImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> PreImputationSet(g).project((0,), max_dim=2)
        []
        """
        return self.poly.project(dims, tol=tol, max_dim=max_dim)


@dataclass(frozen=True)
class ImputationSet:
    """
    Imputation set wrapper (efficiency + individual rationality).

    This wrapper exposes both polyhedral operations and diagnostics specialized
    for imputation membership.

    Parameters
    ----------
    game
        TU game.

    Examples
    --------
    ::

        >>> class _G:
        ...     n_players = 2
        ...     grand_coalition = 3
        ...     def value(self, S: int) -> float:
        ...         return {1: 1.0, 2: 2.0, 3: 5.0}.get(S, 0.0)
        ...
        >>> I = ImputationSet(_G())
        >>> I.contains([1.0, 4.0])
        True
        >>> I.explain([0.5, 4.5])[0].startswith("Violates")
        True
    """

    game: GameProtocol

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation (efficiency + IR bounds).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).poly.n_vars
        2
        """
        return imputation_polyhedron(self.game)

    def contains(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Test membership via the polyhedral representation.

        Parameters
        ----------
        x
            Allocation vector.
        tol
            Numerical tolerance.

        Returns
        -------
        bool
            ``True`` if ``x`` is in the imputation set.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).contains([0.5, 0.5])
        True
        """
        return self.poly.contains(x, tol=tol)

    def check(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL):
        """
        Compute imputation-set diagnostics for $x$.

        Delegates to `tucoopy.diagnostics.imputation_diagnostics.imputation_diagnostics`.

        Parameters
        ----------
        x
            Allocation vector.
        tol
            Numerical tolerance.

        Returns
        -------
        Any
            Diagnostics object with fields such as ``in_set``, ``efficient``,
            ``sum_x``, ``vN`` and potential IR violations.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).check([0.5, 0.5]).in_set
        True
        """
        from ..diagnostics.imputation_diagnostics import imputation_diagnostics

        return imputation_diagnostics(self.game, x, tol=tol)

    def explain(self, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> list[str]:
        """
        Produce a short human-readable explanation of membership.

        Parameters
        ----------
        x
            Allocation vector.
        tol
            Numerical tolerance.

        Returns
        -------
        list[str]
            A list of message lines. Empty list is avoided; when ``x`` is in the
            set, a single affirmative line is returned.

        Examples
        --------
        ::

            >>> class _G:
            ...     n_players = 2
            ...     grand_coalition = 3
            ...     def value(self, S: int) -> float:
            ...         return {1: 1.0, 2: 2.0, 3: 5.0}.get(S, 0.0)
            ...
            >>> ImputationSet(_G()).explain([1.0, 4.0])
            ['In the imputation set.']
        """
        d = self.check(x, tol=tol)
        lines: list[str] = []
        if not d.efficient:
            lines.append(f"Not efficient: sum(x)={d.sum_x:.6g} but v(N)={d.vN:.6g}.")
        if d.violations:
            v0 = d.violations[0]
            lines.append(f"Violates individual rationality: x[{v0.player}]={v0.value:.6g} < {v0.bound:.6g}.")
        if not lines and d.in_set:
            lines.append("In the imputation set.")
        return lines

    def is_empty(self) -> bool:
        """
        Return whether the imputation set is empty.

        Notes
        -----
        The imputation set is empty iff

        $$
            \\sum_{i=1}^n v(\\{i\\}) > v(N).
        $$

        (subject to numerical tolerance inside the polyhedral backend).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Try to sample one point from the imputation set.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Compute the Chebyshev center (if supported by the backend).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points (when representable).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).extreme_points(max_dim=2)
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
        Project the set onto selected coordinates (returns projected vertices when representable).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import ImputationSet
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> ImputationSet(g).project((0,), max_dim=2)
        [[0.0], [1.0]]
        """
        return self.poly.project(dims, tol=tol, max_dim=max_dim)

    def vertices(
        self,
        *,
        tol: float = DEFAULT_IMPUTATION_SAMPLE_TOL,
        max_players: int = DEFAULT_GEOMETRY_MAX_PLAYERS,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        """
        Return the vertices of the imputation set (closed form).

        The imputation set is a **shifted simplex**:

        $$
        \\left\\{ x : \\sum_{i=1}^n x_i = v(N),\\ x_i \\ge v(\\{i\\}) \\right\\}.
        $$

        Let

        $$
        l_i = v(\\{i\\}), \\qquad r = v(N) - \\sum_{i=1}^n l_i.
        $$

        - If $r > 0$, the vertices are $l + r e_i$ for $i=0,\\ldots,n-1$.
        - If $r = 0$, the set is the single point $l$.
        - If $r < 0$, the set is empty.

        Parameters
        ----------
        tol
            Tolerance used to decide whether ``r`` is negative/zero.
        max_players
            Guardrail: this closed-form enumeration is intended for small ``n``.
        max_dim
            Present for API consistency with other geometric sets; not used here.

        Returns
        -------
        list[list[float]]
            Vertices of the imputation set, or ``[]`` if empty.

        Examples
        --------
        3 players, singleton values (1,2,0) and v(N)=10 => r=7, vertices are l+7e_i::

            >>> class _G:
            ...     n_players = 3
            ...     grand_coalition = 7
            ...     def value(self, S: int) -> float:
            ...         return {1: 1.0, 2: 2.0, 4: 0.0, 7: 10.0}.get(S, 0.0)
            ...
            >>> I = ImputationSet(_G())
            >>> I.vertices()
            [[8.0, 2.0, 0.0], [1.0, 9.0, 0.0], [1.0, 2.0, 7.0]]
        """
        _ = max_dim  # part of the shared vertices(...) signature; not used here
        game = self.game
        n = game.n_players
        if n > max_players:
            raise NotSupportedError(f"imputation vertices enumeration is intended for small n (got n={n})")
        vN = float(game.value(game.grand_coalition))
        l = imputation_lower_bounds(game)
        r = vN - sum(l)
        if r < -tol:
            return []
        if abs(r) <= tol:
            return [[float(v) for v in l]]

        verts: list[list[float]] = []
        for i in range(n):
            x = [float(v) for v in l]
            x[i] += float(r)
            verts.append(x)
        return verts


def project_to_imputation(
    game: GameProtocol, x_hat: list[float], *, tol: float = DEFAULT_IMPUTATION_SAMPLE_TOL
) -> ImputationProjection:
    """
    Compute the Euclidean projection onto the imputation set.

    The imputation set is

    $$
    \\mathcal{I}(v) =
    \\left\\{ x : \\sum_{i=1}^n x_i = v(N),\\ x_i \\ge v(\\{i\\}) \\right\\}.
    $$

    This routine returns the closest point (in Euclidean norm) in
    $\\mathcal{I}(v)` to a given vector $\\hat{x} \\in \\mathbb{R}^n`.

    Algorithm
    ---------
    Let ``l_i = v({i})`` and ``r = v(N) - sum_i l_i``.

    1. Shift: ``y = x_hat - l``.
    2. Project ``y`` onto the simplex ``{z: z>=0, sum(z)=r}``.
    3. Shift back: ``x = l + z``.

    Parameters
    ----------
    game
        TU game.
    x_hat
        Arbitrary vector in $\\mathbb{R}^n` (must have length ``n_players``).
    tol
        Numerical tolerance used to decide emptiness / degeneracy of the imputation set.

    Returns
    -------
    ImputationProjection
        ``x`` is the projected allocation. If the imputation set is empty,
        returns ``feasible=False`` and ``x`` is a copy of ``x_hat``.

    Examples
    --------
    2 players, singleton bounds ``(1,2)`` and ``v(N)=5`` => imputation is segment between
    ``(3,2)`` and ``(1,4)``. Projecting ``[10, -10]`` lands at the nearest endpoint::

        >>> class _G:
        ...     n_players = 2
        ...     grand_coalition = 3
        ...     def value(self, S: int) -> float:
        ...         return {1: 1.0, 2: 2.0, 3: 5.0}.get(S, 0.0)
        ...
        >>> proj = project_to_imputation(_G(), [10.0, -10.0])
        >>> proj.feasible
        True
        >>> round(sum(proj.x), 10)
        5.0
        >>> proj.x[0] >= 1.0 and proj.x[1] >= 2.0
        True

    Empty imputation set example (``sum singletons > v(N)``)::

        >>> class _Bad:
        ...     n_players = 2
        ...     grand_coalition = 3
        ...     def value(self, S: int) -> float:
        ...         return {1: 5.0, 2: 5.0, 3: 3.0}.get(S, 0.0)
        ...
        >>> proj = project_to_imputation(_Bad(), [0.0, 0.0])
        >>> proj.feasible
        False

    Notes
    -----
    - This is a *projection* in $\\ell_2$. It does not guarantee any
      game-theoretic property beyond imputation feasibility.
    - If $r$ is numerically zero, the imputation set collapses to the
      single point $l$.
    """
    n = game.n_players
    if len(x_hat) != n:
        raise InvalidParameterError("x_hat must have length n_players")

    vN = float(game.value(game.grand_coalition))
    l = imputation_lower_bounds(game)
    r = vN - sum(l)
    if r < -tol:
        return ImputationProjection(x=[float(v) for v in x_hat], feasible=False)
    if abs(r) <= tol:
        return ImputationProjection(x=[float(v) for v in l], feasible=True)

    y = [float(x_hat[i]) - l[i] for i in range(n)]
    z = _project_to_simplex(y, r)
    x = [l[i] + z[i] for i in range(n)]
    return ImputationProjection(x=x, feasible=True)
