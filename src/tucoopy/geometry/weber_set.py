"""
# Weber set (convex hull of marginal vectors).

The Weber set of a TU game is defined as the convex hull of all marginal payoff
vectors induced by permutations of players.

This module provides:

- functions to compute marginal vectors (exact/enumerative, and sampling-based),
- :class:`WeberSet`, a set-valued object exposing a polyhedral representation for
  small ``n`` via :class:`tucoopy.geometry.PolyhedralSet`.

Notes
-----
The number of permutations is ``n!`` and grows quickly. Exact construction is
intended for small ``n`` only; for larger games, use sampling helpers.

Examples
--------
Compute marginal vectors and build a Weber set for a small game:

>>> from tucoopy import Game
>>> from tucoopy.geometry.weber_set import marginal_vector, WeberSet
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
>>> marginal_vector(g, [0, 1])
[0.0, 1.0]
>>> ws = WeberSet(g)
>>> len(ws.points())
2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import factorial
from random import Random
from typing import Literal, Sequence, overload

from ..base.types import GameProtocol
from ..base.exceptions import BackendError, InvalidParameterError, NotSupportedError
from ..base.config import DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from .polyhedron import PolyhedralSet


def _mask_from_prefix(order: Sequence[int], k: int) -> int:
    """
    Build the coalition mask formed by the first k players of a permutation.

    Parameters
    ----------
    order
        Player order (permutation).
    k
        Prefix length.

    Returns
    -------
    int
        Coalition bitmask containing players ``order[0], ..., order[k-1]``.

    Examples
    --------
    >>> _mask_from_prefix([2, 0, 1], 2)
    5
    """
    mask = 0
    for idx in range(k):
        mask |= 1 << int(order[idx])
    return mask


def marginal_vector(game: GameProtocol, order: Sequence[int]) -> list[float]:
    """
    Marginal contribution vector for a permutation of players.

    Given a permutation (order) $\\pi$ of $N = \\{0,\\dots,n-1\\}$,
    define the growing chain of coalitions

    $$
    S_k = \\{\\pi_1, \\dots, \\pi_k\\}, \\qquad k=0,1,\\dots,n,
    $$

    with $S_0 = \\varnothing$. The **marginal contribution vector**
    $m_\\pi \\in \\mathbb{R}^n$ is defined by

    $$
    m_\\pi[\\pi_k] = v(S_k) - v(S_{k-1}), \\qquad k=1,\\dots,n.
    $$

    Intuition
    ---------
    Walk through the players in the given order. When player i enters, they
    "add" $v(S \\cup \\{i\\}) - v(S)$ to the coalition. Collect these increments
    into a vector.

    Parameters
    ----------
    game
        TU game.
    order
        A permutation of players ``0..n-1``.

    Returns
    -------
    list[float]
        Marginal vector (length ``n_players``).

    Raises
    ------
    InvalidParameterError
        If ``order`` is not a valid permutation of length ``n_players``.

    Examples
    --------
    Minimal 2-player example:

    >>> from tucoopy import Game
    >>> from tucoopy.geometry import marginal_vector
    >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
    >>> marginal_vector(g, [0, 1])
    [0.0, 1.0]
    >>> marginal_vector(g, [1, 0])
    [1.0, 0.0]

    Minimal 3-player "unanimity" game (only grand coalition has value 1):

    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0:0, 1:0, 2:0, 4:0,
    ...     3:0, 5:0, 6:0,
    ...     7:1,
    ... })
    >>> marginal_vector(g, [0, 1, 2])
    [0.0, 0.0, 1.0]
    """
    n = game.n_players
    if len(order) != n:
        raise InvalidParameterError("order must be a permutation of length n_players")
    seen = set(int(i) for i in order)
    if len(seen) != n or any(i < 0 or i >= n for i in seen):
        raise InvalidParameterError("order must be a permutation of players 0..n-1")

    m = [0.0] * n
    prev_v = float(game.value(0))
    for k in range(1, n + 1):
        mask = _mask_from_prefix(order, k)
        vS = float(game.value(mask))
        i = int(order[k - 1])
        m[i] = vS - prev_v
        prev_v = vS
    return m


@overload
def weber_marginal_vectors(
    game: GameProtocol, *, max_permutations: int = 720, return_witness: Literal[False] = False
) -> list[list[float]]: ...


@overload
def weber_marginal_vectors(
    game: GameProtocol, *, max_permutations: int = 720, return_witness: Literal[True]
) -> tuple[list[list[float]], list[tuple[int, ...]]]: ...


def weber_marginal_vectors(
    game: GameProtocol, *, max_permutations: int = 720, return_witness: bool = False
) -> list[list[float]] | tuple[list[list[float]], list[tuple[int, ...]]]:
    """
    Enumerate all marginal contribution vectors (all permutations), if feasible.

    The **Weber set** is the convex hull of marginal vectors, one per permutation.
    This helper returns the generating point set:

    $$
    \\{ m_\\pi : \\pi \\in \\Pi(N) \\}.
    $$

    Since there are $n!$ permutations, full enumeration is only practical
    for small n. This routine raises if $n!$ exceeds ``max_permutations``.

    Parameters
    ----------
    game
        TU game.
    max_permutations
        Safety cap: allow enumeration only if ``n! <= max_permutations``.
        Default 720 corresponds to n<=6.
    return_witness
        If True, also return the permutation ("witness") that generated each
        marginal vector. This is useful for debugging and visualization.

    Returns
    -------
    list[list[float]]
        List of marginal vectors (each of length ``n_players``).
    tuple[list[list[float]], list[tuple[int, ...]]]
        If ``return_witness=True``: ``(points, witnesses)``, where ``witnesses[i]``
        is the permutation that generated ``points[i]``.

    Raises
    ------
    NotSupportedError
        If ``n! > max_permutations``.

    Notes
    -----
    For visualization you often only need the *point cloud* (V-rep). Converting
    the convex hull to an H-representation can be significantly heavier and may
    require additional dependencies.

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.geometry import weber_marginal_vectors
    >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
    >>> weber_marginal_vectors(g)
    [[0.0, 1.0], [1.0, 0.0]]
    >>> pts, perms = weber_marginal_vectors(g, return_witness=True)
    >>> perms
    [(0, 1), (1, 0)]
    """
    n = game.n_players
    total = factorial(n)
    if total > max_permutations:
        raise NotSupportedError(f"Too many permutations (n!={total}); use weber_sample instead")
    out: list[list[float]] = []
    if not return_witness:
        for order in permutations(range(n)):
            out.append(marginal_vector(game, order))
        return out

    witness: list[tuple[int, ...]] = []
    for order in permutations(range(n)):
        order_t = tuple(int(i) for i in order)
        out.append(marginal_vector(game, order_t))
        witness.append(order_t)
    return out, witness


def weber_sample(
    game: GameProtocol,
    *,
    n_samples: int,
    seed: int | None = None,
) -> list[list[float]]:
    """
    Sample marginal vectors by sampling random permutations.

    This is a Monte Carlo approximation to the generating set of the Weber set.
    It returns a list of marginal vectors $m_\\pi$, where each permutation
    $\\pi$ is sampled (approximately) uniformly at random.

    Parameters
    ----------
    game
        TU game.
    n_samples
        Number of random permutations / marginal vectors to generate.
    seed
        Optional RNG seed.

    Returns
    -------
    list[list[float]]
        A list of sampled marginal vectors (each length ``n_players``).

    Raises
    ------
    InvalidParameterError
        If ``n_samples < 1``.

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.geometry import weber_sample
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0:0, 1:0, 2:0, 4:0,
    ...     3:0, 5:0, 6:0,
    ...     7:1,
    ... })
    >>> pts = weber_sample(g, n_samples=5, seed=0)
    >>> len(pts)
    5
    """
    n = game.n_players
    if n_samples < 1:
        raise InvalidParameterError("n_samples must be >= 1")
    rng = Random(seed)
    players = list(range(n))
    out: list[list[float]] = []
    for _ in range(n_samples):
        rng.shuffle(players)
        out.append(marginal_vector(game, players))
    return out


def mean_marginal_vector(game: GameProtocol, *, max_permutations: int = 720) -> list[float]:
    """
    Mean marginal contribution vector across all permutations.

    If all permutations are enumerated, this equals the Shapley value.

    Parameters
    ----------
    game
        TU game.
    max_permutations
        Safety cap: allow enumeration only if ``n! <= max_permutations``.

    Returns
    -------
    list[float]
        Mean marginal vector (length ``n_players``).

    Examples
    --------
    For n=2, the mean marginal vector is easy to compute:

    >>> from tucoopy import Game
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
    >>> mean_marginal_vector(g)
    [0.5, 0.5]
    """
    pts = weber_marginal_vectors(game, max_permutations=max_permutations)
    n = game.n_players
    out = [0.0] * n
    for p in pts:
        for i in range(n):
            out[i] += float(p[i])
    scale = 1.0 / float(len(pts)) if pts else 1.0
    return [float(v) * scale for v in out]


def mean_marginal_sample(
    game: GameProtocol,
    *,
    n_samples: int,
    seed: int | None = None,
) -> list[float]:
    """
    Approximate the mean marginal contribution vector by sampling permutations.

    Parameters
    ----------
    game
        TU game.
    n_samples
        Number of random permutations to sample.
    seed
        Optional RNG seed.

    Returns
    -------
    list[float]
        Approximate mean marginal vector.

    Examples
    --------
    >>> from tucoopy import Game
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0: 0.0,
    ...     1: 1.0, 2: 1.0, 4: 1.0,
    ...     3: 2.0, 5: 2.0, 6: 2.0,
    ...     7: 4.0,
    ... })
    >>> m = mean_marginal_sample(g, n_samples=20, seed=0)
    >>> len(m)
    3
    """
    pts = weber_sample(game, n_samples=n_samples, seed=seed)
    n = game.n_players
    out = [0.0] * n
    for p in pts:
        for i in range(n):
            out[i] += float(p[i])
    scale = 1.0 / float(len(pts)) if pts else 1.0
    return [float(v) * scale for v in out]


def _convex_hull_2d(points: list[tuple[float, float]], *, tol: float) -> list[tuple[float, float]]:
    """
    Compute the convex hull of a point set in 2D (monotone chain).

    Parameters
    ----------
    points
        List of 2D points (x, y).
    tol
        Tolerance used only for de-duplication (not for geometric robustness).

    Returns
    -------
    list[tuple[float, float]]
        Convex hull vertices in counter-clockwise order without repeating the first point.
        If the hull is degenerate, returns one point (singleton) or two points (segment endpoints).

    Examples
    --------
    >>> hull = _convex_hull_2d([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], tol=1e-12)
    >>> len(hull)
    4
    """
    if not points:
        return []

    key_tol = float(tol) if float(tol) > 0 else 1e-12
    uniq: dict[tuple[int, int], tuple[float, float]] = {}
    for x, y in points:
        k = (int(round(float(x) / key_tol)), int(round(float(y) / key_tol)))
        uniq.setdefault(k, (float(x), float(y)))
    pts = sorted(uniq.values())

    if len(pts) <= 2:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        return pts[:1]
    return hull


def _poly_from_hull_2d_in_sum_plane(
    hull: list[tuple[float, float]], *, vN: float, tol: float
) -> PolyhedralSet:
    """
    Build a 3D polyhedron for n=3 from a 2D convex hull in the (x0, x1) chart.

    The Weber set lies in the efficiency plane x0 + x1 + x2 = v(N). We represent
    its convex hull via inequalities in x0,x1 and a single equality for efficiency.

    Examples
    --------
    >>> poly = _poly_from_hull_2d_in_sum_plane([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], vN=1.0, tol=1e-12)
    >>> poly.contains([0.25, 0.25, 0.5])
    True
    """
    A_eq = [[1.0, 1.0, 1.0]]
    b_eq = [float(vN)]

    if len(hull) == 1:
        x0, x1 = hull[0]
        x2 = float(vN) - float(x0) - float(x1)
        bounds = [(float(x0), float(x0)), (float(x1), float(x1)), (float(x2), float(x2))]
        return PolyhedralSet.from_hrep(A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if len(hull) == 2:
        (x0a, x1a), (x0b, x1b) = hull
        dx = float(x0b) - float(x0a)
        dy = float(x1b) - float(x1a)
        # Line equality: (-dy) * x0 + dx * x1 = (-dy) * x0a + dx * x1a
        A_eq2 = A_eq + [[-dy, dx, 0.0]]
        b_eq2 = b_eq + [(-dy) * float(x0a) + dx * float(x1a)]

        # Directional bounds along u = (dx, dy): uÂ·p in [min, max]
        u0, u1 = dx, dy
        t0 = u0 * float(x0a) + u1 * float(x1a)
        t1 = u0 * float(x0b) + u1 * float(x1b)
        tmin, tmax = (t0, t1) if t0 <= t1 else (t1, t0)
        A_ub_seg = [[u0, u1, 0.0], [-u0, -u1, 0.0]]
        b_ub_seg = [tmax + float(tol), -tmin + float(tol)]
        return PolyhedralSet.from_hrep(A_ub=A_ub_seg, b_ub=b_ub_seg, A_eq=A_eq2, b_eq=b_eq2)

    # Polygon hull: for each directed edge p->q in CCW order, keep the left side.
    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    m = len(hull)
    for i in range(m):
        x0p, x1p = hull[i]
        x0q, x1q = hull[(i + 1) % m]
        dx = float(x0q) - float(x0p)
        dy = float(x1q) - float(x1p)
        # dy*x0 - dx*x1 <= dy*x0p - dx*x1p  (equivalent to left-of-edge test)
        A_ub.append([dy, -dx, 0.0])
        b_ub.append(dy * float(x0p) - dx * float(x1p) + float(tol))

    return PolyhedralSet.from_hrep(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)


@dataclass(frozen=True)
class WeberSet:
    """
    Weber set helper (V-representation generator).

    Definition
    ----------
    The **Weber set** is the convex hull of all marginal contribution vectors:

    $$
    W(v) = \\mathrm{conv}\\{ m_\\pi : \\pi \\in \\Pi(N) \\}.
    $$

    Representation in this library
    ------------------------------
    This object provides access to the **generating point set**
    $\\{m_\\pi\\}$ (V-representation) via :meth:`points` and :meth:`sample_points`.

    For ``n_players`` in ``{2, 3}``, this class also provides a lightweight
    H-representation via :attr:`poly`, which is useful for plotting and for
    reusing :class:`~tucoopy.geometry.polyhedron.PolyhedralSet` helpers.

    Parameters
    ----------
    game
        TU game.
    max_permutations
        Safety cap for full enumeration via :meth:`points`.

    Examples
    --------
    For n=3, the exact generator set contains at most ``3! = 6`` points:

    >>> from tucoopy import Game
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0: 0.0,
    ...     1: 1.0, 2: 1.0, 4: 1.0,
    ...     3: 2.0, 5: 2.0, 6: 2.0,
    ...     7: 4.0,
    ... })
    >>> ws = WeberSet(g)
    >>> len(ws.points()) <= 6
    True
    """

    game: GameProtocol
    max_permutations: int = 720
    _poly_cache: PolyhedralSet | None = field(default=None, init=False, repr=False, compare=False)

    def points(self) -> list[list[float]]:
        """
        Return all marginal vectors if the permutation count is small enough.

        Returns
        -------
        list[list[float]]
            All marginal vectors (one per permutation).

        Raises
        ------
    NotSupportedError
        If ``n! > max_permutations``.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import WeberSet
        >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
        >>> WeberSet(g).points()
        [[0.0, 1.0], [1.0, 0.0]]
        """
        return weber_marginal_vectors(self.game, max_permutations=int(self.max_permutations))

    @property
    def poly(self) -> PolyhedralSet:
        """
        Return an H-representation polyhedron for the Weber set (n=2 or n=3).

        For n=2, the Weber set is a line segment on the efficiency line.
        For n=3, the Weber set is a polygon on the efficiency plane and we return a
        halfspace representation (inequalities + the efficiency equality).

        Notes
        -----
        - Implemented only for n=2 and n=3.
        - Requires enumerating all marginal vectors via :meth:`points`, so it is limited
          by ``max_permutations``.
        """
        if self._poly_cache is not None:
            return self._poly_cache

        n = self.game.n_players
        pts = self.points()
        if not pts:
            raise BackendError("Weber set has no points (unexpected)")

        vN = float(self.game.value(self.game.grand_coalition))
        if n == 2:
            xs0 = [float(p[0]) for p in pts]
            x0_min, x0_max = min(xs0), max(xs0)
            A_eq = [[1.0, 1.0]]
            b_eq = [vN]
            bounds = [(x0_min, x0_max), (vN - x0_max, vN - x0_min)]
            poly = PolyhedralSet.from_hrep(A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            object.__setattr__(self, "_poly_cache", poly)
            return poly

        if n == 3:
            pts2 = [(float(p[0]), float(p[1])) for p in pts]
            hull = _convex_hull_2d(pts2, tol=DEFAULT_GEOMETRY_TOL)
            poly = _poly_from_hull_2d_in_sum_plane(hull, vN=vN, tol=DEFAULT_GEOMETRY_TOL)
            object.__setattr__(self, "_poly_cache", poly)
            return poly

        raise NotSupportedError("WeberSet.poly is implemented only for n_players in {2, 3}")

    def sample_points(self, *, n_samples: int, seed: int | None = None) -> list[list[float]]:
        """
        Sample marginal vectors uniformly by sampling random permutations.

        Parameters
        ----------
        n_samples
            Number of sampled permutations.
        seed
            Optional RNG seed.

        Returns
        -------
        list[list[float]]
            Sampled marginal vectors.
        """
        return weber_sample(self.game, n_samples=int(n_samples), seed=seed)

    def mean_marginal(self) -> list[float]:
        """
        Return the mean marginal vector under full enumeration (if feasible).

        Notes
        -----
        When all permutations are enumerated, this equals the Shapley value.
        """
        return mean_marginal_vector(self.game, max_permutations=int(self.max_permutations))

    def mean_marginal_sample(self, *, n_samples: int, seed: int | None = None) -> list[float]:
        """
        Approximate the mean marginal vector by sampling random permutations.
        """
        return mean_marginal_sample(self.game, n_samples=int(n_samples), seed=seed)

    def vertices(
        self,
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_players: int = DEFAULT_GEOMETRY_MAX_PLAYERS,
        max_dim: int = 3,
    ) -> list[list[float]]:
        """
        Enumerate vertices of the Weber set polytope (small n).

        This is available only when :attr:`poly` is available (currently `n=2` or `n=3`).

        Parameters
        ----------
        tol
            Numerical tolerance passed to the underlying polyhedron.
        max_players
            Safety cap (API consistency). This helper is intended for small n.
        max_dim
            Safety limit for vertex enumeration.

        Returns
        -------
        list[list[float]]
            Vertex list (possibly empty if the construction fails).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import WeberSet
        >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
        >>> WeberSet(g).vertices(max_dim=2)
        [[0.0, 1.0], [1.0, 0.0]]
        """
        n = int(self.game.n_players)
        if n > int(max_players):
            raise NotSupportedError(f"WeberSet.vertices is intended for small n (got n={n})")
        return self.poly.extreme_points(tol=float(tol), max_dim=int(max_dim))

    def project(
        self,
        dims: tuple[int, ...] | list[int],
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_dim: int = 3,
    ) -> list[list[float]]:
        """
        Project the Weber set polytope onto selected coordinates (n=2 or n=3).

        Parameters
        ----------
        dims
            Coordinate indices to keep.
        tol
            Numerical tolerance passed to the underlying polyhedron.
        max_dim
            Safety limit used by the underlying enumeration routine.

        Returns
        -------
        list[list[float]]
            Projected points derived from the vertex set.
        """
        return self.poly.project(dims, tol=float(tol), max_dim=int(max_dim))


__all__ = [
    "marginal_vector",
    "weber_marginal_vectors",
    "weber_sample",
    "mean_marginal_vector",
    "mean_marginal_sample",
    "WeberSet",
]
