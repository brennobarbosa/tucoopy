"""
# Polyhedral sets (H-representation) for geometry modules.

This module defines `PolyhedralSet`, a small wrapper around linear
constraints in the form:

- inequalities: $A_{ub} x \\leq b_{ub}$
- equalities: $A_{eq} x = b_{eq}$
- variable bounds: $l_i \\leq x_i \\leq u_i$

It provides core operations used across ``tucoop.geometry``:

- feasibility checks and extracting a sample point (LP),
- membership tests,
- Chebyshev center (LP),
- (small-$n$) vertex enumeration and projection.

Notes
-----
This class is intentionally lightweight and relies on the configured LP backend.
For large-dimensional polytopes, vertex enumeration is not feasible; helpers in
this module are explicitly guarded by ``max_dim`` limits.

Examples
--------
A 2D triangle (simplex): x>=0, y>=0, x+y<=1.

>>> from tucoop.geometry import PolyhedralSet
>>> P = PolyhedralSet.from_hrep(
...     A_ub=[[-1, 0], [0, -1], [1, 1]],
...     b_ub=[0, 0, 1],
... )
>>> P.extreme_points()
[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import isfinite, sqrt
import random
from typing import Any, Sequence

from ..base.config import (
    DEFAULT_GEOMETRY_MAX_DIM,
    DEFAULT_GEOMETRY_TOL,
    DEFAULT_HIT_AND_RUN_BURN_IN,
    DEFAULT_HIT_AND_RUN_THINNING,
    DEFAULT_HIT_AND_RUN_TOL,
)
from ..base.exceptions import BackendError, InvalidParameterError, NotSupportedError
from ..backends.lp import linprog_solve
from ._linalg import solve_linear_system


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Dot product helper with explicit float casts.

    Parameters
    ----------
    a, b
        Numeric vectors of the same length.

    Returns
    -------
    float
        Inner product sum_i a_i b_i.

    Examples
    --------
    >>> _dot([1.0, 2.0], [3.0, 4.0])
    11.0
    """
    return float(sum(float(ai) * float(bi) for ai, bi in zip(a, b)))


def _row_norm(row: Sequence[float]) -> float:
    """
    Euclidean norm of a row vector.

    Used by the Chebyshev center formulation to convert halfspaces
    $a^T x \\leq b$ into $a^T x + \\|a\\| r <\\leq b$.

    Parameters
    ----------
    row
        Row vector.

    Returns
    -------
    float
        sqrt(sum_i row_i^2).

    Examples
    --------
    >>> _row_norm([3.0, 4.0])
    5.0
    """
    return sqrt(sum(float(v) * float(v) for v in row))


def _mat_vec(A: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    # A: m x n, x: n x 1  -> m x 1
    return [_dot(row, x) for row in A]


def _mat_mat(
    A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]
) -> list[list[float]]:
    # A: m x n, B: n x k  -> m x k
    if not A:
        return []
    n = len(A[0])
    if not B:
        return [[] for _ in range(len(A))]
    k = len(B[0])
    out = [[0.0] * k for _ in range(len(A))]
    for i, row in enumerate(A):
        for j in range(k):
            s = 0.0
            for t in range(n):
                s += float(row[t]) * float(B[t][j])
            out[i][j] = float(s)
    return out


def _as_bounds(
    bounds: Sequence[tuple[float | None, float | None]] | None, n: int
) -> list[tuple[float | None, float | None]]:
    """
    Normalize bounds into an explicit list of length $n$.

    Parameters
    ----------
    bounds
        Sequence of (lb, ub) pairs or None. Each entry may be None to denote
        an open side.
    n
        Number of variables.

    Returns
    -------
    list[tuple[float | None, float | None]]
        Bounds with floats (when present) and correct length.

    Raises
    ------
    InvalidParameterError
        If bounds is not None and its length differs from n.

    Examples
    --------
    >>> _as_bounds(None, 2)
    [(None, None), (None, None)]
    >>> _as_bounds([(0.0, None), (None, 1.0)], 2)
    [(0.0, None), (None, 1.0)]
    """
    if bounds is None:
        return [(None, None) for _ in range(n)]
    out = list(bounds)
    if len(out) != n:
        raise InvalidParameterError("bounds must have one (lb, ub) pair per variable")
    return [
        (lb if lb is None else float(lb), ub if ub is None else float(ub))
        for lb, ub in out
    ]


def _quantize_key(x: Sequence[float], *, tol: float) -> tuple[int, ...] | None:
    """
    Quantize coordinates to a tolerance-based integer key for de-duplication.

    Returns None if x contains NaN/inf.

    Examples
    --------
    >>> _quantize_key([0.0, 0.49], tol=0.5)
    (0, 1)
    >>> _quantize_key([float('nan')], tol=1e-6) is None
    True
    """
    t = float(tol)
    if t <= 0.0:
        t = 1e-12
    scale = 1.0 / t
    out: list[int] = []
    for v in x:
        fv = float(v)
        if not isfinite(fv):
            return None
        out.append(int(round(fv * scale)))
    return tuple(out)

def _matrix_rank(A: Sequence[Sequence[float]], *, tol: float) -> int:
    """
    Compute a numerical rank estimate via Gaussian elimination (no dependencies).

    Parameters
    ----------
    A
        Matrix as a sequence of rows.
    tol
        Pivot tolerance. Values with absolute value <= tol are treated as zero.

    Returns
    -------
    int
        Estimated rank.

    Notes
    -----
    This is a lightweight helper for diagnostics (e.g. affine dimension) and is
    not intended to be a fully robust numerical linear algebra routine.

    Examples
    --------
    >>> _matrix_rank([[1.0, 0.0], [0.0, 1.0]], tol=1e-12)
    2
    >>> _matrix_rank([[1.0, 2.0], [2.0, 4.0]], tol=1e-12)
    1
    """
    if not A:
        return 0
    n_cols = len(A[0])
    if n_cols == 0:
        return 0

    t = max(1e-14, float(tol))
    M = [[float(v) for v in row] for row in A]
    m = len(M)

    rank = 0
    for col in range(n_cols):
        pivot = None
        pivot_abs = t
        for r in range(rank, m):
            v = abs(float(M[r][col]))
            if v > pivot_abs:
                pivot_abs = v
                pivot = r
        if pivot is None:
            continue

        M[rank], M[pivot] = M[pivot], M[rank]
        piv = float(M[rank][col])
        if abs(piv) <= t:
            continue

        for r in range(rank + 1, m):
            factor = float(M[r][col]) / piv
            if abs(factor) <= t:
                continue
            for c in range(col, n_cols):
                M[r][c] = float(M[r][c]) - factor * float(M[rank][c])

        rank += 1
        if rank >= m:
            break

    return int(rank)

def _reduce_equalities(
    A_eq: Sequence[Sequence[float]], b_eq: Sequence[float], *, n_vars: int, tol: float
) -> tuple[list[list[float]], list[float], bool]:
    """
    Reduce equality constraints to an independent set (RREF-like), detecting inconsistency.

    Parameters
    ----------
    A_eq, b_eq
        Equality constraints $A_{eq} x = b_{eq}$.
    n_vars
        Number of variables (columns).
    tol
        Pivot tolerance for row reduction.

    Returns
    -------
    (A_eq_red, b_eq_red, inconsistent)
        Reduced equalities and a flag indicating whether the system is inconsistent
        (i.e. implies $0 = c$ for $c \\ne 0$).

    Notes
    -----
    This helper is used primarily to make vertex enumeration robust to redundant
    equality rows.

    Examples
    --------
    >>> A, b, bad = _reduce_equalities([[1.0, 1.0], [2.0, 2.0]], [1.0, 2.0], n_vars=2, tol=1e-12)
    >>> bad
    False
    >>> len(A)
    1
    """
    if not A_eq:
        return [], [], False

    t = max(1e-14, float(tol))
    M = [[float(v) for v in row] + [float(rhs)] for row, rhs in zip(A_eq, b_eq)]
    m = len(M)
    n = int(n_vars)

    row = 0
    for col in range(n):
        pivot = None
        pivot_abs = t
        for r in range(row, m):
            v = abs(float(M[r][col]))
            if v > pivot_abs:
                pivot_abs = v
                pivot = r
        if pivot is None:
            continue
        M[row], M[pivot] = M[pivot], M[row]

        piv = float(M[row][col])
        if abs(piv) <= t:
            continue
        inv = 1.0 / piv
        for c in range(col, n + 1):
            M[row][c] = float(M[row][c]) * inv

        for r in range(m):
            if r == row:
                continue
            factor = float(M[r][col])
            if abs(factor) <= t:
                continue
            for c in range(col, n + 1):
                M[r][c] = float(M[r][c]) - factor * float(M[row][c])

        row += 1
        if row >= m:
            break

    A_red: list[list[float]] = []
    b_red: list[float] = []
    for r in range(m):
        coeffs = M[r][:n]
        rhs = float(M[r][n])
        if all(abs(float(v)) <= t for v in coeffs):
            if abs(rhs) > 10.0 * t:
                return [], [], True
            continue
        A_red.append([float(v) for v in coeffs])
        b_red.append(float(rhs))

    return A_red, b_red, False


@dataclass(frozen=True)
class PolyhedralSet:
    """
    Polyhedral set in H-representation with optional bounds.

    Definition
    ----------
    A polyhedral set (polyhedron) in $\\mathbb{R}$ is described by linear
    equalities, linear inequalities, and coordinate bounds:

    $$
    A_{ub} x \\le b_{ub}, \\\\
    A_{eq} x  = b_{eq}, \\\\
    l_i \\le x_i \\le u_i.
    $$

    This class is a lightweight geometry utility intended to support set-valued
    solution objects (core, epsilon-core, least-core, etc.) and visualization
    tasks such as:
    - feasibility checks,
    - sampling a feasible point,
    - Chebyshev center,
    - enumerating extreme points in small dimension (brute-force),
    - projecting extreme points to selected coordinates.

    Notes
    -----
    - LP-based methods require an LP backend (SciPy/HiGHS or PuLP via your `linprog_solve` wrapper).
    - The `extreme_points` enumerator is **exponential** and intended only for
      small dimension (visualization).

    Attributes
    ----------
    A_ub, b_ub
        Inequality constraints `A_ub x <= b_ub`.
    A_eq, b_eq
        Equality constraints `A_eq x = b_eq`.
    bounds
        Per-coordinate bounds `(lb, ub)`; each side may be None.

    Examples
    --------
    A 2D triangle: x>=0, y>=0, x+y<=1.

    >>> P = PolyhedralSet.from_hrep(
    ...     A_ub=[[ -1,  0], [ 0, -1], [1, 1]],
    ...     b_ub=[  0,  0, 1],
    ... )
    >>> P.contains([0.2, 0.3])
    True
    >>> P.contains([0.8, 0.3])
    False
    """

    A_ub: list[list[float]]
    b_ub: list[float]
    A_eq: list[list[float]]
    b_eq: list[float]
    bounds: list[tuple[float | None, float | None]]

    # Cached H-representation with bounds converted to inequalities.
    _A_ub_all_cache: list[list[float]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _b_ub_all_cache: list[float] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def n_vars(self) -> int:
        """
        Dimension (number of variables) inferred from bounds or constraints.

        Returns
        -------
        int
            The inferred dimension n.

        Raises
        ------
        InvalidParameterError
            If the dimension cannot be inferred (no bounds and empty constraint matrices).

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (None, None)])
        >>> P.n_vars
        2
        """
        if self.bounds:
            return len(self.bounds)
        if self.A_eq:
            return len(self.A_eq[0])
        if self.A_ub:
            return len(self.A_ub[0])
        raise InvalidParameterError("cannot infer dimension (no bounds and empty constraint matrices)")

    def _hrep_with_bounds_as_ub(self) -> tuple[list[list[float]], list[float]]:
        """
        Return a cached pair (A_ub_all, b_ub_all) where variable bounds are encoded
        as additional inequalities.

        This is used by vertex enumeration and other routines that need a single
        "all inequalities" view of the polyhedron.
        """
        if self._A_ub_all_cache is not None and self._b_ub_all_cache is not None:
            return self._A_ub_all_cache, self._b_ub_all_cache

        n = self.n_vars
        A_ub_all: list[list[float]] = [row[:] for row in self.A_ub]
        b_ub_all: list[float] = [float(v) for v in self.b_ub]

        for i, (lb, ub) in enumerate(self.bounds):
            if ub is not None:
                row = [0.0] * n
                row[i] = 1.0
                A_ub_all.append(row)
                b_ub_all.append(float(ub))
            if lb is not None:
                row = [0.0] * n
                row[i] = -1.0
                A_ub_all.append(row)
                b_ub_all.append(-float(lb))

        # Bypass frozen dataclass to store caches.
        object.__setattr__(self, "_A_ub_all_cache", A_ub_all)
        object.__setattr__(self, "_b_ub_all_cache", b_ub_all)
        return A_ub_all, b_ub_all

    @classmethod
    def from_hrep(
        cls,
        *,
        A_ub: Sequence[Sequence[float]] | None = None,
        b_ub: Sequence[float] | None = None,
        A_eq: Sequence[Sequence[float]] | None = None,
        b_eq: Sequence[float] | None = None,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
    ) -> "PolyhedralSet":
        """
        Construct a polyhedron from an H-representation.

        Parameters
        ----------
        A_ub, b_ub
            Inequality constraints `A_ub x <= b_ub`. If omitted, no inequalities.
        A_eq, b_eq
            Equality constraints `A_eq x = b_eq`. If omitted, no equalities.
        bounds
            Optional bounds `(lb, ub)` per coordinate.

        Returns
        -------
        PolyhedralSet
            A normalized polyhedral set.

        Raises
        ------
        InvalidParameterError
            If dimensions mismatch or the dimension cannot be inferred.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> # Unit square in 2D: 0<=x<=1, 0<=y<=1.
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0)])
        >>> P.contains([0.5, 0.5])
        True
        """
        A_ub_list = [list(map(float, row)) for row in (A_ub or [])]
        b_ub_list = [float(v) for v in (b_ub or [])]
        A_eq_list = [list(map(float, row)) for row in (A_eq or [])]
        b_eq_list = [float(v) for v in (b_eq or [])]

        n = 0
        if A_eq_list:
            n = len(A_eq_list[0])
        elif A_ub_list:
            n = len(A_ub_list[0])
        elif bounds is not None:
            n = len(bounds)
        else:
            raise InvalidParameterError("cannot infer dimension (provide constraints or bounds)")

        if any(len(row) != n for row in A_ub_list):
            raise InvalidParameterError("A_ub rows must all have the same number of columns")
        if any(len(row) != n for row in A_eq_list):
            raise InvalidParameterError("A_eq rows must all have the same number of columns")
        if len(b_ub_list) != len(A_ub_list):
            raise InvalidParameterError("b_ub length must match number of A_ub rows")
        if len(b_eq_list) != len(A_eq_list):
            raise InvalidParameterError("b_eq length must match number of A_eq rows")

        return cls(
            A_ub=A_ub_list,
            b_ub=b_ub_list,
            A_eq=A_eq_list,
            b_eq=b_eq_list,
            bounds=_as_bounds(bounds, n),
        )

    def contains(self, x: Sequence[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Check membership with tolerance.

        Parameters
        ----------
        x
            Candidate point.
        tol
            Tolerance for bound checks, inequalities, and equalities.

        Returns
        -------
        bool
            True if x satisfies all constraints within tolerance.

        Notes
        -----
        - Inequalities are accepted if $A_{ub} x \\leq b_{ub} + \\text{tol}$.
        - Equalities are accepted if $|A_{eq} x - b_{eq}| \\leq \\text{tol}$ row-wise.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(A_ub=[[1, 1]], b_ub=[1], bounds=[(0.0, None), (0.0, None)])
        >>> P.contains([0.25, 0.25])
        True
        >>> P.contains([0.75, 0.75])
        False
        """
        n = self.n_vars
        if len(x) != n:
            return False

        xt = [float(v) for v in x]
        t = float(tol)

        for i, (lb, ub) in enumerate(self.bounds):
            if lb is not None and xt[i] < float(lb) - t:
                return False
            if ub is not None and xt[i] > float(ub) + t:
                return False

        for row, rhs in zip(self.A_ub, self.b_ub):
            if _dot(row, xt) > float(rhs) + t:
                return False

        for row, rhs in zip(self.A_eq, self.b_eq):
            if abs(_dot(row, xt) - float(rhs)) > t:
                return False

        return True

    def is_empty(self) -> bool:
        """
        Feasibility check via LP.

        This solves a trivial feasibility LP (zero objective) and returns whether
        the LP solver finds any feasible point.

        Returns
        -------
        bool
            True if the set is infeasible (empty); False otherwise.

        Notes
        -----
        Requires an LP backend at runtime (e.g. SciPy/HiGHS through `linprog_solve`).

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> # Infeasible: x <= 0 and x >= 1.
        >>> P = PolyhedralSet.from_hrep(A_ub=[[1], [-1]], b_ub=[0, -1])
        >>> P.is_empty()  # doctest: +SKIP
        True
        """
        n = self.n_vars
        c = [0.0] * n
        A_ub = self.A_ub if self.A_ub else None
        b_ub = self.b_ub if self.b_ub else None
        A_eq = self.A_eq if self.A_eq else None
        b_eq = self.b_eq if self.b_eq else None
        res = linprog_solve(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            method="highs",
            require_success=False,
            context="PolyhedralSet.is_empty",
        )
        return not bool(getattr(res, "success", False))

    def affine_dimension(self, *, tol: float = DEFAULT_GEOMETRY_TOL) -> int:
        """
        Estimate the affine dimension of the equality/bound hull.

        This estimates the dimension of the affine subspace defined by:

        - equalities: $A_{eq} x = b_{eq}$, and
        - fixed bounds: $x_i = l_i = u_i$.

        The returned value is:

        $$\\dim \\approx n - \\mathrm{rank}(M),$$

        where $M$ stacks the equality rows and the unit vectors for fixed
        coordinates.

        Parameters
        ----------
        tol
            Numerical tolerance used in the rank estimate.

        Returns
        -------
        int
            Estimated affine dimension in $\\{0,\\dots,n\\}$.

        Notes
        -----
        - This method does not check feasibility. For an empty set, the result is
          a diagnostic estimate for the *constraint hull* rather than the set
          itself.
        - If there are no equalities and no fixed bounds, this returns $n$.

        Examples
        --------
        A line segment in 2D has affine dimension 1:

        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(A_eq=[[1.0, 1.0]], b_eq=[1.0], bounds=[(0.0, 1.0), (0.0, 1.0)])
        >>> P.affine_dimension()
        1
        """
        n = int(self.n_vars)
        t = float(tol)
        eq_rows: list[list[float]] = [row[:] for row in self.A_eq]
        for i, (lb, ub) in enumerate(self.bounds):
            if lb is None or ub is None:
                continue
            if abs(float(lb) - float(ub)) > 0.0:
                continue
            row = [0.0] * n
            row[int(i)] = 1.0
            eq_rows.append(row)

        r = _matrix_rank(eq_rows, tol=t)
        dim = n - int(r)
        if dim < 0:
            dim = 0
        if dim > n:
            dim = n
        return int(dim)

    def is_bounded(self, *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
        """
        Check boundedness by optimizing each coordinate (LP-based).

        A polyhedron is bounded iff every linear functional attains a finite
        maximum. As a practical diagnostic, this method checks whether each
        coordinate $x_i$ has a finite minimum and maximum by solving $2n$ LPs.

        Parameters
        ----------
        tol
            Tolerance used to detect fixed bounds and to interpret solver output.

        Returns
        -------
        bool
            True if bounded (or empty), False if an unbounded direction is found.

        Notes
        -----
        - Requires an LP backend at runtime.
        - For large `n`, this can be expensive ($2n$ solves).
        - This method is intended mainly for guarding algorithms like hit-and-run.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0)])
        >>> P.is_bounded()  # doctest: +SKIP
        True
        """
        # Empty is considered bounded (no unbounded rays to witness).
        if self.is_empty():
            return True

        n = int(self.n_vars)
        t = float(tol)

        A_ub = self.A_ub if self.A_ub else None
        b_ub = self.b_ub if self.b_ub else None
        A_eq = self.A_eq if self.A_eq else None
        b_eq = self.b_eq if self.b_eq else None

        def looks_unbounded(res: Any) -> bool:
            status = getattr(res, "status", None)
            msg = str(getattr(res, "message", "")).lower()
            if status == 3:
                return True
            if "unbounded" in msg:
                return True
            return False

        def ok_or_raise(res: Any) -> None:
            if bool(getattr(res, "success", False)):
                return
            if looks_unbounded(res):
                return
            status = getattr(res, "status", None)
            msg = str(getattr(res, "message", "linprog failed"))
            raise BackendError(f"is_bounded: LP failed (status={status}): {msg}")

        for i, (lb, ub) in enumerate(self.bounds):
            if lb is not None and ub is not None and isfinite(float(lb)) and isfinite(float(ub)):
                continue
            if lb is not None and ub is not None and abs(float(lb) - float(ub)) <= 10.0 * t:
                continue

            c_min = [0.0] * n
            c_min[int(i)] = 1.0
            res_min = linprog_solve(
                c_min,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=self.bounds,
                method="highs",
                require_success=False,
                context="PolyhedralSet.is_bounded(min)",
            )
            ok_or_raise(res_min)
            if looks_unbounded(res_min):
                return False

            c_max = [0.0] * n
            c_max[int(i)] = -1.0
            res_max = linprog_solve(
                c_max,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=self.bounds,
                method="highs",
                require_success=False,
                context="PolyhedralSet.is_bounded(max)",
            )
            ok_or_raise(res_max)
            if looks_unbounded(res_max):
                return False

        return True

    def is_empty_with_diag(self) -> tuple[bool, Any]:
        """
        Feasibility check via LP with diagnostics.

        Returns
        -------
        (is_empty, result)
            Where ``result`` is the LP backend result object.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(A_ub=[[1.0], [-1.0]], b_ub=[0.0, -1.0])
        >>> empty, res = P.is_empty_with_diag()  # doctest: +SKIP
        >>> empty  # doctest: +SKIP
        True
        """
        n = self.n_vars
        c = [0.0] * n
        A_ub = self.A_ub if self.A_ub else None
        b_ub = self.b_ub if self.b_ub else None
        A_eq = self.A_eq if self.A_eq else None
        b_eq = self.b_eq if self.b_eq else None
        res = linprog_solve(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            method="highs",
            require_success=False,
            context="PolyhedralSet.is_empty_with_diag",
        )
        return (not bool(getattr(res, "success", False))), res

    def sample_point(self) -> list[float] | None:
        """
        Find any feasible point via LP.

        Returns
        -------
        list[float] | None
            A feasible point if one exists, otherwise None.

        Notes
        -----
        This is a convenience wrapper around a feasibility LP with zero objective.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0)])
        >>> x = P.sample_point()  # doctest: +SKIP
        >>> x is None or P.contains(x)
        True
        """
        n = self.n_vars
        c = [0.0] * n
        A_ub = self.A_ub if self.A_ub else None
        b_ub = self.b_ub if self.b_ub else None
        A_eq = self.A_eq if self.A_eq else None
        b_eq = self.b_eq if self.b_eq else None
        res = linprog_solve(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            method="highs",
            require_success=False,
            context="PolyhedralSet.sample_point",
        )
        if not bool(getattr(res, "success", False)):
            return None
        return [float(v) for v in res.x.tolist()]

    def sample_point_with_diag(self) -> tuple[list[float] | None, Any]:
        """
        Find any feasible point via LP with diagnostics.

        Returns
        -------
        (x, result)
            ``x`` is a feasible point if one exists, otherwise None. ``result`` is
            the LP backend result object.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0)])
        >>> x, res = P.sample_point_with_diag()  # doctest: +SKIP
        >>> x is None or P.contains(x)  # doctest: +SKIP
        True
        """
        n = self.n_vars
        c = [0.0] * n
        A_ub = self.A_ub if self.A_ub else None
        b_ub = self.b_ub if self.b_ub else None
        A_eq = self.A_eq if self.A_eq else None
        b_eq = self.b_eq if self.b_eq else None
        res = linprog_solve(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            method="highs",
            require_success=False,
            context="PolyhedralSet.sample_point_with_diag",
        )
        if not bool(getattr(res, "success", False)):
            return None, res
        return [float(v) for v in res.x.tolist()], res

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Chebyshev center via LP (maximum inscribed ball).

        The Chebyshev center is the point maximizing the radius of an Euclidean ball
        contained in the polyhedron. For inequalities $a_k^T x \\leq b_k$, the ball
        constraint becomes:

        $$
        a_k^T x + \\|a_k\\|_2 \\, r \\le b_k, \\qquad r \\geq 0.
        $$

        Equalities are kept as $a^T x = b$ (they do not involve $r$).

        Returns
        -------
        (center, radius) | None
            The center `x` and radius `r` of the largest inscribed ball, or None if
            infeasible.

        Notes
        -----
        - If there are **no inequalities** (only equalities/bounds), a meaningful
          inscribed-ball radius is not defined in the same way. In that case we
          return any feasible point and $r = +\\infty$ as a sentinel.

        Examples
        --------
        >>> P = PolyhedralSet.from_hrep(A_ub=[[1, 0], [0, 1], [-1, 0], [0, -1]], b_ub=[1, 1, 0, 0])
        >>> center, r = P.chebyshev_center()
        >>> r >= 0
        True
        """
        # If there are no halfspaces, "radius" is not meaningful. Return any feasible point with r=inf-ish.
        if not self.A_ub:
            x = self.sample_point()
            if x is None:
                return None
            return x, float("inf")

        n = self.n_vars
        # Variables: [x0..x_{n-1}, r]
        c = [0.0] * n + [-1.0]

        A_ub2: list[list[float]] = []
        b_ub2: list[float] = []
        for row, rhs in zip(self.A_ub, self.b_ub):
            A_ub2.append([float(v) for v in row] + [_row_norm(row)])
            b_ub2.append(float(rhs))

        A_eq2: list[list[float]] = []
        b_eq2: list[float] = []
        for row, rhs in zip(self.A_eq, self.b_eq):
            A_eq2.append([float(v) for v in row] + [0.0])
            b_eq2.append(float(rhs))

        bounds2 = list(self.bounds) + [(0.0, None)]
        res = linprog_solve(
            c,
            A_ub=A_ub2,
            b_ub=b_ub2,
            A_eq=A_eq2 if A_eq2 else None,
            b_eq=b_eq2 if b_eq2 else None,
            bounds=bounds2,
            method="highs",
            require_success=False,
            context="PolyhedralSet.chebyshev_center",
        )
        if not bool(getattr(res, "success", False)):
            return None
        sol = [float(v) for v in res.x.tolist()]
        return sol[:-1], float(sol[-1])

    def chebyshev_center_with_diag(
        self,
    ) -> tuple[tuple[list[float], float] | None, Any]:
        """
        Chebyshev center via LP with diagnostics.

        Returns
        -------
        (center_radius, result)
            ``center_radius`` is ``(x, r)`` or None if infeasible. ``result`` is the
            LP backend result object.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0)])
        >>> (cc, res) = P.chebyshev_center_with_diag()  # doctest: +SKIP
        >>> cc is None or len(cc[0]) == 1  # doctest: +SKIP
        True
        """
        # Keep behavior consistent with chebyshev_center.
        if not self.A_ub:
            x, res = self.sample_point_with_diag()
            if x is None:
                return None, res
            return (x, float("inf")), res

        n = self.n_vars
        c = [0.0] * n + [-1.0]

        A_ub2: list[list[float]] = []
        b_ub2: list[float] = []
        for row, rhs in zip(self.A_ub, self.b_ub):
            A_ub2.append([float(v) for v in row] + [_row_norm(row)])
            b_ub2.append(float(rhs))

        A_eq2: list[list[float]] = []
        b_eq2: list[float] = []
        for row, rhs in zip(self.A_eq, self.b_eq):
            A_eq2.append([float(v) for v in row] + [0.0])
            b_eq2.append(float(rhs))

        bounds2 = list(self.bounds) + [(0.0, None)]
        res = linprog_solve(
            c,
            A_ub=A_ub2,
            b_ub=b_ub2,
            A_eq=A_eq2 if A_eq2 else None,
            b_eq=b_eq2 if b_eq2 else None,
            bounds=bounds2,
            method="highs",
            require_success=False,
            context="PolyhedralSet.chebyshev_center_with_diag",
        )
        if not bool(getattr(res, "success", False)):
            return None, res
        sol = [float(v) for v in res.x.tolist()]
        return (sol[:-1], float(sol[-1])), res

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points (vertices) for small-dimensional polytopes.

        Method
        ------
        A point is a vertex if it is the unique intersection of $n$ linearly
        independent active constraints (equalities plus tight inequalities/bounds).

        This brute-force enumerator proceeds as follows:

        1. Convert variable bounds to additional inequalities.
        2. Choose $n - m_{eq}$ inequalities to add to the $m_eq$ equalities,
           yielding a square linear system $A x = b$.
        3. Solve this system.
        4. Keep the solution if it satisfies the original polyhedron constraints.
        5. De-duplicate solutions by quantizing coordinates.

        Parameters
        ----------
        tol
            Numerical tolerance used both in the linear solve and membership tests.
        max_dim
            Safety limit: only dimensions `n <= max_dim` are supported.

        Returns
        -------
        list[list[float]]
            List of vertices, sorted lexicographically. Returns an empty list if no
            vertices are found.

        Notes
        -----
        - This routine is exponential in the number of inequalities and is intended
          for visualization (e.g. $n=2,3,4$) rather than serious polyhedral computation.
        - Degenerate polytopes (many redundant/tied constraints) may require a larger
          tolerance for stable de-duplication.

        Examples
        --------
        Triangle in 2D:

        >>> P = PolyhedralSet.from_hrep(A_ub=[[-1,0],[0,-1],[1,1]], b_ub=[0,0,1])
        >>> P.extreme_points()
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        """
        n = self.n_vars
        if n > int(max_dim):
            raise NotSupportedError(f"extreme_points only supported for n<={max_dim} (got n={n})")

        # Convert bounds to inequalities once (cached) so vertices on bounds are found.
        A_ub_all, b_ub_all = self._hrep_with_bounds_as_ub()

        A_eq_all, b_eq_all, bad = _reduce_equalities(self.A_eq, self.b_eq, n_vars=n, tol=float(tol))
        if bad:
            return []

        m_eq = len(A_eq_all)
        if m_eq > n:
            return []
        choose = n - m_eq
        if choose < 0:
            return []

        verts: list[list[float]] = []
        seen: set[tuple[int, ...]] = set()

        for idxs in combinations(range(len(A_ub_all)), choose):
            A = [row[:] for row in A_eq_all]
            b = [float(v) for v in b_eq_all]
            for j in idxs:
                A.append(A_ub_all[j][:])
                b.append(float(b_ub_all[j]))

            # Solve A x = b
            x = solve_linear_system(A, b, eps=float(tol))
            if x is None:
                continue
            if not self.contains(x, tol=float(tol)):
                continue
            key = _quantize_key(x, tol=float(tol))
            if key is None:
                continue
            if key in seen:
                continue
            seen.add(key)
            verts.append([float(v) for v in x])

        verts.sort()
        return verts

    def project(
        self,
        dims: Sequence[int],
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
        approx_n_points: int | None = None,
        approx_seed: int | None = None,
    ) -> list[list[float]]:
        """
        Project enumerated extreme points to selected coordinates (visualization helper).

        Parameters
        ----------
        dims
            Indices of coordinates to keep.
        tol
            Passed to `extreme_points`.
        max_dim
            Passed to `extreme_points`.
        approx_n_points
            If provided and `n_vars > max_dim`, use hit-and-run sampling to
            produce approximately projected points. This requires boundedness
            and an LP backend to find a starting point.
        approx_seed
            Optional RNG seed used by hit-and-run when `approx_n_points` is set.

        Returns
        -------
        list[list[float]]
            Projected vertex list.

        Notes
        -----
        This is not a true polyhedral projection (which would require eliminating
        variables). It simply enumerates vertices in the full space (small $n$) and
        returns the selected coordinates.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
        >>> # Project cube vertices to the first two coordinates.
        >>> verts2 = P.project((0, 1), max_dim=3)
        >>> [0.0, 0.0] in verts2 and [1.0, 1.0] in verts2
        True
        """
        d = [int(i) for i in dims]
        n = self.n_vars
        if any(i < 0 or i >= n for i in d):
            raise InvalidParameterError("dims out of range")
        if approx_n_points is not None and int(n) > int(max_dim):
            if int(approx_n_points) < 1:
                raise InvalidParameterError("approx_n_points must be >= 1")
            x0 = self.sample_point()
            if x0 is None:
                return []
            pts = self.sample_points_hit_and_run(
                int(approx_n_points),
                start_point=x0,
                tol=float(tol),
                seed=approx_seed,
            )
            out: list[list[float]] = []
            seen: set[tuple[int, ...]] = set()
            for p in pts:
                q = [float(p[i]) for i in d]
                key = _quantize_key(q, tol=float(tol))
                if key is None or key in seen:
                    continue
                seen.add(key)
                out.append(q)
            out.sort()
            return out

        verts = self.extreme_points(tol=tol, max_dim=max_dim)
        return [[v[i] for i in d] for v in verts]

    def slack_ub(self, x: Sequence[float]) -> list[float]:
        """
        Compute inequality slacks for the combined $A_{ub} x \\leq b_{ub}$ system.

        This includes bound-inequalities (lb/ub converted to halfspaces) so it is
        suitable for "tight set" diagnostics.

        Parameters
        ----------
        x
            Point in R^n.

        Returns
        -------
        list[float]
            Slacks `b_k - a_k^T x` for each inequality row k.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(A_ub=[[1]], b_ub=[1], bounds=[(0.0, None)])
        >>> P.slack_ub([0.25])[0]
        0.75
        """
        A_ub_all, b_ub_all = self._hrep_with_bounds_as_ub()
        xt = [float(v) for v in x]
        return [float(rhs) - _dot(row, xt) for row, rhs in zip(A_ub_all, b_ub_all)]

    def residual_eq(self, x: Sequence[float]) -> list[float]:
        """
        Compute equality residuals $a_k^T x - b_k$ for each equality row.

        Parameters
        ----------
        x
            Point in R^n.

        Returns
        -------
        list[float]
            Residuals for each equality constraint.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(A_eq=[[1, 1]], b_eq=[1], bounds=[(0.0, 1.0), (0.0, 1.0)])
        >>> P.residual_eq([0.25, 0.75])
        [0.0]
        """
        xt = [float(v) for v in x]
        return [_dot(row, xt) - float(rhs) for row, rhs in zip(self.A_eq, self.b_eq)]

    def slack_eq(self, x: Sequence[float]) -> list[float]:
        """
        Compute absolute equality slacks $|a_k^T x - b_k|$ for each equality row.

        Parameters
        ----------
        x
            Point in R^n.

        Returns
        -------
        list[float]
            Absolute residuals for each equality constraint.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(A_eq=[[1.0, 1.0]], b_eq=[1.0])
        >>> P.slack_eq([0.25, 0.75])
        [0.0]
        """
        return [abs(float(v)) for v in self.residual_eq(x)]

    def sample_points_hit_and_run(
        self,
        n_points: int,
        *,
        start_point: Sequence[float] | None = None,
        burn_in: int = DEFAULT_HIT_AND_RUN_BURN_IN,
        thinning: int = DEFAULT_HIT_AND_RUN_THINNING,
        seed: int | None = None,
        tol: float = DEFAULT_HIT_AND_RUN_TOL,
    ) -> list[list[float]]:
        """
        Sample points using hit-and-run within the polyhedron (approximate uniform sampling).

        This sampler:
        1) starts from a feasible point (either ``start_point`` or `sample_point`),
        2) repeatedly samples a random direction (restricted to the equality subspace),
        3) computes the feasible segment along that direction using all inequalities
           (including bounds encoded as inequalities),
        4) samples uniformly on that segment.

        Parameters
        ----------
        n_points
            Number of returned samples.
        start_point
            Optional feasible start point. If omitted, a start point is found via
            `sample_point` (which requires an LP backend, e.g. ``tucoop[lp]``).
        burn_in
            Number of initial steps discarded.
        thinning
            Keep one sample every `thinning` steps after burn-in.
        seed
            Random seed.
        tol
            Numerical tolerance used in feasibility checks.

        Returns
        -------
        list[list[float]]
            Sampled points (each a length-n vector).

        Notes
        -----
        - Requires a feasible start point. If ``start_point`` is not provided,
          this method calls `sample_point`, which requires an LP backend.
        - If the polyhedron is unbounded in a sampled direction (infinite segment),
          this method raises ValueError because "uniform" sampling is not defined.

        Examples
        --------
        >>> from tucoop.geometry import PolyhedralSet
        >>> P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0)])
        >>> pts = P.sample_points_hit_and_run(3, seed=0)  # doctest: +SKIP
        >>> len(pts)
        3
        """
        m = int(n_points)
        if m < 0:
            raise InvalidParameterError("n_points must be >= 0")
        if m == 0:
            return []
        if burn_in < 0 or thinning < 1:
            raise InvalidParameterError("burn_in must be >= 0 and thinning must be >= 1")

        rng = random.Random(seed)

        n = self.n_vars
        if start_point is None:
            x0 = self.sample_point()
            if x0 is None:
                raise InvalidParameterError("cannot sample: polyhedron is empty (no feasible start point)")
            x = [float(v) for v in x0]
        else:
            x = [float(v) for v in start_point]
            if len(x) != n:
                raise InvalidParameterError("start_point must have length n_vars")

        A_ub_all, b_ub_all = self._hrep_with_bounds_as_ub()
        A_eq = self.A_eq
        b_eq = self.b_eq
        t_tol = float(tol)

        def project_to_eq_nullspace(d: list[float]) -> list[float]:
            if not A_eq:
                return d
            # Orthogonal projection onto nullspace of A_eq using normal equations:
            # d_proj = d - A_eq^T y, where (A_eq A_eq^T) y = A_eq d.
            Ad = _mat_vec(A_eq, d)
            # Build M = A_eq A_eq^T (m x m)
            AT = [[float(A_eq[i][j]) for i in range(len(A_eq))] for j in range(n)]
            M = _mat_mat(A_eq, AT)
            y = solve_linear_system(M, [float(v) for v in Ad], eps=max(1e-12, t_tol))
            if y is None:
                return d
            d_proj = d[:]
            for j in range(n):
                s = 0.0
                for i in range(len(A_eq)):
                    s += float(A_eq[i][j]) * float(y[i])
                d_proj[j] = float(d_proj[j]) - float(s)
            return d_proj

        def satisfies_equalities(xx: list[float]) -> bool:
            if not A_eq:
                return True
            for row, rhs in zip(A_eq, b_eq):
                if abs(_dot(row, xx) - float(rhs)) > 100.0 * t_tol:
                    return False
            return True

        if not self.contains(x, tol=t_tol):
            raise InvalidParameterError("start_point is not feasible within tolerance")

        samples: list[list[float]] = []
        total_steps = int(burn_in) + m * int(thinning)
        for step in range(total_steps):
            # Sample a random direction (Gaussian) and project into equality subspace.
            d = [rng.gauss(0.0, 1.0) for _ in range(n)]
            d = project_to_eq_nullspace(d)
            norm = sqrt(sum(float(v) * float(v) for v in d))
            tries = 0
            while (norm <= 1e-12 or not isfinite(norm)) and tries < 50:
                d = [rng.gauss(0.0, 1.0) for _ in range(n)]
                d = project_to_eq_nullspace(d)
                norm = sqrt(sum(float(v) * float(v) for v in d))
                tries += 1
            if norm <= 1e-12 or not isfinite(norm):
                raise BackendError("failed to sample a valid direction (degenerate equality constraints?)")
            d = [float(v) / float(norm) for v in d]

            # Compute feasible interval [t_min, t_max] along x + t d.
            t_min = float("-inf")
            t_max = float("inf")
            for row, rhs in zip(A_ub_all, b_ub_all):
                denom = _dot(row, d)
                num = float(rhs) - _dot(row, x)
                if abs(denom) <= 1e-14:
                    if num < -t_tol:
                        # numerical inconsistency; current point should be feasible
                        raise BackendError("current point violates a constraint during hit-and-run")
                    continue
                bound = num / denom
                if denom > 0:
                    if bound < t_max:
                        t_max = bound
                else:
                    if bound > t_min:
                        t_min = bound

            if not isfinite(t_min) or not isfinite(t_max):
                raise NotSupportedError("polyhedron appears unbounded; hit-and-run requires a bounded feasible segment")
            if t_min > t_max + 1e-12:
                raise BackendError("empty feasible segment encountered during hit-and-run (numerical issue)")

            t = rng.random() * (t_max - t_min) + t_min
            x = [float(xi) + float(t) * float(di) for xi, di in zip(x, d)]

            # Snap numerical drift on equalities by projecting once (keeps chain on affine subspace).
            if A_eq and not satisfies_equalities(x):
                # One-step correction: solve A_eq delta = (b_eq - A_eq x) and add a minimum-norm correction
                # using A_eq^T y with (A_eq A_eq^T) y = (A_eq x - b_eq).
                r = [(_dot(row, x) - float(rhs)) for row, rhs in zip(A_eq, b_eq)]
                AT = [[float(A_eq[i][j]) for i in range(len(A_eq))] for j in range(n)]
                M = _mat_mat(A_eq, AT)
                y = solve_linear_system(M, [float(v) for v in r], eps=max(1e-12, t_tol))
                if y is not None:
                    for j in range(n):
                        s = 0.0
                        for i in range(len(A_eq)):
                            s += float(A_eq[i][j]) * float(y[i])
                        x[j] = float(x[j]) - float(s)

            if step >= int(burn_in):
                if ((step - int(burn_in)) % int(thinning)) == 0:
                    if not self.contains(x, tol=t_tol):
                        # avoid returning drifted points
                        continue
                    samples.append([float(v) for v in x])
                    if len(samples) >= m:
                        break

        return samples


__all__ = ["PolyhedralSet"]
