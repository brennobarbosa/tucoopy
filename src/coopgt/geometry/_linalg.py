"""
# Internal linear algebra helpers for geometry modules.

This module provides tiny helpers used by `tucoop.geometry.PolyhedralSet`
for "small $n$" operations (e.g. vertex enumeration).

It is internal (prefixed with `_`) and may change without notice.

Examples
--------
>>> from tucoop.geometry._linalg import solve_linear_system
>>> solve_linear_system([[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0], eps=1e-12)
[2.0, 3.0]
"""

from __future__ import annotations

def _solve_linear_system_gauss(A: list[list[float]], b: list[float], eps: float) -> list[float] | None:
    """
    Solve $A \\cdot x = b$ with Gaussian elimination (float).

    Returns None if singular/ill-conditioned.

    Examples
    --------
    >>> _solve_linear_system_gauss([[1.0]], [2.0], eps=1e-12)
    [2.0]
    """
    n = len(A)
    if n == 0:
        return []
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]

    for col in range(n):
        # Pivot
        pivot = col
        best = abs(M[col][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > best:
                best = v
                pivot = r
        if best <= eps:
            return None
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        # Normalize pivot row (only need columns col..n plus RHS).
        piv = M[col][col]
        inv = 1.0 / piv
        for c in range(col, n + 1):
            M[col][c] *= inv

        # Eliminate
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) <= eps:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    return [M[i][n] for i in range(n)]


def solve_linear_system(A: list[list[float]], b: list[float], eps: float) -> list[float] | None:
    """
    Solve $A \\cdot x = b$.

    Prefer NumPy when available (faster/more stable); otherwise use the small
    Gaussian elimination fallback.

    Examples
    --------
    >>> solve_linear_system([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0], eps=1e-12)
    [1.0, 2.0]
    """
    try:
        import numpy as np  # type: ignore

        x = np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))
        return [float(v) for v in x.tolist()]
    except Exception:
        return _solve_linear_system_gauss(A, b, eps)
