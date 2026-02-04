"""
# Coordinate conversions and projections for visualization.

This module contains helpers that map allocations ``x`` in the imputation
hyperplane to convenient 2D/3D coordinates:

- barycentric coordinates on the imputation simplex (``n=3``),
- 2D/3D embeddings used by the static Matplotlib visualizations in ``tucoop.viz``.

Notes
-----
These helpers are intentionally simple and aimed at visualization and diagnostics.
They are not intended as general-purpose dimensionality reduction tools.

Examples
--------
Convert a point in the imputation set to barycentric coordinates (n=3):

>>> from tucoop import Game
>>> from tucoop.geometry.projection import allocation_to_barycentric_imputation
>>> g = Game.from_coalitions(n_players=3, values={
...     0: 0.0,
...     1: 1.0, 2: 1.0, 4: 1.0,
...     3: 2.0, 5: 2.0, 6: 2.0,
...     7: 4.0,
... })
>>> b = allocation_to_barycentric_imputation(g, [1.0, 1.0, 2.0])
>>> (len(b), round(sum(b), 12))
(3, 1.0)
"""

from __future__ import annotations

from typing import Literal

from ..base.config import DEFAULT_IMPUTATION_SAMPLE_TOL
from ..base.types import GameProtocol
from ..base.exceptions import InvalidGameError, InvalidParameterError, NotSupportedError
from ._utils import barycentric_to_cartesian, simplex_vertices_2d, simplex_vertices_3d
from .imputation_set import imputation_lower_bounds


def allocation_to_barycentric_imputation(
    game: GameProtocol, x: list[float], *, tol: float = DEFAULT_IMPUTATION_SAMPLE_TOL
) -> list[float]:
    """
    Convert an allocation to barycentric coordinates of the imputation simplex.

    Background
    ----------
    The imputation set of a TU game can be written as a **shifted simplex**:

    $$
    x = l + r \\cdot b,
    $$

    where

    - $l_i = v(\\{i\\})$ are the individual rationality lower bounds,
    - $r = v(N) - \\sum_i l_i$ is the remaining distributable surplus,
    - $b \\in \\mathbb{R}^n$ satisfies

      $$
      \\sum_i b_i = 1, \\qquad b_i \\ge 0.
      $$

    The vector $b$ is a set of **barycentric coordinates** inside the standard
    simplex, representing the position of $x$ relative to the imputation simplex.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector (length `n_players`) assumed to lie in the imputation set.
    tol
        Numerical tolerance for detecting degenerate simplices.

    Returns
    -------
    list[float]
        Barycentric coordinates `b` such that `sum(b)=1` and `b>=0`.

    Raises
    ------
    InvalidGameError
        If the imputation simplex is empty or degenerate
        ($v(N) - \\sum_i v(\\{i\\}) \\le 0$).

    Notes
    -----
    These coordinates are the natural input for geometric visualizations of
    allocations in 2D (n=3) and 3D (n=4).

    Examples
    --------
    >>> b = allocation_to_barycentric_imputation(g, [1.0, 1.0, 1.0])
    >>> sum(b)
    1.0
    """
    n = game.n_players
    if len(x) != n:
        raise InvalidParameterError("x must have length n_players")
    vN = float(game.value(game.grand_coalition))
    l = imputation_lower_bounds(game)
    r = vN - sum(l)
    if r <= tol:
        raise InvalidGameError("imputation simplex is empty or degenerate")
    return [(float(x[i]) - float(l[i])) / float(r) for i in range(n)]


def project_allocation(
    game: GameProtocol,
    x: list[float],
    *,
    space: Literal["imputation_simplex"] = "imputation_simplex",
) -> list[float]:
    """
    Project an allocation to Euclidean coordinates for visualization.

    Background
    ----------
    For small games, the imputation set is a simplex that can be embedded in
    low-dimensional Euclidean space:

    - For `n=3`, the imputation simplex is a triangle in **2D**.
    - For `n=4`, the imputation simplex is a tetrahedron in **3D**.

    This function:

    1. Converts the allocation to barycentric coordinates in the imputation simplex.
    2. Maps these barycentric coordinates to Cartesian coordinates using
       canonical simplex embeddings.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector in the imputation set.
    space
        Currently only `"imputation_simplex"` is supported.

    Returns
    -------
    list[float]
        2D coordinates (for n=3) or 3D coordinates (for n=4).

    Raises
    ------
    InvalidParameterError
        If the space is unknown.
    NotSupportedError
        If the number of players is not supported for this projection.

    Notes
    -----
    This is intended for geometric plotting of solution concepts such as:

    - Core
    - Epsilon-core
    - Least-core
    - Kernel / Pre-kernel
    - Nucleolus
    - Tau value

    Examples
    --------
    >>> project_allocation(g, [1.0, 1.0, 1.0])
    [x_coord, y_coord]
    """
    if space != "imputation_simplex":
        raise InvalidParameterError("unsupported space")
    n = game.n_players
    b = allocation_to_barycentric_imputation(game, x)
    if n == 3:
        return barycentric_to_cartesian(b, simplex_vertices_2d(3))
    if n == 4:
        return barycentric_to_cartesian(b, simplex_vertices_3d(4))
    raise NotSupportedError("project_allocation only supports n=3 (2D) or n=4 (3D)")


__all__ = ["allocation_to_barycentric_imputation", "project_allocation"]
