"""
# Internal helpers for simplex geometry and barycentric coordinates.

This module contains small utilities used by projection/sampling/viz code, such
as canonical simplex embeddings and barycentric conversions.

It is internal (prefixed with `_`) and may change without notice.

Examples
--------
>>> from tucoopy.geometry._utils import simplex_vertices_2d, barycentric_to_cartesian
>>> verts = simplex_vertices_2d(3)
>>> barycentric_to_cartesian([1.0, 0.0, 0.0], verts)
[0.0, 0.0]
"""

from __future__ import annotations

from math import sqrt
from random import Random

from ..base.exceptions import InvalidParameterError, NotSupportedError


def simplex_vertices_2d(n: int) -> list[list[float]]:
    """
    Canonical simplex vertices embedded in 2D for $n=3$.
    Returns vertices in $\\mathbb{R}^2$.

    Examples
    --------
    >>> simplex_vertices_2d(3)[-1]
    [0.5, 0.8660254037844386]
    """
    if int(n) != 3:
        raise NotSupportedError("simplex_vertices_2d only supports n=3")
    # Equilateral triangle.
    return [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3.0) / 2.0]]


def simplex_vertices_3d(n: int) -> list[list[float]]:
    """
    Canonical simplex vertices embedded in 3D for $n=4$.
    Returns vertices in $\\mathbb{R}^3$.

    Examples
    --------
    >>> len(simplex_vertices_3d(4))
    4
    """
    if int(n) != 4:
        raise NotSupportedError("simplex_vertices_3d only supports n=4")
    # Regular tetrahedron.
    return [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, sqrt(3.0) / 2.0, 0.0],
        [0.5, sqrt(3.0) / 6.0, sqrt(6.0) / 3.0],
    ]


def barycentric_to_cartesian(b: list[float], verts: list[list[float]]) -> list[float]:
    """
    Convert barycentric coords $b$ ($\\sum_{i=1}^n b_{i} = 1$) to cartesian coordinates using verts.

    Examples
    --------
    >>> verts = simplex_vertices_2d(3)
    >>> barycentric_to_cartesian([0.0, 1.0, 0.0], verts)
    [1.0, 0.0]
    """
    if len(b) != len(verts):
        raise InvalidParameterError("barycentric length must match number of vertices")
    d = len(verts[0]) if verts else 0
    if any(len(v) != d for v in verts):
        raise InvalidParameterError("verts must all have the same dimension")
    out = [0.0] * d
    for w, v in zip(b, verts):
        fw = float(w)
        for j in range(d):
            out[j] += fw * float(v[j])
    return out


def sample_dirichlet_uniform(n: int, *, rng: Random) -> list[float]:
    """
    Sample $\\operatorname{Dirichlet(1,...,1)}$ on the $(n-1)$-simplex.

    Examples
    --------
    >>> from random import Random
    >>> b = sample_dirichlet_uniform(3, rng=Random(0))
    >>> round(sum(b), 12)
    1.0
    """
    k = int(n)
    if k < 1:
        raise InvalidParameterError("n must be >= 1")
    e = [rng.expovariate(1.0) for _ in range(k)]
    s = sum(e)
    return [float(v) / float(s) for v in e]


__all__: list[str] = []
