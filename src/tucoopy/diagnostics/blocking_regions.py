"""
# Blocking regions diagnostics (n=3).

This module computes **max-excess blocking regions** inside the imputation simplex.
It is intended for visualization and explanation: regions can be rendered as
polygons in barycentric coordinates.

Notes
-----
The polygon clipping routine is implemented for ``n=3`` only (triangle simplex).
For other ``n`` this module returns an empty set of regions.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.diagnostics.blocking_regions import blocking_regions
>>> g = Game.from_coalitions(n_players=3, values={0: 0, 7: 1})
>>> br = blocking_regions(g)
>>> br.coordinate_system
'barycentric_imputation'
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.coalition import coalition_sums
from ..base.types import GameProtocol
from ..geometry.imputation_set import imputation_lower_bounds
from .core_diagnostics import core_diagnostics


@dataclass(frozen=True)
class BlockingRegion:
    """
    A single blocking region in barycentric coordinates.

    Attributes
    ----------
    coalition_mask
        Coalition mask S that is the (one of the) max-excess blockers in this region.
        For n=2, the only proper non-empty coalitions are {0} and {1} (masks {1,2}).
        For n=3, proper non-empty coalitions are masks in {1,2,3,4,5,6}.
    vertices
        Region vertices in **barycentric coordinates** b of the imputation simplex.
        Each vertex is a length-n vector b with:
          - b_i >= 0
          - sum_i b_i = 1

    Examples
    --------
    >>> r = BlockingRegion(coalition_mask=3, vertices=[[1.0, 0.0, 0.0]])
    >>> r.coalition_mask
    3
    """
    coalition_mask: int
    # Vertices in barycentric coordinates of the imputation simplex (length n, sum=1).
    vertices: list[list[float]]
    # Excess of coalition_mask at each vertex (same length as vertices).
    coalition_excess_at_vertices: list[float] | None = None
    # Max excess among proper coalitions at each vertex (same length as vertices).
    max_excess_at_vertices: list[float] | None = None
    # Ties for max excess at each vertex: list of coalition masks achieving the max (within tol).
    ties: list[list[int]] | None = None


@dataclass(frozen=True)
class BlockingRegions:
    """
    Blocking regions in the imputation simplex (implemented for $n=3$).

    Concept
    -------
    For an allocation $x$, the **excess** of a coalition $S$ is:

    $$
    e(S, x) = v(S) - x(S).
    $$

    A coalition $S$ is said to be a **(max-excess) blocker** at $x$.
    In our sign convention, *blocking* corresponds to **positive excess**
    (the coalition can improve upon $x$):

    1. It can block: $e(S, x) > 0$
    2. It attains the maximum excess among proper non-empty coalitions:
         $e(S, x) \\geq e(T, x)$ for all proper non-empty $T \\neq N$.

    This routine partitions (parts of) the imputation simplex into regions where
    the identity of the max-excess blocker is constant.

    Coordinate system
    -----------------
    The computation is performed in **barycentric coordinates** $b$ over the
    imputation simplex. Let:

    - $l_i = v(\\{i\\})$ (individual rationality lower bounds)
    - $r   = v(N) - \\sum_{i=1}^n l_i$ (simplex "radius")

    Any imputation can be written as:

    $$
    x = l + r b,
    $$

    where $b$ is barycentric:
    
    $$
    b_i \\geq 0, \\;  \\sum_i b_i = 1.
    $$

    Internally, we represent $b$ in 2D using $(b_0, b_1)$, with:
    
    $$
        b_2 = 1 - b_0 - b_1,
    $$

    and the simplex is the triangle with vertices $(b_0, b_1)$ in:
        
    $$
    (1,0), (0,1), (0,0).
    $$

    Notes
    -----
    - This is currently implemented only for $n=3$ because it uses planar polygon
      clipping (half-plane intersection).
    - If the imputation set is empty or degenerates to a point ($r \\leq tol$),
      the result is empty.
    - Regions are computed by intersecting the simplex triangle with linear
      half-planes derived from comparisons:
          
    $$e
    (S, x) \\geq e(T, x)
    $$
    
      and the blocking condition:
    
    $$
    e(S, x) >= 0.
    $$

    Attributes
    ----------
    coordinate_system
        Always "barycentric_imputation".
    regions
        List of `BlockingRegion` polygons.

    See also
    --------
    tucoopy.geometry.imputation.ImputationSet
        The imputation simplex representation used here.

    Examples
    --------
    >>> br = BlockingRegions(coordinate_system="barycentric_imputation", regions=[])
    >>> br.regions
    []
    """
    coordinate_system: str
    regions: list[BlockingRegion]


def _clip_polygon_halfplane(
    poly: list[tuple[float, float]], A: float, B: float, d: float, eps: float
) -> list[tuple[float, float]]:
    """
    Clip a 2D polygon against a half-plane.

    Keeps points satisfying $A \\cdot x + B \\cdot y \\geq d$ within numerical tolerance ``eps``.

    Parameters
    ----------
    poly
        Polygon vertices in 2D, in order.
    A, B, d
        Half-plane coefficients.
    eps
        Tolerance used by the inside test.

    Returns
    -------
    list[tuple[float, float]]
        The clipped polygon (possibly empty).

    Notes
    -----
    Uses Sutherlandâ€“Hodgman clipping. The output preserves input order.
    """
    # Keep points where A*x + B*y >= d (within eps).
    def inside(p: tuple[float, float]) -> bool:
        return A * p[0] + B * p[1] >= d - eps

    def intersect(p0: tuple[float, float], p1: tuple[float, float]) -> tuple[float, float]:
        v0 = A * p0[0] + B * p0[1] - d
        v1 = A * p1[0] + B * p1[1] - d
        denom = v0 - v1
        t = v0 / denom if abs(denom) > 1e-12 else 0.0
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    if not poly:
        return poly
    out: list[tuple[float, float]] = []
    m = len(poly)
    for i in range(m):
        cur = poly[i]
        prev = poly[(i + m - 1) % m]
        cur_in = inside(cur)
        prev_in = inside(prev)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
    return out


def _dedup_polygon(poly: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """
    Remove consecutive near-duplicate vertices from a polygon.

    Parameters
    ----------
    poly
        Polygon vertices in 2D.
    tol
        Absolute tolerance for considering two consecutive vertices equal.

    Returns
    -------
    list[tuple[float, float]]
        De-duplicated polygon vertices.
    """
    if not poly:
        return poly
    out: list[tuple[float, float]] = []
    for p in poly:
        if not out:
            out.append(p)
            continue
        q = out[-1]
        if abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol:
            continue
        out.append(p)
    # Also drop closing duplicate
    if len(out) >= 2:
        p0 = out[0]
        p1 = out[-1]
        if abs(p0[0] - p1[0]) <= tol and abs(p0[1] - p1[1]) <= tol:
            out.pop()
    return out


def _order_polygon_clockwise(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Order polygon vertices in a consistent clockwise order (2D).

    The input may already be ordered (Sutherland-Hodgman clipping preserves order),
    but ordering explicitly makes the output stable across different clipping paths
    and floating-point noise.
    """
    if len(poly) <= 2:
        return poly
    cx = sum(p[0] for p in poly) / float(len(poly))
    cy = sum(p[1] for p in poly) / float(len(poly))
    return sorted(poly, key=lambda p: atan2(p[1] - cy, p[0] - cx), reverse=True)


def blocking_regions(game: GameProtocol, *, tol: float = DEFAULT_GEOMETRY_TOL) -> BlockingRegions:
    """
    Compute max-excess blocking regions in the imputation simplex ($n=3$).

    Definition
    ----------
    For an allocation $x$, define the coalition excess:

    $$
    e(S, x) = v(S) - x(S),\\\\
    x(S) = \\sum_{i \\in S} x_i.
    $$

    A coalition S is a **max-excess blocker** at x if:
      
      1. $e(S, x) \\geq 0$
      2. $e(S, x) \\geq e(T, x)$ for all proper non-empty coalitions $T \\neq S$.

    This routine returns polygonal regions in the **imputation simplex**
    where a fixed coalition S satisfies the two conditions above.

    Coordinate system
    -----------------
    For $n=3$, every imputation can be written as:

    $$
    x = l + r b,
    $$

    where:

      - $l_i = v(\\{i\\})$
      - $r = v(N) - \\sum_i l_i$
      - $b$ is barycentric: $b_i \\geq 0$ and $\\sum_i b_i = 1$

    The returned polygons use $b$ as coordinates (stored explicitly as length-3
    vectors), but internally the clipping is done in the 2D chart $(b_0, b_1)$,
    with $b_2 = 1 - b_0 - b_1$.

    Parameters
    ----------
    game
        TU game.
    tol
        Numerical tolerance used to:
        - detect degenerate imputation simplex (r <= tol),
        - soften half-plane comparisons,
        - de-duplicate nearly-identical polygon vertices.

    Returns
    -------
    BlockingRegions
        A container with `regions`, each giving a coalition mask S and a polygon
        (list of barycentric vertices) describing where S is the max-excess blocker.

    Notes
    -----
    - Implemented for n=3. For n != 3 this returns no regions.
    - If the imputation set is empty or a singleton (r <= tol), returns no regions.
    - Ties between coalitions (multiple maximizers) are not explicitly merged;
      you may see overlapping/adjacent regions due to numerical tolerances.

    Examples
    --------
    >>> from tucoopy import Game
    >>> from tucoopy.diagnostics.blocking_regions import blocking_regions
    >>> g = Game.from_value_function(3, lambda S: float(len(S)))  # additive
    >>> br = blocking_regions(g)
    >>> br.regions
    []
    """
    n = game.n_players
    if n != 3:
        return BlockingRegions(coordinate_system="barycentric_imputation", regions=[])

    grand = game.grand_coalition
    vN = float(game.value(grand))
    l = imputation_lower_bounds(game)
    r = vN - sum(l)
    if r <= tol:
        # Empty or singleton imputation simplex => no meaningful 2D regions.
        return BlockingRegions(coordinate_system="barycentric_imputation", regions=[])

    # Proper non-empty coalitions for n=3: 1..6
    coalitions = [1, 2, 3, 4, 5, 6]

    def c(S: int) -> float:
        s = float(game.value(S))
        for i in range(n):
            if S & (1 << i):
                s -= l[i]
        return s

    def coeff(S: int) -> tuple[float, float, float]:
        # B_S = sum_{i in S} b_i, with b2 = 1 - b0 - b1.
        a0 = 1.0 if (S & 1) else 0.0
        a1 = 1.0 if (S & 2) else 0.0
        a2 = 1.0 if (S & 4) else 0.0
        # B_S = a2 + (a0-a2)*b0 + (a1-a2)*b1
        return (a2, a0 - a2, a1 - a2)

    def to_bary(p: tuple[float, float]) -> list[float]:
        b0, b1 = p
        return [float(b0), float(b1), float(1.0 - b0 - b1)]

    regions: list[BlockingRegion] = []
    eps = tol

    for S in coalitions:
        # Start poly as the full triangle in (b0,b1): (1,0),(0,1),(0,0)
        poly: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]

        cS = c(S)
        c2S, k0S, k1S = coeff(S)

        for T in coalitions:
            if T == S:
                continue
            cT = c(T)
            c2T, k0T, k1T = coeff(T)

            # e_S >= e_T  <=>  r*(B_T - B_S) >= c_T - c_S
            A = r * (k0T - k0S)
            B = r * (k1T - k1S)
            C0 = r * (c2T - c2S)
            rhs = cT - cS
            d = rhs - C0

            poly = _clip_polygon_halfplane(poly, A, B, d, eps)
            if not poly:
                break
        if not poly:
            continue

        # Additionally require that S is actually blocking: e_S >= 0.
        # e_S = c_S - r*B_S, with B_S = c2 + k0*b0 + k1*b1
        # e_S >= 0 <=> r*B_S <= c_S  <=>  -r*k0*b0 - r*k1*b1 >= r*c2 - c_S
        A = -r * k0S
        B = -r * k1S
        d = r * c2S - cS
        poly = _clip_polygon_halfplane(poly, A, B, d, eps)
        poly = _dedup_polygon(poly, tol=1e-10)
        poly = _order_polygon_clockwise(poly)

        if len(poly) < 3:
            continue

        bary = [to_bary(p) for p in poly]

        # Optional diagnostics at vertices: excess of S, max excess, and ties.
        ex_S: list[float] = []
        max_ex: list[float] = []
        ties: list[list[int]] = []
        for b in bary:
            x = [float(l[i]) + float(r) * float(b[i]) for i in range(n)]
            x_sum = coalition_sums(x, n_players=n)
            # Avoid duplicating max/tie logic: reuse the core diagnostics scanner.
            cd = core_diagnostics(game, x, tol=tol, top_k=0)
            max_ex.append(float(cd.max_excess))
            ties.append(list(cd.blocking_coalitions))

            vS = float(game.value(int(S)))
            ex_S.append(float(vS - float(x_sum[int(S)])))

        regions.append(
            BlockingRegion(
                coalition_mask=S,
                vertices=bary,
                coalition_excess_at_vertices=ex_S,
                max_excess_at_vertices=max_ex,
                ties=ties,
            )
        )

    return BlockingRegions(coordinate_system="barycentric_imputation", regions=regions)
