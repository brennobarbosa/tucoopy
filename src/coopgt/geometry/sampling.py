"""
# Sampling helpers for geometry objects.

This module focuses on sampling points from the **imputation set**
(a shifted simplex) via a uniform Dirichlet sampler, which is useful for:

- visual debugging,
- approximate membership probing for non-polyhedral sets (kernel, bargaining set),
- generating example points for documentation.

Examples
--------
Sample imputations from a small 3-player game (deterministic with a seed):

>>> from tucoop import Game
>>> from tucoop.geometry.sampling import sample_imputation_set
>>> g = Game.from_coalitions(n_players=3, values={
...     0: 0.0,
...     1: 1.0, 2: 1.0, 4: 1.0,
...     3: 2.0, 5: 2.0, 6: 2.0,
...     7: 4.0,
... })
>>> pts = sample_imputation_set(g, n_samples=5, seed=0)
>>> len(pts)
5
"""

from __future__ import annotations

from random import Random

from ..base.config import DEFAULT_IMPUTATION_SAMPLE_TOL
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ._utils import sample_dirichlet_uniform
from .imputation_set import imputation_lower_bounds


def sample_imputation_set(
    game: GameProtocol,
    *,
    n_samples: int,
    seed: int | None = None,
    tol: float = DEFAULT_IMPUTATION_SAMPLE_TOL,
) -> list[list[float]]:
    """
    Sample points from the **imputation set** (shifted simplex).

    The imputation set is

    $$
    I(v) = \\left\\{ x \\in \\mathbb{R}^n :
    \\sum_{i=1}^n x_i = v(N),\\; x_i \\ge v(\\{i\\}) \\right\\}.
    $$

    Let $\\ell_i = v(\\{i\\})$ and $r = v(N) - \\sum_i \\ell_i$. When
    $r \\ge 0$, the imputation set is a translation of the standard simplex:

    $$
    x = \\ell + r\\,b, \\qquad b \\ge 0,\\; \\sum_i b_i = 1.
    $$

    This routine samples $b$ using a Dirichlet distribution with all
    parameters equal to 1 (uniform over the simplex in barycentric coordinates),
    then maps back to $x$.

    Parameters
    ----------
    game
        TU game.
    n_samples
        Number of points to sample (must be >= 1).
    seed
        Optional seed for reproducibility (Python's ``random.Random``).
    tol
        Numerical tolerance for detecting an empty or degenerate imputation set.

    Returns
    -------
    list[list[float]]
        A list of allocations in the imputation set.

        - If the imputation set is empty (``r < -tol``), returns ``[]``.
        - If the imputation set is a singleton (``|r| <= tol``), returns ``[l]``.
        - Otherwise returns ``n_samples`` sampled imputations.

    Notes
    -----
    - "Uniform-ish" means uniform with respect to the simplex volume in the
      barycentric coordinates $b$ (Dirichlet(1,...,1)). After the affine
      map $x = \\ell + r b$, this corresponds to the natural uniform measure
      on the shifted simplex as well.
    - This is meant for visualization / Monte Carlo intuition, not for any
      sophisticated MCMC mixing guarantees.

    Examples
    --------
    Minimal 3-player example where the imputation set is non-empty:

    >>> from tucoop import Game
    >>> from tucoop.geometry.sampling import sample_imputation_set
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0: 0, 1: 0, 2: 0, 4: 0,
    ...     3: 0, 5: 0, 6: 0,
    ...     7: 1,
    ... })
    >>> pts = sample_imputation_set(g, n_samples=3, seed=0)
    >>> len(pts)
    3
    >>> all(abs(sum(x) - 1.0) < 1e-9 for x in pts)
    True

    Degenerate case: singleton imputation set (r = 0):

    >>> g2 = Game.from_coalitions(n_players=2, values={0:0, 1:1, 2:2, 3:3})
    >>> sample_imputation_set(g2, n_samples=10)
    [[1.0, 2.0]]
    """
    n = game.n_players
    if n_samples < 1:
        raise InvalidParameterError("n_samples must be >= 1")
    vN = float(game.value(game.grand_coalition))
    l = imputation_lower_bounds(game)
    r = vN - sum(l)
    if r < -tol:
        return []
    if abs(r) <= tol:
        return [[float(v) for v in l]]

    rng = Random(seed)
    out: list[list[float]] = []
    for _ in range(int(n_samples)):
        b = sample_dirichlet_uniform(n, rng=rng)
        out.append([float(l[i]) + float(r) * float(b[i]) for i in range(n)])
    return out


__all__ = ["sample_imputation_set"]
