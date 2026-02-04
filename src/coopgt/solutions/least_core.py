"""
# Least-core (point-valued helpers).

This module provides LP-based computation of the least-core value and selection of a representative point
from the least-core set (e.g. Chebyshev center).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from ..base.types import GameProtocol
from ..base.exceptions import InvalidGameError, InvalidParameterError
from ..base.coalition import all_coalitions
from ..geometry.least_core_set import LeastCore


SelectionMethod: TypeAlias = Literal["chebyshev_center", "any_feasible"]


if TYPE_CHECKING:  # pragma: no cover
    from ..diagnostics.linprog_diagnostics import LinprogDiagnostics


@dataclass(frozen=True)
class LeastCoreResult:
    """
    Result container for the least-core LP.

    Attributes
    ----------
    x : list[float]
        One least-core allocation (length `n_players`).
    epsilon : float
        Least-core value (the minimal epsilon such that the epsilon-core is non-empty).
    tight : list[int] | None
        Coalitions (bitmask) whose least-core constraint is tight at the returned solution,
        i.e. those S for which the inequality is satisfied with equality (within tolerance).
        Proper non-empty coalitions only.
    dual_weights : dict[int, float] | None
        Optional inequality dual multipliers (when provided by the LP backend), keyed by
        coalition mask. These can be interpreted as "weights" on binding coalitions in
        the optimality conditions.
    lp : LinprogDiagnostics | None
        Optional solver diagnostics (SciPy/HiGHS).
    """
    x: list[float]
    epsilon: float
    tight: list[int] | None = None
    dual_weights: dict[int, float] | None = None
    lp: LinprogDiagnostics | None = None


@dataclass(frozen=True)
class LeastCorePointResult:
    """
    Result container for a single point selected from the least-core set.

    Attributes
    ----------
    x : list[float]
        Selected allocation (length `n_players`).
    epsilon : float
        Least-core epsilon value of the underlying least-core set.
    method : str
        Selection method used (e.g. "chebyshev_center" or "any_feasible").
    """
    x: list[float]
    epsilon: float
    method: str


def least_core(game: GameProtocol, *, tol: float = 1e-9) -> LeastCoreResult:
    """
    Compute a least-core allocation and the least-core epsilon.

    Background
    ----------
    The **epsilon-core** of a TU game is the set of allocations x satisfying:

    - Efficiency: $\\sum_{i = 1}^n x_i = v(N)$
    - Relaxed coalitional constraints:
      
    $$
    v(S) - x(S) \\le \\varepsilon
    \\quad \\text{for all proper, non-empty } S \\subset N,
    $$

    The **least-core** is obtained by choosing the smallest such $\\varepsilon$ for which
    the epsilon-core is non-empty.

    Parameters
    ----------
    game
        TU game.
    tol
        Numerical tolerance for tightness detection and for snapping tiny negative
        epsilons to 0 (numerical noise).

    Returns
    -------
    LeastCoreResult
        A least-core epsilon and one corresponding allocation, plus optional diagnostics.

    Notes
    -----
    This function solves the linear program:

    $$ \\text{minimize} \\quad \\varepsilon $$
    
    subject to:
      
      - $\\sum_{i=1}^n x_i = v(N)$
      - $v(S) - \\sum_{i\\in S} x_i \\le \\varepsilon$ for all proper, non-empty coalitions S

    The returned allocation is one optimal solution; the least-core set may contain
    infinitely many allocations.

    Runtime dependency
    ------------------
    Requires SciPy/HiGHS at runtime (`pip install "tucoop[lp]"`).

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.solutions import least_core
    >>>
    >>> # Simple additive game v(S)=|S|
    >>> g = Game.from_value_function(
    ...     n_players=3,
    ...     value_fn=lambda ps: float(len(ps)),
    ... )
    >>> res = least_core(g)
    >>> res.epsilon
    0.0
    >>> res.x
    [1.0, 1.0, 1.0]
    """

    from ..backends.optional_deps import require_module

    np = require_module("numpy", extra="lp", context="least_core")  # type: ignore

    n = game.n_players
    grand = game.grand_coalition
    vN = float(game.value(grand))

    # Variables: [x0..x_{n-1}, eps]
    c = np.zeros(n + 1, dtype=float)
    c[n] = 1.0

    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([vN], dtype=float)

    rows: list[list[float]] = []
    rhs: list[float] = []
    for S in all_coalitions(n):
        if S == 0 or S == grand:
            continue
        row = [0.0] * (n + 1)
        for i in range(n):
            if S & (1 << i):
                row[i] = -1.0
        row[n] = -1.0  # -eps
        rows.append(row)
        rhs.append(-float(game.value(S)))

    A_ub = np.asarray(rows, dtype=float) if rows else None
    b_ub = np.asarray(rhs, dtype=float) if rhs else None

    # x_i are free; eps is free in theory but optimal eps will be finite.
    bounds = [(None, None)] * n + [(None, None)]

    from ..backends.lp import linprog_solve

    res = linprog_solve(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        context="least_core",
    )

    z = res.x.tolist()
    x = [float(v) for v in z[:n]]
    eps = float(z[n])
    # Snap tiny negative eps from numerical noise.
    if abs(eps) <= tol:
        eps = 0.0

    # Diagnostics
    tight: list[int] = []
    duals: dict[int, float] = {}
    if A_ub is not None and b_ub is not None:
        z_np = np.asarray(z, dtype=float)
        slack = b_ub - (A_ub @ z_np)
        for idx, S in enumerate([S for S in all_coalitions(n) if S not in (0, grand)]):
            if idx >= slack.shape[0]:
                break
            if float(slack[idx]) <= tol:
                tight.append(S)
        tight.sort()

        # HiGHS exposes duals in res.ineqlin.marginals (SciPy >= 1.6-ish).
        marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
        if marg is not None:
            for idx, S in enumerate([S for S in all_coalitions(n) if S not in (0, grand)]):
                if idx >= len(marg):
                    break
                w = float(marg[idx])
                if abs(w) > tol:
                    duals[S] = w

    diag = None
    try:
        from ..diagnostics.linprog_diagnostics import linprog_diagnostics

        diag = linprog_diagnostics(res)
    except Exception:
        diag = None

    return LeastCoreResult(x=x, epsilon=eps, tight=tight or None, dual_weights=duals or None, lp=diag)


def least_core_epsilon_star(game: GameProtocol, *, tol: float = 1e-9) -> float:
    """
    Compute the least-core value epsilon*.

    This is a small convenience wrapper around `least_core` that returns only
    the optimal epsilon value. It exists to avoid duplicating the "compute epsilon*"
    logic in other modules (geometry/diagnostics).

    Parameters
    ----------
    game
        TU game.
    tol
        Numerical tolerance forwarded to `least_core`.

    Returns
    -------
    float
        The least-core value epsilon* (snapped for tiny numerical noise as in
        `least_core`).
    """
    return float(least_core(game, tol=float(tol)).epsilon)


def least_core_point(
    game: GameProtocol,
    *,
    restrict_to_imputation: bool = False,
    tol: float = 1e-9,
    method: SelectionMethod = "chebyshev_center",
) -> LeastCorePointResult:
    """
    Select a single allocation from the least-core set.

    This is a **single-valued selector** built on top of the set-valued least-core
    geometry helper `tucoop.geometry.least_core.LeastCore`.

    Parameters
    ----------
    game
        TU game.
    restrict_to_imputation
        If True, intersect the least-core with the imputation constraints
        ($x_i \\ge v(\\{i\\})$).
    tol
        Numerical tolerance forwarded to the underlying geometry helper.
    method
        Selection rule:

        - ``"chebyshev_center"``: return the Chebyshev center of the least-core polyhedron.
        - ``"any_feasible"``: return any feasible point.

    Returns
    -------
    LeastCorePointResult
        A selected allocation and the corresponding least-core epsilon.

    Raises
    ------
    InvalidGameError
        If the least-core set is empty under the requested restrictions.

    Notes
    -----
    - ``least_core()`` solves an LP and returns *one* optimal point.
    - ``least_core_point()`` selects a canonical point from the least-core set.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.solutions import least_core_point
    >>>
    >>> # Majority game on 3 players
    >>> def majority(ps):
    ...     return 1.0 if len(ps) >= 2 else 0.0
    ...
    >>> g = Game.from_value_function(3, majority)
    >>> p = least_core_point(g, method="chebyshev_center")
    >>> p.epsilon >= 0.0
    True
    >>> len(p.x)
    3
    """
    lc = LeastCore(game, restrict_to_imputation=restrict_to_imputation, tol=float(tol))

    if method == "chebyshev_center":
        cc = lc.chebyshev_center()
        if cc is None:
            raise InvalidGameError("least_core_point: least-core set is empty")
        x_cc, _r = cc
        return LeastCorePointResult(x=[float(v) for v in x_cc], epsilon=float(lc.epsilon), method=str(method))

    if method == "any_feasible":
        x_any = lc.sample_point()
        if x_any is None:
            raise InvalidGameError("least_core_point: least-core set is empty")
        return LeastCorePointResult(x=[float(v) for v in x_any], epsilon=float(lc.epsilon), method=str(method))

    raise InvalidParameterError("method must be 'chebyshev_center' or 'any_feasible'")


__all__ = [
    "LeastCoreResult",
    "LeastCorePointResult",
    "least_core",
    "least_core_epsilon_star",
    "least_core_point",
]
