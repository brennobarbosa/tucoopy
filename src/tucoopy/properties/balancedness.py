"""
# Balancedness (Bondareva–Shapley) checks via linear programming.

This module implements an LP-based check of **balancedness**, which by the
Bondareva–Shapley theorem is equivalent to **core non-emptiness** for TU games.

Notes
-----
The LP has one variable per non-empty, non-grand coalition, i.e. ``2^n - 2``.
This grows quickly with ``n`` and typically requires an LP backend (SciPy HiGHS).

Examples
--------
>>> from tucoopy.base.game import Game
>>> from tucoopy.properties.balancedness import balancedness_check
>>> g = Game.from_coalitions(n_players=2, values={(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): 1.0})
>>> result = balancedness_check(g)
>>> result.core_nonempty
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..base.config import DEFAULT_LP_TOL
from ..base.coalition import all_coalitions
from ..base.exceptions import BackendError
from ..base.types import GameProtocol

if TYPE_CHECKING:  # pragma: no cover
    from ..diagnostics.linprog_diagnostics import LinprogDiagnostics


@dataclass(frozen=True)
class BalancednessResult:
    """
    Result of a Bondareva–Shapley balancedness check.
    """

    core_nonempty: bool
    objective: float
    weights: dict[int, float]
    lp: LinprogDiagnostics | None = None


def balancedness_check(game: GameProtocol, *, tol: float = DEFAULT_LP_TOL) -> BalancednessResult:
    """
    Bondareva–Shapley theorem (LP check for core non-emptiness).

    Solve:    
    
    $$\\text{maximize } \\sum_{\\{S \\neq N, S \\neq \\varnothing\\}} \\lambda_S v(S)$$
    
    s.t.    

    $$
    \\begin{cases}
    \\sum\\limits_{S \\ni i} \\lambda_S = 1 \\text{ for each player } i \\\\
    \\lambda_S \\geq 0 
    \\end{cases}
    $$

    If $\\argmax > v(N) + \\text{tol}$, the core is empty and $\\lambda$ is a certificate.

    Warning
    ---
    Requires: SciPy (`pip install "tucoopy[lp]"`)
    """
    from ..backends.optional_deps import require_module

    np = require_module("numpy", extra="lp", context="balancedness_check")  # type: ignore

    n = game.n_players
    N = game.grand_coalition
    vN = float(game.value(N))

    coalitions: list[int] = []
    for S in all_coalitions(n):
        if S == 0 or S == N:
            continue
        coalitions.append(S)

    m = len(coalitions)
    if m == 0:
        return BalancednessResult(core_nonempty=True, objective=0.0, weights={})

    # Objective: maximize sum lambda_S v(S) => minimize -sum lambda_S v(S)
    c = np.zeros(m, dtype=float)
    for j, S in enumerate(coalitions):
        c[j] = -float(game.value(S))

    # Balance constraints: for each i, sum_{S contains i} lambda_S = 1
    A_eq = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j, S in enumerate(coalitions):
            if S & (1 << i):
                A_eq[i, j] = 1.0
    b_eq = np.ones(n, dtype=float)

    bounds = [(0.0, None)] * m

    from ..backends.lp import linprog_solve

    res = linprog_solve(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs", context="balancedness_check")

    fun = getattr(res, "fun", None)
    if fun is None:
        raise BackendError("balancedness_check: LP backend did not provide objective value (fun)")
    x_raw = getattr(res, "x", None)
    if x_raw is None:
        raise BackendError("balancedness_check: LP backend did not provide primal solution (x)")
    if hasattr(x_raw, "tolist"):
        lamb = [float(v) for v in x_raw.tolist()]
    else:
        lamb = [float(v) for v in list(x_raw)]
    obj = float(-float(fun))
    weights = {coalitions[j]: lamb[j] for j in range(m) if lamb[j] > tol}

    core_nonempty = obj <= vN + tol
    diag = None
    try:
        from ..diagnostics.linprog_diagnostics import linprog_diagnostics

        diag = linprog_diagnostics(res)
    except Exception:
        diag = None

    return BalancednessResult(core_nonempty=core_nonempty, objective=obj, weights=weights, lp=diag)


__all__ = ["BalancednessResult", "balancedness_check"]
