"""
# Linear-programming (LP) diagnostics and explanations.

This module contains small, serialization-friendly helpers to extract a stable
subset of information from LP solver results, plus higher-level helpers used by
tucoopy to expose LP-based explanations in analysis outputs.

The low-level extractor `linprog_diagnostics` is designed to work with:

- SciPy/HiGHS ``scipy.optimize.linprog`` result objects, and
- the fallback :class:`tucoopy.backends.lp.LinprogResult` wrapper used by the PuLP backend.

Notes
-----
SciPy is an optional dependency in tucoopy. If SciPy (or the configured LP backend)
is unavailable, higher-level helpers may raise an `ImportError` suggesting
installing extra dependencies (e.g. ``pip install "tucoopy[lp]"``).

Examples
--------
>>> from tucoopy.diagnostics.linprog_diagnostics import LinprogDiagnostics, explain_linprog
>>> d = LinprogDiagnostics(0, "Optimal", 1.0, [1.0, 2.0], [0.0], [0.0], [0.0])
>>> explain_linprog(d)
['status=0', 'fun=1', 'Optimal']
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..base.types import GameProtocol


@dataclass(frozen=True)
class LinprogDiagnostics:
    """
    Diagnostics extracted from SciPy/HiGHS linear programming results.

    Parameters
    ----------
    status : int, optional
        Solver status code.
    message : str, optional
        Solver message.
    fun : float, optional
        Objective function value.
    x : list of float, optional
        Solution vector.
    ineqlin_residual : list of float, optional
        Inequality constraint residuals.
    ineqlin_marginals : list of float, optional
        Inequality constraint multipliers.
    eqlin_marginals : list of float, optional
        Equality constraint multipliers.

    Methods
    -------
    to_dict()
        Returns diagnostics as a dictionary.

    Examples
    --------
    >>> d = LinprogDiagnostics(0, "Optimal", 1.0, [1.0, 2.0], [0.0], [0.0], [0.0])
    >>> d.to_dict()
    {...}
    """
    status: int | None
    message: str | None
    fun: float | None
    x: list[float] | None
    ineqlin_residual: list[float] | None
    ineqlin_marginals: list[float] | None
    eqlin_marginals: list[float] | None

    def to_dict(self) -> dict[str, object]:
        """
        Returns diagnostics as a dictionary.

        Returns
        -------
        dict
            Dictionary representation of diagnostics.

        Examples
        --------
        >>> d = LinprogDiagnostics(0, "Optimal", 1.0, [1.0, 2.0], [0.0], [0.0], [0.0])
        >>> d.to_dict()
        {...}
        """
        return asdict(self)


def linprog_diagnostics(res: Any) -> LinprogDiagnostics:
    """
    Extracts a stable subset of diagnostics from SciPy/HiGHS linprog results.

    Parameters
    ----------
    res : Any
        Result object from linprog solver.

    Returns
    -------
    LinprogDiagnostics
        Diagnostics extracted from the result.

    Examples
    --------
    >>> # res = scipy.optimize.linprog(...)
    >>> diag = linprog_diagnostics(res)  # doctest: +SKIP
    >>> diag.status  # doctest: +SKIP
    0
    """
    status = getattr(res, "status", None)
    message = getattr(res, "message", None)
    fun = getattr(res, "fun", None)
    x = getattr(res, "x", None)
    x_list = [float(v) for v in x.tolist()] if x is not None and hasattr(x, "tolist") else None
    ineqlin = getattr(res, "ineqlin", None)
    eqlin = getattr(res, "eqlin", None)
    ineq_res = getattr(ineqlin, "residual", None) if ineqlin is not None else None
    ineq_mar = getattr(ineqlin, "marginals", None) if ineqlin is not None else None
    eq_mar = getattr(eqlin, "marginals", None) if eqlin is not None else None
    ineq_res_list = (
        [float(v) for v in ineq_res.tolist()]
        if ineq_res is not None and hasattr(ineq_res, "tolist")
        else None
    )
    ineq_mar_list = (
        [float(v) for v in ineq_mar.tolist()]
        if ineq_mar is not None and hasattr(ineq_mar, "tolist")
        else None
    )
    eq_mar_list = (
        [float(v) for v in eq_mar.tolist()] if eq_mar is not None and hasattr(eq_mar, "tolist") else None
    )
    return LinprogDiagnostics(
        status=status,
        message=str(message) if message is not None else None,
        fun=float(fun) if fun is not None else None,
        x=x_list,
        ineqlin_residual=ineq_res_list,
        ineqlin_marginals=ineq_mar_list,
        eqlin_marginals=eq_mar_list,
    )


def explain_linprog(diag: LinprogDiagnostics) -> list[str]:
    """
    Generates a readable summary for UI/debug of LP diagnostics.

    Parameters
    ----------
    diag : LinprogDiagnostics
        Diagnostics extracted from the solver.

    Returns
    -------
    list of str
        Lines of textual summary.

    Examples
    --------
    >>> diag = LinprogDiagnostics(0, "Optimal", 1.0, [1.0, 2.0], [0.0], [0.0], [0.0])
    >>> explain_linprog(diag)
    ['status=0', 'fun=1', 'Optimal']
    """
    lines: list[str] = []
    if diag.status is not None:
        lines.append(f"status={int(diag.status)}")
    if diag.fun is not None:
        lines.append(f"fun={float(diag.fun):.6g}")
    if diag.message:
        lines.append(str(diag.message))
    return lines


def _truncate_list(values: list[Any], max_len: int | None) -> tuple[list[Any], dict[str, Any]]:
    """
    Truncate a list to a maximum number of elements, returning metadata.

    Parameters
    ----------
    values : list
        List of elements.
    max_len : int, optional
        Maximum number of elements to return.

    Returns
    -------
    tuple
        (truncated list, metadata)

    Examples
    --------
    >>> _truncate_list([1, 2, 3], 2)
    ([1, 2], {'count_total': 3, 'count_returned': 2, 'truncated': True})
    """
    if max_len is None:
        return values, {"count_total": len(values), "count_returned": len(values), "truncated": False}
    m = max(0, int(max_len))
    if len(values) <= m:
        return values, {"count_total": len(values), "count_returned": len(values), "truncated": False}
    return values[:m], {"count_total": len(values), "count_returned": m, "truncated": True}


def build_lp_explanations(game: GameProtocol, *, tol: float = 1e-9, max_list: int = 256) -> dict[str, Any]:
    """
    Compute LP-based explanations for cooperative game analysis.

    Parameters
    ----------
    game : Game
        Cooperative game instance.
    tol : float, optional
        Numerical tolerance (default 1e-9).
    max_list : int, optional
        Maximum number of elements in returned lists.

    Returns
    -------
    dict
        Dictionary with balancedness and least core explanations.

    Examples
    --------
    >>> from tucoopy.base.game import Game
    >>> from tucoopy.diagnostics.linprog import build_lp_explanations
    >>> g = Game.from_coalitions(n_players=2, values={(): 0, (0,): 1, (1,): 1, (0, 1): 2})
    >>> build_lp_explanations(g)  # doctest: +SKIP
    {...}
    """
    from ..properties.balancedness import balancedness_check
    from ..solutions.least_core import least_core

    out: dict[str, Any] = {}
    bal = balancedness_check(game, tol=tol)
    weights_list = [{"coalition_mask": int(S), "weight": float(w)} for S, w in bal.weights.items()]
    weights_list.sort(key=lambda r: (-r["weight"], r["coalition_mask"]))
    weights_list, weights_meta = _truncate_list(weights_list, max_list)
    out["balancedness_check"] = {
        "core_nonempty": bool(bal.core_nonempty),
        "objective": float(bal.objective),
        "weights": weights_list,
        "weights_meta": {"computed_by": "tucoopy-py", **weights_meta},
        "lp": bal.lp.to_dict() if bal.lp is not None else None,
        "lp_explanation": explain_linprog(bal.lp) if bal.lp is not None else None,
    }
    lc = least_core(game, tol=tol)
    out["least_core"] = {
        "epsilon": float(lc.epsilon),
        "x": [float(v) for v in lc.x],
        "tight": list(lc.tight) if lc.tight is not None else None,
        "dual_weights": {int(k): float(v) for k, v in (lc.dual_weights or {}).items()} or None,
        "lp": lc.lp.to_dict() if lc.lp is not None else None,
        "lp_explanation": explain_linprog(lc.lp) if lc.lp is not None else None,
    }
    return out


__all__ = [
    "LinprogDiagnostics",
    "linprog_diagnostics",
    "explain_linprog",
    "build_lp_explanations",
]
