"""
# Solution dispatcher.

This module provides `solve`, a single entry point that dispatches to the individual
solution concept implementations under `tucoop.solutions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError


Method: TypeAlias = Literal[
    "shapley",
    "shapley_sample",
    "banzhaf",
    "normalized_banzhaf",
    "nucleolus",
    "prenucleolus",
    "least_core",
    "least_core_point",
    "modiclus",
    "tau",
    "prekernel",
    "kernel",
    "gately",
    "least_squares",
    "myerson",
    "owen",
]


@dataclass(frozen=True)
class SolveResult:
    """
    Result container for solution methods.

    Attributes
    ----------
    method : str
        Name of the solution method used.
    x : list[float]
        Payoff vector (allocation/imputation) returned by the method.
    meta : dict[str, object] | None
        Optional metadata returned by the solver, such as:
        - LP iterations,
        - epsilon values,
        - solver diagnostics,
        - structural information.

    Notes
    -----
    This lightweight container standardizes the output of all solution
    methods exposed through `solve`, enabling uniform downstream
    processing, logging, or visualization.
    """
    method: str
    x: list[float]
    meta: dict[str, object] | None = None


def solve(game: GameProtocol, *, method: Method = "shapley", **kwargs: Any) -> SolveResult:
    """
    Unified dispatcher for cooperative game solution concepts.

    This function provides a single entry point to compute a wide range of
    **point solutions** (allocations/imputations) for TU cooperative games.

    The desired method is selected by name, and any additional keyword
    arguments required by the method can be passed through ``**kwargs``.

    Parameters
    ----------
    game : GameProtocol
        TU cooperative game.
    method : Method, default="shapley"
        Name of the solution concept to compute. Supported values include:

        - "shapley", "shapley_sample"
        - "banzhaf", "normalized_banzhaf"
        - "nucleolus", "prenucleolus"
        - "least_core", "least_core_point"
        - "modiclus"
        - "tau"
        - "prekernel", "kernel"
        - "gately"
        - "least_squares"
        - "myerson"
        - "owen"

    **kwargs
        Additional arguments forwarded to the specific solver. Examples:

        - ``least_core_point``: ``selection``, ``restrict_to_imputation``, ``tol``
        - ``least_squares``: ``x0=[...]`` (initial imputation)
        - ``myerson``: ``edges=[(u,v), ...]``, ``max_players``, ``require_complete``
        - ``owen``: ``unions=[[...], ...]``, ``max_players``, ``require_complete``

    Returns
    -------
    SolveResult
        Container with the solution vector and optional metadata.

    Raises
    ------
    InvalidParameterError
        If the method name is unknown or required parameters are missing/invalid.

    Notes
    -----
    - This function does **not** implement the algorithms itself; it dispatches
      to the appropriate module under `tucoop.solutions`.
    - The goal is to offer a stable, high-level API for experimentation,
      scripting, and integration with analysis/visualization tools.
    - Metadata returned in ``SolveResult.meta`` may include solver diagnostics,
      LP iteration counts, or epsilon values for geometric methods.

    Examples
    --------
    >>> res = solve(g, method="shapley")
    >>> res.x

    >>> res = solve(g, method="nucleolus")
    >>> res.meta["lp_rounds"]

    >>> res = solve(g, method="myerson", edges=[(0,1), (1,2)])
    >>> res.x
    """
    m = str(method).strip().lower()

    if m == "shapley":
        from .shapley import shapley_value

        x = shapley_value(game)
        return SolveResult(method=m, x=[float(v) for v in x])
    
    if m == "shapley_sample":
        from .shapley import shapley_value_sample

        x, stderr = shapley_value_sample(game, n_samples=100)
        return SolveResult(method=m, x=[float(v) for v in x], meta={"standard_error": stderr})

    if m == "banzhaf":
        from .banzhaf import banzhaf_value

        x = banzhaf_value(game)
        return SolveResult(method=m, x=[float(v) for v in x])

    if m == "normalized_banzhaf":
        from .banzhaf import normalized_banzhaf_value

        x = normalized_banzhaf_value(game)
        return SolveResult(method=m, x=[float(v) for v in x])

    if m == "nucleolus":
        from .nucleolus import nucleolus

        nu_res = nucleolus(game)
        return SolveResult(method=m, x=[float(v) for v in nu_res.x], meta={"lp_rounds": len(nu_res.lp_rounds or [])})

    if m == "prenucleolus":
        from .nucleolus import prenucleolus

        pr_res = prenucleolus(game)
        return SolveResult(method=m, x=[float(v) for v in pr_res.x], meta={"lp_rounds": len(pr_res.lp_rounds or [])})

    if m == "least_core":
        from .least_core import least_core

        lc_res = least_core(game)
        lc_meta: dict[str, object] = {"epsilon": float(lc_res.epsilon)}
        if lc_res.lp is not None:
            lc_meta["lp"] = lc_res.lp.to_dict()
        return SolveResult(method=m, x=[float(v) for v in lc_res.x], meta=lc_meta)

    if m == "least_core_point":
        from .least_core import least_core_point
        from .least_core import SelectionMethod

        restrict_to_imputation = bool(kwargs.get("restrict_to_imputation", False))
        tol = float(kwargs.get("tol", 1e-9))
        sel_method_raw = str(kwargs.get("selection", "chebyshev_center")).strip().lower()
        if sel_method_raw not in ("chebyshev_center", "any_feasible"):
            raise InvalidParameterError("least_core_point selection must be 'chebyshev_center' or 'any_feasible'")
        sel_method = cast(SelectionMethod, sel_method_raw)
        lcpt_res = least_core_point(
            game,
            restrict_to_imputation=restrict_to_imputation,
            tol=tol,
            method=sel_method,
        )
        return SolveResult(
            method=m,
            x=[float(v) for v in lcpt_res.x],
            meta={"epsilon": float(lcpt_res.epsilon), "selection": str(sel_method)},
        )

    if m == "modiclus":
        from .modiclus import modiclus

        mod_res = modiclus(game, tol=float(kwargs.get("tol", 1e-9)))
        mod_meta: dict[str, object] = {"lp_rounds": int(len(mod_res.levels))}
        return SolveResult(method=m, x=[float(v) for v in mod_res.x], meta=mod_meta)

    if m == "tau":
        from .tau import tau_value

        x = tau_value(game)
        return SolveResult(method=m, x=[float(v) for v in x])

    if m == "gately":
        from .gately import gately_point

        tol = float(kwargs.get("tol", 1e-12))
        g_res = gately_point(game, tol=tol)
        return SolveResult(method=m, x=[float(v) for v in g_res.x], meta={"d": float(g_res.d), "tol": float(tol)})

    if m == "least_squares":
        from .least_squares import least_squares_imputation

        x0 = kwargs.get("x0", None)
        if x0 is None:
            raise InvalidParameterError("least_squares requires x0=[...]")
        if not isinstance(x0, (list, tuple)):
            raise InvalidParameterError("least_squares requires x0 as a list/tuple")
        x = least_squares_imputation(game, [float(v) for v in x0])
        return SolveResult(method=m, x=[float(v) for v in x], meta={"feasible": True})

    if m == "myerson":
        from .myerson import myerson_value

        edges = kwargs.get("edges", None)
        if edges is None:
            raise InvalidParameterError("myerson requires edges=[(u,v), ...]")
        max_players = int(kwargs.get("max_players", 16))
        require_complete = bool(kwargs.get("require_complete", True))
        my_res = myerson_value(game, edges=edges, max_players=max_players, require_complete=require_complete)
        return SolveResult(method=m, x=[float(v) for v in my_res.x], meta=my_res.meta)

    if m == "owen":
        from .owen import owen_value

        unions = kwargs.get("unions", None)
        if unions is None:
            raise InvalidParameterError("owen requires unions=[[...], ...] or unions=[mask, ...]")
        max_players = int(kwargs.get("max_players", 16))
        require_complete = bool(kwargs.get("require_complete", True))
        ow_res = owen_value(game, unions=unions, max_players=max_players, require_complete=require_complete)
        return SolveResult(method=m, x=[float(v) for v in ow_res.x], meta=ow_res.meta)

    if m == "prekernel":
        from .kernel import prekernel

        pk_res = prekernel(game)
        return SolveResult(method=m, x=[float(v) for v in pk_res.x])

    if m == "kernel":
        from .kernel import kernel

        k_res = kernel(game)
        return SolveResult(method=m, x=[float(v) for v in k_res.x])

    raise InvalidParameterError(f"unknown method={method!r}")


__all__ = ["SolveResult", "solve"]
