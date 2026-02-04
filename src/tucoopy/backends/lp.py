"""
Linear-programming backend adapter.

This module provides the *public* LP adapter used by the rest of the library.
The default implementation targets SciPy (``scipy.optimize.linprog``) when
available.

If SciPy is not available or you explicitly want an alternative solver, this
module can also use PuLP as a fallback backend (enabled via a separate optional
extra).

Notes
-----
The adapter is intentionally small: other modules should depend only on
:func:`linprog_solve` and the result dataclasses, not directly on SciPy.

Warnings
--------
- This module requires an LP solver backend at runtime.
  - Recommended: install SciPy with `pip install "tucoopy[lp]"` (uses `scipy.optimize.linprog` / HiGHS).
  - Alternative: install PuLP with `pip install "tucoopy[lp_alt]"` (uses CBC via PuLP).
- If neither SciPy nor PuLP is available, functions that require LP will raise a
  `MissingOptionalDependencyError` (from `tucoopy.backends.optional_deps`).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .lp_scipy import SciPyLPBackend
from .optional_deps import require_module
from ..base.exceptions import InvalidParameterError, NotSupportedError

@dataclass(frozen=True)
class LinprogFailure(RuntimeError):
    """
    Exception for linear programming failure.

    Attributes
    ----------
    message : str
        Error message.
    status : int, optional
        Solver status code.

    Examples
    --------
    >>> try:
    ...     raise LinprogFailure("LP error", status=1)
    ... except LinprogFailure as e:
    ...     print(str(e))
    LP error (status=1)
    """
    message: str
    status: int | None = None

    def __str__(self) -> str:  # pragma: no cover
        if self.status is None:
            return self.message
        return f"{self.message} (status={self.status})"

@dataclass
class _VectorLike:
    """
    Structure for LP solution variable vector.

    Attributes
    ----------
    values : list of float
        Variable values.

    Examples
    --------
    >>> v = _VectorLike([1.0, 2.0])
    >>> v.tolist()
    [1.0, 2.0]
    """
    values: list[float]

    def tolist(self) -> list[float]:
        """
        Return the values as a list.
        """
        return list(self.values)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.values)

    def __iter__(self):  # pragma: no cover
        return iter(self.values)

    def __getitem__(self, idx: int) -> float:  # pragma: no cover
        return self.values[idx]

@dataclass
class LinprogResult:
    """
    Result of a linear programming call.

    Attributes
    ----------
    success : bool
        Indicates if the solution was optimal.
    status : int
        Solver status code.
    message : str
        Solver message.
    x : _VectorLike
        Solution variable vector.
    fun : float, optional
        Objective function value.

    Examples
    --------
    >>> LinprogResult(True, 0, "Optimal", _VectorLike([1.0, 2.0]), fun=3.0)
    LinprogResult(success=True, status=0, message='Optimal', x=_VectorLike(values=[1.0, 2.0]), fun=3.0)
    """
    success: bool
    status: int
    message: str
    x: _VectorLike
    fun: float | None = None

def _get_scipy_solver():
    """
    Return the solver function from the SciPy backend.
    """
    return SciPyLPBackend().solve

def _get_pulp():
    """
    Return the PuLP module for the alternative backend.
    """
    return require_module("pulp", extra="lp_alt", context="LP routines")

def _normalize_bounds(bounds: Any, n: int) -> list[tuple[float | None, float | None]]:
    """
    Normalize variable bounds to a format accepted by the solver.

    Parameters
    ----------
    bounds : Any
        Variable bounds.
    n : int
        Number of variables.

    Returns
    -------
    list of tuple
        List of (lb, ub) pairs for each variable.

    Raises
    ------
    InvalidParameterError
        If the bounds format is invalid.
    """
    if bounds is None:
        return [(0.0, None) for _ in range(n)]
    if isinstance(bounds, tuple) and len(bounds) == 2:
        lb, ub = bounds
        return [(lb, ub) for _ in range(n)]
    if isinstance(bounds, list) and len(bounds) > 0 and isinstance(bounds[0], (tuple, list)):
        out = []
        for b in bounds:
            if not isinstance(b, (tuple, list)) or len(b) != 2:
                raise InvalidParameterError("bounds must be None, a (lb, ub) pair, or a list of (lb, ub) pairs")
            out.append((b[0], b[1]))
        if len(out) != n:
            raise InvalidParameterError("bounds must have one (lb, ub) pair per variable")
        return out
    if isinstance(bounds, list) and len(bounds) == 2:
        lb, ub = bounds[0], bounds[1]
        return [(lb, ub) for _ in range(n)]
    raise InvalidParameterError("bounds must be None, a (lb, ub) pair, or a list of (lb, ub) pairs")

def _linprog_solve_pulp(
    c: Any,
    *,
    A_ub: Any = None,
    b_ub: Any = None,
    A_eq: Any = None,
    b_eq: Any = None,
    bounds: Any = None,
    options: dict[str, Any] | None = None,
    require_success: bool = True,
    context: str | None = None,
) -> LinprogResult:
    """
    Solve an LP problem using the PuLP backend.

    Parameters
    ----------
    c : Any
        Objective function coefficients.
    A_ub, b_ub : Any, optional
        Inequality constraint matrix and vector.
    A_eq, b_eq : Any, optional
        Equality constraint matrix and vector.
    bounds : Any, optional
        Variable bounds.
    options : dict, optional
        Additional options (not supported).
    require_success : bool, default=True
        If True, raise exception if no optimal solution is found.
    context : str, optional
        Context message for error.

    Returns
    -------
    LinprogResult
        Result from the PuLP solver.

    Raises
    ------
    LinprogFailure
        If no optimal solution is found.
    NotSupportedError
        If options are passed (not supported).
    """
    pulp = _get_pulp()
    if options:
        raise NotSupportedError("pulp backend does not support options yet")
    c_list = [float(v) for v in c]
    n = len(c_list)
    norm_bounds = _normalize_bounds(bounds, n)
    prob = pulp.LpProblem("linprog", pulp.LpMinimize)
    xs = []
    for i, (lb, ub) in enumerate(norm_bounds):
        xs.append(pulp.LpVariable(f"x{i}", lowBound=lb, upBound=ub))
    prob += pulp.lpSum(c_list[i] * xs[i] for i in range(n))
    if A_ub is not None:
        m = len(A_ub)
        for r in range(m):
            rhs = float(b_ub[r])
            prob += pulp.lpSum(float(A_ub[r][j]) * xs[j] for j in range(n)) <= rhs
    if A_eq is not None:
        m = len(A_eq)
        for r in range(m):
            rhs = float(b_eq[r])
            prob += pulp.lpSum(float(A_eq[r][j]) * xs[j] for j in range(n)) == rhs
    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = prob.solve(solver)
    status_str = pulp.LpStatus.get(status_code, str(status_code))
    success = status_str == "Optimal"
    x_vals = [float(pulp.value(v)) for v in xs]
    fun = float(pulp.value(prob.objective))
    res = LinprogResult(success=success, status=int(status_code), message=str(status_str), x=_VectorLike(x_vals), fun=fun)
    if require_success and not success:
        prefix = f"{context}: " if context else ""
        raise LinprogFailure(prefix + res.message, status=res.status)
    return res

def linprog_solve(
    c: Any,
    *,
    A_ub: Any = None,
    b_ub: Any = None,
    A_eq: Any = None,
    b_eq: Any = None,
    bounds: Any = None,
    backend: str = "scipy",
    method: str = "highs",
    options: dict[str, Any] | None = None,
    require_success: bool = True,
    context: str | None = None,
):
    """
    Wrapper to solve LP problems with SciPy or PuLP backend.

    Parameters
    ----------
    c : Any
        Objective function coefficients.
    A_ub, b_ub : Any, optional
        Inequality constraint matrix and vector.
    A_eq, b_eq : Any, optional
        Equality constraint matrix and vector.
    bounds : Any, optional
        Variable bounds.
    backend : str, default='scipy'
        Backend to use ('scipy' or 'pulp').
    method : str, default='highs'
        Solver method (SciPy only).
    options : dict, optional
        Additional options for the solver.
    require_success : bool, default=True
        If True, raise exception if no optimal solution is found.
    context : str, optional
        Context message for error.

    Returns
    -------
    LinprogResult or SciPy solver result
        Result from the chosen solver.

    Warnings
    --------
    - When `backend="scipy"`, SciPy must be installed (`pip install "tucoopy[lp]"`).
    - When `backend="pulp"`, PuLP must be installed (`pip install "tucoopy[lp_alt]"`).
    - This adapter intentionally exposes a minimal contract; callers should not
      depend on SciPy-specific fields unless they explicitly require the SciPy backend.

    Raises
    ------
    LinprogFailure
        If no optimal solution is found.
    InvalidParameterError
        If backend is unknown or options are invalid.

    Examples
    --------
    >>> linprog_solve([1, 2], A_ub=[[1, 1]], b_ub=[2])  # doctest: +SKIP
    LinprogResult(...)
    """
    backend = str(backend).strip().lower()
    if backend == "pulp":
        return _linprog_solve_pulp(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            options=options,
            require_success=require_success,
            context=context,
        )
    if backend != "scipy":
        raise InvalidParameterError(f"Unknown LP backend: {backend!r}")
    solve = _get_scipy_solver()
    res = solve(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        options=options,
    )
    if require_success and not getattr(res, "success", False):
        msg = str(getattr(res, "message", "linprog failed"))
        status = getattr(res, "status", None)
        prefix = f"{context}: " if context else ""
        raise LinprogFailure(prefix + msg, status=status)
    return res

__all__ = ["LinprogFailure", "LinprogResult", "linprog_solve"]


