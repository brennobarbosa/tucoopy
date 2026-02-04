"""
SciPy backend for linear programming (``scipy.optimize.linprog``).

This module provides :class:`SciPyLPBackend`, an implementation of the internal
LP adapter using SciPy's ``linprog``.

The dependency on SciPy is optional and is typically enabled via
``pip install "tucoopy[lp]"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .optional_deps import require_module

@dataclass(frozen=True)
class SciPyLPBackend:
    """
    SciPy `optimize.linprog` backend.

    Attributes
    ----------
    extra
        Extra name used in install hints (default: ``"lp"``).

    Methods
    -------
    solve(...)
        Solve a linear program via ``scipy.optimize.linprog``.

    Examples
    --------
    >>> backend = SciPyLPBackend()
    >>> res = backend.solve([1, 2], A_ub=[[1, 1]], b_ub=[2])  # doctest: +SKIP
    >>> res.success  # doctest: +SKIP
    True
    """

    extra: str = "lp"

    def solve(
        self,
        c: Any,
        *,
        A_ub: Any = None,
        b_ub: Any = None,
        A_eq: Any = None,
        b_eq: Any = None,
        bounds: Any = None,
        method: str | None = "highs",
        options: dict[str, Any] | None = None,
    ) -> Any:
        """
        Solve a linear program using ``scipy.optimize.linprog``.

        Parameters
        ----------
        c
            Objective coefficients.
        A_ub, b_ub
            Inequality constraint matrix and RHS.
        A_eq, b_eq
            Equality constraint matrix and RHS.
        bounds
            Variable bounds accepted by SciPy.
        method
            SciPy method (default: ``"highs"``).
        options
            Extra SciPy options.

        Returns
        -------
        Any
            The SciPy result object.

        Raises
        ------
        ImportError
            If SciPy cannot be imported.

        Examples
        --------
        >>> backend = SciPyLPBackend()
        >>> res = backend.solve([1, 2], A_ub=[[1, 1]], b_ub=[2])  # doctest: +SKIP
        >>> res.success  # doctest: +SKIP
        True
        """
        optimize = require_module("scipy.optimize", extra=self.extra, context="LP routines")
        linprog = optimize.linprog
        return linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options=options,
        )


__all__ = ["SciPyLPBackend"]


