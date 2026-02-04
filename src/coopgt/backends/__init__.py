"""
Optional-dependency backends and runtime adapters.

This package contains runtime adapters for optional dependencies (LP backends,
NumPy fast paths, etc.). Type-level contracts live in :mod:`tucoop.base.types`.

Notes
-----
- The canonical LP entrypoint for the rest of the package is
  :func:`tucoop.backends.lp.linprog_solve` (SciPy-based when available).
- This subpackage is mostly an internal boundary: higher-level modules should
  depend on adapter functions, not on SciPy/PuLP APIs directly.
"""

from __future__ import annotations
from .lp import LinprogFailure, LinprogResult, linprog_solve

__all__ = ["LinprogFailure", "LinprogResult", "linprog_solve"]

