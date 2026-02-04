"""
Optional NumPy helpers.

This module centralizes optional NumPy imports used by fast paths in the
package. It raises a clear error message instructing how to install extras when
NumPy is missing.
"""

from __future__ import annotations

from typing import Any

from .optional_deps import require_module

def require_numpy(*, context: str | None = None) -> Any:
    """
    Import NumPy as an optional dependency.

    Parameters
    ----------
    context
        Context string used in the error message.

    Returns
    -------
    Any
        The imported ``numpy`` module.

    Raises
    ------
    ImportError
        If NumPy cannot be imported.

    Examples
    --------
    >>> require_numpy()  # doctest: +SKIP
    <module 'numpy' ...>
    """
    ctx = context or "NumPy routines"
    return require_module("numpy", extra="fast", context=ctx)

__all__ = ["require_numpy"]

