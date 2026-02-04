"""
Optional dependency importer.

This module provides :func:`require_module`, a small helper that imports an
optional dependency and raises a user-friendly error message indicating the
corresponding ``pip`` extra to install.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..base.exceptions import MissingOptionalDependencyError

def require_module(module: str, *, extra: str, context: str | None = None) -> Any:
    """
    Import an optional dependency with a clear installation hint.

    Parameters
    ----------
    module
        Module name to import (e.g. ``"numpy"``).
    extra
        Extra name to show in the install hint (e.g. ``"viz"``, ``"lp"``).
    context
        Optional context string used in the error message.

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    ImportError
        If the module cannot be imported.

    Examples
    --------
    >>> require_module(\"matplotlib\", extra=\"viz\")  # doctest: +SKIP
    <module 'matplotlib' ...>
    """
    try:
        return importlib.import_module(module)
    except Exception as e:
        prefix = f"{context}: " if context else ""
        raise MissingOptionalDependencyError(
            prefix + f"Missing optional dependency {module!r}. Install with: pip install \"tucoopy[{extra}]\""
        ) from e

__all__ = ["require_module"]

