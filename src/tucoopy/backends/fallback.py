"""
Fallback helpers for missing optional dependencies.

The canonical exception type is :class:`~tucoopy.base.exceptions.MissingOptionalDependencyError`.
"""

from __future__ import annotations
from typing import Any

from tucoopy.base.exceptions import MissingOptionalDependencyError


def raise_missing(*, extra: str, context: str | None = None) -> Any:
    """
    Raise :class:`~tucoopy.base.exceptions.MissingOptionalDependencyError`.

    Parameters
    ----------
    extra
        Name of the pip extra (e.g. ``lp`` or ``viz``).
    context
        Optional context string to prepend to the error message.
    """
    msg = f"missing optional dependency (install with: tucoopy[{extra}])"
    if context:
        msg = f"{context}: {msg}"
    raise MissingOptionalDependencyError(msg)


__all__ = ["raise_missing"]
