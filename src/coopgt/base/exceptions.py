"""
## Shared exception types for `tucoop`.

The project aims to raise a small set of well-named exceptions from core layers
instead of leaking `ValueError`/`RuntimeError` everywhere. This makes it easier
for users to catch and handle errors consistently.

Examples
--------
>>> from tucoop.base.exceptions import InvalidGameError
>>> try:
...     raise InvalidGameError("bad game")
... except InvalidGameError as e:
...     str(e)
'bad game'
"""

from __future__ import annotations


class tucoopError(Exception):
    """
    Base error for tucoop.

    This is the root exception for all errors raised by the tucoop package.

    Examples
    --------
    >>> try:
    ...     raise tucoopError("Generic Error")
    ... except tucoopError as e:
    ...     print(type(e))
    <class 'tucoop.base.exceptions.tucoopError'>
    """


class InvalidGameError(tucoopError, ValueError):
    """
    Raised when a game object is invalid or inconsistent.

    Examples
    --------
    >>> try:
    ...     raise InvalidGameError("Invalid game")
    ... except InvalidGameError as e:
    ...     print(type(e))
    <class 'tucoop.base.exceptions.InvalidGameError'>
    """


class InvalidCoalitionError(tucoopError, ValueError):
    """
    Raised when coalition inputs (bitmasks, player indices) are invalid.

    Examples
    --------
    >>> from tucoop.base.exceptions import InvalidCoalitionError
    >>> try:
    ...     raise InvalidCoalitionError("invalid coalition mask")
    ... except InvalidCoalitionError as e:
    ...     str(e)
    'invalid coalition mask'
    """


class InvalidSpecError(tucoopError, ValueError):
    """
    Raised when a JSON/spec-like object is invalid or inconsistent.

    This is typically raised by :mod:`tucoop.io`.

    Examples
    --------
    >>> from tucoop.base.exceptions import InvalidSpecError
    >>> try:
    ...     raise InvalidSpecError("invalid spec")
    ... except InvalidSpecError as e:
    ...     str(e)
    'invalid spec'
    """


class InvalidParameterError(tucoopError, ValueError):
    """
    Raised when an API parameter is invalid (wrong value/range/shape).

    Examples
    --------
    >>> from tucoop.base.exceptions import InvalidParameterError
    >>> try:
    ...     raise InvalidParameterError("bad parameter")
    ... except InvalidParameterError as e:
    ...     str(e)
    'bad parameter'
    """


class NotSupportedError(tucoopError, ValueError, NotImplementedError):
    """
    Raised when a requested operation is not supported by the implementation.

    Examples
    --------
    >>> from tucoop.base.exceptions import NotSupportedError
    >>> try:
    ...     raise NotSupportedError("not implemented")
    ... except NotSupportedError as e:
    ...     str(e)
    'not implemented'
    """


class BackendError(tucoopError, RuntimeError):
    """
    Raised when a backend fails or violates its contract.

    This is typically used for failures in optional backends (LP, NumPy helpers),
    or when a backend returns an incomplete/invalid result structure.

    Examples
    --------
    >>> from tucoop.base.exceptions import BackendError
    >>> try:
    ...     raise BackendError("backend failed")
    ... except BackendError as e:
    ...     str(e)
    'backend failed'
    """


class ConvergenceError(tucoopError, RuntimeError):
    """
    Raised when an iterative algorithm fails to converge within limits.

    Examples
    --------
    >>> from tucoop.base.exceptions import ConvergenceError
    >>> try:
    ...     raise ConvergenceError("did not converge")
    ... except ConvergenceError as e:
    ...     str(e)
    'did not converge'
    """


class MissingOptionalDependencyError(tucoopError, ImportError):
    """
    Raised when an optional dependency is required but not installed.

    The error message should typically include the corresponding extra, e.g.
    ``pip install "tucoop[lp]"`` or ``pip install "tucoop[viz]"``.

    Examples
    --------
    >>> from tucoop.base.exceptions import MissingOptionalDependencyError
    >>> try:
    ...     raise MissingOptionalDependencyError("install tucoop[lp]")
    ... except MissingOptionalDependencyError as e:
    ...     str(e)
    'install tucoop[lp]'
    """


__all__ = [
    "tucoopError",
    "BackendError",
    "ConvergenceError",
    "InvalidCoalitionError",
    "InvalidGameError",
    "InvalidParameterError",
    "InvalidSpecError",
    "MissingOptionalDependencyError",
    "NotSupportedError",
]
