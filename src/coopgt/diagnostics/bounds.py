"""
# Bound/box-set diagnostics.

This module defines small dataclasses used to report violations of per-player
lower/upper bounds, plus a reusable container for "box-like" sets that also
require efficiency (e.g. reasonable set).

Examples
--------
>>> from tucoop.diagnostics.bounds import BoundViolation, BoxBoundSetDiagnostics
>>> v = BoundViolation(player=0, kind="lower", bound=0.0, value=-0.1)
>>> d = BoxBoundSetDiagnostics(
...     n_players=2,
...     vN=1.0,
...     sum_x=1.0,
...     efficient=True,
...     in_set=False,
...     lower_bounds=[0.0, 0.0],
...     upper_bounds=[1.0, 1.0],
...     violations=[v],
... )
>>> d.to_dict()["in_set"]
False
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BoundViolation:
    """
    One bound violation for a set defined by per-player bounds.

    Attributes
    ----------
    player
        Player index (0-based).
    kind
        `"lower"` or `"upper"`.
    bound
        The violated bound value.
    value
        The actual value `x[player]`.

    Examples
    --------
    >>> BoundViolation(player=1, kind="upper", bound=1.0, value=1.2).kind
    'upper'
    """

    player: int
    kind: str
    bound: float
    value: float


@dataclass(frozen=True)
class BoxBoundSetDiagnostics:
    """
    Diagnostics for efficient sets with per-player lower/upper bounds.

    Examples
    --------
    >>> d = BoxBoundSetDiagnostics(
    ...     n_players=2,
    ...     vN=1.0,
    ...     sum_x=1.0,
    ...     efficient=True,
    ...     in_set=True,
    ...     lower_bounds=[0.0, 0.0],
    ...     upper_bounds=[1.0, 1.0],
    ...     violations=[],
    ... )
    >>> d.in_set
    True
    """

    n_players: int
    vN: float
    sum_x: float
    efficient: bool
    in_set: bool
    lower_bounds: list[float]
    upper_bounds: list[float]
    violations: list[BoundViolation]

    def to_dict(self) -> dict[str, object]:
        """
        Convert diagnostics to a JSON-serializable dictionary.

        Examples
        --------
        >>> d = BoxBoundSetDiagnostics(
        ...     n_players=2,
        ...     vN=1.0,
        ...     sum_x=1.0,
        ...     efficient=True,
        ...     in_set=True,
        ...     lower_bounds=[0.0, 0.0],
        ...     upper_bounds=[1.0, 1.0],
        ...     violations=[],
        ... )
        >>> d.to_dict()["n_players"]
        2
        """
        return asdict(self)


__all__ = [
    "BoundViolation",
    "BoxBoundSetDiagnostics",
]
