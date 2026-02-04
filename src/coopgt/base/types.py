"""
## Shared typing contracts (Protocols).

This module contains small :class:`typing.Protocol` definitions that describe the
minimum interface expected by parts of the library (e.g. LP backends). Keeping
them here avoids circular imports and makes the intended “contract” explicit.

Examples
--------
>>> from tucoop.base.types import GameProtocol
>>> class MyGame:
...     n_players = 2
...     grand_coalition = 3
...     def value(self, coalition_mask: int) -> float:
...         return float(coalition_mask)
>>> g: GameProtocol = MyGame()
>>> g.value(3)
3.0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable, TypeGuard

from .exceptions import InvalidGameError


@runtime_checkable
class GameProtocol(Protocol):
    """
    Minimal protocol for a TU game defined over coalition bitmasks.

    Attributes
    ----------
    n_players : int
        Number of players in the game.

    Methods
    -------
    value(coalition_mask: int) -> float
        Returns the value of the coalition specified by the bitmask.

    Examples
    --------
    >>> class MyGame:
    ...     n_players = 3
    ...     def value(self, coalition_mask: int) -> float:
    ...         return float(coalition_mask)
    >>> g = MyGame()
    >>> g.value(0b011)
    3.0
    """

    @property
    def n_players(self) -> int: ...

    @property
    def grand_coalition(self) -> int: ...

    def value(self, coalition_mask: int) -> float: ...


@runtime_checkable
class TabularGameProtocol(GameProtocol, Protocol):
    """
    Protocol for games that expose a concrete mapping of coalition values.

    This is required by algorithms that need to check completeness
    (i.e. whether all ``2^n`` coalition values are explicitly present).
    """

    v: Mapping[int, float]


def is_tabular_game(game: GameProtocol) -> TypeGuard[TabularGameProtocol]:
    """
    Return True if `game` appears to provide a `.v` mapping.

    Notes
    -----
    This is a structural (duck-typing) check; it does not validate completeness.

    Examples
    --------
    >>> from tucoop.base.types import is_tabular_game
    >>> class MyTabular:
    ...     n_players = 2
    ...     grand_coalition = 3
    ...     v = {0: 0.0, 3: 1.0}
    ...     def value(self, coalition_mask: int) -> float:
    ...         return float(self.v.get(coalition_mask, 0.0))
    >>> is_tabular_game(MyTabular())
    True
    """

    return isinstance(game, TabularGameProtocol)


def require_tabular_game(game: GameProtocol, *, context: str) -> TabularGameProtocol:
    """
    Require that `game` exposes `.v` (tabular characteristic function mapping).

    Raises
    ------
    InvalidGameError
        If the object does not provide a `.v` mapping.

    Examples
    --------
    >>> from tucoop.base.types import require_tabular_game
    >>> class MyGame:
    ...     n_players = 2
    ...     grand_coalition = 3
    ...     def value(self, coalition_mask: int) -> float:
    ...         return 0.0
    >>> require_tabular_game(MyGame(), context="demo")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    tucoop.base.exceptions.InvalidGameError: demo requires a tabular game with a `.v` mapping
    """

    if not is_tabular_game(game):
        raise InvalidGameError(f"{context} requires a tabular game with a `.v` mapping")
    return game


@runtime_checkable
class LPBackend(Protocol):
    """
    Protocol for LP backends (intentionally minimal).

    Methods
    -------
    solve(c: Any, **kwargs: Any) -> Any
        Solves a linear program with given coefficients and options.

    Examples
    --------
    >>> class DummyLP:
    ...     def solve(self, c, **kwargs):
    ...         return 'solved'
    >>> backend = DummyLP()
    >>> backend.solve([1, 2, 3])
    'solved'
    """

    def solve(self, c: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "GameProtocol",
    "LPBackend",
    "TabularGameProtocol",
    "is_tabular_game",
    "require_tabular_game",
]
