"""
## Core game representation.

This module defines `tucoopy.base.game.Game`, the main representation of a
transferable-utility (TU) cooperative game via its characteristic function
$v(S)$.

Conventions
-----------
- Players are indexed ``0..n_players-1``.
- Coalitions are encoded as integer bitmasks (see `tucoopy.base.coalition`).
- Missing coalition values in the mapping are treated as ``0.0``.

Examples
--------
Create a 2-player game where only the grand coalition has value 1:

>>> from tucoopy import Game
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
>>> g.value(0)
0.0
>>> g.value(3)
1.0

Using "Pythonic" coalition keys:

>>> g = Game.from_coalitions(n_players=2, values={(): 0, (0,): 0, (1,): 0, (0, 1): 1})
>>> g.value(g.grand_coalition)
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, overload
from collections.abc import Iterable as AbcIterable, Mapping

from .coalition import all_coalitions, mask_from_players, players
from .coalition import grand_coalition
from .exceptions import InvalidGameError


@dataclass(frozen=True)
class Game:
    """
    TU cooperative game defined by its characteristic function $v(S)$.

    Coalitions are encoded as bitmasks over players $0 \\ldots n-1$.

    Parameters
    ----------
    n_players
        Number of players `n`.
    v
        Mapping from coalition masks (ints) or iterables of player indices to coalition values.
        Missing coalitions are treated as value 0.0 by :meth:`value`.
    player_labels
        Optional display labels for players (length must match `n_players`).

    Notes
    -----
    
    - TU convention: $v( \\varnothing) = 0$ is enforced.
    - The grand coalition mask is `(1 << n_players) - 1`.

    Examples
    --------
    Author with Pythonic coalition keys:

    >>> g = Game.from_coalitions(
    ...     n_players=3,
    ...     values={(): 0.0, (0,): 1.0, (1,): 1.0, (2,): 1.0, (0, 1, 2): 4.0},
    ... )
    >>> g.value(g.grand_coalition)
    4.0
    """

    n_players: int
    v: dict[int, float]
    player_labels: list[str] | None = None

    def __post_init__(self) -> None:
        if self.n_players < 1:
            raise InvalidGameError("n_players must be >= 1")
        normalized: dict[int, float] = {}
        for key, val in list(self.v.items()):
            if isinstance(key, int):
                mask = key
            else:
                mask = mask_from_players(key)
            normalized[int(mask)] = float(val)
        object.__setattr__(self, "v", normalized)
        if 0 not in self.v:
            raise InvalidGameError("v must include coalition 0 (empty set)")
        if self.v[0] != 0:
            raise InvalidGameError("v(0) must be 0 for TU games")
        max_mask = (1 << self.n_players) - 1
        for mask in self.v.keys():
            if mask < 0 or mask > max_mask:
                raise InvalidGameError(f"coalition mask out of range: {mask}")
        if self.player_labels is not None and len(self.player_labels) != self.n_players:
            raise InvalidGameError("player_labels length must match n_players")

    def value(self, coalition_mask: int) -> float:
        """
        Return the coalition worth $v(S)$.

        Parameters
        ----------
        coalition_mask
            Coalition mask (bitmask).

        Returns
        -------
        float
            Coalition value ``v(S)``.

        Notes
        -----
        Missing coalitions in ``self.v`` are treated as value ``0.0``.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g = Game.from_coalitions(n_players=2, values={0:0, 3:1})
        >>> g.value(0b01)  # missing singleton defaults to 0.0
        0.0
        >>> g.value(0b11)
        1.0
        """
        return float(self.v.get(coalition_mask, 0.0))

    # Convenience methods (thin wrappers around tucoopy.properties).
    # Keep core representation minimal while offering OOP ergonomics.

    def is_superadditive(self, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
        """
        Check whether the game is superadditive.

        This is a convenience wrapper around :func:`tucoopy.properties.is_superadditive`.

        Parameters
        ----------
        eps
            Numerical tolerance.
        max_players
            Safety limit. If the game has more than this many players, the check
            raises an error instead of iterating all coalitions.

        Returns
        -------
        bool
            True if the game is superadditive; False otherwise.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 1, 2: 1, 3: 3})
        >>> g.is_superadditive()
        True
        """
        from ..properties import is_superadditive

        return is_superadditive(self, eps=eps, max_players=max_players)

    def is_convex(self, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
        """
        Check whether the game is convex (supermodular).

        This is a convenience wrapper around :func:`tucoopy.properties.is_convex`.

        Parameters
        ----------
        eps
            Numerical tolerance.
        max_players
            Safety limit. If the game has more than this many players, the check
            raises an error instead of iterating all coalitions.

        Returns
        -------
        bool
            True if the game is convex; False otherwise.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
        >>> g.is_convex()
        True
        """
        from ..properties import is_convex

        return is_convex(self, eps=eps, max_players=max_players)

    def is_essential(self, *, eps: float = 1e-12) -> bool:
        """
        Check whether the game is essential.

        A game is essential if ``v(N) > sum_i v({i})`` (up to tolerance).

        This is a convenience wrapper around :func:`tucoopy.properties.is_essential`.

        Parameters
        ----------
        eps
            Numerical tolerance.

        Returns
        -------
        bool
            True if the game is essential; False otherwise.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
        >>> g.is_essential()
        True
        """
        from ..properties import is_essential

        return is_essential(self, eps=eps)

    def is_simple_game(self, *, tol: float = 0.0, max_players: int | None = 20) -> bool:
        """
        Check whether the game is a simple (0/1-valued) game.

        This is a convenience wrapper around :func:`tucoopy.properties.is_simple_game`.

        Parameters
        ----------
        tol
            Tolerance for comparing values to 0 and 1.
        max_players
            Safety limit. If the game has more than this many players, the check
            raises an error instead of iterating all coalitions.

        Returns
        -------
        bool
            True if the game is a simple game; False otherwise.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
        >>> g.is_simple_game()
        True
        """
        from ..properties import is_simple_game

        return is_simple_game(self, tol=tol, max_players=max_players)

    def validate_simple_game(self, *, tol: float = 0.0, max_players: int | None = 20) -> None:
        """
        Validate that the game is a well-formed simple game.

        This is a convenience wrapper around :func:`tucoopy.properties.validate_simple_game`.

        Parameters
        ----------
        tol
            Tolerance for comparing values to 0 and 1.
        max_players
            Safety limit. If the game has more than this many players, the check
            raises an error instead of iterating all coalitions.

        Raises
        ------
        ValueError
            If the game is not a valid simple game.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
        >>> g.validate_simple_game()
        """
        from ..properties import validate_simple_game

        validate_simple_game(self, tol=tol, max_players=max_players)

    @property
    def grand_coalition(self) -> int:
        """
        Return the grand coalition mask (all players).

        Returns
        -------
        int
            Mask ``(1 << n_players) - 1``.

        Examples
        --------
        >>> from tucoopy import Game
        >>> Game.from_coalitions(n_players=3, values={0:0, 7:1}).grand_coalition
        7
        """
        return grand_coalition(self.n_players)

    def with_values(self, v: dict[int, float]) -> "Game":
        """
        Return a copy of the game with a replaced value table.

        Parameters
        ----------
        v
            New characteristic function values (by coalition mask).

        Returns
        -------
        Game
            A new game with the same ``n_players`` and (if present) the same
            ``player_labels``.

        Examples
        --------
        >>> from tucoopy import Game
        >>> g1 = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> g2 = g1.with_values({0: 0, 3: 2})
        >>> g1.value(0b11), g2.value(0b11)
        (1.0, 2.0)
        """
        return Game(n_players=self.n_players, v=v, player_labels=self.player_labels)

    @overload
    @staticmethod
    def from_coalitions(
        *,
        n_players: int,
        values: Mapping[int, float],
        player_labels: list[str] | None = None,
        require_complete: bool = False,
    ) -> "Game": ...

    @overload
    @staticmethod
    def from_coalitions(
        *,
        n_players: int,
        values: Mapping[AbcIterable[int], float],
        player_labels: list[str] | None = None,
        require_complete: bool = False,
    ) -> "Game": ...

    @staticmethod
    def from_coalitions(
        *,
        n_players: int,
        values: Mapping[int, float] | Mapping[AbcIterable[int], float],
        player_labels: list[str] | None = None,
        require_complete: bool = False,
    ) -> "Game":
        """
        Convenience constructor for a TU game from coalition values.

        Accepts coalition keys as either:
        
        - int bitmasks, or
        - iterables of player indices (e.g. ``(0, 2)`` or ``frozenset({1})``).

        Parameters
        ----------
        n_players : int
            Number of players.
        values : Mapping[int, float] or Mapping[Iterable[int], float]
            Coalition values, indexed by bitmask or iterable of player indices.
        player_labels : list of str, optional
            Optional display labels for players.
        require_complete : bool, default=False
            If True, require all $2^n$ coalitions to be specified.

        Returns
        -------
        Game
            Instantiated game object.

        Raises
        ------
        InvalidCoalitionError
            If a coalition key uses invalid player indices.
        InvalidGameError
            If parameters are invalid or required coalitions are missing.

        Examples
        --------
        >>> Game.from_coalitions(n_players=2, values={(): 0, (0,): 1, (1,): 2, (0, 1): 3})
        Game(n_players=2, v={0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0})
        """
        v: dict[int, float] = {}
        for k, val in values.items():
            mask = k if isinstance(k, int) else mask_from_players(k)
            v[int(mask)] = float(val)

        # Ensure v(0)=0 exists for TU games.
        v.setdefault(0, 0.0)

        game = Game(n_players=n_players, v=v, player_labels=player_labels)

        if require_complete:
            for mask in all_coalitions(n_players):
                if mask not in game.v:
                    raise InvalidGameError(f"Missing coalition value for mask={mask}")

        return game

    @staticmethod
    def from_value_function(
        *,
        n_players: int,
        value_fn: Callable[[Sequence[int]], float],
        player_labels: list[str] | None = None,
        include_empty: bool = True,
    ) -> "Game":
        """
        Build a tabular TU game from a Python function defined on coalitions.

        This helper enumerates all coalitions and calls ``value_fn(players(S))``.

        Parameters
        ----------
        n_players
            Number of players.
        value_fn
            Function that receives a *list* of player indices and returns the
            coalition value.
        player_labels
            Optional display labels for players.
        include_empty
            If False, do not call ``value_fn`` on the empty coalition (the value
            is still normalized to ``v(0)=0``).

        Returns
        -------
        Game
            A tabular game storing all coalition values.

        Notes
        -----
        This is an ``O(2^n)`` constructor. Prefer :class:`ValueFunctionGame` when
        you do not want to materialize all coalitions.

        Examples
        --------
        A "size" game (value equals coalition size):

        >>> from tucoopy.base.game import Game
        >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        >>> g.value(0b101)
        2.0
        """
        v: dict[int, float] = {}
        for mask in all_coalitions(n_players):
            if mask == 0 and not include_empty:
                continue
            v[mask] = float(value_fn(players(mask, n_players=n_players)))

        v.setdefault(0, 0.0)
        return Game(n_players=n_players, v=v, player_labels=player_labels)


TabularGame = Game


class ValueFunctionGame:
    """
    TU game backed by a value function ``v(mask)`` with memoization.

    This is useful when $v(S)$ is expensive or you don't want to enumerate all $2^n$
    coalitions up front.

    Notes
    -----
    
    - Missing coalitions are not assumed to be $0$; the function defines them.
    - $v(0)=0$ is enforced (returned regardless of the function).

    Examples
    --------
    Basic usage with a custom value function:

    >>> def v(mask):
    ...     # Example: value is the number of players in the coalition
    ...     return bin(mask).count('1')
    >>> g = ValueFunctionGame(n_players=3, value_fn=v)
    >>> g.value(0b011)  # Players 0 and 1
    2.0
    >>> g.value(g.grand_coalition)
    3.0

    With player labels:

    >>> g = ValueFunctionGame(n_players=2, value_fn=lambda m: 10*m, player_labels=["A", "B"])
    >>> g.value(0b01)  # Only player A
    10.0
    >>> g.value(0b11)  # Both
    30.0
    """

    def __init__(
        self,
        *,
        n_players: int,
        value_fn: Callable[[int], float],
        player_labels: list[str] | None = None,
    ) -> None:
        """
        Initialize a TU game backed by a cached value function.

        Parameters
        ----------
        n_players : int
            Number of players `n`.
        value_fn : Callable[[int], float]
            Function that receives a coalition mask (int) and returns its value.
        player_labels : list[str] | None, optional
            Optional player labels.
        """
        if n_players < 1:
            raise InvalidGameError("n_players must be >= 1")
        self.n_players = int(n_players)
        self._value_fn = value_fn
        self.player_labels = player_labels
        self._cache: dict[int, float] = {0: 0.0}

    @property
    def grand_coalition(self) -> int:
        """
        Return the grand coalition mask (all players).

        Returns
        -------
        int
            Mask of the grand coalition.
        """
        return grand_coalition(self.n_players)

    @property
    def v(self) -> dict[int, float]:
        """
        Return the cache of already computed coalition values.

        Returns
        -------
        dict[int, float]
            Cache of evaluated coalition values.
        """
        return self._cache

    def value(self, coalition_mask: int) -> float:
        """
        Return the value of the coalition specified by the mask.

        Parameters
        ----------
        coalition_mask : int
            Coalition mask.

        Returns
        -------
        float
            Coalition value.
        """
        mask = int(coalition_mask)
        if mask == 0:
            return 0.0
        if mask in self._cache:
            return float(self._cache[mask])
        val = float(self._value_fn(mask))
        self._cache[mask] = val
        return val

    def with_cache(self, v: dict[int, float]) -> "ValueFunctionGame":
        """
        Replace the internal cache of coalition values.

        Parameters
        ----------
        v : dict[int, float]
            New coalition value cache.

        Returns
        -------
        ValueFunctionGame
            The same object with the updated cache.
        """
        self._cache = {int(k): float(val) for k, val in v.items()}
        self._cache.setdefault(0, 0.0)
        return self
