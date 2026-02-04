"""
# Airport (cost) games.

In the standard airport cost game, each player $i$ has a runway requirement
$r_i$, and the cost of a coalition is

$$
c(S) = \\max_{i \\in S} r_i.
$$

This module returns a TU *worth* game by negating the cost:
$v(S) = -c(S)$.

See Also
--------
tucoop.games.mst.mst_game
tucoop.games.flow.flow_game

Examples
--------
>>> from tucoop.games.airport import airport_game
>>> g = airport_game([3, 5, 7])
>>> g.value(0b011)
-5.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game
from ..base.coalition import all_coalitions
from ._utils import _validate_n_players


def airport_game(
    runway_requirements: Sequence[float],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct an **airport** (cost) game as a TU worth game.

    Standard airport cost game:
    
    $$c(S) = \\max_{i \\in S} r_i$$

    This constructor returns a worth game by taking $v(S) = -c(S)$, so that
    cost allocations can be represented as negative payoffs.

    Parameters
    ----------
    runway_requirements : Sequence[float]
        List of runway requirements for each player.
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the airport cost game.

    Raises
    ------
    InvalidParameterError
        If fewer than 1 player is provided.

    Examples
    --------
    >>> from tucoop.games.airport import airport_game
    >>> g = airport_game([3, 5, 7])
    >>> g.n_players
    3
    >>> g.value(1)
    -3.0
    >>> g.value(3)
    -5.0
    >>> g.value(7)
    -7.0
    """
    n = _validate_n_players(len(runway_requirements))

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(n):
        if S == 0:
            continue
        mx = 0.0
        for i in range(n):
            if S & (1 << i):
                mx = max(mx, float(runway_requirements[i]))
        v[S] = -mx

    return Game(n_players=n, v=v, player_labels=player_labels)
