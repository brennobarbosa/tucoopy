"""
# Bankruptcy games.

Bankruptcy games model the division of a limited estate $E$ among players with
claims $c_i$. The standard characteristic function is:

$$
v(S) = \\max\\left(0, E - \\sum_{i \\notin S} c_i\\right).
$$

See Also
--------
tucoop.games.savings.savings_game

Examples
--------
>>> from tucoop.games.bankruptcy import bankruptcy_game
>>> g = bankruptcy_game(10, [4, 7])
>>> g.value(0b01)
3.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game
from ..base.coalition import all_coalitions, grand_coalition
from ._utils import _validate_n_players


def bankruptcy_game(
    estate: float,
    claims: Sequence[float],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct a **bankruptcy game** (TU) as a worth game.

    The standard definition of the characteristic function is:
    
    $$v(S) = \\max\\left(0, E - \\sum_{i \\notin S} c_i\\right)$$
    
    where $E$ is the estate and $c_i$ are claims.

    Parameters
    ----------
    estate : float
        Total estate to be divided.
    claims : Sequence[float]
        Claims of each player.
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the bankruptcy game.

    Raises
    ------
    InvalidParameterError
        If fewer than 1 player is provided.

    Examples
    --------
    >>> from tucoop.games.bankruptcy import bankruptcy_game
    >>> g = bankruptcy_game(10, [4, 7])
    >>> g.n_players
    2
    >>> g.value(1)
    3.0
    >>> g.value(3)
    10.0
    """
    n = _validate_n_players(len(claims))
    E = float(estate)
    c = [float(x) for x in claims]

    v: dict[int, float] = {0: 0.0}
    grand = grand_coalition(n)

    for S in all_coalitions(n):
        if S == 0:
            continue
        comp_sum = 0.0
        comp = grand ^ S
        for i in range(n):
            if comp & (1 << i):
                comp_sum += c[i]
        v[S] = max(0.0, E - comp_sum)

    return Game(n_players=n, v=v, player_labels=player_labels)
