"""
# Glove games.

In a glove game, each player $i$ owns $L_i$ left gloves and $R_i$ right gloves.
The coalition value is proportional to the number of pairs that can be formed:

$$
v(S) = u \\min\\left(\\sum_{i \\in S} L_i, \\sum_{i \\in S} R_i\\right),
$$

where $u$ is the unit value per pair.

Examples
--------
>>> from tucoop.games.glove import glove_game
>>> g = glove_game([1, 0], [0, 1])
>>> g.value(0b11)
1.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game
from ..base.coalition import all_coalitions
from ..base.exceptions import InvalidParameterError
from ._utils import _validate_n_players


def glove_game(
    left_gloves: Sequence[int],
    right_gloves: Sequence[int],
    *,
    unit_value: float = 1.0,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct a **glove game** (TU).

    Each player $i$ owns $L_i$ left gloves and $R_i$ right gloves. A coalition's value is:
    
    $$v(S) = u_v \\min\\left(\\sum_{i \\in S} L_i, \\sum_{i \\in S} R_i\\right)$$

    where $u_v$ is the unit value.

    Parameters
    ----------
    left_gloves : Sequence[int]
        Number of left gloves owned by each player.
    right_gloves : Sequence[int]
        Number of right gloves owned by each player.
    unit_value : float, optional
        Value per pair of gloves (default 1.0).
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the glove game.

    Raises
    ------
    InvalidParameterError
        If left_gloves and right_gloves have different lengths or fewer than 1 player.

    Examples
    --------
    >>> from tucoop.games.glove import glove_game
    >>> g = glove_game([1, 0], [0, 1])
    >>> g.n_players
    2
    >>> g.value(3)
    1.0
    """
    if len(left_gloves) != len(right_gloves):
        raise InvalidParameterError("left_gloves and right_gloves must have the same length")
    n = _validate_n_players(len(left_gloves))

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(n):
        if S == 0:
            continue
        L = 0
        R = 0
        for i in range(n):
            if S & (1 << i):
                L += int(left_gloves[i])
                R += int(right_gloves[i])
        v[S] = float(unit_value) * float(min(L, R))

    return Game(n_players=n, v=v, player_labels=player_labels)
