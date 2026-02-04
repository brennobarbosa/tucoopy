"""
# Apex (simple voting) games.

An apex game is a weighted voting game with a distinguished **apex** player $a$.
A coalition $S$ is winning if it includes the apex and the other members meet
the quota:

$$
a \\in S \\quad \\text{and} \\quad \\sum_{i \\in S,\\ i \\ne a} w_i \\ge q.
$$

See Also
--------
tucoop.games.weighted_voting.weighted_voting_game

Examples
--------
>>> from tucoop.games.apex import apex_game
>>> g = apex_game(0, [2, 3], quota=3)
>>> g.value(0b11)
1.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game
from ..base.exceptions import InvalidParameterError
from ..base.coalition import all_coalitions


def apex_game(
    apex_player: int,
    weights: Sequence[float],
    quota: float,
    *,
    player_labels: list[str] | None = None,
    winning_value: float = 1.0,
    losing_value: float = 0.0,
) -> Game:
    """
    Construct an **apex game** (simple game).

    Let $a$ be the apex player and let $w_i$ be the weights. A coalition $S$ is
    said to be *winning* if:

    - $a \\in S$, and
    - $\\sum_{i \\in S,\\ i \\ne a} w_i \\ge q$.

    The characteristic function is:
     
    $$
    v(S) = \\begin{cases}
    w_v, & \\text{ if } S \\text{ is winning} \\\\
    l_v, & \\text{ otherwise}
    \\end{cases}
    $$

    where $w_v$ is the winning value and $l_v$ is the losing value.

    Parameters
    ----------
    apex_player : int
        Index of the apex player.
    weights : Sequence[float]
        Weights for each player (excluding apex).
    quota : float
        Quota required for a coalition to win (excluding apex).
    player_labels : list of str, optional
        Optional labels for players.
    winning_value : float, optional
        Value assigned to winning coalitions (default 1.0).
    losing_value : float, optional
        Value assigned to losing coalitions (default 0.0).

    Returns
    -------
    Game
        Cooperative game instance representing the apex game.

    Raises
    ------
    InvalidParameterError
        If fewer than 2 players are provided or `apex_player` is out of range.

    Examples
    --------
    >>> from tucoop.games.apex import apex_game
    >>> g = apex_game(0, [2, 3], 3)
    >>> g.n_players
    2
    >>> g.value(1)
    0.0
    >>> g.value(3)
    1.0
    """
    n = len(weights)
    if n < 2:
        raise InvalidParameterError("apex_game needs at least 2 players")
    a = int(apex_player)
    if a < 0 or a >= n:
        raise InvalidParameterError("apex_player out of range")

    v: dict[int, float] = {0: float(losing_value)}
    for S in all_coalitions(n):
        if S == 0:
            continue
        if not (S & (1 << a)):
            v[S] = float(losing_value)
            continue
        w = 0.0
        for i in range(n):
            if i == a:
                continue
            if S & (1 << i):
                w += float(weights[i])
        v[S] = float(winning_value if w >= quota else losing_value)

    v[0] = 0.0
    return Game(n_players=n, v=v, player_labels=player_labels)
