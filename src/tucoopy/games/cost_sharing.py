"""
# Cost-sharing games (via savings).

This module constructs a TU worth game from stand-alone costs $c_i$ and a dense
table of coalition costs $C(S)$ (length $2^n$) using the standard savings
transformation:

$$
v(S) = \\sum_{i \\in S} c_i - C(S).
$$

See Also
--------
tucoopy.games.savings.savings_game

Examples
--------
>>> from tucoopy.games.cost_sharing import cost_sharing_game
>>> g = cost_sharing_game([2, 3], [0, 2, 3, 4])
>>> g.value(0b11)
1.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.coalition import all_coalitions
from ..base.game import Game
from ..base.exceptions import InvalidParameterError
from ._utils import _validate_n_players


def cost_sharing_game(
    individual_costs: Sequence[float],
    coalition_costs: Sequence[float],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct a **cost-sharing game** as a TU worth game (via savings).

    Given:
        
     - individual (stand-alone) costs $c_i$
     - coalition costs $C(S)$ for every coalition mask $S$ (dense, length $2^n$)

    Define the savings game:
    
    $$v(S) = \\sum_{i \\in S} c_i - C(S)$$

    Parameters
    ----------
    individual_costs : Sequence[float]
        Stand-alone costs for each player.
    coalition_costs : Sequence[float]
        Costs for each coalition (length $2^n$).
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the cost-sharing game.

    Raises
    ------
    InvalidParameterError
        If coalition_costs does not have length $2^n$.

    Examples
    --------
    >>> from tucoopy.games.cost_sharing import cost_sharing_game
    >>> g = cost_sharing_game([2, 3], [0, 2, 3, 4])
    >>> g.n_players
    2
    >>> g.value(1)
    0.0
    >>> g.value(3)
    1.0
    """
    n = _validate_n_players(len(individual_costs))
    if len(coalition_costs) != (1 << n):
        raise InvalidParameterError("coalition_costs must have length 2^n")

    c = [float(x) for x in individual_costs]
    C = [float(x) for x in coalition_costs]

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(n):
        if S == 0:
            continue
        s = 0.0
        for i in range(n):
            if S & (1 << i):
                s += c[i]
        v[S] = s - C[S]

    return Game(n_players=n, v=v, player_labels=player_labels)
