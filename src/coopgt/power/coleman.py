"""
# Coleman indices for complete simple games.

This module implements Coleman's system-level and player-level measures:

- collectivity power to act (fraction of winning coalitions),
- power to prevent action (criticality among winning coalitions),
- power to initiate action (pivotality among losing coalitions).
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._simple_utils import require_complete_simple_game


def coleman_collectivity_power_to_act(game: GameProtocol) -> float:
    """
    Compute **Coleman's collectivity power to act** for a complete simple game.

    This quantity measures how often the group, as a whole, is able to take
    collective action. It is defined as the fraction of coalitions that are
    winning:

    $$
    C = \\frac{\\#\\{ S \\subseteq N \\mid v(S) = 1 \\}}{2^n}.
    $$

    Parameters
    ----------
    game : Game
        A *complete* simple game.

    Returns
    -------
    float
        The collectivity power to act.

    Notes
    -----
    - This is not a player-level index, but a **system-level measure** of how
      permissive the decision rule is.
    - A high value means many coalitions can produce action; a low value means
      the decision rule is restrictive.

    Examples
    --------
    >>> coleman_collectivity_power_to_act(g)
    0.5
    """
    require_complete_simple_game(game)

    n = game.n_players
    winning = 0
    for S in range(1 << n):
        if float(game.value(S)) == 1.0:
            winning += 1
    return float(winning) / float(1 << n)


def coleman_prevent_index(game: GameProtocol) -> list[float]:
    """
    Compute **Coleman's power to prevent action** for each player.

    For a winning coalition $S$, a player $i \\in S$ is said to have
    *preventive power* if removing $i$ makes the coalition losing:

    $$
    v(S) = 1 \\quad \\text{and} \\quad v(S \\setminus \\{i\\}) = 0.
    $$

    The index of player $i$ is the fraction of winning coalitions in which
    $i$ is critical:

    $$
    C_i^{\\text{prevent}} =
    \\frac{\\#\\{ S \\mid v(S)=1,\\ i \\text{ critical in } S \\}}
         {\\#\\{ S \\mid v(S)=1 \\}}.
    $$

    Parameters
    ----------
    game : Game
        A *complete* simple game.

    Returns
    -------
    list[float]
        Coleman preventive index for each player.

    Notes
    -----
    - This measures how often a player can **block** collective action.
    - Closely related to the Banzhaf index, but normalized by the number of
      winning coalitions instead of the number of all coalitions.

    Examples
    --------
    >>> coleman_prevent_index(g)
    [0.4, 0.3, 0.3]
    """
    require_complete_simple_game(game)

    n = game.n_players
    W = 0
    swings = [0] * n
    for S in range(1 << n):
        if float(game.value(S)) != 1.0:
            continue
        W += 1
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if float(game.value(S & ~(1 << i))) == 0.0:
                swings[i] += 1

    if W == 0:
        return [0.0 for _ in range(n)]
    return [float(swings[i]) / float(W) for i in range(n)]


def coleman_initiate_index(game: GameProtocol) -> list[float]:
    """
    Compute **Coleman's power to initiate action** for each player.

    For a losing coalition $S$, a player $i \\notin S$ is said to have
    *initiating power* if adding $i$ makes the coalition winning:

    $$
    v(S) = 0 \\quad \\text{and} \\quad v(S \\cup \\{i\\}) = 1.
    $$

    The index of player $i$ is the fraction of losing coalitions in which
    $i$ can turn the outcome into a winning one:

    $$
    C_i^{\\text{initiate}} =
    \\frac{\\#\\{ S \\mid v(S)=0,\\ i \\text{ pivotal for } S \\}}
         {\\#\\{ S \\mid v(S)=0 \\}}.
    $$

    Parameters
    ----------
    game : Game
        A *complete* simple game.

    Returns
    -------
    list[float]
        Coleman initiating index for each player.

    Notes
    -----
    - This measures how often a player can **create** collective action.
    - Dual to the preventive index: one measures blocking power, the other
      initiating power.

    Examples
    --------
    >>> coleman_initiate_index(g)
    [0.6, 0.2, 0.2]
    """
    require_complete_simple_game(game)

    n = game.n_players
    L = 0
    swings = [0] * n
    for S in range(1 << n):
        if float(game.value(S)) != 0.0:
            continue
        L += 1
        for i in range(n):
            if S & (1 << i):
                continue
            if float(game.value(S | (1 << i))) == 1.0:
                swings[i] += 1

    if L == 0:
        return [0.0 for _ in range(n)]
    return [float(swings[i]) / float(L) for i in range(n)]


__all__ = ["coleman_collectivity_power_to_act", "coleman_prevent_index", "coleman_initiate_index"]
