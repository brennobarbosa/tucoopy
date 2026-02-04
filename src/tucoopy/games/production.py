"""
# Production games.

In a (linear) production game, players own resources and coalitions can pool
their resources to produce output that is valued by fixed prices.

This implementation uses a simplified linear model where the coalition worth is
the total market value of the pooled resources:

$$
v(S) = \\sum_{k} p_k \\left(\\sum_{i \\in S} r_{i,k}\\right),
$$

where $r_{i,k}$ is the amount of resource $k$ owned by player $i$.

Examples
--------
>>> from tucoopy.games.production import production_game
>>> g = production_game(resources=[[1, 0], [0, 2]], prices=[10, 5])
>>> g.value(0b11)
20.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game


def production_game(
    resources: Sequence[Sequence[float]],
    prices: Sequence[float],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Generate a **production game**.

    Players own resources, and coalitions can combine resources to
    produce goods with given prices.

    Parameters
    ----------
    resources : Sequence[Sequence[float]]
        resources[i]\\[k\\] = amount of resource k owned by player i.
    prices : Sequence[float]
        Price per unit of each resource.
    player_labels : list[str] | None, optional
        Optional labels.

    Returns
    -------
    Game
        Production cooperative game.

    Notes
    -----
    - Coalition value is total market value of combined resources.
    - This is a simplified linear production model.

    Examples
    --------
    >>> from tucoopy.games.production import production_game
    >>> g = production_game(resources=[[1, 0], [0, 2]], prices=[10, 5])
    >>> g.value(0b01)  # player 0 alone
    10.0
    >>> g.value(0b11)  # pooled
    20.0
    """
    n = len(resources)
    m = len(prices)

    def value_fn(ps: Sequence[int]) -> float:
        total = [0.0] * m
        for i in ps:
            for k in range(m):
                total[k] += float(resources[i][k])
        return float(sum(total[k] * prices[k] for k in range(m)))

    return Game.from_value_function(
        n_players=n,
        value_fn=value_fn,
        player_labels=player_labels,
    )


__all__ = ["production_game"]
