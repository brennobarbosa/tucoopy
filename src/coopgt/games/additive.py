"""
# Additive TU games.

An additive game is defined by player weights $(w_i)$ and the characteristic
function

$$
v(S) = \\sum_{i \\in S} w_i.
$$

These games are the simplest TU games; notably, the Shapley value equals the
weight vector.

See Also
--------
tucoop.solutions.shapley.shapley_value

Examples
--------
>>> from tucoop.games.additive import additive_game
>>> g = additive_game([1.0, 2.0, 3.0])
>>> g.value(0b101)  # players {0,2}
4.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game


def additive_game(
    weights: Sequence[float],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Generate an **additive TU game**.

    The characteristic function is:

    $$
    v(S) = \\sum_{i \\in S} w_i.
    $$

    Parameters
    ----------
    weights : Sequence[float]
        Player weights.
    player_labels : list[str] | None, optional
        Optional labels for players.

    Returns
    -------
    Game
        Additive cooperative game.

    Notes
    -----
    - This is the simplest class of TU games.
    - The Shapley value equals the weight vector.

    Examples
    --------
    >>> from tucoop.games.additive import additive_game
    >>> g = additive_game([1.0, 2.0, 3.0])
    >>> g.n_players
    3
    >>> g.value(0b011)
    3.0
    """
    n = len(weights)

    def value_fn(ps: Sequence[int]) -> float:
        return float(sum(weights[i] for i in ps))

    return Game.from_value_function(
        n_players=n,
        value_fn=value_fn,
        player_labels=player_labels,
    )


__all__ = ["additive_game"]
