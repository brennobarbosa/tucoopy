"""
## Core primitives (base layer).

The `tucoopy.base` package defines the fundamental building blocks used across
the project:

- Coalition bitmask utilities (:mod:`tucoopy.base.coalition`)
- Core TU game representations (:class:`~tucoopy.base.game.Game`)
- Shared defaults (:mod:`tucoopy.base.config`)

This module re-exports the most commonly used names for convenience.

Examples
--------
Iterate coalitions and build a small tabular TU game:

>>> from tucoopy.base import all_coalitions, Game
>>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
>>> list(all_coalitions(g.n_players))
[0, 1, 2, 3]
"""
from .coalition import (
    Coalition,
    all_coalitions,
    coalition_sum,
    coalition_sums,
    grand_coalition,
    mask_from_players,
    players,
    size,
    subcoalitions,
)
from .game import Game, TabularGame, ValueFunctionGame

__all__ = [
    "Coalition",
    "all_coalitions",
    "subcoalitions",
    "players",
    "size",
    "grand_coalition",
    "mask_from_players",
    "coalition_sum",
    "coalition_sums",
    "Game",
    "TabularGame",
    "ValueFunctionGame",
]
