"""
# Game transformations.

This package contains **pure transformations** of TU cooperative games and their
representations, such as:

- Algebraic operations (scale/shift/affine),
- Game combinations (add/subtract),
- Derived games (dual, subgames, restrictions),
- Communication constraints (Myerson restriction),
- Basis transforms (Möbius transform / Harsanyi dividends).

The functions exported here are meant to be the stable public surface of
`tucoopy.transforms`. Internal helpers live in modules prefixed with `_` and are
not re-exported.
"""

from ._utils import to_dense_values
from .algebra import affine_game, scale_game, shift_game
from .harsanyi import harsanyi_dividends
from .mobius import inverse_mobius_transform, mobius_transform
from .combine import add_games, sub_games
from .communication import myerson_restriction
from .derived import dual_game, restrict_to_players, subgame

__all__ = [
    "harsanyi_dividends",
    "mobius_transform",
    "inverse_mobius_transform",
    "to_dense_values",
    "scale_game",
    "shift_game",
    "affine_game",
    "add_games",
    "sub_games",
    "myerson_restriction",
    "dual_game",
    "subgame",
    "restrict_to_players",
]
