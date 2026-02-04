"""
# tucoop (TU cooperative game theory).

This package implements utilities and solution concepts for **transferable-utility
(TU)** cooperative games.

The top-level API is intentionally small and stable. For the full surface area,
import directly from subpackages:

- `tucoop.base` (core primitives: coalitions + games + config)
- `tucoop.games` (game generators)
- `tucoop.solutions` (single-valued solutions: Shapley, nucleolus, kernel, ...)
- `tucoop.geometry` (set-valued objects: core, epsilon-core, least-core, ...)
- `tucoop.power` (power indices for simple/voting games)
- `tucoop.transforms` (game transforms: dual, sum/diff, Myerson restriction, ...)
- `tucoop.properties` (recognizers / property checks)
- `tucoop.io` (JSON/spec I/O and schemas)
- `tucoop.diagnostics` (allocation + set diagnostics)
- `tucoop.backends` (optional dependency adapters)

Optional dependencies
---------------------
Some features rely on optional dependencies and will raise a tucoop-specific
exception with installation instructions when missing:

- LP routines: install `tucoop[lp]` (SciPy backend)
- Matplotlib visualization: install `tucoop[viz]`

Examples
--------
Create a simple 2-player TU game and compute the Shapley value:

>>> from tucoop import Game
>>> from tucoop.solutions import shapley_value
>>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
>>> shapley_value(g)
[0.5, 0.5]
"""

from .base import Game, mask_from_players
from .games import glove_game, weighted_voting_game
from .solutions import shapley_value
from .geometry import Core

# Optional convenience re-exports (still stable, but kept minimal)
from .solutions import nucleolus  # requires SciPy at runtime when called

__all__ = [
    "Game",
    "mask_from_players",
    "glove_game",
    "weighted_voting_game",
    "Core",
    "shapley_value",
    "nucleolus",
]
