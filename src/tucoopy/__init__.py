"""
# tucoopy (TU cooperative game theory).

This package implements utilities and solution concepts for **transferable-utility
(TU)** cooperative games.

The top-level API is intentionally small and stable. For the full surface area,
import directly from subpackages:

- `tucoopy.base` (core primitives: coalitions + games + config)
- `tucoopy.games` (game generators)
- `tucoopy.solutions` (single-valued solutions: Shapley, nucleolus, kernel, ...)
- `tucoopy.geometry` (set-valued objects: core, epsilon-core, least-core, ...)
- `tucoopy.power` (power indices for simple/voting games)
- `tucoopy.transforms` (game transforms: dual, sum/diff, Myerson restriction, ...)
- `tucoopy.properties` (recognizers / property checks)
- `tucoopy.io` (JSON/spec I/O and schemas)
- `tucoopy.diagnostics` (allocation + set diagnostics)
- `tucoopy.backends` (optional dependency adapters)

Optional dependencies
---------------------
Some features rely on optional dependencies and will raise a tucoopy-specific
exception with installation instructions when missing:

- LP routines: install `tucoopy[lp]` (SciPy backend)
- Matplotlib visualization: install `tucoopy[viz]`

Examples
--------
Create a simple 2-player TU game and compute the Shapley value:

>>> from tucoopy import Game
>>> from tucoopy.solutions import shapley_value
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
