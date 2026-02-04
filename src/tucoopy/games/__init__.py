"""
# Game generators (`tucoopy.games`).

This subpackage contains convenience constructors for common transferable-utility
(TU) cooperative games. Each function returns a `tucoopy.base.game.Game`
with a tabular characteristic function $v$ (coalitions encoded as bitmasks).

The modules are organized by classic families:

- Simple / voting games: `weighted_voting_game`, `apex_game`
- Canonical TU examples: `glove_game`, `unanimity_game`
- Cost / operations-research games (returned as worth games): `airport_game`,
  `mst_game`, `flow_game`, `bankruptcy_game`, `savings_game`,
  `cost_sharing_game`, `assignment_game`

Notes
-----
- Most generators enumerate all coalitions and are therefore exponential in
  ``n_players``. Use them mainly for small-to-medium games.
- The TU convention $v(\\varnothing)=0$ is enforced by `tucoopy.base.game.Game`.

Examples
--------
>>> from tucoopy.games import glove_game, weighted_voting_game
>>> g1 = glove_game([1, 0], [0, 1])
>>> g1.n_players
2
>>> g2 = weighted_voting_game([2, 1, 1], quota=3)
>>> g2.value(0b011)
1.0
"""

from .glove import glove_game
from .weighted_voting import weighted_voting_game
from .airport import airport_game
from .bankruptcy import bankruptcy_game
from .cost_sharing import cost_sharing_game
from .assignment import assignment_game
from .flow import OwnedEdge, flow_game
from .mst import mst_game
from .savings import savings_game
from .unanimity import unanimity_game
from .apex import apex_game

__all__ = [
    "glove_game",
    "weighted_voting_game",
    "airport_game",
    "bankruptcy_game",
    "cost_sharing_game",
    "assignment_game",
    "OwnedEdge",
    "flow_game",
    "mst_game",
    "savings_game",
    "unanimity_game",
    "apex_game",
]
