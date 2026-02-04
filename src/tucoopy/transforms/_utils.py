"""
# Small helpers for `tucoopy.transforms`.

These utilities are not meant to be a public API; they support higher-level
transformations (e.g. building a dense vector of coalition values for applying a
fast Möbius transform).
"""

from __future__ import annotations

from ..base.coalition import all_coalitions
from ..base.types import GameProtocol


def to_dense_values(game: GameProtocol) -> list[float]:
    """
    Dense representation of $v(S)$ as a list of length $2^n$, indexed by mask.
    """
    out = [0.0] * (1 << game.n_players)
    for mask in all_coalitions(game.n_players):
        out[mask] = game.value(mask)
    return out
