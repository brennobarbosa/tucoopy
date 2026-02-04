"""
# Internal utilities for `tucoop.properties`.

This module centralizes small helpers used by multiple property checks, mainly:

- enforcing conservative limits for exponential checks, and
- providing a consistent way to iterate over all coalitions.

It is intentionally internal (prefixed with `_`).
"""

from __future__ import annotations

from collections.abc import Iterator

from ..base.types import GameProtocol
from ..base.coalition import all_coalitions
from ..base.exceptions import NotSupportedError


def require_max_players(game: GameProtocol, *, max_players: int | None, context: str) -> None:
    n = game.n_players
    if max_players is not None and n > max_players:
        raise NotSupportedError(f"{context} is exponential in n; use n<={max_players} (got n={n})")


def iter_all_coalitions(game: GameProtocol, *, max_players: int | None, context: str) -> Iterator[int]:
    require_max_players(game, max_players=max_players, context=context)
    yield from all_coalitions(game.n_players)


__all__ = ["require_max_players", "iter_all_coalitions"]
