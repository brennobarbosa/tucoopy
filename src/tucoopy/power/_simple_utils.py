"""
# Internal utilities for power indices on simple games.

This module contains helpers shared across multiple power index implementations,
such as:

- validation that a game is a complete simple game (all $2^n$ coalitions present),
- extracting minimal winning coalitions,
- normalization and validation of integer weights/quota.
"""

from __future__ import annotations

from typing import Sequence

from ..base.types import GameProtocol, require_tabular_game
from ..base.exceptions import InvalidGameError, InvalidParameterError, NotSupportedError
from ..properties.simple_games import validate_simple_game


def require_complete_simple_game(game: GameProtocol, *, max_players: int = 20) -> None:
    """
    Validate that `game` is a complete simple game ($v(S) \\in \\{0,1\\}$ for all $S$),
    with all $2^n$ coalition values explicitly provided.
    """
    validate_simple_game(game)
    n = game.n_players
    if n > max_players:
        raise NotSupportedError(f"requires a complete characteristic function for n<={max_players} (got n={n})")
    tabular = require_tabular_game(game, context="require_complete_simple_game")
    expected = 1 << n
    if len(tabular.v) != expected:
        raise InvalidGameError("requires a complete characteristic function (all coalition values must be provided)")
    present = set(int(m) for m in tabular.v.keys())
    if any(m not in present for m in range(expected)):
        raise InvalidGameError("requires a complete characteristic function (all coalition values must be provided)")


def minimal_winning_coalitions(game: GameProtocol) -> list[int]:
    """
    Minimal winning coalitions for a complete simple game.
    """
    require_complete_simple_game(game)

    n = game.n_players
    out: list[int] = []
    for S in range(1 << n):
        if S == 0:
            continue
        if float(game.value(S)) != 1.0:
            continue
        minimal = True
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if float(game.value(S & ~(1 << i))) == 1.0:
                minimal = False
                break
        if minimal:
            out.append(int(S))
    out.sort()
    return out


def normalize(weights: list[float]) -> list[float]:
    s = float(sum(weights))
    if s <= 0.0:
        return [0.0 for _ in weights]
    return [float(v) / s for v in weights]


def validate_int_weights(weights: Sequence[int], quota: int) -> tuple[list[int], int]:
    w = [int(x) for x in weights]
    if any(x < 0 for x in w):
        raise InvalidParameterError("weights must be non-negative integers")
    q = int(quota)
    if q < 0:
        raise InvalidParameterError("quota must be a non-negative integer")
    return w, q
