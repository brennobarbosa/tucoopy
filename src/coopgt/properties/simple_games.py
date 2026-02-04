"""
# Recognizers for simple games and weighted voting representations.

A **simple game** is a TU game where every coalition is either losing or winning
and the worth is binary:

$v(S) \\in \\{0, 1\\}$ for all coalitions $S$.

This module provides:

- `is_simple_game` / `validate_simple_game` (binary-valued check),
- `find_integer_weighted_voting_representation` (small-n brute force),
- `is_weighted_voting_game`.

Examples
--------
>>> from tucoop.base.game import Game
>>> from tucoop.properties.simple_games import is_simple_game
>>> # 3-player majority: any coalition of size >= 2 is winning.
>>> values = {(): 0.0, (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0, (0, 1, 2): 1.0}
>>> game = Game.from_coalitions(n_players=3, values=values)
>>> is_simple_game(game)
True
"""
from __future__ import annotations

from itertools import product

from ..base.types import GameProtocol
from ..base.coalition import all_coalitions
from ..base.exceptions import InvalidGameError
from .basic import is_monotone
from ._utils import require_max_players


def is_simple_game(game: GameProtocol, *, tol: float = 0.0, max_players: int | None = 20) -> bool:
    """
    Return `True` iff $v(S)$ is (approximately) in $\\{0, 1\\}$ for all coalitions $S$.
    """
    n = game.n_players
    require_max_players(game, max_players=max_players, context="is_simple_game")

    for S in all_coalitions(n):
        v = float(game.value(S))
        if abs(v - 0.0) <= tol or abs(v - 1.0) <= tol:
            continue
        return False
    return True


def validate_simple_game(game: GameProtocol, *, tol: float = 0.0, max_players: int | None = 20) -> None:
    """
    Validate that game is a (TU) simple game: $v(S) \\in \\{0,1\\}$ for all $S$.
    """
    if not is_simple_game(game, tol=tol, max_players=max_players):
        raise InvalidGameError("expected a simple game (all coalition values must be 0 or 1)")


def find_integer_weighted_voting_representation(
    game: GameProtocol,
    *,
    max_weight: int = 10,
    max_players: int | None = 10,
) -> tuple[list[int], int] | None:
    """
    Try to find an integer weighted voting representation $(w, q)$.
    This is an exact brute-force search intended for small $n$.
    Returns (`weights`, `quota`) if successful, else `None`.
    """
    validate_simple_game(game, tol=0.0, max_players=max_players)
    if not is_monotone(game, eps=0.0, max_players=max_players):
        return None

    n = game.n_players
    if n < 1:
        return None

    # Partition coalitions into winning/losing.
    winning: list[int] = []
    losing: list[int] = []
    for S in all_coalitions(n):
        if float(game.value(S)) >= 0.5:
            winning.append(int(S))
        else:
            losing.append(int(S))

    # Need at least one winning coalition for a meaningful representation.
    if not winning:
        return None

    # Brute weights in [0..max_weight], exclude all-zero.
    for w in product(range(max_weight + 1), repeat=n):
        if all(x == 0 for x in w):
            continue
        win_sums = []
        for S in winning:
            s = 0
            for i in range(n):
                if S & (1 << i):
                    s += w[i]
            win_sums.append(s)

        lose_sums = []
        for S in losing:
            s = 0
            for i in range(n):
                if S & (1 << i):
                    s += w[i]
            lose_sums.append(s)

        min_win = min(win_sums)
        max_lose = max(lose_sums) if lose_sums else -1

        if max_lose < min_win:
            # Integer quota works.
            q = max_lose + 1
            return [int(x) for x in w], int(q)

    return None


def is_weighted_voting_game(
    game: GameProtocol,
    *,
    max_weight: int = 10,
    max_players: int | None = 10,
) -> bool:
    """
    Recognize whether a complete simple game can be represented as a weighted voting game.
    For now this is an exact brute-force test for small $n$ only.
    """
    return find_integer_weighted_voting_representation(game, max_weight=max_weight, max_players=max_players) is not None


__all__ = [
    "is_simple_game",
    "validate_simple_game",
    "find_integer_weighted_voting_representation",
    "is_weighted_voting_game",
]
