"""
# Cost game recognizers.

This module provides simple heuristics for recognizing and validating **cost-style**
games, where costs are represented as negative worths (so $v(S) \\leq 0$).

Examples
--------
>>> from tucoop.base.game import Game
>>> from tucoop.properties.cost_games import is_cost_game
>>> g = Game.from_coalitions(n_players=2, values={(): 0.0, (0,): -1.0, (1,): -2.0, (0, 1): -3.0})
>>> is_cost_game(g)
True
"""
from __future__ import annotations

from ..base.types import GameProtocol
from ..base.coalition import all_coalitions
from ..base.exceptions import InvalidGameError
from ._utils import require_max_players


def is_cost_game(game: GameProtocol, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
    """
    Heuristic recognizer for "cost games" represented as negative values.

    Convention used in this library:
    
    - TU games always have $v(\\varnothing)=0$.
    - Many cost-style models store costs as negative worths ($v(S) \\leq 0$).

    This recognizer checks:
    
    - $v(S) \\leq \\epsilon$ for all $S$ (non-positive, up to tolerance)
    - monotone non-increasing: if $S \\subseteq T$ then $v(S) \\geq v(T) - \\epsilon$
    """
    n = game.n_players
    require_max_players(game, max_players=max_players, context="is_cost_game")

    # Non-positive values
    for S in all_coalitions(n):
        if float(game.value(S)) > eps:
            return False

    # Monotone non-increasing (adjacent check): if S ⊆ T then v(S) ≥ v(T).
    for S in all_coalitions(n):
        vS = float(game.value(S))
        for i in range(n):
            bit = 1 << i
            if S & bit:
                continue
            if vS + eps < float(game.value(S | bit)):
                return False
    return True


def validate_cost_game(game: GameProtocol, *, eps: float = 1e-12, max_players: int | None = 12) -> None:
    """
    Validate that a game matches the library's "cost-game" convention.

    Parameters
    ----------
    game
        TU game.
    eps
        Numerical tolerance.
    max_players
        Fail fast if ``game.n_players`` exceeds this bound (check is exponential).

    Raises
    ------
    InvalidGameError
        If the game fails :func:`is_cost_game`.
    """
    if not is_cost_game(game, eps=eps, max_players=max_players):
        raise InvalidGameError("expected a cost-style game (non-positive, monotone non-increasing)")


__all__ = ["is_cost_game", "validate_cost_game"]
