"""
# Convexity and concavity checks for TU cooperative games.

This module provides recognizers for:

- **Convex (supermodular)** games, and
- **Concave (submodular)** games.

Notes
-----
Checking convexity/concavity is exponential, but can be done with a local
condition on pairs of players, yielding an ``O(n^2 2^n)`` check.

Examples
--------
>>> from tucoopy.base.game import Game
>>> from tucoopy.properties.convexity import is_convex, is_concave
>>> g = Game.from_coalitions(n_players=2, values={(): 0.0, (0,): 1.0, (1,): 1.0, (0, 1): 3.0})
>>> is_convex(g)
True
>>> is_concave(g)
False
"""
from __future__ import annotations

from ..base.types import GameProtocol
from ..base.coalition import all_coalitions
from ._utils import require_max_players


def is_convex(game: GameProtocol, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
    """
    **Convexity (supermodularity)**: for all $S$,$T$,
    
    $$v(S) + v(T) \\leq v(S \\cup T) + v(S \\cap T)$$

    """
    n = game.n_players
    require_max_players(game, max_players=max_players, context="is_convex")

    # Equivalent local condition for supermodularity:
    # For all S and i<j not in S:
    #   v(S∪{i}) + v(S∪{j}) <= v(S∪{i,j}) + v(S)
    for S in all_coalitions(n):
        vS = float(game.value(S))
        for i in range(n):
            bit_i = 1 << i
            if S & bit_i:
                continue
            Si = S | bit_i
            vSi = float(game.value(Si))
            for j in range(i + 1, n):
                bit_j = 1 << j
                if S & bit_j:
                    continue
                Sj = S | bit_j
                vSj = float(game.value(Sj))
                vSij = float(game.value(Si | bit_j))
                if vSi + vSj > vSij + vS + eps:
                    return False
    return True


def is_concave(game: GameProtocol, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
    """
    **Concavity (submodularity)**: for all $S$,$T$,
    
    $$v(S) + v(T) \\geq v(S \\cup T) + v(S \\cap T)$$

    """
    n = game.n_players
    require_max_players(game, max_players=max_players, context="is_concave")

    # Equivalent local condition for submodularity (reverse inequality).
    for S in all_coalitions(n):
        vS = float(game.value(S))
        for i in range(n):
            bit_i = 1 << i
            if S & bit_i:
                continue
            Si = S | bit_i
            vSi = float(game.value(Si))
            for j in range(i + 1, n):
                bit_j = 1 << j
                if S & bit_j:
                    continue
                Sj = S | bit_j
                vSj = float(game.value(Sj))
                vSij = float(game.value(Si | bit_j))
                if vSi + vSj + eps < vSij + vS:
                    return False
    return True


__all__ = ["is_convex", "is_concave"]
