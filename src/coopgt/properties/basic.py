"""
# Basic property checks for TU cooperative games.

This module provides quick recognizers for common properties of TU games:

- Normalized: $v(\\varnothing)=0$,
- Monotone: if $S \\subseteq T$ then $v(S) \\leq v(T)$,
- Superadditive: if $S$ and $T$ are disjoint then $v(S)+v(T) \\leq v(S \\cup T)$,
- Essential: $v(N) > \\sum_{i=1}^{n} v(\\{i\\})``.

Notes
-----
Some checks are exponential in the number of players. Functions therefore accept
``max_players`` to fail fast for large games.

Examples
--------
>>> from tucoop.base.game import Game
>>> from tucoop.properties.basic import is_normalized, is_monotone
>>> g = Game.from_coalitions(n_players=2, values={(): 0.0, (0, 1): 1.0})
>>> is_normalized(g)
True
>>> is_monotone(g)
True
"""
from __future__ import annotations

from ..base.coalition import all_coalitions, grand_coalition, subcoalitions
from ..base.types import GameProtocol
from ._utils import require_max_players


def is_normalized(game: GameProtocol, *, eps: float = 0.0) -> bool:
    """
    **Normalized game**: $v(\\varnothing) = 0$.

    Notes
    -----
    :class:`~tucoop.base.game.Game` enforces $v(\\varnothing)=0$, so this is mostly
    provided for completeness and for defensive checks when ingesting external
    game specifications.
    """
    return abs(float(game.value(0))) <= float(eps)


def is_monotone(game: GameProtocol, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
    """
    Check whether a TU game is **monotone**.

    Monotonicity means that adding players to a coalition cannot decrease its
    worth:

    $$S \\subseteq T \\implies v(S) \\leq v(T).$$

    Parameters
    ----------
    game
        TU game.
    eps
        Numerical tolerance.
    max_players
        Fail fast if ``game.n_players`` exceeds this bound (check is exponential).
    """
    n = game.n_players
    require_max_players(game, max_players=max_players, context="is_monotone")

    # Adjacent coalition check: it is enough to verify monotonicity for one-player
    # expansions S -> S ∪ {i}. This reduces complexity to O(n 2^n).
    for S in all_coalitions(n):
        vS = float(game.value(S))
        for i in range(n):
            bit = 1 << i
            if S & bit:
                continue
            if vS > float(game.value(S | bit)) + eps:
                return False
    return True


def is_superadditive(game: GameProtocol, *, eps: float = 1e-12, max_players: int | None = 12) -> bool:
    """
    Check whether a TU game is **superadditive**.

    Superadditivity requires that disjoint coalitions do not lose value by
    merging:

    $$S \\cap T = \\varnothing \\implies v(S) + v(T) \\leq v(S \\cup T).$$

    Parameters
    ----------
    game
        TU game.
    eps
        Numerical tolerance.
    max_players
        Fail fast if ``game.n_players`` exceeds this bound (check is exponential).
    """
    n = game.n_players
    require_max_players(game, max_players=max_players, context="is_superadditive")

    # Iterate only disjoint pairs by enumerating subcoalitions of the complement.
    # This reduces work from ~4^n to ~3^n checks (still exponential but much cheaper).
    N = grand_coalition(n)
    for S in all_coalitions(n):
        vS = float(game.value(S))
        comp = N ^ S
        for T in subcoalitions(comp):
            if vS + float(game.value(T)) > float(game.value(S | T)) + eps:
                return False
    return True


def is_essential(game: GameProtocol, *, eps: float = 1e-12) -> bool:
    """
    Check whether a TU game is **essential**.

    A game is essential if the grand coalition creates strictly more value than
    the sum of singleton coalitions:

    $$v(N) > \\sum_i v(\\{i\\}).$$

    Parameters
    ----------
    game
        TU game.
    eps
        Numerical tolerance.
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    s = 0.0
    for i in range(n):
        s += float(game.value(1 << i))
    return vN > s + eps


__all__ = [
    "is_normalized",
    "is_monotone",
    "is_superadditive",
    "is_essential",
]
