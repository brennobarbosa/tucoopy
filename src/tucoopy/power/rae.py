"""
# Rae satisfaction index.

The Rae index measures how often a player is "satisfied" by a coalition outcome:
winning when included, or losing when excluded.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._simple_utils import require_complete_simple_game


def rae_index(game: GameProtocol) -> list[float]:
    """
    Compute the **Rae index** (satisfaction index) for a complete simple game.

    In a simple game, coalitions are either winning or losing ($v(S) \\in \\{0,1\\}$).
    The Rae index measures how often a player is **satisfied** with the outcome
    of a coalition, assuming all coalitions are equally likely.

    A player $i$ is considered satisfied with a coalition $S$ if:

    - $S$ is winning and $i \\in S$, or
    - $S$ is losing and $i \\notin S$.

    The Rae index of player $i$ is therefore:

    $$
    R_i = \\frac{1}{2^n}
          \\Big(
            \\#\\{ S \\mid v(S)=1,\\ i \\in S \\}
            +
            \\#\\{ S \\mid v(S)=0,\\ i \\notin S \\}
          \\Big).
    $$

    Parameters
    ----------
    game : Game
        A *complete* simple game, i.e. a game where:
        - all $2^n$ coalitions are explicitly defined, and
        - values are in {0,1}.

    Returns
    -------
    list[float]
        Rae index for each player (length `n_players`).

    Raises
    ------
    InvalidGameError
        If the game is not a complete simple game (checked by
        :func:`require_complete_simple_game`).
    NotSupportedError
        If `n_players` exceeds the supported limit for completeness checks.

    Notes
    -----
    - The Rae index can be interpreted as the probability that a randomly
      selected coalition outcome agrees with player $i$'s participation
      (winning with the coalition, or losing outside it).
    - Unlike power indices such as Shapley–Shubik or Banzhaf, the Rae index
      measures *satisfaction* rather than *pivotality*.
    - There is a known relationship between the Rae index and the Banzhaf index.

    Examples
    --------
    >>> rae = rae_index(g)
    >>> len(rae) == g.n_players
    True
    """
    require_complete_simple_game(game)

    n = game.n_players
    win_contains = [0] * n
    lose_not_contains = [0] * n

    for S in range(1 << n):
        vS = float(game.value(S))
        if vS == 1.0:
            for i in range(n):
                if S & (1 << i):
                    win_contains[i] += 1
        else:
            for i in range(n):
                if not (S & (1 << i)):
                    lose_not_contains[i] += 1

    denom = float(1 << n)
    return [(float(win_contains[i] + lose_not_contains[i]) / denom) for i in range(n)]


__all__ = ["rae_index"]
