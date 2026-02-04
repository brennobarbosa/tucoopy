"""
# Deegan–Packel power index.

The Deegan–Packel index is based on minimal winning coalitions and distributes a
unit contribution across members inversely proportional to coalition size.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._simple_utils import minimal_winning_coalitions, normalize


def deegan_packel_index(game: GameProtocol) -> list[float]:
    """
    Compute the **Deegan–Packel power index** for a complete simple game.

    This index is based on the **minimal winning coalitions** of the game.
    A coalition $S$ is minimal winning if:

    - $v(S) = 1$, and
    - for every $i \\in S$, $v(S \\setminus \\{i\\}) = 0$.

    Each minimal winning coalition distributes one unit of power equally
    among its members. That is, for each minimal winning coalition $S$,
    every player $i \\in S$ receives:

    $$
    \\frac{1}{|S|}.
    $$

    The resulting vector is then normalized to sum to $1$.

    Parameters
    ----------
    game : Game
        A *complete* simple game.

    Returns
    -------
    list[float]
        Deegan–Packel index for each player (length `n_players`), normalized to sum to 1.

    Notes
    -----
    - Like the Holler (Public Good) index, this index considers only minimal
      winning coalitions.
    - The key difference is that the Holler index counts how many minimal
      winning coalitions contain a player, while the Deegan–Packel index
      weights each coalition by the inverse of its size.
    - This reflects the idea that being essential in a **small** coalition
      should count more than being essential in a large one.

    Examples
    --------
    >>> dpi = deegan_packel_index(g)
    >>> sum(dpi)
    1.0
    """
    mw = minimal_winning_coalitions(game)
    n = game.n_players
    if not mw:
        return [0.0 for _ in range(n)]

    raw = [0.0] * n
    for S in mw:
        k = int(S).bit_count()
        if k <= 0:
            continue
        share = 1.0 / float(k)
        for i in range(n):
            if S & (1 << i):
                raw[i] += share
    return normalize(raw)


__all__ = ["deegan_packel_index"]
