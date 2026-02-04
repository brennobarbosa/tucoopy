"""
# Koenig–Bräuninger power index.

This index is based on minimal winning coalitions and assigns to each member of
such a coalition a weight proportional to ``1/(|S|-1)``.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._simple_utils import minimal_winning_coalitions, normalize


def koenig_brauninger_index(game: GameProtocol) -> list[float]:
    """
    Koenig-Bräuninger power index for a simple game.

    For each minimal winning coalition S containing player $i$:

    $$KB_i += \\frac{1}{|S| - 1}.$$

    The result is normalized to sum to $1$.

    Parameters
    ----------
    game : Game
        Complete simple game.

    Returns
    -------
    list[float]
        Koenig-Bräuninger index.

    Notes
    -----
    - Based only on minimal winning coalitions.
    - Gives more weight to players in smaller minimal winning coalitions.
    - Implemented as in CoopGame.
    """
    mw = minimal_winning_coalitions(game)
    n = game.n_players
    raw = [0.0] * n

    for S in mw:
        k = int(S).bit_count()
        if k <= 1:
            continue
        share = 1.0 / float(k - 1)
        for i in range(n):
            if S & (1 << i):
                raw[i] += share

    return normalize(raw)


__all__ = ["koenig_brauninger_index"]
