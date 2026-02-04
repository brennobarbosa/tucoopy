"""
# Solidarity value.

The solidarity value distributes each Harsanyi dividend equally among all
members of the coalition.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ..transforms.harsanyi import harsanyi_dividends


def solidarity_value(game: GameProtocol) -> list[float]:
    """
    Solidarity value for a TU game.

    The solidarity value distributes each Harsanyi dividend equally among
    all members of the coalition:

    $$\\phi_i = \\sum_{S \\ni i} \\frac{d(S)}{|S|}.$$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Solidarity value allocation.

    Notes
    -----
    - Uses Harsanyi dividends.
    - Closely related to the Shapley value but conceptually emphasizes
      coalition solidarity instead of marginal pivotality.
    """
    n = game.n_players
    d = harsanyi_dividends(game)
    out = [0.0] * n

    for S, div in d.items():
        if S == 0:
            continue
        k = int(S).bit_count()
        share = float(div) / float(k)
        for i in range(n):
            if S & (1 << i):
                out[i] += share

    return [float(v) for v in out]


__all__ = ["solidarity_value"]
