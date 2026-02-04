"""
# Egalitarian Shapley value.

This value modifies the Shapley allocation by down-weighting each Harsanyi
dividend by the square of the coalition size.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ..transforms.harsanyi import harsanyi_dividends


def egalitarian_shapley_value(game: GameProtocol) -> list[float]:
    """
    Egalitarian Shapley value.

    This value modifies the Shapley allocation by dividing each Harsanyi
    dividend by the square of the coalition size:

    $$\\phi_i = \\sum_{S \\ni i} \\frac{d(S)}{|S|^2}.$$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Egalitarian Shapley allocation.

    Notes
    -----
    - Derived from the Shapley value but emphasizes equality inside coalitions.
    - Implemented directly from Harsanyi dividends.
    """
    n = game.n_players
    d = harsanyi_dividends(game)
    out = [0.0] * n

    for S, div in d.items():
        if S == 0:
            continue
        k = int(S).bit_count()
        share = float(div) / float(k * k)
        for i in range(n):
            if S & (1 << i):
                out[i] += share

    return [float(v) for v in out]


__all__ = ["egalitarian_shapley_value"]
