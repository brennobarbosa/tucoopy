"""
# ENSC value (Equal Non-Separable Contribution).
"""

from __future__ import annotations

from ..base.types import GameProtocol


def ensc_value(game: GameProtocol) -> list[float]:
    """
    Compute the **ENSC value** (Equal Non-Separable Contribution).

    The ENSC value is defined as:

    $$
    x_i = v(N) - \\frac{1}{n-1} \\sum_{j \\ne i} v(N \\setminus \\{j\\}).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        ENSC allocation vector.

    Notes
    -----
    - Based on marginal contributions at the grand coalition.
    - Frequently implemented in CoopGame.

    Examples
    --------
    >>> ensc_value(g)
    """
    n = game.n_players
    if n < 2:
        return [float(game.value(game.grand_coalition))]

    vN = float(game.value(game.grand_coalition))
    out = [0.0] * n

    for i in range(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            s += float(game.value(game.grand_coalition & ~(1 << j)))
        out[i] = vN - s / float(n - 1)

    return out


__all__ = ["ensc_value"]
