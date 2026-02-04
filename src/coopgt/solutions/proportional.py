"""
# Proportional value.

The proportional solution allocates the grand coalition value proportionally to singleton worths.
"""

from __future__ import annotations

from ..base.types import GameProtocol


def proportional_value(game: GameProtocol) -> list[float]:
    """
    Compute the **proportional solution** for a TU game.

    The proportional solution distributes the grand coalition value
    proportionally to singleton values:

    $$
    x_i = \\frac{v(\\{i\\})}{\\sum_j v(\\{j\\})} \\, v(N).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Proportional allocation vector.

    Notes
    -----
    - If all singleton values are zero, returns the zero vector.
    - Used frequently in introductory examples and comparisons.

    Examples
    --------
    >>> proportional_value(g)
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    singletons = [float(game.value(1 << i)) for i in range(n)]
    s = sum(singletons)
    if s == 0.0:
        return [0.0 for _ in range(n)]
    return [v / s * vN for v in singletons]


__all__ = ["proportional_value"]
