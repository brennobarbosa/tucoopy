"""
# Egalitarian value.

The egalitarian solution splits the grand coalition worth equally among players.
"""

from __future__ import annotations

from ..base.types import GameProtocol


def egalitarian_value(game: GameProtocol) -> list[float]:
    """
    Compute the **egalitarian solution** for a TU game.

    The egalitarian solution divides the value of the grand coalition
    equally among all players:

    $$
    x_i = \\frac{v(N)}{n}.
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Egalitarian allocation vector.

    Notes
    -----
    - This solution ignores all coalition structure.
    - Often used as a baseline comparison in cooperative game analysis.

    Examples
    --------
    >>> egalitarian_value(g)
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    share = vN / float(n)
    return [share for _ in range(n)]


__all__ = ["egalitarian_value"]
