"""
# CIS value (Center of the Imputation Set).

The CIS value is a classical imputation defined from singleton values and the grand coalition value.
"""

from __future__ import annotations

from ..base.types import GameProtocol


def cis_value(game: GameProtocol) -> list[float]:
    """
    Compute the **CIS value** (Center of the Imputation Set).

    The CIS value is defined as:

    $$
    x_i = v(\\{i\\}) + \\frac{v(N) - \\sum_j v(\\{j\\})}{n}.
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        CIS allocation vector.

    Notes
    -----
    - Interpreted as the center of the imputation set.
    - Adds an equal share of the surplus to singleton payoffs.

    Examples
    --------
    >>> cis_value(g)
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    v1 = [float(game.value(1 << i)) for i in range(n)]
    surplus = (vN - sum(v1)) / float(n)
    return [v + surplus for v in v1]


__all__ = ["cis_value"]
