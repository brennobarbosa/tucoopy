"""
# ESD value (Equal Surplus Division).
"""

from __future__ import annotations

from ..base.types import GameProtocol


def esd_value(game: GameProtocol) -> list[float]:
    """
    Compute the **Equal Surplus Division (ESD)** value.

    The ESD value distributes the surplus equally on top of singleton values:

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
        ESD allocation vector.

    Notes
    -----
    - Algebraically identical to the CIS value.
    - Included separately for conceptual clarity and compatibility
      with CoopGame terminology.

    Examples
    --------
    >>> esd_value(g)
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    v1 = [float(game.value(1 << i)) for i in range(n)]
    surplus = (vN - sum(v1)) / float(n)
    return [v + surplus for v in v1]


__all__ = ["esd_value"]
