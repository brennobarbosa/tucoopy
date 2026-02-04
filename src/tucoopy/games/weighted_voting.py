"""
# Weighted voting games (simple games).

Given weights $w_i$ and quota $q$, a coalition $S$ is **winning** if
$\\sum_{i \\in S} w_i \\ge q$. The characteristic function takes two values:
`winning_value` and `losing_value` (with $v(\\varnothing)=0$ enforced).

See Also
--------
tucoopy.power.banzhaf.banzhaf_index
tucoopy.power.shapley_shubik.shapley_shubik_index

Examples
--------
>>> from tucoopy.games.weighted_voting import weighted_voting_game
>>> g = weighted_voting_game([2, 1, 1], quota=3)
>>> g.value(0b011)
1.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game
from ..base.coalition import all_coalitions
from ._utils import _validate_n_players


def weighted_voting_game(
    weights: Sequence[float],
    quota: float,
    *,
    player_labels: list[str] | None = None,
    winning_value: float = 1.0,
    losing_value: float = 0.0,
) -> Game:
    """
    Construct a **weighted voting game** (simple game).

    Given weights $w_i$ and quota $q$, coalition $S$ is winning if $\\sum_{i \\in S} w_i \\geq q$.
    The characteristic function is given by:

    $$
    v(S) = \\begin{cases}
    w_v, & \\text{ if } S \\text{ is winning} \\\\
    l_v, & \\text{ otherwise}
    \\end{cases}
    $$

    Parameters
    ----------
    weights : Sequence[float]
        Player weights.
    quota : float
        Winning threshold.
    player_labels : list of str, optional
        Optional display labels for players.
    winning_value : float, optional
        Value assigned to winning coalitions (default 1.0).
    losing_value : float, optional
        Value assigned to losing coalitions (default 0.0). The empty coalition
        is always normalized to 0.0 (TU convention).

    Returns
    -------
    Game
        A TU game with values in {0,1} (by default), suitable for power indices.

    Raises
    ------
    InvalidParameterError
        If fewer than 1 player is provided.

    Notes
    -----
    Commonly: `winning_value=1`, `losing_value=0`.
    The TU convention $v(\\varnothing)=0$ is enforced even if `losing_value` is non-zero.

    Examples
    --------
    >>> from tucoopy.games.weighted_voting import weighted_voting_game
    >>> g = weighted_voting_game([2, 1, 1], quota=3)
    >>> g.value(0b011)  # 2+1 meets quota
    1.0
    >>> g.value(0b001)
    0.0
    """
    n = _validate_n_players(len(weights))

    v: dict[int, float] = {0: float(losing_value)}
    for S in all_coalitions(n):
        if S == 0:
            continue
        w = 0.0
        for i in range(n):
            if S & (1 << i):
                w += float(weights[i])
        v[S] = float(winning_value if w >= quota else losing_value)

    # Normalize v(0)=0 for TU conventions if losing_value was non-zero.
    v[0] = 0.0
    return Game(n_players=n, v=v, player_labels=player_labels)
