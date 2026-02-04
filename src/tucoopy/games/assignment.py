"""
# Assignment games.

An assignment game is defined from a bipartite value matrix. Coalitions can form
matches between the two sides, and their worth is the maximum total matching
value attainable by a one-to-one assignment.

See Also
--------
tucoopy.games.market.market_game

Examples
--------
>>> from tucoopy.games.assignment import assignment_game
>>> g = assignment_game([[1, 2], [3, 4]])
>>> g.n_players
4
>>> g.value(0b1111)  # grand coalition
5.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.coalition import all_coalitions
from ..base.game import Game
from ..base.exceptions import InvalidParameterError
from ._utils import _validate_n_players


def _max_weight_matching_value(values: list[list[float]]) -> float:
    """
    Maximum weight bipartite matching for a complete bipartite graph.

    Uses DP over the right side (works well for small instances).

    Examples
    --------
    >>> _max_weight_matching_value([[1.0, 2.0], [3.0, 4.0]])
    5.0
    """
    if not values:
        return 0.0
    m = len(values)
    n = len(values[0])
    if n == 0:
        return 0.0

    # dp[mask] = best total value using a subset of right nodes
    dp = [-1e300] * (1 << n)
    dp[0] = 0.0
    for i in range(m):
        nxt = [-1e300] * (1 << n)
        for mask in range(1 << n):
            base = dp[mask]
            if base <= -1e200:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                nm = mask | (1 << j)
                nxt[nm] = max(nxt[nm], base + values[i][j])
        dp = nxt
    return max(dp)


def assignment_game(
    values_matrix: Sequence[Sequence[float]],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct an **assignment game** (TU) from a bipartite value matrix.

    Players are split into two sides:
        
    - left players:  $0 \\ldots m-1$
    - right players: $m \\ldots m+n-1$

    For a coalition $S$, let $L(S)$ be the left players in $S$ and $R(S)$ the right players in $S$.
    The coalition worth is the maximum total assignment value achievable by matching
    $L(S)$ to $R(S)$ (one-to-one), using the given value matrix.

    Parameters
    ----------
    values_matrix : Sequence[Sequence[float]]
        Biadjacency matrix (m x n) of the bipartite value graph.
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the assignment game.

    Raises
    ------
    InvalidParameterError
        If the matrix is empty, not rectangular, or has zero rows/columns.

    Examples
    --------
    >>> from tucoopy.games.assignment import assignment_game
    >>> g = assignment_game([[1, 2], [3, 4]])
    >>> g.n_players
    4
    >>> g.value(3)
    0.0
    >>> g.value(15)
    5.0
    """
    m = len(values_matrix)
    if m == 0:
        raise InvalidParameterError("values_matrix must have at least 1 row")
    n = len(values_matrix[0])
    if n == 0:
        raise InvalidParameterError("values_matrix must have at least 1 column")
    for row in values_matrix:
        if len(row) != n:
            raise InvalidParameterError("values_matrix must be rectangular")

    total_players = _validate_n_players(m + n)

    vals = [[float(x) for x in row] for row in values_matrix]

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(total_players):
        if S == 0:
            continue
        left = [i for i in range(m) if S & (1 << i)]
        right = [j for j in range(n) if S & (1 << (m + j))]
        if not left or not right:
            v[S] = 0.0
            continue

        # Build the induced submatrix and match on the smaller side for efficiency.
        if len(left) <= len(right):
            sub = [[vals[i][j] for j in right] for i in left]
            v[S] = _max_weight_matching_value(sub)
        else:
            # Symmetric: swap roles by transposing induced submatrix.
            sub = [[vals[i][j] for i in left] for j in right]
            v[S] = _max_weight_matching_value(sub)

    return Game(n_players=total_players, v=v, player_labels=player_labels)
