"""
# Minimum spanning tree (cost) games.

Given a complete undirected graph over players with edge weights $w(i,j)$, the
coalition cost is the MST cost over the induced subgraph on $S$:

$$
c(S) = \\mathrm{MST}(S).
$$

This module returns a TU worth game by taking $v(S) = -c(S)$.

See Also
--------
tucoop.games.flow.flow_game
tucoop.games.airport.airport_game

Examples
--------
>>> from tucoop.games.mst import mst_game
>>> g = mst_game([[0, 1], [1, 0]])
>>> g.value(0b11)
-1.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.coalition import all_coalitions
from ..base.game import Game
from ..base.exceptions import InvalidParameterError
from ._utils import _validate_n_players


def _mst_cost_complete_graph(nodes: list[int], w: list[list[float]]) -> float:
    if len(nodes) <= 1:
        return 0.0

    import heapq

    total = 0.0
    visited: set[int] = set()
    heap: list[tuple[float, int]] = []
    start = nodes[0]
    visited.add(start)

    for v in nodes:
        if v == start:
            continue
        heapq.heappush(heap, (float(w[start][v]), v))

    while len(visited) < len(nodes):
        while heap and heap[0][1] in visited:
            heapq.heappop(heap)
        if not heap:
            raise InvalidParameterError("graph appears disconnected or has invalid weights")
        cost, nxt = heapq.heappop(heap)
        if not (cost == cost and cost < 1e200):
            raise InvalidParameterError("graph appears disconnected or has invalid weights")
        visited.add(nxt)
        total += cost
        for v in nodes:
            if v in visited:
                continue
            heapq.heappush(heap, (float(w[nxt][v]), v))

    return float(total)


def mst_game(
    weights: Sequence[Sequence[float]],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct a **minimum spanning tree** (cost) game as a TU worth game.

    Given a complete undirected graph over players with edge weights $w(i,j)$,
    define the coalition cost as the MST cost over the induced subgraph on $S$:
    
    $$c(S) = \\text{MST cost}(S)$$

    This constructor returns a worth game by taking $v(S) = -c(S)$.

    Parameters
    ----------
    weights : Sequence[Sequence[float]]
        Edge weights for the complete graph (n x n matrix).
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the MST cost game.

    Raises
    ------
    InvalidParameterError
        If weights is not a square n x n matrix or the graph is disconnected.

    Examples
    --------
    >>> from tucoop.games.mst import mst_game
    >>> g = mst_game([[0, 1], [1, 0]])
    >>> g.n_players
    2
    >>> g.value(3)
    -1.0
    """
    n = _validate_n_players(len(weights))
    w = [[float(x) for x in row] for row in weights]
    for row in w:
        if len(row) != n:
            raise InvalidParameterError("weights must be an n x n matrix")

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(n):
        if S == 0:
            continue
        nodes = [i for i in range(n) if S & (1 << i)]
        cost = _mst_cost_complete_graph(nodes, w)
        v[S] = -cost

    return Game(n_players=n, v=v, player_labels=player_labels)
