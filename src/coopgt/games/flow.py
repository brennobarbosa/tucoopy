"""
# Flow games.

In a flow game, players own edges of a capacitated directed network. A coalition
$S$ may use exactly the edges owned by its members, and its worth is the maximum
$s$-$t$ flow achievable in the induced network.

Notes
-----
This implementation uses the Edmonds–Karp algorithm and is intended for small
networks and examples.

See Also
--------
tucoop.games.mst.mst_game

Examples
--------
>>> from tucoop.games.flow import OwnedEdge, flow_game
>>> edges = [OwnedEdge(0, 1, 1.0, owner=0), OwnedEdge(1, 2, 1.0, owner=1)]
>>> g = flow_game(n_players=2, n_nodes=3, edges=edges)
>>> g.value(0b11)
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..base.coalition import all_coalitions
from ..base.game import Game
from ..base.exceptions import InvalidParameterError
from ._utils import _validate_n_players


@dataclass(frozen=True)
class OwnedEdge:
    """
    An owned directed edge in a flow game.

    Attributes
    ----------
    u, v
        Tail and head node indices.
    capacity
        Non-negative edge capacity.
    owner
        Player index that owns the edge.

    Examples
    --------
    >>> OwnedEdge(0, 1, 2.0, owner=0)
    OwnedEdge(u=0, v=1, capacity=2.0, owner=0)
    """

    u: int
    v: int
    capacity: float
    owner: int


def _max_flow_edmonds_karp(n_nodes: int, edges: Iterable[tuple[int, int, float]]) -> float:
    # Build residual network.
    n = int(n_nodes)
    cap = [[0.0] * n for _ in range(n)]
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v, c in edges:
        uu, vv, cc = int(u), int(v), float(c)
        if cc < 0:
            raise InvalidParameterError("capacity must be >= 0")
        if vv not in adj[uu]:
            adj[uu].append(vv)
        if uu not in adj[vv]:
            adj[vv].append(uu)
        cap[uu][vv] += cc

    source = 0
    sink = n - 1
    flow = 0.0

    while True:
        parent = [-1] * n
        parent[source] = source
        q = [source]
        for x in q:
            for y in adj[x]:
                if parent[y] != -1:
                    continue
                if cap[x][y] <= 1e-12:
                    continue
                parent[y] = x
                q.append(y)
                if y == sink:
                    break
            if parent[sink] != -1:
                break

        if parent[sink] == -1:
            break

        # Augment.
        inc = 1e300
        y = sink
        while y != source:
            x = parent[y]
            inc = min(inc, cap[x][y])
            y = x
        y = sink
        while y != source:
            x = parent[y]
            cap[x][y] -= inc
            cap[y][x] += inc
            y = x
        flow += inc

    return float(flow)


def flow_game(
    *,
    n_players: int,
    n_nodes: int,
    edges: list[OwnedEdge],
    player_labels: list[str] | None = None,
    source: int = 0,
    sink: int | None = None,
) -> Game:
    """
    Construct a **flow game** (TU) from a directed capacitated network with edge ownership.

    Coalition $S$ can use exactly the edges whose `owner` is in $S$, and its worth is
    the maximum $s-t$ flow in that induced network.

    Parameters
    ----------
    n_players : int
        Number of players (edge owners).
    n_nodes : int
        Number of nodes in the network.
    edges : list of OwnedEdge
        List of owned edges.
    player_labels : list of str, optional
        Optional labels for players.
    source : int, optional
        Source node index (default 0).
    sink : int, optional
        Sink node index (default n_nodes-1).

    Returns
    -------
    Game
        Cooperative game instance representing the flow game.

    Raises
    ------
    InvalidParameterError
        If n_nodes < 2, source/sink out of range, or edge owner/endpoints out of range.

    Examples
    --------
    >>> from tucoop.games.flow import flow_game, OwnedEdge
    >>> edges = [OwnedEdge(0, 1, 5, 0), OwnedEdge(1, 2, 3, 1)]
    >>> g = flow_game(n_players=2, n_nodes=3, edges=edges)
    >>> g.n_players
    2
    >>> g.value(3)
    3.0
    """
    n = _validate_n_players(n_players)
    nn = int(n_nodes)
    if nn < 2:
        raise InvalidParameterError("n_nodes must be >= 2")
    s = int(source)
    t = nn - 1 if sink is None else int(sink)
    if s < 0 or s >= nn:
        raise InvalidParameterError("source out of range")
    if t < 0 or t >= nn:
        raise InvalidParameterError("sink out of range")
    if s == t:
        raise InvalidParameterError("source and sink must differ")

    for e in edges:
        if e.owner < 0 or e.owner >= n:
            raise InvalidParameterError("edge owner out of range")
        if e.u < 0 or e.u >= nn or e.v < 0 or e.v >= nn:
            raise InvalidParameterError("edge endpoints out of range")

    # We'll remap source->0 and sink->nn-1 for the solver by relabeling nodes.
    def relabel(node: int) -> int:
        if node == s:
            return 0
        if node == t:
            return nn - 1
        if node == 0:
            return s
        if node == nn - 1:
            return t
        return node

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(n):
        if S == 0:
            continue
        induced = []
        for e in edges:
            if S & (1 << e.owner):
                induced.append((relabel(e.u), relabel(e.v), e.capacity))
        v[S] = _max_flow_edmonds_karp(nn, induced)

    return Game(n_players=n, v=v, player_labels=player_labels)
