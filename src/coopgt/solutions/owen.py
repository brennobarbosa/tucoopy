"""
# Owen value (a priori unions).

The Owen value extends the Shapley value to settings where players are partitioned into unions (blocks)
that act as a priori coalitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..base.coalition import mask_from_players
from ..base.types import GameProtocol, require_tabular_game
from ..base.exceptions import InvalidGameError, InvalidParameterError, NotSupportedError


def _factorials(n: int) -> list[int]:
    f = [1] * (n + 1)
    for k in range(2, n + 1):
        f[k] = f[k - 1] * k
    return f


def _validate_partition(n: int, unions: Sequence[Iterable[int] | int]) -> list[int]:
    masks: list[int] = []
    used = 0
    for u in unions:
        m = int(u) if isinstance(u, int) else int(mask_from_players(u))
        if m == 0:
            raise InvalidParameterError("unions must be non-empty")
        if used & m:
            raise InvalidParameterError("unions must be disjoint")
        used |= m
        masks.append(m)
    full = (1 << n) - 1
    if used != full:
        raise InvalidParameterError("unions must cover all players")
    return masks


@dataclass(frozen=True)
class OwenResult:
    """
    Result container for the Owen value.

    Attributes
    ----------
    x : list[float]
        Allocation vector (length `n_players`).
    meta : dict[str, object]
        Diagnostic metadata about the computation, including:
        - number of unions,
        - configuration parameters.
    """
    x: list[float]
    meta: dict[str, object]


def owen_value(
    game: GameProtocol,
    *,
    unions: Sequence[Iterable[int] | int],
    max_players: int = 16,
    require_complete: bool = True,
) -> OwenResult:
    """
    Compute the **Owen value** for a TU game with a priori unions
    (coalition structure).

    The Owen value extends the Shapley value to games where players are
    partitioned into predefined groups (unions). It is defined by a
    two-level random process:

    1. The unions arrive in a random order (Shapley at the union level).
    2. Within each union, players arrive in a random order (Shapley inside the union).

    Formally, this results in a weighted sum of marginal contributions where
    the weights reflect both the order of unions and the order of players
    within each union.

    Parameters
    ----------
    game : GameProtocol
        TU game.
    unions : Sequence[Iterable[int] | int]
        Partition of players into disjoint non-empty unions. Each element
        may be either:
        - an iterable of player indices, or
        - a coalition mask.
        The unions must be disjoint and cover all players.
    max_players : int, default=16
        Safety bound. The algorithm is exponential in `n_players`.
    require_complete : bool, default=True
        If True, require the game to specify values for all $2^n$ coalitions.

    Returns
    -------
    OwenResult
        Allocation vector and metadata.

    Raises
    ------
    InvalidParameterError
        If:
        - unions are invalid (not disjoint or not covering all players),
    NotSupportedError
        If `n_players > max_players`.
    InvalidGameError
        If the game is incomplete when `require_complete=True`.

    Notes
    -----
    - When there is a single union containing all players, the Owen value
      reduces to the Shapley value.
    - When each player is its own union, the Owen value also reduces to
      the Shapley value.
    - The Owen value is central in cooperative games with **coalition structures**
      and models situations where cooperation is constrained by pre-existing groups.

    Examples
    --------
    >>> # Two unions: {0,1} and {2,3}
    >>> res = owen_value(g, unions=[[0,1], [2,3]])
    >>> res.x
    """
    n = game.n_players
    if n > int(max_players):
        raise NotSupportedError(f"owen_value is exponential; requires n<={max_players} (got n={n})")
    if require_complete:
        expected = 1 << n
        tabular = require_tabular_game(game, context="owen_value")
        if len(tabular.v) < expected or any(int(m) not in tabular.v for m in range(expected)):
            raise InvalidGameError("owen_value requires a complete characteristic function (2^n coalition values)")

    union_masks = _validate_partition(n, unions)
    m = len(union_masks)

    # Dense v.
    v = [0.0] * (1 << n)
    for S in range(1 << n):
        v[S] = float(game.value(S))

    fact_n = _factorials(n)
    fact_m = _factorials(m)

    # Precompute union sizes and per-union factorial denominators.
    union_sizes = [int(u).bit_count() for u in union_masks]
    union_fact = [float(fact_nu) for fact_nu in (fact_n[s] for s in union_sizes)]

    # Helper: union coalition mask from subset of unions.
    def union_of(Tmask: int) -> int:
        out = 0
        for k in range(m):
            if Tmask & (1 << k):
                out |= union_masks[k]
        return out

    # Precompute weights for union-level and within-union.
    # Union-level: |T|!(m-|T|-1)!/m!
    w_union = [0.0] * (1 << m)
    denom_union = float(fact_m[m])
    for T in range(1 << m):
        k = int(T).bit_count()
        if k < m:
            w_union[T] = float(fact_m[k] * fact_m[m - k - 1]) / denom_union

    x = [0.0] * n

    # For each player i, find its union index u_idx.
    player_union: list[int] = [-1] * n
    for ui, um in enumerate(union_masks):
        for i in range(n):
            if um & (1 << i):
                player_union[i] = ui

    # For each union U, precompute within-union weights by subset size:
    # |S|!(|U|-|S|-1)!/|U|!
    w_within_by_union: list[list[float]] = []
    for sz in union_sizes:
        denom = float(fact_n[sz])
        w = [0.0] * (1 << sz)  # indexed by submask over positions 0..sz-1
        # We'll compute by size only; later we iterate subsets explicitly.
        w_by_k = [0.0] * (sz + 1)
        for k in range(sz):
            w_by_k[k] = float(fact_n[k] * fact_n[sz - k - 1]) / denom
        w_within_by_union.append(w_by_k)

    # For each union, list players for mapping to local indices.
    union_players: list[list[int]] = []
    for um in union_masks:
        ps = [i for i in range(n) if um & (1 << i)]
        union_players.append(ps)

    # Main Owen sum:
    for i in range(n):
        ui = player_union[i]
        ps = union_players[ui]
        local_index = ps.index(i)
        sz = len(ps)
        w_by_k = w_within_by_union[ui]

        # Iterate over union subsets T not containing ui.
        for T in range(1 << m):
            if T & (1 << ui):
                continue
            base_mask = union_of(T)
            wT = w_union[T]

            # Iterate over subsets S of U\{i}. Use local bitmask over sz-1 positions.
            # Build mapping from local subset bits to global mask.
            others = [p for p in ps if p != i]
            ocount = len(others)
            for Sbits in range(1 << ocount):
                Smask = 0
                k = int(Sbits).bit_count()
                for b in range(ocount):
                    if Sbits & (1 << b):
                        Smask |= 1 << others[b]
                wS = w_by_k[k]
                a = base_mask | Smask
                bmask = a | (1 << i)
                x[i] += float(wT) * float(wS) * (float(v[bmask]) - float(v[a]))

    return OwenResult(x=[float(v) for v in x], meta={"n_unions": int(m), "max_players": int(max_players), "require_complete": bool(require_complete)})


__all__ = ["OwenResult", "owen_value"]
