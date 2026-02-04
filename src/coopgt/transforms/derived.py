"""
Derived games.

# This module implements standard operators that derive a new game from an
existing one, such as:

- Dual game,
- Subgames and restrictions to subsets of players.
"""

from __future__ import annotations

from ..base.coalition import all_coalitions, grand_coalition, mask_from_players, players
from ..base.game import Game
from ..base.exceptions import InvalidCoalitionError
from ..base.types import GameProtocol


def dual_game(game: GameProtocol) -> Game:
    """
    Compute the **dual game** (sometimes called the complement game) of a TU game.

    Given a TU game $v$ on the player set $N$, its dual game $v^*$ is defined by:

    $$
    v^*(S) =
      \\begin{cases}
        0, & S = \\varnothing \\\\
        v(N) - v(N \\setminus S), & S \\neq \\varnothing
      \\end{cases}
    $$

    where $N \\setminus S$ denotes the complement coalition.

    Parameters
    ----------
    game : Game
        Original TU game $v$.

    Returns
    -------
    Game
        The dual game $v^*$, defined on the same player set.

    Notes
    -----
    - Intuition: $v^*(S)$ measures the **marginal loss** incurred by excluding $S$
      from the grand coalition, i.e., how much value disappears when only the complement
      coalition is allowed to form.
    - The dual operation is an involution for TU games (up to the empty coalition convention):
      applying it twice returns the original game: $(v^*)^* = v$.
    - Duality is useful in studying correspondences between classes of games
      (e.g., cost vs benefit formulations) and for theoretical results involving the core.

    Examples
    --------
    >>> v_star = dual_game(g)
    >>> N = g.grand_coalition
    >>> # For a coalition S, v*(S) = v(N) - v(N \\ S)
    >>> S = 0b011
    >>> v_star.value(S) == g.value(N) - g.value(N & ~S)
    True
    """

    n = game.n_players
    N = grand_coalition(n)
    vN = game.value(N)
    v = {mask: (0.0 if mask == 0 else vN - game.value(N & ~mask)) for mask in all_coalitions(n)}
    v[0] = 0.0
    labels = getattr(game, "player_labels", None)
    return Game(n_players=n, v=v, player_labels=labels)


def subgame(game: GameProtocol, coalition_mask: int) -> Game:
    """
    Restrict a TU game to a coalition, producing the **subgame** on that coalition.

    Given a coalition $T \\subseteq N$, the subgame $v_T$ is defined on subsets of $T$ by:

    $$
    v_T(S) = v(S), \\quad \\forall S \\subseteq T.
    $$

    Since :class:`~tucoop.base.game.Game` encodes coalitions as bitmasks over
    players $0,\\ldots,n-1$, the returned subgame **renumbers** the players of $T$
    to a new index set $0,\\ldots,k-1$ (where $k = |T|$) using the induced order of
    `players(T)`.

    Parameters
    ----------
    game : Game
        Original TU game on player set $N$.
    coalition_mask : int
        Coalition mask $T$ defining the restriction. Must be non-empty.

    Returns
    -------
    Game
        A new TU game with `n_players = |T|` representing the restriction of `game`
        to coalition $T$.

    Raises
    ------
    InvalidCoalitionError
        If `coalition_mask` is empty or out of range.

    Notes
    -----
    - **Player renumbering:** if $T$ corresponds to original players
      `(p0, p1, ..., p_{k-1})`, then in the returned game these players become
      indices `0..k-1` in that order.
    - Player labels are preserved by projection: if `game.player_labels` is set,
      the returned game uses the corresponding subset of labels in the new order.
    - This is a fundamental operation for recursive definitions and solution concepts
      that consider restricted games.

    Examples
    --------
    Restrict a 4-player game to coalition T = {1, 3}:

    >>> from tucoop.base.coalition import mask_from_players
    >>> T = mask_from_players([1, 3])
    >>> gT = subgame(g, T)
    >>> gT.n_players
    2

    The new player 0 corresponds to original player 1, and new player 1 corresponds
    to original player 3, so:

    >>> # In the subgame, coalition {new 0} corresponds to {old 1}
    >>> gT.value(0b01) == g.value(mask_from_players([1]))
    True
    """
    T = int(coalition_mask)
    if T == 0:
        raise InvalidCoalitionError("coalition_mask must be non-empty")
    if T < 0 or T > game.grand_coalition:
        raise InvalidCoalitionError("coalition_mask out of range")

    ps = players(T, n_players=game.n_players)
    mapping = {p: i for i, p in enumerate(ps)}
    k = len(ps)

    def lift(mask_sub: int) -> int:
        # mask_sub is over 0..k-1; map bits back to original player indices.
        out = 0
        for i in range(k):
            if mask_sub & (1 << i):
                out |= 1 << ps[i]
        return out

    v: dict[int, float] = {0: 0.0}
    for mask_sub in all_coalitions(k):
        if mask_sub == 0:
            continue
        v[mask_sub] = game.value(lift(mask_sub))

    labels = None
    game_labels = getattr(game, "player_labels", None)
    if game_labels is not None:
        labels = [game_labels[p] for p in ps]

    return Game(n_players=k, v=v, player_labels=labels)


def restrict_to_players(game: GameProtocol, ps: list[int]) -> Game:
    """
    Restrict a TU game to an explicit list of player indices.

    This is a convenience wrapper around :func:`subgame` that first converts the
    player list to a coalition mask.

    Parameters
    ----------
    game : Game
        Original TU game.
    ps : list[int]
        Player indices to keep. Must be non-empty and within range.

    Returns
    -------
    Game
        Subgame induced by the specified players, with players renumbered to `0..k-1`
        in the order given by `players(mask_from_players(ps))`.

    Examples
    --------
    >>> gT = restrict_to_players(g, [1, 3])
    >>> gT.n_players
    2
    """
    return subgame(game, mask_from_players(ps))
