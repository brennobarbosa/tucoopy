"""
# Holler (Public Good) power index.

The Holler index counts how often a player appears in minimal winning
coalitions, optionally normalized to sum to one.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._simple_utils import minimal_winning_coalitions, normalize


def holler_index(game: GameProtocol, *, normalized: bool = True) -> list[float]:
    """
    Compute the **Holler index** (also known as the **Public Good index**) for a simple game.

    The Holler index is based exclusively on the **minimal winning coalitions**
    of the game. A coalition $S$ is minimal winning if:

    - $v(S) = 1$, and
    - for every $i \\in S$, $v(S \\setminus \\{i\\}) = 0$.

    The Holler index of a player $i$ is the number of minimal winning coalitions
    that contain $i$:

    $$
    H_i = \\#\\{ S \\text{ minimal winning} \\mid i \\in S \\}.
    $$

    Optionally, this vector can be normalized to sum to 1.

    Parameters
    ----------
    game : Game
        A simple game.
    normalized : bool, default=True
        If True, normalize the index to sum to 1.

    Returns
    -------
    list[float]
        Holler index for each player (length `n_players`).

    Notes
    -----
    - This index focuses only on the **essential winning structures** of the game,
      ignoring larger winning coalitions that contain redundant players.
    - It is also called the *Public Good index* because it measures how often
      a player is indispensable in producing the public good (a winning outcome).
    - Unlike Shapley or Banzhaf, this index does not consider coalition sizes
      or permutations—only the structure of minimal winning coalitions.

    Examples
    --------
    >>> hi = holler_index(g)
    >>> len(hi) == g.n_players
    True
    """
    mw = minimal_winning_coalitions(game)
    n = game.n_players
    if not mw:
        return [0.0 for _ in range(n)]

    raw = [0.0] * n
    for S in mw:
        for i in range(n):
            if S & (1 << i):
                raw[i] += 1.0
    return normalize(raw) if normalized else raw


__all__ = ["holler_index"]
