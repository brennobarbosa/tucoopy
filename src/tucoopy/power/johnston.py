"""
# Johnston power index.

The Johnston index refines Banzhaf criticality by splitting each winning
coalition's contribution equally among its critical players.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._simple_utils import normalize, require_complete_simple_game


def johnston_index(game: GameProtocol) -> list[float]:
    """
    Compute the **Johnston power index** for a complete simple game.

    In a simple game ($v(S) \\in \\{0,1\\}$), a player $i$ is **critical**
    in a winning coalition $S$ if removing $i$ makes the coalition losing:

    $$
    C(S) = \\{ i \\in S \\mid v(S) = 1 \\text{ and } v(S \\setminus \\{i\\}) = 0 \\}.
    $$

    The Johnston index assigns to each critical player an equal share of the
    coalition's criticality. That is, each winning coalition $S$ contributes:

    $$
    \\frac{1}{|C(S)|}
    $$

    to each $i \\in C(S)$.

    The resulting vector is then normalized to sum to 1.

    Parameters
    ----------
    game : Game
        A *complete* simple game.

    Returns
    -------
    list[float]
        Johnston index for each player (length `n_players`), normalized to sum to 1.

    Raises
    ------
    InvalidGameError
        If the game is not a complete simple game (checked by
        :func:`require_complete_simple_game`).
    NotSupportedError
        If `n_players` exceeds the supported limit for completeness checks.

    Notes
    -----
    - The Johnston index refines the idea of **criticality** used in the
      (unnormalized) Banzhaf index.
    - In the Banzhaf index, each critical occurrence contributes equally (1).
      In the Johnston index, the contribution of a coalition is divided equally
      among its critical players.
    - This index captures how often a player is decisive *and* how many other
      players share that decisiveness.

    Examples
    --------
    >>> ji = johnston_index(g)
    >>> sum(ji)
    1.0
    """
    require_complete_simple_game(game)

    n = game.n_players
    raw = [0.0] * n
    for S in range(1 << n):
        if S == 0:
            continue
        if float(game.value(S)) != 1.0:
            continue
        critical: list[int] = []
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if float(game.value(S & ~(1 << i))) == 0.0:
                critical.append(i)
        if not critical:
            continue
        share = 1.0 / float(len(critical))
        for i in critical:
            raw[i] += share

    return normalize(raw)


__all__ = ["johnston_index"]
