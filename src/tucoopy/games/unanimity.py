"""
# Unanimity games.

The unanimity game $u_T$ assigns a positive value to coalitions that contain a
fixed coalition $T$:

$$
u_T(S) =
\\begin{cases}
v, & \\text{if } T \\subseteq S, \\\\
0, & \\text{otherwise}.
\\end{cases}
$$

Unanimity games form a convenient basis for TU games (via Harsanyi dividends).

See Also
--------
tucoopy.games.glove.glove_game
tucoopy.transforms.harsanyi.harsanyi_dividends

Examples
--------
>>> from tucoopy.games.unanimity import unanimity_game
>>> g = unanimity_game([0, 1], value=2.0, n_players=3)
>>> g.value(0b111)
2.0
"""

from __future__ import annotations

from typing import Iterable

from ..base.game import Game
from ..base.coalition import all_coalitions, mask_from_players
from ..base.exceptions import InvalidParameterError
from ._utils import _validate_n_players


def unanimity_game(
    coalition: int | Iterable[int],
    *,
    value: float = 1.0,
    n_players: int | None = None,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Construct a **unanimity game** $u_T$ (TU).

    Let $v$ denote the value. Given a coalition $S$ define:
    
    $$
    u_T(S) = 
    \\begin{cases}
    v & \\text{ if } T \\subseteq S \\\\ 
    0 & \\text{ otherwise }
    \\end{cases}
    $$

    Provide either:

     - coalition as bitmask and n_players, or
     - coalition as iterable of players and n_players inferred from labels/explicit arg.

    Parameters
    ----------
    coalition : int or Iterable[int]
        Coalition as bitmask or iterable of player indices.
    value : float, optional
        Value assigned if T is a subset of S (default 1.0).
    n_players : int, optional
        Number of players (required if coalition is bitmask or no labels provided).
    player_labels : list of str, optional
        Optional labels for players.

    Returns
    -------
    Game
        Cooperative game instance representing the unanimity game.

    Raises
    ------
    InvalidParameterError
        If n_players is missing or fewer than 1 player.

    Examples
    --------
    >>> from tucoopy.games.unanimity import unanimity_game
    >>> g = unanimity_game([0, 1], value=2.0, n_players=3)
    >>> g.n_players
    3
    >>> g.value(3)
    2.0
    >>> g.value(7)
    2.0
    """
    if isinstance(coalition, int):
        if n_players is None:
            raise InvalidParameterError("n_players is required when coalition is a bitmask")
        T = int(coalition)
        n = int(n_players)
    else:
        T = mask_from_players(coalition)
        if n_players is None:
            if player_labels is None:
                raise InvalidParameterError("n_players is required when coalition is an iterable and no labels provided")
            n = len(player_labels)
        else:
            n = int(n_players)

    n = _validate_n_players(n)

    v: dict[int, float] = {0: 0.0}
    for S in all_coalitions(n):
        if S == 0:
            continue
        v[S] = float(value) if (S & T) == T else 0.0

    return Game(n_players=n, v=v, player_labels=player_labels)
