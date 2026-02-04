"""
# Market games (buyers vs sellers).

In a market game, players are divided into buyers and sellers. A coalition's
worth is the maximum trade surplus it can realize by matching its internal
buyers and sellers.

Notes
-----
- Players $0 \\ldots B-1$ are buyers and players $B \\ldots B+S-1$ are sellers.
- This implementation uses a greedy matching strategy and is intended as a
  lightweight generator for examples and demos.

See Also
--------
tucoop.games.assignment.assignment_game

Examples
--------
>>> from tucoop.games.market import market_game
>>> g = market_game([10, 8], [3])
>>> g.n_players
3
>>> g.value(0b111)  # grand coalition
7.0
"""

from __future__ import annotations

from typing import Sequence

from ..base.game import Game


def market_game(
    buyers: Sequence[float],
    sellers: Sequence[float],
    *,
    player_labels: list[str] | None = None,
) -> Game:
    """
    Generate a **market game** (buyers vs sellers).

    Coalition value equals the maximum trade surplus achievable
    by matching buyers and sellers inside the coalition.

    Parameters
    ----------
    buyers : Sequence[float]
        Buyer valuations.
    sellers : Sequence[float]
        Seller costs.
    player_labels : list[str] | None, optional
        Optional labels.

    Returns
    -------
    Game
        Market cooperative game.

    Notes
    -----
    - Players $0 \\ldots B-1$ are buyers, $B \\ldots B+S-1$ are sellers.
    - Value is computed by greedy matching of highest surplus pairs.

    Examples
    --------
    >>> from tucoop.games.market import market_game
    >>> g = market_game([10, 8], [3])
    >>> g.value(0b111)
    7.0
    """
    B = len(buyers)
    S = len(sellers)
    n = B + S

    def value_fn(ps: Sequence[int]) -> float:
        bs = [buyers[i] for i in ps if i < B]
        ss = [sellers[i - B] for i in ps if i >= B]

        bs.sort(reverse=True)
        ss.sort()

        total = 0.0
        for b, s in zip(bs, ss):
            if b > s:
                total += (b - s)
            else:
                break
        return float(total)

    return Game.from_value_function(
        n_players=n,
        value_fn=value_fn,
        player_labels=player_labels,
    )


__all__ = ["market_game"]
