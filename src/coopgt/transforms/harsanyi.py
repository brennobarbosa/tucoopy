"""
# Harsanyi dividends (unanimity coordinates).

The Harsanyi dividends provide a basis decomposition of a TU game into unanimity
games. This module exposes a game-level wrapper around the Möbius transform.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ._utils import to_dense_values
from .mobius import mobius_transform


def harsanyi_dividends(game: GameProtocol) -> dict[int, float]:
    """
    Compute the **Harsanyi dividends** (unanimity coordinates) of a TU game.

    The Harsanyi dividends provide a unique decomposition of a cooperative game
    in the basis of **unanimity games**. They are given by the Möbius transform
    of the characteristic function over the subset lattice:

    $$
    d(S) = \\sum_{T \\subseteq S} (-1)^{|S|-|T|} \\, v(T).
    $$

    Conversely, the original game can be reconstructed from the dividends via:

    $$
    v(S) = \\sum_{T \\subseteq S} d(T).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU cooperative game.

    Returns
    -------
    dict[int, float]
        Dictionary mapping coalition masks to their Harsanyi dividend $d(S)$.

    Notes
    -----
    - The Harsanyi dividends express the game as a linear combination of
      **unanimity games** $u_T$, i.e.

      $$
      v = \\sum_{T \\subseteq N} d(T) \\, u_T.
      $$

    - Many solution concepts admit elegant expressions in terms of these dividends.
      For example, the Shapley value can be written directly as a weighted sum of
      Harsanyi dividends.
    - This implementation uses the standard in-place fast Möbius transform,
      running in $O(n 2^n)$ time.

    Examples
    --------
    >>> d = harsanyi_dividends(g)
    >>> # The dividend of the empty coalition is always zero
    >>> d[0]
    0.0
    """
    values = to_dense_values(game)
    dividends = mobius_transform(values, n_players=game.n_players)
    return {int(S): float(dS) for S, dS in enumerate(dividends)}


__all__ = ["harsanyi_dividends"]
