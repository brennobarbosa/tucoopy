"""
# Combining games (sum/difference).

TU games on a fixed player set form a vector space under pointwise addition and
scalar multiplication. This module implements the basic binary operations used
in examples and tests.
"""

from __future__ import annotations

from ..base.coalition import all_coalitions
from ..base.game import Game
from ..base.exceptions import InvalidParameterError
from ..base.types import GameProtocol


def add_games(left: GameProtocol, right: GameProtocol) -> Game:
    """
    Add (sum) two TU cooperative games defined on the same player set.

    The sum game is defined pointwise on coalitions:

    $$
    (v + w)(S) = v(S) + w(S), \\quad \\forall S \\subseteq N,
    $$

    with the TU convention preserved:

    $$
    (v+w)(\\varnothing) = 0.
    $$

    Parameters
    ----------
    left : Game
        First game (characteristic function $v$).
    right : Game
        Second game (characteristic function $w$).

    Returns
    -------
    Game
        The sum game $v+w$.

    Raises
    ------
    InvalidParameterError
        If the two games do not have the same number of players.

    Notes
    -----
    - TU games on a fixed player set form a **vector space** under pointwise
      addition and scalar multiplication. This function implements the
      addition operation.
    - Many solution concepts are **additive** (linearity). For example, the
      Shapley value satisfies:

      $$
      \\phi(v+w) = \\phi(v) + \\phi(w).
      $$

      Additivity is one of the classic axioms characterizing the Shapley value.
    - Player labels: if `left.player_labels` is available it is used;
      otherwise `right.player_labels` is used.

    Examples
    --------
    >>> g_sum = add_games(g1, g2)
    >>> g_sum.value(g_sum.grand_coalition) == g1.value(g1.grand_coalition) + g2.value(g2.grand_coalition)
    True
    """
    if left.n_players != right.n_players:
        raise InvalidParameterError("Both games must have the same n_players")
    n = left.n_players
    v = {mask: left.value(mask) + right.value(mask) for mask in all_coalitions(n)}
    v[0] = 0.0
    left_labels = getattr(left, "player_labels", None)
    right_labels = getattr(right, "player_labels", None)
    labels = left_labels if left_labels is not None else right_labels
    return Game(n_players=n, v=v, player_labels=labels)


def sub_games(left: GameProtocol, right: GameProtocol) -> Game:
    """
    Subtract two TU cooperative games defined on the same player set.

    The difference game is defined pointwise on coalitions:

    $$
    (v - w)(S) = v(S) - w(S), \\quad \\forall S \\subseteq N,
    $$

    with the TU convention preserved:

    $$
    (v-w)(\\varnothing) = 0.
    $$

    Parameters
    ----------
    left : Game
        Minuend game (characteristic function $v$).
    right : Game
        Subtrahend game (characteristic function $w$).

    Returns
    -------
    Game
        The difference game $v-w$.

    Raises
    ------
    InvalidParameterError
        If the two games do not have the same number of players.

    Notes
    -----
    - Subtraction is addition with a negative scalar multiple:
      $v - w = v + (-1) w$.
    - This operation is useful for:
        * comparing games (e.g., incremental contribution of a feature),
        * defining derived games (e.g., $v - u$ where $u$ is an additive/inessential part),
        * diagnostics (checking how far a game is from a reference class).
    - Player labels: if `left.player_labels` is available it is used;
      otherwise `right.player_labels` is used.

    Examples
    --------
    >>> g_diff = sub_games(g1, g2)
    >>> g_diff.value(0)  # empty coalition remains 0
    0.0
    """
    if left.n_players != right.n_players:
        raise InvalidParameterError("Both games must have the same n_players")
    n = left.n_players
    v = {mask: left.value(mask) - right.value(mask) for mask in all_coalitions(n)}
    v[0] = 0.0
    left_labels = getattr(left, "player_labels", None)
    right_labels = getattr(right, "player_labels", None)
    labels = left_labels if left_labels is not None else right_labels
    return Game(n_players=n, v=v, player_labels=labels)
