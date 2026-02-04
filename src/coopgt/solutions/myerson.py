"""
# Myerson value (communication graph).

This module computes the Myerson value by applying the Myerson restriction induced by a communication graph
and then computing the Shapley value of the restricted game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..base.types import GameProtocol
from ..transforms.communication import myerson_restriction
from .shapley import shapley_value


@dataclass(frozen=True)
class MyersonResult:
    """
    Result container for the Myerson value.

    Attributes
    ----------
    x : list[float]
        Allocation vector (length `n_players`).
    meta : dict[str, object]
        Diagnostic metadata about the computation parameters.
    """
    x: list[float]
    meta: dict[str, object]


def myerson_value(
    game: GameProtocol,
    *,
    edges: Iterable[tuple[int, int]],
    max_players: int = 16,
    require_complete: bool = True,
) -> MyersonResult:
    """
    Compute the **Myerson value** for a TU game with communication constraints.

    In cooperative games with restricted communication, players can only
    cooperate effectively within connected components of a communication graph.
    Given an undirected graph $G$, define the **Myerson restricted game**:

    $$
    v_G(S) = \\sum_{C \\in \\mathrm{components}_G(S)} v(C).
    $$

    The Myerson value is then defined as the Shapley value of this
    restricted game:

    $$
    \\phi^{\\text{Myerson}}(v, G) = \\phi(v_G).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.
    edges : Iterable[tuple[int, int]]
        Undirected edges of the communication graph.
    max_players : int, default=16
        Safety bound. The computation is exponential in `n_players`.
    require_complete : bool, default=True
        If True, require the original game to be complete.

    Returns
    -------
    MyersonResult
        Allocation vector and metadata.

    Raises
    ------
    NotSupportedError
        If the number of players exceeds `max_players`.
    InvalidGameError
        If `require_complete=True` and the game is incomplete.
    InvalidParameterError
        If the communication graph `edges` is invalid (e.g. out-of-range endpoints).

    Notes
    -----
    - If the graph is complete (all players connected), the Myerson value
      reduces to the Shapley value.
    - If the graph has no edges, each player acts alone and the value reduces
      to the singleton payoffs.
    - The Myerson value models cooperation under communication constraints
      and is a fundamental concept in network games.

    Examples
    --------
    >>> res = myerson_value(g, edges=[(0,1), (1,2)])
    >>> res.x
    """
    gG = myerson_restriction(game, edges=edges, require_complete=require_complete, max_players=max_players)
    x = shapley_value(gG)
    return MyersonResult(
        x=[float(v) for v in x],
        meta={"max_players": int(max_players), "require_complete": bool(require_complete)},
    )


__all__ = ["MyersonResult", "myerson_value"]
