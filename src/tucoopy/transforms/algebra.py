"""
# Algebraic transformations of TU games.

The functions in this module build new games from an existing one by applying
simple algebraic operations to the characteristic function:

- ``scale_game``: multiply all coalition values by a scalar.
- ``shift_game``: add a constant (with TU conventions preserved).
- ``affine_game``: combine scale and shift.

These are useful for normalization, unit changes, and testing linearity /
scale-equivariance properties of solution concepts.
"""

from __future__ import annotations

from ..base.coalition import all_coalitions
from ..base.game import Game
from ..base.types import GameProtocol


def scale_game(game: GameProtocol, factor: float) -> Game:
    """
    Scale a TU game by a multiplicative factor.

    This transformation defines a new game whose characteristic function is

    $$
    (a v)(S) = a \\, v(S),
    $$

    for every coalition $S$, with the TU convention preserved:

    $$
    (a v)(\\varnothing) = 0.
    $$

    Parameters
    ----------
    game : Game
        Original TU game.
    factor : float
        Scaling factor $a$.

    Returns
    -------
    Game
        New scaled game.

    Notes
    -----
    Many solution concepts are *scale-equivariant*. For example,

    - Shapley value
    - Nucleolus
    - Kernel
    - Core

    all scale proportionally when the game is scaled.

    This operation is useful for normalization and for comparing games
    under different units (e.g., cost vs revenue).

    Examples
    --------
    >>> g2 = scale_game(g, 2.0)
    >>> g2.value(g2.grand_coalition) == 2 * g.value(g.grand_coalition)
    True
    """
    a = float(factor)
    v = {mask: a * game.value(mask) for mask in all_coalitions(game.n_players)}
    v[0] = 0.0
    labels = getattr(game, "player_labels", None)
    return Game(n_players=game.n_players, v=v, player_labels=labels)


def shift_game(game: GameProtocol, constant: float) -> Game:
    """
    Add a constant to all non-empty coalitions of a TU game.

    This transformation defines a new game

    $$
    (v + c)(S) =
        \\begin{cases}
            0, & S = \\varnothing \\\\
            v(S) + c, & S \\neq \\varnothing
        \\end{cases}
    $$

    preserving the TU requirement $v(\\varnothing)=0$.

    Parameters
    ----------
    game : Game
        Original TU game.
    constant : float
        Additive constant $c$.

    Returns
    -------
    Game
        New shifted game.

    Notes
    -----
    This is an **affine translation** of the game.

    Many solution concepts are *translation-invariant*, meaning their
    payoff allocations do not change under this transformation. This includes:

    - Shapley value
    - Nucleolus
    - Kernel
    - Core membership

    This transformation is central when studying **equivalent games**
    and normalizations such as zero-normalized games.

    Examples
    --------
    >>> g2 = shift_game(g, 5.0)
    >>> g2.value(0)  # empty coalition remains 0
    0.0
    >>> g2.value(g2.grand_coalition) == g.value(g.grand_coalition) + 5
    True
    """
    c = float(constant)
    v: dict[int, float] = {}
    for mask in all_coalitions(game.n_players):
        v[mask] = 0.0 if mask == 0 else game.value(mask) + c
    v[0] = 0.0
    labels = getattr(game, "player_labels", None)
    return Game(n_players=game.n_players, v=v, player_labels=labels)


def affine_game(game: GameProtocol, factor: float, constant: float) -> Game:
    """
    Apply an affine transformation to the characteristic function of a TU game.

    The new game is defined by

    $$
    (a v + b)(S) =
        \\begin{cases}
            0, & S = \\varnothing \\\\
            a \\, v(S) + b, & S \\neq \\varnothing
        \\end{cases}
    $$

    which combines scaling and translation while preserving the TU convention.

    Parameters
    ----------
    game : Game
        Original TU game.
    factor : float
        Multiplicative factor $a$.
    constant : float
        Additive constant $b$.

    Returns
    -------
    Game
        New affine-transformed game.

    Notes
    -----
    Affine transformations define an **equivalence class** of TU games
    that share the same strategic structure.

    Most central solution concepts are invariant (up to scaling) under
    affine transformations. Studying games up to affine equivalence is
    standard in cooperative game theory, particularly in:

    - normalization procedures,
    - theoretical characterizations of values,
    - geometric analysis of the core and related sets.

    Examples
    --------
    >>> g2 = affine_game(g, 2.0, 3.0)
    >>> g2.value(0)
    0.0
    """
    a = float(factor)
    b = float(constant)
    v: dict[int, float] = {}
    for mask in all_coalitions(game.n_players):
        v[mask] = 0.0 if mask == 0 else a * game.value(mask) + b
    v[0] = 0.0
    labels = getattr(game, "player_labels", None)
    return Game(n_players=game.n_players, v=v, player_labels=labels)
