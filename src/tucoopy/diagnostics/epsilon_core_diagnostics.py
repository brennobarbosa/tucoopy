"""
# Epsilon-core diagnostics.

This module provides :func:`epsilon_core_diagnostics`, which evaluates whether a
payoff vector $x$ belongs to the epsilon-core of a transferable-utility (TU)
game.

Definitions
-----------
For a TU game with characteristic function $v$ and an allocation $x$:

- Coalition sum: $x(S) = \\sum_{i in S} x_i$
- Excess: $e(S, x) = v(S) - x(S)$

The **epsilon-core** consists of efficient allocations such that:

$$\\max_{\\{S \\subset N, S \\neq \\varnothing\\}} e(S, x) \\leq \\epsilon$$.

Notes
-----
- This diagnostic scans all coalitions (excluding $\\varnothing$ and $N$), so it is
  exponential in ``n_players``.
- Numerical comparisons use ``tol`` for efficiency checks and tie detection.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.diagnostics.epsilon_core_diagnostics import epsilon_core_diagnostics
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> d = epsilon_core_diagnostics(g, [0.5, 0.5], epsilon=0.0)
>>> d.in_set
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..base.coalition import players
from ._excess_scan import scan_excesses

@dataclass(frozen=True)
class EpsilonCoreViolation:
    """
    One epsilon-core inequality evaluation at an allocation $x$.

    The epsilon-core constraints are:

    $v(S) - x(S) \\leq \\epsilon$ for all proper non-empty coalitions $S$.

    Attributes
    ----------
    coalition_mask
        Coalition bitmask.
    players
        List of player indices in the coalition.
    vS
        Coalition value ``v(S)``.
    xS
        Coalition payoff sum ``x(S)``.
    excess
        Excess ``e(S, x) = v(S) - x(S)``.

    Examples
    --------
    >>> v = EpsilonCoreViolation(1, [0], 1.0, 0.8, 0.2)
    >>> v.excess
    0.2
    """

    coalition_mask: int
    players: list[int]
    vS: float
    xS: float
    excess: float


@dataclass(frozen=True)
class EpsilonCoreDiagnostics:
    """
    Diagnostics for epsilon-core membership at an allocation $x$.

    In addition to efficiency, the epsilon-core membership test is:

    $$ \\text{max excess} \\leq \\epsilon $$.

    Attributes
    ----------
    n_players
        Number of players.
    vN
        Grand coalition value ``v(N)``.
    sum_x
        Sum of the allocation vector.
    efficient
        Whether ``sum_x`` matches ``vN`` within ``tol``.
    epsilon
        The epsilon parameter of the epsilon-core.
    in_set
        Whether ``x`` belongs to the epsilon-core with the provided epsilon.
    max_excess
        Maximum excess over all proper non-empty coalitions.
    tight_coalitions
        Coalitions attaining ``max_excess`` (ties within ``tol``).
    violations
        Most violated coalitions, sorted by excess descending.

    Examples
    --------
    >>> d = EpsilonCoreDiagnostics(
    ...     n_players=2,
    ...     vN=1.0,
    ...     sum_x=1.0,
    ...     efficient=True,
    ...     epsilon=0.0,
    ...     in_set=True,
    ...     max_excess=-0.5,
    ...     tight_coalitions=[1, 2],
    ...     violations=[],
    ... )
    >>> d.in_set
    True
    """

    n_players: int
    vN: float
    sum_x: float
    efficient: bool
    epsilon: float
    in_set: bool
    max_excess: float
    tight_coalitions: list[int]
    violations: list[EpsilonCoreViolation]

    def to_dict(self) -> dict[str, object]:
        """
        Convert diagnostics to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation of the diagnostics dataclass.

        Examples
        --------
        >>> d = EpsilonCoreDiagnostics(
        ...     n_players=2,
        ...     vN=1.0,
        ...     sum_x=1.0,
        ...     efficient=True,
        ...     epsilon=0.0,
        ...     in_set=True,
        ...     max_excess=-0.5,
        ...     tight_coalitions=[1, 2],
        ...     violations=[],
        ... )
        >>> d.to_dict()["epsilon"]
        0.0
        """
        return asdict(self)


def epsilon_core_diagnostics(
    game: GameProtocol,
    x: list[float],
    *,
    epsilon: float,
    tol: float = DEFAULT_GEOMETRY_TOL,
    top_k: int = 8,
) -> EpsilonCoreDiagnostics:
    """
    Compute epsilon-core membership diagnostics for an allocation $x$.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector of length ``game.n_players``.
    epsilon
        Epsilon parameter for the epsilon-core.
    tol
        Numerical tolerance used for:
        - efficiency check (``abs(sum(x) - v(N)) <= tol``),
        - tie detection for ``tight_coalitions``.
    top_k
        Maximum number of violating coalitions to include in ``violations``.

    Returns
    -------
    EpsilonCoreDiagnostics
        Diagnostics object including ``in_set`` and the most violated coalitions.

    Raises
    ------
    InvalidParameterError
        If ``x`` does not have length ``game.n_players``.

    Notes
    -----
    Internally, this uses a shared coalition scanner (`scan_excesses`) to
    avoid duplicating the exponential coalition loop across diagnostics.

    Examples
    --------
    A minimal 2-player example (only the grand coalition has value 1):

    >>> from tucoopy import Game
    >>> from tucoopy.diagnostics.epsilon_core_diagnostics import epsilon_core_diagnostics
    >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
    >>> d = epsilon_core_diagnostics(g, [0.5, 0.5], epsilon=0.0)
    >>> d.efficient
    True
    >>> d.in_set
    True
    """
    n = game.n_players
    if len(x) != n:
        raise InvalidParameterError("x must have length n_players")

    grand = game.grand_coalition
    vN = float(game.value(grand))
    sum_x = float(sum(x))
    efficient = abs(sum_x - vN) <= tol

    mx, tight, raw = scan_excesses(game, x, tie_tol=float(tol), violation_threshold=float(epsilon) + float(tol))
    rows: list[EpsilonCoreViolation] = []
    for r in raw:
        rows.append(
            EpsilonCoreViolation(
                coalition_mask=int(r.coalition_mask),
                players=list(players(int(r.coalition_mask), n_players=n)),
                vS=float(r.vS),
                xS=float(r.xS),
                excess=float(r.excess),
            )
        )
    rows.sort(key=lambda rr: (-rr.excess, rr.coalition_mask))
    in_set = efficient and mx <= float(epsilon) + tol
    return EpsilonCoreDiagnostics(
        n_players=n,
        vN=vN,
        sum_x=sum_x,
        efficient=efficient,
        epsilon=float(epsilon),
        in_set=in_set,
        max_excess=float(mx),
        tight_coalitions=list(tight),
        violations=rows[: max(0, int(top_k))],
    )

__all__ = [
    "EpsilonCoreViolation",
    "EpsilonCoreDiagnostics",
    "epsilon_core_diagnostics",
]
