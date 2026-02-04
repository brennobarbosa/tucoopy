"""
# Internal helpers for scanning coalition excesses.

This module contains core-family helper routines that compute max-excess values
and identify ties efficiently using precomputed coalition sums.

It is internal (prefixed with `_`) and may change without notice.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.diagnostics._excess_scan import scan_excesses
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> mx, tight, rows = scan_excesses(g, [0.5, 0.5], tie_tol=1e-9, violation_threshold=0.0)
>>> mx
-0.5
>>> tight
[1, 2]
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.coalition import coalition_sums
from ..base.types import GameProtocol


@dataclass(frozen=True)
class ExcessRow:
    """
    Internal helper row for coalition excess scans.

    Examples
    --------
    >>> r = ExcessRow(coalition_mask=1, vS=0.0, xS=0.5, excess=-0.5)
    >>> r.excess
    -0.5
    """

    coalition_mask: int
    vS: float
    xS: float
    excess: float


def scan_excesses(
    game: GameProtocol,
    x: list[float],
    *,
    tie_tol: float,
    violation_threshold: float,
) -> tuple[float, list[int], list[ExcessRow]]:
    """
    Scan coalition excesses $e(S,x)=v(S)-x(S)$ over proper non-empty coalitions.

    Returns
    -------
    max_excess
        Maximum excess over proper non-empty coalitions.
    argmax_coalitions
        Coalition masks achieving max_excess (ties within tie_tol).
    violations
        Rows with excess > violation_threshold (unsorted).

    Examples
    --------
    >>> from tucoopy import Game
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> mx, tight, rows = scan_excesses(g, [0.5, 0.5], tie_tol=1e-9, violation_threshold=0.0)
    >>> mx
    -0.5
    >>> tight
    [1, 2]
    """
    n = game.n_players
    grand = game.grand_coalition
    x_sum = coalition_sums(x, n_players=n)

    best = float("-inf")
    argmax: list[int] = []
    violations: list[ExcessRow] = []

    for S in range(1 << n):
        if S == 0 or S == grand:
            continue
        vS = float(game.value(int(S)))
        xS = float(x_sum[int(S)])
        e = float(vS - xS)

        # Keep the maximum exact; use `tie_tol` only to decide ties.
        if e > best:
            best = e
            argmax = [int(S)]
        elif abs(e - best) <= tie_tol:
            argmax.append(int(S))

        if e > float(violation_threshold):
            violations.append(ExcessRow(coalition_mask=int(S), vS=vS, xS=xS, excess=e))

    argmax.sort()
    max_excess = float(best if best != float("-inf") else 0.0)
    return max_excess, argmax, violations


__all__ = ["ExcessRow", "scan_excesses"]
