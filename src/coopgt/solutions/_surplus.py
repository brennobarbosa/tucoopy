"""
# Internal surplus computation helpers.

This module contains helper routines for kernel/prekernel computations.
It is internal (prefixed with `_`) and not part of the public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from ..base.coalition import subcoalitions, grand_coalition
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError


def _all_values_list(game: GameProtocol) -> list[float]:
    n = int(game.n_players)
    m = 1 << n
    return [float(game.value(mask)) for mask in range(m)]


def _coalition_sums_dp(x: Sequence[float], *, n_players: int) -> list[float]:
    """
    Compute $x(S)$ for all masks via a subset-sum DP in O(n 2^n).
    """
    n = int(n_players)
    m = 1 << n
    out = [0.0] * m
    xv = [float(v) for v in x]
    for mask in range(1, m):
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        out[mask] = out[mask ^ lsb] + xv[i]
    return out


@dataclass(frozen=True)
class SurplusEvaluator:
    """
    Helper for pairwise surplus computations used by the (pre-)kernel.

    This class precomputes a list of coalition masks for each ordered pair 
    $(i, j)$ with $i \\neq j$, representing all coalitions that contain $i$
    and exclude $j$.

    It also provides an O(n 2^n) DP for $x(S)$, enabling fast evaluation of
    $s_{ij}(x)$ over many pairs without recomputing coalition sums repeatedly.
    """

    n_players: int
    pair_masks: list[list[list[int]]]

    @staticmethod
    def for_n_players(n_players: int) -> "SurplusEvaluator":
        n = int(n_players)
        if n < 1:
            raise InvalidParameterError("n_players must be >= 1")

        N = grand_coalition(n)
        pair_masks: list[list[list[int]]] = [[[] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                base = 1 << i
                free = int(N) & ~(1 << i) & ~(1 << j)
                masks: list[int] = []
                for T in subcoalitions(free):
                    masks.append(int(base | int(T)))
                pair_masks[i][j] = masks

        return SurplusEvaluator(n_players=n, pair_masks=pair_masks)

    def surplus_and_argmax(
        self,
        game: GameProtocol,
        x: Sequence[float],
        i: int,
        j: int,
        *,
        values: list[float] | None = None,
        x_sums: list[float] | None = None,
        tol: float = 1e-15,
        approx_max_coalitions: int | None = None,
        rng: Random | None = None,
    ) -> tuple[float, int]:
        """
        Compute $s_{ij}(x)$ and an argmax coalition.

        If ``approx_max_coalitions`` is provided, evaluates the maximum over a
        random subset of candidate coalitions for $(i, j)$.
        """
        if i == j:
            raise InvalidParameterError("i and j must be different")
        if x_sums is None:
            x_sums = _coalition_sums_dp(x, n_players=self.n_players)

        masks = self.pair_masks[i][j]
        if approx_max_coalitions is not None:
            k = int(approx_max_coalitions)
            if k <= 0:
                raise InvalidParameterError("approx_max_coalitions must be >= 1 when provided")
            if rng is None:
                rng = Random(0)
            if k < len(masks):
                # sample without replacement
                masks = rng.sample(masks, k=k)

        best = -float("inf")
        bestS = 0
        eps = float(tol)
        for S in masks:
            vS = float(values[S]) if values is not None else float(game.value(int(S)))
            e = float(vS) - float(x_sums[S])
            if e > best + eps or (abs(e - best) <= eps and int(S) < int(bestS)):
                best = float(e)
                bestS = int(S)
        return float(best), int(bestS)


__all__ = [
    "SurplusEvaluator",
]
