"""
# Least-squares imputation.

This module provides a simple projection-based imputation, useful as a baseline or initialization.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ..base.exceptions import InvalidGameError
from ..geometry.imputation_set import project_to_imputation


def least_squares_imputation(game: GameProtocol, x0: list[float]) -> list[float]:
    """
    Least-squares imputation: Euclidean projection of $x_0$ onto the imputation set.
    """
    res = project_to_imputation(game, x0)
    if not res.feasible:
        raise InvalidGameError("least_squares_imputation undefined: imputation set is empty")
    return [float(v) for v in res.x]


__all__ = ["least_squares_imputation"]
