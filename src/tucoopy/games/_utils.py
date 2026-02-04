"""
# Internal helpers for game generators.

This module contains small internal validation utilities used across
`tucoopy.games`.

Examples
--------
>>> _validate_n_players(1)
1
"""

from __future__ import annotations

from ..base.exceptions import InvalidParameterError


def _validate_n_players(n: int) -> int:
    """
    Validate that a game generator has at least one player.

    Parameters
    ----------
    n
        Candidate number of players.

    Returns
    -------
    int
        The validated integer value of `n`.

    Raises
    ------
    InvalidParameterError
        If `n < 1`.

    Examples
    --------
    >>> _validate_n_players(3)
    3
    """
    n = int(n)
    if n < 1:
        raise InvalidParameterError("need at least 1 player")
    return n
