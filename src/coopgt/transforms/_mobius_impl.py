"""
# Internal implementations for Möbius transforms.

This module contains the actual transform kernels used by
`tucoop.transforms.mobius.mobius_transform` and
`tucoop.transforms.mobius.inverse_mobius_transform`.

It is kept private (prefixed with `_`) to allow optimization changes without
breaking the public API.
"""

from __future__ import annotations

from typing import Sequence

from ..backends.numpy_fast import require_numpy
from ..base.exceptions import InvalidParameterError


def _infer_n_players_from_len(m: int) -> int:
    if m <= 0 or (m & (m - 1)) != 0:
        raise InvalidParameterError("values length must be a power of two (2^n)")
    return int(m).bit_length() - 1


def _validate(values: Sequence[float], n_players: int | None) -> tuple[int, int]:
    m = len(values)
    n = int(n_players) if n_players is not None else _infer_n_players_from_len(m)
    if m != (1 << n):
        raise InvalidParameterError("values length must be exactly 2^n_players")
    return m, n


# -------------------------
# Fallback (pure Python)
# -------------------------


def _mobius_py(values: Sequence[float], n: int) -> list[float]:
    # branchless subset zeta/mobius butterfly over blocks
    f = [float(v) for v in values]
    m = 1 << n
    for i in range(n):
        half = 1 << i
        block = half << 1
        for base in range(0, m, block):
            lo = base
            hi = base + half
            # update high half using low half (low half is not modified in this stage)
            for j in range(half):
                f[hi + j] -= f[lo + j]
    return f


def _inv_mobius_py(values: Sequence[float], n: int) -> list[float]:
    f = [float(v) for v in values]
    m = 1 << n
    for i in range(n):
        half = 1 << i
        block = half << 1
        for base in range(0, m, block):
            lo = base
            hi = base + half
            for j in range(half):
                f[hi + j] += f[lo + j]
    return f


# -------------------------
# NumPy fast (optional)
# -------------------------


def _mobius_numpy(values: Sequence[float], n: int) -> list[float]:
    np = require_numpy(context="mobius_transform")
    f = np.asarray(values, dtype=float).copy()
    m = 1 << n
    # f is 1D length m
    for i in range(n):
        half = 1 << i
        block = half << 1
        for base in range(0, m, block):
            f[base + half : base + block] -= f[base : base + half]
    # Return python list to keep API stable
    return f.tolist()


def _inv_mobius_numpy(values: Sequence[float], n: int) -> list[float]:
    np = require_numpy(context="inverse_mobius_transform")
    f = np.asarray(values, dtype=float).copy()
    m = 1 << n
    for i in range(n):
        half = 1 << i
        block = half << 1
        for base in range(0, m, block):
            f[base + half : base + block] += f[base : base + half]
    return f.tolist()
