"""
## Default configurations and constants for the tucoop package.

This module centralizes default values used throughout the library, such as
tolerances and conservative limits for potentially expensive computations.

Attributes
----------
DEFAULT_TOL : float
    Default tolerance for numerical checks.
DEFAULT_MAX_PLAYERS : int
    Default maximum number of players for some operations.

Examples
--------
>>> from tucoop.base.config import DEFAULT_TOL
>>> DEFAULT_TOL
1e-9
"""

from __future__ import annotations

DEFAULT_TOL: float = 1e-9

# Tolerances used by different submodules.
DEFAULT_GEOMETRY_TOL: float = DEFAULT_TOL
DEFAULT_LP_TOL: float = DEFAULT_TOL

# Default safety caps for exact geometry routines.
# - max_players: enumeration that is exponential in n_players (e.g., vertices routines).
# - max_dim: projection/extreme-points routines that are exponential in dimension.
DEFAULT_GEOMETRY_MAX_PLAYERS: int = 6
DEFAULT_GEOMETRY_MAX_DIM: int = 6

# Default safety cap for "build_analysis"/demos (overall), independent of geometry internals.
DEFAULT_MAX_PLAYERS: int = 4

# Geometry sampling defaults.
DEFAULT_IMPUTATION_SAMPLE_TOL: float = 1e-12
DEFAULT_HIT_AND_RUN_BURN_IN: int = 50
DEFAULT_HIT_AND_RUN_THINNING: int = 1
DEFAULT_HIT_AND_RUN_TOL: float = DEFAULT_GEOMETRY_TOL

# Bargaining set (heuristic, small-n).
DEFAULT_BARGAINING_TOL: float = 1e-8

# Kernel / pre-kernel (small-n numerical heuristics).
DEFAULT_KERNEL_TOL: float = 1e-8
DEFAULT_KERNEL_MAX_ITER: int = 200
DEFAULT_KERNEL_MAX_PLAYERS: int = 12

# Approximation: when set, evaluate at most this many coalitions per (i, j).
# Keep as None by default for deterministic exact checks.
DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR: int | None = None
DEFAULT_KERNEL_APPROX_SEED: int | None = None
