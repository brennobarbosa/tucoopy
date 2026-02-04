"""
# Power indices for simple cooperative games.

This package keeps a stable public API at `tucoop.power`.
Implementations live in submodules; `__init__` only re-exports.
"""

from __future__ import annotations

from .banzhaf import banzhaf_index, banzhaf_index_weighted_voting
from .coleman import (
    coleman_collectivity_power_to_act,
    coleman_initiate_index,
    coleman_prevent_index,
)
from .deegan_packel import deegan_packel_index
from .holler import holler_index
from .johnston import johnston_index
from .rae import rae_index
from .koenig_brauninger import koenig_brauninger_index
from .shapley_shubik import shapley_shubik_index, shapley_shubik_index_weighted_voting
from .egalitarian_shapley import egalitarian_shapley_value
from .solidarity import solidarity_value

__all__ = [
    "shapley_shubik_index",
    "banzhaf_index",
    "deegan_packel_index",
    "holler_index",
    "johnston_index",
    "shapley_shubik_index_weighted_voting",
    "banzhaf_index_weighted_voting",
    "coleman_collectivity_power_to_act",
    "coleman_prevent_index",
    "coleman_initiate_index",
    "rae_index",
    "koenig_brauninger_index",
    "egalitarian_shapley_value",
    "solidarity_value",
]
