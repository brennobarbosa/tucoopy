"""
# Set-valued (geometric) solution concepts for cooperative games.

The `tucoop.geometry` package groups **set-valued** objects such as:

- `Core`, `EpsilonCore`, `LeastCore`,
- `ImputationSet`, `PreImputationSet`,
- `CoreCover`, `ReasonableSet`, and `WeberSet`.

Many objects expose a polyhedral representation via a ``.poly`` property when
appropriate. Heavy computations (vertex enumeration, projections) are intended
for small $n$ and are guarded by explicit limits in :mod:`tucoop.base.config`.

Examples
--------
Compute the core of a small 3-player game and check membership:

>>> from tucoop import Game
>>> from tucoop.geometry import Core
>>> g = Game.from_coalitions(n_players=3, values={
...     0:0, 1:1, 2:1.2, 3:2.8,
...     4:0.8, 5:2.2, 6:2.0,
...     7:4.0,
... })
>>> core = Core(g)
>>> core.contains([1.5, 1.5, 1.0])
True
"""

from .core_set import Core
from .imputation_set import (
    ImputationProjection,
    ImputationSet,
    PreImputationSet,
    imputation_lower_bounds,
    project_to_imputation,
)
from .epsilon_core_set import EpsilonCore
from .weber_set import WeberSet, marginal_vector, mean_marginal_vector, weber_marginal_vectors, weber_sample
from .core_cover_set import CoreCover
from .reasonable_set import ReasonableSet
from .bargaining_set import BargainingCheckResult, BargainingSet, Objection
from .polyhedron import PolyhedralSet
from .least_core_set import LeastCore
from .kernel_set import KernelSet, PreKernelSet
from .projection import allocation_to_barycentric_imputation, project_allocation
from .sampling import sample_imputation_set

__all__ = [
    "Core",
    "PolyhedralSet",
    "imputation_lower_bounds",
    "project_to_imputation",
    "ImputationProjection",
    "PreImputationSet",
    "ImputationSet",
    "EpsilonCore",
    "marginal_vector",
    "mean_marginal_vector",
    "weber_marginal_vectors",
    "weber_sample",
    "WeberSet",
    "ReasonableSet",
    "CoreCover",
    "LeastCore",
    "PreKernelSet",
    "KernelSet",
    "allocation_to_barycentric_imputation",
    "project_allocation",
    "sample_imputation_set",
    "BargainingSet",
    "BargainingCheckResult",
    "Objection",
]
