"""
# Diagnostics helpers.

This subpackage groups small, UI-friendly routines that explain *why* an
allocation is (or is not) in a given set or satisfies a given property.

Design
------
- Most functions return small dataclasses with a ``to_dict()`` helper so they can
  be serialized into JSON (for demos and analysis reports).
- Coalition scans are shared where possible (to avoid duplicating the same
  exponential loops across modules).

Convenience imports
-------------------
This ``__init__`` re-exports the most frequently used diagnostics so users can:

>>> from tucoopy.diagnostics import core_diagnostics, check_allocation

without needing to remember module paths.

Examples
--------
>>> from tucoopy.diagnostics import core_diagnostics, is_efficient
>>> from tucoopy.base.game import Game
>>> g = Game.from_coalitions(n_players=2, values={(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): 1.0})
>>> x = [0.5, 0.5]
>>> core_diagnostics(g, x).in_core
True
>>> is_efficient(g, x)
True
"""
from .core_diagnostics import (
    CoreDiagnostics,
    CoreViolation,
    TightCoalitions,
    core_diagnostics,
    core_frame_highlight,
    core_violations,
    excesses,
    explain_core_membership,
    is_in_core,
    is_in_epsilon_core,
    max_excess,
    tight_coalitions,
)
from .allocation_diagnostics import (
    AllocationChecks,
    check_allocation,
    is_efficient,
    is_imputation,
    is_individually_rational,
)
from .linprog_diagnostics import LinprogDiagnostics, build_lp_explanations, explain_linprog, linprog_diagnostics
from .bounds import BoundViolation, BoxBoundSetDiagnostics
from .imputation_diagnostics import ImputationDiagnostics, imputation_diagnostics
from .core_cover_diagnostics import core_cover_diagnostics
from .reasonable_diagnostics import reasonable_set_diagnostics
from .epsilon_core_diagnostics import EpsilonCoreDiagnostics, EpsilonCoreViolation, epsilon_core_diagnostics
from .least_core_diagnostics import LeastCoreDiagnostics, least_core_diagnostics
from .blocking_regions import BlockingRegion, BlockingRegions, blocking_regions

__all__ = [
    "core_diagnostics",
    "core_frame_highlight",
    "CoreDiagnostics",
    "CoreViolation",
    "core_violations",
    "excesses",
    "max_excess",
    "is_efficient",
    "is_individually_rational",
    "is_imputation",
    "is_in_core",
    "is_in_epsilon_core",
    "tight_coalitions",
    "TightCoalitions",
    "AllocationChecks",
    "check_allocation",
    "explain_core_membership",
    "LinprogDiagnostics",
    "linprog_diagnostics",
    "explain_linprog",
    "build_lp_explanations",
    "BoundViolation",
    "ImputationDiagnostics",
    "imputation_diagnostics",
    "BoxBoundSetDiagnostics",
    "core_cover_diagnostics",
    "reasonable_set_diagnostics",
    "EpsilonCoreViolation",
    "EpsilonCoreDiagnostics",
    "epsilon_core_diagnostics",
    "LeastCoreDiagnostics",
    "least_core_diagnostics",
    "blocking_regions",
    "BlockingRegions",
    "BlockingRegion",
]
