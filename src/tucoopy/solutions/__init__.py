"""
# Point-valued solution concepts for TU cooperative games.

This package contains algorithms that return a single payoff vector (allocation / imputation),
such as the Shapley value, (normalized) Banzhaf value, nucleolus,
and related solution concepts.

`__init__.py` re-exports the stable public API from the submodules.
"""

from __future__ import annotations

from .banzhaf import banzhaf_value, normalized_banzhaf_value, weighted_banzhaf_value
from .cis import cis_value
from .egalitarian import egalitarian_value
from .ensc import ensc_value
from .esd import esd_value
from .gately import GatelyResult, gately_point
from .myerson import MyersonResult, myerson_value
from .modiclus import ModiclusResult, modiclus
from .nucleolus import NucleolusResult, nucleolus, prenucleolus
from .owen import OwenResult, owen_value
from .proportional import proportional_value
from .shapley import semivalue, shapley_value, shapley_value_sample, weighted_shapley_value
from .solve import SolveResult, solve
from .tau import minimal_rights, tau_value, utopia_payoff

__all__ = [
    "shapley_value",
    "shapley_value_sample",
    "weighted_shapley_value",
    "semivalue",
    "banzhaf_value",
    "normalized_banzhaf_value",
    "weighted_banzhaf_value",
    "solve",
    "SolveResult",
    "nucleolus",
    "prenucleolus",
    "NucleolusResult",
    "tau_value",
    "utopia_payoff",
    "minimal_rights",
    "gately_point",
    "GatelyResult",
    "myerson_value",
    "MyersonResult",
    "modiclus",
    "ModiclusResult",
    "owen_value",
    "OwenResult",
    "cis_value",
    "egalitarian_value",
    "ensc_value",
    "esd_value",
    "proportional_value",
]
