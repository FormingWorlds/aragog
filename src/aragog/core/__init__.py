"""Staged core-evolution module.

Replaces the isothermal-reservoir core boundary condition with a core that
carries its own state: energy budget with inner-core nucleation (B1), entropy
budget and dynamo diagnostics (B2), and stable stratification with
crystallization-regime flags (B3). The module receives the CMB heat flow and
returns the CMB temperature; with every feature disabled it reproduces the
isothermal-reservoir closure, which is the regression anchor.

All radial structure uses the closed-form Gaussian profile family
(Labrosse et al. 2001; Nimmo 2015, Treatise on Geophysics 9.08), so every
budget term is an analytic integral and the whole module stays at ODE cost.
"""

from __future__ import annotations

from aragog.core.budget import CoreEnergyBudget
from aragog.core.entropy import CoreEntropyBudget
from aragog.core.melting import IronMeltingCurve, QuadraticMeltingCurve
from aragog.core.module import CoreModule, build_core_module_budget
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.regime import REGIME_NAMES, crystallization_regime, regime_name
from aragog.core.stratification import adiabatic_ratio, stratification_depth

__all__ = [
    'CoreEnergyBudget',
    'CoreEntropyBudget',
    'CoreModule',
    'GaussianCoreProfiles',
    'IronMeltingCurve',
    'QuadraticMeltingCurve',
    'build_core_module_budget',
    'REGIME_NAMES',
    'crystallization_regime',
    'regime_name',
    'adiabatic_ratio',
    'stratification_depth',
]
