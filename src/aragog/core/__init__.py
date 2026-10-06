"""Core evolution module.

Replaces the isothermal-reservoir core boundary condition with a core that
carries its own state: energy budget with inner-core nucleation, entropy
budget and dynamo diagnostics, and stable stratification with
crystallization-regime flags. The module receives the CMB heat flow and
returns the CMB temperature; with every feature disabled it reproduces the
isothermal-reservoir closure, which is the regression anchor.

All radial structure uses the closed-form Gaussian profile family
(Labrosse et al. 2001; Nimmo 2015, Treatise on Geophysics 9.08), so every
budget term is an analytic integral and the whole module stays at ODE cost.
"""

from __future__ import annotations

from aragog.core.boundary_layer import RA_CRIT_CMB_DEFAULT, cmb_boundary_layer_flux
from aragog.core.budget import CoreEnergyBudget
from aragog.core.entropy import CoreEntropyBudget
from aragog.core.melting import IronMeltingCurve, QuadraticMeltingCurve
from aragog.core.module import CoreModule, build_core_module_budget
from aragog.core.profiles import GaussianCoreProfiles, fit_gaussian_core_profiles
from aragog.core.regime import REGIME_NAMES, crystallization_regime, regime_name
from aragog.core.stratification import adiabatic_ratio, stratification_depth

__all__ = [
    'CoreEnergyBudget',
    'CoreEntropyBudget',
    'CoreModule',
    'GaussianCoreProfiles',
    'fit_gaussian_core_profiles',
    'IronMeltingCurve',
    'QuadraticMeltingCurve',
    'build_core_module_budget',
    'cmb_boundary_layer_flux',
    'RA_CRIT_CMB_DEFAULT',
    'REGIME_NAMES',
    'crystallization_regime',
    'regime_name',
    'adiabatic_ratio',
    'stratification_depth',
]
