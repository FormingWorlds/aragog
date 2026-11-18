"""Core evolution module.

Replaces the isothermal-reservoir core boundary condition with a core that
carries its own state: energy budget with inner-core nucleation, entropy
budget and dynamo diagnostics, and stable stratification with
crystallization-regime flags. The module receives the CMB heat flow and
returns the CMB temperature; with every feature disabled it reproduces the
isothermal-reservoir closure, which is the regression anchor.

The radial structure is the closed-form Gaussian profile family (Labrosse
et al. 2001; Nimmo 2015, Treatise on Geophysics 9.08), whose budget terms
are analytic integrals, or a Zalmoxis core on an isentrope of the PALEOS
liquid iron table, tabulated once per structure; either way the module
stays at ODE cost.
"""

from __future__ import annotations

from aragog.core.alloy import AlloyLiquidus, eutectic
from aragog.core.boundary_layer import (
    RA_CRIT_CMB_DEFAULT,
    RA_CRIT_UPPER_LAYER,
    check_ra_crit,
    cmb_boundary_layer_flux,
    critical_rayleigh_eq16,
)
from aragog.core.budget import CoreEnergyBudget
from aragog.core.entropy import CoreEntropyBudget
from aragog.core.iron_table import LiquidIronTable
from aragog.core.melting import IronMeltingCurve, QuadraticMeltingCurve
from aragog.core.module import CoreModule, build_core_module_budget, load_core_arrays
from aragog.core.profiles import GaussianCoreProfiles, fit_gaussian_core_profiles
from aragog.core.profiles_tabulated import TabulatedCoreProfiles
from aragog.core.regime import REGIME_NAMES, crystallization_regime, regime_name
from aragog.core.stratification import adiabatic_ratio, stratification_depth

__all__ = [
    'AlloyLiquidus',
    'CoreEnergyBudget',
    'CoreEntropyBudget',
    'CoreModule',
    'GaussianCoreProfiles',
    'fit_gaussian_core_profiles',
    'IronMeltingCurve',
    'LiquidIronTable',
    'QuadraticMeltingCurve',
    'TabulatedCoreProfiles',
    'build_core_module_budget',
    'load_core_arrays',
    'check_ra_crit',
    'cmb_boundary_layer_flux',
    'RA_CRIT_CMB_DEFAULT',
    'RA_CRIT_UPPER_LAYER',
    'critical_rayleigh_eq16',
    'REGIME_NAMES',
    'crystallization_regime',
    'eutectic',
    'regime_name',
    'adiabatic_ratio',
    'stratification_depth',
]
