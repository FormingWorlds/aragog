"""Solver package for the Aragog interior dynamics model.

Provides the entropy-formulation solver (EntropySolver) and supporting
classes (BoundaryConditions, EntropyState).
"""

from __future__ import annotations

from scipy import constants as sp_constants

# Time unit conversion: the ODE is integrated in years, but fluxes are in SI (W/m^2).
# dS/dt from flux divergence gives J/kg/K/s; multiply by SECS_PER_YEAR to get per-year.
SECS_PER_YEAR: float = sp_constants.Julian_year  # 31557600.0 s

from aragog.solver.boundary import BoundaryConditions
from aragog.solver.entropy_solver import EntropySolver, SolverOutput
from aragog.solver.entropy_state import EntropyState

__all__ = [
    "BoundaryConditions",
    "EntropySolver",
    "EntropyState",
    "SolverOutput",
    "SECS_PER_YEAR",
]
