"""Verification test B1: Half-space conduction with mesh refinement."""

import numpy as np
import pytest

from aragog.parser import (
    Parameters, _BoundaryConditionsParameters, _EnergyParameters,
    _InitialConditionParameters, _MeshParameters, _PhaseMixedParameters,
    _PhaseParameters, _SolverParameters
)
from aragog.solver.entropy_solver import EntropySolver

pytestmark = [pytest.mark.slow, pytest.mark.timeout(300)]

def test_halfspace_refinement_reduces_boundary_flux_errors():
    """Spherical half-space test at 1e7 yr with top BC 6 and bottom BC 3."""
    YR, T0, TEQ, TC, K, RHO, CP = 3.15576e7, 1500.0, 300.0, 2700.0, 4.0, 4000.0, 1000.0
    R_OUT, R_IN = 6.371e6, 3.48e6
    t_end_yr = 1.0e7
    kappa = K / (RHO * CP)
    L = np.sqrt(np.pi * kappa * t_end_yr * YR)

    def solve_cell(c_km):
        common = dict(density=RHO, heat_capacity=CP, thermal_conductivity=K, thermal_expansivity=3.0e-5)
        p = Parameters(
            boundary_conditions=_BoundaryConditionsParameters(
                outer_boundary_condition=6, outer_boundary_value=TEQ,
                inner_boundary_condition=3, inner_boundary_value=TC,
                emissivity=1.0, equilibrium_temperature=TEQ,
                core_heat_capacity=880.0, core_bc="quasi_steady",
            ),
            energy=_EnergyParameters(
                conduction=True, convection=False, gravitational_separation=False,
                mixing=False, radionuclides=False, tidal=False,
                solver_method="cvode", use_jax_jacobian=False,
            ),
            initial_condition=_InitialConditionParameters(
                initial_condition=1, surface_temperature=T0, basal_temperature=T0
            ),
            mesh=_MeshParameters(
                outer_radius=R_OUT, inner_radius=R_IN, number_of_nodes=100,
                mixing_length_profile="nearest_boundary", core_density=RHO,
                surface_density=RHO, mass_coordinates=False,
                surface_cell_thickness=c_km * 1e3, cmb_cell_thickness=c_km * 1e3,
            ),
            phase_solid=_PhaseParameters(melt_fraction=0.0, viscosity=1e21, **common),
            phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=1e2, **common),
            phase_mixed=_PhaseMixedParameters(
                latent_heat_of_fusion=4.0e5, rheological_transition_melt_fraction=0.4,
                rheological_transition_width=0.15, solidus="solidus.dat", liquidus="liquidus.dat",
                phase="mixed", phase_transition_width=0.01, grain_size=1.0e-3,
                const_properties=True, const_rho=RHO, const_Cp=CP, const_cond=K,
                const_T_ref=T0, const_log10visc=2.0, const_S_ref=3000.0,
            ),
            radionuclides=[],
            solver=_SolverParameters(start_time=0.0, end_time=t_end_yr, atol=1e-8, rtol=1e-8),
        )
        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        s._phi_rheo = 1.2
        s.set_initial_entropy(3000.0)
        s.solve()
        st = s.get_state()
        s.dSdt(s._solution.t[-1], np.asarray(st.S_final, float).ravel())
        F = np.asarray(s.state.heat_flux, float).ravel().copy()
        Ts = float(np.asarray(getattr(st, "T_surface_skin", TEQ)))
        ref_top = K * (T0 - Ts) * (1.0 / L - 1.0 / R_OUT)
        ref_bot = K * (TC - T0) * (1.0 / L + 1.0 / R_IN)
        return F[-1] / ref_top - 1.0, F[0] / ref_bot - 1.0

    err_unif_top, err_unif_bot = solve_cell(0.0)
    err_fine_top, err_fine_bot = solve_cell(1.0)

    assert err_unif_top > 0.15
    assert err_unif_bot > 0.15
    assert abs(err_fine_top) < 0.003
    assert abs(err_fine_bot) < 0.003
