"""Verification test B3: Steady conductive slab."""

import pytest
import numpy as np
from scipy.constants import Stefan_Boltzmann as SIGMA
from scipy.optimize import brentq

from aragog.parser import (
    Parameters, _BoundaryConditionsParameters, _EnergyParameters,
    _InitialConditionParameters, _MeshParameters, _PhaseMixedParameters,
    _PhaseParameters, _SolverParameters
)
from aragog.solver.entropy_solver import EntropySolver
import aragog.solver.entropy_solver as es_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

def test_conductive_slab_flux_converges_below_half_percent():
    _orig = es_mod.EntropySolver._solve_cvode
    def _patched(self, *a, **k):
        k["max_step"] = np.inf
        return _orig(self, *a, **k)
    es_mod.EntropySolver._solve_cvode = _patched

    TEQ = 300.0
    TC = 2700.0
    K = 4.0
    RHO = 4000.0
    CP = 1000.0
    R_OUT = 6.371e6
    R_IN = 3.48e6
    D = R_OUT - R_IN

    G_shell = K * R_IN / (R_OUT * D)
    f_balance = lambda Ts: G_shell * (TC - Ts) - SIGMA * (Ts**4 - TEQ**4)
    Ts_ref = brentq(f_balance, TEQ, TC)
    F_ref = G_shell * (TC - Ts_ref)

    common = dict(density=RHO, heat_capacity=CP, thermal_conductivity=K, thermal_expansivity=3.0e-5)
    errors = []

    for n in [50, 100, 200]:
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
                initial_condition=1, surface_temperature=Ts_ref, basal_temperature=TC
            ),
            mesh=_MeshParameters(
                outer_radius=R_OUT, inner_radius=R_IN, number_of_nodes=n,
                mixing_length_profile="nearest_boundary", core_density=RHO,
                surface_density=RHO, mass_coordinates=False,
                surface_cell_thickness=1.0 * 1e3, cmb_cell_thickness=0.0,
            ),
            phase_solid=_PhaseParameters(melt_fraction=0.0, viscosity=1e21, **common),
            phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=1e2, **common),
            phase_mixed=_PhaseMixedParameters(
                latent_heat_of_fusion=4.0e5, rheological_transition_melt_fraction=0.4,
                rheological_transition_width=0.15, solidus="solidus.dat", liquidus="liquidus.dat",
                phase="mixed", phase_transition_width=0.01, grain_size=1.0e-3,
                const_properties=True, const_rho=RHO, const_Cp=CP, const_cond=K,
                const_T_ref=1500.0, const_log10visc=2.0, const_S_ref=3000.0,
            ),
            radionuclides=[],
            solver=_SolverParameters(start_time=0.0, end_time=1e11, atol=1e-8, rtol=1e-8),
        )
        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        s._phi_rheo = 1.2
        r = s._r_stag_flat
        T_prof = TC - (TC - Ts_ref) * (1.0 / R_IN - 1.0 / r) / (1.0 / R_IN - 1.0 / R_OUT)
        S_prof = 3000.0 + CP * np.log(T_prof / 1500.0)
        s.set_initial_entropy(S_prof)
        s.solve()
        st = s.get_state()
        s.dSdt(s._solution.t[-1], np.asarray(st.S_final, float).ravel())
        F = np.asarray(s.state.heat_flux, float).ravel()
        err_pct = (F[-1] / F_ref - 1.0) * 100.0
        errors.append(abs(err_pct))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.5
