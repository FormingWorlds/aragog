"""Conductive half-space at the surface (outer BC 6) and at the CMB (inner BC 3) on a
mesh refined towards both boundaries.

A uniform shell at ``T0`` has its surface held near ``T_eq`` by the skin balance
(radiative conductance about 6 W/m^2/K against about 0.01 for the half cell) and its
CMB at ``T_c``. For a sphere the exact surface fluxes at time ``t`` are
``k dT (1 / sqrt(pi kappa t) - 1 / R)`` at the top and
``k dT (1 / sqrt(pi kappa t) + 1 / r_c)`` at the CMB (``v = r (T - T_boundary)``
turns the radial problem into the 1-D one). Constant properties, conduction only.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _SolverParameters,
)
from aragog.solver.entropy_solver import EntropySolver

pytestmark = [pytest.mark.slow, pytest.mark.reference_pinned, pytest.mark.timeout(900)]

YR = 3.15576e7
T0, T_EQ, T_C = 1500.0, 300.0, 2700.0
K, RHO, CP = 4.0, 4000.0, 1000.0
R_OUT, R_IN = 6.371e6, 3.48e6
T_END_YR = 1.0e7  # diffusion length sqrt(pi kappa t) = 31.5 km


def _fluxes(cell_km):
    common = dict(density=RHO, heat_capacity=CP, thermal_conductivity=K, thermal_expansivity=3e-5)
    p = Parameters(
        boundary_conditions=_BoundaryConditionsParameters(
            outer_boundary_condition=6,
            outer_boundary_value=0.0,
            inner_boundary_condition=3,
            inner_boundary_value=T_C,
            emissivity=1.0,
            equilibrium_temperature=T_EQ,
            core_heat_capacity=880.0,
            core_bc='quasi_steady',
        ),
        energy=_EnergyParameters(
            conduction=True,
            convection=False,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
            solver_method='cvode',
            use_jax_jacobian=False,
        ),
        initial_condition=_InitialConditionParameters(
            initial_condition=1, surface_temperature=T0, basal_temperature=T0
        ),
        mesh=_MeshParameters(
            outer_radius=R_OUT,
            inner_radius=R_IN,
            number_of_nodes=100,
            mixing_length_profile='nearest_boundary',
            core_density=RHO,
            surface_density=RHO,
            mass_coordinates=False,
            surface_cell_thickness=cell_km * 1e3,
            cmb_cell_thickness=cell_km * 1e3,
        ),
        phase_solid=_PhaseParameters(melt_fraction=0.0, viscosity=1e21, **common),
        phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=1e2, **common),
        phase_mixed=_PhaseMixedParameters(
            latent_heat_of_fusion=4.0e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1.0e-3,
            const_properties=True,
            const_rho=RHO,
            const_Cp=CP,
            const_cond=K,
            const_T_ref=T0,
            const_S_ref=3000.0,
        ),
        radionuclides=[],
        solver=_SolverParameters(start_time=0.0, end_time=T_END_YR, atol=1e-8, rtol=1e-8),
    )
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    s._phi_rheo = 1.2  # const mode reports melt fraction 1: take the solid-skin branch
    s.set_initial_entropy(3000.0)
    s.solve()
    st = s.get_state()
    # get_state's heat_flux holds the boundary nodes before the BCs act; take the applied flux.
    s.dSdt(s._solution.t[-1], np.asarray(st.S_final, float).ravel())
    F = np.asarray(s.state.heat_flux, float).ravel()
    L = np.sqrt(np.pi * K / (RHO * CP) * T_END_YR * YR)
    T_s = float(st.T_surface_skin)
    ref_top = K * (T0 - T_s) * (1.0 / L - 1.0 / R_OUT)
    ref_bot = K * (T_C - T0) * (1.0 / L + 1.0 / R_IN)
    return F[-1] / ref_top - 1.0, F[0] / ref_bot - 1.0


@pytest.mark.physics_invariant
def test_boundary_fluxes_converge_to_the_half_space_flux_with_refinement():
    """N 100 (uniform cell 29 km): the uniform mesh overestimates both fluxes by
    about 17 %; end cells of 8, 4 and 1 km bring the error down monotonically to
    below 0.3 % at both boundaries (N 200 takes it below 0.03 %)."""
    err = {c: _fluxes(c) for c in (0.0, 8.0, 4.0, 1.0)}
    assert err[0.0][0] > 0.1 and err[0.0][1] > 0.1
    for end in (0, 1):
        assert abs(err[8.0][end]) > abs(err[4.0][end])
        assert abs(err[4.0][end]) < 0.005
        assert abs(err[1.0][end]) < 0.003
