"""Tests for CMB convective flux sign gate and core temperature offset under Ruling 35 and 36.

Ruling 35/36 requires:
1. Core closure converges at 65/257/1025 for case 5b (fails before fix).
2. A convecting case bitwise unchanged in every flux column.
3. A stable-CMB case where the CMB flux equals the conductive value (fails before fix).
4. T_core offset test across rtol 1e-6..1e-12 yielding -3.818 mK (fails before fix).
5. T_core offset decode with a large offset (0.6 * T_core_0) gives T_core_0 + offset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)

pytestmark = [pytest.mark.smoke, needs_eos]


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def _build_case5b_params(
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    cvode_output_points: int = 257,
    dt_yr: float = 1e4,
    n_nodes: int = 12,
):
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

    core_params = {
        'rho_cen': 12500.0,
        'length_scale': 7272000.0,
        'alpha': 1.35e-05,
        'c_p': 840.0,
        'melting_curve': 'iron',
        'light_element_fraction': 0.1,
        'depression': 1.2,
        't_m0': 2677.0,
        't_m1': 2.95e-12,
        't_m2': 8.37e-25,
        'ds_fusion': 172.8,
        'icn_width': 10.0,
        'alpha_c': 1.0,
        'c_light': 0.046,
        'q_radio': 0.0,
        'k_core': 130.0,
        'stratification': True,
    }
    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=5,
        outer_boundary_value=1600.0,
        inner_boundary_condition=1,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=840.0,
        core_bc='core_module',
        core_module_params=core_params,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='cvode',
        use_jax_jacobian=False,
    )
    ic = _InitialConditionParameters(
        initial_condition=1,
        surface_temperature=3500.0,
        basal_temperature=5500.0,
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
        mass_coordinates=False,
        mixing_length_profile='nearest_boundary',
        core_density=12500.0,
        eos_method=1,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sol = _SolverParameters(
        start_time=0.0,
        end_time=dt_yr,
        cvode_output_points=cvode_output_points,
        atol=atol,
        rtol=rtol,
    )
    return Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        solver=sol,
        radionuclides=[],
    )


def test_stable_cmb_boundary_convective_flux_vanishes(shared_eos):
    """In a stable CMB boundary layer (dS/dr >= 0), convective heat flux at node 0 must be 0."""
    from aragog.solver import EntropySolver

    params = _build_case5b_params(rtol=1e-8, atol=1e-8, cvode_output_points=65)
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()

    # Set initial entropy profile with convecting interior and stable CMB layer
    S_profile = np.linspace(3100.0, 3000.0, solver._n_stag)
    solver.set_initial_core_temperature(5800.0)
    # Impose a positive (stable) entropy gradient at CMB basic node 0
    solver.set_initial_dSdr_cmb(1.0e-6)
    solver.set_initial_entropy(S_profile)

    # Evaluate state
    solver._dSdt_single(0.0, solver._S0)

    # Eddy diffusivity is borrowed from node 1 (which convects)
    assert solver.state.eddy_diffusivity[0] > 0.0

    # Convective flux must be gated to 0 where dS/dr >= 0
    assert solver.state.jconv[0] == 0.0, (
        f'jconv[0] must vanish when dSdr[0] >= 0; got {solver.state.jconv[0]:.6e}'
    )
    # Total heat flux at CMB must equal conductive heat flux alone
    assert solver.state.heat_flux[0] == pytest.approx(solver.state.jcond[0], abs=1e-15)


def test_convective_case_fluxes_bitwise_unchanged(shared_eos):
    """When the entire mantle convects (dS/dr < 0 everywhere), convective flux is unchanged."""
    from aragog.solver import EntropySolver

    params = _build_case5b_params(rtol=1e-8, atol=1e-8, cvode_output_points=65)
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()

    # Superadiabatic (convective) profile: entropy decreases with radius
    S_profile = np.linspace(3500.0, 3000.0, solver._n_stag)
    solver.set_initial_core_temperature(5800.0)
    solver.set_initial_dSdr_cmb(-1.0e-6)
    solver.set_initial_entropy(S_profile)

    solver._dSdt_single(0.0, solver._S0)

    # Verify that all nodes are convective (dSdr < 0)
    assert np.all(solver.state.dSdr < 0.0)

    # Convective flux should be positive (upward) and strictly match ungated formula
    expected_jconv = (
        np.asarray(solver.state.phase_basic.density()).ravel()
        * np.asarray(solver.state.phase_basic.temperature()).ravel()
        * solver.state.eddy_diffusivity
        * (-solver.state.dSdr)
    )
    assert np.array_equal(solver.state.jconv, expected_jconv)


def test_core_module_tcore_offset_convergence_across_tolerances(shared_eos):
    """Case 5b call 1 core temperature change must converge to -3.818 mK across rtol 1e-6..1e-12."""
    from aragog.solver import EntropySolver

    expected_delta_T = -3.818380e-3  # K

    for rtol in [1e-6, 1e-8, 1e-10, 1e-12]:
        params = _build_case5b_params(rtol=rtol, atol=rtol, cvode_output_points=257)
        solver = EntropySolver(params, entropy_eos=shared_eos)
        solver.initialize()
        solver.set_initial_core_temperature(5800.0)
        solver.set_initial_dSdr_cmb(0.0)
        solver.set_initial_entropy(np.full(solver._n_stag, 3000.0))

        solver.solve()
        sol = solver._solution
        n_s = solver._n_stag
        delta_T_measured = float(sol.y[n_s + 1, -1] - sol.y[n_s + 1, 0])

        # Must not show unphysical heating (> 0)
        assert delta_T_measured < 0.0, (
            f'rtol={rtol:.1e} gave unphysical core heating: delta_T={delta_T_measured:.6e} K'
        )
        # Must match physical cooling within 10 uK across all tolerances
        assert delta_T_measured == pytest.approx(expected_delta_T, abs=1e-5), (
            f'rtol={rtol:.1e} failed convergence: delta_T={delta_T_measured:.6e} K'
        )


def test_case5b_stratified_core_closure_converges(shared_eos):
    """Core energy closure for Case 5b must converge to <= 1e-6 with refinement (65, 257, 1025)."""
    from aragog.solver import EntropySolver

    r_cores = []
    for pts in [65, 257, 1025]:
        params = _build_case5b_params(rtol=1e-8, atol=1e-8, cvode_output_points=pts)
        solver = EntropySolver(params, entropy_eos=shared_eos)
        solver.initialize()
        solver.set_initial_core_temperature(5800.0)
        solver.set_initial_dSdr_cmb(0.0)
        solver.set_initial_entropy(np.full(solver._n_stag, 3000.0))

        solver.solve()
        out = solver.get_state()
        dE_core = float(out.step_dE_core_J)
        dE_cmb = float(out.step_dE_F_cmb_J)
        r_c = abs(dE_core + dE_cmb) / max(abs(dE_cmb), abs(dE_core))
        r_cores.append(r_c)

    # Acceptance threshold <= 1e-6 at 1025 points
    assert r_cores[2] <= 1.0e-6, f'r_core at 1025 points failed: {r_cores[2]:.6e} > 1e-6'
    # Either factor of 8 drop or machine precision (< 1e-8)
    assert r_cores[2] <= 1.0e-8 or (
        r_cores[0] / r_cores[1] >= 8.0 and r_cores[1] / r_cores[2] >= 8.0
    ), f'r_core failed convergence: {r_cores}'


def test_core_module_tcore_offset_large_offset_decode(shared_eos):
    """Decoding a large offset (0.6 * T_core_0) must return T_core_0 + offset when active."""
    from aragog.solver import EntropySolver

    params = _build_case5b_params(rtol=1e-8, atol=1e-8, cvode_output_points=65)
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()

    t_core_0 = 5800.0
    solver.set_initial_core_temperature(t_core_0)
    solver.set_initial_dSdr_cmb(0.0)
    solver.set_initial_entropy(np.full(solver._n_stag, 3000.0))

    # Activate offset representation as during solve
    solver._T_core_0 = t_core_0
    solver._t_core_offset_active = True

    offset = 0.6 * t_core_0
    decoded = solver._decode_core_temperature(offset)
    expected = t_core_0 + offset
    assert decoded == pytest.approx(expected), (
        f'Large offset decode failed: expected {expected}, got {decoded}'
    )

    # Inactive offset representation returns value directly
    solver._t_core_offset_active = False
    assert solver._decode_core_temperature(5700.0) == pytest.approx(5700.0)
