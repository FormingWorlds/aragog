"""Tests for calibration constants, boundary condition limits, and invariants.

These tests pin calibration constants and boundary condition behaviors:
1. lid_contrast_coeff default 2.2 in SolidRheologyParams / rheology_lid.
2. interior_flux_fraction default 0.05 in SolidRheologyParams / rheology_lid.
3. T_v velocity soft-max scale 3.17e-12 in rheology_lid.
4. JAX lid_mask = 1 - w_lid in kappah_floor computation.
5. NumPy p_cutoff in _step_powers and _compute_step_energy_integrals.
6. JAX table-edge cutoff in _apply_surface_bc.
7. Critical Rayleigh number RA_C = 27 pi^4 / 4 in cmb_boundary_layer.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.cmb_boundary_layer import RA_C
from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt
from aragog.jax.solver import BoundaryParams, _apply_surface_bc
from aragog.rheology import SolidRheologyParams
from aragog.rheology_lid import compute_stagnant_lid_state
from aragog.solver.entropy_solver import SECS_PER_YEAR, EntropySolver
from aragog.surface_skin import table_edge_factor

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_lid_contrast_coefficient_scaling():
    """Verify default lid_contrast_coeff is 2.2 and sets T_lid_iso."""
    params = SolidRheologyParams(enabled=True)
    assert params.lid_contrast_coeff == pytest.approx(2.2, rel=1e-12)

    n = 50
    radii = np.linspace(3.48e6, 6.371e6, n)
    pressure = np.linspace(135e9, 0.0, n)
    temperature = np.linspace(2500.0, 1500.0, n)
    total_flux = np.ones(n) * 100.0
    convective_flux = np.ones(n) * 10.0
    melt_fraction = np.zeros(n)
    v_unyielded = np.linspace(1e-9, 1e-8, n)

    out = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=convective_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_fraction,
        params=params,
        unyielded_velocity=v_unyielded,
    )

    expected_T_lid_iso = out['T_i'] - 2.2 * out['dT_rh']
    assert out['T_lid_iso'] == pytest.approx(expected_T_lid_iso, rel=1e-12)


def test_interior_flux_fraction_activation_threshold():
    """Verify interior_flux_fraction threshold gates convective activity."""
    params = SolidRheologyParams(enabled=True)
    assert params.interior_flux_fraction == pytest.approx(0.05, rel=1e-12)

    n = 50
    radii = np.linspace(3.48e6, 6.371e6, n)
    pressure = np.linspace(135e9, 0.0, n)
    temperature = np.linspace(2500.0, 1500.0, n)
    total_flux = np.ones(n) * 100.0
    convective_flux = np.ones(n) * 7.5
    melt_fraction = np.zeros(n)
    v_unyielded = np.linspace(1e-9, 1e-8, n)

    out = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=convective_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_fraction,
        params=params,
        unyielded_velocity=v_unyielded,
    )

    assert out['w_active'] > 0.9


def test_tv_velocity_scale_calibration():
    """Verify soft-max scale T_v sets convective interior velocity v_i."""
    params = SolidRheologyParams(enabled=True)
    n = 50
    radii = np.linspace(3.48e6, 6.371e6, n)
    pressure = np.linspace(135e9, 0.0, n)
    temperature = np.linspace(2500.0, 1500.0, n)
    total_flux = np.ones(n) * 100.0
    convective_flux = np.ones(n) * 7.5
    melt_fraction = np.zeros(n)
    v_unyielded = np.linspace(1e-9, 1e-8, n)

    out = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=convective_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_fraction,
        params=params,
        unyielded_velocity=v_unyielded,
    )

    expected_vi = 9.9876811e-09
    assert out['v_i'] == pytest.approx(expected_vi, rel=1e-6)


def test_jax_lid_mask_in_kappah_floor():
    """Verify JAX kappa_h floor is suppressed inside the stagnant lid."""
    n_basic = 10
    r_basic = jnp.linspace(3.5e6, 6.371e6, n_basic)
    r_stag = 0.5 * (r_basic[1:] + r_basic[:-1])
    P_basic = jnp.linspace(120e9, 1e9, n_basic)
    P_stag = 0.5 * (P_basic[1:] + P_basic[:-1])
    mesh = MeshArrays(
        d_dr_matrix=jnp.zeros((n_basic, n_basic - 1)),
        quantity_matrix=jnp.zeros((n_basic, n_basic - 1)),
        area=jnp.ones(n_basic),
        volume=jnp.ones(n_basic),
        radii_basic=r_basic,
        radii_stag=r_stag,
        mixing_length=jnp.ones(n_basic) * 1.0e2,
        mixing_length_sq=jnp.ones(n_basic) * 1.0e4,
        mixing_length_cu=jnp.ones(n_basic) * 1.0e6,
        P_stag=P_stag,
        P_basic=P_basic,
        dP_dr_basic=jnp.gradient(P_basic, r_basic),
        gravity=jnp.full(n_basic, 9.81),
    )

    temperature = jnp.linspace(2500.0, 1000.0, n_basic)
    ones = jnp.ones(n_basic)
    phase = PhaseProperties(
        temperature=temperature,
        density=ones * 3500.0,
        heat_capacity=ones * 1200.0,
        thermal_expansivity=ones * 3.0e-5,
        dTdPs=ones * 1.0e-8,
        melt_fraction=jnp.zeros(n_basic),
        viscosity=ones * 1.0e21,
        kinematic_viscosity=ones * 1.0e21 / 3500.0,
        thermal_conductivity=ones * 4.0,
        latent_heat=ones * 4.0e5,
        capacitance=ones * 3500.0 * 1200.0,
        eta_diff=ones * 1.0e21,
        tau_y=ones * 1.0e8,
        visc_solid_weight=ones,
    )

    params = PhaseParams(
        enabled=True,
        stress_closure_mode='lid',
        kappah_floor=1000.0,
        phi_rheo=0.4,
        phi_width=0.15,
        eddy_diff_thermal=1.0,
    )

    dSdr = jnp.full(n_basic, -1.0e-4)
    kh, _ = compute_mlt(dSdr, phase, mesh, params)

    assert float(kh[-1]) < 1.0


def test_numpy_p_cutoff_energy_integral():
    """Verify NumPy p_cutoff and energy integrals capture surface cutoff."""
    solver = EntropySolver.__new__(EntropySolver)
    solver._table_edge_cutoff = True
    solver._core_bc = 'quasi_steady'
    solver._surface_flux_nominal = 1000.0
    solver._volume_flat = np.ones(5) * 1e18
    solver._r_basic_flat = np.linspace(3.5e6, 6.371e6, 6)
    solver._P_stag_flat = np.linspace(120e9, 1e9, 5)

    A_int = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2

    class MockMesh:
        staggered_effective_density = np.ones(5) * 3500.0

    class MockEvaluator:
        mesh = MockMesh()

    solver.evaluator = MockEvaluator()

    class MockPhase:
        def density(self):
            return np.ones(5) * 3500.0

        def temperature(self):
            return np.ones(5) * 2000.0

    class MockState:
        _heat_flux = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 600.0])
        phase_staggered = MockPhase()
        heating_radio = np.zeros(5)
        heating_tidal = np.zeros(5)
        _pb_cache_hits = 0
        _pb_cache_misses = 0

        def capacitance_staggered(self):
            return np.ones(5) * 3500.0 * 1200.0

    solver.state = MockState()

    class MockEOS:
        def density(self, P, S):
            return np.ones(5) * 3500.0

    solver.entropy_eos = MockEOS()
    solver._dSdt_single = lambda t, y: np.zeros(5)
    solver._stag_entropy = lambda y: y

    y_col = np.ones(5) * 1000.0
    powers = solver._step_powers(0.0, y_col)

    expected_p_cutoff = (1000.0 - 600.0) * A_int
    assert powers[7] == pytest.approx(expected_p_cutoff, rel=1e-12)
    assert powers[7] > 0.0

    class MockSolution:
        t = np.array([0.0, 1.0])
        y = np.ones((5, 2)) * 1000.0

        def get(self, key):
            return None

    solver._solution = MockSolution()
    solver._step_heat_content = lambda S0, Sf: 0.0
    integrals = solver._compute_step_energy_integrals()

    expected_integral = expected_p_cutoff * 1.0 * SECS_PER_YEAR
    assert integrals['surface_cutoff'] == pytest.approx(expected_integral, rel=1e-12)


def test_jax_table_edge_cutoff():
    """Verify JAX table-edge cutoff reduces surface flux at table edge."""
    bc = BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=1000.0,
        emissivity=1.0,
        T_eq=250.0,
        inner_bc_type=1,
        inner_bc_value=4000.0,
        core_density=7000.0,
        core_heat_capacity=800.0,
        tfac_core_avg=1.0,
        table_edge_cutoff=True,
        S_table_edge=500.0,
    )
    heat_flux = jnp.zeros(10)
    S_top = jnp.array(600.0)
    phase_basic_T = jnp.full(10, 1500.0)

    out_flux = _apply_surface_bc(
        heat_flux=heat_flux,
        bc=bc,
        phase_basic_T=phase_basic_T,
        S_top=S_top,
    )

    expected_factor = float(table_edge_factor(S_top, bc.S_table_edge, xp=jnp))
    assert expected_factor == pytest.approx(0.5, rel=1e-12)
    assert float(out_flux[-1]) == pytest.approx(500.0, rel=1e-12)


def test_ra_c_literal_value():
    """Verify critical Rayleigh number RA_C matches Chandrasekhar onset value."""
    expected_ra_c = 27.0 * (np.pi**4) / 4.0
    assert expected_ra_c == pytest.approx(657.5113644795163, rel=1e-12)
    assert RA_C == pytest.approx(657.5113644795163, rel=1e-12)
