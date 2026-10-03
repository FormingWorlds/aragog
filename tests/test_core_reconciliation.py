"""Tests for core-evolution reconciliation with aragog main.

Verifies boundary flux reporting, core temperature slot extraction, retry snapshots,
and radiogenic heating accounting.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    None,
)


@pytest.fixture(scope='module')
def shared_eos():
    if EOS_DIR is None or not EOS_DIR.exists():
        pytest.skip(
            f'SPIDER EOS tables not found. ARAGOG_TEST_EOS_DIR={os.environ.get("ARAGOG_TEST_EOS_DIR")}'
        )
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


CORE_MODULE_PARAMS = {
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
    'stratification': False,
}


def _build_solver(
    core_bc: str = 'core_module',
    shared_eos=None,
    core_module_params=None,
    n_nodes: int = 10,
    end_time: float = 1.0,
    solver_method: str = 'radau',
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
    from aragog.solver import EntropySolver

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=1,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc=core_bc,
        core_module_params=core_module_params or CORE_MODULE_PARAMS,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method=solver_method,
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
    sv = _SolverParameters(
        start_time=0.0,
        end_time=end_time,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
    params = Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        solver=sv,
    )
    return EntropySolver(params, entropy_eos=shared_eos)


def test_reported_cmb_flux_matches_rhs_applied(shared_eos):
    """get_state() reports the CMB flux that the RHS integrated."""
    solver = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver.initialize()
    s_init = np.linspace(2950.0, 2600.0, solver._n_stag)
    solver.set_initial_entropy(s_init)
    solver.set_initial_core_temperature(5800.0)
    solver.solve()
    out = solver.get_state()

    # Re-evaluate RHS at final trajectory column
    y_final = solver._solution.y[:, -1]
    t_final = solver._solution.t[-1]
    solver._dSdt_single(t_final, y_final)
    flux_rhs = solver.state.heat_flux[0]
    assert out.heat_flux[0] == pytest.approx(flux_rhs, rel=1e-12)


def test_core_temperature_from_column_reads_slot(shared_eos):
    """_core_temperature_from_column reads T_core slot for core_module.

    Guards against fallback to basal EOS temperature when T_core differs.
    """
    solver = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver.initialize()
    n_stag = solver._n_stag
    y_col = np.zeros(n_stag + 2)
    y_col[:n_stag] = 2500.0  # basal node entropy corresponding to ~4739 K
    y_col[n_stag] = 0.0  # dSdr_cmb
    y_col[n_stag + 1] = 6543.21  # distinct T_core ODE slot value

    reported_t = solver._core_temperature_from_column(y_col)
    assert reported_t == pytest.approx(6543.21, rel=1e-6)


def test_retry_snapshot_restore_t_core(shared_eos):
    """get_current_core_temperature snapshots T_core and set_initial restores it.

    Ensures that a failed solve attempt does not leave T_core at the failed state
    and can be restored on retry.
    """

    class _Sol:
        pass

    solver = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver.initialize()
    n_stag = solver._n_stag
    sol = _Sol()
    sol.y = np.zeros((n_stag + 2, 3))
    sol.y[n_stag, -1] = -1.5e-4
    sol.y[n_stag + 1, -1] = 5925.5
    solver._solution = sol

    # Snapshot current core temperature
    assert hasattr(solver, 'get_current_core_temperature')
    t_core_snap = solver.get_current_core_temperature()
    assert t_core_snap == pytest.approx(5925.5)

    # When core_bc is energy_balance, returns None
    eb_solver = _build_solver(core_bc='energy_balance', shared_eos=shared_eos)
    eb_solver.initialize()
    sol_eb = _Sol()
    sol_eb.y = np.zeros((eb_solver._n_stag + 1, 3))
    sol_eb.y[eb_solver._n_stag, -1] = -1.5e-4
    eb_solver._solution = sol_eb
    assert eb_solver.get_current_core_temperature() is None

    # Simulate retry: restore from snapshot
    solver.set_initial_core_temperature(t_core_snap)
    solver.set_initial_entropy(np.full(n_stag, 2900.0))
    assert solver._S0[n_stag + 1] == pytest.approx(5925.5)


def test_core_module_heating_counted_once(shared_eos):
    """core_module counts bottom-cell radiogenic heating in the mantle once.

    Verifies that bottom cell heating enters the mantle cell's dSdt once and is
    not added to or subtracted from the CMB heat flux or core cooling rate.
    """
    from aragog.parser import _Radionuclide

    solver = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver.parameters.energy.radionuclides = False
    solver.initialize()
    n_stag = solver._n_stag
    y_col = np.zeros(n_stag + 2)
    y_col[:n_stag] = 2900.0
    y_col[n_stag] = -1.0e-5  # dSdr_cmb
    y_col[n_stag + 1] = 5800.0  # T_core

    dy_no_heat = solver._dSdt_single(0.0, y_col)
    flux_no_heat = float(solver.state.heat_flux[0])
    dT_core_no_heat = float(dy_no_heat[n_stag + 1])

    # Now create solver with uniform radiogenic heating H_0 = 1e-10 W/kg
    H_0 = 1.0e-10
    solver_heat = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver_heat.parameters.energy.radionuclides = True
    solver_heat.parameters.radionuclides = [_Radionuclide('X', 0.0, 1.0, 1.0, H_0, 1e20)]
    solver_heat.initialize()

    dy_heat = solver_heat._dSdt_single(0.0, y_col)
    flux_heat = float(solver_heat.state.heat_flux[0])
    dT_core_heat = float(dy_heat[n_stag + 1])

    # 1. CMB heat flux is purely state-derived (from dSdr_cmb), unchanged by mantle heating
    assert flux_heat == pytest.approx(flux_no_heat, rel=1e-12)

    # 2. Core cooling rate dT_core/dt is unchanged by mantle heating
    assert dT_core_heat == pytest.approx(dT_core_no_heat, rel=1e-12)

    # 3. Mantle bottom cell dS/dt increases by exactly H_0 / T_0 per second
    T_0 = float(np.asarray(solver_heat.state.phase_staggered.temperature()).flat[0])
    sec_per_yr = 3.15576e7
    dSdt_diff = (dy_heat[0] - dy_no_heat[0]) / sec_per_yr
    assert dSdt_diff == pytest.approx(H_0 / T_0, rel=1e-6)

    # Canary: if bottom cell heating Q_0 were incorrectly subtracted from or added
    # to the CMB flux (as quasi_steady does with alpha*(F_1 - Q_0/A_1)), the CMB flux
    # would change by ~Q_0/A_cmb and dT_core/dt would change by ~Q_0/C_eff.
    vol_0 = float(solver_heat._volume_flat[0])
    rho_0 = float(np.asarray(solver_heat.state.phase_staggered.density()).flat[0])
    Q_0 = H_0 * rho_0 * vol_0
    A_cmb = float(solver_heat._cmb_area)
    double_count_flux_shift = Q_0 / A_cmb
    assert double_count_flux_shift > 0.01  # Significant shift (> 10 mW/m^2)
    # Confirm that actual flux differs from double-counted flux
    assert abs(flux_heat - (flux_no_heat - double_count_flux_shift)) > 0.005


def test_core_module_hot_start_reads_t_core_slot(shared_eos):
    """Automatic hot-start from prev_sol reads T_core slot (n_stag + 1), not dSdr_cmb."""
    from types import SimpleNamespace

    solver = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver.initialize()
    n_stag = solver._n_stag
    sol = SimpleNamespace()
    sol.y = np.zeros((n_stag + 2, 3))
    sol.y[n_stag, -1] = -1.5e-4
    sol.y[n_stag + 1, -1] = 5925.5
    solver._solution = sol
    solver._T_core_init = None
    solver.set_initial_entropy(np.full(n_stag, 2900.0))
    assert solver._S0[n_stag + 1] == pytest.approx(5925.5)


def test_core_module_rhs_evaluates_at_t_core_state_not_t_cmb_basic(shared_eos):
    """_dSdt_single passes state t_core to core RHS, not T_cmb_basic."""
    solver = _build_solver(core_bc='core_module', shared_eos=shared_eos)
    solver.initialize()
    n_stag = solver._n_stag
    y_col = np.zeros(n_stag + 2)
    y_col[:n_stag] = 2600.0
    y_col[n_stag] = -1.0e-5
    y_col[n_stag + 1] = 4000.0  # t_core in partially frozen regime
    dy = solver._dSdt_single(0.0, y_col)
    dT_dt_actual = float(dy[n_stag + 1])

    T_cmb_basic = float(np.asarray(solver.state.phase_basic.temperature()).flat[0])
    assert abs(T_cmb_basic - 4000.0) > 500.0

    cap_at_t_core = float(solver._core_module_budget.effective_capacity(4000.0))
    cap_at_t_cmb = float(solver._core_module_budget.effective_capacity(T_cmb_basic))
    assert abs(cap_at_t_core - cap_at_t_cmb) > 1e27

    sec_per_yr = 3.15576e7
    F_cmb = float(solver.state.heat_flux[0])
    area = float(solver._core_module_budget.profiles.r_cmb**2 * 4.0 * np.pi)
    dT_dt_expected = -F_cmb * area / cap_at_t_core * sec_per_yr
    dT_dt_mutant = -F_cmb * area / cap_at_t_cmb * sec_per_yr
    assert dT_dt_actual == pytest.approx(dT_dt_expected, rel=1e-3)
    assert abs(dT_dt_actual - dT_dt_mutant) > 1e-3 * abs(dT_dt_actual)
