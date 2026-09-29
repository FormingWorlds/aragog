"""Tests for core-evolution reconciliation with aragog main.

Exercises reconciliation items R3, R4, R5, R6, R7 from stream plan section 4.1.
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


def test_r3_reported_cmb_flux_matches_rhs_applied(shared_eos):
    """R3: get_state() reports the CMB flux that the RHS integrated."""
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


def test_r4_core_temperature_from_column_reads_slot(shared_eos):
    """R4: _core_temperature_from_column reads T_core slot for core_module.

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
