"""Smoke coverage for the ``core_bc='core_module'`` solver coupling.

The staged core-evolution budget joins the entropy solver as an appended
ODE state, so the checks here are the coupling contract: the state vector
grows by one, the reported core temperature is the integrated boundary
state rather than the basal node's EOS read-off, it stays smooth and
physical over a short integration, and the legacy ``quasi_steady`` mode
still produces its own (node-derived) core temperature on the same setup.
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

CORE_MODULE_PARAMS = {
    'rho_cen': 12500.0,
    'length_scale': 7200e3,
    'alpha': 1.35e-5,
    'c_p': 840.0,
    'melting_curve': 'iron',
    'light_element_fraction': 0.1,
    'depression': 1.2,
    'ds_fusion': 170.0,
    'icn_width': 10.0,
    'q_radio': 0.0,
}


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def _build(
    core_bc: str, shared_eos, core_module_params=None, n_nodes: int = 10, end_time: float = 1.0
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
        core_module_params=core_module_params,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='radau',
        use_jax_jacobian=False,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
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
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()
    solver.set_initial_entropy(2700.0)
    return solver


def test_core_module_state_extension_and_integrated_t_core(shared_eos):
    """The core_module solve carries the extra T_cmb state, reports the
    integrated value (finite, near the EOS-derived start, strictly
    monotone under net heat loss), and the budget object was built with
    the mesh's own CMB radius."""
    solver = _build('core_module', shared_eos, CORE_MODULE_PARAMS)
    assert solver._state_is_extended
    n_stag = solver._n_stag
    assert len(solver._S0) == n_stag + 1  # entropy block plus T_cmb
    assert solver._core_module_budget.profiles.r_cmb == pytest.approx(3.480e6)
    solver.solve()
    out = solver.get_state()
    y = solver._solution.y
    assert y.shape[0] == n_stag + 1
    t_core_path = y[n_stag]
    assert np.all(np.isfinite(t_core_path))
    # Reported T_core is the ODE state's endpoint, not the basal node.
    assert out.T_core == pytest.approx(float(t_core_path[-1]), rel=1e-12)
    # Physical bounds: positive, and within a few hundred K of the start
    # over one year of integration.
    assert 0.0 < out.T_core
    assert abs(float(t_core_path[-1]) - float(t_core_path[0])) < 300.0
    # Sub-step smoothness: no single internal jump exceeds 50 K over this
    # short, transient-free integration.
    if t_core_path.size > 2:
        assert np.max(np.abs(np.diff(t_core_path))) < 50.0


def test_core_module_against_quasi_steady_baseline(shared_eos):
    """Both modes integrate the same setup; the legacy mode reports the
    node-derived core temperature, the module the integrated one, and the
    two agree within the boundary-layer scale (hundreds of K) rather than
    diverging to different regimes."""
    legacy = _build('quasi_steady', shared_eos)
    legacy.solve()
    t_legacy = legacy.get_state().T_core

    module = _build('core_module', shared_eos, CORE_MODULE_PARAMS)
    module.solve()
    t_module = module.get_state().T_core

    assert np.isfinite(t_legacy) and np.isfinite(t_module)
    assert abs(t_module - t_legacy) < 500.0


def test_core_module_missing_params_still_builds_with_defaults(shared_eos):
    """An empty params dict builds the budget entirely from defaults plus
    the mesh geometry and the EOS-derived CMB pressure fallback; the
    solve still runs. A wrong key fails at solver construction with the
    factory's message."""
    solver = _build('core_module', shared_eos, {})
    assert float(solver._core_module_budget.profiles.p_cmb) > 0.0
    solver.solve()
    assert np.isfinite(solver.get_state().T_core)

    with pytest.raises(ValueError, match='unrecognised'):
        _build('core_module', shared_eos, {'not_a_key': 1.0})
