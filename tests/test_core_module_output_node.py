"""Unit checks of the core_module CMB node: the default core start and the output node.

Both sit at the bottom cell carried to the CMB pressure, the mantle side of the contrast the
boundary-layer flux acts on, so a default start carries no CMB flux and the gradient slot
enters no reported node-0 value. No solve runs here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tests.conftest import entropy_eos_copy, needs_eos

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build  # noqa: E402

pytestmark = [pytest.mark.unit, needs_eos]


@pytest.fixture(scope='module')
def solver():
    return _build('core_module', entropy_eos_copy(), CORE_MODULE_PARAMS, s_init='driven')


def _t_mantle(solver, s_bottom):
    p_cmb = solver._P_basic_flat[:1]
    return float(np.asarray(solver.entropy_eos.temperature(p_cmb, np.array([s_bottom]))).item())


@pytest.mark.physics_invariant
def test_the_default_core_start_is_the_mantle_side_of_the_cmb(solver):
    """Without an explicit core temperature the core starts at T(S_bottom, P_cmb), so the
    boundary-layer flux is exactly zero at the start."""
    n_stag = solver._n_stag
    t_core = float(solver._S0[n_stag + 1])
    assert t_core == pytest.approx(_t_mantle(solver, float(solver._S0[0])), rel=1e-12)
    solver.dSdt(0.0, solver._S0)
    assert float(solver.state.heat_flux[0]) == 0.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('slot', [5.0e-4, -5.0e-4])
def test_the_output_cmb_node_is_the_bottom_cell_at_the_cmb_pressure(solver, slot):
    """With the core 50 K above the mantle, every node-0 output is the bottom cell at the
    CMB pressure whatever the gradient slot holds, and the applied outward flux is reported
    as conduction."""
    n_stag = solver._n_stag
    y = np.array(solver._S0, dtype=float)
    y[n_stag], y[n_stag + 1] = slot, y[n_stag + 1] + 50.0
    solver._solution = OptimizeResult(
        t=np.array([0.0, 1.0]), y=np.stack([y, y], axis=1), status=0, success=True, message=''
    )
    out = solver.get_state()
    assert float(out.T_basic[0]) == pytest.approx(_t_mantle(solver, y[0]), rel=1e-12)
    assert float(out.dSdr_b[0]) == 0.0
    assert float(out.jcond_b[0]) == float(out.heat_flux[0]) > 0.0
    assert [out.jconv_b[0], out.jgrav_b[0], out.jmix_b[0]] == [0.0, 0.0, 0.0]
    eos = solver.entropy_eos
    p_cmb, s0 = solver._P_basic_flat[:1], np.array([y[0]])
    assert float(out.phi_basic[0]) == pytest.approx(
        float(eos.melt_fraction(p_cmb, s0)[0]), rel=1e-10
    )
    assert float(out.rho_basic[0]) == pytest.approx(float(eos.density(p_cmb, s0)[0]), rel=1e-10)


@pytest.mark.physics_invariant
def test_the_gradient_slot_sets_no_other_rate(solver):
    """The dSdr_cmb slot keeps the energy_balance layout but feeds nothing back: changing it
    changes its own rate only, with the core 50 K above the mantle."""
    n_stag = solver._n_stag
    y = np.array(solver._S0, dtype=float)
    y[n_stag + 1] += 50.0
    rates = []
    for slot in (0.0, 5.0e-4, -5.0e-4):
        y[n_stag] = slot
        rates.append(np.array(solver.dSdt(0.0, y), dtype=float))
    for r in rates[1:]:
        np.testing.assert_array_equal(np.delete(r, n_stag), np.delete(rates[0], n_stag))
    assert rates[0][n_stag + 1] != 0.0


def test_the_cmb_flux_refuses_a_solver_without_entropy_tables(solver, monkeypatch):
    """The boundary-layer flux evaluates the bottom cell at the CMB pressure, which needs the
    entropy EOS tables; a solver without them (const_properties) is refused with the reason,
    while the same call with the tables returns a finite flux."""
    solver.dSdt(0.0, solver._S0)  # the flux reads the bottom cell's state
    assert np.isfinite(solver._core_module_cmb_flux(5000.0, float(solver._S0[0])))
    monkeypatch.setattr(solver, 'entropy_eos', None)
    with pytest.raises(ValueError, match='needs the entropy EOS tables'):
        solver._core_module_cmb_flux(5000.0, float(solver._S0[0]))


def test_the_cmb_flux_asks_for_an_evaluated_state():
    """Before any right-hand side evaluation the solver state is empty; the flux says so
    instead of failing on an empty array."""
    fresh = _build('core_module', entropy_eos_copy(), CORE_MODULE_PARAMS, s_init='driven')
    with pytest.raises(RuntimeError, match='evaluate the right-hand side'):
        fresh._core_module_cmb_flux(5000.0, float(fresh._S0[0]))
    fresh.dSdt(0.0, fresh._S0)
    assert np.isfinite(fresh._core_module_cmb_flux(5000.0, float(fresh._S0[0])))


def test_a_reset_keeps_the_budget_until_its_inputs_change():
    """A reset with unchanged module parameters and CMB geometry reuses the budget and its
    compiled functions; a changed parameter (the refit after an impact) rebuilds it."""
    params = dict(CORE_MODULE_PARAMS)
    fresh = _build('core_module', entropy_eos_copy(), params, s_init='driven')
    budget, dtcmb = fresh._core_module_budget, fresh._core_module_budget_dtcmb_dt
    fresh.reset()
    assert fresh._core_module_budget is budget
    assert fresh._core_module_budget_dtcmb_dt is dtcmb
    fresh.parameters.boundary_conditions.core_module_params['rho_cen'] = 12000.0
    fresh.reset()
    assert fresh._core_module_budget is not budget
    assert float(fresh._core_module_budget.profiles.rho_cen) == 12000.0
    for name in ('_P_basic_flat', '_r_basic_flat'):
        rebuilt = fresh._core_module_budget
        setattr(fresh, name, getattr(fresh, name) * 1.01)
        fresh._cache_bc_constants()
        assert fresh._core_module_budget is not rebuilt
    profiles = fresh._core_module_budget.profiles
    assert float(profiles.p_cmb) == float(fresh._P_basic_flat[0])
    assert float(profiles.r_cmb) == float(fresh._r_basic_flat[0])


def test_a_failed_budget_build_fails_again_on_the_next_reset():
    """A rejected parameter set leaves no cache entry behind, so a second reset with the
    same parameters raises again instead of keeping the previous budget."""
    fresh = _build('core_module', entropy_eos_copy(), dict(CORE_MODULE_PARAMS), s_init='driven')
    fresh.parameters.boundary_conditions.core_module_params['ds_fusion'] = -1.0
    for _ in range(2):
        with pytest.raises(ValueError, match='ds_fusion must be positive'):
            fresh.reset()
