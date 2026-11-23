"""Core energy ledger of the CMB closure: the core heat change equals the CMB heat flow.

The energy_balance and gradient boundary conditions move the CMB basic-node entropy so that
the node temperature follows the core balance ``c_core T_hat M_core dT_cmb/dt = -E`` (Bower
et al. 2018, Eqs. 19-21). The node temperature is the phase blend ``T(P, S)``, so the
closure must convert with ``(dT/dS)_P`` of that blend. These tests run the solver with the
CMB node in the mush, where ``c_p/T`` of the blend differs from ``(dS/dT)_P`` by
``T_avg/T_mix``, and require the ledger to close over the call.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EOS_DIR, entropy_eos_copy, needs_eos
from tests.test_entropy_solver_integration import _build_parameters

pytestmark = [pytest.mark.smoke, needs_eos, pytest.mark.timeout(600)]

# Tight tolerances, so the solver error floor sits far below the closure bias (about 0.1);
# the gradient layout sums the node entropy over all gradient states and needs the tighter one.
RTOL = {'energy_balance': 1.0e-10, 'gradient': 1.0e-12}
LEDGER_TOL = 1.0e-4


def _jax_args(solver):
    """JAX RHS arguments of an energy_balance solver: eos, phases, mesh, BC, heating, radio."""
    from aragog.jax.solver import BoundaryParams
    from tests.test_jax_dsdt_core_module import _build_jax_pieces

    for name, value in (
        ('_core_module_budget', None),
        ('_core_module_q_radio', 0.0),
        ('_core_module_ra_crit_cmb', None),
    ):
        setattr(solver, name, value)
    args = list(_build_jax_pieces(solver))
    bc = solver.parameters.boundary_conditions
    args[3] = BoundaryParams(
        outer_bc_type=bc.outer_boundary_condition,
        outer_bc_value=bc.outer_boundary_value,
        emissivity=bc.emissivity,
        T_eq=bc.equilibrium_temperature,
        inner_bc_type=5,
        inner_bc_value=0.0,
        core_density=float(solver._core_density),
        core_heat_capacity=float(solver._core_cp),
        tfac_core_avg=float(solver._core_tfac),
        cmb_area=float(solver._cmb_area),
        core_M=float(solver._core_M),
        cmb_dr_cmb=float(solver._cmb_dr_cmb),
    )
    return tuple(args[:6])


def _jax_factory(solver):
    """Option Z factory for an energy_balance solver (JAX RHS and Jacobian)."""
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    args = _jax_args(solver)
    calls = {'n': 0}

    def factory(scales, mode):
        calls['n'] += 1
        rhs, jac, _ = build_jax_rhs_and_jacobian(
            eos_jax=args[0],
            phase_params=args[1],
            mesh_arrays=args[2],
            boundary_params=args[3],
            heating_array=np.zeros(solver._n_stag),
            scales=scales,
            core_bc_mode=mode,
        )
        return rhs, jac

    return factory, calls


def _node_temperature(solver, t, y) -> float:
    """CMB basic-node temperature [K] of the solver state ``y`` at ``t`` [yr]."""
    solver._dSdt_single(float(t), np.asarray(y, dtype=float))
    return float(np.asarray(solver.state.phase_basic.temperature()).flat[0])


def _profile(solver, eos, g0):
    """Entropy at melt coordinate ``g0`` on every staggered node."""
    P = np.asarray(solver._P_stag_flat, dtype=float)
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    return S_sol + g0 * (S_liq - S_sol)


def _ledger(core_bc, method, g0, jax_rhs=False, end_time=20.0):
    """Run one call from a mush profile at melt coordinate ``g0``; return the ledger terms."""
    from aragog.solver.entropy_solver import EntropySolver

    params = _build_parameters(
        core_bc=core_bc,
        n_nodes=15,
        end_time=end_time,
        inner_boundary_condition=1,
        solver_method=method,
        use_jax_jacobian=jax_rhs,
    )
    params.solver.rtol = params.solver.atol = RTOL[core_bc]
    eos = entropy_eos_copy(EOS_DIR)
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(_profile(solver, eos, g0))
    calls = None
    if jax_rhs:
        factory, calls = _jax_factory(solver)
        solver.set_jax_cvode_factory(factory)
    solver.solve()
    out = solver.get_state()
    sol = solver._solution
    c_core = float(solver._core_cp * solver._core_tfac * solver._core_M)
    d_t = _node_temperature(solver, sol.t[-1], sol.y[:, -1]) - _node_temperature(
        solver, sol.t[0], sol.y[:, 0]
    )
    q = float(out.step_dE_F_cmb_J)
    return c_core * d_t, q, calls, solver


# no gradient case at g = 0.3: its CMB heat is 5e25 J, and the 1e-8 atol floor on its 14
# gradient slots holds the node entropy only to about 0.03 J/kg/K per step
CASES = [
    ('energy_balance', 'radau', False, 0.9),
    ('energy_balance', 'radau', False, 0.3),
    ('energy_balance', 'cvode', False, 0.9),
    ('energy_balance', 'cvode', False, 0.3),
    ('energy_balance', 'cvode', True, 0.9),
    ('energy_balance', 'cvode', True, 0.3),
    ('gradient', 'radau', False, 0.9),
]


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('core_bc', 'method', 'jax_rhs', 'g0'), CASES)
def test_core_heat_change_equals_the_cmb_heat_flow(core_bc, method, jax_rhs, g0):
    """``c_core T_hat M_core dT_node + int E dt = 0`` over a call with the CMB node in the mush."""
    if method == 'cvode':
        pytest.importorskip('scikits_odes_sundials')
    if jax_rhs:
        pytest.importorskip('equinox')
    d_e, q, calls, solver = _ledger(core_bc, method, g0, jax_rhs=jax_rhs)
    check = solver.cmb_ledger
    if jax_rhs:
        assert calls['n'] > 0, 'the JAX factory was not used'
    assert q > 0.0, f'the fixture must cool the core, q = {q:.4e} J'
    miss = (d_e + q) / abs(q)
    assert abs(miss) < LEDGER_TOL, (
        f'core ledger misses by {miss:+.5f} of the CMB heat flow '
        f'(core change {d_e:+.4e} J, CMB heat {q:+.4e} J)'
    )
    assert not check['excluded'] and abs(check['r']) <= check['bound'], check


@pytest.mark.physics_invariant
def test_the_cp_over_t_closure_raises_the_closure_fault(monkeypatch):
    """The ``c_p/T`` closure of the blend misses the ledger by about 9 %, above 10x the bound."""
    from aragog.solver.entropy_solver import EntropySolver

    def cp_of_the_blend(self):
        return float(np.asarray(self.state.phase_basic.heat_capacity()).flat[0])

    monkeypatch.setattr(EntropySolver, '_cmb_closure_cp', cp_of_the_blend)
    with pytest.raises(RuntimeError, match='CMB closure fault'):
        _ledger('energy_balance', 'radau', 0.9)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('stratified', 'method'),
    [(False, 'radau'), pytest.param(True, 'cvode', marks=pytest.mark.slow)],
)
def test_core_module_node_follows_the_core_top_temperature(stratified, method):
    """In core_module the CMB node rides on the core top: both change by the same amount."""
    from tests.test_entropy_solver_core_module_smoke import (
        CORE_MODULE_PARAMS,
        STRATIFIED_PARAMS,
        _build,
    )

    if method == 'cvode':
        pytest.importorskip('scikits_odes_sundials')
    eos = entropy_eos_copy(EOS_DIR)
    params = STRATIFIED_PARAMS if stratified else CORE_MODULE_PARAMS
    solver = _build('core_module', eos, params, end_time=2.0, solver_method=method)
    S = _profile(solver, eos, 0.9)
    solver.set_initial_entropy(S)
    t_m = float(np.asarray(eos.temperature(solver._P_basic_flat[:1], S[:1])).item())
    solver.set_initial_core_temperature(t_m + 50.0)
    solver.set_initial_entropy(S)
    solver.solve()
    sol = solver._solution
    top = -1 if stratified else solver._n_stag + 1
    d_top = float(sol.y[top, -1] - sol.y[top, 0])
    d_node = _node_temperature(solver, sol.t[-1], sol.y[:, -1]) - _node_temperature(
        solver, sol.t[0], sol.y[:, 0]
    )
    assert d_top < 0.0, f'the fixture must cool the core top, dT_top = {d_top:+.4e} K'
    assert d_node / d_top == pytest.approx(1.0, rel=LEDGER_TOL), (
        f'the CMB node changes by {d_node:+.6e} K while the core top changes by {d_top:+.6e} K'
    )


def _table_slope(solver, S_node):
    """Central difference of the evaluated node temperature in S [K kg K/J]."""
    ev = solver._cmb_node_phase
    ev.set_pressure(np.full(2, float(solver._P_basic_flat[0])))
    ev.set_entropy(np.array([S_node + 1.0e-3, S_node - 1.0e-3]))
    ev.update()
    T = np.ravel(ev.temperature())
    return float(T[0] - T[1]) / 2.0e-3


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('g0', [-0.03, 0.3, 0.9, 1.004])
@pytest.mark.parametrize('core_bc', ['energy_balance', 'gradient', 'core_module'])
def test_the_rhs_moves_the_node_temperature_at_the_core_rate(core_bc, g0):
    """At one state the NumPy RHS gives ``C (dT/dS)_P dS_node/dt = -E`` (energy_balance,
    gradient) and ``(dT/dS)_P dS_node/dt = dT_top/dt`` (core_module): the chain rule at fixed
    pressure with the core balance of Bower et al. (2018), Eqs. 19-21. In the mush (g 0.3,
    0.9) the slope is also the central difference of the evaluated node temperature, so there
    the node moves at the core rate; in the solid and the band (g -0.03, 1.004) the identity
    holds with the closure slope, which differs from the tabulated slope there."""
    from scipy.constants import Julian_year

    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

    eos = entropy_eos_copy(EOS_DIR)
    if core_bc == 'core_module':
        solver = _build('core_module', eos, CORE_MODULE_PARAMS)
        S = _profile(solver, eos, g0)
        solver.set_initial_entropy(S)
        t_m = float(np.asarray(eos.temperature(solver._P_basic_flat[:1], S[:1])).item())
        solver.set_initial_core_temperature(t_m + 50.0)
        solver.set_initial_entropy(S)
    else:
        params = _build_parameters(core_bc=core_bc, n_nodes=15, inner_boundary_condition=1)
        solver = EntropySolver(params, entropy_eos=eos)
        solver.initialize()
        solver.set_initial_entropy(_profile(solver, eos, g0))
    n, y = solver._n_stag, np.asarray(solver._S0, dtype=float)
    f = np.asarray(solver._dSdt_single(0.0, y), dtype=float) / Julian_year
    slope = float(np.asarray(solver.state.phase_basic.temperature_slope()).flat[0])
    # the node offset of each RHS: half the basic spacing, the staggered radius in gradient
    off, sum_abs = -0.5 * float(solver._cmb_dr_cmb), 0.0
    if core_bc == 'gradient':
        off = float(solver._r_basic_flat[0] - solver._r_stag_flat[0])
        terms = np.r_[f[n + 1], -np.diff(solver._r_stag_flat) * f[1:n], off * f[0]]
        dS_node, sum_abs = float(np.sum(terms)), float(np.sum(np.abs(terms)))
    else:
        dS_node = f[0] + off * f[n]
    if core_bc == 'gradient':
        S_node = y[n + 1] - np.diff(solver._r_stag_flat) @ y[1:n] + y[0] * off
    else:
        S_node = y[0] + y[n] * off
    if 0.0 < g0 < 1.0:  # in the mush the closure slope is the slope of the evaluated T
        assert slope == pytest.approx(_table_slope(solver, float(S_node)), rel=1.0e-6)
    if core_bc == 'core_module':
        assert slope * dS_node == pytest.approx(f[n + 1], rel=1.0e-10)
        return
    E = float(solver.state._heat_flux[0]) * float(solver._cmb_area)
    C = float(solver._core_cp * solver._core_tfac * solver._core_M)
    assert abs(E) > 0.0
    # the gradient node rate sums rates up to 8e9 times its size; 50 eps of that sum
    cancel = 50.0 * np.finfo(float).eps * C * slope * sum_abs
    assert C * slope * dS_node == pytest.approx(-E, rel=1.0e-10, abs=cancel)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('g0', [-0.03, 0.9, 1.004])
def test_the_jax_rhs_moves_the_node_temperature_at_the_core_rate(g0):
    """The JAX energy_balance RHS gives ``C (dT/dS)_P dS_node/dt = -E`` at one state (chain rule
    at fixed pressure, Bower et al. 2018, Eqs. 19-21)."""
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('equinox')
    from scipy.constants import Julian_year

    from aragog.jax.phase import evaluate_phase
    from aragog.jax.solver import _dSdt_energy_balance_parts
    from aragog.solver.entropy_solver import EntropySolver

    eos = entropy_eos_copy(EOS_DIR)
    params = _build_parameters(
        core_bc='energy_balance', n_nodes=15, inner_boundary_condition=1, solver_method='cvode'
    )
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(_profile(solver, eos, g0))
    args = _jax_args(solver)
    n, y = solver._n_stag, jnp.asarray(solver._S0, dtype=float)
    parts = _dSdt_energy_balance_parts(0.0, y, args)
    f = np.asarray(parts.rate) / Julian_year
    off = -0.5 * float(solver._cmb_dr_cmb)
    S_node = float(y[0]) + float(y[n]) * off
    node = evaluate_phase(args[0], args[1], args[2].P_basic[:1], jnp.array([S_node]))
    slope = float(node.temperature_slope[0])
    E = float(parts.heat_flux[0]) * float(solver._cmb_area)
    C = float(solver._core_cp * solver._core_tfac * solver._core_M)
    assert C * slope * (f[0] + off * f[n]) == pytest.approx(-E, rel=1.0e-10)


def test_a_call_through_a_padded_cell_is_left_out_of_the_check(caplog):
    """Just above the liquidus near the CMB pressure the CI melt table is flat in S (a padded
    cell): the table slope is 0, so the call is left out of the ledger check with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger='fwl.aragog'):
        d_e, q, _, solver = _ledger('energy_balance', 'radau', 1.0005, end_time=5.0)
    check = solver.cmb_ledger
    assert check['excluded'] and check['rel'] == pytest.approx(1.0, rel=1.0e-6)
    assert solver._cmb_ledger_counts == {'missed': 0, 'excluded': 1}
    assert 'CMB closure check skipped' in caplog.text
    # the padded cell holds the node temperature while the core loses heat
    assert (d_e + q) / abs(q) > 1.0e-3


def test_a_small_closure_miss_warns_and_is_counted(monkeypatch, caplog):
    """A closure 3e-4 off misses the ledger by about 3 bounds: the run warns and goes on."""
    import logging

    from aragog.solver.entropy_solver import EntropySolver

    exact = EntropySolver._cmb_closure_cp
    monkeypatch.setattr(EntropySolver, '_cmb_closure_cp', lambda self: exact(self) * (1 + 3e-4))
    with caplog.at_level(logging.WARNING, logger='fwl.aragog'):
        _, _, _, solver = _ledger('energy_balance', 'radau', 0.9)
    check = solver.cmb_ledger
    assert check['bound'] < abs(check['r']) < 10.0 * check['bound'], check
    assert solver._cmb_ledger_counts == {'missed': 1, 'excluded': 0}
    assert 'CMB closure ledger miss' in caplog.text


def test_a_non_positive_node_slope_stops_the_rhs_and_the_check(monkeypatch):
    """A node slope of 0 or below is refused with a message, in the RHS (before a division by 0)
    and in the ledger check over the accepted states."""
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_solver import EntropySolver

    eos = entropy_eos_copy(EOS_DIR)
    params = _build_parameters(core_bc='energy_balance', n_nodes=15, inner_boundary_condition=1)
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(_profile(solver, eos, 0.9))
    y = np.asarray(solver._S0, dtype=float)
    solver._call_tol = (1.0e-8, 1.0e-8)  # as solve() sets them
    for value in (0.0, -1.0, np.nan):
        monkeypatch.setattr(
            EntropyPhaseEvaluator, 'temperature_slope', lambda self, v=value: np.full(1, v)
        )
        with pytest.raises(RuntimeError, match='finite positive slope'):
            solver._dSdt_single(0.0, y)
        with pytest.raises(RuntimeError, match='node slope'):
            solver._check_cmb_ledger(np.array([0.0, 1.0]), np.c_[y, y], np.zeros(2), 0.0)
