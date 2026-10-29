"""Tests of the sub-grid lid in the NumPy path of ``EntropySolver``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aragog import surface_lid as sl
from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(300), needs_eos]


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


HEATING = [None, 'tidal', 'radio']


def _heat(p, heating):
    """Internal heating of about 3e-7 W/kg: tidal, or one radionuclide."""
    from aragog.parser import _Radionuclide

    if heating == 'tidal':
        p.energy.tidal = True
        p.energy.tidal_array = np.array([3.0e-7])
    elif heating == 'radio':
        p.energy.radionuclides = True
        p.radionuclides = [_Radionuclide('X', 0.0, 1.0, 3.0e-4, 1.0e-3, 1.0e9)]


def _solver(eos, lid=True, end=1.0, core_bc='quasi_steady', S0=3050.0, heating=None, **lidkw):
    from aragog.solver.entropy_solver import EntropySolver

    inner = 2 if core_bc == 'quasi_steady' else 1
    p = _build(core_bc=core_bc, outer_bc=1, inner_bc=inner, n_nodes=10, end_time=end)
    _heat(p, heating)
    p.boundary_conditions.subgrid_lid = lid
    for k, v in lidkw.items():
        setattr(p.boundary_conditions, k, v)
    p.boundary_conditions.normalize()
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(S0)
    return s


def test_state_helpers_run_on_a_carrier_without_the_lid_property(eos):
    """A carrier object that borrows the state helpers and holds the solver attributes, as a
    test double of a coupling code does, works with the lid off although it has no
    ``_n_lid`` property."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    s = _solver(eos, lid=False, core_bc='energy_balance')

    class Carrier:
        set_initial_entropy = EntropySolver.set_initial_entropy
        get_current_dSdr_cmb = EntropySolver.get_current_dSdr_cmb
        _final_extra_state = EntropySolver._final_extra_state
        _resolve_dSdr_cmb_init = EntropySolver._resolve_dSdr_cmb_init
        _n_shell = EntropySolver._n_shell
        _core_shell = EntropySolver._core_shell

    c = Carrier()
    c.__dict__.update(vars(s))
    c._solution = SimpleNamespace(y=np.full((s._n_stag + 1, 2), 7.0e-4))
    assert c.get_current_dSdr_cmb() == 7.0e-4
    c.set_initial_entropy(3000.0)
    assert c._S0.size == s._n_stag + 1


def test_lid_off_keeps_the_state_layout(eos):
    """With the default (lid off) the state has no lid entries and no lid is evaluated."""
    s = _solver(eos, lid=False)
    assert s._n_lid == 0
    assert s._S0.size == s._n_stag
    s._dSdt_single(0.0, s._S0)
    assert getattr(s, '_lid_last', None) is None


@pytest.mark.physics_invariant
def test_lid_starts_as_the_phantom_lid(eos):
    """The lid states sit at the end of the state; the start is the phantom lid (delta_lo,
    linear profile), with the mean between the base and the surface temperature."""
    s = _solver(eos)
    assert s._S0.size == s._n_stag + 2
    delta, E = s._S0[-2:]
    assert delta == pytest.approx(0.5, abs=0)
    s._dSdt_single(0.0, s._S0)
    ev = s._lid_last
    T_s = sl.linear_lid_surface(0.5, ev.T_b, s._lid_k, 1.0, 255.0)
    assert ev.T_m == pytest.approx(0.5 * (ev.T_b + T_s), rel=1e-9)
    assert T_s < ev.T_m < ev.T_b


def test_set_initial_lid_and_hot_start(eos):
    """A set lid is used as given; without it a second solve starts from the first's end."""
    s = _solver(eos)
    s.set_initial_lid(2.0, -1.0e9)
    s.set_initial_entropy(3050.0)
    np.testing.assert_array_equal(s._S0[-2:], [2.0, -1.0e9])
    s.set_initial_lid(None)
    s.solve()
    end = s._solution.y[-2:, -1].copy()
    s.set_initial_entropy(3050.0)
    np.testing.assert_array_equal(s._S0[-2:], end)
    with pytest.raises(ValueError, match='finite delta > 0'):
        s.set_initial_lid(-1.0, 0.0)
    with pytest.raises(ValueError, match='finite delta > 0'):
        s.set_initial_lid(1.0, float('nan'))


@pytest.mark.parametrize('heating', HEATING)
@pytest.mark.physics_invariant
@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    frac=st.floats(1e-5, 0.8),
    dT=st.floats(-300.0, 300.0),
    S_top=st.floats(2700.0, 3300.0),
    S_bot=st.floats(2700.0, 3300.0),
)
def test_ledger_per_call_closes(eos, heating, frac, dT, S_top, S_bot):
    """The per-call energy identity closes on random valid states: the lid at any thickness up
    to the growth stop, its mean off the steady value, any entropy profile (so the lid hidden,
    partly hidden or full), with and without internal heating; residual within 1e-10 of the
    largest power."""
    s = _solver(eos, heating=heating)
    delta = max(frac * s._lid_h_top, s._lid_delta_lo)
    y = s._S0.copy()
    y[: s._n_stag] = np.linspace(S_bot, S_top, s._n_stag)
    s._dSdt_single(0.0, y)
    ev = s._lid_last
    T_m = float(np.clip(ev.T_m + dT, 300.0, ev.T_b - 1.0))
    y[-2:] = (delta, delta * float(s._lid_hv.H(T_m)))
    pw = s._step_powers(0.0, y)
    scale = max(abs(pw[0]), abs(pw[1]), s._lid_area * abs(s._lid_last.rates.dE), 1.0)
    assert abs(pw[-1]) <= 1e-10 * scale
    assert 0.0 <= s._lid_last.rates.w <= 1.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('heating', ['tidal', 'radio'])
def test_ledger_per_call_closes_with_internal_heating(eos, heating):
    """With internal heating the per-call energy identity closes too: the lid heats the mass
    of its profile, delta times the mean density over the profile, in the ledger as in the
    lid energy rate."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    _heat(p, heating)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3050.0)
    y = s._S0.copy()
    y[-2:] = (300.0, 300.0 * float(s._lid_hv.H(900.0)))
    pw = s._step_powers(0.0, y)
    ev = s._lid_last
    assert ev.rates.H_lid > 0.0
    scale = max(abs(pw[0]), abs(pw[1]), s._lid_area * abs(ev.rates.dE), 1.0)
    assert abs(pw[-1]) <= 1e-10 * scale


@pytest.mark.physics_invariant
@pytest.mark.parametrize('heating', HEATING)
def test_ledger_over_a_solve(eos, heating):
    """Over a solve the step ledger closes to 1e-10 of the surface power and the state heat to
    1e-2 of the budget (its quadrature accuracy), with and without internal heating, and the lid
    grows well past its phantom start."""
    s = _solver(eos, heating=heating)
    s.solve()
    ei = s._solution.energy_integrals
    budget = ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal']
    assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])
    assert abs(ei['state_heat'] - budget) <= 1e-2 * abs(budget)
    assert s._lid_last.delta > 5.0 * s._lid_delta_lo


@pytest.mark.physics_invariant
@pytest.mark.parametrize('n_nodes', [10, 20, 40])
def test_lid_growth_sanity_against_quasi_static_neumann(eos, n_nodes, monkeypatch):
    """Sanity check: with no heat from the melt (F_b forced to 0), the lid over a molten column
    grows as the Neumann law taken step by step along the run, delta^2 = 4 int kappa
    lambda^2(St) dt, with St and kappa from the run's own states (the secant heat capacity of
    the crust), within 1 % at 10, 20 and 40 nodes (measured -0.45, -0.30, -0.03 %). The grey
    surface cools during the run, so a law with one surface temperature does not apply. The
    reference takes its heat capacity from the run, so it cannot tell a heat content at the
    base or mean temperature from the EOS one; the resolved-mesh check of V2 does."""
    from scipy.optimize import brentq
    from scipy.special import erf

    from aragog.solver import entropy_solver as es
    from aragog.solver.entropy_solver import EntropySolver

    rates = es._lid.lid_rates
    monkeypatch.setattr(es._lid, 'lid_rates', lambda **kw: rates(**{**kw, 'F_b': 0.0}))
    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=n_nodes, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3050.0)
    s.solve()
    sol = s._solution
    t_nd, y = (np.asarray(v) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    kappa_lam2 = []
    for k in range(t_nd.size):
        s._dSdt_single(t_nd[k], y[:, k])
        ev = s._lid_last
        H_b = float(s._lid_H_of_S(ev.S_b))
        cp = (H_b - float(s._lid_hv.H(ev.surf.T_s))) / (ev.rho * (ev.T_b - ev.surf.T_s))
        St = cp * (ev.T_b - ev.surf.T_s) / (ev.h_f - H_b / ev.rho)
        lam = brentq(lambda x: x * np.exp(x * x) * erf(x) - St / np.sqrt(np.pi), 1e-9, 10.0)
        kappa_lam2.append(s._lid_k / (ev.rho * cp) * lam**2)
    span = (s._solution.t[-1] - s._solution.t[0]) * 3.15576e7
    t = (t_nd - t_nd[0]) / (t_nd[-1] - t_nd[0]) * span
    assert y[-2, -1] == pytest.approx(np.sqrt(4.0 * np.trapezoid(kappa_lam2, t)), rel=1e-2)


def test_delta_on_must_fit_the_top_cell(eos):
    """lid_delta_on at or above the growth stop of the top cell is refused, naming the width."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    s0 = EntropySolver(p, entropy_eos=eos)
    s0.initialize()
    h_top = float(s0._r_basic_flat[-1] - s0._r_basic_flat[-2])
    with pytest.raises(ValueError, match='top-cell width'):
        _solver(eos, lid_delta_on=sl.LID_CAP_FRAC * h_top)
    _solver(eos, lid_delta_on=0.99 * sl.LID_CAP_FRAC * h_top)


def test_handover_cells_must_fit_the_mesh(eos):
    """A hand-over below the whole mesh is refused at setup."""
    with pytest.raises(ValueError, match='needs more than'):
        _solver(eos, lid_handover_cells=9)


def _phantom_energy(s, delta, T_m):
    s._dSdt_single(0.0, s._S0)
    ev = s._lid_last
    return delta * float(s._lid_hv.H(T_m)), ev


def test_lid_above_the_growth_stop_fails_the_step_check(eos):
    """An accepted state with the lid past the growth stop raises the lid invariant."""
    s = _solver(eos)
    d = 1.01 * sl.LID_CAP_FRAC * s._lid_h_top
    E, _ = _phantom_energy(s, d, 0.5 * (s._lid_T_sol + 300.0))
    s.set_initial_lid(d, E)
    s.set_initial_entropy(3050.0)
    with pytest.raises(RuntimeError, match='lid invariant violated at step 0') as err:
        s.solve()
    assert 'outside' in str(err.value)


def test_growth_stop_warns_once(eos, caplog):
    """A growing lid at the growth stop holds below LID_CAP_FRAC h_top and warns once."""
    import logging

    s = _solver(eos, end=10.0)
    d = (sl.LID_CAP_FRAC - 0.5 * sl.LID_CAP_WIDTH) * s._lid_h_top
    # a lid mean just above T_eq gives the steep base gradient of a lid that grows here
    E, ev = _phantom_energy(s, d, 257.0)
    s.set_initial_lid(d, E)
    s.set_initial_entropy(3050.0)
    with caplog.at_level(logging.WARNING, logger='fwl.aragog.solver.entropy_solver'):
        s.solve()
    msgs = [r.getMessage() for r in caplog.records if 'growth stop holds' in r.getMessage()]
    assert len(msgs) == 1
    assert s._solution.y[-2].max() <= sl.LID_CAP_FRAC * s._lid_h_top * (1 + 1e-6)
    assert s._solution.y[-2].max() > d


def test_outputs_and_netcdf(eos, tmp_path):
    """get_state reports the lid with the lid on and NaN with it off; the netCDF snapshot
    carries the lid scalars only with the lid on."""
    import netCDF4

    on = _solver(eos)
    on.solve()
    out = on.get_state()
    ev = on._lid_last
    assert out.lid_delta == pytest.approx(on._solution.y[-2, -1], rel=1e-15)
    assert out.lid_T_surface < out.lid_T_mean < out.lid_T_base
    assert out.lid_delta_eff == pytest.approx(ev.delta / sl.shape(ev.surf.s)[0], rel=1e-15)
    assert (
        out.lid_delta_eff < out.lid_delta
    )  # a growing lid (s > 0) conducts better than linear
    assert 0.0 <= out.lid_weight <= 1.0
    off = _solver(eos, lid=False)
    off.solve()
    out_off = off.get_state()
    assert np.isnan(out_off.lid_delta) and np.isnan(out_off.lid_T_base)
    for o, name in ((out, 'on.nc'), (out_off, 'off.nc')):
        o.to_netcdf(tmp_path / name, time=1.0)
    with netCDF4.Dataset(tmp_path / 'on.nc') as ds:
        assert float(ds['lid_delta'][...]) == pytest.approx(out.lid_delta, rel=1e-12)
    with netCDF4.Dataset(tmp_path / 'off.nc') as ds:
        assert 'lid_delta' not in ds.variables


def _cold_top_solver(eos, end=1.0, cells=1):
    """A 20-node column with the solid rheology ('lid' closure) and two cold top cells, where
    the lid-base isotherm lies below the face of the top cell at the start."""
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=20, end_time=end)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_handover_cells = cells
    p.boundary_conditions.normalize()
    p.phase_solid.rheology = SolidRheologyParams(enabled=True)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    S = np.linspace(3150.0, 3000.0, s._n_stag)
    S[-2:] = 1500.0
    s.set_initial_entropy(S)
    return s


@pytest.mark.physics_invariant
def test_lid_depth_is_the_isotherm_depth(eos):
    """The hand-over depth is where the column with the lid reaches T_lid_iso: the profile
    interpolated at that depth gives T_lid_iso, below the top-cell face here."""
    s = _cold_top_solver(eos)
    g = s._lid_root_value(0.0, s._S0)
    z = s._lid_depth()
    T_iso = float(np.asarray(s.state.lid_state['T_lid_iso']))
    T = np.asarray(s.state.phase_staggered.temperature()).ravel()[::-1]
    ev = s._lid_last
    zs = np.r_[0.0, ev.delta, s._lid_z_stag]
    Ts = np.r_[ev.surf.T_s, ev.T_b, T]
    assert np.interp(z, zs, Ts) == pytest.approx(T_iso, rel=1e-12)
    assert z > s._lid_handover_depth
    assert g == pytest.approx((z - s._lid_handover_depth) / s._lid_handover_depth, rel=1e-12)


@pytest.mark.physics_invariant
def test_merge_is_energy_exact(eos):
    """The merge keeps the heat content: V_N H(S') = V_N H(S_f) + A (E_lid - rho h_f delta),
    with H checked by an independent adaptive quadrature of rho T dS."""
    from scipy.integrate import quad

    s = _cold_top_solver(eos)
    y = s._S0.copy()
    y[-2:] = (20.0, 40.0 * y[-1])  # a 20 m lid with 40 times the phantom energy
    y_new, resid = s._lid_merge(y)
    eos_, P = s.entropy_eos, s._lid_P

    def H(S):
        def f(x):
            return float(eos_.density(P, x) * eos_.temperature(P, x))

        pts = [p for p in (s._lid_S_liq,) if min(s._lid_S_sol, S) < p < max(s._lid_S_sol, S)]
        return quad(f, s._lid_S_sol, S, points=pts or None, epsrel=1e-12, limit=200)[0]

    i = s._n_stag - 1
    T, phi = (
        float(np.asarray(fn(P, y[i])).item()) for fn in (eos_.temperature, eos_.melt_fraction)
    )
    base = s._lid_base_state(float(y[i]), T, phi)
    V, A = float(s._volume_flat[-1]), s._lid_area
    before = V * H(float(y[i])) + A * (y[-1] - base['rho'] * base['h_f'] * y[-2])
    after = V * H(float(y_new[i]))
    assert after == pytest.approx(before, rel=1e-7)
    assert resid <= 1e-10
    np.testing.assert_array_equal(y_new[-2:], [0.0, 0.0])
    np.testing.assert_array_equal(y_new[:i], y[:i])
    assert y_new[i] != y[i]


def test_trigger_passed_at_the_start_hands_over_at_the_start(eos, caplog):
    """A lid whose hand-over trigger lies behind the call start hands over at the start; the
    lid states stay zero for the whole call."""
    import logging

    s = _cold_top_solver(eos)
    with caplog.at_level(logging.INFO, logger='fwl.aragog.solver.entropy_solver'):
        s.solve()
    out = s.get_state()
    assert out.lid_handover_time == pytest.approx(0.0, abs=0)
    assert out.lid_delta == pytest.approx(0.0, abs=0)
    assert np.all(s._solution.y[-2:] == 0.0)
    assert any('handed over to the mesh at t=0.0' in r.getMessage() for r in caplog.records)


def _methods():
    from aragog.solver import entropy_solver as es

    cv = ['cvode-fixed', 'cvode-rate'] if es._CVODE_AVAILABLE else []
    return ['radau', *cv]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('method', _methods())
def test_handover_in_the_call(eos, method):
    """A hand-over at t* inside the call merges there and continues to the call end on every
    integrator path (scipy, plain CVODE, CVODE segments); the ledger closes across it.

    The trigger is replaced by g = (t - t*) / t* on this mushy column, which has no stagnant
    lid; the physical trigger has its own test."""
    t_star = 0.4
    s = _solver(eos, end=1.0)
    if method != 'radau':
        s.parameters.energy.solver_method = 'cvode'
        s.parameters.energy.use_jax_jacobian = False
        s.parameters.energy.phase_boundary_cap = method.split('-')[1]
    s._lid_handover_armed = lambda: True
    s._lid_root_value = lambda t, y: -1.0 if y[-2] == 0.0 else (t - t_star) / t_star
    s.solve()
    sol, out = s._solution, s.get_state()
    assert out.lid_handover_time == pytest.approx(t_star, rel=1e-6)
    assert sol.t[-1] == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.diff(sol.t) > 0)
    assert np.all(sol.y[-2:, -1] == 0.0) and sol.y[-2, 0] > 0.0
    tr_t = sol.energy_trace[0]
    assert np.count_nonzero(np.isclose(tr_t, out.lid_handover_time, rtol=0, atol=1e-12)) == 2
    ei = sol.energy_integrals
    budget = ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal']
    assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])
    assert abs(ei['state_heat'] - budget) <= 1e-2 * abs(budget)
    if method == 'cvode-rate':
        assert any(seg[1] == 'lid' for seg in sol.segments)


@pytest.mark.physics_invariant
def test_resume_after_handover_matches_the_lid_off_run(eos):
    """A run resumed with the handed-over lid (0, 0) evolves as the run without the lid."""
    s = _solver(eos)
    s.set_initial_lid(0.0, 0.0)
    s.set_initial_entropy(3050.0)
    s.solve()
    off = _solver(eos, lid=False)
    off.solve()
    np.testing.assert_allclose(
        s._solution.y[: s._n_stag, -1], off._solution.y[:, -1], rtol=1e-6
    )
    np.testing.assert_array_equal(s._solution.y[-2:, -1], [0.0, 0.0])
    assert s.get_state().lid_delta == pytest.approx(0.0, abs=0)


@pytest.mark.parametrize('closure', ['local', None])
def test_no_handover_without_the_lid_closure(eos, closure):
    """The hand-over needs the stagnant-lid state of the 'lid' stress closure: with the 'local'
    closure, or without the rheology, the column whose trigger has passed keeps its lid."""
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=20, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_handover_cells = 1
    p.boundary_conditions.normalize()
    if closure is not None:
        p.phase_solid.rheology = SolidRheologyParams(enabled=True, stress_closure_mode=closure)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    S = np.linspace(3150.0, 3000.0, s._n_stag)
    S[-2:] = 1500.0
    s.set_initial_entropy(S)
    assert not s._lid_handover_armed()
    assert s._lid_root_value(0.0, s._S0) == pytest.approx(-1.0, abs=0)
    s.solve()
    out = s.get_state()
    assert np.isnan(out.lid_handover_time)
    assert out.lid_delta > 0.0


@pytest.mark.physics_invariant
def test_hidden_lid_on_a_hot_top_cell(eos):
    """A hot convecting top cell carries more heat than a crust at T_b can radiate: the lid is
    hidden (w = 0), the surface loses the grid flux, the lid relaxes to the phantom thickness and
    the per-call ledger closes."""
    s = _solver(eos)
    y = s._S0.copy()
    y[: s._n_stag] = np.linspace(3300.0, 4500.0, s._n_stag)
    y[-2:] = (3.0, y[-1] * 6.0)
    off = _solver(eos, lid=False)
    off._dSdt_single(0.0, y[: s._n_stag])
    f_grid = float(off.state._heat_flux[-1])
    pw = s._step_powers(0.0, y)
    ev = s._lid_last
    assert ev.rates.w == pytest.approx(0.0, abs=0)
    assert ev.rates.F_top == pytest.approx(f_grid, rel=1e-12)
    assert ev.rates.ddelta == pytest.approx(
        -(3.0 - s._lid_delta_lo) / sl.LID_TAU_HIDDEN, rel=1e-12
    )
    assert abs(pw[-1]) <= 1e-10 * max(abs(pw[0]), abs(pw[1]), 1.0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('source', ['fixed', 'rheological', 'lid_state'])
def test_solid_lid_base_is_the_rheological_temperature(eos, source, monkeypatch):
    """Below the rheological transition the lid base lies at the rheological temperature of
    the solid rheology on the EOS, not on a constant-cp estimate of its entropy."""
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import _R_GAS, EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    mode = 'fixed' if source == 'fixed' else 'rheological'
    p.phase_solid.rheology = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='local',
        lid_base_mode=mode,
        lid_base_temperature=1300.0,
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    T_f = 1350.0
    S_f = eos.entropy_at_temperature(s._lid_P, T_f)
    rh = {'fixed': 1300.0, 'lid_state': 1250.0}
    expected = rh.get(source) or T_f - 2.2 * _R_GAS * T_f**2 / s._lid_H
    lid_state = {'T_lid_iso': 1250.0} if source == 'lid_state' else None
    monkeypatch.setattr(s.state, '_lid_state', lid_state, raising=False)
    assert 600.0 < expected < s._lid_T_sol
    base = s._lid_base_state(S_f, T_f, 0.0)
    assert base['T_b'] == pytest.approx(expected, rel=1e-9)
