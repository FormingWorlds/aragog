"""Per-call energy quadrature over CVODE's accepted internal steps (``EntropySolver._energy_trace``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from aragog.solver import entropy_solver as es

pytestmark = [
    pytest.mark.skipif(es._scikits_cvode is None, reason='scikits-odes-sundials not installed'),
]

AMP, T_PULSE, WIDTH = 3.0, 0.37, 3e-3


def _pulse(t: float) -> float:
    return np.exp(-(((t - T_PULSE) / WIDTH) ** 2))


def _solve(n_out=2, rootfn=None):
    """Integrate y' = -AMP pulse(t) over [0, 1] through ``_solve_cvode``."""
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._core_bc, s._cvode_output_points, s._max_steps = 'quasi_steady', n_out, 100000
    # A time-only pulse has no state signature, so a step cap makes CVODE step through it.
    return s._solve_cvode(
        start_time=0.0,
        end_time=1.0,
        y0=np.array([0.0]),
        atol=1e-14,
        rtol=1e-10,
        max_step=0.01,
        rhs=lambda t, y: np.array([-AMP * _pulse(t)]),
        phi_cap_rootfn=rootfn,
    )


def _trap(t, p):
    return float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(t)))


@pytest.mark.unit
def test_trace_filters_nodes():
    t, y = np.array([0.0, 1.0]), np.array([[1.0, 5.0], [2.0, 6.0]])
    nodes = [(0.3, [1.1, 2.1]), (0.2, [9, 9]), (0.5, [1.2, 2.2]), (0.5, [9, 9]), (1.5, [9, 9])]
    ts, ys = es.EntropySolver._energy_trace(nodes, t, y)
    np.testing.assert_array_equal(ts, [0.0, 0.3, 0.5, 1.0])
    np.testing.assert_array_equal(ys, [[1, 1.1, 1.2, 5], [2, 2.1, 2.2, 6]])


@pytest.mark.unit
def test_pulse_between_outputs_is_integrated():
    res = _solve()
    t, y = res.energy_trace
    exact = -AMP * np.sqrt(np.pi) * WIDTH
    assert res.t.size == 2 and t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    assert np.array_equal(y[:, -1], res.y[:, -1])
    assert abs(_trap(t, -AMP * _pulse(t)) / exact - 1.0) < 1e-3
    assert abs(_trap(res.t, -AMP * _pulse(res.t)) / exact) < 1e-6


@pytest.mark.unit
def test_trace_holds_every_output():
    res = _solve(n_out=9)
    assert np.all(np.isin(res.t, res.energy_trace[0]))


@pytest.mark.unit
def test_trace_ends_at_the_root():
    half = -0.5 * AMP * np.sqrt(np.pi) * WIDTH

    def root(t, y, g):
        g[0] = y[0] - half

    root.phi0 = 0.0  # read when a cap root fires
    res = _solve(n_out=65, rootfn=root)
    t, y = res.energy_trace
    assert res.cvode_flag == 2 and t[-1] == res.t[-1] == pytest.approx(T_PULSE, abs=1e-6)
    assert np.all(np.diff(t) > 0) and y[0, -1] == res.y[0, -1]
    assert abs(_trap(t, -AMP * _pulse(t)) / half - 1.0) < 1e-3


@pytest.mark.unit
def test_a_never_firing_root_leaves_the_solution_unchanged():
    """The recording root function relies on this CVODE property."""

    def rhs(t, y, ydot):
        ydot[0] = -AMP * _pulse(t) - 0.1 * y[0]
        return 0

    def never(t, y, g):
        g[0] = 1.0

    def run(**extra):
        s = es._scikits_cvode(
            rhs, old_api=False, rtol=1e-10, atol=1e-14, max_step_size=0.01, **extra
        )
        return s.solve(np.linspace(0.0, 1.0, 9) ** 2, np.array([0.0])), s.get_info()

    (ref, ref_info), (sol, info) = run(), run(rootfn=never, nr_rootfns=1)
    assert np.array_equal(ref.values.t, sol.values.t)
    assert np.array_equal(ref.values.y, sol.values.y)
    assert ref_info['NumSteps'] == info['NumSteps']


@pytest.mark.unit
def test_recording_a_firing_root_leaves_the_solution_unchanged():
    """Outputs before the root and the root time match an unwrapped root function."""

    def rhs(t, y, ydot):
        ydot[0] = -np.cos(t)
        return 0

    def root(t, y, g):
        g[0] = y[0] + 0.5

    seen = []

    def recorded(t, y, g):
        seen.append(t)
        return root(t, y, g)

    def run(fn):
        s = es._scikits_cvode(
            rhs, old_api=False, rtol=1e-10, atol=1e-14, rootfn=fn, nr_rootfns=1
        )
        # Smooth y, so the step that holds the root at pi/6 also holds the output at 0.52.
        return s.solve(np.array([0.0, 0.45, 0.52, 1.0]), np.array([0.0]))

    ref, sol = run(root), run(recorded)
    assert sol.flag == 2 and np.array_equal(sol.values.t, [0.0, 0.45, 0.52])
    assert np.array_equal(ref.values.y, sol.values.y) and ref.roots.t[0] == sol.roots.t[0]
    assert np.any(np.diff(seen) < 0), 'the root search must revisit earlier times'


@pytest.mark.smoke
def test_cmb_energy_matches_the_core_energy_change():
    """bower2018 evolves T_core by -F_cmb A_cmb / C_core, so int F_cmb A_cmb dt = -C_core dT_core.

    With only the two call endpoints as outputs, the integral over the internal steps
    still closes the core energy budget.
    """
    from .test_entropy_verification import EOS_DIR, TestCvodeEnergyOutputGrid

    if not EOS_DIR.exists():
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}')
    s = TestCvodeEnergyOutputGrid._build_greybody_solver('cvode', n_out=2, core_bc='bower2018')
    s.solve()
    core = -s._core_cap * float(s._solution.y[-1, -1] - s._solution.y[-1, 0])
    assert abs(core) > 1e24, 'the core must exchange a non-trivial amount of heat'
    assert s._compute_step_energy_integrals()['F_cmb'] == pytest.approx(core, rel=1e-6)


@pytest.mark.smoke
def test_integral_up_to_a_fired_cap_matches_a_dense_reference():
    """A temperature cap ends the call at a root after a steep grey-body flux decay.

    The kept nodes increase in time and give the surface integral of a dense-output
    run over the same span; the call's two endpoints alone miss it by 25 percent.
    """
    from .test_entropy_verification import EOS_DIR, TestCvodeEnergyOutputGrid

    if not EOS_DIR.exists():
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}')
    s = TestCvodeEnergyOutputGrid._build_greybody_solver('cvode', n_out=65)
    s.parameters.energy.temperature_step_cap = 1000.0
    s.solve()
    sol = s._solution
    t = sol.energy_trace[0]
    assert sol.cap_label == 'temperature' and t[-1] == sol.t[-1]
    assert t.size > 100 and np.all(np.diff(t) > 0)
    ref = TestCvodeEnergyOutputGrid._build_greybody_solver('cvode', n_out=1025)
    ref.parameters.solver.end_time = float(sol.t[-1])
    ref.solve()
    ref._solution.energy_trace = None
    F_ref = ref._compute_step_energy_integrals()['F_int']
    assert s._compute_step_energy_integrals()['F_int'] == pytest.approx(F_ref, rel=1e-5)
    sol.energy_trace = (sol.t[[0, -1]], sol.y[:, [0, -1]])
    assert abs(s._compute_step_energy_integrals()['F_int'] / F_ref - 1.0) > 0.1


class _Cap:
    """Root function object in the step-cap shape: CVODE reaches it through ``evaluate``."""

    phi0, binding_cap, cap, evals = 0.0, 'phi', 0.5, 0

    def evaluate(self, t, y, g, userdata=None):
        g[0] = y[0] + 0.5 * AMP * np.sqrt(np.pi) * WIDTH
        return 0


@pytest.mark.unit
@pytest.mark.parametrize('cap', [None, _Cap()])
def test_recording_leaves_the_solution_bitwise_unchanged(monkeypatch, cap):
    real, spies = es._scikits_cvode, []

    class Spy:
        def __init__(self, rhs, **opts):
            self.rhs, self.opts, self._s = rhs, opts, real(rhs, **opts)
            spies.append(self)

        def solve(self, tspan, y0):
            self.args = (np.array(tspan), np.array(y0))
            return self._s.solve(tspan, y0)

        def __getattr__(self, name):
            return getattr(self._s, name)

    monkeypatch.setattr(es, '_scikits_cvode', Spy)
    res = _solve(n_out=9, rootfn=cap)
    opts = dict(spies[0].opts)
    if cap is None:
        del opts['rootfn'], opts['nr_rootfns']
    else:
        opts['rootfn'] = cap.evaluate
    ref = real(spies[0].rhs, **opts).solve(*spies[0].args)
    if cap is None:
        assert ref.flag == 0 and res.cvode_flag == 0
        assert np.array_equal(res.t, ref.values.t) and np.array_equal(res.y, ref.values.y.T)
    else:
        assert ref.flag == 2 and res.cvode_flag == 2 and res.cap_fired
        assert res.t[-1] == ref.roots.t[-1] and np.array_equal(res.y[:, -1], ref.roots.y[-1])


def _fake_solver(sol):
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._solution, s.entropy_eos = sol, object()
    s._volume_flat, s._r_basic_flat = np.ones(1), np.array([1.0, 2.0])
    s.evaluator = SimpleNamespace(mesh=SimpleNamespace(staggered_effective_density=np.ones(1)))
    s.state = SimpleNamespace(_pb_cache_hits=3, _pb_cache_misses=4)
    s._stag_entropy, s._step_heat_content = (lambda y: y), (lambda a, b: 0.0)
    seen = []

    def powers(t, y, geom):
        seen.append((t, y.copy()))
        s.state._pb_cache_hits += 1
        return np.array([t, 2 * t, 3 * t, 4 * t, 5 * t, 6 * t, 7 * t])

    s._step_powers = powers
    return s, seen


@pytest.mark.unit
@pytest.mark.parametrize('traced', [True, False])
def test_integrals_evaluate_the_powers_at_every_node(traced):
    sol = OptimizeResult(t=np.array([0.0, 2.0]), y=np.array([[1.0, 3.0]]))
    if traced:
        sol.energy_trace = (np.array([0.0, 0.5, 2.0]), np.array([[1.0, 1.5, 3.0]]))
    nodes_t, nodes_y = sol.energy_trace if traced else (sol.t, sol.y)
    s, seen = _fake_solver(sol)
    out = s._compute_step_energy_integrals()
    assert [t for t, _ in seen] == list(nodes_t)
    assert all(np.array_equal(y, nodes_y[:, i]) for i, (_, y) in enumerate(seen))
    t_int = _trap(nodes_t, nodes_t) * es.SECS_PER_YEAR
    keys = ['F_int', 'F_cmb', 'Q_radio', 'Q_tidal', 'Q_radio_cons', 'Q_tidal_cons']
    for k, key in enumerate(keys + ['solver_residual'], start=1):
        assert out[key] == pytest.approx(k * t_int, rel=1e-12)
    assert (s.state._pb_cache_hits, s.state._pb_cache_misses) == (3, 4)


@pytest.mark.unit
def test_a_trace_short_of_the_steps_warns(monkeypatch, caplog):
    monkeypatch.setattr(
        es.EntropySolver, '_energy_trace', staticmethod(lambda nodes, t, y: (t, y))
    )
    with caplog.at_level('WARNING', logger=es.logger.name):
        res = _solve()
    assert any(f'2 nodes for {res.cvode_nst} CVODE steps' in r.message for r in caplog.records)


def _heated_bower_call(tidal_rate=None, **source):
    """Solve a 500-yr bower2018 grey-body call with one heat source; check the budget.

    ``tidal_rate(t)`` [W/kg] replaces the constant tidal rate at every RHS evaluation.
    """
    from .test_entropy_verification import EOS_DIR, TestCvodeEnergyOutputGrid

    if not EOS_DIR.exists():
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}')
    s = TestCvodeEnergyOutputGrid._build_greybody_solver(
        'cvode', n_out=2, core_bc='bower2018', **source
    )
    s.parameters.solver.end_time = 500.0
    if tidal_rate is not None:
        update = s.state.update

        def timed_update(entropy, time, **kw):
            s.state._tidal_array = [tidal_rate(float(time))]
            return update(entropy, time, **kw)

        s.state.update = timed_update
    s.solve()
    assert (s.state._pb_cache_hits, s.state._pb_cache_misses) == (0, 0)
    d, t = s._solution.energy_integrals, s._solution.t
    mass = float(np.dot(np.ravel(s.evaluator.mesh.staggered_effective_density), s._volume_flat))
    # The heat the RHS adds follows the live cell mass, so the budget closes with Q_radio
    # and Q_tidal (8e-5 measured); without the source it is off by a factor of about 6.
    heat = d['Q_radio'] + d['Q_tidal']
    assert abs(heat) > 1.0 * abs(d['state_heat'])
    assert d['F_int'] + d['F_cmb'] + heat == pytest.approx(d['state_heat'], rel=5e-4)
    assert abs(d['solver_residual']) < 1e-9 * abs(d['F_int'])
    return d, float(t[0]), float(t[-1]), mass


@pytest.mark.smoke
def test_radiogenic_energy_matches_the_decay_integral():
    """A source decaying by 1.2 percent over the call integrates to the analytic value."""
    from aragog.parser import _Radionuclide

    iso = _Radionuclide(
        'X',
        t0_years=0.0,
        abundance=1.0,
        concentration=1e3,
        heat_production=1.0,
        half_life_years=3e4,
    )
    d, t0, t1, mass = _heated_bower_call(radionuclides=[iso])
    k = np.log(2) / 3e4
    exact = mass * 1e-3 / k * (np.exp(-k * t0) - np.exp(-k * t1)) * es.SECS_PER_YEAR
    assert d['Q_radio_cons'] == pytest.approx(exact, rel=1e-8)
    assert d['Q_tidal'] == d['Q_tidal_cons'] == 0.0


@pytest.mark.smoke
def test_tidal_energy_follows_a_time_varying_rate():
    """A 70-yr oscillation in the tidal rate integrates to the analytic value over the nodes.

    The trapezoid over the call's two endpoints misses it by 19 percent.
    """
    w = 2.0 * np.pi / 70.0
    d, t0, t1, mass = _heated_bower_call(
        tidal=1e-3, tidal_rate=lambda t: 1e-3 * (1 + 0.5 * np.sin(w * t))
    )
    exact = (
        mass
        * 1e-3
        * ((t1 - t0) + 0.5 * (np.cos(w * t0) - np.cos(w * t1)) / w)
        * es.SECS_PER_YEAR
    )
    assert d['Q_tidal_cons'] == pytest.approx(exact, rel=1e-5)
    assert d['Q_radio'] == d['Q_radio_cons'] == 0.0


@pytest.mark.unit
def test_a_one_node_trace_integrates_to_zero():
    sol = OptimizeResult(t=np.array([0.0, 2.0]), y=np.ones((1, 2)))
    sol.energy_trace = (np.array([0.0]), np.ones((1, 1)))
    out = _fake_solver(sol)[0]._compute_step_energy_integrals()
    assert out['F_int'] == out['solver_residual'] == 0.0


@pytest.mark.unit
def test_a_failed_call_does_not_warn_about_the_trace(monkeypatch, caplog):
    monkeypatch.setattr(
        es.EntropySolver, '_energy_trace', staticmethod(lambda nodes, t, y: (t, y))
    )
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._core_bc, s._cvode_output_points, s._max_steps = 'quasi_steady', 2, 200
    with caplog.at_level('WARNING', logger=es.logger.name):
        res = s._solve_cvode(
            start_time=0.0,
            end_time=1.0,
            y0=np.array([0.0]),
            atol=1e-14,
            rtol=1e-10,
            max_step=0.01,
            rhs=lambda t, y: np.array([-AMP * _pulse(t)]),
        )
    assert res.cvode_flag < 0 and res.cvode_nst >= 200
    assert not any('CVODE steps' in r.message for r in caplog.records)


def _const_properties_solver(max_steps=100000):
    """10-node const-properties solver on CVODE; no EOS tables, one solve takes ~10 ms."""
    from .test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

    p = _build_const_properties_parameters(n_nodes=10, end_time=5.0)
    p.energy.solver_method, p.solver.max_steps = 'cvode', max_steps
    s = es.EntropySolver(p, entropy_eos=None)
    s.initialize()
    s.set_initial_entropy(3050.0)
    return s


_FIELDS = {
    'F_int': 'step_dE_F_int_J',
    'F_cmb': 'step_dE_F_cmb_J',
    'F_cmb_step_avg': 'F_cmb',
    'Q_radio': 'step_dE_Q_radio_J',
    'Q_tidal': 'step_dE_Q_tidal_J',
    'Q_radio_cons': 'step_dE_Q_radio_cons_J',
    'Q_tidal_cons': 'step_dE_Q_tidal_cons_J',
    'solver_residual': 'step_solver_residual_J',
    'state_heat': 'step_dE_state_heat_J',
}


@pytest.mark.unit
def test_get_state_reports_the_integrals_of_one_solve(monkeypatch):
    values = {key: float(i + 1) for i, key in enumerate(_FIELDS)}
    traces = []

    def integrals(self):
        traces.append(self._solution.energy_trace[0].copy())
        return dict(values)

    monkeypatch.setattr(es.EntropySolver, '_compute_step_energy_integrals', integrals)
    s = _const_properties_solver()
    s.solve()
    first, second = s.get_state(), s.get_state()
    assert len(traces) == 1
    np.testing.assert_array_equal(traces[0][[0, -1]], [0.0, 5.0])
    for key, field in _FIELDS.items():
        assert getattr(first, field) == getattr(second, field) == values[key]


@pytest.mark.unit
def test_a_failed_solve_reports_zero_integrals(monkeypatch):
    calls = []
    monkeypatch.setattr(
        es.EntropySolver, '_compute_step_energy_integrals', lambda self: calls.append(1) or {}
    )
    s = _const_properties_solver(max_steps=1)
    s.solve()
    out = s.get_state()
    assert s.stop_early and not calls
    assert [getattr(out, f) for k, f in _FIELDS.items() if k != 'F_cmb_step_avg'] == [0.0] * 8
