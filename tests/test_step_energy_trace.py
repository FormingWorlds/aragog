"""Per-call energy quadrature over CVODE's accepted internal steps (``EntropySolver._energy_trace``)."""

from __future__ import annotations

import numpy as np
import pytest

from aragog.solver import entropy_solver as es

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(es._scikits_cvode is None, reason='scikits-odes-sundials not installed'),
]

AMP, T_PULSE, WIDTH = 3.0, 0.37, 3e-3


def _pulse(t: float) -> float:
    return np.exp(-(((t - T_PULSE) / WIDTH) ** 2))


def _rhs(t, y, ydot):
    ydot[0] = -AMP * _pulse(t)
    return 0


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


def test_pulse_between_outputs_is_integrated():
    res = _solve()
    t, y = res.energy_trace
    exact = -AMP * np.sqrt(np.pi) * WIDTH
    assert res.t.size == 2 and t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    assert np.array_equal(y[:, -1], res.y[:, -1])
    assert abs(_trap(t, -AMP * _pulse(t)) / exact - 1.0) < 1e-3
    assert abs(_trap(res.t, -AMP * _pulse(res.t)) / exact) < 1e-6


def test_trace_holds_every_output():
    res = _solve(n_out=9)
    assert np.all(np.isin(res.t, res.energy_trace[0]))


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
