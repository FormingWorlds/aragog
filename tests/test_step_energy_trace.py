"""Per-call energy integrals over CVODE's accepted internal steps (``_cvode_solve_stepwise``)."""

from __future__ import annotations

from types import SimpleNamespace

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


def _cvode(**extra):
    # A time-only pulse has no state signature, so a step cap makes CVODE step through it.
    return es._scikits_cvode(
        _rhs,
        old_api=False,
        rtol=1e-10,
        atol=1e-14,
        lmm_type='BDF',
        max_steps=100000,
        max_step_size=0.01,
        **extra,
    )


def _stub():
    """The two attributes ``_cvode_solve_stepwise`` reads: the power is y' itself."""
    return SimpleNamespace(
        entropy_eos=object(),
        _step_powers=lambda t, y: np.array([-AMP * _pulse(t)]),
    )


def _trap(t, p):
    return float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(t)))


def test_pulse_between_outputs_is_integrated():
    tspan = np.array([0.0, 1.0])
    sol, (t, P) = es.EntropySolver._cvode_solve_stepwise(_stub(), _cvode(), tspan, [0.0])
    exact = -AMP * np.sqrt(np.pi) * WIDTH
    assert sol.flag == 0 and t[0] == 0.0 and t[-1] == 1.0
    assert abs(_trap(t, P[:, 0]) / exact - 1.0) < 1e-3
    assert abs(_trap(tspan, np.array([_pulse(0.0), _pulse(1.0)])) * -AMP / exact) < 1e-6


def test_outputs_and_steps_equal_normal_mode():
    tspan = np.linspace(0.0, 1.0, 9) ** 2
    ref_solver = _cvode()
    ref = ref_solver.solve(tspan, np.array([0.0]))
    solver = _cvode()
    sol, _ = es.EntropySolver._cvode_solve_stepwise(_stub(), solver, tspan, [0.0])
    assert np.array_equal(sol.values.t, np.asarray(ref.values.t))
    assert np.array_equal(sol.values.y, np.asarray(ref.values.y))
    assert solver.get_info()['NumSteps'] == ref_solver.get_info()['NumSteps']


def test_trace_ends_at_the_root():
    half = -0.5 * AMP * np.sqrt(np.pi) * WIDTH

    def root(t, y, g):
        g[0] = y[0] - half

    sol, (t, P) = es.EntropySolver._cvode_solve_stepwise(
        _stub(), _cvode(rootfn=root, nr_rootfns=1), np.array([0.0, 1.0]), [0.0]
    )
    assert sol.flag == 2 and t[-1] == pytest.approx(T_PULSE, abs=1e-6)
    assert t[-1] == sol.roots.t[0]
    assert abs(_trap(t, P[:, 0]) / half - 1.0) < 1e-3


def test_no_trace_without_eos():
    stub = SimpleNamespace(entropy_eos=None, _step_powers=None)
    sol, trace = es.EntropySolver._cvode_solve_stepwise(
        stub, _cvode(), np.array([0.0, 1.0]), [0.0]
    )
    assert trace is None and sol.flag == 0 and sol.values.t[-1] == 1.0


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
