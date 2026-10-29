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


def _solver(eos, lid=True, end=1.0, core_bc='quasi_steady', S0=3050.0, **lidkw):
    from aragog.solver.entropy_solver import EntropySolver

    inner = 2 if core_bc == 'quasi_steady' else 1
    p = _build(core_bc=core_bc, outer_bc=1, inner_bc=inner, n_nodes=10, end_time=end)
    p.boundary_conditions.subgrid_lid = lid
    for k, v in lidkw.items():
        setattr(p.boundary_conditions, k, v)
    p.boundary_conditions.normalize()
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(S0)
    return s


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
def test_ledger_per_call_closes(eos, frac, dT, S_top, S_bot):
    """The per-call energy identity closes on random valid states: the lid at any thickness up
    to the growth stop, its mean off the steady value, any entropy profile (so the lid hidden,
    partly hidden or full)."""
    s = _solver(eos)
    delta = max(frac * s._lid_h_top, s._lid_delta_lo)
    y = s._S0.copy()
    y[: s._n_stag] = np.linspace(S_bot, S_top, s._n_stag)
    s._dSdt_single(0.0, y)
    ev = s._lid_last
    T_m = float(np.clip(ev.T_m + dT, 300.0, ev.T_b - 1.0))
    y[-2:] = (delta, sl.lid_energy(ev.rho, delta, ev.h_b, ev.cp, ev.T_b, T_m))
    pw = s._step_powers(0.0, y)
    scale = max(abs(pw[0]), abs(pw[1]), s._lid_area * abs(s._lid_last.rates.dE), 1.0)
    assert abs(pw[-1]) <= 1e-10 * scale
    assert 0.0 <= s._lid_last.rates.w <= 1.0


@pytest.mark.physics_invariant
def test_ledger_over_a_solve(eos):
    """Over a solve the step ledger closes to round-off and the state heat to the stated
    quadrature accuracy; the lid grows as Neumann with the run's Stefan number within 2 %."""
    from scipy.optimize import brentq
    from scipy.special import erf

    s = _solver(eos)
    s.solve()
    ei = s._solution.energy_integrals
    budget = ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal']
    assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])
    assert abs(ei['state_heat'] - budget) <= 1e-2 * abs(budget)
    ev = s._lid_last
    St = ev.cp * (ev.T_b - ev.surf.T_s) / (ev.h_f - ev.h_b)
    lam = brentq(lambda x: x * np.exp(x * x) * erf(x) - St / np.sqrt(np.pi), 1e-9, 10.0)
    t = (s._solution.t[-1] - s._solution.t[0]) * 3.15576e7
    neumann = 2.0 * lam * np.sqrt(s._lid_k / (ev.rho * ev.cp) * t)
    assert ev.delta == pytest.approx(neumann, rel=0.02)
    assert ev.delta > 5.0 * s._lid_delta_lo


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


def test_jax_factory_refused_with_the_lid(eos):
    """The lid has no JAX right-hand side: a registered factory with the lid on is refused."""
    s = _solver(eos)
    s.parameters.energy.solver_method = 'cvode'
    s.parameters.energy.use_jax_jacobian = True
    s._jax_cvode_factory = lambda *a, **k: pytest.fail('factory must not be called')
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    with pytest.raises(ValueError, match='no JAX right-hand side'):
        s.solve()
