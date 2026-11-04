"""Sub-grid lid on a constant-property column: the explicit mode, its closed forms and its
refusals."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad

from tests.conftest import EOS_DIR, entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import _build, needs_eos

T_REF, S_REF, RHO, CP, K = 1600.0, 3000.0, 3300.0, 1250.0, 3.0
T_SURF = 300.0  # surface temperature of outer boundary 5 [K]

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


def _params(closure='lid', rheology=True, lid=True, outer_bc=1):
    from aragog.parser import SolidRheologyParams

    p = _build(
        core_bc='quasi_steady',
        outer_bc=outer_bc,
        inner_bc=2,
        n_nodes=20,
        end_time=1.0,
        outer_bc_value=T_SURF,
    )
    p.boundary_conditions.subgrid_lid = lid
    p.boundary_conditions.lid_handover_cells = 1
    p.boundary_conditions.normalize()
    pm = p.phase_mixed
    pm.const_properties = True
    pm.const_T_ref, pm.const_S_ref = T_REF, S_REF
    pm.const_rho, pm.const_Cp, pm.const_cond = RHO, CP, K
    if rheology:
        p.phase_solid.rheology = SolidRheologyParams(enabled=True, stress_closure_mode=closure)
    return p


def _solver(eos=None, **kw):
    from aragog.solver.entropy_solver import EntropySolver

    s = EntropySolver(_params(**kw), entropy_eos=eos)
    s.initialize()
    return s


def _T(S):
    return T_REF * np.exp((np.asarray(S, dtype=float) - S_REF) / CP)


def test_constant_property_lid_sets_up_with_the_lid_closure():
    """No entropy EOS, the solid rheology and the 'lid' closure: the lid sets up in the
    constant-property mode."""
    s = _solver()
    assert s._lid_on and s._lid_const


@needs_eos
def test_constant_property_lid_is_refused_with_an_entropy_eos(eos):
    """An attached EOS with constant properties would give the lid other physics than the
    column: refused by name."""
    with pytest.raises(ValueError, match='constant-property lid needs entropy_eos=None'):
        _solver(eos=eos)


@pytest.mark.parametrize(
    ('kw', 'match'),
    [({'rheology': False}, 'solid rheology'), ({'closure': 'local'}, "closure 'lid'")],
    ids=['no-rheology', 'local'],
)
def test_constant_property_lid_is_refused_outside_its_mode(kw, match):
    """Without the solid rheology or with the 'local' closure the mode has no lid base: refused
    by name; the same column without the lid sets up."""
    with pytest.raises(ValueError, match=match):
        _solver(**kw)
    assert not _solver(lid=False, **kw)._lid_on


def test_lid_without_tables_or_constant_properties_is_refused():
    """A lid on neither the entropy tables nor constant properties stops at setup by name."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _params()
    p.phase_mixed.const_properties = False
    with pytest.raises(ValueError, match='subgrid_lid needs the entropy EOS tables'):
        EntropySolver(p, entropy_eos=None).initialize()


def test_constant_property_lid_refuses_the_jax_rhs():
    """The JAX right-hand side has no constant-property mode: the lid data refuses by name."""
    with pytest.raises(ValueError, match='NumPy right-hand side'):
        _solver().jax_lid_params()


def test_constant_property_lid_takes_the_column_conductivity():
    """One conductivity per lid evaluation: the column's const_cond."""
    assert _solver()._lid_k == K


@settings(max_examples=40, deadline=None)
@given(a=st.floats(1500.0, 4500.0), b=st.floats(1500.0, 4500.0))
def test_constant_property_heat_content_is_the_integral_of_rho_t(a, b):
    """W(b) - W(a) equals the integral of rho T dS, and W = rho Cp T (zero at T = 0)."""
    s = _solver()
    W = s._lid_W([a, b])
    ref = quad(lambda S: RHO * float(_T(S)), a, b, epsrel=1e-13)[0]
    assert W[1] - W[0] == pytest.approx(ref, rel=1e-11, abs=1e-6 * RHO * CP)
    np.testing.assert_allclose(W, RHO * CP * _T([a, b]), rtol=1e-14)


@settings(max_examples=40, deadline=None)
@given(T=st.floats(1.0, 6000.0))
def test_constant_property_base_entropy_round_trips(T):
    """S(T) and T(S) are inverse: the lid-base entropy of a temperature gives that
    temperature back."""
    s = _solver()
    S = s._lid_props.entropy_at_temperature(s._lid_P, T)
    assert float(s._lid_props.temperature(s._lid_P, S)) == pytest.approx(T, rel=1e-13)


@pytest.mark.parametrize('width', [0.01, 0.5, 5.0])
@pytest.mark.parametrize('T_rh', [900.0, 1500.0, 2500.0])
def test_constant_property_lid_base_is_the_lower_of_the_isotherm_and_the_fluid(width, T_rh):
    """T_b = min(T_rh, T_f) for any transition width: the mode has no solidus and no melt."""
    s = _solver()
    s.state.phase_staggered._phi_width = width
    S_f = 3000.0
    T_f = float(_T(S_f))
    s.state._lid_state = {'T_lid_iso': np.asarray(T_rh)}
    base = s._lid_base_state(T_f, 0.0, S_f)
    assert base['T_b'] == pytest.approx(min(T_rh, T_f), rel=1e-13)
    assert base['rho'] == RHO


@settings(max_examples=30, deadline=None)
@given(
    S_f=st.floats(2600.0, 3400.0),
    frac=st.floats(1e-3, 0.8),
    m=st.floats(0.05, 0.95),
)
def test_constant_property_merge_conserves_energy(S_f, frac, m):
    """The hand-over merge keeps V_N W(S') = V_N W(S_f) + A (E - W(S_f) delta) to round-off."""
    s = _solver()
    S = np.full(s._n_stag, S_f)
    s.set_initial_entropy(S)
    y = np.array(s._S0, dtype=float)
    delta = frac * s._lid_h_top
    T_f = float(_T(S_f))
    E = delta * RHO * CP * (300.0 + m * (T_f - 300.0))
    y[-2:] = delta, E
    out, resid = s._lid_merge(y)
    V, A = float(s._volume_flat[-1]), s._lid_area
    lhs = V * float(s._lid_W(out[s._n_stag - 1]))
    rhs = V * float(s._lid_W(S_f)) + A * (E - float(s._lid_W(S_f)) * delta)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert resid <= 1e-12


def _solved(calls=None, **kw):
    """A 1-year solve of the 20-node column from 3200 J/kg/K at the bottom to 2800 at the top;
    ``calls`` collects the arguments of every lid step check."""
    s = _solver(**kw)
    s.set_initial_entropy(np.linspace(3200.0, 2800.0, s._n_stag))
    if calls is not None:
        check = s._check_lid_step
        s._check_lid_step = lambda *a: (calls.append(a), check(*a))[1]
    s.solve()
    return s


@pytest.mark.physics_invariant
@pytest.mark.parametrize('outer_bc', [1, 5])
def test_constant_property_lid_ledger_closes_over_a_solve(outer_bc):
    """With constant properties the step ledger runs on both outer boundaries: it closes to
    1e-10 of the surface power and the state heat, lid included, to 2e-3 of the budget."""
    s = _solved(outer_bc=outer_bc)
    assert s._lid_last.delta > s._lid_delta_lo
    ei = s._solution.energy_integrals
    budget = ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal']
    assert ei['F_int'] < 0.0
    assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])
    assert abs(ei['state_heat'] - budget) <= 2e-3 * abs(budget)


def test_constant_property_lid_checks_every_accepted_step():
    """The lid step check runs on the accepted states of a constant-property solve."""
    calls = []
    s = _solved(calls)
    assert len(calls) >= 2
    assert calls[-1][2] == pytest.approx(float(s._solution.t[-1]))


@settings(max_examples=60, deadline=None)
@given(
    F_b=st.floats(1e-6, 1e4),
    T_b=st.floats(150.0, 3000.0),
    T_surf=st.floats(150.0, 3000.0),
    eps=st.floats(0.1, 1.0),
)
def test_fixed_surface_existence_weight(F_b, T_b, T_surf, eps):
    """With a fixed surface temperature the steady crust is ``k (T_b - T_surf) / F_b``, with
    no lid when the base is not hotter than the surface; the grey-body values do not enter."""
    from aragog import surface_lid as sl

    w = sl.existence_weight(F_b, T_b, K, eps, 255.0, 500.0, T_surf=T_surf)
    if T_b <= T_surf:
        assert w == 0.0
        return
    d_eq = K * (T_b - T_surf) / F_b
    ref = 0.5 * (1.0 + np.tanh(np.log10(d_eq / 500.0) / sl.LID_EXIST_WIDTH))
    assert w == (pytest.approx(ref, rel=1e-12) if ref > sl.LID_WEIGHT_FLOOR else 0.0)
    assert w == sl.existence_weight(F_b, T_b, K, 1.0, 100.0, 500.0, T_surf=T_surf)


def _bc5(lid=True):
    s = _solver(outer_bc=5, lid=lid)
    s.set_initial_entropy(np.linspace(3200.0, 2800.0, s._n_stag))
    return s


def test_fixed_surface_lid_starts_and_stays_at_the_surface_temperature():
    """Under outer boundary 5 the phantom lid has its surface at T_surf, and so has every lid
    profile the right-hand side solves."""
    s = _bc5()
    S_top = float(s._S0[s._n_stag - 1])
    T_b = s._lid_base_state(float(_T(S_top)), 0.0, S_top)['T_b']
    d = s._lid_delta_lo
    assert s._S0[-2] == d
    heat = d * float(s._lid_hv.profile_heat(0.0, T_SURF, T_b))
    assert s._S0[-1] == pytest.approx(heat, rel=1e-12)
    s._dSdt_single(0.0, s._S0)
    assert s._lid_last.surf.T_s == T_SURF
    y = s._S0.copy()
    y[-1] *= 1.05
    s._dSdt_single(0.0, y)
    assert s._lid_last.surf.T_s == T_SURF
    assert s._lid_last.surf.s != pytest.approx(0.0, abs=1e-6)


def test_fixed_surface_lid_weight_reads_the_surface_temperature(monkeypatch):
    """The existence weight of the right-hand side takes T_surf under outer boundary 5."""
    from aragog import surface_lid as sl

    s, seen, real = _bc5(), [], sl.existence_weight

    def spy(*args, **kw):
        seen.append(kw.get('T_surf'))
        return real(*args, **kw)

    monkeypatch.setattr(sl, 'existence_weight', spy)
    s._dSdt_single(0.0, s._S0)
    assert seen and set(seen) == {T_SURF}


def test_constant_property_lid_heat_is_the_heat_of_the_column():
    """The heat table of the lid is rho cp T, and at the temperature of an entropy it equals
    the column heat W(S)."""
    s = _bc5()
    T = np.array([300.0, 1000.0, 1900.0])
    np.testing.assert_allclose(s._lid_hv.H(T), RHO * CP * T, rtol=1e-14)
    S = np.array([2800.0, 3000.0, 3200.0])
    np.testing.assert_allclose(s._lid_hv.H(_T(S)), s._lid_W(S), rtol=1e-14)


def test_fixed_surface_lid_blends_the_grid_flux_of_the_surface_face(monkeypatch):
    """F_grid of the lid is the outer boundary 5 flux of the top face, the flux of the same
    column without the lid."""
    from aragog import surface_lid as sl

    seen = {}
    rates = sl.lid_rates
    monkeypatch.setattr(sl, 'lid_rates', lambda **kw: (seen.update(kw), rates(**kw))[1])
    s, off = _bc5(), _bc5(lid=False)
    s._dSdt_single(0.0, s._S0)
    off._dSdt_single(0.0, off._S0)
    assert seen['F_grid'] == float(np.asarray(off.state._heat_flux).ravel()[-1])
    assert seen['F_grid'] > 0.0


@needs_eos
def test_fixed_surface_lid_refuses_the_jax_rhs(eos):
    """The JAX lid has the grey-body surface only: under outer boundary 5 the lid data refuses
    by name, also with an entropy EOS."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=5, inner_bc=2, n_nodes=10, end_time=1.0)
    p.boundary_conditions.outer_boundary_value = 1000.0  # inside the test tables
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    with pytest.raises(ValueError, match='fixed surface temperature'):
        s.jax_lid_params()


@pytest.mark.parametrize('core_bc', ['energy_balance', 'core_module'])
def test_constant_property_lid_is_refused_on_another_core(core_bc):
    """A trial state of the energy-balance core gives the lid a NaN base; a core mode other than
    quasi_steady stops at setup by name."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _params()
    p.boundary_conditions.core_bc = core_bc
    with pytest.raises(
        ValueError, match="constant-property lid needs core_bc = 'quasi_steady'"
    ):
        EntropySolver(p, entropy_eos=None).initialize()


def test_constant_property_heated_mass_is_rho_v():
    """Internal heating enters the ledger with the mass rho V of the constant density; the lid
    holds about 1e-5 of the volume, so a mass without it misses the tolerance."""
    from aragog.solver import SECS_PER_YEAR
    from aragog.solver.entropy_solver import EntropySolver

    p = _params()
    p.energy.tidal, p.energy.tidal_array = True, np.array([1.0e-10])
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    s.set_initial_entropy(np.linspace(3200.0, 2800.0, s._n_stag))
    s.solve()
    t = s._solution.t[-1] - s._solution.t[0]
    seconds = float(s._solution.energy_integrals['Q_tidal']) / (
        RHO * float(np.sum(s._volume_flat)) * 1.0e-10
    )
    assert t > 0.0
    assert seconds == pytest.approx(t * SECS_PER_YEAR, rel=1e-12)


def test_constant_property_lid_temperature_is_the_column_temperature():
    """The lid and the column read one T(S) in this mode."""
    s = _solved()
    S = s._solution.y[: s._n_stag, -1]
    s.dSdt(s._solution.t[-1], s._solution.y[:, -1])
    T = np.asarray(s.state.phase_staggered.temperature()).ravel()
    np.testing.assert_array_equal(np.asarray(s._lid_props.temperature(None, S)).ravel(), T)


def test_constant_property_lid_hands_over_at_once_under_a_thick_closure_lid():
    """With a stiff solid the isotherm of the closure lies below the hand-over face at the start:
    the lid hands over at the start and its states are zero."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _params(outer_bc=5)
    p.mesh.inner_radius = 6071000.0  # a shell 300 km thick, as the tutorial
    p.phase_solid.viscosity = 1.0e21
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    s.set_initial_entropy(np.linspace(3200.0, 2800.0, s._n_stag))
    s.solve()
    assert s._lid_handover_time == pytest.approx(0.0, abs=1.0)
    assert s._solution.y[-2, -1] == 0.0 and s._solution.y[-1, -1] == 0.0
    assert not s._lid_active


@pytest.mark.parametrize(
    ('F_b', 'T_b', 'w'), [(100.0, T_SURF, 0.0), (-1.0, T_SURF, 0.0), (-1.0, T_SURF + 1.0, 1.0)]
)
def test_fixed_surface_existence_weight_edges(F_b, T_b, w):
    """A base at the surface temperature carries no lid at any flux; a hotter base with no flux
    from below carries a full one."""
    from aragog import surface_lid as sl

    assert sl.existence_weight(F_b, T_b, K, 1.0, 255.0, 500.0, T_surf=T_SURF) == w


@needs_eos
def test_table_lid_runs_under_a_fixed_surface_temperature(eos):
    """On the tables under outer boundary 5 the lid surface stays at the boundary temperature and
    the ledger closes, with the base at the solidus without the rheology."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(
        core_bc='quasi_steady',
        outer_bc=5,
        inner_bc=2,
        n_nodes=10,
        end_time=1.0,
        outer_bc_value=1000.0,
    )
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_handover_cells = 1
    p.boundary_conditions.normalize()
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(np.full(s._n_stag, 3050.0))
    s.solve()
    ev, ei = s._lid_last, s._solution.energy_integrals
    assert ev.delta > s._lid_delta_lo
    assert ev.surf.T_s == 1000.0
    assert ev.T_b == pytest.approx(s._lid_T_sol, rel=1e-12)
    budget = ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal']
    assert abs(ei['state_heat'] - budget) <= 2e-3 * abs(budget)
