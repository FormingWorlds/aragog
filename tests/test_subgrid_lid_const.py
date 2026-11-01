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

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


def _params(closure='lid', rheology=True, lid=True):
    from aragog.parser import SolidRheologyParams

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=20, end_time=1.0)
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
    s.state.lid_state = {'T_lid_iso': np.asarray(T_rh)}
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
