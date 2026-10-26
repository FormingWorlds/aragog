"""Prescribed surface temperature (outer boundary condition 5) in the entropy solver.

The surface basic node takes the entropy of the prescribed temperature at its pressure, and its
entropy gradient is set so that the temperature gradient equals the half-cell difference
``2 (T_surf - T_stag[-1]) / delta_mesh[-1] * dxidr[-1]``. The conductive face flux is then
``k (T_stag[-1] - T_surf) / (dr / 2)`` with ``dr / 2 = delta_mesh[-1] / dxidr[-1] / 2``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.constants import Julian_year

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

K_COND, CP, RHO, T_REF, S_REF = (
    4.0,
    1000.0,
    4000.0,
    3000.0,
    3000.0,
)  # the builder's const_properties


def _solver(
    t_surf=300.0,
    *,
    profile='nearest_boundary',
    floor=0.0,
    mass_coordinates=True,
    utbl=False,
    core_bc='quasi_steady',
):
    """A const_properties solver under outer BC 5, with S falling 20 J/kg/K to the surface."""
    p = _build_const_properties_parameters(n_nodes=20)
    b = p.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, t_surf
    b.param_utbl, b.core_bc = utbl, core_bc
    p.mesh.mixing_length_profile, p.mesh.mass_coordinates = profile, mass_coordinates
    p.energy.kappah_floor = floor
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    r = np.asarray(s._r_stag_flat)
    s.set_initial_entropy(S_REF + 20.0 * (r[-1] - r) / (r[-1] - r[0]))
    return s


def _rhs(s):
    return np.asarray(s.dSdt(0.0, np.asarray(s._S0, dtype=float))).ravel()


def _half_cell(s):
    """``dr / 2`` of the surface face as the temperature-gradient expression defines it [m]."""
    m = s.evaluator.mesh
    return float(np.ravel(m.basic.delta_mesh)[-1] / np.ravel(m.dxidr)[-1]) / 2.0


def _top(s, name):
    return float(np.ravel(getattr(s.state, name))[-1])


@pytest.mark.physics_invariant
def test_the_prescribed_temperature_changes_the_rhs():
    """A hotter surface lowers the face flux; only the top cell rate changes, by the face flux
    change over the cell heat capacity: d(dS/dt) = A dF / (rho T V) per Julian year."""
    cold, hot = _solver(300.0), _solver(2000.0)
    f_cold, f_hot = _rhs(cold), _rhs(hot)
    flux_cold, flux_hot = _top(cold, 'heat_flux'), _top(hot, 'heat_flux')
    assert flux_hot < flux_cold
    m = cold.evaluator.mesh
    area, vol = float(np.ravel(m.basic.area)[-1]), float(np.ravel(m.basic.volume)[-1])
    t_top_cell = float(np.ravel(cold.state.phase_staggered.temperature())[-1])
    expected = area * (flux_cold - flux_hot) / (RHO * t_top_cell * vol) * Julian_year
    assert f_hot[-1] - f_cold[-1] == pytest.approx(expected, rel=1e-5)
    np.testing.assert_array_equal(f_cold[:-1], f_hot[:-1])


@pytest.mark.physics_invariant
@pytest.mark.parametrize('t_surf', [300.0, 2000.0, 3500.0])
def test_the_surface_face_carries_the_half_cell_conduction(t_surf):
    """With mixing length 0 at the surface and no floor, the face flux is the half-cell
    conduction, positive outward below the top cell temperature and negative above it, and
    the surface node sits at the prescribed temperature."""
    s = _solver(t_surf)
    _rhs(s)
    t_top_cell = float(np.ravel(s.state.phase_staggered.temperature())[-1])
    expected = K_COND * (t_top_cell - t_surf) / _half_cell(s)
    assert _top(s, 'heat_flux') == pytest.approx(expected, rel=1e-12)
    assert _top(s, '_jconv') == 0.0
    assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
        t_surf, rel=1e-12
    )
    assert math.copysign(1.0, _top(s, 'heat_flux')) == math.copysign(1.0, t_top_cell - t_surf)
    # A full cell or the geometric half cell gives another flux on this mass-coordinate mesh.
    assert _top(s, 'heat_flux') != pytest.approx(expected / 2.0, rel=1e-3)


@pytest.mark.physics_invariant
def test_the_half_cell_is_the_temperature_gradient_expression_on_a_mass_mesh():
    """On the mass-coordinate mesh the face distance is ``delta_mesh / dxidr / 2``, which differs
    from the geometric distance between the surface node and the top cell centre by more than
    0.1 %; on a radius mesh the two agree to round-off."""
    for mass, differs in ((True, True), (False, False)):
        s = _solver(300.0, mass_coordinates=mass)
        _rhs(s)
        m = s.evaluator.mesh
        geometric = float(np.ravel(m.basic.radii)[-1] - np.ravel(m.staggered.radii)[-1])
        t_top_cell = float(np.ravel(s.state.phase_staggered.temperature())[-1])
        assert _top(s, 'heat_flux') == pytest.approx(
            K_COND * (t_top_cell - 300.0) / _half_cell(s), rel=1e-12
        )
        assert (abs(_half_cell(s) / geometric - 1.0) > 1e-3) is differs


@pytest.mark.physics_invariant
def test_the_convective_face_flux_uses_the_matched_gradient():
    """With a constant mixing length and the kappa_h floor the surface face also carries a
    convective flux, built from the same entropy gradient that gives the half-cell conduction:
    dS/dr = (Cp / T_surf) (dT/dr - dT/dP|_S dP/dr)."""
    s = _solver(300.0, profile='constant', floor=10.0)
    _rhs(s)
    st_ = s.state
    t_top_cell = float(np.ravel(st_.phase_staggered.temperature())[-1])
    g_t = (300.0 - t_top_cell) / _half_cell(s)
    dtdps = float(np.ravel(st_.phase_basic.dTdPs())[-1])
    dpdr = float(np.ravel(st_._dP_dr_basic)[-1])
    assert float(st_._dSdr[-1]) == pytest.approx((CP / 300.0) * (g_t - dtdps * dpdr), rel=1e-10)
    assert _top(s, '_jcond') == pytest.approx(-K_COND * g_t, rel=1e-12)
    assert _top(s, '_jconv') > 0.0
    assert _top(s, 'heat_flux') == pytest.approx(
        _top(s, '_jcond') + _top(s, '_jconv'), rel=1e-12
    )


@pytest.mark.parametrize(
    ('kw', 'match'),
    [
        ({'utbl': True}, 'param_utbl'),
        ({'core_bc': 'gradient'}, 'gradient'),
    ],
)
def test_bc5_refuses_the_settings_it_cannot_honour(kw, match):
    """The UTBL correction and the gradient core have no meaning with a prescribed surface
    temperature; the solver refuses them at setup."""
    with pytest.raises(ValueError, match=match):
        _solver(300.0, **kw)


_BAD = st.one_of(
    st.booleans(),
    st.just(float('nan')),
    st.just(float('inf')),
    st.just(float('-inf')),
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=5),
)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(value=_BAD)
def test_bc5_refuses_a_surface_temperature_outside_its_domain(value):
    """A TOML true, a string, NaN, an infinity, zero or a negative value is refused with an
    error that names the setting."""
    with pytest.raises(ValueError, match='outer_boundary_value'):
        _solver(value)


@pytest.mark.physics_invariant
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(t_surf=st.floats(min_value=1.0, max_value=1.0e4))
def test_bc5_inside_its_domain_gives_a_finite_rhs_at_the_prescribed_node(t_surf):
    """Any positive finite surface temperature gives a finite rate in every cell, the surface
    node at that temperature, and the analytic const_properties entropy at the node."""
    s = _solver(t_surf)
    f = _rhs(s)
    assert np.all(np.isfinite(f))
    assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
        t_surf, rel=1e-12
    )
    s_surf = S_REF + CP * math.log(t_surf / T_REF)
    assert float(np.ravel(s.state._entropy_basic)[-1]) == pytest.approx(
        s_surf, rel=1e-12, abs=1e-9
    )


@pytest.mark.physics_invariant
def test_a_changed_surface_temperature_takes_effect_on_the_next_call():
    """A caller that changes outer_boundary_value between calls, as the prescribed-flux mode
    allows, gets the new surface temperature at the node and in the face flux."""
    s = _solver(300.0)
    _rhs(s)
    s.evaluator.boundary_conditions._settings.outer_boundary_value = 2000.0
    _rhs(s)
    t_top_cell = float(np.ravel(s.state.phase_staggered.temperature())[-1])
    assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
        2000.0, rel=1e-12
    )
    expected = K_COND * (t_top_cell - 2000.0) / _half_cell(s)
    assert _top(s, 'heat_flux') == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match='outer_boundary_value'):
        s.evaluator.boundary_conditions._settings.outer_boundary_value = float('nan')
        _rhs(s)


@pytest.mark.physics_invariant
def test_the_surface_gradient_uses_the_heat_capacity_floor_of_the_conduction():
    """Below the 100 J/kg/K floor of the conductive flux the matched gradient uses the same
    floored heat capacity, so the face flux stays the half-cell conduction; the unfloored value
    50 J/kg/K would double the face flux."""
    p = _build_const_properties_parameters(n_nodes=20)
    b = p.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, 300.0
    p.phase_mixed.const_Cp = 50.0
    p.energy.kappah_floor, p.energy.convection = 0.0, False
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    r = np.asarray(s._r_stag_flat)
    s.set_initial_entropy(S_REF + 1.0 * (r[-1] - r) / (r[-1] - r[0]))
    _rhs(s)
    t_top_cell = float(np.ravel(s.state.phase_staggered.temperature())[-1])
    expected = K_COND * (t_top_cell - 300.0) / _half_cell(s)
    assert _top(s, 'heat_flux') == pytest.approx(expected, rel=1e-12)
    assert _top(s, 'heat_flux') != pytest.approx(2.0 * expected, rel=1e-3)
