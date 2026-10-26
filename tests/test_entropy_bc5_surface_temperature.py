"""Prescribed surface temperature (outer boundary condition 5) in the entropy solver.

The surface basic node takes the entropy of the prescribed temperature at its pressure, and its
entropy gradient is set so that the temperature gradient equals the half-cell difference
``2 (T_surf - T_stag[-1]) / delta_mesh[-1] * dxidr[-1]``. The conductive face flux is then
``k (T_stag[-1] - T_surf) / (dr / 2)`` with ``dr / 2 = delta_mesh[-1] / dxidr[-1] / 2``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.constants import Julian_year

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

K_COND, CP, RHO, T_REF, S_REF = 4.0, 1000.0, 4000.0, 3000.0, 3000.0  # builder values


def _solver(
    t_surf=300.0,
    *,
    profile='nearest_boundary',
    fraction=0.25,
    eddy=1.0,
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
    p.mesh.mixing_length_constant_fraction = fraction
    p.energy.kappah_floor, p.energy.eddy_diffusivity_thermal = floor, eddy
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


@pytest.mark.physics_invariant
def test_the_half_cell_is_the_temperature_gradient_expression_on_a_mass_mesh():
    """On the mass-coordinate mesh the face distance is ``delta_mesh / dxidr / 2``, which
    differs from the geometric distance between the surface node and the top cell centre by
    more than 0.1 %; on a radius mesh the two agree to round-off."""
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


@pytest.mark.parametrize('fraction', [0.25, -0.1])
def test_bc5_refuses_a_constant_mixing_length(fraction):
    """A constant mixing length that is not 0 puts an eddy flux on a surface face held at its
    temperature, driven by the steep half-cell gradient; the setup refuses it with BC 5, and
    nearest_boundary, mixing length 0 at the surface, is accepted."""
    with pytest.raises(Exception) as err:
        _solver(300.0, profile='constant', fraction=fraction)
    assert err.type is ValueError, repr(err.value)
    assert "mixing_length_profile = 'constant'" in str(err.value)
    assert _solver(300.0, profile='nearest_boundary').surface_temperature_bc[1] == 300.0


@pytest.mark.parametrize(
    'kw',
    [{'profile': 'constant', 'fraction': 0.0}, {'eddy': 0.0}],
    ids=['fraction-0', 'eddy-0'],
)
def test_bc5_accepts_the_settings_without_a_surface_eddy_flux(kw):
    """A constant mixing length of 0 and eddy_diffusivity_thermal 0 leave no eddy flux at the
    surface; the setup accepts both and the surface face carries no convective flux."""
    try:
        s = _solver(300.0, **kw)
    except Exception as exc:
        pytest.fail(f'BC 5 refused {kw}: {exc!r}')
    assert np.all(np.isfinite(_rhs(s)))
    assert _top(s, '_jconv') == 0.0
    assert _top(s, 'heat_flux') > 0.0


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
    g_t = (300.0 - t_top_cell) / _half_cell(s)
    assert float(s.state._dSdr[-1]) == pytest.approx(100.0 / 300.0 * g_t, rel=1e-10)


@pytest.mark.physics_invariant
def test_the_matched_gradient_carries_the_adiabatic_term():
    """With a nonzero dP/dr the surface gradient subtracts dT/dP|_S dP/dr, so the conductive
    face flux is still -k dT/dr; without that term the flux would differ by about 0.4 %."""
    s = _solver(300.0)
    s.state._dP_dr_basic = np.full(np.ravel(s.evaluator.mesh.basic.radii).size, -4.0e4)
    _rhs(s)
    t_top_cell = float(np.ravel(s.state.phase_staggered.temperature())[-1])
    g_t = (300.0 - t_top_cell) / _half_cell(s)
    dtdr_ad = float(np.ravel(s.state.phase_basic.dTdPs())[-1]) * -4.0e4
    assert abs(dtdr_ad / g_t) > 1e-3
    assert _top(s, '_jcond') == pytest.approx(-K_COND * g_t, rel=1e-12)
    assert float(s.state._dSdr[-1]) == pytest.approx(CP / 300.0 * (g_t - dtdr_ad), rel=1e-12)


@pytest.mark.physics_invariant
def test_the_kappah_floor_does_not_act_at_the_prescribed_surface():
    """The kappa_h floor lifts the eddy diffusivity of molten cells below the convective
    threshold, but not at a surface held at a prescribed temperature: with mixing length 0
    there, the face carries only the half-cell conduction, while the floor still acts below."""
    s = _solver(300.0, floor=10.0)
    _rhs(s)
    t_top_cell = float(np.ravel(s.state.phase_staggered.temperature())[-1])
    assert _top(s, '_jconv') == 0.0
    assert _top(s, 'heat_flux') == pytest.approx(
        K_COND * (t_top_cell - 300.0) / _half_cell(s), rel=1e-12
    )
    assert float(np.ravel(s.state._eddy_diffusivity)[-2]) > 1.0


def test_bc5_refuses_conduction_off():
    """Without conduction the prescribed temperature sets no face flux under nearest_boundary,
    so the setup refuses the pair."""
    p = _build_const_properties_parameters(n_nodes=20)
    b = p.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, 300.0
    p.energy.conduction = False
    s = EntropySolver(p, entropy_eos=None)
    with pytest.raises(ValueError, match='conduction'):
        s.initialize()
    assert getattr(s.state, '_surface_temperature_bc', None) is None


def test_the_solver_gives_the_surface_entropy_temperature_and_half_cell():
    """surface_temperature_bc returns the inverted surface entropy, the surface temperature and
    the face half cell, follows a changed outer_boundary_value without a right-hand-side call,
    and is None under another outer mode."""
    s = _solver(300.0)
    s_surf, t_surf, half = s.surface_temperature_bc
    assert s_surf == pytest.approx(S_REF + CP * math.log(300.0 / T_REF), rel=1e-12)
    assert (t_surf, half) == (300.0, pytest.approx(_half_cell(s), rel=1e-15))
    s.evaluator.boundary_conditions._settings.outer_boundary_value = 2000.0
    assert s.surface_temperature_bc[:2] == (
        pytest.approx(S_REF + CP * math.log(2000.0 / T_REF), rel=1e-12),
        2000.0,
    )
    p = _build_const_properties_parameters(n_nodes=20)
    other = EntropySolver(p, entropy_eos=None)
    other.initialize()
    assert other.surface_temperature_bc is None


def test_solve_checks_a_changed_surface_temperature_before_the_first_rhs():
    """solve() inverts a changed outer_boundary_value before the integrator starts, so a value
    outside the domain stops it without a right-hand-side call."""
    s = _solver(300.0)
    s.evaluator.boundary_conditions._settings.outer_boundary_value = float('nan')

    def no_rhs(*_):
        raise AssertionError('right-hand side called')

    s._dSdt_single = no_rhs
    with pytest.raises(ValueError, match='outer_boundary_value'):
        s.solve()


@pytest.mark.parametrize(
    ('kw', 'match'),
    [
        ({'surf_dr_half': 100.0}, 'S_surf'),
        ({'S_surf': float('nan'), 'surf_dr_half': 100.0}, 'S_surf'),
        ({'S_surf': 2000.0, 'surf_dr_half': 0.0}, 'surf_dr_half'),
        ({'S_surf': 2000.0, 'surf_dr_half': -5.0}, 'surf_dr_half'),
        ({'S_surf': 2000.0, 'surf_dr_half': float('nan')}, 'surf_dr_half'),
        ({'S_surf': 2000.0, 'surf_dr_half': float('inf')}, 'surf_dr_half'),
        (
            {'S_surf': 2000.0, 'surf_dr_half': 100.0, 'outer_bc_value': float('nan')},
            'outer_bc_value',
        ),
        ({'S_surf': 2000.0, 'surf_dr_half': 100.0, 'outer_bc_value': -5.0}, 'outer_bc_value'),
        ({'S_surf': 2000.0, 'surf_dr_half': 100.0, 'param_utbl': True}, 'param_utbl'),
    ],
)
def test_boundary_params_refuse_bc5_without_its_surface_values(kw, match):
    """BoundaryParams of type 5 needs a finite surface entropy, a positive finite surface
    temperature and a finite positive half cell, and refuses the UTBL correction of the
    grey-body flux; a default or a broken value is refused."""
    pytest.importorskip('equinox')
    from aragog.jax.solver import BoundaryParams

    base = dict(emissivity=1.0, T_eq=255.0, inner_bc_type=2, inner_bc_value=0.0)
    base.update(core_density=1e4, core_heat_capacity=880.0, tfac_core_avg=1.147)
    with pytest.raises(Exception) as err:
        BoundaryParams(outer_bc_type=5, **{'outer_bc_value': 1500.0, **kw}, **base)
    assert err.type is ValueError, repr(err.value)
    assert match in str(err.value)
    assert (
        float(BoundaryParams(outer_bc_type=4, outer_bc_value=0.0, **base).surf_dr_half) == 0.0
    )


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('cp', 't_node', 't_top', 'dtdps', 'dp_dr'),
    [(50.0, 300.0, 400.0, 1e-8, -4e4), (1000.0, 1500.0, 2000.0, 0.0, 0.0)],
    ids=['heat-capacity-floor', 'no-floor'],
)
def test_the_jax_surface_gradient(cp, t_node, t_top, dtdps, dp_dr):
    """The JAX matcher gives (Cp_safe / T_node) ((T_node - T_top) / dr_half - dT/dP|_S dP/dr),
    with Cp floored at 100 J/kg/K as in its conductive flux (at Cp = 50 the gradient doubles);
    T_node is the temperature of the surface node, the matcher takes no prescribed value."""
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('equinox')
    from types import SimpleNamespace

    import aragog.jax.phase as jp

    pb = SimpleNamespace(
        heat_capacity=jnp.array([cp]), temperature=jnp.array([t_node]), dTdPs=jnp.array([dtdps])
    )
    mesh = SimpleNamespace(dP_dr_basic=jnp.array([dp_dr]))
    out = jp._matched_surface_gradient(jnp.zeros(4), 50.0, t_top, pb, mesh)
    expected = max(cp, 100.0) / t_node * ((t_node - t_top) / 50.0 - dtdps * dp_dr)
    assert float(out[-1]) == pytest.approx(expected, rel=1e-12)
    assert float(out[0]) == 0.0


@pytest.mark.parametrize(
    ('params', 'mixing_length', 'match'),
    [
        ({'conduction': 0.0}, None, 'conduction'),
        ({'conduction': 1.0}, np.array([0.0, 7.2e5, 7.2e5]), 'mixing length'),
        *[
            (
                {'conduction': 1.0, 'eddy_diff_thermal': e},
                np.zeros(3),
                'eddy_diffusivity_thermal',
            )
            for e in (-1e3, -1e-12, math.nan, math.inf)
        ],
    ],
    ids=[
        'no-conduction',
        'surface-mixing-length',
        'eddy-negative',
        'eddy-tiny-negative',
        'eddy-nan',
        'eddy-inf',
    ],
)
def test_the_jax_factory_refuses_bc5_without_a_conductive_surface(params, mixing_length, match):
    """The JAX CVODE factory refuses type 5 without conduction, with a mixing length that is
    not 0 at the surface node (as with 'constant'), or with a negative or non-finite
    eddy_diffusivity_thermal, which gives the surface node a kappa_h; each leaves the surface
    face without the half-cell conduction it prescribes."""
    pytest.importorskip('equinox')
    from types import SimpleNamespace

    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    scales = NonDimScales(state_scale=np.ones(4), t_ref=1.0)
    with pytest.raises(Exception) as err:
        build_jax_rhs_and_jacobian(
            eos_jax=None,
            phase_params=SimpleNamespace(**params),
            mesh_arrays=SimpleNamespace(mixing_length=mixing_length),
            boundary_params=SimpleNamespace(outer_bc_type=5),
            heating_array=np.zeros(4),
            scales=scales,
        )
    assert err.type is ValueError, repr(err.value)
    assert match in str(err.value)


def test_the_jax_factory_applies_the_bc5_refusals_only_to_type_5():
    """With outer_bc_type 4 the factory builds without conduction, with a mixing length that is
    not 0 at the surface node and with a negative eddy_diffusivity_thermal."""
    pytest.importorskip('equinox')
    from types import SimpleNamespace

    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    try:
        build_jax_rhs_and_jacobian(
            eos_jax=None,
            phase_params=SimpleNamespace(conduction=0.0, eddy_diff_thermal=-1e3),
            mesh_arrays=SimpleNamespace(mixing_length=np.array([0.0, 7.2e5, 7.2e5])),
            boundary_params=SimpleNamespace(outer_bc_type=4),
            heating_array=np.zeros(4),
            scales=NonDimScales(state_scale=np.ones(4), t_ref=1.0),
        )
    except Exception as exc:
        pytest.fail(f'type 4 refused: {exc!r}')


@pytest.mark.parametrize('eddy', [-1e3, -1e-12, math.nan, math.inf])
def test_bc5_refuses_a_negative_or_non_finite_eddy_diffusivity_thermal(eddy):
    """A negative eddy_diffusivity_thermal sets kappa_h to its magnitude at every node, the
    surface node too, and NaN or an infinity gives no finite kappa_h there; the setup refuses
    each with BC 5."""
    with pytest.raises(Exception) as err:
        _solver(300.0, eddy=eddy)
    assert err.type is ValueError, repr(err.value)
    assert 'eddy_diffusivity_thermal' in str(err.value)


@pytest.mark.parametrize(
    'value',
    [10**400, Fraction(1, 10**400)],
    ids=['int-1e400', 'fraction-1e-400'],
)
def test_bc5_refuses_a_real_value_that_is_not_a_finite_positive_float(value):
    """An integer too large for a float and a Fraction that rounds to 0 K are refused with the
    ValueError of the setting, not an OverflowError or a log of zero."""
    with pytest.raises(Exception) as err:
        _solver(value)
    assert err.type is ValueError, repr(err.value)
    assert 'outer_boundary_value' in str(err.value)


def test_a_fraction_surface_temperature_is_inverted_once():
    """A non-dyadic Fraction compares unequal to its float; the surface entropy is still
    inverted once at setup and not again on each right-hand side."""
    calls = []
    s = _solver(Fraction(16001, 10))
    orig = s._invert_surface_temperature
    s._invert_surface_temperature = lambda t_surf: calls.append(1) or orig(t_surf)
    for _ in range(5):
        _rhs(s)
    assert calls == []
    assert s.surface_temperature_bc[1] == pytest.approx(1600.1, rel=1e-15)


def test_bc5_takes_a_fraction_as_a_surface_temperature():
    """Any real number type is a valid surface temperature; a Fraction gives the same node
    temperature as the float, not a TypeError from the finiteness test."""
    s = _solver(Fraction(1600))
    _rhs(s)
    assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
        1600.0, rel=1e-12
    )
    assert s.surface_temperature_bc[1] == 1600.0
