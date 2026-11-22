"""Unit tests for ``aragog.core.cmb_boundary_layer_flux``.

The flux must carry the sign of ``T_core - T_mantle`` (heat never flows from
the colder side), equal the Thiriet et al. (2019) boundary-layer value for a hotter
core whatever the mesh, conduct across the half cell only for a colder core (or a
layer that is not buoyant), take the critical Rayleigh number from their eq. 16 when
none is set, and stay differentiable for the JAX Jacobian.
"""

from __future__ import annotations

from unittest import mock

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.core import (
    RA_CRIT_CMB_DEFAULT,
    RA_CRIT_UPPER_LAYER,
    boundary_layer,
    check_ra_crit,
    cmb_boundary_layer_flux,
    critical_rayleigh_eq16,
)

pytestmark = pytest.mark.unit

# Bottom-cell properties: liquid magma (eta 0.1 Pa s) and solid rock (eta 1e21 Pa s), at the
# fixed critical Rayleigh number 450 (the option ra_crit_cmb = 450).
_COMMON = dict(
    ra_crit=RA_CRIT_UPPER_LAYER,
    conductivity=4.0,
    density=4000.0,
    heat_capacity=1000.0,
    expansivity=3e-5,
    gravity=10.0,
    dr_half=1e4,
)
_LIQUID = dict(_COMMON, viscosity=0.1)
_SOLID = dict(_COMMON, viscosity=1e21)


def _flux(dT, props, t_mantle=4000.0, **kw):
    return float(cmb_boundary_layer_flux(t_mantle + dT, t_mantle, **{**props, **kw}))


@pytest.mark.physics_invariant
@pytest.mark.parametrize('props', [_LIQUID, _SOLID], ids=['liquid', 'solid'])
def test_flux_sign_follows_the_temperature_contrast(props):
    """Second law at the CMB: F > 0 exactly when the core is hotter, F < 0 when colder."""
    dTs = np.array([-1000.0, -10.0, -1e-3, 0.0, 1e-3, 10.0, 1000.0])
    fluxes = np.asarray(cmb_boundary_layer_flux(4000.0 + dTs, 4000.0, **props))
    assert np.array_equal(np.sign(fluxes), np.sign(dTs))


@pytest.mark.physics_invariant
def test_convective_branch_matches_the_hand_computed_boundary_layer():
    """Liquid base, dT = 100 K: delta = (450 * 1e-6 * 0.1 / (3e-5 * 4000 * 10 * 100))^(1/3)
    = 7.2112e-3 m, so q = 4 * 100 / delta = 55468.90 W/m^2, far above conduction (0.04)."""
    assert _flux(100.0, _LIQUID) == pytest.approx(55468.90194805076, rel=1e-12)
    assert RA_CRIT_UPPER_LAYER == 450.0 and RA_CRIT_CMB_DEFAULT is None


@pytest.mark.physics_invariant
@pytest.mark.parametrize('dT', [-500.0, -5.0])
def test_a_colder_core_conducts_across_the_half_cell(dT):
    """A colder core (a stable layer) gets conduction across the half cell, for any viscosity."""
    expected = 4.0 * dT / 1e4
    assert _flux(dT, _SOLID) == pytest.approx(expected, rel=1e-12)
    assert _flux(dT, _LIQUID) == pytest.approx(expected, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('dT', [5.0, 500.0])
def test_a_hotter_core_under_a_solid_base_takes_the_boundary_layer_flux(dT):
    """Solid base: delta_BL (4.2e5 m at 5 K) exceeds the 1e4 m half cell; the flux is still the
    boundary-layer value k dT / delta_BL, far below the conduction across the half cell."""
    delta = (450.0 * 1e-6 * 1e21 / (3e-5 * 4000.0 * 10.0 * dT)) ** (1.0 / 3.0)
    assert _flux(dT, _SOLID) == pytest.approx(4.0 * dT / delta, rel=1e-12)
    assert _flux(dT, _SOLID) < 4.0 * dT / 1e4


@pytest.mark.physics_invariant
@settings(max_examples=200, deadline=None)
@given(
    dT=st.floats(1e-6, 3e3),
    eta=st.floats(1e-2, 1e24),
    dr_half=st.floats(1.0, 1e5),
)
def test_a_hotter_core_flux_does_not_depend_on_the_cell(dT, eta, dr_half):
    """For T_core > T_mantle the flux has no term in dr_half: the same layer state gives the
    same flux on every mesh (Validation V1)."""
    props = dict(_COMMON, viscosity=eta, dr_half=dr_half)
    other = dict(props, dr_half=1e4)
    assert float(cmb_boundary_layer_flux(4000.0 + dT, 4000.0, **props)) == float(
        cmb_boundary_layer_flux(4000.0 + dT, 4000.0, **other)
    )


# The layer state of a 1 Earth-mass coupled run at 5.03 Myr (1500 m CMB cell): layer properties
# at (T_core + T_m)/2, interior viscosity at T_m, mantle thickness and the core-to-top contrast.
_STATE = dict(
    conductivity=4.0,
    density=4706.5,
    heat_capacity=3002.0,
    expansivity=3.147e-5,
    viscosity=1.152e19,
    interior_viscosity=8.506e22,
    gravity=11.012,
    dr_half=750.0,
    shell_thickness=3670.5e3,
    shell_contrast=3332.0,
)


@pytest.mark.physics_invariant
def test_eq16_critical_rayleigh_number_by_hand():
    """Thiriet et al. (2019) eq. 3 and eq. 16 (0.28 Ra_i^0.21, Deschamps and Sotin 2000), eqs.
    13-14 with beta = 1/3, at the stored state: Ra_i = 1.12e7 with the interior viscosity,
    Ra_crit = 8.46, delta = 3.06 km with the mid-layer viscosity, q = 0.769 W/m^2 (against
    0.205 W/m^2 at Ra_crit 450)."""
    t_core, t_m = 4902.8, 4313.9
    k, rho, cp, alpha, eta, g = (
        _STATE[x]
        for x in (
            'conductivity',
            'density',
            'heat_capacity',
            'expansivity',
            'viscosity',
            'gravity',
        )
    )
    kappa = k / (rho * cp)
    ra_i = alpha * rho * g * 3332.0 * 3670.5e3**3 / (kappa * _STATE['interior_viscosity'])
    ra_crit = 0.28 * ra_i**0.21
    delta = (ra_crit * kappa * eta / (alpha * rho * g * (t_core - t_m))) ** (1.0 / 3.0)
    q = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=None, **_STATE))
    assert q == pytest.approx(k * (t_core - t_m) / delta, rel=1e-12)
    assert (ra_i, ra_crit, delta, q) == pytest.approx(
        (1.116e7, 8.456, 3062.0, 0.7693), rel=2e-3
    )
    q450 = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=450.0, **_STATE))
    assert q450 == pytest.approx(0.2045, rel=2e-3)


def test_eq16_needs_the_shell():
    """With no critical Rayleigh number set, the shell thickness and contrast and the interior
    viscosity are required."""
    with pytest.raises(
        ValueError, match='shell_thickness, shell_contrast and interior_viscosity'
    ):
        cmb_boundary_layer_flux(5000.0, 4000.0, **{**_LIQUID, 'ra_crit': None})
    state = {k: v for k, v in _STATE.items() if k != 'interior_viscosity'}
    with pytest.raises(
        ValueError, match='shell_thickness, shell_contrast and interior_viscosity'
    ):
        cmb_boundary_layer_flux(4902.8, 4313.9, ra_crit=None, **state)


@pytest.mark.physics_invariant
def test_eq16_rayleigh_number_takes_the_interior_viscosity():
    """Ra_i (eq. 3) takes the interior viscosity, the layer law (eq. 14) the mid-layer one: an
    8x interior viscosity scales q by 8^(0.21/3) through Ra_crit only, an 8x layer viscosity
    halves q and leaves Ra_crit alone. A fixed Ra_crit ignores the interior viscosity."""
    t_core, t_m = 4902.8, 4313.9

    def q(**kw):
        return float(cmb_boundary_layer_flux(t_core, t_m, **{**_STATE, 'ra_crit': None, **kw}))

    eta_i, eta = _STATE['interior_viscosity'], _STATE['viscosity']
    assert q(interior_viscosity=8.0 * eta_i) == pytest.approx(q() * 8.0 ** (0.07), rel=1e-12)
    assert q(viscosity=8.0 * eta) == pytest.approx(q() / 2.0, rel=1e-12)
    fixed = {**_STATE, 'ra_crit': 450.0}
    assert float(cmb_boundary_layer_flux(t_core, t_m, **fixed)) == float(
        cmb_boundary_layer_flux(t_core, t_m, **{**fixed, 'interior_viscosity': 1.0})
    )


# 6 states of a 1 M_E coupled run (CMB cell 1500 m): time [Myr], T_core, T_m, top cell T [K],
# mantle thickness [m], layer k, rho, cp, alpha, g at the mean temperature, layer and interior
# viscosity [Pa s], Ra_i, Ra_crit, q [W/m^2] as logged (Ra_crit 4 decimals, else 5 sig. digits).
_RUN_STATES = [
    (0.0983, 5118.811704813429, 5083.533916503925, 2381.400790730236, 3670481.5816515163, 4.0, 4332.787038192286, 5644.44744909633, 5.3480562222107306e-05, 11.012233868575926, 5880.986890556698, 10215.40949939358, 2.0674e+26, 94058.1465, 140.88),
    (0.4996, 4939.818551378954, 4664.90397794359, 1904.2924786796275, 3670481.5816515163, 4.0, 4551.9367805792235, 4042.34997843334, 4.0571510416005524e-05, 11.012233868575926, 2009662022598.5461, 2.888313099508526e+17, 4.8621e+12, 129.2381, 23.614),
    (0.9996, 4933.2693144609475, 4476.473589334745, 1851.9265309624834, 3670481.5816515163, 4.0, 4628.313008403075, 3519.6520287194166, 3.607255608918772e-05, 11.012233868575926, 1.2981748697774718e+16, 3.043132352645164e+21, 3.7491e+08, 17.6882, 4.4957),
    (2.4980, 4922.568201321717, 4347.101038455513, 1687.778531128513, 3670481.5816515163, 4.0, 4684.783131124418, 3144.145014622264, 3.274617417439798e-05, 11.012233868575926, 2.2777956448009418e+18, 4.82608729203438e+22, 2.0619e+07, 9.6196, 1.2584),
    (5.0319, 4902.783984810352, 4313.858952868887, 1570.4865457801793, 3670481.5816515163, 4.0, 4706.52169888826, 3001.9930368828554, 3.146566024635326e-05, 11.012233868575926, 1.1522746073845006e+19, 8.505628079231984e+22, 1.1160e+07, 8.4561, 0.76922),
    (6.9755, 4887.673385940018, 4308.210350087209, 1548.757275183266, 3670481.5816515163, 4.0, 4715.08715445909, 2946.3421343237537, 3.09611106698747e-05, 11.012233868575926, 2.0666773683355165e+19, 1.0073213819539438e+23, 9.1517e+06, 8.1110, 0.62174),
]  # fmt: skip


@pytest.mark.physics_invariant
@pytest.mark.parametrize('row', _RUN_STATES, ids=[f'{r[0]}Myr' for r in _RUN_STATES])
def test_eq16_reproduces_the_states_of_a_coupled_run(row):
    """At six states from molten (0.1 Myr) to solid (7 Myr), eq. 16 and the layer flux equal
    eqs. 3, 13, 14 and 16 by hand to 1e-9, and the hand values equal the run log to its
    printed digits."""
    t, t_core, t_m, t_top, depth, k, rho, cp, alpha, g, eta, eta_i, ra_log, rc_log, q_log = row
    kappa = k / (rho * cp)
    ra_i = alpha * rho * g * (t_core - t_top) * depth**3 / (kappa * eta_i)
    ra_crit = 0.28 * ra_i**0.21
    delta = (ra_crit * kappa * eta / (alpha * rho * g * (t_core - t_m))) ** (1.0 / 3.0)
    layer = dict(conductivity=k, density=rho, heat_capacity=cp, expansivity=alpha, gravity=g)
    shell = dict(shell_thickness=depth, shell_contrast=t_core - t_top)
    assert float(critical_rayleigh_eq16(viscosity=eta_i, **layer, **shell)) == pytest.approx(
        ra_crit, rel=1e-9
    )
    q = float(
        cmb_boundary_layer_flux(
            t_core, t_m, viscosity=eta, interior_viscosity=eta_i, dr_half=1500.0,
            ra_crit=None, **layer, **shell,
        )
    )  # fmt: skip
    assert q == pytest.approx(k * (t_core - t_m) / delta, rel=1e-9)
    assert ra_i == pytest.approx(ra_log, rel=5e-5)
    assert ra_crit == pytest.approx(rc_log, abs=6e-5)
    assert q == pytest.approx(q_log, rel=5e-5)


def test_non_buoyant_layer_conducts():
    """Zero or negative expansivity removes the convective branch."""
    for alpha in (0.0, -1e-5):
        props = dict(_LIQUID, expansivity=alpha)
        assert _flux(100.0, props) == pytest.approx(4.0 * 100.0 / 1e4, rel=1e-9)


def test_ra_crit_scales_the_convective_flux():
    """q_conv scales as Ra_crit^(-1/3): a 8x larger Ra_crit halves the flux."""
    base = _flux(100.0, _LIQUID)
    assert _flux(100.0, _LIQUID, ra_crit=8.0 * RA_CRIT_UPPER_LAYER) == pytest.approx(
        base / 2.0, rel=1e-12
    )


@pytest.mark.physics_invariant
def test_eq16_flux_has_the_sign_of_the_contrast_and_vanishes_at_zero():
    """At the 5.03 Myr state with eq. 16, from dT = -50 K to +50 K the flux has the sign of dT,
    a colder core conducts across the half cell (k dT / dr_half), and the flux is below
    1e-6 W/m^2 at |dT| = 1e-6 K on both sides (Validation V4)."""
    dTs = np.concatenate([np.linspace(-50.0, 50.0, 101), [-1e-6, 1e-6]])
    t_m = 4313.9
    q = np.asarray(cmb_boundary_layer_flux(t_m + dTs, t_m, ra_crit=None, **_STATE))
    dT = (t_m + dTs) - t_m  # the contrast after rounding at t_m
    assert np.array_equal(np.sign(q), np.sign(dT))
    cold = dT <= 0.0
    np.testing.assert_allclose(q[cold], 4.0 * dT[cold] / 750.0, rtol=1e-12, atol=0.0)
    assert np.abs(q[-2:]).max() < 1e-6


def test_flux_is_continuous_and_differentiable_through_zero():
    """The derivative is finite on both sides of dT = 0 (the JAX Jacobian needs it), and the
    flux tends to zero from both sides."""

    def f(t_core):
        return cmb_boundary_layer_flux(t_core, 4000.0, **_LIQUID)

    t_core = np.array([3999.0, 4000.0 - 1e-9, 4000.0, 4000.0 + 1e-9, 4001.0])
    assert np.isfinite(np.asarray(jax.vmap(jax.grad(f))(t_core))).all()
    assert np.abs(np.asarray(f(t_core))[1:4]).max() < 1e-6


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('change', 'factor'),
    [
        ({'dT': 800.0}, 16.0),
        ({'viscosity': 0.8}, 0.5),
        ({'expansivity': 2.4e-4}, 2.0),
        ({'conductivity': 32.0}, 4.0),
    ],
    ids=['dT', 'viscosity', 'expansivity', 'conductivity'],
)
def test_convective_flux_follows_the_boundary_layer_scaling(change, factor):
    """On the convective branch q = k dT / delta with delta ~ (kappa eta / (alpha dT))^(1/3)
    and kappa = k / (rho c_p), so q ~ dT^(4/3) eta^(-1/3) alpha^(1/3) k^(2/3): an 8x change
    of each gives 16, 1/2, 2 and 4 times the flux."""
    props = dict(_LIQUID, **{k: v for k, v in change.items() if k != 'dT'})
    got = _flux(change.get('dT', 100.0), props)
    assert got == pytest.approx(factor * _flux(100.0, _LIQUID), rel=1e-12)


def test_flux_slope_on_each_side_of_zero_contrast():
    """A colder core conducts across the half cell, slope k / dr_half = 4e-4 W m^-2 K^-1.
    A hotter core over a liquid base stays on the convective branch down to about 1e-17 K,
    since delta grows only as dT^(-1/3) (33 m at 1e-9 K against the 1e4 m half cell), so
    there q = q(100 K) (dT / 100 K)^(4/3) with slope 4 q / (3 dT)."""

    def f(t_core):
        return cmb_boundary_layer_flux(t_core, 4000.0, **_LIQUID)

    t_hot = 4000.0 + 1e-9
    dT = t_hot - 4000.0  # the contrast the flux sees after rounding at 4000 K
    slopes = np.asarray(jax.vmap(jax.grad(f))(np.array([4000.0 - 1e-9, t_hot])))
    q_hot = _flux(100.0, _LIQUID) * (dT / 100.0) ** (4.0 / 3.0)
    assert slopes[0] == pytest.approx(4.0e-4, rel=1e-9, abs=0.0)
    assert float(f(t_hot)) == pytest.approx(q_hot, rel=1e-9, abs=0.0)
    assert slopes[1] == pytest.approx(4.0 * q_hot / (3.0 * dT), rel=1e-9, abs=0.0)


@pytest.mark.parametrize(('ra_crit', 'calls'), [(450.0, 0), (None, 1), (float('nan'), 1)])
def test_eq16_is_evaluated_only_when_the_law_needs_it(ra_crit, calls):
    """A fixed critical Rayleigh number does not evaluate eq. 16; None and the NaN sentinel
    evaluate it once."""
    eq16 = boundary_layer.critical_rayleigh_eq16
    with mock.patch.object(boundary_layer, 'critical_rayleigh_eq16', side_effect=eq16) as spy:
        q = float(cmb_boundary_layer_flux(4902.8, 4313.9, ra_crit=ra_crit, **_STATE))
    assert np.isfinite(q) and q > 0.0
    assert spy.call_count == calls


@pytest.mark.parametrize('bad', [True, '450', [450.0], 0, -450.0, float('nan'), float('inf')])
def test_check_ra_crit_refuses_anything_but_a_positive_finite_number(bad):
    """Only None (eq. 16) or a positive, finite real number passes, the number as a float; a boolean, a string or a
    sequence is refused with the same message as a non-physical value."""
    assert check_ra_crit(1800) == 1800.0 and isinstance(check_ra_crit(np.float32(450.0)), float)
    assert check_ra_crit(None) is None  # None selects Thiriet eq. 16
    with pytest.raises(ValueError, match='ra_crit_cmb must be positive and finite'):
        check_ra_crit(bad)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('shell', [0.0, -100.0, 10.0])
def test_eq16_shell_contrast_is_at_least_the_cmb_contrast(shell):
    """The shell spans the CMB layer, so its contrast is held at the CMB contrast or more: a
    shell contrast at or below the CMB contrast gives the flux of the CMB contrast, never the
    unbounded flux of a vanishing Ra_i."""
    t_core, t_m = 4902.8, 4313.9
    state = dict(_STATE, shell_contrast=shell)
    floor = dict(_STATE, shell_contrast=t_core - t_m)
    q = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=None, **state))
    assert q == float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=None, **floor))
    assert 0.0 < q < float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=None, **_STATE)) * 2.0
