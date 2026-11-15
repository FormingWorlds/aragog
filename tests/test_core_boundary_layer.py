"""Unit tests for ``aragog.core.cmb_boundary_layer_flux``.

The flux must carry the sign of ``T_core - T_mantle`` (heat never flows from
the colder side), equal the Thiriet et al. (2019) boundary-layer value for a hotter
core whatever the mesh, conduct across the half cell only for a colder core (or a
layer that is not buoyant), take the critical Rayleigh number from their eq. 16 when
none is set, and stay differentiable for the JAX Jacobian.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.core import (
    RA_CRIT_CMB_DEFAULT,
    RA_CRIT_UPPER_LAYER,
    check_ra_crit,
    cmb_boundary_layer_flux,
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
# at (T_core + T_m)/2, mantle thickness and the core-to-top contrast of the shell.
_STATE = dict(
    conductivity=4.0,
    density=4706.5,
    heat_capacity=3002.0,
    expansivity=3.147e-5,
    viscosity=1.152e19,
    gravity=11.012,
    dr_half=750.0,
    shell_thickness=3670.5e3,
    shell_contrast=3332.0,
)


@pytest.mark.physics_invariant
def test_eq16_critical_rayleigh_number_by_hand():
    """Thiriet et al. (2019) eq. 3 and eq. 16 (0.28 Ra_i^0.21, Deschamps and Sotin 2000), eqs.
    13-14 with beta = 1/3, at the stored state: Ra_i = 8.2e10, Ra_crit = 54.9, delta = 5.71 km,
    q = 0.412 W/m^2 (against 0.205 W/m^2 at Ra_crit 450)."""
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
    ra_i = alpha * rho * g * 3332.0 * 3670.5e3**3 / (kappa * eta)
    ra_crit = 0.28 * ra_i**0.21
    delta = (ra_crit * kappa * eta / (alpha * rho * g * (t_core - t_m))) ** (1.0 / 3.0)
    q = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=None, **_STATE))
    assert q == pytest.approx(k * (t_core - t_m) / delta, rel=1e-12)
    assert (ra_i, ra_crit, delta, q) == pytest.approx(
        (8.238e10, 54.89, 5713.0, 0.4124), rel=2e-3
    )
    q450 = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=450.0, **_STATE))
    assert q450 == pytest.approx(0.2045, rel=2e-3)


def test_eq16_needs_the_shell():
    """With no critical Rayleigh number set, the shell thickness and contrast are required."""
    with pytest.raises(ValueError, match='shell_thickness and shell_contrast'):
        cmb_boundary_layer_flux(5000.0, 4000.0, **{**_LIQUID, 'ra_crit': None})


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


@pytest.mark.parametrize('bad', [True, '450', [450.0], 0, -450.0, float('nan'), float('inf')])
def test_check_ra_crit_refuses_anything_but_a_positive_finite_number(bad):
    """Only None (eq. 16) or a positive, finite real number passes, the number as a float; a boolean, a string or a
    sequence is refused with the same message as a non-physical value."""
    assert check_ra_crit(1800) == 1800.0 and isinstance(check_ra_crit(np.float32(450.0)), float)
    assert check_ra_crit(None) is None  # None selects Thiriet eq. 16
    with pytest.raises(ValueError, match='ra_crit_cmb must be positive and finite'):
        check_ra_crit(bad)
