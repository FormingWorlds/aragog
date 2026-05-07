"""Unit tests for ``aragog.mesh.pressure_eos.AdamsWilliamsonEOS``.

The Adams-Williamson EOS is the SPIDER-parity exponential density
profile that fixes the mesh pressure and density. SPIDER's
``eos_adamswilliamson.c`` is the reference; the previous Aragog
implementation used a rational (hyperbolic) form derived from
``K_S``, which differed from SPIDER by up to 6 % at CMB depth and
caused 3 % R_int mismatches. The current code switched to the
exponential form, but the entire 371 LOC has had no direct unit
tests.

These tests verify the closed-form physics:
  rho(z) = rho_s * exp(beta * z)         (exponential density)
  P(z)   = (rho_s g / beta) * (e^(beta z) - 1) + P_s   (pressure)
  dP/dr  = -g * rho                      (hydrostatic balance)
  r(P) is the analytic inverse of P(r)   (round-trip identity)

All tests use SI units. Defaults pick beta from the K_S derivation
(``adams_williamson_beta = 0`` → beta = rho_s g / K_S).
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.mesh.pressure_eos import AdamsWilliamsonEOS
from aragog.parser import _MeshParameters, _ScalingsParameters

pytestmark = pytest.mark.unit


def _earth_settings(beta_override: float = 0.0) -> _MeshParameters:
    """Build an Earth-like ``_MeshParameters`` with vestigial scalings.

    Defaults: rho_s = 4000 kg/m^3, g = 9.81 m/s^2, K_S = 260 GPa, P_s = 0.

    With beta_override = 0 the EOS derives beta = rho_s g / K_S
    ≈ 1.51e-7 m^-1, the SPIDER-parity Earth value. With
    beta_override > 0 that override path is exercised instead.
    """
    s = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=100,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
        surface_density=4000.0,
        gravitational_acceleration=9.81,
        adiabatic_bulk_modulus=260e9,
        adams_williamson_beta=beta_override,
        surface_pressure=0.0,
    )
    s.scale_attributes(_ScalingsParameters())  # all scales = 1
    return s


def _earth_radii(N: int = 32) -> np.ndarray:
    """Basic-node radii from CMB to surface."""
    return np.linspace(3.480e6, 6.371e6, N)


# ---- Density physics -------------------------------------------------------


def test_density_at_surface_equals_rho_s_exactly():
    """At r = R_surf the depth z = 0 so rho(z) = rho_s.

    Discriminator: a regression to ``rho = rho_s * exp(-beta * z)``
    (sign flip on depth) would still give rho_s here BUT silently
    invert the depth dependence — discriminated by the next test.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)

    rho_at_surface = float(eos.basic_density[-1])
    assert rho_at_surface == pytest.approx(
        settings.surface_density,
        rel=1e-12,
    )


def test_density_increases_monotonically_with_depth():
    """Density must increase from surface to CMB. Catches a sign
    flip on the exponent (rho_s * exp(-beta*z)) that the
    surface-equality test alone would not notice.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)
    # eos.basic_density is ordered surface (max r) -> CMB (min r) flipped:
    # actually basic_density(r_basic) at r_basic ascending, so [0] = CMB,
    # [-1] = surface. CMB density should be larger.
    rho_cmb = float(eos.basic_density[0])
    rho_surf = float(eos.basic_density[-1])
    assert rho_cmb > rho_surf, (
        'Density at CMB ({:.1f} kg/m^3) is not greater than surface '
        '({:.1f}); exponent sign is wrong.'.format(rho_cmb, rho_surf)
    )


def test_density_matches_closed_form_exponential():
    """For a non-trivial physical depth, density must equal the
    closed form rho_s * exp(beta * (R_surf - r)) to machine
    precision.

    Discriminator vs the legacy rational form
    rho = rho_s * K_S / (K_S - rho_s g z): at z = 1500 km the
    exponential and rational forms differ by ~3 %.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)
    beta = eos._beta
    rho_s = settings.surface_density
    R_surf = settings.outer_radius

    expected = rho_s * np.exp(beta * (R_surf - radii))
    np.testing.assert_allclose(
        np.asarray(eos.basic_density).ravel(),
        expected,
        rtol=1e-12,
    )


def test_beta_override_takes_precedence_over_K_S_derivation():
    """When ``adams_williamson_beta > 0`` the EOS must use that
    value instead of deriving from K_S.

    Discriminator: a regression that ignored the override would
    silently fall back to the derived value; pick a beta_override
    that is 10x the derived one so the resulting CMB density is
    different by orders of magnitude.
    """
    settings_default = _earth_settings(beta_override=0.0)
    radii = _earth_radii()
    eos_default = AdamsWilliamsonEOS(settings_default, radii)
    beta_default = eos_default._beta

    beta_override = 10.0 * beta_default
    settings_override = _earth_settings(beta_override=beta_override)
    eos_override = AdamsWilliamsonEOS(settings_override, radii)
    assert eos_override._beta == pytest.approx(beta_override, rel=1e-12)
    # CMB densities must differ by exp(9 * beta_default * z_cmb), order 1e8
    z_cmb = settings_default.outer_radius - settings_default.inner_radius
    ratio = float(eos_override.basic_density[0]) / float(eos_default.basic_density[0])
    expected_ratio = np.exp(9.0 * beta_default * z_cmb)
    assert ratio == pytest.approx(expected_ratio, rel=1e-9), (
        'beta override was silently ignored; CMB density ratio '
        f'{ratio:.3e} does not match exp(9 beta_default z_cmb) '
        f'{expected_ratio:.3e}.'
    )


# ---- Pressure physics ------------------------------------------------------


def test_pressure_at_surface_equals_surface_pressure():
    """At r = R_surf the pressure equals P_surf exactly (depth = 0
    makes the exponential bracket zero).
    """
    settings = _earth_settings()
    settings.surface_pressure = 1.0e5  # 1 atm overburden
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)

    P_surf = float(eos.basic_pressure[-1])
    assert P_surf == pytest.approx(1.0e5, rel=1e-12)


def test_pressure_at_cmb_within_earth_order_of_magnitude():
    """Earth's CMB pressure is ~135 GPa. The Adams-Williamson value
    with the K_S-derived beta and Earth surface density should be
    within a factor of 2 of that. This is a sanity check, not a
    parity check (real Earth has a non-uniform density profile).

    Edge case: a regression that mishandled the exponential would
    give a CMB pressure of 0 or 1e30 Pa.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)
    P_cmb = float(eos.basic_pressure[0])
    assert 5.0e10 < P_cmb < 5.0e11, (
        f'CMB pressure {P_cmb:.3e} Pa is outside the plausible'
        ' range [5e10, 5e11] for an Earth-like profile.'
    )


def test_pressure_radius_round_trip_is_identity():
    """Property: ``radii -> pressure -> radii`` must round-trip to
    machine precision via the analytic inverse. Catches sign and
    log-base errors in get_radii_from_pressure.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)

    pressures = eos.get_pressure_from_radii(radii)
    radii_back = eos.get_radii_from_pressure(pressures)
    np.testing.assert_allclose(radii_back, radii, rtol=1e-12)


def test_hydrostatic_balance_dPdr_equals_minus_g_rho():
    """Property: dP/dr = -g rho everywhere (hydrostatic balance).

    The closed-form pressure derivative computed analytically from
    the exponential P(r) must equal the helper ``get_pressure_gradient``
    output. Catches a sign flip OR a missing factor of g.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)

    P = np.asarray(eos.get_pressure_from_radii(radii)).ravel()
    dPdr = np.asarray(eos.get_pressure_gradient(P)).ravel()
    rho = np.asarray(eos.basic_density).ravel()
    expected = -settings.gravitational_acceleration * rho
    np.testing.assert_allclose(dPdr, expected, rtol=1e-12)


# ---- Mass integrals --------------------------------------------------------


def test_mass_within_radii_is_zero_at_inner_boundary():
    """Edge case: M(r_inner) must equal 0 by construction (the
    integral runs from r_inner to r). Catches a regression where
    the anchor was mis-set.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)

    M_at_inner = float(np.asarray(eos.get_mass_within_radii(radii[0])).item())
    assert M_at_inner == pytest.approx(0.0, abs=1e-3)


def test_mass_within_radii_is_monotonic_outwards():
    """Property: M(r) increases monotonically with r (a shell of
    positive volume and positive density adds positive mass).

    Discriminator: a sign error in the antiderivative would give a
    monotonically DECREASING mass integral. Pick non-trivial
    radii to avoid a near-zero discriminator at the boundaries.
    """
    settings = _earth_settings()
    radii = _earth_radii(N=20)
    eos = AdamsWilliamsonEOS(settings, radii)
    M = np.asarray([eos.get_mass_within_radii(r).item() for r in radii])
    assert np.all(np.diff(M) > 0), (
        'Mass integral is not monotonic in r; antiderivative has a sign error.'
    )


def test_mass_within_radii_total_within_earth_order_of_magnitude():
    """Sanity: total mass between CMB and surface is the mantle
    mass. For the SPIDER-parity Earth defaults
    (rho_s=4000 kg/m^3, g=9.81, K_S=260 GPa), the mantle integral
    should be within a factor of 2 of Earth's actual mantle
    (4e24 kg).

    Edge case: a regression that scaled by 1/(4 pi) instead of
    4 pi would give 1e23 — discriminated by this bound.
    """
    settings = _earth_settings()
    radii = _earth_radii(N=64)
    eos = AdamsWilliamsonEOS(settings, radii)
    M_total = float(np.asarray(eos.get_mass_within_radii(radii[-1])).item())
    assert 1.0e24 < M_total < 1.0e25, (
        f'Mantle mass {M_total:.3e} kg is outside the plausible'
        ' range [1e24, 1e25] for an Earth-like profile.'
    )


def test_get_mass_element_at_surface_matches_4pi_r2_rho():
    """``get_mass_element(r) = 4 π r² ρ(r)``. At r = R_surf this is
    4 π R_surf² ρ_s. Discriminator vs a missing 4π factor (only 1/4
    the value) and vs r³ (off by a factor of R_surf).
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)
    r = settings.outer_radius
    expected = 4.0 * np.pi * r**2 * settings.surface_density
    actual = float(np.asarray(eos.get_mass_element(np.array([r]))).item())
    assert actual == pytest.approx(expected, rel=1e-12)


def test_get_mass_element_array_input_matches_pointwise_formula():
    """Property: array input returns 4πr²ρ(r) elementwise. Use 3
    distinct radii where ρ values differ by orders of magnitude
    (exponential profile) so a wrong vectorisation is caught.
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)
    test_r = np.array([3.5e6, 5.0e6, 6.3e6])
    expected = 4.0 * np.pi * test_r**2 * np.asarray(eos.get_density_from_radii(test_r))
    actual = np.asarray(eos.get_mass_element(test_r))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


# ---- UserDefinedEOS --------------------------------------------------------


def _user_defined_settings() -> _MeshParameters:
    """Settings with eos_method=2 and a hand-built linear EOS."""
    s = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=10,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=2,
        eos_file='',  # filled in by the test fixture below
    )
    return s


def _attach_user_eos_arrays(settings: _MeshParameters) -> _MeshParameters:
    """Stamp eos_radius / eos_pressure / eos_density / eos_gravity arrays
    on the settings object directly (bypassing scale_attributes which
    would try to load from eos_file).
    """
    n = 32
    settings.eos_radius = np.linspace(3.480e6, 6.371e6, n)
    # Linear pressure profile (P=0 at surface, monotonically rising into depth).
    settings.eos_pressure = np.linspace(1.35e11, 0.0, n)
    # Linear density: 5500 kg/m^3 at CMB, 4000 kg/m^3 at surface.
    settings.eos_density = np.linspace(5500.0, 4000.0, n)
    settings.eos_gravity = np.full(n, 9.81)
    return settings


def test_user_defined_eos_basic_pressure_matches_input_at_eos_nodes():
    """Constructor must interpolate eos_pressure onto basic_radii.
    When basic_radii equal eos_radius, basic_pressure must match
    the input eos_pressure exactly (PchipInterpolator is local-quadratic
    on uniform grids and identity on its anchor nodes).
    """
    from aragog.mesh.pressure_eos import UserDefinedEOS

    settings = _attach_user_eos_arrays(_user_defined_settings())
    basic_radii = settings.eos_radius.copy()
    eos = UserDefinedEOS(settings, basic_radii)
    np.testing.assert_allclose(
        np.asarray(eos.basic_pressure).ravel(),
        settings.eos_pressure,
        rtol=1e-9,
    )


def test_user_defined_eos_staggered_density_is_neighbour_average():
    """Property: staggered_effective_density[i] = 0.5 * (basic[i] + basic[i+1]).
    Catches a regression that swapped the average for a min/max or used
    only one neighbour.
    """
    from aragog.mesh.pressure_eos import UserDefinedEOS

    settings = _attach_user_eos_arrays(_user_defined_settings())
    basic_radii = settings.eos_radius.copy()
    eos = UserDefinedEOS(settings, basic_radii)
    basic_rho = np.asarray(eos.basic_density).ravel()
    expected_staggered = 0.5 * (basic_rho[:-1] + basic_rho[1:])
    np.testing.assert_allclose(
        np.asarray(eos.staggered_effective_density).ravel(),
        expected_staggered,
        rtol=1e-9,
    )


def test_user_defined_eos_set_staggered_pressure_writes_property():
    """``set_staggered_pressure(staggered_radii)`` populates
    ``staggered_pressure``. Property: midpoint radius gives a
    midpoint-ish pressure (linear profile; PCHIP is not exact-linear
    but is monotone-preserving, so within the bracket).
    """
    from aragog.mesh.pressure_eos import UserDefinedEOS

    settings = _attach_user_eos_arrays(_user_defined_settings())
    basic_radii = settings.eos_radius.copy()
    eos = UserDefinedEOS(settings, basic_radii)
    staggered = 0.5 * (basic_radii[:-1] + basic_radii[1:])
    eos.set_staggered_pressure(staggered)
    sp = np.asarray(eos.staggered_pressure).ravel()
    assert sp.shape == (len(basic_radii) - 1,)
    # Monotone with depth (input pressure is monotone-decreasing in r).
    assert np.all(np.diff(sp) < 0)


def test_user_defined_eos_mass_within_radii_zero_at_inner_anchor():
    """``get_mass_within_radii(r_inner)`` = 0 because the cumulative
    integral is anchored at r_eos[0]. Edge case at the anchor point.
    """
    from aragog.mesh.pressure_eos import UserDefinedEOS

    settings = _attach_user_eos_arrays(_user_defined_settings())
    basic_radii = settings.eos_radius.copy()
    eos = UserDefinedEOS(settings, basic_radii)
    M0 = float(np.asarray(eos.get_mass_within_radii(settings.eos_radius[0])).item())
    assert M0 == pytest.approx(0.0, abs=1e-3)


def test_user_defined_eos_mass_within_radii_monotone_outwards():
    """Property: M(r) increases with r. Use a non-trivial linear ρ(r)
    profile so a sign error in the cumulative trapezoid is caught.
    """
    from aragog.mesh.pressure_eos import UserDefinedEOS

    settings = _attach_user_eos_arrays(_user_defined_settings())
    basic_radii = settings.eos_radius.copy()
    eos = UserDefinedEOS(settings, basic_radii)
    M = np.asarray(
        [float(np.asarray(eos.get_mass_within_radii(r)).item()) for r in settings.eos_radius]
    )
    assert np.all(np.diff(M) > 0), 'cumulative mass must be monotone in r'


def test_get_density_from_pressure_matches_spider_linear_form():
    """SPIDER parity: rho(P) = rho_s + P * beta / g.

    This is the algebraic relation used in SPIDER's
    eos_adamswilliamson.c::GetRho. Catches a regression that
    swapped the formula to rho_s * (1 + P beta / (rho_s g)) (which
    has the same surface limit but different derivative).
    """
    settings = _earth_settings()
    radii = _earth_radii()
    eos = AdamsWilliamsonEOS(settings, radii)
    P = np.array([0.0, 1.0e9, 5.0e10, 1.0e11])
    rho = eos.get_density(P)
    expected = settings.surface_density + P * eos._beta / settings.gravitational_acceleration
    np.testing.assert_allclose(rho, expected, rtol=1e-12)
