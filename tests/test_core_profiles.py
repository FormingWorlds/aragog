"""Unit tests for ``aragog.core.profiles.GaussianCoreProfiles``.

The profile family is the analytic foundation every core-module budget term
integrates over, so these tests pin it to physics identities rather than to
its own implementation: mass against independent quadrature, gravity against
its centre limit and Newton's shell integral, pressure against hydrostatic
balance, and the adiabat against the thermodynamic gradient identity that
defines its length scale. Parameters are Earth-like so the pins discriminate
(a compressible Gaussian differs from constant density by ~15% in mass).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy import constants as sp_constants
from scipy.integrate import quad

from aragog.core import GaussianCoreProfiles, fit_gaussian_core_profiles

pytestmark = pytest.mark.unit

G = sp_constants.G

# Earth-like core: centre density and a length scale chosen so the density
# at the CMB is rho_cen * exp(-(r_cmb/L)^2) ~ 9900 kg/m3, the seismological
# CMB-side value; both well inside the family's regime of validity.
EARTH = dict(
    rho_cen=12500.0,
    length_scale=7200e3,
    r_cmb=3480e3,
    p_cmb=136e9,
    alpha=1.35e-5,
    c_p=840.0,
)


@pytest.fixture(scope='module')
def prof():
    """One Earth-like profile shared by the whole file; construction is pure."""
    return GaussianCoreProfiles(**EARTH)


@pytest.mark.physics_invariant
def test_density_positive_monotone_and_mass_matches_quadrature(prof):
    """Density is positive and decreasing outward; the closed-form enclosed
    mass agrees with independent numerical quadrature of 4 pi rho r^2, and
    the compressible mass is discriminably below the constant-density mass."""
    r = np.linspace(0.0, prof.r_cmb, 200)
    rho = np.asarray(prof.density(r))
    assert np.all(rho > 0.0)
    assert np.all(np.diff(rho) < 0.0)
    # rho at the CMB lands on the seismological ~9900 kg/m3 the length scale
    # was chosen for (loose: the choice, not the physics, sets the value).
    assert rho[-1] == pytest.approx(9900.0, rel=0.02)

    m_closed = float(prof.enclosed_mass(prof.r_cmb))
    m_quad, err = quad(lambda s: 4.0 * np.pi * float(prof.density(s)) * s**2, 0.0, prof.r_cmb)
    assert m_closed == pytest.approx(m_quad, rel=1e-9)
    assert err < 1e-6 * abs(m_quad)  # the reference itself converged

    # Discrimination guard: constant-density mass at rho_cen would be ~15%
    # higher; the closed form must not degenerate to it.
    m_const = 4.0 / 3.0 * np.pi * prof.r_cmb**3 * prof.rho_cen
    assert m_const / m_closed > 1.10
    # Sanity against Earth's core mass (1.94e24 kg): same order, within 10%.
    assert m_closed == pytest.approx(1.94e24, rel=0.10)


@pytest.mark.physics_invariant
def test_gravity_centre_limit_and_shell_integral(prof):
    """Gravity vanishes at the centre with the analytic linear slope and
    equals G M(r)/r^2 at finite radius; positive and finite everywhere."""
    # Centre limit: g -> (4 pi / 3) G rho_cen r, exact as r -> 0.
    r_small = 1.0  # 1 m: deep inside the linear regime
    expected = 4.0 / 3.0 * np.pi * G * prof.rho_cen * r_small
    assert float(prof.gravity(r_small)) == pytest.approx(expected, rel=1e-6)
    assert float(prof.gravity(0.0)) == pytest.approx(0.0, abs=1e-12)

    # Finite radius: Newton's shell theorem against the closed-form mass.
    for r in (0.3, 0.7, 1.0):
        radius = r * prof.r_cmb
        newton = G * float(prof.enclosed_mass(radius)) / radius**2
        assert float(prof.gravity(radius)) == pytest.approx(newton, rel=1e-10)

    r = np.linspace(0.0, prof.r_cmb, 100)
    g = np.asarray(prof.gravity(r))
    assert np.all(g >= 0.0) and np.all(np.isfinite(g))
    # Discrimination guard: at the CMB the linear centre law overestimates
    # true gravity by >5% for this compressibility, so the exact branch is
    # measurably not the small-r limit.
    linear_at_cmb = 4.0 / 3.0 * np.pi * G * prof.rho_cen * prof.r_cmb
    assert linear_at_cmb / g[-1] > 1.05


@pytest.mark.physics_invariant
def test_pressure_hydrostatic_balance_and_anchor(prof):
    """dP/dr = -rho g to quadrature accuracy, the CMB anchor is honoured
    exactly, and pressure increases monotonically toward the centre."""
    assert float(prof.pressure(prof.r_cmb)) == pytest.approx(prof.p_cmb, rel=1e-12)

    # Central finite difference against the analytic RHS at three radii.
    h = 50.0  # metres; small vs the ~7e6 m profile scale
    for r in (0.25, 0.5, 0.85):
        radius = r * prof.r_cmb
        dp_dr = (float(prof.pressure(radius + h)) - float(prof.pressure(radius - h))) / (2 * h)
        rhs = -float(prof.density(radius)) * float(prof.gravity(radius))
        assert dp_dr == pytest.approx(rhs, rel=1e-6)

    r = np.linspace(0.0, prof.r_cmb, 100)
    p = np.asarray(prof.pressure(r))
    assert np.all(np.diff(p) < 0.0)  # decreasing outward
    # Centre pressure lands in the Earth ballpark (PREM ~364 GPa); loose
    # bounds because alpha/cp do not enter P and the pin is on the family.
    assert 300e9 < p[0] < 420e9


@pytest.mark.physics_invariant
def test_adiabat_gradient_identity_and_anchor(prof):
    """The adiabat obeys d ln T / dr = -alpha g / c_p in the small-radius
    limit (the identity defining D), anchors at t_cmb, and is hotter inward;
    the identity deviates at the CMB where gravity is sub-linear."""
    t_cmb = 4000.0
    assert float(prof.adiabat(prof.r_cmb, t_cmb)) == pytest.approx(t_cmb, rel=1e-12)
    assert float(prof.t_cen(t_cmb)) > t_cmb

    # Identity at small radius: -2 r / D^2 vs -alpha g / cp with linear g.
    radius = 1e3  # 1 km
    h = 10.0
    t_hi = float(prof.adiabat(radius + h, t_cmb))
    t_lo = float(prof.adiabat(radius - h, t_cmb))
    dlnt_dr = (np.log(t_hi) - np.log(t_lo)) / (2 * h)
    rhs = -prof.alpha * float(prof.gravity(radius)) / prof.c_p
    assert dlnt_dr == pytest.approx(rhs, rel=1e-4)

    # Discrimination guard: at the CMB the same identity must NOT hold to
    # that tolerance (gravity is ~5% below linear there), proving the test
    # resolves the approximation structure rather than passing vacuously.
    t_hi = float(prof.adiabat(prof.r_cmb, t_cmb))
    t_lo = float(prof.adiabat(prof.r_cmb - 2 * h, t_cmb))
    dlnt_cmb = (np.log(t_hi) - np.log(t_lo)) / (2 * h)
    rhs_cmb = -prof.alpha * float(prof.gravity(prof.r_cmb - h)) / prof.c_p
    assert abs(dlnt_cmb / rhs_cmb - 1.0) > 0.02

    # Monotone: temperature decreases outward along the whole profile.
    r = np.linspace(0.0, prof.r_cmb, 100)
    t = np.asarray(prof.adiabat(r, t_cmb))
    assert np.all(t > 0.0) and np.all(np.diff(t) < 0.0)


def test_constructor_error_contract_and_jit_compatibility():
    """Non-positive parameters and an out-of-regime length scale raise
    eagerly with the offending name; evaluation is jit-safe and matches
    the eager path bitwise-closely."""
    bad = dict(EARTH)
    bad['rho_cen'] = -1.0
    with pytest.raises(ValueError, match='rho_cen'):
        GaussianCoreProfiles(**bad)
    tail = dict(EARTH)
    tail['length_scale'] = EARTH['r_cmb'] / 4.0  # r_cmb > 3 L: far Gaussian tail
    with pytest.raises(ValueError, match='length scales'):
        GaussianCoreProfiles(**tail)
    with pytest.raises(ValueError, match='unknown pressure_mode'):
        GaussianCoreProfiles(**EARTH, pressure_mode='invalid')

    prof = GaussianCoreProfiles(**EARTH)
    r = np.linspace(0.0, prof.r_cmb, 17)
    for fn in (prof.density, prof.gravity, prof.pressure):
        eager = np.asarray(fn(r))
        jitted = np.asarray(jax.jit(fn)(r))
        np.testing.assert_allclose(jitted, eager, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jax.jit(prof.adiabat)(r, 4000.0)),
        np.asarray(prof.adiabat(r, 4000.0)),
        rtol=1e-12,
    )


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mode', ['quadrature', 'labrosse'])
def test_fit_gaussian_core_profiles_round_trip_recovers_earth_defaults(mode):
    """The root solve reproduces the Earth parameters and constraints to 1e-6 in a round trip."""
    ref = GaussianCoreProfiles(**EARTH, pressure_mode=mode)
    m_core = float(ref.enclosed_mass(ref.r_cmb))
    p_cen = float(ref.pressure(0.0))

    fitted = fit_gaussian_core_profiles(
        m_core=m_core,
        p_cen=p_cen,
        r_cmb=ref.r_cmb,
        p_cmb=ref.p_cmb,
        alpha=ref.alpha,
        c_p=ref.c_p,
        pressure_mode=mode,
    )
    assert fitted.rho_cen == pytest.approx(EARTH['rho_cen'], rel=1e-6)
    assert fitted.length_scale == pytest.approx(EARTH['length_scale'], rel=1e-6)
    assert float(fitted.enclosed_mass(fitted.r_cmb)) == pytest.approx(m_core, rel=1e-6)
    assert float(fitted.pressure(0.0)) == pytest.approx(p_cen, rel=1e-6)


@pytest.mark.physics_invariant
def test_fit_gaussian_core_profiles_zalmoxis_earth_structure():
    """Fitted profile reproduces real Zalmoxis Earth structure and quantifies C_eff change."""
    # Real Zalmoxis tutorial_earth structure output
    m_core_zal = 1.8915686024e24
    p_cen_zal = 341392448890.0
    r_cmb_zal = 3409138.308
    p_cmb_zal = 102988393650.0

    ref_default = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=136e9,
        alpha=1.35e-5,
        c_p=840.0,
    )
    m_core_old = float(ref_default.enclosed_mass(ref_default.r_cmb))
    ratio = m_core_old / m_core_zal
    assert ratio == pytest.approx(1.018653, rel=1e-5)

    fitted = fit_gaussian_core_profiles(
        m_core=m_core_zal,
        p_cen=p_cen_zal,
        r_cmb=r_cmb_zal,
        p_cmb=p_cmb_zal,
        alpha=1.35e-5,
        c_p=840.0,
    )
    assert fitted.rho_cen == pytest.approx(13559.7897, rel=1e-4)
    assert fitted.length_scale == pytest.approx(6281399.14, rel=1e-4)
    assert float(fitted.enclosed_mass(r_cmb_zal)) == pytest.approx(m_core_zal, rel=1e-6)
    assert float(fitted.pressure(0.0)) == pytest.approx(p_cen_zal, rel=1e-6)

    # Effective capacity comparison at T_cmb = 4000 K
    from aragog.core.budget import CoreEnergyBudget
    from aragog.core.melting import QuadraticMeltingCurve

    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    budget_old = CoreEnergyBudget(
        ref_default,
        curve,
        ds_fusion=170.0,
        icn_width=10.0,
        latent_heat=750e3,
        alpha_c=1.0,
        c_light=560.0 / 12150.0,
    )
    budget_new = CoreEnergyBudget(
        fitted,
        curve,
        ds_fusion=170.0,
        icn_width=10.0,
        latent_heat=750e3,
        alpha_c=1.0,
        c_light=560.0 / 12150.0,
    )
    c_eff_old = float(budget_old.effective_capacity(4000.0))
    c_eff_new = float(budget_new.effective_capacity(4000.0))
    # At 4000 K, old Earth geometry is nucleating (4119 K) while Zalmoxis geometry is liquid (3954 K).
    # Geometry shift causes -76%; fit at equal Zalmoxis geometry gives +5.3% and shifts onset +53 K.
    assert c_eff_old == pytest.approx(7.322507e27, rel=1e-5)
    assert c_eff_new == pytest.approx(1.852134e27, rel=1e-5)
    rel_change = (c_eff_new - c_eff_old) / c_eff_old
    assert rel_change == pytest.approx(-0.747063, rel=1e-4)


@pytest.mark.physics_invariant
def test_fit_gaussian_core_profiles_reproduces_structure_constraints_on_exoplanet():
    """Fitted profiles match real Zalmoxis structure output from zalmoxis_resume_mesh to 1e-6."""
    m_exo = 9.8365400909e23
    p_cen_exo = 184117858870.0
    r_cmb_exo = 2867012.4963
    p_cmb_exo = 54848888186.0

    fitted = fit_gaussian_core_profiles(
        m_core=m_exo,
        p_cen=p_cen_exo,
        r_cmb=r_cmb_exo,
        p_cmb=p_cmb_exo,
        alpha=1.35e-5,
        c_p=840.0,
    )
    assert fitted.rho_cen == pytest.approx(11902.885, rel=1e-4)
    assert fitted.length_scale == pytest.approx(5221789.65, rel=1e-4)
    assert float(fitted.enclosed_mass(r_cmb_exo)) == pytest.approx(m_exo, rel=1e-6)
    assert float(fitted.pressure(0.0)) == pytest.approx(p_cen_exo, rel=1e-6)


def test_fit_gaussian_core_profiles_error_contract_and_non_convergence(recwarn):
    """The solver raises ValueError without warnings when parameters are invalid or unbracketed."""
    ref = GaussianCoreProfiles(**EARTH)
    ok = dict(
        m_core=float(ref.enclosed_mass(ref.r_cmb)),
        p_cen=float(ref.pressure(0.0)),
        r_cmb=ref.r_cmb,
        p_cmb=ref.p_cmb,
        alpha=ref.alpha,
        c_p=ref.c_p,
    )
    cases = [
        ('must exceed p_cmb', {'p_cen': ref.p_cmb}),
        ('incompressible central pressure', {'p_cen': ref.p_cmb + 1e6}),
        ('valid Gaussian regime', {'p_cen': ok['p_cen'] * 20.0}),
        ('must be positive', {'m_core': -1.0}),
    ]
    for match, override in cases:
        with pytest.raises(ValueError, match=match):
            fit_gaussian_core_profiles(**{**ok, **override})

    # Strict contract: no warnings emitted, raise only
    assert len(recwarn) == 0
