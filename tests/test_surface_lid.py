"""Tests of the sub-grid lid functions in ``aragog.surface_lid``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf, erfi

from aragog import surface_lid as sl

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _f(s, x):
    if s == 0.0:
        return x
    if s > 0.0:
        return erf(np.sqrt(s) * x) / erf(np.sqrt(s))
    return erfi(np.sqrt(-s) * x) / erfi(np.sqrt(-s))


def _neumann_lambda(St):
    return brentq(lambda lam: lam * np.exp(lam**2) * erf(lam) - St / np.sqrt(np.pi), 1e-9, 5.0)


def test_constants_pinned():
    """The fixed design values of the lid keep the values the validation was run with."""
    assert sl.LID_CAP_FRAC == pytest.approx(0.8, abs=0)
    assert sl.LID_CAP_WIDTH == pytest.approx(0.02, abs=0)
    assert sl.LID_WEIGHT_FLOOR == pytest.approx(1e-10, abs=0)
    assert sl.LID_EXIST_WIDTH == pytest.approx(0.1, abs=0)
    assert sl.LID_DH_MIN == pytest.approx(1e3, abs=0)
    assert sl.LID_TAU_HIDDEN == pytest.approx(1e10, abs=0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('s', [-200.0, -3.0, -1.0e-7, 0.0, 1.0e-7, 0.5, 3.0, 200.0])
def test_shape_matches_profile(s):
    """Slopes and mean of the closed forms equal the derivative and the integral of f_s."""
    f0, f1, m = sl.shape(s)
    h = 1e-5
    d0 = (_f(s, h) - _f(s, -h)) / (2 * h) if s > -10 else None
    d1 = (_f(s, 1.0 + h) - _f(s, 1.0 - h)) / (2 * h) if s < 10 else None
    mean = quad(lambda x: _f(s, x), 0.0, 1.0, epsabs=1e-13, epsrel=1e-12)[0]
    assert m == pytest.approx(mean, rel=1e-8, abs=1e-12)
    if d0 is not None:
        assert f0 == pytest.approx(d0, rel=2e-5)
    if d1 is not None:
        assert f1 == pytest.approx(d1, rel=2e-5)
    # f0 / f1 = exp(s) for the whole family: the most likely slip (a lost exp) breaks it
    assert f0 / f1 == pytest.approx(np.exp(s), rel=1e-10)


@pytest.mark.physics_invariant
def test_shape_continuous_at_series_switch():
    """The series form near s = 0 joins the closed forms without a step."""
    for sign in (1.0, -1.0):
        a = np.array(sl.shape(sign * 0.999999e-6))
        b = np.array(sl.shape(sign * 1.000001e-6))
        np.testing.assert_allclose(a, b, rtol=1e-11)


@pytest.mark.physics_invariant
def test_mean_monotone_and_bounds():
    """The mean rises with s, and the shape bounds sit at the accepted mean ratios."""
    s = np.r_[-np.geomspace(5e3, 1e-3, 60), 0.0, np.geomspace(1e-3, 3e3, 60)]
    m = np.array([sl.shape(x)[2] for x in s])
    assert np.all(np.diff(m) > 0)
    assert sl.shape(sl.LID_SHAPE_MIN)[2] == pytest.approx(sl.LID_MEAN_RATIO_MIN, rel=1e-9)
    assert sl.shape(sl.LID_SHAPE_MAX)[2] == pytest.approx(sl.LID_MEAN_RATIO_MAX, rel=1e-9)


def _neumann_case(St, delta=10.0, T_s=300.0, T_b=1400.0, k=4.0, T_eq=200.0):
    lam = _neumann_lambda(St)
    s = lam**2
    f0, f1, m = sl.shape(s)
    F_s = k * (T_b - T_s) * f0 / delta
    eps = F_s / (Stefan_Boltzmann * (T_s**4 - T_eq**4))
    return dict(
        lam=lam,
        s=s,
        m=m,
        f1=f1,
        eps=eps,
        T_m=T_s + m * (T_b - T_s),
        T_s=T_s,
        T_b=T_b,
        k=k,
        T_eq=T_eq,
        delta=delta,
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('St', [0.1, 1.0, 3.381])
def test_surface_state_recovers_neumann(St):
    """The surface solve returns the Neumann profile of the one-phase Stefan problem.

    Reference: Carslaw and Jaeger (1959), the Neumann solution; the profile is the similarity
    family with s = lambda^2, lambda exp(lambda^2) erf(lambda) = St / sqrt(pi).
    """
    c = _neumann_case(St)
    out = sl.surface_state(c['delta'], c['T_m'], c['T_b'], c['k'], c['eps'], c['T_eq'], 0.0)
    assert out.clipped == 0
    assert out.s == pytest.approx(c['s'], rel=1e-9)
    assert out.T_s == pytest.approx(c['T_s'], rel=1e-10)
    assert out.q_b == pytest.approx(
        c['k'] * (c['T_b'] - c['T_s']) * c['f1'] / c['delta'], rel=1e-9
    )
    assert out.F_s == pytest.approx(sl.grey_flux(out.T_s, c['eps'], c['T_eq']), rel=1e-9)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('St', [0.1, 1.0, 3.381])
def test_lid_energy_follows_neumann_growth(St):
    """On the Neumann solution the lid rates keep the mean fixed and grow as Neumann.

    Neumann: d(delta)/dt = 2 lambda^2 kappa / delta and T_m constant, so
    dE_lid/dt = rho cp T_m d(delta)/dt. A linear profile (s = 0) misses the growth rate by
    more than 5 % at each Stefan number here (-0.8 %, -7 % and -16 % for the energy-consistent
    linear law), which the second check guards.
    """
    rho, cp = 3000.0, 1200.0
    c = _neumann_case(St)
    L = cp * (c['T_b'] - c['T_s']) / St
    surf = sl.surface_state(c['delta'], c['T_m'], c['T_b'], c['k'], c['eps'], c['T_eq'], 0.0)
    h_b = cp * c['T_b']
    E = sl.lid_energy(rho, c['delta'], h_b, cp, c['T_b'], c['T_m'])
    rates = sl.lid_rates(
        delta=c['delta'],
        E_lid=E,
        surf=surf,
        F_b=0.0,
        F_grid=0.0,
        w=1.0,
        rho=rho,
        cp=cp,
        h_b=h_b,
        h_f=h_b + L,
        T_b=c['T_b'],
        h_top=1e6,
        delta_lo=1e-3,
        heating=0.0,
    )
    kappa = c['k'] / (rho * cp)
    assert rates.ddelta == pytest.approx(2.0 * c['lam'] ** 2 * kappa / c['delta'], rel=1e-9)
    assert rates.dE == pytest.approx(rho * cp * c['T_m'] * rates.ddelta, rel=1e-9)
    assert rates.F_fluid == pytest.approx(0.0, abs=1e-9 * surf.q_b)
    r_linear = c['k'] * (c['T_b'] - c['T_s']) / c['delta'] / (rho * L)
    if St > 0.5:
        assert abs(r_linear / rates.ddelta - 1.0) > 0.05


@pytest.mark.physics_invariant
def test_surface_state_steady_linear_profile():
    """A lid whose mean is the mid temperature has the linear profile and q_b = F_s."""
    T_b, T_s, k, d, T_eq = 1400.0, 500.0, 4.0, 100.0, 255.0
    eps = k * (T_b - T_s) / d / (Stefan_Boltzmann * (T_s**4 - T_eq**4))
    out = sl.surface_state(d, 0.5 * (T_b + T_s), T_b, k, eps, T_eq, 0.0)
    assert out.s == pytest.approx(0.0, abs=1e-9)
    assert out.q_b == pytest.approx(out.F_s, rel=1e-8)
    assert out.T_s == pytest.approx(T_s, rel=1e-10)


@pytest.mark.physics_invariant
def test_surface_state_clips_at_both_ends():
    """A mean too hot or too cold for any profile uses the bound and reports it."""
    hot = sl.surface_state(1.0, 1399.999, 1400.0, 4.0, 1.0, 255.0, 0.0)
    cold = sl.surface_state(1.0, 255.05, 1400.0, 4.0, 1.0, 255.0, 0.0)
    assert hot.clipped == 1
    assert hot.s == pytest.approx(sl.LID_SHAPE_MAX, rel=1e-12)
    assert cold.clipped == -1
    assert cold.s == pytest.approx(sl.LID_SHAPE_MIN, rel=1e-12)
    for out in (hot, cold):
        assert np.isfinite(out.F_s) and np.isfinite(out.q_b)


@pytest.mark.physics_invariant
def test_surface_state_floor_bounds_flux():
    """At zero thickness the conduction floor keeps the fluxes finite and the surface below T_b."""
    out = sl.surface_state(0.0, 1000.0, 1400.0, 4.0, 1.0, 255.0, 1e-3)
    assert np.isfinite(out.F_s) and out.F_s < 1e12
    assert 255.0 < out.T_s < 1400.0


@pytest.mark.physics_invariant
def test_convective_base_flux_scaling():
    """The boundary-layer flux scales as dT^(4/3) and eta^(-1/3) and vanishes for dT <= 0."""
    args = dict(k=4.0, rho=3000.0, cp=1200.0, alpha=3e-5, g=9.8, ra_crit=450.0)
    q1 = sl.convective_base_flux(10.0, eta=1e2, **args)
    q2 = sl.convective_base_flux(80.0, eta=1e2, **args)
    q3 = sl.convective_base_flux(10.0, eta=8e2, **args)
    assert q2 / q1 == pytest.approx(16.0, rel=1e-12)
    assert q1 / q3 == pytest.approx(2.0, rel=1e-12)
    assert sl.convective_base_flux(0.0, eta=1e2, **args) == pytest.approx(0.0, abs=0)
    assert sl.convective_base_flux(-5.0, eta=1e2, **args) == pytest.approx(0.0, abs=0)
    kappa = 4.0 / (3000.0 * 1200.0)
    assert q1 == pytest.approx(
        4.0 * 10.0 ** (4 / 3) * (3e-5 * 3000.0 * 9.8 / (450.0 * kappa * 1e2)) ** (1 / 3),
        rel=1e-12,
    )


@pytest.mark.physics_invariant
def test_stefan_rate_stops():
    """Growth stops at LID_CAP_FRAC h_top, melt-back at delta_lo, and dh has a floor."""
    h, lo = 1000.0, 0.5
    r, f = sl.stefan_rate(10.0, 0.0, 3000.0, 4e5, 0.8 * h, h, lo)
    assert (r, f) == (0.0, 0.0)
    r, f = sl.stefan_rate(10.0, 0.0, 3000.0, 4e5, 0.78 * h, h, lo)
    assert f == pytest.approx(1.0, abs=1e-12) and r == pytest.approx(
        10.0 / (3000.0 * 4e5), rel=1e-12
    )
    r, f = sl.stefan_rate(10.0, 0.0, 3000.0, 4e5, 0.79 * h, h, lo)
    assert f == pytest.approx(0.5, rel=1e-9)
    r, f = sl.stefan_rate(0.0, 10.0, 3000.0, 4e5, lo, h, lo)
    assert r == pytest.approx(0.0, abs=0) and f == pytest.approx(0.0, abs=0)
    r, f = sl.stefan_rate(0.0, 10.0, 3000.0, 4e5, 2 * lo, h, lo)
    assert f == pytest.approx(1.0, abs=0) and r < 0.0
    r, _ = sl.stefan_rate(10.0, 0.0, 3000.0, -5.0, 1.0, h, lo)
    assert r == pytest.approx(10.0 / (3000.0 * sl.LID_DH_MIN), rel=1e-12)


@pytest.mark.physics_invariant
def test_existence_weight_limits():
    """No lid where the fluid outruns radiation at T_b, a full lid without base flux, 1/2 at delta_on."""
    k, eps, T_eq, T_b = 4.0, 1.0, 255.0, 1400.0
    F_rad = sl.grey_flux(T_b, eps, T_eq)
    assert sl.existence_weight(1.01 * F_rad, T_b, k, eps, T_eq, 1.0) == pytest.approx(
        0.0, abs=0
    )
    assert sl.existence_weight(0.0, T_b, k, eps, T_eq, 1.0) == pytest.approx(1.0, abs=0)
    F_b = 1.0e4
    T_s = (F_b / (eps * Stefan_Boltzmann) + T_eq**4) ** 0.25
    d_eq = k * (T_b - T_s) / F_b
    assert sl.existence_weight(F_b, T_b, k, eps, T_eq, d_eq) == pytest.approx(0.5, rel=1e-12)
    assert (
        sl.existence_weight(F_b, T_b, k, eps, T_eq, d_eq * 10 ** (5 * sl.LID_EXIST_WIDTH))
        < 1e-4
    )
    w = [
        sl.existence_weight(F, T_b, k, eps, T_eq, 1.0)
        for F in np.geomspace(1.0, 0.99 * F_rad, 50)
    ]
    assert np.all(np.diff(w) <= 0.0)
    assert sl.existence_weight(0.999 * F_rad, T_b, k, eps, T_eq, 1e3) == pytest.approx(
        0.0, abs=0
    )


@pytest.mark.physics_invariant
def test_lid_rates_full_lid_delivers_base_flux():
    """With the full lid and no stop the fluid loses F_b, whatever the lid does."""
    surf = sl.surface_state(50.0, 900.0, 1400.0, 4.0, 1.0, 255.0, 0.0)
    kw = dict(
        delta=50.0,
        E_lid=3.0e11,
        surf=surf,
        w=1.0,
        rho=3000.0,
        cp=1200.0,
        h_b=1.0e6,
        h_f=1.4e6,
        T_b=1400.0,
        h_top=1.0e4,
        delta_lo=0.5,
        heating=0.0,
    )
    for F_b in (0.3 * surf.q_b, 3.0 * surf.q_b):
        rates = sl.lid_rates(F_b=F_b, F_grid=0.0, **kw)
        assert rates.stop == pytest.approx(1.0, abs=0)
        assert rates.F_fluid == pytest.approx(F_b, rel=1e-12)
        assert rates.F_top == pytest.approx(surf.F_s, rel=1e-15)


@pytest.mark.physics_invariant
def test_lid_rates_hidden_lid_relaxes_to_phantom():
    """A hidden lid (w = 0) shows the grid flux and relaxes to delta_lo; the energy balance holds."""
    surf = sl.surface_state(5.0, 900.0, 1400.0, 4.0, 1.0, 255.0, 0.0)
    rates = sl.lid_rates(
        delta=5.0,
        E_lid=1.0e10,
        surf=surf,
        F_b=1e3,
        F_grid=2.5e4,
        w=0.0,
        rho=3000.0,
        cp=1200.0,
        h_b=1.0e6,
        h_f=1.4e6,
        T_b=1400.0,
        h_top=1.0e4,
        delta_lo=0.5,
        heating=1e-9,
    )
    assert rates.F_top == pytest.approx(2.5e4, rel=1e-15)
    assert rates.ddelta == pytest.approx(-(5.0 - 0.5) / sl.LID_TAU_HIDDEN, rel=1e-12)
    H_lid = 1e-9 * 3000.0 * 5.0
    lhs = rates.F_fluid + 3000.0 * 1.4e6 * rates.ddelta - rates.dE + H_lid
    assert lhs == pytest.approx(rates.F_top, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('St', [0.1, 3.381])
def test_fixed_surface_recovers_the_neumann_shape(St):
    """With the surface temperature fixed, the mean of the Neumann profile gives s = lambda^2
    and the Neumann base and surface fluxes."""
    c = _neumann_case(St)
    out = sl.fixed_surface_state(c['delta'], c['T_m'], c['T_b'], c['k'], c['T_s'], 0.0)
    assert out.clipped == 0
    assert out.s == pytest.approx(c['s'], rel=1e-9)
    assert out.T_s == pytest.approx(c['T_s'], abs=0)
    assert out.q_b == pytest.approx(
        c['k'] * (c['T_b'] - c['T_s']) * c['f1'] / c['delta'], rel=1e-9
    )


@pytest.mark.physics_invariant
def test_fixed_surface_reports_a_mean_outside_the_family():
    """A mean at the base or at the surface temperature uses the bound of the family and says so."""
    hot = sl.fixed_surface_state(10.0, 1399.9999, 1400.0, 4.0, 300.0, 0.0)
    cold = sl.fixed_surface_state(10.0, 300.0001, 1400.0, 4.0, 300.0, 0.0)
    assert (hot.clipped, cold.clipped) == (1, -1)
    assert hot.s == pytest.approx(sl.LID_SHAPE_MAX, rel=1e-9)
    assert cold.s == pytest.approx(sl.LID_SHAPE_MIN, rel=1e-9)
