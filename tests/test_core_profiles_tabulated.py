"""Tests of ``aragog.core.profiles_tabulated.TabulatedCoreProfiles``."""

from __future__ import annotations

import math
import pathlib

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.optimize import brentq

from aragog.core.iron_table import FIELDS, LiquidIronTable
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.profiles_tabulated import TabulatedCoreProfiles

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

CUT = pathlib.Path(__file__).parent / 'validation' / 'data' / 'paleos_iron_liquid_cut_3tpa.npz'
EARTH = dict(
    rho_cen=12500.0, length_scale=7272e3, r_cmb=3480e3, p_cmb=136e9, alpha=1.35e-5, c_p=840.0
)
T_RANGE = (2500.0, 7000.0)


def gaussian_core(n=201):
    """Core arrays of the Earth Gaussian profile on ``n`` uniform nodes, and the profile."""
    g = GaussianCoreProfiles(**EARTH)
    r = np.linspace(0.0, g.r_cmb, n)
    arrays = {
        'r': r,
        'm': np.asarray(g.enclosed_mass(r)),
        'P': np.asarray(g.pressure(r)),
        'rho': np.asarray(g.density(r)),
        'g': np.asarray(g.gravity(r)),
    }
    return arrays, g


@pytest.fixture(scope='module')
def table():
    """PALEOS liquid iron cut, 45 GPa to 3 TPa."""
    return LiquidIronTable.from_file(CUT)


@pytest.fixture(scope='module')
def paleos(table):
    """Earth structure on the PALEOS liquid isentrope, c_p 840."""
    return TabulatedCoreProfiles(gaussian_core()[0], table, t_cmb_range=T_RANGE)


def test_radial_fields_reproduce_the_structure_between_nodes(paleos):
    """Density, mass, pressure, gravity and potential between the nodes agree with the
    Gaussian profile they were sampled from, to the Hermite error of 201 nodes (measured:
    density 4.2e-7 relative, the others below 4e-11 of their largest value)."""
    _, g = gaussian_core()
    r = np.linspace(0.0, g.r_cmb, 1777)[1:-1]
    for name, tol in (
        ('density', 1e-6),
        ('enclosed_mass', 1e-10),
        ('pressure', 1e-10),
        ('gravity', 1e-10),
        ('potential', 1e-9),
    ):
        ref = np.asarray(getattr(g, name)(r))
        got = np.asarray(getattr(paleos, name)(r))
        assert np.max(np.abs(got - ref)) <= tol * np.max(np.abs(ref)), name
    assert float(paleos.potential(g.r_cmb)) == 0.0
    assert paleos.m_core == pytest.approx(float(g.enclosed_mass(g.r_cmb)), rel=1e-15)


@pytest.mark.physics_invariant
def test_the_isentrope_holds_the_cmb_entropy_between_grid_points(paleos, table):
    """Between the nodes and between the tabulated CMB temperatures, the interpolated
    temperature keeps ``s(P, T_a) = s(P_cmb, T_cmb)`` to 1e-6 (invariant I5), equals T_cmb at
    the CMB and rises monotonically inward."""
    rng = np.random.default_rng(3)
    r = rng.uniform(0.0, paleos.r_cmb, 64)
    for t_cmb in rng.uniform(*T_RANGE, 8):
        t_a = np.asarray(paleos.adiabat(r, t_cmb))
        s_c = table.value('s', paleos.p_cmb, t_cmb)
        miss = np.abs(table.value('s', np.asarray(paleos.pressure(r)), t_a) / s_c - 1.0)
        assert miss.max() < 1e-6
    assert float(paleos.adiabat(paleos.r_cmb, 4321.0)) == pytest.approx(4321.0, rel=1e-12)
    # The centre is hotter than the CMB and T_a rises monotonically inward.
    t_a = np.asarray(paleos.adiabat(np.linspace(0, paleos.r_cmb, 400), 4000.0))
    assert np.all(np.diff(t_a) < 0.0)


@pytest.mark.physics_invariant
def test_the_isentrope_matches_an_independent_inversion(paleos, table):
    """At off-grid points the centre temperature equals a brentq root of the table entropy,
    and the radial gradient equals ``(dT/dP)_s dP/dr`` with ``(dT/dP)_s = nabla_ad T / P``."""
    for t_cmb in (2711.0, 4400.5, 6789.0):
        s_c = table.value('s', paleos.p_cmb, t_cmb)
        p_cen = float(paleos.pressure(0.0))
        t_ref = brentq(lambda t: table.value('s', p_cen, t) - s_c, 2000.0, 19000.0, xtol=1e-10)
        assert float(paleos.t_cen(t_cmb)) == pytest.approx(t_ref, rel=1e-7)
        r = 2.1e6
        t_a, p = float(paleos.adiabat(r, t_cmb)), float(paleos.pressure(r))
        slope = table.value('nabla_ad', p, t_a) * t_a / p
        dtdr = slope * -float(paleos.density(r)) * float(paleos.gravity(r))
        assert float(paleos.adiabat_gradient(r, t_cmb)) == pytest.approx(dtdr, rel=1e-4)


@pytest.mark.physics_invariant
def test_secular_capacity_is_cp_over_tcmb_times_the_integral_of_t_dm(paleos, table):
    """``C_s = (840 / T_cmb) int T_a dm`` against scipy quadrature over radius with the
    temperature from brentq inversion; the isothermal form ``840 M`` misses by over 10 %."""
    for t_cmb in (3100.0, 4777.7):
        s_c = table.value('s', paleos.p_cmb, t_cmb)

        def integrand(r, s_c=s_c):
            p = float(paleos.pressure(r))
            t = brentq(lambda x: table.value('s', p, x) - s_c, 2000.0, 19000.0, xtol=1e-10)
            return t * float(paleos.density(r)) * 4.0 * math.pi * r**2

        ref = 840.0 / t_cmb * quad(integrand, 0.0, paleos.r_cmb, epsrel=1e-10, limit=200)[0]
        assert float(paleos.secular_capacity(t_cmb)) == pytest.approx(ref, rel=1e-6)
        assert abs(840.0 * paleos.m_core / ref - 1.0) > 0.1


def test_secular_capacity_without_override_takes_the_table_cp_at_the_cmb(table):
    """With ``c_p=None`` the capacity scales with the table c_p at the CMB."""
    arrays, _ = gaussian_core()
    a = TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE)
    b = TabulatedCoreProfiles(arrays, table, c_p=None, t_cmb_range=T_RANGE)
    cp = table.value('cp', a.p_cmb, 4000.0)
    ratio = float(b.secular_capacity(4000.0)) / float(a.secular_capacity(4000.0))
    assert ratio == pytest.approx(cp / 840.0, rel=1e-6)
    assert 400.0 < cp < 700.0  # the 3nR liquid of PALEOS, well below 840


@pytest.mark.physics_invariant
def test_heat_change_is_the_integral_of_the_capacity_it_reads(paleos):
    """``secular_heat_change`` equals the quadrature of ``secular_capacity`` across grid
    points, is antisymmetric, additive, and zero for a zero change; its capacity derivative
    equals the finite difference (the Jacobian sees the same function)."""
    cap = lambda t: float(paleos.secular_capacity(t))  # noqa: E731
    for t0, dt in ((4000.0, -55.0), (3333.3, 0.7), (6900.0, 99.0), (2600.0, -100.0)):
        ref = quad(cap, t0, t0 + dt, epsrel=1e-13, points=None, limit=200)[0]
        assert paleos.secular_heat_change(t0, dt) == pytest.approx(ref, rel=1e-11)
        back = paleos.secular_heat_change(t0 + dt, -dt)
        assert back == pytest.approx(-paleos.secular_heat_change(t0, dt), rel=1e-13)
    whole = paleos.secular_heat_change(3000.0, 500.0)
    parts = paleos.secular_heat_change(3000.0, 123.4) + paleos.secular_heat_change(
        3123.4, 376.6
    )
    assert whole == pytest.approx(parts, rel=1e-13)
    assert paleos.secular_heat_change(4000.0, 0.0) == 0.0
    d = float(jax.jit(jax.grad(paleos.secular_capacity))(4000.0))
    assert d == pytest.approx((cap(4000.01) - cap(3999.99)) / 0.02, rel=1e-6)


def test_capacity_is_positive_and_flat_outside_the_grid(paleos):
    """C_s > 0 over the grid (invariant I4), and a temperature outside the grid reads the
    edge value (the solver refuses those states before they are booked)."""
    t = np.linspace(*T_RANGE, 500)
    assert np.all(np.asarray(paleos.secular_capacity(t)) > 0.0)
    assert float(paleos.secular_capacity(1000.0)) == float(paleos.secular_capacity(T_RANGE[0]))


def test_centre_inversion_and_calibration_fraction(paleos):
    """``t_cmb_at_centre`` inverts ``t_cen``; an Earth core at 4000 K is wholly below the
    8000 K floor of the Luo window, so the outside fraction is 1."""
    t = paleos.t_cmb_at_centre(float(paleos.t_cen(4567.0)))
    assert t == pytest.approx(4567.0, rel=1e-10)
    with pytest.raises(ValueError, match='no tabulated isentrope'):
        paleos.t_cmb_at_centre(50000.0)
    assert paleos.outside_calibration_fraction(4000.0) == pytest.approx(1.0, abs=1e-15)


def test_calibration_fraction_inside_the_window(table):
    """A hot core between 50 and 1300 GPa with its isentrope above 8000 K lies inside."""
    arrays, _ = gaussian_core()
    hot = TabulatedCoreProfiles(arrays, table, t_cmb_range=(8100.0, 9000.0))
    assert hot.outside_calibration_fraction(8500.0) == pytest.approx(0.0, abs=1e-15)


def test_a_range_that_leaves_the_table_is_refused(table):
    """An isentrope whose centre passes 2e4 K is refused at install, naming the pressure."""
    arrays, _ = gaussian_core()
    with pytest.raises(ValueError, match='leaves the liquid iron table'):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=(4000.0, 19000.0))


def test_the_core_mass_check(table):
    """``m_core`` must equal the last node mass to 1e-10."""
    arrays, _ = gaussian_core()
    m = arrays['m'][-1]
    TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE, m_core=m * (1 + 5e-11))
    with pytest.raises(ValueError, match='core arrays hold'):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE, m_core=m * (1 + 2e-10))


@pytest.mark.parametrize(
    ('key', 'mutate', 'msg'),
    [
        ('r', lambda a: a[::-1], 'start at the centre'),
        ('r', lambda a: np.r_[a[:5], a[4], a[6:]], 'strictly increasing'),
        ('m', lambda a: np.r_[a[:5], a[4], a[6:]], 'strictly increasing'),
        ('P', lambda a: np.r_[a[:5], a[4], a[6:]], 'strictly decreasing'),
        ('rho', lambda a: np.r_[a[:5], 0.0, a[6:]], 'density must be positive'),
        ('g', lambda a: -a, 'gravity non-negative'),
        ('rho', lambda a: a[:-1], 'has shape'),
        ('P', lambda a: np.r_[a[:5], np.nan, a[6:]], 'non-finite'),
    ],
)
def test_malformed_core_arrays_are_refused(table, key, mutate, msg):
    """Each malformed array is refused with a message that names the fault."""
    arrays, _ = gaussian_core(41)
    arrays[key] = mutate(arrays[key])
    with pytest.raises(ValueError, match=msg):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE)


def test_missing_arrays_and_too_few_nodes_are_refused(table):
    """All of r, m, P, rho, g are needed, on at least 4 nodes."""
    arrays, _ = gaussian_core(41)
    del arrays['g']
    with pytest.raises(ValueError, match=r"missing \['g'\]"):
        TabulatedCoreProfiles(arrays, table)
    arrays, _ = gaussian_core(3)
    with pytest.raises(ValueError, match='at least 4 nodes'):
        TabulatedCoreProfiles(arrays, table)


@settings(max_examples=30, deadline=None)
@given(
    c_p=st.one_of(
        st.sampled_from([0.0, -840.0, math.nan, math.inf, -math.inf, True, False, '840']),
    )
)
def test_a_bad_heat_capacity_override_is_refused(table, c_p):
    """The override must be a positive finite number or None; a TOML bool, a string, zero,
    a negative value, NaN and inf are refused."""
    arrays, _ = gaussian_core(41)
    with pytest.raises(ValueError, match='c_p must be'):
        TabulatedCoreProfiles(arrays, table, c_p=c_p, t_cmb_range=T_RANGE)


@settings(max_examples=30, deadline=None)
@given(
    lo=st.sampled_from([0.0, -1.0, math.nan, 7000.0, math.inf]),
    step=st.sampled_from([20.0, 0.0, -5.0, math.nan, math.inf]),
)
def test_a_bad_temperature_grid_is_refused(table, lo, step):
    """The grid needs 0 < low < high, both finite, and a positive finite step."""
    arrays, _ = gaussian_core(41)
    with pytest.raises(ValueError, match='bad CMB temperature grid'):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=(lo, 7000.0), t_cmb_step=step)


def test_extra_z2_keys_are_accepted(table):
    """The Zalmoxis keys the profile does not use (T, phase, w_S, scalars) are accepted."""
    arrays, _ = gaussian_core(41)
    arrays.update(T=np.full(41, 5000.0), phase=np.zeros(41), w_S=np.full(41, 0.1), M_ic=0.0)
    p = TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE)
    assert p.r_cmb == pytest.approx(EARTH['r_cmb'], rel=1e-15)
    assert sorted(FIELDS) == sorted(table.fields)
