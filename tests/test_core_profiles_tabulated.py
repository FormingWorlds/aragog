"""Tests of ``aragog.core.profiles_tabulated.TabulatedCoreProfiles``."""

from __future__ import annotations

import math
import pathlib
import re

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.optimize import brentq

from aragog.core.iron_table import FIELDS, LiquidIronTable
from aragog.core.module import load_core_arrays
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.profiles_tabulated import TabulatedCoreProfiles
from tests.conftest import PALEOS_CUT as CUT
from tests.conftest import gaussian_core_arrays

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

EARTH = dict(
    rho_cen=12500.0, length_scale=7272e3, r_cmb=3480e3, p_cmb=136e9, alpha=1.35e-5, c_p=840.0
)
T_RANGE = (2500.0, 7000.0)


def gaussian_core(n=201):
    """Core arrays of the Earth Gaussian profile on ``n`` uniform nodes, and the profile."""
    g = GaussianCoreProfiles(**EARTH)
    return gaussian_core_arrays(g, n), g


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
    temperature keeps ``s(P, T_a) = s(P_cmb, T_cmb)`` to 1e-6, equals T_cmb at
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
    """C_s > 0 over the grid, and a temperature outside the grid reads the
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


def _pressure_slice(table, lo, hi):
    """The table restricted to pressures in [lo, hi] Pa."""
    keep = (table.pressure >= lo) & (table.pressure <= hi)
    fields = {f: table.fields[f][keep] for f in FIELDS}
    return LiquidIronTable(table.pressure[keep], table.temperature, fields, table.header)


def test_a_core_outside_the_pressure_range_of_its_table_is_refused(table):
    """The pressure range comes from the table: cut at 1.5 TPa it holds the 3 M_E core (centre
    1.0 TPa) and refuses the 5 M_E core (2.1 TPa); from 200 GPa up it refuses the 1 M_E core
    (CMB 128 GPa). The message names the pressures of the core and of the table."""
    data = pathlib.Path(__file__).parent / 'validation' / 'data'
    core = {
        k: load_core_arrays(data / f'zalmoxis_core_arrays_{k}.npz')
        for k in ('1me_3000K', '3me_3500K', '5me_2510K')
    }
    top, bottom = _pressure_slice(table, 0.0, 1.5e12), _pressure_slice(table, 2e11, np.inf)
    TabulatedCoreProfiles(core['3me_3500K'], top)
    for name, tab in (('5me_2510K', top), ('1me_3000K', bottom)):
        p = core[name]['P']
        msg = (
            f'core pressures {p.min():.4e} to {p.max():.4e} Pa leave the pressure range of the '
            f'liquid iron table ({tab.pressure[0]:.4e} to {tab.pressure[-1]:.4e} Pa)'
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            TabulatedCoreProfiles(core[name], tab)


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


def test_the_isentrope_check_refuses_a_miss(table, monkeypatch):
    """Invariant I5 is asserted at install: with its bound set to 0 the round-off of the
    Newton inversion (about 1e-14) is refused, so the check runs on every node."""
    import aragog.core.profiles_tabulated as pt

    arrays, _ = gaussian_core(41)
    TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE)
    monkeypatch.setattr(pt, 'ISENTROPE_RTOL', 0.0)
    with pytest.raises(ValueError, match='isentrope misses s_c'):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE)


@pytest.mark.physics_invariant
def test_derivatives_at_the_ends_of_the_tables_are_the_one_sided_slopes(paleos):
    """At the CMB and at the edges of the temperature grid, autodiff gives the slope of the
    interpolant there, not half of it (a clamp whose gradient splits at the bound would)."""
    r_cmb = paleos.r_cmb
    inside = float(paleos.adiabat_gradient(r_cmb - 1.0, 4000.0))
    assert float(paleos.adiabat_gradient(r_cmb, 4000.0)) == pytest.approx(inside, rel=1e-4)
    d_rho = jax.grad(paleos.density)
    assert float(d_rho(r_cmb)) == pytest.approx(float(d_rho(r_cmb - 1.0)), rel=1e-4)
    d_cs = jax.grad(paleos.secular_capacity)
    for k in (0, -1):
        slope = float(paleos._c_s[1][k])
        assert float(d_cs(float(paleos.t_grid[k]))) == pytest.approx(slope, rel=1e-9)


def test_a_coarse_grid_that_misses_the_isentrope_between_grid_points_is_refused(table):
    """The entropy check runs at the midpoints of the temperature grid too: a grid of 2
    temperatures 4500 K apart misses the isentrope at their midpoint by 1.6e-6 (measured; a
    2250 K step misses by 1.2e-7 at the midpoints) and is refused, a 10 K step passes."""
    arrays, _ = gaussian_core(41)
    TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE, t_cmb_step=10.0)
    with pytest.raises(ValueError, match='between grid temperatures'):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE, t_cmb_step=4500.0)


def test_a_table_too_cold_for_any_isentrope_is_refused_with_its_range(table):
    """A table that ends at 2600 K holds no isentrope from CMB to centre of the Earth core
    (a cut at 3000 K still holds 1996 to 2003 K); the default range says so and names the
    table temperatures."""
    keep = table.temperature <= 2600.0
    small = LiquidIronTable(
        table.pressure,
        table.temperature[keep],
        {f: v[:, keep] for f, v in table.fields.items()},
        header='# Kind: liquid',
    )
    with pytest.raises(ValueError, match='no CMB temperature keeps the isentrope inside'):
        TabulatedCoreProfiles(gaussian_core(41)[0], small)


def test_check_t_cmb_refuses_outside_the_grid_and_nan(paleos):
    """``check_t_cmb`` passes the grid ends and refuses below, above, NaN and inf."""
    paleos.check_t_cmb([float(paleos.t_grid[0]), float(paleos.t_grid[-1])])
    for bad in (2400.0, 7100.0, math.nan, math.inf):
        with pytest.raises(ValueError, match='outside the tabulated isentropes'):
            paleos.check_t_cmb([4000.0, bad])


def test_the_install_size_is_bounded_by_its_cells(table):
    """The install refuses more than 2,000,000 node-temperature cells, whatever the split: 2001
    nodes on a 2 K step (4.5 million cells) are refused, on the default 10 K step (0.9 million)
    they are built."""
    arrays, _ = gaussian_core(2001)
    with pytest.raises(ValueError, match='at most 2000000 cells'):
        TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE, t_cmb_step=2.0)
    assert TabulatedCoreProfiles(arrays, table, t_cmb_range=T_RANGE).n_nodes == 2001


def test_a_core_too_coarse_for_the_isentrope_between_its_nodes_is_refused(table):
    """The entropy check also runs between the radial nodes: an Earth core on 7 nodes misses the
    isentrope there by 1.3e-5 and is refused; on 41 nodes (1.3e-8) it is built (measured; the
    Zalmoxis cores of the validation data miss by 4.2e-9 at most)."""
    TabulatedCoreProfiles(gaussian_core(41)[0], table, t_cmb_range=T_RANGE)
    with pytest.raises(ValueError, match='between radial nodes'):
        TabulatedCoreProfiles(gaussian_core(7)[0], table, t_cmb_range=T_RANGE)


def test_a_core_that_misses_the_isentrope_inside_a_cell_is_refused(table):
    """The entropy check covers the centres of the grid cells too: an Earth core on 15 nodes
    with one 4000 K temperature interval misses by 9.2e-7 between the temperatures and 6.2e-7
    between the nodes, both below 1e-6, but by 1.1e-6 at the cell centres, and is refused."""
    with pytest.raises(ValueError, match='inside the grid cells'):
        TabulatedCoreProfiles(
            gaussian_core(15)[0], table, t_cmb_range=(2500.0, 6500.0), t_cmb_step=4000.0
        )


def test_the_out_of_grid_message_tells_the_value_from_the_bound(paleos):
    """A temperature 0.004 K beyond either end prints a value that differs from the bound."""
    import re

    for bad in (float(paleos.t_grid[0]) - 0.004, float(paleos.t_grid[-1]) + 0.004):
        with pytest.raises(ValueError) as err:
            paleos.check_t_cmb([bad])
        pattern = r'T_core = ([0-9.]+) K .*\(([0-9.]+) to ([0-9.]+) K\)'
        value, lo, hi = re.search(pattern, str(err.value)).groups()
        assert value not in (lo, hi), str(err.value)
