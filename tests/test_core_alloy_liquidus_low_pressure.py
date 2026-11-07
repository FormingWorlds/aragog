"""Alloy liquidus of the core below the Simon fits: the Fe-S eutectic of Fei et al. (1997) joined
into Mori et al. (2017) over 18 to 21 GPa, FeS units up to 18 GPa blended into Fe3S units over 18
to 26 GPa, and the Fe-C eutectic from 1 bar joined into Mashino et al. (2019) at 5 GPa."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.optimize import brentq

from aragog.core import alloy
from aragog.core.alloy import AlloyLiquidus, binary_liquidus, eutectic
from aragog.core.melting import IronMeltingCurve
from tests.conftest import assert_legend_clear, opaque_legend

pytestmark = pytest.mark.unit

DATA = Path(__file__).parent / 'validation' / 'data' / 'core_eutectics' / 'low_pressure'
T_FE = IronMeltingCurve.t_melt_pure
MOLAR = {'Fe': 55.845, 'S': 32.06}
FES, FE3S = (1.0, 1.0), (3.0, 1.0)


def _csv(name):
    """Rows of a data file as lists of strings, without the comment lines and the header."""
    lines = (DATA / name).read_text().splitlines()
    return [ln.split(',') for ln in lines if not ln.startswith('#')][1:]


def _w_s(x_fes):
    """Mass fraction of S from the mole fraction of FeS in Fe-FeS."""
    return MOLAR['S'] * x_fes / (MOLAR['Fe'] + MOLAR['S'] * x_fes)


def _form(w, p_gpa, unit):
    """Fe-S liquidus [K] of the ideal form on ``unit`` through pure Fe and the shipped eutectic."""
    t_e, w_e = (float(v) for v in eutectic('S', p_gpa * 1e9))
    return float(binary_liquidus('S', w, T_FE(p_gpa * 1e9), t_e, w_e, unit))


def _slopes(f, p, h=1e-6):
    """One-sided slopes (left, right) of ``f`` at ``p`` [GPa], per GPa."""
    return (f(p) - f(p - h)) / h, (f(p + h) - f(p)) / h


def _t_e(el):
    return lambda p: float(eutectic(el, p * 1e9)[0])


def _w_e(el):
    return lambda p: float(eutectic(el, p * 1e9)[1])


def _smootherstep(z):
    return float(alloy._smootherstep(z))


def _blend(p_gpa):
    """Weight of the Fe3S form at ``p_gpa``, the smootherstep over the unit blend window."""
    s0, s1 = alloy.S_UNIT_BLEND_GPA
    return _smootherstep((p_gpa - s0) / (s1 - s0))


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_fe_s_eutectic_nodes_are_the_mean_of_two_reads_of_fei_1997():
    """At 1 bar, 10 and 14 GPa the eutectic is the mean of 2 reads of Fei et al. (1997, Fig. 2),
    which agree to 1.5 C and 0.03 wt%; at 21 GPa it is 1348 K (Fei et al. 2000, the start of the
    Mori fit) and from the Mori table start (20 GPa) the composition is the table."""
    for p, ta, tb, wa, wb in (map(float, r) for r in _csv('fei1997_fig2_eutectic.csv')):
        assert abs(ta - tb) < 1.6 and abs(wa - wb) < 0.035
        t_e, w_e = (float(v) for v in eutectic('S', p * 1e9))
        assert t_e == pytest.approx(0.5 * (ta + tb) + 273.15, rel=1e-12)
        assert w_e == pytest.approx(0.5 * (wa + wb) / 100.0, rel=1e-12)
    assert _t_e('S')(21.0) == pytest.approx(1348.0, rel=1e-12)
    tab = alloy._table('fe_s_composition')
    for p, w in tab[:3]:
        assert _w_e('S')(p) == pytest.approx(w, rel=1e-12)


def _hermite_by_hand(p, x0, x1, y0, y1, d0, d1):
    """Cubic Hermite on [x0, x1] with the end values y and slopes d, written out."""
    h, t = x1 - x0, (p - x0) / (x1 - x0)
    return (
        (2 * t**3 - 3 * t**2 + 1) * y0
        + (t**3 - 2 * t**2 + t) * h * d0
        + (-2 * t**3 + 3 * t**2) * y1
        + (t**3 - t**2) * h * d1
    )


def _fritsch_carlson(h0, h1, s0, s1):
    """Interior node slope of the monotone Hermite: the weighted harmonic mean of the 2 secants,
    0 where they change sign."""
    return (
        0.0 if s0 * s1 <= 0.0 else 3.0 * (h0 + h1) / ((2 * h1 + h0) / s0 + (h1 + 2 * h0) / s1)
    )


def _s_hermite_by_hand():
    """Nodes and slopes of T_e [K] and w_e of Fe-S below the join: 1 bar end slope = first
    chord, end slope = the Simon fit at 21 GPa and the first table segment at 20 GPa."""
    tab = alloy._table('fe_s_composition')
    t = ((0.0, 10.0, 14.0, 21.0), (1261.25, 1187.9, 1135.25, 1348.0), 1348.0 / (36.5 * 2.07))
    w_end = (tab[1, 1] - tab[0, 1]) / (tab[1, 0] - tab[0, 0])
    w = ((0.0, 10.0, 14.0, tab[0, 0]), (0.31175, 0.2057, 0.18305, tab[0, 1]), w_end)
    out = []
    for x, y, end in (t, w):
        s = [(y[i + 1] - y[i]) / (x[i + 1] - x[i]) for i in range(3)]
        h = [x[i + 1] - x[i] for i in range(3)]
        d = [
            s[0],
            _fritsch_carlson(h[0], h[1], s[0], s[1]),
            _fritsch_carlson(h[1], h[2], s[1], s[2]),
            end,
        ]
        out.append((x, y, d))
    return out


def _from_nodes(p, x, y, d):
    i = 0 if p < x[1] else (1 if p < x[2] else 2)
    return _hermite_by_hand(p, x[i], x[i + 1], y[i], y[i + 1], d[i], d[i + 1])


@pytest.mark.physics_invariant
def test_the_fe_s_eutectic_is_the_monotone_hermite_through_the_nodes():
    """T_e and w_e are the monotone cubic Hermite (Fritsch-Carlson) through the nodes and the
    line ends (1348 K at 21 GPa; the composition table at 20 GPa): the first chord at 1 bar, the
    weighted harmonic mean of the secants at 10 GPa, slope 0 at the 14 GPa minimum of T_e and
    the slope of the target at the join end. T_e, w_e and the liquidus of 2, 8 and 12 wt% S have
    no slope step at 10 and 14 GPa; T_e lies up to 25.94 K below the chord from 14 to 21 GPa, at
    15.91 GPa."""
    (tx, ty, td), (wx, wy, wd) = _s_hermite_by_hand()
    assert td[1:3] == pytest.approx([-9.819165, 0.0], abs=1e-6)
    for p in (5.0, 12.0, 16.0, 17.5):
        assert _t_e('S')(p) == pytest.approx(_from_nodes(p, tx, ty, td), rel=1e-12)
        assert _w_e('S')(p) == pytest.approx(_from_nodes(p, wx, wy, wd), rel=1e-12)
    for i, p in ((1, 10.0), (2, 14.0)):
        assert _slopes(_t_e('S'), p) == pytest.approx((td[i], td[i]), abs=1e-3)
        assert _slopes(_w_e('S'), p) == pytest.approx((wd[i], wd[i]), abs=1e-7)
        for w in (0.02, 0.08, 0.12):
            liq = AlloyLiquidus({'S': w})
            left, right = _slopes(lambda q: float(liq.t_melt(q * 1e9)), p)
            assert abs(left - right) < 0.01
    p = np.arange(14.0, 21.0001, 0.01)
    below = (
        1135.25 + (p - 14.0) * (1348.0 - 1135.25) / 7.0 - np.array([_t_e('S')(x) for x in p])
    )
    assert below.max() == pytest.approx(25.94, abs=0.01)
    assert p[below.argmax()] == pytest.approx(15.91, abs=0.015)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('el', 'p_gpa', 't_slope', 'w_slope'),
    [
        ('S', 18.0, 41.746026, -0.0058937),
        ('S', 21.0, 1348.0 / (2.07 * 36.5), -0.000592),
        ('C', 5.0, 1513.0 / (4.1 * 8.4), None),
    ],
)
def test_the_eutectic_joins_the_fits_with_a_continuous_slope(el, p_gpa, t_slope, w_slope):
    """At the edges of the S join (18 and 21 GPa) and at the C join (5 GPa) the slopes of T_e
    [K/GPa] and w_e [1/GPa] are continuous: the low-pressure Hermite at 18 GPa, the Simon fit
    and the first segment of the table at 21 and 5 GPa."""
    for f, slope, tol in ((_t_e(el), t_slope, 2e-3), (_w_e(el), w_slope, 2e-7)):
        if slope is None:
            continue
        left, right = _slopes(f, p_gpa)
        assert left == pytest.approx(right, abs=tol)
        assert right == pytest.approx(slope, abs=tol)


@pytest.mark.physics_invariant
def test_the_fe_s_eutectic_inside_the_join_is_the_smootherstep_mean():
    """Inside 18 to 21 GPa the eutectic is (1 - b) low-pressure Hermite + b fit, with b the
    smootherstep of (P - 18)/3; at 19.5 GPa b = 1/2."""
    tab = alloy._table('fe_s_composition')
    (tx, ty, td), (wx, wy, wd) = _s_hermite_by_hand()
    line_t, line_w = _from_nodes(19.5, tx, ty, td), _from_nodes(19.5, wx, wy, wd)
    simon = 1348.0 * ((19.5 - 21.0) / 36.5 + 1.0) ** (1.0 / 2.07)
    first = tab[0, 1] + (19.5 - tab[0, 0]) * (tab[1, 1] - tab[0, 1]) / (tab[1, 0] - tab[0, 0])
    assert _smootherstep(0.5) == 0.5
    assert _t_e('S')(19.5) == pytest.approx(0.5 * (line_t + simon), rel=1e-12)
    assert _w_e('S')(19.5) == pytest.approx(0.5 * (line_w + first), rel=1e-12)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_fe_c_eutectic_below_5_gpa_is_the_monotone_cubic():
    """From 1 bar (1416.15 K, 4.3 wt% C: the Fe-graphite eutectic of Bens & Elliott 1961 as
    Nakajima et al. 2009, p. 207 give it) to the Mashino fit at 5 GPa, T_e and w_e are the cubic
    with zero curvature at 1 bar and the value of the fit and the table at 5 GPa, with the slope
    of the fit (43.93 K/GPa) and, for w_e, the slope 0, since the table falls where the chord
    rises: both are monotone (Fritsch-Carlson). T_e is 1416.15, 1423.73, 1450.68, 1475.95,
    1513.00 K at 0, 1, 3, 4, 5 GPa, rising by 7.09 to 43.93 K/GPa, 2.1 times the 21.0 K/GPa of
    Fei & Brosh (2014) from 5 to 10 GPa at the top; Lord et al. (2009) give 1420 K at 1 bar. The
    w_e slope steps from 0 to the table at 5 GPa, and the liquidus slope of 2 wt% C with it
    (36.56 to 34.93 K/GPa)."""
    tab = alloy._table('fe_c_composition')

    def cubic(p, y0, y1, s1):
        d = (s1 * 5.0 - (y1 - y0)) / 250.0
        return y0 + (s1 - 75.0 * d) * p + d * p**3

    for p, t_stated in ((0.0, 1416.15), (1.0, 1423.73), (3.0, 1450.68), (4.0, 1475.95)):
        t_e, w_e = (float(v) for v in eutectic('C', p * 1e9))
        assert t_e == pytest.approx(cubic(p, 1416.15, 1513.0, 1513.0 / 34.44), rel=1e-12)
        assert t_e == pytest.approx(t_stated, abs=0.01)
        assert w_e == pytest.approx(cubic(p, 0.043, tab[0, 1], 0.0), rel=1e-12)
    w = np.array([_w_e('C')(p) for p in np.linspace(0.0, 5.0, 501)])
    assert np.all(np.diff(w) >= 0.0)
    assert _slopes(_w_e('C'), 5.0) == pytest.approx(
        (0.0, (tab[1, 1] - tab[0, 1]) / (tab[1, 0] - tab[0, 0])), abs=2e-7
    )
    liq = AlloyLiquidus({'C': 0.02})
    left_right = _slopes(lambda p: float(liq.t_melt(p * 1e9)), 5.0, 1e-4)
    assert left_right == pytest.approx((36.56, 34.93), abs=0.01)
    slopes = np.diff([_t_e('C')(p) for p in np.linspace(0.0, 5.0, 501)]) / 0.01
    assert slopes.min() > 7.0 and slopes.max() < 43.94
    fei = [
        (float(r[2]), float(r[3]))
        for r in _csv('low_pressure_eutectic_checks.csv')
        if r[0].startswith('Fei & Brosh')
    ]
    assert (fei[1][1] - fei[0][1]) / (fei[1][0] - fei[0][0]) == pytest.approx(21.0, abs=1e-9)
    assert abs(_t_e('C')(0.0) - 1420.0) < 4.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('el', ['S', 'C'])
def test_the_s_and_c_data_start_at_1_bar(el):
    """The S and C data reach down to 1 bar, so nothing below 21 or 5 GPa is flagged."""
    assert alloy._data()['range'][el][0] == 0.0
    flags = AlloyLiquidus({el: 0.01}).status(np.array([0.0, 3.0, 10.0, 20.0]) * 1e9)
    assert not flags['extrapolated'].any()


@pytest.mark.physics_invariant
@pytest.mark.parametrize('w', [0.01, 0.04, 0.12])
def test_the_sulfur_units_are_fes_below_18_and_fe3s_above_26_gpa(w):
    """Below 18 GPa the Fe-S liquidus is the ideal form on FeS units, above 26 GPa on Fe3S units,
    and at 22 GPa, the middle of the blend, the mean of both."""
    for p_gpa in (0.0, 6.0, 14.0, 17.99):
        t = float(AlloyLiquidus({'S': w}).t_melt(p_gpa * 1e9))
        assert t == pytest.approx(_form(w, p_gpa, FES), rel=1e-12)
    for p_gpa in (26.0, 60.0):
        t = float(AlloyLiquidus({'S': w}).t_melt(p_gpa * 1e9))
        assert t == pytest.approx(_form(w, p_gpa, FE3S), rel=1e-12)
    mid = 0.5 * (_form(w, 22.0, FES) + _form(w, 22.0, FE3S))
    assert float(AlloyLiquidus({'S': w}).t_melt(22e9)) == pytest.approx(mid, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('w', 'added', 'top'),
    [(0.02, 29.6, 51.4), (0.04, 53.6, 73.9), (0.08, 81.5, 98.4), (0.10, 81.8, 96.7)]
    + [(0.12, 67.4, 80.0)],
)
def test_the_unit_blend_adds_the_stated_slope(w, added, top):
    """Over 18 to 26 GPa the blend adds c'(P) (T_Fe3S - T_FeS) to the weighted slope of the 2
    unit forms, at most 29.6, 53.6, 81.5, 81.8 and 67.4 K/GPa near 22 GPa for 2, 4, 8, 10 and 12
    wt% S; the liquidus slope peaks at 51.4, 73.9, 98.4, 96.7 and 80.0 K/GPa."""
    p = np.linspace(18.001, 25.999, 801)
    g = np.asarray(AlloyLiquidus({'S': w}).gradient(p * 1e9)) * 1e9
    c = np.array([_blend(x) for x in p])
    ends = [
        np.array([(_form(w, x + 1e-4, u) - _form(w, x - 1e-4, u)) / 2e-4 for x in p])
        for u in (FES, FE3S)
    ]
    extra = g - ((1.0 - c) * ends[0] + c * ends[1])
    assert extra.max() == pytest.approx(added, abs=0.2)
    assert abs(p[extra.argmax()] - 22.0) < 0.2
    assert g.max() == pytest.approx(top, abs=0.2)


@pytest.mark.physics_invariant
def test_the_dilute_depression_of_s_falls_over_the_unit_blend():
    """The depression of the liquidus per wt% S in the dilute limit, (T_Fe - T) / w at 0.01 wt%,
    falls from 92.72 K at 18 GPa (FeS units) to 26.80 K at 26 GPa (Fe3S units), 3.46 times."""
    dil = [
        (float(T_FE(p * 1e9)) - float(AlloyLiquidus({'S': 1e-4}).t_melt(p * 1e9))) / 0.01
        for p in (18.0, 26.0)
    ]
    assert dil == pytest.approx([92.72, 26.80], abs=0.01)
    assert dil[0] / dil[1] == pytest.approx(3.46, abs=0.005)


def _a(el, p_gpa, unit):
    """A of the Fe-``el`` ideal form on ``unit`` through pure Fe and the shipped eutectic."""
    t_e, w_e = (float(v) for v in eutectic(el, p_gpa * 1e9))
    return float(alloy._binary_a(el, float(T_FE(p_gpa * 1e9)), t_e, w_e, unit))


@pytest.mark.physics_invariant
def test_the_stated_forms_a_values_dilute_slopes_and_crossings():
    """The numbers the docs state: the Fe3S and FeS forms differ by up to 457.7 K (10.32 wt% S)
    at 18 GPa and 357.7 K (9.15 wt%) at 21 GPa; A is 3.497 and 0.454 for S on FeS at 1 bar and
    21 GPa, 2.467, 1.788 and 0.213 on Fe3S at 18, 21 and 330 GPa, at most 1.728 (16 GPa) and
    0.503 (17 GPa) for C on Fe3C and Fe7C3 (0.401 and 0.122 at 330 GPa), 1.074 for O at 1 bar
    and at most 3.449 (50 GPa); the dilute slopes (T_Fe - T) per wt% at 1e-6 wt%; the tutorial
    alloy and 10 wt% O leave their eutectic at 20.535 and 49.752 GPa, the slope stepping from
    27.97 to 41.49 and from 10.24 to 15.90 K/GPa."""
    for p, diff, w_at in ((18.0, 457.7, 10.32), (21.0, 357.7, 9.15)):
        w = np.linspace(0.001, _w_e('S')(p), 2000)
        d = np.abs([_form(x, p, FE3S) - _form(x, p, FES) for x in w])
        assert (d.max(), 100 * w[d.argmax()]) == pytest.approx((diff, w_at), abs=0.05)
    stated = [(0, (1, 1), 3.497), (21, (1, 1), 0.454), (18, (3, 1), 2.467), (21, (3, 1), 1.788)]
    for p, unit, a in stated + [(330, (3, 1), 0.213)]:
        assert _a('S', p, unit) == pytest.approx(a, abs=5e-4)
    grid = np.arange(5.0, 330.01, 1.0)
    for unit, top, p_top, end in (((3, 1), 1.728, 16, 0.401), ((7, 3), 0.503, 17, 0.122)):
        a = np.array([_a('C', x, unit) for x in grid])
        assert (a.max(), grid[a.argmax()], a[-1]) == pytest.approx((top, p_top, end), abs=5e-4)
    a = np.array([_a('O', x, (1, 1)) for x in np.arange(0.0, 325.01, 1.0)])
    assert (a[0], a.max(), float(a.argmax())) == pytest.approx((1.074, 3.449, 50.0), abs=5e-4)
    table = {
        'S': (9.1, 73.5, 124.7, 510.4),
        'O': (59.3, 33.0, 66.8, 137.7),
        'C': (54.7, 65.7, 226.9, 792.0),
    }
    for el, row in table.items():
        for p, stated_k in zip((0.0, 21.0, 135.0, 330.0), row):
            t = float(AlloyLiquidus({el: 1e-6}).t_melt(p * 1e9))
            assert (float(T_FE(p * 1e9)) - t) / 1e-4 == pytest.approx(stated_k, abs=0.05)
    crossings = (
        ({'S': 0.02, 'C': 0.005, 'O': 0.03, 'Si': 0.04}, (20.0, 21.0), 20.535, 27.97, 41.49),
        ({'O': 0.10}, (49.0, 50.5), 49.752, 10.24, 15.90),
    )
    for comp, bracket, p_x, before, after in crossings:
        liq = AlloyLiquidus(comp)
        x = brentq(
            lambda q: float(liq.status(np.array([q * 1e9]))['past_eutectic'][0]) - 0.5,
            *bracket,
            xtol=1e-6,
        )
        g = np.asarray(liq.gradient(np.array([x - 1e-3, x + 1e-3]) * 1e9)) * 1e9
        assert (x, *g) == pytest.approx((p_x, before, after), abs=0.005)


@pytest.mark.physics_invariant
def test_the_fe3s_units_hold_the_eutectic_from_the_start_of_the_blend():
    """At 18 GPa w_e is 15.50 wt% S, below the 16.06 wt% that Fe3S units can hold (w_e crosses
    it at 17.14 GPa), and it stays below over the whole blend, so the Fe3S form is defined
    there; its x_Fe at the eutectic is 0.115 at 18 GPa."""
    cap = MOLAR['S'] / (3.0 * MOLAR['Fe'] + MOLAR['S'])
    w_e = _w_e('S')(18.0)
    assert w_e == pytest.approx(0.15498, abs=1e-5)
    assert _w_e('S')(17.13) > cap > _w_e('S')(17.14)
    assert all(_w_e('S')(p) < cap for p in np.linspace(*alloy.S_UNIT_BLEND_GPA, 81))
    u = w_e / MOLAR['S']
    free = (1.0 - w_e) / MOLAR['Fe'] - 3.0 * u
    assert free / (free + u) == pytest.approx(0.1151, abs=2e-4)


@pytest.mark.parametrize(
    'comp', [{'S': 0.02}, {'S': 0.15}, {'S': 0.02, 'C': 0.01, 'O': 0.03}, {'C': 0.03}]
)
def test_a_low_pressure_alloy_has_a_nan_free_gradient(comp):
    """Below 30 GPa, where the Fe3S form cannot hold the eutectic and the join and blend act,
    the liquidus gradient passes jax.debug_nans and is finite."""
    p = np.array([0.0, 2.0, 5.0, 10.0, 14.0, 17.99, 18.0, 18.01, 19.5, 21.0, 22.0, 26.0, 29.0])
    with jax.debug_nans(True):
        g = np.asarray(AlloyLiquidus(comp).gradient(p * 1e9))
    assert np.all(np.isfinite(g))


def _liquidus_points():
    """Measured liquidus points (P GPa, w, T K) below 15 GPa with S and on the Fe-rich side of the
    shipped eutectic: read A of the 1 bar points of Buono & Walker (2011, Fig. 6), their Table 1 at
    6 GPa, and read A of Chen et al. (2008, Fig. 1) at 10 and 14 GPa."""
    one_bar = [
        (0.0, _w_s(float(x)), float(t))
        for _, x, t, *_ in _csv('buono_walker2011_fig6_1bar.csv')
    ]
    bw = [
        (6.0, _w_s(float(x)), float(t)) for _, t, x in _csv('buono_walker2011_table1_6GPa.csv')
    ]
    chen = [
        (float(p), float(w) / 100.0, float(t))
        for p, kind, w, t, *_ in _csv('chen2008_fig1_liquidus.csv')
        if kind == 'liquid'
    ]
    return [(p, w, t) for p, w, t in one_bar + bw + chen if 0.0 < w < _w_e('S')(p)]


@pytest.mark.reference_pinned
def test_the_two_reads_of_the_1_bar_points_agree(validation_plot):
    """Read A (vector marker centres) and read B (600 dpi raster, markers found without read A) of
    the 23 points of Buono & Walker (2011, Fig. 6) agree within 0.5 pt of the figure (357.08 pt per
    unit X_FeS, 0.22779 pt per K); the 2 pure-iron points and the 2 points past the shipped 1 bar
    eutectic (X_FeS 0.81 and 0.86) are left out of the misfit."""
    rows = [(s, *map(float, r)) for s, *r in _csv('buono_walker2011_fig6_1bar.csv')]
    assert len(rows) == 23
    for _, xa, ta, xb, tb in rows:
        assert abs(xa - xb) * 357.08 < 0.5 and abs(ta - tb) * 0.22779 < 0.5
    assert sum(_w_s(xa) >= _w_e('S')(0.0) for _, xa, *_ in rows) == 2
    assert sum(xa == 0.0 for _, xa, *_ in rows) == 2

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        w = np.linspace(0.0, _w_e('S')(0.0), 12)
        liq = AlloyLiquidus({})
        ax.plot(100 * w, [float(liq.t_melt(0.0, {'S': x})) for x in w], label='Aragog')
        for src, mark in (('Friedrich 1910', '^'), ('Miyazaki 1928', 's')):
            pts = np.array([(100 * _w_s(xa), ta) for s, xa, ta, *_ in rows if s == src]).T
            ax.plot(*pts, mark, ms=4, mfc='none', label=src)
        ax.set(xlabel='S [wt%]', ylabel='Liquidus at 1 bar [K]')
        opaque_legend(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'core_alloy_liquidus_one_bar')


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('p_gpa', 'n', 'rms', 'mean'),
    [(0.0, 19, 78.60, -33.32), (6.0, 21, 153.15, -141.36), (10.0, 7, 88.95, -82.55)]
    + [(14.0, 4, 150.90, -12.45)],
)
def test_the_low_pressure_liquidus_misses_the_measured_points_as_stated(p_gpa, n, rms, mean):
    """The ideal form on FeS units lies below the measured non-ideal liquidus: RMS and mean misfit
    [K] at 1 bar (Friedrich 1910 and Miyazaki 1928 in Buono & Walker 2011, Fig. 6), at 6 GPa
    (Buono & Walker 2011, Table 1) and at 10 and 14 GPa (Chen et al. 2008), to 0.05 K."""
    pts = [(w, t) for p, w, t in _liquidus_points() if p == p_gpa]
    res = np.array([float(AlloyLiquidus({'S': w}).t_melt(p_gpa * 1e9)) - t for w, t in pts])
    assert len(pts) == n
    assert math.sqrt(np.mean(res**2)) == pytest.approx(rms, abs=0.05)
    assert res.mean() == pytest.approx(mean, abs=0.05)


@pytest.mark.reference_pinned
def test_the_two_reads_of_the_chen_points_agree():
    """Read A (vector symbols) and read B (600 dpi raster) of the 11 points of Chen et al. (2008,
    Fig. 1) read twice agree to 0.28 wt% and 3.2 K."""
    both = [r for r in _csv('chen2008_fig1_liquidus.csv') if r[4]]
    assert len(both) == 11
    for _, _, wa, ta, wb, tb in both:
        assert abs(float(wa) - float(wb)) <= 0.28 and abs(float(ta) - float(tb)) <= 3.2


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_from_10_to_14_gpa_the_liquidus_falls_from_a_higher_s_content_than_the_data(
    validation_plot,
):
    """From 10 to 14 GPa the measured liquidus of Chen et al. (2008, linear between their points
    and their pure-Fe point) changes by -42.05, -39.06 and -15.53 K/GPa at 4, 8 and 12 wt% S,
    the ideal form by +3.05, -14.49 and -26.89 K/GPa. The liquidus falls from 10 to 14 GPa from
    4.61 wt% S in the code and from 1.01 wt% S in the data, where the 10 GPa data are the chord
    from the pure-Fe point (0.002 wt%) to the first liquid point (7.05 wt%); the code's pure
    iron melts 12.9 and 35.0 K above the pure-Fe points at 10 and 14 GPa."""
    rows = _csv('chen2008_fig1_liquidus.csv')

    def data(p, w):
        pts = sorted((float(r[2]), float(r[3])) for r in rows if float(r[0]) == p)
        return np.interp(w, *zip(*pts))

    def code(w):
        liq = AlloyLiquidus({'S': w / 100.0})
        return (float(liq.t_melt(14e9)) - float(liq.t_melt(10e9))) / 4.0

    assert brentq(code, 1.0, 8.0) == pytest.approx(4.61, abs=0.005)
    assert brentq(lambda w: data(14.0, w) - data(10.0, w), 0.5, 1.5) == pytest.approx(
        1.01, abs=0.005
    )
    for p, t_fe, offset in ((10.0, 2117.9, 12.9), (14.0, 2203.3, 35.0)):
        assert data(p, 0.0) == pytest.approx(t_fe, abs=0.05)
        assert float(T_FE(p * 1e9)) - t_fe == pytest.approx(offset, abs=0.05)

    for w, stated_data, stated_code in (
        (4, -42.05, 3.05),
        (8, -39.06, -14.49),
        (12, -15.53, -26.89),
    ):
        assert (data(14.0, w) - data(10.0, w)) / 4.0 == pytest.approx(stated_data, abs=0.005)
        assert code(w) == pytest.approx(stated_code, abs=0.05)

    def figure():
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        pts, liq = _liquidus_points(), AlloyLiquidus({})
        for p_gpa in (6.0, 10.0, 14.0):
            w = np.linspace(0.0, float(eutectic('S', p_gpa * 1e9)[1]), 12)
            t = [float(liq.t_melt(p_gpa * 1e9, {'S': x})) for x in w]
            (line,) = ax.plot(100 * w, t, label=f'{p_gpa:g}')
            meas = np.array([(100 * x, t) for p, x, t in pts if p == p_gpa]).T
            ax.plot(*meas, 'o', ms=3, mfc='none', color=line.get_color())
        ink = plt.rcParams['text.color']
        keys = [
            Line2D([], [], color=ink, alpha=0.55, label='Aragog'),
            Line2D(
                [],
                [],
                color=ink,
                alpha=0.55,
                ls='none',
                marker='o',
                mfc='none',
                ms=3,
                label='measured',
            ),
        ]
        handles = ax.get_legend_handles_labels()[0] + keys
        opaque_legend(
            ax, handles=handles, title='P [GPa]', loc='upper left', bbox_to_anchor=(1.02, 1.0)
        )
        ax.set(xlabel='S [wt%]', ylabel='Liquidus [K]')
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'core_alloy_liquidus_low_pressure_liquidus')


# code minus source, T_e [K] and w_e [wt%], by source and pressure [GPa] of low_pressure_eutectic_checks.csv
_CHECK_MISS = {
    ('Buono & Walker 2011 Table 1', '6'): (-42.18, -1.443),
    ('Ryzhenko & Kennedy 1973 via Buono & Walker 2011 Table 1', '6'): (-42.18, -1.317),
    ('Usselman 1975 via Buono & Walker 2011 Table 1', '6'): (-38.18, -3.445),
    ('Morard et al. 2007 via Buono & Walker 2011 Table 1', '6'): (80.82, -2.780),
    ('Fei et al. 2000 p. 1831', '21'): (0.0, -0.669),
    ('Kamada et al. 2010 p. 98; T_e = 1030(40) + 13.4(0.7) P', '21'): (36.60, None),
    ('Kamada et al. 2010 p. 98; T_e = 1030(40) + 13.4(0.7) P', '25'): (52.45, None),
    ('Chudinovskikh & Boehler 2007 Table 1', '21.9'): (-148.04, 1.678),
    ('Chudinovskikh & Boehler 2007 Table 1', '23.0'): (-66.81, None),
    ('Lord et al. 2009 p. 158 (after Chipman 1972)', '0'): (-3.85, 0.0),
    ('Fei & Brosh 2014 p. 158', '5'): (0.0, 0.311),
    ('Fei & Brosh 2014 p. 158', '10'): (77.54, 0.505),
}


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_low_pressure_eutectic_against_the_check_points(validation_plot):
    """Eutectic points that set no node, code minus source: T_e [K] and w_e [wt%]; the 6 GPa
    eutectic is 42 K and 1.4 wt% below Buono & Walker (2011), Kamada et al. (2010) lie inside
    their fit errors, the 21.9 and 23.0 GPa quench runs (upper bounds of T_e) lie above, the liquid
    at 21.9 GPa (on the Fe-rich liquidus) holds less S than w_e and lies 55.62 K below the
    liquidus at its 13 wt% S, and Fe-C at 10 GPa is the Mashino
    fit, 77.5 K above Fei & Brosh (2014). The 2 anchors of the fits (Fei et al. 2000 at 21 GPa,
    Fei & Brosh 2014 at 5 GPa) check the composition only."""
    rows = _csv('low_pressure_eutectic_checks.csv')
    assert {(r[0], r[2]) for r in rows} == set(_CHECK_MISS) and len(rows) == len(_CHECK_MISS)
    assert [r[2] for r in rows if r[8] == 'anchor'] == ['21', '5']
    for src, el, p, t, t_err, comp, _, unit, kind in rows:
        dt, dw = _CHECK_MISS[(src, p)]
        t_e, w_e = (float(v) for v in eutectic(el, float(p) * 1e9))
        assert t_e - float(t) == pytest.approx(dt, abs=0.02), src
        if kind == 'fit':
            assert abs(t_e - float(t)) < float(t_err), src
        if kind == 'upper_bound_T':
            assert t_e < float(t), src
            assert not comp or w_e > float(comp) / 100.0, src
            if comp:  # the liquid on the Fe-rich liquidus: the liquidus at its S content
                liq = float(AlloyLiquidus({'S': float(comp) / 100.0}).t_melt(float(p) * 1e9))
                assert liq - float(t) == pytest.approx(55.62, abs=0.02), src
        if comp:
            ref = _w_s(float(comp)) if unit == 'X_FeS' else float(comp) / 100.0
            assert 100.0 * (w_e - ref) == pytest.approx(dw, abs=0.002), src

    p = np.unique(np.r_[0.0:40.0:2.0, 3.0:5.0:0.5, 10.0, 14.0, 18.0:21.0:0.5, 40.0])
    fei = np.array([tuple(map(float, r)) for r in _csv('fei1997_fig2_eutectic.csv')])
    nodes = (fei[:, 1:3].mean(1) + 273.15, fei[:, 3:5].mean(1))

    def figure(col):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        scale = 1.0 if col == 0 else 100.0
        for el in ('S', 'C'):
            (line,) = ax.plot(
                p, [scale * float(eutectic(el, x * 1e9)[col]) for x in p], label=f'Fe-{el}'
            )
            chk = [r for r in rows if r[1] == el and r[3 + 2 * col]]
            y = [
                float(r[3])
                if col == 0
                else 100 * _w_s(float(r[5]))
                if r[7] == 'X_FeS'
                else float(r[5])
                for r in chk
            ]
            err = [float(r[4]) if r[4] and col == 0 else 0.0 for r in chk]
            ax.errorbar(
                [float(r[2]) for r in chk],
                y,
                yerr=err,
                fmt='s',
                ms=3,
                mfc='none',
                color=line.get_color(),
                alpha=0.7,
            )
        ax.plot(fei[:, 0], nodes[col], 'o', ms=4, color='C0', label='nodes')
        ax.errorbar(
            [], [], yerr=[], fmt='s', ms=3, mfc='none', color='C0', alpha=0.7, label='checks'
        )
        ax.set(
            xlabel='P [GPa]', ylabel='Eutectic T [K]' if col == 0 else 'Eutectic liquid [wt%]'
        )
        opaque_legend(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(lambda: figure(0), 'core_alloy_liquidus_low_pressure_eutectic_t')
    validation_plot(lambda: figure(1), 'core_alloy_liquidus_low_pressure_eutectic_w')


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('w', 'slope_min', 'p_min', 'falling'),
    [
        (0.01, 19.531, 13.15, None),
        (0.03, 7.241, 12.56, None),
        (0.048, -2.368, 12.35, (10.78, 13.79)),
        (0.08, -16.413, 12.07, (4.99, 15.57)),
        (0.12, -29.381, 11.77, (1.74, 16.41)),
    ],
)
def test_below_26_gpa_the_s_liquidus_falls_where_the_eutectic_falls(
    w, slope_min, p_min, falling
):
    """Up to 14 GPa T_e falls (Fei et al. 1997) and it rises slowly from its minimum there, so S
    alloys from about 4 wt% fall with pressure around 14 GPa: the least slope [K/GPa] and the
    first and last falling pressures [GPa] on a 0.01 GPa grid; 1 and 3 wt% S keep rising."""
    p = np.arange(0.01, 26.0, 0.01)
    g = np.asarray(AlloyLiquidus({'S': w}).gradient(p * 1e9)) * 1e9
    assert g.min() == pytest.approx(slope_min, abs=0.02)
    assert p[g.argmin()] == pytest.approx(p_min, abs=0.015)
    down = p[g <= 0.0]
    if falling is None:
        assert down.size == 0
    else:
        assert (down[0], down[-1]) == pytest.approx(falling, abs=0.015)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('p_gpa', [1.0, 3.0, 10.0])
def test_below_18_gpa_the_fe_c_liquidus_is_the_ideal_form_on_fe3c_units(p_gpa):
    """Below the S blend the Fe-C liquidus is the ideal form on Fe3C units through pure Fe and the
    eutectic, written out by hand; on C atoms it is another curve."""
    t_e, w_e = (float(v) for v in eutectic('C', p_gpa * 1e9))
    t_fe = float(T_FE(p_gpa * 1e9))

    def x_fe(w, k):
        u = w / 12.011
        free = (1.0 - w) / MOLAR['Fe'] - k * u
        return free / (free + u)

    for k, tol in ((3.0, 1e-12), (0.0, None)):
        a = math.log(x_fe(w_e, k)) / (1.0 - t_fe / t_e)
        for w in (0.4 * w_e, 0.8 * w_e):
            hand = t_fe / (1.0 - math.log(x_fe(w, k)) / a)
            code = float(AlloyLiquidus({'C': w}).t_melt(p_gpa * 1e9))
            if tol:
                assert code == pytest.approx(hand, rel=tol)
            else:
                assert abs(code - hand) > 5.0


def _harmonic_by_hand(p_gpa, comp, s_unit):
    """Harmonic-rule liquidus [K] of an S, C, O mixture with Fe3C and FeO units, by hand."""
    units = {'S': s_unit, 'C': (3.0, 1.0), 'O': (1.0, 1.0)}
    molar = {'S': 32.06, 'C': 12.011, 'O': 15.999}
    t_fe = float(T_FE(p_gpa * 1e9))
    free, u, inv_a = (1.0 - sum(comp.values())) / MOLAR['Fe'], {}, {}
    for el, w in comp.items():
        t_e, w_e = (float(v) for v in eutectic(el, p_gpa * 1e9))
        k, m = units[el]
        ue = w_e / molar[el] / m
        fe = (1.0 - w_e) / MOLAR['Fe'] - k * ue
        inv_a[el] = (1.0 - t_fe / t_e) / math.log(fe / (fe + ue))
        u[el] = w / molar[el] / m
        free -= k * u[el]
    total = free + sum(u.values())
    big = sum(u.values()) / total
    return t_fe / (1.0 - math.log1p(-big) * sum(u[e] / total / big * inv_a[e] for e in comp))


@pytest.mark.physics_invariant
def test_a_mixture_takes_the_sulfur_units_of_its_pressure():
    """A mixture of 2 wt% S, 0.5 wt% C and 1 wt% O, not past its eutectic, follows the harmonic rule
    on FeS units at 10 GPa, on Fe3S units at 40 GPa and their smootherstep blend at 20, 22 and
    24 GPa, written out by hand."""
    comp = {'S': 0.02, 'C': 0.005, 'O': 0.01}
    liq = AlloyLiquidus(comp)
    assert not liq.status(np.array([10e9, 22e9, 40e9]))['past_eutectic'].any()
    assert float(liq.t_melt(10e9)) == pytest.approx(
        _harmonic_by_hand(10.0, comp, FES), rel=1e-12
    )
    assert float(liq.t_melt(40e9)) == pytest.approx(
        _harmonic_by_hand(40.0, comp, FE3S), rel=1e-12
    )
    for p in (20.0, 22.0, 24.0):
        b = _blend(p)
        hand = (1.0 - b) * _harmonic_by_hand(p, comp, FES) + b * _harmonic_by_hand(
            p, comp, FE3S
        )
        assert float(liq.t_melt(p * 1e9)) == pytest.approx(hand, rel=1e-12)
    assert abs(_harmonic_by_hand(22.0, comp, FES) - _harmonic_by_hand(22.0, comp, FE3S)) > 10.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('el', ['S', 'C', 'O'])
def test_below_1_bar_the_eutectic_is_held_homologous_and_flagged(el):
    """Below the start of the data (1 bar, written 0 GPa) the eutectic composition is its 1 bar
    value and the eutectic temperature keeps its 1 bar ratio to the pure-iron curve; the flag
    marks it."""
    t0, w0 = (float(v) for v in eutectic(el, 0.0))
    t_e, w_e = (float(v) for v in eutectic(el, -2e9))
    assert w_e == w0
    assert t_e / float(T_FE(-2e9)) == pytest.approx(t0 / float(T_FE(0.0)), rel=1e-12)
    assert t_e != pytest.approx(t0, rel=1e-6)
    flags = AlloyLiquidus({el: 0.01}).status(np.array([-2e9, 0.0]))['extrapolated']
    assert flags.tolist() == [True, False]


@pytest.mark.physics_invariant
def test_the_joins_and_blends_sit_where_the_data_need_them():
    """The S unit blend ends below the C unit blend starts, so the 2 act one at a time; the S join
    ends at the Simon reference pressure, the Mori composition table starts inside it, and the
    Mashino table starts at the Fe-C Simon reference pressure, where the cubic joins."""
    assert (
        alloy.S_UNIT_BLEND_GPA[1] * 1e9
        < alloy.C_UNIT_SWITCH_PA - alloy.C_UNIT_BLEND_HALF_WIDTH_PA
    )
    start = alloy._table('fe_s_composition')[0, 0]
    assert alloy.S_UNIT_BLEND_GPA[0] < start < alloy._SIMON['S'][1]
    assert alloy._table('fe_c_composition')[0, 0] == alloy._SIMON['C'][1]
