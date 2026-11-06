"""Alloy liquidus of the core below the Simon fits: the Fe-S eutectic of Fei et al. (1997) joined
into Mori et al. (2017) over 18 to 21 GPa, FeS units up to 18 GPa blended into Fe3S units over 18
to 26 GPa, and the Fe-C eutectic from 1 bar joined into Mashino et al. (2019) at 5 GPa."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import numpy as np
import pytest

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
    z = min(max(z, 0.0), 1.0)
    return z**3 * (z * (6.0 * z - 15.0) + 10.0)


@pytest.mark.reference_pinned
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


@pytest.mark.physics_invariant
def test_the_fe_s_eutectic_is_linear_between_the_nodes_with_kinks_at_10_and_14_gpa():
    """T_e and w_e are linear between the nodes; the slopes step at 10 GPa (-7.335 to -13.163
    K/GPa) and at 14 GPa, the invariant point where Fe3S2 appears (-13.163 to +30.393 K/GPa)."""
    assert _t_e('S')(5.0) == pytest.approx(0.5 * (_t_e('S')(0.0) + _t_e('S')(10.0)), rel=1e-12)
    assert _w_e('S')(12.0) == pytest.approx(
        0.5 * (_w_e('S')(10.0) + _w_e('S')(14.0)), rel=1e-12
    )
    for p, left, right in ((10.0, -7.335, -13.1625), (14.0, -13.1625, 30.392857)):
        assert _slopes(_t_e('S'), p) == pytest.approx((left, right), abs=1e-3)
    assert _slopes(_w_e('S'), 14.0) == pytest.approx((-0.0056625, -0.0058583), abs=1e-6)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('el', 'p_gpa', 't_slope', 'w_slope'),
    [
        ('S', 18.0, 30.392857, -0.0058583),
        ('S', 21.0, 1348.0 / (2.07 * 36.5), -0.000592),
        ('C', 5.0, 1513.0 / (4.1 * 8.4), -0.000212),
    ],
)
def test_the_eutectic_joins_the_fits_with_a_continuous_slope(el, p_gpa, t_slope, w_slope):
    """At the edges of the S join (18 and 21 GPa) and at the C join (5 GPa) the slopes of T_e
    [K/GPa] and w_e [1/GPa] are continuous: the low-pressure line at 18 GPa, the Simon fit and the
    first segment of the table at 21 and 5 GPa."""
    for f, slope, tol in ((_t_e(el), t_slope, 2e-3), (_w_e(el), w_slope, 2e-7)):
        left, right = _slopes(f, p_gpa)
        assert left == pytest.approx(right, abs=tol)
        assert right == pytest.approx(slope, abs=tol)


@pytest.mark.physics_invariant
def test_the_fe_s_eutectic_inside_the_join_is_the_smootherstep_mean():
    """Inside 18 to 21 GPa the eutectic is (1 - b) line + b fit, with b the smootherstep of
    (P - 18)/3; at 19.5 GPa b = 1/2."""
    tab = alloy._table('fe_s_composition')
    line_t = 1135.25 + (19.5 - 14.0) * (1348.0 - 1135.25) / 7.0
    line_w = 0.18305 + (19.5 - 14.0) * (tab[0, 1] - 0.18305) / (tab[0, 0] - 14.0)
    simon = 1348.0 * ((19.5 - 21.0) / 36.5 + 1.0) ** (1.0 / 2.07)
    first = tab[0, 1] + (19.5 - tab[0, 0]) * (tab[1, 1] - tab[0, 1]) / (tab[1, 0] - tab[0, 0])
    assert _smootherstep(0.5) == 0.5
    assert _t_e('S')(19.5) == pytest.approx(0.5 * (line_t + simon), rel=1e-12)
    assert _w_e('S')(19.5) == pytest.approx(0.5 * (line_w + first), rel=1e-12)


@pytest.mark.reference_pinned
def test_the_fe_c_eutectic_below_5_gpa_is_the_natural_cubic():
    """From 1 bar (1416.15 K, 4.3 wt% C: Nakajima et al. 2009, p. 207) to the Mashino fit at
    5 GPa, T_e and w_e are the cubic with zero curvature at 1 bar and the value and slope of the
    fit and the table at 5 GPa: 1416.15, 1423.73, 1450.68, 1475.95, 1513.00 K at 0, 1, 3, 4, 5
    GPa, rising by 7.09 to 43.93 K/GPa; Lord et al. (2009) give 1420 K at 1 bar."""
    tab = alloy._table('fe_c_composition')

    def cubic(p, y0, y1, s1):
        d = (s1 * 5.0 - (y1 - y0)) / 250.0
        return y0 + (s1 - 75.0 * d) * p + d * p**3

    s_w = (tab[1, 1] - tab[0, 1]) / (tab[1, 0] - tab[0, 0])
    for p, t_stated in ((0.0, 1416.15), (1.0, 1423.73), (3.0, 1450.68), (4.0, 1475.95)):
        t_e, w_e = (float(v) for v in eutectic('C', p * 1e9))
        assert t_e == pytest.approx(cubic(p, 1416.15, 1513.0, 1513.0 / 34.44), rel=1e-12)
        assert t_e == pytest.approx(t_stated, abs=0.01)
        assert w_e == pytest.approx(cubic(p, 0.043, tab[0, 1], s_w), rel=1e-12)
    slopes = np.diff([_t_e('C')(p) for p in np.linspace(0.0, 5.0, 501)]) / 0.01
    assert slopes.min() > 7.0 and slopes.max() < 43.94
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
    ('w', 'added', 'top'), [(0.02, 29.6, 51.4), (0.04, 53.6, 73.9), (0.08, 81.5, 98.4)]
)
def test_the_unit_blend_adds_the_stated_slope(w, added, top):
    """Over 18 to 26 GPa the blend adds c'(P) (T_Fe3S - T_FeS) to the weighted slope of the 2
    unit forms, at most 29.6, 53.6 and 81.5 K/GPa near 22 GPa for 2, 4 and 8 wt% S; the liquidus
    slope peaks at 51.4, 73.9 and 98.4 K/GPa."""
    p = np.linspace(18.001, 25.999, 801)
    g = np.asarray(AlloyLiquidus({'S': w}).gradient(p * 1e9)) * 1e9
    c = np.array([_smootherstep((x - 18.0) / 8.0) for x in p])
    ends = [
        np.array([(_form(w, x + 1e-4, u) - _form(w, x - 1e-4, u)) / 2e-4 for x in p])
        for u in (FES, FE3S)
    ]
    extra = g - ((1.0 - c) * ends[0] + c * ends[1])
    assert extra.max() == pytest.approx(added, abs=0.2)
    assert abs(p[extra.argmax()] - 22.0) < 0.2
    assert g.max() == pytest.approx(top, abs=0.2)


@pytest.mark.physics_invariant
def test_the_fe3s_units_hold_the_eutectic_from_the_start_of_the_blend():
    """At 18 GPa w_e is 15.96 wt% S, below the 16.06 wt% that Fe3S units can hold, so the Fe3S form
    is defined over the whole blend; its x_Fe at the eutectic is 0.022 there."""
    cap = MOLAR['S'] / (3.0 * MOLAR['Fe'] + MOLAR['S'])
    w_e = _w_e('S')(18.0)
    assert w_e == pytest.approx(0.15962, abs=1e-5)
    assert w_e < cap
    u = w_e / MOLAR['S']
    free = (1.0 - w_e) / MOLAR['Fe'] - 3.0 * u
    assert free / (free + u) == pytest.approx(0.0221, abs=2e-4)


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
    """Measured liquidus points (P GPa, w, T K) below 15 GPa: Buono & Walker (2011, Table 1) at
    6 GPa without the eutectic run, and read A of Chen et al. (2008, Fig. 1) at 10 and 14 GPa."""
    bw = [
        (6.0, _w_s(float(x)), float(t)) for _, t, x in _csv('buono_walker2011_table1_6GPa.csv')
    ]
    chen = [
        (float(p), float(w) / 100.0, float(t))
        for p, kind, w, t, *_ in _csv('chen2008_fig1_liquidus.csv')
        if kind == 'liquid'
    ]
    return [r for r in bw if r[1] < _w_s(0.604)] + chen


@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    ('p_gpa', 'n', 'rms', 'mean'),
    [(6.0, 23, 141.4, -130.6), (10.0, 7, 89.0, -82.6), (14.0, 4, 150.9, -12.4)],
)
def test_the_low_pressure_liquidus_misses_the_measured_points_as_stated(p_gpa, n, rms, mean):
    """The ideal form on FeS units lies below the measured non-ideal liquidus: RMS and mean misfit
    [K] at 6 GPa (Buono & Walker 2011) and at 10 and 14 GPa (Chen et al. 2008), not growing by
    more than 1 K."""
    pts = [(w, t) for p, w, t in _liquidus_points() if p == p_gpa]
    res = np.array([float(AlloyLiquidus({'S': w}).t_melt(p_gpa * 1e9)) - t for w, t in pts])
    assert len(pts) == n
    assert math.sqrt(np.mean(res**2)) < rms + 1.0
    assert res.mean() == pytest.approx(mean, abs=1.0)


@pytest.mark.reference_pinned
def test_from_10_to_14_gpa_the_liquidus_falls_less_than_the_data(validation_plot):
    """Between 10 and 14 GPa the measured liquidus of Chen et al. (2008, linear between their
    points and their pure-Fe point) falls by 42.05, 39.06 and 15.53 K/GPa at 4, 8 and 12 wt% S; the
    ideal form changes by +3.0, -14.5 and -26.9 K/GPa."""
    rows = _csv('chen2008_fig1_liquidus.csv')

    def data(p, w):
        pts = sorted((float(r[2]), float(r[3])) for r in rows if float(r[0]) == p)
        return np.interp(w, *zip(*pts))

    for w, stated_data, stated_code in (
        (4, -42.05, 3.0),
        (8, -39.06, -14.5),
        (12, -15.53, -26.9),
    ):
        assert (data(14.0, w) - data(10.0, w)) / 4.0 == pytest.approx(stated_data, abs=0.005)
        liq = AlloyLiquidus({'S': w / 100.0})
        code = (float(liq.t_melt(14e9)) - float(liq.t_melt(10e9))) / 4.0
        assert code == pytest.approx(stated_code, abs=0.05)

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


# code minus source, T_e [K] and w_e [wt%], in the row order of low_pressure_eutectic_checks.csv
_CHECK_MISS = [
    (-45.76, -0.935),
    (-45.76, -0.808),
    (-41.76, -2.937),
    (77.24, -2.272),
    (0.0, -0.669),
    (36.60, None),
    (52.45, None),
    (-148.04, 1.678),
    (-66.81, None),
    (-3.85, 0.0),
    (0.0, 0.311),
    (77.54, 0.505),
]


@pytest.mark.reference_pinned
def test_the_low_pressure_eutectic_against_the_check_points(validation_plot):
    """Eutectic points that set no node, code minus source: T_e [K] and w_e [wt%]; the 6 GPa
    eutectic is 46 K and 0.9 wt% below Buono & Walker (2011), Kamada et al. (2010) lie within
    their fit errors, the 21.9 and 23.0 GPa quench runs (upper bounds of T_e) lie above, and
    Fe-C at 10 GPa is the Mashino fit, 77.5 K above Fei & Brosh (2014)."""
    rows = _csv('low_pressure_eutectic_checks.csv')
    assert len(rows) == len(_CHECK_MISS)
    for (src, el, p, t, t_err, comp, _, unit, kind), (dt, dw) in zip(rows, _CHECK_MISS):
        t_e, w_e = (float(v) for v in eutectic(el, float(p) * 1e9))
        assert t_e - float(t) == pytest.approx(dt, abs=0.02), src
        if kind == 'fit':
            assert abs(t_e - float(t)) < float(t_err), src
        if kind == 'upper_bound_T':
            assert t_e < float(t), src
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
        (0.01, 18.123, 13.99, None),
        (0.03, 4.436, 13.99, None),
        (0.048, -5.816, 13.99, (9.33, 13.99)),
        (0.08, -19.977, 13.99, (5.49, 14.00)),
        (0.12, -34.304, 9.99, (2.08, 14.00)),
    ],
)
def test_below_26_gpa_the_s_liquidus_falls_where_the_eutectic_falls(
    w, slope_min, p_min, falling
):
    """Below 14 GPa T_e falls (Fei et al. 1997), so S alloys from about 4 wt% fall with pressure
    there: the least slope [K/GPa] and the first and last falling pressures [GPa] on a 0.01 GPa
    grid; 1 and 3 wt% S keep rising."""
    p = np.arange(0.01, 26.0, 0.01)
    g = np.asarray(AlloyLiquidus({'S': w}).gradient(p * 1e9)) * 1e9
    assert g.min() == pytest.approx(slope_min, abs=0.02)
    assert p[g.argmin()] == pytest.approx(p_min, abs=0.015)
    down = p[g <= 0.0]
    if falling is None:
        assert down.size == 0
    else:
        assert (down[0], down[-1]) == pytest.approx(falling, abs=0.015)
