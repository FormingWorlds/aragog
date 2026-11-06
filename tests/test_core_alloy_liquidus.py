"""Alloy liquidus of the core: ideal-solution Fe-X binaries joined by a harmonic A
(``aragog.core.alloy``), against the curves the source papers draw."""

from __future__ import annotations

import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.core import alloy
from aragog.core.alloy import AlloyLiquidus, binary_liquidus, eutectic
from aragog.core.melting import IronMeltingCurve
from tests.conftest import assert_legend_clear, opaque_legend

pytestmark = pytest.mark.unit

DATA = Path(__file__).parent / 'validation' / 'data' / 'core_eutectics'
T_FE = IronMeltingCurve.t_melt_pure
MOLAR = {'Fe': 55.845, 'S': 32.06, 'C': 12.011, 'O': 15.999}
UNIT = {'S': (3.0, 1.0), 'O': (1.0, 1.0), 'C': (3.0, 1.0), 'C7': (7.0, 3.0)}


def _read(name):
    """Header lines and the 3 columns (x, read A, read B with NaN where absent) of a data file."""
    lines = (DATA / name).read_text().splitlines()
    head = [ln for ln in lines if ln.startswith('#')]
    rows = [ln.split(',') for ln in lines if not ln.startswith('#')][1:]
    cols = np.array([[float(v) if v else np.nan for v in r] for r in rows])
    return head, cols[:, 0], cols[:, 1], cols[:, 2]


def _units(w, el, unit):
    """Mole fraction of free Fe among Fe and the Fe_k X_m units of a binary, by hand."""
    k, m = unit
    u = w / MOLAR[el] / m
    free = (1.0 - w) / MOLAR['Fe'] - k * u
    return free / (free + u)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('el', ['S', 'C', 'O'])
@pytest.mark.parametrize('p_gpa', [10.0, 60.0, 135.0, 254.0, 330.0, 450.0])
def test_each_binary_runs_through_pure_iron_and_its_eutectic(el, p_gpa):
    """At zero light element the liquidus is the pure-iron curve; at the eutectic composition it is
    the eutectic temperature, also outside the data range where both are extrapolated."""
    p = p_gpa * 1e9
    t_e, w_e = map(float, eutectic(el, p))
    assert float(AlloyLiquidus({el: 0.0}).t_melt(p)) == float(T_FE(p))
    assert float(AlloyLiquidus({el: w_e}).t_melt(p)) == pytest.approx(t_e, rel=1e-12)


def test_si_h_and_n_lower_no_melting_point():
    """Si, H and N alone leave the liquidus at pure iron."""
    p = np.array([60e9, 135e9, 330e9])
    t = AlloyLiquidus({'Si': 0.06, 'H': 0.005, 'N': 0.01}).t_melt(p)
    np.testing.assert_array_equal(np.asarray(t), np.asarray(T_FE(p)))


def test_h_and_n_leave_the_mole_basis():
    """H and N have no liquidus data at core pressures: the alloy melts like the same alloy with
    its H and N removed and the rest renormalised."""
    p = np.array([60e9, 135e9, 330e9])
    with_hn = np.asarray(AlloyLiquidus({'S': 0.02, 'O': 0.03, 'H': 0.01, 'N': 0.01}).t_melt(p))
    without = np.asarray(AlloyLiquidus({'S': 0.02 / 0.98, 'O': 0.03 / 0.98}).t_melt(p))
    np.testing.assert_allclose(with_hn, without, rtol=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('extra', [{'H': 0.02}, {'N': 0.05}, {'H': 0.01, 'N': 0.02}])
def test_h_and_n_leave_the_eutectic_check_too(extra):
    """Near and past the eutectic an alloy with H or N melts and is flagged like the same alloy
    without them, renormalised: never below the eutectic temperature."""
    p = np.array([135e9])
    t_e, w_e = map(float, eutectic('S', p[0]))
    keep = 1.0 - sum(extra.values())
    for ratio in (0.995, 1.01, 1.1):  # 1.01: past the eutectic only without H and N
        with_hn = AlloyLiquidus({'S': ratio * w_e * keep, **extra})
        without = AlloyLiquidus({'S': ratio * w_e})
        assert float(with_hn.t_melt(p[0])) == pytest.approx(
            float(without.t_melt(p[0])), rel=1e-12
        )
        assert float(with_hn.t_melt(p[0])) >= t_e * (1.0 - 1e-12)
        assert with_hn.status(p)['past_eutectic'][0] == without.status(p)['past_eutectic'][0]


def test_si_counts_with_the_iron_solvent():
    """Si atoms join the Fe solvent: they dilute the S units, so S with Si melts closer to iron
    than S alone, by the mole-fraction change written out here."""
    p = 135e9
    t_fe = float(T_FE(p))
    t_e, w_e = map(float, eutectic('S', p))
    a = math.log(_units(w_e, 'S', UNIT['S'])) / (1.0 - t_fe / t_e)
    w_s, w_si = 0.02, 0.06
    u = w_s / MOLAR['S']
    solvent = (1.0 - w_s - w_si) / MOLAR['Fe'] - 3.0 * u + w_si / 28.085
    by_hand = t_fe / (1.0 - math.log(solvent / (solvent + u)) / a)
    with_si = float(AlloyLiquidus({'S': w_s, 'Si': w_si}).t_melt(p))
    assert with_si == pytest.approx(by_hand, rel=1e-12)
    assert with_si > float(AlloyLiquidus({'S': w_s}).t_melt(p)) + 10.0


@pytest.mark.parametrize('el', ['S', 'O'])
def test_one_element_is_the_binary_ideal_form_on_component_units(el):
    """With one element the harmonic rule is the binary form, written out here by hand on Fe3S or
    FeO units; S atoms in place of Fe3S units give another curve."""
    p = 135e9
    t_e, w_e = map(float, eutectic(el, p))
    t_fe = float(T_FE(p))
    a = math.log(_units(w_e, el, UNIT[el])) / (1.0 - t_fe / t_e)
    for w in (0.01, 0.4 * w_e, 0.8 * w_e):
        by_hand = t_fe / (1.0 - math.log(_units(w, el, UNIT[el])) / a)
        assert float(AlloyLiquidus({el: w}).t_melt(p)) == pytest.approx(by_hand, rel=1e-12)
        assert float(binary_liquidus(el, w, t_fe, t_e, w_e, UNIT[el])) == pytest.approx(
            by_hand, rel=1e-12
        )
    a_atoms = math.log(_units(w_e, el, (0.0, 1.0))) / (1.0 - t_fe / t_e)
    w = 0.4 * w_e
    atoms = t_fe / (1.0 - math.log(_units(w, el, (0.0, 1.0))) / a_atoms)
    if el == 'S':
        assert abs(float(AlloyLiquidus({el: w}).t_melt(p)) - atoms) > 20.0


def _multi_by_hand(p, comp):
    """Depression [K] of the harmonic, composition-weighted and binary-sum rules, by hand, with the
    C units of pressure ``p`` below 230 GPa (Fe3C) or above 280 GPa (Fe7C3)."""
    t_fe = float(T_FE(p))
    units = {'S': UNIT['S'], 'O': UNIT['O'], 'C': UNIT['C'] if p < 230e9 else UNIT['C7']}
    a = {}
    free = (1.0 - sum(comp.values())) / MOLAR['Fe']
    u = {}
    for el, w in comp.items():
        t_e, w_e = map(float, eutectic(el, p))
        a[el] = math.log(_units(w_e, el, units[el])) / (1.0 - t_fe / t_e)
        k, m = units[el]
        u[el] = w / MOLAR[el] / m
        free -= k * u[el]
    total = free + sum(u.values())
    x = {el: u[el] / total for el in comp}
    big = sum(x.values())

    def t_of(inv_a):
        return t_fe / (1.0 - math.log1p(-big) * inv_a)

    harmonic = t_fe - t_of(sum(x[e] / big / a[e] for e in comp))
    weighted = t_fe - t_of(1.0 / sum(x[e] / big * a[e] for e in comp))
    binaries = sum(t_fe - t_fe / (1.0 - math.log1p(-x[e]) / a[e]) for e in comp)
    dilute = t_fe * sum(x[e] / a[e] for e in comp)
    return harmonic, weighted, binaries, dilute


@pytest.mark.reference_pinned
@pytest.mark.parametrize(('p_gpa', 'expected'), [(135.0, 641.0), (330.0, 1749.0)])
def test_the_harmonic_rule_gives_the_stated_depressions(p_gpa, expected, validation_plot):
    """For S 2, C 0.5, O 3 wt% the harmonic rule lowers the melting point by 641 K at 135 GPa and
    1749 K at 330 GPa (the values the docs state, pinned); the composition-weighted rule gives a
    far smaller depression."""
    p = p_gpa * 1e9
    comp = {'S': 0.02, 'C': 0.005, 'O': 0.03}
    depression = float(T_FE(p) - AlloyLiquidus(comp).t_melt(p))
    harmonic, weighted, binaries, _ = _multi_by_hand(p, comp)
    assert depression == pytest.approx(expected, abs=1.0)
    assert depression == pytest.approx(harmonic, rel=1e-12)
    assert depression - weighted > 100.0
    assert weighted < depression < binaries

    def figure():
        import matplotlib.pyplot as plt

        scales = np.linspace(0.02, 1.0, 20)
        rows = np.array(
            [_multi_by_hand(p, {k: s * v for k, v in comp.items()}) for s in scales]
        )
        total = 100.0 * sum(comp.values()) * scales
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for col, label in zip(
            range(4), ('harmonic (Aragog)', 'weighted', 'binary sum', 'dilute')
        ):
            ax.plot(total, rows[:, col], label=label, ls='-' if col == 0 else '--')
        ax.set(xlabel='S + C + O [wt%]', ylabel='Depression [K]')
        opaque_legend(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, f'core_alloy_liquidus_rules_{p_gpa:g}gpa')


@pytest.mark.physics_invariant
def test_the_harmonic_rule_meets_the_dilute_limit():
    """As the light elements vanish, the depression tends to T_Fe sum x_i / A_i."""
    p = 135e9
    base = {'S': 0.02, 'C': 0.005, 'O': 0.03}
    ratios = []
    for s in (1e-1, 1e-2, 1e-3):
        comp = {k: s * v for k, v in base.items()}
        harmonic, _, _, dilute = _multi_by_hand(p, comp)
        code = float(T_FE(p) - AlloyLiquidus(comp).t_melt(p))
        assert code == pytest.approx(harmonic, rel=1e-9)
        ratios.append(code / dilute)
    assert abs(ratios[-1] - 1.0) < 1e-3
    assert abs(ratios[0] - 1.0) > abs(ratios[-1] - 1.0)


def _curves_figure(curves, xlabel, source):
    """Model lines and paper points per pressure, one legend entry per pressure plus a key for the
    2 marks; colours from the active style."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for p_gpa, (w, model, paper) in curves.items():
        (line,) = ax.plot(w[::10], model[::10], label=f'{p_gpa}')
        keep = ~np.isnan(paper)
        ax.plot(
            w[keep][1:-1:18],
            paper[keep][1:-1:18],
            'o',
            ms=3,
            mfc='none',
            color=line.get_color(),
        )
    ink = plt.rcParams['text.color']
    keys = [
        Line2D([], [], color=ink, alpha=0.55, label='Aragog'),
        Line2D(
            [], [], color=ink, alpha=0.55, ls='none', marker='o', mfc='none', ms=3, label=source
        ),
    ]
    handles = ax.get_legend_handles_labels()[0] + keys
    opaque_legend(
        ax, handles=handles, title='P [GPa]', loc='upper left', bbox_to_anchor=(1.02, 1.0)
    )
    ax.set(xlabel=xlabel, ylabel='Liquidus [K]')
    fig.tight_layout()
    assert_legend_clear(fig)
    return fig


@pytest.mark.reference_pinned
def test_the_fe_s_liquidus_curves_of_mori_are_the_ideal_form_on_fe3s_units(validation_plot):
    """Through each curve's own end points the binary form on Fe3S units reproduces the 3 liquidus
    curves Mori et al. (2017, Fig. 9) draw at 30, 130 and 250 GPa, on both reads; on S atoms it
    misses by over 100 K."""
    curves = {}
    for p_gpa in (30, 130, 250):
        _, w, ta, tb = _read(f'mori_liquidus_{p_gpa}GPa.csv')
        t_fe, t_e, w_e = ta[0], ta[-1], w[-1] / 100.0
        model = np.asarray(binary_liquidus('S', w / 100.0, t_fe, t_e, w_e, UNIT['S']))
        atoms = np.asarray(binary_liquidus('S', w / 100.0, t_fe, t_e, w_e, (0.0, 1.0)))
        b = ~np.isnan(tb)
        assert np.sqrt(np.mean((model - ta) ** 2)) < 5.0
        assert np.sqrt(np.mean((model[b] - tb[b]) ** 2)) < 6.0
        assert np.sqrt(np.mean((atoms - ta) ** 2)) > 100.0
        curves[p_gpa] = (w, ta, tb, model)

    def figure():
        return _curves_figure(
            {p: (w, model, tb) for p, (w, ta, tb, model) in curves.items()},
            'S [wt%]',
            'Mori 2017',
        )

    validation_plot(figure, 'core_alloy_liquidus_fe_s_mori2017')


@pytest.mark.reference_pinned
def test_the_fe_c_misfit_against_mashino_stays_as_stated(validation_plot):
    """Mashino et al. (2019, Fig. 8) draw near-straight Fe-C curves; the ideal form on Fe3C (below
    255 GPa) and Fe7C3 units through each curve's end points, the zero-C end taken at w = 0 by
    extrapolation of the first 2 points, misses them by the RMS the docs state (read A, 0.1 wt%
    grid of the test data); the C units are not verified, and this test keeps the misfit from
    growing."""
    stated = {
        54: ('C', 48.7),
        138: ('C', 110.1),
        203: ('C', 120.8),
        255: ('C7', 114.5),
        330: ('C7', 101.5),
    }
    curves = {}
    for p_gpa, (unit, rms_stated) in stated.items():
        _, w, ta, _ = _read(f'mashino_liquidus_{p_gpa}GPa.csv')
        t_fe = ta[0] - w[0] * (ta[1] - ta[0]) / (w[1] - w[0])
        model = np.asarray(
            binary_liquidus('C', w / 100.0, t_fe, ta[-1], w[-1] / 100.0, UNIT[unit])
        )
        rms = np.sqrt(np.mean((model - ta) ** 2))
        assert rms < rms_stated + 1.0
        curves[p_gpa] = (w, ta, model)

    def figure():
        return _curves_figure(
            {p: (w, model, ta) for p, (w, ta, model) in curves.items()},
            'C [wt%]',
            'Mashino 2019',
        )

    validation_plot(figure, 'core_alloy_liquidus_fe_c_mashino2019')


@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    ('el', 'p_gpa', 'expected', 'tol'),
    [('S', 254.0, 3550.0, 50.0), ('C', 330.0, 3750.0, 50.0)],
)
def test_the_eutectic_temperatures_meet_the_papers(el, p_gpa, expected, tol):
    """Fe-S 3550 K at 254 GPa (Mori et al. 2017, run 11) and Fe-C about 3750 K at 330 GPa (Mashino
    et al. 2019, p. 6)."""
    assert float(eutectic(el, p_gpa * 1e9)[0]) == pytest.approx(expected, abs=tol)


@pytest.mark.reference_pinned
def test_the_fe_o_eutectic_temperature_meets_the_second_read(validation_plot):
    """The Fe-O eutectic temperature lies on read B of Oka et al. (2019, Fig. 4), the read the
    table does not come from, within the full drawn line width (0.354 pt) in figure points, from
    0 to 325 GPa, the end of the Fe-O data."""
    _, p_gpa, _, tb = _read('oka2019_fig4_T.csv')
    keep = (p_gpa >= 0.0) & (p_gpa <= 325.0)
    got = np.asarray(eutectic('O', p_gpa[keep] * 1e9)[0])
    assert np.sum(~np.isnan(tb[keep])) > 40
    assert _perpendicular(p_gpa[keep], got, tb[keep], 0.4654, 0.04946) <= 0.354

    def figure():
        import matplotlib.pyplot as plt

        q = np.linspace(1.0, 400.0, 60)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for el in ('S', 'C', 'O'):
            ax.plot(q, np.asarray(eutectic(el, q * 1e9)[0]), label=f'Fe-{el}')
        b = ~np.isnan(tb) & keep
        ax.plot(p_gpa[b][::10], tb[b][::10], 'o', ms=3, mfc='none', label='Oka 2019')
        ax.plot([254.0], [3550.0], 's', ms=5, mfc='none', label='Mori 2017')
        ax.plot([330.0], [3750.0], 'D', ms=5, mfc='none', label='Mashino 2019')
        ax.set(xlabel='Pressure [GPa]', ylabel='$T_e$ [K]')
        opaque_legend(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'core_alloy_liquidus_eutectic_temperatures')


def _read_distance(name):
    """Largest read-B-to-read-A distance [pt] of a data file and its tolerance [pt]."""
    head, x, ya, yb = _read(name)
    scale = next(ln for ln in head if 'figure scale' in ln)
    sx, sy, _, tol = map(float, re.findall(r'\d+\.?\d*', scale))
    return _perpendicular(x, ya, yb, sx, sy), tol


def _perpendicular(x, ya, yb, sx, sy):
    """Largest distance [pt] from the read B points to the read A polyline in figure points."""
    ax_, ay_ = x * sx, ya * sy
    keep = ~np.isnan(yb)
    s = np.linspace(0.0, 1.0, 20001)
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(ax_), np.diff(ay_)))]
    px, py = np.interp(s * arc[-1], arc, ax_), np.interp(s * arc[-1], arc, ay_)
    return max(
        np.min(np.hypot(px - bx, py - by)) for bx, by in zip(x[keep] * sx, yb[keep] * sy)
    )


@pytest.mark.reference_pinned
@pytest.mark.parametrize('name', sorted(p.name for p in DATA.glob('*.csv')))
def test_the_two_reads_of_each_figure_curve_agree(name):
    """Read A (vector path) and read B (raster trace) of each digitised curve agree within the
    tolerance in the file header, in points of the source figure."""
    assert np.sum(~np.isnan(_read(name)[3])) > 10
    distance, tol = _read_distance(name)
    assert distance <= tol + 0.02


def _short(name):
    """Short label of a data file: X and the pressure [GPa] of a liquidus curve, X w for an
    eutectic composition, O T for the Fe-O eutectic temperature."""
    el = {'mori': 'S', 'mashino': 'C'}
    if '_liquidus_' in name:
        src, _, p_gpa = name.removesuffix('GPa.csv').split('_')
        return f'{el[src]}{p_gpa}'
    last = name.removesuffix('.csv').split('_')[-1]
    return 'O T' if last == 'T' else f'{last} w'


@pytest.mark.reference_pinned
def test_the_two_reads_figure(validation_plot):
    """The largest distance between the 2 reads of every data file, over its tolerance."""
    names = sorted(p.name for p in DATA.glob('*.csv'))
    ratios = np.array([d / tol for d, tol in map(_read_distance, names)])
    assert ratios.max() < 1.0 + 0.02 / 0.35

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.barh(range(len(names)), ratios, alpha=0.7)
        ax.axvline(1.0, color=plt.rcParams['text.color'], lw=0.8, ls='--')
        ax.set_yticks(range(len(names)), [_short(n) for n in names])
        ax.set(xlabel='Read distance / tolerance', xlim=(0.0, 1.1))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'core_alloy_liquidus_two_reads')


@pytest.mark.physics_invariant
def test_a_composition_past_the_eutectic_is_held_at_its_boundary():
    """Past the Fe-rich side the liquidus is that of the boundary composition and the flag is set;
    below it the flag is clear."""
    p = 135e9
    t_e, w_e = map(float, eutectic('S', p))
    past = AlloyLiquidus({'S': 1.5 * w_e})
    assert float(past.t_melt(p)) == pytest.approx(t_e, rel=1e-12)
    assert past.status(np.array([p]))['past_eutectic'].tolist() == [True]
    assert AlloyLiquidus({'S': 0.5 * w_e}).status(np.array([p]))['past_eutectic'].tolist() == [
        False
    ]
    just_past = AlloyLiquidus({'S': 1.05 * w_e})
    assert float(just_past.t_melt(p)) == pytest.approx(t_e, rel=1e-12)


@pytest.mark.physics_invariant
def test_beyond_the_fade_the_eutectic_temperature_is_homologous_and_flagged():
    """Above the end of the data the eutectic composition is held at its end value (5 wt% S,
    3.35 wt% C, 15.068 wt% O); from the end plus the fade length the eutectic temperature keeps
    its ratio to the pure-iron curve, taken 20 GPa above the end; the flag marks the
    extrapolation."""
    for el in ('S', 'C', 'O'):
        hi = alloy._data()['range'][el][1]
        t_end, w_end = map(float, alloy._eutectic_data(el, hi))
        h = 0.5 * alloy.HOMOLOGOUS_FADE_GPA
        t_fade = float(eutectic(el, (hi + 2.0 * h) * 1e9)[0])
        for p_gpa in (hi + 1e-3, hi + 1.0, hi + h, hi + 2.0 * h, 500.0):
            assert float(eutectic(el, p_gpa * 1e9)[1]) == pytest.approx(w_end, rel=1e-12)
        for p_gpa in (hi + 2.0 * h, 500.0):
            ratio = float(eutectic(el, p_gpa * 1e9)[0]) / float(T_FE(p_gpa * 1e9))
            assert ratio == pytest.approx(t_fade / float(T_FE((hi + 2.0 * h) * 1e9)), rel=1e-12)
        assert t_fade / float(T_FE((hi + 2.0 * h) * 1e9)) != t_end / float(T_FE(hi * 1e9))
    assert float(eutectic('S', 600e9)[1]) == pytest.approx(0.05, rel=1e-12)
    assert float(eutectic('C', 600e9)[1]) == pytest.approx(0.0335, rel=1e-12)
    s = AlloyLiquidus({'S': 0.02}).status(np.array([135e9, 500e9]))
    assert s['extrapolated'].tolist() == [False, True]
    assert AlloyLiquidus({'Si': 0.02}).status(np.array([500e9]))['extrapolated'].tolist() == [
        False
    ]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('w_c', [0.005, 0.01, 0.02, 0.03])
def test_the_carbon_unit_switch_adds_no_slope_dip(w_c, monkeypatch):
    """Across the 230 to 280 GPa blend of the Fe3C and Fe7C3 units the liquidus slope falls under
    1 K/GPa below its lower edge value (0.76 K/GPa at most, at 2 wt% C, against edge slopes of 1.6
    to 9.3 K/GPa), so the blend sets no false snow margin; a 10 GPa blend dips 3.7 to 8.9 K/GPa."""

    def dip(liq):
        g = np.asarray(liq.gradient(np.linspace(230e9, 280e9, 101))) * 1e9
        edges = np.asarray(liq.gradient(np.array([229e9, 281e9]))) * 1e9
        return edges.min() - g.min()

    assert dip(AlloyLiquidus({'C': w_c})) < 1.0
    monkeypatch.setattr(alloy, 'C_UNIT_BLEND_HALF_WIDTH_PA', 5.0e9)
    if w_c in (0.01, 0.02):
        assert dip(AlloyLiquidus({'C': w_c})) > 3.0


_bad_value = st.one_of(
    st.just(True),
    st.just('0.01'),
    st.none(),
    st.just(float('nan')),
    st.just(float('inf')),
    st.floats(max_value=-1e-12, allow_nan=False, allow_infinity=False),
    st.just(1j),
)


@settings(max_examples=60, deadline=None)
@given(el=st.sampled_from(alloy.ELEMENTS), bad=_bad_value)
def test_a_bad_mass_fraction_is_refused(el, bad):
    with pytest.raises(ValueError, match='mass fraction'):
        AlloyLiquidus({el: bad})


@pytest.mark.parametrize('comp', [{'Mg': 0.01}, {'S': 0.6, 'O': 0.4}, [('S', 0.01)]])
def test_an_unknown_element_a_full_alloy_or_a_non_dict_is_refused(comp):
    with pytest.raises(ValueError):
        AlloyLiquidus(comp)


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@given(
    s=st.floats(0.0, 0.08),
    c=st.floats(0.0, 0.03),
    o=st.floats(0.0, 0.10),
    si=st.floats(0.0, 0.08),
    p_gpa=st.floats(20.0, 600.0),
)
def test_the_liquidus_lies_below_iron_and_falls_with_sulfur(s, c, o, si, p_gpa):
    """Any valid alloy melts at or below pure iron, and on the Fe-rich side more S never raises
    the liquidus."""
    p = p_gpa * 1e9
    liq, more = (
        AlloyLiquidus({'S': s, 'C': c, 'O': o, 'Si': si}),
        AlloyLiquidus({'S': s + 0.005, 'C': c, 'O': o, 'Si': si}),
    )
    t = float(liq.t_melt(p))
    assert 0.0 < t <= float(T_FE(p)) * (1.0 + 1e-12)
    if not more.status(np.array([p]))['past_eutectic'][0]:  # past it, the boundary rule holds T
        assert float(more.t_melt(p)) <= t * (1.0 + 1e-12)


@pytest.mark.physics_invariant
def test_a_binary_with_a_non_positive_a_is_refused(monkeypatch):
    """A eutectic temperature above pure iron gives A <= 0; the concrete call refuses it."""
    monkeypatch.setitem(alloy._SIMON, 'S', (1.0e5, 21.0, 36.5, 2.07))
    with pytest.raises(ValueError, match='not positive|out of'):
        AlloyLiquidus({'S': 0.02}).t_melt(135e9)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('bad', [1.01, float('nan'), -1.0])
def test_the_liquidus_guard_refuses_a_temperature_outside_zero_to_pure_iron(bad):
    """The guard of a concrete call refuses a liquidus above pure iron, non-finite or not
    positive, with every binary A valid."""
    p = np.array([135e9])
    t = np.asarray(T_FE(p)) * bad if bad != -1.0 else np.array([-1.0])
    with pytest.raises(ValueError, match='out of'):
        alloy._assert_invariants(p, t, {'S': np.array([0.5])})
    alloy._assert_invariants(p, np.asarray(T_FE(p)) * 0.9, {'S': np.array([0.5])})


@pytest.mark.physics_invariant
def test_a_bad_a_of_an_absent_element_is_refused(monkeypatch):
    """A binary with A <= 0 is refused even when its element is absent from the alloy, where the
    liquidus itself stays valid."""
    monkeypatch.setitem(alloy._SIMON, 'S', (1.0e5, 21.0, 36.5, 2.07))
    with pytest.raises(ValueError, match='not positive'):
        AlloyLiquidus({'O': 0.02}).t_melt(135e9)


_RISING = (
    [{'C': w} for w in (0.005, 0.01, 0.02, 0.03, 0.032)]
    + [{'S': w} for w in (0.01, 0.03, 0.048)]
    + [{'O': w} for w in (0.01, 0.05, 0.1)]
    + [
        {'S': 0.02, 'C': 0.005, 'O': 0.03, 'Si': 0.04},
        {'S': 0.03, 'O': 0.02},
    ]
)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'comp', _RISING, ids=lambda c: '+'.join(f'{k}{v:g}' for k, v in c.items())
)
def test_the_liquidus_rises_with_pressure(comp):
    """dT_liq/dP > 0 over 1 to 600 GPa wherever the alloy is not past the eutectic, so no table
    node, anchor or end of the data sets a false top-down regime (S up to 4.8 wt%; the next test
    pins the falling liquidus of 5.0 to 5.8 wt% S)."""
    p = np.linspace(1e9, 600e9, 2393)
    liq = AlloyLiquidus(comp)
    inside = ~liq.status(p)['past_eutectic']
    slope = np.asarray(liq.gradient(p))[inside] * 1e9
    assert inside.sum() > 100
    assert slope.min() > 0.0, (p[inside][slope.argmin()] / 1e9, slope.min())


@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    ('comp', 'slope_min', 'p_from', 'p_to'),
    [({'S': 0.050}, -0.161, 272.3, 274.9), ({'S': 0.054}, -0.637, 259.0, 274.9)]
    + [({'S': 0.058}, -0.508, 243.9, 261.4), ({'S': 0.04, 'C': 0.01}, -0.372, 270.0, 273.4)],
    ids=lambda v: '+'.join(f'{k}{x:g}' for k, x in v.items()) if isinstance(v, dict) else None,
)
def test_near_the_eutectic_the_liquidus_falls_where_the_read_curve_falls(
    comp, slope_min, p_from, p_to
):
    """The eutectic composition of Mori et al. (2017, Fig. 5) falls by 0.021 wt%/GPa at 250 to
    275 GPa, so near the eutectic, with dT/dP ~ dT_e/dP - |dT/dw| |dw_e/dP|, the liquidus of 5.0
    to 5.8 wt% S, and of S with C near their eutectic, falls with pressure there; the pinned
    interval [GPa] and least slope [K/GPa] show a later change of the data or the code."""
    p = np.linspace(200e9, 320e9, 1201)
    liq = AlloyLiquidus(comp)
    inside = ~liq.status(p)['past_eutectic']
    g = np.asarray(liq.gradient(p)) * 1e9
    falling = p[inside & (g <= 0.0)] / 1e9
    assert g[inside].min() == pytest.approx(slope_min, abs=0.01)
    assert (falling[0], falling[-1]) == pytest.approx((p_from, p_to), abs=0.2)
    assert np.all(g[inside & ((p < 240e9) | (p > 280e9))] > 0.0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('comp', 'step_325', 'step_330'),
    [
        ({'S': 0.048}, 0.0, 3.671),
        ({'C': 0.01}, 0.0, 1.273),
        ({'C': 0.03}, 0.0, 2.693),
        ({'O': 0.1}, 1.587, 0.0),
        ({'S': 0.02, 'C': 0.005, 'O': 0.03}, 0.404, 2.333),
    ],
)
def test_the_held_eutectic_composition_steps_the_slope_at_the_end_of_the_data(
    comp, step_325, step_330
):
    """The eutectic composition stops changing at the end of the data (325 GPa for O, 330 GPa for
    S and C), so the liquidus slope steps there by the stated K/GPa between points 0.1 GPa apart;
    elsewhere over 325 to 380 GPa it moves by under 0.1 K/GPa, the fade of the temperature ratio
    included."""
    g = np.asarray(AlloyLiquidus(comp).gradient(np.linspace(325e9, 380e9, 551))) * 1e9
    d = np.abs(np.diff(g))
    assert (d[0], d[50]) == pytest.approx((step_325, step_330), abs=0.01)
    assert np.delete(d, [0, 50]).max() < 0.1


@pytest.mark.physics_invariant
@pytest.mark.parametrize('el', ['S', 'C', 'O'])
def test_the_composition_slope_at_zero_is_the_dilute_slope(el):
    """dT_liq/dw of a traced composition at w = 0 is the dilute slope, equal to the finite
    difference, not 0."""
    liq = AlloyLiquidus({})
    full = {e: 0.0 for e in alloy.ELEMENTS}

    def t_of(w):
        return liq.t_melt(135e9, {**full, el: w})

    at_zero = float(jax.grad(t_of)(0.0))
    h = 1e-7
    assert at_zero < -1.0e3
    assert at_zero == pytest.approx((float(t_of(h)) - float(t_of(0.0))) / h, rel=1e-4)


def test_a_traced_composition_takes_zero_for_missing_keys():
    """A traced composition with only some keys behaves like the full one; an unknown key still
    fails."""
    liq = AlloyLiquidus({})
    part = jax.grad(lambda s: liq.t_melt(135e9, {'S': s}))(0.02)
    full = jax.grad(lambda s: liq.t_melt(135e9, {**{e: 0.0 for e in alloy.ELEMENTS}, 'S': s}))(
        0.02
    )
    assert float(part) == float(full)
    with pytest.raises(ValueError, match='unknown light elements'):
        jax.grad(lambda s: liq.t_melt(135e9, {'Mg': s}))(0.02)


@pytest.mark.parametrize('zero_d', [jnp.asarray(0.02), np.asarray(0.02), np.float32(0.02)])
def test_a_zero_dimensional_array_is_a_mass_fraction(zero_d):
    """A 0-d NumPy or JAX array is a mass fraction like the float; a 1-d array is refused."""
    liq = AlloyLiquidus({})
    expected = float(liq.t_melt(135e9, {'S': float(zero_d)}))
    assert float(liq.t_melt(135e9, {'S': zero_d})) == expected
    with pytest.raises(ValueError, match='mass fraction'):
        AlloyLiquidus({'S': np.array([0.02])})


def test_o_is_flagged_beyond_its_composition_table():
    """The Fe-O composition table ends at 325 GPa, so O is extrapolated from there."""
    flags = AlloyLiquidus({'O': 0.05}).status(np.array([320e9, 327e9]))['extrapolated']
    assert flags.tolist() == [False, True]


@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    ('table', 'read'),
    [('fe_s_composition', 'mori2017_fig5_S'), ('fe_c_composition', 'mashino2019_fig3_C')]
    + [('fe_o_composition', 'oka2019_fig3_O'), ('fe_o_temperature', 'oka2019_fig4_T')],
)
def test_the_shipped_tables_are_the_first_read(table, read):
    """Every shipped eutectic table row inside the read range lies on read A of its figure."""
    shipped = alloy._table(table)
    _, x, ya, _ = _read(f'{read}.csv')
    scale = 1.0 if table == 'fe_o_temperature' else 100.0
    inside = (shipped[:, 0] >= x[0]) & (shipped[:, 0] <= x[-1])
    on_read = np.interp(shipped[inside, 0], x, ya) / scale
    assert inside.sum() > 40
    np.testing.assert_allclose(shipped[inside, 1], on_read, atol=0.06 if scale == 1.0 else 6e-6)


@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    ('name', 'el', 'p_gpa', 'rms_stated'),
    [('mori', 'S', 30, 2.6), ('mori', 'S', 130, 2.0), ('mori', 'S', 250, 4.6)]
    + [('mashino', 'C', 54, 46.5), ('mashino', 'C', 138, 175.1), ('mashino', 'C', 203, 103.5)]
    + [('mashino', 'C', 255, 105.1), ('mashino', 'C', 330, 70.1)],
)
def test_the_shipped_liquidus_misses_the_figure_curves_as_stated(name, el, p_gpa, rms_stated):
    """AlloyLiquidus with its own eutectic data against read A of the liquidus curves of Mori et
    al. (2017, Fig. 9) and Mashino et al. (2019, Fig. 8): the RMS [K] the docs state, kept from
    growing by more than 1 K; for C the shipped eutectic points differ from the ends of the drawn
    curves."""
    _, w, ta, _ = _read(f'{name}_liquidus_{p_gpa}GPa.csv')
    keep = w >= 0.0
    liq = AlloyLiquidus({})
    model = np.array([float(liq.t_melt(p_gpa * 1e9, {el: x / 100.0})) for x in w[keep]])
    assert np.sqrt(np.mean((model - ta[keep]) ** 2)) < rms_stated + 1.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_liquidus_slope_against_pressure(validation_plot):
    """The slope of the liquidus over 1 to 600 GPa for 4 alloys: positive for 2 wt% C, 4.8 wt% S
    and the mixed alloy, negative at 259 to 275 GPa for 5.4 wt% S near its eutectic."""
    p = np.linspace(1e9, 600e9, 1198)
    comps = {
        'C 2': {'C': 0.02},
        'S 4.8': {'S': 0.048},
        'S 5.4': {'S': 0.054},
        'mixed': {'S': 0.02, 'C': 0.005, 'O': 0.03, 'Si': 0.04},
    }
    slopes = {k: np.asarray(AlloyLiquidus(c).gradient(p)) * 1e9 for k, c in comps.items()}
    for k, g in slopes.items():
        low = (p > 259e9) & (p < 275e9)
        assert (g[low].min() < 0.0) if k == 'S 5.4' else (g.min() > 0.0), k

    def figure():
        import matplotlib.pyplot as plt

        win = (p >= 150e9) & (p <= 400e9)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for k, g in slopes.items():
            ax.plot(p[win][::8] / 1e9, g[win][::8], label=k)
        ax.axhline(0.0, color=plt.rcParams['text.color'], lw=0.8, ls='--')
        ax.set(xlabel='Pressure [GPa]', ylabel='dT/dP [K/GPa]')
        opaque_legend(ax, title='wt%', loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'core_alloy_liquidus_slope')


@pytest.mark.reference_pinned
def test_up_to_the_end_of_the_data_the_eutectic_is_the_data():
    """At every table node up to the end of each binary's data the eutectic composition (and the
    Fe-O temperature) is the shipped table, and at 330 GPa the anchors: 5 wt% S, 3.35 wt% C and
    the Fe-C Simon fit."""
    for el in ('S', 'C', 'O'):
        lo, hi = alloy._data()['range'][el]
        tab = alloy._table(f'fe_{el.lower()}_composition')
        inside = (tab[:, 0] >= lo) & (tab[:, 0] <= hi)
        got = np.asarray(eutectic(el, tab[inside, 0] * 1e9)[1])
        np.testing.assert_allclose(got, tab[inside, 1], rtol=1e-12)
    t_o = alloy._table('fe_o_temperature')
    inside = t_o[:, 0] <= alloy._data()['range']['O'][1]
    got = np.asarray(eutectic('O', t_o[inside, 0] * 1e9)[0])
    np.testing.assert_allclose(got, t_o[inside, 1], rtol=1e-12)
    assert float(eutectic('S', 330e9)[1]) == pytest.approx(0.05, rel=1e-12)
    assert float(eutectic('C', 330e9)[1]) == pytest.approx(0.0335, rel=1e-12)
    t_ref, p_ref, a, c = alloy._SIMON['C']
    simon = t_ref * ((330.0 - p_ref) / a + 1.0) ** (1.0 / c)
    assert float(eutectic('C', 330e9)[0]) == pytest.approx(simon, rel=1e-12)


def test_the_composition_curvature_at_zero_is_the_finite_difference():
    """d2T/dw2 of a traced composition at w = 0 equals the finite difference of dT/dw."""
    liq = AlloyLiquidus({})

    def t_of(w):
        return liq.t_melt(135e9, {'S': w})

    h = 1e-7
    fd = (float(jax.grad(t_of)(h)) - float(jax.grad(t_of)(0.0))) / h
    assert float(jax.hessian(t_of)(0.0)) == pytest.approx(fd, rel=1e-3)


def test_a_fraction_is_a_mass_fraction():
    """A numbers.Real such as Fraction(1, 50) is a mass fraction like 0.02."""
    from fractions import Fraction

    with_fraction = float(AlloyLiquidus({'S': Fraction(1, 50)}).t_melt(135e9))
    assert with_fraction == float(AlloyLiquidus({'S': 0.02}).t_melt(135e9))


def test_a_huge_fraction_is_refused_as_a_mass_fraction():
    """A numbers.Real too large for a float is refused with the mass-fraction error."""
    from fractions import Fraction

    with pytest.raises(ValueError, match='mass fraction'):
        AlloyLiquidus({'S': Fraction(10**400)})
