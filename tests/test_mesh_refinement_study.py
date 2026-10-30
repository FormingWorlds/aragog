"""Validation of the refined mesh from the stored convergence study of abe_solid on the PALEOS
low-temperature table (``tests/validation/data/mesh_refinement_study.json``, written by
``tools/verification/run_mesh_refinement_study.py``). The runs need that table, which CI does
not download, so the test reads the stored raw values, computes the noise floor, the spreads
and the verdicts from them, and checks every number that docs/Validation/mesh_refinement.md
quotes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.reference_pinned, pytest.mark.timeout(60)]

DATA = Path(__file__).parent / 'validation' / 'data' / 'mesh_refinement_study.json'
COMMIT = 'bbef9c9f139d7d200c0de98e5e010462d597d77c'
EOS_MD5 = '18d6f9eb5ba654d58fe3fad30ce118f0'
PAIR = {100: ('T50-100p', 'T50-100m'), 200: ('P+', 'P-')}
H_TOP = (200, 100, 50)


def _study():
    return json.loads(DATA.read_text())


def _metrics(v):
    """The compared results of one run: the global melt fraction, the CMB temperature, the heat
    lost through the surface and the mean surface heat flux, and the time-mean global melt
    fraction (a diagnostic)."""
    assert v['t_yr'][:4] == [1e2, 1e3, 1e4, 1e5]
    x = {}
    for i, t in ((1, '1e3'), (2, '1e4'), (3, '1e5')):
        x[f'Phi {t}'] = v['Phi'][i]
        x[f'T_cmb {t}'] = v['T_cmb_K'][i]
    x['heat_loss 1e5'] = -sum(v['F_int_J'][:4])
    for i, w in ((1, '1e2-1e3'), (2, '1e3-1e4'), (3, '1e4-1e5')):
        x[f'F_top {w}'] = v['F_top_mean_W_m2'][i]
    for t, m in v.get('Phi_time_mean', {}).items():
        x[f'Phi_mean {float(t):.0e}'.replace('+0', '')] = m
    return x


def _verdict(e, s_c, s_f, threshold=0.75):
    """Pass: s_f / s_c <= threshold or s_f <= 3e; with s_c <= 3e, converged only when s_f <= 3e."""
    if s_c <= 3 * e:
        return 'converged' if s_f <= 3 * e else 'no pass'
    if s_f <= 3 * e or s_f / s_c <= threshold:
        return 'pass'
    return 'no pass'


def _convergence(study, n):
    X = {r: _metrics(study[r]) for r in [f'T{q}-{n}' for q in H_TOP] + list(PAIR[n])}
    out = {}
    for k in X[f'T50-{n}']:
        e = abs(X[PAIR[n][0]][k] / X[PAIR[n][1]][k] - 1)
        s_c = abs(X[f'T200-{n}'][k] / X[f'T100-{n}'][k] - 1)
        s_f = abs(X[f'T100-{n}'][k] / X[f'T50-{n}'][k] - 1)
        out[k] = dict(e=e, s_c=s_c, s_f=s_f, verdict=_verdict(e, s_c, s_f))
    return X, out


def _recommendation(study):
    """Per result at 200 nodes: the deviation of each top cell from T50-200, the tolerance
    max(3 e, 1 %) and the largest top cell with it and every thinner top cell within the
    tolerance; and the node-count change |X(Tq-200) / X(Tq-100) - 1| per top cell q."""
    X2, c2 = _convergence(study, 200)
    X1, _ = _convergence(study, 100)
    out = {}
    for k, c in c2.items():
        tol = max(3 * c['e'], 0.01)
        dev = {q: abs(X2[f'T{q}-200'][k] / X2['T50-200'][k] - 1) for q in H_TOP}
        rec = next(q for q in H_TOP if all(dev[p] <= tol for p in H_TOP if p <= q))
        node = {q: abs(X2[f'T{q}-200'][k] / X1[f'T{q}-100'][k] - 1) for q in H_TOP}
        out[k] = dict(tol=tol, dev=dev, rec=rec, node=node)
    return out


_MAIN = [
    'Phi 1e3',
    'Phi 1e4',
    'Phi 1e5',
    'heat_loss 1e5',
    'F_top 1e2-1e3',
    'F_top 1e3-1e4',
    'F_top 1e4-1e5',
    'T_cmb 1e3',
    'T_cmb 1e4',
    'T_cmb 1e5',
]  # the 10 results of the pass rule; the time-mean melt fraction is a diagnostic
_PANELS = ('Phi 1e5', 'heat_loss 1e5', 'F_top 1e4-1e5', 'T_cmb 1e5')  # plot order
_TITLES = ('melt fraction', 'heat lost', 'surface heat flux', 'CMB temperature')
_FILES = ('melt_fraction', 'heat_lost', 'surface_heat_flux', 'cmb_temperature')
_MAX_SVG = 40_000  # bytes per validation plot file
FIGSIZE = (4.4, 4.4)  # inches, every plot of docs/Validation/mesh_refinement.md


def _check_size(figure, name):
    """Both theme files of the plot ``name`` are at most _MAX_SVG bytes."""
    from proteus_mpl.validation import render

    files, _ = render(figure, name)
    assert [len(data) <= _MAX_SVG for _, data in files] == [True, True], [
        (f, len(d)) for f, d in files
    ]


def _check_figure(fig, titles, series, legend):
    """The panel titles, the y-data of the plotted lines per panel and the legend texts."""
    axes = fig.axes
    assert [
        ''.join(ax.get_title(loc) for loc in ('left', 'center', 'right')) for ax in axes
    ] == list(titles)
    for ax, ys in zip(axes, series):
        got = [np.asarray(line.get_ydata(), float) for line in ax.get_lines()]
        assert len(got) == len(ys)
        for g, y in zip(got, ys):
            np.testing.assert_array_equal(g, y)
    texts = [
        t.get_text()
        for leg in [ax.get_legend() for ax in axes] + fig.legends
        if leg
        for t in leg.get_texts()
    ]
    assert texts == list(legend)


def _check_layout(figure):
    """In both themes every legend and text of ``figure()`` lies inside the figure, covers no
    point of a drawn line and no other text (display coordinates, 2 px margin), every text has
    the style size of its role and the figure has the size ``FIGSIZE`` of every page plot."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox
    from proteus_mpl import use
    from proteus_mpl.validation import _VARIANTS

    for variant, _ in _VARIANTS:
        use(variant)
        fig = figure()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        legends = [leg for leg in [ax.get_legend() for ax in fig.axes] + fig.legends if leg]
        boxes = [(f'legend {i}', leg.get_window_extent(r)) for i, leg in enumerate(legends)]
        role = {t: 'legend.fontsize' for leg in legends for t in leg.get_texts()}
        points = []
        for ax in fig.axes:
            texts = {t: 'axes.titlesize' for t in (ax.title, ax._left_title, ax._right_title)}
            texts |= {ax.xaxis.label: 'axes.labelsize', ax.yaxis.label: 'axes.labelsize'}
            for axis, key in ((ax.xaxis, 'xtick.labelsize'), (ax.yaxis, 'ytick.labelsize')):
                lo, hi = sorted(axis.get_view_interval())
                texts |= {
                    t.label1: key for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi
                }
            texts = {t: k for t, k in texts.items() if t.get_text()}
            boxes += [(t.get_text(), t.get_window_extent(r)) for t in texts]
            role |= texts
            for line in ax.get_lines():
                xy = line.get_transform().transform(np.c_[line.get_xdata(), line.get_ydata()])
                if line.get_linestyle() not in ('None', '') and len(xy) > 1:
                    s = np.linspace(0, 1, 50)[:, None, None]
                    xy = (xy[:-1] + s * (xy[1:] - xy[:-1])).reshape(-1, 2)
                points.append(xy[_inside(ax.bbox, xy)])
        points = np.concatenate(points)
        want = {
            k: FontProperties(size=plt.rcParams[k]).get_size_in_points() for k in role.values()
        }
        size = tuple(fig.get_size_inches())
        bad = [
            (t.get_text(), t.get_fontsize())
            for t, k in role.items()
            if t.get_fontsize() != want[k]
        ]
        bad += [('figure size', size)] if size != FIGSIZE else []
        bad += [
            (n, 'outside the figure')
            for n, b in boxes
            if not _inside(fig.bbox, b.get_points()).all()
        ]
        bad += [(n, 'covers a line') for n, b in boxes if _inside(b.padded(2), points).any()]
        bad += [
            (n1, n2)
            for i, (n1, b1) in enumerate(boxes)
            for n2, b2 in boxes[i + 1 :]
            if Bbox.intersection(b1.padded(2), b2.padded(2))
        ]
        plt.close(fig)
        assert not bad, (variant, bad)


def _inside(box, xy):
    """Whether each point of the (n, 2) array ``xy`` lies in ``box`` (edges included)."""
    (x0, y0), (x1, y1) = box.get_points()
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def _one_way(study, n):
    """Per result: whether X200, X100, X50 change in one direction at ``n`` nodes."""
    X, _ = _convergence(study, n)
    a, b, c = (X[f'T{q}-{n}'] for q in H_TOP)
    return {k: (b[k] - a[k]) * (c[k] - b[k]) > 0 for k in a}


def _frozen(v):
    f = v['first_frozen']
    return f['thaw_yr'] - f['freeze_yr']


def test_the_results_change_as_the_top_cell_gets_thinner(validation_plot):
    """The pass rule of ``_verdict`` per node count, the strict recommendation of
    ``_recommendation`` at 200 nodes, and every number of the case on the Validation page."""
    study = _study()
    head = study['header']
    assert head['commits'] == [COMMIT]
    runs = {k: v for k, v in study.items() if k != 'header'}
    assert all(v['eos_md5'] == EOS_MD5 and set(v['status']) == {0} for v in runs.values())
    for n, cmb in ((100, 9327.0), (200, 4638.0)):
        for q in H_TOP:
            v = runs[f'T{q}-{n}']
            assert v['nodes'] == n and v['cmb_cell_m'] == cmb and v['s_init'] == 9466.0
            assert abs(v['h_top_m'] / q - 1) < 1e-6 and abs(v['h_cmb_m'] / cmb - 1) < 1e-6
    for p, m in PAIR.values():
        assert runs[p]['s_init'] == 9466.0 * (1 + 1e-12) and runs[m]['s_init'] == 9466.0 * (
            1 - 1e-12
        )

    c100, c200 = _convergence(study, 100)[1], _convergence(study, 200)[1]
    passed = {k for k in _MAIN if c200[k]['verdict'] == 'pass'}
    assert passed == {'Phi 1e3', 'Phi 1e4', 'T_cmb 1e3', 'heat_loss 1e5', 'F_top 1e4-1e5'}
    assert sum(c100[k]['verdict'] == 'pass' for k in _MAIN) == 5
    ratio = {
        n: [c[k]['s_f'] / c[k]['s_c'] for k in _MAIN] for n, c in ((100, c100), (200, c200))
    }
    assert [f'{min(x for x in ratio[n] if x > 0.75):.3f}' for n in (100, 200)] == [
        '0.776',
        '0.790',
    ]
    lo = max(x for n in ratio for x in ratio[n] if x <= 0.75)
    hi = min(x for n in ratio for x in ratio[n] if x > 0.75)
    assert (f'{lo:.3f}', f'{hi:.3f}') == ('0.626', '0.776')
    for th in (lo, np.nextafter(hi, 0.0)):
        assert [
            sum(_verdict(c[k]['e'], c[k]['s_c'], c[k]['s_f'], th) == 'pass' for k in _MAIN)
            for c in (c100, c200)
        ] == [5, 5]
    assert [c200[f'T_cmb {t}']['s_f'] > c200[f'T_cmb {t}']['s_c'] for t in ('1e4', '1e5')] == [
        True,
        True,
    ]
    assert c100['T_cmb 1e5']['s_f'] < c100['T_cmb 1e5']['s_c']
    one_way = {n: _one_way(study, n) for n in (100, 200)}
    assert [
        sum(not one_way[n][k] for k in _MAIN if c[k]['verdict'] == 'pass')
        for n, c in ((100, c100), (200, c200))
    ] == [3, 4]
    assert c200['Phi 1e5']['verdict'] == c200['F_top 1e3-1e4']['verdict'] == 'no pass'
    three_e = [3 * c200[k]['e'] for k in _MAIN]
    assert f'{min(three_e):.1e}' == '7.4e-08' and f'{max(three_e):.1e}' == '1.4e-03'
    assert c200['T_cmb 1e4']['verdict'] == c200['T_cmb 1e5']['verdict'] == 'no pass'
    assert [c200[f'Phi_mean {t}']['verdict'] for t in ('1e3', '1e4', '1e5')] == [
        'pass',
        'pass',
        'no pass',
    ]

    rec = _recommendation(study)
    t_cmb_dev = max(
        rec[k]['dev'][q] for k in _MAIN if k.startswith('T_cmb') for q in (200, 100)
    )
    assert f'{t_cmb_dev:.1e}' == '3.4e-05'
    need_50 = {k for k in _MAIN if rec[k]['rec'] == 50}
    assert need_50 == {'Phi 1e3', 'Phi 1e4', 'Phi 1e5', 'F_top 1e3-1e4', 'F_top 1e4-1e5'}
    assert min(rec[k]['rec'] for k in _MAIN) == 50
    assert f'{rec["Phi 1e5"]["dev"][100]:.1%}' == '2.2%'
    assert f'{rec["F_top 1e4-1e5"]["dev"][100]:.1%}' == '1.7%'
    node = {
        k: rec[k]['node'][50]
        for k in ('T_cmb 1e5', 'heat_loss 1e5', 'Phi 1e5', 'F_top 1e4-1e5')
    }
    assert [f'{node[k]:.1e}' for k in ('T_cmb 1e5', 'heat_loss 1e5')] == ['4.0e-04', '3.5e-03']
    assert f'{node["Phi 1e5"]:.1%}' == '5.1%' and f'{node["F_top 1e4-1e5"]:.1%}' == '2.7%'
    s_f = {k: c200[k]['s_f'] for k in node}
    assert [f'{s_f[k]:.1e}' for k in ('T_cmb 1e5', 'heat_loss 1e5')] == ['3.2e-05', '5.8e-04']
    assert f'{s_f["Phi 1e5"]:.1%}' == '2.2%'
    assert all(node[k] > s_f[k] for k in node)

    no_core = [k for k, v in runs.items() if v['core_bc'] != 'core_module']
    assert len(no_core) == 17
    freeze = [runs[k]['first_frozen']['freeze_yr'] for k in no_core]
    assert (round(min(freeze)), round(max(freeze))) == (768, 791)
    assert [sum(runs[k]['top_cell_freezes'][:4]) for k in ('T50-100', 'T50-200')] == [61, 72]
    assert f'{runs["U100"]["h_top_m"] / 1e3:.1f}' == '10.9'
    frozen = {n: [_frozen(runs[f'T{q}-{n}']) for q in H_TOP] for n in (100, 200)}
    assert {n: [f'{x:.0f}' for x in f] for n, f in frozen.items()} == {
        100: ['85', '68', '64'],
        200: ['79', '58', '51'],
    }
    for n, f in frozen.items():
        e = abs(_frozen(runs[PAIR[n][0]]) / _frozen(runs[PAIR[n][1]]) - 1)
        assert _verdict(e, abs(f[0] / f[1] - 1), abs(f[1] / f[2] - 1)) == 'pass'
    assert [round(_frozen(runs[k])) for k in ('U100', 'U200', 'U400')] == [2851, 513, 255]
    uni = {r: _metrics(runs[r]) for r in ('U100', 'U200', 'U400')}
    step = [
        f'{abs(uni[b][k] / uni[a][k] - 1):.{d}%}'
        for k, a, b, d in (
            ('Phi 1e3', 'U100', 'U200', 2),
            ('Phi 1e3', 'U200', 'U400', 1),
            ('F_top 1e3-1e4', 'U100', 'U200', 1),
            ('F_top 1e3-1e4', 'U200', 'U400', 1),
        )
    ]
    assert step == ['0.21%', '7.6%', '3.4%', '3.6%']
    steps = [sum(runs[k]['nst'][:4]) for k in ('U100', 'T200-100', 'T50-100')]
    assert [f'{x:.1e}' for x in steps] == ['2.4e+05', '1.4e+06', '1.9e+06']

    L, T = runs['L50'], runs['T50-100']
    assert L['config_hash'] == T['config_hash'] and L['t_yr'][-1] == 1e8
    assert all(L[k][:4] == T[k][:4] for k in ('Phi', 'T_cmb_K', 'F_top_mean_W_m2', 'F_int_J'))
    assert [f'{x:.2g}' for x in L['Phi'][4:]] == ['0.065', '3.3e-05', '2.1e-05']
    assert (
        f'{L["F_top_mean_W_m2"][4]:.0f}' == '163'
        and f'{L["F_top_mean_W_m2"][6]:.3f}' == '0.096'
    )
    assert [f'{x:.1f}' for x in L['T_cmb_K'][4:]] == ['3465.0', '3440.3', '3440.3']
    assert abs(L['T_cmb_K'][6] / L['T_cmb_K'][5] - 1) < 1e-9

    assert f'{max(3 * c200[k]["e"] for k in _PANELS):.2%}' == '0.04%'
    assert f'{max(3 * c100[k]["e"] for k in _PANELS):.2%}' == '0.17%'
    assert [f'{runs[u]["h_cmb_m"]:.1f}' for u in ('U100', 'U200')] == ['9327.4', '4638.4']
    X = {r: _metrics(runs[r]) for r in runs if r[0] == 'T' and r[-3:] in ('100', '200')}
    ref = X['T50-200']

    series = {
        k: [
            np.array([100 * (X[f'T{q}-{n}'][k] / ref[k] - 1) for q in H_TOP])
            for n in (100, 200)
        ]
        for k in _PANELS
    }
    assert all(series[k][1][-1] == 0.0 for k in _PANELS)
    assert (
        f'{series["Phi 1e5"][1][1]:.1f}' == '2.2' or f'{-series["Phi 1e5"][1][1]:.1f}' == '2.2'
    )

    import matplotlib.pyplot as plt

    def make(k):
        def figure():
            fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
            fig.get_layout_engine().set(h_pad=0.05)
            band = plt.rcParams['text.color']
            ax.axhspan(-1, 1, color=band, alpha=0.12, lw=0, label='1 %')
            for (n, marker), y in zip(((100, 'o-'), (200, 's-')), series[k]):
                ax.plot(H_TOP, y, marker, label=f'{n} nodes')
            span = 1.3 * max(1.0, *(np.abs(y).max() for y in series[k]))
            ax.set(xscale='log', ylim=(-span, span))
            ax.set(xlabel='top cell [m]', ylabel='change [%]')
            ax.set_title(_TITLES[_PANELS.index(k)])
            ax.set_xticks(H_TOP, [str(q) for q in H_TOP])
            ax.minorticks_off()
            ax.locator_params(axis='y', nbins=4)
            fig.legend(loc='outside lower center', ncols=3, handlelength=1.2, columnspacing=1.0)
            return fig

        return figure

    for k, tail in zip(_PANELS, _FILES):
        figure = make(k)
        fig = figure()
        _check_figure(
            fig, [_TITLES[_PANELS.index(k)]], [series[k]], ['1 %', '100 nodes', '200 nodes']
        )
        plt.close(fig)
        _check_layout(figure)
        _check_size(figure, f'mesh_refinement_convergence_{tail}')
        validation_plot(figure, f'mesh_refinement_convergence_{tail}')


def _pattern(v, i):
    """Sign changes of S - S_sol between neighbours in the lowest 48 cells at the end of part
    i, and the number of cells within 5 J/kg/K of the solidus."""
    d = np.subtract(v['S_end'][i], v['S_sol'])
    sign = np.sign(d[:48])
    return int(np.sum(sign[1:] != sign[:-1])), int(np.sum(np.abs(d) < 5))


def test_the_alternating_pattern_from_a_perturbed_start_and_at_200_nodes():
    """The 50 m run from an initial entropy changed by 1e-12 and the 50 m top cell at 200 nodes,
    both to 10^8 years: the sign changes of S - S_sol and the cells near the solidus, as on the
    page."""
    from tests.test_mesh_refinement import _mesh

    study = _study()
    L, Lp, T, T0 = study['L50'], study['L50p'], study['T50-200-long'], study['T50-200']
    assert abs(Lp['s_init'] / L['s_init'] - 1 - 1e-12) < 1e-15 and set(Lp['status']) == {0}
    assert [_pattern(Lp, i)[0] for i in (5, 6)] == [37, 10]
    sign = [
        np.sign(np.subtract(v['S_end'][i], v['S_sol'])[:48]) for i in (5, 6) for v in (L, Lp)
    ]
    assert [int(np.sum(sign[k] == sign[k + 1])) for k in (0, 2)] == [46, 48]

    assert T['config_hash'] == T0['config_hash'] and T['t_yr'][-1] == 1e8
    assert set(T['status']) == {0}
    assert all(T[k][:4] == T0[k][:4] for k in ('Phi', 'T_cmb_K', 'F_top_mean_W_m2', 'F_int_J'))
    mid = []
    for v in (L, T):
        r = np.ravel(_mesh(v['surface_cell_m'], v['cmb_cell_m'], True, v['nodes']).basic.radii)
        mid.append(0.5 * (r[1:] + r[:-1]))
    sel = (mid[1] >= mid[0][0]) & (mid[1] <= mid[0][47])
    d = [np.subtract(T['S_end'][i], T['S_sol']) for i in (5, 6)]
    assert int(sel.sum()) == 102 and d[0].size == 199
    assert [int(np.sum(np.diff(np.sign(x[sel])) != 0)) for x in d] == [36, 12]
    assert [int(np.sum(np.abs(x) < 5)) for x in d] == [116, 82]


def test_the_long_run_and_the_tables():
    """The 10^8 year runs on the 50 m and the uniform mesh, the fill plateau of the solid table
    at the CMB and the melt table against the PALEOS liquid table at 1 bar, as on the page."""
    study = _study()
    head, L, U, U0 = study['header'], study['L50'], study['U100-long'], study['U100']
    assert head['md5']['temperature_solid.dat'] == EOS_MD5
    assert (
        U['config_hash'] == U0['config_hash']
        and U['t_yr'][-1] == 1e8
        and set(U['status']) == {0}
    )
    assert all(U[k][:4] == U0[k][:4] for k in ('Phi', 'T_cmb_K', 'F_top_mean_W_m2', 'F_int_J'))
    rows = [
        [f'{v["Phi"][i]:.2g}' for i in (4, 5, 6)]
        + [f'{v["T_cmb_K"][i]:.1f}' for i in (4, 5, 6)]
        + [f'{v["F_top_mean_W_m2"][i]:.2g}' for i in (4, 5, 6)]
        for v in (U, L)
    ]
    assert rows[0] == [
        '0.12',
        '0.12',
        '0.079',
        '3495.6',
        '3493.1',
        '3463.3',
        '55',
        '0.61',
        '0.71',
    ]
    assert rows[1] == [
        '0.065',
        '3.3e-05',
        '2.1e-05',
        '3465.0',
        '3440.3',
        '3440.3',
        '1.6e+02',
        '8',
        '0.096',
    ]

    assert f'{L["F_top_mean_W_m2"][5]:.1f}' == '8.0'
    assert [_pattern(L, i) for i in (4, 5, 6)] == [(0, 0), (39, 53), (10, 38)]
    assert [_pattern(U, i) for i in (4, 5, 6)] == [(0, 0)] * 3
    closest = min(np.abs(np.subtract(e, U['S_sol'])).min() for e in U['S_end'])
    assert 144 <= closest < 145

    c = head['solid_at_cmb']
    (p1, p2), (s_valid, s_fill), T = c['P_nodes_Pa'], c['S_nodes'], c['T_K']
    assert c['valid'] == [[1, 1], [0, 0]] and T[1] == T[0] and p1 < c['P_cmb_Pa'] < p2
    w = (c['P_cmb_Pa'] - p1) / (p2 - p1)
    plateau = T[1][0] + w * (T[1][1] - T[1][0])
    assert all(abs(L['T_cmb_K'][i] / plateau - 1) < 1e-12 for i in (5, 6))
    s_bot = L['S_bottom_cell'][5:]
    assert s_valid < s_bot[1] < s_bot[0] < s_fill and s_valid < L['S_sol'][0]
    assert all(x > s_fill for x in U['S_bottom_cell'][5:])
    assert f'{U["F_cmb_J"][6] / L["F_cmb_J"][6]:.0f}' == '85'
    assert all(U['Phi'][i] / L['Phi'][i] > 1e3 for i in (5, 6)) and len(L['S_sol']) == 99
    assert [f'{x:.2f}' for x in (s_valid, s_fill, *s_bot)] == [
        '3070.70',
        '3150.82',
        '3111.09',
        '3110.24',
    ]
    assert [f'{x:.2f}' for x in T[0]] == [
        '3439.49',
        '3443.24',
    ] and f'{c["P_cmb_Pa"] / 1e9:.2f}' == '43.38'

    lv, sol = head['solid_last_valid'], head['solidus']
    p_node, s_last, grid = (np.array(lv[k]) for k in ('P_nodes_Pa', 'S', 'S_grid'))
    assert p_node.size == 1350
    s_sol = np.interp(p_node, sol['P_Pa'], sol['S'])
    between = np.array([np.sum((grid > a) & (grid < b)) for a, b in zip(s_last, s_sol)])
    above = s_sol > s_last
    assert f'{np.diff(grid).min():.1f}' == f'{np.diff(grid).max():.1f}' == '80.1'
    assert [
        int(above.sum()),
        int(np.sum(above & (between == 0))),
        int(np.sum(between == 1)),
    ] == [1301, 853, 448]
    cells = {}
    for k, v in study.items():
        if k != 'header':
            j = np.searchsorted(p_node, v['P_cell_Pa'])
            cells[k] = np.array(v['S_sol']) > np.minimum(s_last[j - 1], s_last[j])
    assert all(c[-1] for c in cells.values())
    assert [
        int(cells[k].sum()) for k in ('T50-100', 'L50', 'C-cmb9327', 'T50-200', 'U100')
    ] == [93, 93, 93, 185, 86]

    m, r = head['melt_at_1bar'], head['raw_liquid_at_1bar']
    assert r['md5'] == '4b5c14255ca845880a2e940110861df1'
    tr, rr, sr = (np.array(r[k]) for k in ('T_K', 'rho', 's'))
    tm = np.array(m['T_K'])
    above = tr >= tm.min()
    assert np.all(np.diff(sr[above]) > 0) and tm.max() <= tr.max()
    d_rho = np.abs(np.array(m['rho']) / np.interp(tm, tr, rr) - 1).max()
    d_s = np.abs(np.array(m['S']) - np.interp(tm, tr, sr)).max()
    assert len(tm) == 122 and [f'{x:.0f}' for x in (tm.min(), tm.max())] == ['1831', '4862']
    assert f'{d_rho:.1e}' == '3.4e-05' and f'{d_s:.2f}' == '0.34'
    fall = np.flatnonzero(np.diff(sr) <= 0)
    assert fall.size == 1 and [f'{tr[fall[0] + q]:.0f}' for q in (0, 1)] == ['1595', '1620']
    assert [f'{rr[fall[0] + q]:.0f}' for q in (0, 1)] == ['2227', '3863']


def test_a_coarsened_cmb_cell_changes_the_core_results(validation_plot):
    """The core-module pair with a 9327 m and an unset CMB cell against the noise floor of a
    perturbed copy of the first, as quoted on the page."""
    study = _study()
    on, off, p = study['C-cmb9327'], study['C-off'], study['C-on_p']
    for v in (on, off, p):
        assert v['core_bc'] == 'core_module' and set(v['status']) == {0}
        assert (
            v['t_yr'] == [1e2, 1e3, 1e4, 1e5]
            and v['nodes'] == 100
            and v['surface_cell_m'] == 50
        )
    assert (
        on['s_init'] == 9466.0
        and p['s_init'] == 9466.0 * (1 + 1e-12)
        and off['s_init'] == 9466.0
    )
    assert on['cmb_cell_m'] == p['cmb_cell_m'] == 9327.0 and off['cmb_cell_m'] == 0.0
    assert [f'{v["h_cmb_m"]:.0f}' for v in (on, off)] == ['9327', '38456']

    d, e = {}, {}
    for k in ('T_core_K', 'F_cmb_J'):
        d[k] = np.abs(np.array(on[k]) / np.array(off[k]) - 1)
        e[k] = np.abs(np.array(on[k]) / np.array(p[k]) - 1)
    ratio = np.concatenate([d[k] / (3 * e[k]) for k in d])
    assert ratio.min() > 1 and [f'{x:.1e}' for x in (ratio.min(), ratio.max())] == [
        '3.4e+02',
        '3.8e+03',
    ]
    dT = np.array(on['T_core_K']) - np.array(off['T_core_K'])
    assert np.all(dT < 0) and f'{-dT.min():.2f}' == '0.51' and int(np.argmin(dT)) == 2
    assert f'{d["T_core_K"].max():.1e}' == '1.2e-04' and f'{d["F_cmb_J"].max():.2%}' == '0.42%'
    assert int(np.argmax(d['F_cmb_J'])) == 3 and int(np.argmax(d['T_core_K'])) == 2
    gap = np.array(on['T_cmb_K']) - np.array(on['T_core_K'])
    assert [f'{x:.0f}' for x in gap] == ['-2', '23', '37', '39']

    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    titles = {'T_core_K': 'core temperature', 'F_cmb_J': 'heat across the CMB'}
    files = {'T_core_K': 'core_temperature', 'F_cmb_J': 'heat'}

    def make(k):
        def figure():
            fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
            fig.get_layout_engine().set(h_pad=0.05)
            ax.plot(on['t_yr'], d[k], 'o-', label='difference')
            ax.plot(on['t_yr'], 3 * e[k], 's--', label='noise floor 3e')
            ax.set(xscale='log', yscale='log', xlabel='time [yr]', ylabel='relative difference')
            ax.set_title(titles[k])
            ax.yaxis.set_major_locator(LogLocator(numticks=4))
            ax.minorticks_off()
            fig.legend(loc='outside lower center', ncols=2, handlelength=1.6)
            return fig

        return figure

    for k in d:
        figure = make(k)
        fig = figure()
        _check_figure(fig, [titles[k]], [[d[k], 3 * e[k]]], ['difference', 'noise floor 3e'])
        plt.close(fig)
        _check_layout(figure)
        _check_size(figure, f'mesh_refinement_cmb_{files[k]}')
        validation_plot(figure, f'mesh_refinement_cmb_{files[k]}')


@pytest.mark.parametrize(
    ('e', 's_c', 's_f', 'verdict'),
    [
        (1.0, 3.0, 3.0, 'converged'),
        (1.0, 3.0, 3.1, 'no pass'),
        (1.0, 4.0, 3.0, 'pass'),
        (1.0, 8.0, 6.0, 'pass'),
        (1.0, 8.0, 6.1, 'no pass'),
        (0.0, 1.0, 0.5, 'pass'),
    ],
)
def test_the_pass_rule_on_every_branch(e, s_c, s_f, verdict):
    """Converged needs both spreads within 3e; otherwise s_f within 3e or s_f / s_c <= 0.75."""
    assert _verdict(e, s_c, s_f) == verdict


# Largest staggered radius shift between platforms that the stored-mesh test allows [m]:
# 6 times the 16.6 mm measured between a Linux runner and the stored macOS rebuild.
PLATFORM_DR = 0.1


def test_every_stored_mesh_is_rebuilt_at_this_commit():
    """The mesh of each stored run, built here from abe_solid.toml with the run's nodes and end
    cells, has the stored top and CMB cell to 1e-9 and, at each staggered node, the staggered
    pressure that the study tool rebuilt to the local |dP/dr| times PLATFORM_DR; each run warns
    with its largest pressure difference."""
    import warnings

    from tests.test_mesh_refinement import _mesh

    h, p = [], []
    for name, v in _study().items():
        if name == 'header':
            continue
        m = _mesh(v['surface_cell_m'], v['cmb_cell_m'], True, v['nodes'])
        r, rs = np.ravel(m.basic.radii), np.ravel(m.staggered.radii)
        P = np.ravel(m.staggered_pressure)
        h.append(
            np.abs(np.array([r[-1] - r[-2], r[1] - r[0]]) / [v['h_top_m'], v['h_cmb_m']] - 1)
        )
        dP = np.abs(P - v['P_cell_Pa'])
        p.append(np.max(dP / (np.abs(np.gradient(P, rs)) * PLATFORM_DR)))
        warnings.warn(f'{name}: max |P - stored| {dP.max():.6g} Pa', stacklevel=1)
    assert len(h) == 20
    assert np.max(h) <= 1e-9
    assert max(p) <= 1.0
