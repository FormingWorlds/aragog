"""The lid hands over in one case, and the mesh carries the crust after it: the
1000 km column on 100, 200 and 400 nodes against 2 runs without the lid on refined top cells, from
stored runs (``tests/reference/build_surface_lid_v8.py``); they take days on a cluster.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_legend_clear, opaque_legend

pytestmark = [pytest.mark.unit]

REF = Path(__file__).resolve().parent / 'reference' / 'surface_lid_v8.npz'
MERGE = REF.with_name('surface_lid_v8_merge.json')
MESHES = ('n100', 'n200', 'n400')
PART_ENDS = np.arange(2, 11) * 1.0e8
WINDOW = 2.0e7  # years: longer than the swings of the flux after the transition
LATE = 5.0e8  # years: the transition of the interior lies before this
R_MAX = 0.75  # finest mesh step over the coarser step after the last hand-over
EVENT_LIMIT = 0.05  # limit on the flux step at a hand-over, missed by all 3 meshes


@pytest.fixture(scope='module')
def d():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def _at(d, run, key, t):
    return np.interp(t, d[f'{run}_t'], d[f'{run}_{key}'])


def _window_mean(d, run, key, t, start):
    """Time mean of a series over ``WINDOW`` around each ``t``, from ``start`` on."""
    keep = d[f'{run}_t'] >= start
    x, v = d[f'{run}_t'][keep], d[f'{run}_{key}'][keep].astype(float)
    area = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(x))]
    return (np.interp(t + WINDOW / 2, x, area) - np.interp(t - WINDOW / 2, x, area)) / WINDOW


def _last_handover(d):
    return max(float(d[f'{m}_handover_yr']) for m in MESHES)


def _late_mean(d, run, key):
    t = np.linspace(LATE, 1.0e9 - WINDOW, 1000)
    return float(np.mean(_window_mean(d, run, key, t, _last_handover(d))))


def _flux_step(d, m):
    """Flux just before and just after the merge of mesh ``m``, at its stored states."""
    t, F, delta = d[f'{m}_t'], d[f'{m}_F_top'].astype(float), d[f'{m}_delta']
    i = int(np.flatnonzero(delta == 0.0)[0])
    return t[i - 1], F[i - 1], F[i]


@pytest.mark.reference_pinned
def test_v8_the_lid_hands_over_at_the_growth_stop_on_all_meshes(d):
    """The lid hands over on all 3 meshes; each lid sits at 0.8 of its top cell before the hand-over,
    and the hand-over times agree within 3 %."""
    times = [float(d[f'{m}_handover_yr']) for m in MESHES]
    assert max(times) / min(times) < 1.03
    for m, cap in zip(MESHES, (8693.27, 4326.35, 2158.14)):
        t, delta = d[f'{m}_t'], d[f'{m}_delta']
        before = (t > 1.0e6) & (t < float(d[f'{m}_handover_yr']))
        assert delta[before] == pytest.approx(cap, rel=1e-5)
        assert np.all(delta[t > float(d[f'{m}_handover_yr']) + 2.0e6] == 0.0)
        assert d[f'{m}_w_active'][-1] == 1.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v8_the_mesh_converges_after_the_hand_over(d, validation_plot):
    """After the last hand-over the step from 200 to 400 nodes is at most R_MAX of the step from
    100 to 200 nodes in the surface flux and the lid-base isotherm depth, at the part ends. In the
    late time mean the flux of 200 and 400 nodes lies within 1 % of the reference and that of 100
    nodes more than 20 % below it; the 2 references agree within 1e-3 there."""
    for key in ('F_top', 'z_lid'):
        s_c = np.abs(_at(d, 'n100', key, PART_ENDS) / _at(d, 'n200', key, PART_ENDS) - 1)
        s_f = np.abs(_at(d, 'n200', key, PART_ENDS) / _at(d, 'n400', key, PART_ENDS) - 1)
        assert s_f.max() / s_c.max() <= R_MAX
    ref = _late_mean(d, 's500', 'F_top')
    assert _late_mean(d, 's1000', 'F_top') == pytest.approx(ref, rel=1e-3)
    for m in ('n200', 'n400'):
        assert _late_mean(d, m, 'F_top') == pytest.approx(ref, rel=1e-2)
    assert _late_mean(d, 'n100', 'F_top') < 0.8 * ref
    ref_z = _late_mean(d, 's500', 'z_lid')
    assert _late_mean(d, 's1000', 'z_lid') == pytest.approx(ref_z, rel=2e-2)
    assert _late_mean(d, 'n400', 'z_lid') == pytest.approx(ref_z, rel=5e-2)
    assert _late_mean(d, 'n200', 'z_lid') == pytest.approx(ref_z, rel=1e-1)
    flux = [_late_mean(d, m, 'F_top') / ref for m in ('n100', 'n200', 'n400', 's1000')]
    depth = [_late_mean(d, m, 'z_lid') / ref_z for m in ('n100', 'n200', 'n400', 's1000')]
    np.testing.assert_allclose(flux, (0.777, 1.008, 0.999, 1.000), atol=5.0e-4)
    np.testing.assert_allclose(depth, (1.55, 1.09, 1.05, 1.01), atol=5.0e-3)

    def figure():
        import matplotlib.pyplot as plt

        start = _last_handover(d)
        t = np.linspace(start + WINDOW, 1.0e9 - WINDOW, 18)
        fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.4), sharex=True)
        ink = plt.rcParams['text.color']
        for key, ax in zip(('F_top', 'z_lid'), axes):
            base = _window_mean(d, 's500', key, t, start)
            for run, c, ls in (
                ('n100', 'C1', '-'),
                ('n200', 'C2', '--'),
                ('n400', 'C3', '-.'),
                ('s1000', ink, ':'),
            ):
                label = f'{run[1:]} nodes' if run != 's1000' else '1000 m cell'
                ratio = _window_mean(d, run, key, t, start) / base
                ax.plot(t / 1e8, ratio, color=c, ls=ls, label=label)
            ax.axhline(1.0, color=ink, lw=0.8)
            ax.grid(False)
        axes[0].set_ylabel('flux ratio')
        axes[1].set_ylabel('depth ratio')
        axes[1].set_xlabel('time (100 Myr)')
        axes[0].set_ylim(0.5, 1.25)
        axes[1].set_ylim(0.9, 2.1)
        opaque_legend(axes[0], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4)
        fig.tight_layout()
        assert_legend_clear(fig)
        fig.canvas.draw()
        boxes = [ax.yaxis.label.get_window_extent() for ax in axes]
        assert not boxes[0].overlaps(boxes[1])
        return fig

    validation_plot(figure, 'surface_lid_v8_after_the_hand_over')


@pytest.mark.reference_pinned
def test_v8_the_flux_before_the_hand_over_lies_below_the_reference(d):
    """From 1e6 yr, the first stored state, to the last stored state with the lid the flux lies
    below both references. Against 500 m by up to 13 % on 100 and 200 nodes and 4.8 % on 400 nodes,
    against 1000 m by 13, 12 and 4.1 %; by 2, 0.9 and 0.4 % at that last state."""
    ratio, coarse = {}, {}
    for m in MESHES:
        i = int(np.flatnonzero(d[f'{m}_delta'] == 0.0)[0])
        t, F = d[f'{m}_t'][:i], d[f'{m}_F_top'][:i].astype(float)
        assert t[0] == 1.0e6
        ratio[m] = F / np.interp(t, d['s500_t'], d['s500_F_top'].astype(float))
        coarse[m] = F / np.interp(t, d['s1000_t'], d['s1000_F_top'].astype(float))
    assert all(r.max() < 1.0 for r in coarse.values())
    np.testing.assert_allclose(
        [r.min() for r in coarse.values()], (0.872, 0.881, 0.959), atol=1e-3
    )
    assert ratio['n200'].argmin() == ratio['n400'].argmin() == 0
    assert all(r.max() < 1.0 for r in ratio.values())
    np.testing.assert_allclose(
        [r.min() for r in ratio.values()], (0.870, 0.874, 0.952), atol=1e-3
    )
    np.testing.assert_allclose(
        [r[-1] for r in ratio.values()], (0.981, 0.991, 0.996), atol=1e-3
    )


@pytest.mark.reference_pinned
def test_v8_the_flux_step_at_the_hand_over_misses_its_limit(d):
    """The limit of EVENT_LIMIT on the flux step at a hand-over is missed: from the stored
    state at the hand-over to the next one the flux rises by 15 to 27 %, at least 80 times the step
    of the reference over the same time. The excess falls below 5 % sooner on a finer mesh."""
    for m, low, high in zip(MESHES, (0.25, 0.13, 0.18), (0.30, 0.17, 0.22)):
        t0, before, after = _flux_step(d, m)
        step = after / before - 1
        assert step > EVENT_LIMIT
        assert low < step < high
        i = int(np.flatnonzero(d[f'{m}_delta'] == 0.0)[0])
        ref_step = abs(_at(d, 's500', 'F_top', d[f'{m}_t'][i]) - _at(d, 's500', 'F_top', t0))
        assert after - before > 80.0 * ref_step
        assert before == pytest.approx(_at(d, 's500', 'F_top', t0), rel=2.5e-2)
    late = []
    for m in MESHES:
        t = d[f'{m}_t']
        after = t > float(d[f'{m}_handover_yr']) + 1.0e5
        excess = d[f'{m}_F_top'][after] / _at(d, 's500', 'F_top', t[after]) - 1
        late.append((t[after][np.argmax(excess < 0.05)] - float(d[f'{m}_handover_yr'])) / 1e6)
    np.testing.assert_allclose(late, (28.9, 8.0, 2.1), rtol=0.05)


@pytest.mark.reference_pinned
def test_v8_the_merge_states_and_the_offset_of_the_growth_stop():
    """From ``tools/verification/run_surface_lid_merge_state.py``: the merge keeps the energy, the
    surface node falls below the equilibrium temperature, so the flux is negative for up to about
    900 yr, and the top node before the merge is warmer than the run without the lid by the offset
    (delta - h/2) (T_b - T_s) / delta within 5 %."""
    runs = json.loads(MERGE.read_text())
    assert list(runs) == list(MESHES)
    for d, years in zip(runs.values(), (923.0, 448.0, 195.0)):
        assert d['merge_residual'] < 1e-15
        assert d['after']['flux'] < -20.0 < 0.0 < d['before']['flux']
        assert d['after']['T_node'] < 273.0 < d['before']['T_node']
        assert d['after']['T_cell'] < d['before']['T_cell']
        assert d['negative_yr'] == pytest.approx(years, rel=1e-2)
        assert d['offset_measured'] == pytest.approx(d['offset_predicted'], rel=5e-2)
    assert runs['n100']['offset_measured'] > 1.8 * runs['n200']['offset_measured']
    assert runs['n200']['offset_measured'] > 1.8 * runs['n400']['offset_measured']
