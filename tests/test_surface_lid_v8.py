"""The lid hands over in a case with its gate open and the mesh carries the crust after it: the
1000 km column on 100, 200 and 400 nodes against 2 runs without the lid on refined top cells, from
stored runs (``tests/reference/build_surface_lid_v8.py``); they take days on a cluster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_legend_clear, opaque_legend

pytestmark = [pytest.mark.unit]

REF = Path(__file__).resolve().parent / 'reference' / 'surface_lid_v8.npz'
MESHES = ('n100', 'n200', 'n400')
PART_ENDS = np.arange(2, 11) * 1.0e8
WINDOW = 2.0e7  # years: longer than the swings of the flux after the transition
LATE = 5.0e8  # years: the transition of the interior lies before this
R_MAX = 0.75  # approved: finest mesh step over the coarser step after the last hand-over
EVENT_LIMIT = 0.05  # approved limit on the flux step at a hand-over, missed by all 3 meshes


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
    """The gate opens on all 3 meshes, each lid sits at 0.8 of its top cell before the hand-over,
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

    def figure():
        import matplotlib.pyplot as plt

        start = _last_handover(d)
        t = np.linspace(start + WINDOW, 1.0e9 - WINDOW, 45)
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
                label = (
                    f'{run[1:]} nodes with lid' if run != 's1000' else 'reference, 1000 m cell'
                )
                ratio = _window_mean(d, run, key, t, start) / base
                ax.plot(t / 1e8, ratio, color=c, ls=ls, label=label)
            ax.axhline(1.0, color=ink, lw=0.8)
        axes[0].set_ylabel('flux ratio')
        axes[1].set_ylabel('depth ratio')
        axes[1].set_xlabel('time ($10^8$ yr)')
        opaque_legend(axes[0], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
        fig.tight_layout()
        assert_legend_clear(fig)
        fig.canvas.draw()
        boxes = [ax.yaxis.label.get_window_extent() for ax in axes]
        assert not boxes[0].overlaps(boxes[1])
        return fig

    validation_plot(figure, 'surface_lid_v8_after_the_hand_over')


@pytest.mark.reference_pinned
def test_v8_the_flux_step_at_the_hand_over_misses_its_limit(d):
    """The approved limit of EVENT_LIMIT on the flux step at a hand-over is missed: the flux rises
    by 15 to 27 % at the merge, at least 80 times the step of the reference over the same time."""
    for m, low, high in zip(MESHES, (0.25, 0.13, 0.18), (0.30, 0.17, 0.22)):
        t0, before, after = _flux_step(d, m)
        step = after / before - 1
        assert step > EVENT_LIMIT
        assert low < step < high
        i = int(np.flatnonzero(d[f'{m}_delta'] == 0.0)[0])
        ref_step = abs(_at(d, 's500', 'F_top', d[f'{m}_t'][i]) - _at(d, 's500', 'F_top', t0))
        assert after - before > 80.0 * ref_step
        assert before == pytest.approx(_at(d, 's500', 'F_top', t0), rel=2.5e-2)
