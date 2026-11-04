"""The sub-grid lid on the stagnant-lid benchmark rows of the SSC harness.

The rows of Korenaga (2009, Table 1) and of Deschamps and Vilella (2021, Table 1) run at the
calibration point (0.25, 0.25) on 100 nodes, with constant properties and a fixed surface
temperature, in 3 ways: lid off with the 'local' closure (the calibrated runs), lid off with the
'lid' closure, and lid on with the 'lid' closure. ``tools/verification/run_surface_lid_v5_rows.py``
stores the runs in ``tests/reference/surface_lid_v5_rows.json``, whose ``meta.commit`` names the
code that wrote them. The first test checks the stored runs; the slow tests run rows again.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_legend_clear, opaque_legend
from tools.verification.run_surface_lid_v5_rows import OBSERVABLES, run_row

REF = Path(__file__).resolve().parent / 'reference' / 'surface_lid_v5_rows.json'
CALIBRATION = (
    Path(__file__).resolve().parent / 'reference' / 'ssc_mlt_top_slope_calibration.json'
)
SAME = 1e-5  # lid on against lid off under the 'lid' closure, each observable
GUARD = 1e-4  # gross ledger guard per call: time quadrature of the start transient
SUB = 1e-2  # lid sub-ledger per call that starts with the lid
FIRST_OUTPUT_YR = 5e9  # first output of both runners: 4e10 yr in 8 parts, 2e11 yr in 40
TABLES = {'K': 'korenaga_2009_rows', 'D': 'deschamps_vilella_2021_rows'}


def _assert_text_clear_of_lines(fig, ax) -> None:
    """Assert that no text of the axes covers a marker or a line point (50 per segment)."""
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in ax.texts]
    for line in ax.lines:
        xy = line.get_transform().transform(line.get_xydata())
        pts = xy  # a marker-only series has no segments
        if line.get_linestyle() not in ('None', '', ' ') and len(xy) > 1:
            pts = np.concatenate([np.linspace(a, b, 50) for a, b in zip(xy[:-1], xy[1:])])
        for box in boxes:
            inside = (box.x0 <= pts[:, 0]) & (pts[:, 0] <= box.x1)
            assert not (inside & (box.y0 <= pts[:, 1]) & (pts[:, 1] <= box.y1)).any(), box


def _change(on: dict, off: dict, table: str) -> float:
    """Largest relative change of an observable of a row from lid off to lid on."""
    return max(abs(on[k] / off[k] - 1.0) for k in OBSERVABLES[table])


@pytest.mark.unit
def test_stored_rows_keep_their_values_with_the_lid(validation_plot):
    """Every row with the lid equals the row without it within SAME and hands the lid to the
    mesh before the first output; each call closes the gross ledger within GUARD and each call
    with the lid its lid sub-ledger within SUB; the lid off 'local' runs equal the calibration
    and the counts outside the published tolerances agree in all 3 ways."""
    d = json.loads(REF.read_text())
    cal = json.loads(CALIBRATION.read_text())['evaluated_points']['minimax_0p25_0p25']
    assert (d['meta']['s_top_inf'], d['meta']['c_theta'], d['meta']['n_nodes']) == (
        0.25,
        0.25,
        100,
    )
    change = {}
    for t, name in TABLES.items():
        rows = sorted(cal[name])
        counts = {w: 0 for w in ('off:local', 'off:lid', 'on:lid')}
        for row in rows:
            r = {w: d[f'{t}:{row}:{w}'] for w in counts}
            for w in counts:
                counts[w] += r[w]['exceeds']
            for k in OBSERVABLES[t]:
                assert r['off:local'][k] == cal[name][row][k], (row, k)
            on = r['on:lid']
            change[(t, row)] = _change(on, r['off:lid'], t)
            assert 0.0 < change[(t, row)] <= SAME, row
            assert on['handover_t'] is not None and on['handover_t'] < FIRST_OUTPUT_YR, row
            assert on['guard_max'] <= GUARD, row
            assert on['sub'] and max(abs(x) for x in on['sub']) <= SUB, row
        assert len(set(counts.values())) == 1, (t, counts)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ink = plt.rcParams['text.color']
        pos, ticks, labels = 0, [], []
        for t, head in (('K', 'Korenaga 2009'), ('D', 'Deschamps and Vilella 2021')):
            rows = sorted(r for tt, r in change if tt == t)
            x = pos + np.arange(1, len(rows) + 1)
            y = np.array([change[(t, r)] for r in rows])
            ax.semilogy(x, y, 'o', color='C1', ms=4, label='lid on vs off')
            ax.text(x.mean(), 1e-4, head, ha='center', va='center', fontsize=8)
            ticks += [x[0], x[-1]]
            labels += ['1', str(len(rows))]
            pos = x[-1] + 2
            if t == 'K':
                ax.axvline(pos - 0.5, color=ink, ls=':', lw=0.8)
        ax.axhline(SAME, color='C3', ls='--', lw=1.0, label='tolerance')
        ax.set_xticks(ticks, labels)
        ax.set_ylim(1e-17, 1e-3)
        ax.set_yticks([1e-15, 1e-10, 1e-5])
        ax.minorticks_off()
        ax.set_xlabel('row in the table')
        ax.set_ylabel('largest relative change')
        handles, names = ax.get_legend_handles_labels()
        keep = dict(zip(names, handles))
        opaque_legend(
            ax,
            handles=list(keep.values()),
            labels=list(keep.keys()),
            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=8,
        )
        fig.tight_layout()
        assert_legend_clear(fig)
        _assert_text_clear_of_lines(fig, ax)
        return fig

    validation_plot(figure, 'surface_lid_v5_rows')


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(('table', 'row'), [('K', 'E230a'), ('D', 'f60_H4')])
def test_row_keeps_its_values_with_the_lid(table, row):
    """A row run now with and without the lid under the 'lid' closure gives the same
    observables within SAME, hands the lid over before the first output and closes its lid
    sub-ledger within SUB."""
    on = run_row(table, row, True, 'lid')
    off = run_row(table, row, False, 'lid')
    assert _change(on['result'], off['result'], table) <= SAME
    assert on['handover_t'] is not None and on['handover_t'] < FIRST_OUTPUT_YR
    sub = [c['lid']['miss_rel'] for c in on['calls'] if 'lid' in c]
    assert sub and max(abs(x) for x in sub) <= SUB


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_deschamps_vilella_row_with_the_lid_at_the_growth_stop_completes():
    """In this row the lid sits at the growth stop for about 7e6 yr before it hands over; the
    integrator leaves it a little above the stop, inside the slack of the lid step check."""
    r = run_row('D', 'f70_15p8_1e7_H3', True, 'lid')
    assert math.isfinite(r['result']['Tm'])
    assert r['handover_t'] is not None and r['handover_t'] > 1.0e6
