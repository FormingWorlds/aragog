"""The repeat of a refused CMB ledger call in stored coupled PROTEUS runs.

A check of stored runs; it runs no aragog code (the repeat in the solver is tested in
``tests/test_cmb_ledger_repeat.py``). The 4 runs are the PROTEUS layer-trapping tutorial
(``trap_mode = "none"``, 1 Earth mass, energy_balance core) at rtol 1e-8, each from a start
changed at round-off level, to the tutorial end. The base run is the first start on the code
without the repeat. The stored file (``tests/validation/data/cmb_ledger_repeat_runs.json``, built
by ``tests/reference/build_cmb_ledger_repeat_runs.py``) holds each refused attempt with its
repeat, a summary of each run and the commits and start of each run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_figure_legend_below

pytestmark = [pytest.mark.unit]

DATA = Path(__file__).parent / 'validation' / 'data' / 'cmb_ledger_repeat_runs.json'
N_RUNS = 4
T_END = 3.32e6  # [yr] the runs end when the mantle solidifies, near 3.33 Myr
# The numbers that docs/Validation/core.md (section 19) quotes, with the rounding it uses.
PAGE = {
    'refused calls': '13',
    'refused calls per run': '2 to 4',
    'refused miss over B': '10.5 to 15.5',
    'repeat miss over B': '0.0087 to 0.56',
    'calls per run': '847 or 848',
    'kept calls between B and 10 B': '69 to 101',
    'kept share [%]': '8.1 to 11.9',
    'repeat share of the steps [%]': '0.15 to 0.45',
    'refused share of the steps [%]': '0.13 to 0.18',
    'base run stop [Myr]': '0.51',
    'base run refused calls': '3',
}


@pytest.fixture(scope='module')
def stored():
    """The stored runs and their refused calls."""
    return json.loads(DATA.read_text())


def test_the_stored_runs_name_their_commits_and_starts(stored):
    """Each run names its aragog, PROTEUS and Zalmoxis commits, its start and its log hash; the
    runs share one aragog commit and differ in their start; the base run has the first start on
    another commit."""
    prov = stored['provenance']
    assert set(prov) == {'config', 'runs', 'builder', 'note'}
    runs, base = stored['runs'], stored['base_runs']
    assert set(prov['runs']) == set(runs) | set(base) and len(runs) == N_RUNS and len(base) == 1
    for p in prov['runs'].values():
        assert all(
            re.fullmatch('[0-9a-f]{40}', p[k]) for k in ('aragog', 'proteus', 'zalmoxis')
        )
        assert re.fullmatch('[0-9a-f]{64}', p['log_sha256'])
    tip = {prov['runs'][k]['aragog'] for k in runs}
    starts = sorted(prov['runs'][k]['delta_T_super'] for k in runs)
    (b,) = base
    assert len(tip) == 1 and len(set(starts)) == N_RUNS
    assert (
        prov['runs'][b]['delta_T_super'] == starts[0] and prov['runs'][b]['aragog'] not in tip
    )


@pytest.mark.physics_invariant
def test_every_refused_call_passes_on_its_repeat_and_no_run_stops(stored, validation_plot):
    """Every repeat at rtol 1e-10 misses by at most the bound B; every run reaches its end with no
    failed repeat and no kept call above 10 B, while the base run stops on refused calls."""
    rows, runs = stored['rows'], stored['runs']
    repeat = np.array([abs(r['r_repeat']) / r['bound_repeat'] for r in rows])
    assert rows and np.all(repeat <= 1.0)
    for name, run in runs.items():
        assert run['t_end'] > T_END and run['status'] == 'Completed (solidified)', (name, run)
        assert run['repeats_failed'] == run['refusals_without_repeat'] == 0, (name, run)
        assert run['kept_above_10_B'] == 0, (name, run)
        assert run['repeats_passed'] == sum(r['run'] == name for r in rows) > 0, (name, run)
    (base,) = stored['base_runs'].values()
    assert base['t_end'] < T_END and base['refusals_without_repeat'] >= 3
    assert not base['status'].startswith('Completed')
    assert_figure_legend_below(_figure(rows))
    validation_plot(lambda: _figure(rows), 'core_cmb_ledger_repeat')


def _span(values, fmt):
    """``'<min> to <max>'`` of ``values`` in the format of the page."""
    return f'{fmt(min(values))} to {fmt(max(values))}'


def test_the_numbers_on_the_page_match_the_stored_runs(stored):
    """Every number of section 19 of the core validation page, from the stored file."""
    rows, runs = stored['rows'], stored['runs']
    (base,) = stored['base_runs'].values()
    v = list(runs.values())
    per_run = [sum(r['run'] == k for r in rows) for k in runs]
    calls = sorted({x['calls'] for x in v})
    kept = [x['kept_1_to_10_B'] for x in v]
    found = {
        'refused calls': str(len(rows)),
        'refused calls per run': _span(per_run, str),
        'refused miss over B': _span([abs(r['r']) / r['bound'] for r in rows], '{:.1f}'.format),
        'repeat miss over B': _span(
            [abs(r['r_repeat']) / r['bound_repeat'] for r in rows], '{:.2g}'.format
        ),
        'calls per run': ' or '.join(map(str, calls)),
        'kept calls between B and 10 B': _span(kept, str),
        'kept share [%]': _span(
            [100 * x['kept_1_to_10_B'] / x['calls'] for x in v], '{:.1f}'.format
        ),
        'repeat share of the steps [%]': _span(
            [100 * x['steps_repeat'] / x['steps'] for x in v], '{:.2f}'.format
        ),
        'refused share of the steps [%]': _span(
            [100 * x['steps_refused'] / x['steps'] for x in v], '{:.2f}'.format
        ),
        'base run stop [Myr]': f'{base["t_end"] / 1e6:.2f}',
        'base run refused calls': str(base['refusals_without_repeat']),
    }
    assert found == PAGE


def _figure(rows):
    """Miss over bound of each refused call at rtol 1e-8 and of its repeat at 1e-10, against the
    time of the call."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = np.array([r['t'] for r in rows]) / 1.0e6
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    text = plt.rcParams['axes.labelcolor']
    for level, style in ((1.0, ':'), (10.0, '--')):
        ax.axhline(level, color=text, lw=1, ls=style, alpha=0.6)
    ax.plot(t, [abs(r['r']) / r['bound'] for r in rows], 'o', ms=4, color='C1')
    ax.plot(t, [abs(r['r_repeat']) / r['bound_repeat'] for r in rows], 's', ms=4, color='C0')
    ax.set_yscale('log')
    ax.margins(y=0.1)
    ax.set(xlabel='time of the call [Myr]', ylabel='|r| / B')
    handles = [
        Line2D([], [], color='C1', marker='o', ls='none', label='refused, rtol 1e-8'),
        Line2D([], [], color='C0', marker='s', ls='none', label='repeat, rtol 1e-10'),
        Line2D([], [], color=text, ls='--', alpha=0.6, label='10 B: refusal'),
        Line2D([], [], color=text, ls=':', alpha=0.6, label='B: warning'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.2, 1.0, 1.0))
    return fig
