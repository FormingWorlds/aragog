"""The repeat of a refused CMB ledger call in stored coupled PROTEUS runs.

A check of stored runs: 8 runs of the PROTEUS layer-trapping tutorial (``trap_mode = "none"``,
1 Earth mass, energy_balance core) at rtol 1e-8, each from a start changed at round-off level, to
the tutorial end. At 1e-8 these starts stop the run on refused calls. With the repeat, each refused
call is solved again at rtol 1e-10. The stored file
(``tests/validation/data/cmb_ledger_repeat_runs.json``, built by
``tests/reference/build_cmb_ledger_repeat_runs.py``) holds each refused attempt with its repeat
and a summary of each run; its provenance names the commits of each run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_figure_legend_below

pytestmark = [pytest.mark.reference_pinned, pytest.mark.unit]

DATA = Path(__file__).parent / 'validation' / 'data' / 'cmb_ledger_repeat_runs.json'
N_RUNS = 8
T_END = 3.3e6  # [yr] the tutorial ends at about 3.33 Myr


@pytest.fixture(scope='module')
def stored():
    """The stored runs and their refused calls."""
    return json.loads(DATA.read_text())


def test_the_stored_runs_name_their_commits_and_starts(stored):
    """Each run names its aragog, PROTEUS and Zalmoxis commits, its start and its log hash, and
    no 2 runs share both the aragog commit and the start."""
    prov = stored['provenance']
    assert set(prov) == {'config', 'runs', 'builder', 'note'}
    assert set(prov['runs']) == set(stored['runs']) and len(stored['runs']) == N_RUNS
    for p in prov['runs'].values():
        assert all(
            re.fullmatch('[0-9a-f]{40}', p[k]) for k in ('aragog', 'proteus', 'zalmoxis')
        )
        assert re.fullmatch('[0-9a-f]{64}', p['log_sha256'])
    starts = {(p['aragog'], p['delta_T_super']) for p in prov['runs'].values()}
    assert len(starts) == N_RUNS


@pytest.mark.physics_invariant
def test_every_refused_call_passes_on_its_repeat_and_no_run_stops(stored, validation_plot):
    """Every refused attempt misses by more than 10 B at rtol 1e-8, and its repeat at 1e-10 stays
    within the bound B; every run reaches the tutorial end, with no failed repeat and no kept call
    above 10 B."""
    rows, runs = stored['rows'], stored['runs']
    refused = np.array([abs(r['r']) / r['bound'] for r in rows])
    repeat = np.array([abs(r['r_repeat']) / r['bound_repeat'] for r in rows])
    assert rows and np.all(refused > 10.0) and np.all(repeat <= 1.0)
    for name, run in runs.items():
        assert run['t_end'] > T_END and run['status'].startswith('Completed'), (name, run)
        assert run['repeats_failed'] == run['refusals_without_repeat'] == 0, (name, run)
        assert run['kept_above_10_B'] == 0, (name, run)
        assert run['repeats_passed'] == sum(r['run'] == name for r in rows) > 0, (name, run)
    assert_figure_legend_below(_figure(rows))
    validation_plot(lambda: _figure(rows), 'core_cmb_ledger_repeat')


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
