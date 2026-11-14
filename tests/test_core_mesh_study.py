"""Validation of the default CMB cell from the stored CMB mesh study
(``tests/validation/data/core_mesh_study.json``, written by
``tools/verification/run_core_mesh_study.py`` from runs of
``tools/verification/run_core_mesh_case.py``). The runs take hours and need a PALEOS table set
that CI does not build, so the test reads the stored values and checks every number that
docs/Validation/mesh_refinement.md quotes for this case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_mesh_refinement_study import FIGSIZE, _check_figure, _check_layout, _check_size

pytestmark = [pytest.mark.unit, pytest.mark.reference_pinned, pytest.mark.timeout(60)]

DATA = Path(__file__).parent / 'validation' / 'data' / 'core_mesh_study.json'
RUNS_COMMIT = 'f9b87e9a70ccc4d45aad4eaa73418ce210f7dda5'
CHECK_COMMIT = RUNS_COMMIT
COMPARED = ('U160', 'U320', 'D80', 'D160')
TIMES = (3.0, 50.0, 100.0)


def _study():
    return json.loads(DATA.read_text())


def _at(run):
    return {x['t_myr']: x for x in run['at']}


def _relative(study):
    """q_cmb of each compared run over that of U640 minus 1 [%], at TIMES and on the grid."""
    ref = study['runs']['U640']
    at_ref, grid_ref = _at(ref), np.array(ref['q_grid'])
    rel, grid = {}, {}
    for name in COMPARED:
        run = study['runs'][name]
        rel[name] = [100 * (_at(run)[t]['q_cmb'] / at_ref[t]['q_cmb'] - 1) for t in TIMES]
        grid[name] = 100 * (np.array(run['q_grid']) / grid_ref - 1)
    return rel, grid


def test_the_stored_runs_and_their_mesh():
    """The 5 runs, the left-out run, the commits, the start and the cells quoted on the page."""
    study = _study()
    head, runs = study['header'], study['runs']
    assert set(runs) == {'U160', 'U320', 'U640', 'D80', 'D160'}
    assert list(head['left_out']) == ['U80']
    # In every call of Aragog's CMB law the conductive flux is the larger one.
    assert all(v['max_q_conv_minus_q_cond'] < 0.0 for v in runs.values())
    assert f'{head["left_out"]["U80"]["share_conv_above_cond"]:.0%}' == '59%'
    assert head['runs_commit'] == RUNS_COMMIT
    check = head['check']
    assert (check['run'], check['commit']) == ('D80', CHECK_COMMIT)
    # The rerun of D80 reproduces the stored outputs bit for bit.
    md5 = runs['D80']['md5']
    assert check['md5'] == {k: md5[k] for k in ('segments', '.cmb.csv', '.nodes.bin', '.S.npy')}
    assert head['times_yr'] == [3e6, 5e7, 1e8]
    assert all(v['end_myr'] > 100.0 for v in runs.values())
    for name, v in runs.items():
        for t, x in zip(TIMES, v['at']):
            assert x['t_myr'] == t
            # States are stored every 0.05 Myr.
            assert abs(x['t_q_myr'] - t) < 0.01 and abs(x['t_state_myr'] - t) <= 0.05, name
    build = head['eos_build']
    assert (build['P_max_Pa'], build['nP'], build['nS']) == (350e9, 1350, 280)
    assert build['mushy_zone_factor'] == 0.8
    assert [runs[k]['n'] for k in ('U160', 'U320', 'U640', 'D80', 'D160')] == [
        160, 320, 640, 80, 160
    ]  # fmt: skip
    cmb = {k: f'{v["cmb_cell_m"] / 1e3:.2f}' for k, v in runs.items()}
    assert cmb == {
        'U160': '18.18',
        'U320': '9.06',
        'U640': '4.52',
        'D80': '3.00',
        'D160': '3.00',
    }
    assert [f'{runs[k]["max_cell_ratio"]:.3f}' for k in ('D80', 'D160')] == ['1.865', '1.621']
    for k in ('U160', 'U320', 'U640'):
        assert runs[k]['cmb_cell_m'] == pytest.approx(runs[k]['top_cell_m'], rel=1e-9)
    assert runs['D80']['top_cell_m'] == pytest.approx(36594.9, abs=0.1)


def test_the_default_cmb_cell_against_finer_uniform_meshes(validation_plot):
    """q_cmb of the default mesh at 80 and 160 nodes and of the uniform mesh at 160 and 320 nodes
    against the uniform 640-node mesh, the core and mid-mantle temperatures, and the start
    transient, as quoted on the page."""
    study = _study()
    runs = study['runs']
    rel, grid = _relative(study)
    got = {k: [f'{x:+.2f}' for x in v] for k, v in rel.items()}
    assert got == {
        'U160': ['-10.08', '-5.99', '-4.35'],
        'U320': ['+6.68', '+3.00', '+2.69'],
        'D80': ['-1.60', '-0.45', '-0.35'],
        'D160': ['-1.42', '-0.23', '-0.15'],
    }
    ref = _at(runs['U640'])
    dT = {}
    for name in ('D80', 'D160'):
        at = _at(runs[name])
        dT[name] = max(abs(at[t][k] - ref[t][k]) for t in TIMES for k in ('T_core', 'T_mid'))
    assert {k: f'{v:.2f}' for k, v in dT.items()} == {'D80': '0.10', 'D160': '0.04'}

    g = np.array(study['header']['grid_myr'])
    transient = {}
    for name in ('D80', 'D160'):
        r = grid[name]
        i = int(np.argmin(r))
        settled = int(np.where(np.abs(r) > 0.5)[0].max()) + 1
        transient[name] = (f'{r[0]:+.0f}', f'{r[i]:.1f}', f'{g[i]:.2f}', f'{g[settled]:.1f}')
    assert transient == {
        'D80': ('+44', '-7.9', '0.53', '28.2'),
        'D160': ('+42', '-7.7', '0.71', '7.5'),
    }
    assert g[0] == 0.01

    import matplotlib.pyplot as plt

    style = {'U160': ':', 'U320': '-.', 'D80': '-', 'D160': '--'}

    def over_time():
        fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
        fig.get_layout_engine().set(h_pad=0.05)
        for name in COMPARED:
            ax.plot(g[::4], grid[name][::4], style[name], lw=1.2, label=name)
        ax.axhline(0.0, color='0.5', lw=0.6)
        ax.set(xscale='log', xlabel='time [Myr]', ylabel='$q_\\mathrm{cmb}$ vs U640 [%]')
        ax.set_ylim(-80, 60)
        ax.minorticks_off()
        fig.legend(loc='outside lower center', ncols=2, handlelength=1.6)
        return fig

    def at_times():
        fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
        fig.get_layout_engine().set(h_pad=0.05)
        cells = [runs[k]['cmb_cell_m'] / 1e3 for k in ('U160', 'U320', 'D160')]
        for j, (t, mark) in enumerate(zip(TIMES, 'os^')):
            y = [rel[k][j] for k in ('U160', 'U320', 'D160')]
            ax.plot(cells, y, mark + '-', lw=1.0, ms=5, label=f'{t:.0f} Myr')
        ax.axhline(0.0, color='0.5', lw=0.6)
        ax.set(xscale='log', xlabel='CMB cell [km]', ylabel='$q_\\mathrm{cmb}$ vs U640 [%]')
        ax.set_xticks([3, 10, 20], ['3', '10', '20'])
        ax.set_yticks([-10, -5, 0, 5])
        ax.set_ylim(-12, 9)
        ax.minorticks_off()
        fig.legend(loc='outside lower center', ncols=3, handlelength=1.6)
        return fig

    fig = over_time()
    _check_figure(
        fig, [''], [[*(grid[k][::4] for k in COMPARED), [0.0, 0.0]]],
        list(COMPARED),
    )  # fmt: skip
    plt.close(fig)
    for figure, name in ((over_time, 'core_mesh_flux_over_time'), (at_times, 'core_mesh_flux_at_times')):  # fmt: skip
        _check_layout(figure)
        _check_size(figure, f'mesh_refinement_{name}')
        validation_plot(figure, f'mesh_refinement_{name}')
