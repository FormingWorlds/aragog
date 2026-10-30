"""The study tools ``run_mesh_refinement_study.py`` and ``run_mesh_refinement_case.py`` of
``tools/verification``: the first frozen interval, the time-mean windows, the argument and table
guards, the resume guard and the config of every stored run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

_PATH = Path(__file__).parents[1] / 'tools' / 'verification' / 'run_mesh_refinement_study.py'
_SPEC = importlib.util.spec_from_file_location('run_mesh_refinement_study', _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)
_CASE = importlib.util.spec_from_file_location(
    'run_mesh_refinement_case', _PATH.with_name('run_mesh_refinement_case.py')
)
case = importlib.util.module_from_spec(_CASE)
_CASE.loader.exec_module(case)
DATA = Path(__file__).parent / 'validation' / 'data' / 'mesh_refinement_study.json'


def test_the_first_frozen_interval_is_interpolated_between_rows():
    """The top cell crosses the solidus downward between rows 1 and 2 and upward between rows 3
    and 4; a second freeze later does not count."""
    t = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    d = np.array([2.0, 1.0, -1.0, -3.0, 1.0, -1.0, 2.0])
    f = tool._first_frozen(t, d, 0.0)
    assert f == {'freeze_yr': 15.0, 'thaw_yr': 37.5}
    assert tool._first_frozen(t, np.abs(d) + 1, 0.0) is None
    assert tool._first_frozen(t[:3], d[:3], 0.0) == {'freeze_yr': 15.0, 'thaw_yr': None}


def _run_folder(tmp_path, t_rows):
    """A run folder with 1 part to 1e3 yr and stored melt-fraction rows at ``t_rows``."""
    run = tmp_path / 'R'
    run.mkdir()
    rec = dict(
        part=0,
        sha='a',
        config_hash='h',
        nodes=3,
        h_top=1.0,
        h_cmb=1.0,
        s_init=1.0,
        t_yr=1e3,
        status=0,
        Phi_global=0.5,
        T_cmb=4e3,
        F_top_mean_step_integral=1.0,
        energy_integrals={'F_int': -1.0, 'F_cmb': 1.0},
        cvode={'nst': 1},
    )
    (run / 'za.json').write_text(json.dumps([rec]))
    y = np.array([[3.0, 2.0], [3.0, 2.0], [3.0, 2.0]])
    np.savez(run / 'za_part0.npz', t=np.array([0.0, 1e3]), y=y, S_sol=np.array([1.0, 1.0, 1.0]))
    rows = tmp_path / 'rows'
    rows.mkdir()
    np.savez(rows / 'R.npz', t=np.asarray(t_rows), Phi=np.full(len(t_rows), 0.5))
    return run, rows


@pytest.mark.parametrize('start', [500.0, 524.0])
def test_a_time_mean_window_that_starts_on_time_is_kept(tmp_path, start, monkeypatch):
    """A run to 1e3 yr gets the mean of its 1e3 yr window only, over the stored rows."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    run, rows = _run_folder(tmp_path, np.linspace(start, 1e3, 8))
    v = tool.run_values(run, rows)
    assert v['Phi_time_mean'] == {'1e+03': 0.5}
    assert (v['commit'], v['config_hash'], v['nodes'], v['t_yr'], v['nst']) == (
        'a',
        'h',
        3,
        [1e3],
        [1],
    )
    assert (
        v['S_bottom_cell'] == [2.0]
        and v['top_cell_freezes'] == [0]
        and v['first_frozen'] is None
    )
    assert (
        v['S_sol'] == [1.0, 1.0, 1.0]
        and v['P_cell_Pa'] == [1.0, 2.0, 3.0]
        and 'T_core_K' not in v
    )


@pytest.mark.parametrize('start', [526.0, 600.0])
def test_a_time_mean_window_that_starts_late_is_refused(tmp_path, start, monkeypatch):
    """Rows from more than 5 % of t/2 after t/2 do not cover [t/2, t]: refused."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    run, rows = _run_folder(tmp_path, np.linspace(start, 1e3, 8))
    with pytest.raises(ValueError, match=f'window \\[500, 1000\\] yr starts at {start:g}'):
        tool.run_values(run, rows)


def test_the_tables_need_a_cmb_pressure_inside_the_table(tmp_path):
    with pytest.raises(SystemExit):
        tool.main([str(tmp_path), str(tmp_path / 'o.json'), 'R', '--tables', str(tmp_path)])
    d = tmp_path / 'tab'
    d.mkdir()
    head = '#\n#\n#\n#\n# 1.0 1.0 1.0\n'
    rows = ''.join(f'{p} {s} {p + s}\n' for s in (1.0, 2.0) for p in (10.0, 20.0, 30.0))
    (d / 'temperature_solid.dat').write_text(head + rows)
    (d / 'valid_mask_solid.dat').write_text('# mask\n1 1 1\n0 1 1\n')
    for p_cmb in (5.0, 10.0, 35.0):
        with pytest.raises(Exception) as err:
            tool.table_values(d, p_cmb)
        assert isinstance(err.value, ValueError) and 'outside the pressure nodes' in str(
            err.value
        )


def test_the_case_driver_rebuilds_the_config_of_every_stored_run(tmp_path):
    """The command line of each stored run, read from its record, gives its config hash."""
    study = json.loads(DATA.read_text())
    for name, v in study.items():
        if name == 'header':
            continue
        argv = [str(tmp_path), '--eos-dir', str(tmp_path), '--nodes', str(v['nodes'])]
        argv += [
            '--surface-cell',
            repr(v['surface_cell_m']),
            '--cmb-cell',
            repr(v['cmb_cell_m']),
        ]
        argv += ['--s-init', repr(v['s_init'])]
        if v['core_module_params']:
            f = tmp_path / f'{name}.json'
            f.write_text(json.dumps(v['core_module_params']))
            argv += ['--core-module-params', str(f)]
        assert case.case_config(case.parse(argv), _PATH.parents[2])[2] == v['config_hash'], name
    with pytest.raises(SystemExit):
        case.parse([str(tmp_path)])


def test_the_driver_refuses_to_resume_after_a_failed_part(tmp_path):
    """A last part that did not reach its time ends the resume before any table is read."""
    eos = tmp_path / 'eos'
    eos.mkdir()
    (eos / 'temperature_solid.dat').write_text('x')
    out = tmp_path / 'run'
    out.mkdir()
    a = case.parse([str(out), '--eos-dir', str(eos)])
    sha = case.subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=_PATH.parents[2]
    ).stdout.strip()
    rec = dict(
        sha=sha, config_hash=case.case_config(a, _PATH.parents[2])[2], reached=False, status=0
    )
    (out / 'za.json').write_text(json.dumps([rec]))
    with pytest.raises(BaseException) as err:
        case.main([str(out), '--eos-dir', str(eos), '--resume'])
    assert isinstance(err.value, SystemExit) and 'did not reach its time' in str(err.value)


def test_a_core_run_keeps_the_core_temperature_and_the_freezes_per_part(tmp_path, monkeypatch):
    """Two parts of a core-module run: the core temperature is the state after the staggered
    entropy and the CMB gradient, and each top-cell freeze counts in the part where it falls."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0])
    run = tmp_path / 'C'
    run.mkdir()
    base = dict(sha='a', config_hash='h', nodes=3, h_top=1.0, h_cmb=1.0, s_init=1.0, status=0)
    base |= dict(Phi_global=0.5, T_cmb=4e3, F_top_mean_step_integral=1.0, core_bc='core_module')
    base |= dict(energy_integrals={'F_int': -1.0, 'F_cmb': 1.0}, cvode={'nst': 1})
    recs, top = [], {0: [2.0, -1.0, 1.0, -1.0], 1: [-1.0, 1.0, -1.0, 2.0]}
    for k, t in ((0, [0.0, 1.0, 2.0, 3.0]), (1, [3.0, 4.0, 5.0, 6.0])):
        recs.append(base | dict(part=k, t_yr=t[-1]))
        y = np.array([[0.0] * 4, top[k]])
        np.savez(
            run / f'za_part{k}.npz',
            t=np.array(t),
            y=y,
            S_sol=np.zeros(2),
            y_end=np.array([0.0, top[k][-1], -0.5, 4000.0 + k]),
        )
    (run / 'za.json').write_text(json.dumps(recs))
    v = tool.run_values(run)
    assert v['T_core_K'] == [4000.0, 4001.0] and v['core_bc'] == 'core_module'
    assert v['top_cell_freezes'] == [2, 1]
    assert v['first_frozen'] == {'freeze_yr': 2.0 / 3.0, 'thaw_yr': 1.5}
