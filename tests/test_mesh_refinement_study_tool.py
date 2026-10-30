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


def _run_folder(tmp_path, t_rows, phi=None):
    """A run folder with 1 part to 1e3 yr and stored melt-fraction rows ``phi`` (0.5 if None)
    at ``t_rows``."""
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
    phi = np.full(len(t_rows), 0.5) if phi is None else np.asarray(phi)
    np.savez(rows / 'R.npz', t=np.asarray(t_rows), Phi=phi)
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


def test_the_time_mean_is_the_trapezoid_mean_over_the_window_rows(tmp_path, monkeypatch):
    """Phi = t / 1e3 on rows that are denser near t/2, with rows before t/2 and after t: the mean
    over [t/2, t] is 0.75; the sample mean or a window without either bound gives another value."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    t = np.r_[400.0, 450.0, 500.0 + 500.0 * np.linspace(0.0, 1.0, 11) ** 2, 1050.0, 1100.0]
    run, rows = _run_folder(tmp_path, t, t / 1e3)
    inside = t[(t >= 500.0) & (t <= 1e3)] / 1e3
    assert abs(inside.mean() - 0.75) > 1e-2
    assert tool.run_values(run, rows)['Phi_time_mean']['1e+03'] == pytest.approx(
        0.75, rel=1e-12
    )


@pytest.mark.parametrize(
    't',
    [np.linspace(500.0, 1e3, 5), np.r_[np.linspace(500.0, 990.0, 8), 1010.0]],
    ids=['5 rows', 'no row at t'],
)
def test_a_time_mean_window_without_6_rows_to_t_is_refused(tmp_path, t, monkeypatch):
    """A window of fewer than 6 rows, or one whose last row is before t, gives a ValueError."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    run, rows = _run_folder(tmp_path, t)
    with pytest.raises(ValueError, match='too few stored rows in'):
        tool.run_values(run, rows)


@pytest.mark.parametrize('start', [526.0, 600.0])
def test_a_time_mean_window_that_starts_late_is_refused(tmp_path, start, monkeypatch):
    """Rows from more than 5 % of t/2 after t/2 do not cover [t/2, t]: refused."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    run, rows = _run_folder(tmp_path, np.linspace(start, 1e3, 8))
    with pytest.raises(ValueError, match=f'window \\[500, 1000\\] yr starts at {start:g}'):
        tool.run_values(run, rows)


def test_the_tables_need_a_cmb_pressure_inside_the_table(tmp_path, capsys):
    """--tables without --cmb-pressure ends the study tool with the argument error, and a CMB
    pressure outside the pressure nodes of the table gives a ValueError."""
    with pytest.raises(SystemExit) as err:
        tool.main([str(tmp_path), str(tmp_path / 'o.json'), 'R', '--tables', str(tmp_path)])
    assert err.value.code == 2
    assert '--tables and --cmb-pressure go together' in capsys.readouterr().err
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


def _write_table(path, p, s, values):
    """An aragog P-S table: 5 header lines, the last with the unit scales, rows S-outer P-inner."""
    rows = [(pi, si, values[a, b]) for a, si in enumerate(s) for b, pi in enumerate(p)]
    head = '# a\n# b\n# c\n# d\n# 1 1 1\n'
    path.write_text(head + '\n'.join(f'{x:.17g} {y:.17g} {z:.17g}' for x, y, z in rows) + '\n')


def test_the_table_values_of_a_synthetic_table_set(tmp_path):
    """On a 3 x 3 table set the CMB entries come from the last entropy node valid at both
    pressure nodes around the CMB pressure, the last valid node per pressure node from the
    mask, and the melt values at 1 bar from the valid melt rows."""
    p, s = np.array([1e5, 1e9, 2e9]), np.array([1000.0, 2000.0, 3000.0])
    t = 1000.0 + s[:, None] / 10 + p[None, :] / 1e7
    vm = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
    for name, v in (
        ('temperature_solid.dat', t),
        ('temperature_melt.dat', t + 50.0),
        ('density_melt.dat', 3000.0 + 0 * t),
    ):
        _write_table(tmp_path / name, p, s, v)
    np.savetxt(tmp_path / 'valid_mask_solid.dat', vm)
    np.savetxt(tmp_path / 'valid_mask_melt.dat', vm)
    (tmp_path / 'solidus_P-S.dat').write_text(
        '# a\n# b\n# c\n# d\n# 1 1\n1e5 1500.0\n2e9 2500.0\n'
    )
    v = tool.table_values(tmp_path, 1.5e9)
    c = v['solid_at_cmb']
    assert c['P_nodes_Pa'] == [1e9, 2e9] and c['S_nodes'] == [2000.0, 3000.0]
    assert c['T_K'] == t[1:3, 1:3].tolist() and c['valid'] == [[1.0, 1.0], [0.0, 0.0]]
    assert v['solid_last_valid']['S'] == [3000.0, 2000.0, 2000.0]
    assert v['solidus'] == {'P_Pa': [1e5, 2e9], 'S': [1500.0, 2500.0]}
    assert v['melt_at_1bar']['S'] == s.tolist()
    assert v['melt_at_1bar']['T_K'] == (t[:, 0] + 50.0).tolist()


def test_the_cell_pressure_of_a_run_record_is_the_mesh_pressure():
    """The study tool maps the record fields nodes, surface_cell and cmb_cell onto the mesh of
    abe_solid.toml: its staggered pressure equals that of the mesh built directly."""
    import aragog
    from aragog.mesh import Mesh
    from aragog.parser import Parameters

    p = Parameters.from_file(Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml')
    p.mesh.number_of_nodes, p.mesh.surface_cell_thickness, p.mesh.cmb_cell_thickness = (
        100,
        50.0,
        9327.0,
    )
    rec = dict(nodes=100, surface_cell=50.0, cmb_cell=9327.0)
    assert tool._cell_pressure(rec) == np.ravel(Mesh(p).staggered_pressure).tolist()


def test_the_driver_refuses_to_resume_after_a_part_with_a_negative_status(tmp_path):
    """A last part that reached its time with CVODE status -1 also ends the resume."""
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
        sha=sha, config_hash=case.case_config(a, _PATH.parents[2])[2], reached=True, status=-1
    )
    (out / 'za.json').write_text(json.dumps([rec]))
    with pytest.raises(BaseException) as err:
        case.main([str(out), '--eos-dir', str(eos), '--resume'])
    assert isinstance(err.value, SystemExit) and 'did not reach its time' in str(err.value)


def test_the_cvode_counters_parse_the_solver_log_line():
    """The CVODE line built from the solver's own format string is summed by _CvodeStats."""
    import logging
    import re

    src = (_PATH.parents[2] / 'src/aragog/solver/entropy_solver.py').read_text()
    m = re.search(r"'(CVODE stats: [^']*)'\s*'([^']*)'\s*'([^']*)'", src)
    assert m, 'no CVODE stats format in the solver'
    fmt = ''.join(m.groups())
    args = (12, 34, 5, 6, 3, 2.5e-3, 1.0e-3, 40, 3.3, 0.42)
    h = case._CvodeStats()
    h.cv = dict(calls=0, nst=0, nfe=0, netf=0, last_step=None)
    for _ in range(2):
        h.emit(logging.LogRecord('x', logging.INFO, '', 0, fmt, args, None))
    assert h.cv == dict(calls=2, nst=24, nfe=68, netf=12, last_step=2.5e-3)


def test_the_rows_mode_refuses_a_run_folder_of_another_config(tmp_path):
    """--rows evaluates stored rows only with the solver of the run's own config hash."""
    eos = tmp_path / 'eos'
    eos.mkdir()
    (eos / 'temperature_solid.dat').write_text('x')
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'za.json').write_text(json.dumps([dict(sha='a', config_hash='other')]))
    with pytest.raises(BaseException) as err:
        case.main([str(out), '--eos-dir', str(eos), '--rows', str(tmp_path / 'r.npz')])
    assert isinstance(err.value, SystemExit) and 'comes from another config' in str(err.value)
    assert not (tmp_path / 'r.npz').exists()
