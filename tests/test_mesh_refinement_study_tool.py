"""The study tool ``tools/verification/run_mesh_refinement_study.py`` on synthetic input: the first
frozen interval of the top cell, the time-mean window check and the argument and table guards."""

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


@pytest.mark.parametrize('start', [500.0, 512.0])
def test_a_time_mean_window_that_starts_on_time_is_kept(tmp_path, start, monkeypatch):
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    run, rows = _run_folder(tmp_path, np.linspace(start, 1e3, 8))
    with pytest.raises(ValueError, match='too few stored rows in \\[5000, 10000\\]'):
        tool.run_values(run, rows)


def test_a_time_mean_window_that_starts_late_is_refused(tmp_path, monkeypatch):
    """Rows from 0.6 t to t cover only 80 % of [t/2, t]: refused, not averaged over the rest."""
    monkeypatch.setattr(tool, '_cell_pressure', lambda r0: [1.0, 2.0, 3.0])
    run, rows = _run_folder(tmp_path, np.linspace(600.0, 1e3, 8))
    with pytest.raises(ValueError, match='window \\[500, 1000\\] yr starts at 600'):
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
        with pytest.raises(ValueError, match='outside the pressure nodes'):
            tool.table_values(d, p_cmb)
