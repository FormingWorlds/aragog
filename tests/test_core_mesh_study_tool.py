"""The CMB mesh study tools ``run_core_mesh_case.py`` and ``run_core_mesh_study.py`` of
``tools/verification``: the config of a run, the record of the CMB flux calls and the stored
states, the solid start, and the values and checksums the study file holds."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

_DIR = Path(__file__).parents[1] / 'tools' / 'verification'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


case = _load('run_core_mesh_case')
study = _load('run_core_mesh_study')
BASE = (_DIR / 'data' / 'core_mesh_study.toml').read_text()


def _args(*extra):
    return case.parse(
        ['c.toml', 'o', '--eos-dir', 'e', '--solid-s0', '1', '--t-core', '1', *extra]
    )


def test_the_run_config_sets_only_the_given_mesh_values():
    """Each option replaces its one line; an unset option keeps the base value."""
    text = case.case_config(
        BASE, _args('--nodes', '160', '--cmb-cell', '0', '--surface-cell', '0')
    )
    changed = [(a, b) for a, b in zip(BASE.splitlines(), text.splitlines()) if a != b]
    assert changed == [
        ('number_of_nodes = 80', 'number_of_nodes = 160'),
        ('cmb_cell_thickness = 3000.0', 'cmb_cell_thickness = 0.0'),
        ('surface_cell_thickness = "uniform"', 'surface_cell_thickness = 0.0'),
    ]
    assert case.case_config(BASE, _args()) == BASE
    assert 'surface_cell_thickness = "uniform"' in case.case_config(
        BASE.replace('"uniform"', '0.0'), _args('--surface-cell', 'uniform')
    )


@pytest.mark.parametrize('count', [0, 2])
def test_the_run_config_refuses_a_key_that_is_not_there_once(count):
    text = BASE.replace('number_of_nodes = 80\n', 'number_of_nodes = 80\n' * count)
    with pytest.raises(AssertionError, match='number_of_nodes: '):
        case.case_config(text, _args('--nodes', '160'))


def _law(t_hot, t_m, **kw):
    return kw['conductivity'] * (t_hot - t_m) / kw['dr_half'] + 1.0


KW = dict(conductivity=4.0, density=4000.0, heat_capacity=1000.0, expansivity=3e-5,
          viscosity=1e20, gravity=10.0, dr_half=1500.0, ra_crit=660.0)  # fmt: skip


def _recorder(tmp_path, node_dt=10.0):
    rec = case._Recorder(str(tmp_path / 'r'), node_dt)
    rec.law = _law
    rec.scales = SimpleNamespace(t_ref=2.0, state_scale=np.array([10.0, 1.0]), state_offset=1.0)
    return rec


def test_a_flux_call_records_its_caller_and_the_two_fluxes(tmp_path):
    """A call from the CVODE right-hand side records its time in years and returns the law's
    flux; a call from elsewhere records no time; a cold core has no convective flux."""
    rec = _recorder(tmp_path)

    def rhs_fn(t):
        return rec.flux(2500.0, 2000.0, **KW)

    assert rhs_fn(3.0) == _law(2500.0, 2000.0, **KW)
    assert rec.flux(2000.0, 2500.0, **KW) == _law(2000.0, 2500.0, **KW)
    rec.close()
    rows = list(csv.DictReader(open(tmp_path / 'r.cmb.csv')))
    assert [(r['src'], r['t_yr'], r['n_nodes']) for r in rows] == [
        ('cvode', '6.0', '-1'),
        ('other', 'nan', '-1'),
    ]
    q_cond = 4.0 * 500.0 / 1500.0
    kappa = 4.0 / (4000.0 * 1000.0)
    q_conv = 4.0 * 500.0 ** (4 / 3) * (3e-5 * 4000.0 * 10.0 / (660.0 * kappa * 1e20)) ** (1 / 3)
    assert float(rows[0]['q_cond']) == q_cond and float(rows[0]['q']) == q_cond + 1.0
    assert float(rows[0]['q_conv']) == pytest.approx(q_conv, rel=1e-14)
    assert rows[1]['q_conv'] == 'nan'
    meta = json.loads((tmp_path / 'r.nodes.json').read_text())
    assert meta == {'n_y': None, 'n_written': 0, 'node_dt_yr': 10.0, 'calls': 2}


def test_the_accepted_states_are_stored_once_per_interval(tmp_path):
    """States of the solve are written in physical units when the time has advanced by the
    interval since the last written state; a call from the energy pass is a trace call."""
    rec = _recorder(tmp_path, node_dt=10.0)

    def _solve_cvode(nodes):
        def rhs_fn(t):
            return rec.flux(2500.0, 2000.0, **KW)

        return rhs_fn(nodes[-1][0])

    def _compute_step_energy_integrals():
        return rec.flux(2500.0, 2000.0, **KW)

    for t in (0.0, 2.0, 6.0, 7.0):  # 0, 4, 12 and 14 yr
        _solve_cvode([(t, [0.5, 2.0])])
    _compute_step_energy_integrals()
    rec.close()
    rows = np.fromfile(tmp_path / 'r.nodes.bin', '<f8').reshape(-1, 3)
    np.testing.assert_array_equal(rows, [[0.0, 6.0, 3.0], [12.0, 6.0, 3.0]])
    calls = list(csv.DictReader(open(tmp_path / 'r.cmb.csv')))
    assert [r['src'] for r in calls] == ['cvode'] * 4 + ['trace']
    assert json.loads((tmp_path / 'r.nodes.json').read_text())['n_written'] == 2


def test_the_solid_start_refuses_a_cell_with_melt():
    solver = SimpleNamespace(_P_stag_flat=np.array([1e9, 2e9]))
    solid = SimpleNamespace(melt_fraction=lambda P, s: np.zeros_like(P))
    np.testing.assert_array_equal(case.solid_start(solver, solid, 2045.0), [2045.0, 2045.0])
    melt = SimpleNamespace(melt_fraction=lambda P, s: np.array([0.0, 1e-9]))
    with pytest.raises(AssertionError, match='solid start has melt'):
        case.solid_start(solver, melt, 2045.0)


def _write_run(tmp_path, wall='1.5', q_conv=('0.1', 'nan')):
    out = str(tmp_path / 'run.csv')
    Path(out).write_text(f'seg,t_end_yr,wall_s,nst\n1,2000.0,{wall},7\n')
    with open(out + '.cmb.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(case.CMB_COLS)
        w.writerow(['cvode', '3e6', '1', '2500', '2000', '1e20', '0.2', q_conv[0], '0.2'])
        w.writerow(['trace', 'nan', '1', '2500', '2000', '1e20', '0.2', '0.3', '0.2'])
        w.writerow(['cvode', '1e6', '1', '2500', '2000', '1e20', '0.4', q_conv[1], '0.4'])
    np.save(out + '.S.npy', np.ones(3))
    np.array([[0.0, 1.0, 2.0], [5.0, 3.0, 4.0]], '<f8').tofile(out + '.nodes.bin')
    Path(out + '.nodes.json').write_text(json.dumps({'n_y': 2}))
    return out


def test_the_study_reads_the_calls_and_the_stored_states(tmp_path):
    """CVODE calls in file order; the flux gap covers every call, also the trace call; the
    share of calls with a convective flux above the conductive one counts every call."""
    out = _write_run(tmp_path)
    t, q, gap = study.cvode_records(Path(out + '.cmb.csv'))
    np.testing.assert_array_equal(t, [3e6, 1e6])
    np.testing.assert_array_equal(q, [0.2, 0.4])
    assert gap == pytest.approx(0.1)
    assert study.law_share(Path(out + '.cmb.csv')) == {
        'calls': 3,
        'share_conv_above_cond': 1 / 3,
    }
    np.testing.assert_array_equal(study.stored_states(out), [[0.0, 1.0, 2.0], [5.0, 3.0, 4.0]])
    Path(out + '.nodes.json').write_text(json.dumps({'n_y': 4}))
    with pytest.raises(AssertionError):
        study.stored_states(out)


def test_the_rerun_checksums_ignore_only_the_wall_clock(tmp_path):
    for name in 'abc':
        (tmp_path / name).mkdir()
    a = study.outputs_md5(_write_run(tmp_path / 'a'))
    b = study.outputs_md5(_write_run(tmp_path / 'b', wall='9.9'))
    assert a == b and set(a) == {'segments', '.cmb.csv', '.nodes.bin', '.S.npy'}
    out = _write_run(tmp_path / 'c')
    Path(out).write_text('seg,t_end_yr,wall_s,nst\n1,2000.0,1.5,8\n')
    assert study.outputs_md5(out)['segments'] != a['segments']
