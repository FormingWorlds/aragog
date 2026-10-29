"""``tools/verification/compare_v5_runs.py``: config pins, version checks and verdicts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'compare_v5_runs',
    Path(__file__).resolve().parents[1] / 'tools' / 'verification' / 'compare_v5_runs.py',
)
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)


def _run(root, case, aragog, T_magma=1000.0, temp=(1.0, 2.0), late=None):
    """Write a PROTEUS output folder with one helpfile row per time and 2 snapshots."""
    pin = cmp.pins(cmp.CONFIGS[case])
    out = root / 'output'
    (out / 'data').mkdir(parents=True)
    rows = [
        dict.fromkeys(cmp.COLUMNS, 0.0),
        {**dict.fromkeys(cmp.COLUMNS, 1.0), 'T_magma': T_magma},
    ]
    lines = ['\t'.join(cmp.COLUMNS)] + ['\t'.join(str(r[k]) for k in cmp.COLUMNS) for r in rows]
    (out / 'runtime_helpfile.csv').write_text('\n'.join(lines) + '\n')
    (out / 'proteus_00.log').write_text(
        f'[ INFO  ] PROTEUS git hash  {pin["PROTEUS"]}\n'
        f'[ INFO  ] Interior module   aragog version {aragog}\n'
    )
    for name, T in (('99p000_int.nc', late or temp), ('1000p500_int.nc', temp)):
        with netCDF4.Dataset(out / 'data' / name, 'w') as nc:
            nc.createDimension('s', len(T))
            for k in cmp.PROFILES:
                nc.createVariable(k, 'f8', ('s',))[:] = np.asarray(T)
    return root


BRANCH, MAIN = '26.10.29.dev12+g507e408a', '26.10.22.dev8+g4ae8167d'


def test_both_configs_record_the_proteus_and_aragog_pins():
    """Each case config header names a 40-digit PROTEUS commit and the Aragog commit of MAIN."""
    for case in cmp.CONFIGS:
        pin = cmp.pins(cmp.CONFIGS[case])
        assert all(len(v) == 40 and int(v, 16) >= 0 for v in pin.values())
        assert pin['Aragog'].startswith(MAIN.rsplit('+g', 1)[1])


def test_v5_passes_only_on_bitwise_equal_end_states(tmp_path):
    """Equal end states pass; one helpfile digit or one profile entry fails the case."""
    run = _run(tmp_path / 'run', 'v5', BRANCH, late=(5.0, 5.0))
    main = _run(tmp_path / 'main', 'v5', MAIN)
    res = cmp.compare('v5', run, main)
    assert res['passed'] and res['versions']['main']['Aragog'] == MAIN
    off = _run(tmp_path / 'off', 'v5', BRANCH, T_magma=1000.0 + 1e-9)
    assert cmp.compare('v5', off, main)['columns']['T_magma'] == dict(equal=False, passed=False)
    prof = _run(tmp_path / 'prof', 'v5', BRANCH, temp=(1.0, 2.0 + 1e-12))
    res = cmp.compare('v5', prof, main)
    assert not res['passed'] and not res['columns']['temp_s']['passed']


def test_v5b_compares_the_errors_against_the_reference(tmp_path):
    """RUN passes a column when its error is not larger than the error of MAIN."""
    ref = _run(tmp_path / 'ref', 'v5b', BRANCH, T_magma=1000.0, temp=(1.0, 2.0))
    main = _run(tmp_path / 'main', 'v5b', MAIN, T_magma=1000.5, temp=(1.0, 2.5))
    run = _run(tmp_path / 'run', 'v5b', BRANCH, T_magma=999.5, temp=(1.0, 1.5))
    res = cmp.compare('v5b', run, main, ref)
    assert res['passed']
    assert res['columns']['T_magma'] == dict(error_run=0.5, error_main=0.5, passed=True)
    worse = _run(tmp_path / 'worse', 'v5b', BRANCH, T_magma=1000.0, temp=(1.0, 2.75))
    res = cmp.compare('v5b', worse, main, ref)
    assert not res['passed'] and res['columns']['temp_s']['error_run'] == 0.75


def test_runs_with_other_versions_are_refused(tmp_path):
    """A MAIN run on another Aragog, or RUN on the pinned one, stops the comparison."""
    run = _run(tmp_path / 'run', 'v5', BRANCH)
    with pytest.raises(SystemExit, match='main: versions'):
        cmp.compare('v5', run, _run(tmp_path / 'm2', 'v5', BRANCH))
    with pytest.raises(SystemExit, match='run: versions'):
        cmp.compare('v5', _run(tmp_path / 'r2', 'v5', MAIN), _run(tmp_path / 'm3', 'v5', MAIN))
    other = _run(tmp_path / 'p', 'v5', MAIN)
    log = other / 'output' / 'proteus_00.log'
    log.write_text(log.read_text().replace(cmp.pins(cmp.CONFIGS['v5'])['PROTEUS'], 'f' * 40))
    with pytest.raises(SystemExit, match='main: versions'):
        cmp.compare('v5', run, other)


def test_main_writes_the_result_and_exits_1_on_a_failed_column(tmp_path, capsys):
    """main() prints the verdict, writes the JSON, and needs --reference exactly for v5b."""
    run = _run(tmp_path / 'run', 'v5', BRANCH, T_magma=2.0)
    main = _run(tmp_path / 'main', 'v5', MAIN)
    out = tmp_path / 'r.json'
    assert cmp.main(['--case', 'v5', str(run), str(main), '--output', str(out)]) == 1
    assert 'v5: FAIL' in capsys.readouterr().out
    assert json.loads(out.read_text())['columns']['T_magma']['passed'] is False
    with pytest.raises(SystemExit) as exc:
        cmp.main(['--case', 'v5b', str(run), str(main)])
    assert exc.value.code == 2
