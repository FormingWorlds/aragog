"""``tools/compare_snapshots.py``: reported differences and exit codes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import netCDF4
import numpy as np
import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'compare_snapshots', Path(__file__).resolve().parents[1] / 'tools' / 'compare_snapshots.py'
)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _write(path, floats, labels, extra=(), scalar=0.0):
    with netCDF4.Dataset(path, 'w') as f:
        f.createDimension('n', 3)
        for name, values in floats.items():
            f.createVariable(name, 'f8', ('n',))[:] = values
        f.createVariable('scalar', 'f8', (), fill_value=np.nan).assignValue(scalar)
        f.createVariable('label', str, ('n',))[:] = np.array(labels, dtype=object)
        for name in extra:
            f.createVariable(name, 'f8', ('n',))[:] = 0.0


@pytest.fixture
def pair(tmp_path):
    a, b = tmp_path / 'a.nc', tmp_path / 'b.nc'
    _write(
        a,
        {'same': [1, 2, 3], 'E': [1, 2, 4], 'nan': [1, np.nan, 3], 'inf': [np.inf, 1, 2]},
        ['x', 'y', 'z'],
        ['only_a'],
        scalar=np.nan,
    )
    _write(
        b,
        {'same': [1, 2, 3], 'E': [1, 2, 5], 'nan': [1, 2, 3], 'inf': [-np.inf, 1, 3]},
        ['x', 'y', 'w'],
    )
    return str(a), str(b)


def test_differences_are_described(pair):
    diff, only = cs.compare(*pair)
    assert diff == {
        'E': 'max relative difference 0.25',
        'nan': 'non-finite mismatch at 1 points',
        'inf': 'max relative difference 0.5, non-finite mismatch at 1 points',
        'scalar': 'non-finite mismatch at 1 points',
        'label': 'values differ',
    }
    assert only == {'only_a'}


def test_exit_code_follows_the_ignore_list(pair, capsys):
    assert cs.main([*pair]) == 1
    assert cs.main([*pair, '--ignore', 'E,nan,inf,scalar,label']) == 1
    assert cs.main(['--ignore', 'E,nan,inf', '--ignore', 'scalar,label,only_a', *pair]) == 0
    out = capsys.readouterr().out
    assert (
        'only_a: in one file only (ignored)' in out and 'label: values differ (ignored)' in out
    )


def test_unreadable_file_exits_2(pair, tmp_path, capsys):
    assert cs.main([pair[0], str(tmp_path / 'missing.nc')]) == 2
    assert capsys.readouterr().err.startswith('error:')
