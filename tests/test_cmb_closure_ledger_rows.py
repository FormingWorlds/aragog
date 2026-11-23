"""Schema of the stored rows of the melt-band ledger case in ``tests/test_cmb_closure_validation.py``.

The case draws its plot from these rows; this file reads only the JSON, so it needs no table.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from tests.test_cmb_closure_validation import (
    G0,
    MUSH_TOL,
    PREDICTION_TOL,
    REQUIRED_TABLES,
    ROW_KEYS,
    ROWS_FILE,
    ROWS_TOL,
    TABLES,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_the_stored_rows_match_the_case():
    """One row per start of ``G0`` with float values, a melt coordinate at its start, the solver
    tolerances of the case, a tolerance for each start outside the band, the pass criteria of the
    case, and the provenance of the writer with one sha256 for each table file the case reads."""
    data = json.loads(ROWS_FILE.read_text())
    rows, provenance = data['rows'], data['provenance']
    assert set(provenance) == {'commit', 'platform', 'tables', 'sha256', 'builder'}
    assert re.fullmatch('[0-9a-f]{40}', provenance['commit'])
    sha = provenance['sha256']
    assert set(REQUIRED_TABLES) <= set(sha) <= set(TABLES) and all(
        re.fullmatch('[0-9a-f]{64}', h) for h in sha.values()
    ), sha
    assert [row['g0'] for row in rows] == G0
    assert {g0 for g0 in G0 if not 0.0 < g0 < 1.0} == set(ROWS_TOL) - {'band'}
    for row in rows:
        assert set(row) == {*ROW_KEYS, 'atol', 'rtol'}, row
        assert all(type(v) is float and np.isfinite(v) for v in row.values()), row
        assert abs(row['g'] - row['g0']) < 1.0e-6, row
        assert row['rtol'] == 1.0e-10 and row['atol'] == (
            1.0e-7 if row['g0'] < 0 else 1.0e-8
        ), row
        assert abs(row['miss'] - row['pred']) < PREDICTION_TOL, row
    band = [row for row in rows if 0.0 < row['g0'] < 1.0]
    assert all(abs(row['miss']) < MUSH_TOL for row in band)
    assert max(abs(row['pred_cp']) for row in band) > 0.05


def test_a_changed_start_list_is_named_and_not_raised():
    """Rows are matched by start: a start without a stored row and a stored row without a start
    are both named, so a write can add or drop a start."""
    from tests.test_cmb_closure_validation import _row_faults

    row = {'g': 0.1, 'miss': 1e-4, 'pred': 0.0, 'pred_cp': 0.1, 'atol': 1e-8, 'rtol': 1e-10}
    faults = _row_faults(
        [dict(row, g0=0.1), dict(row, g0=0.5)], [dict(row, g0=0.1), dict(row, g0=0.7)]
    )
    assert faults == ['g0 0.5: no stored row', 'stored g0 0.7: no start']


def test_write_mode_takes_a_missing_rows_file_as_no_stored_rows(tmp_path, monkeypatch):
    """Without the rows file the write starts from no stored rows; the compare refuses it."""
    import tests.test_cmb_closure_validation as V

    monkeypatch.setattr(V, 'ROWS_FILE', tmp_path / 'rows.json')
    with pytest.warns(UserWarning, match='rows.json'):
        assert V._stored_rows(write=True) == {'rows': [], 'provenance': {'sha256': {}}}
    with pytest.raises(pytest.fail.Exception, match='rows.json'):
        V._stored_rows(write=False)


def test_the_write_guard_covers_tools_and_the_imported_aragog(monkeypatch):
    """The write refuses local changes under tools/ too, and an aragog imported from outside the
    repository's src."""
    import aragog
    import tests.test_cmb_closure_validation as V

    calls = []
    monkeypatch.setattr(V, '_git', lambda *a: calls.append(a) or '')
    monkeypatch.setattr(
        aragog, '__file__', str(V.ROWS_FILE.parents[3] / 'src/aragog/__init__.py')
    )
    V._check_write_tree()
    assert 'tools' in calls[0]
    monkeypatch.setattr(aragog, '__file__', '/elsewhere/aragog/__init__.py')
    with pytest.raises(pytest.fail.Exception, match='/elsewhere/aragog'):
        V._check_write_tree()


def test_the_tables_are_hashed_by_the_files_that_exist(tmp_path):
    """The optional thermal_exp tables may be absent; a missing required table stops the case
    before its solves, and a table missing against the stored hashes is named."""
    from tests.test_cmb_closure_validation import REQUIRED_TABLES, _table_faults, _table_hashes

    for f in REQUIRED_TABLES:
        (tmp_path / f).write_text(f)
    sha = _table_hashes(tmp_path)
    assert set(sha) == set(REQUIRED_TABLES) < set(TABLES)
    stored = dict(sha, **{'thermal_exp_melt.dat': '0' * 64})
    assert _table_faults(sha, stored) == ['table thermal_exp_melt.dat missing']
    (tmp_path / 'density_melt.dat').unlink()
    with pytest.raises(pytest.fail.Exception, match='density_melt.dat'):
        _table_hashes(tmp_path)


def test_a_start_without_a_tolerance_is_named():
    """A stored start outside the band with no tolerance of its own is named, not a KeyError."""
    from tests.test_cmb_closure_validation import _row_faults

    row = {
        'g0': -0.03,
        'g': -0.03,
        'miss': 1e-3,
        'pred': 1e-3,
        'pred_cp': 0.1,
        'atol': 1e-7,
        'rtol': 1e-10,
    }
    assert _row_faults([row], [row]) == ['g0 -0.03: no tolerance']


@pytest.mark.parametrize('text', ['', '{bad', '{}', '{"rows": []}'])
def test_write_mode_takes_a_bad_rows_file_as_no_stored_rows(tmp_path, monkeypatch, text):
    """A rows file that does not parse or lacks its blocks counts as no stored rows in write mode,
    with a warning, and is refused without it."""
    import tests.test_cmb_closure_validation as V

    path = tmp_path / 'rows.json'
    path.write_text(text)
    monkeypatch.setattr(V, 'ROWS_FILE', path)
    with pytest.warns(UserWarning, match='rows.json'):
        assert V._stored_rows(write=True) == {'rows': [], 'provenance': {'sha256': {}}}
    with pytest.raises(pytest.fail.Exception, match='rows.json'):
        V._stored_rows(write=False)


@pytest.mark.parametrize('rows', [[{'g': 0.1}], [{}]])
def test_write_mode_takes_rows_that_do_not_compare_as_no_stored_rows(
    tmp_path, monkeypatch, rows
):
    """A stored row without g0, or an empty row, makes the compare fail: in write mode the file
    counts as no stored rows, with a warning; otherwise it is refused."""
    import json

    import tests.test_cmb_closure_validation as V

    path = tmp_path / 'rows.json'
    path.write_text(json.dumps({'rows': rows, 'provenance': {'sha256': {}}}))
    monkeypatch.setattr(V, 'ROWS_FILE', path)
    fresh = [
        {
            'g0': 0.1,
            'g': 0.1,
            'miss': 0.0,
            'pred': 0.0,
            'pred_cp': 0.1,
            'atol': 1e-8,
            'rtol': 1e-10,
        }
    ]
    with pytest.warns(UserWarning, match='rows.json'):
        data, faults = V._stored_faults({}, fresh, write=True)
    assert data['rows'] == [] and faults == ['g0 0.1: no stored row']
    with pytest.raises(pytest.fail.Exception, match='rows.json'):
        V._stored_faults({}, fresh, write=False)
