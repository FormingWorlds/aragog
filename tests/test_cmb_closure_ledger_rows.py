"""Schema of the stored rows of the melt-band ledger case in ``tests/test_cmb_closure_validation.py``.

The case draws its plot from these rows; this file reads only the JSON, so it needs no table.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from tests.test_cmb_closure_validation import G0, ROW_KEYS, ROWS_FILE, ROWS_TOL

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]


def test_the_stored_rows_match_the_case():
    """One row per start of ``G0`` with float values, a melt coordinate at its start, the solver
    tolerances of the case, a tolerance for each start outside the band, and the provenance of
    the writer with one sha256 per table file."""
    data = json.loads(ROWS_FILE.read_text())
    rows, provenance = data['rows'], data['provenance']
    assert set(provenance) == {'commit', 'platform', 'tables', 'sha256', 'builder'}
    assert re.fullmatch('[0-9a-f]{40}', provenance['commit'])
    sha = provenance['sha256']
    assert sha and all(
        re.fullmatch(r'[\w.-]+\.dat', f) and re.fullmatch('[0-9a-f]{64}', h)
        for f, h in sha.items()
    ), sha
    assert [row['g0'] for row in rows] == G0
    assert {g0 for g0 in G0 if not 0.0 < g0 < 1.0} == set(ROWS_TOL) - {'band'}
    for row in rows:
        assert set(row) == {*ROW_KEYS, 'atol', 'rtol'}, row
        assert all(type(v) is float and np.isfinite(v) for v in row.values()), row
        assert abs(row['g'] - row['g0']) < 1.0e-6, row
        assert row['rtol'] == 1.0e-10 and row['atol'] in (1.0e-8, 1.0e-7), row
