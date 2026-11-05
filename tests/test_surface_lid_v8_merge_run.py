"""The lid hand-over run again: the column of ``tests/test_surface_lid_v8.py`` on 100 nodes restarts
from a stored state 3e5 yr before the hand-over and passes it, about 40 s. It needs the extended
PALEOS table of ``ARAGOG_LOWT_EOS_DIR`` and skips without it; CI does not hold that table.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

from tests.reference.surface_lid_v2_column import LOWT_EOS_MD5
from tools.verification.run_surface_lid_merge_state import over_the_merge

REF = Path(__file__).resolve().parent / 'reference'
TOL = 1e-6  # relative, against the stored run of another machine: measured at most 2.6e-8


@pytest.mark.slow
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.timeout(900)
def test_v8_restart_over_the_merge_gives_the_stored_merge_state(monkeypatch):
    """The merge time, the lid thickness at it, the state on both sides of it and how long the
    flux stays negative equal the stored run within TOL; the merge keeps the energy."""
    table = os.environ.get('ARAGOG_LOWT_EOS_DIR')
    if not table:
        pytest.skip('ARAGOG_LOWT_EOS_DIR is not set')
    md5 = hashlib.md5((Path(table) / 'temperature_solid.dat').read_bytes()).hexdigest()
    assert md5 == LOWT_EOS_MD5
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    start = json.loads((REF / 'surface_lid_v8_merge_start.json').read_text())
    ref = json.loads((REF / 'surface_lid_v8_merge.json').read_text())['n100']
    a = types.SimpleNamespace(eos_dir=Path(table), nodes=100, eta=1e18, heating=1e-10)
    got, _, offset = over_the_merge(a, start['t0'], np.array(start['y0']), start['t_ho'])
    assert got['merge_residual'] < 1e-15
    for key in ('t_merge', 'delta', 'negative_yr'):
        assert got[key] == pytest.approx(ref[key], rel=TOL), key
    for side in ('before', 'after'):
        for key in ('flux', 'T_cell', 'T_node'):
            assert got[side][key] == pytest.approx(ref[side][key], rel=TOL), (side, key)
    assert offset == pytest.approx(ref['offset_predicted'], rel=TOL)
