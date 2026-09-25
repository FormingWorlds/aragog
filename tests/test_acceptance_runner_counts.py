"""CVODE counters reported by the solid-state acceptance runner.

The runner's per-interval step counts and step sizes must come from CVODE's own
counters, not from the dense output grid of ``sol.t`` (65 points per solve).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from aragog.solver.entropy_solver import EntropySolver

pytestmark = [pytest.mark.slow, pytest.mark.timeout(600)]

_REPO = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
]
EOS_DIR = next((Path(p) for p in _CANDIDATES if p and Path(p).exists()), None)


@pytest.mark.skipif(EOS_DIR is None, reason='EOS_DIR not found')
def test_runner_counts_come_from_cvode(tmp_path, monkeypatch):
    pytest.importorskip('jax')
    pytest.importorskip('scikits_odes_sundials')
    sys.path.insert(0, str(_REPO / 'tools'))
    from verification.run_ssc_acceptance import run_acceptance

    sols = []
    solve = EntropySolver.solve

    def recording_solve(self):
        solve(self)
        sols.append(self._solution)

    monkeypatch.setattr(EntropySolver, 'solve', recording_solve)
    config = _REPO / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    res = run_acceptance(config, EOS_DIR, tmp_path, checkpoints=(1.0,))

    info = [s['cvode_info'] for s in sols]
    assert res['cvode_steps'][0] == sum(i['NumSteps'] for i in info) >= 1
    assert res['cvode_err_test_fails'][0] == sum(i['NumErrTestFails'] for i in info)
    assert res['cvode_jac_setups'][0] == sum(i['NumLinSolvSetups'] for i in info) >= 1
    h_min = min(s.cvode_last_step for s in sols)
    assert res['cvode_last_step_min'][0] == res['global_last_step_min'] == h_min
    assert 0.0 < h_min <= 1.0
