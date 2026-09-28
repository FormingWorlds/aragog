"""CVODE counters reported by the solid-state acceptance runner.

The runner's per-interval step counts and step sizes must come from CVODE's own
counters, not from the dense output grid of ``sol.t``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

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
    monkeypatch.syspath_prepend(str(_REPO / 'tools'))
    from verification.run_ssc_acceptance import cvode_counts, run_acceptance

    with pytest.raises(KeyError):
        cvode_counts(OptimizeResult(t=np.zeros(2)))

    sols = []
    solve = EntropySolver.solve

    def recording_solve(self):
        solve(self)
        sols.append((self.parameters.solver.end_time, self._solution))

    monkeypatch.setattr(EntropySolver, 'solve', recording_solve)
    config = _REPO / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    res = run_acceptance(config, EOS_DIR, tmp_path, checkpoints=(1.0, 2.0))

    assert all(s.t.size == 9 for _, s in sols)
    h_mins = []
    for i, t_end in enumerate((1.0, 2.0)):
        interval = [s for t, s in sols if t_end - 1.0 < t <= t_end]
        info = [s.cvode_info for s in interval]
        assert res['cvode_steps'][i] == sum(x['NumSteps'] for x in info) >= 1
        assert res['cvode_err_test_fails'][i] == sum(x['NumErrTestFails'] for x in info)
        assert res['cvode_jac_setups'][i] == sum(x['NumLinSolvSetups'] for x in info) >= 1
        h_mins.append(min(s.cvode_last_step for s in interval))
        assert res['cvode_last_step_min'][i] == h_mins[i]
    assert res['global_last_step_min'] == min(h_mins)
    assert 0.0 < min(h_mins) <= 1.0

    ckpt = tmp_path / 'acceptance_checkpoint.npz'
    data = dict(np.load(ckpt))
    del data['cvode_jac_setups']
    np.savez(ckpt, **data)
    with pytest.raises(ValueError, match='without resume'):
        run_acceptance(config, EOS_DIR, tmp_path, resume=True, checkpoints=(1.0, 2.0, 3.0))


@pytest.mark.skipif(EOS_DIR is None, reason='EOS_DIR not found')
def test_runner_mushy_solve_length(tmp_path, monkeypatch):
    """While Phi_global >= 0.01 each solve call spans ``MUSHY_SOLVE_YR``, the last the remainder."""
    pytest.importorskip('jax')
    pytest.importorskip('scikits_odes_sundials')
    monkeypatch.syspath_prepend(str(_REPO / 'tools'))
    from verification import run_ssc_acceptance as rsa

    assert rsa.MUSHY_SOLVE_YR == 1000.0
    monkeypatch.setattr(rsa, 'MUSHY_SOLVE_YR', 0.4)
    spans = []
    solve = EntropySolver.solve

    def recording_solve(self):
        spans.append(self.parameters.solver.end_time - self.parameters.solver.start_time)
        solve(self)

    monkeypatch.setattr(EntropySolver, 'solve', recording_solve)
    config = _REPO / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    rsa.run_acceptance(config, EOS_DIR, tmp_path, checkpoints=(1.0,))
    assert spans == pytest.approx([0.4, 0.4, 0.2])
