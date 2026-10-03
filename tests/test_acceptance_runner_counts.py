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
    '/tmp/aragog-test-data/spider_eos',
    str(_REPO.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
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


@pytest.mark.skipif(EOS_DIR is None, reason='EOS_DIR not found')
def test_runner_step_count_on_10yr_run_equals_cvode_numsteps(tmp_path, monkeypatch):
    """On a 10 yr run the reported step count equals CVODE NumSteps."""
    pytest.importorskip('jax')
    pytest.importorskip('scikits_odes_sundials')
    monkeypatch.syspath_prepend(str(_REPO / 'tools'))
    from verification.run_ssc_acceptance import run_acceptance

    sols = []
    solve = EntropySolver.solve

    def recording_solve(self):
        solve(self)
        sols.append(self._solution)

    monkeypatch.setattr(EntropySolver, 'solve', recording_solve)
    config = _REPO / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    res = run_acceptance(config, EOS_DIR, tmp_path, checkpoints=(10.0,))

    assert len(sols) == 1
    sol = sols[0]
    expected_steps = sol.cvode_info['NumSteps']
    assert expected_steps == sol.cvode_nst
    assert expected_steps > sol.t.size
    assert 'MinLastStep' in sol.cvode_info
    assert 'LastStep' in sol.cvode_info
    assert sol.cvode_info['MinLastStep'] <= sol.cvode_info['LastStep']
    assert hasattr(sol, 'cvode_min_last_step')
    assert sol.cvode_min_last_step <= sol.cvode_last_step
    assert res['cvode_steps'][0] == expected_steps
    assert res['total_cvode_steps'] == expected_steps
    assert res['total_cvode_steps'] > 50


def test_cvode_info_distinguishes_final_segment_and_minimum_last_step():
    """Verify LastStep retains final segment value while MinLastStep records multi-segment minimum."""
    from unittest.mock import MagicMock

    from aragog.config import Config

    solver = object.__new__(EntropySolver)
    solver.parameters = Config()
    solver._output_grid = MagicMock(return_value=np.array([1.0]))

    seg1 = OptimizeResult(
        t=np.array([0.0, 0.5]),
        y=np.ones((5, 2)),
        status=0,
        cvode_flag=2,
        nfev=20,
        cvode_nst=10,
        cvode_nfe=20,
        cvode_info={
            'NumSteps': 10,
            'NumRhsEvals': 20,
            'NumErrTestFails': 1,
            'NumLinSolvSetups': 2,
            'LastStep': 1.5,
        },
    )

    seg2 = OptimizeResult(
        t=np.array([0.5, 1.0]),
        y=np.ones((5, 2)),
        status=0,
        cvode_flag=0,
        nfev=30,
        cvode_nst=15,
        cvode_nfe=30,
        cvode_info={
            'NumSteps': 15,
            'NumRhsEvals': 30,
            'NumErrTestFails': 2,
            'NumLinSolvSetups': 3,
            'LastStep': 15.0,
        },
    )

    solver._solve_cvode = MagicMock(side_effect=[seg1, seg2])
    solver._nondimensional_rhs = MagicMock()

    roots = MagicMock()
    roots.fired = MagicMock(return_value='step')
    roots.inside = False

    res = solver._solve_cvode_segments(
        start_time=0.0,
        end_time=1.0,
        y0=np.ones(5),
        h_at=lambda t, y: 1.0,
        roots_at=lambda y, inside: roots,
        t_ref=1.0,
        h_min=1.0,
        max_segments=2,
    )

    assert res.cvode_info['LastStep'] == 15.0
    assert res.cvode_info['MinLastStep'] == 1.5
    assert res.cvode_info['MinLastStep'] < res.cvode_info['LastStep']
    assert res.cvode_info['NumSteps'] == 25
    assert res.cvode_info['NumRhsEvals'] == 50
