"""CVODE counters reported by the solid-state acceptance runner.

The runner's per-interval step counts and step sizes must come from CVODE's own
counters, not from the dense output grid of ``sol.t`` (65 points per solve).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from aragog.cli import _derive_initial_entropy_from_config
from aragog.solver.entropy_solver import EntropySolver

pytestmark = [pytest.mark.slow, pytest.mark.timeout(300)]

_REPO = Path(__file__).resolve().parent.parent
_EOS = os.environ.get('ARAGOG_TEST_EOS_DIR')


@pytest.mark.skipif(not (_EOS and Path(_EOS).is_dir()), reason='ARAGOG_TEST_EOS_DIR not set')
def test_runner_counts_come_from_cvode():
    sys.path.insert(0, str(_REPO / 'tools'))
    from verification.run_ssc_acceptance import cvode_counts

    config = _REPO / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    solver = EntropySolver.from_file(str(config), eos_dir=Path(_EOS))
    solver.parameters.energy.use_jax_jacobian = False
    solver.initialize()
    solver.set_initial_entropy(_derive_initial_entropy_from_config(solver))
    solver.parameters.solver.start_time = 0.0
    solver.parameters.solver.end_time = 1.0
    solver.solve()
    sol = solver.solution

    nst, netf, nsetups, h_last = cvode_counts(sol)
    assert nst == sol.cvode_nst == sol['cvode_info']['NumSteps']
    assert netf == sol['cvode_info']['NumErrTestFails']
    assert nsetups == sol['cvode_info']['NumLinSolvSetups'] >= 1
    assert 0.0 < h_last <= 1.0
