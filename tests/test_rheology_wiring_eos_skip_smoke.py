"""The rheology wiring tests that read the SPIDER P-S tables skip without them."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.conftest import _CANDIDATES, _REPO_ROOT

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(600)]

NEEDS_TABLES = [
    'tests/test_rheology_solver_wiring.py::test_arrhenius_reference_viscosity_ci_table_vs_linear_solidus',
    'tests/test_rheology_solver_wiring.py::test_update_stores_the_pass_two_viscosity_when_yielding_is_active',
    'tests/test_rheology_solver_wiring_smoke.py::test_rhs_numpy_jax_parity_rheology_on',
    'tests/test_rheology_solver_wiring_smoke.py::test_rhs_parity_rheology_on_energy_balance_mode',
    'tests/test_rheology_solver_wiring_smoke.py::test_rhs_parity_rheology_on_core_module_mode',
    'tests/test_rheology_solver_wiring_integration.py::test_rheology_off_preserves_main_rhs_and_solve',
    'tests/test_rheology_solver_wiring_integration.py::test_solve_short_trajectory_rheology_on_lid_and_local',
    'tests/test_rheology_solver_wiring_integration.py::test_solve_rheology_invariants_checked_inside_step_energy_integrals_without_double_replay',
    'tests/test_rheology_solver_wiring_integration.py::test_step_energy_integrals_checks_eos_entropy_range_on_every_step_both_branches',
]


def test_rheology_wiring_tests_skip_without_the_eos_tables(tmp_path) -> None:
    """Without the tables, each of these tests skips with the conftest reason and none fails.

    The child run points ``ARAGOG_TEST_EOS_DIR`` at a path that does not exist and unsets
    ``FWL_DATA``. The parity cache next to the repository is the last place conftest looks,
    so the check cannot run where that cache exists.
    """
    if os.path.exists(_CANDIDATES[-1]):
        pytest.skip(f'the parity cache {_CANDIDATES[-1]} provides the tables')
    env = {k: v for k, v in os.environ.items() if k != 'FWL_DATA'}
    env['ARAGOG_TEST_EOS_DIR'] = str(tmp_path / 'no_tables')
    src = str(_REPO_ROOT / 'src')
    env['PYTHONPATH'] = os.pathsep.join(
        filter(None, [src, str(_REPO_ROOT), env.get('PYTHONPATH')])
    )
    run = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            *NEEDS_TABLES,
            '-o',
            'addopts=',
            '-p',
            'no:cacheprovider',
            '-p',
            'no:randomly',
            '-q',
            '-rs',
            '--tb=no',
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=540,
    )
    out = run.stdout
    skipped = [line for line in out.splitlines() if 'SPIDER P-S tables not found' in line]
    assert run.returncode == 0, out[-2000:]
    assert ' failed' not in out, out[-2000:]
    assert f'{len(NEEDS_TABLES)} skipped' in out, out[-2000:]
    assert len(skipped) == len(NEEDS_TABLES), out[-2000:]
