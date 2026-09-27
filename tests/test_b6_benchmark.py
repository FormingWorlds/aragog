"""Slow benchmark test for Benchmark B6: Euen et al. (2023) Case A7.

Evaluates the stagnant-lid thermal convection case A7 at n = 100 nodes,
verifying steady-state convergence, Frank-Kamenetskii viscosity rheology,
and surface-to-base global energy balance.
"""

from __future__ import annotations

import pytest

from tools.verification.b6_euen2023 import run_simulation

pytestmark = [pytest.mark.slow, pytest.mark.timeout(60)]


def test_b6_a7_benchmark_energy_balance():
    """Case A7 steady-state run achieves global surface-to-base energy balance."""
    res = run_simulation('A7', n_nodes=100, law_on=True, convection_on=True, t_end_yr=2.0e11)

    assert res['status'] == 'CONVERGED'
    assert res['balance_error'] <= 1.0e-3
    assert res['fk_check'] <= 1.0e-10
    assert res['dual_flux_error_top'] <= 1.0e-3
    assert res['dual_flux_error_bot'] <= 1.0e-3
    assert res['dT_ad_ratio'] <= 1.0e-3
