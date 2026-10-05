"""Fixture entering stagnant lid formation and plastic yield closure.

References
----------
Moresi & Solomatov (1998), p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x
    (viscosity switch / minimum closure eta_eff = min(eta_diff, eta_y))
Foley & Bercovici (2014), GJI 199, pp. 580-603, doi:10.1093/gji/ggu275
    (convective driving shear stress tau_d = 2 * mu_i * v_m / d)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('scikits_odes_sundials')
pytest.importorskip('jax')

from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from tests.fixture_helpers import get_cold_top_lid_initial_entropy

_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    '/Users/timlichtenberg/work/ssc-verify-task6/test-data/spider_eos',
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
]
EOS_DIR = next((Path(p) for p in _CANDIDATES if p and Path(p).exists()), None)


@pytest.fixture(scope='module')
def shared_eos():
    if EOS_DIR is None:
        pytest.skip('EOS_DIR not found')
    return EntropyEOS(EOS_DIR)


@pytest.mark.unit
@pytest.mark.parametrize('mode,tau_y_val', [('yielding', 1.0e4), ('sub_yield', 5.0e4)])
def test_stagnant_lid_yielding_fixture(shared_eos, mode, tau_y_val):
    """Verify stagnant lid yielding active and sub-yield contracts in integrated solve.

    Parameters
    ----------
    shared_eos : EntropyEOS
        Shared equation of state evaluator.
    mode : str
        Yielding mode ('yielding' or 'sub_yield').
    tau_y_val : float
        Target yield stress [Pa].

    Notes
    -----
    Per Ruling 119, pins lid_mask > 0 at top nodes, w_y > 0.9, and the top-node
    eta_eff == eta_diff * tau_y / tau_d to rtol 1e-8 for yielding, and
    eta_eff == eta_diff exactly for sub-yield. Runs in under 3 s per case.
    """
    config_file = 'tests/configs/cold_top_lid.toml'
    cfg = Config.from_file(config_file)
    cfg.phase_solid.yield_stress_mu = 0.0
    cfg.phase_solid.yield_stress_c = tau_y_val
    cfg.phase_solid.yield_stress_max = tau_y_val
    cfg.solver.end_time = 0.01

    solver = EntropySolver(cfg, entropy_eos=shared_eos)
    solver.initialize()
    s0 = get_cold_top_lid_initial_entropy(solver.evaluator.mesh)
    solver.set_initial_entropy(s0)
    solver.solve()

    output = solver.get_state()
    assert output.status == 0

    st = solver.state
    lid = st.lid_state
    w_y = float(lid['w_y'])
    tau_d = float(lid['tau_d'])
    tau_y = float(lid['tau_y_lid'])
    visc_eff = st.visc_eff
    eta_diff = st.phase_basic.eta_diff

    # 1. Lid formation: upper cells in solid lid regime
    assert output.lid_mask_b[-1] > 0.0
    assert output.lid_mask_b[-2] > 0.0

    if mode == 'yielding':
        # 2. Active yielding: w_y > 0.9
        assert w_y > 0.9
        assert tau_d > tau_y
        # 3. Top-node effective viscosity matches minimum closure to rtol 1e-8
        expected = eta_diff[-1] * (tau_y / tau_d)
        rel_diff = abs(visc_eff[-1] - expected) / expected
        assert rel_diff <= 1.0e-8
    else:
        # 4. Sub-yield: tau_d < tau_y and eta_eff == eta_diff exactly
        assert tau_d < tau_y
        assert visc_eff[-1] == eta_diff[-1]
