"""Unit checks of the core_module CMB node: the default core start and the output node.

Both sit at the bottom cell carried to the CMB pressure, the mantle side of the contrast the
boundary-layer flux acts on, so a default start carries no CMB flux and the gradient slot
enters no reported node-0 value. No solve runs here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tests.conftest import entropy_eos_copy, needs_eos

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build  # noqa: E402

pytestmark = [pytest.mark.unit, needs_eos]


@pytest.fixture(scope='module')
def solver():
    return _build('core_module', entropy_eos_copy(), CORE_MODULE_PARAMS, s_init='driven')


def _t_mantle(solver, s_bottom):
    p_cmb = solver._P_basic_flat[:1]
    return float(np.asarray(solver.entropy_eos.temperature(p_cmb, np.array([s_bottom]))).item())


@pytest.mark.physics_invariant
def test_the_default_core_start_is_the_mantle_side_of_the_cmb(solver):
    """Without an explicit core temperature the core starts at T(S_bottom, P_cmb), so the
    boundary-layer flux is exactly zero at the start."""
    n_stag = solver._n_stag
    t_core = float(solver._S0[n_stag + 1])
    assert t_core == pytest.approx(_t_mantle(solver, float(solver._S0[0])), rel=1e-12)
    solver.dSdt(0.0, solver._S0)
    assert float(solver.state.heat_flux[0]) == 0.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('slot', [5.0e-4, -5.0e-4])
def test_the_output_cmb_node_is_the_bottom_cell_at_the_cmb_pressure(solver, slot):
    """With the core 50 K above the mantle, every node-0 output is the bottom cell at the
    CMB pressure whatever the gradient slot holds, and the applied outward flux is reported
    as conduction."""
    n_stag = solver._n_stag
    y = np.array(solver._S0, dtype=float)
    y[n_stag], y[n_stag + 1] = slot, y[n_stag + 1] + 50.0
    solver._solution = OptimizeResult(
        t=np.array([0.0, 1.0]), y=np.stack([y, y], axis=1), status=0, success=True, message=''
    )
    out = solver.get_state()
    assert float(out.T_basic[0]) == pytest.approx(_t_mantle(solver, y[0]), rel=1e-12)
    assert float(out.dSdr_b[0]) == 0.0
    assert float(out.jcond_b[0]) == float(out.heat_flux[0]) > 0.0
    assert [out.jconv_b[0], out.jgrav_b[0], out.jmix_b[0]] == [0.0, 0.0, 0.0]
    eos = solver.entropy_eos
    p_cmb, s0 = solver._P_basic_flat[:1], np.array([y[0]])
    assert float(out.phi_basic[0]) == pytest.approx(
        float(eos.melt_fraction(p_cmb, s0)[0]), rel=1e-10
    )
    assert float(out.rho_basic[0]) == pytest.approx(float(eos.density(p_cmb, s0)[0]), rel=1e-10)
