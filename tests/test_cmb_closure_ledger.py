"""Core energy ledger of the CMB closure: the core heat change equals the CMB heat flow.

The energy_balance and gradient boundary conditions move the CMB basic-node entropy so that
the node temperature follows the core balance ``c_core T_hat M_core dT_cmb/dt = -E`` (Bower
et al. 2018, Eqs. 19-21). The node temperature is the phase blend ``T(P, S)``, so the
closure must convert with ``(dT/dS)_P`` of that blend. These tests run the solver with the
CMB node in the mush, where ``c_p/T`` of the blend differs from ``(dS/dT)_P`` by
``T_avg/T_mix``, and require the ledger to close over the call.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EOS_DIR, entropy_eos_copy, needs_eos
from tests.test_entropy_solver_integration import _build_parameters

pytestmark = [pytest.mark.smoke, needs_eos, pytest.mark.timeout(600)]

# Tight tolerances, so the solver error floor sits far below the closure bias (about 0.1).
RTOL = 1.0e-10
LEDGER_TOL = 1.0e-4


def _jax_factory(solver):
    """Option Z factory for an energy_balance solver (JAX RHS and Jacobian)."""
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    from tests.test_jax_dsdt_core_module import _build_jax_pieces

    for name, value in (
        ('_core_module_budget', None),
        ('_core_module_q_radio', 0.0),
        ('_core_module_ra_crit_cmb', None),
    ):
        setattr(solver, name, value)
    args = list(_build_jax_pieces(solver))
    bc = solver.parameters.boundary_conditions
    args[3] = BoundaryParams(
        outer_bc_type=bc.outer_boundary_condition,
        outer_bc_value=bc.outer_boundary_value,
        emissivity=bc.emissivity,
        T_eq=bc.equilibrium_temperature,
        inner_bc_type=5,
        inner_bc_value=0.0,
        core_density=float(solver._core_density),
        core_heat_capacity=float(solver._core_cp),
        tfac_core_avg=float(solver._core_tfac),
        cmb_area=float(solver._cmb_area),
        core_M=float(solver._core_M),
        cmb_dr_cmb=float(solver._cmb_dr_cmb),
    )
    calls = {'n': 0}

    def factory(scales, mode):
        calls['n'] += 1
        rhs, jac, _ = build_jax_rhs_and_jacobian(
            eos_jax=args[0],
            phase_params=args[1],
            mesh_arrays=args[2],
            boundary_params=args[3],
            heating_array=np.zeros(solver._n_stag),
            scales=scales,
            core_bc_mode=mode,
        )
        return rhs, jac

    return factory, calls


def _node_temperature(solver, t, y) -> float:
    """CMB basic-node temperature [K] of the solver state ``y`` at ``t`` [yr]."""
    solver._dSdt_single(float(t), np.asarray(y, dtype=float))
    return float(np.asarray(solver.state.phase_basic.temperature()).flat[0])


def _ledger(core_bc, method, g0, jax_rhs=False, end_time=20.0):
    """Run one call from a mush profile at melt coordinate ``g0``; return the ledger terms."""
    from aragog.solver.entropy_solver import EntropySolver

    params = _build_parameters(
        core_bc=core_bc,
        n_nodes=15,
        end_time=end_time,
        inner_boundary_condition=1,
        solver_method=method,
        use_jax_jacobian=jax_rhs,
    )
    params.solver.rtol = params.solver.atol = RTOL
    eos = entropy_eos_copy(EOS_DIR)
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    P = np.asarray(solver._P_stag_flat, dtype=float)
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    solver.set_initial_entropy(S_sol + g0 * (S_liq - S_sol))
    calls = None
    if jax_rhs:
        factory, calls = _jax_factory(solver)
        solver.set_jax_cvode_factory(factory)
    solver.solve()
    out = solver.get_state()
    sol = solver._solution
    c_core = float(solver._core_cp * solver._core_tfac * solver._core_M)
    d_t = _node_temperature(solver, sol.t[-1], sol.y[:, -1]) - _node_temperature(
        solver, sol.t[0], sol.y[:, 0]
    )
    q = float(out.step_dE_F_cmb_J)
    return c_core * d_t, q, calls


@pytest.mark.parametrize('g0', [0.9, 0.3])
@pytest.mark.parametrize(
    ('core_bc', 'method', 'jax_rhs'),
    [
        ('energy_balance', 'radau', False),
        ('energy_balance', 'cvode', False),
        ('energy_balance', 'cvode', True),
        ('gradient', 'radau', False),
    ],
)
def test_core_heat_change_equals_the_cmb_heat_flow(core_bc, method, jax_rhs, g0):
    """``c_core T_hat M_core dT_node + int E dt = 0`` over a call with the CMB node in the mush."""
    if method == 'cvode':
        pytest.importorskip('scikits_odes_sundials')
    if jax_rhs:
        pytest.importorskip('equinox')
    d_e, q, calls = _ledger(core_bc, method, g0, jax_rhs=jax_rhs)
    if jax_rhs:
        assert calls['n'] > 0, 'the JAX factory was not used'
    assert q > 0.0, f'the fixture must cool the core, q = {q:.4e} J'
    miss = (d_e + q) / abs(q)
    assert abs(miss) < LEDGER_TOL, (
        f'core ledger misses by {miss:+.5f} of the CMB heat flow '
        f'(core change {d_e:+.4e} J, CMB heat {q:+.4e} J)'
    )
