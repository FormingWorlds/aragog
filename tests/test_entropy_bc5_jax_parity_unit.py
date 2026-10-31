"""NumPy-JAX parity of the BC 5 right-hand side in the unit tier, so that PR CI checks the JAX
wiring of the prescribed surface temperature on the test tables."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import needs_eos

pytestmark = [pytest.mark.unit, needs_eos, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
def test_the_jax_rhs_matches_numpy_under_bc5_on_a_graded_profile():
    """quasi_steady, nearest_boundary, a mushy surface node at 1500 K and an entropy falling
    from 2900 to 2600 J/kg/K: the JAX right-hand side equals the NumPy one in every cell, and
    the top cell differs from the prescribed-flux mode with the temperature read as a flux."""
    pytest.importorskip('jax')
    pytest.importorskip('equinox')
    from tests.test_entropy_bc5_jax_parity_smoke import _bc5_solver, _jax_rhs, _jax_rhs_and_args

    s = _bc5_solver('quasi_steady')
    y = np.asarray(s._S0, dtype=float)
    f_np = np.asarray(s.dSdt(0.0, y)).ravel()
    np.testing.assert_allclose(_jax_rhs(s, y), f_np, rtol=1e-10)
    fn4, args4 = _jax_rhs_and_args(s, outer_bc_type=4)
    f4 = np.asarray(fn4(0.0, y, args4)).ravel()
    assert f4[-1] != pytest.approx(f_np[-1], rel=1e-3)


def _bc5_rheology_rhs(t_surf, top_entropy, **rheology):
    """Solver, NumPy and JAX right-hand sides under BC 5 at ``t_surf`` [K] with solid rheology on.

    ``top_entropy(s_sol, s_liq)`` gives the top-cell entropy [J/(kg K)] of a linear profile
    that starts 65 J/(kg K) below the bottom solidus.
    """
    pytest.importorskip('jax')
    pytest.importorskip('equinox')
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.parser import SolidRheologyParams
    from tests.conftest import EOS_DIR, entropy_eos_copy
    from tests.test_entropy_bc5_jax_parity_smoke import _jax_rhs_and_args
    from tests.test_entropy_solver_core_module_smoke import _build

    s = _build('quasi_steady', entropy_eos_copy(EOS_DIR, identity_alpha=True), None, n_nodes=20)
    b = s.parameters.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, t_surf
    s.parameters.phase_mixed.matprop_smooth_width = 0.0
    s.parameters.phase_solid.rheology = SolidRheologyParams(enabled=True, **rheology)
    s.initialize()
    eos = s.state.phase_basic._eos
    p_b = np.asarray(s.state.phase_basic.pressure, dtype=float)
    s_sol = np.asarray(eos.solidus_entropy(p_b), dtype=float)
    s_liq = np.asarray(eos.liquidus_entropy(p_b), dtype=float)
    s.set_initial_entropy(np.linspace(s_sol[0] - 65.0, top_entropy(s_sol, s_liq), s._n_stag))

    y = np.asarray(s._S0, dtype=float)
    f_np = np.asarray(s.dSdt(0.0, y)).ravel()
    fn, args = _jax_rhs_and_args(s)
    args = list(args)
    args[1] = PhaseParams.from_parameters(s.parameters)
    args[2] = MeshArrays.from_numpy_mesh(s.evaluator.mesh, validate_rheology=True)
    f_jax = np.asarray(fn(0.0, y, tuple(args))).ravel()
    return s, f_np, f_jax


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mode', ['lid', 'local'])
def test_the_jax_rhs_matches_numpy_under_bc5_with_solid_rheology(mode):
    """Solid rheology on, a 1500 K mushy surface over a top cell 30 % into the melting
    interval: dS/dr < 0 at the surface, so the kappa_h floor would act there in either MLT pass
    of the JAX path; the JAX right-hand side equals the NumPy one in every cell."""
    s, f_np, f_jax = _bc5_rheology_rhs(
        1500.0, lambda sol, liq: sol[-1] + 0.3 * (liq[-1] - sol[-1]), stress_closure_mode=mode
    )
    assert s.state._dSdr[-1] < 0.0
    assert np.asarray(s.state.phase_basic.melt_fraction()).ravel()[-1] > 0.0
    np.testing.assert_allclose(f_jax, f_np, rtol=1e-10, atol=1e-10 * np.max(np.abs(f_np)))


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mode', ['lid', 'local'])
def test_the_jax_rhs_matches_numpy_under_bc5_with_an_active_yield_branch(mode):
    """A 1000 K surface and a low yield stress make at least one node yield; the JAX
    right-hand side still equals the NumPy one in every cell."""
    s, f_np, f_jax = _bc5_rheology_rhs(
        1000.0,
        lambda sol, liq: sol[-1] - 400.0,
        stress_closure_mode=mode,
        yield_stress_c=1.0e3,
        yield_stress_mu=0.0,
    )
    eta_eff = np.asarray(s.state._eta_eff_basic, dtype=float)
    eta_diff = np.asarray(s.state.phase_basic.eta_diff(), dtype=float)
    assert int(np.sum(eta_eff < 0.99 * eta_diff)) > 0, 'the yield branch must be active'
    np.testing.assert_allclose(f_jax, f_np, rtol=1e-10, atol=1e-10 * np.max(np.abs(f_np)))
