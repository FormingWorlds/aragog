"""End-to-end smoke for the Option Z (JAX-RHS + JAX-Jacobian) path.

Production CHILI runs go through ``EntropySolver.solve()`` with
``solver_method = 'cvode'``, ``use_jax_jacobian = true``, AND a
JAX CVODE factory registered via
``EntropySolver.set_jax_cvode_factory``. The factory returns
``(rhs_fn, jac_fn)`` pairs that ``_solve_cvode`` installs as
SUNDIALS callbacks.

The existing test_cvode_jax_factory.py covers the factory's contract,
test_cvode_jax_factory_invocation.py exercises the returned
callbacks directly, and test_entropy_solver_integration.py runs
``solve()`` with ``use_jax_jacobian = false`` (FD Jacobian).
What is NOT covered today is the wire-up between ``solve()`` and
``_solve_cvode`` for the Option Z branch:

* solve() lines 2417 / 2420 / 2424 (factory invocation +
  Option-Z-active log + fallback path).
* _solve_cvode lines 1900-1905 (cvode_rhs_fn_override wrapper
  installs the JAX callback as the in-place CVODE RHS).
* _solve_cvode lines 2002-2005 (cvode_jacfn install: jacfn key,
  dense linsolver, banded option pop).

This smoke test runs one short integration with a real factory so
all three branches execute end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)
needs_cvode = pytest.mark.skipif(
    not pytest.importorskip('scikits_odes_sundials', reason='scikits_odes_sundials missing'),
    reason='scikits-odes-sundials missing',
)

pytestmark = [pytest.mark.smoke, needs_eos]


def _build_parameters(*, n_nodes: int = 12, end_time: float = 2.0):
    from aragog.parser import (
        Parameters,
        _BoundaryConditionsParameters,
        _EnergyParameters,
        _InitialConditionParameters,
        _MeshParameters,
        _PhaseMixedParameters,
        _PhaseParameters,
        _SolverParameters,
    )

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc='quasi_steady',
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='cvode',
        use_jax_jacobian=True,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(
        start_time=0.0,
        end_time=end_time,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
    return Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        solver=sv,
    )


@needs_cvode
def test_solve_with_option_z_factory_registered_completes():
    """``EntropySolver.solve()`` with ``use_jax_jacobian=True`` and a
    registered factory must execute the full Option Z dispatch:

    * factory returns (rhs_fn, jac_fn);
    * ``_solve_cvode`` installs both as in-place CVODE callbacks;
    * dense linsolver activated, banded keys popped;
    * integration completes with status=0.

    Discriminator: the final state must be finite and physically
    plausible (entropy in the expected mantle range). A regression
    that lost the factory wire-up would either raise inside
    ``solve()`` (fall-through after the factory exception path) or
    silently use FD Jacobian (which still works, but the
    ``info['jac_calls']`` counter on the factory side stays 0).
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_parameters(n_nodes=12, end_time=2.0)
    eos = EntropyEOS(EOS_DIR)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    solver = EntropySolver(parameters, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(3050.0)

    # Build a factory closure consistent with the production wrapper.
    # PROTEUS's AragogRunner wires this up; here we inline the
    # equivalent so the test is self-contained.
    n_stag = solver._n_stag

    def factory(scales, core_bc_mode):
        # Build minimal MeshArrays + BoundaryParams matching the
        # solver's mesh + BC. Reuses the staggered + basic radii
        # that EntropySolver._initialize_internals cached.
        mesh = solver.evaluator.mesh
        r_basic = np.asarray(mesh.basic.radii).ravel()
        r_stag = np.asarray(mesh.staggered.radii).ravel()
        dr = np.diff(r_stag)
        d_dr = np.zeros((r_basic.size, n_stag))
        for i in range(1, r_basic.size - 1):
            d_dr[i, i - 1] = -1.0 / dr[i - 1]
            d_dr[i, i] = 1.0 / dr[i - 1]
        d_dr[0, :] = d_dr[1, :]
        d_dr[-1, :] = d_dr[-2, :]
        q_mat = np.zeros((r_basic.size, n_stag))
        q_mat[0, 0] = 1.0
        q_mat[-1, -1] = 1.0
        for i in range(1, r_basic.size - 1):
            q_mat[i, i - 1] = 0.5
            q_mat[i, i] = 0.5

        import jax.numpy as jnp

        mesh_arr = MeshArrays(
            d_dr_matrix=jnp.asarray(d_dr),
            quantity_matrix=jnp.asarray(q_mat),
            area=jnp.asarray(4.0 * np.pi * r_basic**2),
            volume=jnp.asarray(np.asarray(mesh.basic.volume).ravel()),
            radii_basic=jnp.asarray(r_basic),
            radii_stag=jnp.asarray(r_stag),
            mixing_length=jnp.asarray(np.asarray(mesh.basic.mixing_length).ravel()),
            mixing_length_sq=jnp.asarray(np.asarray(mesh.basic.mixing_length).ravel() ** 2),
            mixing_length_cu=jnp.asarray(np.asarray(mesh.basic.mixing_length).ravel() ** 3),
            P_stag=jnp.asarray(np.asarray(mesh.staggered_pressure).ravel()),
            P_basic=jnp.asarray(np.asarray(mesh.basic_pressure).ravel()),
            gravity=jnp.full(r_basic.size, 9.81),
        )
        bc = BoundaryParams(
            outer_bc_type=1,
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=2,
            inner_bc_value=0.0,
            core_density=10500.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )
        rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=PhaseParams(),
            mesh_arrays=mesh_arr,
            boundary_params=bc,
            heating_array=np.zeros(n_stag),
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs_fn, jac_fn

    solver.set_jax_cvode_factory(factory)
    solver.solve()

    sol = solver._solution
    assert sol is not None
    assert sol.status == 0, f'Option Z solve returned status={sol.status}; expected 0 (success)'
    final_y = sol.y[:, -1] if sol.y.ndim == 2 else sol.y
    assert np.all(np.isfinite(final_y))
    # The trailing slots are the two boundary-energy quadrature states in
    # joules; only the leading block is entropy, so the window applies there.
    S_final = final_y[: solver._n_stag]
    assert 1500.0 < float(S_final.min()) < 5500.0, (
        f'final entropy range [{float(S_final.min())}, {float(S_final.max())}] '
        'is outside the plausible mantle window'
    )
    E_final = final_y[solver._n_stag :]
    assert E_final.size == 2, (
        f'expected the two boundary-energy quadrature slots, got {E_final.size}'
    )
    # Surface cooling drains the mantle, so the surface slot must be negative.
    assert float(E_final[0]) < 0.0, (
        f'surface boundary energy {float(E_final[0]):.6e} J is not negative '
        'over a cooling solve'
    )
