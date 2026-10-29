"""Probe of the solve path with the sub-grid lid off, for the off-path reference.

``probe(rheology, jax)`` builds a 12-node quasi_steady column with the grey-body surface, with
or without the solid rheology (the 'lid' stress closure), on CVODE with the finite-difference
Jacobian or with a JAX factory, and returns the mesh, one right-hand side at the start state and
a 2-yr solve. ``build_subgrid_lid_v0_reference.py`` stores it for the commit the lid branches
from; ``tests/test_subgrid_lid_off_path.py`` compares the current code with it.
"""

from __future__ import annotations

import numpy as np

N_NODES = 12
END_YR = 2.0
S_PROFILE = (3150.0, 3000.0)
CASES = [(False, False), (True, False), (False, True), (True, True)]


def case_name(rheology: bool, jax: bool) -> str:
    return f'{"rheo" if rheology else "plain"}_{"jax" if jax else "numpy"}'


def _parameters(rheology: bool, jax: bool):
    from aragog.parser import SolidRheologyParams
    from tests.test_entropy_solver_option_z_smoke import _build_parameters

    p = _build_parameters(n_nodes=N_NODES, end_time=END_YR)
    p.energy.use_jax_jacobian = jax
    if rheology:
        p.phase_solid.rheology = SolidRheologyParams(enabled=True)
    return p


def _factory(solver, eos_dir, rheology: bool, info_box: list, factory_kw: dict):
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    from tests.conftest import entropy_eos_jax

    eos_jax = entropy_eos_jax(eos_dir)

    def factory(scales, core_bc_mode):
        bc = solver.parameters.boundary_conditions
        rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=PhaseParams.from_parameters(solver.parameters),
            mesh_arrays=MeshArrays.from_numpy_mesh(
                solver.evaluator.mesh, validate_rheology=rheology
            ),
            boundary_params=BoundaryParams(
                outer_bc_type=1,
                outer_bc_value=0.0,
                emissivity=float(bc.emissivity),
                T_eq=float(bc.equilibrium_temperature),
                inner_bc_type=2,
                inner_bc_value=0.0,
                core_density=float(solver._core_density),
                core_heat_capacity=float(solver._core_cp),
                tfac_core_avg=float(solver._core_tfac),
            ),
            heating_array=np.zeros(solver._n_stag),
            scales=scales,
            core_bc_mode=core_bc_mode,
            **factory_kw,
        )
        info_box.append(info)
        return rhs_fn, jac_fn

    return factory


def probe(rheology: bool, jax: bool, eos_dir, factory_kw=None) -> dict[str, np.ndarray]:
    """Mesh, start right-hand side and 2-yr solve of one case (arrays, SI and yr);
    ``factory_kw`` goes to ``build_jax_rhs_and_jacobian``."""
    from aragog.solver.entropy_solver import EntropySolver
    from tests.conftest import entropy_eos_copy

    s = EntropySolver(_parameters(rheology, jax), entropy_eos=entropy_eos_copy(eos_dir))
    s.initialize()
    S = np.linspace(*S_PROFILE, s._n_stag)
    s.set_initial_entropy(S)
    info_box: list = []
    if jax:
        s.set_jax_cvode_factory(_factory(s, eos_dir, rheology, info_box, factory_kw or {}))
    mesh = s.evaluator.mesh
    out = dict(
        r_basic=np.asarray(mesh.basic.radii, dtype=float).ravel(),
        r_stag=np.asarray(mesh.staggered.radii, dtype=float).ravel(),
        volume=np.asarray(mesh.basic.volume, dtype=float).ravel(),
        rhs=np.asarray(s.dSdt(0.0, np.asarray(s._S0, dtype=float)), dtype=float).ravel(),
    )
    s.solve()
    sol = s._solution
    out.update(
        t=np.asarray(sol.t, dtype=float),
        y_end=np.asarray(sol.y, dtype=float)[:, -1],
        status=np.array([int(sol.status)]),
        jax_rhs_calls=np.array([int(info_box[-1]['rhs_calls']) if info_box else 0]),
    )
    return out
