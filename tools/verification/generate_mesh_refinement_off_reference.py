#!/usr/bin/env python3
"""Write the mesh, RHS and short-solve arrays of abe_solid with the mesh refinement off.

Run from the root of a clean aragog tree with ``ARAGOG_TEST_EOS_DIR`` set to the SPIDER
P-S test tables. For each mesh mode (mass coordinates, radius) the file holds the basic and
staggered mesh arrays, one NumPy RHS evaluation, and the end state of a 100-year solve on
the NumPy RHS and on the JAX CVODE factory. ``<mode>/jax/rhs_calls`` counts the calls of
the JAX RHS during the solve; the script stops when it is 0, because ``solve()`` falls back
to the NumPy RHS when the factory fails. The tree's commit SHA is stored as
``source_commit``.

Usage: ``python tools/verification/generate_mesh_refinement_off_reference.py <out.npz>``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import aragog  # noqa: E402
from aragog.mesh import Mesh  # noqa: E402
from aragog.parser import Parameters  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402
from tests.conftest import entropy_eos_copy, entropy_eos_jax  # noqa: E402
from tests.test_step_powers_batch import _make_phase_params, _set_initial_entropy  # noqa: E402

CFG = ROOT / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'
MESH_ARRAYS = ('basic.radii', 'staggered.radii', 'basic.volume', 'basic.mixing_length',
               'basic_pressure', 'staggered_pressure', 'staggered_effective_density',
               '_d_dr_transform')


def _commit() -> str:
    """Commit SHA of the clean tree at ``ROOT``; refuses a tree with changes."""
    git = ['git', '-C', str(ROOT)]
    if subprocess.run([*git, 'status', '--porcelain'], capture_output=True, text=True,
                      check=True).stdout.strip():
        raise SystemExit(f'{ROOT} has uncommitted changes')
    return subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


def _parameters(mass_coordinates: bool) -> Parameters:
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = mass_coordinates
    p.solver.end_time = 100.0
    return p


def _solver(p: Parameters) -> EntropySolver:
    s = EntropySolver(p, entropy_eos=entropy_eos_copy(identity_alpha=True))
    s.initialize()
    _set_initial_entropy(s)
    return s


def _jax_factory(s: EntropySolver, p: Parameters, calls: list[int]):
    """CVODE factory of the JAX RHS and Jacobian on the solver's mesh and boundary settings;
    ``calls[0]`` counts the RHS calls."""
    from aragog.jax.phase import MeshArrays
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    b = p.boundary_conditions
    mesh = MeshArrays.from_numpy_mesh(s.evaluator.mesh)
    r = np.asarray(mesh.radii_basic)
    bc = BoundaryParams(
        outer_bc_type=int(b.outer_boundary_condition), outer_bc_value=float(b.outer_boundary_value),
        emissivity=float(b.emissivity), T_eq=float(b.equilibrium_temperature),
        inner_bc_type=int(b.inner_boundary_condition), inner_bc_value=float(b.inner_boundary_value),
        core_density=float(p.mesh.core_density), core_heat_capacity=float(b.core_heat_capacity),
        tfac_core_avg=float(s._core_tfac), cmb_area=4.0 * np.pi * float(r[0]) ** 2,
        core_M=float(s._core_cap / b.core_heat_capacity), cmb_dr_cmb=float(r[1] - r[0]),
        param_utbl=bool(b.param_utbl), param_utbl_const=float(b.param_utbl_const))

    def factory(scales, core_bc_mode):
        rhs, jac, _ = build_jax_rhs_and_jacobian(
            eos_jax=entropy_eos_jax(), phase_params=_make_phase_params(p), mesh_arrays=mesh,
            boundary_params=bc, heating_array=np.zeros(s._n_stag), scales=scales,
            core_bc_mode=core_bc_mode)

        def rhs_counted(*a, **k):
            calls[0] += 1
            return rhs(*a, **k)

        return rhs_counted, jac

    return factory


def _case(mass_coordinates: bool) -> dict[str, np.ndarray]:
    """Mesh, RHS and end-state arrays of one mesh mode."""
    out = {}
    mesh = Mesh(_parameters(mass_coordinates))
    for name in MESH_ARRAYS:
        obj = mesh
        for part in name.split('.'):
            obj = getattr(obj, part)
        out[f'mesh/{name}'] = np.asarray(obj, dtype=float)
    s = _solver(_parameters(mass_coordinates))
    out['rhs/numpy'] = np.asarray(s.dSdt(0.0, np.asarray(s._S0, float)), float)
    s.solve()
    out['solve/numpy/y_end'] = np.asarray(s._solution.y[:, -1], float)
    p = _parameters(mass_coordinates)
    s = _solver(p)
    calls = [0]
    s.set_jax_cvode_factory(_jax_factory(s, p, calls))
    s.solve()
    if calls[0] == 0:
        raise SystemExit('the JAX solve did not call the JAX RHS')
    out['solve/jax/y_end'] = np.asarray(s._solution.y[:, -1], float)
    out['solve/jax/rhs_calls'] = np.asarray(calls[0])
    return out


def main(path: Path) -> None:
    arrays = {'source_commit': np.asarray(_commit())}
    for mode, mass in (('mass', True), ('radius', False)):
        arrays |= {f'{mode}/{k}': v for k, v in _case(mass).items()}
    np.savez_compressed(path, **arrays)
    print(f'{path}: {len(arrays)} arrays from {aragog.__file__} at {arrays["source_commit"]}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
