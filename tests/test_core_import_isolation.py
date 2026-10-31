"""The energy_balance and rheology solver paths load no part of ``aragog.core``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

_SUBPROCESS_SOLVE_CODE = """\
import sys
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import aragog.jax.solver
import aragog.solver.entropy_solver
import aragog.solver.cvode_jax

print(sorted(m for m in sys.modules if m.startswith('aragog.core')))

from aragog.config.phases import SolidRheologyParams
from aragog.eos.entropy import EntropyEOS
from aragog.jax.eos import EntropyEOS_JAX
from aragog.jax.phase import MeshArrays, PhaseParams
from aragog.jax.solver import BoundaryParams, _no_radio, dSdt
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
from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
from aragog.solver.entropy_solver import EntropySolver

with tempfile.TemporaryDirectory() as tmp:
    eos_dir = Path(tmp)
    P = np.linspace(0.0, 1.5e11, 4)
    S = np.linspace(1000.0, 4000.0, 4)

    def _write_2d(fname, vals):
        lines = ['# 2 4 4', '# 1.0 1.0 1.0']
        for s in S:
            for p, v in zip(P, vals(P, s)):
                lines.append(f'{p:.6e} {s:.6e} {v:.6e}')
        (eos_dir / fname).write_text('\\n'.join(lines) + '\\n')

    _write_2d('temperature_solid.dat', lambda p, s: 1000.0 + 0.5 * s + 5e-9 * p)
    _write_2d('temperature_melt.dat', lambda p, s: 1200.0 + 0.5 * s + 5e-9 * p)
    _write_2d('density_solid.dat', lambda p, s: np.full_like(p, 4200.0))
    _write_2d('density_melt.dat', lambda p, s: np.full_like(p, 4000.0))
    _write_2d('heat_capacity_solid.dat', lambda p, s: np.full_like(p, 1200.0))
    _write_2d('heat_capacity_melt.dat', lambda p, s: np.full_like(p, 1200.0))
    _write_2d('adiabat_temp_grad_solid.dat', lambda p, s: np.full_like(p, 1e-8))
    _write_2d('adiabat_temp_grad_melt.dat', lambda p, s: np.full_like(p, 1e-8))
    for fname, s_val in [('solidus_P-S.dat', 2400.0), ('liquidus_P-S.dat', 3000.0)]:
        lines = ['# 2 4', '# 1.0 1.0'] + [f'{p:.6e} {s_val:.6e}' for p in P]
        (eos_dir / fname).write_text('\\n'.join(lines) + '\\n')

    rheo = SolidRheologyParams(enabled=True)
    params = Parameters(
        boundary_conditions=_BoundaryConditionsParameters(
            outer_boundary_condition=1,
            outer_boundary_value=1200.0,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=255.0,
            core_heat_capacity=880.0,
            core_bc='energy_balance',
        ),
        energy=_EnergyParameters(
            conduction=True,
            convection=True,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
            solver_method='scipy',
        ),
        initial_condition=_InitialConditionParameters(
            initial_condition=1,
            surface_temperature=1500.0,
            basal_temperature=2000.0,
        ),
        mesh=_MeshParameters(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=5,
            mixing_length_profile='nearest_boundary',
            core_density=10500.0,
            eos_method=1,
        ),
        phase_solid=_PhaseParameters(
            density=4200.0,
            heat_capacity=1200.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            viscosity=1e21,
            rheology=rheo,
        ),
        phase_liquid=_PhaseParameters(
            density=4000.0,
            heat_capacity=1200.0,
            melt_fraction=1.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            viscosity=10.0,
        ),
        phase_mixed=_PhaseMixedParameters(
            latent_heat_of_fusion=4e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='none',
            liquidus='none',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1e-3,
        ),
        radionuclides=[],
        solver=_SolverParameters(start_time=0.0, end_time=1.0, atol=1e-5, rtol=1e-5),
    )
    eos_np = EntropyEOS(eos_dir)
    solver = EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()
    solver.set_initial_entropy(1800.0)
    solver.solve()
    assert solver._solution.status == 0
    print(sorted(m for m in sys.modules if m.startswith('aragog.core')))

    eos_jax = EntropyEOS_JAX(eos_dir)
    mesh = solver.evaluator.mesh
    mesh_arr = MeshArrays.from_numpy_mesh(mesh)
    r_basic = np.asarray(mesh.basic.radii).ravel()
    n_stag = np.asarray(mesh.staggered.radii).size
    r_cmb = float(r_basic[0])
    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=1200.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=5,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=(4.0 / 3.0) * np.pi * r_cmb**3 * 10500.0,
        cmb_dr_cmb=float(r_basic[1] - r_basic[0]),
    )
    phase_params = PhaseParams(rheology=rheo)
    rhs_fn, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=phase_params,
        mesh_arrays=mesh_arr,
        boundary_params=bc,
        heating_array=np.zeros(n_stag),
        scales=solver._build_nondim_scales(),
        core_bc_mode='energy_balance',
    )
    ydot_eb = np.zeros(n_stag + 1)
    rhs_fn(0.0, np.ones(n_stag + 1), ydot_eb)
    args = (eos_jax, phase_params, mesh_arr, bc, jnp.zeros(n_stag), _no_radio)
    out_qs = jax.jit(lambda s: dSdt(0.0, s, args))(jnp.full(n_stag, 1800.0))
    assert np.all(np.isfinite(ydot_eb)) and np.all(np.isfinite(np.asarray(out_qs)))
    print(sorted(m for m in sys.modules if m.startswith('aragog.core')))

import aragog.core
print('aragog.core.layer' in sys.modules)
"""


def test_energy_balance_paths_load_no_core_module_at_import_or_runtime_solve():
    """The numpy solver and the JAX RHS load no part of ``aragog.core`` at import or during a
    short rheology-enabled solve, and ``aragog.core`` loads the experimental layer only when a
    stratified budget is built."""
    src = str(Path(__file__).resolve().parents[1] / 'src')
    env = os.environ | {'PYTHONPATH': src}
    out = subprocess.run(
        [sys.executable, '-c', _SUBPROCESS_SOLVE_CODE],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ['[]', '[]', '[]', 'False'], (
        f'Unexpected aragog.core modules loaded during import or runtime solve: {out.stdout}'
    )
