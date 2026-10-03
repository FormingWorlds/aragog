"""Run abe_mixed.cfg on aragog main or branch with NumPy or JAX solver.

Saves final profiles (S, T, heat_flux), trajectory (t, y), CVODE status,
step counts (nst), and root events to a compressed npz file for reproducibility
analysis across runs and platforms.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def get_git_commit(repo_dir: Path) -> str:
    """Get git commit hash for repo."""
    try:
        out = subprocess.check_output(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], text=True
        ).strip()
        return out
    except Exception:
        return 'unknown'


def run_one(
    aragog_dir: Path,
    solver_type: str,
    config_file: Path,
    eos_dir: Path,
    end_time: float,
    output_npz: Path,
    run_id: str = '0',
) -> dict:
    """Execute one solve of abe_mixed."""
    sys.path.insert(0, str(aragog_dir / 'src'))

    try:
        from aragog.parser import Parameters
    except ImportError:
        from aragog.parser import Config as Parameters  # fallback

    try:
        from aragog.cli import _derive_initial_entropy_from_config
    except ImportError:

        def _derive_initial_entropy_from_config(s):
            return None

    try:
        import scikits.odes  # noqa: F401
    except ImportError:
        try:
            import types

            import scikits_odes_sundials

            scikits = types.ModuleType('scikits')
            scikits.odes = scikits_odes_sundials
            sys.modules['scikits'] = scikits
            sys.modules['scikits.odes'] = scikits_odes_sundials
        except ImportError:
            pass

    from aragog.eos.entropy import EntropyEOS
    from aragog.solver.entropy_solver import EntropySolver

    cfg = Parameters.from_file(config_file)
    cfg.solver.end_time = float(end_time)

    eos = EntropyEOS(str(eos_dir))
    solver = EntropySolver(cfg, entropy_eos=eos)
    solver.initialize()

    s0 = _derive_initial_entropy_from_config(solver)
    if s0 is not None:
        solver.set_initial_entropy(s0)

    if solver_type == 'jax':
        from aragog.jax.eos import EntropyEOS_JAX
        from aragog.jax.phase import MeshArrays, PhaseParams
        from aragog.jax.solver import BoundaryParams
        from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

        eos_jax = EntropyEOS_JAX(str(eos_dir))
        mesh = solver.evaluator.mesh
        mesh_arr = MeshArrays.from_numpy_mesh(mesh)

        bc_cfg = solver.parameters.boundary_conditions
        bc = BoundaryParams(
            outer_bc_type=int(bc_cfg.outer_boundary_condition),
            outer_bc_value=float(bc_cfg.outer_boundary_value),
            emissivity=float(bc_cfg.emissivity),
            T_eq=float(bc_cfg.equilibrium_temperature),
            inner_bc_type=(
                5
                if solver._core_bc == 'energy_balance'
                else int(bc_cfg.inner_boundary_condition)
            ),
            inner_bc_value=float(bc_cfg.inner_boundary_value),
            core_density=float(solver.parameters.mesh.core_density),
            core_heat_capacity=float(bc_cfg.core_heat_capacity),
            tfac_core_avg=float(getattr(bc_cfg, 'tfac_core_avg', 1.147)),
            cmb_area=float(getattr(solver, '_cmb_area', 0.0)),
            core_M=float(getattr(solver, '_core_M', 0.0)),
            cmb_dr_cmb=float(getattr(solver, '_cmb_dr_cmb', 0.0)),
            param_utbl=bool(getattr(bc_cfg, 'param_utbl', False)),
            param_utbl_const=float(getattr(bc_cfg, 'param_utbl_const', 0.0)),
        )

        ps = solver.parameters.phase_solid
        pl = solver.parameters.phase_liquid
        pm = solver.parameters.phase_mixed
        en = solver.parameters.energy

        params_jax = PhaseParams(
            phi_rheo=float(pm.rheological_transition_melt_fraction),
            phi_width=float(pm.rheological_transition_width),
            viscosity_solid=float(ps.viscosity),
            viscosity_liquid=float(pl.viscosity),
            grain_size=float(pm.grain_size),
            k_solid=float(ps.thermal_conductivity),
            k_liquid=float(pl.thermal_conductivity),
            matprop_smooth_width=float(getattr(pm, 'matprop_smooth_width', 0.0)),
            conduction=bool(en.conduction),
            convection=bool(en.convection),
            grav_sep=bool(en.gravitational_separation),
            mixing=bool(en.mixing),
            eddy_diff_thermal=float(en.eddy_diffusivity_thermal),
            eddy_diff_chemical=float(en.eddy_diffusivity_chemical),
            kappah_floor=float(en.kappah_floor),
            bottom_up_grav_sep=bool(en.bottom_up_grav_sep),
            phase_smoothing=str(en.phase_smoothing),
            phase_smoothing_width=0.01,
            separation_viscosity=str(pm.separation_viscosity),
        )

        n_stag = solver._n_stag
        heating = np.zeros(n_stag)

        def jax_factory(scales, core_bc_mode):
            rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
                eos_jax=eos_jax,
                phase_params=params_jax,
                mesh_arrays=mesh_arr,
                boundary_params=bc,
                heating_array=heating,
                scales=scales,
                core_bc_mode=core_bc_mode,
            )
            return rhs_fn, jac_fn

        solver.parameters.energy.use_jax_jacobian = True
        solver.set_jax_cvode_factory(jax_factory)
    else:
        solver.parameters.energy.use_jax_jacobian = False
        solver.set_jax_cvode_factory(None)

    t_start = time.perf_counter()
    solver.solve()
    wall_time = time.perf_counter() - t_start

    output = solver.get_state()
    sol = getattr(solver, '_solution', None)

    nst = int(getattr(sol, 'cvode_nst', getattr(output, 'cvode_nst', -1)))
    flag = int(getattr(sol, 'cvode_flag', getattr(output, 'cvode_flag', -1)))
    status = int(output.status)
    t_steps = np.asarray(getattr(sol, 't', np.array([])), dtype=float)
    y_traj = np.asarray(getattr(sol, 'y', np.array([])), dtype=float)
    roots = getattr(sol, 'roots', None)
    roots_t = (
        np.asarray(getattr(roots, 't', np.array([])), dtype=float)
        if roots is not None
        else np.array([])
    )

    commit = get_git_commit(aragog_dir)

    data = {
        'S': output.S_final,
        'T': output.T_stag,
        'heat_flux': output.heat_flux,
        'cvode_status': status,
        'cvode_flag': flag,
        'cvode_nst': nst,
        'output_step_times': t_steps,
        'y_trajectory': y_traj,
        'roots_t': roots_t,
        'commit': commit,
        'solver': solver_type,
        'run_id': str(run_id),
        'wall_time_s': wall_time,
    }

    # Include all other float/array diagnostics from output
    for k, v in output.__dict__.items():
        if k not in data and isinstance(v, (np.ndarray, float, int, str, bool, np.generic)):
            data[k] = v

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **data)
    print(
        f'Run {run_id} ({solver_type}) completed in {wall_time:.2f}s: status={status}, nst={nst}, wrote {output_npz}'
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description='Run abe_mixed.cfg for reproducibility check')
    parser.add_argument(
        '--aragog-dir', type=Path, required=True, help='Path to aragog repo root'
    )
    parser.add_argument(
        '--solver', choices=['numpy', 'jax'], default='numpy', help='Solver type (numpy or jax)'
    )
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--eos-dir', type=Path, default=None, help='Path to EOS directory')
    parser.add_argument('--end-time', type=float, default=100.0, help='End time in years')
    parser.add_argument(
        '--output-npz', type=Path, required=True, help='Path to write output npz'
    )
    parser.add_argument('--run-id', type=str, default='0', help='Run repetition identifier')
    args = parser.parse_args()

    cfg_file = args.config
    if cfg_file is None:
        cfg_file = args.aragog_dir / 'src' / 'aragog' / 'cfg' / 'abe_mixed.cfg'

    eos_dir = args.eos_dir
    if eos_dir is None:
        env_eos = os.environ.get('ARAGOG_TEST_EOS_DIR')
        if env_eos and Path(env_eos).exists():
            eos_dir = Path(env_eos)
        else:
            candidates = [
                Path('/Users/timlichtenberg/work/ssc-verify-task6/test-data/spider_eos'),
                Path('tests/test_data/spider_eos'),
            ]
            eos_dir = next((c for c in candidates if c.exists()), None)
            if eos_dir is None:
                raise FileNotFoundError('No valid EOS directory found')

    run_one(
        aragog_dir=args.aragog_dir.resolve(),
        solver_type=args.solver,
        config_file=cfg_file.resolve(),
        eos_dir=eos_dir.resolve(),
        end_time=args.end_time,
        output_npz=args.output_npz.resolve(),
        run_id=args.run_id,
    )


if __name__ == '__main__':
    main()
