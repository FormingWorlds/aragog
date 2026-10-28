#!/usr/bin/env python3
"""Generate bitwise reference outputs for rheology-off Aragog comparisons.

This script executes 42 quasi-steady RHS evaluations, 6 energy balance RHS
evaluations, and 6 short 100-year solve trajectories on a main branch tree.
It saves all 54 resulting arrays plus the source commit SHA to a compressed
.npz reference file for regression verification.
"""

from __future__ import annotations

import argparse
import importlib.resources
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


def _resolve_commit_sha(main_tree: Path, explicit_sha: str | None) -> str:
    """Resolve the git commit SHA of the reference main tree.

    Parameters
    ----------
    main_tree : Path
        Root of the aragog git tree at the main commit to record.
    explicit_sha : str or None
        SHA to record instead, for a tree that is not a git checkout.

    Returns
    -------
    str
        Full 40-character commit SHA.

    Raises
    ------
    ValueError
        If ``main_tree`` is not the root of a git tree and no SHA is given,
        or if the tree has uncommitted changes or untracked files.
    """
    if explicit_sha:
        return explicit_sha.strip()
    res = subprocess.run(
        ['git', '-C', str(main_tree), 'rev-parse', '--show-toplevel', 'HEAD'],
        capture_output=True,
        text=True,
    )
    out = res.stdout.split()
    if res.returncode != 0 or len(out) != 2 or Path(out[0]).resolve() != main_tree.resolve():
        raise ValueError(f'{main_tree} is not the root of a git tree; pass --commit-sha')
    status = subprocess.run(
        ['git', '-C', str(main_tree), 'status', '--porcelain'],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise ValueError(
            f'{main_tree} has uncommitted changes; the SHA would not match the arrays'
        )
    return out[1]


def generate_reference(
    main_tree: Path,
    eos_dir: Path,
    output_path: Path,
    commit_sha: str,
) -> None:
    """Compute and store reference arrays and source commit SHA.

    Parameters
    ----------
    main_tree : Path
        Directory containing aragog tree for main branch.
    eos_dir : Path
        Directory containing EOS reference tables.
    output_path : Path
        Destination .npz file path.
    commit_sha : str
        Commit SHA corresponding to the main tree.

    Raises
    ------
    RuntimeError
        If the imported ``aragog`` package does not lie inside ``main_tree``.
    """
    sys.path.insert(0, str(main_tree / 'src'))
    sys.path.insert(0, str(main_tree))
    os.environ['ARAGOG_TEST_EOS_DIR'] = str(eos_dir)

    import aragog

    aragog_file = Path(aragog.__file__).resolve()
    print(f'aragog imported from {aragog_file}')
    if main_tree.resolve() not in aragog_file.parents:
        raise RuntimeError(f'aragog at {aragog_file} is not inside the main tree {main_tree}')

    import jax
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)

    from aragog.config import Config
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import (
        BoundaryParams,
        _no_radio,
        dSdt,
        dSdt_energy_balance,
    )
    from aragog.parser import Parameters
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    from aragog.solver.entropy_solver import EntropySolver
    from tests.conftest import entropy_eos_copy, entropy_eos_jax

    res: dict[str, np.ndarray] = {
        'source_commit': np.array(commit_sha),
    }

    # 1. 42 quasi-steady RHS arrays across 7 energy configurations
    configs = {
        'default': {},
        'floor0': {'kappah_floor': 0.0},
        'gravsep': {'gravitational_separation': True},
        'mixing': {'mixing': True},
        'gravmix': {'gravitational_separation': True, 'mixing': True},
        'nocond': {'conduction': False},
        'noconv': {'convection': False},
    }

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        raw_params = Parameters.from_file(p)

    for cfg_name, en in configs.items():
        params = replace(
            raw_params,
            mesh=replace(raw_params.mesh, number_of_nodes=20),
            boundary_conditions=replace(
                raw_params.boundary_conditions,
                core_bc='quasi_steady',
                inner_boundary_condition=1,
            ),
            energy=replace(raw_params.energy, **en),
        )
        solver = EntropySolver(
            params, entropy_eos=entropy_eos_copy(eos_dir, identity_alpha=True)
        )
        solver.initialize()
        eos = solver.state.phase_basic._eos
        ps = np.asarray(solver.state.phase_staggered.pressure, float)
        ssol = np.asarray(eos.solidus_entropy(ps), float)
        sliq = np.asarray(eos.liquidus_entropy(ps), float)
        n = solver._n_stag
        profs = {
            'solid': np.linspace(ssol[0] - 65.0, ssol[-1] - 120.0, n),
            'mushy': ssol + (sliq - ssol) * np.linspace(0.1, 0.9, n),
            'melt': sliq + np.linspace(50.0, 300.0, n),
        }
        mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
        pj = PhaseParams(
            phi_rheo=float(params.phase_mixed.rheological_transition_melt_fraction),
            phi_width=float(params.phase_mixed.rheological_transition_width),
            viscosity_solid=float(params.phase_solid.viscosity),
            viscosity_liquid=float(params.phase_liquid.viscosity),
            grain_size=float(params.phase_mixed.grain_size),
            k_solid=float(params.phase_solid.thermal_conductivity),
            k_liquid=float(params.phase_liquid.thermal_conductivity),
            matprop_smooth_width=float(params.phase_mixed.matprop_smooth_width),
            conduction=params.energy.conduction,
            convection=params.energy.convection,
            grav_sep=params.energy.gravitational_separation,
            mixing=params.energy.mixing,
            eddy_diff_thermal=float(params.energy.eddy_diffusivity_thermal),
            eddy_diff_chemical=float(params.energy.eddy_diffusivity_chemical),
            kappah_floor=float(params.energy.kappah_floor),
            bottom_up_grav_sep=params.energy.bottom_up_grav_sep,
            phase_smoothing=params.energy.phase_smoothing,
            separation_viscosity=params.phase_mixed.separation_viscosity,
        )
        r_cmb = float(mesh_jax.radii_basic[0])
        r_next = float(mesh_jax.radii_basic[1])
        bc = BoundaryParams(
            outer_bc_type=int(params.boundary_conditions.outer_boundary_condition),
            outer_bc_value=float(params.boundary_conditions.outer_boundary_value),
            emissivity=float(params.boundary_conditions.emissivity),
            T_eq=float(params.boundary_conditions.equilibrium_temperature),
            inner_bc_type=int(params.boundary_conditions.inner_boundary_condition),
            inner_bc_value=float(params.boundary_conditions.inner_boundary_value),
            core_density=float(params.mesh.core_density),
            core_heat_capacity=float(params.boundary_conditions.core_heat_capacity),
            tfac_core_avg=float(solver._core_tfac),
            cmb_area=4.0 * np.pi * r_cmb**2,
            core_M=float(solver._core_cap / params.boundary_conditions.core_heat_capacity),
            cmb_dr_cmb=r_next - r_cmb,
            param_utbl=bool(params.boundary_conditions.param_utbl),
            param_utbl_const=float(params.boundary_conditions.param_utbl_const),
        )
        args_jax = (
            entropy_eos_jax(eos_dir),
            pj,
            mesh_jax,
            bc,
            jnp.zeros(solver._n_stag, dtype=jnp.float64),
            _no_radio,
        )
        for prof_name, s_profile in profs.items():
            res[f'rhs/{cfg_name}/{prof_name}/np'] = np.asarray(solver.dSdt(0.0, s_profile))
            res[f'rhs/{cfg_name}/{prof_name}/jax'] = np.asarray(
                dSdt(0.0, jnp.asarray(s_profile), args_jax)
            )

    # 2. 6 energy balance RHS arrays
    params_eb = replace(
        raw_params,
        mesh=replace(raw_params.mesh, number_of_nodes=20),
        boundary_conditions=replace(
            raw_params.boundary_conditions,
            core_bc='energy_balance',
            inner_boundary_condition=1,
        ),
    )
    solver_eb = EntropySolver(
        params_eb, entropy_eos=entropy_eos_copy(eos_dir, identity_alpha=True)
    )
    solver_eb.initialize()
    eos_eb = solver_eb.state.phase_basic._eos
    ps_eb = np.asarray(solver_eb.state.phase_staggered.pressure, float)
    ssol_eb = np.asarray(eos_eb.solidus_entropy(ps_eb), float)
    sliq_eb = np.asarray(eos_eb.liquidus_entropy(ps_eb), float)
    n_eb = solver_eb._n_stag
    profs_eb = {
        'solid': np.concatenate(
            [np.linspace(ssol_eb[0] - 65.0, ssol_eb[-1] - 120.0, n_eb), [-1e-4]]
        ),
        'mushy': np.concatenate(
            [ssol_eb + (sliq_eb - ssol_eb) * np.linspace(0.1, 0.9, n_eb), [-1e-4]]
        ),
        'melt': np.concatenate([sliq_eb + np.linspace(50.0, 300.0, n_eb), [-1e-4]]),
    }
    mesh_jax_eb = MeshArrays.from_numpy_mesh(solver_eb.evaluator.mesh)
    pj_eb = PhaseParams(
        phi_rheo=float(params_eb.phase_mixed.rheological_transition_melt_fraction),
        phi_width=float(params_eb.phase_mixed.rheological_transition_width),
        viscosity_solid=float(params_eb.phase_solid.viscosity),
        viscosity_liquid=float(params_eb.phase_liquid.viscosity),
        grain_size=float(params_eb.phase_mixed.grain_size),
        k_solid=float(params_eb.phase_solid.thermal_conductivity),
        k_liquid=float(params_eb.phase_liquid.thermal_conductivity),
        matprop_smooth_width=float(params_eb.phase_mixed.matprop_smooth_width),
        conduction=params_eb.energy.conduction,
        convection=params_eb.energy.convection,
        grav_sep=params_eb.energy.gravitational_separation,
        mixing=params_eb.energy.mixing,
        eddy_diff_thermal=float(params_eb.energy.eddy_diffusivity_thermal),
        eddy_diff_chemical=float(params_eb.energy.eddy_diffusivity_chemical),
        kappah_floor=float(params_eb.energy.kappah_floor),
        bottom_up_grav_sep=params_eb.energy.bottom_up_grav_sep,
        phase_smoothing=params_eb.energy.phase_smoothing,
        separation_viscosity=params_eb.phase_mixed.separation_viscosity,
    )
    r_cmb_eb = float(mesh_jax_eb.radii_basic[0])
    r_next_eb = float(mesh_jax_eb.radii_basic[1])
    bc_eb = BoundaryParams(
        outer_bc_type=int(params_eb.boundary_conditions.outer_boundary_condition),
        outer_bc_value=float(params_eb.boundary_conditions.outer_boundary_value),
        emissivity=float(params_eb.boundary_conditions.emissivity),
        T_eq=float(params_eb.boundary_conditions.equilibrium_temperature),
        inner_bc_type=int(params_eb.boundary_conditions.inner_boundary_condition),
        inner_bc_value=float(params_eb.boundary_conditions.inner_boundary_value),
        core_density=float(params_eb.mesh.core_density),
        core_heat_capacity=float(params_eb.boundary_conditions.core_heat_capacity),
        tfac_core_avg=float(solver_eb._core_tfac),
        cmb_area=4.0 * np.pi * r_cmb_eb**2,
        core_M=float(solver_eb._core_cap / params_eb.boundary_conditions.core_heat_capacity),
        cmb_dr_cmb=r_next_eb - r_cmb_eb,
        param_utbl=bool(params_eb.boundary_conditions.param_utbl),
        param_utbl_const=float(params_eb.boundary_conditions.param_utbl_const),
    )
    args_eb = (
        entropy_eos_jax(eos_dir),
        pj_eb,
        mesh_jax_eb,
        bc_eb,
        jnp.zeros(solver_eb._n_stag, dtype=jnp.float64),
        _no_radio,
    )
    for name, y in profs_eb.items():
        res[f'rhs/energy_balance/{name}/np'] = np.asarray(solver_eb.dSdt(0.0, y))
        res[f'rhs/energy_balance/{name}/jax'] = np.asarray(
            dSdt_energy_balance(0.0, jnp.asarray(y), args_eb)
        )

    # 3. 6 short solve() trajectory arrays (100 yr, 20 nodes)
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        c_solve = Config.from_file(str(p))
    c_solve.mesh.number_of_nodes = 20
    c_solve.solver.end_time = 100.0
    c_solve.solver.cvode_max_steps = 100
    eos_inst = EntropyEOS(str(eos_dir))

    # NumPy solve
    s_np = EntropySolver(c_solve, entropy_eos=eos_inst)
    s_np.initialize()
    s_np.set_initial_entropy(2500.0)
    s_np.solve()
    out_np = s_np.get_state()
    res['solve/np/S_final'] = np.asarray(out_np.S_final)
    res['solve/np/T_stag'] = np.asarray(out_np.T_stag)
    res['solve/np/heat_flux'] = np.asarray(out_np.heat_flux)

    # JAX solve
    s_jax = EntropySolver(c_solve, entropy_eos=eos_inst)
    eos_jx_inst = entropy_eos_jax(str(eos_dir))
    s_jax.set_jax_cvode_factory(lambda slv: build_jax_rhs_and_jacobian(slv, eos_jx_inst))
    s_jax.initialize()
    s_jax.set_initial_entropy(2500.0)
    s_jax.solve()
    out_jax = s_jax.get_state()
    res['solve/jax/S_final'] = np.asarray(out_jax.S_final)
    res['solve/jax/T_stag'] = np.asarray(out_jax.T_stag)
    res['solve/jax/heat_flux'] = np.asarray(out_jax.heat_flux)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **res)
    print(f'Saved {len(res) - 1} arrays and commit SHA to {output_path}')


def main() -> int:
    """CLI entry point for generating rheology-off main reference.

    Returns
    -------
    int
        Exit status code.
    """
    parser = argparse.ArgumentParser(
        description='Generate rheology-off reference outputs from a main tree.'
    )
    parser.add_argument(
        '--main-tree',
        type=Path,
        required=True,
        help='Path to aragog source tree at main branch.',
    )
    parser.add_argument(
        '--commit-sha',
        type=str,
        default=None,
        help='Explicit source commit SHA string.',
    )
    parser.add_argument(
        '--eos-dir',
        type=Path,
        default=None,
        help='Directory containing EOS tables (default: $ARAGOG_TEST_EOS_DIR).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('tests/reference/rheology_off_main_reference.npz'),
        help='Destination path for .npz file.',
    )
    args = parser.parse_args()

    env_eos = os.environ.get('ARAGOG_TEST_EOS_DIR', '').strip()
    eos_dir = args.eos_dir or (Path(env_eos) if env_eos else None)
    if eos_dir is None:
        print('Error: pass --eos-dir or set ARAGOG_TEST_EOS_DIR', file=sys.stderr)
        return 1
    if not eos_dir.exists():
        print(f'Error: EOS directory does not exist: {eos_dir}', file=sys.stderr)
        return 1

    try:
        commit_sha = _resolve_commit_sha(args.main_tree, args.commit_sha)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    print(f'Generating reference using commit {commit_sha}...')
    try:
        generate_reference(args.main_tree, eos_dir, args.output, commit_sha)
    except RuntimeError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
