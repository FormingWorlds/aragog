#!/usr/bin/env python3
"""Generate bitwise reference outputs for rheology-off Aragog comparisons.

This script executes 42 quasi-steady RHS evaluations, 6 energy balance RHS
evaluations, and 6 short 100-year solve trajectories on a main branch tree.
It saves all 54 resulting arrays, the source commit SHA and the flag
``solve/jax/option_z_active`` to a compressed .npz reference file for regression
verification. The JAX solve stops with an error when solve() does not run the JAX RHS.
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


def _phase_params(params):
    """Build ``PhaseParams`` with the constructor of the imported (main) aragog.

    Parameters
    ----------
    params : Parameters or Config
        Run parameters of the solver.

    Returns
    -------
    PhaseParams
        Phase parameters; fields not passed keep the defaults of the main constructor.
    """
    from aragog.jax.phase import PhaseParams

    return PhaseParams(
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


def _boundary_params(params, solver, mesh_jax):
    """Build ``BoundaryParams`` for ``solver`` with the imported (main) aragog.

    Parameters
    ----------
    params : Parameters or Config
        Run parameters of the solver.
    solver : EntropySolver
        Initialized solver.
    mesh_jax : MeshArrays
        JAX mesh of ``solver``.

    Returns
    -------
    BoundaryParams
        Boundary parameters of the JAX RHS.
    """
    from aragog.jax.solver import BoundaryParams

    bcs = params.boundary_conditions
    r_cmb = float(mesh_jax.radii_basic[0])
    r_next = float(mesh_jax.radii_basic[1])
    return BoundaryParams(
        outer_bc_type=int(bcs.outer_boundary_condition),
        outer_bc_value=float(bcs.outer_boundary_value),
        emissivity=float(bcs.emissivity),
        T_eq=float(bcs.equilibrium_temperature),
        inner_bc_type=int(bcs.inner_boundary_condition),
        inner_bc_value=float(bcs.inner_boundary_value),
        core_density=float(params.mesh.core_density),
        core_heat_capacity=float(bcs.core_heat_capacity),
        tfac_core_avg=float(solver._core_tfac),
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=float(solver._core_cap / bcs.core_heat_capacity),
        cmb_dr_cmb=r_next - r_cmb,
        param_utbl=bool(bcs.param_utbl),
        param_utbl_const=float(bcs.param_utbl_const),
    )


class _RecordedFactory:
    """A 2-argument JAX CVODE factory that records whether ``solve()`` used it.

    ``EntropySolver.solve()`` catches any error of the factory and continues with the NumPy
    RHS and a finite-difference Jacobian, so a JAX solve needs this record to show its path.

    Parameters
    ----------
    build : callable
        ``build(scales, core_bc_mode) -> (rhs_fn, jac_fn)``.
    """

    def __init__(self, build):
        self._build = build
        self.built = 0
        self.failed = 0

    def __call__(self, scales, core_bc_mode):
        """Build the callbacks and count the result."""
        try:
            out = self._build(scales, core_bc_mode)
        except Exception:
            self.failed += 1
            raise
        self.built += 1
        return out

    def require_option_z(self) -> None:
        """Raise unless every factory call built the JAX callbacks.

        Raises
        ------
        RuntimeError
            If the factory was never called or a call failed.
        """
        if self.built == 0 or self.failed:
            raise RuntimeError(
                f'the JAX solve did not run on the JAX RHS (factory built {self.built}, '
                f'failed {self.failed}); solve() used the NumPy RHS'
            )


def _solve_factory(solver, eos_jax) -> _RecordedFactory:
    """Return the recorded JAX CVODE factory for ``solver`` with the main aragog API.

    Parameters
    ----------
    solver : EntropySolver
        Initialized solver of the JAX solve.
    eos_jax : EntropyEOS_JAX
        JAX EOS tables.

    Returns
    -------
    _RecordedFactory
        Factory to pass to ``set_jax_cvode_factory``.
    """
    from aragog.jax.phase import MeshArrays
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    def build(scales, core_bc_mode):
        params = solver.parameters
        mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=_phase_params(params),
            mesh_arrays=mesh_jax,
            boundary_params=_boundary_params(params, solver, mesh_jax),
            heating_array=np.zeros(solver._n_stag),
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs_fn, jac_fn

    return _RecordedFactory(build)


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
        or if the JAX solve does not run on the JAX RHS.
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
    from aragog.jax.phase import MeshArrays
    from aragog.jax.solver import (
        _no_radio,
        dSdt,
        dSdt_energy_balance,
    )
    from aragog.parser import Parameters
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
        pj = _phase_params(params)
        bc = _boundary_params(params, solver, mesh_jax)
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
    pj_eb = _phase_params(params_eb)
    bc_eb = _boundary_params(params_eb, solver_eb, mesh_jax_eb)
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
    s_jax.initialize()
    s_jax.set_initial_entropy(2500.0)
    factory = _solve_factory(s_jax, entropy_eos_jax(str(eos_dir)))
    s_jax.set_jax_cvode_factory(factory)
    s_jax.solve()
    factory.require_option_z()
    out_jax = s_jax.get_state()
    res['solve/jax/S_final'] = np.asarray(out_jax.S_final)
    res['solve/jax/T_stag'] = np.asarray(out_jax.T_stag)
    res['solve/jax/heat_flux'] = np.asarray(out_jax.heat_flux)
    res['solve/jax/option_z_active'] = np.array(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **res)
    print(f'Saved {len(res) - 2} arrays, the option Z flag and commit SHA to {output_path}')


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
