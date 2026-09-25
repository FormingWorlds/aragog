"""Acceptance verification run for solid-state convection to 4.5 Gyr.

Executes the long-term Earth-like cooling run of
tools/verification/configs/ssc_earth_4p5gyr.toml to 4.5 Gyr with:
- 100 mesh nodes
- Radiogenic heating enabled
- CVODE with the JAX right-hand side and analytic Jacobian
- Stagnant lid stress closure and diagnostics

Verifies:
1. Integration completes to 4.5 Gyr.
2. Solidification time (Phi_global < 0.05).
3. Lid regime is 0 (no lid) while liquid exists at the surface.
4. Discrete energy residual relative to surface flux is < 1e-3 at every output.
5. Generates 4-panel diagnostic plot saved as vector PDF and PNG.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure aragog and local modules are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'src'))
if str(_REPO_ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'tools'))

import jax.numpy as jnp  # noqa: E402
from verification.figures._style import apply_rc, panel_label, save  # noqa: E402

from aragog.cli import _derive_initial_entropy_from_config  # noqa: E402
from aragog.jax.eos import EntropyEOS_JAX  # noqa: E402
from aragog.jax.phase import MeshArrays, PhaseParams  # noqa: E402
from aragog.jax.solver import BoundaryParams  # noqa: E402
from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Checkpoint times [yr]: dense during magma ocean solidification, sparser in secular cooling.
CHECKPOINTS_YR = (
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
    300000.0,
    1.0e6,
    3.0e6,
    1.0e7,
    3.0e7,
    1.0e8,
    3.0e8,
    1.0e9,
    1.5e9,
    2.0e9,
    2.5e9,
    3.0e9,
    3.5e9,
    4.0e9,
    4.5e9,
)


def cvode_counts(sol) -> tuple[int, int, int, float]:
    """Return CVODE's steps, error-test failures, Jacobian setups and last step [yr].

    ``sol.t`` is the output grid, not the CVODE step sequence, so the counts come from
    the ``cvode_info`` that ``EntropySolver`` attaches to a CVODE solution.
    """
    info = sol['cvode_info']
    return (
        int(info['NumSteps']),
        int(info['NumErrTestFails']),
        int(info['NumLinSolvSetups']),
        float(sol['cvode_last_step']),
    )


def run_acceptance(
    config_path: Path,
    eos_dir: Path,
    output_dir: Path,
    resume: bool = False,
    checkpoints: tuple[float, ...] = CHECKPOINTS_YR,
) -> dict:
    """Run 4.5 Gyr acceptance integration with JAX CVODE factory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'acceptance_run.log'

    file_mode = 'a' if resume and (output_dir / 'acceptance_checkpoint.npz').is_file() else 'w'
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info('Starting solid-state convection acceptance run: %s', config_path)
    logger.info('Using EOS directory: %s', eos_dir)
    logger.info('Outputs will be saved to: %s', output_dir)

    solver = EntropySolver.from_file(str(config_path), eos_dir=eos_dir)
    solver.parameters.energy.use_jax_jacobian = True
    solver.initialize()

    S0 = _derive_initial_entropy_from_config(solver)
    solver.set_initial_entropy(S0)

    eos_jax = EntropyEOS_JAX(eos_dir)
    mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
    n_stag = solver._n_stag
    p = solver.parameters

    phase_params = PhaseParams(
        phi_rheo=p.phase_mixed.rheological_transition_melt_fraction,
        phi_width=p.phase_mixed.rheological_transition_width,
        viscosity_solid=p.phase_solid.viscosity,
        viscosity_liquid=p.phase_liquid.viscosity,
        grain_size=p.phase_mixed.grain_size,
        k_solid=p.phase_solid.thermal_conductivity,
        k_liquid=p.phase_liquid.thermal_conductivity,
        matprop_smooth_width=p.phase_mixed.matprop_smooth_width,
        conduction=p.energy.conduction,
        convection=p.energy.convection,
        grav_sep=p.energy.gravitational_separation,
        mixing=p.energy.mixing,
        kappah_floor=p.energy.kappah_floor,
        rheology=p.phase_solid.rheology,
    )

    _cached_cvode = None

    def cvode_factory(scales, core_bc_mode):
        nonlocal _cached_cvode
        if _cached_cvode is not None:
            return _cached_cvode

        bc_cfg = solver.parameters.boundary_conditions
        bc_jax = BoundaryParams(
            outer_bc_type=bc_cfg.outer_boundary_condition,
            outer_bc_value=bc_cfg.outer_boundary_value,
            emissivity=bc_cfg.emissivity,
            T_eq=bc_cfg.equilibrium_temperature,
            inner_bc_type=(
                5 if core_bc_mode == 'energy_balance' else bc_cfg.inner_boundary_condition
            ),
            inner_bc_value=bc_cfg.inner_boundary_value,
            core_density=solver.parameters.mesh.core_density,
            core_heat_capacity=bc_cfg.core_heat_capacity,
            tfac_core_avg=bc_cfg.tfac_core_avg,
            cmb_area=float(solver._cmb_area),
            core_M=float(solver._core_M),
            cmb_dr_cmb=float(solver._cmb_dr_cmb),
            param_utbl=bool(bc_cfg.param_utbl),
            param_utbl_const=float(bc_cfg.param_utbl_const),
        )
        heating_static = jnp.zeros(n_stag)
        radionuclides = solver.parameters.radionuclides
        if radionuclides:
            radio_isotope_params = (
                np.array([float(r.heat_production) for r in radionuclides]),
                np.array([float(r.abundance) for r in radionuclides]),
                np.array([float(r.concentration) for r in radionuclides]),
                np.array([float(r.t0_years) for r in radionuclides]),
                np.array([float(r.half_life_years) for r in radionuclides]),
            )
        else:
            radio_isotope_params = ()

        rhs_fn, jac_fn, _info = build_jax_rhs_and_jacobian(
            eos_jax,
            phase_params,
            mesh_jax,
            bc_jax,
            heating_static,
            scales,
            core_bc_mode=core_bc_mode,
            radio_isotope_params=radio_isotope_params,
        )
        _cached_cvode = (rhs_fn, jac_fn)
        return _cached_cvode

    solver.set_jax_cvode_factory(cvode_factory)

    r_basic = np.asarray(solver.evaluator.mesh.basic.radii).ravel()
    r_surf = r_basic[-1]
    a_surf = 4.0 * np.pi * r_surf**2

    # Storage arrays
    results = {
        'time_yr': [],
        'phi_global': [],
        'phi_surf': [],
        'T_interior': [],
        'T_pot': [],
        'T_core': [],
        'F_surf': [],
        'F_cmb': [],
        'lid_thickness': [],
        'lid_base_temperature': [],
        'lid_stress': [],
        'theta': [],
        'lid_regime': [],
        'energy_residual': [],
        'relative_residual': [],
        'cvode_steps': [],
        'cvode_err_test_fails': [],
        'cvode_jac_setups': [],
        'cvode_last_step_min': [],
    }

    ckpt_path = output_dir / 'acceptance_checkpoint.npz'
    t_start_wall = time.perf_counter()
    t_cur = 0.0
    total_cvode_steps = 0
    global_last_step_min = np.inf
    solidification_time_yr = None
    start_idx = 0
    st = None

    if resume and ckpt_path.is_file():
        logger.info('Found existing checkpoint at %s, resuming...', ckpt_path)
        ckpt_data = np.load(ckpt_path)
        missing = [k for k in (*results, 'global_last_step_min') if k not in ckpt_data]
        if missing:
            raise ValueError(f'{ckpt_path} lacks {missing}; start the run without resume')
        for k in results:
            results[k] = list(ckpt_data[k])
        t_cur = float(ckpt_data['t_cur'])
        solver.set_initial_entropy(ckpt_data['S_current'])
        total_cvode_steps = int(ckpt_data['total_cvode_steps'])
        global_last_step_min = float(ckpt_data['global_last_step_min'])
        sol_time = float(ckpt_data['solidification_time_yr'])
        solidification_time_yr = sol_time if sol_time > 0.0 else None
        start_idx = int(ckpt_data['checkpoint_idx']) + 1
        logger.info(
            'Resumed from checkpoint %d: t=%.2e yr, Phi_glob=%.4f',
            start_idx - 1,
            t_cur,
            results['phi_global'][-1] if results['phi_global'] else -1.0,
        )

    print(
        f'{"Idx":>3} {"Time [yr]":>11} {"Phi_glob":>9} {"T_int [K]":>10} {"d_lid [km]":>11} '
        f'{"Regime":>7} {"F_surf [W/m2]":>14} {"Rel Resid":>11} {"Steps":>7} {"h_last_min [yr]":>15}',
        flush=True,
    )
    print('-' * 105, flush=True)

    for i, t_target in enumerate(checkpoints):
        if i < start_idx or t_target <= t_cur + 1.0e-6 * max(t_cur, 1.0):
            continue

        interval_steps = interval_netf = interval_setups = 0
        interval_last_step_min = np.inf

        while t_cur < (t_target - 1.0e-6 * max(t_target, 1.0)):
            dt_rem = t_target - t_cur
            # Adaptive sub-step sizing:
            # During magma ocean and mushy layer crystallization (Phi_global >= 0.01),
            # take sub-steps of at most 100 yr so CVODE resolves front crossings cleanly
            # without multiple coupled phase boundary transitions in one solve call.
            # Once mantle is solid (Phi_global < 0.01), take macro steps.
            if st is not None and st.Phi_global < 0.01:
                if t_cur < 1.0e8:
                    dt_sub = min(dt_rem, 1.0e6)
                else:
                    dt_sub = min(dt_rem, 1.0e8)
            else:
                dt_sub = min(dt_rem, 100.0)

            t_sub_target = t_cur + dt_sub
            solver.parameters.solver.start_time = t_cur
            solver.parameters.solver.end_time = t_sub_target

            solver.solve()
            sol = solver._solution

            t_actual = float(sol.t[-1])
            if sol.status not in (0, 1):
                if t_actual > t_cur:
                    logger.warning(
                        'CVODE returned status %d at t=%.3e yr (sub-target %.3e yr): %s. Accepting progress.',
                        sol.status,
                        t_actual,
                        t_sub_target,
                        sol.message,
                    )
                else:
                    raise RuntimeError(
                        f'CVODE integration failed at t={t_cur:e} yr with status {sol.status}: {sol.message}'
                    )

            st = solver.get_state()

            nst, netf, nsetups, h_last = cvode_counts(sol)
            interval_steps += nst
            interval_netf += netf
            interval_setups += nsetups
            interval_last_step_min = min(interval_last_step_min, h_last)

            t_cur = t_actual
            solver.set_initial_entropy(st.S_final)

        phi_glob = float(st.Phi_global)
        phi_surf = float(st.phi_stag[-1])
        t_int = float(st.interior_temperature)
        t_pot = float(st.T_magma)
        t_core = float(st.T_core)
        f_surf = float(st.heat_flux[-1])
        f_cmb = float(st.heat_flux[0])
        d_lid = float(st.lid_thickness)
        t_lid_base = float(st.lid_base_temperature)
        tau_d = float(st.lid_stress)
        theta_val = float(st.theta)
        regime = float(st.lid_regime)

        # Residual calculation
        e_resid = float(st.energy_residual)
        denom = max(abs(f_surf) * a_surf, 1.0)
        rel_resid = abs(e_resid) / denom

        total_cvode_steps += interval_steps
        global_last_step_min = min(global_last_step_min, interval_last_step_min)

        if solidification_time_yr is None and phi_glob < 0.05:
            solidification_time_yr = t_cur

        results['time_yr'].append(t_cur)
        results['phi_global'].append(phi_glob)
        results['phi_surf'].append(phi_surf)
        results['T_interior'].append(t_int)
        results['T_pot'].append(t_pot)
        results['T_core'].append(t_core)
        results['F_surf'].append(f_surf)
        results['F_cmb'].append(f_cmb)
        results['lid_thickness'].append(d_lid)
        results['lid_base_temperature'].append(t_lid_base)
        results['lid_stress'].append(tau_d)
        results['theta'].append(theta_val)
        results['lid_regime'].append(regime)
        results['energy_residual'].append(e_resid)
        results['relative_residual'].append(rel_resid)
        results['cvode_steps'].append(interval_steps)
        results['cvode_err_test_fails'].append(interval_netf)
        results['cvode_jac_setups'].append(interval_setups)
        results['cvode_last_step_min'].append(interval_last_step_min)

        print(
            f'{i:3d} {t_cur:11.2e} {phi_glob:9.4f} {t_int:10.1f} {d_lid / 1e3:11.2f} '
            f'{regime:7.0f} {f_surf:14.3e} {rel_resid:11.2e} {interval_steps:7d} {interval_last_step_min:15.3e}',
            flush=True,
        )

        logger.info(
            'Checkpoint %d: t=%.2e yr, Phi_glob=%.4f, T_int=%.1f K, d_lid=%.1f km, '
            'regime=%.0f, F_surf=%.2e W/m2, rel_resid=%.2e, CVODE steps=%d, '
            'error-test fails=%d, Jacobian setups=%d, min last step=%.2e yr',
            i,
            t_cur,
            phi_glob,
            t_int,
            d_lid / 1e3,
            regime,
            f_surf,
            rel_resid,
            interval_steps,
            interval_netf,
            interval_setups,
            interval_last_step_min,
        )

        # Save intermediate checkpoint
        save_dict = {k: np.asarray(v) for k, v in results.items()}
        save_dict['S_current'] = st.S_final
        save_dict['t_cur'] = t_cur
        save_dict['total_cvode_steps'] = total_cvode_steps
        save_dict['global_last_step_min'] = global_last_step_min
        save_dict['solidification_time_yr'] = (
            solidification_time_yr if solidification_time_yr is not None else -1.0
        )
        save_dict['checkpoint_idx'] = i
        np.savez(ckpt_path, **save_dict)
        logger.debug('Saved checkpoint to %s (checkpoint %d, t=%.2e yr)', ckpt_path, i, t_cur)

    elapsed_wall = time.perf_counter() - t_start_wall
    logger.info('Integration completed in %.1f s (%.2f min)', elapsed_wall, elapsed_wall / 60.0)

    # Convert results to arrays
    for k in results:
        results[k] = np.asarray(results[k])

    results['total_cvode_steps'] = total_cvode_steps
    results['global_last_step_min'] = global_last_step_min
    results['solidification_time_yr'] = (
        solidification_time_yr if solidification_time_yr is not None else -1.0
    )
    results['elapsed_wall_s'] = elapsed_wall

    # Save raw data
    npz_path = output_dir / 'acceptance_data.npz'
    np.savez(npz_path, **results)
    logger.info('Saved raw acceptance run data to %s', npz_path)

    # Save final NetCDF snapshot
    nc_path = output_dir / 'ssc_earth_4p5gyr.nc'
    st.to_netcdf(str(nc_path), description='4.5 Gyr acceptance run snapshot')
    logger.info('Saved final snapshot to %s', nc_path)

    # Plot results
    plot_path = output_dir / 'acceptance_run'
    plot_acceptance(results, plot_path)
    logger.info('Saved acceptance diagnostic plots to %s.{pdf,png}', plot_path)

    return results


def plot_acceptance(data: dict, out_base_path: Path) -> None:
    """Generate publication-quality diagnostic figure for acceptance run."""
    apply_rc()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    time_yr = data['time_yr']

    # Panel (a): Lid thickness
    ax = axes[0, 0]
    ax.plot(
        time_yr, data['lid_thickness'] / 1e3, color='#1f77b4', lw=1.8, label='Lid thickness'
    )
    ax.set_xscale('log')
    ax.set_ylabel('Lid thickness $d_\\mathrm{lid}$ [km]')
    ax.set_ylim(bottom=0)
    ax.grid(True, ls=':', alpha=0.6)
    panel_label(ax, '(a)')

    # Panel (b): Temperatures (Interior and Potential)
    ax = axes[0, 1]
    ax.plot(time_yr, data['T_interior'], color='#d62728', lw=1.8, label='Interior $T_i$')
    ax.plot(
        time_yr,
        data['T_pot'],
        color='#ff7f0e',
        ls='--',
        lw=1.4,
        label='Potential $T_\\mathrm{pot}$',
    )
    ax.plot(
        time_yr,
        data['T_core'],
        color='#7f7f7f',
        ls=':',
        lw=1.4,
        label='Core $T_\\mathrm{core}$',
    )
    ax.set_xscale('log')
    ax.set_ylabel('Temperature [K]')
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, ls=':', alpha=0.6)
    panel_label(ax, '(b)')

    # Panel (c): Surface and CMB heat flows
    ax = axes[1, 0]
    ax.plot(
        time_yr, data['F_surf'], color='#1f77b4', lw=1.8, label='Surface $F_\\mathrm{surf}$'
    )
    ax.plot(
        time_yr, data['F_cmb'], color='#2ca02c', ls='--', lw=1.4, label='CMB $F_\\mathrm{cmb}$'
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Heat flux [W m$^{-2}$]')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, ls=':', alpha=0.6)
    panel_label(ax, '(c)')

    # Panel (d): Frank-Kamenetskii theta
    ax = axes[1, 1]
    ax.plot(time_yr, data['theta'], color='#9467bd', lw=1.8, label='$\\theta$')
    ax.axhline(9.0, color='gray', ls='--', lw=1.2, label='$\\theta = 9$ threshold')
    ax.set_xscale('log')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Frank-Kamenetskii parameter $\\theta$ [-]')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, ls=':', alpha=0.6)
    panel_label(ax, '(d)')

    fig.tight_layout()
    save(fig, str(out_base_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run solid-state convection acceptance to 4.5 Gyr.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=_REPO_ROOT / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml',
        help='Path to acceptance run TOML configuration',
    )
    parser.add_argument(
        '--eos-dir',
        type=Path,
        default=None,
        help='Path to SPIDER P-S tables directory',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=_REPO_ROOT / 'output' / 'ssc_acceptance',
        help='Directory to store output logs, data, and diagnostic plots',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint if available in output directory',
    )

    args = parser.parse_args()

    eos_dir = args.eos_dir
    if eos_dir is None:
        candidates = [
            os.environ.get('ARAGOG_TEST_EOS_DIR'),
            os.environ.get('FWL_DATA') + '/spider/lookup-fs'
            if os.environ.get('FWL_DATA')
            else None,
            str(
                _REPO_ROOT.parent
                / 'PROTEUS'
                / 'output'
                / 'coupled_parity'
                / 'spider'
                / 'data'
                / 'spider_eos'
            ),
            '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
        ]
        eos_dir = next((Path(p) for p in candidates if p and Path(p).is_dir()), None)

    if eos_dir is None or not eos_dir.is_dir():
        sys.exit(f'Error: EOS directory not found. Candidates searched: {candidates}')

    results = run_acceptance(args.config, eos_dir, args.out_dir, resume=args.resume)

    # Acceptance verification assertions
    print('\n--- Acceptance Run Verification Summary ---')
    print(f'Total simulated time: {results["time_yr"][-1]:.2e} yr')
    print(f'Solidification time (Phi < 0.05): {results["solidification_time_yr"]:.2e} yr')
    print(f'Total CVODE steps: {results["total_cvode_steps"]}')
    print(f'Minimum last CVODE step: {results["global_last_step_min"]:.3e} yr')
    print(f'Max relative energy residual: {np.max(results["relative_residual"]):.3e}')

    # Check 1: Completes to 4.5 Gyr
    assert np.isclose(results['time_yr'][-1], 4.5e9, rtol=1e-3), (
        'Run did not complete to 4.5 Gyr'
    )

    # Check 2: Lid regime is 0 while liquid exists at surface
    for t, p_surf, regime in zip(
        results['time_yr'], results['phi_surf'], results['lid_regime']
    ):
        if p_surf > 0.01:
            assert regime == 0.0, (
                f'Expected lid_regime 0 while surface has melt (t={t:e}, phi_surf={p_surf:.3f}, regime={regime})'
            )

    # Check 3: Discrete energy residual < 1e-3
    max_resid = np.max(results['relative_residual'])
    assert max_resid < 1.0e-3, (
        f'Energy residual exceeds 1e-3: max relative residual = {max_resid:.3e}'
    )

    print('ALL ACCEPTANCE CHECKS PASSED.')


if __name__ == '__main__':
    main()
