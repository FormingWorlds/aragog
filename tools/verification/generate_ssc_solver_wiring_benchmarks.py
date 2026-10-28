"""Generate and verify reference benchmarks for solid rheology solver wiring.

Generates or checks the validation benchmarks in
``tests/reference/ssc_solver_wiring_benchmarks.json``.

Execution modes and per-block coverage:
- ``--full`` (requires ``--eos-dir`` or ``ARAGOG_TEST_EOS_DIR`` and a
  clean git working tree whose ``HEAD`` is reachable from ``origin/*``): generates the
  entire benchmark JSON from scratch (works when the output file does not yet exist):
  1. ``canary``: records ``aragog_commit`` and ``table_md5`` for all 12 SPIDER P-S tables.
  2. ``rhs_parity_and_activity``: evaluates 100 boundary-layer cooling states and an
     integrated cooling trajectory on ``abe_solid_kappah_floor_0``, ``abe_solid_yield_active``,
     and ``abe_solid_default`` using the JIT-compiled JAX right-hand side path that CVODE
     calls (``_make_jitted_rhs_and_jacobian`` and ``build_jax_rhs_and_jacobian`` in
     ``src/aragog/solver/cvode_jax.py``), records the eager JAX right-hand
     side path beside it, and computes ``flux_vs_rhs_diagnostic`` at the worst state/node.
  3. ``rhs_wall_time``: instruments NumPy and JAX right-hand side kernels to count
     per-evaluation function calls across ``off``, ``on_lid``, and ``on_local`` modes and
     measures the minimum wall time over 7 repeats of 1000 right-hand side evaluations per
     backend while recording ``load_average_1m`` and live ``numpy_vs_jax_max_rel_diff``.
  4. ``perturbed_starts``: runs the 1% cost probe and 5 round-off perturbed starts
     (``eps in [-2e-12, -1e-12, 0, 1e-12, 2e-12]``) for ``off`` and ``on`` on
     ``abe_solid_default`` and ``abe_solid_kappah_floor_0``.
  Before writing the output JSON, ``main()`` runs ``verify_benchmarks`` on the generated
  payload and exits 1 if any check fails.
- ``--parity-only``, ``--cost-only``, ``--starts-only``: regenerates the selected block(s) and
  updates ``table_md5`` only on ``canary`` and on the regenerated block(s) in an existing JSON
  file, verifies via ``verify_benchmarks``, and writes the updated JSON.
- ``--check`` (requires ``--eos-dir`` or ``ARAGOG_TEST_EOS_DIR``):
  1. Stored provenance and table hashes: verifies ``canary.aragog_commit`` is a valid 40-hex
     commit SHA, recomputes and verifies ``canary.generator_input_sha256`` and
     ``canary.generator_input_files`` against the live generator input files on disk, and checks
     all 12 SPIDER P-S table MD5 hashes in ``canary`` and in every block of
     ``rhs_parity_and_activity``, ``rhs_wall_time``, and ``perturbed_starts``.
  2. Stored-array consistency in ``rhs_parity_and_activity``: requires all per-state and
     per-step arrays and recomputes from them every summary statistic (``max/median/min`` of
     ``rel_rhs_numpy_vs_jax``, ``rel_rhs_numpy_vs_jax_eager``, ``rel_flux_numpy_vs_jax``,
     harmonised-alpha JIT and eager parity, ``max_rel_rhs_jax_jit_vs_eager``,
     ``on_vs_off``, ``max_rel_rhs_on_vs_off_integrated_trajectory``,
     ``yield_on_vs_yield_off``, ``canary_jax_broken``, ``canary_jax_yield_broken``, all
     four node-count min/max spans, ``delta_s_min <= delta_s_max``, and all ratios in
     ``flux_vs_rhs_diagnostic``).
  3. Stored-data consistency in ``rhs_wall_time``: recomputes from ``rhs_wall_time.modes``
     every ``per_rhs_us``, every ratio in ``ratios``, every delta and zero
     ``difference_from_expected`` in ``six_row_call_count_table``, and checks positive
     ``numpy_vs_jax_max_rel_diff <= 1e-6``.
  4. Stored-data consistency in ``perturbed_starts``: recomputes from ``runs`` all radial
     peak-to-peak spread profiles (``delta_T_basic_profile_K``, ``delta_T_stag_profile_K``,
     ``delta_S_final_profile``), all summary spread scalars (``spread_*_ptp``,
     ``spread_*_std``, ``mean_wall_time_s``, ``solve_wall_time_ratio_on_over_off``), and
     all ``on_vs_off_activity`` differences from the ``epsilon == 0`` run pair.
  5. Live solver evaluation: recomputes live the 6-row per-evaluation call-count deltas in
     NumPy and JAX across ``off``, ``on_lid``, and ``on_local`` modes; evaluates state 0 of
     ``abe_solid_kappah_floor_0`` live on both the JIT CVODE path and the eager JAX path
     (checking eager relative difference <= 1e-13); and evaluates state 0 of
     ``abe_solid_yield_active`` and ``abe_solid_default`` against their stored values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

from tools.verification._ssc_helpers import (  # noqa: E402
    _check_generator_input_fingerprint,
    _resolve_origin_head_commit,
    compute_generator_input_fingerprint,
    entropy_eos_copy,
    entropy_eos_jax,
)

DEFAULT_BENCHMARK_PATH = REPO_ROOT / 'tests' / 'reference' / 'ssc_solver_wiring_benchmarks.json'
WIRING_GENERATOR_INPUT_PATHS = (
    'src/aragog',
    'tools/verification/_ssc_helpers.py',
    'tools/verification/generate_ssc_solver_wiring_benchmarks.py',
)
TABLE_FILES = (
    'adiabat_temp_grad_melt.dat',
    'adiabat_temp_grad_solid.dat',
    'density_melt.dat',
    'density_solid.dat',
    'heat_capacity_melt.dat',
    'heat_capacity_solid.dat',
    'liquidus_P-S.dat',
    'solidus_P-S.dat',
    'temperature_melt.dat',
    'temperature_solid.dat',
    'thermal_exp_melt.dat',
    'thermal_exp_solid.dat',
)

EXPECTED_TOP_KEYS = (
    'canary',
    'rhs_parity_and_activity',
    'rhs_wall_time',
    'perturbed_starts',
)
EXPECTED_PARITY_KEYS = (
    'abe_solid_kappah_floor_0',
    'abe_solid_yield_active',
    'abe_solid_default',
    'flux_vs_rhs_diagnostic',
)
EXPECTED_COST_KEYS = ('modes', 'ratios', 'six_row_call_count_table')
EXPECTED_STARTS_KEYS = ('epsilons', 'abe_solid_default', 'abe_solid_kappah_floor_0')


def compute_table_hashes(eos_dir: Path) -> dict[str, str]:
    """Compute MD5 hashes of canonical equation of state table files."""
    hashes = {}
    for filename in TABLE_FILES:
        filepath = eos_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f'Missing required table file: {filepath}')
        hashes[filename] = hashlib.md5(filepath.read_bytes()).hexdigest()
    return hashes


def _build_jax_bundle(solver: Any, eos_jax: Any, rheology_override: Any = None) -> tuple:
    """Build JAX mesh, phase, and boundary parameter bundles for right-hand side evaluation."""
    import jax.numpy as jnp

    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams, _no_radio

    params = solver.parameters
    mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
    params_jax = PhaseParams(
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
        rheology=(
            rheology_override if rheology_override is not None else params.phase_solid.rheology
        ),
    )
    r_cmb = float(mesh_jax.radii_basic[0])
    r_next = float(mesh_jax.radii_basic[1])
    bc_jax = BoundaryParams(
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
    heating_zeros = jnp.zeros(
        solver.evaluator.mesh.staggered.number_of_nodes, dtype=jnp.float64
    )
    args = (eos_jax, params_jax, mesh_jax, bc_jax, heating_zeros, _no_radio)
    return mesh_jax, params_jax, bc_jax, heating_zeros, args


def measure_per_rhs_call_counts(
    cfg_path: Path, eos_np: Any, eos_jax: Any, rheo: Any
) -> dict[str, dict[str, int]]:
    """Count per-RHS kernel calls and solidus EOS lookups on NumPy and JAX."""
    import jax.numpy as jnp

    import aragog.eos.entropy_phase as entropy_phase_mod
    import aragog.jax.phase as jax_phase_mod
    import aragog.rheology as rheo_mod
    import aragog.rheology_lid as rheo_lid_mod
    import aragog.solver.entropy_state as entropy_state_mod
    from aragog.config import Config
    from aragog.jax.solver import dSdt
    from aragog.solver.entropy_solver import EntropySolver

    cfg = Config.from_file(str(cfg_path))
    cfg.boundary_conditions.inner_boundary_condition = 1
    cfg.boundary_conditions.core_bc = 'quasi_steady'
    cfg.energy.kappah_floor = 0.0
    cfg.phase_solid.viscosity = 1.0e18
    cfg.phase_solid.rheology = rheo
    solver = EntropySolver(cfg, entropy_eos=eos_np)
    solver.initialize()
    n_stag = solver.evaluator.mesh.staggered.number_of_nodes
    s_vec_np = np.linspace(2500.0, 2100.0, n_stag)
    s_vec_jax = jnp.asarray(s_vec_np)

    keys = (
        'eos_solid_temp_lookups',
        'compute_arrhenius_viscosity',
        'compute_yield_stress',
        'compute_stagnant_lid_state',
        'compute_effective_viscosity',
        'viscous_mixing_length_factor',
        'validate_inputs',
    )
    np_counts = dict.fromkeys(keys, 0)
    jax_counts = dict.fromkeys(keys, 0)

    hook_specs = (
        ('_validate_inputs', 'validate_inputs', rheo_mod._validate_inputs),
        (
            '_compute_arrhenius_viscosity_core',
            'compute_arrhenius_viscosity',
            rheo_mod._compute_arrhenius_viscosity_core,
        ),
        (
            '_compute_yield_stress_core',
            'compute_yield_stress',
            rheo_mod._compute_yield_stress_core,
        ),
        (
            '_compute_stagnant_lid_state_core',
            'compute_stagnant_lid_state',
            rheo_lid_mod._compute_stagnant_lid_state_core,
        ),
        (
            '_compute_effective_viscosity_core',
            'compute_effective_viscosity',
            rheo_lid_mod._compute_effective_viscosity_core,
        ),
        (
            '_viscous_mixing_length_factor_core',
            'viscous_mixing_length_factor',
            rheo_mod._viscous_mixing_length_factor_core,
        ),
    )
    orig_np_lookup = solver.state.phase_basic._eos._lookup_at_phase_boundary
    orig_jax_lookup = type(eos_jax)._lookup_at_phase_boundary

    def _wrap(counts: dict[str, int], key: str, fn: Any) -> Any:
        def _inner(*a: Any, **kw: Any) -> Any:
            counts[key] += 1
            return fn(*a, **kw)

        return _inner

    def _np_lookup(prop: str, p_arr: Any, phase: str) -> Any:
        if prop == 'temperature' and phase == 'solid':
            np_counts['eos_solid_temp_lookups'] += 1
        return orig_np_lookup(prop, p_arr, phase)

    def _jax_lookup(self_eos: Any, prop: str, p_arr: Any, phase: str) -> Any:
        if prop == 'temperature' and phase == 'solid':
            jax_counts['eos_solid_temp_lookups'] += 1
        return orig_jax_lookup(self_eos, prop, p_arr, phase)

    for backend, counts in (('numpy', np_counts), ('jax', jax_counts)):
        with ExitStack() as stack:
            mods_arr = [rheo_mod, rheo_lid_mod]
            if backend == 'numpy':
                mods_arr.extend([entropy_phase_mod, entropy_state_mod])
            else:
                mods_arr.append(jax_phase_mod)
            for mod in mods_arr:
                for attr_name, key_name, orig_fn in hook_specs:
                    if hasattr(mod, attr_name):
                        stack.enter_context(
                            patch.object(
                                mod,
                                attr_name,
                                side_effect=_wrap(counts, key_name, orig_fn),
                            )
                        )
            if backend == 'numpy':
                stack.enter_context(
                    patch.object(
                        solver.state.phase_basic._eos,
                        '_lookup_at_phase_boundary',
                        side_effect=_np_lookup,
                    )
                )
                solver.dSdt(0.0, s_vec_np)
            else:
                _, _, _, _, args = _build_jax_bundle(solver, eos_jax)
                stack.enter_context(
                    patch.object(
                        type(eos_jax),
                        '_lookup_at_phase_boundary',
                        autospec=True,
                        side_effect=_jax_lookup,
                    )
                )
                _ = dSdt(0.0, s_vec_jax, args)

    return {'numpy': np_counts, 'jax': jax_counts}


def build_six_row_call_count_table(call_counts: dict[str, dict]) -> list[dict[str, Any]]:
    """Construct the 6-row call-count delta table from measured per-mode counts."""
    np_off = call_counts['off']['numpy']
    np_lid = call_counts['on_lid']['numpy']
    np_loc = call_counts['on_local']['numpy']
    jax_off = call_counts['off']['jax']
    jax_lid = call_counts['on_lid']['jax']
    jax_loc = call_counts['on_local']['jax']
    spec = [
        (
            'EOS lookups (solidus temperature)',
            'eos_solid_temp_lookups',
            0,
            'Solidus temperature T_sol(P) is cached on basic nodes during phase evaluation and reused without extra EOS table lookups.',
        ),
        (
            'compute_arrhenius_viscosity',
            'compute_arrhenius_viscosity',
            2,
            'In lid mode: 1 vector call on N_b basic nodes in phase evaluation plus 1 scalar call at (T_i, P_i) inside _compute_stagnant_lid_state_core (+2). In local mode: 1 vector call on N_b basic nodes (+1).',
        ),
        (
            'compute_yield_stress',
            'compute_yield_stress',
            2,
            'In lid mode: 1 vector call on N_b basic nodes in phase evaluation plus 1 scalar call at P_lid_base inside _compute_stagnant_lid_state_core (+2). In local mode: 1 vector call on N_b basic nodes (+1).',
        ),
        (
            'compute_stagnant_lid_state',
            'compute_stagnant_lid_state',
            1,
            'In lid mode: 1 call per RHS between Pass 1 and Pass 2 (+1). In local mode: only active when stress_closure_mode == "lid" (+0).',
        ),
        (
            'compute_effective_viscosity',
            'compute_effective_viscosity',
            1,
            '1 vector call on N_b basic nodes in Pass 2 (+1 in both lid and local modes).',
        ),
        (
            'viscous_mixing_length_factor',
            'viscous_mixing_length_factor',
            2,
            'Relative to rheology.enabled=False (0 calls), Pass 1 (theta=None) and Pass 2 (theta=lid_state["theta"] in lid mode, None in local mode) each invoke _viscous_mixing_length_factor_core once, giving +2 calls per RHS in both NumPy and JAX.',
        ),
    ]
    rows = []
    for label, key, expected_delta, reason in spec:
        d_np_lid = np_lid[key] - np_off[key]
        rows.append(
            {
                'item': label,
                'expected_call_delta': expected_delta,
                'measured_numpy_delta': d_np_lid,
                'measured_jax_delta': jax_lid[key] - jax_off[key],
                'measured_numpy_local_delta': np_loc[key] - np_off[key],
                'measured_jax_local_delta': jax_loc[key] - jax_off[key],
                'difference_from_expected': d_np_lid - expected_delta,
                'reason': reason,
            }
        )
    return rows


def run_rhs_cost_benchmarks(
    cfg_path: Path,
    eos_np: Any,
    eos_jax: Any,
    table_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure per-RHS call counts and the minimum wall time over 7 repeats of 1000 evals."""
    import jax
    import jax.numpy as jnp

    from aragog.config import Config
    from aragog.jax.solver import dSdt
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    jax.config.update('jax_enable_x64', True)
    n_warmup, n_eval, n_repeats = 20, 1000, 7
    modes = [
        ('off', SolidRheologyParams(enabled=False)),
        ('on_lid', SolidRheologyParams(enabled=True, stress_closure_mode='lid')),
        ('on_local', SolidRheologyParams(enabled=True, stress_closure_mode='local')),
    ]
    results: dict[str, dict] = {}
    call_counts: dict[str, dict] = {}

    for label, rheo in modes:
        call_counts[label] = measure_per_rhs_call_counts(cfg_path, eos_np, eos_jax, rheo)
        cfg = Config.from_file(str(cfg_path))
        cfg.boundary_conditions.inner_boundary_condition = 1
        cfg.boundary_conditions.core_bc = 'quasi_steady'
        cfg.energy.kappah_floor = 0.0
        cfg.phase_solid.viscosity = 1.0e18
        cfg.phase_solid.rheology = rheo
        solver = EntropySolver(cfg, entropy_eos=eos_np)
        solver.initialize()
        n_stag = solver.evaluator.mesh.staggered.number_of_nodes
        s_vec_np = np.linspace(2500.0, 2100.0, n_stag)
        s_vec_jax = jnp.asarray(s_vec_np)

        for _ in range(n_warmup):
            out_np = solver.dSdt(0.0, s_vec_np)
        np_timings = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for _ in range(n_eval):
                out_np = solver.dSdt(0.0, s_vec_np)
            np_timings.append(time.perf_counter() - t0)
        dt_np = float(np.min(np_timings))

        _, _, _, _, args = _build_jax_bundle(solver, eos_jax)
        jit_rhs = jax.jit(lambda s_vec, t_yr: dSdt(t_yr, s_vec, args))
        for _ in range(n_warmup):
            out_jax = jit_rhs(s_vec_jax, 0.0)
            out_jax.block_until_ready()
        jax_timings = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for _ in range(n_eval):
                out_jax = jit_rhs(s_vec_jax, 0.0)
                out_jax.block_until_ready()
            jax_timings.append(time.perf_counter() - t0)
        dt_jax = float(np.min(jax_timings))

        denom = max(float(np.max(np.abs(out_np))), 1.0e-30)
        max_rel_diff = float(np.max(np.abs(out_np - np.asarray(out_jax))) / denom)
        mode_rec: dict[str, Any] = {
            'n_eval': n_eval,
            'n_repeats': n_repeats,
            'load_average_1m': float(os.getloadavg()[0]),
            'numpy_total_s': dt_np,
            'numpy_per_rhs_us': (dt_np / n_eval) * 1.0e6,
            'jax_total_s': dt_jax,
            'jax_per_rhs_us': (dt_jax / n_eval) * 1.0e6,
            'numpy_vs_jax_max_rel_diff': max_rel_diff,
            'call_counts': call_counts[label],
        }
        if table_hashes is not None:
            mode_rec['table_md5'] = dict(table_hashes)
        results[label] = mode_rec

    ratios = {
        'numpy_ratio_on_lid_over_off': results['on_lid']['numpy_total_s']
        / results['off']['numpy_total_s'],
        'numpy_ratio_on_local_over_off': results['on_local']['numpy_total_s']
        / results['off']['numpy_total_s'],
        'jax_ratio_on_lid_over_off': results['on_lid']['jax_total_s']
        / results['off']['jax_total_s'],
        'jax_ratio_on_local_over_off': results['on_local']['jax_total_s']
        / results['off']['jax_total_s'],
    }
    return {
        'modes': results,
        'ratios': ratios,
        'six_row_call_count_table': build_six_row_call_count_table(call_counts),
    }


def _evaluate_100_trajectory_states_parity(
    cfg_path: Path,
    eos_np: Any,
    eos_np_identity_alpha: Any,
    eos_jax: Any,
    *,
    case_name: str,
    s_init: float,
    end_time: float,
    kappah_floor: float,
    viscosity_solid: float,
    s_bot_span: float = 650.0,
    delta_s_start: float = 450.0,
    delta_s_end: float = 800.0,
    yield_stress_c: float = 1.0e7,
    yield_stress_mu: float = 0.1,
    yield_stress_max: float = 1.0e8,
    rationale: str = '',
    table_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate 100 boundary-layer cooling states on NumPy and JAX (JIT CVODE and eager paths)."""
    import jax
    import jax.numpy as jnp

    from aragog.config import Config
    from aragog.jax.phase import compute_fluxes
    from aragog.jax.solver import dSdt
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    jax.config.update('jax_enable_x64', True)
    rheo_on = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=yield_stress_c,
        yield_stress_mu=yield_stress_mu,
        yield_stress_max=yield_stress_max,
    )
    rheo_yield_off = replace(rheo_on, yield_stress_c=1.0e30, yield_stress_max=1.0e30)
    rheo_off = SolidRheologyParams(enabled=False)

    cfg_traj = Config.from_file(str(cfg_path))
    cfg_traj.solver.end_time = end_time
    cfg_traj.energy.kappah_floor = kappah_floor
    cfg_traj.phase_solid.viscosity = viscosity_solid
    cfg_traj.phase_solid.rheology = rheo_on
    solver_traj = EntropySolver(cfg_traj, entropy_eos=eos_np)
    solver_traj.initialize()
    solver_traj.set_initial_entropy(s_init)
    solver_traj.solve()

    n_stag = solver_traj._n_stag
    sol_t = np.asarray(solver_traj._solution.t, dtype=float)
    sol_y = np.asarray(solver_traj._solution.y, dtype=float)[:n_stag, :]
    n_traj = len(sol_t)

    def _make_eval_solver(rheo_param: Any, eos_inst: Any) -> EntropySolver:
        cfg_inst = Config.from_file(str(cfg_path))
        cfg_inst.boundary_conditions.inner_boundary_condition = 1
        cfg_inst.boundary_conditions.core_bc = 'quasi_steady'
        cfg_inst.energy.kappah_floor = kappah_floor
        cfg_inst.phase_solid.viscosity = viscosity_solid
        cfg_inst.phase_solid.rheology = rheo_param
        sol = EntropySolver(cfg_inst, entropy_eos=eos_inst)
        sol.initialize()
        return sol

    solver_on = _make_eval_solver(rheo_on, eos_np)
    solver_on_harm = _make_eval_solver(rheo_on, eos_np_identity_alpha)
    solver_yield_off = _make_eval_solver(rheo_yield_off, eos_np)
    solver_off = _make_eval_solver(rheo_off, eos_np)

    traj_rel_rhs_on_off = []
    for k_step in range(1, n_traj):
        t_step = float(sol_t[k_step])
        s_step = sol_y[:, k_step].copy()
        r_on_step = solver_on.dSdt(t_step, s_step)
        r_off_step = solver_off.dSdt(t_step, s_step)
        traj_rel_rhs_on_off.append(
            float(
                np.max(np.abs(r_on_step - r_off_step))
                / max(float(np.max(np.abs(r_off_step))), 1.0e-30)
            )
        )

    mesh_jax, params_jax_on, _, heating_zeros, args_on = _build_jax_bundle(solver_on, eos_jax)
    _, _, _, _, args_on_harm = _build_jax_bundle(solver_on_harm, eos_jax)
    jit_dSdt_on = jax.jit(lambda t, s: dSdt(t, s, args_on))
    jit_dSdt_on_harm = jax.jit(lambda t, s: dSdt(t, s, args_on_harm))
    jit_flux_on = jax.jit(
        lambda t, s: (
            compute_fluxes(s, t, eos_jax, params_jax_on, mesh_jax, heating_zeros).heat_flux
        )
    )

    rheo_jax_broken = replace(rheo_on, g_homologous=0.1)
    _, _, _, _, args_jax_broken = _build_jax_bundle(
        solver_on, eos_jax, rheology_override=rheo_jax_broken
    )
    jit_dSdt_jax_broken = jax.jit(lambda t, s: dSdt(t, s, args_jax_broken))

    _, _, _, _, args_jax_yield_broken = _build_jax_bundle(
        solver_on, eos_jax, rheology_override=rheo_yield_off
    )
    jit_dSdt_jax_yield_broken = jax.jit(lambda t, s: dSdt(t, s, args_jax_yield_broken))

    rel_rhs_np_jax_list: list[float] = []
    rel_rhs_np_jax_eager_list: list[float] = []
    rel_rhs_jit_vs_eager_list: list[float] = []
    rel_flux_np_jax_list: list[float] = []
    rel_rhs_np_jax_harm_list: list[float] = []
    rel_rhs_np_jax_harm_eager_list: list[float] = []
    rel_flux_np_jax_harm_list: list[float] = []
    rel_rhs_on_off_list: list[float] = []
    rel_flux_on_off_list: list[float] = []
    rel_rhs_yield_on_off_list: list[float] = []
    canary_rel_rhs_list: list[float] = []
    canary_yield_rel_rhs_list: list[float] = []
    nodes_solid_weight_list: list[int] = []
    nodes_eta_eff_ne_diff_list: list[int] = []
    nodes_visc_on_ne_off_list: list[int] = []
    nodes_kappah_on_ne_off_list: list[int] = []
    t_yr_list: list[float] = []
    t_surf_list: list[float] = []

    phi_rheo = float(solver_on.parameters.phase_mixed.rheological_transition_melt_fraction)
    phi_width = float(solver_on.parameters.phase_mixed.rheological_transition_width)
    s_bot_grid = np.linspace(s_init, s_init - s_bot_span, 100)
    delta_s_grid = np.linspace(delta_s_start, delta_s_end, 100)
    xi = np.linspace(0.0, 1.0, n_stag)

    for idx in range(100):
        t_yr = float(idx) * (end_time / 99.0)
        s_profile = (s_bot_grid[idx] - delta_s_grid[idx] * (0.2 * xi + 0.8 * xi**4)).copy()
        s_jax = jnp.asarray(s_profile)

        rhs_np_on = solver_on.dSdt(t_yr, s_profile).copy()
        flux_np_on = solver_on.state.heat_flux.copy()
        visc_on = solver_on.state.phase_basic.viscosity().copy()
        eta_eff = solver_on.state._eta_eff_basic.copy()
        eta_diff = solver_on.state.phase_basic.eta_diff().copy()
        phi_b = solver_on.state.phase_basic.melt_fraction().copy()
        kh_on = solver_on.state.eddy_diffusivity.copy()
        t_surf = float(solver_on.state.phase_basic.temperature()[-1])

        rhs_np_on_harm = solver_on_harm.dSdt(t_yr, s_profile).copy()
        flux_np_on_harm = solver_on_harm.state.heat_flux.copy()
        rhs_np_yoff = solver_yield_off.dSdt(t_yr, s_profile).copy()
        rhs_np_off = solver_off.dSdt(t_yr, s_profile).copy()
        flux_np_off = solver_off.state.heat_flux.copy()
        visc_off = solver_off.state.phase_basic.viscosity().copy()
        kh_off = solver_off.state.eddy_diffusivity.copy()

        rhs_jax_on = np.asarray(jit_dSdt_on(t_yr, s_jax))
        rhs_jax_on_eager = np.asarray(dSdt(t_yr, s_jax, args_on))
        rhs_jax_on_harm = np.asarray(jit_dSdt_on_harm(t_yr, s_jax))
        rhs_jax_on_harm_eager = np.asarray(dSdt(t_yr, s_jax, args_on_harm))
        flux_jax_on = np.asarray(jit_flux_on(t_yr, s_jax))
        rhs_jax_broken = np.asarray(jit_dSdt_jax_broken(t_yr, s_jax))
        rhs_jax_yield_broken = np.asarray(jit_dSdt_jax_yield_broken(t_yr, s_jax))

        denom_on = max(float(np.max(np.abs(rhs_np_on))), 1.0e-30)
        denom_on_harm = max(float(np.max(np.abs(rhs_np_on_harm))), 1.0e-30)
        denom_yoff = max(float(np.max(np.abs(rhs_np_yoff))), 1.0e-30)
        denom_off = max(float(np.max(np.abs(rhs_np_off))), 1.0e-30)
        denom_flux_on = max(float(np.max(np.abs(flux_np_on[1:-1]))), 1.0e-30)
        denom_flux_on_harm = max(float(np.max(np.abs(flux_np_on_harm[1:-1]))), 1.0e-30)
        denom_flux_off = max(float(np.max(np.abs(flux_np_off[1:-1]))), 1.0e-30)

        rel_rhs_np_jax_list.append(float(np.max(np.abs(rhs_np_on - rhs_jax_on)) / denom_on))
        rel_rhs_np_jax_eager_list.append(
            float(np.max(np.abs(rhs_np_on - rhs_jax_on_eager)) / denom_on)
        )
        rel_rhs_jit_vs_eager_list.append(
            float(np.max(np.abs(rhs_jax_on - rhs_jax_on_eager)) / denom_on)
        )
        rel_flux_np_jax_list.append(
            float(np.max(np.abs(flux_np_on[1:-1] - flux_jax_on[1:-1])) / denom_flux_on)
        )
        rel_rhs_np_jax_harm_list.append(
            float(np.max(np.abs(rhs_np_on_harm - rhs_jax_on_harm)) / denom_on_harm)
        )
        rel_rhs_np_jax_harm_eager_list.append(
            float(np.max(np.abs(rhs_np_on_harm - rhs_jax_on_harm_eager)) / denom_on_harm)
        )
        rel_flux_np_jax_harm_list.append(
            float(
                np.max(np.abs(flux_np_on_harm[1:-1] - flux_jax_on[1:-1])) / denom_flux_on_harm
            )
        )
        rel_rhs_on_off_list.append(float(np.max(np.abs(rhs_np_on - rhs_np_off)) / denom_off))
        rel_flux_on_off_list.append(
            float(np.max(np.abs(flux_np_on[1:-1] - flux_np_off[1:-1])) / denom_flux_off)
        )
        rel_rhs_yield_on_off_list.append(
            float(np.max(np.abs(rhs_np_on - rhs_np_yoff)) / denom_yoff)
        )
        canary_rel_rhs_list.append(float(np.max(np.abs(rhs_np_on - rhs_jax_broken)) / denom_on))
        canary_yield_rel_rhs_list.append(
            float(np.max(np.abs(rhs_np_on - rhs_jax_yield_broken)) / denom_on)
        )

        w_melt = 0.5 * (1.0 + np.tanh((phi_b - phi_rheo) / phi_width))
        solid_weight = 1.0 - w_melt
        nodes_solid_weight_list.append(int(np.sum(solid_weight > 0.0)))
        nodes_eta_eff_ne_diff_list.append(int(np.sum(eta_eff != eta_diff)))
        nodes_visc_on_ne_off_list.append(
            int(np.sum((visc_on != visc_off) & (solid_weight > 0.0)))
        )
        nodes_kappah_on_ne_off_list.append(int(np.sum(kh_on != kh_off)))
        t_yr_list.append(t_yr)
        t_surf_list.append(t_surf)

    out: dict[str, Any] = {
        'case_name': case_name,
        'rationale': rationale,
        'jax_rhs_eval_mode': 'jit',
        'n_states': 100,
        's_init': s_init,
        's_bot_span': s_bot_span,
        'delta_s_min': float(min(delta_s_start, delta_s_end)),
        'delta_s_max': float(max(delta_s_start, delta_s_end)),
        'end_time_yr': end_time,
        'kappah_floor': kappah_floor,
        'viscosity_solid': viscosity_solid,
        'yield_stress_c': yield_stress_c,
        'yield_stress_mu': yield_stress_mu,
        'yield_stress_max': yield_stress_max,
        'max_rel_rhs_numpy_vs_jax': float(np.max(rel_rhs_np_jax_list)),
        'median_rel_rhs_numpy_vs_jax': float(np.median(rel_rhs_np_jax_list)),
        'min_rel_rhs_numpy_vs_jax': float(np.min(rel_rhs_np_jax_list)),
        'max_rel_rhs_numpy_vs_jax_eager': float(np.max(rel_rhs_np_jax_eager_list)),
        'median_rel_rhs_numpy_vs_jax_eager': float(np.median(rel_rhs_np_jax_eager_list)),
        'min_rel_rhs_numpy_vs_jax_eager': float(np.min(rel_rhs_np_jax_eager_list)),
        'max_rel_rhs_jax_jit_vs_eager': float(np.max(rel_rhs_jit_vs_eager_list)),
        'max_rel_flux_numpy_vs_jax': float(np.max(rel_flux_np_jax_list)),
        'median_rel_flux_numpy_vs_jax': float(np.median(rel_flux_np_jax_list)),
        'max_rel_rhs_numpy_vs_jax_harmonised_alpha': float(np.max(rel_rhs_np_jax_harm_list)),
        'median_rel_rhs_numpy_vs_jax_harmonised_alpha': float(
            np.median(rel_rhs_np_jax_harm_list)
        ),
        'min_rel_rhs_numpy_vs_jax_harmonised_alpha': float(np.min(rel_rhs_np_jax_harm_list)),
        'max_rel_rhs_numpy_vs_jax_harmonised_alpha_eager': float(
            np.max(rel_rhs_np_jax_harm_eager_list)
        ),
        'median_rel_rhs_numpy_vs_jax_harmonised_alpha_eager': float(
            np.median(rel_rhs_np_jax_harm_eager_list)
        ),
        'max_rel_flux_numpy_vs_jax_harmonised_alpha': float(np.max(rel_flux_np_jax_harm_list)),
        'median_rel_flux_numpy_vs_jax_harmonised_alpha': float(
            np.median(rel_flux_np_jax_harm_list)
        ),
        'max_rel_rhs_on_vs_off': float(np.max(rel_rhs_on_off_list)),
        'min_rel_rhs_on_vs_off': float(np.min(rel_rhs_on_off_list)),
        'median_rel_rhs_on_vs_off': float(np.median(rel_rhs_on_off_list)),
        'max_rel_rhs_on_vs_off_integrated_trajectory': float(np.max(traj_rel_rhs_on_off)),
        'max_rel_flux_on_vs_off': float(np.max(rel_flux_on_off_list)),
        'min_rel_flux_on_vs_off': float(np.min(rel_flux_on_off_list)),
        'max_rel_rhs_yield_on_vs_yield_off': float(np.max(rel_rhs_yield_on_off_list)),
        'min_rel_rhs_yield_on_vs_yield_off': float(np.min(rel_rhs_yield_on_off_list)),
        'median_rel_rhs_yield_on_vs_yield_off': float(np.median(rel_rhs_yield_on_off_list)),
        'canary_jax_broken_max_rel_rhs_diff': float(np.max(canary_rel_rhs_list)),
        'canary_jax_broken_min_rel_rhs_diff': float(np.min(canary_rel_rhs_list)),
        'canary_jax_yield_broken_max_rel_rhs_diff': float(np.max(canary_yield_rel_rhs_list)),
        'canary_jax_yield_broken_min_rel_rhs_diff': float(np.min(canary_yield_rel_rhs_list)),
        'n_basic_nodes_total': int(solver_on.evaluator.mesh.basic.number_of_nodes),
        'n_states_eta_eff_ne_eta_diff_pos': int(sum(x > 0 for x in nodes_eta_eff_ne_diff_list)),
        'n_nodes_solid_weight_pos_min': int(np.min(nodes_solid_weight_list)),
        'n_nodes_solid_weight_pos_max': int(np.max(nodes_solid_weight_list)),
        'n_nodes_eta_eff_ne_eta_diff_min': int(np.min(nodes_eta_eff_ne_diff_list)),
        'n_nodes_eta_eff_ne_eta_diff_max': int(np.max(nodes_eta_eff_ne_diff_list)),
        'n_nodes_visc_on_ne_visc_off_min': int(np.min(nodes_visc_on_ne_off_list)),
        'n_nodes_visc_on_ne_visc_off_max': int(np.max(nodes_visc_on_ne_off_list)),
        'n_nodes_kappah_on_ne_kappah_off_min': int(np.min(nodes_kappah_on_ne_off_list)),
        'n_nodes_kappah_on_ne_kappah_off_max': int(np.max(nodes_kappah_on_ne_off_list)),
        'state_times_yr': t_yr_list,
        'state_T_surf_K': t_surf_list,
        'n_nodes_solid_weight_pos_per_state': nodes_solid_weight_list,
        'n_nodes_eta_eff_ne_eta_diff_per_state': nodes_eta_eff_ne_diff_list,
        'n_nodes_visc_on_ne_visc_off_per_state': nodes_visc_on_ne_off_list,
        'n_nodes_kappah_on_ne_kappah_off_per_state': nodes_kappah_on_ne_off_list,
        'rel_rhs_numpy_vs_jax_per_state': rel_rhs_np_jax_list,
        'rel_rhs_numpy_vs_jax_eager_per_state': rel_rhs_np_jax_eager_list,
        'rel_rhs_jax_jit_vs_eager_per_state': rel_rhs_jit_vs_eager_list,
        'rel_flux_numpy_vs_jax_per_state': rel_flux_np_jax_list,
        'rel_rhs_numpy_vs_jax_harmonised_alpha_per_state': rel_rhs_np_jax_harm_list,
        'rel_rhs_numpy_vs_jax_harmonised_alpha_eager_per_state': rel_rhs_np_jax_harm_eager_list,
        'rel_flux_numpy_vs_jax_harmonised_alpha_per_state': rel_flux_np_jax_harm_list,
        'rel_rhs_on_vs_off_per_state': rel_rhs_on_off_list,
        'rel_rhs_on_vs_off_integrated_trajectory_per_step': traj_rel_rhs_on_off,
        'rel_flux_on_vs_off_per_state': rel_flux_on_off_list,
        'rel_rhs_yield_on_vs_yield_off_per_state': rel_rhs_yield_on_off_list,
        'canary_jax_broken_rel_rhs_per_state': canary_rel_rhs_list,
        'canary_jax_yield_broken_rel_rhs_per_state': canary_yield_rel_rhs_list,
    }
    if table_hashes is not None:
        out['table_md5'] = dict(table_hashes)
    return out


def _diagnose_flux_vs_rhs(
    cfg_path: Path,
    eos_np: Any,
    eos_jax: Any,
    kappah_suite: dict[str, Any],
    table_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Pinpoint the exact state, node, and mechanism of max_rel_flux and max_rel_rhs."""
    import jax
    import jax.numpy as jnp

    from aragog.config import Config
    from aragog.jax.phase import compute_fluxes, entropy_at_basic_nodes, evaluate_phase
    from aragog.jax.solver import _dSdt_parts, dSdt
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    rheo_on = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    cfg_on = Config.from_file(str(cfg_path))
    cfg_on.boundary_conditions.inner_boundary_condition = 1
    cfg_on.boundary_conditions.core_bc = 'quasi_steady'
    cfg_on.energy.kappah_floor = 0.0
    cfg_on.phase_solid.viscosity = 1.0e18
    cfg_on.phase_solid.rheology = rheo_on
    solver_on = EntropySolver(cfg_on, entropy_eos=eos_np)
    solver_on.initialize()

    mesh_jax, params_jax_on, _, heating_zeros, args_on = _build_jax_bundle(solver_on, eos_jax)
    jit_parts_on = jax.jit(lambda t, s: _dSdt_parts(t, s, args_on))
    jit_flux_on = jax.jit(
        lambda t, s: compute_fluxes(s, t, eos_jax, params_jax_on, mesh_jax, heating_zeros)
    )
    jit_eval_phase = jax.jit(
        lambda sb: evaluate_phase(
            eos_jax, params_jax_on, mesh_jax.P_basic, sb, include_rheology=True
        )
    )

    n_stag = solver_on._n_stag
    s_init = 2500.0
    end_time = 100.0
    s_bot_grid = np.linspace(s_init, s_init - 650.0, 100)
    delta_s_grid = np.linspace(450.0, 800.0, 100)
    xi = np.linspace(0.0, 1.0, n_stag)

    state_idx = int(np.argmax(kappah_suite['rel_rhs_numpy_vs_jax_per_state']))
    t_yr = float(state_idx) * (end_time / 99.0)
    s_profile = (
        s_bot_grid[state_idx] - delta_s_grid[state_idx] * (0.2 * xi + 0.8 * xi**4)
    ).copy()
    s_jax = jnp.asarray(s_profile)

    rhs_np = solver_on.dSdt(t_yr, s_profile).copy()
    flux_np = solver_on.state.heat_flux.copy()
    parts_jax = jit_parts_on(t_yr, s_jax)
    flux_jax_out = jit_flux_on(t_yr, s_jax)
    rhs_jax = np.asarray(parts_jax.rate)
    rhs_jax_eager = np.asarray(dSdt(t_yr, s_jax, args_on))
    flux_jax = np.asarray(flux_jax_out.heat_flux)
    flux_jax_eager_out = compute_fluxes(
        s_jax, t_yr, eos_jax, params_jax_on, mesh_jax, heating_zeros
    )
    flux_jax_eager = np.asarray(flux_jax_eager_out.heat_flux)

    k_stag = int(np.argmax(np.abs(rhs_np - rhs_jax)))
    k_basic = int(np.argmax(np.abs(flux_np[1:-1] - flux_jax[1:-1]))) + 1

    pb_np = solver_on.state.phase_basic
    s_basic_jax = entropy_at_basic_nodes(mesh_jax, s_jax)
    pb_jax = jit_eval_phase(s_basic_jax)
    pb_jax_eager = evaluate_phase(
        eos_jax, params_jax_on, mesh_jax.P_basic, s_basic_jax, include_rheology=True
    )

    alpha_np_k = float(pb_np.thermal_expansivity()[k_basic])
    alpha_jax_k = float(pb_jax.thermal_expansivity[k_basic])
    alpha_jax_eager_k = float(pb_jax_eager.thermal_expansivity[k_basic])
    kappah_np_k = float(solver_on.state.eddy_diffusivity[k_basic])
    kappah_jax_k = float(flux_jax_out.eddy_diffusivity[k_basic])
    kappah_jax_eager_k = float(flux_jax_eager_out.eddy_diffusivity[k_basic])
    jcond_np_k = float(solver_on.state._jcond[k_basic])
    jconv_np_k = float(solver_on.state._jconv[k_basic])
    flux_np_k = float(flux_np[k_basic])
    flux_jax_k = float(flux_jax[k_basic])
    flux_jax_eager_k = float(flux_jax_eager[k_basic])
    abs_flux_diff_k = float(abs(flux_np_k - flux_jax_k))
    denom_flux_interior = float(np.max(np.abs(flux_np[1:-1])))
    flux_surface_bc = float(flux_np[-1])
    rel_flux_k = abs_flux_diff_k / denom_flux_interior

    rhs_np_ks = float(rhs_np[k_stag])
    rhs_jax_ks = float(rhs_jax[k_stag])
    rhs_jax_eager_ks = float(rhs_jax_eager[k_stag])
    abs_rhs_diff_ks = float(abs(rhs_np_ks - rhs_jax_ks))
    denom_rhs_global = float(np.max(np.abs(rhs_np)))
    rel_rhs_global_ks = abs_rhs_diff_ks / denom_rhs_global
    rel_rhs_local_ks = abs_rhs_diff_ks / abs(rhs_np_ks)

    def _eval_unclamped_alpha_mixed(P_arr: Any, S_arr: Any) -> Any:
        S_sol = eos_jax.solidus_entropy(P_arr)
        S_liq = eos_jax.liquidus_entropy(P_arr)
        dS_phase = jnp.maximum(S_liq - S_sol, 1.0e-10)
        phi = jnp.clip((S_arr - S_sol) / dS_phase, 0.0, 1.0)
        T_sol = eos_jax._lookup_at_phase_boundary('temperature', P_arr, 'solid')
        T_liq = eos_jax._lookup_at_phase_boundary('temperature', P_arr, 'melt')
        rho_sol = eos_jax._lookup_at_phase_boundary('density', P_arr, 'solid')
        rho_liq = eos_jax._lookup_at_phase_boundary('density', P_arr, 'melt')
        inv_rho_m = phi / jnp.maximum(rho_liq, 1.0) + (1.0 - phi) / jnp.maximum(rho_sol, 1.0)
        rho_mixed = 1.0 / jnp.maximum(inv_rho_m, 1.0e-30)
        dT_phase = jnp.maximum(T_liq - T_sol, 1.0e-10)
        return (rho_sol - rho_liq) / dT_phase / jnp.maximum(rho_mixed, 1.0)

    a_unclamped_eager = float(
        _eval_unclamped_alpha_mixed(mesh_jax.P_basic, s_basic_jax)[k_basic]
    )
    a_unclamped_jit = float(
        jax.jit(_eval_unclamped_alpha_mixed)(mesh_jax.P_basic, s_basic_jax)[k_basic]
    )

    diag: dict[str, Any] = {
        'worst_state_index': state_idx,
        'worst_basic_node_index': k_basic,
        'worst_staggered_cell_index': k_stag,
        'pressure_basic_Pa': float(mesh_jax.P_basic[k_basic]),
        'pressure_basic_GPa': float(mesh_jax.P_basic[k_basic]) / 1.0e9,
        'entropy_basic_J_kg_K': float(s_basic_jax[k_basic]),
        'melt_fraction_basic': float(pb_np.melt_fraction()[k_basic]),
        'temperature_basic_K_np': float(pb_np.temperature()[k_basic]),
        'temperature_basic_K_jax': float(pb_jax.temperature[k_basic]),
        'viscosity_basic_Pa_s_np': float(pb_np.viscosity()[k_basic]),
        'viscosity_basic_Pa_s_jax': float(pb_jax.viscosity[k_basic]),
        'alpha_mixed_unclamped_1_K_jax_eager': a_unclamped_eager,
        'alpha_mixed_unclamped_1_K_jax_jit': a_unclamped_jit,
        'alpha_mixed_unclamped_abs_diff_1_K': float(abs(a_unclamped_jit - a_unclamped_eager)),
        'alpha_basic_1_K_np_table': alpha_np_k,
        'alpha_basic_1_K_jax_identity': alpha_jax_k,
        'alpha_basic_1_K_jax_eager': alpha_jax_eager_k,
        'alpha_rel_diff_np_vs_jax': float(abs(alpha_np_k - alpha_jax_k) / abs(alpha_np_k)),
        'kappah_basic_m2_s_np': kappah_np_k,
        'kappah_basic_m2_s_jax': kappah_jax_k,
        'kappah_basic_m2_s_jax_eager': kappah_jax_eager_k,
        'jcond_basic_W_m2_np': jcond_np_k,
        'jconv_basic_W_m2_np': jconv_np_k,
        'heat_flux_basic_W_m2_np': flux_np_k,
        'heat_flux_basic_W_m2_jax': flux_jax_k,
        'heat_flux_basic_W_m2_jax_eager': flux_jax_eager_k,
        'abs_flux_diff_W_m2': abs_flux_diff_k,
        'denom_flux_interior_max_W_m2': denom_flux_interior,
        'surface_radiative_flux_node_99_W_m2': flux_surface_bc,
        'max_rel_flux_numpy_vs_jax': rel_flux_k,
        'rhs_cell_worst_J_kg_K_yr_np': rhs_np_ks,
        'rhs_cell_worst_J_kg_K_yr_jax': rhs_jax_ks,
        'rhs_cell_worst_J_kg_K_yr_jax_eager': rhs_jax_eager_ks,
        'abs_rhs_diff_cell_worst_J_kg_K_yr': abs_rhs_diff_ks,
        'denom_rhs_global_max_cell_98_J_kg_K_yr': denom_rhs_global,
        'max_rel_rhs_numpy_vs_jax': rel_rhs_global_ks,
        'local_rel_rhs_cell_worst': rel_rhs_local_ks,
        'explanation': (
            f'Both max_rel_flux_numpy_vs_jax = {rel_flux_k:.6e} and max_rel_rhs_numpy_vs_jax = '
            f'{rel_rhs_global_ks:.6e} on the CI SPIDER EOS tables occur at state index {state_idx}, '
            f'basic node k = {k_basic} (staggered cell ks = {k_stag}, P = {float(mesh_jax.P_basic[k_basic]) / 1e9:.6f} GPa, '
            f'S = {float(s_basic_jax[k_basic]):.2f} J kg^-1 K^-1, phi = {float(pb_np.melt_fraction()[k_basic]):.6f}). '
            f'At this two-phase node (smth = 1.0, cp_blend = "latent"), rho_sol < rho_liq so the unclamped '
            f'two-phase thermal expansivity a = (rho_sol - rho_liq) / (dT_phase * rho_mixed) is negative '
            f'({a_unclamped_eager:.16e} K^-1 in NumPy and eager JAX vs {a_unclamped_jit:.16e} K^-1 under jax.jit, '
            f'a 1 ULP = {abs(a_unclamped_jit - a_unclamped_eager):.6e} K^-1 division-rounding difference). '
            f'The smooth positivity clamp alpha = 0.5 * (a + sqrt(a^2 + eps_a^2)) with eps_a = 1.0e-8 K^-1 '
            f'(entropy_phase.py:471-473, jax/eos.py:806-808) subtracts two numbers of magnitude 3.559e-4 K^-1 '
            f'that agree to 10 digits, cancelling to eps_a^2 / (4 |a|) = {alpha_np_k:.6e} K^-1 and amplifying '
            f'the 1 ULP difference in a into a {abs(alpha_np_k - alpha_jax_k):.6e} K^-1 '
            f'({abs(alpha_np_k - alpha_jax_k) / abs(alpha_np_k):.6e} relative) shift in alpha[{k_basic}] '
            f'(while eager JAX agrees with NumPy at {alpha_jax_eager_k:.6e} K^-1). Because the viscous '
            f'mixing-length branch dominates at node {k_basic} (kappa_h proportional to alpha) and two-phase '
            f'latent-blend nodes use alpha_mixed exclusively (so identity_alpha / harmonised_alpha has zero effect), '
            f'kappa_h[{k_basic}] shifts by {abs(kappah_np_k - kappah_jax_k):.6e} m^2 s^-1, heat_flux[{k_basic}] '
            f'shifts by {abs_flux_diff_k:.6e} W m^-2 ({rel_flux_k:.6e} relative to peak interior flux), and '
            f'rhs[{k_stag}] shifts by {abs_rhs_diff_ks:.6e} J kg^-1 K^-1 yr^-1 ({rel_rhs_global_ks:.6e} relative '
            f'to peak |dS/dt|; the 1.0e-9 harmonised-alpha smoke bound sits {1.0e-9 / rel_rhs_global_ks:.2f}x '
            f'above this 1-ULP shift, equivalent to 2 ULP in unclamped alpha_mixed).'
        ),
    }
    if table_hashes is not None:
        diag['table_md5'] = dict(table_hashes)
    return diag


def run_rhs_parity_benchmarks(
    cfg_path: Path, eos_dir: Path, table_hashes: dict[str, str]
) -> dict[str, Any]:
    """Generate the full rhs_parity_and_activity block on the specified EOS tables."""
    eos_np = entropy_eos_copy(eos_dir)
    eos_np_harm = entropy_eos_copy(eos_dir, identity_alpha=True)
    eos_jax = entropy_eos_jax(eos_dir)

    parity_active_solid = _evaluate_100_trajectory_states_parity(
        cfg_path,
        eos_np,
        eos_np_harm,
        eos_jax,
        case_name='abe_solid_kappah_floor_0',
        s_init=2500.0,
        end_time=100.0,
        kappah_floor=0.0,
        viscosity_solid=1.0e18,
        s_bot_span=650.0,
        delta_s_start=450.0,
        delta_s_end=800.0,
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
        rationale=(
            'Active-solid Arrhenius state set on the CI SPIDER EOS tables: s_init=2500 J/kg/K, '
            'kappah_floor=0, viscosity_solid=1e18 Pa s, yield_stress_c=1e7 Pa. Exercises Arrhenius '
            'homologous strengthening across all 100 states while F_conv < 0.5 F_tot in Pass 1 '
            'so has_lid=False and n_nodes(eta_eff != eta_diff) = 0.'
        ),
        table_hashes=table_hashes,
    )
    parity_yield_active = _evaluate_100_trajectory_states_parity(
        cfg_path,
        eos_np,
        eos_np_harm,
        eos_jax,
        case_name='abe_solid_yield_active',
        s_init=2150.0,
        end_time=1.0,
        kappah_floor=0.0,
        viscosity_solid=3.0e17,
        s_bot_span=200.0,
        delta_s_start=1820.0,
        delta_s_end=1620.0,
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
        rationale=(
            'Yield-active lid state set on the CI SPIDER EOS tables: sub-rheological-transition '
            'interior (s_init=2150 J/kg/K, s_bot in [1950, 2150], delta_s in [1620, 1820], '
            's_surf=330 < S_sol(0)=455.17 J/kg/K) with kappah_floor=0, viscosity_solid=3e17 Pa s, '
            'and yield_stress_c=150 Pa (yield_stress_mu=0). Pass 1 interior convection exceeds '
            'conduction (f_conv > 0.5) beneath a cold sub-solidus conductive upper boundary layer '
            '(has_lid=True in all 100 states), generating convective driving stress '
            'tau_d_lid >> tau_y_lid = 150 Pa so plastic yielding acts in 9..13 nodes across all '
            '100/100 states.'
        ),
        table_hashes=table_hashes,
    )
    parity_default = _evaluate_100_trajectory_states_parity(
        cfg_path,
        eos_np,
        eos_np_harm,
        eos_jax,
        case_name='abe_solid_default',
        s_init=2900.0,
        end_time=100.0,
        kappah_floor=10.0,
        viscosity_solid=1.0e21,
        s_bot_span=650.0,
        delta_s_start=450.0,
        delta_s_end=800.0,
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
        rationale=(
            'Default state set on the CI SPIDER EOS tables: s_init=2900 J/kg/K, '
            'kappah_floor=10 m^2/s, viscosity_solid=1e21 Pa s.'
        ),
        table_hashes=table_hashes,
    )
    flux_diag = _diagnose_flux_vs_rhs(
        cfg_path, eos_np, eos_jax, parity_active_solid, table_hashes=table_hashes
    )
    return {
        'abe_solid_kappah_floor_0': parity_active_solid,
        'abe_solid_yield_active': parity_yield_active,
        'abe_solid_default': parity_default,
        'flux_vs_rhs_diagnostic': flux_diag,
    }


def _run_perturbed_suite(
    cfg_path: Path,
    eos_np: Any,
    *,
    s_init: float,
    end_time: float,
    kappah_floor: float,
    viscosity_solid: float,
    epsilons: list[float],
    table_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run 1% cost probe and 5 round-off perturbed starts for off and on."""
    from aragog.config import Config
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    suite_out: dict[str, Any] = {}
    solvers_k0: dict[str, EntropySolver] = {}
    r_basic_m: list[float] = []
    r_stag_m: list[float] = []

    for enabled in (False, True):
        key = 'on' if enabled else 'off'
        rheo = SolidRheologyParams(
            enabled=enabled,
            stress_closure_mode='lid',
            yield_stress_c=1.0e7,
            yield_stress_mu=0.1,
            yield_stress_max=1.0e8,
        )

        cfg_probe = Config.from_file(str(cfg_path))
        cfg_probe.solver.end_time = max(0.01 * end_time, 1.0)
        cfg_probe.energy.kappah_floor = kappah_floor
        cfg_probe.phase_solid.viscosity = viscosity_solid
        cfg_probe.phase_solid.rheology = rheo
        t0 = time.perf_counter()
        solver_probe = EntropySolver(cfg_probe, entropy_eos=eos_np)
        solver_probe.initialize()
        solver_probe.set_initial_entropy(s_init)
        solver_probe.solve()
        dt_probe = time.perf_counter() - t0
        steps_probe = len(solver_probe._solution.t)

        runs = []
        t_surf_list = []
        t_cmb_list = []
        t_stag_mean_list = []
        t_basic_arrays = []
        t_stag_arrays = []
        s_final_arrays = []

        for idx, eps in enumerate(epsilons):
            cfg = Config.from_file(str(cfg_path))
            cfg.solver.end_time = end_time
            cfg.energy.kappah_floor = kappah_floor
            cfg.phase_solid.viscosity = viscosity_solid
            cfg.phase_solid.rheology = rheo

            s0_pert = s_init * (1.0 + eps)
            t0 = time.perf_counter()
            solver = EntropySolver(cfg, entropy_eos=eos_np)
            solver.initialize()
            solver.set_initial_entropy(s0_pert)
            solver.solve()
            dt_run = time.perf_counter() - t0

            if enabled:
                solver._check_rheology_invariants()

            if not r_basic_m:
                r_basic_m = [
                    float(x) for x in np.asarray(solver.evaluator.mesh.basic.radii).ravel()
                ]
                r_stag_m = [
                    float(x) for x in np.asarray(solver.evaluator.mesh.staggered.radii).ravel()
                ]

            st = solver.get_state()
            s_final = np.asarray(st.S_final, dtype=float).copy()
            y_end = np.asarray(solver._solution.y[:, -1], dtype=float).copy()
            solver.dSdt(end_time, y_end)
            t_basic = np.asarray(solver.state.phase_basic.temperature(), dtype=float).copy()
            t_stag = np.asarray(solver.state.phase_staggered.temperature(), dtype=float).copy()
            n_steps = len(solver._solution.t)

            if eps == 0.0:
                solvers_k0[key] = solver

            t_surf = float(t_basic[-1])
            t_cmb = float(t_basic[0])
            t_stag_mean = float(np.mean(t_stag))

            t_surf_list.append(t_surf)
            t_cmb_list.append(t_cmb)
            t_stag_mean_list.append(t_stag_mean)
            t_basic_arrays.append(t_basic)
            t_stag_arrays.append(t_stag)
            s_final_arrays.append(s_final)

            runs.append(
                {
                    'index': idx,
                    'epsilon': eps,
                    's0_perturbed': s0_pert,
                    'wall_time_s': dt_run,
                    'n_steps': n_steps,
                    'T_surface_K': t_surf,
                    'T_cmb_K': t_cmb,
                    'T_stag_mean_K': t_stag_mean,
                    'S_final_mean': float(np.mean(s_final)),
                    'T_basic_K': [float(x) for x in t_basic],
                    'T_stag_K': [float(x) for x in t_stag],
                    'S_final': [float(x) for x in s_final],
                }
            )

        tb_stack = np.vstack(t_basic_arrays)
        ts_stack = np.vstack(t_stag_arrays)
        sf_stack = np.vstack(s_final_arrays)
        delta_t_basic_profile = np.ptp(tb_stack, axis=0)
        delta_t_stag_profile = np.ptp(ts_stack, axis=0)
        delta_s_final_profile = np.ptp(sf_stack, axis=0)

        suite_out[key] = {
            'probe_1pct_end_time_yr': cfg_probe.solver.end_time,
            'probe_1pct_wall_time_s': dt_probe,
            'probe_1pct_n_steps': steps_probe,
            'end_time_yr': end_time,
            's_init': s_init,
            'kappah_floor': kappah_floor,
            'viscosity_solid': viscosity_solid,
            'r_basic_m': r_basic_m,
            'r_stag_m': r_stag_m,
            'runs': runs,
            'delta_T_basic_profile_K': [float(x) for x in delta_t_basic_profile],
            'delta_T_stag_profile_K': [float(x) for x in delta_t_stag_profile],
            'delta_S_final_profile': [float(x) for x in delta_s_final_profile],
            'spread_T_surface_ptp_K': float(np.ptp(t_surf_list)),
            'spread_T_surface_std_K': float(np.std(t_surf_list)),
            'spread_T_cmb_ptp_K': float(np.ptp(t_cmb_list)),
            'spread_T_cmb_std_K': float(np.std(t_cmb_list)),
            'spread_T_stag_mean_ptp_K': float(np.ptp(t_stag_mean_list)),
            'spread_T_stag_mean_std_K': float(np.std(t_stag_mean_list)),
            'spread_T_basic_max_node_ptp_K': float(np.max(delta_t_basic_profile)),
            'spread_T_stag_max_node_ptp_K': float(np.max(delta_t_stag_profile)),
            'spread_S_final_max_node_ptp': float(np.max(delta_s_final_profile)),
            'mean_wall_time_s': float(np.mean([r['wall_time_s'] for r in runs])),
        }

    sol_off = solvers_k0['off']
    sol_on = solvers_k0['on']
    s_end_off = np.asarray(sol_off.get_state().S_final, dtype=float)
    s_end_on = np.asarray(sol_on.get_state().S_final, dtype=float)
    y_end_off = np.asarray(sol_off._solution.y[:, -1], dtype=float).copy()
    y_end_on = np.asarray(sol_on._solution.y[:, -1], dtype=float).copy()
    rhs_end_off = sol_off.dSdt(end_time, y_end_off).copy()
    tb_end_off = np.asarray(sol_off.state.phase_basic.temperature(), dtype=float).copy()
    ts_end_off = np.asarray(sol_off.state.phase_staggered.temperature(), dtype=float).copy()
    visc_end_off = sol_off.state.phase_basic.viscosity().copy()
    kh_end_off = sol_off.state.eddy_diffusivity.copy()

    rhs_end_on_at_on = sol_on.dSdt(end_time, y_end_on).copy()
    tb_end_on = np.asarray(sol_on.state.phase_basic.temperature(), dtype=float).copy()
    ts_end_on = np.asarray(sol_on.state.phase_staggered.temperature(), dtype=float).copy()
    visc_end_on = sol_on.state.phase_basic.viscosity().copy()
    eta_eff_end = sol_on.state._eta_eff_basic.copy()
    eta_diff_end = sol_on.state.phase_basic.eta_diff().copy()
    phi_end = sol_on.state.phase_basic.melt_fraction().copy()
    kh_end_on = sol_on.state.eddy_diffusivity.copy()

    rhs_end_on_same_state = sol_on.dSdt(end_time, y_end_off).copy()
    denom_rhs_off = max(float(np.max(np.abs(rhs_end_off))), 1.0e-30)

    phi_rheo = float(sol_on.parameters.phase_mixed.rheological_transition_melt_fraction)
    phi_width = float(sol_on.parameters.phase_mixed.rheological_transition_width)
    w_melt = 0.5 * (1.0 + np.tanh((phi_end - phi_rheo) / phi_width))
    solid_weight = 1.0 - w_melt

    suite_out['on_vs_off_activity'] = {
        'identical_final_entropy': bool(np.array_equal(s_end_on, s_end_off)),
        'identical_final_T_basic': bool(np.array_equal(tb_end_on, tb_end_off)),
        'identical_final_T_stag': bool(np.array_equal(ts_end_on, ts_end_off)),
        'max_rel_diff_S_final_on_vs_off': float(
            np.max(np.abs(s_end_on - s_end_off))
            / max(float(np.max(np.abs(s_end_off))), 1.0e-30)
        ),
        'max_rel_diff_T_basic_on_vs_off': float(
            np.max(np.abs(tb_end_on - tb_end_off))
            / max(float(np.max(np.abs(tb_end_off))), 1.0e-30)
        ),
        'max_abs_diff_T_basic_on_vs_off_K': float(np.max(np.abs(tb_end_on - tb_end_off))),
        'max_rel_diff_T_stag_on_vs_off': float(
            np.max(np.abs(ts_end_on - ts_end_off))
            / max(float(np.max(np.abs(ts_end_off))), 1.0e-30)
        ),
        'max_abs_diff_T_stag_on_vs_off_K': float(np.max(np.abs(ts_end_on - ts_end_off))),
        'max_rel_rhs_on_vs_off_at_t_end': float(
            np.max(np.abs(rhs_end_on_at_on - rhs_end_off)) / denom_rhs_off
        ),
        'max_rel_rhs_on_vs_off_same_state_t_end': float(
            np.max(np.abs(rhs_end_on_same_state - rhs_end_off)) / denom_rhs_off
        ),
        'n_basic_nodes_total': int(sol_on.evaluator.mesh.basic.number_of_nodes),
        'n_nodes_solid_weight_pos': int(np.sum(solid_weight > 0.0)),
        'n_nodes_eta_eff_ne_eta_diff': int(np.sum(eta_eff_end != eta_diff_end)),
        'n_nodes_visc_on_ne_visc_off': int(
            np.sum((visc_end_on != visc_end_off) & (solid_weight > 0.0))
        ),
        'n_nodes_kappah_on_ne_kappah_off': int(np.sum(kh_end_on != kh_end_off)),
    }
    suite_out['solve_wall_time_ratio_on_over_off'] = (
        suite_out['on']['mean_wall_time_s'] / suite_out['off']['mean_wall_time_s']
    )
    if table_hashes is not None:
        suite_out['table_md5'] = dict(table_hashes)
    return suite_out


def run_perturbed_starts_benchmarks(
    cfg_path: Path, eos_dir: Path, table_hashes: dict[str, str]
) -> dict[str, Any]:
    """Generate the full perturbed_starts block on the specified EOS tables."""
    eos_np = entropy_eos_copy(eos_dir)
    epsilons = [-2.0e-12, -1.0e-12, 0.0, 1.0e-12, 2.0e-12]
    starts_default = _run_perturbed_suite(
        cfg_path,
        eos_np,
        s_init=2900.0,
        end_time=100.0,
        kappah_floor=10.0,
        viscosity_solid=1.0e21,
        epsilons=epsilons,
        table_hashes=table_hashes,
    )
    starts_active_solid = _run_perturbed_suite(
        cfg_path,
        eos_np,
        s_init=2500.0,
        end_time=100.0,
        kappah_floor=0.0,
        viscosity_solid=1.0e18,
        epsilons=epsilons,
        table_hashes=table_hashes,
    )
    return {
        'epsilons': epsilons,
        'abe_solid_default': starts_default,
        'abe_solid_kappah_floor_0': starts_active_solid,
    }


def _verify_live_fast_subset(data: dict[str, Any], eos_dir: Path) -> list[str]:
    """Recompute the fast live subset (call counts and state-0 JIT/eager parity) against JSON."""
    import jax
    import jax.numpy as jnp

    from aragog.config import Config
    from aragog.jax.solver import dSdt
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    jax.config.update('jax_enable_x64', True)
    failures: list[str] = []
    cfg_path = Path(str(importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')))
    eos_np = entropy_eos_copy(eos_dir)
    eos_jax = entropy_eos_jax(eos_dir)

    modes = [
        ('off', SolidRheologyParams(enabled=False)),
        ('on_lid', SolidRheologyParams(enabled=True, stress_closure_mode='lid')),
        ('on_local', SolidRheologyParams(enabled=True, stress_closure_mode='local')),
    ]
    call_counts = {
        label: measure_per_rhs_call_counts(cfg_path, eos_np, eos_jax, rheo)
        for label, rheo in modes
    }
    live_rows = build_six_row_call_count_table(call_counts)
    live_np_lid = [r['measured_numpy_delta'] for r in live_rows]
    live_jax_lid = [r['measured_jax_delta'] for r in live_rows]
    live_np_loc = [r['measured_numpy_local_delta'] for r in live_rows]
    live_jax_loc = [r['measured_jax_local_delta'] for r in live_rows]
    if live_np_lid != [0, 2, 2, 1, 1, 2] or live_jax_lid != [0, 2, 2, 1, 1, 2]:
        failures.append(f'Live lid call-count deltas NP={live_np_lid} JAX={live_jax_lid}')
    if live_np_loc != [0, 1, 1, 0, 1, 2] or live_jax_loc != [0, 1, 1, 0, 1, 2]:
        failures.append(f'Live local call-count deltas NP={live_np_loc} JAX={live_jax_loc}')

    def _make_solver(
        kappah_floor: float, viscosity: float, rheo: SolidRheologyParams
    ) -> EntropySolver:
        cfg = Config.from_file(str(cfg_path))
        cfg.boundary_conditions.inner_boundary_condition = 1
        cfg.boundary_conditions.core_bc = 'quasi_steady'
        cfg.energy.kappah_floor = kappah_floor
        cfg.phase_solid.viscosity = viscosity
        cfg.phase_solid.rheology = rheo
        sol = EntropySolver(cfg, entropy_eos=eos_np)
        sol.initialize()
        return sol

    solver_on = _make_solver(
        0.0,
        1.0e18,
        SolidRheologyParams(
            enabled=True,
            stress_closure_mode='lid',
            yield_stress_c=1.0e7,
            yield_stress_mu=0.1,
            yield_stress_max=1.0e8,
        ),
    )
    n_stag = solver_on._n_stag
    xi = np.linspace(0.0, 1.0, n_stag)
    s_profile = (2500.0 - 450.0 * (0.2 * xi + 0.8 * xi**4)).copy()
    s_jax = jnp.asarray(s_profile)
    rhs_np = solver_on.dSdt(0.0, s_profile).copy()
    _, _, _, _, args_on = _build_jax_bundle(solver_on, eos_jax)
    jit_dSdt = jax.jit(lambda t, s: dSdt(t, s, args_on))
    rhs_jax_jit = np.asarray(jit_dSdt(0.0, s_jax))
    rhs_jax_eager = np.asarray(dSdt(0.0, s_jax, args_on))
    denom = max(float(np.max(np.abs(rhs_np))), 1.0e-30)
    rel_diff_0_jit = float(np.max(np.abs(rhs_np - rhs_jax_jit)) / denom)
    rel_diff_0_eager = float(np.max(np.abs(rhs_np - rhs_jax_eager)) / denom)

    kappah_rec = data['rhs_parity_and_activity']['abe_solid_kappah_floor_0']
    stored_0_jit = float(kappah_rec['rel_rhs_numpy_vs_jax_per_state'][0])
    tol_0_jit = 1.0e-3 * max(stored_0_jit, 1.0e-16)
    if rel_diff_0_jit > 1.0e-6 or abs(rel_diff_0_jit - stored_0_jit) > tol_0_jit:
        failures.append(
            f'Live state-0 JIT rel_rhs_numpy_vs_jax {rel_diff_0_jit:.6e} vs stored '
            f'{stored_0_jit:.6e} exceeds tolerance {tol_0_jit:.3e}'
        )
    stored_0_eager = float(kappah_rec['rel_rhs_numpy_vs_jax_eager_per_state'][0])
    if rel_diff_0_eager > 1.0e-13:
        failures.append(
            f'Live state-0 eager rel_rhs_numpy_vs_jax {rel_diff_0_eager:.6e} vs stored '
            f'{stored_0_eager:.6e} exceeds upper bound 1.0e-13'
        )

    def _state0_rel_diff(
        s_bot: float,
        delta_s: float,
        kappah_floor: float,
        viscosity: float,
        rheo_a: SolidRheologyParams,
        rheo_b: SolidRheologyParams,
    ) -> float:
        sol_a = _make_solver(kappah_floor, viscosity, rheo_a)
        sol_b = _make_solver(kappah_floor, viscosity, rheo_b)
        prof = (s_bot - delta_s * (0.2 * xi + 0.8 * xi**4)).copy()
        rhs_a = sol_a.dSdt(0.0, prof).copy()
        rhs_b = sol_b.dSdt(0.0, prof).copy()
        return float(np.max(np.abs(rhs_a - rhs_b)) / max(float(np.max(np.abs(rhs_b))), 1.0e-30))

    # Also evaluate state 0 of abe_solid_yield_active and abe_solid_default so a merge
    # carrying another branch's JSON fails even when git history is unavailable.
    rheo_y_on = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
    )
    live_y_on_off_0 = _state0_rel_diff(
        2150.0,
        1820.0,
        0.0,
        3.0e17,
        rheo_y_on,
        replace(rheo_y_on, yield_stress_c=1.0e30, yield_stress_max=1.0e30),
    )
    stored_y_on_off_0 = float(
        data['rhs_parity_and_activity']['abe_solid_yield_active'][
            'rel_rhs_yield_on_vs_yield_off_per_state'
        ][0]
    )
    _check_scalar_match(
        failures,
        'abe_solid_yield_active.state_0_rel_rhs_yield_on_vs_yield_off',
        stored_y_on_off_0,
        live_y_on_off_0,
        rtol=1.0e-6,
    )

    live_d_on_off_0 = _state0_rel_diff(
        2900.0,
        450.0,
        10.0,
        1.0e21,
        SolidRheologyParams(
            enabled=True,
            stress_closure_mode='lid',
            yield_stress_c=1.0e7,
            yield_stress_mu=0.1,
            yield_stress_max=1.0e8,
        ),
        SolidRheologyParams(enabled=False),
    )
    stored_d_on_off_0 = float(
        data['rhs_parity_and_activity']['abe_solid_default']['rel_rhs_on_vs_off_per_state'][0]
    )
    _check_scalar_match(
        failures,
        'abe_solid_default.state_0_rel_rhs_on_vs_off',
        stored_d_on_off_0,
        live_d_on_off_0,
        rtol=1.0e-6,
    )
    return failures


def _check_scalar_match(
    failures: list[str],
    label: str,
    stored: Any,
    expected: float,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-24,
) -> None:
    """Append a failure if ``stored`` is not a finite float matching ``expected``."""
    if not isinstance(stored, (int, float)) or not np.isfinite(float(stored)):
        failures.append(f'{label}: non-finite or missing value {stored!r}')
        return
    s_val = float(stored)
    tol = atol + rtol * max(abs(expected), abs(s_val))
    if abs(s_val - expected) > tol:
        failures.append(
            f'{label}: stored {s_val:.6e} != recomputed {expected:.6e} (tol {tol:.3e})'
        )


def _check_table_md5_dict(
    failures: list[str], label: str, expected_hashes: Any, actual_hashes: dict[str, str]
) -> None:
    """Verify a stored ``table_md5`` dictionary against ``actual_hashes``."""
    if not isinstance(expected_hashes, dict) or tuple(expected_hashes.keys()) != TABLE_FILES:
        failures.append(
            f'{label} keys {tuple(expected_hashes.keys()) if isinstance(expected_hashes, dict) else None} != {TABLE_FILES}'
        )
        return
    for fname, expected_hash in expected_hashes.items():
        actual_hash = actual_hashes.get(fname)
        if actual_hash != expected_hash:
            failures.append(
                f'{label} hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}'
            )


def _verify_parity_block(
    parity_block: dict[str, Any], actual_hashes: dict[str, str] | None
) -> list[str]:
    """Recompute and verify all summary statistics and invariants in ``rhs_parity_and_activity``."""
    failures: list[str] = []
    for req_key in EXPECTED_PARITY_KEYS:
        if not isinstance(parity_block.get(req_key), dict):
            failures.append(f"Missing '{req_key}' in parity block")
    if failures:
        return failures

    for suite_name in (
        'abe_solid_kappah_floor_0',
        'abe_solid_yield_active',
        'abe_solid_default',
    ):
        suite = parity_block[suite_name]
        if actual_hashes is not None:
            _check_table_md5_dict(
                failures, f"{suite_name}['table_md5']", suite.get('table_md5'), actual_hashes
            )
        if suite.get('n_states') != 100 or suite.get('n_basic_nodes_total') != 100:
            failures.append(
                f"'{suite_name}' n_states={suite.get('n_states')} or "
                f'n_basic_nodes_total={suite.get("n_basic_nodes_total")} != 100'
            )
        d_min = suite.get('delta_s_min')
        d_max = suite.get('delta_s_max')
        if (
            not isinstance(d_min, (int, float))
            or not isinstance(d_max, (int, float))
            or float(d_min) > float(d_max)
        ):
            failures.append(f"'{suite_name}' delta_s_min={d_min} > delta_s_max={d_max}")

        arr_specs = (
            (
                'rel_rhs_numpy_vs_jax_per_state',
                (
                    ('max_rel_rhs_numpy_vs_jax', np.max),
                    ('median_rel_rhs_numpy_vs_jax', np.median),
                    ('min_rel_rhs_numpy_vs_jax', np.min),
                ),
            ),
            (
                'rel_rhs_numpy_vs_jax_eager_per_state',
                (
                    ('max_rel_rhs_numpy_vs_jax_eager', np.max),
                    ('median_rel_rhs_numpy_vs_jax_eager', np.median),
                    ('min_rel_rhs_numpy_vs_jax_eager', np.min),
                ),
            ),
            (
                'rel_rhs_jax_jit_vs_eager_per_state',
                (('max_rel_rhs_jax_jit_vs_eager', np.max),),
            ),
            (
                'rel_flux_numpy_vs_jax_per_state',
                (
                    ('max_rel_flux_numpy_vs_jax', np.max),
                    ('median_rel_flux_numpy_vs_jax', np.median),
                ),
            ),
            (
                'rel_rhs_numpy_vs_jax_harmonised_alpha_per_state',
                (
                    ('max_rel_rhs_numpy_vs_jax_harmonised_alpha', np.max),
                    ('median_rel_rhs_numpy_vs_jax_harmonised_alpha', np.median),
                    ('min_rel_rhs_numpy_vs_jax_harmonised_alpha', np.min),
                ),
            ),
            (
                'rel_rhs_numpy_vs_jax_harmonised_alpha_eager_per_state',
                (
                    ('max_rel_rhs_numpy_vs_jax_harmonised_alpha_eager', np.max),
                    ('median_rel_rhs_numpy_vs_jax_harmonised_alpha_eager', np.median),
                ),
            ),
            (
                'rel_flux_numpy_vs_jax_harmonised_alpha_per_state',
                (
                    ('max_rel_flux_numpy_vs_jax_harmonised_alpha', np.max),
                    ('median_rel_flux_numpy_vs_jax_harmonised_alpha', np.median),
                ),
            ),
            (
                'rel_rhs_on_vs_off_per_state',
                (
                    ('max_rel_rhs_on_vs_off', np.max),
                    ('min_rel_rhs_on_vs_off', np.min),
                    ('median_rel_rhs_on_vs_off', np.median),
                ),
            ),
            (
                'rel_flux_on_vs_off_per_state',
                (
                    ('max_rel_flux_on_vs_off', np.max),
                    ('min_rel_flux_on_vs_off', np.min),
                ),
            ),
            (
                'rel_rhs_yield_on_vs_yield_off_per_state',
                (
                    ('max_rel_rhs_yield_on_vs_yield_off', np.max),
                    ('min_rel_rhs_yield_on_vs_yield_off', np.min),
                    ('median_rel_rhs_yield_on_vs_yield_off', np.median),
                ),
            ),
            (
                'canary_jax_broken_rel_rhs_per_state',
                (
                    ('canary_jax_broken_max_rel_rhs_diff', np.max),
                    ('canary_jax_broken_min_rel_rhs_diff', np.min),
                ),
            ),
            (
                'canary_jax_yield_broken_rel_rhs_per_state',
                (
                    ('canary_jax_yield_broken_max_rel_rhs_diff', np.max),
                    ('canary_jax_yield_broken_min_rel_rhs_diff', np.min),
                ),
            ),
        )
        for arr_key, stat_specs in arr_specs:
            arr = suite.get(arr_key)
            if not isinstance(arr, list) or len(arr) != 100:
                failures.append(f"'{suite_name}.{arr_key}' missing or length != 100")
                continue
            arr_np = np.asarray(arr, dtype=float)
            if not np.all(np.isfinite(arr_np)) or np.any(arr_np < 0.0):
                failures.append(
                    f"'{suite_name}.{arr_key}' contains non-finite or negative entries"
                )
                continue
            for stat_key, stat_fn in stat_specs:
                _check_scalar_match(
                    failures,
                    f'{suite_name}.{stat_key}',
                    suite.get(stat_key),
                    float(stat_fn(arr_np)),
                )

        traj_arr = suite.get('rel_rhs_on_vs_off_integrated_trajectory_per_step')
        if not isinstance(traj_arr, list) or len(traj_arr) == 0:
            failures.append(
                f"'{suite_name}.rel_rhs_on_vs_off_integrated_trajectory_per_step' missing or empty"
            )
        else:
            traj_np = np.asarray(traj_arr, dtype=float)
            if not np.all(np.isfinite(traj_np)) or np.any(traj_np < 0.0):
                failures.append(
                    f"'{suite_name}.rel_rhs_on_vs_off_integrated_trajectory_per_step' invalid"
                )
            else:
                _check_scalar_match(
                    failures,
                    f'{suite_name}.max_rel_rhs_on_vs_off_integrated_trajectory',
                    suite.get('max_rel_rhs_on_vs_off_integrated_trajectory'),
                    float(np.max(traj_np)),
                )

        for node_arr_key, min_key, max_key in (
            (
                'n_nodes_solid_weight_pos_per_state',
                'n_nodes_solid_weight_pos_min',
                'n_nodes_solid_weight_pos_max',
            ),
            (
                'n_nodes_eta_eff_ne_eta_diff_per_state',
                'n_nodes_eta_eff_ne_eta_diff_min',
                'n_nodes_eta_eff_ne_eta_diff_max',
            ),
            (
                'n_nodes_visc_on_ne_visc_off_per_state',
                'n_nodes_visc_on_ne_visc_off_min',
                'n_nodes_visc_on_ne_visc_off_max',
            ),
            (
                'n_nodes_kappah_on_ne_kappah_off_per_state',
                'n_nodes_kappah_on_ne_kappah_off_min',
                'n_nodes_kappah_on_ne_kappah_off_max',
            ),
        ):
            nodes_arr = suite.get(node_arr_key)
            if not isinstance(nodes_arr, list) or len(nodes_arr) != 100:
                failures.append(f"'{suite_name}.{node_arr_key}' invalid")
            else:
                if suite.get(min_key) != min(nodes_arr):
                    failures.append(f"'{suite_name}.{min_key}' mismatch")
                if suite.get(max_key) != max(nodes_arr):
                    failures.append(f"'{suite_name}.{max_key}' mismatch")
                if node_arr_key == 'n_nodes_eta_eff_ne_eta_diff_per_state' and suite.get(
                    'n_states_eta_eff_ne_eta_diff_pos'
                ) != sum(x > 0 for x in nodes_arr):
                    failures.append(f"'{suite_name}.n_states_eta_eff_ne_eta_diff_pos' mismatch")

        if float(suite.get('max_rel_rhs_numpy_vs_jax', 1.0)) > 1.0e-6:
            failures.append(
                f"'{suite_name}' max_rel_rhs_numpy_vs_jax {suite.get('max_rel_rhs_numpy_vs_jax')} > 1e-6"
            )
        if float(suite.get('max_rel_rhs_numpy_vs_jax_harmonised_alpha', 1.0)) > 1.0e-9:
            failures.append(
                f"'{suite_name}' max_rel_rhs_numpy_vs_jax_harmonised_alpha "
                f'{suite.get("max_rel_rhs_numpy_vs_jax_harmonised_alpha")} > 1e-9'
            )

    kfl = parity_block['abe_solid_kappah_floor_0']
    if float(kfl.get('min_rel_rhs_on_vs_off', 0.0)) < 4.0e-3:
        failures.append('abe_solid_kappah_floor_0 min_rel_rhs_on_vs_off < 4e-3')
    if float(kfl.get('max_rel_rhs_on_vs_off', 1.0)) > 2.0e-2:
        failures.append('abe_solid_kappah_floor_0 max_rel_rhs_on_vs_off > 2e-2')
    if float(kfl.get('canary_jax_broken_min_rel_rhs_diff', 0.0)) < 1.0e-3:
        failures.append('abe_solid_kappah_floor_0 canary_jax_broken_min_rel_rhs_diff < 1e-3')

    yld = parity_block['abe_solid_yield_active']
    min_nodes = yld.get('n_nodes_eta_eff_ne_eta_diff_min')
    max_nodes = yld.get('n_nodes_eta_eff_ne_eta_diff_max')
    if min_nodes != 9 or max_nodes != 13:
        failures.append(f'Yield-active node count span ({min_nodes}, {max_nodes}) != (9, 13)')
    # Yielding effect must exceed 10x the 1e-6 parity tolerance across all 100 states.
    min_y_on_off = float(yld.get('min_rel_rhs_yield_on_vs_yield_off', 0.0))
    if min_y_on_off < 1.0e-5:
        failures.append(f'Yield-active min on vs off {min_y_on_off:.3e} below 1.0e-5')
    if float(yld.get('canary_jax_yield_broken_min_rel_rhs_diff', 0.0)) < 1.0e-5:
        failures.append('Yield-active canary_jax_yield_broken_min_rel_rhs_diff below 1.0e-5')

    fdiag = parity_block['flux_vs_rhs_diagnostic']
    if actual_hashes is not None:
        _check_table_md5_dict(
            failures,
            "flux_vs_rhs_diagnostic['table_md5']",
            fdiag.get('table_md5'),
            actual_hashes,
        )
    _check_scalar_match(
        failures,
        'flux_vs_rhs_diagnostic.max_rel_flux_numpy_vs_jax',
        fdiag.get('max_rel_flux_numpy_vs_jax'),
        float(kfl.get('max_rel_flux_numpy_vs_jax', -1.0)),
    )
    _check_scalar_match(
        failures,
        'flux_vs_rhs_diagnostic.max_rel_rhs_numpy_vs_jax',
        fdiag.get('max_rel_rhs_numpy_vs_jax'),
        float(kfl.get('max_rel_rhs_numpy_vs_jax', -1.0)),
    )
    p_pa = float(fdiag.get('pressure_basic_Pa', 0.0))
    _check_scalar_match(
        failures,
        'flux_vs_rhs_diagnostic.pressure_basic_GPa',
        fdiag.get('pressure_basic_GPa'),
        p_pa / 1.0e9,
    )
    a_np = float(fdiag.get('alpha_basic_1_K_np_table', 0.0))
    a_jx = float(fdiag.get('alpha_basic_1_K_jax_identity', 0.0))
    if a_np != 0.0:
        _check_scalar_match(
            failures,
            'flux_vs_rhs_diagnostic.alpha_rel_diff_np_vs_jax',
            fdiag.get('alpha_rel_diff_np_vs_jax'),
            abs(a_np - a_jx) / abs(a_np),
        )
    f_np = float(fdiag.get('heat_flux_basic_W_m2_np', 0.0))
    f_jx = float(fdiag.get('heat_flux_basic_W_m2_jax', 0.0))
    f_den = float(fdiag.get('denom_flux_interior_max_W_m2', 0.0))
    _check_scalar_match(
        failures,
        'flux_vs_rhs_diagnostic.abs_flux_diff_W_m2',
        fdiag.get('abs_flux_diff_W_m2'),
        abs(f_np - f_jx),
    )
    if f_den > 0.0:
        _check_scalar_match(
            failures,
            'flux_vs_rhs_diagnostic.max_rel_flux_ratio',
            fdiag.get('max_rel_flux_numpy_vs_jax'),
            abs(f_np - f_jx) / f_den,
        )
    return failures


def _verify_cost_block(
    cost_block: dict[str, Any], actual_hashes: dict[str, str] | None
) -> list[str]:
    """Recompute and verify all fields in ``rhs_wall_time``."""
    failures: list[str] = []
    for req_key in EXPECTED_COST_KEYS:
        if req_key not in cost_block:
            failures.append(f"Missing '{req_key}' in cost block")
    if failures:
        return failures

    modes = cost_block.get('modes')
    expected_modes = ('off', 'on_lid', 'on_local')
    if not isinstance(modes, dict) or tuple(modes.keys()) != expected_modes:
        failures.append(
            f"cost_block['modes'] keys {tuple(modes.keys()) if isinstance(modes, dict) else None} != {expected_modes}"
        )
        return failures

    call_counts: dict[str, dict] = {}
    for m_key in expected_modes:
        m_rec = modes[m_key]
        if not isinstance(m_rec, dict):
            failures.append(f"modes['{m_key}'] is not a dict")
            continue
        if actual_hashes is not None:
            _check_table_md5_dict(
                failures,
                f"modes['{m_key}']['table_md5']",
                m_rec.get('table_md5'),
                actual_hashes,
            )
        n_eval = m_rec.get('n_eval')
        if n_eval != 1000:
            failures.append(f"modes['{m_key}'].n_eval={n_eval} != 1000")
            continue
        if m_rec.get('n_repeats') != 7:
            failures.append(f"modes['{m_key}'].n_repeats={m_rec.get('n_repeats')} != 7")
        dt_np = float(m_rec.get('numpy_total_s', -1.0))
        dt_jx = float(m_rec.get('jax_total_s', -1.0))
        if dt_np <= 0.0 or dt_jx <= 0.0:
            failures.append(f"modes['{m_key}'] non-positive total time np={dt_np} jax={dt_jx}")
        else:
            _check_scalar_match(
                failures,
                f"modes['{m_key}'].numpy_per_rhs_us",
                m_rec.get('numpy_per_rhs_us'),
                (dt_np / n_eval) * 1.0e6,
                rtol=1.0e-9,
            )
            _check_scalar_match(
                failures,
                f"modes['{m_key}'].jax_per_rhs_us",
                m_rec.get('jax_per_rhs_us'),
                (dt_jx / n_eval) * 1.0e6,
                rtol=1.0e-9,
            )
        rel_d = m_rec.get('numpy_vs_jax_max_rel_diff')
        if (
            not isinstance(rel_d, (int, float))
            or not np.isfinite(float(rel_d))
            or float(rel_d) <= 0.0
            or float(rel_d) > 1.0e-6
        ):
            failures.append(
                f"modes['{m_key}'].numpy_vs_jax_max_rel_diff={rel_d} outside (0, 1e-6]"
            )
        cc = m_rec.get('call_counts')
        if (
            not isinstance(cc, dict)
            or not isinstance(cc.get('numpy'), dict)
            or not isinstance(cc.get('jax'), dict)
        ):
            failures.append(f"modes['{m_key}'].call_counts invalid")
        else:
            call_counts[m_key] = cc

    ratios = cost_block.get('ratios', {})
    if not isinstance(ratios, dict):
        failures.append("cost_block['ratios'] is not a dict")
    elif len(call_counts) == 3:
        expected_ratios = {
            'numpy_ratio_on_lid_over_off': modes['on_lid']['numpy_total_s']
            / modes['off']['numpy_total_s'],
            'numpy_ratio_on_local_over_off': modes['on_local']['numpy_total_s']
            / modes['off']['numpy_total_s'],
            'jax_ratio_on_lid_over_off': modes['on_lid']['jax_total_s']
            / modes['off']['jax_total_s'],
            'jax_ratio_on_local_over_off': modes['on_local']['jax_total_s']
            / modes['off']['jax_total_s'],
        }
        for r_key, exp_val in expected_ratios.items():
            _check_scalar_match(
                failures, f'ratios.{r_key}', ratios.get(r_key), float(exp_val), rtol=1.0e-9
            )
        for ratio_key in ('jax_ratio_on_lid_over_off', 'jax_ratio_on_local_over_off'):
            val = ratios.get(ratio_key)
            # 1.5 guards against a gross regression; it is not a performance target
            if val is None or float(val) > 1.5 or float(val) < 0.5:
                failures.append(f'Ratio {ratio_key} value {val} outside [0.5, 1.5]')
        for ratio_key in ('numpy_ratio_on_lid_over_off', 'numpy_ratio_on_local_over_off'):
            val = ratios.get(ratio_key)
            if val is None or float(val) > 2.5 or float(val) < 0.8:
                failures.append(f'Ratio {ratio_key} value {val} outside [0.8, 2.5]')

    table = cost_block.get('six_row_call_count_table', [])
    if len(table) != 6:
        failures.append(f'Call count table has {len(table)} rows, expected 6')
    else:
        expected_deltas = [0, 2, 2, 1, 1, 2]
        expected_local_deltas = [0, 1, 1, 0, 1, 2]
        actual_np = [row.get('measured_numpy_delta') for row in table]
        actual_jax = [row.get('measured_jax_delta') for row in table]
        actual_np_loc = [row.get('measured_numpy_local_delta') for row in table]
        actual_jax_loc = [row.get('measured_jax_local_delta') for row in table]
        if actual_np != expected_deltas or actual_jax != expected_deltas:
            failures.append(
                f'Stored call count deltas NP={actual_np} JAX={actual_jax} != {expected_deltas}'
            )
        if actual_np_loc != expected_local_deltas or actual_jax_loc != expected_local_deltas:
            failures.append(
                f'Stored local call count deltas NP={actual_np_loc} '
                f'JAX={actual_jax_loc} != {expected_local_deltas}'
            )
        if len(call_counts) == 3:
            recomp_table = build_six_row_call_count_table(call_counts)
            for idx, (row, exp_row) in enumerate(zip(table, recomp_table)):
                if row.get('difference_from_expected') != 0:
                    failures.append(
                        f'six_row_call_count_table[{idx}].difference_from_expected='
                        f'{row.get("difference_from_expected")!r} != 0'
                    )
                for fld in (
                    'item',
                    'expected_call_delta',
                    'measured_numpy_delta',
                    'measured_jax_delta',
                    'measured_numpy_local_delta',
                    'measured_jax_local_delta',
                    'difference_from_expected',
                ):
                    if row.get(fld) != exp_row[fld]:
                        failures.append(
                            f'six_row_call_count_table[{idx}].{fld}={row.get(fld)!r} != {exp_row[fld]!r}'
                        )
    return failures


def _verify_starts_block(
    starts_block: dict[str, Any], actual_hashes: dict[str, str] | None
) -> list[str]:
    """Recompute and verify all profiles, spread scalars, and activity metrics in ``perturbed_starts``."""
    failures: list[str] = []
    for req_key in EXPECTED_STARTS_KEYS:
        if req_key not in starts_block:
            failures.append(f"Missing '{req_key}' in starts block")
    if failures:
        return failures

    epsilons = starts_block.get('epsilons')
    expected_eps = [-2.0e-12, -1.0e-12, 0.0, 1.0e-12, 2.0e-12]
    if epsilons != expected_eps:
        failures.append(f'Perturbation epsilons {epsilons} != {expected_eps}')
        return failures

    for suite_key in ('abe_solid_default', 'abe_solid_kappah_floor_0'):
        suite = starts_block.get(suite_key)
        if not isinstance(suite, dict):
            failures.append(f"Invalid or missing '{suite_key}' dictionary in starts block")
            continue
        if actual_hashes is not None:
            _check_table_md5_dict(
                failures, f"{suite_key}['table_md5']", suite.get('table_md5'), actual_hashes
            )
        for mode_key in ('off', 'on'):
            m_rec = suite.get(mode_key)
            if not isinstance(m_rec, dict):
                failures.append(f"Missing '{suite_key}.{mode_key}' dict")
                continue
            runs = m_rec.get('runs')
            if not isinstance(runs, list) or len(runs) != 5:
                failures.append(f"'{suite_key}.{mode_key}.runs' missing or length != 5")
                continue
            s_init = float(m_rec.get('s_init', 0.0))
            tb_list, ts_list, sf_list = [], [], []
            tsurf_list, tcmb_list, tsmean_list, wtime_list = [], [], [], []
            for idx, r_rec in enumerate(runs):
                if r_rec.get('index') != idx or r_rec.get('epsilon') != expected_eps[idx]:
                    failures.append(
                        f"'{suite_key}.{mode_key}.runs[{idx}]' index/epsilon mismatch"
                    )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.{mode_key}.runs[{idx}].s0_perturbed',
                    r_rec.get('s0_perturbed'),
                    s_init * (1.0 + expected_eps[idx]),
                )
                tb = np.asarray(r_rec.get('T_basic_K', []), dtype=float)
                ts = np.asarray(r_rec.get('T_stag_K', []), dtype=float)
                sf = np.asarray(r_rec.get('S_final', []), dtype=float)
                if tb.shape != (100,) or ts.shape != (99,) or sf.shape != (99,):
                    failures.append(
                        f"'{suite_key}.{mode_key}.runs[{idx}]' profile shape mismatch "
                        f'tb={tb.shape} ts={ts.shape} sf={sf.shape}'
                    )
                    break
                _check_scalar_match(
                    failures,
                    f'{suite_key}.{mode_key}.runs[{idx}].T_surface_K',
                    r_rec.get('T_surface_K'),
                    float(tb[-1]),
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.{mode_key}.runs[{idx}].T_cmb_K',
                    r_rec.get('T_cmb_K'),
                    float(tb[0]),
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.{mode_key}.runs[{idx}].T_stag_mean_K',
                    r_rec.get('T_stag_mean_K'),
                    float(np.mean(ts)),
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.{mode_key}.runs[{idx}].S_final_mean',
                    r_rec.get('S_final_mean'),
                    float(np.mean(sf)),
                )
                tb_list.append(tb)
                ts_list.append(ts)
                sf_list.append(sf)
                tsurf_list.append(float(r_rec['T_surface_K']))
                tcmb_list.append(float(r_rec['T_cmb_K']))
                tsmean_list.append(float(r_rec['T_stag_mean_K']))
                wtime_list.append(float(r_rec['wall_time_s']))

            if len(tb_list) != 5:
                continue
            re_dtb = np.ptp(np.vstack(tb_list), axis=0)
            re_dts = np.ptp(np.vstack(ts_list), axis=0)
            re_dsf = np.ptp(np.vstack(sf_list), axis=0)
            for p_key, re_arr in (
                ('delta_T_basic_profile_K', re_dtb),
                ('delta_T_stag_profile_K', re_dts),
                ('delta_S_final_profile', re_dsf),
            ):
                st_arr = np.asarray(m_rec.get(p_key, []), dtype=float)
                if st_arr.shape != re_arr.shape or not np.allclose(
                    st_arr, re_arr, rtol=1.0e-9, atol=1.0e-15
                ):
                    failures.append(f"'{suite_key}.{mode_key}.{p_key}' does not match runs ptp")
            for s_key, exp_v in (
                ('spread_T_surface_ptp_K', float(np.ptp(tsurf_list))),
                ('spread_T_surface_std_K', float(np.std(tsurf_list))),
                ('spread_T_cmb_ptp_K', float(np.ptp(tcmb_list))),
                ('spread_T_cmb_std_K', float(np.std(tcmb_list))),
                ('spread_T_stag_mean_ptp_K', float(np.ptp(tsmean_list))),
                ('spread_T_stag_mean_std_K', float(np.std(tsmean_list))),
                ('spread_T_basic_max_node_ptp_K', float(np.max(re_dtb))),
                ('spread_T_stag_max_node_ptp_K', float(np.max(re_dts))),
                ('spread_S_final_max_node_ptp', float(np.max(re_dsf))),
                ('mean_wall_time_s', float(np.mean(wtime_list))),
            ):
                _check_scalar_match(
                    failures,
                    f'{suite_key}.{mode_key}.{s_key}',
                    m_rec.get(s_key),
                    exp_v,
                    rtol=1.0e-9,
                    atol=1.0e-15,
                )

        if isinstance(suite.get('on'), dict) and isinstance(suite.get('off'), dict):
            _check_scalar_match(
                failures,
                f'{suite_key}.solve_wall_time_ratio_on_over_off',
                suite.get('solve_wall_time_ratio_on_over_off'),
                float(suite['on']['mean_wall_time_s'])
                / float(suite['off']['mean_wall_time_s']),
                rtol=1.0e-9,
            )
            act = suite.get('on_vs_off_activity')
            if not isinstance(act, dict):
                failures.append(f"Missing '{suite_key}.on_vs_off_activity' dict")
            elif (
                len(suite['on'].get('runs', [])) == 5 and len(suite['off'].get('runs', [])) == 5
            ):
                r_on0 = suite['on']['runs'][2]
                r_off0 = suite['off']['runs'][2]
                tb_on0 = np.asarray(r_on0['T_basic_K'], dtype=float)
                tb_off0 = np.asarray(r_off0['T_basic_K'], dtype=float)
                ts_on0 = np.asarray(r_on0['T_stag_K'], dtype=float)
                ts_off0 = np.asarray(r_off0['T_stag_K'], dtype=float)
                sf_on0 = np.asarray(r_on0['S_final'], dtype=float)
                sf_off0 = np.asarray(r_off0['S_final'], dtype=float)
                _check_scalar_match(
                    failures,
                    f'{suite_key}.on_vs_off_activity.max_rel_diff_S_final_on_vs_off',
                    act.get('max_rel_diff_S_final_on_vs_off'),
                    float(np.max(np.abs(sf_on0 - sf_off0)) / np.max(np.abs(sf_off0))),
                    rtol=1.0e-9,
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.on_vs_off_activity.max_rel_diff_T_basic_on_vs_off',
                    act.get('max_rel_diff_T_basic_on_vs_off'),
                    float(np.max(np.abs(tb_on0 - tb_off0)) / np.max(np.abs(tb_off0))),
                    rtol=1.0e-9,
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.on_vs_off_activity.max_abs_diff_T_basic_on_vs_off_K',
                    act.get('max_abs_diff_T_basic_on_vs_off_K'),
                    float(np.max(np.abs(tb_on0 - tb_off0))),
                    rtol=1.0e-9,
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.on_vs_off_activity.max_rel_diff_T_stag_on_vs_off',
                    act.get('max_rel_diff_T_stag_on_vs_off'),
                    float(np.max(np.abs(ts_on0 - ts_off0)) / np.max(np.abs(ts_off0))),
                    rtol=1.0e-9,
                )
                _check_scalar_match(
                    failures,
                    f'{suite_key}.on_vs_off_activity.max_abs_diff_T_stag_on_vs_off_K',
                    act.get('max_abs_diff_T_stag_on_vs_off_K'),
                    float(np.max(np.abs(ts_on0 - ts_off0))),
                    rtol=1.0e-9,
                )
    return failures


def verify_benchmarks(
    data: dict[str, Any],
    eos_dir: Path | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    strict_fingerprint: bool = False,
) -> list[str]:
    """Verify that benchmark data satisfy schema keys, invariants, and live fast checks."""
    failures: list[str] = []

    if tuple(data.keys()) != EXPECTED_TOP_KEYS:
        failures.append(f'Top-level keys {tuple(data.keys())} != {EXPECTED_TOP_KEYS}')

    actual_hashes: dict[str, str] | None = None
    canary = data.get('canary')
    if not isinstance(canary, dict):
        failures.append("Missing or invalid 'canary' section")
    else:
        commit_sha = canary.get('aragog_commit')
        input_sha256 = canary.get('generator_input_sha256')
        input_files = canary.get('generator_input_files')
        if not commit_sha:
            failures.append("Missing 'aragog_commit' in canary")
        elif not input_sha256:
            failures.append("Missing 'generator_input_sha256' in canary")
        else:
            failures.extend(
                _check_generator_input_fingerprint(
                    commit_sha,
                    input_sha256,
                    WIRING_GENERATOR_INPUT_PATHS,
                    recorded_files=input_files
                    if isinstance(input_files, (list, dict))
                    else None,
                    repo_root=repo_root,
                    regenerate_cmd=(
                        'python tools/verification/generate_ssc_solver_wiring_benchmarks.py --full'
                    ),
                    strict=strict_fingerprint,
                )
            )
        if eos_dir is None or not eos_dir.is_dir():
            failures.append(
                'Missing or invalid EOS table directory (--eos-dir or ARAGOG_TEST_EOS_DIR); '
                'required for table_md5 and live solver verification'
            )
        else:
            try:
                actual_hashes = compute_table_hashes(eos_dir)
            except FileNotFoundError as exc:
                failures.append(str(exc))
            else:
                _check_table_md5_dict(
                    failures, "canary['table_md5']", canary.get('table_md5'), actual_hashes
                )

    parity_block = data.get('rhs_parity_and_activity')
    if not isinstance(parity_block, dict):
        failures.append("Missing 'rhs_parity_and_activity' section")
    else:
        failures.extend(_verify_parity_block(parity_block, actual_hashes))

    cost_block = data.get('rhs_wall_time')
    if not isinstance(cost_block, dict):
        failures.append("Missing 'rhs_wall_time' section")
    else:
        failures.extend(_verify_cost_block(cost_block, actual_hashes))

    starts_block = data.get('perturbed_starts')
    if not isinstance(starts_block, dict):
        failures.append("Missing 'perturbed_starts' section")
    else:
        failures.extend(_verify_starts_block(starts_block, actual_hashes))

    if eos_dir is not None and not failures:
        failures.extend(_verify_live_fast_subset(data, eos_dir))

    return failures


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the benchmark generator."""
    parser = argparse.ArgumentParser(
        description='Generate or check validation benchmarks for solid rheology.'
    )
    env_eos = os.environ.get('ARAGOG_TEST_EOS_DIR', '').strip()
    parser.add_argument('--output', type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument(
        '--eos-dir',
        type=Path,
        default=Path(env_eos) if env_eos else None,
    )
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--cost-only', action='store_true')
    parser.add_argument('--starts-only', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for benchmark generation and checking."""
    args = parse_arguments(argv)
    output_path = args.output.resolve()
    has_eos_dir = args.eos_dir is not None and args.eos_dir.is_dir()

    if args.check:
        if not output_path.is_file():
            sys.stderr.write(f'Benchmark file not found: {output_path}\n')
            return 1
        data = json.loads(output_path.read_text(encoding='utf-8'))
        eos_dir = args.eos_dir if has_eos_dir else None
        failures = verify_benchmarks(data, eos_dir=eos_dir)
        if failures:
            sys.stderr.write(f'Validation failed with {len(failures)} issue(s):\n')
            for f in failures:
                sys.stderr.write(f'  - {f}\n')
            return 1
        print(f'Verification passed: {output_path}')
        return 0

    if not args.full and not output_path.is_file():
        sys.stderr.write(f'Baseline benchmark file not found: {output_path}\n')
        return 1
    if not has_eos_dir:
        sys.stderr.write(
            'Missing or invalid EOS table directory (--eos-dir or ARAGOG_TEST_EOS_DIR)\n'
        )
        return 1

    try:
        head_sha = _resolve_origin_head_commit(REPO_ROOT)
    except RuntimeError as exc:
        sys.stderr.write(f'{exc}\n')
        return 1

    table_hashes = compute_table_hashes(args.eos_dir)
    input_sha256, input_files = compute_generator_input_fingerprint(
        WIRING_GENERATOR_INPUT_PATHS, repo_root=REPO_ROOT
    )
    cfg_path = Path(str(importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')))

    if args.full:
        data = {
            'canary': {
                'aragog_commit': head_sha,
                'generator_input_sha256': input_sha256,
                'generator_input_files': input_files,
                'table_md5': table_hashes,
            },
            'rhs_parity_and_activity': run_rhs_parity_benchmarks(
                cfg_path, args.eos_dir, table_hashes
            ),
            'rhs_wall_time': run_rhs_cost_benchmarks(
                cfg_path,
                entropy_eos_copy(args.eos_dir),
                entropy_eos_jax(args.eos_dir),
                table_hashes=table_hashes,
            ),
            'perturbed_starts': run_perturbed_starts_benchmarks(
                cfg_path, args.eos_dir, table_hashes
            ),
        }
    else:
        data = json.loads(output_path.read_text(encoding='utf-8'))
        existing_canary = data.get('canary', {})
        fp_errs = _check_generator_input_fingerprint(
            existing_canary.get('aragog_commit'),
            existing_canary.get('generator_input_sha256'),
            WIRING_GENERATOR_INPUT_PATHS,
            recorded_files=existing_canary.get('generator_input_files')
            if isinstance(existing_canary.get('generator_input_files'), (list, dict))
            else None,
            repo_root=REPO_ROOT,
            regenerate_cmd=(
                'python tools/verification/generate_ssc_solver_wiring_benchmarks.py --full'
            ),
            strict=True,
        )
        if fp_errs:
            for err in fp_errs:
                sys.stderr.write(f'{err}\n')
            return 1
        data.setdefault('canary', {})['table_md5'] = table_hashes
        data.setdefault('canary', {})['aragog_commit'] = head_sha
        data['canary']['generator_input_sha256'] = input_sha256
        data['canary']['generator_input_files'] = input_files
        if args.parity_only:
            data['rhs_parity_and_activity'] = run_rhs_parity_benchmarks(
                cfg_path, args.eos_dir, table_hashes
            )
        if args.cost_only:
            data['rhs_wall_time'] = run_rhs_cost_benchmarks(
                cfg_path,
                entropy_eos_copy(args.eos_dir),
                entropy_eos_jax(args.eos_dir),
                table_hashes=table_hashes,
            )
        if args.starts_only:
            data['perturbed_starts'] = run_perturbed_starts_benchmarks(
                cfg_path, args.eos_dir, table_hashes
            )

    failures = verify_benchmarks(data, eos_dir=args.eos_dir, strict_fingerprint=True)
    if failures:
        sys.stderr.write(
            f'Generated benchmark verification failed with {len(failures)} issue(s):\n'
        )
        for f in failures:
            sys.stderr.write(f'  - {f}\n')
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f'Benchmarks written to {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
