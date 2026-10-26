"""Generate and verify reference benchmarks for solid rheology solver wiring.

Generates or checks the validation benchmarks in
``tests/reference/ssc_wp2_validation_benchmarks.json``.

Execution modes and per-block coverage:
- ``--full`` or ``--a2-4-only`` (~5 s on 1 CPU core; requires ``--eos-dir`` or
  ``ARAGOG_TEST_EOS_DIR`` and exits 1 if no valid EOS table directory is provided):
  recomputes ``canary.table_md5`` for all 12 read SPIDER P-S table files, updates
  ``canary.aragog_commit``, and recomputes ``a2_4_rhs_wall_time`` in place by
  instrumenting NumPy and JAX right-hand side kernels to count per-evaluation function
  calls across ``off``, ``on_lid``, and ``on_local`` modes and timing 1000 right-hand
  side evaluations per backend.
- ``--check`` (~2 s; requires ``--eos-dir`` or ``ARAGOG_TEST_EOS_DIR`` and exits 1
  if no valid EOS table directory is provided): verifies expected schema keys,
  canary MD5 hashes for all 12 read SPIDER P-S table files, and stored numerical
  bounds across ``a2_2_rhs_parity_and_activity``, ``a2_4_rhs_wall_time``, and
  ``a2_3_perturbed_starts``, and recomputes a fast live subset:
  1. ``a2_2_rhs_parity_and_activity``: recomputes state 0 of
     ``abe_solid_kappah_floor_0`` in NumPy and JAX and verifies that the live
     relative right-hand side difference matches the stored element within ``1e-6``
     and stays below ``1e-6``.
  2. ``a2_4_rhs_wall_time``: recomputes the 6-row per-evaluation call-count deltas
     in both NumPy and JAX and verifies exact equality ``[0, 2, 2, 1, 1, 2]`` in
     lid mode and ``[0, 1, 1, 0, 1, 2]`` in local mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / 'tests' / 'reference' / 'ssc_wp2_validation_benchmarks.json'
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
    'a2_2_rhs_parity_and_activity',
    'a2_4_rhs_wall_time',
    'a2_3_perturbed_starts',
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
    import aragog.solver.entropy_state as entropy_state_mod
    from aragog.config import Config
    from aragog.jax.solver import dSdt
    from aragog.solver.entropy_solver import EntropySolver

    try:
        import aragog.rheology_lid as rheo_lid_mod
    except ImportError:
        rheo_lid_mod = rheo_mod

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

    orig_arr = rheo_mod._compute_arrhenius_viscosity_core
    orig_yld = rheo_mod._compute_yield_stress_core
    orig_lid = rheo_lid_mod._compute_stagnant_lid_state_core
    orig_eff = rheo_lid_mod._compute_effective_viscosity_core
    orig_mlt = rheo_mod._viscous_mixing_length_factor_core
    orig_val = rheo_mod._validate_inputs
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
            mods_arr = [rheo_mod] + ([rheo_lid_mod] if rheo_lid_mod is not rheo_mod else [])
            if backend == 'numpy':
                mods_arr.extend([entropy_phase_mod, entropy_state_mod])
            else:
                mods_arr.append(jax_phase_mod)
            for mod in mods_arr:
                if hasattr(mod, '_validate_inputs'):
                    stack.enter_context(
                        patch.object(
                            mod,
                            '_validate_inputs',
                            side_effect=_wrap(counts, 'validate_inputs', orig_val),
                        )
                    )
                if hasattr(mod, '_compute_arrhenius_viscosity_core'):
                    stack.enter_context(
                        patch.object(
                            mod,
                            '_compute_arrhenius_viscosity_core',
                            side_effect=_wrap(counts, 'compute_arrhenius_viscosity', orig_arr),
                        )
                    )
                if hasattr(mod, '_compute_yield_stress_core'):
                    stack.enter_context(
                        patch.object(
                            mod,
                            '_compute_yield_stress_core',
                            side_effect=_wrap(counts, 'compute_yield_stress', orig_yld),
                        )
                    )
                if hasattr(mod, '_compute_stagnant_lid_state_core'):
                    stack.enter_context(
                        patch.object(
                            mod,
                            '_compute_stagnant_lid_state_core',
                            side_effect=_wrap(counts, 'compute_stagnant_lid_state', orig_lid),
                        )
                    )
                if hasattr(mod, '_compute_effective_viscosity_core'):
                    stack.enter_context(
                        patch.object(
                            mod,
                            '_compute_effective_viscosity_core',
                            side_effect=_wrap(counts, 'compute_effective_viscosity', orig_eff),
                        )
                    )
                if hasattr(mod, '_viscous_mixing_length_factor_core'):
                    stack.enter_context(
                        patch.object(
                            mod,
                            '_viscous_mixing_length_factor_core',
                            side_effect=_wrap(counts, 'viscous_mixing_length_factor', orig_mlt),
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
            1,
            'Relative to rheology.enabled=False (0 calls), Pass 1 (theta=None) and Pass 2 (theta=lid_state["theta"] in lid mode, None in local mode) each invoke _viscous_mixing_length_factor_core once, giving +2 calls per RHS in both NumPy and JAX.',
        ),
    ]
    rows = []
    for label, key, plan_delta, reason in spec:
        d_np_lid = np_lid[key] - np_off[key]
        rows.append(
            {
                'item': label,
                'plan_a2_4_delta': plan_delta,
                'measured_numpy_delta': d_np_lid,
                'measured_jax_delta': jax_lid[key] - jax_off[key],
                'measured_numpy_local_delta': np_loc[key] - np_off[key],
                'measured_jax_local_delta': jax_loc[key] - jax_off[key],
                'difference_from_plan': d_np_lid - plan_delta,
                'reason': reason,
            }
        )
    return rows


def run_a2_4_rhs_benchmarks(cfg_path: Path, eos_np: Any, eos_jax: Any) -> dict[str, Any]:
    """Measure per-RHS call counts and wall-time ratio on/off for NumPy and JAX (1000 evals)."""
    import jax
    import jax.numpy as jnp

    from aragog.config import Config
    from aragog.jax.solver import dSdt
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    jax.config.update('jax_enable_x64', True)
    n_warmup, n_eval = 20, 1000
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
            solver.dSdt(0.0, s_vec_np)
        t0 = time.perf_counter()
        for _ in range(n_eval):
            out_np = solver.dSdt(0.0, s_vec_np)
        dt_np = time.perf_counter() - t0

        _, _, _, _, args = _build_jax_bundle(solver, eos_jax)
        jit_rhs = jax.jit(lambda s_vec, t_yr: dSdt(t_yr, s_vec, args))
        for _ in range(n_warmup):
            jit_rhs(s_vec_jax, 0.0).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_eval):
            out_jax = jit_rhs(s_vec_jax, 0.0)
            out_jax.block_until_ready()
        dt_jax = time.perf_counter() - t0

        denom = max(float(np.max(np.abs(out_np))), 1.0e-30)
        max_rel_diff = float(np.max(np.abs(out_np - np.asarray(out_jax))) / denom)
        results[label] = {
            'n_eval': n_eval,
            'numpy_total_s': dt_np,
            'numpy_per_rhs_us': (dt_np / n_eval) * 1.0e6,
            'jax_total_s': dt_jax,
            'jax_per_rhs_us': (dt_jax / n_eval) * 1.0e6,
            'numpy_vs_jax_max_rel_diff': max_rel_diff,
            'call_counts': call_counts[label],
        }

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


def _verify_live_fast_subset(data: dict[str, Any], eos_dir: Path) -> list[str]:
    """Recompute the fast live subset (call counts and state-0 parity) against stored JSON."""
    import jax
    import jax.numpy as jnp

    from aragog.config import Config
    from aragog.jax.solver import dSdt
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver
    from tests.conftest import entropy_eos_copy, entropy_eos_jax

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

    cfg_on = Config.from_file(str(cfg_path))
    cfg_on.boundary_conditions.inner_boundary_condition = 1
    cfg_on.boundary_conditions.core_bc = 'quasi_steady'
    cfg_on.energy.kappah_floor = 0.0
    cfg_on.phase_solid.viscosity = 1.0e18
    cfg_on.phase_solid.rheology = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    solver_on = EntropySolver(cfg_on, entropy_eos=eos_np)
    solver_on.initialize()
    n_stag = solver_on._n_stag
    xi = np.linspace(0.0, 1.0, n_stag)
    s_profile = (2500.0 - 450.0 * (0.2 * xi + 0.8 * xi**4)).copy()
    rhs_np = solver_on.dSdt(0.0, s_profile).copy()
    _, _, _, _, args_on = _build_jax_bundle(solver_on, eos_jax)
    rhs_jax = np.asarray(dSdt(0.0, jnp.asarray(s_profile), args_on))
    rel_diff_0 = float(
        np.max(np.abs(rhs_np - rhs_jax)) / max(float(np.max(np.abs(rhs_np))), 1.0e-30)
    )
    stored_0 = float(
        data['a2_2_rhs_parity_and_activity']['abe_solid_kappah_floor_0'][
            'rel_rhs_numpy_vs_jax_per_state'
        ][0]
    )
    if rel_diff_0 > 1.0e-6 or abs(rel_diff_0 - stored_0) > 1.0e-6:
        failures.append(
            f'Live state-0 rel_rhs_numpy_vs_jax {rel_diff_0:.3e} vs stored {stored_0:.3e}'
        )
    return failures


def _is_commit_on_origin(sha: str, repo_root: Path = REPO_ROOT) -> bool:
    """Return True if ``sha`` is a 40-hex git commit reachable from any ``origin/*`` branch."""
    if not isinstance(sha, str) or len(sha) != 40 or sha == 'uncommitted':
        return False
    try:
        obj_type = subprocess.check_output(
            ['git', '-C', str(repo_root), 'cat-file', '-t', sha],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if obj_type != 'commit':
            return False
        branches_out = subprocess.check_output(
            ['git', '-C', str(repo_root), 'branch', '-r', '--contains', sha],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return any(line.strip().startswith('origin/') for line in branches_out.splitlines())


def _resolve_origin_head_commit(repo_root: Path = REPO_ROOT) -> str:
    """Return ``HEAD`` SHA only when the tracked tree is clean and ``HEAD`` is on ``origin``."""
    status_out = subprocess.check_output(
        ['git', '-C', str(repo_root), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if status_out:
        raise RuntimeError(
            'Working tree has uncommitted tracked changes; commit and push before generating benchmarks'
        )
    head_sha = subprocess.check_output(
        ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if not _is_commit_on_origin(head_sha, repo_root):
        raise RuntimeError(
            f'HEAD {head_sha} is not reachable from any origin/* branch; push before generating benchmarks'
        )
    return head_sha


def verify_benchmarks(data: dict[str, Any], eos_dir: Path | None = None) -> list[str]:
    """Verify that benchmark data satisfy schema keys, invariants, and live fast checks."""
    failures: list[str] = []

    if tuple(data.keys()) != EXPECTED_TOP_KEYS:
        failures.append(f'Top-level keys {tuple(data.keys())} != {EXPECTED_TOP_KEYS}')

    canary = data.get('canary')
    if not isinstance(canary, dict):
        failures.append("Missing or invalid 'canary' section")
    else:
        commit_sha = canary.get('aragog_commit')
        if not commit_sha:
            failures.append("Missing 'aragog_commit' in canary")
        elif not _is_commit_on_origin(str(commit_sha)):
            failures.append(
                f"canary['aragog_commit'] {commit_sha!r} is not a commit reachable from origin/*"
            )
        if eos_dir is None or not str(eos_dir) or not eos_dir.is_dir():
            failures.append(
                'Missing or invalid EOS table directory (--eos-dir or ARAGOG_TEST_EOS_DIR); '
                'required for table_md5 and live solver verification'
            )
        else:
            expected_hashes = canary.get('table_md5')
            if (
                not isinstance(expected_hashes, dict)
                or tuple(expected_hashes.keys()) != TABLE_FILES
            ):
                failures.append(
                    f"canary['table_md5'] keys {tuple(expected_hashes.keys()) if isinstance(expected_hashes, dict) else None} != {TABLE_FILES}"
                )
            else:
                try:
                    actual_hashes = compute_table_hashes(eos_dir)
                except FileNotFoundError as exc:
                    failures.append(str(exc))
                else:
                    for fname, expected_hash in expected_hashes.items():
                        actual_hash = actual_hashes.get(fname)
                        if actual_hash != expected_hash:
                            failures.append(
                                f'Table hash mismatch for {fname}: '
                                f'expected {expected_hash}, got {actual_hash}'
                            )

    parity_block = data.get('a2_2_rhs_parity_and_activity')
    if not isinstance(parity_block, dict):
        failures.append("Missing 'a2_2_rhs_parity_and_activity' section")
    else:
        for req_key in EXPECTED_PARITY_KEYS:
            if not isinstance(parity_block.get(req_key), dict):
                failures.append(f"Missing '{req_key}' in parity block")
        for suite_name in (
            'abe_solid_kappah_floor_0',
            'abe_solid_yield_active',
            'abe_solid_default',
        ):
            suite = parity_block.get(suite_name)
            if not isinstance(suite, dict):
                continue
            if suite.get('n_states') != 100:
                failures.append(
                    f"'{suite_name}' state count is {suite.get('n_states')}, expected 100"
                )
            rel_rhs = suite.get('rel_rhs_numpy_vs_jax_per_state')
            if not isinstance(rel_rhs, list) or len(rel_rhs) != 100:
                failures.append(f"'{suite_name}' rel_rhs array missing or not length 100")
            elif max(rel_rhs) > 1.0e-6:
                failures.append(f"'{suite_name}' max rel_rhs {max(rel_rhs):.3e} exceeds 1.0e-6")

        yld = parity_block.get('abe_solid_yield_active', {})
        if isinstance(yld, dict) and yld:
            min_nodes = yld.get('n_nodes_eta_eff_ne_eta_diff_min')
            max_nodes = yld.get('n_nodes_eta_eff_ne_eta_diff_max')
            if min_nodes != 9 or max_nodes != 13:
                failures.append(
                    f'Yield-active node count span ({min_nodes}, {max_nodes}) != (9, 13)'
                )
            min_y_on_off = yld.get('min_rel_rhs_yield_on_vs_yield_off', 0.0)
            if min_y_on_off < 1.0e-4:
                failures.append(f'Yield-active min on vs off {min_y_on_off:.3e} below 1.0e-4')

    cost_block = data.get('a2_4_rhs_wall_time')
    if not isinstance(cost_block, dict):
        failures.append("Missing 'a2_4_rhs_wall_time' section")
    else:
        for req_key in EXPECTED_COST_KEYS:
            if req_key not in cost_block:
                failures.append(f"Missing '{req_key}' in cost block")
        ratios = cost_block.get('ratios', {})
        for ratio_key in ('jax_ratio_on_lid_over_off', 'jax_ratio_on_local_over_off'):
            val = ratios.get(ratio_key)
            if val is None or float(val) > 1.25:
                failures.append(f'Ratio {ratio_key} value {val} exceeds 1.25')
        table = cost_block.get('six_row_call_count_table', [])
        if len(table) != 6:
            failures.append(f'Call count table has {len(table)} rows, expected 6')
        else:
            expected_deltas = [0, 2, 2, 1, 1, 2]
            actual_np = [row.get('measured_numpy_delta') for row in table]
            actual_jax = [row.get('measured_jax_delta') for row in table]
            if actual_np != expected_deltas or actual_jax != expected_deltas:
                failures.append(
                    f'Stored call count deltas NP={actual_np} JAX={actual_jax} != {expected_deltas}'
                )
            expected_local_deltas = [0, 1, 1, 0, 1, 2]
            actual_np_loc = [row.get('measured_numpy_local_delta') for row in table]
            actual_jax_loc = [row.get('measured_jax_local_delta') for row in table]
            if (
                actual_np_loc != expected_local_deltas
                or actual_jax_loc != expected_local_deltas
            ):
                failures.append(
                    f'Stored local call count deltas NP={actual_np_loc} '
                    f'JAX={actual_jax_loc} != {expected_local_deltas}'
                )

    starts_block = data.get('a2_3_perturbed_starts')
    if not isinstance(starts_block, dict):
        failures.append("Missing 'a2_3_perturbed_starts' section")
    else:
        for req_key in EXPECTED_STARTS_KEYS:
            if req_key not in starts_block:
                failures.append(f"Missing '{req_key}' in starts block")
        for suite_key in ('abe_solid_default', 'abe_solid_kappah_floor_0'):
            if not isinstance(starts_block.get(suite_key), dict):
                failures.append(f"Invalid or missing '{suite_key}' dictionary in starts block")
        epsilons = starts_block.get('epsilons')
        expected_eps = [-2.0e-12, -1.0e-12, 0.0, 1.0e-12, 2.0e-12]
        if epsilons != expected_eps:
            failures.append(f'Perturbation epsilons {epsilons} != {expected_eps}')

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
    parser.add_argument('--a2-4-only', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for benchmark generation and checking."""
    args = parse_arguments(argv)
    output_path = args.output.resolve()
    has_eos_dir = args.eos_dir is not None and bool(str(args.eos_dir)) and args.eos_dir.is_dir()

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

    if not output_path.is_file():
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

    data = json.loads(output_path.read_text(encoding='utf-8'))
    table_hashes = compute_table_hashes(args.eos_dir)
    data.setdefault('canary', {})['table_md5'] = table_hashes
    if args.a2_4_only or args.full:
        from tests.conftest import entropy_eos_copy, entropy_eos_jax

        cfg_path = Path(str(importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')))
        data['a2_4_rhs_wall_time'] = run_a2_4_rhs_benchmarks(
            cfg_path, entropy_eos_copy(args.eos_dir), entropy_eos_jax(args.eos_dir)
        )
    data.setdefault('canary', {})['aragog_commit'] = head_sha

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f'Benchmarks written to {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
