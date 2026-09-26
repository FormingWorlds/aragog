"""B7: Mutation testing across branches.

For each mutation:
- Backup target file inside the worktree
- Assert anchor exists exactly once
- Mutate target file
- Run named pytest file
- Record CAUGHT (good: test failed) or MISSED (bad: test passed)
- Save full test output to data/B/B7_<branch>_<mutation_id>.log
- Restore file from backup
- Verify git status --porcelain is empty
- Write results to ~/.shared-agent-state/ssc-verify-task6/mutations.csv
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/Users/timlichtenberg/work/ssc-verify-task6')
DATA_B = BASE / 'data' / 'B'
DATA_B.mkdir(parents=True, exist_ok=True)
R_DIR = Path(os.path.expanduser('~/.shared-agent-state/ssc-verify-task6'))
R_DIR.mkdir(parents=True, exist_ok=True)
MUTATIONS_CSV = R_DIR / 'mutations.csv'

# Define all mutations
# Each entry: (branch_name, worktree_dir, target_rel_path, mutation_id, anchor, replacement, test_file)
MUTATIONS = [
    # ---------------------------------------------------------
    # Branch 1: adaptive-phase-cap
    # ---------------------------------------------------------
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_always_rate',
        "if cap_mode != 'rate' or self._core_bc == 'gradient':",
        "if False:",
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_drop_gradient_guard',
        "if cap_mode != 'rate' or self._core_bc == 'gradient':",
        "if cap_mode != 'rate':",
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_ignore_direction',
        'target = np.where(dSdt < 0.0, down, up)',
        'target = down',
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_drop_near_mask',
        't_c = t_c[near & np.isfinite(t_c)]',
        't_c = t_c[np.isfinite(t_c)]',
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_drop_upper_clip',
        'return float(np.clip(fraction * t_c.min(), *bounds)) if t_c.size else bounds[1]',
        'return float(np.maximum(fraction * t_c.min(), bounds[0])) if t_c.size else bounds[1]',
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_max_for_min',
        'fraction * t_c.min()',
        'fraction * t_c.max()',
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_zero_rate_not_dropped',
        't_c = np.abs(target - S) / np.abs(dSdt)',
        't_c = np.where(dSdt == 0.0, 0.0, np.abs(target - S) / np.where(dSdt == 0.0, 1.0, np.abs(dSdt)))',
        'tests/test_phase_boundary_cap.py',
    ),
    # New mutations
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_wrong_fraction',
        'fraction: float = 0.1,',
        'fraction: float = 0.5,',
        'tests/test_phase_boundary_cap.py',
    ),
    (
        'adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'src/aragog/solver/entropy_solver.py',
        'cap_clip_bounds_reversed',
        'bounds: tuple[float, float] = (1.0, 100.0),',
        'bounds: tuple[float, float] = (5.0, 100.0),',
        'tests/test_phase_boundary_cap.py',
    ),
    # ---------------------------------------------------------
    # Branch 2: ssc-mlt-calibration
    # ---------------------------------------------------------
    (
        'ssc-mlt-calibration',
        'wt-aragog-tl_ssc-mlt-calibration',
        'src/aragog/solver/entropy_state.py',
        'mlt_re_q_switch',
        '''        reynolds = viscous_velocity * mixing_length / nu
        if q is not None:
            reynolds = q * reynolds''',
        '''        reynolds = viscous_velocity * mixing_length / nu
        if False:
            reynolds = q * reynolds''',
        'tests/test_mlt_calibration.py',
    ),
    # New mutations
    (
        'ssc-mlt-calibration',
        'wt-aragog-tl_ssc-mlt-calibration',
        'src/aragog/rheology.py',
        'mlt_q_fourth_power_to_first',
        'return xp.where(positive, (l_v / xp.where(positive, mixing_length, 1.0)) ** 4, 1.0)',
        'return xp.where(positive, (l_v / xp.where(positive, mixing_length, 1.0)) ** 2, 1.0)',
        'tests/test_mlt_calibration.py',
    ),
    (
        'ssc-mlt-calibration',
        'wt-aragog-tl_ssc-mlt-calibration',
        'src/aragog/rheology.py',
        'mlt_visc_factor_inverted',
        'l_v = xp.minimum(bottom_slope * (radii - r_in), top_slope * (r_out - radii))',
        'l_v = xp.maximum(bottom_slope * (radii - r_in), top_slope * (r_out - radii))',
        'tests/test_mlt_calibration.py',
    ),
    # ---------------------------------------------------------
    # Branch 3: ssc-cmb-bl-law
    # ---------------------------------------------------------
    (
        'ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'src/aragog/solver/entropy_solver.py',
        'law_wrong_face',
        'face = self._cmb_law_face',
        'face = 1 - self._cmb_law_face',
        'tests/test_cmb_boundary_layer.py',
    ),
    (
        'ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'src/aragog/solver/entropy_solver.py',
        'law_tm_from_cell0',
        '''        if self.entropy_eos is not None:
            T_m = float(
                self.entropy_eos.temperature_scalar(float(self._P_basic_flat[face]), S_int)
            )
        else:
            pm = self.parameters.phase_mixed
            T_m = pm.const_T_ref * np.exp((S_int - pm.const_S_ref) / pm.const_Cp)''',
        '        T_m = float(np.asarray(ps.temperature()).ravel()[0])',
        'tests/test_cmb_boundary_layer.py',
    ),
    (
        'ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'src/aragog/jax/solver.py',
        'law_jax_face_not_set',
        '        heat_flux = heat_flux.at[1].set(q)',
        '        pass',
        'tests/test_cmb_boundary_layer.py',
    ),
    (
        'ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'src/aragog/solver/entropy_solver.py',
        'law_ts_top_node_with_bc5',
        'self._outer_bc_value\n            if self._outer_bc_kind == 5\n            else float(self.state.top_temperature.item())',
        'float(self.state.top_temperature.item())',
        'tests/test_cmb_boundary_layer.py',
    ),
    # New mutations
    (
        'ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'src/aragog/cmb_boundary_layer.py',
        'law_exponent_flip',
        'RA_CRIT_EXPONENT = 0.21',
        'RA_CRIT_EXPONENT = 0.20',
        'tests/test_cmb_boundary_layer.py',
    ),
    (
        'ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'src/aragog/cmb_boundary_layer.py',
        'law_sign_flip_temperature_jump',
        'dT = T_c - T_m',
        'dT = T_m - T_c',
        'tests/test_cmb_boundary_layer.py',
    ),
    # ---------------------------------------------------------
    # Branch 4: aragog-mesh-refinement
    # ---------------------------------------------------------
    (
        'aragog-mesh-refinement',
        'wt-aragog-tl_aragog-mesh-refinement',
        'src/aragog/mesh/__init__.py',
        'mesh_mass_xtol_1m',
        '_xi_of_r, xi_target, r_lo, r_hi, node=j, xtol=1e-3 if self._refined else 1.0',
        '_xi_of_r, xi_target, r_lo, r_hi, node=j, xtol=1.0',
        'tests/test_mesh_refinement.py',
    ),
    (
        'aragog-mesh-refinement',
        'wt-aragog-tl_aragog-mesh-refinement',
        'src/aragog/mesh/__init__.py',
        'mesh_no_stretching_radius_mode',
        'if self._refined:\n                r_in, r_out = self.settings.inner_radius, self.settings.outer_radius',
        'if False:\n                r_in, r_out = self.settings.inner_radius, self.settings.outer_radius',
        'tests/test_mesh_refinement.py',
    ),
    # New mutations
    (
        'aragog-mesh-refinement',
        'wt-aragog-tl_aragog-mesh-refinement',
        'src/aragog/mesh/stretching.py',
        'mesh_stretching_swap_ends',
        '''    if first == 0.0 and last == 0.0:
        return xi''',
        '''    first, last = last, first
    if first == 0.0 and last == 0.0:
        return xi''',
        'tests/test_mesh_refinement.py',
    ),
    (
        'aragog-mesh-refinement',
        'wt-aragog-tl_aragog-mesh-refinement',
        'src/aragog/parser.py',
        'mesh_surface_cell_thickness_sign_check_dropped',
        'if not (math.isfinite(val) and 0.0 < val < uniform):',
        'if not (math.isfinite(val)):',
        'tests/test_mesh_refinement.py',
    ),
    # ---------------------------------------------------------
    # Branch 5: aragog-skin-temperature
    # ---------------------------------------------------------
    (
        'aragog-skin-temperature',
        'wt-aragog-tl_aragog-skin-temperature',
        'src/aragog/surface_skin.py',
        'skin_newton_2_iter',
        'SKIN_NEWTON_ITERATIONS = 12',
        'SKIN_NEWTON_ITERATIONS = 2',
        'tests/test_surface_skin.py',
    ),
    (
        'aragog-skin-temperature',
        'wt-aragog-tl_aragog-skin-temperature',
        'src/aragog/surface_skin.py',
        'skin_weight_tanh',
        'return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)',
        'return 0.5 * (1.0 - xp.tanh((phi - phi_rheo) / half_width))',
        'tests/test_surface_skin.py',
    ),
    # New mutations
    (
        'aragog-skin-temperature',
        'wt-aragog-tl_aragog-skin-temperature',
        'src/aragog/surface_skin.py',
        'skin_table_edge_offset_flipped',
        'TABLE_EDGE_OFFSET = 100.0',
        'TABLE_EDGE_OFFSET = -100.0',
        'tests/test_surface_skin.py',
    ),
    (
        'aragog-skin-temperature',
        'wt-aragog-tl_aragog-skin-temperature',
        'src/aragog/surface_skin.py',
        'skin_flux_blend_sign',
        'f = es * (T**4 - T_eq**4) - G * (T_top - T)',
        'f = es * (T**4 - T_eq**4) + G * (T_top - T)',
        'tests/test_surface_skin.py',
    ),
    # ---------------------------------------------------------
    # Branch 6: aragog-fixed-cmb-temperature
    # ---------------------------------------------------------
    (
        'aragog-fixed-cmb-temperature',
        'wt-aragog-tl_aragog-fixed-cmb-temperature',
        'src/aragog/solver/entropy_solver.py',
        'f35_bc3_dr_instead_of_dr_half',
        'k_first * (self._inner_bc_value - T_first) / self._cmb_dr_half',
        'k_first * (self._inner_bc_value - T_first) / self._cmb_dr_cmb',
        'tests/test_entropy_solver_bc_dispatch_smoke.py',
    ),
    (
        'aragog-fixed-cmb-temperature',
        'wt-aragog-tl_aragog-fixed-cmb-temperature',
        'src/aragog/parser.py',
        'f35_drop_loader_rule',
        "elif self.inner_boundary_condition == 3 and self.core_bc != 'quasi_steady':",
        "elif False and self.inner_boundary_condition == 3 and self.core_bc != 'quasi_steady':",
        'tests/test_parser.py',
    ),
    # New mutations
    (
        'aragog-fixed-cmb-temperature',
        'wt-aragog-tl_aragog-fixed-cmb-temperature',
        'src/aragog/solver/entropy_solver.py',
        'f35_sign_flip_prescribed_cmb_flux',
        'k_first * (self._inner_bc_value - T_first) / self._cmb_dr_half',
        'k_first * (T_first - self._inner_bc_value) / self._cmb_dr_half',
        'tests/test_entropy_solver_bc_dispatch_smoke.py',
    ),
    (
        'aragog-fixed-cmb-temperature',
        'wt-aragog-tl_aragog-fixed-cmb-temperature',
        'src/aragog/jax/solver.py',
        'f35_jax_dr_instead_of_dr_half',
        'dr_half = 0.5 * (mesh.radii_basic[1] - mesh.radii_basic[0])',
        'dr_half = 1.0 * (mesh.radii_basic[1] - mesh.radii_basic[0])',
        'tests/test_jax_dsdt_energy_balance.py',
    ),
]


def run_mutation(entry) -> dict[str, str | bool]:
    branch, wt_dir, rel_path, mut_id, anchor, replacement, test_file = entry
    wt_path = BASE / wt_dir
    file_path = wt_path / rel_path
    log_file = DATA_B / f'B7_{branch}_{mut_id}.log'

    print(f'Running mutation {branch} / {mut_id} on {rel_path}...')

    # 1. Assert anchor exists exactly once
    content = file_path.read_text()
    count = content.count(anchor)
    if count != 1:
        print(f'  ERROR: Anchor found {count} times (expected 1)!')
        return {
            'branch': branch,
            'mutation': mut_id,
            'test_file': test_file,
            'anchor_found': False,
            'result': 'MISSED',
            'restored_clean': True,
        }

    # 2. Backup file
    backup_path = file_path.with_suffix(file_path.suffix + '.b7_bak')
    shutil.copy2(file_path, backup_path)

    # 3. Apply mutation
    mutated = content.replace(anchor, replacement)
    file_path.write_text(mutated)

    # 4. Run test
    cmd = [
        'pytest',
        test_file,
        '-v',
    ]
    env = dict(os.environ)
    env['PYTHONPATH'] = f'{wt_path}/src'
    env['FWL_DATA'] = '/Users/timlichtenberg/work/fwl_data-dev-2'
    env['ARAGOG_TEST_EOS_DIR'] = '/Users/timlichtenberg/work/ssc-verify-task6/test-data/spider_eos'

    try:
        proc = subprocess.run(
            cmd,
            cwd=wt_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        test_output = proc.stdout + '\n' + proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        test_output = f'TIMEOUT after 180s:\n{exc.stdout}\n{exc.stderr}'
        exit_code = 124

    # Save log
    log_file.write_text(test_output)

    # Test failure means the mutation was CAUGHT. Test success means MISSED.
    if exit_code != 0:
        result = 'CAUGHT'
    else:
        result = 'MISSED'

    print(f'  Result: {result} (exit code {exit_code})')

    # 5. Restore file
    shutil.copy2(backup_path, file_path)
    backup_path.unlink()

    # 6. Verify worktree is clean
    git_proc = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=wt_path,
        capture_output=True,
        text=True,
    )
    restored_clean = (git_proc.stdout.strip() == '')
    if not restored_clean:
        print(f'  WARNING: Worktree {wt_dir} not clean after restore! Output:\n{git_proc.stdout}')
        # Clean up forcefully
        subprocess.run(['git', 'checkout', '--', '.'], cwd=wt_path)

    return {
        'branch': branch,
        'mutation': mut_id,
        'test_file': test_file,
        'anchor_found': True,
        'result': result,
        'restored_clean': restored_clean,
    }


def main():
    rows = []
    findings = []

    for entry in MUTATIONS:
        res = run_mutation(entry)
        rows.append(res)
        if res['result'] == 'MISSED':
            findings.append(res)

    # Write mutations.csv
    with open(MUTATIONS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['branch', 'mutation', 'test_file', 'anchor_found', 'result', 'restored_clean']
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nWrote {len(rows)} mutation results to {MUTATIONS_CSV}')
    print(f'Caught: {sum(1 for r in rows if r["result"] == "CAUGHT")}')
    print(f'Missed: {sum(1 for r in rows if r["result"] == "MISSED")}')

    if findings:
        print('\nFINDINGS (missed mutations):')
        for f in findings:
            print(f"  {f['branch']}: {f['mutation']} on {f['test_file']}")


if __name__ == '__main__':
    main()
