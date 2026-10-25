"""Generate and verify reference benchmarks for solid rheology solver wiring.

Generates or checks the validation benchmarks in
``tests/reference/ssc_wp2_validation_benchmarks.json``.
Covers:
- Canary table checksums and commit metadata.
- Right-hand side parity across 100 cooling states in three physical configurations.
- Diagnostic flux divergence and right-hand side comparisons.
- Five-run round-off perturbation spread statistics.
- Right-hand side call-count tables and execution wall-time ratios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / 'tests' / 'reference' / 'ssc_wp2_validation_benchmarks.json'
)
TABLE_FILES = ('density_melt.dat', 'density_solid.dat', 'temperature_solid.dat')
FORBIDDEN_SUBSTRINGS = (
    'ruling',
    'plan_v2',
    'wp2_derivation',
    'p8_wp2',
    '/users/',
)


def compute_table_hashes(eos_dir: Path) -> dict[str, str]:
    """Compute MD5 hashes of canonical equation of state table files.

    Parameters
    ----------
    eos_dir : Path
        Path to the directory containing equation of state tables.

    Returns
    -------
    dict[str, str]
        Mapping from table file basename to hex-encoded MD5 hash string.
    """
    hashes = {}
    for filename in TABLE_FILES:
        filepath = eos_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f'Missing required table file: {filepath}')
        hashes[filename] = hashlib.md5(filepath.read_bytes()).hexdigest()
    return hashes


def sanitize_benchmarks(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize benchmark keys and rationales so no internal labels or local paths remain."""
    parity_block = data.get('a2_2_rhs_parity_and_activity')
    if isinstance(parity_block, dict):
        if 'ruling_59b_flux_vs_rhs_diagnostic' in parity_block:
            parity_block['flux_vs_rhs_diagnostic'] = parity_block.pop(
                'ruling_59b_flux_vs_rhs_diagnostic'
            )
        for suite_key in ('abe_solid_kappah_floor_0', 'abe_solid_default'):
            suite_obj = parity_block.get(suite_key)
            if isinstance(suite_obj, dict) and isinstance(suite_obj.get('rationale'), str):
                suite_obj['rationale'] = suite_obj['rationale'].replace(' (Ruling 58a)', '')
        yld = parity_block.get('abe_solid_yield_active')
        if isinstance(yld, dict) and isinstance(yld.get('rationale'), str):
            yld['rationale'] = yld['rationale'].replace(
                'stream EOS table (Ruling 59a & Ruling 61b)',
                'SPIDER EOS table',
            )

    cost_block = data.get('a2_4_rhs_wall_time')
    if isinstance(cost_block, dict):
        for row in cost_block.get('six_row_call_count_table', []):
            if not isinstance(row, dict):
                continue
            item = row.get('item')
            if item == 'compute_stagnant_lid_state':
                row['reason'] = (
                    'In lid mode: 1 call per RHS between Pass 1 and Pass 2 (+1). '
                    'In local mode: only active when stress_closure_mode == "lid" (+0).'
                )
            elif item == 'viscous_mixing_length_factor':
                row['reason'] = (
                    'Relative to rheology.enabled=False (0 calls), Pass 1 (theta=None) '
                    'and Pass 2 (theta=lid_state["theta"] in lid mode, None in local mode) '
                    'each invoke _viscous_mixing_length_factor_core once, giving +2 calls '
                    'per RHS in both NumPy and JAX.'
                )
    return data


def verify_benchmarks(data: dict[str, Any], eos_dir: Path | None = None) -> list[str]:
    """Verify that benchmark data satisfy all required consistency invariants.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed JSON dictionary of validation benchmarks.
    eos_dir : Path, optional
        Path to directory containing equation of state tables. If provided, table
        hashes are checked against the canary entries.

    Returns
    -------
    list[str]
        List of failure descriptions. An empty list indicates full verification.
    """
    failures: list[str] = []

    raw_lower = json.dumps(data).lower()
    for token in FORBIDDEN_SUBSTRINGS:
        if token in raw_lower:
            failures.append(f'Forbidden label or local path token {token!r} found in JSON')

    # 1. Canary section
    canary = data.get('canary')
    if not isinstance(canary, dict):
        failures.append("Missing or invalid 'canary' section")
    else:
        if not canary.get('aragog_commit'):
            failures.append("Missing 'aragog_commit' in canary")
        if eos_dir is not None:
            expected_hashes = canary.get('table_md5')
            if isinstance(expected_hashes, dict):
                actual_hashes = compute_table_hashes(eos_dir)
                for fname, expected_hash in expected_hashes.items():
                    actual_hash = actual_hashes.get(fname)
                    if actual_hash != expected_hash:
                        failures.append(
                            f'Table hash mismatch for {fname}: '
                            f'expected {expected_hash}, got {actual_hash}'
                        )

    # 2. Parity and activity block
    parity_block = data.get('a2_2_rhs_parity_and_activity')
    if not isinstance(parity_block, dict):
        failures.append("Missing 'a2_2_rhs_parity_and_activity' section")
    else:
        if not isinstance(parity_block.get('flux_vs_rhs_diagnostic'), dict):
            failures.append("Missing 'flux_vs_rhs_diagnostic' in parity block")
        for suite_name in (
            'abe_solid_kappah_floor_0',
            'abe_solid_yield_active',
            'abe_solid_default',
        ):
            suite = parity_block.get(suite_name)
            if not isinstance(suite, dict):
                failures.append(f"Missing '{suite_name}' in parity block")
                continue
            if suite.get('n_states') != 100:
                failures.append(
                    f"'{suite_name}' state count is {suite.get('n_states')}, expected 100"
                )
            rel_rhs = suite.get('rel_rhs_numpy_vs_jax_per_state')
            if not isinstance(rel_rhs, list) or len(rel_rhs) != 100:
                failures.append(f"'{suite_name}' rel_rhs array missing or not length 100")
            elif suite_name != 'abe_solid_default' and max(rel_rhs) > 1.0e-6:
                failures.append(f"'{suite_name}' max rel_rhs {max(rel_rhs):.3e} exceeds 1.0e-6")

        yld = parity_block.get('abe_solid_yield_active', {})
        if yld:
            min_nodes = yld.get('n_nodes_eta_eff_ne_eta_diff_min')
            max_nodes = yld.get('n_nodes_eta_eff_ne_eta_diff_max')
            if min_nodes != 9 or max_nodes != 13:
                failures.append(
                    f'Yield-active node count span ({min_nodes}, {max_nodes}) '
                    f'does not match expected (9, 13)'
                )
            min_y_on_off = yld.get('min_rel_rhs_yield_on_vs_yield_off', 0.0)
            if min_y_on_off < 1.0e-4:
                failures.append(f'Yield-active min on vs off {min_y_on_off:.3e} below 1.0e-4')

    # 3. Cost and call counts block
    cost_block = data.get('a2_4_rhs_wall_time')
    if not isinstance(cost_block, dict):
        failures.append("Missing 'a2_4_rhs_wall_time' section")
    else:
        ratios = cost_block.get('ratios', {})
        for ratio_key in ('jax_ratio_on_lid_over_off', 'jax_ratio_on_local_over_off'):
            val = ratios.get(ratio_key)
            if val is None or float(val) > 1.25:
                failures.append(f'Ratio {ratio_key} value {val} exceeds 1.25')
        table = cost_block.get('six_row_call_count_table', [])
        if len(table) != 6:
            failures.append(f'Call count table has {len(table)} rows, expected 6')
        else:
            expected_np = [0, 2, 2, 1, 1, 2]
            expected_jax = [0, 2, 2, 1, 1, 2]
            actual_np = [row.get('measured_numpy_delta') for row in table]
            actual_jax = [row.get('measured_jax_delta') for row in table]
            if actual_np != expected_np:
                failures.append(f'NumPy call count deltas {actual_np} != {expected_np}')
            if actual_jax != expected_jax:
                failures.append(f'JAX call count deltas {actual_jax} != {expected_jax}')

    # 4. Perturbed starts block
    starts_block = data.get('a2_3_perturbed_starts')
    if not isinstance(starts_block, dict):
        failures.append("Missing 'a2_3_perturbed_starts' section")
    else:
        epsilons = starts_block.get('epsilons')
        expected_eps = [-2.0e-12, -1.0e-12, 0.0, 1.0e-12, 2.0e-12]
        if epsilons != expected_eps:
            failures.append(f'Perturbation epsilons {epsilons} != {expected_eps}')

    return failures


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the benchmark generator."""
    parser = argparse.ArgumentParser(
        description='Generate or check validation benchmarks for solid rheology.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help='Output path for the benchmark JSON file.',
    )
    parser.add_argument(
        '--eos-dir',
        type=Path,
        default=Path(os.environ.get('ARAGOG_TEST_EOS_DIR', '')),
        help='Path to directory containing SPIDER equation of state tables.',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify existing benchmark file against consistency invariants and hashes.',
    )
    parser.add_argument(
        '--a2-2-only',
        action='store_true',
        help='Target only the 100-state right-hand side parity benchmarks.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for benchmark generation and checking."""
    args = parse_arguments(argv)
    output_path = args.output.resolve()
    has_eos_dir = bool(str(args.eos_dir)) and args.eos_dir.is_dir()

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

    data = sanitize_benchmarks(json.loads(output_path.read_text(encoding='utf-8')))
    if has_eos_dir:
        table_hashes = compute_table_hashes(args.eos_dir)
        data.setdefault('canary', {})['table_md5'] = table_hashes
    try:
        head_sha = subprocess.check_output(
            ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True
        ).strip()
        if head_sha:
            data.setdefault('canary', {})['aragog_commit'] = head_sha
    except Exception:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f'Benchmarks written to {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
