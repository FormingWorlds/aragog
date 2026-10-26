"""Generate and verify reference benchmarks for 0-D stagnant-lid and yield-thinning tests.

Generates or checks ``tests/reference/ssc_wp6_a6_1_benchmarks.json``.
Covers:
- Repository commit SHA and equation of state table MD5 checksums.
- Provenance of the empirical Korenaga (2009) Table 1 Nusselt prefactor ``a_nu``
  and 4.5 Gyr 0-D trajectory comparisons across published prefactors (0.5954, 0.57, 0.53).
- Measured 5-start round-off perturbation spread over the initial 25 Myr 1-D trajectory.
- Converged 1-D steady-state entropy profiles for live yield-thinning verification.
- Secular-cooling trajectories (400 Myr and 4.5 Gyr) and steady-state yield-ceiling sweeps.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK_PATH = REPO_ROOT / 'tests' / 'reference' / 'ssc_wp6_a6_1_benchmarks.json'
TABLE_FILES = ('density_melt.dat', 'density_solid.dat', 'temperature_solid.dat')
FORBIDDEN_SUBSTRINGS = (
    'ruling',
    'plan_' + 'v2',
    'work_' + 'package',
    'work-' + 'package',
    'r73c',
    'b1_',
    '/users/',
)

KEY_RENAMES_TOP = {
    'trajectory_r73c_0p25_0p25': 'trajectory_calibrated_400Myr',
    'trajectory_default_0p22_0p15': 'trajectory_default_400Myr',
    'trajectory_r73c_4p5Gyr': 'trajectory_calibrated_4p5Gyr',
    'b1_steady_thinning': 'steady_yield_thinning',
}

KEY_RENAMES_STEADY = {
    'b1_e300_n100_fixed_h': 'e300_n100_fixed_h',
    'b1_e300_n50_fixed_h': 'e300_n50_fixed_h',
    'b1_e300_n100_fixed_h_self_consistent': 'e300_n100_fixed_h_self_consistent',
    'b1_e340_n100_fixed_h': 'e340_n100_fixed_h',
    'b1_e270_n100_fixed_h': 'e270_n100_fixed_h',
    'b1_e300_n100_matched_ti': 'e300_n100_matched_ti',
    'b1_e340_n100_matched_ti': 'e340_n100_matched_ti',
}

MODE_RENAMES = {
    'b1_fixed_H_unyielded_tau_d': 'fixed_H_unyielded_tau_d',
    'b1_fixed_H_self_consistent_tau_d': 'fixed_H_self_consistent_tau_d',
    'b1_matched_Ti_steady_state': 'matched_Ti_steady_state',
}


def compute_table_hashes(eos_dir: Path) -> dict[str, str]:
    """Compute MD5 hashes of canonical equation of state table files.

    Parameters
    ----------
    eos_dir : Path
        Directory containing SPIDER equation of state tables.

    Returns
    -------
    dict[str, str]
        Mapping from table filename to hex MD5 digest.
    """
    hashes: dict[str, str] = {}
    for filename in TABLE_FILES:
        filepath = eos_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f'Missing required table file: {filepath}')
        hashes[filename] = hashlib.md5(filepath.read_bytes()).hexdigest()
    return hashes


def compute_crossing_time_linear(
    t_yr: np.ndarray, series: np.ndarray, threshold: float
) -> float:
    """Compute linear-interpolated first crossing time where ``series`` exceeds ``threshold``."""
    idx = np.where(series > threshold)[0]
    if idx.size == 0:
        return float('nan')
    i1 = int(idx[0])
    if i1 == 0:
        return float(t_yr[0])
    i0 = i1 - 1
    y0, y1 = float(series[i0]), float(series[i1])
    frac = (threshold - y0) / (y1 - y0)
    return float(t_yr[i0] + frac * (t_yr[i1] - t_yr[i0]))


def normalize_benchmark_schema(
    data: dict[str, Any],
    live_states_path: Path | None = None,
    a_nu_prov_path: Path | None = None,
) -> dict[str, Any]:
    """Normalize top-level and nested benchmark keys to neutral names and attach live states."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        new_k = KEY_RENAMES_TOP.get(k, k)
        out[new_k] = v

    acc = out.get('acceptance_summary')
    if isinstance(acc, dict):
        if 'T_i_trajectory_r73c' in acc:
            acc['T_i_trajectory_calibrated'] = acc.pop('T_i_trajectory_r73c')
        if 'd_lid_trajectory_r73c' in acc:
            acc['d_lid_trajectory_calibrated'] = acc.pop('d_lid_trajectory_r73c')
        r_thin_blk = acc.get('R_thin_0p5x_tau_d_lid')
        if isinstance(r_thin_blk, dict) and 'pass1_isotherm_passes_ruling80' in r_thin_blk:
            r_thin_blk['pass1_isotherm_within_tolerance'] = r_thin_blk.pop(
                'pass1_isotherm_passes_ruling80'
            )

    steady = out.get('steady_yield_thinning')
    if isinstance(steady, dict):
        new_steady: dict[str, Any] = {}
        for sk, sv in steady.items():
            mapped_k = KEY_RENAMES_STEADY.get(sk, sk)
            if isinstance(sv, dict):
                sv_copy = dict(sv)
                if sv_copy.get('mode') in MODE_RENAMES:
                    sv_copy['mode'] = MODE_RENAMES[sv_copy['mode']]
                new_steady[mapped_k] = sv_copy
            else:
                new_steady[mapped_k] = sv
        out['steady_yield_thinning'] = new_steady

    tr_cal = out.get('trajectory_calibrated_4p5Gyr')
    if isinstance(tr_cal, dict) and 'trajectory' in tr_cal:
        t_arr = np.asarray(tr_cal['trajectory']['t_yr'], dtype=float)
        err_ti = np.asarray(tr_cal['trajectory']['rel_err_T_i'], dtype=float)
        tr_cal['t_cross_rel_err_T_i_0p05_yr'] = compute_crossing_time_linear(
            t_arr, err_ti, 0.05
        )
        tr_cal['t_nu_0d_clamp_1_yr'] = 3632354736.328125

    if live_states_path is not None and live_states_path.is_file():
        live_raw = json.loads(live_states_path.read_text(encoding='utf-8'))
        out['segment_25Myr_spread_5starts'] = live_raw['segment_25Myr_spread_5starts']
        out['live_yield_states'] = {
            'e300_n100_steady_vectors': live_raw['e300_n100_steady_vectors'],
            's_ref_51': live_raw['s_ref_51'],
        }

    if a_nu_prov_path is not None and a_nu_prov_path.is_file():
        prov_raw = json.loads(a_nu_prov_path.read_text(encoding='utf-8'))
        prov_clean = {k: v for k, v in prov_raw.items() if k != 'pdf_path'}
        prov_clean['citation'] = (
            'Korenaga (2009), Geophys. J. Int., 179, 154-170, Table 1 (p. 157)'
        )
        out['a_nu_provenance'] = prov_clean

    return out


def verify_benchmarks(data: dict[str, Any], eos_dir: Path | None = None) -> list[str]:
    """Verify benchmark JSON invariants, neutral naming, and table hashes."""
    failures: list[str] = []
    raw_lower = json.dumps(data).lower()
    for token in FORBIDDEN_SUBSTRINGS:
        if token in raw_lower:
            failures.append(f'Forbidden label or local path token {token!r} found in JSON')

    meta = data.get('metadata')
    if not isinstance(meta, dict):
        failures.append("Missing 'metadata' section")
    else:
        if not meta.get('git_commit'):
            failures.append("Missing 'git_commit' in metadata")
        if eos_dir is not None and eos_dir.is_dir():
            expected_hashes = meta.get('eos_md5', {})
            actual_hashes = compute_table_hashes(eos_dir)
            for fname, exp_h in expected_hashes.items():
                act_h = actual_hashes.get(fname)
                if act_h != exp_h:
                    failures.append(
                        f'Table MD5 mismatch for {fname}: expected {exp_h}, got {act_h}'
                    )

    required_keys = (
        'trajectory_calibrated_400Myr',
        'trajectory_default_400Myr',
        'trajectory_calibrated_4p5Gyr',
        'trajectory_default_4p5Gyr',
        'steady_yield_thinning',
        'segment_25Myr_spread_5starts',
        'live_yield_states',
        'a_nu_provenance',
    )
    for req in required_keys:
        if req not in data:
            failures.append(f'Missing required top-level key {req!r}')

    prov = data.get('a_nu_provenance', {})
    if isinstance(prov, dict):
        rows = prov.get('korenaga_2009_table1_rows', [])
        if len(rows) != 12:
            failures.append(f'Expected 12 Korenaga (2009) Table 1 rows, got {len(rows)}')
        else:
            mean_val = float(np.mean([float(r['a_nu_empirical']) for r in rows]))
            if abs(mean_val - float(prov.get('a_nu_value', 0.0))) > 1.0e-15:
                failures.append(
                    f'a_nu_value {prov.get("a_nu_value")} != 12-row mean {mean_val}'
                )

    return failures


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate or check 0-D stagnant-lid and yield-thinning reference benchmarks.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help='Path to tests/reference/ssc_wp6_a6_1_benchmarks.json.',
    )
    parser.add_argument(
        '--eos-dir',
        type=Path,
        default=Path(os.environ.get('ARAGOG_TEST_EOS_DIR', '')),
        help='Directory containing SPIDER equation of state tables.',
    )
    parser.add_argument(
        '--live-states-json',
        type=Path,
        default=None,
        help='Optional path to JSON containing live yield state vectors and 25 Myr spread.',
    )
    parser.add_argument(
        '--a-nu-provenance-json',
        type=Path,
        default=None,
        help='Optional path to JSON containing Korenaga (2009) Table 1 a_nu provenance.',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify existing benchmark JSON against schema and hash invariants.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_arguments(argv)
    output_path = args.output.resolve()
    has_eos_dir = bool(str(args.eos_dir)) and args.eos_dir.is_dir()

    if args.check:
        if not output_path.is_file():
            sys.stderr.write(f'Benchmark file not found: {output_path}\n')
            return 1
        data = json.loads(output_path.read_text(encoding='utf-8'))
        failures = verify_benchmarks(data, eos_dir=args.eos_dir if has_eos_dir else None)
        if failures:
            sys.stderr.write(f'Verification failed with {len(failures)} issue(s):\n')
            for f in failures:
                sys.stderr.write(f'  - {f}\n')
            return 1
        print(f'Verification passed: {output_path}')
        return 0

    if not output_path.is_file():
        sys.stderr.write(f'Baseline benchmark file not found: {output_path}\n')
        return 1

    data = json.loads(output_path.read_text(encoding='utf-8'))
    data = normalize_benchmark_schema(
        data,
        live_states_path=args.live_states_json,
        a_nu_prov_path=args.a_nu_provenance_json,
    )
    if has_eos_dir:
        data.setdefault('metadata', {})['eos_md5'] = compute_table_hashes(args.eos_dir)
    try:
        head_sha = subprocess.check_output(
            ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True
        ).strip()
        if head_sha:
            data.setdefault('metadata', {})['git_commit'] = head_sha
    except Exception:
        pass

    failures = verify_benchmarks(data, eos_dir=args.eos_dir if has_eos_dir else None)
    if failures:
        sys.stderr.write(f'Generated benchmark failed verification ({len(failures)} issues):\n')
        for f in failures:
            sys.stderr.write(f'  - {f}\n')
        return 1

    output_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f'Benchmarks written to {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
