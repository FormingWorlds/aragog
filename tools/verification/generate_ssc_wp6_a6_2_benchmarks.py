"""Generate and verify reference benchmarks for internal-heating stagnant-lid scaling tests.

Generates or checks ``tests/reference/ssc_wp6_a6_2_benchmarks.json``.
Covers:
- Repository commit SHA and equation of state table MD5 checksums.
- Twelve-case planar internal-heating stagnant-lid sweep results and case specifications.
- Ordinary least-squares scaling exponent fits on both design parameters ``(Ra_i, theta)``
  and realized steady-state parameters ``(Ra_i_realized_kor, theta_realized_kor)``.
- Analytical and discrete N = 100 staggered-grid pure-conduction floor derivations.
- Converged steady-state entropy profiles and live residual bounds for five benchmark
  cases (``case_01``, ``case_03``, ``case_06``, ``case_07``, ``case_12``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK_PATH = REPO_ROOT / 'tests' / 'reference' / 'ssc_wp6_a6_2_benchmarks.json'
TABLE_FILES = ('density_melt.dat', 'density_solid.dat', 'temperature_solid.dat')
FORBIDDEN_SUBSTRINGS = (
    'ruling',
    'plan_' + 'v2',
    'work_' + 'package',
    'work-' + 'package',
    'a6_2_',
    '/users/',
)
REQUIRED_SNAPSHOT_IDS = ('case_01', 'case_03', 'case_06', 'case_07', 'case_12')
STEADY_RESIDUAL_BOUND = 1.0e-9


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


def _ols(x_cols: list[np.ndarray], y_vec: np.ndarray) -> dict[str, Any]:
    """Fit ``y = c_0 + sum_j c_j x_j`` by ordinary least squares with standard errors.

    Parameters
    ----------
    x_cols : list[np.ndarray]
        Predictor columns (excluding the constant intercept column).
    y_vec : np.ndarray
        Response vector.

    Returns
    -------
    dict[str, Any]
        Fitted coefficients, standard errors, and coefficient of determination ``r_squared``.
    """
    n = len(y_vec)
    x_mat = np.column_stack([np.ones(n), *x_cols])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    resid = y_vec - x_mat @ coeffs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    dof = max(n - x_mat.shape[1], 1)
    cov = (ss_res / dof) * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    if len(x_cols) == 2:
        ln_a, alpha, beta = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return {
            'a': float(np.exp(ln_a)),
            'ln_a': ln_a,
            'alpha': alpha,
            'beta': beta,
            'std_err_ln_a': float(se[0]),
            'std_err_alpha': float(se[1]),
            'std_err_beta': float(se[2]),
            'r_squared': r2,
            'n_cases': int(n),
            'dof': int(dof),
        }
    return {
        'slope': float(coeffs[1]),
        'std_err_slope': float(se[1]),
        'intercept_log10': float(coeffs[0]),
        'r_squared': r2,
        'n_cases': int(n),
        'dof': int(dof),
    }


def compute_discrete_conduction_floor_n100() -> dict[str, float]:
    """Compute exact analytical, planar grid, and spherical-shell N=100 pure-conduction floors."""
    n_stag = 99
    z_stag = (np.arange(n_stag, dtype=float) + 0.5) / float(n_stag)
    z_desc = z_stag[::-1]
    t_norm = z_desc * (1.0 - 0.5 * z_desc)

    def _kor_fixed(t_arr: np.ndarray, z_arr: np.ndarray) -> float:
        order = np.argsort(z_arr)
        z_s, t_s = z_arr[order], t_arr[order]
        ti = float(t_s.max())
        for _ in range(100):
            dl = ti
            zz = np.concatenate([[dl], z_s[z_s > dl]])
            tt = np.interp(zz, z_s, t_s)
            ti_new = float(np.trapezoid(tt, zz) / (zz[-1] - dl))
            if abs(ti_new - ti) < 1.0e-14:
                return ti_new
            ti = ti_new
        raise RuntimeError('Conduction floor fixed point did not converge')

    ti_planar_n100 = _kor_fixed(t_norm, z_desc)

    r_b = np.linspace(99.0, 100.0, 100)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])
    dr = 1.0 / 99.0
    f_top_norm = (100.0**3 - 99.0**3) / (3.0 * 100.0**2)
    t_sph = np.zeros(99)
    t_sph[-1] = 0.5 * dr * f_top_norm
    for i in range(98, 0, -1):
        f_i = (r_b[i] ** 3 - 99.0**3) / (3.0 * r_b[i] ** 2)
        t_sph[i - 1] = t_sph[i] + dr * f_i
    ti_sph_n100 = _kor_fixed(t_sph, 100.0 - r_s)

    dl_analytic = float(np.sqrt(6.0) - 2.0)
    return {
        'Nu_analytic_parabola_peak': 2.0,
        'Nu_analytic_volume_mean': 3.0,
        'dl_korenaga_analytic_planar': dl_analytic,
        'Nu_korenaga_analytic_planar': float(1.0 + 0.5 * np.sqrt(6.0)),
        'T_i_korenaga_norm_planar_N100': ti_planar_n100,
        'Nu_korenaga_conduction_floor_planar_N100': float(1.0 / ti_planar_n100),
        'T_i_korenaga_norm_spherical_N100': ti_sph_n100,
        'Nu_korenaga_conduction_floor_spherical_N100': float(1.0 / ti_sph_n100),
    }


def _strip_prefix(val: str) -> str:
    """Strip leading ``a6_2_`` prefix from case IDs and labels."""
    return re.sub(r'^a6_2_', '', val)


def normalize_benchmark_schema(
    data: dict[str, Any],
    snapshots_path: Path | None = None,
) -> dict[str, Any]:
    """Normalize case identifiers, compute realized-state fits, and attach 5 snapshots."""
    meta = data.setdefault('metadata', {})
    win = meta.setdefault('acceptance_window', {})
    win['source'] = (
        'Solomatov & Moresi (2000), J. Geophys. Res., 105(B9), 21795-21817, '
        'Tables 5-6 and Eq. (23)'
    )
    fp = meta.setdefault('fixed_parameters', {})
    floor_info = compute_discrete_conduction_floor_n100()
    fp['conduction_floor_Nu_analytic_peak'] = floor_info['Nu_analytic_parabola_peak']
    fp['conduction_floor_Nu_korenaga_analytic_planar'] = floor_info[
        'Nu_korenaga_analytic_planar'
    ]
    fp['conduction_floor_Nu_korenaga_planar_N100'] = floor_info[
        'Nu_korenaga_conduction_floor_planar_N100'
    ]
    fp['conduction_floor_Nu_korenaga_spherical_N100'] = floor_info[
        'Nu_korenaga_conduction_floor_spherical_N100'
    ]
    fp.pop('conduction_floor_Nu_analytic', None)
    fp.pop('conduction_floor_Nu_korenaga_N100', None)
    fp['steady_residual_bound_J_kg_K_yr'] = STEADY_RESIDUAL_BOUND

    for section_key in ('cases_spec', 'cases_results'):
        rows = data.get(section_key, [])
        for row in rows:
            if isinstance(row, dict):
                for fld in ('case_id', 'label', 'name'):
                    if fld in row:
                        row[fld] = _strip_prefix(str(row[fld]))

    diag_fits = data.get('diagnostic_fits', {})
    if isinstance(diag_fits, dict):
        for df_val in diag_fits.values():
            if isinstance(df_val, dict) and isinstance(df_val.get('case_ids'), list):
                df_val['case_ids'] = [_strip_prefix(str(cid)) for cid in df_val['case_ids']]

    cases = data.get('cases_results', [])
    if len(cases) == 12:
        ra_real = np.array([float(c['Ra_i_realized_kor']) for c in cases])
        th_real = np.array([float(c['theta_realized_kor']) for c in cases])
        nu_kor = np.array([float(c['Nu_korenaga']) for c in cases])
        nu_est = np.array([float(c['Nu_est']) for c in cases])
        mask_ra = np.array([c['sweep'] == 'Ra_i' for c in cases])
        mask_th = np.array([c['sweep'] == 'theta' for c in cases])
        mask_c7 = nu_est >= 2.8

        data['realized_state_fits'] = {
            'regressor_definition': (
                'Ra_i_realized_kor = rho * g * alpha * (T_i_korenaga - T_surf) * D^3 / '
                '(kappa * eta(T_i_korenaga)); '
                'theta_realized_kor = E_a * (T_i_korenaga - T_surf) / (R * T_i_korenaga^2)'
            ),
            'all_12_nu_korenaga_on_realized_kor': {
                'multilinear': _ols([-np.log(th_real), np.log(ra_real)], np.log(nu_kor)),
                'ra_sweep_1d': _ols([np.log10(ra_real[mask_ra])], np.log10(nu_kor[mask_ra])),
                'theta_sweep_1d': _ols(
                    [-np.log10(th_real[mask_th])], np.log10(nu_kor[mask_th])
                ),
            },
            'convecting_7_nu_korenaga_on_realized_kor': {
                'case_ids': [str(c['case_id']) for c, m in zip(cases, mask_c7) if m],
                'multilinear': _ols(
                    [-np.log(th_real[mask_c7]), np.log(ra_real[mask_c7])],
                    np.log(nu_kor[mask_c7]),
                ),
                'ra_sweep_1d': _ols(
                    [np.log10(ra_real[mask_ra & mask_c7])],
                    np.log10(nu_kor[mask_ra & mask_c7]),
                ),
                'theta_sweep_1d': _ols(
                    [-np.log10(th_real[mask_th & mask_c7])],
                    np.log10(nu_kor[mask_th & mask_c7]),
                ),
            },
        }

    snaps_in = data.get('steady_state_snapshots', {})
    if snapshots_path is not None and snapshots_path.is_file():
        snaps_in = json.loads(snapshots_path.read_text(encoding='utf-8'))

    snaps_clean: dict[str, Any] = {}
    for k, snap in snaps_in.items():
        clean_id = _strip_prefix(str(k))
        snap_copy = dict(snap)
        snap_copy['case_id'] = clean_id
        rhs_arr = np.asarray(snap_copy['rhs_final'], dtype=float)
        snap_copy['max_abs_rhs_J_kg_K_yr'] = float(np.max(np.abs(rhs_arr)))
        snaps_clean[clean_id] = snap_copy
    data['steady_state_snapshots'] = snaps_clean

    return data


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

    if 'realized_state_fits' not in data:
        failures.append("Missing 'realized_state_fits' section")

    snaps = data.get('steady_state_snapshots', {})
    for cid in REQUIRED_SNAPSHOT_IDS:
        if cid not in snaps:
            failures.append(f'Missing steady-state snapshot for {cid!r}')
        else:
            max_rhs = float(snaps[cid].get('max_abs_rhs_J_kg_K_yr', 1.0))
            if max_rhs > STEADY_RESIDUAL_BOUND:
                failures.append(
                    f'Snapshot {cid} max_abs_rhs {max_rhs:.3e} exceeds {STEADY_RESIDUAL_BOUND:.1e}'
                )

    return failures


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate or check stagnant-lid scaling reference benchmarks.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help='Path to tests/reference/ssc_wp6_a6_2_benchmarks.json.',
    )
    parser.add_argument(
        '--eos-dir',
        type=Path,
        default=Path(os.environ.get('ARAGOG_TEST_EOS_DIR', '')),
        help='Directory containing SPIDER equation of state tables.',
    )
    parser.add_argument(
        '--snapshots-json',
        type=Path,
        default=None,
        help='Optional path to JSON containing converged steady-state snapshots.',
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
    data = normalize_benchmark_schema(data, snapshots_path=args.snapshots_json)
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
