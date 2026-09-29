"""Diagnostic of clamped thermal expansivity alpha nodes.

Evaluates the count and radial distribution of nodes carrying clamped alpha
(negative raw alpha smoothed to ~1e-13 1/K by the eps_a = 1e-8 guard)
in the 175 kyr state and the 283 kyr (melt fraction <= 0.05) state.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.parser import Parameters
from aragog.solver.entropy_solver import EntropySolver


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    fwl_data = os.environ.get('FWL_DATA')
    candidates = [
        os.environ.get('ARAGOG_TEST_EOS_DIR'),
        f'{fwl_data}/aragog/spider_eos' if fwl_data else None,
        str(repo_root.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
        '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
        '/Users/timlichtenberg/work/ssc-verify-task6/test-data/spider_eos',
    ]
    eos_dir = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if eos_dir is None:
        raise FileNotFoundError('SPIDER EOS directory not found')

    eos = EntropyEOS(eos_dir)
    cfg_path = repo_root / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    cfg = Parameters.from_file(str(cfg_path))

    states = [
        (
            '175 kyr',
            175000.0,
            Path('/Users/timlichtenberg/work/ssc-speed-dev3/acc_states/state_175000yr.npz'),
        ),
        (
            '283 kyr (phi <= 0.05)',
            282900.0,
            Path('/Users/timlichtenberg/work/ssc-speed-dev3/acc_states/state_283kyr_ref.npz'),
        ),
    ]

    out_dir = Path(
        os.environ.get('ARAGOG_DIAGNOSTIC_DIR', repo_root / 'output' / 'diagnostics')
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    node_records = []

    for label, time_yr, st_path in states:
        solver = EntropySolver(cfg, entropy_eos=eos)
        solver.initialize()

        data = np.load(st_path)
        s_arr = data['S']
        time_s = time_yr * 365.25 * 86400.0
        solver.state.update(s_arr, time=time_s)

        alpha_b = np.asarray(solver.state.phase_basic.thermal_expansivity()).ravel()
        alpha_s = np.asarray(solver.state.phase_staggered.thermal_expansivity()).ravel()
        phi_b = np.asarray(solver.state.phase_basic.melt_fraction()).ravel()
        phi_s = np.asarray(solver.state.phase_staggered.melt_fraction()).ravel()
        t_b = np.asarray(solver.state.phase_basic.temperature()).ravel()
        t_s = np.asarray(solver.state.phase_staggered.temperature()).ravel()
        p_b = np.asarray(solver._P_basic_flat).ravel()
        p_s = np.asarray(solver._P_stag_flat).ravel()
        r_b = np.asarray(solver.evaluator.mesh.basic.radii).ravel()
        r_s = np.asarray(solver.evaluator.mesh.staggered.radii).ravel()

        clamped_b = np.where(alpha_b < 1.0e-8)[0]
        clamped_s = np.where(alpha_s < 1.0e-8)[0]

        summary_records.append(
            {
                'state': label,
                'time_yr': time_yr,
                'n_basic_total': len(alpha_b),
                'n_basic_clamped': len(clamped_b),
                'basic_clamped_indices': clamped_b.tolist(),
                'n_staggered_total': len(alpha_s),
                'n_staggered_clamped': len(clamped_s),
                'staggered_clamped_indices': clamped_s.tolist(),
                'min_alpha_basic': float(np.min(alpha_b)),
                'min_alpha_staggered': float(np.min(alpha_s)),
            }
        )

        for idx in clamped_b:
            node_records.append(
                {
                    'state': label,
                    'grid': 'basic',
                    'index': int(idx),
                    'radius_m': float(r_b[idx]),
                    'pressure_gpa': float(p_b[idx] / 1.0e9),
                    'temperature_k': float(t_b[idx]),
                    'melt_fraction': float(phi_b[idx]),
                    'clamped_alpha': float(alpha_b[idx]),
                }
            )

        for idx in clamped_s:
            node_records.append(
                {
                    'state': label,
                    'grid': 'staggered',
                    'index': int(idx),
                    'radius_m': float(r_s[idx]),
                    'pressure_gpa': float(p_s[idx] / 1.0e9),
                    'temperature_k': float(t_s[idx]),
                    'melt_fraction': float(phi_s[idx]),
                    'clamped_alpha': float(alpha_s[idx]),
                }
            )

    # Write JSON summary
    json_path = out_dir / 'f21_clamped_alpha_diagnostic.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary_records, 'nodes': node_records}, f, indent=2)

    # Write CSV summary
    csv_path = out_dir / 'f21_clamped_alpha_diagnostic.csv'
    fieldnames = [
        'state',
        'grid',
        'index',
        'radius_m',
        'pressure_gpa',
        'temperature_k',
        'melt_fraction',
        'clamped_alpha',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(node_records)

    print(f'Wrote {json_path} and {csv_path}')
    for s in summary_records:
        print(
            f'{s["state"]}: {s["n_basic_clamped"]}/{s["n_basic_total"]} basic clamped, '
            f'{s["n_staggered_clamped"]}/{s["n_staggered_total"]} stag clamped'
        )


if __name__ == '__main__':
    main()
