#!/usr/bin/env python3
"""Execute discrimination tests for Step A2 and write A2_discrimination.csv."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path('/Users/timlichtenberg/work/ssc-verify-task6')
WT_MAIN = BASE_DIR / 'wt-aragog-main'
DATA_A = BASE_DIR / 'data' / 'A'
DATA_A.mkdir(parents=True, exist_ok=True)

out_csv = DATA_A / 'A2_discrimination.csv'

# Test items: (branch, source_wt, test_rel_path, target_name)
ITEMS = [
    (
        'tl/adaptive-phase-cap',
        'wt-aragog-tl_adaptive-phase-cap',
        'tests/test_phase_boundary_cap.py',
        'tmp_test_phase_boundary_cap.py',
    ),
    (
        'tl/aragog-mesh-refinement',
        'wt-aragog-tl_aragog-mesh-refinement',
        'tests/test_mesh_refinement.py',
        'tmp_test_mesh_refinement.py',
    ),
    (
        'tl/aragog-skin-temperature',
        'wt-aragog-tl_aragog-skin-temperature',
        'tests/test_surface_skin.py',
        'tmp_test_surface_skin.py',
    ),
    (
        'tl/aragog-fixed-cmb-temperature',
        'wt-aragog-tl_aragog-fixed-cmb-temperature',
        'tests/test_jax_dsdt_energy_balance.py',
        'tmp_test_jax_dsdt_energy_balance.py',
    ),
    (
        'tl/aragog-fixed-cmb-temperature',
        'wt-aragog-tl_aragog-fixed-cmb-temperature',
        'tests/test_parser.py',
        'tmp_test_parser.py',
    ),
    (
        'tl/ssc-mlt-calibration',
        'wt-aragog-tl_ssc-mlt-calibration',
        'tests/test_mlt_calibration.py',
        'tmp_test_mlt_calibration.py',
    ),
    (
        'tl/ssc-mlt-calibration',
        'wt-aragog-tl_ssc-mlt-calibration',
        'tests/test_mesh_refinement_halfspace.py',
        'tmp_test_mesh_refinement_halfspace.py',
    ),
    (
        'tl/ssc-cmb-bl-law',
        'wt-aragog-tl_ssc-cmb-bl-law',
        'tests/test_cmb_boundary_layer.py',
        'tmp_test_cmb_boundary_layer.py',
    ),
]

def main():
    rows = ['branch,test_file,result_on_main,reason']

    for branch, src_wt, test_file, tmp_name in ITEMS:
        src_path = BASE_DIR / src_wt / test_file
        dest_path = WT_MAIN / 'tests' / tmp_name

        print(f"=== Testing discrimination of {branch}:{test_file} on main ===", flush=True)
        shutil.copyfile(src_path, dest_path)

        cmd = [
            sys.executable, '-m', 'pytest',
            f'tests/{tmp_name}',
            '-q',
            '-p', 'no:cacheprovider'
        ]

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        res = subprocess.run(
            cmd,
            cwd=WT_MAIN,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            start_new_session=True
        )

        dest_path.unlink()

        # Verify git status is clean
        status_res = subprocess.run(
            ['git', '-C', str(WT_MAIN), 'status', '--porcelain'],
            capture_output=True, text=True, check=True
        )
        if status_res.stdout.strip():
            raise RuntimeError(f"wt-aragog-main dirty after {tmp_name}: {status_res.stdout}")

        # Determine pass/fail
        if res.returncode == 0:
            result = "PASS"
            reason = "test passed on main (undiscriminated)"
        else:
            result = "FAIL"
            # Extract main error reason from output
            out = res.stdout + res.stderr
            reason_line = "unknown error"
            for line in out.splitlines():
                if line.startswith('E   ') or 'ModuleNotFoundError' in line or 'ImportError' in line or 'Failed: ' in line or 'FAILED ' in line:
                    reason_line = line.strip().replace('"', "'").replace(',', ';')
                    break
            reason = reason_line

        print(f"Result: {result} ({reason})", flush=True)
        rows.append(f"{branch},{test_file},{result},\"{reason}\"")

    out_csv.write_text('\n'.join(rows) + '\n')
    print(f"Wrote {out_csv}", flush=True)

if __name__ == '__main__':
    main()
