#!/usr/bin/env python3
"""Run A1 unit regression tier across aragog branches."""
import argparse
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path('/Users/timlichtenberg/work/ssc-verify-task6')
DATA_A = BASE_DIR / 'data' / 'A'
DATA_A.mkdir(parents=True, exist_ok=True)

ALL_BRANCHES = [
    ('tl/adaptive-phase-cap', 'wt-aragog-tl_adaptive-phase-cap'),
    ('tl/aragog-fixed-cmb-temperature', 'wt-aragog-tl_aragog-fixed-cmb-temperature'),
    ('tl/aragog-mesh-refinement', 'wt-aragog-tl_aragog-mesh-refinement'),
    ('tl/aragog-skin-temperature', 'wt-aragog-tl_aragog-skin-temperature'),
    ('tl/ssc-cmb-bl-law', 'wt-aragog-tl_ssc-cmb-bl-law'),
    ('tl/ssc-mlt-calibration', 'wt-aragog-tl_ssc-mlt-calibration'),
]

def run_branch(branch_name: str, wt_name: str, summary_csv: Path) -> dict:
    wt_dir = BASE_DIR / wt_name
    print(f"=== Testing {branch_name} in {wt_name} ===", flush=True)

    # Get git tip
    tip_res = subprocess.run(
        ['git', '-C', str(wt_dir), 'rev-parse', '--short=12', 'HEAD'],
        capture_output=True, text=True, check=True
    )
    tip = tip_res.stdout.strip()

    clean_name = branch_name.replace('/', '_')
    log_file = DATA_A / f"A1_{clean_name}.log"

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['ARAGOG_TEST_EOS_DIR'] = str(BASE_DIR / 'test-data' / 'spider_eos')

    cmd = [
        sys.executable, '-m', 'pytest',
        '-m', 'unit and not slow',
        '-n', '4',
        '-q',
        '-p', 'no:cacheprovider'
    ]

    is_timeout = False
    with open(log_file, 'w') as out_f:
        proc = subprocess.Popen(
            cmd,
            cwd=wt_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            is_timeout = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            out_f.write("\nTIMEOUT: process timed out after 1800 seconds\n")

    log_content = log_file.read_text()
    if is_timeout:
        print(f"FAILED: {branch_name} hit 1800s timeout", flush=True)
        return {
            'branch': branch_name, 'tip': tip, 'passed': 0, 'failed': 0,
            'skipped': 0, 'branch_only': 'TIMEOUT', 'status': 'TIMEOUT'
        }

    # Verify summary line exists
    # Pytest format: "752 passed, 4 warnings in 64.78s" or "X failed, Y passed in ..."
    summary_match = re.search(r'(=*\s*[\d\w\s,]+in\s+[\d\.]+s.*|(\d+)\s+passed.*in\s+[\d\.]+s)', log_content)
    if not summary_match:
        # Check if any passed/failed lines exist
        m_pass = re.search(r'(\d+)\s+passed', log_content)
        m_fail = re.search(r'(\d+)\s+failed', log_content)
        if not m_pass and not m_fail:
            raise RuntimeError(f"Pytest summary line missing for {branch_name}! Log has {len(log_content)} bytes.")

    passed = 0
    failed = 0
    skipped = 0

    m_pass = re.search(r'(\d+)\s+passed', log_content)
    if m_pass:
        passed = int(m_pass.group(1))
    m_fail = re.search(r'(\d+)\s+failed', log_content)
    if m_fail:
        failed = int(m_fail.group(1))
    m_skip = re.search(r'(\d+)\s+skipped', log_content)
    if m_skip:
        skipped = int(m_skip.group(1))

    if passed == 0 and failed == 0:
        raise RuntimeError(f"Pytest reported 0 passed and 0 failed for {branch_name}! Invalid result.")

    # Extract failed test names
    failed_tests = re.findall(r'FAILED\s+([^\s:]+)', log_content)
    branch_only_str = ";".join(failed_tests) if failed_tests else "none"

    print(f"{branch_name} ({tip}): {passed} passed, {failed} failed, {skipped} skipped, failures={branch_only_str}", flush=True)

    return {
        'branch': branch_name, 'tip': tip, 'passed': passed, 'failed': failed,
        'skipped': skipped, 'branch_only': branch_only_str, 'status': 'OK'
    }

def main():
    parser = argparse.ArgumentParser(description="Run A1 regression tests")
    parser.add_argument('--branch', type=str, help="Specific branch to test")
    parser.add_argument('--init-csv', action='store_true', help="Initialize A1_summary.csv")
    args = parser.parse_args()

    summary_csv = DATA_A / 'A1_summary.csv'

    if args.init_csv or not summary_csv.exists():
        summary_csv.write_text('branch,tip,passed,failed,skipped,branch_only_failures\n')

    if args.branch:
        targets = [b for b in ALL_BRANCHES if b[0] == args.branch]
        if not targets:
            sys.exit(f"Unknown branch: {args.branch}")
    else:
        targets = ALL_BRANCHES
        summary_csv.write_text('branch,tip,passed,failed,skipped,branch_only_failures\n')

    for branch_name, wt_name in targets:
        res = run_branch(branch_name, wt_name, summary_csv)
        # Append to CSV if not already present or updating
        lines = summary_csv.read_text().splitlines()
        header = lines[0]
        rows = [line for line in lines[1:] if not line.startswith(branch_name + ',')]
        new_row = f"{res['branch']},{res['tip']},{res['passed']},{res['failed']},{res['skipped']},\"{res['branch_only']}\""
        rows.append(new_row)
        summary_csv.write_text(header + '\n' + '\n'.join(rows) + '\n')

    print("=== A1 run complete ===", flush=True)

if __name__ == '__main__':
    main()
