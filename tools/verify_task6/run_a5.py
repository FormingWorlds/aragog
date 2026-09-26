#!/usr/bin/env python3
"""Run Step A5: PROTEUS unit tier across paired worktrees."""
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

PAIRS = [
    (
        'main',
        'wt-proteus-main',
        'wt-aragog-main',
    ),
    (
        'tl/aragog-surface-half-cell',
        'wt-proteus-tl_aragog-surface-half-cell',
        'wt-aragog-tl_aragog-skin-temperature',
    ),
    (
        'tl/aragog-mesh-refinement',
        'wt-proteus-tl_aragog-mesh-refinement',
        'wt-aragog-tl_aragog-mesh-refinement',
    ),
]

def run_proteus(branch_name: str, wt_proteus: str, wt_aragog: str) -> dict:
    wt_p_dir = BASE_DIR / wt_proteus
    wt_a_dir = BASE_DIR / wt_aragog
    print(f"=== Testing PROTEUS {branch_name} paired with {wt_aragog} ===", flush=True)

    # Get git tip
    tip_res = subprocess.run(
        ['git', '-C', str(wt_p_dir), 'rev-parse', '--short=12', 'HEAD'],
        capture_output=True, text=True, check=True
    )
    tip = tip_res.stdout.strip()

    clean_name = branch_name.replace('/', '_')
    log_file = DATA_A / f"A5_{clean_name}.log"

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = f"{wt_p_dir / 'src'}:{wt_a_dir / 'src'}"

    # Verify import paths first
    check_code = f"""
import os, sys, proteus, aragog
assert os.path.realpath(proteus.__file__).startswith(os.path.realpath('{wt_p_dir}')), proteus.__file__
assert os.path.realpath(aragog.__file__).startswith(os.path.realpath('{wt_a_dir}')), aragog.__file__
"""
    check_res = subprocess.run(
        [sys.executable, '-c', check_code],
        env=env, cwd=wt_p_dir, capture_output=True, text=True
    )
    if check_res.returncode != 0:
        print(f"Import check failed for {branch_name}: {check_res.stderr}", flush=True)
        return {
            'branch': branch_name, 'tip': tip, 'passed': 0, 'failed': 0,
            'skipped': 0, 'branch_only': f'IMPORT_FAIL: {check_res.stderr.strip()}', 'status': 'BLOCKED'
        }

    cmd = [
        sys.executable, '-m', 'pytest',
        '-m', 'unit and not skip and not slow and not integration',
        '-n', '4',
        '-q',
        '-p', 'no:cacheprovider'
    ]

    is_timeout = False
    with open(log_file, 'w') as out_f:
        proc = subprocess.Popen(
            cmd,
            cwd=wt_p_dir,
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
        # Check if pytest failed with import/collection error
        first_err = "No tests ran or collection failed"
        for line in log_content.splitlines():
            if 'ERROR' in line or 'Error' in line:
                first_err = line.strip()
                break
        return {
            'branch': branch_name, 'tip': tip, 'passed': 0, 'failed': 0,
            'skipped': skipped, 'branch_only': first_err, 'status': 'BLOCKED'
        }

    failed_tests = re.findall(r'FAILED\s+([^\s:]+)', log_content)
    branch_only_str = ";".join(failed_tests) if failed_tests else "none"

    print(f"{branch_name} ({tip}): {passed} passed, {failed} failed, {skipped} skipped, failures={branch_only_str}", flush=True)

    return {
        'branch': branch_name, 'tip': tip, 'passed': passed, 'failed': failed,
        'skipped': skipped, 'branch_only': branch_only_str, 'status': 'OK',
        'failed_list': failed_tests
    }

def main():
    parser = argparse.ArgumentParser(description="Run A5 PROTEUS tests")
    parser.add_argument('--branch', type=str, help="Specific branch to test")
    args = parser.parse_args()

    summary_csv = DATA_A / 'A5_summary.csv'
    summary_csv.write_text('branch,tip,passed,failed,skipped,branch_only_failures\n')

    targets = PAIRS
    if args.branch:
        targets = [p for p in PAIRS if p[0] == args.branch]
        if not targets:
            sys.exit(f"Unknown branch {args.branch}")

    main_failures = set()

    for branch_name, wt_p, wt_a in targets:
        res = run_proteus(branch_name, wt_p, wt_a)
        if branch_name == 'main':
            main_failures = set(res.get('failed_list', []))
            branch_only = "none" if not main_failures else ";".join(sorted(main_failures))
        else:
            cur_failures = set(res.get('failed_list', []))
            diff = cur_failures - main_failures
            branch_only = "none" if not diff else ";".join(sorted(diff))

        with open(summary_csv, 'a') as f:
            f.write(f"{res['branch']},{res['tip']},{res['passed']},{res['failed']},{res['skipped']},\"{branch_only}\"\n")

    print(f"=== A5 completed, wrote {summary_csv} ===", flush=True)

if __name__ == '__main__':
    main()
