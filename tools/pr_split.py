#!/usr/bin/env python3
"""Compute and verify modular pull-request splits against origin/main.

Reads mapping.csv defining the assignment of every changed file (and hunk
ranges for multi-concern files) across stream branches to pull requests,
validates that every changed file is accounted for in exactly one PR, and
writes split.csv containing exact diff statistics.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_hunks(fpath: str, repo_root: Path) -> list[dict]:
    """Parse git diff hunks for a file."""
    diff = subprocess.run(
        ['git', 'diff', 'origin/main...HEAD', fpath],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    hunk_splits = re.split(r'(?m)^(?=@@)', diff)
    hunks = []
    for i, h in enumerate(hunk_splits[1:], 1):
        lines = h.split('\n')
        hdr = lines[0]
        plus = len([l for l in lines[1:] if l.startswith('+') and not l.startswith('+++')])
        minus = len([l for l in lines[1:] if l.startswith('-') and not l.startswith('---')])
        hunks.append({'idx': i, 'hdr': hdr, 'plus': plus, 'minus': minus})
    return hunks


def expand_ranges(s: str) -> list[int]:
    """Expand comma-separated list of hunk indices and ranges."""
    if not s:
        return []
    res = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            res.extend(range(lo, hi + 1))
        else:
            res.append(int(part))
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mapping',
        type=Path,
        default=Path(
            '/Users/timlichtenberg/.shared-agent-state/streams/ssc-step1/data/A3-split/mapping.csv'
        ),
        help='Path to mapping CSV file',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(
            '/Users/timlichtenberg/.shared-agent-state/streams/ssc-step1/data/A3-split/split.csv'
        ),
        help='Output path for split CSV',
    )
    parser.add_argument(
        '--repairs-dir',
        type=Path,
        default=Path('/Users/timlichtenberg/work/stream-ssc-step1/wt-repairs'),
        help='Worktree path for wt-repairs',
    )
    parser.add_argument(
        '--bc6-dir',
        type=Path,
        default=Path('/Users/timlichtenberg/work/stream-ssc-step1/wt-bc6-rule'),
        help='Worktree path for wt-bc6-rule',
    )
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('/Users/timlichtenberg/work/stream-ssc-step1/wt-docs'),
        help='Worktree path for wt-docs',
    )
    args = parser.parse_args()

    # Load mapping
    mapping: dict[str, list[dict]] = {}
    with open(args.mapping) as f:
        reader = csv.DictReader(f)
        for r in reader:
            fpath = r['file_path']
            pr = int(r['pr'])
            hunks = r['hunks'].strip()
            desc = r['description'].strip()
            if fpath not in mapping:
                mapping[fpath] = []
            mapping[fpath].append({'pr': pr, 'hunks': hunks, 'description': desc})

    # Collect changed files across all 3 branches
    branches = {
        'repairs': args.repairs_dir,
        'docs': args.docs_dir,
        'bc6': args.bc6_dir,
    }
    all_branch_files = set()
    for _, bpath in branches.items():
        res = subprocess.run(
            ['git', 'diff', '--name-only', 'origin/main...HEAD'],
            cwd=bpath,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.strip().split('\n'):
            if line:
                all_branch_files.add(line)

    missing_files = all_branch_files - set(mapping.keys())
    extra_files = set(mapping.keys()) - all_branch_files
    if missing_files:
        sys.stderr.write(f'ERROR: Files missing from mapping: {missing_files}\n')
        return 1
    if extra_files:
        sys.stderr.write(f'ERROR: Extra files in mapping not in git diff: {extra_files}\n')
        return 1

    pr_totals = {pr: {'files': 0, 'insertions': 0, 'deletions': 0} for pr in range(1, 8)}
    split_rows = []

    for fpath in sorted(mapping.keys()):
        entries = mapping[fpath]
        cwd = args.bc6_dir if fpath == 'tests/test_bc6_loader_rule.py' else args.repairs_dir
        numstat = subprocess.run(
            ['git', 'diff', '--numstat', 'origin/main...HEAD', fpath],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        if not numstat:
            ins_tot, del_tot = 0, 0
        else:
            parts = numstat.split('\t')
            ins_tot = int(parts[0]) if parts[0] != '-' else 0
            del_tot = int(parts[1]) if parts[1] != '-' else 0

        if len(entries) == 1 and not entries[0]['hunks']:
            pr = entries[0]['pr']
            desc = entries[0]['description']
            pr_totals[pr]['files'] += 1
            pr_totals[pr]['insertions'] += ins_tot
            pr_totals[pr]['deletions'] += del_tot
            split_rows.append(
                {
                    'file_path': fpath,
                    'pr': pr,
                    'hunks': 'all',
                    'insertions': ins_tot,
                    'deletions': del_tot,
                    'net_lines': ins_tot - del_tot,
                    'description': desc,
                }
            )
        else:
            hunks = parse_hunks(fpath, cwd)
            total_hunks_count = len(hunks)
            assigned_hunks = set()
            for ent in entries:
                pr = ent['pr']
                desc = ent['description']
                hrange = expand_ranges(ent['hunks'])
                for hidx in hrange:
                    if hidx in assigned_hunks:
                        sys.stderr.write(
                            f'ERROR: Hunk {hidx} in {fpath} assigned multiple times\n'
                        )
                        return 1
                    assigned_hunks.add(hidx)
                sub_ins = sum(h['plus'] for h in hunks if h['idx'] in hrange)
                sub_del = sum(h['minus'] for h in hunks if h['idx'] in hrange)
                pr_totals[pr]['files'] += 1
                pr_totals[pr]['insertions'] += sub_ins
                pr_totals[pr]['deletions'] += sub_del
                split_rows.append(
                    {
                        'file_path': fpath,
                        'pr': pr,
                        'hunks': ent['hunks'],
                        'insertions': sub_ins,
                        'deletions': sub_del,
                        'net_lines': sub_ins - sub_del,
                        'description': desc,
                    }
                )
            if len(assigned_hunks) != total_hunks_count:
                sys.stderr.write(
                    f'ERROR: Unassigned hunks in {fpath}: '
                    f'{set(range(1, total_hunks_count + 1)) - assigned_hunks}\n'
                )
                return 1

    # Write split.csv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        fieldnames = [
            'file_path',
            'pr',
            'hunks',
            'insertions',
            'deletions',
            'net_lines',
            'description',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in split_rows:
            writer.writerow(row)

    # Print summary
    print('PR Split Verification Summary:')
    print('=' * 72)
    grand_ins = 0
    grand_del = 0
    for pr in range(1, 8):
        t = pr_totals[pr]
        grand_ins += t['insertions']
        grand_del += t['deletions']
        net = t['insertions'] - t['deletions']
        print(
            f'PR {pr}: {t["files"]:2d} file references | +{t["insertions"]:5d} / -{t["deletions"]:3d} (net {net:+5d})'
        )
    print('=' * 72)
    print(
        f'Total: {len(all_branch_files)} files accounted for | +{grand_ins} / -{grand_del} (net {grand_ins - grand_del:+d})'
    )
    print(f'Wrote detailed breakdown to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
