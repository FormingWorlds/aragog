#!/usr/bin/env python3
"""Build data/MANIFEST.tsv.

Lists every file in ~/work/ssc-verify-task6/data, tab-separated:
path_relative_to_D  sha256  bytes  step  status
"""

import hashlib
import os
from pathlib import Path

DATA_DIR = Path('/Users/timlichtenberg/work/ssc-verify-task6/data')
CHECKPOINT_FILE = Path(
    '/Users/timlichtenberg/.shared-agent-state/ssc-verify-task6/CHECKPOINT.md'
)
MANIFEST_FILE = DATA_DIR / 'MANIFEST.tsv'


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoints() -> dict[str, str]:
    cp = {}
    if not CHECKPOINT_FILE.exists():
        return cp
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5 and parts[1] not in ('Step', '---'):
                step = parts[1]
                status = parts[2]
                cp[step] = status
    return cp


def determine_step(rel_path: str) -> str:
    parts = rel_path.split('/')
    top = parts[0]
    fname = parts[-1]

    if top == 'S':
        if 'import_canary' in fname:
            return 'S2'
        elif 'base_aragog' in fname:
            return 'S_base'
        else:
            return 'S1'
    elif top == 'A':
        for s in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
            if s in fname:
                return s
        return 'A'
    elif top == 'B':
        for s in ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']:
            if s in fname:
                return s
        return 'B'
    elif top == 'C':
        if len(parts) >= 2 and parts[1].startswith('C'):
            return parts[1]
        return 'C'
    elif top == 'D':
        if fname in ('b6_reference.csv', 'b6_sources.csv', 'README.rtf', 'License.txt', 'paper.html', 'paper.pdf', 'paper.txt', 'paper_layout.txt'):
            return 'D1'
        elif fname == 'b6_mapping.md':
            return 'D2'
        elif fname == 'b6_run.py':
            return 'D3'
        elif fname in ('b6_runs.csv', 'b6_A1.log', 'b6_A7.log'):
            return 'D4'
        return 'D'
    return top


def main():
    cp = load_checkpoints()
    records = []

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname == 'MANIFEST.tsv':
                continue
            full_path = Path(root) / fname
            rel_path = str(full_path.relative_to(DATA_DIR))
            file_bytes = full_path.stat().st_size
            file_sha = sha256_file(full_path)
            step = determine_step(rel_path)
            status = cp.get(step, 'DONE')
            records.append((rel_path, file_sha, str(file_bytes), step, status))

    records.sort(key=lambda x: x[0])

    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        for r in records:
            f.write('\t'.join(r) + '\n')

    print(f'Wrote {len(records)} entries to {MANIFEST_FILE}')


if __name__ == '__main__':
    main()
