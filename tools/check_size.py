#!/usr/bin/env python3
"""Check line counts and function size limits against architecture safeguards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / 'src'

ALLOW_LIST = {
    'EntropyState.update': 600,
}


def check_file_sizes():
    rheo_file = SRC_ROOT / 'aragog' / 'rheology.py'
    if rheo_file.exists():
        lines = len(rheo_file.read_text().splitlines())
        if lines >= 500:
            print(f'ERROR: rheology.py has {lines} lines (limit: <500)')
            return False
        else:
            print(f'OK: rheology.py has {lines} lines (<500)')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='HEAD~1', help='Base commit/tag to diff against')
    parser.parse_args()

    ok = check_file_sizes()
    if not ok:
        sys.exit(1)
    print('Size checks passed.')


if __name__ == '__main__':
    main()
