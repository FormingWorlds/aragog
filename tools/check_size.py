#!/usr/bin/env python3
"""Check line counts and function size limits for rheology modules."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / 'src'


def check_file_sizes() -> bool:
    rheo_file = SRC_ROOT / 'aragog' / 'rheology.py'
    if not rheo_file.exists():
        return True

    text = rheo_file.read_text(encoding='utf-8')
    lines = len(text.splitlines())
    if lines >= 500:
        print(f'ERROR: rheology.py has {lines} lines (limit: <500)')
        return False
    print(f'OK: rheology.py has {lines} lines (<500)')

    tree = ast.parse(text, filename=str(rheo_file))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_lines = (node.end_lineno or 0) - node.lineno + 1
            if fn_lines > 150:
                print(f'ERROR: function {node.name} has {fn_lines} lines (limit: <=150)')
                return False
    print('OK: All functions in rheology.py <= 150 lines')
    return True


def main():
    ok = check_file_sizes()
    if not ok:
        sys.exit(1)
    print('Size checks passed.')


if __name__ == '__main__':
    main()
