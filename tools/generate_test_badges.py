"""Generate shields.io endpoint-badge JSON files for aragog test counts.

For each marker expression the script invokes ``pytest --collect-only -q``
to count tests without executing them, then writes a JSON file under the
``--out`` directory in the shields.io endpoint-badge schema:

    {"schemaVersion": 1, "label": "tests", "message": "<count>", "color": "blue"}

The website at FormingWorlds.github.io fetches each JSON via the raw
GitHub URL and shields.io renders the badge.

Usage
-----
    python tools/generate_test_badges.py --out .github/badges/

Notes
-----
Running the script does not execute the test suite; only collection is
triggered. Pytest exit code 5 ("no tests collected") is treated as a
successful zero count. Any other non-zero exit is a hard failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_COLLECT_RE = re.compile(r'^(\d+)(?:/\d+)?\s+tests?\s+collected\b', re.MULTILINE)

_MARKERS: tuple[tuple[str, str, str], ...] = (
    ('total', 'tests', 'not skip'),
    ('unit', 'unit', 'unit and not skip'),
    ('integration', 'integration', 'integration and not skip'),
    ('smoke', 'smoke', 'smoke and not skip'),
    ('slow', 'slow', 'slow and not skip'),
)


def count_tests(marker_expr: str) -> int:
    """Run pytest collection and return the number of selected tests.

    Parameters
    ----------
    marker_expr : str
        Pytest marker expression passed via ``-m``.

    Returns
    -------
    int
        Number of tests pytest collected for the given marker. Exit
        code 5 ("no tests collected") is mapped to 0.

    Raises
    ------
    RuntimeError
        If pytest exits with a non-zero code other than 5, or if the
        trailing summary line cannot be parsed from stdout.
    """
    proc = subprocess.run(
        ['pytest', '--collect-only', '-q', '-m', marker_expr],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 5:
        return 0
    if proc.returncode != 0:
        raise RuntimeError(
            f'pytest --collect-only -m {marker_expr!r} exited with '
            f'code {proc.returncode}\n'
            f'--- stdout ---\n{proc.stdout}\n'
            f'--- stderr ---\n{proc.stderr}'
        )
    match = _COLLECT_RE.search(proc.stdout)
    if match is None:
        raise RuntimeError(
            f'pytest summary line not found for marker {marker_expr!r}\n'
            f'--- stdout ---\n{proc.stdout}'
        )
    return int(match.group(1))


def write_badge(out_dir: Path, name: str, label: str, count: int) -> Path:
    """Write a shields.io endpoint-badge JSON file.

    Parameters
    ----------
    out_dir : Path
        Directory to write the JSON file into. Must already exist.
    name : str
        Suffix used in the filename ``tests-<name>.json``.
    label : str
        Badge label rendered on the left side of the shield.
    count : int
        Badge message rendered on the right side of the shield.

    Returns
    -------
    Path
        Path of the written JSON file.
    """
    payload = {
        'schemaVersion': 1,
        'label': label,
        'message': str(count),
        'color': 'blue',
    }
    out_path = out_dir / f'tests-{name}.json'
    out_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return out_path


def remove_stale_badge(out_dir: Path, name: str) -> bool:
    """Remove a previously written badge JSON if it exists.

    Parameters
    ----------
    out_dir : Path
        Directory the JSON file lives in.
    name : str
        Suffix in the filename ``tests-<name>.json``.

    Returns
    -------
    bool
        True if a file was removed, False if no file existed.
    """
    out_path = out_dir / f'tests-{name}.json'
    if out_path.exists():
        out_path.unlink()
        return True
    return False


def main() -> int:
    """Entry point.

    Returns
    -------
    int
        Process exit code (always 0 on success; failures raise).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Directory to write tests-<name>.json badge files into.',
    )
    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, label, expr in _MARKERS:
        count = count_tests(expr)
        if count > 0:
            write_badge(out_dir, name, label, count)
            print(f'{label}: {count}')
        else:
            removed = remove_stale_badge(out_dir, name)
            suffix = ' (stale badge removed)' if removed else ''
            print(f'{label}: 0{suffix}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
