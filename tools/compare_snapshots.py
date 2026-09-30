"""Compare two aragog NetCDF snapshots variable by variable.

usage: python tools/compare_snapshots.py A.nc B.nc [--ignore NAME[,NAME...]]

Prints each variable that differs (bitwise, NaN equal to NaN) with its largest relative
difference against A and the number of positions where only one file holds NaN, and each
variable present in one file only; a variable named in ``--ignore`` is marked ``(ignored)``.
Exits 0 when nothing outside ``--ignore`` differs, 1 when something does, and 2 when a file
cannot be read.
"""

from __future__ import annotations

import argparse
import sys

import netCDF4
import numpy as np


def compare(path_a: str, path_b: str) -> tuple[dict[str, str], set[str]]:
    """Return ``{name: description}`` for differing variables, and the names in one file only."""
    with netCDF4.Dataset(path_a) as a, netCDF4.Dataset(path_b) as b:
        only = set(a.variables) ^ set(b.variables)
        diff = {}
        for name in sorted(set(a.variables) & set(b.variables)):
            x, y = np.asarray(a.variables[name][:]), np.asarray(b.variables[name][:])
            if x.shape != y.shape:
                diff[name] = f'shape {x.shape} against {y.shape}'
            elif x.dtype.kind not in 'fiu' or y.dtype.kind not in 'fiu':
                if not np.array_equal(x, y):
                    diff[name] = 'values differ'
            elif not np.array_equal(x, y, equal_nan=True):
                x, y = x.astype(float), y.astype(float)
                nan_x, nan_y = np.isnan(x), np.isnan(y)
                both = ~(nan_x | nan_y)
                with np.errstate(over='ignore', invalid='ignore'):
                    rel = np.abs(x[both] - y[both]) / np.maximum(
                        np.abs(x[both]), np.finfo(float).tiny
                    )
                parts = [f'max relative difference {rel.max():.3g}'] if np.any(rel) else []
                if np.any(nan_x != nan_y):
                    parts.append(f'NaN in one file at {int(np.sum(nan_x != nan_y))} points')
                diff[name] = ', '.join(parts)
    return diff, only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument(
        '--ignore',
        action='extend',
        type=lambda s: s.split(','),
        help='comma-separated variables allowed to differ (repeatable)',
    )
    args = parser.parse_args(argv)
    ignored = set(args.ignore or [])
    try:
        diff, only = compare(args.a, args.b)
    except OSError as err:
        print(f'error: {err}', file=sys.stderr)
        return 2
    for name, what in diff.items():
        print(f'{name}: {what}' + (' (ignored)' if name in ignored else ''))
    for name in sorted(only):
        print(f'{name}: in one file only' + (' (ignored)' if name in ignored else ''))
    return int(bool((only | set(diff)) - ignored))


if __name__ == '__main__':
    sys.exit(main())
