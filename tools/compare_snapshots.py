"""Compare two aragog NetCDF snapshots variable by variable.

usage: python tools/compare_snapshots.py A.nc B.nc [--ignore NAME ...]

Prints every variable whose values differ (bitwise, NaN equal to NaN) with its largest
relative difference, and every variable present in only one file. Exits 1 when a
variable outside ``--ignore`` differs or is missing, 0 otherwise.
"""

from __future__ import annotations

import argparse
import sys

import netCDF4
import numpy as np


def compare(path_a: str, path_b: str) -> tuple[dict[str, float], set[str]]:
    """Return ``{name: max relative difference}`` for differing variables, and the names in one file only."""
    with netCDF4.Dataset(path_a) as a, netCDF4.Dataset(path_b) as b:
        only = set(a.variables) ^ set(b.variables)
        diff = {}
        for name in sorted(set(a.variables) & set(b.variables)):
            x = np.asarray(a.variables[name][:], dtype=float)
            y = np.asarray(b.variables[name][:], dtype=float)
            if x.shape != y.shape:
                diff[name] = float('inf')
            elif not np.array_equal(x, y, equal_nan=True):
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.abs(x - y) / np.maximum(np.abs(x), np.finfo(float).tiny)
                diff[name] = float(np.nanmax(rel))
    return diff, only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--ignore', nargs='*', default=[], help='variables allowed to differ')
    args = parser.parse_args(argv)
    diff, only = compare(args.a, args.b)
    for name, rel in diff.items():
        print(
            f'{name}: max relative difference {rel:.3g}'
            + (' (ignored)' if name in args.ignore else '')
        )
    for name in sorted(only):
        print(f'{name}: in one file only')
    return int(bool((only | set(diff)) - set(args.ignore)))


if __name__ == '__main__':
    sys.exit(main())
