"""Compare the coupled PROTEUS runs of the step-controller cases V5 and V5b.

V5 passes when the end state of RUN is bitwise equal to that of MAIN. V5b passes when, on each
of the 11 columns, the error of RUN against REFERENCE is not larger than the error of MAIN.
The 11 columns are the 9 helpfile columns in ``COLUMNS`` and the staggered temperature and
entropy profiles of the last interior snapshot. Each argument is a PROTEUS output folder (with
``runtime_helpfile.csv``, ``proteus_00.log`` and ``data/<time>_int.nc``) or a folder that holds
one as ``output``.

RUN uses this Aragog code. MAIN uses the Aragog commit without the controller rules near and
below the solidus. REFERENCE is ``run_v5b_proteus.py --reference`` with the code of RUN. The
header of each case config records the PROTEUS commit of the case and the Aragog commit of MAIN.
The script refuses runs made with other versions and writes the versions of each run to its
result file. Run from the repository root::

    python tools/verification/compare_v5_runs.py --case v5 RUN MAIN [--output FILE]
    python tools/verification/compare_v5_runs.py --case v5b RUN MAIN --reference REF [--output FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
CONFIGS = {
    'v5': DATA / 'v5_aragog_janus.toml',
    'v5b': DATA / 'v5b_zalmoxis_aragog_calliope.toml',
}
COLUMNS = (
    'Time',
    'T_magma',
    'T_surf',
    'F_int',
    'F_atm',
    'Phi_global',
    'M_mantle_liquid',
    'M_mantle_solid',
    'R_int',
)
PROFILES = ('temp_s', 'entropy_s')


def pins(config):
    """Return the PROTEUS and Aragog commits in the comment header of a case config."""
    head = ''.join(
        line for line in Path(config).read_text().splitlines(True) if line[:1] == '#'
    )
    return {k: re.search(k + r' ([0-9a-f]{40})', head).group(1) for k in ('PROTEUS', 'Aragog')}


def _folder(path):
    path = Path(path)
    return path / 'output' if (path / 'output' / 'runtime_helpfile.csv').exists() else path


def versions(path):
    """Return the PROTEUS git hash and the Aragog version from the log of one run."""
    log = (_folder(path) / 'proteus_00.log').read_text()
    return dict(
        PROTEUS=re.search(r'PROTEUS git hash\s+(\S+)', log).group(1),
        Aragog=re.search(r'Interior module\s+aragog version\s+(\S+)', log).group(1),
    )


def end_state(path):
    """Return the last helpfile row and the profiles of the last interior snapshot of one run."""
    out = _folder(path)
    with open(out / 'runtime_helpfile.csv') as f:
        row = list(csv.DictReader(f, delimiter='\t'))[-1]
    snap = max(
        (out / 'data').glob('*_int.nc'), key=lambda p: float(p.name[:-7].replace('p', '.'))
    )
    with netCDF4.Dataset(snap) as nc:
        prof = {k: np.asarray(nc.variables[k][:], dtype=float) for k in PROFILES}
    return {**{k: float(row[k]) for k in COLUMNS}, **prof}


def check_versions(case, runs):
    """Stop unless every run used the pinned PROTEUS and only MAIN used the pinned Aragog."""
    pin = pins(CONFIGS[case])
    for name, v in runs.items():
        on_pin = v['Aragog'].rsplit('+g', 1)[-1][:7] == pin['Aragog'][:7]
        if v['PROTEUS'] != pin['PROTEUS'] or on_pin != (name == 'main'):
            raise SystemExit(f'{case} {name}: versions {v} do not match the config pins {pin}')


def compare(case, run, main, reference=None):
    """Return the per-column result and the verdict of one case."""
    paths = dict(run=run, main=main, **({'reference': reference} if case == 'v5b' else {}))
    found = {k: versions(p) for k, p in paths.items()}
    check_versions(case, found)
    st = {k: end_state(p) for k, p in paths.items()}
    columns = {}
    for k in (*COLUMNS, *PROFILES):
        if case == 'v5':
            columns[k] = dict(equal=bool(np.array_equal(st['run'][k], st['main'][k])))
            columns[k]['passed'] = columns[k]['equal']
        else:
            err = {
                n: float(np.max(np.abs(np.subtract(st[n][k], st['reference'][k]))))
                for n in ('run', 'main')
            }
            columns[k] = dict(error_run=err['run'], error_main=err['main'])
            columns[k]['passed'] = err['run'] <= err['main']
    passed = all(c['passed'] for c in columns.values())
    return dict(case=case, passed=passed, versions=found, columns=columns)


def main(argv=None) -> int:
    """Compare the runs of one case, write the result and exit 1 when a column fails."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--case', required=True, choices=sorted(CONFIGS))
    ap.add_argument('run', type=Path)
    ap.add_argument('main', type=Path)
    ap.add_argument('--reference', type=Path, default=None)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args(argv)
    if (args.case == 'v5b') != (args.reference is not None):
        ap.error('--reference is required for v5b and not used for v5')
    result = compare(args.case, args.run, args.main, args.reference)
    for k, c in result['columns'].items():
        print(f'{k:16s} ' + ' '.join(f'{n}={v}' for n, v in c.items()))
    print(f'{args.case}: ' + ('PASS' if result['passed'] else 'FAIL'))
    out = args.output or Path(f'step_cap_{args.case}_compare.json')
    out.write_text(json.dumps(result) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
