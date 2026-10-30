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
The script stops unless every run logged that PROTEUS commit (the log gives the commit, not the
state of the tree), only the MAIN runs used that Aragog commit, no Aragog version carries a
``.d<date>`` suffix (a modified tree), and RUN and REFERENCE used one Aragog version. The last
interior snapshot of each run must be at the time of its last helpfile row. For v5b it also
checks the driver flags of each run (``v5b_result.json``): REFERENCE at ``--reference-rtol``
(default 1e-10), the round-off runs of ``--floor`` at ``s0_eps`` = +-1e-12 with the end time and
row count of their base run, the other runs at ``s0_eps`` = 0, and an end ``Phi_global`` of 0;
for v5 an end ``Phi_global`` of 1. The result file holds the versions of each run. Run from the
repository root::

    python tools/verification/compare_v5_runs.py --case v5 RUN MAIN [--output FILE]
    python tools/verification/compare_v5_runs.py --case v5b RUN MAIN --reference REF \
        [--reference-rtol RTOL] [--floor RUN_P RUN_M MAIN_P MAIN_M] [--output FILE]
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
    'v5b': DATA / 'v5b_aragog_calliope.toml',
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
SOLID_STEPS = 3  # coupling steps with Phi_global = 0 at the end of a v5b run
S0_EPS = 1.0e-12  # relative start offset of the v5b round-off runs


def pins(config):
    """Return the PROTEUS and Aragog commits in the comment header of a case config."""
    head = ''.join(
        line for line in Path(config).read_text().splitlines(True) if line[:1] == '#'
    )
    found = {k: re.search(k + r' ([0-9a-f]{40})', head) for k in ('PROTEUS', 'Aragog')}
    if not all(found.values()):
        raise SystemExit(f'{config}: the header needs a PROTEUS and an Aragog commit')
    return {k: m.group(1) for k, m in found.items()}


def _folder(path):
    path = Path(path)
    return path / 'output' if (path / 'output' / 'runtime_helpfile.csv').exists() else path


def versions(path):
    """Return the PROTEUS git hash, the Aragog version and the driver flags of one run."""
    log = (_folder(path) / 'proteus_00.log').read_text()
    found = dict(
        PROTEUS=re.search(r'PROTEUS git hash\s+(\S+)', log).group(1),
        Aragog=re.search(r'Interior module\s+aragog version\s+(\S+)', log).group(1),
    )
    driver = Path(path) / 'v5b_result.json'
    if driver.exists():
        d = json.loads(driver.read_text())
        keys = ('reference', 'reference_rtol', 'solid_steps', 's0_eps')
        found.update({k: d[k] for k in keys if k in d})
    return found


def end_state(path):
    """Return the last helpfile row and the profiles of the last interior snapshot of one run."""
    out = _folder(path)
    with open(out / 'runtime_helpfile.csv') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    snaps = {float(p.name[:-7].replace('p', '.')): p for p in (out / 'data').glob('*_int.nc')}
    t_snap = max(snaps)
    if abs(t_snap - float(rows[-1]['Time'])) > 1.0e-3:  # the file name keeps 3 decimals
        raise SystemExit(
            f'{out}: last snapshot at {t_snap}, last helpfile Time {rows[-1]["Time"]}'
        )
    with netCDF4.Dataset(snaps[t_snap]) as nc:
        prof = {k: np.asarray(nc.variables[k][:], dtype=float) for k in PROFILES}
    return {**{k: float(rows[-1][k]) for k in COLUMNS}, **prof, 'rows': len(rows)}


def check_runs(case, runs, state, reference_rtol=1.0e-10):
    """Stop unless each run fits its role: versions, driver flags, end time and melt fraction."""
    pin = pins(CONFIGS[case])
    for name, v in runs.items():
        role, _, tag = name.partition('_')
        sign = {'p': 1, 'm': -1}.get(tag, 0)
        bad = bool(re.search(r'\.d\d{8}', v['Aragog'])) or v['PROTEUS'] != pin['PROTEUS']
        bad = bad or v['Aragog'] != runs['main' if role == 'main' else 'run']['Aragog']
        bad = bad or state[name]['Phi_global'] != (0.0 if case == 'v5b' else 1.0)
        if case == 'v5b':
            want = dict(
                reference=role == 'reference',
                reference_rtol=reference_rtol if role == 'reference' else None,
                solid_steps=SOLID_STEPS,
                s0_eps=sign * S0_EPS,
            )
            bad = bad or any(k not in v or v[k] != w for k, w in want.items())
            end = [(state[n]['Time'], state[n]['rows']) for n in (name, role)]
            bad = bad or end[0] != end[1]
        # The logged version is the string set when Aragog was installed, not read from git.
        on_pin = v['Aragog'].rsplit('+g', 1)[-1][:7] == pin['Aragog'][:7]
        if bad or on_pin != (role == 'main'):
            raise SystemExit(f'{case} {name}: run {v} does not fit its role (pins {pin})')


def compare(case, run, main, reference=None, floor=None, reference_rtol=1.0e-10):
    """Return the per-column result and the verdict of one case.

    ``floor`` holds the v5b round-off runs RUN_P, RUN_M, MAIN_P, MAIN_M (``--s0-eps`` +- eps);
    a column then fails only when the error of RUN exceeds that of MAIN by more than the
    largest round-off offset of either code.
    """
    paths = dict(run=run, main=main, **({'reference': reference} if case == 'v5b' else {}))
    paths.update(zip(('run_p', 'run_m', 'main_p', 'main_m'), floor or ()))
    if len({Path(p).resolve() for p in paths.values()}) < len(paths):
        raise SystemExit('each run folder must be given once')
    found = {k: versions(p) for k, p in paths.items()}
    st = {k: end_state(p) for k, p in paths.items()}
    check_runs(case, found, st, reference_rtol)
    columns = {}
    for k in (*COLUMNS, *PROFILES):
        if case == 'v5':
            eq = bool(np.array_equal(st['run'][k], st['main'][k]))
            columns[k] = dict(equal=eq, passed=eq)
            continue
        err = {n: _maxdiff(st[n][k], st['reference'][k]) for n in ('run', 'main')}
        pert = [_maxdiff(st[n][k], st[n.split('_')[0]][k]) for n in paths if '_' in n]
        fl = max(pert, default=0.0)
        passed = err['run'] <= err['main'] + fl
        columns[k] = dict(error_run=err['run'], error_main=err['main'], floor=fl, passed=passed)
    passed = all(c['passed'] for c in columns.values())
    return dict(case=case, passed=passed, versions=found, columns=columns)


def _maxdiff(a, b):
    return float(np.max(np.abs(np.subtract(a, b))))


def main(argv=None) -> int:
    """Compare the runs of one case, write the result and exit 1 when a column fails."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--case', required=True, choices=sorted(CONFIGS))
    ap.add_argument('run', type=Path)
    ap.add_argument('main', type=Path)
    ap.add_argument('--reference', type=Path, default=None)
    ap.add_argument('--reference-rtol', type=float, default=1.0e-10)
    ap.add_argument('--floor', type=Path, nargs=4, default=None, metavar='DIR')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args(argv)
    if (args.case == 'v5b') != (args.reference is not None) or args.case == 'v5' and args.floor:
        ap.error('--reference is required for v5b; v5 takes neither --reference nor --floor')
    result = compare(
        args.case, args.run, args.main, args.reference, args.floor, args.reference_rtol
    )
    for k, c in result['columns'].items():
        print(f'{k:16s} ' + ' '.join(f'{n}={v}' for n, v in c.items()))
    print(f'{args.case}: ' + ('PASS' if result['passed'] else 'FAIL'))
    out = args.output or Path(f'step_cap_{args.case}_compare.json')
    out.write_text(json.dumps(result) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
