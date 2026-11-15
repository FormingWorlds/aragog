"""Collect the runs of the CMB mesh study into ``tests/validation/data/core_mesh_study.json``.

Each run is the output of ``tools/verification/run_core_mesh_case.py``. For each run the file
gets the mesh (node count, CMB and top cell, largest cell over the mean cell), and at 3, 50 and
100 Myr:

- ``q_cmb``: the CMB heat flux [W/m^2] of the CVODE right-hand side call nearest the time, the
  flux the solver used, with the time of that call;
- ``T_core``: the core temperature [K] of the stored state nearest the time, with its time;
- ``T_mid``: the mantle temperature [K] at the mid radius (r_cmb + r_surf) / 2, linear in radius
  between the two cells around it, from the entropy of the same stored state.

It also gets ``q_cmb`` on a time grid (the first CVODE call at or after each grid time) for the
plot, the largest convective minus conductive CMB flux over the CVODE calls, and the md5 of
every table file, of the run outputs and of the outputs of the rerun of one run (``--check-out``).

Usage::

    python tools/verification/run_core_mesh_study.py --eos-dir DIR --runs-commit SHA
        --check-commit SHA --check-run NAME --check-out <out> NAME=<out> [NAME=<out> ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.parser import Parameters
from aragog.solver.entropy_solver import EntropySolver

TIMES_YR = (3.0e6, 5.0e7, 1.0e8)
GRID_MYR = np.logspace(-2, 2, 161)
OUT = Path(__file__).parents[2] / 'tests' / 'validation' / 'data' / 'core_mesh_study.json'


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _segments_md5(path: Path) -> str:
    """md5 of the segment table without its wall-clock column."""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    j = rows[0].index('wall_s')
    text = '\n'.join(','.join(r[:j] + r[j + 1 :]) for r in rows)
    return hashlib.md5(text.encode()).hexdigest()


def outputs_md5(out: str) -> dict:
    """md5 of the outputs of a run that a rerun must reproduce."""
    md5 = {ext: _md5(Path(out + ext)) for ext in ('.cmb.csv', '.nodes.bin', '.S.npy')}
    return {'segments': _segments_md5(Path(out))} | md5


def cvode_records(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Time [yr] and flux [W/m^2] of the CVODE calls of a ``.cmb.csv`` file, in file order, and
    the largest convective minus conductive flux over all calls [W/m^2]."""
    t, q, gap = [], [], -np.inf
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            conv = float(row['q_conv'])
            gap = max(gap, conv - float(row['q_cond']) if conv == conv else -np.inf)
            if row['src'] == 'cvode':
                t.append(float(row['t_yr']))
                q.append(float(row['q']))
    t, q = np.array(t), np.array(q)
    assert t.size and np.isfinite(t).all() and np.isfinite(q).all(), path
    return t, q, gap


def stored_states(out: str) -> np.ndarray:
    """The stored states of a run, one row ``[t_yr, y...]`` each."""
    n_y = json.loads(Path(out + '.nodes.json').read_text())['n_y']
    raw = np.fromfile(out + '.nodes.bin', '<f8')
    n = raw.size // (n_y + 1)
    assert n > 0 and raw.size == n * (n_y + 1), (out, raw.size, n_y)
    return raw.reshape(n, n_y + 1)


def run_values(out: str, eos: EntropyEOS) -> dict:
    """The mesh, the values at ``TIMES_YR`` and the gridded flux of one run."""
    s = EntropySolver(Parameters.from_file(out + '.toml'), entropy_eos=eos)
    s.initialize()
    n_stag = s._n_stag
    r_b = np.ravel(s.evaluator.mesh.basic.radii)
    r_s = np.ravel(s.evaluator.mesh.staggered.radii)
    cells = np.diff(r_b)
    P = np.ravel(s._P_stag_flat)
    r_mid = 0.5 * (r_b[0] + r_b[-1])
    k_mid = int(np.searchsorted(r_s, r_mid))
    assert 0 < k_mid < n_stag and r_s[k_mid - 1] <= r_mid <= r_s[k_mid], (k_mid, r_mid)
    w = (r_mid - r_s[k_mid - 1]) / (r_s[k_mid] - r_s[k_mid - 1])
    t, q, gap = cvode_records(Path(out + '.cmb.csv'))
    a = stored_states(out)
    assert a.shape[1] - 1 in (n_stag + 1, n_stag + 2), (a.shape, n_stag)
    at = []
    for tq in TIMES_YR:
        i = int(np.argmin(np.abs(t - tq)))
        k = int(np.argmin(np.abs(a[:, 0] - tq)))
        T = np.ravel(s.entropy_eos.temperature(P, a[k, 1 : n_stag + 1]))
        at.append(
            {
                't_myr': tq / 1e6,
                'q_cmb': float(q[i]),
                't_q_myr': float(t[i] / 1e6),
                'T_core': float(a[k, -1]),
                't_state_myr': float(a[k, 0] / 1e6),
                'T_mid': float((1 - w) * T[k_mid - 1] + w * T[k_mid]),
            }
        )
    o = np.argsort(t, kind='stable')
    idx = np.clip(np.searchsorted(t[o] / 1e6, GRID_MYR), 0, t.size - 1)
    return {
        'n': int(r_b.size),
        'cmb_cell_m': float(cells[0]),
        'top_cell_m': float(cells[-1]),
        'max_cell_ratio': float(cells.max() / cells.mean()),
        'at': at,
        'q_grid': [float(x) for x in q[o][idx]],
        'end_myr': float(t.max() / 1e6),
        'max_q_conv_minus_q_cond': float(gap),
        'md5': outputs_md5(out) | {'.toml': _md5(Path(out + '.toml'))},
    }


def law_share(path: Path) -> dict:
    """The number of calls of a ``.cmb.csv`` file and the share with a convective flux above
    the conductive one."""
    n = above = 0
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            n += 1
            above += float(row['q_conv']) > float(row['q_cond'])
    return {'calls': n, 'share_conv_above_cond': above / n}


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--eos-dir', type=Path, required=True)
    ap.add_argument('--runs-commit', required=True, help='aragog commit of the stored runs')
    ap.add_argument('--check-commit', required=True, help='aragog commit of the rerun check')
    ap.add_argument('--check-run', required=True, help='name of the run rerun bit for bit')
    ap.add_argument('--check-out', required=True, help='<out> of the rerun')
    ap.add_argument(
        '--left-out', action='append', default=[], help='NAME=<out> of a run left out'
    )
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('runs', nargs='+', help='NAME=<out> of run_core_mesh_case.py')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    eos = EntropyEOS(str(a.eos_dir))
    runs = dict(r.split('=', 1) for r in a.runs)
    assert a.check_run in runs, a.check_run
    build = json.loads((a.eos_dir / 'BUILD_INFO.json').read_text())
    keep = ('P_max_Pa', 'nP', 'nS', 'mushy_zone_factor', 'mass_M_E', 'zalmoxis_version')
    study = {
        'header': {
            'runs_commit': a.runs_commit,
            'check': {'run': a.check_run, 'commit': a.check_commit}
            | {'md5': outputs_md5(a.check_out)},
            'times_yr': list(TIMES_YR),
            'grid_myr': [float(x) for x in GRID_MYR],
            'eos_build': {k: build[k] for k in keep} | {'sources_md5': build['sources_md5']},
            'eos_md5': {p.name: _md5(p) for p in sorted(a.eos_dir.glob('*.dat'))},
            'left_out': {
                name: law_share(Path(out + '.cmb.csv'))
                for name, out in (x.split('=', 1) for x in a.left_out)
            },
        },
        'runs': {name: run_values(out, eos) for name, out in runs.items()},
    }
    a.out.write_text(json.dumps(study, indent=1) + '\n')
    print('wrote', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
