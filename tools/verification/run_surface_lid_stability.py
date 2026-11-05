"""Density sign and Rayleigh-Taylor time scales of the lid along 2 real runs, for the Validation page.

``solid`` evaluates the output fields at stored states of a run of ``run_surface_lid_case`` (the
hand-over column on 100 nodes); ``molten`` runs the 10-node shell of
``tests/reference/surface_lid_v2_column.py`` from a molten start with the solid rheology. Run it from
the root of a checkout, with its ``src`` and the root on ``PYTHONPATH``::

    python tools/verification/run_surface_lid_stability.py RUN OUT.json --eos-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ssc_helpers import entropy_eos_copy  # noqa: E402
from run_surface_lid_case import STEM  # noqa: E402
from run_surface_lid_merge_state import _solver  # noqa: E402

FIELDS = ('lid_drho', 'lid_L', 'lid_tau_rt', 'lid_tau_th', 'lid_L_crit', 'lid_rt_flag')
MOLTEN_TIMES_YR = np.geomspace(0.1, 300.0, 12)
SOLID_ROWS = 30  # stored rows kept: this many equal in log time and about this many denser
COLUMN = dict(eta=1e18, heating=1e-10)  # the hand-over column of the Validation page


def _row(s, t, y) -> dict:
    """Output fields of the solver ``s`` at the state ``y`` of time ``t`` [yr]."""
    s._dSdt_single(t, np.array(y))
    ev = s._lid_last
    fields = dict.fromkeys(FIELDS, float('nan')) | s._lid_stability(ev)
    return dict(t=float(t), delta=ev.delta, w=ev.rates.w, T_b=ev.T_b, **fields)


def _table(rows: list[dict]) -> dict:
    """The rows as one list per column."""
    return {k: [row[k] for row in rows] for k in rows[0]}


def solid(run: Path, eos_dir: Path) -> dict:
    """Fields at every stored state of the run folder ``run`` before its hand-over: the counts of
    all rows and a sample of them."""
    a = SimpleNamespace(eos_dir=eos_dir, nodes=100, **COLUMN)
    s = _solver(a, lid=True, nodes=100, cell=None, end=1.0)
    n = len(json.loads((run / f'{STEM}.json').read_text()))
    parts = [np.load(run / f'{STEM}_part{k}.npz') for k in range(n)]
    t = np.concatenate([p['t'] for p in parts])
    y = np.concatenate([p['y'] for p in parts], axis=1)
    live = np.flatnonzero((t > 0.0) & (y[-2] > 0.0))
    rows = [_row(s, t[i], y[:, i]) for i in live]
    dense = np.array([r['lid_drho'] > 0.0 for r in rows])
    times = t[live]
    wide = np.searchsorted(times, np.geomspace(times[0], times[-1], SOLID_ROWS)).clip(
        max=live.size - 1
    )
    denser = np.flatnonzero(dense)[:: max(1, int(dense.sum()) // SOLID_ROWS)]
    picks = np.unique(np.concatenate([wide, denser]))
    return dict(
        rows=len(rows),
        rows_denser=int(dense.sum()),
        rows_flag_1=sum(r['lid_rt_flag'] == 1.0 for r in rows),
        **_table([rows[i] for i in picks]),
    )


def molten(eos_dir: Path) -> dict:
    """Fields along chained solves of the 10-node shell from a molten start."""
    from tests.reference.surface_lid_v2_column import molten_start, shell

    eos, rows, y, t_prev = entropy_eos_copy(eos_dir), [], None, 0.0
    for t in MOLTEN_TIMES_YR:
        s = shell(10, True, eos, float(t), t_prev, rheology=True, convection=True)
        if y is not None:
            s.set_initial_lid(float(y[-2]), float(y[-1]))
        s.set_initial_entropy(np.asarray(molten_start(s, eos) if y is None else y[: s._n_stag]))
        s.solve()
        y, t_prev = np.asarray(s._solution.y[:, -1]), float(t)
        o = s.get_state()
        rows.append(
            dict(
                t=float(t),
                delta=o.lid_delta,
                w=o.lid_weight,
                T_b=o.lid_T_base,
                **{k: getattr(o, k) for k in FIELDS},
            )
        )
    return _table(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--eos-dir', type=Path, required=True)
    a = p.parse_args()
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
    ).stdout.strip()
    md5 = hashlib.md5((a.eos_dir / 'temperature_solid.dat').read_bytes()).hexdigest()
    run_sha = json.loads((a.run / f'{STEM}.json').read_text())[0]['sha']
    data = dict(
        meta=dict(commit=head, run_commit=run_sha, md5=md5, **COLUMN),
        solid=solid(a.run, a.eos_dir),
        molten=molten(a.eos_dir),
    )
    body = ',\n'.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in data.items())
    a.out.write_text('{' + body + '}\n')


if __name__ == '__main__':
    main()
