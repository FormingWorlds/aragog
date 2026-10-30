"""Write the raw values of the mesh refinement study runs to one compact JSON file.

Each run folder holds ``za.json`` (one record per part, written at the part end) and
``za_part<k>.npz`` (the stored output rows of part k: ``t`` [yr], ``y`` with the staggered
entropy in its first rows, ``S_sol`` the staggered solidus entropy). For every run the file
keeps the mesh (node count, measured top and CMB cell), the config hash and, per part, the end
time, status, global melt fraction, CMB temperature, the time-mean surface heat flux of the
part and the energy integrals ``F_int`` (heat lost at the surface, J) and ``F_cmb`` (heat
gained at the CMB, J). It also keeps the first frozen interval of the top cell: from the first
time the top cell entropy is at or below the solidus (linear between stored rows) to the first
time after it that it is above the solidus again. With ``--phi-rows DIR`` it adds the time mean
of the global melt fraction over [t/2, t] for t = 1e3, 1e4, 1e5 yr, from ``DIR/<run>.npz``
(arrays ``t`` and ``Phi`` on the stored rows; a run without that file has no time mean). No
derived number is stored: ``tests/test_mesh_refinement_study.py`` computes them.

Usage::

    python tools/verification/run_mesh_refinement_study.py <run root> <out.json> <run>...
        [--phi-rows DIR] [--note TEXT]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _first_frozen(t, s_top, s_sol):
    """First frozen interval of the top cell [yr], or None when it never freezes."""
    d = s_top - s_sol

    def cross(i):
        return float(t[i - 1] + (t[i] - t[i - 1]) * d[i - 1] / (d[i - 1] - d[i]))

    fz = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    if not fz.size:
        return None
    th = fz[0] + np.flatnonzero((d[fz[0] - 1 : -1] <= 0) & (d[fz[0] :] > 0))
    thaw = cross(th[0]) if th.size else None
    return {'freeze_yr': cross(fz[0]), 'thaw_yr': thaw}


def run_values(folder: Path, phi_rows: Path | None = None) -> dict:
    """Raw values of one run folder."""
    rows = json.loads((folder / 'za.json').read_text())
    parts = sorted(
        folder.glob('za_part*.npz'),
        key=lambda p: int(re.search(r'(\d+)\.npz$', p.name).group(1)),
    )
    if len(parts) != len(rows) or [r['part'] for r in rows] != list(range(len(rows))):
        raise ValueError(f'{folder}: {len(parts)} part files for {len(rows)} records')
    if len({r['sha'] for r in rows}) != 1 or len({r.get('config_hash') for r in rows}) != 1:
        raise ValueError(f'{folder}: parts from more than one tree or config')
    t, s_top, s_sol = [], [], None
    for p in parts:
        z = np.load(p)
        s_sol = z['S_sol'][-1] if s_sol is None else s_sol
        t.append(z['t'])
        s_top.append(z['y'][z['S_sol'].size - 1])
    t, idx = np.unique(np.concatenate(t), return_index=True)
    r0 = rows[0]
    out = {
        'tree': r0['sha'],
        'config_hash': r0.get('config_hash'),
        'nodes': r0['nodes'],
        'surface_cell_m': r0.get('surface_cell', 0.0),
        'cmb_cell_m': r0.get('cmb_cell', 0.0),
        'h_top_m': r0['h_top'],
        'h_cmb_m': r0['h_cmb'],
        's_init': r0['s_init'],
        'core_bc': r0.get('core_bc'),
        'eos_md5': r0.get('eos_md5'),
        't_yr': [r['t_yr'] for r in rows],
        'status': [r['status'] for r in rows],
        'Phi': [r['Phi_global'] for r in rows],
        'T_cmb_K': [r['T_cmb'] for r in rows],
        'F_top_mean_W_m2': [r['F_top_mean_step_integral'] for r in rows],
        'F_int_J': [r['energy_integrals']['F_int'] for r in rows],
        'F_cmb_J': [r['energy_integrals']['F_cmb'] for r in rows],
        'nst': [r['cvode']['nst'] for r in rows],
        'first_frozen': _first_frozen(t, np.concatenate(s_top)[idx], s_sol),
    }
    if phi_rows is not None and (phi_rows / f'{folder.name}.npz').exists():
        z = np.load(phi_rows / f'{folder.name}.npz')
        tp, phi = z['t'], z['Phi']
        mean = {}
        for ts in (1e3, 1e4, 1e5):
            m = (tp >= 0.5 * ts) & (tp <= ts * (1 + 1e-9))
            if m.sum() < 6 or tp[m][-1] < ts * (1 - 1e-9):
                raise ValueError(
                    f'{folder.name}: too few stored rows in [{ts / 2:g}, {ts:g}] yr'
                )
            mean[f'{ts:.0e}'] = float(np.trapezoid(phi[m], tp[m]) / (tp[m][-1] - tp[m][0]))
        out['Phi_time_mean'] = mean
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--phi-rows', type=Path)
    ap.add_argument('--note', default='')
    a = ap.parse_args(argv)
    res = {
        'header': {
            'generator': 'tools/verification/run_mesh_refinement_study.py',
            'note': a.note,
        }
    }
    res.update({r: run_values(a.root / r, a.phi_rows) for r in a.runs})
    trees = {v['tree'] for k, v in res.items() if k != 'header'}
    res['header']['trees'] = sorted(trees)
    text = (
        '{\n'
        + ',\n'.join(
            f'{json.dumps(k)}: {json.dumps(v, separators=(",", ":"))}' for k, v in res.items()
        )
        + '\n}\n'
    )
    a.out.write_text(text)
    print(a.out, len(a.runs), 'runs')


if __name__ == '__main__':
    main()
