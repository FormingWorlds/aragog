"""Write the raw values of the mesh refinement study runs to one compact JSON file.

Each run folder holds ``za.json`` (one record per part, written at the part end) and
``za_part<k>.npz`` (the stored output rows of part k: ``t`` [yr], ``y`` with the staggered
entropy in its first rows, ``S_sol`` the staggered solidus entropy). For every run the file
keeps the mesh (node count, measured top and CMB cell), the config hash and, per part, the end
time, status, global melt fraction, CMB temperature, the time-mean surface heat flux of the
part and the energy integrals ``F_int`` (heat lost at the surface, J) and ``F_cmb`` (heat
gained at the CMB, J). It also keeps the first frozen interval of the top cell: from the first
time the top cell entropy is at or below the solidus (linear between stored rows) to the first
time after it that it is above the solidus again, and the number of times per part that the top cell
falls to or below the solidus. With ``--phi-rows DIR`` it adds the time mean
of the global melt fraction over [t/2, t] for t = 1e3, 1e4, 1e5 yr, from ``DIR/<run>.npz``
(arrays ``t`` and ``Phi`` on the stored rows; a run without that file has no time mean). No
derived number is stored: ``tests/test_mesh_refinement_study.py`` computes them. With ``--tables DIR``
(the aragog P-S tables of the runs) and ``--cmb-pressure PA`` the header also keeps the solid
temperature table at the two pressure nodes around the CMB pressure, for the last entropy node valid at
both and the next node, and the melt temperature and density on the valid nodes at 1 bar; with
``--raw-liquid FILE`` it keeps the rows at 1 bar of that PALEOS (P, T) liquid table. Each table file is
named with its md5, and the header keeps the entropy nodes and the largest valid entropy of the solid
table at every pressure node, and the solidus of the tables. Every run keeps its staggered solidus entropy and the staggered pressure of its mesh, rebuilt from
``abe_solid.toml`` with the run's node count and end cells; the runs named with ``--profiles`` also keep
the staggered entropy at every part end.

Usage::

    python tools/verification/run_mesh_refinement_study.py <run root> <out.json> <run>...
        [--phi-rows DIR] [--note TEXT] [--tables DIR --cmb-pressure PA] [--raw-liquid FILE] [--profiles RUN...]
"""

from __future__ import annotations

import argparse
import hashlib
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


def _freezes(t, d, rows):
    """Number of downward crossings of the solidus by the top cell in each part."""
    at = t[1:][(d[:-1] > 0) & (d[1:] <= 0)]
    ends = [0.0] + [r['t_yr'] for r in rows]
    return [int(np.sum((at > a) & (at <= b))) for a, b in zip(ends[:-1], ends[1:])]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _ps_table(path: Path):
    """Pressure nodes [Pa], entropy nodes [J/kg/K] and the quantity on the (S, P) grid [SI]."""
    head = path.read_text().splitlines()[4].lstrip('#').split()
    a = np.loadtxt(path) * np.array(head, dtype=float)
    p, s = np.unique(a[:, 0]), np.unique(a[:, 1])
    return p, s, a[:, 2].reshape(s.size, p.size)


def table_values(tables: Path, p_cmb: float, raw_liquid: Path | None = None) -> dict:
    """Raw table entries around the CMB pressure (solid) and at 1 bar (melt, raw PALEOS liquid)."""
    p, s, t = _ps_table(tables / 'temperature_solid.dat')
    vm = np.loadtxt(tables / 'valid_mask_solid.dat')
    k = int(np.searchsorted(p, p_cmb))
    if not 0 < k < p.size:
        raise ValueError(f'CMB pressure {p_cmb} Pa is outside the pressure nodes of {tables}')
    j = int(np.flatnonzero((vm[:, k - 1] == 1) & (vm[:, k] == 1)).max())
    names = ['temperature_solid.dat', 'valid_mask_solid.dat', 'temperature_melt.dat']
    names += ['density_melt.dat', 'valid_mask_melt.dat', 'solidus_P-S.dat']
    last = vm.shape[0] - 1 - np.argmax(vm[::-1] == 1, axis=0)
    out = {
        'tables': tables.name,
        'md5': {n: _md5(tables / n) for n in names},
        'solid_at_cmb': {
            'P_cmb_Pa': p_cmb,
            'P_nodes_Pa': p[k - 1 : k + 1].tolist(),
            'S_nodes': s[j : j + 2].tolist(),
            'T_K': t[j : j + 2, k - 1 : k + 1].tolist(),
            'valid': vm[j : j + 2, k - 1 : k + 1].tolist(),
        },
        'solid_last_valid': {
            'P_nodes_Pa': p.tolist(),
            'S': s[last].tolist(),
            'S_grid': s.tolist(),
        },
    }
    sol = np.loadtxt(tables / 'solidus_P-S.dat') * np.array(
        (tables / 'solidus_P-S.dat').read_text().splitlines()[4].lstrip('#').split(),
        dtype=float,
    )
    out['solidus'] = {'P_Pa': sol[:, 0].tolist(), 'S': sol[:, 1].tolist()}
    p, s, t = _ps_table(tables / 'temperature_melt.dat')
    rho = _ps_table(tables / 'density_melt.dat')[2]
    if abs(p[0] / 1e5 - 1) > 1e-12:
        raise ValueError(f'{tables}: the first melt pressure node is {p[0]} Pa, not 1 bar')
    ok = np.loadtxt(tables / 'valid_mask_melt.dat')[:, 0] == 1
    out['melt_at_1bar'] = {
        'S': s[ok].tolist(),
        'T_K': t[ok, 0].tolist(),
        'rho': rho[ok, 0].tolist(),
    }
    if raw_liquid is not None:
        r = np.loadtxt(raw_liquid, usecols=(0, 1, 2, 4))
        r = r[r[:, 0] == 1e5]
        r = r[np.argsort(r[:, 1])]
        out['raw_liquid_at_1bar'] = {
            'file': raw_liquid.name,
            'md5': _md5(raw_liquid),
            'T_K': r[:, 1].tolist(),
            'rho': r[:, 2].tolist(),
            's': r[:, 3].tolist(),
        }
    return out


def _cell_pressure(r0: dict) -> list:
    """Staggered pressure [Pa] of the run's mesh: abe_solid.toml with its nodes and end cells."""
    import aragog
    from aragog.mesh import Mesh
    from aragog.parser import Parameters

    p = Parameters.from_file(Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml')
    p.mesh.number_of_nodes = r0['nodes']
    p.mesh.surface_cell_thickness = r0.get('surface_cell', 0.0)
    p.mesh.cmb_cell_thickness = r0.get('cmb_cell', 0.0)
    return np.ravel(Mesh(p).staggered_pressure).tolist()


def run_values(folder: Path, phi_rows: Path | None = None, profiles: bool = False) -> dict:
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
    t, s_top, s_sol, s_bot, s_end = [], [], None, [], []
    for p in parts:
        z = np.load(p)
        s_sol = z['S_sol'][-1] if s_sol is None else s_sol
        s_bot.append(float(z['y'][0, -1]))
        s_end.append(z['y'][: z['S_sol'].size, -1].tolist())
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
        'S_bottom_cell': s_bot,
        'first_frozen': _first_frozen(t, np.concatenate(s_top)[idx], s_sol),
        'top_cell_freezes': _freezes(t, np.concatenate(s_top)[idx] - s_sol, rows),
    }
    sol = [np.load(p)['S_sol'] for p in parts]
    if any(not np.array_equal(x, sol[0]) for x in sol):
        raise ValueError(f'{folder}: the solidus entropy differs between parts')
    out['S_sol'] = sol[0].tolist()
    out['P_cell_Pa'] = _cell_pressure(r0)
    if profiles:
        out['S_end'] = s_end
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
            if tp[m][0] > 0.525 * ts:
                raise ValueError(
                    f'{folder.name}: window [{ts / 2:g}, {ts:g}] yr starts at {tp[m][0]:g} yr'
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
    ap.add_argument('--tables', type=Path)
    ap.add_argument('--cmb-pressure', type=float)
    ap.add_argument('--raw-liquid', type=Path)
    ap.add_argument('--profiles', nargs='*', default=[])
    a = ap.parse_args(argv)
    if (a.tables is None) != (a.cmb_pressure is None):
        ap.error('--tables and --cmb-pressure go together')
    res = {
        'header': {
            'generator': 'tools/verification/run_mesh_refinement_study.py',
            'note': a.note,
        }
    }
    res.update({r: run_values(a.root / r, a.phi_rows, r in a.profiles) for r in a.runs})
    trees = {v['tree'] for k, v in res.items() if k != 'header'}
    res['header']['trees'] = sorted(trees)
    if a.tables is not None:
        res['header'].update(table_values(a.tables, a.cmb_pressure, a.raw_liquid))
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
