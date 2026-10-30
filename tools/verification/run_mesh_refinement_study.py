"""Write the raw values of the mesh refinement study runs to one compact JSON file.

Each run folder holds ``za.json`` (one record per part) and ``za_part<k>.npz`` (the stored rows
of part k), as ``tools/verification/run_mesh_refinement_case.py`` writes them. Per run the file
keeps the mesh, the config hash, the staggered solidus entropy, the staggered pressure of the
mesh rebuilt from ``abe_solid.toml`` and, per part, the end time, status, global melt fraction,
CMB temperature, time-mean surface heat flux, the energy integrals ``F_int`` (through the
surface, negative for heat loss) and ``F_cmb`` (through the CMB) in J, the CVODE step count, the
bottom-cell entropy, the number of times the top cell falls to or below the solidus and, for a
core-module run, the core temperature. It also keeps the first frozen interval of the top cell
(linear between stored rows); ``--phi-rows DIR`` adds the time mean of the global melt fraction
from the first stored row at or after t/2 (at most 0.525 t) to t, for each t = 1e3, 1e4, 1e5 yr
that the run reaches, and ``--profiles RUN...`` the staggered entropy at every part end. All
other derived numbers are computed by ``tests/test_mesh_refinement_study.py``.

With ``--tables DIR --cmb-pressure PA`` the header keeps, each table file named with its md5:
the solid temperature table around the CMB pressure, its entropy nodes and last valid entropy
per pressure node, the solidus, and the melt temperature and density at 1 bar; ``--raw-liquid
FILE`` adds the 1 bar rows of a PALEOS (P, T) liquid table.

Usage::

    python tools/verification/run_mesh_refinement_study.py <run root> <out.json> <run>...
        [--phi-rows DIR] [--note TEXT] [--tables DIR --cmb-pressure PA] [--raw-liquid FILE]
        [--profiles RUN...]
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
    """Raw table entries around the CMB pressure and at 1 bar, and the solidus."""
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
    """Staggered pressure [Pa] of the mesh of abe_solid.toml with the run's nodes and cells."""
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
    if len({json.dumps(r['table_md5'], sort_keys=True) for r in rows if 'table_md5' in r}) > 1:
        raise ValueError(f'{folder}: parts from more than one table set')
    t, s_top, s_bot, s_end = [], [], [], []
    zs = [np.load(p) for p in parts]
    s_sol = zs[0]['S_sol'][-1]
    for z in zs:
        s_bot.append(float(z['y'][0, -1]))
        s_end.append(z['y'][: z['S_sol'].size, -1].tolist())
        t.append(z['t'])
        s_top.append(z['y'][z['S_sol'].size - 1])
    t, idx = np.unique(np.concatenate(t), return_index=True)
    r0 = rows[0]
    out = {
        'commit': r0['sha'],
        'tree': r0.get('tree'),
        'config_hash': r0.get('config_hash'),
        'nodes': r0['nodes'],
        'surface_cell_m': r0.get('surface_cell', 0.0),
        'cmb_cell_m': r0.get('cmb_cell', 0.0),
        'h_top_m': r0['h_top'],
        'h_cmb_m': r0['h_cmb'],
        's_init': r0['s_init'],
        'core_bc': r0.get('core_bc'),
        'core_module_params': r0.get('core_module_params'),
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
    if r0.get('core_bc') == 'core_module':
        from aragog.solver.entropy_solver import EXTRA_STATE_SLOTS

        k = EXTRA_STATE_SLOTS['core_module'].index('T_core')
        out['T_core_K'] = [float(z['y_end'][z['S_sol'].size + k]) for z in zs]
    sol = [z['S_sol'] for z in zs]
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
        for ts in [x for x in (1e3, 1e4, 1e5) if x <= rows[-1]['t_yr']]:
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
    res['header']['commits'] = sorted({v['commit'] for k, v in res.items() if k != 'header'})
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
