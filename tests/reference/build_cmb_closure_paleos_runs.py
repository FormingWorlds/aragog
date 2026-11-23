"""Build ``tests/validation/data/cmb_closure_paleos_runs.npz`` from the stored states of 2
coupled PROTEUS runs, one per CMB closure.

The runs couple aragog (energy_balance core, JAX right-hand side) to Zalmoxis with PALEOS:MgSiO3
mantle and PALEOS:iron core tables, 1 Earth mass, from ``input/all_options.toml``:
``cp_over_t`` with the ``c_p/T`` closure, ``slope`` with ``(dS/dT)_P`` of the blend. The file holds the run's mantle
tables cropped to the CMB pressure window and, per helpfile row, the CMB node entropy at the
row start and end (``S_stag0 - dr_cmb/2 dSdr_cmb`` from the interior snapshots), the CMB
pressure, the CMB heat of the row and the core heat change ``c_core M_core T_hat dT_node``.

Usage: ``python tests/reference/build_cmb_closure_paleos_runs.py <c_p/T run> <its aragog commit>
<slope run> <its aragog commit> <PROTEUS commit>``
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
from scipy.constants import Julian_year

TABLES = ('temperature', 'heat_capacity')  # the blend T and its slope read only these
OUT = Path(__file__).resolve().parents[1] / 'validation' / 'data'
P_WINDOW = (104.0e9, 131.0e9)  # [Pa] around the run's CMB pressures, 106.6 to 128.1 GPa
S_MARGIN = 500.0  # [J/kg/K] below the solidus and above the liquidus


def _table(path):
    """P grid, S grid and values ``(n_P, n_S)`` of a SPIDER P-S table, in SI units."""
    head = path.read_text().splitlines()[:5]
    n_p, n_s = (int(x) for x in head[0].split()[2:4])
    scales = [float(x) for x in head[4].lstrip('#').split()]
    data = np.loadtxt(path, skiprows=5)
    P = np.unique(data[:, 0] * scales[0])
    S = np.unique(data[:, 1] * scales[1])
    return P, S, (data[:, 2] * scales[2]).reshape(n_s, n_p).T


def _boundary(path):
    """P and S of a SPIDER phase-boundary file, in SI units."""
    head = path.read_text().splitlines()[:5]
    scales = [float(x) for x in head[4].lstrip('#').split()]
    data = np.loadtxt(path, skiprows=5)
    return data[:, 0] * scales[0], data[:, 1] * scales[1]


def _tfac(run):
    """``core_tfac_avg`` of the run's coupler configuration."""
    config = tomllib.loads((run / 'init_coupler.toml').read_text())
    return float(config['interior_energetics']['core_tfac_avg'])


def _run_label(run, commit, closure):
    """The aragog commit of the run, its closure and its time span from the helpfile."""
    t = np.loadtxt(run / 'runtime_helpfile.csv', skiprows=1, usecols=0)
    return f'aragog {commit} ({closure} closure), {t[0] / 1e3:.3g} to {t[-1] / 1e3:.3g} kyr'


def _rows(run):
    """Per helpfile row: time, CMB pressure, node entropy at the start and end, q and dE."""
    import netCDF4
    import pandas as pd

    h = pd.read_csv(run / 'runtime_helpfile.csv', sep=r'\s+')
    stems = {
        float(p.name.split('_')[0].replace('p', '.')): p
        for p in (run / 'data').glob('*_int.nc')
    }
    times = np.array(sorted(stems))

    def node(t):
        near = float(times[np.argmin(np.abs(times - t))])
        if abs(near - t) > 2.0e-3:  # no snapshot of this row
            return np.nan
        with netCDF4.Dataset(stems[near]) as d:
            v = {
                k: np.ravel(d[k][:]).astype(float)
                for k in ('radius_b', 'radius_s', 'entropy_s', 'dSdr_cmb_state')
            }
        rb = np.sort(v['radius_b']) * 1.0e3
        return (
            v['entropy_s'][np.argmin(v['radius_s'])]
            - 0.5 * (rb[1] - rb[0]) * v['dSdr_cmb_state'][0]
        )

    t, S_node = h.Time.to_numpy(), np.array([node(x) for x in h.Time])
    c = h.core_heatcap.to_numpy() * h.M_core.to_numpy() * _tfac(run)
    structure = (np.diff(h.M_core.to_numpy()) != 0) | (np.diff(h.core_heatcap.to_numpy()) != 0)
    return {
        't_end': t[1:],
        'dt_s': np.diff(t) * Julian_year,
        'P_cmb': h.P_cmb.to_numpy()[1:],
        'S_start': S_node[:-1],
        'S_end': S_node[1:],
        'q': h.step_dE_F_cmb_J.to_numpy()[1:],
        'dE': c[1:] * np.diff(h.T_cmb_node.to_numpy()),
        'T_node_start': h.T_cmb_node.to_numpy()[:-1],
        'T_node_end': h.T_cmb_node.to_numpy()[1:],
        'kept': ~(structure | np.r_[False, structure[:-1]])
        & np.isfinite(S_node[:-1] + S_node[1:]),
    }


def main(cp_over_t, cp_commit, slope, slope_commit, proteus_commit):
    eos = cp_over_t / 'data' / 'spider_eos'
    names = [f'{n}_{ph}.dat' for n in TABLES for ph in ('solid', 'melt')]
    for name in (*names, 'solidus_P-S.dat', 'liquidus_P-S.dat'):  # both runs read one table set
        assert (eos / name).read_bytes() == (
            slope / 'data' / 'spider_eos' / name
        ).read_bytes(), name
    out, sha = {}, {}
    P_sol, S_sol = _boundary(eos / 'solidus_P-S.dat')
    P_liq, S_liq = _boundary(eos / 'liquidus_P-S.dat')
    in_p = (P_sol >= P_WINDOW[0]) & (P_sol <= P_WINDOW[1])
    s_lo, s_hi = (
        S_sol[in_p].min() - S_MARGIN,
        S_liq[(P_liq >= P_WINDOW[0]) & (P_liq <= P_WINDOW[1])].max() + S_MARGIN,
    )
    for name in TABLES:
        for phase in ('solid', 'melt'):
            path = eos / f'{name}_{phase}.dat'
            P, S, v = _table(path)
            keep = (P >= P_WINDOW[0]) & (P <= P_WINDOW[1])
            ks = (S >= s_lo) & (S <= s_hi)
            out[f'P_{name}_{phase}'], out[f'S_{name}_{phase}'] = P[keep], S[ks]
            out[f'{name}_{phase}'] = v[keep][:, ks]
            sha[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('solidus', 'liquidus'):
        path = eos / f'{name}_P-S.dat'
        P, S = _boundary(path)
        keep = (P >= P_WINDOW[0]) & (P <= P_WINDOW[1])
        out[f'P_{name}'], out[f'S_{name}'] = P[keep], S[keep]
        sha[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for label, run in (('cp_over_t', cp_over_t), ('slope', slope)):
        for key, value in _rows(run).items():
            out[f'{label}_{key}'] = value
        sha[f'{label}/runtime_helpfile.csv'] = hashlib.sha256(
            (run / 'runtime_helpfile.csv').read_bytes()
        ).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'cmb_closure_paleos_runs.npz', **out)
    provenance = {
        'runs': {
            'cp_over_t': _run_label(cp_over_t, cp_commit, 'c_p/T'),
            'slope': _run_label(slope, slope_commit, '(dS/dT)_P'),
        },
        'config': 'PROTEUS input/all_options.toml, 1 M_earth, PALEOS:MgSiO3 mantle, PALEOS:iron core',
        'proteus_commit': proteus_commit,
        'p_window_Pa': list(P_WINDOW),
        's_margin': S_MARGIN,
        'tables': list(TABLES),
        'sha256': sha,
        'builder': 'tests/reference/build_cmb_closure_paleos_runs.py',
    }
    body = ',\n'.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in provenance.items())
    (OUT / 'cmb_closure_paleos_runs_provenance.json').write_text('{\n' + body + '\n}\n')
    print(
        {
            k: v.shape
            for k, v in out.items()
            if k.startswith(('temperature_melt', 'cp_over_t_t', 'slope_t'))
        }
    )


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5])
