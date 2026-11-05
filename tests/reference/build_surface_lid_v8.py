"""Build ``tests/reference/surface_lid_v8.npz`` from the series of ``run_surface_lid_series.py``:
the column of the hand-over case on 100, 200 and 400 nodes with the lid, and without the lid on
400 nodes with top cells of 500 and 1000 m, from 1e6 yr on, where the two references agree.

The runs take days on a cluster, so they are not repeated here. Run from the repository root:
``python tests/reference/build_surface_lid_v8.py n100.npz n200.npz n400.npz s500.npz s1000.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'surface_lid_v8.npz'
RUNS = ('n100', 'n200', 'n400', 's500', 's1000')
T_FIRST = 1.0e6

if __name__ == '__main__':
    assert len(sys.argv) == 6, 'usage: build_surface_lid_v8.py <5 series files>'
    out = {}
    for name, path in zip(RUNS, sys.argv[1:]):
        d = np.load(path)
        keep = d['t'] >= T_FIRST
        out[f'{name}_t'] = d['t'][keep]
        for key in ('F_top', 'z_lid', 'delta'):
            out[f'{name}_{key}'] = d[key][keep].astype(np.float32)
        out[f'{name}_handover_yr'], out[f'{name}_sha'] = d['handover_yr'], d['sha']
        out[f'{name}_part_yr'], out[f'{name}_w_active'] = d['part_yr'], d['w_active']
        out[f'{name}_f_int_max'] = d['f_int_max']
    np.savez_compressed(OUT, **out)
    print(OUT, OUT.stat().st_size)
