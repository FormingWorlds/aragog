"""Cut a Zalmoxis liquid iron side table to a pressure and temperature box for the core tests.

Usage::

    python tools/verification/make_core_liquid_cut.py <source .dat> <out dir> \\
        P_lo P_hi T_lo T_hi <name> [--keep-nan-edges]

The cut keeps 1 grid line beyond each box edge. Without ``--keep-nan-edges`` a non-finite
cell in the cut stops the script; with it, whole pressure rows of non-finite cells at the top
or bottom of the cut are kept (the reader drops them) and any other non-finite cell stops it.
Writes ``<name>.npz`` (compressed; keys ``P_Pa``, ``T_K``, the 7 property arrays
``[P index, T index]`` and ``header``) and ``<name>_provenance.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

COLS = ['P', 'T', 'rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad']


def main(argv):
    """Write the cut and its provenance; print the md5 and the axes of the cut."""
    keep_nan = '--keep-nan-edges' in argv
    argv = [a for a in argv if a != '--keep-nan-edges']
    src, out = Path(argv[1]), Path(argv[2])
    p_lo, p_hi, t_lo, t_hi = (float(x) for x in argv[3:7])
    name = argv[7]
    raw = src.read_bytes()
    lines = raw.decode().splitlines()
    header = [line for line in lines if line.startswith('#')]
    rows = [line.split() for line in lines if line and not line.startswith('#')]
    assert all(r[-1] in ('liquid', 'out-of-range') for r in rows) and all(
        len(r) == 10 for r in rows
    )
    data = np.array([[float(x) for x in r[:9]] for r in rows])
    p_ax, t_ax = np.unique(data[:, 0]), np.unique(data[:, 1])
    assert len(p_ax) * len(t_ax) == len(data), 'not a full rectangle'
    grid = data[np.lexsort((data[:, 1], data[:, 0]))].reshape(len(p_ax), len(t_ax), 9)
    assert np.array_equal(grid[:, 0, 0], p_ax) and np.array_equal(grid[0, :, 1], t_ax)
    ip = slice(
        max(np.searchsorted(p_ax, p_lo) - 1, 0),
        min(np.searchsorted(p_ax, p_hi, 'right') + 1, len(p_ax)),
    )
    it = slice(
        max(np.searchsorted(t_ax, t_lo) - 1, 0),
        min(np.searchsorted(t_ax, t_hi, 'right') + 1, len(t_ax)),
    )
    cut = grid[ip, it]
    bad = ~np.all(np.isfinite(cut), axis=(1, 2))
    if keep_nan:
        inner = np.flatnonzero(~bad)
        assert inner.size and not bad[inner[0] : inner[-1] + 1].any(), 'non-finite cell inside'
    else:
        assert not bad.any(), 'non-finite cell in the box'
    arrays = {'P_Pa': cut[:, 0, 0], 'T_K': cut[0, :, 1]}
    arrays.update({c: cut[:, :, k] for k, c in enumerate(COLS) if k >= 2})
    arrays['header'] = np.array('\n'.join(header))
    npz = out / f'{name}.npz'
    np.savez_compressed(npz, **arrays)
    md5 = lambda b: hashlib.md5(b).hexdigest()  # noqa: E731
    data_rows = '\n'.join(line for line in lines if not line.startswith('#')).encode()
    prov = {
        'source_file': src.name,
        'source_md5': md5(raw),
        'source_data_rows_md5': md5(data_rows),
        'source_grid': [len(p_ax), len(t_ax)],
        'box': {'P_Pa': [p_lo, p_hi], 'T_K': [t_lo, t_hi]},
        'cut_P_Pa': [float(arrays['P_Pa'][0]), float(arrays['P_Pa'][-1]), len(arrays['P_Pa'])],
        'cut_T_K': [float(arrays['T_K'][0]), float(arrays['T_K'][-1]), len(arrays['T_K'])],
        'nan_edge_rows': int(bad.sum()),
        'units': {
            'P_Pa': 'Pa',
            'T_K': 'K',
            'rho': 'kg/m^3',
            'u': 'J/kg',
            's': 'J/(kg K)',
            'cp': 'J/(kg K)',
            'cv': 'J/(kg K)',
            'alpha': '1/K',
            'nabla_ad': '-',
        },
        'layout': (
            'each property is a 2D array [P index, T index]; h = u + P/rho; the source header '
            'names bilinear interpolation in (log10 P, log10 T); aragog LiquidIronTable reads '
            'the cut with a bicubic spline in (log10 P, log10 T)'
        ),
        'header_lines': header,
        'npz_file': npz.name,
        'npz_md5': md5(npz.read_bytes()),
        'npz_bytes': npz.stat().st_size,
        'script': 'tools/verification/make_core_liquid_cut.py',
    }
    body = ',\n'.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in prov.items())
    (out / f'{name}_provenance.json').write_text('{\n' + body + '\n}\n')
    print(json.dumps({k: prov[k] for k in ('cut_P_Pa', 'cut_T_K', 'nan_edge_rows', 'npz_md5')}))


if __name__ == '__main__':
    main(sys.argv)
