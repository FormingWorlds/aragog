"""Build ``tests/reference/surface_lid_v2_grid.npz``: the surface flux of the V2 shell with the
lid on 10 nodes and without the lid on 10, 500, 1000 and 4000 nodes, at ``TIMES_YR``, on the
PALEOS low-temperature table named by ``ARAGOG_LOWT_EOS_DIR``. About 25 min on one core.

Run from the repository root:
``ARAGOG_LOWT_EOS_DIR=<table> PYTHONPATH=src:. python tests/reference/build_surface_lid_v2_grid.py``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from tests.conftest import entropy_eos_copy
from tests.reference.surface_lid_v2_column import LOWT_EOS_MD5, TIMES_YR, surface_flux

GRIDS = (500, 1000, 4000)
OUT = Path(__file__).resolve().parent / 'surface_lid_v2_grid.npz'

if __name__ == '__main__':
    table = Path(os.environ['ARAGOG_LOWT_EOS_DIR'])
    md5 = hashlib.md5((table / 'temperature_solid.dat').read_bytes()).hexdigest()
    assert md5 == LOWT_EOS_MD5, md5
    eos = entropy_eos_copy(table)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    out = {'lid': surface_flux(10, True, TIMES_YR, eos)}
    out['coarse'] = surface_flux(10, False, TIMES_YR, eos)
    print('lid', out['lid'], flush=True)
    for n in GRIDS:
        out[f'grid_{n}'] = surface_flux(n, False, TIMES_YR, eos)
        print(n, out[f'grid_{n}'], flush=True)
    meta = dict(times_yr=TIMES_YR, commit=head.stdout.strip(), table=table.name, md5=md5)
    np.savez(OUT, **meta, **out)
    print(OUT)
