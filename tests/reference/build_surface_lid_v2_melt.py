"""Build ``tests/reference/surface_lid_v2_melt.npz``: the surface flux and crust thickness of
the V2 shell started in the mush (``melt_column``) with the lid on 10 nodes, and without the
lid on 10, 1000 and 4000 nodes, plus a 1000-node member with the start entropy changed by
1e-12 (the round-off floor), at ``MELT_TIMES_YR``, on the PALEOS low-temperature table named
by ``ARAGOG_LOWT_EOS_DIR``. About 17 min on one core.

Run from the repository root:
``ARAGOG_LOWT_EOS_DIR=<table> PYTHONPATH=src:. python tests/reference/build_surface_lid_v2_melt.py``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from tests.conftest import entropy_eos_copy
from tests.reference.surface_lid_v2_column import LOWT_EOS_MD5, MELT_TIMES_YR, melt_column

OUT = Path(__file__).resolve().parent / 'surface_lid_v2_melt.npz'

if __name__ == '__main__':
    table = Path(os.environ['ARAGOG_LOWT_EOS_DIR'])
    md5 = hashlib.md5((table / 'temperature_solid.dat').read_bytes()).hexdigest()
    assert md5 == LOWT_EOS_MD5, md5
    eos = entropy_eos_copy(table)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    out = {}
    for name, n, lid, perturb in (
        ('lid', 10, True, 0.0),
        ('coarse', 10, False, 0.0),
        ('grid_1000', 1000, False, 0.0),
        ('grid_1000_p', 1000, False, 1e-12),
        ('grid_4000', 4000, False, 0.0),
    ):
        out[f'{name}_flux'], out[f'{name}_crust'] = melt_column(
            n, lid, MELT_TIMES_YR, eos, perturb
        )
        print(name, out[f'{name}_flux'], out[f'{name}_crust'], flush=True)
    meta = dict(times_yr=MELT_TIMES_YR, commit=head.stdout.strip(), table=table.name, md5=md5)
    np.savez(OUT, **meta, **out)
    print(OUT)
