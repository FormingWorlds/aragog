"""Build ``tests/reference/surface_lid_c3_molten.npz``: the surface flux of the V2 shell started
molten and convecting, without rheology (``molten_column``), at ``MOLTEN_TIMES_YR`` on the
PALEOS low-temperature table named by ``ARAGOG_LOWT_EOS_DIR``: with the lid on 10 nodes, with the
lid and its hand-over root held at -1, and without the lid on 10 nodes; the stored runs without
the lid on 400 and 800 nodes with top cells of 0.25 and 0.125 m in radius coordinates (the
800-node run is the reference); the lid run on the SPIDER test tables at ``MELT_TIMES_YR`` as
the pin of CI; and, on both tables, the top-cell entropy of the 10-node run without the lid at
each step (``molten_top_cell``), whose crossing of the solidus checks the hand-over time.

The mesh runs take hours, so they are read from their row files (JSON with a list ``rows`` of
``t`` and ``F_s``): ``python tests/reference/build_surface_lid_c3_molten.py <n400.json>
<n800.json>``. Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.conftest import entropy_eos_copy
from tests.reference.surface_lid_v2_column import (
    LOWT_EOS_MD5,
    MELT_TIMES_YR,
    MOLTEN_TIMES_YR,
    molten_column,
    molten_top_cell,
)
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR

OUT = Path(__file__).resolve().parent / 'surface_lid_c3_molten.npz'


def _rows(path: str, n_nodes: int) -> tuple[np.ndarray, str, str]:
    """Surface flux [W/m^2] of a molten-start run without the lid on ``n_nodes`` at
    ``MOLTEN_TIMES_YR``, the md5 of its row file and the commit of its code."""
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    setup = (d['n_nodes'], d['lid'], d['start'], d['rheo'], d['convection'])
    assert setup == (n_nodes, False, 1.1, 'none', True), setup
    np.testing.assert_allclose([r['t'] for r in d['rows']], MOLTEN_TIMES_YR, rtol=1e-12)
    F_s = np.array([r['F_s'] for r in d['rows']], dtype=float)
    return F_s, hashlib.md5(raw).hexdigest(), d['sha']


if __name__ == '__main__':
    from aragog.solver.entropy_solver import EntropySolver

    assert len(sys.argv) == 3, 'usage: build_surface_lid_c3_molten.py <n400.json> <n800.json>'
    table = Path(os.environ['ARAGOG_LOWT_EOS_DIR'])
    md5 = hashlib.md5((table / 'temperature_solid.dat').read_bytes()).hexdigest()
    assert md5 == LOWT_EOS_MD5, md5
    eos = entropy_eos_copy(table)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    out = {}
    for key, n, path in zip(('refined_400', 'refined_800'), (400, 800), sys.argv[1:3]):
        out[key], out[f'{key}_md5'], out[f'{key}_sha'] = _rows(path, n)
    out['lid'], out['handover_yr'] = molten_column(10, True, MOLTEN_TIMES_YR, eos)
    out['coarse'], _ = molten_column(10, False, MOLTEN_TIMES_YR, eos)
    root = EntropySolver._lid_solidus_root
    EntropySolver._lid_solidus_root = lambda self, y: -1.0
    try:
        out['lid_kept'], _ = molten_column(10, True, MOLTEN_TIMES_YR, eos)
    finally:
        EntropySolver._lid_solidus_root = root
    out['pin'], out['pin_handover_yr'] = molten_column(
        10, True, MELT_TIMES_YR, entropy_eos_copy(EOS_DIR)
    )
    for key, tab in (('nolid', eos), ('pin_nolid', entropy_eos_copy(EOS_DIR))):
        out[f'{key}_t'], out[f'{key}_S_top'], out[f'{key}_S_sol'] = molten_top_cell(tab)
    for k, v in out.items():
        print(k, v, flush=True)
    meta = dict(times_yr=MOLTEN_TIMES_YR, pin_times_yr=MELT_TIMES_YR, md5=md5)
    np.savez(OUT, commit=head.stdout.strip(), table=table.name, **meta, **out)
    print(OUT)
