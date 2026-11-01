"""Build ``tests/reference/surface_lid_c3_molten.npz``: the surface flux of the V2 shell started
molten and convecting, without rheology (``molten_column``), at ``MOLTEN_TIMES_YR`` on the
PALEOS low-temperature table named by ``ARAGOG_LOWT_EOS_DIR``: with the lid on 10 nodes, with the
lid and its hand-over root held at -1, and without the lid on 10 nodes; the stored reference
without the lid on 1000 nodes; the stored refinement bound, the same shell without the lid on
400 and 800 nodes with top cells of 0.25 and 0.125 m in radius coordinates; and the lid run on
the SPIDER test tables at ``MELT_TIMES_YR`` as the pin of CI.

The mesh runs take hours, so they are read from their row files (JSON with a list ``rows`` of
``t`` and ``F_s``):
``python tests/reference/build_surface_lid_c3_molten.py <reference.json> <n400.json> <n800.json>
[<last.json>]``; ``<last.json>``, a run of the same reference, gives its last row. Run from the
repository root with ``PYTHONPATH=src:.``.
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
)
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR

OUT = Path(__file__).resolve().parent / 'surface_lid_c3_molten.npz'


def _rows(path: str, n_nodes: int | None = None) -> tuple[np.ndarray, str, str, int]:
    """Surface flux [W/m^2] of a molten-start run without the lid at ``MOLTEN_TIMES_YR``, the
    md5 of its row file, the commit of its code and its node count (``n_nodes`` when given)."""
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    setup = (d['lid'], d['start'], d['rheo'], d['convection'])
    assert setup == (False, 1.1, 'none', True), setup
    assert n_nodes is None or d['n_nodes'] == n_nodes, d['n_nodes']
    np.testing.assert_allclose([r['t'] for r in d['rows']], MOLTEN_TIMES_YR, rtol=1e-12)
    F_s = np.array([r['F_s'] for r in d['rows']], dtype=float)
    return F_s, hashlib.md5(raw).hexdigest(), d['sha'], int(d['n_nodes'])


if __name__ == '__main__':
    from aragog.solver.entropy_solver import EntropySolver

    table = Path(os.environ['ARAGOG_LOWT_EOS_DIR'])
    md5 = hashlib.md5((table / 'temperature_solid.dat').read_bytes()).hexdigest()
    assert md5 == LOWT_EOS_MD5, md5
    eos = entropy_eos_copy(table)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    out = {}
    out['grid_ref'], out['grid_ref_md5'], out['grid_ref_sha'], out['n_ref'] = _rows(sys.argv[1])
    for key, n, path in zip(('refined_400', 'refined_800'), (400, 800), sys.argv[2:4]):
        out[key], out[f'{key}_md5'], out[f'{key}_sha'], _ = _rows(path, n)
    out['grid_ref_last_md5'] = out['grid_ref_last_sha'] = ''
    if len(sys.argv) > 4:  # the last row of the reference from another run of it
        last, out['grid_ref_last_md5'], out['grid_ref_last_sha'], n = _rows(sys.argv[4])
        assert n == out['n_ref'], n
        out['grid_ref'][-1] = last[-1]
    out['lid'], out['handover_yr'] = molten_column(10, True, MOLTEN_TIMES_YR, eos)
    out['coarse'], _ = molten_column(10, False, MOLTEN_TIMES_YR, eos)
    root = EntropySolver._lid_solidus_root
    EntropySolver._lid_solidus_root = lambda self, y: -1.0
    out['lid_kept'], _ = molten_column(10, True, MOLTEN_TIMES_YR, eos)
    EntropySolver._lid_solidus_root = root
    out['pin'], out['pin_handover_yr'] = molten_column(
        10, True, MELT_TIMES_YR, entropy_eos_copy(EOS_DIR)
    )
    for k, v in out.items():
        print(k, v, flush=True)
    meta = dict(times_yr=MOLTEN_TIMES_YR, pin_times_yr=MELT_TIMES_YR, md5=md5)
    np.savez(OUT, commit=head.stdout.strip(), table=table.name, **meta, **out)
    print(OUT)
