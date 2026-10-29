"""Build ``tests/reference/subgrid_lid_v0_reference.npz``: the off-path probe on the main commit
the sub-grid lid branches from.

Run from the repository root with that commit's ``src`` first on the path, for example
``PYTHONPATH=<checkout of the base>/src:. python tests/reference/build_subgrid_lid_v0_reference.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

import aragog
from tests.reference.subgrid_lid_v0_probe import CASES, case_name, probe

BASE_COMMIT = 'dd6c2b98b4bbaf7dcf8710ef6b7317635d01a0f7'
OUT = Path(__file__).resolve().parent / 'subgrid_lid_v0_reference.npz'

if __name__ == '__main__':
    src = Path(aragog.__file__).resolve().parent.parent
    head = subprocess.run(
        ['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True
    )
    assert head.stdout.strip() == BASE_COMMIT, f'aragog from {src} is at {head.stdout.strip()}'
    arrays = {}
    for rheology, jax in CASES:
        for key, value in probe(rheology, jax, os.environ['ARAGOG_TEST_EOS_DIR']).items():
            arrays[f'{case_name(rheology, jax)}/{key}'] = value
    np.savez_compressed(OUT, base_commit=np.array(BASE_COMMIT), **arrays)
    print(OUT, len(arrays), 'arrays')
