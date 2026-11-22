"""Build ``tests/reference/subgrid_lid_v0_reference.npz``: the off-path probe on the main commit
the sub-grid lid branches from.

Run from the repository root with that commit's ``src`` first on the path, for example
``PYTHONPATH=<checkout of the base>/src:. python <this file>``. With ``--core-module`` it
replaces only the core_module entries, from ``CORE_MODULE_COMMIT`` (the boundary-layer CMB flux
with the eq. 16 critical Rayleigh number of the interior viscosity), and with ``--energy-balance``
only the energy_balance entries, from ``ENERGY_BALANCE_COMMIT`` (the CMB closure with
``(dS/dT)_P`` of the node blend); each stores its commit in ``mode_<mode>/commit``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import aragog
from tests.reference.subgrid_lid_v0_probe import CASES, MODES, case_name, mode_probe, probe

BASE_COMMIT = '65e0d18ef13606515da9fa6ecba917e0620208d1'
CORE_MODULE_COMMIT = '03962559665104b50d7893c1a2c1cefec38d43aa'
ENERGY_BALANCE_COMMIT = '444f858b16053931a9ad99aa0a7e7e15dc9c7d85'
MODE_COMMITS = {
    '--core-module': ('core_module', CORE_MODULE_COMMIT),
    '--energy-balance': ('energy_balance', ENERGY_BALANCE_COMMIT),
}
OUT = Path(__file__).resolve().parent / 'subgrid_lid_v0_reference.npz'

if __name__ == '__main__':
    src = Path(aragog.__file__).resolve().parent.parent
    head = subprocess.run(
        ['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True
    )
    eos_dir = os.environ['ARAGOG_TEST_EOS_DIR']
    flag = next((f for f in MODE_COMMITS if f in sys.argv[1:]), None)
    if flag:
        mode, commit = MODE_COMMITS[flag]
        assert head.stdout.strip() == commit, f'aragog from {src} is at {head.stdout}'
        with np.load(OUT) as old:
            arrays = {k: old[k] for k in old.files if not k.startswith(f'mode_{mode}/')}
        for key, value in mode_probe(mode, eos_dir).items():
            arrays[f'mode_{mode}/{key}'] = value
        arrays[f'mode_{mode}/commit'] = np.array(commit)
        np.savez_compressed(OUT, **arrays)
        print(OUT, len(arrays), 'arrays')
        sys.exit(0)
    assert head.stdout.strip() == BASE_COMMIT, f'aragog from {src} is at {head.stdout.strip()}'
    arrays = {}
    for rheology, jax in CASES:
        for key, value in probe(rheology, jax, os.environ['ARAGOG_TEST_EOS_DIR']).items():
            arrays[f'{case_name(rheology, jax)}/{key}'] = value
    for mode in MODES:
        for key, value in mode_probe(mode, os.environ['ARAGOG_TEST_EOS_DIR']).items():
            arrays[f'mode_{mode}/{key}'] = value
    np.savez_compressed(OUT, base_commit=np.array(BASE_COMMIT), **arrays)
    print(OUT, len(arrays), 'arrays')
