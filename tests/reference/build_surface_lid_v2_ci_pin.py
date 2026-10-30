"""Build ``tests/reference/surface_lid_v2_ci_pin.npz``: the surface flux of the V2 shell with
the lid on 10 nodes on the SPIDER test tables (``ARAGOG_TEST_EOS_DIR``), at ``TIMES_YR``, and
the surface flux and lid thickness of its melt variant at ``MELT_TIMES_YR``. The lid keeps
every node above the 300 K floor of those tables, so CI reruns both cases.

Run from the repository root: ``PYTHONPATH=src:. python tests/reference/build_surface_lid_v2_ci_pin.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from tests.conftest import entropy_eos_copy
from tests.reference.surface_lid_v2_column import (
    MELT_TIMES_YR,
    TIMES_YR,
    melt_column,
    surface_flux,
)

OUT = Path(__file__).resolve().parent / 'surface_lid_v2_ci_pin.npz'

if __name__ == '__main__':
    eos = entropy_eos_copy(os.environ['ARAGOG_TEST_EOS_DIR'])
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    flux = surface_flux(10, True, TIMES_YR, eos)
    melt_flux, melt_crust = melt_column(10, True, MELT_TIMES_YR, eos)
    np.savez(
        OUT,
        times_yr=TIMES_YR,
        lid=flux,
        melt_times_yr=MELT_TIMES_YR,
        melt_flux=melt_flux,
        melt_crust=melt_crust,
        commit=head.stdout.strip(),
    )
    print(OUT, flux, melt_flux, melt_crust)
