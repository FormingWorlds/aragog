"""V2 cross-check: the sub-grid lid on a coarse mesh against a resolved mesh without the lid.

The shell is ``tests/reference/surface_lid_v2_column.py``. Its resolved runs cool below the
300 K floor of the SPIDER test tables, so the cross-check data come from the PALEOS table
extended to 67 K (``tests/reference/build_surface_lid_v2_grid.py``). On the test tables the
lid run stays above the floor and is rerun against a stored pin
(``tests/reference/build_surface_lid_v2_ci_pin.py``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import entropy_eos_copy
from tests.reference.surface_lid_v2_column import LOWT_EOS_MD5, TIMES_YR, surface_flux
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(900), needs_eos]

REF = Path(__file__).resolve().parent / 'reference'
TOL = 1e-2


def _stored(name):
    with np.load(REF / name) as d:
        np.testing.assert_array_equal(d['times_yr'], TIMES_YR)
        return {k: d[k] for k in d.files}


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v2_lid_matches_the_resolved_mesh(validation_plot):
    """On the extended table, a 10-node shell with the lid loses heat through its surface as a
    4000-node shell without the lid, within TOL from 300 to 5000 yr; the 10-node shell without
    the lid misses by more than 10 %. With ``ARAGOG_LOWT_EOS_DIR`` set to that table, the lid
    run is repeated here."""
    d = _stored('surface_lid_v2_grid.npz')
    assert str(d['md5']) == LOWT_EOS_MD5
    lid, grid = d['lid'], d['grid_4000']
    np.testing.assert_allclose(lid, grid, rtol=TOL)
    assert np.all(np.abs(d['coarse'] / grid - 1) > 0.1)
    table = os.environ.get('ARAGOG_LOWT_EOS_DIR')
    if table:
        md5 = hashlib.md5((Path(table) / 'temperature_solid.dat').read_bytes()).hexdigest()
        assert md5 == LOWT_EOS_MD5
        live = surface_flux(10, True, TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(live, lid, rtol=1e-6)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(1 - TOL, 1 + TOL, color='C0', alpha=0.15, label=f'{TOL:.0%} band')
        ax.axhline(1.0, color='0.5', lw=0.8)
        bound = np.abs(grid - d['grid_1000']) / grid
        ax.errorbar(
            TIMES_YR,
            np.ones_like(grid),
            yerr=bound,
            fmt='none',
            color='0.4',
            capsize=3,
            label='grid bound',
        )
        ax.plot(TIMES_YR, lid / grid, 'o', color='C1', ms=5, label='10 nodes with lid')
        ax.set_xscale('log')
        ax.set_ylim(0.97, 1.03)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel('lid / resolved flux')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'surface_lid_v2_resolved_mesh')


@pytest.mark.reference_pinned
def test_v2_lid_run_on_the_test_tables():
    """The V2 lid run on the SPIDER test tables equals its stored pin."""
    pin = _stored('surface_lid_v2_ci_pin.npz')['lid']
    eos = entropy_eos_copy(EOS_DIR)
    np.testing.assert_allclose(surface_flux(10, True, TIMES_YR, eos), pin, rtol=1e-6)
