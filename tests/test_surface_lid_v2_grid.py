"""V2 cross-check: the sub-grid lid on a coarse mesh against a resolved mesh without the lid,
for a solid shell and for the same shell started in the mush.

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

from tests.conftest import assert_legend_clear, entropy_eos_copy
from tests.reference.surface_lid_v2_column import (
    LOWT_EOS_MD5,
    MELT_TIMES_YR,
    TIMES_YR,
    melt_column,
    surface_flux,
)
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(900), needs_eos]

REF = Path(__file__).resolve().parent / 'reference'
TOL = 1e-2
MELT_TOL, FLOOR = 2e-2, 1e-5  # melt start: tolerance, round-off floor of the mesh run


def _stored(name, times=TIMES_YR):
    with np.load(REF / name) as d:
        np.testing.assert_array_equal(d['times_yr'], times)
        return {k: d[k] for k in d.files}


def _lowt_table():
    """The extended PALEOS table named by ``ARAGOG_LOWT_EOS_DIR``, checked by md5, or None."""
    table = os.environ.get('ARAGOG_LOWT_EOS_DIR')
    if table:
        md5 = hashlib.md5((Path(table) / 'temperature_solid.dat').read_bytes()).hexdigest()
        assert md5 == LOWT_EOS_MD5
    return table


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
    table = _lowt_table()
    if table:
        live = surface_flux(10, True, TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(live, lid, rtol=1e-6)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(1 - TOL, 1 + TOL, color='C0', alpha=0.15, label=f'{TOL:.0%} band')
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=0.8)
        bound = np.abs(grid - d['grid_1000']) / grid
        ax.errorbar(
            TIMES_YR,
            np.ones_like(grid),
            yerr=bound,
            fmt='none',
            color=plt.rcParams['text.color'],
            capsize=3,
            label='grid bound',
        )
        ax.plot(TIMES_YR, lid / grid, 'o', color='C1', ms=5, label='10 nodes with lid')
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks([300, 1000, 3000], ['300', '1000', '3000'])
        ax.set_ylim(0.97, 1.03)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel('lid / resolved flux')
        ax.legend(fontsize=8)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v2_resolved_mesh')


@pytest.mark.reference_pinned
def test_v2_lid_run_on_the_test_tables():
    """The V2 lid run on the SPIDER test tables equals its stored pin."""
    pin = _stored('surface_lid_v2_ci_pin.npz')['lid']
    eos = entropy_eos_copy(EOS_DIR)
    np.testing.assert_allclose(surface_flux(10, True, TIMES_YR, eos), pin, rtol=1e-6)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v2_melt_lid_matches_the_resolved_mesh(validation_plot):
    """Started in the mush, without rheology, a 10-node shell with the lid gives the surface
    flux and the crust thickness of a 4000-node shell without the lid within MELT_TOL from 100
    to 5000 yr: the lid freezes melt with its latent heat, so the case tests the Stefan jump.
    The 10-node shell without the lid misses both by more than 25 %; a round-off member of the
    1000-node run stays within FLOOR. With ``ARAGOG_LOWT_EOS_DIR`` set to the extended table,
    the lid run is repeated."""
    d = _stored('surface_lid_v2_melt.npz', MELT_TIMES_YR)
    assert str(d['md5']) == LOWT_EOS_MD5
    for q in ('flux', 'crust'):
        grid = d[f'grid_4000_{q}']
        np.testing.assert_allclose(d[f'grid_1000_p_{q}'], d[f'grid_1000_{q}'], rtol=FLOOR)
        np.testing.assert_allclose(d[f'lid_{q}'], grid, rtol=MELT_TOL)
        assert np.all(np.abs(d[f'coarse_{q}'] / grid - 1) > 0.25)
    table = _lowt_table()
    if table:
        live = melt_column(10, True, MELT_TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(live, (d['lid_flux'], d['lid_crust']), rtol=1e-6)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(
            1 - MELT_TOL, 1 + MELT_TOL, color='C0', alpha=0.15, label=f'{MELT_TOL:.0%} band'
        )
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=0.8)
        for q, mark, color in (('flux', 'o', 'C1'), ('crust', 's', 'C2')):
            ratio = d[f'lid_{q}'] / d[f'grid_4000_{q}']
            ax.plot(MELT_TIMES_YR, ratio, mark, color=color, ms=5, label=q)
        ax.set_xscale('log')
        ax.set_ylim(0.97, 1.03)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel('lid / resolved')
        ax.legend(fontsize=8)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v2_melt_resolved_mesh')


@pytest.mark.reference_pinned
def test_v2_melt_lid_run_on_the_test_tables():
    """The V2 melt lid run on the SPIDER test tables equals its stored pin."""
    pin = _stored('surface_lid_v2_ci_pin.npz')
    np.testing.assert_array_equal(pin['melt_times_yr'], MELT_TIMES_YR)
    live = melt_column(10, True, MELT_TIMES_YR, entropy_eos_copy(EOS_DIR))
    np.testing.assert_allclose(live, (pin['melt_flux'], pin['melt_crust']), rtol=1e-6)
