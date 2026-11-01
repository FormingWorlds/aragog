"""The stagnant-lid benchmark rows of the SSC harness run with the sub-grid lid."""

from __future__ import annotations

import math

import pytest

from tools.verification.run_surface_lid_v5_rows import run_row

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]


def test_deschamps_vilella_row_runs_with_the_lid():
    """A Deschamps and Vilella (2021) row runs to steady state with the lid, which hands over
    to the mesh."""
    r = run_row('D', 'f60_H4', True, 'lid')
    assert math.isfinite(r['result']['Tm'])
    assert r['handover_t'] is not None and r['handover_t'] > 0.0


def test_deschamps_vilella_row_with_the_lid_at_the_growth_stop_completes():
    """In this row the lid sits at the growth stop for about 7e6 yr before it hands over; the
    integrator leaves it a little above the stop, inside the slack of the lid step check."""
    r = run_row('D', 'f70_15p8_1e7_H3', True, 'lid')
    assert math.isfinite(r['result']['Tm'])
    assert r['handover_t'] is not None and r['handover_t'] > 1.0e6
