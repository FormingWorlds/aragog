"""Test B5: NumPy vs JAX parity verification."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from tools.verify_task6.run_b5 import (
    verify_bc6_surface_flux,
    verify_cmb_law_flux,
    verify_mlt_slope_factor,
    verify_mesh_arrays,
)

pytestmark = [pytest.mark.physics_invariant]


def test_b5_bc6_surface_flux_parity():
    max_diff = verify_bc6_surface_flux(n_states=20, seed=42)
    assert max_diff < 1e-9


def test_b5_cmb_law_flux_parity():
    if 'ARAGOG_TEST_EOS_DIR' not in os.environ:
        pytest.skip('ARAGOG_TEST_EOS_DIR not set')
    max_diff = verify_cmb_law_flux(n_states=20, seed=42)
    assert max_diff < 1e-9


def test_b5_mlt_slope_factor_parity():
    max_diff = verify_mlt_slope_factor(n_states=20, seed=42)
    assert max_diff < 1e-9


def test_b5_mesh_arrays_parity():
    max_diff = verify_mesh_arrays(n_states=20, seed=42)
    assert max_diff < 1e-9
