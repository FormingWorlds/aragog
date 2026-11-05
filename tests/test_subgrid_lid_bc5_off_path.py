"""With the sub-grid lid off, the outer boundary 5 solve path equals main.

The reference (``tests/reference/subgrid_lid_bc5_off_reference.npz``) is the probe of
``tests/reference/subgrid_lid_bc5_off_probe.py`` on main: one right-hand side and a 5-yr solve of
a constant-property column with the surface at 300 K, without and with the solid rheology. The
right-hand side must match to rtol 1e-10 and the solve to rtol 1e-9, above the round-off between
the macOS and Linux builds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.reference.subgrid_lid_bc5_off_probe import CASES, case_name, probe

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]

REFERENCE = Path(__file__).resolve().parent / 'reference' / 'subgrid_lid_bc5_off_reference.npz'


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('rheology', CASES, ids=[case_name(c) for c in CASES])
def test_lid_off_path_under_a_fixed_surface_temperature_equals_main(rheology):
    """Off, the lid leaves the outer boundary 5 right-hand side and a short solve as on main,
    without and with the solid rheology."""
    got, name = probe(rheology), case_name(rheology)
    with np.load(REFERENCE) as ref:
        np.testing.assert_allclose(got['rhs'], ref[f'{name}/rhs'], rtol=1e-10, atol=0)
        assert int(got['status'][0]) == int(ref[f'{name}/status'][0]) == 0
        assert got['t'][-1] == pytest.approx(float(ref[f'{name}/t'][-1]), rel=1e-12)
        np.testing.assert_allclose(got['y_end'], ref[f'{name}/y_end'], rtol=1e-9, atol=0)


@settings(max_examples=25, deadline=None)
@given(top=st.floats(2700.0, 3300.0), drop=st.floats(-200.0, 200.0))
def test_lid_off_path_under_a_fixed_surface_temperature_calls_no_lid_code(top, drop):
    """Off, the outer boundary 5 right-hand side calls no function of the lid module."""
    import aragog.surface_lid as sl
    from tests.test_entropy_bc5_surface_temperature import _solver

    s = _solver()
    r = np.asarray(s._r_stag_flat)
    S = top + drop * (r[-1] - r) / (r[-1] - r[0])
    ref = np.asarray(s.dSdt(0.0, S), dtype=float).copy()
    names = [n for n in dir(sl) if callable(getattr(sl, n)) and not n.startswith('__')]
    saved = {n: getattr(sl, n) for n in names}

    def refuse(*a, **k):
        raise AssertionError('lid code called with the lid off')

    try:
        for n in names:
            setattr(sl, n, refuse)
        got = np.asarray(s.dSdt(0.0, S), dtype=float)
    finally:
        for n, f in saved.items():
            setattr(sl, n, f)
    np.testing.assert_array_equal(got, ref)
