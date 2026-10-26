"""NumPy-JAX parity of the BC 5 right-hand side in the unit tier, so that PR CI checks the JAX
wiring of the prescribed surface temperature on the test tables."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import needs_eos

pytestmark = [pytest.mark.unit, needs_eos, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
def test_the_jax_rhs_matches_numpy_under_bc5_on_a_graded_profile():
    """quasi_steady, nearest_boundary, a mushy surface node at 1500 K and an entropy falling
    from 2900 to 2600 J/kg/K: the JAX right-hand side equals the NumPy one in every cell, and
    the top cell differs from the prescribed-flux mode with the temperature read as a flux."""
    pytest.importorskip('jax')
    pytest.importorskip('equinox')
    from tests.test_entropy_bc5_jax_parity_smoke import _bc5_solver, _jax_rhs, _jax_rhs_and_args

    s = _bc5_solver('quasi_steady')
    y = np.asarray(s._S0, dtype=float)
    f_np = np.asarray(s.dSdt(0.0, y)).ravel()
    np.testing.assert_allclose(_jax_rhs(s, y), f_np, rtol=1e-10)
    fn4, args4 = _jax_rhs_and_args(s, outer_bc_type=4)
    f4 = np.asarray(fn4(0.0, y, args4)).ravel()
    assert f4[-1] != pytest.approx(f_np[-1], rel=1e-3)
