"""The guards of gravitational separation: the gravity check and the output sign check.

The separation velocity takes gravity as a positive magnitude, so
``positive_gravity`` refuses a zero, negative or non-finite value at any node.
``_check_separation_sign`` refuses an accepted output whose separation heat flux
opposes ``rho_s - rho_l`` above ``JGRAV_SIGN_FLOOR``, or that holds a non-finite
flux or contrast.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.solver.entropy_solver import JGRAV_SIGN_FLOOR, _check_separation_sign
from aragog.utilities import positive_gravity

pytestmark = pytest.mark.unit


def _accepted(j, c, viscosity='melt', buoyancy='solid'):
    """Fail the test, not error it, when the sign check refuses a valid row."""
    try:
        _check_separation_sign(np.asarray(j), np.asarray(c), viscosity, buoyancy)
    except RuntimeError as exc:
        pytest.fail(f'valid row refused: {exc}')


@pytest.mark.parametrize('bad', [0.0, -0.0, -9.8, np.nan, np.inf, -np.inf])
def test_positive_gravity_refuses_a_bad_node(bad):
    """A zero, negative or non-finite value at node 2 raises with the node and value named."""
    g = np.array([10.0, 9.9, bad, 9.7])
    with pytest.raises(ValueError, match=r'gravity on the test mesh .* node 2 has') as exc:
        positive_gravity(g, 'test mesh')
    assert 'np.float64' not in str(exc.value)


def test_positive_gravity_passes_a_valid_profile_unchanged():
    """A finite positive profile comes back as the same float values."""
    g = [10.0, 9.5, 1.0e-300]
    out = positive_gravity(g, 'test mesh')
    assert out.dtype == float
    np.testing.assert_array_equal(out, np.array(g))


@pytest.mark.parametrize(
    ('jgrav', 'contrast'),
    [(np.nan, 5.0), (1.0e-3, np.nan), (np.inf, 5.0), (-1.0e-3, -np.inf)],
)
def test_sign_check_refuses_a_non_finite_row(jgrav, contrast):
    """A NaN or inf flux or contrast at nodes 1 and 3 raises with the first node and the
    count named."""
    j = np.array([1.0e-3, jgrav, 0.0, jgrav])
    c = np.array([5.0, contrast, 5.0, contrast])
    with pytest.raises(RuntimeError, match='not finite at node 1') as exc:
        _check_separation_sign(j, c, 'melt', 'solid')
    assert 'np.float64' not in str(exc.value) and ', 2 such nodes)' in str(exc.value)


def test_sign_check_names_node_values_and_modes():
    """Fluxes against the contrast at nodes 2 and 3 raise with the first node, its values,
    both modes and the count."""
    j = np.array([1.0e-3, -2.0e-3, -4.0e-3, 6.0e-3])
    c = np.array([5.0, -5.0, 250.0, -8.0])
    with pytest.raises(RuntimeError) as exc:
        _check_separation_sign(j, c, 'mixture', 'solid')
    msg = str(exc.value)
    assert 'against the density contrast at node 2' in msg
    assert '-4.000000e-03' in msg and '2.500000e+02' in msg
    assert "separation_viscosity 'mixture'" in msg and "separation_buoyancy 'solid'" in msg
    assert ', 2 such nodes)' in msg


def test_sign_check_accepts_a_denser_melt_and_zero_contrast():
    """A negative flux under a negative contrast, and a zero or nonzero flux at 0
    contrast, pass."""
    j = np.array([-3.0e-2, 4.0e-2, -0.0, 0.0, 1.0e-3, -1.0e-3])
    c = np.array([-120.0, 300.0, 5.0, 0.0, 0.0, 0.0])
    _accepted(j, c, buoyancy='mixture')


def test_sign_check_floor_bounds_the_refused_flux():
    """A flux against the contrast at 2x the floor raises; at the floor and at half of it
    the flux passes."""
    c = np.array([7.0])
    with pytest.raises(RuntimeError, match='against the density contrast'):
        _check_separation_sign(np.array([-2.0 * JGRAV_SIGN_FLOOR]), c, 'melt', 'solid')
    _accepted([-0.5 * JGRAV_SIGN_FLOOR], c)
    _accepted([-JGRAV_SIGN_FLOOR], c)
    assert JGRAV_SIGN_FLOOR == 1.0e-12
