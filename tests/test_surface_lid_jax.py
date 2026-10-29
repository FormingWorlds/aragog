"""JAX functions of the sub-grid lid against their NumPy and SciPy counterparts."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]


def _arguments():
    x = np.concatenate([np.geomspace(1e-8, 1.0, 2000), np.linspace(1.0, 75.0, 8000)])
    return np.concatenate([-x[::-1], [0.0], x])


def test_dawson_matches_scipy():
    """The sampling sum matches scipy.special.dawsn to relative 1e-12 for |x| <= 75, which
    covers the lid range sqrt(-LID_SHAPE_MIN) = 70.7, and is exactly 0 at 0."""
    from scipy.special import dawsn

    from aragog import surface_lid as sl
    from aragog.jax.surface_lid import dawson

    assert np.sqrt(-sl.LID_SHAPE_MIN) < 75.0
    x = _arguments()
    got, ref = np.asarray(jax.jit(dawson)(x)), dawsn(x)
    nz = ref != 0.0
    np.testing.assert_allclose(got[nz], ref[nz], rtol=1e-12, atol=0)
    assert got[x == 0.0] == 0.0


def test_dawson_derivative():
    """The JAX derivative equals F'(x) = 1 - 2 x F(x), including F'(0) = 1."""
    from scipy.special import dawsn

    from aragog.jax.surface_lid import dawson

    x = _arguments()
    got = np.asarray(jax.vmap(jax.grad(dawson))(x))
    np.testing.assert_allclose(got, 1.0 - 2.0 * x * dawsn(x), rtol=0, atol=1e-13)
    assert float(jax.grad(dawson)(0.0)) == pytest.approx(1.0, rel=1e-14)
