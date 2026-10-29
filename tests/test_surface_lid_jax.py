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


def _shapes():
    from aragog import surface_lid as sl

    lo, hi = sl.LID_SHAPE_MIN, sl.LID_SHAPE_MAX
    s = np.concatenate([-np.geomspace(-lo, 1e-8, 400), [0.0], np.geomspace(1e-8, hi, 400)])
    return np.concatenate([s, [-1e-6, 1e-6, -1.0000001e-6, 1.0000001e-6]])


def test_shape_matches_numpy():
    """The JAX shape function equals the NumPy one over the full shape range of the lid, both
    sides of the series switch included: the slopes to relative 1e-12, the mean to absolute
    1e-14 (for s << 0 both forms lose digits in m = 1 + expm1(s) / (2 nu F(nu)), m >= 1e-4)."""
    from aragog import surface_lid as sl
    from aragog.jax.surface_lid import shape

    s = _shapes()
    got = np.array(jax.jit(jax.vmap(shape))(s)).T
    ref = np.array([sl.shape(v) for v in s])
    np.testing.assert_allclose(
        got[:, :2], ref[:, :2], rtol=1e-12, atol=1e-300
    )  # XLA flushes subnormals
    np.testing.assert_allclose(got[:, 2], ref[:, 2], rtol=0, atol=1e-14)


def test_shape_gradient_is_finite_and_matches_differences():
    """The derivative of each output in s is finite everywhere and equals a central
    difference of the NumPy shape function away from the series switch."""
    from aragog import surface_lid as sl
    from aragog.jax.surface_lid import shape

    s = _shapes()
    jac = np.array([jax.vmap(jax.grad(lambda v, i=i: shape(v)[i]))(s) for i in range(3)]).T
    assert np.all(np.isfinite(jac))
    far = (np.abs(s) > 1e-3) & (s > sl.LID_SHAPE_MIN * 0.99) & (s < sl.LID_SHAPE_MAX * 0.99)
    h = 1e-6 * np.maximum(np.abs(s[far]), 1.0)
    fd = np.array(
        [
            (np.array(sl.shape(v + d)) - np.array(sl.shape(v - d))) / (2 * d)
            for v, d in zip(s[far], h)
        ]
    )
    np.testing.assert_allclose(jac[far], fd, rtol=1e-6, atol=1e-12)
