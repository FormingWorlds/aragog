"""Verify the JAX EOS bilinear-interp Jacobian within a single cell.

The bilinear interpolation in `aragog.jax.eos._bilinear_interp` uses
`jax.lax.stop_gradient` on the integer searchsorted indices. The
audit (A5) flagged this as potentially making the analytic Jacobian
"inexact at grid-boundary crossings". In fact the Jacobian is the
EXACT analytic Jacobian of the bilinear function within each cell
(only the integer indices are non-differentiable, and the bilinear
function is smooth in tp, ts within a cell).

This test pins that property: at a query point safely inside one
grid cell, the JAX-jacrev gradient must match the analytical
bilinear gradient to machine precision (rtol=1e-12). At a query
near a cell boundary, the gradient is naturally discontinuous (a
property of piecewise-bilinear functions, not a bug); we verify
the gradient matches the analytic value INSIDE one of the two
adjacent cells.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from aragog.jax.eos import _bilinear_interp


@pytest.fixture
def grid_2x2():
    """A simple 4x4 P-S grid with a known smooth values pattern."""
    P = jnp.array([1.0, 2.0, 3.0, 4.0])
    S = jnp.array([10.0, 20.0, 30.0, 40.0])
    # values[i, j] = 100 * i + j → pure-bilinear interpolant has
    # constant gradient w.r.t. P (= 100/dP) and w.r.t. S (= 1/dS)
    # within a cell, so the analytical answer is independent of
    # the query position inside the cell.
    vals = jnp.array(
        [[100.0 * i + j for j in range(4)] for i in range(4)]
    )
    return P, S, vals


@pytest.mark.unit
def test_jacobian_dP_within_cell(grid_2x2):
    """Within a cell, ∂f/∂P must match analytical bilinear gradient."""
    P, S, vals = grid_2x2

    # Query at (P=1.5, S=15.0): inside cell [1,2] x [10,20].
    # Within this cell, with values[i,j] = 100i + j:
    # f(P,S) = (100*0 + 0)*(1-tp)*(1-ts) + (100*1 + 0)*tp*(1-ts)
    #       + (100*0 + 1)*(1-tp)*ts + (100*1 + 1)*tp*ts
    #       = 100*tp + ts
    # where tp = (P-1)/1 = (P-1), ts = (S-10)/10
    # So ∂f/∂P = 100, ∂f/∂S = 0.1, both constant inside the cell.
    def f(P_q):
        S_q = jnp.array([15.0])
        return _bilinear_interp(P, S, vals, P_q, S_q)[0]

    df_dP = jax.grad(f)(jnp.array([1.5]))
    np.testing.assert_allclose(np.asarray(df_dP), [100.0], rtol=1e-12)


@pytest.mark.unit
def test_jacobian_dS_within_cell(grid_2x2):
    """Within a cell, ∂f/∂S must match analytical bilinear gradient."""
    P, S, vals = grid_2x2

    def f(S_q):
        P_q = jnp.array([1.5])
        return _bilinear_interp(P, S, vals, P_q, S_q)[0]

    df_dS = jax.grad(f)(jnp.array([15.0]))
    np.testing.assert_allclose(np.asarray(df_dS), [0.1], rtol=1e-12)


@pytest.mark.unit
def test_jacobian_at_different_cell_positions(grid_2x2):
    """Same gradient at multiple positions inside the same cell."""
    P, S, vals = grid_2x2

    def f(query):
        P_q, S_q = query[:1], query[1:]
        return _bilinear_interp(P, S, vals, P_q, S_q)[0]

    jac = jax.jacrev(f)
    # Three distinct points all safely inside cell [1,2] x [10,20]:
    for P_pos, S_pos in [(1.1, 11.0), (1.5, 15.0), (1.9, 19.0)]:
        g = jac(jnp.array([P_pos, S_pos]))
        np.testing.assert_allclose(
            np.asarray(g), [100.0, 0.1], rtol=1e-12,
            err_msg=f'(P={P_pos}, S={S_pos}) gradient shifted',
        )


@pytest.mark.unit
def test_jacobian_in_neighboring_cell(grid_2x2):
    """Cell [2,3] x [10,20] gives the same constant gradient.

    For values[i,j] = 100i + j and a uniform grid with dP = 1,
    every cell has ∂f/∂P = 100. The Jacobian is constant across
    the table for this synthetic linear field, so cell-boundary
    discontinuity doesn't appear here. (Discontinuity appears
    only when values[i+1,j] - values[i,j] differs from
    values[i+2,j] - values[i+1,j], which our linear field
    avoids by construction.)
    """
    P, S, vals = grid_2x2

    def f(query):
        P_q, S_q = query[:1], query[1:]
        return _bilinear_interp(P, S, vals, P_q, S_q)[0]

    jac = jax.jacrev(f)
    g_left = jac(jnp.array([1.5, 15.0]))   # cell [1,2]
    g_right = jac(jnp.array([2.5, 15.0]))  # cell [2,3]
    np.testing.assert_allclose(np.asarray(g_left), np.asarray(g_right), rtol=1e-12)


@pytest.mark.unit
def test_jacobian_discontinuity_at_cell_boundary_is_documented():
    """Verify that for a non-linear values field, ∂f/∂P jumps across
    a cell boundary. This documents the inherent property of
    piecewise-bilinear interpolation referenced in eos.py and is
    the "gradient discontinuity at cell boundaries" the audit (A5)
    flagged. CVODE absorbs this as a stiff-region step rejection.
    """
    P = jnp.array([1.0, 2.0, 3.0])
    S = jnp.array([10.0, 20.0])
    # Non-linear values: jumps slope between cells [1,2] and [2,3]
    # row P=1.0:  v = [0, 1]
    # row P=2.0:  v = [10, 11]   (slope ∂v/∂P = 10 in cell [1,2])
    # row P=3.0:  v = [200, 201] (slope ∂v/∂P = 190 in cell [2,3])
    vals = jnp.array([[0.0, 1.0], [10.0, 11.0], [200.0, 201.0]])

    def f(P_q):
        S_q = jnp.array([15.0])
        return _bilinear_interp(P, S, vals, P_q, S_q)[0]

    df_dP_left = jax.grad(f)(jnp.array([1.5]))   # cell [1,2]
    df_dP_right = jax.grad(f)(jnp.array([2.5]))  # cell [2,3]
    # In cell [1,2]: ∂f/∂P = 10, in cell [2,3]: ∂f/∂P = 190.
    np.testing.assert_allclose(np.asarray(df_dP_left), [10.0], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(df_dP_right), [190.0], rtol=1e-12)
    # Gradient jumps across the cell boundary at P=2.0 — by design.
    assert abs(float(df_dP_right[0]) - float(df_dP_left[0])) > 100.0
