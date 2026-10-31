"""Exception paths inside ``build_jax_rhs_and_jacobian``'s rhs_fn /
jacfn callbacks.

The factory's contract is that any uncaught exception inside the
JAX-traced RHS or Jacobian must be swallowed (logged + return 1) so
CVODE can fall back to its own finite-difference Jacobian rather
than aborting the integrator partway through a long solve.

The contract tests in test_cvode_jax_factory.py and the invocation
tests in test_cvode_jax_factory_invocation.py exercise the happy
paths but not these except branches (lines 221-223, 235-237 of
solver/cvode_jax.py). Force the failures by giving the in-place
output buffers the wrong shape so the trailing ``ydot[:] = ...`` /
``J[...] = ...`` write raises during numpy broadcast.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import EOS_DIR, entropy_eos_jax

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
eqx = pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


def _make_const_property_mesh(N: int = 8):
    """Minimal MeshArrays for cheap factory tests."""
    from aragog.jax.phase import MeshArrays

    R_INNER = 3.480e6
    R_OUTER = 6.371e6
    r_stag = np.linspace(R_INNER, R_OUTER, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_INNER
    r_basic[-1] = R_OUTER
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_INNER, R_OUTER - r_basic), 1.0)
    d_dr = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr[i, i - 1] = -1.0 / dr[i - 1]
        d_dr[i, i] = 1.0 / dr[i - 1]
    d_dr[0, :] = d_dr[1, :]
    d_dr[-1, :] = d_dr[-2, :]
    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5
    P_stag = np.linspace(135e9, 1e5, N)
    P_basic = q_mat @ P_stag
    return MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(area),
        volume=jnp.asarray(volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.asarray(P_stag),
        P_basic=jnp.asarray(P_basic),
        gravity=jnp.full(N + 1, 10.0),
    )


def _make_bc():
    from aragog.jax.solver import BoundaryParams

    return BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=2,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )


def _build_factory(radio_isotope_params=()):
    """Build a factory at quasi_steady mode against a real EOS, with the given radio 5-tuple."""
    if not EOS_DIR.exists():
        pytest.skip(f'EOS unavailable at {EOS_DIR}')

    from aragog.jax.nondim import NonDimScales
    from aragog.jax.phase import PhaseParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    eos_jax = entropy_eos_jax(EOS_DIR)
    params = PhaseParams()
    mesh = _make_const_property_mesh(N=8)
    bc = _make_bc()
    n_stag = int(mesh.P_stag.shape[0])
    state_scale = np.full(n_stag, 3.0e3)
    scales = NonDimScales(state_scale=state_scale, t_ref=1.0)
    heating = np.zeros(n_stag)
    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh,
        boundary_params=bc,
        heating_array=heating,
        scales=scales,
        core_bc_mode='quasi_steady',
        radio_isotope_params=radio_isotope_params,
    )
    return rhs_fn, jac_fn, info, n_stag


@needs_eos
def test_factory_rhs_fn_returns_one_on_buffer_shape_mismatch(caplog):
    """Passing an ``ydot_nd`` whose length disagrees with the JAX
    output forces the in-place write ``ydot_nd[:] = np.asarray(result)``
    to raise. The except branch must catch, log an ERROR-level
    message naming the failure, and return 1 so CVODE drops to its
    own FD-Jacobian.

    Discriminator: silence + return 0 would let CVODE proceed with
    a stale ydot, producing a corrupt trajectory. The non-zero
    return AND the error log are both required.
    """
    rhs_fn, _, info, n_stag = _build_factory()
    y_nd = np.full(n_stag, 3050.0 / 3.0e3)
    ydot_nd = np.zeros(n_stag - 3)  # wrong length: 5 vs n_stag=8

    with caplog.at_level(logging.ERROR, logger='fwl.aragog.solver.cvode_jax'):
        rc = rhs_fn(0.0, y_nd, ydot_nd)

    assert rc == 1, f'shape-mismatch rhs_fn must return 1; got {rc}'
    err_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any('JAX RHS failed' in m for m in err_msgs), (
        f'expected "JAX RHS failed" error log; got {err_msgs}'
    )


@needs_eos
def test_factory_jacfn_returns_one_on_buffer_shape_mismatch(caplog):
    """Same exception contract for the Jacobian callback. Wrong-shape
    J buffer forces the in-place write ``J[...] = np.asarray(jac)``
    to raise. The except branch must catch, log an ERROR, and return
    1 so CVODE drops to its FD Jacobian.
    """
    _, jacfn, info, n_stag = _build_factory()
    y_nd = np.full(n_stag, 3050.0 / 3.0e3)
    fy_nd = np.zeros(n_stag)
    J = np.zeros((n_stag - 1, n_stag - 1))  # wrong shape: (7,7) vs (8,8)

    with caplog.at_level(logging.ERROR, logger='fwl.aragog.solver.cvode_jax'):
        rc = jacfn(0.0, y_nd, fy_nd, J)

    assert rc == 1, f'shape-mismatch jacfn must return 1; got {rc}'
    err_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any('JAX Jacobian failed' in m for m in err_msgs), (
        f'expected "JAX Jacobian failed" error log; got {err_msgs}'
    )


@needs_eos
def test_factory_with_radio_isotope_params_builds_callable_radio_heating():
    """The radio 5-tuple (heat_prod, abundance, concentration, t0_years, half_life_years)
    adds a finite heating to the RHS: the RHS with an Al26-like isotope at t0 = 0 is finite
    and differs from the RHS without radionuclides."""
    rhs_off, _, _, n_stag = _build_factory()
    rhs_fn, _, info, _ = _build_factory((0.3583, 1.0, 1.0e-9, 0.0, 7.17e5))
    y_nd = np.full(n_stag, 3050.0 / 3.0e3)
    off, ydot_nd = np.zeros(n_stag), np.zeros(n_stag)
    assert rhs_off(0.0, y_nd, off) == 0 and rhs_fn(0.0, y_nd, ydot_nd) == 0
    assert info['rhs_calls'] == 1
    assert np.all(np.isfinite(ydot_nd)) and np.all(ydot_nd > off)


@needs_eos
@pytest.mark.parametrize('half_life', [0.0, np.nan, 1.0e-310])
def test_factory_rejects_a_half_life_that_is_not_positive(half_life):
    """The CVODE factory refuses a zero, NaN or subnormal half-life before any RHS call."""
    with pytest.raises(ValueError, match='half_life_years must be positive'):
        _build_factory((0.3583, 1.0, 1.0e-9, 4.55e9, half_life))


@needs_eos
def test_factory_zero_amplitude_isotope_with_a_tiny_half_life_heats_zero():
    """A zero-amplitude isotope with a 1e-300 yr half-life leaves the RHS and the Jacobian
    bitwise equal to those without radionuclides."""
    rhs_off, jac_off, _, n_stag = _build_factory()
    rhs_zero, jac_zero, _, _ = _build_factory((0.3583, 0.0, 1.0e-9, 4.55e9, 1.0e-300))
    y_nd = np.full(n_stag, 3050.0 / 3.0e3)
    off, zero = np.zeros(n_stag), np.zeros(n_stag)
    j_off, j_zero = np.zeros((n_stag, n_stag)), np.zeros((n_stag, n_stag))
    assert rhs_off(1.0, y_nd, off) == 0 and rhs_zero(1.0, y_nd, zero) == 0
    assert jac_off(1.0, y_nd, off, j_off) == 0 and jac_zero(1.0, y_nd, zero, j_zero) == 0
    assert np.any(off != 0.0) and np.all(np.isfinite(zero)) and np.all(np.isfinite(j_zero))
    np.testing.assert_array_equal(zero, off)
    np.testing.assert_array_equal(j_zero, j_off)
