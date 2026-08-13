"""Invocation tests for ``build_jax_rhs_and_jacobian``.

The contract tests in ``test_cvode_jax_factory.py`` verify the factory
constructs and rejects bad shapes. They do NOT call the returned
``rhs_fn`` / ``jacfn`` callbacks. The body of those callbacks (lines
213-237 of ``solver/cvode_jax.py``) and the underlying ``_rhs_phys`` /
``_rhs_nondim`` closures (lines 187, 193-196) are the remaining
coverage gap on this module.

These tests build the factory with realistic JAX pytrees and then
invoke ``rhs_fn`` and ``jacfn`` with stub ydot / J buffers, exercising
the JIT-compile + counter logic and the SECS_PER_YEAR + scaling
plumbing on the way through.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
eqx = pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


def _make_const_property_mesh(N: int = 10):
    """Same factory as test_jax_dsdt_energy_balance, kept local to avoid
    a cross-test import dependency.
    """
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


@needs_eos
def test_factory_rhs_fn_writes_finite_ydot_in_place():
    """Calling ``rhs_fn(t, y, ydot)`` with a finite state must fill
    ``ydot`` in place with finite values and return 0 (CVODE success).

    Discriminator: a regression that lost the ``ydot[:] = ...`` write
    would leave ydot zero, and any downstream BDF Newton iterate would
    silently make zero progress. ``rhs_calls`` counter must increment
    so the diagnostic logger reports realistic counts.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.nondim import NonDimScales
    from aragog.jax.phase import PhaseParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    eos_jax = EntropyEOS_JAX(EOS_DIR)
    params = PhaseParams()
    mesh = _make_const_property_mesh(N=10)
    bc = _make_bc()
    n_stag = int(mesh.P_stag.shape[0])

    state_scale = np.concatenate([np.full(n_stag, 3.0e3), np.full(2, 1.0e32)])
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
    )

    # State and ydot in nondim space (S0/state_scale ~ 1.01).
    y_nd = np.concatenate([np.full(n_stag, 3050.0 / 3.0e3), np.zeros(2)])
    ydot_nd = np.zeros(n_stag + 2)

    rc = rhs_fn(0.0, y_nd, ydot_nd)
    assert rc == 0, f'rhs_fn returned {rc}; expected 0 (CVODE success)'
    assert info['rhs_calls'] == 1, (
        f'rhs_calls counter is {info["rhs_calls"]}, expected 1 after one call'
    )
    assert info['first_rhs_compile_done'], (
        'first_rhs_compile_done flag must be set after the JIT compile'
    )
    assert np.all(np.isfinite(ydot_nd)), f'ydot_nd has non-finite entries: {ydot_nd}'
    # The state is at moderate entropy, well clear of the boundary;
    # ydot should be a non-trivial finite vector with at least one
    # non-zero entry. A regression that left ydot zero would fail.
    assert np.any(np.abs(ydot_nd) > 1e-12), (
        'ydot_nd is all-zero after rhs_fn; expected non-trivial dS/dt'
    )


@needs_eos
def test_factory_jacfn_writes_finite_jacobian_in_place():
    """Calling ``jacfn(t, y, fy, J)`` must fill the J buffer in place
    with a finite (n, n) Jacobian and return 0.

    Discriminator: a regression in the autodiff path (e.g. wrong
    argnums) would either return a non-square Jacobian or zero out
    the matrix.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.nondim import NonDimScales
    from aragog.jax.phase import PhaseParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    eos_jax = EntropyEOS_JAX(EOS_DIR)
    params = PhaseParams()
    mesh = _make_const_property_mesh(N=8)
    bc = _make_bc()
    n_stag = int(mesh.P_stag.shape[0])

    state_scale = np.concatenate([np.full(n_stag, 3.0e3), np.full(2, 1.0e32)])
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
    )

    y_nd = np.concatenate([np.full(n_stag, 3050.0 / 3.0e3), np.zeros(2)])
    fy_nd = np.zeros(n_stag + 2)
    J = np.zeros((n_stag + 2, n_stag + 2), dtype=float)

    rc = jac_fn(0.0, y_nd, fy_nd, J)
    assert rc == 0, f'jacfn returned {rc}, expected 0'
    assert info['jac_calls'] == 1
    assert info['first_jac_compile_done']
    assert J.shape == (n_stag + 2, n_stag + 2)
    assert np.all(np.isfinite(J)), 'jacobian has non-finite entries'
    # A non-trivial Jacobian must have non-zero entries (the entropy
    # equation couples nodes via the flux divergence). All-zero would
    # mean autodiff dropped the dependency.
    assert np.any(np.abs(J) > 1e-20), 'jacobian is all-zero; autodiff path is broken'


def test_factory_radio_isotope_5_tuple_must_be_full_length():
    """Bad-length ``radio_isotope_params`` (e.g. 4-tuple instead of 5)
    must raise ValueError with a clear message that names the
    expected layout.

    Discriminator: a regression that silently truncated would leave
    the radio-heating function broken at runtime; raising at factory-
    build time is the correct surface.
    """
    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    n = 4
    state_scale = np.concatenate([np.full(n, 3.0e3), np.full(2, 1.0e32)])
    scales = NonDimScales(state_scale=state_scale, t_ref=1.0)
    with pytest.raises(ValueError, match='radio_isotope_params must be a 5-tuple'):
        build_jax_rhs_and_jacobian(
            eos_jax=None,
            phase_params=None,
            mesh_arrays=None,
            boundary_params=None,
            heating_array=np.zeros(n),
            scales=scales,
            core_bc_mode='quasi_steady',
            radio_isotope_params=(1.0, 2.0, 3.0, 4.0),  # 4 instead of 5
        )
