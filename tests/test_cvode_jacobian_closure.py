"""Verification of the analytic CVODE Jacobian with stagnant lid stress closure.

Asserts row-by-row agreement between the analytic Jacobian from jax.jacrev
and central finite differences on the entropy ODE state vector, with special
focus on rows inside the stagnant lid (w_lid > 0.01).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    None,
)

needs_eos = pytest.mark.skipif(
    EOS_DIR is None,
    reason='SPIDER P-S tables not found',
)

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

jax.config.update('jax_enable_x64', True)

from aragog.jax import rheology as jax_rheo  # noqa: E402
from aragog.jax.eos import EntropyEOS_JAX  # noqa: E402
from aragog.jax.phase import MeshArrays, PhaseParams  # noqa: E402
from aragog.jax.solver import BoundaryParams, _no_radio, dSdt  # noqa: E402


@pytest.fixture(scope='module')
def shared_eos_jax():
    if EOS_DIR is None:
        pytest.skip('EOS_DIR not found')
    return EntropyEOS_JAX(EOS_DIR)


@pytest.mark.unit
@pytest.mark.physics_invariant
@needs_eos
def test_analytic_jacobian_matches_finite_differences_lid(shared_eos_jax):
    """Verify analytic Jacobian matches central finite differences row-by-row in lid."""
    n_stag = 20
    n_basic = n_stag + 1

    r_basic = np.linspace(3.48e6, 6.371e6, n_basic)
    r_stag = 0.5 * (r_basic[:-1] + r_basic[1:])

    d_dr_matrix = np.zeros((n_basic, n_stag))
    for i in range(1, n_basic - 1):
        dr = r_stag[i] - r_stag[i - 1]
        d_dr_matrix[i, i] = 1.0 / dr
        d_dr_matrix[i, i - 1] = -1.0 / dr

    quantity_matrix = np.zeros((n_basic, n_stag))
    for i in range(1, n_basic - 1):
        quantity_matrix[i, i - 1] = 0.5
        quantity_matrix[i, i] = 0.5
    quantity_matrix[0, 0] = 1.0
    quantity_matrix[-1, -1] = 1.0

    area = 4.0 * np.pi * r_basic**2
    volume = 4.0 / 3.0 * np.pi * (r_basic[1:] ** 3 - r_basic[:-1] ** 3)
    mixing_length = 0.1 * (r_basic[-1] - r_basic[0]) * np.ones(n_basic)
    p_stag = np.linspace(135.0e9, 1.0e9, n_stag)
    p_basic = np.linspace(135.0e9, 0.0, n_basic)
    dp_dr = np.gradient(p_basic, r_basic)
    g_basic = np.full(n_basic, 9.81)

    mesh = MeshArrays(
        d_dr_matrix=jnp.array(d_dr_matrix),
        quantity_matrix=jnp.array(quantity_matrix),
        area=jnp.array(area),
        volume=jnp.array(volume),
        radii_basic=jnp.array(r_basic),
        radii_stag=jnp.array(r_stag),
        mixing_length=jnp.array(mixing_length),
        mixing_length_sq=jnp.array(mixing_length**2),
        mixing_length_cu=jnp.array(mixing_length**3),
        P_stag=jnp.array(p_stag),
        P_basic=jnp.array(p_basic),
        dP_dr_basic=jnp.array(dp_dr),
        gravity=jnp.array(g_basic),
    )

    params = PhaseParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_max=50.0e6,
    )

    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=1500.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=2,
        inner_bc_value=0.0,
        core_density=10000.0,
        core_heat_capacity=800.0,
        tfac_core_avg=1.0,
    )

    entropy = jnp.linspace(3200.0, 2600.0, n_stag)
    args = (shared_eos_jax, params, mesh, bc, jnp.zeros(n_stag), _no_radio)

    def _rhs(s):
        return dSdt(0.0, s, args)

    j_analytic = jax.jacrev(_rhs)(entropy)
    assert jnp.all(jnp.isfinite(j_analytic)), 'Analytic Jacobian must be finite'

    # Compute central finite differences
    eps = 1.0e-4
    j_fd = np.zeros_like(j_analytic)
    for j in range(n_stag):
        s_plus = np.array(entropy)
        s_plus[j] += eps
        s_minus = np.array(entropy)
        s_minus[j] -= eps
        j_fd[:, j] = (
            np.array(_rhs(jnp.array(s_plus))) - np.array(_rhs(jnp.array(s_minus)))
        ) / (2.0 * eps)

    # Determine lid nodes via stagnant lid state
    v_dummy = np.zeros(n_basic)
    lid_state = jax_rheo.compute_stagnant_lid_state(
        radii=mesh.radii_basic,
        temperature=jnp.linspace(3400.0, 1500.0, n_basic),
        pressure=mesh.P_basic,
        convective_flux=jnp.full(n_basic, 100.0),
        total_flux=jnp.full(n_basic, 120.0),
        solidus_temperature=None,
        melt_fraction=jnp.zeros(n_basic),
        params=params,
        unyielded_velocity=jnp.array(v_dummy),
        viscosity_solid=1.0e21,
    )
    w_lid_basic = np.array(lid_state['w_lid'])
    # Map basic lid mask to staggered rows
    w_lid_stag = 0.5 * (w_lid_basic[:-1] + w_lid_basic[1:])
    lid_rows = np.where(w_lid_stag > 0.01)[0]

    assert len(lid_rows) > 0, 'Test configuration must produce at least one lid row'

    for row in lid_rows:
        row_an = np.array(j_analytic[row, :])
        row_fd = j_fd[row, :]
        scale = np.maximum(np.abs(row_fd), 1.0e-10)
        rel_diff = np.abs(row_an - row_fd) / scale
        assert np.max(rel_diff) < 1.0e-4, (
            f'Jacobian mismatch in lid row {row}: max rel diff {np.max(rel_diff)}'
        )
