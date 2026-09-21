"""JAX and NumPy float64 parity.

Pins float64 numerical parity at a stiff, no-yield operating point between JAX and NumPy backends
for the mixing length theory (MLT) closure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60), pytest.mark.physics_invariant]
pytest.importorskip('jax')

from aragog.jax.phase import compute_mlt as jax_compute_mlt  # noqa: E402


def test_jax_numpy_float64_parity():
    """Verify that JAX and NumPy yield identical numerical results within a relative tolerance of 1e-8
    for the mixing length theory (MLT) closure and effective viscosity.
    """
    import jax

    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(__file__))
    from test_convection_scaling import _make_mesh, _make_state

    mesh = _make_mesh(n=50)
    state = _make_state(mesh, log10visc=20.0)

    ds_dr = -1e-6
    rs = np.asarray(mesh.staggered.radii).ravel()
    entropy = 3000.0 + ds_dr * (rs - rs.mean())
    state.update(entropy, 0.0)

    kappa_numpy = np.array(state.eddy_diffusivity).ravel()

    # Execute in JAX
    from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties

    n_basic = len(mesh.basic.radii)
    n_stag = len(mesh.staggered.radii)

    mesh_jax = MeshArrays(
        d_dr_matrix=jnp.array(np.zeros((n_basic, n_stag))),
        quantity_matrix=jnp.array(np.zeros((n_basic, n_stag))),
        area=jnp.array(mesh.basic.area).ravel(),
        volume=jnp.array(mesh.basic.volume).ravel(),
        radii_basic=jnp.array(mesh.basic.radii).ravel(),
        radii_stag=jnp.array(mesh.staggered.radii).ravel(),
        mixing_length=jnp.array(state._mixing_length).ravel(),
        mixing_length_sq=jnp.array(state._mixing_length_sq).ravel(),
        mixing_length_cu=jnp.array(state._mixing_length_cu).ravel(),
        P_stag=jnp.array(state.phase_staggered.pressure).ravel(),
        P_basic=jnp.array(state.phase_basic.pressure).ravel(),
        dP_dr_basic=jnp.array(state._dP_dr_basic).ravel(),
        gravity=jnp.array(state.phase_basic.gravitational_acceleration()).ravel(),
    )

    phase_jax = PhaseProperties(
        temperature=jnp.array(state.T_basic_diag).ravel(),
        density=jnp.array(state.rho_basic_diag).ravel(),
        heat_capacity=jnp.array(state.cp_basic_diag).ravel(),
        thermal_expansivity=jnp.array(state.phase_basic.thermal_expansivity()).ravel(),
        dTdPs=jnp.array(state._dS_liq_dP_basic).ravel(),
        melt_fraction=jnp.array(state.phi_basic_diag).ravel(),
        viscosity=jnp.array(state.phase_basic.viscosity()).ravel(),
        kinematic_viscosity=jnp.array(
            state.phase_basic.viscosity() / state.rho_basic_diag
        ).ravel(),
        thermal_conductivity=jnp.array(state.phase_basic.thermal_conductivity()).ravel(),
        latent_heat=jnp.array(state.phase_basic.latent_heat()).ravel(),
        capacitance=jnp.array(state.cp_basic_diag * state.rho_basic_diag).ravel(),
        eta_diff=jnp.array(state.phase_basic.eta_diff).ravel() if state.phase_basic.eta_diff is not None else None,
        tau_y=jnp.array(state.phase_basic.tau_y).ravel() if state.phase_basic.tau_y is not None else None,
        visc_solid_weight=jnp.array(np.ones_like(state.rho_basic_diag)).ravel(),
    )

    params = PhaseParams(
        kappah_floor=0.0,
        stress_closure_mode='local',
        lid_base_mode='rheological',
        lid_contrast_coeff=2.2,
        activation_energy=300e3,
        activation_volume=1.5e-6,
    )

    ds_dr_jax = jnp.array(state._dSdr).ravel()
    k_h_jax, _ = jax_compute_mlt(ds_dr_jax, phase_jax, mesh_jax, params)

    assert k_h_jax.dtype == jnp.float64
    np.testing.assert_allclose(np.array(k_h_jax), kappa_numpy, rtol=1e-8, atol=1e-30)
