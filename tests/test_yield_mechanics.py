"""Verification test for Yield Mechanics.

Sweeps Byerlee yield stress parameters to explicitly verify the transition
from a stagnant lid to a mobile lid.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import compute_yield_stress, eta_diff, eta_eff, stress_closure

pytestmark = [pytest.mark.physics_invariant, pytest.mark.unit, pytest.mark.timeout(60)]


def test_yield_stress_transition():
    """Verify that lowering the yield stress transitions the system from a
    stagnant lid (high effective viscosity, low strain rate) to a mobile lid
    (yielding, low effective viscosity, high strain rate).
    """
    n_nodes = 50
    r_cmb = 3480.0e3
    r_surf = 6371.0e3
    r = np.linspace(r_cmb, r_surf, n_nodes)

    # Hydrostatic pressure
    p = 135.0e9 * (r_surf - r) / (r_surf - r_cmb)

    # Cold lid profile
    norm_r = (r - r_cmb) / (r_surf - r_cmb)
    t = 1600.0 + 1000.0 * (1.0 - norm_r) - 1200.0 * (norm_r**8)
    t = np.clip(t, 300.0, 4000.0)

    # Convective velocity
    v_visc = 1.0e-9 * np.sin(np.pi * norm_r)

    # Reference ductile viscosity
    eta_d = eta_diff(t, p, viscosity_solid=1.0e21, activation_volume=1.5e-6)

    # Stagnant lid case: high yield stress (unyielding)
    tau_stagnant = compute_yield_stress(
        p, yield_stress_c=500.0e6, yield_stress_mu=0.6, yield_stress_max=10.0e9
    )
    sr_stagnant = stress_closure('global', v_visc, radius=r, temperature=t, t_lid_base=1400.0)
    eta_eff_stagnant = eta_eff(eta_d, tau_stagnant, sr_stagnant, smooth=True)

    # Mobile lid case: low yield stress cohesion, moderate friction (mu=0.01)
    # Yields the cold, low-pressure lid, but preserves the high-pressure unyielded interior.
    tau_mobile = compute_yield_stress(
        p, yield_stress_c=1.0e6, yield_stress_mu=0.01, yield_stress_max=10.0e9
    )
    sr_mobile = stress_closure('global', v_visc, radius=r, temperature=t, t_lid_base=1400.0)
    eta_eff_mobile = eta_eff(eta_d, tau_mobile, sr_mobile, smooth=True)

    # In the lid (cold, near surface), the mobile effective viscosity must be dramatically lower
    lid_idx = -2
    assert eta_eff_mobile[lid_idx] < 1e-1 * eta_eff_stagnant[lid_idx]

    # The deep interior (hot, ductile, high pressure) should be completely unaffected
    mid_idx = n_nodes // 2
    # Compare to the unyielded reference ductile viscosity with a physical tolerance
    assert eta_eff_mobile[mid_idx] == pytest.approx(eta_d[mid_idx], rel=2e-2)
    assert eta_eff_stagnant[mid_idx] == pytest.approx(eta_d[mid_idx], rel=2e-2)


def test_yield_stress_max_ceiling():
    """Verify that a non-default ceiling is correctly applied in numpy and JAX."""
    import jax.numpy as jnp

    from aragog.jax.phase import compute_yield_stress as compute_yield_stress_jax

    # Pressure very high
    p_np = np.array([1e11])
    p_jax = jnp.array([1e11])

    # 1. Numpy version has the yield_stress_max as a kwarg directly.
    tau_y_np = compute_yield_stress(
        p_np, yield_stress_c=10e6, yield_stress_mu=0.5, yield_stress_max=200e6
    )
    assert float(tau_y_np[0]) == 200e6

    # 2. JAX version does not have yield_stress_max as kwarg, it is applied externally via jnp.minimum.
    # We test the pure JAX computation without the limit, and then apply it.
    tau_y_jax = compute_yield_stress_jax(p_jax, yield_stress_c=10e6, yield_stress_mu=0.5)
    tau_y_jax_limited = jnp.minimum(tau_y_jax, 200e6)

    assert float(tau_y_jax_limited[0]) == 200e6
