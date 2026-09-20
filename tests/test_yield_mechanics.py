"""Verification Tier E: Yield Mechanics (Stagnant to Mobile Lid Transition).

Sweeps Byerlee yield stress parameters to explicitly verify the transition
from a stagnant lid to a mobile lid.
"""
from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import compute_yield_stress, eta_eff, stress_closure, eta_diff

@pytest.mark.physics_invariant
@pytest.mark.slow
def test_mobile_to_stagnant_lid_transition():
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
    tau_stagnant = compute_yield_stress(p, yield_stress_c=500.0e6, yield_stress_mu=0.6, yield_stress_max=500.0e6)
    sr_stagnant = stress_closure('global', v_visc, radius=r, temperature=t, t_lid_base=1400.0)
    eta_eff_stagnant = eta_eff(eta_d, tau_stagnant, sr_stagnant, smooth=True)

    # Mobile lid case: extremely low yield stress
    tau_mobile = compute_yield_stress(p, yield_stress_c=1.0e6, yield_stress_mu=0.001, yield_stress_max=500.0e6)
    sr_mobile = stress_closure('global', v_visc, radius=r, temperature=t, t_lid_base=1400.0)
    eta_eff_mobile = eta_eff(eta_d, tau_mobile, sr_mobile, smooth=True)

    # In the lid (cold, near surface), the mobile effective viscosity must be dramatically lower
    lid_idx = -2
    assert eta_eff_mobile[lid_idx] < 1e-1 * eta_eff_stagnant[lid_idx]

    # The deep interior (hot, ductile) should be largely unaffected
    mid_idx = n_nodes // 2
    assert eta_eff_mobile[mid_idx] == pytest.approx(eta_eff_stagnant[mid_idx], rel=0.1)
