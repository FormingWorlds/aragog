#!/usr/bin/env python3
"""Generator for the Arrhenius viscosity and yielding verification plot."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from aragog.rheology import compute_yield_stress, eta_eff, stress_closure, eta_diff

def main():
    n_nodes = 500
    r_cmb = 3480.0e3
    r_surf = 6371.0e3
    r = np.linspace(r_cmb, r_surf, n_nodes)

    p = 135.0e9 * (r_surf - r) / (r_surf - r_cmb)

    norm_r = (r - r_cmb) / (r_surf - r_cmb)
    t = 1600.0 + 1000.0 * (1.0 - norm_r) - 1200.0 * (norm_r**8)
    t = np.clip(t, 300.0, 4000.0)

    v_visc = 1.0e-9 * np.sin(np.pi * norm_r)

    eta_d = eta_diff(t, p, viscosity_solid=1.0e21, activation_volume=1.5e-6)

    # Stagnant lid
    tau_stagnant = compute_yield_stress(p, yield_stress_c=500.0e6, yield_stress_mu=0.6, yield_stress_max=500.0e6)
    sr_stagnant = stress_closure('global', v_visc, radius=r, temperature=t, t_lid_base=1400.0)
    eta_eff_stagnant = eta_eff(eta_d, tau_stagnant, sr_stagnant, smooth=True)

    # Mobile lid
    tau_mobile = compute_yield_stress(p, yield_stress_c=1.0e6, yield_stress_mu=0.001, yield_stress_max=500.0e6)
    sr_mobile = stress_closure('global', v_visc, radius=r, temperature=t, t_lid_base=1400.0)
    eta_eff_mobile = eta_eff(eta_d, tau_mobile, sr_mobile, smooth=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    depth = (r_surf - r) / 1e3
    
    ax.semilogx(eta_eff_stagnant, depth, label='Stagnant Lid (High Yield Stress)', color='C0')
    ax.semilogx(eta_eff_mobile, depth, label='Mobile Lid (Low Yield Stress)', color='C1', linestyle='--')
    
    ax.set_ylabel('Depth [km]')
    ax.set_xlabel('Effective Viscosity [Pa s]')
    ax.set_title('Byerlee Yield Mechanics: Transition to Mobile Lid')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs('docs/Explanations/assets', exist_ok=True)
    out_path = 'docs/Explanations/assets/arrhenius_yielding.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated {out_path}")

if __name__ == '__main__':
    main()
