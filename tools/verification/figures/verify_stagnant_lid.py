#!/usr/bin/env python3
"""Generator for the stagnant-lid verification plot.

Runs the analytical convection scaling laws and generates the Nusselt-Rayleigh
exponent and boundary layer thickness plots for the documentation.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Make sure we can import aragog modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tests.test_convection_scaling import _probe, _GRADS, _S_REF

def main():
    flux_molten = []
    kappa_molten = []
    flux_stiff = []
    
    for g in _GRADS:
        kh, jc = _probe(1.0, g)
        kappa_molten.append(kh)
        flux_molten.append(jc)
        
        _, jcs = _probe(20.0, g)
        flux_stiff.append(jcs)
        
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(-_GRADS, flux_molten, 'o-', label=r'Free-fall (molten), $\beta \approx 1.5$')
    ax.loglog(-_GRADS, flux_stiff, 's-', label=r'Viscous (stiff), $\beta \approx 2.0$')
    
    ax.set_xlabel(r'Superadiabatic Gradient $|dS/dr|$ [J/kg/K/m]')
    ax.set_ylabel(r'Convective Flux $J_{conv}$ [W/m$^2$]')
    ax.set_title('Convection Scaling Regimes')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs('docs/Explanations/assets', exist_ok=True)
    out_path = 'docs/Explanations/assets/stagnant_lid_scaling.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated {out_path}")

if __name__ == '__main__':
    main()
