#!/usr/bin/env python3
"""Generator for the flux-gradient scaling verification plot.

Runs the analytical convection scaling laws and generates the
flux-gradient exponent plots for the documentation.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from aragog.solver.entropy_state import EntropyState
from aragog.eos.entropy_phase import EntropyPhaseEvaluator

_R_CMB = 3480e3
_R_SURF = 6371e3
_S_REF = 3000.0

def _make_mesh(n=60):
    r_stag = np.linspace(_R_CMB, _R_SURF, n)
    dr = np.diff(r_stag)
    r_basic = np.zeros(n + 1)
    r_basic[0], r_basic[-1] = _R_CMB, _R_SURF
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    p_stag = np.linspace(135e9, 1e5, n)
    p_basic = np.interp(r_basic, r_stag, p_stag)

    class _Sub:
        def __init__(self, r, p):
            self.radii = r
            self.pressure = p
            self.mass_radii = r
            self.mass_radii = r
            self.area = 4 * np.pi * r**2
            self.volume = np.ones_like(r) * 1e20

    class _Mesh:
        def __init__(self):
            self.n_cells = n
            self.basic = _Sub(r_basic, p_basic)
            self.staggered = _Sub(r_stag, p_stag)
            
            from scipy.sparse import diags
            self.d_dr_matrix = diags([-1/1e4, 1/1e4], [0, 1], shape=(n+1, n))
            self.dxidr = np.ones_like(r_basic)
            self.quantity_matrix = diags([0.5, 0.5], [0, 1], shape=(n+1, n))
            
            d2surf = _R_SURF - r_basic
            d2cmb = r_basic - _R_CMB
            self.basic.mixing_length = np.minimum(d2surf, d2cmb)
            self.basic.mixing_length[0] = self.basic.mixing_length[-1] = 1.0
            self.basic.mixing_length_squared = self.basic.mixing_length**2
            self.basic.mixing_length_cubed = self.basic.mixing_length**3

        def quantity_at_basic_nodes(self, x):
            res = np.zeros(self.n_cells + 1)
            res[1:-1] = 0.5 * (x[:-1] + x[1:])
            res[0] = x[0]
            res[-1] = x[-1]
            return res
            
        def gradient_at_basic_nodes(self, x):
            res = np.zeros(self.n_cells + 1)
            dr = np.diff(self.staggered.radii)
            res[1:-1] = np.diff(x) / dr
            res[0] = res[1]
            res[-1] = res[-2]
            return res
            
    return _Mesh()

def _probe(log10visc, grad):
    mesh = _make_mesh()
    
    def _evaluator(pressure):
        ev = EntropyPhaseEvaluator(
            entropy_eos=None, gravitational_acceleration=9.81,
            const_properties=True, const_rho=4000.0, const_Cp=1200.0,
            const_alpha=3e-5, const_cond=4.0, const_log10visc=log10visc,
            const_T_ref=2000.0, const_S_ref=_S_REF,
            yield_stress_c=1e30, yield_stress_mu=0.0, yield_stress_max=1e40,
            stress_closure_mode='local'
        )
        ev.set_pressure(pressure)
        return ev
    
    class _Eval: pass
    evaluator = _Eval()
    evaluator.mesh = mesh
    
    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=_evaluator(mesh.staggered.pressure),
        phase_basic=_evaluator(mesh.basic.pressure),
        conduction=True, convection=True
    )
    
    rs = np.asarray(mesh.staggered.radii).ravel()
    entropy = _S_REF + grad * (rs - rs.mean())
    state.update(entropy, 0.0)
    
    kappa_h = np.abs(np.asarray(state.eddy_diffusivity).ravel()[1:-1])
    jconv = np.abs(np.asarray(state.jconv).ravel()[1:-1])
    return float(np.max(kappa_h)), float(np.max(jconv))

def main():
    _GRADS = np.array([-1.0e-7, -3.0e-7, -1.0e-6, -3.0e-6, -1.0e-5])
    flux_molten = []
    flux_stiff = []
    
    for g in _GRADS:
        _, jc = _probe(1.0, g)
        flux_molten.append(jc)
        
        _, jcs = _probe(20.0, g)
        flux_stiff.append(jcs)
        
    try:
        import _style
        _style.apply_rc()
        c0 = _style.PALETTE["numpy"]
        c1 = _style.PALETTE["jax"]
    except ImportError:
        c0, c1 = 'C0', 'C1'

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(-_GRADS, flux_molten, 'o-', color=c0, label=r'Free-fall (molten), $\beta \approx 1.5$')
    ax.loglog(-_GRADS, flux_stiff, 's-', color=c1, label=r'Viscous (stiff), $\beta \approx 2.0$')
    
    ax.set_xlabel(r'Superadiabatic Gradient $|dS/dr|$ [J/kg/K/m]')
    ax.set_ylabel(r'Convective Flux $J_{conv}$ [W/m$^2$]')
    ax.set_title('Flux-Gradient Scaling Regimes')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../docs/figures/vv'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'flux_gradient_scaling.png')
    try:
        import _style
        _style.save(fig, os.path.join(out_dir, 'flux_gradient_scaling'))
    except ImportError:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../output/aragog_vv_data'))
    os.makedirs(data_dir, exist_ok=True)
    npz_path = os.path.join(data_dir, 'flux_gradient_scaling.npz')
    np.savez(npz_path, grads=-_GRADS, flux_molten=flux_molten, flux_stiff=flux_stiff)
    
    print(f"Generated {out_path}")
    print(f"Generated {npz_path}")

if __name__ == '__main__':
    main()
