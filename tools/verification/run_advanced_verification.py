"""Advanced first-principles verification plots for the tech document.

Produces a 4-panel figure:
(a) Conduction steady state: T(r) vs analytical A/r + B
(b) Eigenvalue decay: perturbation amplitude vs time with exp(-t/tau) fit
(c) Nu-Ra scaling: Nusselt vs Rayleigh number across viscosity sweep
(d) Boundary layer: T(r) profiles at two viscosities showing BL thinning

All tests use constant-property physics: T = T_ref * exp((S-S_ref)/Cp).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'tests'))
from test_entropy_advanced import (
    RHO, CP, K_COND, ALPHA, T_REF, S_REF, KAPPA,
    R_INNER, R_OUTER, D_SHELL, G, SECS_PER_YEAR,
    T_to_S, S_to_T, analytical_T, ConstPropState, make_const_mesh,
)

OUT_DIR = Path(__file__).parent

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 15,
    'legend.fontsize': 10, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'lines.linewidth': 2.0, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


def run_conduction_test():
    """Panel (a): conduction steady state T(r) = A/r + B."""
    print('  (a) Conduction steady state...')
    results = {}
    for N, ls in [(50, '--'), (100, '-'), (200, ':')]:
        mesh = make_const_mesh(N)
        state = ConstPropState(mesh, convection=False)
        T_inner, T_outer = 4000.0, 1500.0
        T_ss = analytical_T(mesh.r_stag, T_inner, T_outer)
        S_ss = T_to_S(T_ss)

        a, b = R_INNER, R_OUTER
        A_c = (T_inner - T_outer) * a * b / (b - a)
        Q_an = 4 * np.pi * K_COND * A_c
        F_in = Q_an / (4 * np.pi * R_INNER**2)
        F_out = Q_an / (4 * np.pi * R_OUTER**2)

        def rhs(t, S, _s=state, _fi=F_in, _fo=F_out):
            return _s.compute_dSdt(t, S, F_inner=_fi, F_outer=_fo)

        sol = solve_ivp(rhs, (0, 1e6), S_ss, method='BDF', atol=0.01, rtol=1e-8)
        T_final = S_to_T(sol.y[:, -1])
        results[N] = (mesh.r_stag, T_ss, T_final, ls)

    return results


def run_eigenvalue_test():
    """Panel (b): eigenvalue decay."""
    print('  (b) Eigenvalue decay...')
    k_test = 4000.0
    kappa_test = k_test / (RHO * CP)
    tau_yr = D_SHELL**2 / (np.pi**2 * kappa_test) / SECS_PER_YEAR

    N = 100
    mesh = make_const_mesh(N)
    T_inner, T_outer = 3500.0, 3000.0
    T_ss = analytical_T(mesh.r_stag, T_inner, T_outer)
    S_ss = T_to_S(T_ss)

    delta = 100.0
    r = mesh.r_stag
    pert = delta * np.sin(np.pi * (r - R_INNER) / D_SHELL) * ((R_INNER + R_OUTER) / 2) / r
    S_pert = T_to_S(T_ss + pert)

    a, b = R_INNER, R_OUTER
    A_c = (T_inner - T_outer) * a * b / (b - a)
    Q_an = 4 * np.pi * k_test * A_c
    F_in = Q_an / (4 * np.pi * R_INNER**2)
    F_out = Q_an / (4 * np.pi * R_OUTER**2)

    class HighKState(ConstPropState):
        def dSdt(self, t, S):
            S = np.asarray(S).flatten()
            T_s = S_to_T(S)
            T_b = self.mesh.quantity_at_basic_nodes(T_s)
            dTdr = self.mesh.d_dr_at_basic_nodes(T_s)
            self.heat_flux = -k_test * dTdr
            self._T_basic = T_b
            return self.heat_flux

    state = HighKState(mesh, convection=False)

    def rhs(t, S):
        state.dSdt(t, S)
        state.heat_flux[0] = F_in
        state.heat_flux[-1] = F_out
        ef = state.heat_flux * mesh.area
        cap = RHO * S_to_T(S) * mesh.volume
        dsdt = -np.diff(ef) / cap * SECS_PER_YEAR
        relax = 1e6
        dsdt[0] += relax * (S_ss[0] - S[0])
        dsdt[-1] += relax * (S_ss[-1] - S[-1])
        return dsdt

    t_end = 2.0 * tau_yr
    sol = solve_ivp(rhs, (0, t_end), S_pert, method='BDF',
                    atol=1e-4, rtol=1e-8, dense_output=True)

    n_samp = 50
    times = np.linspace(0.01 * tau_yr, t_end, n_samp)
    amps = np.array([np.max(np.abs(S_to_T(sol.sol(t)) - T_ss)) for t in times])

    def exp_decay(t, A0, tau):
        return A0 * np.exp(-t / tau)
    popt, _ = curve_fit(exp_decay, times, amps, p0=[delta, tau_yr])

    return times, amps, tau_yr, popt


def run_nura_test():
    """Panel (c): Nu-Ra scaling."""
    print('  (c) Nu-Ra scaling...')
    N = 50
    mesh = make_const_mesh(N)
    T_inner, T_outer = 4000.0, 1500.0
    DT = T_inner - T_outer

    A_c = DT * R_INNER * R_OUTER / D_SHELL
    Q_cond = 4 * np.pi * K_COND * A_c
    F_cond_surf = Q_cond / (4 * np.pi * R_OUTER**2)

    alpha_conv = 1e-5
    viscosities = np.logspace(10, 19, 10)
    Nu_vals, Ra_vals = [], []

    for visc in viscosities:
        state = ConstPropState(mesh, viscosity=visc, convection=True, alpha=alpha_conv)
        T_uniform = 0.5 * (T_inner + T_outer)
        S_init = T_to_S(np.full(N, T_uniform))
        S_in_bc = T_to_S(T_inner)
        S_out_bc = T_to_S(T_outer)
        F_in = Q_cond / (4 * np.pi * R_INNER**2)
        F_out = Q_cond / (4 * np.pi * R_OUTER**2)

        def rhs(t, S, _s=state, _fi=F_in, _fo=F_out, _si=S_in_bc, _so=S_out_bc):
            _s.dSdt(t, S)
            _s.heat_flux[0] = _fi
            _s.heat_flux[-1] = _fo
            ef = _s.heat_flux * mesh.area
            cap = RHO * S_to_T(S) * mesh.volume
            dsdt = -np.diff(ef) / cap * SECS_PER_YEAR
            dsdt[0] += 1e6 * (_si - S[0])
            dsdt[-1] += 1e6 * (_so - S[-1])
            return dsdt

        sol = solve_ivp(rhs, (0, 1e6), S_init, method='BDF', atol=0.01, rtol=1e-6)
        if sol.status != 0:
            continue

        state.dSdt(0, sol.y[:, -1])
        F_surf = abs(state.heat_flux[-1])
        Nu = F_surf / abs(F_cond_surf) if abs(F_cond_surf) > 0 else 1.0
        nu = visc / RHO
        Ra = RHO * G * alpha_conv * DT * D_SHELL**3 / (nu * KAPPA)
        Nu_vals.append(Nu)
        Ra_vals.append(Ra)

    return np.array(Ra_vals), np.array(Nu_vals)


def run_bl_test():
    """Panel (d): T(r) profiles at two viscosities."""
    print('  (d) Boundary layer profiles...')
    N = 100
    mesh = make_const_mesh(N)
    mesh.mixing_length = np.maximum(
        np.minimum(mesh.r_basic - R_INNER, R_OUTER - mesh.r_basic), 1.0)

    T_inner, T_outer = 4000.0, 1500.0
    DT = T_inner - T_outer
    A_c = DT * R_INNER * R_OUTER / D_SHELL
    Q_cond = 4 * np.pi * K_COND * A_c
    F_in = Q_cond / (4 * np.pi * R_INNER**2)
    F_out = Q_cond / (4 * np.pi * R_OUTER**2)
    S_in_bc = T_to_S(T_inner)
    S_out_bc = T_to_S(T_outer)

    alpha_conv = 1e-5
    profiles = {}

    for visc, label in [(1e16, r'$\eta = 10^{16}$'), (1e13, r'$\eta = 10^{13}$'),
                         (1e10, r'$\eta = 10^{10}$')]:
        state = ConstPropState(mesh, viscosity=visc, convection=True, alpha=alpha_conv)
        T_uniform = 0.5 * (T_inner + T_outer)
        S_init = T_to_S(np.full(N, T_uniform))

        def rhs(t, S, _s=state, _fi=F_in, _fo=F_out, _si=S_in_bc, _so=S_out_bc):
            _s.dSdt(t, S)
            _s.heat_flux[0] = _fi
            _s.heat_flux[-1] = _fo
            ef = _s.heat_flux * mesh.area
            cap = RHO * S_to_T(S) * mesh.volume
            dsdt = -np.diff(ef) / cap * SECS_PER_YEAR
            dsdt[0] += 1e6 * (_si - S[0])
            dsdt[-1] += 1e6 * (_so - S[-1])
            return dsdt

        sol = solve_ivp(rhs, (0, 1e6), S_init, method='BDF', atol=0.01, rtol=1e-6)
        if sol.status == 0:
            T_final = S_to_T(sol.y[:, -1])
            profiles[label] = (mesh.r_stag / 1e3, T_final)

    # Analytical conduction profile for reference
    r_fine = np.linspace(R_INNER, R_OUTER, 200)
    T_cond = analytical_T(r_fine)
    profiles['Conduction'] = (r_fine / 1e3, T_cond)

    return profiles


def main():
    print('Generating advanced verification plots...')

    cond_results = run_conduction_test()
    eig_times, eig_amps, eig_tau, eig_fit = run_eigenvalue_test()
    nura_Ra, nura_Nu = run_nura_test()
    bl_profiles = run_bl_test()

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel (a): Conduction steady state
    ax = axes[0, 0]
    r_an = np.linspace(R_INNER, R_OUTER, 200)
    ax.plot(r_an / 1e3, analytical_T(r_an), 'k-', linewidth=2.5, label='Analytical $A/r + B$')
    for N, (r, T_ss, T_final, ls) in cond_results.items():
        err = np.max(np.abs(T_final - T_ss))
        ax.plot(r / 1e3, T_final, ls, linewidth=1.5,
                label=f'N={N} (max err {err:.1f} K)')
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(a) Conduction steady state')
    ax.legend(fontsize=9)

    # Panel (b): Eigenvalue decay
    ax = axes[0, 1]
    ax.semilogy(eig_times / eig_tau, eig_amps, 'bo', markersize=4, label='Numerical')
    t_fit = np.linspace(eig_times[0], eig_times[-1], 100)
    ax.semilogy(t_fit / eig_tau, eig_fit[0] * np.exp(-t_fit / eig_fit[1]),
                'r-', linewidth=2, label=f'Fit: $\\tau$ = {eig_fit[1]:.2e} yr')
    ax.semilogy(t_fit / eig_tau, eig_fit[0] * np.exp(-t_fit / eig_tau),
                'k--', linewidth=1.5, label=f'Analytical: $\\tau$ = {eig_tau:.2e} yr')
    rel_err = abs(eig_fit[1] - eig_tau) / eig_tau
    ax.set_xlabel(r'Time [$\tau_\mathrm{analytical}$]')
    ax.set_ylabel('Perturbation amplitude [K]')
    ax.set_title(f'(b) Eigenvalue decay ({rel_err:.0%} error)')
    ax.legend(fontsize=9)

    # Panel (c): Nu-Ra scaling
    ax = axes[1, 0]
    if len(nura_Ra) > 0:
        ax.loglog(nura_Ra, nura_Nu, 'bs-', markersize=7, label='Entropy solver')
        # Reference slopes
        Ra_ref = np.logspace(np.log10(nura_Ra.min()), np.log10(nura_Ra.max()), 50)
        ax.loglog(Ra_ref, 0.1 * (Ra_ref / Ra_ref[0]), 'k--', alpha=0.4,
                  label=r'$\mathrm{Nu} \propto \mathrm{Ra}^1$ (viscous MLT)')
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5, label='Nu = 1 (conduction)')
    ax.set_xlabel('Rayleigh number')
    ax.set_ylabel('Nusselt number')
    ax.set_title(r'(c) Nu-Ra scaling ($\alpha = 10^{-5}$)')
    ax.legend(fontsize=9)

    # Panel (d): BL profiles
    ax = axes[1, 1]
    colors = {'Conduction': 'k', r'$\eta = 10^{16}$': '#4477AA',
              r'$\eta = 10^{13}$': '#EE6677', r'$\eta = 10^{10}$': '#228833'}
    for label, (r_km, T) in bl_profiles.items():
        ls = '--' if label == 'Conduction' else '-'
        lw = 1.5 if label == 'Conduction' else 2.0
        ax.plot(r_km, T, ls, color=colors.get(label, 'gray'),
                linewidth=lw, label=label)
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(d) Boundary layer structure')
    ax.legend(fontsize=9)

    fig.suptitle('Advanced verification: constant-property entropy solver', fontsize=15, y=1.005)
    fig.tight_layout()

    fname = OUT_DIR / 'verify_entropy_advanced.pdf'
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved: {fname}')


if __name__ == '__main__':
    main()
