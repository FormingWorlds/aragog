"""Boundary condition regime verification for the Aragog entropy solver.

Produces a 3-panel figure (1 row, 3 columns) verifying three distinct
BC regimes using the constant-property entropy model:

(a) Dirichlet BCs: conduction vs convection
    Two T(r) profiles on the same panel. Conduction-only (dashed) shows
    the classical A/r + B profile. Convection (solid, viscosity=1e12,
    alpha=1e-5) shows a well-mixed interior with thin boundary layers.

(b) Neumann BCs: prescribed-flux energy conservation
    Thermal energy E(t) vs time against the expected linear relation
    E_0 - F*A*t. Insulating core, constant surface flux of 100 W/m^2.

(c) Mixed BCs: Dirichlet inner + grey-body outer
    T(r) profiles at multiple times showing surface cooling while the
    CMB temperature stays fixed at 4000 K.

All tests use constant-property physics: T = T_ref * exp((S-S_ref)/Cp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import solve_ivp

# Import constant-property utilities from the test module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'tests'))
from test_entropy_advanced import (
    RHO, CP, K_COND,
    R_INNER, R_OUTER, D_SHELL, G, SECS_PER_YEAR,
    T_to_S, S_to_T, analytical_T, ConstPropState, make_const_mesh,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / 'output' / 'entropy_verification'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match existing plot style from run_advanced_verification.py
plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 9, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'lines.linewidth': 2.0, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})

# Colors
C_COND = '#332288'     # dark blue-violet for conduction
C_CONV = '#EE6677'     # red for convection
C_ENERGY = '#4477AA'   # blue for energy
C_EXPECT = 'k'         # black for expected/analytical


def run_dirichlet_test():
    """Panel (a): Dirichlet BCs with conduction-only vs convection.

    Returns
    -------
    r_km : ndarray
        Radial positions in km.
    T_cond : ndarray
        Steady-state T(r) for conduction-only.
    T_conv : ndarray
        Steady-state T(r) with convection (viscosity=1e12, alpha=1e-5).
    r_an_km : ndarray
        Fine radial grid in km for analytical reference.
    T_an : ndarray
        Analytical conduction profile A/r + B.
    """
    print('  (a) Dirichlet BCs: conduction vs convection...')
    N = 100
    T_inner, T_outer = 4000.0, 1500.0
    S_in_bc = T_to_S(T_inner)
    S_out_bc = T_to_S(T_outer)
    relax = 1e6  # strong relaxation rate [1/yr]

    # Start from linear T (not the analytical solution, to test convergence)
    S_init = T_to_S(np.linspace(T_inner, T_outer, N))

    # -- Conduction-only --
    mesh_cond = make_const_mesh(N)
    state_cond = ConstPropState(mesh_cond, convection=False)

    def rhs_cond(t, S, _s=state_cond, _m=mesh_cond,
                 _si=S_in_bc, _so=S_out_bc):
        _s.dSdt(t, S)
        _s.heat_flux[0] = 0.0
        _s.heat_flux[-1] = 0.0
        ef = _s.heat_flux * _m.area
        cap = RHO * S_to_T(S) * _m.volume
        dsdt = -np.diff(ef) / cap * SECS_PER_YEAR
        dsdt[0] += relax * (_si - S[0])
        dsdt[-1] += relax * (_so - S[-1])
        return dsdt

    sol_cond = solve_ivp(rhs_cond, (0, 1e8), S_init, method='BDF',
                         atol=0.01, rtol=1e-8)
    T_cond = S_to_T(sol_cond.y[:, -1])

    # -- Convection (viscosity=1e12, alpha=1e-5) --
    mesh_conv = make_const_mesh(N)
    mesh_conv.mixing_length = np.maximum(
        np.minimum(mesh_conv.r_basic - R_INNER, R_OUTER - mesh_conv.r_basic),
        1.0)
    state_conv = ConstPropState(mesh_conv, viscosity=1e12, convection=True,
                                alpha=1e-5)

    def rhs_conv(t, S, _s=state_conv, _m=mesh_conv,
                 _si=S_in_bc, _so=S_out_bc):
        _s.dSdt(t, S)
        _s.heat_flux[0] = 0.0
        _s.heat_flux[-1] = 0.0
        ef = _s.heat_flux * _m.area
        cap = RHO * S_to_T(S) * _m.volume
        dsdt = -np.diff(ef) / cap * SECS_PER_YEAR
        dsdt[0] += relax * (_si - S[0])
        dsdt[-1] += relax * (_so - S[-1])
        return dsdt

    sol_conv = solve_ivp(rhs_conv, (0, 1e6), S_init, method='BDF',
                         atol=0.01, rtol=1e-6)
    T_conv = S_to_T(sol_conv.y[:, -1])

    # Analytical conduction reference on a fine grid
    r_an = np.linspace(R_INNER, R_OUTER, 300)
    T_an = analytical_T(r_an, T_inner, T_outer)

    return (mesh_cond.r_stag / 1e3, T_cond, T_conv,
            r_an / 1e3, T_an)


def run_neumann_test():
    """Panel (b): prescribed-flux energy conservation.

    Returns
    -------
    times_yr : ndarray
        Time samples in years.
    E_numerical : ndarray
        Thermal energy E(t) from the solver at each time.
    E_expected : ndarray
        Expected E(t) = E_0 - F * A_surf * t.
    rel_residual : float
        Relative residual |E_num - E_exp| / |E_0 - E_exp| at final time.
    """
    print('  (b) Neumann BCs: energy conservation...')
    N = 50
    mesh = make_const_mesh(N)
    state = ConstPropState(mesh, convection=False)

    T_uniform = 3000.0
    S_init = T_to_S(np.full(N, T_uniform))
    F_prescribed = 100.0  # W/m^2 outward
    A_surf = mesh.area[-1]

    def rhs(t, S, _s=state, _m=mesh):
        _s.dSdt(t, S)
        _s.heat_flux[0] = 0.0        # insulating core
        _s.heat_flux[-1] = F_prescribed
        ef = _s.heat_flux * _m.area
        cap = RHO * S_to_T(S) * _m.volume
        return -np.diff(ef) / cap * SECS_PER_YEAR

    t_end = 1e4  # yr
    sol = solve_ivp(rhs, (0, t_end), S_init, method='BDF',
                    atol=0.01, rtol=1e-8, dense_output=True)

    # Sample E(t) at uniform times
    n_samples = 200
    times_yr = np.linspace(0, t_end, n_samples)
    E_numerical = np.zeros(n_samples)
    for i, t in enumerate(times_yr):
        S_t = sol.sol(t)
        T_t = S_to_T(S_t)
        E_numerical[i] = np.sum(RHO * CP * T_t * mesh.volume)

    E_0 = E_numerical[0]
    E_expected = E_0 - F_prescribed * A_surf * times_yr * SECS_PER_YEAR

    dE_total = E_expected[-1] - E_0  # total expected energy loss (negative)
    rel_residual = abs(E_numerical[-1] - E_expected[-1]) / abs(dE_total)

    return times_yr, E_numerical, E_expected, rel_residual


def run_mixed_test():
    """Panel (c): Dirichlet inner + grey-body outer.

    Returns
    -------
    r_km : ndarray
        Radial positions in km.
    profiles : dict
        {t_yr: T_array} for selected snapshot times.
    """
    print('  (c) Mixed BCs: Dirichlet inner + grey-body outer...')
    N = 50
    mesh = make_const_mesh(N)
    mesh.mixing_length = np.maximum(
        np.minimum(mesh.r_basic - R_INNER, R_OUTER - mesh.r_basic), 1.0)
    state = ConstPropState(mesh, viscosity=1e15, convection=True, alpha=1e-5)

    T_inner = 4000.0
    S_inner = T_to_S(T_inner)
    T_eq = 255.0
    emissivity = 1.0
    relax = 1e6

    # Start from uniform T midway between inner and outer
    S_init = T_to_S(np.full(N, 0.5 * (T_inner + 1500.0)))

    def rhs(t, S, _s=state, _m=mesh, _si=S_inner):
        _s.dSdt(t, S)
        # Grey-body at surface
        T_surf = S_to_T(S[-1])
        _s.heat_flux[-1] = (
            emissivity * Stefan_Boltzmann * (T_surf**4 - T_eq**4))
        # Zero internal flux at CMB; relaxation pins S
        _s.heat_flux[0] = 0.0
        ef = _s.heat_flux * _m.area
        cap = RHO * S_to_T(S) * _m.volume
        dsdt = -np.diff(ef) / cap * SECS_PER_YEAR
        # Relaxation BC at inner boundary only
        dsdt[0] += relax * (_si - S[0])
        return dsdt

    t_end = 1e6  # yr
    sol = solve_ivp(rhs, (0, t_end), S_init, method='BDF',
                    atol=0.01, rtol=1e-6, dense_output=True)

    # Extract profiles at selected times
    snapshot_times = [0, 1e3, 1e4, 1e5, 1e6]
    profiles = {}
    for t in snapshot_times:
        if t <= sol.t[-1]:
            T_t = S_to_T(sol.sol(t))
            profiles[t] = T_t

    return mesh.r_stag / 1e3, profiles


def main():
    """Generate the 3-panel BC verification figure."""
    print('Generating BC regime verification plots...')

    # Run all three tests
    (r_km, T_cond, T_conv, r_an_km, T_an) = run_dirichlet_test()
    (times_yr, E_num, E_exp, rel_res) = run_neumann_test()
    (r_km_mixed, profiles) = run_mixed_test()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # -- Panel (a): Dirichlet BCs ------------------------------------------
    ax = axes[0]
    ax.plot(r_an_km, T_an, '--', color=C_COND, linewidth=2.5,
            label='Conduction ($A/r + B$)')
    ax.plot(r_km, T_cond, '-', color=C_COND, linewidth=1.5, alpha=0.6,
            label='Conduction (numerical)')
    ax.plot(r_km, T_conv, '-', color=C_CONV, linewidth=2.0,
            label=r'Convection ($\eta = 10^{12}$, $\alpha = 10^{-5}$)')
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(a) Dirichlet BCs')
    ax.legend(fontsize=8, loc='upper right')

    # Annotate the well-mixed interior
    n = len(T_conv)
    T_int = T_conv[n // 10: 9 * n // 10]
    T_spread = np.max(T_int) - np.min(T_int)
    ax.text(0.05, 0.05,
            f'Interior $\\Delta T$ = {T_spread:.0f} K\n'
            f'(well-mixed: {T_spread/(4000-1500)*100:.0f}% of $\\Delta T_{{BC}}$)',
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # -- Panel (b): Neumann BCs --------------------------------------------
    ax = axes[1]
    ax.plot(times_yr, E_num, '-', color=C_ENERGY, linewidth=2.0,
            label='$E(t)$ (numerical)')
    ax.plot(times_yr, E_exp, '--', color=C_EXPECT, linewidth=2.0,
            label='$E_0 - F \\cdot A \\cdot t$')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Thermal energy [J]')
    ax.set_title('(b) Neumann BCs')
    ax.legend(fontsize=8, loc='upper right')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Annotation with residual
    ax.text(0.05, 0.05,
            f'Residual: {rel_res:.2%}',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round',
                      facecolor='lightgreen' if rel_res < 0.05 else 'lightyellow',
                      alpha=0.8))

    # -- Panel (c): Mixed BCs ----------------------------------------------
    ax = axes[2]
    n_prof = len(profiles)
    colors_mixed = plt.cm.plasma(np.linspace(0.1, 0.9, n_prof))
    for i, (t, T_t) in enumerate(profiles.items()):
        if t == 0:
            lbl = 't = 0'
        elif t < 1e3:
            lbl = f't = {t:.0f} yr'
        elif t < 1e6:
            lbl = f't = {t/1e3:.0f} kyr'
        else:
            lbl = f't = {t/1e6:.0f} Myr'
        ax.plot(r_km_mixed, T_t, '-', color=colors_mixed[i],
                linewidth=2.0, label=lbl)
    # Mark the fixed CMB temperature
    ax.axhline(4000.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(r_km_mixed[0] + 20, 4000 + 40, '$T_{\\mathrm{CMB}}$ = 4000 K',
            fontsize=8, color='gray')
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(c) Mixed BCs (Dirichlet inner + grey-body outer)')
    ax.legend(fontsize=8, loc='center right')

    fig.suptitle('BC regime verification: constant-property entropy solver',
                 fontsize=15, y=1.01)
    fig.tight_layout()

    fname = OUT_DIR / 'verify_bc_regimes.pdf'
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f'\nSaved: {fname}')

    # Print summary
    print('\n' + '=' * 60)
    print('Summary:')
    # (a) Conduction vs analytical
    T_an_on_mesh = analytical_T(np.array(r_km) * 1e3, 4000.0, 1500.0)
    cond_err = np.max(np.abs(T_cond - T_an_on_mesh))
    n = len(T_conv)
    T_int = T_conv[n // 10: 9 * n // 10]
    T_spread = np.max(T_int) - np.min(T_int)
    print(f'  (a) Conduction max error vs analytical: {cond_err:.2f} K')
    print(f'      Convective interior DT spread: {T_spread:.0f} K '
          f'({T_spread / 2500 * 100:.1f}% of DT_BC)')
    # (b) Energy conservation
    print(f'  (b) Energy residual: {rel_res:.2%}')
    # (c) Mixed BCs
    t_keys = sorted(profiles.keys())
    T_cmb_final = profiles[t_keys[-1]][0]
    T_surf_final = profiles[t_keys[-1]][-1]
    print(f'  (c) Final T_CMB = {T_cmb_final:.0f} K (target: 4000 K, '
          f'drift: {abs(T_cmb_final - 4000):.0f} K)')
    print(f'      Final T_surf = {T_surf_final:.0f} K '
          f'(cooled from initial)')
    print('=' * 60)


if __name__ == '__main__':
    main()
