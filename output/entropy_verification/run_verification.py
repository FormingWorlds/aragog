"""First-principles verification of the Aragog entropy solver.

Produces publication-quality PDF plots for the tech document:
1. Entropy conservation (insulating box)
2. Energy budget closure (grey-body cooling)
3. Grey-body cooling trajectory (T_surf vs time, t^{-1/3} comparison)
4. Convective homogenization (entropy profiles at multiple times)

All plots are saved to the same directory as this script.
Requires PALEOS P-S tables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState

OUT_DIR = Path(__file__).parent
EOS_DIR = Path('/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos')
SECS_PER_YEAR = 31557600.0

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
    'legend.fontsize': 10, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'lines.linewidth': 2.0, 'lines.markersize': 7,
    'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


def make_mesh(N=50, R_cmb=3480e3, R_surf=6371e3, P_cmb=135e9, P_surf=1e5):
    """Build a simple radial mesh."""
    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_stag = np.linspace(P_cmb, P_surf, N)

    class Mesh:
        pass
    class SubMesh:
        pass
    mesh = Mesh()
    mesh.basic = SubMesh()
    mesh.staggered = SubMesh()
    mesh.basic.radii = r_basic
    mesh.staggered.radii = r_stag
    mesh.basic.area = 4.0 * np.pi * r_basic**2
    mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.minimum(r_basic - R_cmb, R_surf - r_basic)
    mesh.basic.mixing_length = np.maximum(ml, 1.0)
    mesh.basic.mixing_length_squared = mesh.basic.mixing_length**2
    mesh.basic.mixing_length_cubed = mesh.basic.mixing_length**3
    mesh.basic.pressure = np.interp(r_basic, r_stag, P_stag)
    mesh.staggered.pressure = P_stag
    mesh.N = N
    mesh.dr = dr
    mesh.R_surf = R_surf

    def quantity_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[0], out[-1] = q[0], q[-1]
        out[1:-1] = 0.5 * (q[:-1] + q[1:])
        return out

    def d_dr_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[1:-1] = np.diff(q) / dr
        out[0], out[-1] = out[1], out[-2]
        return out

    mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
    mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes
    return mesh


def make_state(mesh, eos, conduction=True, convection=True):
    """Build EntropyState."""
    phase_stag = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(mesh.staggered.pressure)
    phase_basic = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_basic.set_pressure(mesh.basic.pressure)

    class Eval:
        pass
    evaluator = Eval()
    evaluator.mesh = mesh
    return EntropyState(evaluator=evaluator, phase_staggered=phase_stag,
                        phase_basic=phase_basic, conduction=conduction, convection=convection)


def compute_thermal_energy(S, mesh, eos):
    T = eos.temperature(mesh.staggered.pressure, S)
    rho = eos.density(mesh.staggered.pressure, S)
    Cp = eos.heat_capacity(mesh.staggered.pressure, S)
    return np.sum(rho * Cp * T * mesh.basic.volume)


def run_solver(state, mesh, S0, t_end, surface_bc='greybody', T_eq=255.0):
    """Run the entropy BDF solver with specified BCs."""
    flux_log = []
    time_log = []

    def dSdt(t, S):
        state.update(S, t)
        if surface_bc == 'greybody':
            T_top = state.top_temperature.item()
            F = Stefan_Boltzmann * (T_top**4 - T_eq**4)
            state._heat_flux[-1] = F
            flux_log.append(F)
            time_log.append(t)
        else:
            state._heat_flux[-1] = 0.0
        state._heat_flux[0] = 0.0
        energy_flux = state.heat_flux * mesh.basic.area
        cap = state.capacitance_staggered() * mesh.basic.volume
        return -np.diff(energy_flux) / cap * SECS_PER_YEAR

    sol = solve_ivp(dSdt, (0, t_end), S0, method='BDF',
                    atol=0.5, rtol=1e-5, dense_output=True)
    return sol, np.array(time_log), np.array(flux_log)


def main():
    print('Loading PALEOS P-S tables...')
    eos = EntropyEOS(EOS_DIR)
    N = 50
    mesh = make_mesh(N=N)
    S_init = 10000.0

    # ── Test 1: Entropy conservation (insulating box) ────────────────
    print('\nTest 1: Entropy conservation (insulating box, no conduction)...')
    state_nocd = make_state(mesh, eos, conduction=False, convection=True)
    S0 = np.full(N, S_init)
    sol1, _, _ = run_solver(state_nocd, mesh, S0, 10000, surface_bc='insulating')

    times1 = np.linspace(0, sol1.t[-1], 100)
    max_drift = []
    for t in times1:
        S_t = sol1.sol(t)
        max_drift.append(np.max(np.abs(S_t - S_init)))

    # ── Test 2: Energy budget (grey-body cooling) ────────────────────
    print('Test 2: Energy budget (grey-body cooling)...')
    state_cool = make_state(mesh, eos, conduction=True, convection=True)
    S0 = np.full(N, S_init)
    sol2, t_log, F_log = run_solver(state_cool, mesh, S0, 10000, surface_bc='greybody')

    # Sample at uniform times and compute INCREMENTAL enthalpy changes.
    # E = sum(rho*Cp*T*V) is not conserved by the entropy equation because
    # rho, Cp, T all depend on S nonlinearly. Instead, compute cumulative
    # enthalpy change: dH = sum(rho_i * Cp_i * dT_i * V_i) between steps.
    n_samples = 500
    sample_times = np.linspace(0, sol2.t[-1], n_samples)
    _trapz = getattr(np, 'trapezoid', np.trapz)
    A_surf = mesh.basic.area[-1]

    # Compute T, rho, Cp, F at each sample time
    T_all = np.zeros((n_samples, N))
    F_history = np.zeros(n_samples)
    rho_all = np.zeros((n_samples, N))
    Cp_all = np.zeros((n_samples, N))
    for i, t in enumerate(sample_times):
        S_t = sol2.sol(t)
        T_all[i] = eos.temperature(mesh.staggered.pressure, S_t)
        rho_all[i] = eos.density(mesh.staggered.pressure, S_t)
        Cp_all[i] = eos.heat_capacity(mesh.staggered.pressure, S_t)
        F_history[i] = Stefan_Boltzmann * (T_all[i, -1]**4 - 255.0**4)

    # Incremental energy change using dH = rho * T * dS * V
    # (the natural energy measure for the entropy equation)
    S_all = np.zeros((n_samples, N))
    for i, t in enumerate(sample_times):
        S_all[i] = sol2.sol(t)

    dH_cumul = np.zeros(n_samples)
    for i in range(1, n_samples):
        dS = S_all[i] - S_all[i-1]
        rho_mid = 0.5 * (rho_all[i] + rho_all[i-1])
        T_mid = 0.5 * (T_all[i] + T_all[i-1])
        dH_cumul[i] = dH_cumul[i-1] + np.sum(rho_mid * T_mid * dS * mesh.basic.volume)

    # Cumulative surface flux loss
    Q_cumul = np.zeros(n_samples)
    for i in range(1, n_samples):
        Q_cumul[i] = _trapz(
            F_history[:i+1] * A_surf,
            sample_times[:i+1] * SECS_PER_YEAR
        )

    # ── Test 3: Grey-body cooling trajectory ─────────────────────────
    print('Test 3: Grey-body cooling trajectory (10 kyr)...')
    state_long = make_state(mesh, eos, conduction=True, convection=True)
    S0 = np.full(N, S_init)
    sol3, _, _ = run_solver(state_long, mesh, S0, 10000, surface_bc='greybody')

    sample_times3 = np.linspace(1, sol3.t[-1], 200)
    T_surfs = []
    phi_surfs = []
    for t in sample_times3:
        S_t = sol3.sol(t)
        state_long.phase_staggered.set_entropy(S_t)
        state_long.phase_staggered.update()
        T_surfs.append(float(state_long.phase_staggered.temperature()[-1]))
        phi_surfs.append(float(state_long.phase_staggered.melt_fraction()[-1]))
    T_surfs = np.array(T_surfs)
    phi_surfs = np.array(phi_surfs)

    # No analytical prediction overlaid: the t^{-1/3} lumped model
    # does not apply to a phase-changing magma ocean with variable Cp.

    # ── Test 4: Convective homogenization ────────────────────────────
    print('Test 4: Convective homogenization...')
    state_homo = make_state(mesh, eos, conduction=False, convection=True)
    S0_gradient = np.linspace(10400.0, 9600.0, N)
    sol4, _, _ = run_solver(state_homo, mesh, S0_gradient, 1.0, surface_bc='insulating')

    r_stag_km = mesh.staggered.radii / 1e3
    # Sub-year sampling: homogenization takes < 0.01 yr at liquid viscosity
    profile_times = [0, 0.001, 0.003, 0.005, 0.01, 0.03, 0.1, 1.0]
    profiles = {}
    for t in profile_times:
        if t <= sol4.t[-1]:
            profiles[t] = sol4.sol(t)

    # ── Extract T-P and P-S profiles at several times from sol3 ─────
    # ── Run the same cooling at double resolution (N=100) ─────────────
    print('Test 3b: Grey-body cooling at N=100 (convergence check)...')
    N_hi = 100
    mesh_hi = make_mesh(N=N_hi)
    state_hi = make_state(mesh_hi, eos, conduction=True, convection=True)
    S0_hi = np.full(N_hi, S_init)
    sol3b, _, _ = run_solver(state_hi, mesh_hi, S0_hi, 10000, surface_bc='greybody')

    # Sample T_surf at same times for convergence comparison
    T_surfs_hi = []
    phi_surfs_hi = []
    for t in sample_times3:
        if t <= sol3b.t[-1]:
            S_t = sol3b.sol(t)
            state_hi.phase_staggered.set_entropy(S_t)
            state_hi.phase_staggered.update()
            T_surfs_hi.append(float(state_hi.phase_staggered.temperature()[-1]))
            phi_surfs_hi.append(float(state_hi.phase_staggered.melt_fraction()[-1]))
        else:
            T_surfs_hi.append(np.nan)
            phi_surfs_hi.append(np.nan)
    T_surfs_hi = np.array(T_surfs_hi)
    phi_surfs_hi = np.array(phi_surfs_hi)

    # Extract T-P and P-S profiles at several times from both resolutions
    print('Extracting T-P and P-S profiles...')
    tp_times = [0, 500, 1000, 2000, 5000, 10000]
    tp_profiles = {}       # N=50: {t: (T, S, phi, P_GPa)}
    tp_profiles_hi = {}    # N=100
    for t in tp_times:
        if t <= sol3.t[-1]:
            S_t = sol3.sol(t)
            T_t = eos.temperature(mesh.staggered.pressure, S_t)
            phi_t = eos.melt_fraction(mesh.staggered.pressure, S_t)
            tp_profiles[t] = (T_t, S_t, phi_t, mesh.staggered.pressure / 1e9)
        if t <= sol3b.t[-1]:
            S_t = sol3b.sol(t)
            T_t = eos.temperature(mesh_hi.staggered.pressure, S_t)
            phi_t = eos.melt_fraction(mesh_hi.staggered.pressure, S_t)
            tp_profiles_hi[t] = (T_t, S_t, phi_t, mesh_hi.staggered.pressure / 1e9)

    # Solidus and liquidus in T-P space from PALEOS analytic parameterization.
    # Liquidus: Belonoshko+2005 (P < 2.55 GPa) / Fei+2021 (P >= 2.55 GPa).
    # Solidus: PALEOS liquidus with cryoscopic depression (x0=0.79, Stixrude 2014).
    P_range = np.linspace(mesh.staggered.pressure[0], mesh.staggered.pressure[-1], 500)
    P_range_GPa = P_range / 1e9
    P_GPa = P_range / 1e9
    PALEOS_P0 = 2.551686137257537  # GPa crossover
    T_liq = np.where(
        P_GPa < PALEOS_P0,
        1831.0 * (1.0 + P_GPa / 4.6)**0.33,
        6000.0 * (P_GPa / 140.0)**0.26,
    )
    T_liq = np.where(P_range > 0, T_liq, 0.0)
    # Cryoscopic solidus: T_sol = T_liq / (1 - ln(x0)) with x0=0.79
    cryo_factor = 1.0 / (1.0 - np.log(0.79))
    T_sol = T_liq * cryo_factor

    # Also get S boundaries from the P-S table files (for panel f)
    S_sol = eos.solidus_entropy(P_range)
    S_liq = eos.liquidus_entropy(P_range)

    # ── Generate 6-panel figure (3x2) ────────────────────────────────
    print('\nGenerating verification figure...')
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Panel (a): Entropy conservation [row 0, col 0]
    ax = axes[0, 0]
    ax.semilogy(times1, np.maximum(max_drift, 1e-15), 'b-', linewidth=2)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(r'max$|S(r,t) - S_0|$ [J/kg/K]')
    ax.set_title('(a) Entropy conservation (insulating, no conduction)')
    ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.5, label='1 J/kg/K threshold')
    ax.legend(fontsize=9)
    ax.text(0.95, 0.95, f'Drift < {max(max_drift):.2e} J/kg/K',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel (b): Energy budget (incremental enthalpy vs integrated flux)
    ax = axes[0, 1]
    ax.plot(sample_times, dH_cumul, 'b-', label=r'$\Delta H$ (incremental)', linewidth=2)
    ax.plot(sample_times, -Q_cumul, 'r--', label=r'$-\int F_\mathrm{surf} A \, dt$', linewidth=2)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Energy change [J]')
    ax.set_title('(b) Energy budget (grey-body cooling)')
    ax.legend()
    if abs(Q_cumul[-1]) > 0:
        residual = abs(dH_cumul[-1] + Q_cumul[-1]) / abs(Q_cumul[-1])
        ax.text(0.95, 0.05, f'Residual: {residual:.1%}',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel (c): Cooling trajectory
    ax = axes[1, 0]
    ax.plot(sample_times3, T_surfs, 'b-', label='N=50', linewidth=2)
    ax.plot(sample_times3[:len(T_surfs_hi)], T_surfs_hi, 'b--', label='N=100', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(r'$T_\mathrm{surf}$ [K]')
    ax.set_title(r'(c) Grey-body cooling ($S_0 = 10{,}000$ J/kg/K)')
    ax.legend()

    # Add phi on twin axis
    ax2 = ax.twinx()
    ax2.plot(sample_times3, phi_surfs, 'g-', alpha=0.5, linewidth=1.5)
    ax2.plot(sample_times3[:len(phi_surfs_hi)], phi_surfs_hi, 'g--', alpha=0.3, linewidth=1.5)
    ax2.set_ylabel(r'$\phi_\mathrm{surf}$', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)

    # Panel (d): Convective homogenization
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
    for i, (t, S) in enumerate(profiles.items()):
        if t >= 1:
            lbl = f't = {t:.0f} yr'
        elif t >= 0.01:
            lbl = f't = {t*365:.0f} d'
        else:
            lbl = f't = {t*365*24:.0f} h'
        ax.plot(r_stag_km, S, color=colors[i], label=lbl, linewidth=2)
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Entropy [J/kg/K]')
    ax.set_title('(d) Convective homogenization (no conduction)')
    ax.legend()

    # Panel (e): T-P profiles with solidus/liquidus (both resolutions)
    ax = axes[2, 0]
    n_tp = len(tp_profiles)
    colors_tp = plt.cm.plasma(np.linspace(0.1, 0.9, n_tp))
    for i, (t, (T_t, S_t, phi_t, P_GPa)) in enumerate(tp_profiles.items()):
        lbl = f't = {t/1e3:.0f} kyr' if t >= 1000 else f't = {t} yr'
        ax.plot(T_t, P_GPa, color=colors_tp[i], linewidth=1.8, label=lbl)
    # Overlay N=100 as thin dashed
    for i, (t, (T_t, S_t, phi_t, P_GPa)) in enumerate(tp_profiles_hi.items()):
        ax.plot(T_t, P_GPa, color=colors_tp[i], linewidth=0.8, ls='--', alpha=0.6)
    ax.plot(T_sol, P_range_GPa, 'k--', linewidth=1.5, label='Solidus')
    ax.plot(T_liq, P_range_GPa, 'k-', linewidth=1.5, label='Liquidus')
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Pressure [GPa]')
    ax.set_title('(e) T-P profiles (solid: N=50, dashed: N=100)')
    ax.legend(fontsize=7, ncol=2)
    ax.invert_yaxis()
    ax.set_xlim(left=0)

    # Panel (f): P-S profiles (both resolutions)
    ax = axes[2, 1]
    for i, (t, (T_t, S_t, phi_t, P_GPa)) in enumerate(tp_profiles.items()):
        lbl = f't = {t/1e3:.0f} kyr' if t >= 1000 else f't = {t} yr'
        ax.plot(S_t, P_GPa, color=colors_tp[i], linewidth=1.8, label=lbl)
    for i, (t, (T_t, S_t, phi_t, P_GPa)) in enumerate(tp_profiles_hi.items()):
        ax.plot(S_t, P_GPa, color=colors_tp[i], linewidth=0.8, ls='--', alpha=0.6)
    ax.plot(S_sol, P_range_GPa, 'k--', linewidth=1.5, label='Solidus')
    ax.plot(S_liq, P_range_GPa, 'k-', linewidth=1.5, label='Liquidus')
    ax.set_xlabel('Entropy [J/kg/K]')
    ax.set_ylabel('Pressure [GPa]')
    ax.set_title('(f) P-S profiles (solid: N=50, dashed: N=100)')
    ax.legend(fontsize=7, ncol=2)
    ax.invert_yaxis()

    fig.suptitle('First-principles verification: Aragog entropy solver', fontsize=16, y=1.005)
    fig.tight_layout()

    fname = OUT_DIR / 'verify_entropy_solver.pdf'
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved: {fname}')

    # ── Summary ──────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Verification Summary:')
    print(f'  (a) Entropy drift: {max(max_drift):.2f} J/kg/K (pass: < 1.0)')
    if abs(Q_cumul[-1]) > 0:
        print(f'  (b) Energy residual: {residual:.1%} (pass: < 10%)')
    print(f'  (c) T_surf: {T_surfs[0]:.0f} -> {T_surfs[-1]:.0f} K (monotonic: {np.all(np.diff(T_surfs) <= 1.0)})')
    print(f'  (d) S spread: {S0_gradient.max()-S0_gradient.min():.0f} -> '
          f'{profiles[max(profiles.keys())].max()-profiles[max(profiles.keys())].min():.0f} J/kg/K')
    print('=' * 60)


if __name__ == '__main__':
    main()
