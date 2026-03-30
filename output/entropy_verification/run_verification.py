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
    S_init = 3200.0

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
    sol2, t_log, F_log = run_solver(state_cool, mesh, S0, 2000, surface_bc='greybody')

    E_start = compute_thermal_energy(S0, mesh, eos)
    sample_times = np.linspace(0, sol2.t[-1], 50)
    E_history = []
    for t in sample_times:
        S_t = sol2.sol(t)
        E_history.append(compute_thermal_energy(S_t, mesh, eos))
    E_history = np.array(E_history)

    # Integrated surface flux
    idx = np.argsort(t_log)
    _trapz = getattr(np, 'trapezoid', np.trapz)
    Q_cumul = np.zeros(len(sample_times))
    for i, t in enumerate(sample_times):
        mask = t_log[idx] <= t
        if np.sum(mask) > 1:
            Q_cumul[i] = _trapz(
                F_log[idx][mask] * mesh.basic.area[-1],
                t_log[idx][mask] * SECS_PER_YEAR
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

    # Analytical t^{-1/3} reference (lumped model)
    T0 = T_surfs[0]
    M_mantle = 4.0 / 3.0 * np.pi * (6371e3**3 - 3480e3**3) * 4000.0
    Cp_eff = 1200.0
    A_surf = 4.0 * np.pi * 6371e3**2
    tau_rad = M_mantle * Cp_eff / (3 * Stefan_Boltzmann * A_surf * T0**3)
    T_analytical = T0 * (1 + sample_times3 * SECS_PER_YEAR / tau_rad)**(-1.0/3.0)

    # ── Test 4: Convective homogenization ────────────────────────────
    print('Test 4: Convective homogenization...')
    state_homo = make_state(mesh, eos, conduction=False, convection=True)
    S0_gradient = np.linspace(3400.0, 3000.0, N)
    sol4, _, _ = run_solver(state_homo, mesh, S0_gradient, 500, surface_bc='insulating')

    r_stag_km = mesh.staggered.radii / 1e3
    profile_times = [0, 10, 50, 100, 500]
    profiles = {}
    for t in profile_times:
        if t <= sol4.t[-1]:
            profiles[t] = sol4.sol(t)

    # ── Generate 4-panel figure ──────────────────────────────────────
    print('\nGenerating verification figure...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel (a): Entropy conservation
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

    # Panel (b): Energy budget
    ax = axes[0, 1]
    dE = E_history - E_start
    ax.plot(sample_times, dE, 'b-', label=r'$\Delta E_\mathrm{th}$', linewidth=2)
    ax.plot(sample_times, -Q_cumul, 'r--', label=r'$-\int F_\mathrm{surf} A \, dt$', linewidth=2)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Energy change [J]')
    ax.set_title('(b) Energy budget (grey-body cooling)')
    ax.legend()
    if abs(Q_cumul[-1]) > 0:
        residual = abs(dE[-1] + Q_cumul[-1]) / abs(Q_cumul[-1])
        ax.text(0.95, 0.05, f'Residual: {residual:.1%}',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel (c): Cooling trajectory
    ax = axes[1, 0]
    ax.plot(sample_times3, T_surfs, 'b-', label='Aragog (entropy)', linewidth=2)
    ax.plot(sample_times3, T_analytical, 'r--', label=r'$T_0 (1 + t/\tau)^{-1/3}$', linewidth=1.5)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(r'$T_\mathrm{surf}$ [K]')
    ax.set_title(r'(c) Grey-body cooling ($S_0 = 3200$ J/kg/K)')
    ax.legend()

    # Add phi on twin axis
    ax2 = ax.twinx()
    ax2.plot(sample_times3, phi_surfs, 'g-', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel(r'$\phi_\mathrm{surf}$', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)

    # Panel (d): Convective homogenization
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
    for i, (t, S) in enumerate(profiles.items()):
        ax.plot(r_stag_km, S, color=colors[i], label=f't = {t} yr', linewidth=2)
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Entropy [J/kg/K]')
    ax.set_title('(d) Convective homogenization (no conduction)')
    ax.legend()

    fig.suptitle('First-principles verification: Aragog entropy solver', fontsize=16, y=1.01)
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
