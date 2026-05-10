"""IC sensitivity and heating verification for the Aragog entropy solver.

Produces a 2x2 figure with four verification panels:

(a) IC sweep: T_surf(t) for different initial entropy values
    S0 = 2500 (mostly solid), 3200 (SPIDER default), 5000 (partially liquid),
    10000 (fully liquid). Grey-body surface, insulating core, N=30.

(b) IC sweep: Phi_global(t) for the same runs.
    Volume-weighted melt fraction across all cells.

(c) Radiogenic heating in an insulating box.
    S0=3200, zero-flux BCs, H=1e-11 W/kg, conduction+convection.
    Mean entropy <S>(t) vs time over 1 Myr. Overlaid: expected dS/dt = H/T.

(d) Core cooling: F_CMB and T_CMB vs time.
    S0=3200, grey-body surface, Bower+2018 core cooling BC.
    F_CMB on left y-axis (blue), T_CMB on right y-axis (red).

All plots use PALEOS P-S tables.
Output: Aragog/output/entropy_verification/
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

# Add Aragog src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState

OUT_DIR = Path(__file__).resolve().parent.parent.parent / 'output' / 'entropy_verification'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EOS_DIR = Path(os.environ.get(
    'ARAGOG_TEST_EOS_DIR',
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
))
SECS_PER_YEAR = 31557600.0

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
    'legend.fontsize': 10, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'lines.linewidth': 2.0, 'lines.markersize': 7,
    'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


# ---------------------------------------------------------------------------
# Shared infrastructure (matches run_verification.py and test_entropy_verification.py)
# ---------------------------------------------------------------------------

def make_mesh(N=30, R_cmb=3480e3, R_surf=6371e3, P_cmb=135e9, P_surf=1e5):
    """Build a simple radial mesh for verification tests."""
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
    mesh.R_cmb = R_cmb

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
    """Build EntropyState from mesh and EOS."""
    phase_stag = EntropyPhaseEvaluator(
        entropy_eos=eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(mesh.staggered.pressure)
    phase_basic = EntropyPhaseEvaluator(
        entropy_eos=eos, gravitational_acceleration=10.0)
    phase_basic.set_pressure(mesh.basic.pressure)

    class Eval:
        pass
    evaluator = Eval()
    evaluator.mesh = mesh
    return EntropyState(
        evaluator=evaluator, phase_staggered=phase_stag,
        phase_basic=phase_basic, conduction=conduction, convection=convection)


# ---------------------------------------------------------------------------
# Panel (a) and (b): IC sweep
# ---------------------------------------------------------------------------

def run_ic_sweep(eos):
    """Run grey-body cooling from 4 different initial entropy values.

    Returns
    -------
    results : dict
        {S0_val: {'times': array, 'T_surf': array, 'phi_global': array}}
    """
    S0_values = [2500.0, 3200.0, 5000.0, 10000.0]
    t_end = 10000.0  # 10 kyr
    n_samples = 80

    results = {}
    for S0_val in S0_values:
        print(f'  IC sweep: S0 = {S0_val:.0f} J/kg/K ...')
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)
        S0 = np.full(N, S0_val)

        def dSdt(t, S, _s=state, _m=mesh):
            _s.update(S, t)
            T_top = _s.top_temperature.item()
            F_surf = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            _s._heat_flux[-1] = F_surf
            _s._heat_flux[0] = 0.0
            energy_flux = _s.heat_flux * _m.basic.area
            cap = _s.capacitance_staggered() * _m.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, t_end), S0, method='BDF',
                        atol=0.5, rtol=1e-5, dense_output=True)
        if sol.status != 0:
            print(f'    WARNING: solver failed for S0={S0_val} '
                  f'(status={sol.status})')

        t_final = min(t_end, sol.t[-1])
        times = np.linspace(0, t_final, n_samples)
        T_surf_arr = np.zeros(n_samples)
        phi_global_arr = np.zeros(n_samples)

        for i, t in enumerate(times):
            S_t = sol.sol(t)
            # Update phase evaluator for T_surf
            state.phase_staggered.set_entropy(S_t)
            state.phase_staggered.update()
            T_surf_arr[i] = float(state.phase_staggered.temperature()[-1])
            # Phi_global: volume-weighted melt fraction
            phi = np.asarray(state.phase_staggered.melt_fraction()).flatten()
            vol = mesh.basic.volume
            phi_global_arr[i] = np.sum(phi * vol) / np.sum(vol)

        results[S0_val] = {
            'times': times,
            'T_surf': T_surf_arr,
            'phi_global': phi_global_arr,
        }
        print(f'    T_surf: {T_surf_arr[0]:.0f} -> {T_surf_arr[-1]:.0f} K '
              f'over {t_final:.0f} yr')

    return results


# ---------------------------------------------------------------------------
# Panel (c): Radiogenic heating
# ---------------------------------------------------------------------------

def run_radiogenic_heating(eos):
    """Insulating box with radiogenic heating: <S>(t).

    Returns
    -------
    times : array
        Sampling times [yr].
    S_mean : array
        Mean entropy at each time [J/kg/K].
    dSdt_expected : float
        Expected dS/dt = H / <T> from initial state [J/kg/K/yr].
    """
    print('  Radiogenic heating: insulating box ...')
    N = 30
    mesh = make_mesh(N=N)
    state = make_state(mesh, eos, conduction=True, convection=True)

    S0 = np.full(N, 3200.0)
    H_rate = 1e-11  # W/kg (Earth-like radiogenic)
    t_end = 1e6  # 1 Myr
    n_samples = 80

    # Expected dS/dt from initial state for reference line
    T_init = eos.temperature(mesh.staggered.pressure, S0)
    T_mean_init = float(np.mean(T_init))
    # dS/dt = H/T in SI (J/kg/K/s) -> convert to per year
    dSdt_expected_per_yr = H_rate / T_mean_init * SECS_PER_YEAR

    def dSdt(t, S, _s=state, _m=mesh, _eos=eos):
        _s.update(S, t)
        # Zero-flux BCs (insulating box)
        _s._heat_flux[-1] = 0.0
        _s._heat_flux[0] = 0.0
        energy_flux = _s.heat_flux * _m.basic.area
        cap = _s.capacitance_staggered() * _m.basic.volume
        dsdt = -np.diff(energy_flux) / cap * SECS_PER_YEAR
        # Add radiogenic heating: dS/dt += H / T (units: J/kg/K/yr)
        T_stag = _eos.temperature(_m.staggered.pressure,
                                  np.asarray(S).flatten())
        dsdt += H_rate / np.maximum(T_stag, 1.0) * SECS_PER_YEAR
        return dsdt

    sol = solve_ivp(dSdt, (0, t_end), S0, method='BDF',
                    atol=0.01, rtol=1e-8, dense_output=True)
    if sol.status != 0:
        print(f'    WARNING: solver status={sol.status}')

    t_final = min(t_end, sol.t[-1])
    times = np.linspace(0, t_final, n_samples)
    S_mean = np.zeros(n_samples)
    for i, t in enumerate(times):
        S_t = sol.sol(t)
        S_mean[i] = np.mean(S_t)

    dS_total = S_mean[-1] - S_mean[0]
    print(f'    <S> change: {dS_total:.4f} J/kg/K over {t_final:.0e} yr')
    print(f'    Expected dS/dt: {dSdt_expected_per_yr:.2e} J/kg/K/yr')

    return times, S_mean, dSdt_expected_per_yr


# ---------------------------------------------------------------------------
# Panel (d): Core cooling
# ---------------------------------------------------------------------------

def run_core_cooling(eos):
    """Grey-body + core cooling BC: F_CMB and T_CMB vs time.

    Returns
    -------
    times : array
        Sampling times [yr].
    F_cmb_arr : array
        CMB heat flux [W/m^2].
    T_cmb_arr : array
        Temperature at CMB node [K].
    """
    print('  Core cooling: Bower+2018 BC ...')
    N = 30
    mesh = make_mesh(N=N)
    state = make_state(mesh, eos, conduction=True, convection=True)

    S0 = np.full(N, 3200.0)
    t_end = 1000.0  # 1000 yr
    n_samples = 80

    # Core parameters (Bower+2018)
    core_density = 12000.0   # kg/m^3
    core_cp = 800.0          # J/kg/K
    tfac = 1.147
    r_cmb = float(np.asarray(mesh.basic.radii).flat[0])
    r_above = float(np.asarray(mesh.basic.radii).flat[1])
    core_vol = 4.0 / 3.0 * np.pi * r_cmb**3
    core_cap = core_vol * core_density * core_cp

    def dSdt(t, S, _s=state, _m=mesh):
        _s.update(S, t)
        # Grey-body at surface
        T_top = _s.top_temperature.item()
        _s._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
        # Core cooling at CMB (Bower+2018 Eq. 37)
        rho_first = float(np.asarray(
            _s.phase_staggered.density()).flat[0])
        cp_first = float(np.asarray(
            _s.phase_staggered.heat_capacity()).flat[0])
        vol_first = float(np.asarray(_m.basic.volume).flat[0])
        cell_cap = vol_first * rho_first * cp_first
        alpha_core = (r_above / r_cmb)**2 / (
            cell_cap / (core_cap * tfac) + 1.0)
        _s._heat_flux[0] = alpha_core * _s._heat_flux[1]

        energy_flux = _s.heat_flux * _m.basic.area
        cap = _s.capacitance_staggered() * _m.basic.volume
        return -np.diff(energy_flux) / cap * SECS_PER_YEAR

    sol = solve_ivp(dSdt, (0, t_end), S0, method='BDF',
                    atol=0.5, rtol=1e-5, dense_output=True)
    if sol.status != 0:
        print(f'    WARNING: solver status={sol.status}')

    t_final = min(t_end, sol.t[-1])
    times = np.linspace(0, t_final, n_samples)
    F_cmb_arr = np.zeros(n_samples)
    T_cmb_arr = np.zeros(n_samples)

    for i, t in enumerate(times):
        S_t = sol.sol(t)
        state.update(S_t, t)
        # Recompute CMB flux with the same BC formula
        rho_first = float(np.asarray(
            state.phase_staggered.density()).flat[0])
        cp_first = float(np.asarray(
            state.phase_staggered.heat_capacity()).flat[0])
        vol_first = float(np.asarray(mesh.basic.volume).flat[0])
        cell_cap = vol_first * rho_first * cp_first
        alpha_core = (r_above / r_cmb)**2 / (
            cell_cap / (core_cap * tfac) + 1.0)
        F_cmb_arr[i] = alpha_core * state._heat_flux[1]
        # T at CMB node (innermost staggered node)
        state.phase_staggered.set_entropy(S_t)
        state.phase_staggered.update()
        T_cmb_arr[i] = float(state.phase_staggered.temperature()[0])

    print(f'    F_CMB: {F_cmb_arr[0]:.2e} -> {F_cmb_arr[-1]:.2e} W/m^2')
    print(f'    T_CMB: {T_cmb_arr[0]:.0f} -> {T_cmb_arr[-1]:.0f} K')

    return times, F_cmb_arr, T_cmb_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not EOS_DIR.exists():
        print(f'ERROR: PALEOS P-S tables not found at {EOS_DIR}')
        print('Set ARAGOG_TEST_EOS_DIR to the directory containing the tables.')
        sys.exit(1)

    print(f'Loading PALEOS P-S tables from {EOS_DIR} ...')
    eos = EntropyEOS(EOS_DIR)
    print(f'  S range: [{eos.S_min:.0f}, {eos.S_max:.0f}] J/kg/K')
    print(f'  P range: [{eos.P_min:.2e}, {eos.P_max:.2e}] Pa')

    # ── Run all four experiments ──────────────────────────────────────
    print('\nPanel (a)/(b): IC sensitivity sweep')
    ic_results = run_ic_sweep(eos)

    print('\nPanel (c): Radiogenic heating')
    heat_times, heat_S_mean, heat_dSdt_exp = run_radiogenic_heating(eos)

    print('\nPanel (d): Core cooling')
    core_times, core_F_cmb, core_T_cmb = run_core_cooling(eos)

    # ── Build 2x2 figure ─────────────────────────────────────────────
    print('\nGenerating figure ...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Color map for IC sweep
    colors = {
        2500.0: '#2166ac',
        3200.0: '#4daf4a',
        5000.0: '#ff7f00',
        10000.0: '#e41a1c',
    }
    labels = {
        2500.0: r'$S_0 = 2500$ (mostly solid)',
        3200.0: r'$S_0 = 3200$ (SPIDER default)',
        5000.0: r'$S_0 = 5000$ (partially liquid)',
        10000.0: r'$S_0 = 10{,}000$ (fully liquid)',
    }

    # ── Panel (a): T_surf(t) for different S0 ────────────────────────
    ax = axes[0, 0]
    for S0_val, data in ic_results.items():
        ax.plot(data['times'], data['T_surf'],
                color=colors[S0_val], label=labels[S0_val], linewidth=2)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(r'$T_\mathrm{surf}$ [K]')
    ax.set_title('(a) IC sweep: surface temperature')
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ── Panel (b): Phi_global(t) for different S0 ────────────────────
    ax = axes[0, 1]
    for S0_val, data in ic_results.items():
        ax.plot(data['times'], data['phi_global'],
                color=colors[S0_val], label=labels[S0_val], linewidth=2)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(r'$\Phi_\mathrm{global}$ (volume-weighted)')
    ax.set_title('(b) IC sweep: global melt fraction')
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.05)

    # ── Panel (c): Radiogenic heating ────────────────────────────────
    ax = axes[1, 0]
    ax.plot(heat_times / 1e6, heat_S_mean, 'b-', linewidth=2,
            label=r'$\langle S \rangle(t)$ (solver)')
    # Overlay expected linear increase
    S0_mean = heat_S_mean[0]
    S_expected = S0_mean + heat_dSdt_exp * heat_times
    ax.plot(heat_times / 1e6, S_expected, 'r--', linewidth=1.5, alpha=0.7,
            label=r'$S_0 + (H/\langle T \rangle) \cdot t$')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$\langle S \rangle$ [J/kg/K]')
    ax.set_title('(c) Radiogenic heating (insulating box)')
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    # Annotate the heating rate
    ax.text(0.95, 0.05,
            f'H = $10^{{-11}}$ W/kg\n'
            fr'$\Delta S$ = {heat_S_mean[-1] - heat_S_mean[0]:.3f} J/kg/K',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Panel (d): Core cooling: F_CMB and T_CMB ─────────────────────
    ax_left = axes[1, 1]
    ax_right = ax_left.twinx()

    ln1 = ax_left.plot(core_times, core_F_cmb, 'b-', linewidth=2,
                       label=r'$F_\mathrm{CMB}$')
    ax_left.set_xlabel('Time [yr]')
    ax_left.set_ylabel(r'$F_\mathrm{CMB}$ [W/m$^2$]', color='#2166ac')
    ax_left.tick_params(axis='y', labelcolor='#2166ac')
    ax_left.set_xlim(left=0)

    ln2 = ax_right.plot(core_times, core_T_cmb, 'r-', linewidth=2,
                        label=r'$T_\mathrm{CMB}$')
    ax_right.set_ylabel(r'$T_\mathrm{CMB}$ [K]', color='#e41a1c')
    ax_right.tick_params(axis='y', labelcolor='#e41a1c')

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc='center right', fontsize=10)
    ax_left.set_title('(d) Core cooling (Bower+2018 BC)')
    # Annotate core parameters
    ax_left.text(0.02, 0.05,
                 r'$\rho_\mathrm{core}$ = 12000 kg/m$^3$, '
                 r'$c_p$ = 800 J/kg/K, $f$ = 1.147',
                 transform=ax_left.transAxes, ha='left', va='bottom',
                 fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           alpha=0.8))

    fig.suptitle('Aragog entropy solver: IC sensitivity and heating verification',
                 fontsize=16, y=1.005)
    fig.tight_layout()

    # Save
    fname_pdf = OUT_DIR / 'verify_ic_heating.pdf'
    fname_png = OUT_DIR / 'verify_ic_heating.png'
    fig.savefig(fname_pdf)
    fig.savefig(fname_png)
    plt.close(fig)
    print(f'\nSaved: {fname_pdf}')
    print(f'Saved: {fname_png}')

    # ── Save raw data as .npz for replotting ─────────────────────────
    npz_data = {}
    for S0_val, data in ic_results.items():
        key = f'ic_{int(S0_val)}'
        npz_data[f'{key}_times'] = data['times']
        npz_data[f'{key}_T_surf'] = data['T_surf']
        npz_data[f'{key}_phi_global'] = data['phi_global']
    npz_data['heat_times'] = heat_times
    npz_data['heat_S_mean'] = heat_S_mean
    npz_data['heat_dSdt_expected'] = np.array([heat_dSdt_exp])
    npz_data['core_times'] = core_times
    npz_data['core_F_cmb'] = core_F_cmb
    npz_data['core_T_cmb'] = core_T_cmb

    npz_path = OUT_DIR / 'verify_ic_heating_data.npz'
    np.savez(npz_path, **npz_data)
    print(f'Saved: {npz_path}')

    # ── Console summary ──────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('Verification Summary')
    print('=' * 65)
    for S0_val, data in ic_results.items():
        T0, Tf = data['T_surf'][0], data['T_surf'][-1]
        phi0, phif = data['phi_global'][0], data['phi_global'][-1]
        mono = np.all(np.diff(data['T_surf']) <= 5.0)
        print(f'  S0={S0_val:6.0f}: T_surf {T0:6.0f} -> {Tf:6.0f} K, '
              f'Phi {phi0:.3f} -> {phif:.3f}, monotonic={mono}')
    dS = heat_S_mean[-1] - heat_S_mean[0]
    print(f'  Heating: dS_mean = {dS:.4f} J/kg/K '
          f'(expected ~ {heat_dSdt_exp * heat_times[-1]:.4f})')
    print(f'  Core: F_CMB {core_F_cmb[0]:.2e} -> {core_F_cmb[-1]:.2e} W/m^2, '
          f'T_CMB {core_T_cmb[0]:.0f} -> {core_T_cmb[-1]:.0f} K')
    print('=' * 65)


if __name__ == '__main__':
    main()
