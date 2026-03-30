"""SPIDER vs Aragog parity: prescribed-flux surface BC sweep.

Runs both solvers with identical setup except the surface boundary
condition is a prescribed constant flux (not grey-body). This removes
the nonlinear T^4 feedback and tests the solvers in a simpler regime.

Flux values: 1e2, 1e4, 1e6, 1e8 W/m^2

Produces a 3x2 figure:
  Row 1-2: T-P profiles at 5 time snapshots (one panel per flux case)
  Row 3: (e) Phi_global(t) for all fluxes, (f) relative Phi difference
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ARAGOG_ROOT = SCRIPT_DIR.parent.parent
PROTEUS_ROOT = ARAGOG_ROOT.parent
SPIDER_DIR = PROTEUS_ROOT / "SPIDER"
SPIDER_BIN = SPIDER_DIR / "spider"
EOS_DIR = PROTEUS_ROOT / "output" / "coupled_parity" / "spider" / "data" / "spider_eos"
OUT_DIR = ARAGOG_ROOT / "output" / "entropy_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ARAGOG_ROOT / "src"))

# Physics parameters
R_SURF = 6.371e6
R_CORE = 3.480e6
G = 9.81
S_INIT = 3200.0
DSDR_INIT = -1e-6
N_NODES = 100
SECS_PER_YEAR = 31557600.0

# Flux cases: {flux_W_m2: (t_end_yr, dt_macro_yr)}
FLUX_CASES = {
    1e2:  (50000, 500),
    1e4:  (5000,  50),
    1e6:  (500,   5),
    1e8:  (50,    0.5),
}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 8, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'lines.linewidth': 1.8, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


# ── Helpers ───────────────────────────────────────────────────────────

def get_spider_array(data_dict, key):
    entry = data_dict[key]
    vals = np.array([float(v) for v in entry["values"]])
    return vals * float(entry["scaling"])


def read_spider_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    t_yr = float(data.get("time_years", 0))
    d = data["data"]
    out = {"t_yr": t_yr}
    for key in ["temp_s", "S_s", "radius_s", "radius_b", "phi_s", "Jtot_b"]:
        try:
            out[key] = get_spider_array(d, key)
        except (KeyError, TypeError):
            pass
    return out


def generate_spider_mesh(outpath):
    r_basic = np.linspace(R_SURF, R_CORE, N_NODES)
    rho_ref = 4000.0
    P_basic = rho_ref * G * (R_SURF - r_basic)
    r_stag = 0.5 * (r_basic[:-1] + r_basic[1:])
    P_stag = rho_ref * G * (R_SURF - r_stag)
    with open(outpath, "w") as f:
        f.write(f"# {N_NODES} {N_NODES - 1}\n")
        for i in range(N_NODES):
            f.write(f"{r_basic[i]:.10e} {P_basic[i]:.10e} {rho_ref:.10e} {-G:.10e}\n")
        for i in range(N_NODES - 1):
            f.write(f"{r_stag[i]:.10e} {P_stag[i]:.10e} {rho_ref:.10e} {-G:.10e}\n")


def make_aragog_mesh(N, R_cmb, R_surf, P_cmb=135e9, P_surf=1e5):
    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_stag = np.linspace(P_cmb, P_surf, N)
    D = R_surf - R_cmb

    class SubMesh:
        pass
    class Mesh:
        pass
    mesh = Mesh()
    mesh.basic = SubMesh()
    mesh.staggered = SubMesh()
    mesh.basic.radii = r_basic
    mesh.staggered.radii = r_stag
    mesh.basic.area = 4.0 * np.pi * r_basic**2
    mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    mesh.basic.mixing_length = np.full(N + 1, D / 4.0)
    mesh.basic.mixing_length_squared = mesh.basic.mixing_length**2
    mesh.basic.mixing_length_cubed = mesh.basic.mixing_length**3
    mesh.basic.pressure = np.interp(r_basic, r_stag, P_stag)
    mesh.staggered.pressure = P_stag
    mesh.N = N
    mesh.dr = dr

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


# ── SPIDER run ────────────────────────────────────────────────────────

def run_spider_flux(F_surf, t_end_yr, dt_macro_yr):
    tag = f"F{F_surf:.0e}"
    spider_out = OUT_DIR / f"spider_flux_{tag}"
    spider_out.mkdir(parents=True, exist_ok=True)

    mesh_path = spider_out / "mesh.dat"
    generate_spider_mesh(mesh_path)
    n_steps = max(int(t_end_yr / dt_macro_yr), 10)

    opts = f"""# SPIDER prescribed-flux: F = {F_surf:.0e} W/m^2
-nstepsmacro {n_steps}
-dtmacro {dt_macro_yr}
-n {N_NODES}
-ts_sundials_atol 1.0e-8
-ts_sundials_rtol 1.0e-8
-ts_sundials_type bdf
-entropy0 2.60E3
-radius0 1.0E8
-time0 3.154E7
-MESH_SOURCE 1
-mesh_external_filename {mesh_path}
-radius {R_SURF}
-coresize {R_CORE / R_SURF:.6f}
-gravity {-G}
-phase_names melt,solid
-melt_TYPE 1
-melt_alpha_filename {EOS_DIR}/thermal_exp_melt.dat
-melt_cp_filename {EOS_DIR}/heat_capacity_melt.dat
-melt_dTdPs_filename {EOS_DIR}/adiabat_temp_grad_melt.dat
-melt_rho_filename {EOS_DIR}/density_melt.dat
-melt_temp_filename {EOS_DIR}/temperature_melt.dat
-melt_log10visc 2.0
-melt_cond 4.0
-melt_phase_boundary_filename {EOS_DIR}/liquidus_P-S.dat
-solid_TYPE 1
-solid_alpha_filename {EOS_DIR}/thermal_exp_solid.dat
-solid_cp_filename {EOS_DIR}/heat_capacity_solid.dat
-solid_dTdPs_filename {EOS_DIR}/adiabat_temp_grad_solid.dat
-solid_rho_filename {EOS_DIR}/density_solid.dat
-solid_temp_filename {EOS_DIR}/temperature_solid.dat
-solid_log10visc 21.0
-solid_cond 4.0
-solid_phase_boundary_filename {EOS_DIR}/solidus_P-S.dat
-CONDUCTION 1
-CONVECTION 1
-MIXING 0
-SEPARATION 0
-mixing_length 2
-grain 1.0E-3
-matprop_smooth_width 1.0E-2
-phi_critical 0.4
-phi_width 0.15
-eddy_diffusivity_thermal 1.0
-eddy_diffusivity_chemical 1.0
-IC_INTERIOR 1
-ic_adiabat_entropy {S_INIT}
-ic_dsdr {DSDR_INIT}
-SURFACE_BC 4
-surface_bc_value {F_surf}
-CORE_BC 2
-core_bc_value 0.0
-outputDirectory {spider_out}
"""
    opts_path = spider_out / f"spider_{tag}.opts"
    with open(opts_path, "w") as f:
        f.write(opts)

    env = os.environ.copy()
    petsc_dir = PROTEUS_ROOT / "petsc"
    if petsc_dir.exists():
        env["PETSC_DIR"] = str(petsc_dir)
        env["PETSC_ARCH"] = "arch-darwin-c-opt"

    result = subprocess.run(
        [str(SPIDER_BIN), "-options_file", str(opts_path)],
        env=env, capture_output=True, text=True,
        cwd=str(SPIDER_DIR), timeout=600)
    if result.returncode != 0:
        print(f"    SPIDER FAILED: {result.stderr[-300:]}")
        return None

    json_files = sorted(
        [f for f in spider_out.iterdir() if f.suffix == ".json"],
        key=lambda f: float(f.stem))
    print(f"    SPIDER: {len(json_files)} files")

    # Read ALL snapshots (for T-P profiles and Phi)
    snapshots = []
    for jf in json_files:
        snapshots.append(read_spider_json(jf))

    # SPIDER arrays: index 0 = surface, index -1 = CMB
    times = np.array([s["t_yr"] for s in snapshots])
    phi_global = np.array([np.mean(s["phi_s"]) if "phi_s" in s else np.nan
                           for s in snapshots])

    # Pressure from radius (simple hydrostatic)
    rho_ref = 4000.0
    if "radius_s" in snapshots[0]:
        r_s = snapshots[0]["radius_s"]
        P_s = rho_ref * G * (R_SURF - r_s)  # surface=0, CMB=max
    else:
        P_s = None

    return {
        "times": times,
        "phi_global": phi_global,
        "snapshots": snapshots,
        "P_s_GPa": P_s / 1e9 if P_s is not None else None,
    }


# ── Aragog run ────────────────────────────────────────────────────────

def run_aragog_flux(F_surf, t_end_yr):
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    eos = EntropyEOS(EOS_DIR)
    N = N_NODES - 1
    mesh = make_aragog_mesh(N, R_CORE, R_SURF)

    phase_kwargs = dict(
        entropy_eos=eos, gravitational_acceleration=G,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15, grain_size=1e-3,
        viscosity_solid=1e21, viscosity_liquid=1e2,
        thermal_conductivity_solid=4.0, thermal_conductivity_liquid=4.0)
    phase_stag = EntropyPhaseEvaluator(**phase_kwargs)
    phase_stag.set_pressure(mesh.staggered.pressure)
    phase_basic = EntropyPhaseEvaluator(**phase_kwargs)
    phase_basic.set_pressure(mesh.basic.pressure)

    class _Eval:
        pass
    evaluator = _Eval()
    evaluator.mesh = mesh
    evaluator.radionuclides = []

    state = EntropyState(
        evaluator=evaluator, phase_staggered=phase_stag,
        phase_basic=phase_basic, conduction=True, convection=True,
        gravitational_separation=False, mixing=False,
        radionuclides=False, tidal=False, tidal_array=[0.0],
        eddy_diffusivity_thermal=1.0, eddy_diffusivity_chemical=1.0,
        kappah_floor=0.0)

    S0 = np.full(N, S_INIT)

    def dSdt(t, S):
        state.update(S, t)
        state._heat_flux[-1] = F_surf
        state._heat_flux[0] = 0.0
        energy_flux = state.heat_flux * mesh.basic.area
        cap = state.capacitance_staggered() * mesh.basic.volume
        return -np.diff(energy_flux) / cap * SECS_PER_YEAR

    sol = solve_ivp(dSdt, (0, t_end_yr), S0, method='BDF',
                    atol=0.5, rtol=1e-5, dense_output=True)
    if sol.status != 0:
        print(f"    Aragog FAILED: {sol.message}")
        return None
    print(f"    Aragog: {sol.t[-1]:.0f} yr, {len(sol.t)} steps")

    # Sample at 5 profile times + 100 points for Phi(t)
    t_actual = min(t_end_yr, sol.t[-1])
    profile_fracs = [0.0, 0.1, 0.3, 0.6, 1.0]
    profile_times = [f * t_actual for f in profile_fracs]

    P_stag_GPa = mesh.staggered.pressure / 1e9

    profiles = []
    for t in profile_times:
        S_t = sol.sol(t)
        T_t = eos.temperature(mesh.staggered.pressure, S_t)
        profiles.append({"t_yr": t, "T": np.asarray(T_t).flatten(),
                         "P_GPa": P_stag_GPa})

    n_phi = 100
    phi_times = np.linspace(0, t_actual, n_phi)
    phi_global = []
    for t in phi_times:
        S_t = sol.sol(t)
        state.update(S_t, t)
        phi = np.asarray(phase_stag.melt_fraction()).flatten()
        phi_global.append(np.dot(phi, mesh.basic.volume) / np.sum(mesh.basic.volume))

    return {
        "times": phi_times,
        "phi_global": np.array(phi_global),
        "profiles": profiles,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("SPIDER vs Aragog: prescribed-flux parity sweep")
    print("=" * 60)

    if not SPIDER_BIN.exists():
        print(f"SPIDER binary not found: {SPIDER_BIN}"); return
    if not EOS_DIR.exists():
        print(f"PALEOS EOS not found: {EOS_DIR}"); return

    results = {}
    for F_val, (t_end, dt_macro) in FLUX_CASES.items():
        print(f"\n  F = {F_val:.0e} W/m^2 (t_end = {t_end} yr)")
        spider = run_spider_flux(F_val, t_end, dt_macro)
        aragog = run_aragog_flux(F_val, t_end)
        results[F_val] = {"spider": spider, "aragog": aragog}

    # ── Compute solidus/liquidus in T-P space for mush zone shading ──
    from aragog.eos.entropy import EntropyEOS
    from scipy.signal import savgol_filter
    eos = EntropyEOS(EOS_DIR)
    P_curve = np.linspace(1e5, 135e9, 500)
    T_sol_raw = np.array([eos.temperature(np.array([p]),
                           np.array([eos.solidus_entropy(p)])).item()
                           for p in P_curve])
    T_liq_raw = np.array([eos.temperature(np.array([p]),
                           np.array([eos.liquidus_entropy(p)])).item()
                           for p in P_curve])
    # Smooth kinks from bilinear interpolation on 500x200 P-S grid.
    # TODO: regenerate PALEOS-2phase tables at higher resolution to
    # eliminate the need for post-hoc smoothing.
    window = min(51, len(P_curve) // 4 * 2 + 1)
    T_sol = savgol_filter(T_sol_raw, window, 3)
    T_liq = savgol_filter(T_liq_raw, window, 3)
    P_curve_GPa = P_curve / 1e9

    # ── Figure: 3 rows x 2 columns ───────────────────────────────────
    flux_list = sorted(FLUX_CASES.keys())
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    import matplotlib.cm as cm

    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, F_val in enumerate(flux_list):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        r = results[F_val]
        sp = r["spider"]
        ar = r["aragog"]

        # Mush zone shading (between solidus and liquidus)
        ax.fill_betweenx(P_curve_GPa, T_sol, T_liq,
                         color='#cccccc', alpha=0.4, zorder=0)
        ax.plot(T_sol, P_curve_GPa, 'k-', linewidth=0.7, alpha=0.5)
        ax.plot(T_liq, P_curve_GPa, 'k-', linewidth=0.7, alpha=0.5)

        # SPIDER snapshots
        if sp is not None and sp["snapshots"]:
            snaps = sp["snapshots"]
            n_snap = len(snaps)
            snap_idxs = [0] + [int(f * (n_snap - 1))
                                for f in [0.1, 0.3, 0.6, 1.0]]
            snap_idxs = sorted(set(snap_idxs))

            cmap_s = cm.Blues
            for i, si in enumerate(snap_idxs):
                s = snaps[si]
                if "temp_s" not in s or sp["P_s_GPa"] is None:
                    continue
                t_yr = s["t_yr"]
                color = cmap_s(0.3 + 0.7 * i / max(len(snap_idxs) - 1, 1))
                label = f'S: {t_yr:.0f} yr' if i in [0, len(snap_idxs)-1] else None
                ax.plot(s["temp_s"], sp["P_s_GPa"], '-',
                        color=color, linewidth=1.5, label=label)

        # Aragog profiles
        if ar is not None and ar["profiles"]:
            cmap_a = cm.Reds
            for i, prof in enumerate(ar["profiles"]):
                color = cmap_a(0.3 + 0.7 * i / max(len(ar["profiles"]) - 1, 1))
                label = f'A: {prof["t_yr"]:.0f} yr' if i in [0, len(ar["profiles"])-1] else None
                ax.plot(prof["T"], prof["P_GPa"], '--',
                        color=color, linewidth=1.5, label=label)

        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [GPa]')
        ax.set_title(f'{panel_labels[idx]} $F = 10^{{{int(np.log10(F_val))}}}$ W/m$^2$')
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc='lower left')

    # ── Row 3, left: Phi_global(t) ────────────────────────────────────
    ax = axes[2, 0]
    colors = {1e2: '#2166ac', 1e4: '#4daf4a', 1e6: '#ff7f00', 1e8: '#e41a1c'}
    for F_val in flux_list:
        r = results[F_val]
        c = colors[F_val]
        lbl = f'$10^{{{int(np.log10(F_val))}}}$'
        if r["spider"] is not None:
            ax.plot(r["spider"]["times"], r["spider"]["phi_global"],
                    '-', color=c, linewidth=2, label=f'S: {lbl}')
        if r["aragog"] is not None:
            ax.plot(r["aragog"]["times"], r["aragog"]["phi_global"],
                    '--', color=c, linewidth=2, label=f'A: {lbl}')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(e) Global melt fraction')
    ax.set_xscale('log')
    ax.set_xlim(1, None)
    ax.legend(fontsize=7, ncol=2, loc='upper right')

    # ── Row 3, right: relative Phi difference ─────────────────────────
    ax = axes[2, 1]
    for F_val in flux_list:
        r = results[F_val]
        c = colors[F_val]
        sp, ar = r["spider"], r["aragog"]
        if sp is None or ar is None:
            continue
        from scipy.interpolate import interp1d
        t_common = sp["times"][sp["times"] > 0]
        if len(t_common) < 3:
            continue
        f_phi = interp1d(ar["times"], ar["phi_global"],
                         bounds_error=False, fill_value="extrapolate")
        phi_s = np.interp(t_common, sp["times"], sp["phi_global"])
        phi_a = f_phi(t_common)
        # Absolute difference (Phi is already 0-1)
        abs_diff = np.abs(phi_a - phi_s)
        lbl = f'$10^{{{int(np.log10(F_val))}}}$'
        ax.plot(t_common, abs_diff, '-', color=c, linewidth=2, label=lbl)
    ax.axhline(0.05, color='k', ls='--', alpha=0.5, label='0.05 target')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$|\\Delta\\Phi_\\mathrm{global}|$')
    ax.set_title('(f) Absolute $\\Phi$ difference')
    ax.set_xscale('log')
    ax.set_xlim(1, None)
    ax.set_ylim(0, 0.35)
    ax.legend(fontsize=8)

    fig.suptitle('SPIDER vs Aragog: prescribed-flux parity (T-P profiles + $\\Phi$)',
                 fontsize=15, y=1.005)
    fig.tight_layout()

    fname = OUT_DIR / "verify_spider_parity_flux.pdf"
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"\nSaved: {fname}")

    # Summary
    print("\n" + "=" * 60)
    print("Parity Summary:")
    for F_val in flux_list:
        r = results[F_val]
        sp, ar = r["spider"], r["aragog"]
        phi_s_f = sp["phi_global"][-1] if sp is not None else np.nan
        phi_a_f = ar["phi_global"][-1] if ar is not None else np.nan
        dphi = abs(phi_a_f - phi_s_f)
        print(f"  F={F_val:.0e}: SPIDER Phi_final={phi_s_f:.3f}, "
              f"Aragog Phi_final={phi_a_f:.3f}, |dPhi|={dphi:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
