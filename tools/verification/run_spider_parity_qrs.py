"""SPIDER vs Aragog parity: Tests Q (multi-mass), R (core cooling), S (mixing).

Test Q: Grey-body cooling at 0.5, 1.0, 2.0, 5.0 M_Earth.
        Compare Phi_global(t) and solidification timescale.
Test R: Grey-body + core cooling BC (Bower+2018) at 1 M_Earth.
        Compare T_CMB(t) and F_CMB(t).
Test S: Grey-body + MIXING=1 + SEPARATION=1 at 1 M_Earth.
        Compare Phi_global(t).

All use PALEOS P-S tables (wide range: 1e5 to 500 GPa for multi-mass).

Produces a 3x2 figure:
  (a) Test Q: T_magma(t) for 4 masses
  (b) Test Q: Phi_global(t) for 4 masses
  (c) Test R: T_CMB(t) with core cooling
  (d) Test R: F_CMB(t) with core cooling
  (e) Test S: Phi_global(t) with mixing+separation
  (f) Summary: solidification timescale vs mass
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
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
ARAGOG_ROOT = SCRIPT_DIR.parent.parent
PROTEUS_ROOT = ARAGOG_ROOT.parent
SPIDER_DIR = PROTEUS_ROOT / "SPIDER"
SPIDER_BIN = SPIDER_DIR / "spider"
# Wide-range EOS for multi-mass (up to 500 GPa)
EOS_DIR = ARAGOG_ROOT / "output" / "entropy_verification" / "spider_eos_wide"
OUT_DIR = ARAGOG_ROOT / "output" / "entropy_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ARAGOG_ROOT / "src"))

SECS_PER_YEAR = 31557600.0
S_INIT = 3200.0
DSDR_INIT = -1e-6
N_NODES = 50  # reduced from 80 for tractable runtime at 5 M_Earth
EMISSIVITY = 1.0
T_EQM = 255.0

# Planet parameters: M [M_Earth] -> (R_surf [m], R_core [m], g [m/s^2])
R_E = 6.371e6
G_E = 9.81
CMF = 0.5466  # Earth core radius fraction

MASSES = {
    0.5: {'R_surf': R_E * 0.5**0.27, 'g': G_E * 0.5 / 0.5**0.54},
    1.0: {'R_surf': R_E, 'g': G_E},
    2.0: {'R_surf': R_E * 2.0**0.27, 'g': G_E * 2.0 / 2.0**0.54},
    5.0: {'R_surf': R_E * 5.0**0.27, 'g': G_E * 5.0 / 5.0**0.54},
}
for m in MASSES.values():
    m['R_core'] = CMF * m['R_surf']

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 8, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'lines.linewidth': 2.0, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


# ── Helpers ───────────────────────────────────────────────────────────

def get_spider_array(d, key):
    e = d[key]
    return np.array([float(v) for v in e["values"]]) * float(e["scaling"])

def read_spider_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    t_yr = float(data.get("time_years", 0))
    d = data["data"]
    out = {"t_yr": t_yr}
    for key in ["temp_s", "S_s", "radius_s", "phi_s", "Jtot_b"]:
        try:
            out[key] = get_spider_array(d, key)
        except (KeyError, TypeError):
            pass
    return out

def generate_mesh(outpath, R_surf, R_core, g_surf):
    r_basic = np.linspace(R_surf, R_core, N_NODES)
    rho_ref = 4000.0
    P_basic = rho_ref * g_surf * (R_surf - r_basic)
    r_stag = 0.5 * (r_basic[:-1] + r_basic[1:])
    P_stag = rho_ref * g_surf * (R_surf - r_stag)
    with open(outpath, "w") as f:
        f.write(f"# {N_NODES} {N_NODES - 1}\n")
        for i in range(N_NODES):
            f.write(f"{r_basic[i]:.10e} {P_basic[i]:.10e} {rho_ref:.10e} {-g_surf:.10e}\n")
        for i in range(N_NODES - 1):
            f.write(f"{r_stag[i]:.10e} {P_stag[i]:.10e} {rho_ref:.10e} {-g_surf:.10e}\n")

def make_aragog_mesh(N, R_cmb, R_surf, g_surf):
    rho_ref = 4000.0
    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb; r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_cmb = rho_ref * g_surf * (R_surf - R_cmb)
    P_stag = np.linspace(P_cmb, 1e5, N)
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
    mesh.N = N; mesh.dr = dr
    mesh.R_surf = R_surf; mesh.R_cmb = R_cmb

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


# ── SPIDER runner ─────────────────────────────────────────────────────

def run_spider(tag, R_surf, R_core, g_surf, t_end_yr, dt_macro,
               core_bc=2, core_bc_value=0.0,
               mixing=0, separation=0,
               rho_core=10738.0, cp_core=880.0, coresize_frac=None):
    """Run SPIDER with given parameters. Returns dict or None."""
    spider_out = OUT_DIR / f"spider_{tag}"
    spider_out.mkdir(parents=True, exist_ok=True)
    mesh_path = spider_out / "mesh.dat"
    generate_mesh(mesh_path, R_surf, R_core, g_surf)
    n_steps = max(int(t_end_yr / dt_macro), 10)
    if coresize_frac is None:
        coresize_frac = R_core / R_surf

    opts = f"""-nstepsmacro {n_steps}
-dtmacro {dt_macro}
-n {N_NODES}
-ts_sundials_atol 1.0e-8
-ts_sundials_rtol 1.0e-8
-ts_sundials_type bdf
-entropy0 2.60E3
-radius0 1.0E8
-time0 3.154E7
-MESH_SOURCE 1
-mesh_external_filename {mesh_path}
-radius {R_surf}
-coresize {coresize_frac:.6f}
-gravity {-g_surf}
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
-MIXING {mixing}
-SEPARATION {separation}
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
-SURFACE_BC 1
-emissivity0 {EMISSIVITY}
-teqm {T_EQM}
-PARAM_UTBL 0
-CORE_BC {core_bc}
-core_bc_value {core_bc_value}
-coresize {coresize_frac:.6f}
-rho_core {rho_core}
-cp_core {cp_core}
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
        print(f"    SPIDER FAILED: {result.stderr[-200:]}")
        return None

    json_files = sorted(
        [f for f in spider_out.iterdir() if f.suffix == ".json"],
        key=lambda f: float(f.stem))
    print(f"    SPIDER: {len(json_files)} files")

    # SPIDER: index 0 = surface, index -1 = CMB
    times, T_magma, phi_global, T_cmb, F_cmb = [], [], [], [], []
    for jf in json_files:
        d = read_spider_json(jf)
        times.append(d["t_yr"])
        T_magma.append(d["temp_s"][0] if "temp_s" in d else np.nan)
        T_cmb.append(d["temp_s"][-1] if "temp_s" in d else np.nan)
        phi_global.append(np.mean(d["phi_s"]) if "phi_s" in d else np.nan)
        if "Jtot_b" in d:
            F_cmb.append(d["Jtot_b"][-1])  # CMB = last index
        else:
            F_cmb.append(np.nan)

    return {k: np.array(v) for k, v in
            {"times": times, "T_magma": T_magma, "phi_global": phi_global,
             "T_cmb": T_cmb, "F_cmb": F_cmb}.items()}


# ── Aragog runner ─────────────────────────────────────────────────────

def run_aragog(tag, R_surf, R_core, g_surf, t_end_yr,
               core_cooling=False, grav_sep=False, mixing=False):
    """Run Aragog entropy solver. Returns dict or None."""
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    eos = EntropyEOS(EOS_DIR)
    N = N_NODES - 1
    mesh = make_aragog_mesh(N, R_core, R_surf, g_surf)

    phase_kwargs = dict(
        entropy_eos=eos, gravitational_acceleration=g_surf,
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
        gravitational_separation=grav_sep, mixing=mixing,
        radionuclides=False, tidal=False, tidal_array=[0.0],
        eddy_diffusivity_thermal=1.0, eddy_diffusivity_chemical=1.0,
        kappah_floor=0.0)

    S0 = np.full(N, S_INIT)
    sigma = 5.670374419e-8

    # Core cooling parameters
    rho_core = 10738.0; cp_core = 880.0; tfac = 1.147
    r_cmb = float(mesh.basic.radii[0])
    r_above = float(mesh.basic.radii[1])
    core_vol = 4.0 / 3.0 * np.pi * r_cmb**3
    core_cap = core_vol * rho_core * cp_core

    def dSdt(t, S):
        state.update(S, t)
        # Grey-body surface
        T_top = state.top_temperature.item()
        state._heat_flux[-1] = EMISSIVITY * sigma * (T_top**4 - T_EQM**4)
        # CMB BC
        if core_cooling:
            rho_f = float(np.asarray(state.phase_staggered.density()).flat[0])
            cp_f = float(np.asarray(state.phase_staggered.heat_capacity()).flat[0])
            vol_f = float(mesh.basic.volume[0])
            cell_cap = vol_f * rho_f * cp_f
            alpha_c = (r_above / r_cmb)**2 / (cell_cap / (core_cap * tfac) + 1.0)
            state._heat_flux[0] = alpha_c * state._heat_flux[1]
        else:
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

    n_samples = 80
    t_actual = min(t_end_yr, sol.t[-1])
    times = np.linspace(0, t_actual, n_samples)
    T_magma, phi_global, T_cmb_arr, F_cmb_arr = [], [], [], []

    for t in times:
        S_t = sol.sol(t)
        state.update(S_t, t)
        T_magma.append(state.top_temperature.item())
        phi = np.asarray(phase_stag.melt_fraction()).flatten()
        phi_global.append(np.dot(phi, mesh.basic.volume) / np.sum(mesh.basic.volume))
        T_bot = eos.temperature(
            np.array([mesh.staggered.pressure[0]]),
            np.array([S_t[0]])).item()
        T_cmb_arr.append(T_bot)
        F_cmb_arr.append(float(state._heat_flux[0]))

    return {k: np.array(v) for k, v in
            {"times": times, "T_magma": T_magma, "phi_global": phi_global,
             "T_cmb": T_cmb_arr, "F_cmb": F_cmb_arr}.items()}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("SPIDER vs Aragog: Tests Q, R, S")
    print("=" * 60)

    if not SPIDER_BIN.exists():
        print(f"SPIDER not found: {SPIDER_BIN}"); return
    if not EOS_DIR.exists():
        print(f"EOS not found: {EOS_DIR}"); return

    # ── Test Q: Multi-mass ────────────────────────────────────────────
    print("\n--- Test Q: Multi-mass grey-body ---")
    q_results = {}
    for M_ME, params in MASSES.items():
        R_s, R_c, g = params['R_surf'], params['R_core'], params['g']
        t_end = 10000.0
        dt = 100.0
        print(f"  M = {M_ME} ME (R={R_s/1e6:.2f} Mm, g={g:.1f} m/s^2)")
        sp = run_spider(f"Q_M{M_ME}", R_s, R_c, g, t_end, dt)
        ar = run_aragog(f"Q_M{M_ME}", R_s, R_c, g, t_end)
        q_results[M_ME] = {"spider": sp, "aragog": ar}

    # ── Test R: Core cooling ──────────────────────────────────────────
    print("\n--- Test R: Core cooling (1 ME) ---")
    p = MASSES[1.0]
    sp_r = run_spider("R_core", p['R_surf'], p['R_core'], p['g'],
                      10000, 100, core_bc=1, rho_core=10738, cp_core=880)
    ar_r = run_aragog("R_core", p['R_surf'], p['R_core'], p['g'],
                      10000, core_cooling=True)

    # ── Test S: Mixing + separation ───────────────────────────────────
    print("\n--- Test S: Mixing + separation (1 ME) ---")
    sp_s = run_spider("S_mix", p['R_surf'], p['R_core'], p['g'],
                      10000, 100, mixing=1, separation=1)
    ar_s = run_aragog("S_mix", p['R_surf'], p['R_core'], p['g'],
                      10000, grav_sep=True, mixing=True)

    # ── Figure: 3x2 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    colors = {0.5: '#2166ac', 1.0: '#4daf4a', 2.0: '#ff7f00', 5.0: '#e41a1c'}

    # (a) Test Q: T_magma(t)
    ax = axes[0, 0]
    for M_ME in sorted(q_results.keys()):
        r = q_results[M_ME]; c = colors[M_ME]
        lbl = f'{M_ME} $M_\\oplus$'
        if r["spider"] is not None:
            ax.plot(r["spider"]["times"], r["spider"]["T_magma"],
                    '-', color=c, linewidth=2, label=f'S: {lbl}')
        if r["aragog"] is not None:
            ax.plot(r["aragog"]["times"], r["aragog"]["T_magma"],
                    '--', color=c, linewidth=2, label=f'A: {lbl}')
    ax.set_xlabel('Time [yr]'); ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
    ax.set_title('(a) Test Q: multi-mass $T_\\mathrm{magma}$')
    ax.set_xscale('log'); ax.set_xlim(1, None)
    ax.legend(fontsize=7, ncol=2)

    # (b) Test Q: Phi_global(t)
    ax = axes[0, 1]
    for M_ME in sorted(q_results.keys()):
        r = q_results[M_ME]; c = colors[M_ME]
        if r["spider"] is not None:
            ax.plot(r["spider"]["times"], r["spider"]["phi_global"],
                    '-', color=c, linewidth=2)
        if r["aragog"] is not None:
            ax.plot(r["aragog"]["times"], r["aragog"]["phi_global"],
                    '--', color=c, linewidth=2)
    ax.set_xlabel('Time [yr]'); ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(b) Test Q: multi-mass $\\Phi_\\mathrm{global}$')
    ax.set_xscale('log'); ax.set_xlim(1, None)

    # (c) Test R: T_CMB(t)
    ax = axes[1, 0]
    if sp_r is not None:
        ax.plot(sp_r["times"], sp_r["T_cmb"], 'b-', linewidth=2, label='SPIDER')
    if ar_r is not None:
        ax.plot(ar_r["times"], ar_r["T_cmb"], 'r--', linewidth=2, label='Aragog')
    ax.set_xlabel('Time [yr]'); ax.set_ylabel('$T_\\mathrm{CMB}$ [K]')
    ax.set_title('(c) Test R: core cooling $T_\\mathrm{CMB}$')
    ax.legend()

    # (d) Test R: F_CMB(t)
    ax = axes[1, 1]
    if sp_r is not None:
        ax.plot(sp_r["times"][1:], np.abs(sp_r["F_cmb"][1:]),
                'b-', linewidth=2, label='SPIDER')
    if ar_r is not None:
        ax.plot(ar_r["times"][1:], np.abs(ar_r["F_cmb"][1:]),
                'r--', linewidth=2, label='Aragog')
    ax.set_xlabel('Time [yr]'); ax.set_ylabel('$|F_\\mathrm{CMB}|$ [W/m$^2$]')
    ax.set_title('(d) Test R: core cooling $F_\\mathrm{CMB}$')
    ax.set_yscale('log')
    ax.legend()

    # (e) Test S: Phi_global with mixing+separation
    ax = axes[2, 0]
    # Also plot the no-mixing case for reference (1 ME from Test Q)
    ref = q_results.get(1.0)
    if ref and ref["spider"] is not None:
        ax.plot(ref["spider"]["times"], ref["spider"]["phi_global"],
                'b-', linewidth=1, alpha=0.5, label='S: no mix')
    if ref and ref["aragog"] is not None:
        ax.plot(ref["aragog"]["times"], ref["aragog"]["phi_global"],
                'r--', linewidth=1, alpha=0.5, label='A: no mix')
    if sp_s is not None:
        ax.plot(sp_s["times"], sp_s["phi_global"],
                'b-', linewidth=2, label='S: mix+sep')
    if ar_s is not None:
        ax.plot(ar_s["times"], ar_s["phi_global"],
                'r--', linewidth=2, label='A: mix+sep')
    ax.set_xlabel('Time [yr]'); ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(e) Test S: mixing + separation')
    ax.set_xscale('log'); ax.set_xlim(1, None)
    ax.legend(fontsize=8)

    # (f) Summary: solidification timescale vs mass
    ax = axes[2, 1]
    # Define solidification as Phi < 0.05
    phi_threshold = 0.05
    spider_tsol, aragog_tsol = [], []
    mass_list = sorted(q_results.keys())
    for M_ME in mass_list:
        r = q_results[M_ME]
        for solver, data, arr in [("spider", r["spider"], spider_tsol),
                                   ("aragog", r["aragog"], aragog_tsol)]:
            if data is None:
                arr.append(np.nan); continue
            idx = np.where(data["phi_global"] < phi_threshold)[0]
            if len(idx) > 0:
                arr.append(data["times"][idx[0]])
            else:
                arr.append(data["times"][-1])  # not solidified yet

    x = np.arange(len(mass_list))
    w = 0.35
    ax.bar(x - w/2, spider_tsol, w, label='SPIDER', color='#4477AA')
    ax.bar(x + w/2, aragog_tsol, w, label='Aragog', color='#EE6677')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}' for m in mass_list])
    ax.set_xlabel('Planet mass [$M_\\oplus$]')
    ax.set_ylabel('Solidification time [yr]')
    ax.set_title(f'(f) Solidification timescale ($\\Phi < {phi_threshold}$)')
    ax.set_yscale('log')
    ax.legend()

    fig.suptitle('SPIDER vs Aragog: Tests Q (multi-mass), R (core cooling), S (mixing)',
                 fontsize=14, y=1.005)
    fig.tight_layout()

    fname = OUT_DIR / "verify_spider_parity_qrs.pdf"
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"\nSaved: {fname}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Q (multi-mass):")
    for M_ME in mass_list:
        r = q_results[M_ME]
        sp_phi = r["spider"]["phi_global"][-1] if r["spider"] is not None else np.nan
        ar_phi = r["aragog"]["phi_global"][-1] if r["aragog"] is not None else np.nan
        print(f"  M={M_ME}: SPIDER Phi={sp_phi:.3f}, Aragog Phi={ar_phi:.3f}, "
              f"|dPhi|={abs(ar_phi-sp_phi):.3f}")
    print("Test R (core cooling):")
    if sp_r and ar_r:
        print(f"  SPIDER T_CMB: {sp_r['T_cmb'][0]:.0f} -> {sp_r['T_cmb'][-1]:.0f} K")
        print(f"  Aragog T_CMB: {ar_r['T_cmb'][0]:.0f} -> {ar_r['T_cmb'][-1]:.0f} K")
    print("Test S (mixing+separation):")
    if sp_s and ar_s:
        print(f"  SPIDER Phi: {sp_s['phi_global'][0]:.3f} -> {sp_s['phi_global'][-1]:.3f}")
        print(f"  Aragog Phi: {ar_s['phi_global'][0]:.3f} -> {ar_s['phi_global'][-1]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
