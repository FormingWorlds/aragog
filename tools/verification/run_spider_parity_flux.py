"""SPIDER vs Aragog parity: prescribed-flux surface BC sweep.

Runs both solvers with identical setup except the surface boundary
condition is a prescribed constant flux (not grey-body). This removes
the nonlinear T^4 feedback and tests the solvers in a simpler regime.

Flux values: 1e2, 1e4, 1e6, 1e8 W/m^2
- 1e2: weak flux, slow cooling (conduction-dominated)
- 1e4: moderate flux (comparable to grey-body at ~1500 K)
- 1e6: strong flux (comparable to grey-body at ~4000 K)
- 1e8: extreme flux (rapid surface draining)

Setup:
    - 1 M_Earth, R_surf = 6371 km, R_core = 3480 km
    - PALEOS MgSiO3 P-S EOS tables
    - Prescribed constant surface flux, insulating core
    - S0 = 3200 J/kg/K (uniform)
    - Conduction + convection, no mixing, no separation
    - N = 100 nodes
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

# Flux cases: {flux_W_m2: (t_end_yr, dt_macro_yr, label)}
FLUX_CASES = {
    1e2:  (50000, 500,  r'$F = 10^2$ W/m$^2$'),
    1e4:  (5000,  50,   r'$F = 10^4$ W/m$^2$'),
    1e6:  (500,   5,    r'$F = 10^6$ W/m$^2$'),
    1e8:  (50,    0.5,  r'$F = 10^8$ W/m$^2$'),
}

plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 9, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'lines.linewidth': 2.0, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


# ── Helpers (reused from run_spider_parity.py) ────────────────────────

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
    for key in ["temp_s", "S_s", "radius_s", "phi_s", "Jtot_b"]:
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

def run_spider_flux(F_surf_prescribed, t_end_yr, dt_macro_yr):
    """Run SPIDER with prescribed constant surface flux."""
    tag = f"F{F_surf_prescribed:.0e}"
    spider_out = OUT_DIR / f"spider_flux_{tag}"
    spider_out.mkdir(parents=True, exist_ok=True)

    mesh_path = spider_out / "mesh.dat"
    generate_spider_mesh(mesh_path)
    n_steps = max(int(t_end_yr / dt_macro_yr), 10)

    opts = f"""# SPIDER prescribed-flux test: F = {F_surf_prescribed:.0e} W/m^2
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
-surface_bc_value {F_surf_prescribed}
-CORE_BC 2
-core_bc_value 0.0

-outputDirectory {spider_out}
"""
    opts_path = spider_out / f"spider_flux_{tag}.opts"
    with open(opts_path, "w") as f:
        f.write(opts)

    env = os.environ.copy()
    petsc_dir = PROTEUS_ROOT / "petsc"
    if petsc_dir.exists():
        env["PETSC_DIR"] = str(petsc_dir)
        env["PETSC_ARCH"] = "arch-darwin-c-opt"

    cmd = [str(SPIDER_BIN), "-options_file", str(opts_path)]
    print(f"    SPIDER cmd: {' '.join(cmd[-3:])}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           cwd=str(SPIDER_DIR), timeout=600)
    if result.returncode != 0:
        print(f"    SPIDER FAILED: {result.stderr[-300:]}")
        return None

    json_files = sorted(
        [f for f in spider_out.iterdir() if f.suffix == ".json"],
        key=lambda f: float(f.stem),
    )
    print(f"    SPIDER: {len(json_files)} output files")

    times, T_magma, phi_global = [], [], []
    for jf in json_files:
        d = read_spider_json(jf)
        times.append(d["t_yr"])
        T_magma.append(d["temp_s"][0] if "temp_s" in d else np.nan)
        phi_global.append(np.mean(d["phi_s"]) if "phi_s" in d else np.nan)

    return {
        "times": np.array(times),
        "T_magma": np.array(T_magma),
        "phi_global": np.array(phi_global),
    }


# ── Aragog run ────────────────────────────────────────────────────────

def run_aragog_flux(F_surf_prescribed, t_end_yr):
    """Run Aragog with prescribed constant surface flux."""
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
        thermal_conductivity_solid=4.0, thermal_conductivity_liquid=4.0,
    )
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
        kappah_floor=0.0,
    )

    S0 = np.full(N, S_INIT)

    def dSdt(t, S):
        state.update(S, t)
        state._heat_flux[-1] = F_surf_prescribed
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

    n_samples = 100
    t_actual_end = min(t_end_yr, sol.t[-1])
    times = np.linspace(0, t_actual_end, n_samples)
    T_magma, phi_global = [], []

    for t in times:
        S_t = sol.sol(t)
        state.update(S_t, t)
        T_magma.append(state.top_temperature.item())
        phi = np.asarray(phase_stag.melt_fraction()).flatten()
        phi_global.append(np.dot(phi, mesh.basic.volume) / np.sum(mesh.basic.volume))

    return {
        "times": np.array(times),
        "T_magma": np.array(T_magma),
        "phi_global": np.array(phi_global),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("SPIDER vs Aragog parity: prescribed-flux sweep")
    print("=" * 60)

    if not SPIDER_BIN.exists():
        print(f"SPIDER binary not found: {SPIDER_BIN}")
        return
    if not EOS_DIR.exists():
        print(f"PALEOS EOS not found: {EOS_DIR}")
        return

    # Run all flux cases
    results = {}
    for F_val, (t_end, dt_macro, label) in FLUX_CASES.items():
        print(f"\n  F = {F_val:.0e} W/m^2 (t_end = {t_end} yr)")
        spider = run_spider_flux(F_val, t_end, dt_macro)
        aragog = run_aragog_flux(F_val, t_end)
        results[F_val] = {"spider": spider, "aragog": aragog, "label": label}

    # ── Figure: 2x2 panels ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {1e2: '#2166ac', 1e4: '#4daf4a', 1e6: '#ff7f00', 1e8: '#e41a1c'}

    # (a) T_magma(t) for all fluxes
    ax = axes[0, 0]
    for F_val in sorted(results.keys()):
        r = results[F_val]
        c = colors[F_val]
        label = r["label"]
        if r["spider"] is not None:
            t_s = r["spider"]["times"]
            # Normalize time to fraction of t_end for comparison
            ax.plot(t_s, r["spider"]["T_magma"], '-', color=c, linewidth=2,
                    label=f'SPIDER {label}')
        if r["aragog"] is not None:
            ax.plot(r["aragog"]["times"], r["aragog"]["T_magma"], '--', color=c,
                    linewidth=2, label=f'Aragog {label}')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
    ax.set_title('(a) Surface temperature')
    ax.set_xscale('log')
    ax.set_xlim(1, None)
    ax.legend(fontsize=7, ncol=2)

    # (b) Phi_global(t) for all fluxes
    ax = axes[0, 1]
    for F_val in sorted(results.keys()):
        r = results[F_val]
        c = colors[F_val]
        if r["spider"] is not None:
            ax.plot(r["spider"]["times"], r["spider"]["phi_global"],
                    '-', color=c, linewidth=2)
        if r["aragog"] is not None:
            ax.plot(r["aragog"]["times"], r["aragog"]["phi_global"],
                    '--', color=c, linewidth=2)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(b) Global melt fraction')
    ax.set_xscale('log')
    ax.set_xlim(1, None)

    # (c) Relative T_magma difference per flux
    ax = axes[1, 0]
    for F_val in sorted(results.keys()):
        r = results[F_val]
        c = colors[F_val]
        s, a = r["spider"], r["aragog"]
        if s is None or a is None:
            continue
        # Interpolate Aragog to SPIDER times
        from scipy.interpolate import interp1d
        t_common = s["times"][s["times"] > 0]
        if len(t_common) < 3:
            continue
        f_T = interp1d(a["times"], a["T_magma"],
                       bounds_error=False, fill_value="extrapolate")
        T_s = np.interp(t_common, s["times"], s["T_magma"])
        T_a = f_T(t_common)
        rel = np.abs(T_a - T_s) / np.maximum(T_s, 1.0) * 100
        ax.plot(t_common, rel, '-', color=c, linewidth=2, label=r["label"])
    ax.axhline(5.0, color='k', ls='--', alpha=0.5, label='5% target')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$|\\Delta T / T|$ [%]')
    ax.set_title('(c) Relative $T_\\mathrm{magma}$ difference')
    ax.set_xscale('log')
    ax.set_xlim(1, None)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8)

    # (d) Summary: final T_magma for each flux
    ax = axes[1, 1]
    F_vals_sorted = sorted(results.keys())
    spider_finals, aragog_finals = [], []
    for F_val in F_vals_sorted:
        r = results[F_val]
        sf = r["spider"]["T_magma"][-1] if r["spider"] is not None else np.nan
        af = r["aragog"]["T_magma"][-1] if r["aragog"] is not None else np.nan
        spider_finals.append(sf)
        aragog_finals.append(af)
    x = np.arange(len(F_vals_sorted))
    w = 0.35
    ax.bar(x - w/2, spider_finals, w, label='SPIDER', color='#4477AA')
    ax.bar(x + w/2, aragog_finals, w, label='Aragog', color='#EE6677')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$10^{{{int(np.log10(f))}}}$' for f in F_vals_sorted])
    ax.set_xlabel('Prescribed flux [W/m$^2$]')
    ax.set_ylabel('Final $T_\\mathrm{magma}$ [K]')
    ax.set_title('(d) Final surface temperature')
    ax.legend()

    fig.suptitle('SPIDER vs Aragog: prescribed-flux parity sweep',
                 fontsize=15, y=1.005)
    fig.tight_layout()

    fname = OUT_DIR / "verify_spider_parity_flux.pdf"
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"\nSaved: {fname}")

    # Summary
    print("\n" + "=" * 60)
    print("Parity Summary (prescribed flux):")
    for F_val in F_vals_sorted:
        r = results[F_val]
        s, a = r["spider"], r["aragog"]
        T_s = s["T_magma"][-1] if s is not None else np.nan
        T_a = a["T_magma"][-1] if a is not None else np.nan
        rel = abs(T_a - T_s) / max(T_s, 1) * 100
        print(f"  F={F_val:.0e}: SPIDER T_final={T_s:.0f} K, "
              f"Aragog T_final={T_a:.0f} K, diff={rel:.1f}%")
    print("=" * 60)

    np.savez(OUT_DIR / "spider_parity_flux_data.npz",
             **{f"F{F_val:.0e}_spider_t": results[F_val]["spider"]["times"]
                for F_val in F_vals_sorted if results[F_val]["spider"] is not None},
             **{f"F{F_val:.0e}_spider_T": results[F_val]["spider"]["T_magma"]
                for F_val in F_vals_sorted if results[F_val]["spider"] is not None},
             **{f"F{F_val:.0e}_aragog_t": results[F_val]["aragog"]["times"]
                for F_val in F_vals_sorted if results[F_val]["aragog"] is not None},
             **{f"F{F_val:.0e}_aragog_T": results[F_val]["aragog"]["T_magma"]
                for F_val in F_vals_sorted if results[F_val]["aragog"] is not None})


if __name__ == "__main__":
    main()
