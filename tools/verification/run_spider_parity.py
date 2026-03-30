"""SPIDER vs Aragog parity comparison: standalone grey-body cooling.

Runs both SPIDER (C/PETSc) and Aragog (Python/entropy) with identical
PALEOS EOS tables, initial conditions, and boundary conditions, then
compares T_magma(t), Phi_global(t), and F_surf(t).

Setup:
    - 1 M_Earth, R_surf = 6371 km, R_core = 3480 km
    - PALEOS MgSiO3 P-S EOS tables (same files for both solvers)
    - Grey-body surface (emissivity=1, T_eq=255 K)
    - Insulating core (F_CMB = 0)
    - Initial entropy S0 = 3200 J/kg/K (uniform)
    - Conduction + convection, no mixing, no separation
    - N = 100 nodes, 10 kyr integration

Output: output/entropy_verification/verify_spider_parity.pdf
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
R_SURF = 6.371e6   # m
R_CORE = 3.480e6   # m (Earth CMB)
G = 9.81            # m/s^2
S_INIT = 3200.0     # J/kg/K (SPIDER default)
DSDR_INIT = -1e-6   # Initial entropy gradient (SPIDER default)
EMISSIVITY = 1.0
T_EQM = 255.0       # K
N_NODES = 100
T_END_KYR = 10.0    # 10 kyr
DT_MACRO = 100.0    # yr per SPIDER macro step
N_STEPS = int(T_END_KYR * 1000 / DT_MACRO)

SECS_PER_YEAR = 31557600.0

plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'lines.linewidth': 2.0, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


# ── SPIDER helpers ────────────────────────────────────────────────────

def get_spider_array(data_dict, key):
    """Extract a dimensional array from SPIDER JSON output."""
    entry = data_dict[key]
    vals = np.array([float(v) for v in entry["values"]])
    scale = float(entry["scaling"])
    return vals * scale


def read_spider_json(fpath):
    """Read a SPIDER JSON output file and return key arrays."""
    with open(fpath) as f:
        data = json.load(f)
    t_yr = float(data.get("time_years", 0))
    d = data["data"]
    out = {"t_yr": t_yr}
    for key in ["temp_s", "S_s", "radius_s", "radius_b", "phi_s",
                "Jcond_b", "Jconv_b", "Jtot_b"]:
        try:
            out[key] = get_spider_array(d, key)
        except (KeyError, TypeError):
            pass
    return out


def generate_spider_mesh(outpath):
    """Generate a uniform mesh file for SPIDER (surface to CMB ordering)."""
    r_basic = np.linspace(R_SURF, R_CORE, N_NODES)
    # Simple hydrostatic P = rho * g * (R_surf - r)
    # Use a reference density for the mesh (SPIDER will use EOS for actual rho)
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


def run_spider():
    """Run SPIDER standalone with grey-body cooling."""
    print("Running SPIDER...")
    spider_out = OUT_DIR / "spider_parity"
    spider_out.mkdir(parents=True, exist_ok=True)

    # Generate mesh
    mesh_path = spider_out / "mesh.dat"
    generate_spider_mesh(mesh_path)

    # Generate options file
    opts = f"""# SPIDER parity test: grey-body cooling, PALEOS EOS
-nstepsmacro {N_STEPS}
-dtmacro {DT_MACRO:.0f}
-n {N_NODES}
-ts_sundials_atol 1.0e-8
-ts_sundials_rtol 1.0e-8
-ts_sundials_type bdf

# Scaling
-entropy0 2.60E3
-radius0 1.0E8
-time0 3.154E7

# Mesh
-MESH_SOURCE 1
-mesh_external_filename {mesh_path}
-radius {R_SURF}
-coresize {R_CORE / R_SURF:.6f}
-gravity {-G}

# PALEOS EOS tables
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

# Physics
-CONDUCTION 1
-CONVECTION 1
-MIXING 0
-SEPARATION 0
-mixing_length 2

# Material
-grain 1.0E-3
-matprop_smooth_width 1.0E-2
-phi_critical 0.4
-phi_width 0.15
-eddy_diffusivity_thermal 1.0
-eddy_diffusivity_chemical 1.0

# Initial condition
-IC_INTERIOR 1
-ic_adiabat_entropy {S_INIT}
-ic_dsdr {DSDR_INIT}

# BCs: grey-body surface, insulating core
-SURFACE_BC 1
-emissivity0 {EMISSIVITY}
-teqm {T_EQM}
-PARAM_UTBL 0
-CORE_BC 2
-core_bc_value 0.0

# Output
-outputDirectory {spider_out}
"""
    opts_path = spider_out / "spider_parity.opts"
    with open(opts_path, "w") as f:
        f.write(opts)

    # Run SPIDER
    env = os.environ.copy()
    petsc_dir = PROTEUS_ROOT / "petsc"
    if petsc_dir.exists():
        env["PETSC_DIR"] = str(petsc_dir)
        env["PETSC_ARCH"] = "arch-darwin-c-opt"

    cmd = [str(SPIDER_BIN), "-options_file", str(opts_path)]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           cwd=str(SPIDER_DIR), timeout=600)
    if result.returncode != 0:
        print(f"  SPIDER FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[-1000:]}")
        return None

    # Read output
    json_files = sorted(
        [f for f in spider_out.iterdir() if f.suffix == ".json"],
        key=lambda f: float(f.stem),
    )
    print(f"  SPIDER produced {len(json_files)} output files")

    # SPIDER arrays are ordered surface-to-CMB:
    # index 0 = surface, index -1 = CMB
    times, T_magma, phi_global, F_surf = [], [], [], []
    for jf in json_files:
        d = read_spider_json(jf)
        times.append(d["t_yr"])
        T_magma.append(d["temp_s"][0] if "temp_s" in d else np.nan)
        if "phi_s" in d:
            phi_global.append(np.mean(d["phi_s"]))
        else:
            phi_global.append(np.nan)
        if "Jtot_b" in d:
            F_surf.append(d["Jtot_b"][0])
        else:
            F_surf.append(np.nan)

    return {
        "times": np.array(times),
        "T_magma": np.array(T_magma),
        "phi_global": np.array(phi_global),
        "F_surf": np.array(F_surf),
    }


# ── Aragog helpers ────────────────────────────────────────────────────

def make_mesh(N, R_cmb, R_surf, P_cmb=135e9, P_surf=1e5):
    """Build a mesh matching the EntropyState interface."""
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
    # Mixing length type 2 (constant = D/4, matching SPIDER -mixing_length 2)
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


def run_aragog():
    """Run Aragog EntropySolver with grey-body cooling."""
    print("Running Aragog entropy solver...")
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    eos = EntropyEOS(EOS_DIR)
    N = N_NODES - 1  # staggered nodes
    mesh = make_mesh(N, R_CORE, R_SURF)

    # Create phase evaluators matching SPIDER parameters
    phase_kwargs = dict(
        entropy_eos=eos,
        gravitational_acceleration=G,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        grain_size=1e-3,
        viscosity_solid=1e21,
        viscosity_liquid=1e2,
        thermal_conductivity_solid=4.0,
        thermal_conductivity_liquid=4.0,
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
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        tidal_array=[0.0],
        eddy_diffusivity_thermal=1.0,
        eddy_diffusivity_chemical=1.0,
        kappah_floor=0.0,
    )

    # Initial condition
    S0 = np.full(N, S_INIT)

    sigma = 5.670374419e-8

    def dSdt(t, S):
        state.update(S, t)
        T_top = state.top_temperature.item()
        state._heat_flux[-1] = EMISSIVITY * sigma * (T_top**4 - T_EQM**4)
        state._heat_flux[0] = 0.0
        energy_flux = state.heat_flux * mesh.basic.area
        cap = state.capacitance_staggered() * mesh.basic.volume
        return -np.diff(energy_flux) / cap * SECS_PER_YEAR

    t_end = T_END_KYR * 1000  # yr
    sol = solve_ivp(dSdt, (0, t_end), S0, method='BDF',
                    atol=0.5, rtol=1e-5, dense_output=True)

    if sol.status != 0:
        print(f"  Aragog FAILED: {sol.message}")
        return None

    print(f"  Aragog completed: {sol.t[-1]:.0f} yr, {len(sol.t)} steps")

    # Sample at uniform times
    n_samples = 100
    times = np.linspace(0, min(t_end, sol.t[-1]), n_samples)
    T_magma, phi_global, F_surf_arr = [], [], []

    for t in times:
        S_t = sol.sol(t)
        state.update(S_t, t)
        T_top = state.top_temperature.item()
        T_magma.append(T_top)

        phi = np.asarray(phase_stag.melt_fraction()).flatten()
        phi_g = np.dot(phi, mesh.basic.volume) / np.sum(mesh.basic.volume)
        phi_global.append(phi_g)

        F_s = EMISSIVITY * sigma * (T_top**4 - T_EQM**4)
        F_surf_arr.append(F_s)

    return {
        "times": np.array(times),
        "T_magma": np.array(T_magma),
        "phi_global": np.array(phi_global),
        "F_surf": np.array(F_surf_arr),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("SPIDER vs Aragog parity comparison")
    print("=" * 60)

    # Check prerequisites
    if not SPIDER_BIN.exists():
        print(f"SPIDER binary not found: {SPIDER_BIN}")
        return
    if not EOS_DIR.exists():
        print(f"PALEOS EOS directory not found: {EOS_DIR}")
        return

    # Run both solvers
    spider = run_spider()
    aragog = run_aragog()

    if spider is None or aragog is None:
        print("One or both solvers failed. Cannot compare.")
        return

    # ── Comparison figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (a) T_magma(t)
    ax = axes[0, 0]
    ax.plot(spider["times"], spider["T_magma"], 'b-', linewidth=2,
            label='SPIDER')
    ax.plot(aragog["times"], aragog["T_magma"], 'r--', linewidth=2,
            label='Aragog (entropy)')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
    ax.set_title('(a) Surface temperature')
    ax.legend()

    # (b) Phi_global(t)
    ax = axes[0, 1]
    ax.plot(spider["times"], spider["phi_global"], 'b-', linewidth=2,
            label='SPIDER')
    ax.plot(aragog["times"], aragog["phi_global"], 'r--', linewidth=2,
            label='Aragog (entropy)')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(b) Global melt fraction')
    ax.legend()

    # (c) F_surf(t)
    ax = axes[1, 0]
    ax.semilogy(spider["times"][1:], spider["F_surf"][1:], 'b-', linewidth=2,
                label='SPIDER')
    ax.semilogy(aragog["times"][1:], aragog["F_surf"][1:], 'r--', linewidth=2,
                label='Aragog (entropy)')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$F_\\mathrm{surf}$ [W/m$^2$]')
    ax.set_title('(c) Surface flux')
    ax.legend()

    # (d) Relative difference
    ax = axes[1, 1]
    # Interpolate Aragog to SPIDER times for comparison
    from scipy.interpolate import interp1d
    t_common = spider["times"][spider["times"] > 0]
    if len(t_common) > 2:
        f_T_a = interp1d(aragog["times"], aragog["T_magma"],
                         bounds_error=False, fill_value="extrapolate")
        f_phi_a = interp1d(aragog["times"], aragog["phi_global"],
                           bounds_error=False, fill_value="extrapolate")
        T_spider = np.interp(t_common, spider["times"], spider["T_magma"])
        T_aragog = f_T_a(t_common)
        phi_spider = np.interp(t_common, spider["times"], spider["phi_global"])
        phi_aragog = f_phi_a(t_common)

        rel_T = np.abs(T_aragog - T_spider) / np.maximum(T_spider, 1.0) * 100
        rel_phi = np.abs(phi_aragog - phi_spider) / np.maximum(phi_spider, 0.01) * 100

        ax.plot(t_common, rel_T, 'b-', linewidth=2, label='$T_\\mathrm{magma}$')
        ax.plot(t_common, rel_phi, 'g-', linewidth=2, label='$\\Phi_\\mathrm{global}$')
        ax.axhline(5.0, color='k', ls='--', alpha=0.5, label='5% target')
        ax.set_ylim(0, 50)

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Relative difference [%]')
    ax.set_title('(d) Parity: Aragog vs SPIDER')
    ax.legend()

    fig.suptitle('SPIDER vs Aragog parity: standalone grey-body cooling',
                 fontsize=15, y=1.005)
    fig.tight_layout()

    fname = OUT_DIR / "verify_spider_parity.pdf"
    fig.savefig(fname)
    fig.savefig(str(fname).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"\nSaved: {fname}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Parity Summary:")
    print(f"  SPIDER: T_magma {spider['T_magma'][0]:.0f} -> {spider['T_magma'][-1]:.0f} K")
    print(f"  Aragog: T_magma {aragog['T_magma'][0]:.0f} -> {aragog['T_magma'][-1]:.0f} K")
    if len(t_common) > 2:
        print(f"  Max |dT/T|: {np.max(rel_T):.1f}%")
        print(f"  Max |dPhi/Phi|: {np.max(rel_phi[phi_spider > 0.01]):.1f}%"
              if np.any(phi_spider > 0.01) else "  Phi too small to compare")
        within_5 = np.mean(rel_T < 5.0) * 100
        print(f"  T_magma within 5%: {within_5:.0f}% of time points")
    print("=" * 60)

    # Save data
    np.savez(OUT_DIR / "spider_parity_data.npz",
             spider_times=spider["times"], spider_T=spider["T_magma"],
             spider_phi=spider["phi_global"], spider_F=spider["F_surf"],
             aragog_times=aragog["times"], aragog_T=aragog["T_magma"],
             aragog_phi=aragog["phi_global"], aragog_F=aragog["F_surf"])


if __name__ == "__main__":
    main()
