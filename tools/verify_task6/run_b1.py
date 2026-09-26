#!/usr/bin/env python3
"""Step B1: Half-space conduction with mesh refinement.

Derivation of the spherical half-space reference flux:
Spherically symmetric heat conduction in a medium with constant thermal conductivity k,
density rho, and heat capacity cp (thermal diffusivity kappa = k / (rho * cp)) satisfies:
    dT/dt = (kappa / r^2) * d/dr(r^2 * dT/dr)
Using the substitution u(r, t) = r * (T(r, t) - T0), where T0 is the initial uniform temperature,
the spherical diffusion equation simplifies to the 1-D Cartesian heat equation:
    du/dt = kappa * d^2u/dr^2
For a boundary at r = R held at constant temperature T_b (step dT = T_b - T0), the boundary
condition for u is u(R, t) = R * dT, with initial condition u(r, 0) = 0.
The semi-infinite half-space solution for u is:
    u(r, t) = R * dT * erfc( |r - R| / (2 * sqrt(kappa * t)) )
Returning to temperature T(r, t) = T0 + u(r, t) / r:
    T(r, t) = T0 + (R / r) * dT * erfc( |r - R| / (2 * sqrt(kappa * t)) )
The conductive heat flux is F = -k * dT/dr.
Evaluating the radial gradient at r = R:
1. Outer boundary (r = R_out): The mantle domain is r <= R_out.
   With T_skin < T0, cooling step dT_top = T0 - T_skin > 0.
   Evaluating the outward radial flux:
       F_top = k * dT_top * ( 1 / sqrt(pi * kappa * t) - 1 / R_out )
2. Inner boundary (r = R_in): The mantle domain is r >= R_in.
   With T_cmb > T0, heating step dT_bot = T_cmb - T0 > 0.
   Evaluating the inward radial flux into the mantle:
       F_bot = k * dT_bot * ( 1 / sqrt(pi * kappa * t) + 1 / R_in )

This script runs aragog with constant properties, conduction only, N=100 nodes,
end-cell thicknesses: uniform (0), 16, 8, 4, 2, 1 km.
Applied fluxes are computed from dSdt.
Deliverables:
- data/B/B1_halfspace.csv
- data/B/B1_order.json
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

# Insert worktree with full mesh refinement and BC 6 support
sys.path.insert(0, "/Users/timlichtenberg/work/ssc-verify-task6/wt-verify/src")
import aragog
from aragog.parser import (
    Parameters, _BoundaryConditionsParameters, _EnergyParameters,
    _InitialConditionParameters, _MeshParameters, _PhaseMixedParameters,
    _PhaseParameters, _SolverParameters
)
from aragog.solver.entropy_solver import EntropySolver
import numpy as np

def run_halfspace():
    root = Path("/Users/timlichtenberg/work/ssc-verify-task6")
    data_b = root / "data/B"
    data_b.mkdir(parents=True, exist_ok=True)

    YR = 3.15576e7
    T0 = 1500.0
    TEQ = 300.0
    TC = 2700.0
    K = 4.0
    RHO = 4000.0
    CP = 1000.0
    R_OUT = 6.371e6
    R_IN = 3.48e6
    T_END_YR = 1.0e7
    N_NODES = 100
    kappa = K / (RHO * CP)
    L = np.sqrt(np.pi * kappa * T_END_YR * YR)

    cells = [0.0, 16.0, 8.0, 4.0, 2.0, 1.0]
    dev2_err = {
        "top": [17.606, 6.263, 0.782, 0.113, -0.065, -0.146],
        "bottom": [17.068, 6.826, 1.289, 0.294, 0.040, -0.038]
    }

    results = []
    print(f"Running Step B1 on {aragog.__file__}...")

    common = dict(density=RHO, heat_capacity=CP, thermal_conductivity=K, thermal_expansivity=3.0e-5)

    for c in cells:
        t0 = time.time()
        p = Parameters(
            boundary_conditions=_BoundaryConditionsParameters(
                outer_boundary_condition=6,
                outer_boundary_value=TEQ,
                inner_boundary_condition=3,
                inner_boundary_value=TC,
                emissivity=1.0,
                equilibrium_temperature=TEQ,
                core_heat_capacity=880.0,
                core_bc="quasi_steady",
            ),
            energy=_EnergyParameters(
                conduction=True,
                convection=False,
                gravitational_separation=False,
                mixing=False,
                radionuclides=False,
                tidal=False,
                solver_method="cvode",
                use_jax_jacobian=False,
            ),
            initial_condition=_InitialConditionParameters(
                initial_condition=1, surface_temperature=T0, basal_temperature=T0
            ),
            mesh=_MeshParameters(
                outer_radius=R_OUT,
                inner_radius=R_IN,
                number_of_nodes=N_NODES,
                mixing_length_profile="nearest_boundary",
                core_density=RHO,
                surface_density=RHO,
                mass_coordinates=False,
                surface_cell_thickness=c * 1e3,
                cmb_cell_thickness=c * 1e3,
            ),
            phase_solid=_PhaseParameters(melt_fraction=0.0, viscosity=1e21, **common),
            phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=1e2, **common),
            phase_mixed=_PhaseMixedParameters(
                latent_heat_of_fusion=4.0e5,
                rheological_transition_melt_fraction=0.4,
                rheological_transition_width=0.15,
                solidus="solidus.dat",
                liquidus="liquidus.dat",
                phase="mixed",
                phase_transition_width=0.01,
                grain_size=1.0e-3,
                const_properties=True,
                const_rho=RHO,
                const_Cp=CP,
                const_cond=K,
                const_T_ref=T0,
                const_log10visc=2.0,
                const_S_ref=3000.0,
            ),
            radionuclides=[],
            solver=_SolverParameters(start_time=0.0, end_time=T_END_YR, atol=1e-8, rtol=1e-8),
        )

        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        s._phi_rheo = 1.2
        s.set_initial_entropy(3000.0)
        s.solve()
        st = s.get_state()
        s.dSdt(s._solution.t[-1], np.asarray(st.S_final, float).ravel())
        F = np.asarray(s.state.heat_flux, float).ravel().copy()
        wall = time.time() - t0

        Ts = float(np.asarray(getattr(st, "T_surface_skin", TEQ)))
        ref_top = K * (T0 - Ts) * (1.0 / L - 1.0 / R_OUT)
        ref_bot = K * (TC - T0) * (1.0 / L + 1.0 / R_IN)
        err_top_pct = (F[-1] / ref_top - 1.0) * 100.0
        err_bot_pct = (F[0] / ref_bot - 1.0) * 100.0

        r = np.asarray(st.r_basic, float).ravel()
        dr_top = r[-1] - r[-2]
        dr_bot = r[1] - r[0]

        results.append({
            "cell_km": c,
            "dr_top_km": dr_top / 1e3,
            "dr_bot_km": dr_bot / 1e3,
            "F_top": F[-1],
            "ref_top": ref_top,
            "err_top_pct": err_top_pct,
            "F_bot": F[0],
            "ref_bot": ref_bot,
            "err_bot_pct": err_bot_pct,
            "Ts": Ts,
            "wall_s": wall,
        })
        print(f"Cell {c:4.1f} km: wall={wall:.1f}s | Top err={err_top_pct:+.4f}% (ref={ref_top:.6f}) | Bot err={err_bot_pct:+.4f}% (ref={ref_bot:.6f})")

    # Write CSV: side,cell_km,flux_model,flux_ref,err_pct,dev2_err_pct
    csv_path = data_b / "B1_halfspace.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "cell_km", "flux_model", "flux_ref", "err_pct", "dev2_err_pct"])
        for i, row in enumerate(results):
            writer.writerow(["top", row["cell_km"], f"{row['F_top']:.8e}", f"{row['ref_top']:.8e}", f"{row['err_top_pct']:.4f}", f"{dev2_err['top'][i]:.4f}"])
        for i, row in enumerate(results):
            writer.writerow(["bottom", row["cell_km"], f"{row['F_bot']:.8e}", f"{row['ref_bot']:.8e}", f"{row['err_bot_pct']:.4f}", f"{dev2_err['bottom'][i]:.4f}"])

    # Compute observed order from 3 finest cells (4 km, 2 km, 1 km: indices 3, 4, 5)
    # F_4 = results[3], F_2 = results[4], F_1 = results[5]
    top_F4, top_F2, top_F1 = results[3]["F_top"], results[4]["F_top"], results[5]["F_top"]
    bot_F4, bot_F2, bot_F1 = results[3]["F_bot"], results[4]["F_bot"], results[5]["F_bot"]

    order_top = math.log2(abs(top_F4 - top_F2) / abs(top_F2 - top_F1))
    order_bot = math.log2(abs(bot_F4 - bot_F2) / abs(bot_F2 - bot_F1))

    richardson_top = top_F1 - (top_F2 - top_F1)**2 / (top_F4 - 2 * top_F2 + top_F1)
    richardson_bot = bot_F1 - (bot_F2 - bot_F1)**2 / (bot_F4 - 2 * bot_F2 + bot_F1)

    richardson_top_err_pct = (richardson_top / results[5]["ref_top"] - 1.0) * 100.0
    richardson_bot_err_pct = (richardson_bot / results[5]["ref_bot"] - 1.0) * 100.0

    # Verification checks
    agree_top = all(abs(results[i]["err_top_pct"] - dev2_err["top"][i]) <= 0.05 for i in range(len(cells)))
    agree_bot = all(abs(results[i]["err_bot_pct"] - dev2_err["bottom"][i]) <= 0.05 for i in range(len(cells)))

    order_summary = {
        "status": "DONE",
        "observed_order_top": round(order_top, 3),
        "observed_order_bottom": round(order_bot, 3),
        "richardson_top_err_pct": round(richardson_top_err_pct, 4),
        "richardson_bottom_err_pct": round(richardson_bot_err_pct, 4),
        "dev2_order_top": 1.1,
        "dev2_order_bottom": 1.7,
        "agrees_with_dev2_within_0p05_percentage_points": bool(agree_top and agree_bot),
        "passed": bool(agree_top and agree_bot)
    }

    json_path = data_b / "B1_order.json"
    with open(json_path, "w") as f:
        json.dump(order_summary, f, indent=2)

    print("\nB1 Order summary:", json.dumps(order_summary, indent=2))

if __name__ == "__main__":
    run_halfspace()
