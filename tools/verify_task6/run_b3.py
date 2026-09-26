#!/usr/bin/env python3
"""Step B3: Steady conductive slab with BC 6 skin and BC 3 fixed CMB temperature.

Setup:
- Constant properties, conduction only, no heating.
- Inner boundary: fixed CMB temperature TC = 2700 K (BC 3).
- Outer boundary: grey body with conductive skin across top half cell (BC 6), TEQ = 300 K, eps = 1.0.
- Top cell thickness: 1 km (surface_cell_thickness = 1000 m).
- Grid resolutions: N = 50, 100, 200 nodes.

Analytical reference derivation:
In steady state without heating or convection, heat conduction through the spherical shell satisfies:
    d/dr(r^2 * k * dT/dr) = 0  =>  r^2 * k * dT/dr = -Q / (4*pi) = const
Integrating from inner radius R_in to outer radius R_out:
    T(r) = TC - (TC - Ts) * (1/R_in - 1/r) / (1/R_in - 1/R_out)
The total heat flow rate Q is:
    Q = 4 * pi * k * (TC - Ts) / (1/R_in - 1/R_out)
The surface heat flux at the outer radius R_out is:
    F_surf = Q / (4 * pi * R_out^2) = G_shell * (TC - Ts)
where the effective shell conductance referred to the outer surface area is:
    G_shell = k * R_in / (R_out * (R_out - R_in))
At the outer boundary, the skin temperature Ts satisfies the radiative balance:
    F_surf = eps * sigma * (Ts^4 - TEQ^4)
Equating the conductive flux through the shell and the radiated skin flux gives:
    G_shell * (TC - Ts) = eps * sigma * (Ts^4 - TEQ^4)
Solving this 1-D root by hand (via scipy.optimize.brentq) gives the reference skin temperature
Ts and the steady surface reference flux F_ref.

Deliverable:
- data/B/B3_slab.csv (columns: n, flux_model, flux_ref, err_pct)
"""

import csv
import sys
import time
from pathlib import Path

# Add worktree to path
sys.path.insert(0, "/Users/timlichtenberg/work/ssc-verify-task6/wt-verify/src")
import aragog
from aragog.parser import (
    Parameters, _BoundaryConditionsParameters, _EnergyParameters,
    _InitialConditionParameters, _MeshParameters, _PhaseMixedParameters,
    _PhaseParameters, _SolverParameters
)
from aragog.solver.entropy_solver import EntropySolver
import aragog.solver.entropy_solver as es_mod
import numpy as np
from scipy.constants import Stefan_Boltzmann as SIGMA
from scipy.optimize import brentq

# Lift 100 yr step cap for const properties
_orig = es_mod.EntropySolver._solve_cvode
def _patched(self, *a, **k):
    k["max_step"] = np.inf
    return _orig(self, *a, **k)
es_mod.EntropySolver._solve_cvode = _patched

def main():
    root = Path("/Users/timlichtenberg/work/ssc-verify-task6")
    data_b = root / "data/B"
    data_b.mkdir(parents=True, exist_ok=True)

    YR = 3.15576e7
    TEQ = 300.0
    TC = 2700.0
    K = 4.0
    RHO = 4000.0
    CP = 1000.0
    R_OUT = 6.371e6
    R_IN = 3.48e6
    D = R_OUT - R_IN
    T_END_YR = 1.0e11

    # Analytical reference calculation
    G_shell = K * R_IN / (R_OUT * D)
    f_balance = lambda Ts: G_shell * (TC - Ts) - SIGMA * (Ts**4 - TEQ**4)
    Ts_ref = brentq(f_balance, TEQ, TC)
    F_ref = G_shell * (TC - Ts_ref)

    print(f"B3 Reference: Ts={Ts_ref:.4f} K, F_ref={F_ref:.8e} W/m^2")

    common = dict(density=RHO, heat_capacity=CP, thermal_conductivity=K, thermal_expansivity=3.0e-5)
    node_counts = [50, 100, 200]
    results = []

    for n in node_counts:
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
                initial_condition=1, surface_temperature=Ts_ref, basal_temperature=TC
            ),
            mesh=_MeshParameters(
                outer_radius=R_OUT,
                inner_radius=R_IN,
                number_of_nodes=n,
                mixing_length_profile="nearest_boundary",
                core_density=RHO,
                surface_density=RHO,
                mass_coordinates=False,
                surface_cell_thickness=1.0 * 1e3,
                cmb_cell_thickness=0.0,
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
                const_T_ref=1500.0,
                const_log10visc=2.0,
                const_S_ref=3000.0,
            ),
            radionuclides=[],
            solver=_SolverParameters(start_time=0.0, end_time=T_END_YR, atol=1e-8, rtol=1e-8),
        )

        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        s._phi_rheo = 1.2

        # Initialize with steady profile
        r = s._r_stag_flat
        T_prof = TC - (TC - Ts_ref) * (1.0 / R_IN - 1.0 / r) / (1.0 / R_IN - 1.0 / R_OUT)
        S_prof = 3000.0 + CP * np.log(T_prof / 1500.0)
        s.set_initial_entropy(S_prof)

        s.solve()
        st = s.get_state()
        s.dSdt(s._solution.t[-1], np.asarray(st.S_final, float).ravel())
        F = np.asarray(s.state.heat_flux, float).ravel()
        F_model = F[-1]
        err_pct = (F_model / F_ref - 1.0) * 100.0
        wall = time.time() - t0

        results.append({
            "n": n,
            "flux_model": F_model,
            "flux_ref": F_ref,
            "err_pct": err_pct,
            "wall_s": wall,
        })
        print(f"N={n:3d}: F_model={F_model:.8e}, F_ref={F_ref:.8e}, err={err_pct:+.4f}% (wall={wall:.2f}s)")

    # Check acceptance: error falls with N and is below 0.5% at N=200
    errors = [abs(r["err_pct"]) for r in results]
    assert errors[0] > errors[1] > errors[2], f"Errors do not decrease monotonically: {errors}"
    assert errors[-1] < 0.5, f"Error at N=200 ({errors[-1]:.4f}%) is not below 0.5%"

    # Write deliverable CSV
    csv_path = data_b / "B3_slab.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "flux_model", "flux_ref", "err_pct"])
        for r in results:
            writer.writerow([r["n"], f"{r['flux_model']:.8e}", f"{r['flux_ref']:.8e}", f"{r['err_pct']:.4f}"])

    print(f"\nDeliverable written to {csv_path}")

if __name__ == "__main__":
    main()
