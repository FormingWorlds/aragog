#!/usr/bin/env python3
"""Step B4: Energy closure per call.

Fast-crossing case on wt-aragog-tl_ssc-cmb-bl-law:
- Config: noyield_floor0_bc1.toml
- Uniform S = 3100 J/kg/K, grey body T_eq = 1500 K, 100 yr calls, numpy, tables, n = 100, energy_balance core.
- 50 calls.
- Per call: compute H = sum_i V_i * int_{S_start}^{S_end} rho T dS (trapezoid quadrature, 64 points in S per cell)
  and I = dE_int + dE_cmb from solver step integrals.
- Measure rel_err = |H - I| / |H|.
- Canary: sum of H over 50 calls equals composite path quadrature from S_0 to S_50 within 1e-12 relative.

Deliverables:
- data/B/B4_closure.csv (call, H, I, rel_err)
- data/B/B4.json (keys: median, max, canary_rel)
"""

import csv
import json
import sys
import time
from pathlib import Path

# Add worktree to path
sys.path.insert(0, "/Users/timlichtenberg/work/ssc-verify-task6/wt-verify/src")
import aragog
from aragog.solver.entropy_solver import EntropySolver
from aragog.eos.entropy import EntropyEOS
import numpy as np

def quad_cell_heat(eos, P, vol, S_start, S_end, nq=64):
    """Quadrature of sum_i V_i * int rho T dS from S_start to S_end."""
    dS = S_end - S_start
    w = np.linspace(0.0, 1.0, nq)
    integrand = np.empty((nq, len(P)), dtype=float)
    for j, wj in enumerate(w):
        S_j = S_start + wj * dS
        rho_j = np.asarray(eos.density(P, S_j)).ravel()
        T_j = np.asarray(eos.temperature(P, S_j)).ravel()
        integrand[j] = rho_j * T_j
    dw = 1.0 / (nq - 1)
    weights = np.full(nq, dw)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    cell_int = (weights[:, None] * integrand).sum(axis=0) * dS
    return float(np.sum(cell_int * vol))

def main():
    root = Path("/Users/timlichtenberg/work/ssc-verify-task6")
    data_b = root / "data/B"
    data_b.mkdir(parents=True, exist_ok=True)

    CFG = root / "cfg/noyield_floor0_bc1.toml"
    if not CFG.exists():
        CFG = Path("/Users/timlichtenberg/work/ssc-step1-dev2/cfg/noyield_floor0_bc1.toml")
    EOS_DIR = root / "test-data/spider_eos"

    eos = EntropyEOS(EOS_DIR)
    s = EntropySolver.from_file(CFG, eos_dir=EOS_DIR)
    bc = s.parameters.boundary_conditions
    bc.outer_boundary_condition = 1
    bc.equilibrium_temperature = 1500.0
    bc.emissivity = 1.0
    bc.inner_boundary_condition = 1
    bc.core_bc = "energy_balance"
    s.parameters.energy.use_jax_jacobian = False
    s.parameters.energy.kappah_floor = 0.0
    s.initialize()

    P = s._P_stag_flat
    vol = s._volume_flat
    n_calls = 50
    call_yr = 100.0

    S_cur = np.full(s._n_stag, 3100.0)
    S_history = [S_cur.copy()]
    t = 0.0

    results = []
    print(f"Running Step B4 on {aragog.__file__} ({n_calls} calls of {call_yr} yr)...")
    t0 = time.time()

    for call in range(1, n_calls + 1):
        s.parameters.solver.start_time = t
        s.parameters.solver.end_time = t + call_yr
        if call > 1:
            s.reset()
        s.set_initial_entropy(S_cur)
        s.solve()
        st = s.get_state()
        S_next = np.asarray(st.S_final, float)
        t += float(st.dt_actual)

        H_call = quad_cell_heat(eos, P, vol, S_cur, S_next, nq=64)
        I_call = float(st.step_dE_F_int_J + st.step_dE_F_cmb_J)
        rel_err = abs(H_call - I_call) / abs(H_call)

        results.append({
            "call": call,
            "H": H_call,
            "I": I_call,
            "rel_err": rel_err
        })
        S_cur = S_next
        S_history.append(S_cur.copy())

        if call % 10 == 0 or call <= 5:
            print(f"Call {call:2d}/{n_calls}: H={H_call:.6e}, I={I_call:.6e}, rel_err={rel_err:.6e} (elapsed {time.time()-t0:.1f}s)")

    wall_total = time.time() - t0
    rel_errors = [r["rel_err"] for r in results]
    median_err = float(np.median(rel_errors))
    max_err = float(np.max(rel_errors))

    # Canary: sum of H over all calls vs composite path quadrature
    sum_H = sum(r["H"] for r in results)
    # Composite path quadrature along the exact sequence of 50 segments
    path_H = 0.0
    for i in range(len(S_history) - 1):
        path_H += quad_cell_heat(eos, P, vol, S_history[i], S_history[i+1], nq=64)
    canary_rel = abs(sum_H - path_H) / abs(sum_H)

    print(f"\nB4 complete in {wall_total:.1f}s:")
    print(f"Median rel_err: {median_err:.6e} (dev-2 claim: ~9.2e-4)")
    print(f"Max rel_err:    {max_err:.6e}")
    print(f"Canary rel diff: {canary_rel:.6e} (must be < 1e-12)")

    # Deliverables
    csv_path = data_b / "B4_closure.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["call", "H", "I", "rel_err"])
        for r in results:
            writer.writerow([r["call"], f"{r['H']:.8e}", f"{r['I']:.8e}", f"{r['rel_err']:.6e}"])

    out_json = {
        "status": "DONE",
        "n_calls": n_calls,
        "median": median_err,
        "max": max_err,
        "canary_rel": canary_rel,
        "dev2_claim_median": 9.2e-4,
        "canary_passed": bool(canary_rel < 1e-12)
    }

    json_path = data_b / "B4.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"Deliverables written to {csv_path} and {json_path}")

if __name__ == "__main__":
    main()
