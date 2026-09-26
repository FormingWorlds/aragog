#!/usr/bin/env python3
"""Step B2: BC 6 root: Newton vs an independent root finder (scipy.optimize.brentq).

Solves the skin balance:
    eps * sigma * (T_s^4 - T_eq^4) = G * (T_top - T_s)
across 10,000 random cases.
Compares aragog.surface_skin.skin_temperature (Newton, 12 iterations) with brentq.
Canary: cuts Newton to 2 iterations and confirms many cases fail tolerance 1e-12.

Deliverables:
- data/B/B2_newton_brentq.npz
- data/B/B2.json
"""

import json
import os
import sys
from pathlib import Path

# Add worktree to path
sys.path.insert(0, "/Users/timlichtenberg/work/ssc-verify-task6/wt-verify/src")
import aragog
from aragog.surface_skin import skin_temperature
import numpy as np
from scipy.constants import Stefan_Boltzmann as SIGMA
from scipy.optimize import brentq

def skin_temperature_2iter(T_top, G, emissivity, T_eq, sigma: float, xp=np):
    es = emissivity * sigma
    T_u = (T_eq**4 + G * xp.maximum(T_top - T_eq, 0.0) / es) ** 0.25
    T = xp.where(T_top >= T_eq, xp.minimum(T_top, T_u), T_eq + 0.0 * T_top)
    for _ in range(2):
        f = es * (T**4 - T_eq**4) - G * (T_top - T)
        T = T - f / (4.0 * es * T**3 + G)
    return T

def main():
    root = Path("/Users/timlichtenberg/work/ssc-verify-task6")
    data_b = root / "data/B"
    data_b.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = np.random.default_rng(seed)
    n_cases = 10000

    T_top = rng.uniform(200.0, 4000.0, n_cases)
    T_eq = rng.uniform(50.0, 2000.0, n_cases)
    G = 10.0 ** rng.uniform(-5.0, 3.0, n_cases)
    eps = rng.uniform(0.1, 1.0, n_cases)

    # Newton solution from aragog
    T_newton = skin_temperature(T_top, G, eps, T_eq, SIGMA)

    # Brentq reference solution
    T_brentq = np.empty(n_cases, dtype=float)
    for i in range(n_cases):
        tt, ge, ep, te = T_top[i], G[i], eps[i], T_eq[i]
        f = lambda T: ep * SIGMA * (T**4 - te**4) - ge * (tt - T)
        hi = max(tt, te)
        T_brentq[i] = brentq(f, 0.0, hi, xtol=1e-13, rtol=1e-15)

    rel_diff = np.abs(T_newton - T_brentq) / T_brentq
    max_rel_diff = float(np.max(rel_diff))

    # Canary: 2 iterations
    T_canary = skin_temperature_2iter(T_top, G, eps, T_eq, SIGMA)
    canary_diff = np.abs(T_canary - T_brentq) / T_brentq
    canary_failures = int(np.sum(canary_diff > 1e-12))

    # Save deliverables
    npz_path = data_b / "B2_newton_brentq.npz"
    np.savez_compressed(
        npz_path,
        T_top=T_top,
        T_eq=T_eq,
        G=G,
        eps=eps,
        T_newton=T_newton,
        T_brentq=T_brentq,
    )

    out_json = {
        "status": "DONE",
        "seed": seed,
        "n_cases": n_cases,
        "max_rel_diff": max_rel_diff,
        "canary_failures": canary_failures,
        "passed": bool(max_rel_diff < 1e-12 and canary_failures > 0)
    }

    json_path = data_b / "B2.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"B2 completed: max_rel_diff = {max_rel_diff:.4e}, canary_failures = {canary_failures}")
    print(f"Deliverables written to {npz_path} and {json_path}")

if __name__ == "__main__":
    main()
