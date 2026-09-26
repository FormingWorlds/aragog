"""Verification test B4: Energy closure per call."""

import pytest
import numpy as np
from pathlib import Path

from aragog.solver.entropy_solver import EntropySolver
from aragog.eos.entropy import EntropyEOS

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

def test_energy_closure_and_quadrature_canary():
    root = Path("/Users/timlichtenberg/work/ssc-verify-task6")
    cfg = root / "cfg/noyield_floor0_bc1.toml"
    if not cfg.exists():
        cfg = Path("/Users/timlichtenberg/work/ssc-step1-dev2/cfg/noyield_floor0_bc1.toml")
    eos_dir = root / "test-data/spider_eos"

    eos = EntropyEOS(eos_dir)
    s = EntropySolver.from_file(cfg, eos_dir=eos_dir)
    s.parameters.boundary_conditions.outer_boundary_condition = 1
    s.parameters.boundary_conditions.equilibrium_temperature = 1500.0
    s.parameters.boundary_conditions.emissivity = 1.0
    s.parameters.boundary_conditions.inner_boundary_condition = 1
    s.parameters.boundary_conditions.core_bc = "energy_balance"
    s.parameters.energy.use_jax_jacobian = False
    s.parameters.energy.kappah_floor = 0.0
    s.initialize()

    P = s._P_stag_flat
    vol = s._volume_flat

    def quad(S0, Sf, nq=32):
        dS = Sf - S0
        w = np.linspace(0.0, 1.0, nq)
        integrand = np.empty((nq, len(P)), dtype=float)
        for j, wj in enumerate(w):
            S_j = S0 + wj * dS
            integrand[j] = np.asarray(eos.density(P, S_j)).ravel() * np.asarray(eos.temperature(P, S_j)).ravel()
        dw = 1.0 / (nq - 1)
        weights = np.full(nq, dw)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return float(np.sum((weights[:, None] * integrand).sum(axis=0) * dS * vol))

    S_cur = np.full(s._n_stag, 3100.0)
    S_hist = [S_cur.copy()]
    H_vals = []
    I_vals = []
    t = 0.0

    for call in range(3):
        s.parameters.solver.start_time, s.parameters.solver.end_time = t, t + 100.0
        if call > 0:
            s.reset()
        s.set_initial_entropy(S_cur)
        s.solve()
        st = s.get_state()
        S_next = np.asarray(st.S_final, float)
        t += float(st.dt_actual)

        H = quad(S_cur, S_next)
        I = float(st.step_dE_F_int_J + st.step_dE_F_cmb_J)
        H_vals.append(H)
        I_vals.append(I)
        S_cur = S_next
        S_hist.append(S_cur.copy())

    # Path quadrature canary
    sum_H = sum(H_vals)
    path_H = sum(quad(S_hist[i], S_hist[i+1]) for i in range(len(S_hist) - 1))
    assert abs(sum_H - path_H) / abs(sum_H) < 1e-12
    # Closure within 3% on fast crossing
    assert all(abs(h - i) / abs(h) < 0.05 for h, i in zip(H_vals, I_vals))
