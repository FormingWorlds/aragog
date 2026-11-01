"""Run the stagnant-lid benchmark rows of the SSC harness with the sub-grid lid.

Each row of Korenaga (2009, Table 1) or Deschamps and Vilella (2021, Table 1) runs through
``evaluate_mlt_top_slope_calibration`` at a calibration point, with the lid on or off and the
stress closure set on the harness solver, and with the real energy integrals. Per row the result
holds the harness observables, the lid hand-over time, the accepted steps whose Stefan jump lies
below ``LID_DH_MIN``, and per solver call two ledger checks:

- the gross guard ``|miss| / (|F_int| + |F_cmb| + |Q_radio| + |Q_tidal| + |dX|)``, with ``miss``
  the state heat minus the integrated powers and ``dX`` the lid heat change of the call;
- for a call that starts with the lid, the lid sub-ledger: the change of the lid excess heat
  ``X = A (E - W_f delta)`` from the states against the trapezoid integral of
  ``dX/dt = A (dE/dt - W_f ddelta/dt - delta rho_f T_f dS_f/dt)`` up to the last node with the lid.

Usage: ``python tools/verification/run_surface_lid_v5_rows.py OUT.jsonl on|off lid|local K:E230a D:f60_H4``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tools.verification.evaluate_mlt_top_slope_calibration as ev  # noqa: E402
from aragog import surface_lid as sl  # noqa: E402
from aragog.solver import entropy_solver as es_mod  # noqa: E402

POWERS = ('F_int', 'F_cmb', 'Q_radio', 'Q_tidal')


def lid_subledger(s, t: np.ndarray, y: np.ndarray) -> dict:
    """Lid excess heat change from the states and the trapezoid integral of its rate, up to the
    last node that carries the lid, for the trace ``t`` [yr], ``y`` of one call."""
    last = int(np.flatnonzero(y[-2] > 0.0)[-1])
    A, p, P = s._lid_area, s._lid_props, s._lid_P
    rate = np.empty(last + 1)
    for i in range(last + 1):
        dS = np.asarray(s._dSdt_single(float(t[i]), y[:, i])).ravel() / es_mod.SECS_PER_YEAR
        e = s._lid_last
        S_f = float(s._stag_entropy(y[:, i])[-1])
        rhoT = float(p.density(P, S_f)) * float(p.temperature(P, S_f))
        dS_f = float(s._stag_entropy(dS)[-1])
        rate[i] = A * (
            e.rates.dE - float(s._lid_W(S_f)) * e.rates.ddelta - e.delta * rhoT * dS_f
        )
    ts = t[: last + 1] * es_mod.SECS_PER_YEAR
    integral = float(np.sum(0.5 * (rate[:-1] + rate[1:]) * np.diff(ts)))
    change = float(s._lid_heat_change(y[:, 0], y[:, last]))
    return {
        't_last_lid': float(t[last]),
        'dX': change,
        'dX_rate_integral': integral,
        'miss_rel': (change - integral) / abs(change) if change else 0.0,
    }


def _solver_class(lid: bool, closure: str, rec: dict):
    class LidHarnessSolver(es_mod.EntropySolver):
        def __init__(self, p, **kw):
            rheo = p.phase_solid.rheology
            p.phase_solid.rheology = dataclasses.replace(rheo, stress_closure_mode=closure)
            p.boundary_conditions.subgrid_lid = lid
            p.boundary_conditions.normalize()
            super().__init__(p, **kw)

        def solve(self, *a, **k):
            out = super().solve(*a, **k)
            sol = self._solution
            ei = sol.energy_integrals
            dX = float(self._lid_heat_change(sol.y[:, 0], sol.y[:, -1])) if lid else 0.0
            miss = ei['state_heat'] - sum(ei[n] for n in POWERS)
            call = {
                't0': float(sol.t[0]),
                't1': float(sol.t[-1]),
                'guard': abs(miss) / max(sum(abs(ei[n]) for n in POWERS) + abs(dX), 1.0),
            }
            if lid and float(sol.y[-2, 0]) > 0.0:
                t, y = (
                    np.asarray(v, dtype=float)
                    for v in (sol.get('energy_trace') or (sol.t, sol.y))
                )
                call['lid'] = lid_subledger(self, t, y)
            rec['calls'].append(call)
            h = self.get_state().lid_handover_time if lid else float('nan')
            if rec['handover_t'] is None and np.isfinite(h):
                rec['handover_t'] = float(h)
            return out

        def _check_lid_step(self, *a):
            e = self._lid_last
            rec['checked_steps'] += 1
            rec['floor_steps'] += int((e.H_f - e.H_b) / e.rho < sl.LID_DH_MIN)
            return super()._check_lid_step(*a)

    return LidHarnessSolver


def run_row(
    table: str,
    row: str,
    lid: bool,
    closure: str,
    n_nodes: int = 100,
    s_top_inf: float = 0.25,
    c_theta: float = 0.25,
) -> dict:
    """Run one row ('K' Korenaga 2009, 'D' Deschamps and Vilella 2021) with the lid on or off."""
    rec = {'calls': [], 'handover_t': None, 'floor_steps': 0, 'checked_steps': 0}
    install, solver = ev._install_benchmark_patches, ev.EntropySolver

    def install_with_ledger():
        install()
        es_mod.EntropySolver._compute_step_energy_integrals = ev._orig_energy_integrals

    ev._install_benchmark_patches, ev.EntropySolver = (
        install_with_ledger,
        _solver_class(lid, closure, rec),
    )
    run = ev.run_korenaga_row if table == 'K' else ev.run_deschamps_vilella_row
    t0 = time.time()
    try:
        result = run(row, n_nodes, s_top_inf, c_theta)
    finally:
        ev._install_benchmark_patches, ev.EntropySolver = install, solver
    keep = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}
    return {
        'table': table,
        'row': row,
        'lid': lid,
        'closure': closure,
        'wall_s': time.time() - t0,
        'result': keep,
        **rec,
    }


def main(argv: list[str]) -> int:
    out, lid, closure, rows = Path(argv[0]), argv[1] == 'on', argv[2], argv[3:]
    with out.open('a') as f:
        for tr in rows:
            table, row = tr.split(':', 1)
            f.write(json.dumps(run_row(table, row, lid, closure)) + '\n')
            f.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
