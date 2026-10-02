"""Accuracy of ``phase_boundary_cap = 'rate'`` against a tight ``'fixed'`` run.

Computes the numbers in the accuracy table of ``docs/Explanations/energy_equation.md``.
Each case integrates one solver call with the CVODE entropy solver on the
test EOS (``ARAGOG_TEST_EOS_DIR``) and compares the cell temperatures at the
call end against ``'fixed'`` at rtol = atol = 1e-10:

====  ==============  ========  =====================  ========  ======  =====
Case  core_bc         Call [yr] Start                  F [W/m2]  w       Nodes
====  ==============  ========  =====================  ========  ======  =====
A     quasi_steady    1000      mushy (uniform)        1e4       0.01    24
B     quasi_steady    1e4       S_liq(P) + 230 J/kg/K  1e5       0.01    24
C     quasi_steady    2000      mushy (uniform)        1e4       0       24
D     energy_balance  3000      mushy (uniform)        1e4       0.01    16
====  ==============  ========  =====================  ========  ======  =====

The mushy start is the midpoint of the solidus and liquidus entropy at
140 GPa, in every cell. F is a prescribed surface flux
(``outer_boundary_condition = 4``); w is ``matprop_smooth_width``. The other
parameters are those of ``_build_mushy_parameters`` in
``tests/test_phi_step_cap_armed_smoke.py`` with ``phi_step_cap`` off and
``max_steps`` 1e7. Run from the repository root::

    ARAGOG_TEST_EOS_DIR=<dir> python tools/verification/run_phase_boundary_cap_accuracy.py [--check] [A B C D]

``--check`` exits with status 1 when ``'rate'`` at 1e-8 is more than ``RATE_BOUND_K``
from the reference in any case.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from tests.test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S

CASES = {
    'A': dict(core='quasi_steady', end=1000.0, flux=1e4, width=0.01, start='mushy', n=24),
    'B': dict(core='quasi_steady', end=1e4, flux=1e5, width=0.01, start='liq+230', n=24),
    'C': dict(core='quasi_steady', end=2000.0, flux=1e4, width=0.0, start='mushy', n=24),
    'D': dict(core='energy_balance', end=3000.0, flux=1e4, width=0.01, start='mushy', n=16),
}
RUNS = (('fixed', 1e-8), ('rate', 1e-8), ('rate', 1e-10))
RATE_BOUND_K = 1.3e-4  # the bound the energy-equation docs state for 'rate' at 1e-8


def run_case(eos, case, mode, tol):
    """Integrate one call of ``case``; return the end temperatures [K] and the CVODE step count."""
    p = _build_mushy_parameters(solver_method='cvode', n_nodes=case['n'], end_time=case['end'])
    p.energy = dataclasses.replace(p.energy, phase_boundary_cap=mode, phi_step_cap=None)
    p.boundary_conditions = dataclasses.replace(
        p.boundary_conditions,
        outer_boundary_condition=4,
        outer_boundary_value=case['flux'],
        core_bc=case['core'],
    )
    p.phase_mixed = dataclasses.replace(p.phase_mixed, matprop_smooth_width=case['width'])
    p.solver.rtol = p.solver.atol = tol
    p.solver.max_steps = 10_000_000
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    if case['start'] == 'mushy':
        s.set_initial_entropy(_pick_mushy_S(eos))
    else:
        s.set_initial_entropy(np.asarray(eos.liquidus_entropy(s._P_stag_flat)).ravel() + 230.0)
    s.solve()
    sol = s._solution
    if sol.status != 0:
        raise RuntimeError(f'{mode} at {tol:g}: solver status {sol.status}')
    T = np.asarray(eos.temperature(s._P_stag_flat, np.asarray(sol.y)[: s._n_stag, -1])).ravel()
    return T, int(sol.get('cvode_nst') or 0)


def main(names, check=False):
    """Print one markdown table row per case; return 1 when ``check`` and a bound fails."""
    eos_dir = os.environ.get('ARAGOG_TEST_EOS_DIR')
    if not eos_dir:
        sys.exit(
            'ARAGOG_TEST_EOS_DIR is not set: point it at the test EOS (SPIDER P-S tables).'
        )
    eos = EntropyEOS(eos_dir)
    print(
        r'| Case | Call [yr] | steps fixed / rate at 1e-8 | max \|dT\| fixed 1e-8 | rate 1e-8 '
        '| rate 1e-10 |'
    )
    print('|---|---|---|---|---|---|')
    failed = []
    for name in names:
        case = CASES[name]
        T_ref, _ = run_case(eos, case, 'fixed', 1e-10)
        dT, nst = [], []
        for mode, tol in RUNS:
            T, n = run_case(eos, case, mode, tol)
            dT.append(float(np.abs(T - T_ref).max()))
            nst.append(n)
        print(
            f'| {name} | {case["end"]:g} | {nst[0]} / {nst[1]} | '
            + ' | '.join(f'{d:.2g} K' for d in dT)
            + ' |',
            flush=True,
        )
        if dT[1] > RATE_BOUND_K:
            failed.append(name)
    if check and failed:
        print(f'rate at 1e-8 exceeds {RATE_BOUND_K:g} K in case(s) {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--check']
    sys.exit(main(args or list(CASES), check='--check' in sys.argv[1:]))
