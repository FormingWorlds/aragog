"""Heated sub-solidus cases of the step controller (``phase_boundary_cap = 'rate'``).

Computes the numbers of the cases "Heated sub-solidus segment that ends at the solidus" and
"Long steps while a cell heats to the solidus" in ``docs/Validation/numerics.md``. Every run is
a 12-node column on the CI test table whose top cell starts a gap below its solidus
(``_heated_subsolidus`` of ``tests/test_phase_boundary_cap.py``). The cases:

- ``heated``: 400 W/m^2 into the top, the other cells 300 J/(kg K) below their solidus, the
  top cell 2 below, 10 kyr.
- ``long_steps``: an isothermal column with 1e-8 W/kg internal heating and an insulated top,
  the top cell 2 below its solidus, 15 kyr.
- ``sweep``: ``heated`` with top gaps 0, 0.5, 2 and 10 J/(kg K), and with 200 and 800 W/m^2 at
  gap 2 (a known limit of the case).
- ``limit``: ``heated`` with top gaps 80, 100 and 120 J/(kg K); the cell stays below its solidus
  (a known limit of the case); gap 100 also with the rate run at rtol = atol = 1e-10.

Per run: the largest |T - T_ref| over the output times both runs share and at the end, the
largest CVODE step before the solidus, the solidus root time and the segments; the time of
the largest error. Each run
compares ``'rate'`` at rtol = atol = 1e-8 with ``'fixed'`` (1 yr steps) at rtol 1e-10 and a
physical atol of 1e-10. ``heated`` and ``long_steps`` also run 5 starts S0 (1 + m 1e-12),
m = -2..2, for the round-off envelope (the largest difference of a perturbed run from the
m = 0 run over the shared output times). With ``--check`` the tool exits 1 when ``heated`` or
``long_steps`` is above 1.3e-4 K, or when ``long_steps`` takes no step of 1000 yr or more
before the solidus; ``sweep`` and ``limit`` only report. Run from the repository root::

    python tools/verification/run_subsolidus_heating_cases.py --case heated [--check]

``--eos-dir`` defaults to ``ARAGOG_TEST_EOS_DIR``. The result goes to ``--output`` (default
``subsolidus_heating_<case>.json`` in the working directory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from aragog.eos.entropy import EntropyEOS  # noqa: E402

BOUND_K = 1.3e-4  # the V3-R4 tolerance of the sub-solidus controller
LONG_STEP_YR = 1000.0
RUNS = {
    'heated': [dict(gap=2.0)],
    'long_steps': [dict(gap=2.0, isothermal=True)],
    'sweep': [dict(gap=g) for g in (0.0, 0.5, 2.0, 10.0)]
    + [dict(gap=2.0, flux=f) for f in (200.0, 800.0)],
    'limit': [dict(gap=g) for g in (80.0, 100.0, 120.0)] + [dict(gap=100.0, rate_tol=1e-10)],
}


def one(eos, run):
    """Integrate one run and its reference; return its metrics."""
    from tests.test_phase_boundary_cap import (
        _heated_subsolidus,
        _largest_step_before,
        _max_dT_at_shared_outputs,
    )

    kw = {k: v for k, v in run.items() if k != 'rate_tol'}
    t0 = time.perf_counter()
    ref, t_ref, _, _, T_ref = _heated_subsolidus(eos, 'fixed', 1e-10, **kw)
    wall_ref = time.perf_counter() - t0
    s, t, S, S_sol, T = _heated_subsolidus(eos, 'rate', run.get('rate_tol', 1e-8), **kw)
    for res, name in ((ref._solution, 'fixed'), (s._solution, 'rate')):
        if res.status != 0:
            raise RuntimeError(f'{run} {name}: solver status {res.status}')
    segs = [(float(a), b, float(c)) for a, b, c in s._solution.segments]
    root = next((a for a, b, _ in segs if b == 'solidus'), None)
    shared = [(i, int(np.argmin(np.abs(t_ref - x)))) for i, x in enumerate(t)]
    worst = max(
        (float(np.abs(T[:, i] - T_ref[:, j]).max()), float(t[i]))
        for i, j in shared
        if abs(t_ref[j] - t[i]) <= 1e-9 * max(1.0, t[i])
    )
    return dict(
        run=run,
        max_dT_K=_max_dT_at_shared_outputs((t_ref, T_ref), (t, T)),
        max_dT_time_yr=worst[1],
        end_dT_K=float(np.abs(T[:, -1] - T_ref[:, -1]).max()),
        largest_step_before_solidus_yr=_largest_step_before(s, root if root else t[-1]),
        solidus_root_yr=root,
        segments=segs,
        steps=int(s._solution.cvode_nst),
        reference_steps=int(ref._solution.cvode_nst),
        reference_wall_s=wall_ref,
        top_end_above_solidus=float(S[-1, -1] - S_sol[-1]),
    )


def envelope(eos, run):
    """Round-off envelope [K]: largest difference of the 5 starts from the m = 0 start."""
    from tests.test_phase_boundary_cap import _heated_subsolidus, _max_dT_at_shared_outputs

    out = {}
    for m in (-2, -1, 0, 1, 2):
        _, t, _, _, T = _heated_subsolidus(eos, 'rate', 1e-8, eps=m * 1e-12, **run)
        out[m] = (t, T)
    return max(_max_dT_at_shared_outputs(out[0], out[m]) for m in (-2, -1, 1, 2))


def check(case, result):
    """Return the failed tolerances of the docs page for one case result."""
    if case in ('sweep', 'limit'):
        return []
    fails = [
        f'{r["run"]}: |dT| > {BOUND_K} K' for r in result['runs'] if r['max_dT_K'] > BOUND_K
    ]
    if case == 'long_steps':
        r = result['runs'][0]
        fails += [] if r['largest_step_before_solidus_yr'] >= LONG_STEP_YR else ['no long step']
    return fails


def main(argv=None) -> int:
    """Run one case, write its JSON result and print the result line."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--case', required=True, choices=list(RUNS))
    ap.add_argument('--eos-dir', default=os.environ.get('ARAGOG_TEST_EOS_DIR'))
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--check', action='store_true', help='exit 1 when a tolerance fails')
    args = ap.parse_args(argv)
    if not args.eos_dir:
        ap.error('--eos-dir or ARAGOG_TEST_EOS_DIR is required')
    os.environ['ARAGOG_TEST_EOS_DIR'] = str(args.eos_dir)
    eos = EntropyEOS(str(args.eos_dir))
    result = dict(case=args.case, runs=[one(eos, run) for run in RUNS[args.case]])
    if args.case in ('heated', 'long_steps'):
        result['round_off_envelope_K'] = envelope(eos, RUNS[args.case][0])
    for r in result['runs']:
        print(json.dumps({k: r[k] for k in r if k != 'segments'}))
    if 'round_off_envelope_K' in result:
        print(f'round-off envelope {result["round_off_envelope_K"]:.3e} K')
    out = args.output or Path(f'subsolidus_heating_{args.case}.json')
    out.write_text(json.dumps(dict(eos_dir=str(args.eos_dir), **result)) + '\n')
    fails = check(args.case, result)
    print(f'{args.case}: ' + ('FAIL ' + ', '.join(fails) if fails else 'PASS'))
    return 1 if args.check and fails else 0


if __name__ == '__main__':
    sys.exit(main())
