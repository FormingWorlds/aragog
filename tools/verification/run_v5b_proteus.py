"""Coupled PROTEUS run of the step-controller case V5b, with its stop rule and its reference.

The case config is ``tools/verification/data/v5b_zalmoxis_aragog_calliope.toml``; its header
records the PROTEUS commit of the case. The driver sets case settings that are not PROTEUS
config options before the run starts:

- the run stops at the 3rd coupling step with ``Phi_global`` = 0
  (``proteus.utils.terminate.check_termination``);
- ``--reference`` integrates every CVODE call at ``--reference-rtol`` (default 1e-10), a
  physical atol of the same value and 1 yr steps (``EntropySolver._solve_cvode``), so the
  reference does not depend on the step controller;
- ``--s0-eps`` multiplies the state at the first CVODE call by 1 + eps for a round-off start
  (PROTEUS sets the start entropy more than once before the integration starts).

The result file records the CVODE steps of all calls and the wall time of the run. The driver
stops with an error when one of these hooks is missing in the installed PROTEUS or Aragog. It
needs PROTEUS and its data (``FWL_DATA``). Run from the repository root::

    python tools/verification/run_v5b_proteus.py [--reference] [--s0-eps EPS] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parent / 'data' / 'v5b_zalmoxis_aragog_calliope.toml'
SOLID_STEPS = 3
COLUMNS = (
    'Time',
    'T_magma',
    'T_surf',
    'F_int',
    'F_atm',
    'Phi_global',
    'M_mantle_liquid',
    'M_mantle_solid',
    'R_int',
)


def hook(owner, name):
    """Return ``owner.name``; stop when it is missing, so the run cannot drop a case setting."""
    fn = getattr(owner, name, None)
    if fn is None:
        raise SystemExit(
            f'{getattr(owner, "__name__", owner)}.{name} is missing: V5b cannot run'
        )
    return fn


def install(reference, s0_eps=0.0, reference_rtol=1.0e-10):
    """Patch the stop rule, count the CVODE steps, and set the reference and start options."""
    import proteus.utils.terminate as term

    import aragog.solver.entropy_solver as es

    check = hook(term, 'check_termination')
    solve_cvode = hook(es.EntropySolver, '_solve_cvode')
    solid, steps, first = [0], [0], [True]

    def check_termination(handler):
        if float(handler.hf_row.get('Phi_global', 1.0)) == 0.0:
            solid[0] += 1
            if solid[0] >= SOLID_STEPS:
                handler.finished_both = handler.finished_prev = True
                return True
        return check(handler)

    def counted(self, start_time, end_time, y0, atol, rtol, max_step, **kw):
        if first[0]:
            first[0] = False
            y0 = np.asarray(y0, dtype=float) * (1.0 + s0_eps)
        if reference:
            sc = self._build_nondim_scales()
            rtol, atol = reference_rtol, reference_rtol / sc.state_scale
            max_step = 1.0 / sc.t_ref
        res = solve_cvode(self, start_time, end_time, y0, atol, rtol, max_step, **kw)
        steps[0] += int(res.get('cvode_nst', 0))
        return res

    term.check_termination = check_termination  # PROTEUS imports it inside Proteus.start()
    es.EntropySolver._solve_cvode = counted
    return solid, steps


def main(argv=None) -> int:
    """Run V5b once and print the last helpfile row."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--reference-rtol', type=float, default=1.0e-10)
    ap.add_argument('--s0-eps', type=float, default=0.0)
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args(argv)
    out = (args.output_dir or Path(tempfile.mkdtemp(prefix='v5b_'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import toml  # a PROTEUS dependency; Aragog itself does not install it

    cfg = tomllib.loads(CONFIG.read_text())
    cfg['params']['out']['path'] = str(out / 'output')
    (out / 'config.toml').write_text(toml.dumps(cfg))

    solid, steps = install(args.reference, args.s0_eps, args.reference_rtol)
    from proteus import Proteus

    runner = Proteus(config_path=str(out / 'config.toml'))
    t0 = time.perf_counter()
    runner.start(offline=True)
    row = runner.hf_all.iloc[-1]
    result = dict(
        reference=args.reference,
        reference_rtol=args.reference_rtol if args.reference else None,
        s0_eps=args.s0_eps,
        cvode_steps=steps[0],
        wall_s=time.perf_counter() - t0,
        solid_steps=solid[0],
        rows=len(runner.hf_all),
        **{k: float(row[k]) for k in COLUMNS},
    )
    print(json.dumps(result))
    (out / 'v5b_result.json').write_text(json.dumps(result) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
