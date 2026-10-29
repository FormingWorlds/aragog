"""Coupled PROTEUS run of the step-controller case V5b, with its stop rule and its reference.

The case config is ``tools/verification/data/v5b_zalmoxis_aragog_calliope.toml``; the case was
run with PROTEUS a8bbd8751cb5040aedd90f7acb4a85755afae26b. Two settings of the case are not
PROTEUS config options, so this driver sets them before the run starts:

- the run stops 3 coupling steps after ``Phi_global`` first reaches 0
  (``proteus.utils.terminate.check_termination``);
- ``--reference`` integrates every rate-cap segment that starts with no cell above the solidus at
  rtol = 1e-10, atol = 1e-10 / state scale and 1 yr steps (``EntropySolver._solve_cvode``).

The driver stops with an error when one of these hooks is missing in the installed PROTEUS or
Aragog. It needs PROTEUS and its data (``FWL_DATA``). Run from the repository root::

    python tools/verification/run_v5b_proteus.py [--reference] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import toml

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


def install(reference):
    """Patch the stop rule and, with ``reference``, the Regime 4 segment setting."""
    import proteus.utils.terminate as term

    import aragog.solver.entropy_solver as es

    check = hook(term, 'check_termination')
    solve_cvode = hook(es.EntropySolver, '_solve_cvode')
    root_type = hook(es, '_PhaseBoundarySegmentRoot')
    solid = [0]

    def check_termination(handler):
        if float(handler.hf_row.get('Phi_global', 1.0)) == 0.0:
            solid[0] += 1
            if solid[0] >= SOLID_STEPS:
                handler.finished_both = handler.finished_prev = True
                return True
        return check(handler)

    def regime4_reference(self, start_time, end_time, y0, atol, rtol, max_step, **kw):
        root = kw.get('phi_cap_rootfn')
        if isinstance(root, root_type) and np.all(root.S0 <= root.S_sol):
            sc = self._build_nondim_scales()
            rtol, atol, max_step = 1.0e-10, 1.0e-10 / sc.state_scale, 1.0 / sc.t_ref
        return solve_cvode(self, start_time, end_time, y0, atol, rtol, max_step, **kw)

    term.check_termination = check_termination  # PROTEUS imports it inside Proteus.start()
    if reference:
        es.EntropySolver._solve_cvode = regime4_reference
    return solid


def main(argv=None) -> int:
    """Run V5b once and print the last helpfile row."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args(argv)
    out = (args.output_dir or Path(tempfile.mkdtemp(prefix='v5b_'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = tomllib.loads(CONFIG.read_text())
    cfg['params']['out']['path'] = str(out / 'output')
    (out / 'config.toml').write_text(toml.dumps(cfg))

    solid = install(args.reference)
    from proteus import Proteus

    runner = Proteus(config_path=str(out / 'config.toml'))
    runner.start(offline=True)
    row = runner.hf_all.iloc[-1]
    result = dict(
        reference=args.reference,
        solid_steps=solid[0],
        rows=len(runner.hf_all),
        **{k: float(row[k]) for k in COLUMNS},
    )
    print(json.dumps(result))
    (out / 'v5b_result.json').write_text(json.dumps(result) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
