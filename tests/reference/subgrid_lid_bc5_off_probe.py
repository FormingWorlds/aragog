"""Probe of the lid-off solve path under outer boundary 5, for the off-path reference.

``probe(rheology)`` builds the 20-node constant-property column of
``tests/test_entropy_bc5_surface_temperature.py`` (surface at 300 K), without or with the solid
rheology (the 'lid' stress closure), and returns one right-hand side at the start state and a
5-yr solve. Run as a script with the code of the reference commit first on the path, it writes
``tests/reference/subgrid_lid_bc5_off_reference.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CASES = [False, True]


def case_name(rheology: bool) -> str:
    return 'rheo' if rheology else 'plain'


def probe(rheology: bool) -> dict:
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_entropy_bc5_surface_temperature import _solver

    s = _solver()
    p = s.parameters
    if rheology:
        p.phase_solid.rheology = SolidRheologyParams(enabled=True, mlt_bottom_slope=1.0)
        S0 = np.array(s._S0, dtype=float)
        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        s.set_initial_entropy(S0)
    rhs = np.asarray(s.dSdt(0.0, np.asarray(s._S0, dtype=float)), dtype=float).ravel()
    s.solve()
    sol = s._solution
    return {
        'rhs': rhs,
        'status': np.array([sol.status]),
        't': np.asarray(sol.t, dtype=float),
        'y_end': np.asarray(sol.y, dtype=float)[:, -1],
    }


if __name__ == '__main__':
    out = Path(sys.argv[1])
    data = {f'{case_name(r)}/{k}': v for r in CASES for k, v in probe(r).items()}
    np.savez_compressed(out, **data)
    print(f'wrote {out}: {sorted(data)}')
