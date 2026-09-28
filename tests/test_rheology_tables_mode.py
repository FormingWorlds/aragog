"""Verification test for solid rheology in tables mode (Step 3).

Verifies that EntropySolver with EOS lookup tables (const_properties=False),
solid-state rheology enabled in 'lid' stress closure mode, runs for 1 Myr
and yields a finite, depth-monotonic solid viscosity profile.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from aragog.eos.entropy import EntropyEOS
from aragog.rheology import SolidRheologyParams
from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_integration import _build_parameters

_FWL_DATA = os.environ.get('FWL_DATA')
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=(
        f'SPIDER P-S tables not found at {EOS_DIR}. Set ARAGOG_TEST_EOS_DIR '
        'or populate $FWL_DATA/aragog/spider_eos.'
    ),
)

pytestmark = [pytest.mark.smoke, needs_eos]


@pytest.mark.smoke
def test_rheology_tables_mode_1myr_solve():
    """Verify 1 Myr solve in tables mode with lid rheology enabled.

    Requires:
    - const_properties = False (SPIDER EOS lookup tables).
    - enabled = True with stress_closure_mode = 'lid'.
    - 1 Myr integration time completes.
    - Solid viscosity profile is finite and monotonic in depth.
    """
    eos = EntropyEOS(EOS_DIR)
    params = _build_parameters(end_time=1.0e6, n_nodes=15)
    params.phase_solid.rheology = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params.boundary_conditions.outer_boundary_condition = 4
    params.boundary_conditions.outer_boundary_value = 0.0  # insulating surface (zero flux)

    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    # Subsolidus mantle entropy (T ~ 1500-2300 K) for solid-state convection
    solver.set_initial_entropy(400.0)
    solver.solve()

    sol = solver._solution
    assert sol.status == 0, (
        f'Integration failed with status {sol.status}: {getattr(sol, "message", None)}'
    )
    assert np.isclose(sol.t[-1], 1.0e6, rtol=1e-5), f'Expected t_final 1e6 yr, got {sol.t[-1]}'

    state = solver.get_state()
    eta_diff = state.eta_diff_b
    visc_eff = state.visc_eff_b

    # Verify solid viscosity is finite throughout the mantle
    assert np.all(np.isfinite(eta_diff)), 'eta_diff_b contains non-finite values'
    assert np.all(np.isfinite(visc_eff)), 'visc_eff_b contains non-finite values'
    assert np.all(eta_diff > 0.0), 'eta_diff_b must be strictly positive'

    # Verify monotonic variation in depth (from CMB to surface, eta_diff decreases)
    diffs = np.diff(eta_diff)
    assert np.all(diffs <= 0.0), f'eta_diff_b is not monotonic with depth: diffs={diffs}'
