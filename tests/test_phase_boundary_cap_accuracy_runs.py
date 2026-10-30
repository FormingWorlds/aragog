"""``tools/verification/run_phase_boundary_cap_accuracy.py``: short real runs of the reference."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from .conftest import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, needs_eos]

_spec = importlib.util.spec_from_file_location(
    'run_phase_boundary_cap_accuracy',
    Path(__file__).resolve().parents[1]
    / 'tools'
    / 'verification'
    / 'run_phase_boundary_cap_accuracy.py',
)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
es = sys.modules[v1.EntropySolver.__module__]


@pytest.mark.skipif(not es._CVODE_AVAILABLE, reason='scikits.odes CVODE not installed')
def test_reference_caps_every_cvode_solve_at_one_year(monkeypatch):
    """The reference of a short case B solves at max_step <= 1 yr (100 yr without one_year)."""
    seen = []
    solve = v1.EntropySolver._solve_cvode

    def record(self, *args, **kw):
        seen.append(kw['max_step'] * self._build_nondim_scales().t_ref)
        return solve(self, *args, **kw)

    monkeypatch.setattr(v1.EntropySolver, '_solve_cvode', record)
    case = dict(v1.CASES['B'], end=300.0, n=12)
    v1.run_case(v1.EntropyEOS(str(EOS_DIR)), case, 'fixed', 1e-10, one_year=True)
    assert seen and max(seen) <= 1.0 + 1e-12


def test_reference_refuses_a_run_without_cvode(monkeypatch):
    """Without CVODE the solver falls back to Radau; the 1 yr reference then stops."""
    monkeypatch.setattr(es, '_CVODE_AVAILABLE', False)
    with pytest.raises(RuntimeError, match='the 1 yr reference ran no CVODE solve'):
        case = dict(v1.CASES['A'], end=5.0, n=12)
        v1.run_case(v1.EntropyEOS(str(EOS_DIR)), case, 'fixed', 1e-10, one_year=True)
