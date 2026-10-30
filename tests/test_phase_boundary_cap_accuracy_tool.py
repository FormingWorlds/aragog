"""``tools/verification/run_phase_boundary_cap_accuracy.py``: the 1 yr reference of the accuracy table."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'run_phase_boundary_cap_accuracy',
    Path(__file__).resolve().parents[1]
    / 'tools'
    / 'verification'
    / 'run_phase_boundary_cap_accuracy.py',
)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)


def test_one_year_steps_replace_every_max_step():
    """Each CVODE solve runs at max_step = 1 yr / t_ref, whatever the caller passes."""
    seen = []
    s = types.SimpleNamespace(
        _solve_cvode=lambda *a, **kw: seen.append(kw['max_step']),
        _build_nondim_scales=lambda: types.SimpleNamespace(t_ref=0.25),
    )
    v1._one_year_steps(s)
    for h in (31557.6, np.inf, 0.1):
        s._solve_cvode(0.0, 1.0, max_step=h)
    assert seen == [4.0, 4.0, 4.0]


def test_main_runs_only_the_reference_at_one_year(monkeypatch):
    """The 'fixed' 1e-10 reference forces 1 yr steps; the three compared runs do not."""
    calls = []

    def run_case(eos, case, mode, tol, one_year=False):
        calls.append((mode, tol, one_year))
        return np.zeros(2), 1, 1

    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', 'E')
    monkeypatch.setattr(v1, 'EntropyEOS', lambda _eos_dir: None)
    monkeypatch.setattr(v1, 'run_case', run_case)
    assert v1.main(['A']) == 0
    assert calls == [
        ('fixed', 1e-10, True),
        ('fixed', 1e-8, False),
        ('rate', 1e-8, False),
        ('rate', 1e-10, False),
    ]
