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


@pytest.fixture
def fake_main(monkeypatch):
    """Run ``main`` with the given ``run_case`` and no EOS table."""
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', 'E')
    monkeypatch.setattr(v1, 'EntropyEOS', lambda _eos_dir: None)
    return lambda run_case: monkeypatch.setattr(v1, 'run_case', run_case)


def test_main_runs_only_the_reference_at_one_year(fake_main):
    """The 'fixed' 1e-10 reference forces max_step 1 yr; the three compared runs do not."""
    calls = []

    def run_case(eos, case, mode, tol, one_year=False):
        calls.append((mode, tol, one_year))
        return np.zeros(2), 1, 1

    fake_main(run_case)
    assert v1.main(['A']) == 0
    assert calls == [
        ('fixed', 1e-10, True),
        ('fixed', 1e-8, False),
        ('rate', 1e-8, False),
        ('rate', 1e-10, False),
    ]


@pytest.mark.parametrize(
    'scale, check, code', [(2.0, False, 0), (2.0, True, 1), (0.5, True, 0)]
)
def test_main_check_exits_1_only_when_rate_exceeds_the_bound(
    fake_main, capsys, scale, check, code
):
    """With --check, a 'rate' run at 1e-8 above the bound names its case and returns 1."""

    def run_case(eos, case, mode, tol, one_year=False):
        T = np.full(2, v1.RATE_BOUND_K * scale if (mode, tol) == ('rate', 1e-8) else 0.0)
        return T, 1, 1

    fake_main(run_case)
    assert v1.main(['A', 'B'], check=check) == code
    assert ('case(s) A, B' in capsys.readouterr().out) == bool(code)
