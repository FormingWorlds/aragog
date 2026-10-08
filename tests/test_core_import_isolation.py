"""The energy_balance paths load no part of ``aragog.core`` at import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_energy_balance_paths_load_no_core_module_at_import():
    """The numpy solver and the JAX RHS load no part of ``aragog.core``, and ``aragog.core``
    loads the experimental layer only when a stratified budget is built."""
    code = (
        'import sys, aragog.jax.solver, aragog.solver.entropy_solver, aragog.solver.cvode_jax\n'
        "print(sorted(m for m in sys.modules if m.startswith('aragog.core')))\n"
        'import aragog.core\n'
        "print('aragog.core.layer' in sys.modules)"
    )
    src = str(Path(__file__).resolve().parents[1] / 'src')
    env = os.environ | {'PYTHONPATH': src}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['[]', 'False']
