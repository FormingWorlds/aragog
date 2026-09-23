"""Subprocess tests verifying import origin and lazy loading.

Checks that:
1. `import aragog` resolves to the local worktree src/ directory.
2. `import aragog` does not eagerly import jax or equinox into sys.modules.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_aragog_import_origin():
    """Verify import aragog resolves to the repository src/ directory."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    expected_init = (repo_root / 'src' / 'aragog' / '__init__.py').resolve()

    code = (
        'import aragog, pathlib; '
        'origin = pathlib.Path(aragog.__file__).resolve(); '
        f'expected = pathlib.Path({str(expected_init)!r}); '
        "assert origin == expected, f'Expected origin {expected}, got {origin}'"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f'Import origin check failed:\n{result.stderr}'


def test_aragog_does_not_eagerly_import_jax():
    """Verify import aragog does not load jax or equinox into sys.modules."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    code = (
        'import sys, aragog; '
        "assert 'jax' not in sys.modules, 'jax leaked into sys.modules on import aragog'; "
        "assert 'equinox' not in sys.modules, 'equinox leaked into sys.modules on import aragog'"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f'Lazy import check failed:\n{result.stderr}'
