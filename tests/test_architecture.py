from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent / 'src' / 'aragog'


@pytest.mark.unit
def test_rheology_imports():
    """rheology.py imports only approved core math/typing/numpy libraries."""
    rheo_file = SRC_ROOT / 'rheology.py'
    tree = ast.parse(rheo_file.read_text(), filename=str(rheo_file))

    allowed = {'numpy', 'numpy.typing', 'math', 'dataclasses', 'typing', '__future__'}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                assert root in allowed, f'Unapproved import in rheology.py: {alias.name}'
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            root = mod.split('.')[0]
            assert root in allowed, f'Unapproved import-from in rheology.py: {mod}'


@pytest.mark.unit
def test_eos_and_jax_do_not_import_solver():
    """No module under aragog/eos/ or aragog/jax/ imports aragog.solver."""
    for sub in ['eos', 'jax']:
        sub_dir = SRC_ROOT / sub
        for py_file in sub_dir.glob('**/*.py'):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith('aragog.solver'), (
                            f'{py_file} imports {alias.name}'
                        )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ''
                    assert not mod.startswith('aragog.solver'), f'{py_file} imports from {mod}'


@pytest.mark.unit
def test_cvode_jax_all_exports():
    """cvode_jax.__all__ matches the fixed set of public re-exports."""
    from aragog.solver import cvode_jax

    expected = {
        'build_jax_rhs_and_jacobian',
        'verify_jax_vs_numpy_rhs',
    }
    assert set(cvode_jax.__all__) == expected


@pytest.mark.unit
def test_aragog_import_is_clean_of_jax():
    """Subprocess check that import aragog leaves jax and equinox out of sys.modules."""
    code = (
        'import sys\n'
        'import aragog\n'
        "assert 'jax' not in sys.modules, f'jax leaked: {sys.modules[\"jax\"]}'\n"
        "assert 'equinox' not in sys.modules, f'equinox leaked: {sys.modules[\"equinox\"]}'\n"
    )
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert res.returncode == 0, f'Import test failed:\n{res.stderr}'


@pytest.mark.unit
def test_top_level_modules_list():
    """Literal list of top-level modules in aragog."""
    expected_modules = {
        '__init__.py',
        'cfg',
        'cli.py',
        'config',
        'eos',
        'jax',
        'mesh',
        'output',
        'parser.py',
        'rheology.py',
        'solver',
        'utilities.py',
    }
    actual = {
        p.name
        for p in SRC_ROOT.iterdir()
        if p.name not in ('__pycache__', '_version.py') and not p.name.startswith('.')
    }
    assert actual == expected_modules
