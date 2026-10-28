"""Refusals of tools/verification/generate_rheology_off_reference.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'generate_rheology_off_reference',
    Path(__file__).resolve().parents[1]
    / 'tools'
    / 'verification'
    / 'generate_rheology_off_reference.py',
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _git(tree: Path, *args: str) -> str:
    """Run git in ``tree`` with a fixed identity and return its stdout."""
    cmd = ['git', '-C', str(tree), '-c', 'user.name=t', '-c', 'user.email=t@t', *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def git_tree(tmp_path: Path) -> Path:
    """A git tree with one committed file."""
    tree = tmp_path / 'main_tree'
    tree.mkdir()
    _git(tree, 'init', '-q')
    (tree / 'a.py').write_text('x = 1\n')
    _git(tree, 'add', 'a.py')
    _git(tree, 'commit', '-q', '-m', 'init')
    return tree


def test_clean_tree_gives_head_sha(git_tree: Path) -> None:
    """A clean tree root resolves to its HEAD SHA."""
    assert (
        gen._resolve_commit_sha(git_tree, None) == _git(git_tree, 'rev-parse', 'HEAD').strip()
    )


@pytest.mark.parametrize('change', ['modified', 'untracked'])
def test_dirty_tree_is_refused(git_tree: Path, change: str) -> None:
    """A modified or untracked file in the main tree is refused."""
    if change == 'modified':
        (git_tree / 'a.py').write_text('x = 2\n')
    else:
        (git_tree / 'b.py').write_text('y = 1\n')
    with pytest.raises(ValueError, match='uncommitted changes'):
        gen._resolve_commit_sha(git_tree, None)


def test_dirty_tree_gives_rc1_and_no_file(
    git_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 1 and writes nothing for a dirty main tree."""
    (git_tree / 'a.py').write_text('x = 2\n')
    out = tmp_path / 'ref.npz'
    argv = [
        'gen',
        '--main-tree',
        str(git_tree),
        '--eos-dir',
        str(tmp_path),
        '--output',
        str(out),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    try:
        rc = gen.main()
    except Exception as exc:
        pytest.fail(f'main() raised {exc!r} instead of returning 1')
    assert rc == 1
    assert not out.exists()


def test_aragog_imported_outside_main_tree_is_refused(
    git_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run stops before any array when aragog does not come from the main tree."""
    monkeypatch.setattr(sys, 'path', list(sys.path))
    out = tmp_path / 'ref.npz'
    try:
        gen.generate_reference(git_tree, tmp_path, out, 'f' * 40)
    except Exception as exc:
        assert isinstance(exc, RuntimeError), f'expected RuntimeError, got {exc!r}'
        assert 'not inside the main tree' in str(exc)
    else:
        pytest.fail('generate_reference did not refuse an aragog outside the main tree')
    assert not out.exists()
