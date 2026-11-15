"""The Python blocks of docs/Tutorials/core_tabulated.md run and print what the page shows."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'docs' / 'Tutorials' / 'core_tabulated.md'

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]


def test_core_tabulated_tutorial_blocks_run_and_match_page(capsys, monkeypatch):
    """Every Python block of the page runs in order from the repository root, and every line
    of its output blocks is printed exactly."""
    monkeypatch.chdir(ROOT)
    text = PAGE.read_text(encoding='utf-8')
    code = re.findall(r'```python\n(.*?)```', text, flags=re.DOTALL)
    shown = re.findall(r'```text\n(.*?)```', text, flags=re.DOTALL)
    assert len(code) == 5 and len(shown) == 5
    ns: dict[str, object] = {'__name__': '__main__'}
    for block in code:
        exec(compile(block, str(PAGE), 'exec'), ns)  # noqa: S102
    printed = capsys.readouterr().out.splitlines()
    missing = [
        line for block in shown for line in block.strip().splitlines() if line not in printed
    ]
    assert not missing, (missing, printed)
