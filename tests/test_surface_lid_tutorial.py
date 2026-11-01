"""The Python blocks of docs/Tutorials/surface_lid.md run and print the output the page shows."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import EOS_DIR
from tests.test_entropy_solver_bc_dispatch_smoke import needs_eos

PAGE = Path(__file__).resolve().parents[1] / 'docs' / 'Tutorials' / 'surface_lid.md'

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800), needs_eos]


def test_surface_lid_tutorial_blocks_run_and_match_page(tmp_path, monkeypatch, capsys):
    """Every Python block of the page runs in order, and every line of its output blocks is
    printed exactly."""
    text = PAGE.read_text(encoding='utf-8')
    code = re.findall(r'```python\n(.*?)```', text, flags=re.DOTALL)
    shown = re.findall(r'```text\n(.*?)```', text, flags=re.DOTALL)
    assert len(code) == 4 and len(shown) == 3
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', str(EOS_DIR))
    ns: dict[str, object] = {'__name__': '__main__'}
    for block in code:
        exec(compile(block, str(PAGE), 'exec'), ns)  # noqa: S102
    printed = capsys.readouterr().out.splitlines()
    for block in shown:
        for line in block.strip().splitlines():
            assert line in printed, line
