"""The Python blocks of docs/Tutorials/cmb_eddy_diffusivity.md run and print the output the page
shows."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / 'docs' / 'Tutorials' / 'cmb_eddy_diffusivity.md'


@pytest.mark.unit
def test_cmb_eddy_diffusivity_tutorial_blocks_run_and_match_page(capsys):
    """Every Python block of the page runs in order, and every line of its output blocks is
    printed exactly."""
    pytest.importorskip('jax')
    text = PAGE.read_text(encoding='utf-8')
    code = re.findall(r'```python\n(.*?)```', text, flags=re.DOTALL)
    shown = re.findall(r'```text\n(.*?)```', text, flags=re.DOTALL)
    assert len(code) == 2 and len(shown) == 2
    ns: dict[str, object] = {'__name__': '__main__'}
    for block in code:
        exec(compile(block, str(PAGE), 'exec'), ns)  # noqa: S102
    printed = capsys.readouterr().out.splitlines()
    for block in shown:
        for line in block.strip().splitlines():
            assert line in printed, line
