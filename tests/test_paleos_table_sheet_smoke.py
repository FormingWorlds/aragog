"""The table sheet on a real table set."""

from __future__ import annotations

import hashlib

import pytest

from aragog.eos.entropy import EntropyEOS

from .conftest import EOS_DIR, needs_eos
from .test_paleos_table_sheet import sheet_tool

pytestmark = [pytest.mark.smoke, needs_eos]


def test_the_sheet_reads_a_real_table_set(capsys):
    """A full sheet: statuses known, the md5 of each table read, a legend and the sources."""
    assert sheet_tool.main([str(EOS_DIR)]) == 0
    out = capsys.readouterr().out
    for name in ('temperature_solid.dat', 'density_melt.dat', 'solidus_P-S.dat'):
        digest = hashlib.md5((EOS_DIR / name).read_bytes()).hexdigest()
        assert f'{name} md5 {digest}' in out
    assert 'not a failure' in out and 'MK22A: ' in out
    rows = sheet_tool.sheet(EntropyEOS(EOS_DIR))
    assert rows and {r.status for r in rows} <= {'pass', 'outside', 'n/a'}
