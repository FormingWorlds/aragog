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
    assert 'not verdicts' in out and 'MK22A: ' in out
    rows = sheet_tool.sheet(EntropyEOS(EOS_DIR))
    assert rows and {r.status for r in rows} <= {'pass', 'outside', 'report', 'n/a'}
    values = {r.check: r.value for r in rows}
    assert values['rho_melt - rho_solid at 1 GPa [kg/m3]'] == '449.2'
    assert values['T_liquidus - T_solidus at 50 GPa [K]'] == '1029'
    assert values['Delta S at 100 GPa / lowest pressure'] == '3.243'
    assert (
        values['solid temperature rows copied above S_min, lowest pressure [J/kg/K]'] == '321.7'
    )
