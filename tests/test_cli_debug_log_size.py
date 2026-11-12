"""The DEBUG file log of ``aragog run`` grows with the segments of a run, not with its RHS calls."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

_SRC = Path(__file__).resolve().parents[1] / 'src'
_RECORD = re.compile(r'^\[\d{4}-\d\d-\d\d ', re.M)


@needs_eos
def test_debug_file_log_has_no_record_per_rhs_call(tmp_path):
    """A 1 yr solid run writes far fewer log records than RHS calls and keeps its segment lines.

    A DEBUG line in a mesh transform writes about 3 records per RHS call.
    """
    cfg = _SRC / 'aragog' / 'cfg' / 'abe_solid.toml'
    cmd = [sys.executable, '-c', 'from aragog.cli import cli; cli()', 'run', str(cfg)]
    cmd += ['--eos-dir', str(EOS_DIR), '--set', 'solver.end_time=1', '--log-dir', str(tmp_path)]
    cmd += ['--out', str(tmp_path / 'out.nc')]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(_SRC), str(_SRC.parent)])}
    proc = subprocess.run(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=55
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    log = (tmp_path / 'aragog.log').read_text(encoding='utf-8')
    rhs = sum(int(n) for n in re.findall(r'(\d+) RHS evals', log))
    records = len(_RECORD.findall(log))
    assert records > 0, 'no record matches the log line pattern'
    assert rhs > 300, f'{rhs} RHS calls: the run is too short to show a per-call line'
    assert records < rhs / 10, f'{records} log records for {rhs} RHS calls'
    assert 'rate cap segment' in log
    assert 'quantity_at_basic_nodes' not in log
