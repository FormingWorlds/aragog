"""Layout of the stored solid-state convection reference JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.verification._ssc_helpers import write_compact_json

pytestmark = pytest.mark.unit

REFERENCE_DIR = Path(__file__).resolve().parent / 'reference'
STORED_FILES = (
    'ssc_0d_reference_benchmarks.json',
    'ssc_mlt_top_slope_calibration.json',
    'ssc_scaling_exponents_benchmarks.json',
    'ssc_solver_wiring_benchmarks.json',
)


def test_compact_json_loads_equal_to_an_indented_dump(tmp_path: Path) -> None:
    """The compact file loads to the same object, one top-level block per line."""
    data = {
        'canary': {'aragog_commit': 'a' * 40, 'files': ['src/aragog', 'tools/x.py']},
        'values': [0.1, 1.0e-300, 2.220446049250313e-16, -0.0, 3, None, True],
        'nested': {'a': {'b': [[1.5, 2.5], []], 'c': {}}, 'text': 'μ, "q"'},
        'scalar': 1.2105794672131882,
    }
    out = tmp_path / 'out.json'
    write_compact_json(out, data)
    text = out.read_text(encoding='utf-8')
    lines = text.split('\n')

    assert json.loads(text) == json.loads(json.dumps(data, indent=2)) == data
    assert lines[0] == '{' and lines[-2] == '}' and lines[-1] == ''
    assert len(lines) == len(data) + 3
    for line, (key, value) in zip(lines[1:-2], data.items(), strict=True):
        assert line.rstrip(',') == f'"{key}": ' + json.dumps(value, separators=(',', ':'))
    assert all(line.endswith(',') for line in lines[1:-3])
    assert not lines[-3].endswith(',')


def test_compact_json_keeps_non_finite_floats(tmp_path: Path) -> None:
    """NaN and infinity load back as they do from an indented dump."""
    out = tmp_path / 'out.json'
    write_compact_json(out, {'x': [math.nan, math.inf, -math.inf]})
    x = json.loads(out.read_text(encoding='utf-8'))['x']
    assert math.isnan(x[0]) and x[1] == math.inf and x[2] == -math.inf


def test_compact_json_writes_an_empty_object(tmp_path: Path) -> None:
    """An empty mapping gives ``{}`` on one line."""
    out = tmp_path / 'out.json'
    write_compact_json(out, {})
    assert out.read_text(encoding='utf-8') == '{}\n'


def test_compact_json_refuses_a_non_string_key(tmp_path: Path) -> None:
    """A top-level key that is not a string raises and writes no file."""
    out = tmp_path / 'out.json'
    with pytest.raises(TypeError, match='top-level JSON keys must be str'):
        write_compact_json(out, {1: 'a'})
    assert not out.exists()


@pytest.mark.parametrize('name', STORED_FILES)
def test_stored_reference_file_is_compact(name: str, tmp_path: Path) -> None:
    """Each stored file equals the compact rendering of its own content."""
    stored = REFERENCE_DIR / name
    text = stored.read_text(encoding='utf-8')
    rewritten = tmp_path / name
    write_compact_json(rewritten, json.loads(text))
    assert text == rewritten.read_text(encoding='utf-8')
