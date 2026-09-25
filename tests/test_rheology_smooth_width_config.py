"""The solid rheology requires a positive phase-edge smoothing width."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from aragog.config import Config
from aragog.parser import Parameters

pytestmark = pytest.mark.unit

_PROBE = Path(__file__).resolve().parent / 'configs' / 'yielding_active_probe.toml'


def _probe_text(width: float, enabled: bool = True) -> str:
    text = _PROBE.read_text()
    for old, new in (
        ('matprop_smooth_width = 0.01', f'matprop_smooth_width = {width}'),
        ('enabled = true', f'enabled = {str(enabled).lower()}'),
    ):
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text


@pytest.mark.parametrize('width', [0.0, -0.01])
def test_rheology_rejects_non_positive_width(width, tmp_path):
    with pytest.raises(ValueError, match='matprop_smooth_width > 0'):
        Config.from_dict(tomllib.loads(_probe_text(width)))
    path = tmp_path / 'probe.toml'
    path.write_text(_probe_text(width))
    with pytest.raises(ValueError, match='matprop_smooth_width > 0'):
        Parameters.from_file(path)


def test_rheology_accepts_positive_width():
    params = Config.from_dict(tomllib.loads(_probe_text(0.01)))
    assert params.phase_mixed.matprop_smooth_width == 0.01


def test_rheology_off_accepts_zero_width():
    params = Config.from_dict(tomllib.loads(_probe_text(0.0, enabled=False)))
    assert params.phase_mixed.matprop_smooth_width == 0.0
