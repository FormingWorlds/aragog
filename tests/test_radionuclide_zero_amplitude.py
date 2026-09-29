"""Zero-amplitude radionuclides contribute no heating; an overflowing live one is rejected."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from aragog.cli import cli
from aragog.config.radionuclides import RadionuclideConfig
from aragog.parser import Parameters, _Radionuclide

from .test_phi_step_cap_armed_smoke import EOS_DIR, needs_eos

BUNDLED_LOOKUP = Path(__file__).resolve().parents[1] / 'src/aragog/cfg/abe_mixed_lookup.cfg'
AL26_ZERO = dict(
    name='Al26',
    t0_years=4.55e9,
    abundance=0.0,
    concentration=0.0,
    heat_production=0.3583,
    half_life_years=0.717e6,
)


def _lookup_cfg(tmp_path, *, end_time=None, al26_live=False) -> Path:
    """Copy of the bundled lookup cfg with a shorter end time or a live Al26."""
    text = BUNDLED_LOOKUP.read_text()
    if end_time is not None:
        text = text.replace('end_time = 1000\n', f'end_time = {end_time}\n')
        assert f'end_time = {end_time}\n' in text
    if al26_live:
        head, al26 = text.split('[radionuclide_Al26]')
        al26, fe60 = al26.split('[radionuclide_Fe60]')
        al26, n = re.subn(r'(?m)^(abundance|concentration) = 0$', r'\1 = 1', al26)
        assert n == 2
        text = head + '[radionuclide_Al26]' + al26 + '[radionuclide_Fe60]' + fe60
    path = tmp_path / 'lookup.cfg'
    path.write_text(text)
    return path


@pytest.mark.unit
@pytest.mark.parametrize('cls', [_Radionuclide, RadionuclideConfig])
def test_zero_amplitude_isotope_heats_zero_where_exp_overflows(cls):
    r = cls(**AL26_ZERO)
    assert r.get_heating(0.0) == 0.0
    np.testing.assert_array_equal(r.get_heating(np.array([0.0, 1e3])), [0.0, 0.0])
    with np.errstate(divide='ignore'):
        assert cls(**{**AL26_ZERO, 'half_life_years': 0.0}).get_heating(0.0) == 0.0


@pytest.mark.unit
def test_bundled_lookup_cfg_heating_is_finite():
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    heat = {r.name: r.get_heating(p.solver.start_time) for r in p.radionuclides}
    assert heat.pop('Al26') == heat.pop('Fe60') == 0.0
    assert all(np.isfinite(h) and h > 0.0 for h in heat.values())


@pytest.mark.unit
def test_non_positive_half_life_raises(tmp_path):
    cfg = _lookup_cfg(tmp_path)
    cfg.write_text(
        cfg.read_text().replace('half_life_years = 0.717E6', 'half_life_years = -0.717E6')
    )
    with pytest.raises(ValueError, match='Al26: half_life_years must be positive'):
        Parameters.from_file(str(cfg))


@pytest.mark.unit
def test_live_isotope_overflowing_at_start_raises(tmp_path):
    cfg = _lookup_cfg(tmp_path, al26_live=True)
    with pytest.raises(ValueError, match=r'Al26.*start time 0.*t0_years = 4550000000'):
        Parameters.from_file(str(cfg))


@pytest.mark.unit
def test_jax_zero_amplitude_value_and_jacobian_are_finite():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from aragog.jax.solver import make_radio_heating_fn

    rs = Parameters.from_file(str(BUNDLED_LOOKUP)).radionuclides
    keys = ('heat_production', 'abundance', 'concentration', 't0_years', 'half_life_years')
    h = make_radio_heating_fn(*(np.array([getattr(r, k) for r in rs]) for k in keys))
    value, slope = float(h(0.0)), float(jax.jacrev(h)(0.0))
    assert np.isfinite(slope) and slope < 0.0
    assert value == pytest.approx(sum(r.get_heating(0.0) for r in rs), rel=1e-12)


@pytest.mark.smoke
@needs_eos
def test_bundled_lookup_cfg_runs_with_finite_state(tmp_path):
    import netCDF4

    out = tmp_path / 'o.nc'
    args = [
        'run',
        str(_lookup_cfg(tmp_path, end_time=1)),
        '--eos-dir',
        str(EOS_DIR),
        '--out',
        str(out),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert 'integration failed' not in result.output
    with netCDF4.Dataset(out) as d:
        assert float(d['time'][:]) == pytest.approx(1.0)
        assert np.isfinite(float(d['T_magma'][:]))
