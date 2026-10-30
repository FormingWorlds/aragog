"""``tools/verification/run_step_cap_validation.py``: short real runs of the v3 and v4 paths."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from .conftest import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, needs_eos]

_spec = importlib.util.spec_from_file_location(
    'run_step_cap_validation',
    Path(__file__).resolve().parents[1]
    / 'tools'
    / 'verification'
    / 'run_step_cap_validation.py',
)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


@pytest.mark.parametrize('name', ['v3_r3', 'v3_r4'])
def test_v3_rate_run_counts_its_segments_inside_the_step_ceiling(name):
    """The m = 0 rate run lands slowly in v3_r3 and reaches infinite caps in v3_r4."""
    row = tool.v3_run((str(EOS_DIR), name, 'rate', 1e-8, 0, None))
    assert (row['mode'], row['m']) == ('rate', 0)
    assert 0 < row['steps'] <= tool.V3[name]['max_steps']
    assert row['segments'] >= 1 and len(row['T_end']) > 0
    if name == 'v3_r3':
        assert row['slow_landing'] >= 1 and row['inf_caps'] == 0
    else:
        assert row['subsolidus'] >= 1 and row['inf_caps'] >= 1


def test_v4_rheo_on_runs_with_the_jax_factory(monkeypatch):
    """A 10 yr rheo_on run on the CI table (table guard set to it) calls the JAX factory."""
    table = (Path(EOS_DIR) / 'temperature_solid.dat').read_bytes()
    monkeypatch.setattr(tool, 'PALEOS_LOWT_MD5', hashlib.md5(table).hexdigest())
    calls, install = [], tool.EntropySolver.set_jax_cvode_factory

    def spy(self, factory):
        def counted(*args):
            calls.append(args)
            return factory(*args)

        return install(self, counted)

    monkeypatch.setattr(tool.EntropySolver, 'set_jax_cvode_factory', spy)
    r = tool.run_v4(EOS_DIR, 'v4_rheo_on', 10.0)
    assert calls, 'the JAX factory was never called'
    assert r['status'] == 0 and r['t_yr'] == pytest.approx(10.0) and r['segments'] >= 1
    assert r['steps'] > 0 and 0.0 < r['phi_max'] < 1.0 and r['inside_table']
    assert not r['passed_stall_point'] and not r['solidified']
    assert tool.check('v4_rheo_on', r) == ['passed_stall_point', 'solidified']


def test_v3_starts_place_the_cells_against_the_solidus():
    """v3_r3 puts cells 4 and 5 above the stiff zone and the rest 15 below; v3_r4 is uniform."""
    eos = tool.EntropyEOS(str(EOS_DIR))
    S_sol = np.asarray(
        eos.solidus_entropy(tool._v3_solver(eos, tool.V3['v3_r3'], 'rate', 1e-8)._P_stag_flat)
    ).ravel()
    offset = tool.v3_start(eos, 'v3_r3') - S_sol
    np.testing.assert_allclose(offset[[4, 5]], [11.0, 12.5], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.delete(offset, [4, 5]), -15.0, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(
        tool.v3_start(eos, 'v3_r4'), np.full_like(S_sol, S_sol[-1] + 5.0)
    )


def test_v3_cases_run_at_width_zero_unless_replaced():
    """The v3 controller cases fix matprop_smooth_width = 0; --width replaces it."""
    eos = tool.EntropyEOS(str(EOS_DIR))
    for name in tool.V3:
        s = tool._v3_solver(eos, tool.V3[name], 'rate', 1e-8)
        assert s.parameters.phase_mixed.matprop_smooth_width == 0.0
        s = tool._v3_solver(eos, tool.V3[name], 'rate', 1e-8, width=0.01)
        assert s.parameters.phase_mixed.matprop_smooth_width == 0.01
