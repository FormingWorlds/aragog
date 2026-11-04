"""The JAX right-hand side and the options of ``tools/verification/run_surface_lid_case.py``."""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos

jax = pytest.importorskip('jax')

pytestmark = [pytest.mark.timeout(600)]

DRIVER = Path(__file__).parents[1] / 'tools' / 'verification' / 'run_surface_lid_case.py'


@functools.cache
def _driver():
    spec = importlib.util.spec_from_file_location('run_surface_lid_case', DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rates(eos, heating, jax_rhs):
    """dS/dt [J/(kg K yr)] of the 10-node lid column at its start, NumPy or through the driver's
    JAX factory, with internal heating [W/kg] or without."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    if heating:
        p.energy.tidal, p.energy.tidal_array = True, np.array([heating])
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3050.0)
    y = np.asarray(s._S0, float).copy()
    if not jax_rhs:
        return np.asarray(s.dSdt(0.0, y), float)[: s._n_stag]
    _driver()._install_factory(s, EOS_DIR)
    sc = s._build_nondim_scales()
    rhs, _ = s._jax_cvode_factory(sc, s._core_bc)
    off = 0.0 if sc.state_offset is None else np.asarray(sc.state_offset)
    y_nd = (y - off) / np.asarray(sc.state_scale)
    yd = np.zeros_like(y_nd)
    rhs(0.0, y_nd, yd)
    return (yd * np.asarray(sc.state_scale) / float(sc.t_ref))[: s._n_stag]


@pytest.mark.slow
@needs_eos
def test_driver_jax_rhs_carries_the_internal_heating():
    """The heating set in the config reaches the JAX right-hand side of the driver: its heated
    minus unheated rate equals that of the NumPy right-hand side in every cell."""
    eos = entropy_eos_copy(EOS_DIR)
    H = 3.0e-7
    d_np = _rates(eos, H, False) - _rates(eos, None, False)
    d_jax = _rates(eos, H, True) - _rates(eos, None, True)
    assert np.all(d_np > 0.0)
    np.testing.assert_allclose(d_jax, d_np, rtol=1e-9, atol=0.0)


def _options(**kw):
    import argparse

    return argparse.Namespace(**{'eta': None, 'heating': None, 'surface_cell': None, **kw})


@pytest.mark.unit
@pytest.mark.parametrize(
    ('name', 'accept', 'message'),
    [
        ('eta', lambda v: v > 0.0, '--eta must be finite and positive'),
        ('heating', lambda v: v >= 0.0, '--heating must be finite and non-negative'),
        ('surface_cell', lambda v: v >= 2.0, '--surface-cell must be finite and at least 2 m'),
    ],
)
@given(st.floats(allow_nan=True, allow_infinity=True))
@example(0.0)
@example(2.0)
@settings(max_examples=200, deadline=None)
def test_options_accept_only_their_range(name, accept, message, value):
    """Each option passes inside its finite range, and stops the run outside it."""
    check, options = _driver()._check_options, _options(**{name: value})
    if np.isfinite(value) and accept(value):
        check(options)
    else:
        with pytest.raises(SystemExit, match=message):
            check(options)


@pytest.mark.unit
@pytest.mark.parametrize('flag', ['--eta', '--heating', '--surface-cell'])
def test_options_refuse_a_value_that_is_not_a_number(flag):
    """A value that is not a number (as a TOML true) stops the parser; no value keeps the
    default None."""
    d = _driver()
    with pytest.raises(SystemExit):
        d.parse(['out', flag, 'true'])
    assert getattr(d.parse(['out']), flag[2:].replace('-', '_')) is None


@pytest.mark.slow
@needs_eos
def test_driver_runs_without_the_lid_on_a_refined_top_cell(tmp_path):
    """--no-lid with --surface-cell runs the column without the lid on a mesh whose top cell has
    the set thickness; the part rows hold no lid states and record both options."""
    d = _driver()
    assert (
        d.main(
            [
                str(tmp_path),
                '--eos-dir',
                str(EOS_DIR),
                '--samples',
                '1e1',
                '--nodes',
                '20',
                '--no-lid',
                '--surface-cell',
                '1000',
            ]
        )
        == 0
    )
    row = json.loads((tmp_path / f'{d.STEM}.json').read_text())[-1]
    assert row['lid'] is False and row['surface_cell'] == 1000.0
    assert row['delta'] is None and row['E'] is None and row['handover_root'] is None
    r = np.load(tmp_path / f'{d.STEM}_part0.npz')['r_basic']
    assert r[-1] - r[-2] == pytest.approx(1000.0, rel=1e-6)  # radii from the Newton mass grid


@needs_eos
@pytest.mark.slow
@needs_eos
def test_driver_records_the_flux_of_the_lid(tmp_path, monkeypatch):
    """The part row holds the surface flux of the lid, not the flux of the grid face that the
    hand-over root leaves in the state."""
    from aragog.solver.entropy_solver import EntropySolver

    seen = []
    root = EntropySolver._lid_root_value

    def spy(self, t, y):
        last = getattr(self, '_lid_last', None)
        seen.append(None if last is None else float(last.rates.F_top))
        return root(self, t, y)

    monkeypatch.setattr(EntropySolver, '_lid_root_value', spy)
    d = _driver()
    args = [str(tmp_path), '--eos-dir', str(EOS_DIR), '--samples', '1e1', '--nodes', '20']
    assert d.main(args) == 0
    row = json.loads((tmp_path / f'{d.STEM}.json').read_text())[-1]
    assert row['delta'] > 0.0
    assert row['F_top'] == pytest.approx(seen[-1], rel=1e-12)
