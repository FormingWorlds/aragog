"""The JAX right-hand side and the options of ``tools/verification/run_surface_lid_case.py``."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
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
SERIES = DRIVER.with_name('run_surface_lid_series.py')


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


def _collect_one(tmp_path, record):
    """``--collect`` of one run folder whose only part ends in the state ``[..., 7000, -1e13]``;
    ``record`` holds the fields of the record layout under test. Returns the written file."""
    d, run = _driver(), tmp_path / 'run'
    run.mkdir()
    base = dict(t_yr=10.0, sha='abc', eos_dir='/x/tab', F_top=2.5, dh_min=4.0, nodes=20)
    (run / f'{d.STEM}.json').write_text(json.dumps([{**base, **record}]))
    np.savez(run / f'{d.STEM}_part0.npz', y_end=np.array([2400.0, 7000.0, -1e13]))
    assert d.main(['--collect', str(tmp_path / 'out.npz'), f'a={run}']) == 0
    return np.load(tmp_path / 'out.npz')


@pytest.mark.unit
@pytest.mark.parametrize(
    'record',
    [
        {'T_i': 1400.0},
        {'lid': {'T_i': 1400.0}},
        {'lid': True, 'T_i': 1400.0},
    ],
    ids=['old-top-level', 'old-lid-entry', 'new-with-lid'],
)
def test_collect_reads_every_record_layout_with_a_lid(tmp_path, record):
    """A run with the lid gives its thickness from the part file and its T_i from the record."""
    out = _collect_one(tmp_path, record)
    assert out['a_delta'].tolist() == [7000.0]
    assert out['a_T_i'].tolist() == [1400.0]
    assert out['a_F_top'].tolist() == [2.5]


@pytest.mark.unit
def test_collect_gives_a_run_without_the_lid_zero_thickness(tmp_path):
    """The state of a --no-lid run has no lid thickness, so ``y_end[-2]`` is not read as one; the
    missing T_i is NaN in a plain float array."""
    out = _collect_one(tmp_path, {'lid': False, 'T_i': None})
    assert out['a_delta'].tolist() == [0.0]
    assert out['a_T_i'].dtype == float and np.isnan(out['a_T_i']).all()


@functools.cache
def _series():
    path = list(sys.path)
    spec = importlib.util.spec_from_file_location('run_surface_lid_series', SERIES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


@pytest.mark.unit
def test_series_depth_finds_the_rise_above_a_cold_node():
    """A cold node below the rise does not hide it: 300 to 1800 K over 1000 m crosses 1200 K at
    600 m."""
    T, z = np.array([300.0, 1800.0, 1000.0]), np.array([0.0, 1000.0, 2000.0])
    assert _series().isotherm_depth(T, z, 1200.0) == pytest.approx(600.0, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('T', 'T_iso'),
    [([1300.0, 1400.0], 1200.0), ([1000.0, 1100.0], 1200.0), ([1300.0, 1000.0], 1200.0)]
    + [([1000.0, 1300.0], float('nan'))],
    ids=['above', 'below', 'falls', 'no-isotherm'],
)
def test_series_depth_is_nan_without_a_rise_in_the_column(T, T_iso):
    """No rise in the mesh column gives NaN, the value that the series stores."""
    assert np.isnan(_series().isotherm_depth(np.array(T), np.array([10.0, 20.0]), T_iso))


@pytest.mark.unit
def test_series_depth_is_nan_where_the_solver_finds_the_isotherm_in_the_lid():
    """The series sees only the mesh column. Above the isotherm, a live lid holds the crossing, so
    the depth is not 0: the solver gives 200 m here and the series has no value to give."""
    from aragog.solver.entropy_solver import _isotherm_depth

    T, z = np.array([1500.0, 1600.0]), np.array([500.0, 1500.0])
    assert _isotherm_depth(1000.0, 400.0, 1500.0, lambda: 500.0, z, T) == pytest.approx(200.0)
    assert np.isnan(_series().isotherm_depth(T, z, 1000.0))


@pytest.mark.unit
@settings(deadline=None, max_examples=300)
@given(
    T=st.lists(st.floats(200.0, 3000.0), min_size=2, max_size=12),
    T_iso=st.floats(300.0, 2500.0),
)
@example(T=[300.0, 1800.0, 1000.0], T_iso=1200.0)
@example(T=[300.0, 1800.0, 1000.0, 1500.0], T_iso=1200.0)
def test_series_depth_agrees_with_the_solver_on_the_mesh_column(T, T_iso):
    """Where the column holds a rise, the series gives the depth of the solver function; where it
    holds none, the solver finds no crossing below the top cell."""
    from aragog.solver.entropy_solver import _isotherm_depth

    T, z = np.array(T), 1000.0 * np.arange(1.0, len(T) + 1.0)
    got = _series().isotherm_depth(T, z, T_iso)
    want = _isotherm_depth(T_iso, 500.0, 100.0, lambda: 100.0, z, T)
    if np.isnan(got):
        assert want is None or want <= z[0]
    else:
        assert got == want


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
    assert d.main(['--collect', str(tmp_path / 'o.npz'), f'a={tmp_path}']) == 0
    assert np.load(tmp_path / 'o.npz')['a_delta'].tolist() == [0.0]


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
    assert d.main(['--collect', str(tmp_path / 'o.npz'), f'a={tmp_path}']) == 0
    out = np.load(tmp_path / 'o.npz')
    assert out['a_delta'].tolist() == [row['delta']] and out['a_T_i'].tolist() == [row['T_i']]


@pytest.mark.unit
@pytest.mark.parametrize('surface_cell', [None, 5000.0])
def test_the_stored_column_is_the_uniform_mesh_with_slope_1(surface_cell):
    """The tools rebuild the mesh and the rheology of the stored runs whatever the defaults
    are: no CMB refinement, a uniform top cell or the given one, MLT bottom slope 1.0."""
    from aragog.config import Config
    from aragog.mesh import Mesh

    d = _driver()
    cfg = Path(d.__file__).resolve().parents[2] / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'
    c = Config.from_file(str(cfg))
    d.stored_column(c, surface_cell)
    assert c.phase_solid.rheology.mlt_bottom_slope == 1.0
    assert c.phase_solid.rheology.stress_closure_mode == 'lid'
    ref = Config.from_file(str(cfg))  # the mesh of the stored runs, set field by field
    ref.mesh.cmb_cell_thickness = 0.0
    ref.mesh.surface_cell_thickness = 0.0 if surface_cell is None else surface_cell
    assert np.array_equal(np.ravel(Mesh(c).basic.radii), np.ravel(Mesh(ref).basic.radii))


@pytest.mark.unit
def test_a_part_file_on_another_mesh_stops_the_tools(tmp_path):
    """check_stored_mesh passes the rebuilt mesh and stops on a part file with other radii."""
    from aragog.config import Config
    from aragog.mesh import Mesh

    d = _driver()
    cfg = Path(d.__file__).resolve().parents[2] / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'
    c = Config.from_file(str(cfg))
    d.stored_column(c, None)
    mesh = Mesh(c)
    r = np.ravel(mesh.basic.radii)
    np.savez(tmp_path / 'same.npz', r_basic=r)
    np.savez(tmp_path / 'other.npz', r_basic=r * (1 + 1e-12))
    d.check_stored_mesh(np.load(tmp_path / 'same.npz'), mesh, 'same')
    with pytest.raises(SystemExit, match='other: the stored mesh differs'):
        d.check_stored_mesh(np.load(tmp_path / 'other.npz'), mesh, 'other')
