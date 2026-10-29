"""Basic mesh refined towards the surface and the CMB.

Module under test: ``aragog.mesh.stretching.stretched_unit_grid`` and its use in
``Mesh`` through ``mesh.surface_cell_thickness`` and ``mesh.cmb_cell_thickness``.
Invariants: 0 at both ends is the uniform grid; a positive value is the radial
thickness of that end cell, in radius and in mass-coordinate mode; the grid is
strictly increasing and keeps its end radii; the external EOS (``eos_method = 2``)
and the JAX mesh arrays follow the refined radii.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.interpolate import PchipInterpolator

import aragog
from aragog.mesh import Mesh
from aragog.mesh.stretching import stretched_unit_grid
from aragog.parser import Parameters
from tests.conftest import needs_eos

pytestmark = pytest.mark.unit

D = 2.891e6  # Earth-like mantle depth [m]
CFG = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'


def _mesh(top=0.0, bottom=0.0, mass_coordinates=False, n=None):
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = mass_coordinates
    p.mesh.surface_cell_thickness, p.mesh.cmb_cell_thickness = top, bottom
    if n is not None:
        p.mesh.number_of_nodes = n
    return Mesh(p)


def _radii(m):
    return np.asarray(m.basic.radii).ravel()


def test_unit_grid_hits_both_end_cells_over_a_wide_range():
    """End cells of 0.5 to 16 km on a 2891 km span at 20 to 800 nodes, each end on
    or off: both targets to 1e-10, exact end points, strictly increasing; from 100
    nodes the neighbour-cell ratio stays below 1.2."""
    for n, f, l in itertools.product((20, 100, 400, 800), (0, 0.5, 2, 16), (0, 0.5, 2, 16)):
        first, last = f * 1e3 / D, l * 1e3 / D
        if max(first, last) >= 1.0 / (n - 1):
            continue
        s = stretched_unit_grid(n, first, last)
        d = np.diff(s)
        assert s[0] == 0.0 and s[-1] == 1.0 and np.all(d > 0.0)
        if first:
            assert d[0] == pytest.approx(first, rel=1e-10)
        if last:
            assert d[-1] == pytest.approx(last, rel=1e-10)
        if n >= 100:
            assert np.max(np.maximum(d[1:] / d[:-1], d[:-1] / d[1:])) < 1.2
    np.testing.assert_array_equal(stretched_unit_grid(50), np.linspace(0.0, 1.0, 50))


def test_unit_grid_rejects_cells_at_or_above_the_uniform_cell():
    for bad in (-1e-3, 1.0 / 99, 0.5, np.nan):
        with pytest.raises(ValueError, match='cell fraction'):
            stretched_unit_grid(100, bad, 0.0)
        with pytest.raises(ValueError, match='cell fraction'):
            stretched_unit_grid(100, 1e-4, bad)


def test_unit_grid_refuses_an_end_cell_below_the_smallest_fraction():
    """Below 1e-7 of the span the one-sided map loses the end cell (2e-5 relative at 1e-12)
    and two nodes merge (at 1e-16), and the two-sided map does not converge."""
    for bad in (5e-324, 1e-16, 1e-12, 9.9e-8):
        for first, last in ((bad, 0.0), (0.0, bad), (bad, 1e-3), (1e-3, bad)):
            with pytest.raises(ValueError, match='cell fraction'):
                stretched_unit_grid(100, first, last)


@pytest.mark.physics_invariant
@settings(max_examples=150, deadline=None)
@given(
    n=st.sampled_from([10, 100, 200, 400]),
    first=st.floats(min_value=0.0, max_value=1.0),
    last=st.floats(min_value=0.0, max_value=1.0),
)
def test_unit_grid_end_cells_hold_their_targets_from_the_smallest_fraction(n, first, last):
    """Each end fraction in [1e-7, 1 / (n - 1)) or 0, on one or both ends: the grid runs
    from 0 to 1 exactly, increases strictly and each set end cell equals its target to
    1e-6 relative."""
    h, lo = 1.0 / (n - 1), np.log10(1e-7)
    first, last = (
        0.0
        if u < 0.2
        else min(10 ** (lo + (u - 0.2) / 0.8 * (np.log10(h) - lo)), np.nextafter(h, 0))
        for u in (first, last)
    )
    s = stretched_unit_grid(n, first, last)
    d = np.diff(s)
    assert (s[0], s[-1]) == (0.0, 1.0)
    assert np.all(d > 0.0)
    for target, cell in ((first, d[0]), (last, d[-1])):
        if target:
            assert cell == pytest.approx(target, rel=1e-6)


XI = np.linspace(0.0, 1.0, 100)


def test_unit_grid_is_the_one_sided_tanh_map():
    """With the last cell of tanh(3 xi) / tanh(3) as target the grid is that map at every
    node, and the mirrored target gives the mirrored map, to 1e-10."""
    s = np.tanh(3.0 * XI) / np.tanh(3.0)
    np.testing.assert_allclose(
        stretched_unit_grid(100, 0.0, 1.0 - s[-2]), s, rtol=0, atol=1e-10
    )
    m = 1.0 - s[::-1]
    np.testing.assert_allclose(stretched_unit_grid(100, m[1], 0.0), m, rtol=0, atol=1e-10)


def test_unit_grid_is_the_two_sided_vinokur_map():
    """With the end cells of Vinokur's map at delta 4 and A 0.7 as targets the grid is that
    map at every node to 1e-9, and equal end cells give a grid symmetric about 1/2."""
    u = 0.5 * (1.0 + np.tanh(4.0 * (XI - 0.5)) / np.tanh(2.0))
    s = u / (0.7 + 0.3 * u)
    np.testing.assert_allclose(
        stretched_unit_grid(100, s[1], 1.0 - s[-2]), s, rtol=0, atol=1e-9
    )
    g = stretched_unit_grid(100, 1e-3, 1e-3)
    np.testing.assert_allclose(g + g[::-1], 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize('n', [100, 200])
@pytest.mark.parametrize('ends', [(0, 1), (1, 0), (1, 1)], ids=['last', 'first', 'both'])
def test_unit_grid_reaches_the_uniform_grid_at_the_uniform_cell(n, ends):
    """The largest float below the uniform cell, at one or both ends, gives the uniform
    grid to 1e-12."""
    f = np.nextafter(1.0 / (n - 1), 0.0)
    s = stretched_unit_grid(n, ends[0] * f, ends[1] * f)
    np.testing.assert_allclose(s, np.linspace(0.0, 1.0, n), rtol=0, atol=1e-12)


def test_two_sided_grid_refuses_end_cells_that_three_nodes_cannot_hold():
    """With 3 nodes both end cells must sum to 1; for 0.2 and 0.2 the solve ends on a
    monotone grid with both cells at 0.5, which the residual check refuses."""
    with pytest.raises(ValueError, match='did not converge'):
        stretched_unit_grid(3, 0.2, 0.2)


def test_radius_mode_keeps_the_shell_radii_of_a_thick_shell():
    """For these radii r_in + (r_out - r_in) misses r_out by 1 ulp; the refined radius-mode
    mesh still starts and ends exactly at the shell radii."""
    r_in, r_out = 504437.1555063366, 6871404.744724282
    assert r_in + (r_out - r_in) != r_out
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = False
    p.mesh.inner_radius, p.mesh.outer_radius = r_in, r_out
    p.mesh.surface_cell_thickness = 1e3
    r = _radii(Mesh(p))
    assert (r[0], r[-1]) == (r_in, r_out)
    assert np.diff(r)[-1] == pytest.approx(1e3, rel=1e-6)


@pytest.mark.parametrize('mass_coordinates', [False, True])
def test_mesh_end_cells_have_the_radial_thickness(mass_coordinates):
    """abe_solid (1000 km shell, 100 nodes, uniform cell 10.1 km): a 1 km top cell,
    a 2 km CMB cell, or both, to 1 mm; the end radii stay at the shell radii. In
    mass-coordinate mode the uniform grid has neither end cell at 1 or 2 km."""
    base = _radii(_mesh(mass_coordinates=mass_coordinates))
    assert np.diff(base)[-1] > 5e3 and np.diff(base)[0] > 5e3
    for top, bottom in ((1e3, 0.0), (0.0, 2e3), (1e3, 2e3)):
        r = _radii(_mesh(top, bottom, mass_coordinates))
        d = np.diff(r)
        assert (r[0], r[-1]) == (base[0], base[-1])
        assert np.all(d > 0.0)
        if top:
            assert d[-1] == pytest.approx(top, abs=1e-3)
        if bottom:
            assert d[0] == pytest.approx(bottom, abs=1e-3)
        # The other end moves away from the refined end: the cells there grow.
        if not bottom:
            assert d[0] > np.diff(base)[0]
        if not top:
            assert d[-1] > np.diff(base)[-1]


def test_mesh_rejects_a_cell_thicker_than_the_uniform_cell():
    p = Parameters.from_file(CFG)
    with pytest.raises(ValueError, match='surface_cell_thickness'):
        type(p.mesh)(**{**vars(p.mesh), 'surface_cell_thickness': 20e3})
    with pytest.raises(ValueError, match='cmb_cell_thickness'):
        type(p.mesh)(**{**vars(p.mesh), 'cmb_cell_thickness': -1.0})


def test_mass_mode_refuses_a_cmb_cell_between_its_uniform_cell_and_the_radial_bound():
    """abe_solid in mass-coordinate mode has a uniform CMB cell of 9.33 km, below the
    10.1 km radial cell the parser checks: 9.5 km passes the parser and the mesh refuses
    it with both field values. In radius mode 9.5 km is the CMB cell."""
    p = Parameters.from_file(CFG)
    radial = (p.mesh.outer_radius - p.mesh.inner_radius) / (p.mesh.number_of_nodes - 1)
    assert np.diff(_radii(_mesh(mass_coordinates=True)))[0] < 9.5e3 < radial
    type(p.mesh)(**{**vars(p.mesh), 'cmb_cell_thickness': 9.5e3})
    with pytest.raises(
        ValueError,
        match='mesh.cmb_cell_thickness = 9500.0 m and '
        'mesh.surface_cell_thickness = 0.0 m give no refined grid',
    ):
        _mesh(0.0, 9.5e3, mass_coordinates=True)
    assert np.diff(_radii(_mesh(0.0, 9.5e3)))[0] == pytest.approx(9.5e3, abs=1e-3)


@pytest.mark.physics_invariant
def test_derivative_operator_is_exact_for_a_linear_field_on_the_refined_grid():
    """d/dr of a staggered field linear in radius equals the slope at every basic node
    on the refined grid (top 1 km, CMB 2 km, radius mode), both extrapolated end rows
    included."""
    m = _mesh(1e3, 2e3)
    rs = np.asarray(m.staggered.radii)
    grad = (m._d_dr_transform @ (3.0e-3 * rs)).ravel()
    np.testing.assert_allclose(grad, 3.0e-3, rtol=1e-9)


def test_external_eos_and_jax_arrays_follow_the_refined_radii(tmp_path):
    """``eos_method = 2`` (a 4-column r, P, rho, g file), mass-coordinate mode, top
    1 km: the basic pressure is the PCHIP of the file at the refined radii and the
    JAX ``MeshArrays`` carry the same radii, pressure and gravity."""
    pytest.importorskip('jax')
    from aragog.jax.phase import MeshArrays
    from tests.test_entropy_solver_eos_method2_smoke import (
        _build_parameters_eos_method_2,
        _write_synthetic_external_mesh,
    )

    ef = tmp_path / 'mesh.dat'
    _write_synthetic_external_mesh(ef, n=30)
    p = _build_parameters_eos_method_2(ef, n_nodes=40)
    p.mesh.mass_coordinates = True
    p.mesh.surface_cell_thickness = 1e3
    m = Mesh(p)
    r = _radii(m)
    assert np.diff(r)[-1] == pytest.approx(1e3, abs=1e-3)
    tab = np.loadtxt(ef)
    P_ref = PchipInterpolator(tab[:, 0], tab[:, 1])(r)
    np.testing.assert_allclose(np.asarray(m.basic_pressure).ravel(), P_ref, rtol=1e-12)
    ma = MeshArrays.from_numpy_mesh(m)
    np.testing.assert_array_equal(np.asarray(ma.radii_basic), r)
    np.testing.assert_array_equal(np.asarray(ma.P_basic), np.asarray(m.basic_pressure).ravel())
    np.testing.assert_allclose(
        np.asarray(ma.gravity), PchipInterpolator(tab[:, 0], tab[:, 3])(r), rtol=1e-12
    )


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mass_coordinates', [False, True])
@pytest.mark.parametrize(
    ('grid', 'message'),
    [
        (lambda n, first, last: np.linspace(0.0, 1.0, n), 'gives an end cell of'),
        (lambda n, first, last: np.r_[0.0, 0.5, 0.4, np.linspace(0.6, 1.0, n - 3)],
         'not strictly increasing|Non-monotonic basic mesh'),
    ],
    ids=['end_cell', 'monotone'],
)  # fmt: skip
def test_a_refined_mesh_that_breaks_an_invariant_is_refused(
    monkeypatch, mass_coordinates, grid, message
):
    """A refined grid whose end cells miss their thickness or whose radii do not
    increase raises a ValueError in Mesh."""
    monkeypatch.setattr('aragog.mesh.stretched_unit_grid', grid)
    with pytest.raises(ValueError, match=message):
        _mesh(1e3, 2e3, mass_coordinates)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mass_coordinates', [False, True])
def test_the_refined_radii_check_refuses_end_radii_off_the_shell(mass_coordinates):
    """Both mesh modes set the end radii to the shell radii; the check passes them and
    refuses the same radii shifted by 1 um, with the end cells unchanged."""
    m = _mesh(1e3, 2e3, mass_coordinates)
    r = _radii(m)
    m._check_refined_radii(r)
    with pytest.raises(ValueError, match='end radii'):
        m._check_refined_radii(r + 1e-6)


def _jax_args(mass_coordinates, top, bottom):
    """abe_solid solver at its start state and the JAX ``dSdt_energy_balance`` arguments
    with the config boundary conditions."""
    import dataclasses

    from aragog.solver.entropy_solver import EntropySolver
    from tests.conftest import entropy_eos_copy
    from tests.test_jax_dsdt_core_module import _build_jax_pieces
    from tests.test_step_powers_batch import _set_initial_entropy

    params = Parameters.from_file(CFG)
    params.mesh.mass_coordinates = mass_coordinates
    params.mesh.surface_cell_thickness, params.mesh.cmb_cell_thickness = top, bottom
    s = EntropySolver(params, entropy_eos=entropy_eos_copy(identity_alpha=True))
    s.initialize()
    _set_initial_entropy(s)
    s._core_module_budget, s._core_module_q_radio, s._core_module_ra_crit_cmb = None, 0, 0
    args = list(_build_jax_pieces(s))[:6]
    b = params.boundary_conditions
    args[3] = dataclasses.replace(
        args[3],
        outer_bc_type=int(b.outer_boundary_condition),
        outer_bc_value=float(b.outer_boundary_value),
        inner_bc_type=int(b.inner_boundary_condition),
        inner_bc_value=float(b.inner_boundary_value),
    )
    return s, tuple(args)


@needs_eos
@pytest.mark.physics_invariant
@pytest.mark.parametrize('mass_coordinates', [False, True])
def test_the_jax_rhs_matches_numpy_on_the_refined_mesh(mass_coordinates):
    """abe_solid with a 1 km top cell and a 2 km CMB cell: the JAX energy-balance right-hand
    side equals the NumPy one to 1e-10 in the cells and 1e-9 in the CMB closure; on the
    uniform mesh arrays it misses by more than 10 %, so the refined mesh reaches JAX."""
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('equinox')
    from aragog.jax.solver import dSdt_energy_balance

    s, args = _jax_args(mass_coordinates, 1e3, 2e3)
    assert s._core_bc == 'energy_balance'
    assert np.diff(np.ravel(s.evaluator.mesh.basic.radii))[-1] == pytest.approx(1e3, abs=2e-3)
    y = np.asarray(s._S0, dtype=float)
    f_np, n = np.ravel(s.dSdt(0.0, y)), s._n_stag
    f_jax = np.ravel(dSdt_energy_balance(0.0, jnp.asarray(y), args))
    scale = np.abs(f_np[:n]).max()
    np.testing.assert_allclose(f_jax[:n], f_np[:n], rtol=1e-10, atol=1e-10 * scale)
    assert f_jax[n] == pytest.approx(f_np[n], rel=1e-9)
    _, uniform = _jax_args(mass_coordinates, 0.0, 0.0)
    f_u = np.ravel(dSdt_energy_balance(0.0, jnp.asarray(y), (*args[:2], uniform[2], *args[3:])))
    assert np.abs(f_u[:n] - f_np[:n]).max() > 0.1 * scale


def _cell_density(m):
    """Mass over volume of each cell between the basic radii, from the mesh EOS [kg/m^3]."""
    r = np.asarray(m.basic.radii)
    mass = np.diff(m.eos.get_mass_within_radii(r), axis=0)
    return np.ravel(mass / (4.0 / 3.0 * np.pi * np.diff(r**3, axis=0)))


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mass_coordinates', [False, True])
def test_the_staggered_density_is_the_cell_mass_over_volume_on_a_refined_mesh(mass_coordinates):
    """abe_solid with a 1 km top cell and a 2 km CMB cell, both modes: each cell's effective
    density is its mass over its volume on the refined radii, to 1e-12."""
    m = _mesh(1e3, 2e3, mass_coordinates)
    np.testing.assert_allclose(
        np.ravel(m.staggered_effective_density), _cell_density(m), rtol=1e-12
    )


@pytest.mark.physics_invariant
def test_the_external_eos_staggered_density_follows_the_refined_radii_in_radius_mode(tmp_path):
    """``eos_method = 2`` in radius mode with a 1 km top cell: the effective density of each
    cell is its mass over its volume on the refined radii, to 1e-12."""
    from tests.test_entropy_solver_eos_method2_smoke import (
        _build_parameters_eos_method_2,
        _write_synthetic_external_mesh,
    )

    ef = tmp_path / 'mesh.dat'
    _write_synthetic_external_mesh(ef, n=30)
    p = _build_parameters_eos_method_2(ef, n_nodes=40)
    p.mesh.mass_coordinates = False
    p.mesh.surface_cell_thickness = 1e3
    m = Mesh(p)
    assert np.diff(_radii(m))[-1] == pytest.approx(1e3, rel=1e-6)
    np.testing.assert_allclose(
        np.ravel(m.staggered_effective_density), _cell_density(m), rtol=1e-12
    )


@pytest.mark.parametrize('value', [True, '50', 0.5])
def test_mesh_refuses_a_cell_thickness_set_after_the_config_is_read(value):
    """A value set on the parsed parameters gets the config check in Mesh: a bool is not
    read as 1 m, a string is refused with the field name, and 0.5 m is below the floor."""
    p = Parameters.from_file(CFG)
    p.mesh.surface_cell_thickness = value
    with pytest.raises(ValueError, match='mesh.surface_cell_thickness must be 0'):
        Mesh(p)


@pytest.mark.parametrize('end', ['top', 'cmb'])
def test_mass_mode_puts_the_staggered_node_inside_a_2_m_end_cell(end):
    """Mass-coordinate mode with a 2 m end cell, the smallest allowed: the end cell holds
    2 m and its staggered node lies in its middle half, as in every other cell."""
    m = _mesh(2.0, 0.0, True) if end == 'top' else _mesh(0.0, 2.0, True)
    rb, rs = _radii(m), np.ravel(m.staggered.radii)
    frac = (rs - rb[:-1]) / np.diff(rb)
    cell = np.diff(rb)[-1 if end == 'top' else 0]
    assert cell == pytest.approx(2.0, abs=2e-3)
    assert np.all((frac > 0.25) & (frac < 0.75)), frac.min()
    assert frac[-1 if end == 'top' else 0] == pytest.approx(0.5, abs=1e-2)
