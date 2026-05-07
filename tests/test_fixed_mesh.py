"""Unit tests for ``aragog.mesh.fixed_mesh.FixedMesh``.

FixedMesh is the geometric building block underneath the higher-level
``Mesh`` (which composes a ``basic`` and a ``staggered`` FixedMesh). It
owns the cached_property accessors for area, delta_mesh, depth, height,
mixing_length, volume, and total_volume. These are pure geometry, no
physics, so a regression here is easy to catch with closed-form values.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.mesh.fixed_mesh import FixedMesh
from aragog.parser import _MeshParameters

pytestmark = pytest.mark.unit


def _settings(mixing_length_profile: str = 'nearest_boundary') -> _MeshParameters:
    """Minimal MeshParameters instance for FixedMesh construction."""
    return _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=4,
        mixing_length_profile=mixing_length_profile,
        core_density=10500.0,
    )


def _basic_mesh_4_nodes(*, mixing_length_profile: str = 'nearest_boundary') -> FixedMesh:
    """4-node uniform mesh between r=3.48e6 and r=6.371e6.

    Uses 1-D arrays so the post_init monotonicity check (which
    operates on ``np.diff(axis=-1)``) is actually exercised.
    """
    radii = np.linspace(3.48e6, 6.371e6, 4)
    mass_radii = radii.copy()  # unit-density coordinate
    return FixedMesh(
        settings=_settings(mixing_length_profile=mixing_length_profile),
        radii=radii,
        mass_radii=mass_radii,
        outer_boundary=6.371e6,
        inner_boundary=3.480e6,
    )


# ---- post_init validation --------------------------------------------------


def test_fixed_mesh_rejects_non_monotonic_radii():
    """Non-monotonic radii must raise. Discriminator: catches a regression
    that loosened the strict-greater check (e.g. accepting plateaus).
    """
    bad_radii = np.array([1.0, 3.0, 2.0, 4.0])
    with pytest.raises(ValueError, match='monotonically increasing'):
        FixedMesh(
            settings=_settings(),
            radii=bad_radii,
            mass_radii=bad_radii,
            outer_boundary=4.0,
            inner_boundary=1.0,
        )


def test_fixed_mesh_rejects_duplicate_radii_at_node_boundary():
    """Edge case: two adjacent identical radii (a degenerate mesh) must
    raise. A regression to ``np.diff >= 0`` would silently accept this
    and produce delta_mesh == 0 → division by zero downstream.
    """
    radii = np.array([1.0, 2.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='monotonically increasing'):
        FixedMesh(
            settings=_settings(),
            radii=radii,
            mass_radii=radii,
            outer_boundary=3.0,
            inner_boundary=1.0,
        )


# ---- cached property accessors --------------------------------------------


def test_fixed_mesh_area_uses_4pi_r_squared():
    """area = 4 π r².  At r=outer_boundary=6.371e6:
    A = 4π × (6.371e6)² = 5.10e14 m². Use a value that distinguishes
    4π r² from naive r² and from 2π r (sphere vs cylinder).
    """
    fm = _basic_mesh_4_nodes()
    expected_outer = 4 * np.pi * (6.371e6) ** 2
    assert float(fm.area[-1]) == pytest.approx(expected_outer, rel=1e-12)
    # Inner area should be smaller than outer (monotonic); discriminate
    # against a regression that swapped indices.
    assert float(fm.area[0]) < float(fm.area[-1])
    # Sanity: nonzero, positive everywhere.
    assert np.all(np.asarray(fm.area) > 0)


def test_fixed_mesh_delta_mesh_uniform_for_uniform_radii():
    """delta_mesh = diff(mass_radii); for 4 uniform nodes, three deltas,
    all equal to (outer - inner) / 3.
    """
    fm = _basic_mesh_4_nodes()
    expected_delta = (6.371e6 - 3.480e6) / 3.0
    np.testing.assert_allclose(
        np.asarray(fm.delta_mesh).ravel(),
        np.full(3, expected_delta),
        rtol=1e-12,
    )


def test_fixed_mesh_depth_zero_at_outer_boundary():
    """depth(r) = outer_boundary - r.  At r=outer the depth is 0; at
    r=inner the depth is the full shell thickness.
    """
    fm = _basic_mesh_4_nodes()
    assert float(fm.depth[-1]) == pytest.approx(0.0, abs=1e-9)
    assert float(fm.depth[0]) == pytest.approx(6.371e6 - 3.480e6, rel=1e-12)
    # Edge case: monotonically decreasing with index.
    assert np.all(np.diff(np.asarray(fm.depth).ravel()) < 0)


def test_fixed_mesh_height_zero_at_inner_boundary():
    """height(r) = r - inner_boundary.  Mirrors depth."""
    fm = _basic_mesh_4_nodes()
    assert float(fm.height[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(fm.height[-1]) == pytest.approx(6.371e6 - 3.480e6, rel=1e-12)
    assert np.all(np.diff(np.asarray(fm.height).ravel()) > 0)


def test_fixed_mesh_volume_matches_spherical_shell_formula():
    """volume_i = 4/3 π (r_{i+1}^3 - r_i^3).  Compute the bottom shell
    closed-form and compare. Use mid-mesh values, not boundary, to
    distinguish 4/3π r³ from 4π r² δr (thin-shell approximation).
    """
    fm = _basic_mesh_4_nodes()
    radii = np.asarray(fm.radii).ravel()
    expected_bottom = 4.0 / 3.0 * np.pi * (radii[1] ** 3 - radii[0] ** 3)
    assert float(fm.volume[0]) == pytest.approx(expected_bottom, rel=1e-12)
    # Volumes increase outward (each shell is geometrically larger)
    vols = np.asarray(fm.volume).ravel()
    assert np.all(np.diff(vols) > 0), (
        'shell volumes do not increase outward; index ordering is wrong'
    )


def test_fixed_mesh_total_volume_matches_outer_minus_inner_sphere():
    """total_volume = 4/3 π (R_outer^3 - R_inner^3). Earth-mantle
    closed-form check.
    """
    fm = _basic_mesh_4_nodes()
    expected = 4.0 / 3.0 * np.pi * ((6.371e6) ** 3 - (3.480e6) ** 3)
    assert fm.total_volume == pytest.approx(expected, rel=1e-12)
    # Edge case: total_volume == sum(volume) for a contiguous mesh
    assert fm.total_volume == pytest.approx(float(np.sum(fm.volume)), rel=1e-12)


def test_fixed_mesh_number_of_nodes_matches_radii_size():
    """number_of_nodes is a derived count; must equal radii.size."""
    fm = _basic_mesh_4_nodes()
    assert fm.number_of_nodes == 4
    # Edge case: mesh of 2 nodes (the smallest valid).
    radii_2 = np.array([1.0, 2.0])
    fm2 = FixedMesh(
        settings=_settings(),
        radii=radii_2,
        mass_radii=radii_2,
        outer_boundary=2.0,
        inner_boundary=1.0,
    )
    assert fm2.number_of_nodes == 2


# ---- mixing_length dispatch ------------------------------------------------


def test_fixed_mesh_mixing_length_nearest_boundary():
    """nearest_boundary profile: l(r) = min(R_o - r, r - R_i).  At the
    midpoint r = (R_i + R_o) / 2, l = (R_o - R_i) / 2.  At the
    boundaries, l = 0.
    """
    fm = _basic_mesh_4_nodes(mixing_length_profile='nearest_boundary')
    ml = np.asarray(fm.mixing_length).ravel()
    # Boundary values
    assert ml[0] == pytest.approx(0.0, abs=1e-9)
    assert ml[-1] == pytest.approx(0.0, abs=1e-9)
    # Interior values are positive
    assert np.all(ml[1:-1] > 0)


def test_fixed_mesh_mixing_length_constant():
    """constant profile: l = 0.25 * (R_o - R_i) everywhere.

    Discriminator: 0.25 vs 0.5 vs 1.0 — using 4 nodes and Earth-like
    bounds gives ml = 0.25 * 2.891e6 = 7.2275e5 everywhere.
    """
    fm = _basic_mesh_4_nodes(mixing_length_profile='constant')
    expected = 0.25 * (6.371e6 - 3.480e6)
    np.testing.assert_allclose(
        np.asarray(fm.mixing_length).ravel(),
        np.full(4, expected),
        rtol=1e-12,
    )


def test_fixed_mesh_mixing_length_unknown_profile_raises():
    """Edge case: an unknown profile string must raise ValueError."""
    fm_bad = FixedMesh(
        settings=_MeshParameters(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=4,
            mixing_length_profile='not_a_profile',
            core_density=10500.0,
        ),
        radii=np.linspace(3.480e6, 6.371e6, 4),
        mass_radii=np.linspace(3.480e6, 6.371e6, 4),
        outer_boundary=6.371e6,
        inner_boundary=3.480e6,
    )
    with pytest.raises(ValueError, match='Mixing length profile'):
        _ = fm_bad.mixing_length


def test_fixed_mesh_mixing_length_squared_and_cubed_match_powers():
    """Properties: ml_squared == ml**2, ml_cubed == ml**3.

    Tests that the powers were not transposed (e.g. cube returning
    square would be invisible in a single test).
    """
    fm = _basic_mesh_4_nodes(mixing_length_profile='constant')
    ml = np.asarray(fm.mixing_length)
    np.testing.assert_allclose(np.asarray(fm.mixing_length_squared), ml**2, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fm.mixing_length_cubed), ml**3, rtol=1e-12)
    # Discriminator: ml_cubed must NOT equal ml_squared (use a non-unit value).
    assert not np.allclose(np.asarray(fm.mixing_length_cubed), ml**2, rtol=1e-3)
