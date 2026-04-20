"""Regression test for the JAX mesh gravity-fallback bug.

When the aragog mesh's pressure-EOS object is provided externally (e.g.,
a Zalmoxis-derived PALEOS-2phase EOS), it does not expose the private
attribute ``_gravitational_acceleration`` that the JAX mesh builder was
reading. Before the fix, ``MeshArrays.from_numpy_mesh`` silently fell
back to ``0.0``, zeroing every MLT-convection quantity downstream and
making the JAX RHS return values ~1e6 to 1e7 smaller than the numpy
RHS on PALEOS-2phase. See memory ``stage1a_jax_paleos2phase_rhs_bug``.

This test builds a minimal stub mesh with ``mesh.eos`` missing the
attribute, and asserts that ``MeshArrays.from_numpy_mesh`` reads the
correct value from ``mesh.settings.gravitational_acceleration`` instead.

Without the fix: gravity array is zeros ⇒ MLT collapses ⇒ RHS under-
responds by orders of magnitude.
With the fix: gravity array carries the configured value end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)

from aragog.jax.phase import MeshArrays


class _StubEOS:
    """Stands in for a Zalmoxis/PALEOS-provided EOS: lacks the private
    ``_gravitational_acceleration`` attribute."""


class _StubSettings:
    def __init__(self, g: float):
        self.gravitational_acceleration = g


class _StubBasic:
    def __init__(self, n: int):
        self.radii = np.linspace(3.48e6, 6.37e6, n)
        self.area = np.ones(n)
        self.volume = np.ones(n)
        self.mixing_length = np.full(n, 9.0e5)
        self.mixing_length_squared = self.mixing_length ** 2
        self.mixing_length_cubed = self.mixing_length ** 3


class _StubStaggered:
    def __init__(self, n: int):
        self.radii = np.linspace(3.5e6, 6.35e6, n)


class _StubMesh:
    """Minimal mesh stub reproducing the bug trigger: mesh.eos without
    ``_gravitational_acceleration`` AND mesh.settings with the attribute."""
    def __init__(self, configured_g: float = 9.81, n_basic: int = 10):
        self.eos = _StubEOS()
        self.settings = _StubSettings(configured_g)
        self.basic = _StubBasic(n_basic)
        self.staggered = _StubStaggered(n_basic - 1)
        self.basic_pressure = np.linspace(1.5e11, 1.0e5, n_basic)
        self.staggered_pressure = np.linspace(1.4e11, 1.1e5, n_basic - 1)
        self._d_dr_transform = np.eye(n_basic, n_basic - 1)
        self._quantity_transform = np.eye(n_basic, n_basic - 1)


@pytest.mark.unit
def test_mesh_gravity_falls_back_to_settings_when_eos_lacks_attribute():
    """MeshArrays.from_numpy_mesh must use mesh.settings.gravitational_acceleration
    when mesh.eos._gravitational_acceleration is absent. The pre-fix fallback was
    0.0, which silently zeroed MLT convection on PALEOS-2phase runs."""
    configured_g = 9.81
    mesh = _StubMesh(configured_g=configured_g)
    mesh_jax = MeshArrays.from_numpy_mesh(mesh)
    gravity = np.asarray(mesh_jax.gravity)
    assert gravity.shape == mesh.basic.radii.shape
    assert np.all(gravity > 0.0), (
        f'Gravity zeroed: {gravity}. Regression of PALEOS-2phase bug.'
    )
    assert np.allclose(gravity, configured_g, atol=0.0, rtol=1e-12), (
        f'Expected uniform gravity={configured_g}, got {gravity}'
    )


@pytest.mark.unit
def test_mesh_gravity_prefers_eos_attribute_when_available():
    """When mesh.eos._gravitational_acceleration exists, it wins over the
    settings fallback. This preserves the WB2018 code path that stored
    surface-density * g on the pressure-EOS object."""
    eos_g = 7.977
    settings_g = 1.234
    mesh = _StubMesh(configured_g=settings_g)
    mesh.eos._gravitational_acceleration = eos_g
    mesh_jax = MeshArrays.from_numpy_mesh(mesh)
    gravity = np.asarray(mesh_jax.gravity)
    assert np.allclose(gravity, eos_g, atol=0.0, rtol=1e-12), (
        f'Expected eos override {eos_g}, got {gravity}'
    )


@pytest.mark.unit
def test_mesh_gravity_uses_per_node_profile_when_eos_gravity_present():
    """Stage 1c regression: when the external mesh supplies eos_radius +
    eos_gravity columns, MeshArrays.from_numpy_mesh must interpolate that
    profile onto the basic-node radii rather than broadcasting a single
    scalar. Earth-like radial profile: g rising from ~8 m/s^2 at CMB to
    ~10 m/s^2 at surface."""
    mesh = _StubMesh(configured_g=9.81)
    # Populate mesh.parameters with the external eos arrays, matching the
    # shape that entropy_solver.reset() writes when eos_method=2.
    import types
    mesh.parameters = types.SimpleNamespace()
    mesh.parameters.eos_radius = np.linspace(3.48e6, 6.37e6, 60)
    mesh.parameters.eos_gravity = np.linspace(8.0, 10.0, 60)  # CMB to surface
    mesh_jax = MeshArrays.from_numpy_mesh(mesh)
    gravity = np.asarray(mesh_jax.gravity)
    assert gravity.shape == mesh.basic.radii.shape
    # Per-node profile: gravity should NOT be uniform
    assert gravity.max() - gravity.min() > 0.5, (
        f'Expected non-uniform g; got min={gravity.min()}, max={gravity.max()}'
    )
    # Endpoints should match the supplied profile at the mesh boundaries
    assert np.isclose(gravity[0], 8.0, atol=5e-2), (
        f'Inner gravity {gravity[0]} far from CMB source 8.0'
    )
    assert np.isclose(gravity[-1], 10.0, atol=5e-2), (
        f'Outer gravity {gravity[-1]} far from surface source 10.0'
    )


@pytest.mark.unit
def test_mesh_gravity_profile_beats_eos_scalar_attribute():
    """When both the profile (mesh.parameters.eos_gravity) AND the scalar
    (mesh.eos._gravitational_acceleration) are available, the per-node
    profile wins. A run that ships a full Zalmoxis profile should not be
    silently flattened by a stale scalar attribute."""
    import types
    mesh = _StubMesh(configured_g=9.81)
    mesh.eos._gravitational_acceleration = 5.0  # scalar intentionally wrong
    mesh.parameters = types.SimpleNamespace()
    mesh.parameters.eos_radius = np.linspace(3.48e6, 6.37e6, 30)
    mesh.parameters.eos_gravity = np.linspace(8.0, 10.0, 30)
    mesh_jax = MeshArrays.from_numpy_mesh(mesh)
    gravity = np.asarray(mesh_jax.gravity)
    assert gravity.max() - gravity.min() > 0.5, (
        'Stale scalar attribute flattened the per-node profile'
    )
    assert gravity.min() > 7.0 and gravity.max() < 11.0, (
        f'Profile corrupted; got range [{gravity.min()}, {gravity.max()}]'
    )
