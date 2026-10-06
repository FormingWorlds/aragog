"""Unit tests for ``aragog.core.cmb_boundary_layer_flux``.

The flux must carry the sign of ``T_core - T_mantle`` (heat never flows from
the colder side), equal the Foley & Driscoll / Thiriet boundary-layer value
where convection beats conduction across the half cell, fall back to that
conduction elsewhere, and stay differentiable for the JAX Jacobian.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from aragog.core import RA_CRIT_CMB_DEFAULT, cmb_boundary_layer_flux

pytestmark = pytest.mark.unit

# Bottom-cell properties: liquid magma (eta 0.1 Pa s) and solid rock (eta 1e21 Pa s).
_COMMON = dict(
    conductivity=4.0,
    density=4000.0,
    heat_capacity=1000.0,
    expansivity=3e-5,
    gravity=10.0,
    dr_half=1e4,
)
_LIQUID = dict(_COMMON, viscosity=0.1)
_SOLID = dict(_COMMON, viscosity=1e21)


def _flux(dT, props, t_mantle=4000.0, **kw):
    return float(cmb_boundary_layer_flux(t_mantle + dT, t_mantle, **props, **kw))


@pytest.mark.physics_invariant
@pytest.mark.parametrize('props', [_LIQUID, _SOLID], ids=['liquid', 'solid'])
def test_flux_sign_follows_the_temperature_contrast(props):
    """Second law at the CMB: F > 0 exactly when the core is hotter, F < 0 when colder."""
    dTs = np.array([-1000.0, -10.0, -1e-3, 0.0, 1e-3, 10.0, 1000.0])
    fluxes = np.asarray(cmb_boundary_layer_flux(4000.0 + dTs, 4000.0, **props))
    assert np.array_equal(np.sign(fluxes), np.sign(dTs))


@pytest.mark.physics_invariant
def test_convective_branch_matches_the_hand_computed_boundary_layer():
    """Liquid base, dT = 100 K: delta = (450 * 1e-6 * 0.1 / (3e-5 * 4000 * 10 * 100))^(1/3)
    = 7.2112e-3 m, so q = 4 * 100 / delta = 55468.90 W/m^2, far above conduction (0.04)."""
    assert _flux(100.0, _LIQUID) == pytest.approx(55468.90194805076, rel=1e-12)
    assert RA_CRIT_CMB_DEFAULT == 450.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('dT', [-500.0, 5.0, 500.0])
def test_solid_base_and_cold_core_conduct_across_the_half_cell(dT):
    """Solid base: delta_BL (4.2e5 m at 5 K) exceeds the 1e4 m half cell, so q = k dT / dr_half;
    a colder core gets the same conduction for any viscosity."""
    expected = 4.0 * dT / 1e4
    assert _flux(dT, _SOLID) == pytest.approx(expected, rel=1e-12)
    if dT < 0:
        assert _flux(dT, _LIQUID) == pytest.approx(expected, rel=1e-12)


def test_non_buoyant_layer_conducts():
    """Zero or negative expansivity removes the convective branch."""
    for alpha in (0.0, -1e-5):
        props = dict(_LIQUID, expansivity=alpha)
        assert _flux(100.0, props) == pytest.approx(4.0 * 100.0 / 1e4, rel=1e-9)


def test_ra_crit_scales_the_convective_flux():
    """q_conv scales as Ra_crit^(-1/3): a 8x larger Ra_crit halves the flux."""
    base = _flux(100.0, _LIQUID)
    assert _flux(100.0, _LIQUID, ra_crit=8.0 * RA_CRIT_CMB_DEFAULT) == pytest.approx(
        base / 2.0, rel=1e-12
    )


def test_flux_is_continuous_and_differentiable_through_zero():
    """The derivative is finite on both sides of dT = 0 (the JAX Jacobian needs it), and the
    flux tends to zero from both sides."""

    def f(t_core):
        return cmb_boundary_layer_flux(t_core, 4000.0, **_LIQUID)

    t_core = np.array([3999.0, 4000.0 - 1e-9, 4000.0, 4000.0 + 1e-9, 4001.0])
    assert np.isfinite(np.asarray(jax.vmap(jax.grad(f))(t_core))).all()
    assert np.abs(np.asarray(f(t_core))[1:4]).max() < 1e-6


def _kinked_temperature(s):
    """T(S) [K] with a nearly flat stretch (slope 1e-9 K kg K/J) on 9000 < S < 10500."""
    import jax.numpy as jnp

    flat = 5000.0 + 1e-9 * (s - 9000.0)
    below = 5000.0 + 5.0 * (s - 9000.0)
    above = flat + 5.0 * (s - 10500.0)
    return jnp.where(s < 9000.0, below, jnp.where(s > 10500.0, above, flat))


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    't_core', [4700.0, 5000.0 + 1e-10, 6000.0], ids=['below', 'flat', 'above']
)
def test_cmb_node_entropy_hits_the_core_temperature_from_a_flat_start(t_core):
    """The node entropy solves T(S_node) = T_core even when the bottom cell sits on a
    nearly flat stretch of T(S), where a plain Newton step from S0 jumps to S ~ -1e8.
    Pinned roots: 8940 below, any point of the flat stretch, and 10700 above."""
    from aragog.core.boundary_layer import cmb_node_gradient

    s0, dr_offset = 10300.0, -2.0e4
    s_node = s0 + dr_offset * float(
        cmb_node_gradient(t_core, s0, _kinked_temperature, dr_offset)
    )
    assert float(_kinked_temperature(s_node)) == pytest.approx(t_core, abs=1e-6)
    if t_core < 5000.0:
        assert s_node == pytest.approx(8940.0, abs=1e-6)
    elif t_core > 5001.0:
        assert s_node == pytest.approx(10700.0, abs=1e-3)


def test_cmb_node_gradient_carries_the_core_temperature_sensitivity():
    """d(S_node)/d(T_core) = 1 / T'(S_node) for the JAX Jacobian (1/5 on the steep part),
    and S_node does not depend on the bottom entropy it starts from."""
    import jax

    from aragog.core.boundary_layer import cmb_node_gradient

    def s_node(t_core, s0):
        return s0 + (-2.0e4) * cmb_node_gradient(t_core, s0, _kinked_temperature, -2.0e4)

    d_tc, d_s0 = jax.grad(s_node, argnums=(0, 1))(4700.0, 10300.0)
    assert float(d_tc) == pytest.approx(0.2, rel=1e-6)
    assert abs(float(d_s0)) < 1e-6
