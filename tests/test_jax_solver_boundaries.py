"""Tests for JAX exports and boundary condition branches.

Exercises exported rheology functions in ``aragog.jax`` and boundary condition
error and flux law branches in ``aragog.jax.solver``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax import (
    BoundaryParams,
    compute_arrhenius_viscosity,
    compute_effective_viscosity,
    compute_yield_stress,
)
from aragog.jax.solver import _apply_cmb_bc, _apply_surface_bc


def _make_bc(**kwargs) -> BoundaryParams:
    """Helper to build BoundaryParams with standard default values."""
    defaults = {
        'outer_bc_type': 1,
        'outer_bc_value': 0.0,
        'emissivity': 1.0,
        'T_eq': 255.0,
        'inner_bc_type': 1,
        'inner_bc_value': 0.0,
        'core_density': 10500.0,
        'core_heat_capacity': 880.0,
        'tfac_core_avg': 1.147,
    }
    defaults.update(kwargs)
    return BoundaryParams(**defaults)


@pytest.mark.unit
def test_jax_arrhenius_viscosity_export_contract():
    """compute_arrhenius_viscosity exported from aragog.jax matches analytic scaling."""
    R = 8.314462618
    T = 1600.0
    P = 3.0e9
    V = 1.0e-5
    eta_0 = 1.0e20
    expected = eta_0 * np.exp((P * V) / (R * T))
    res = compute_arrhenius_viscosity(
        temperature=T,
        pressure=P,
        viscosity_solid=eta_0,
        activation_energy=0.0,
        activation_volume=V,
        arrhenius_t_ref=T,
        xp=np,
    )
    np.testing.assert_allclose(res, expected, rtol=1e-12)


@pytest.mark.unit
def test_jax_yield_stress_export_contract():
    """compute_yield_stress exported from aragog.jax evaluates Byerlee envelope."""
    P = 1.0e8
    C = 1.0e7
    mu = 0.6
    tau_max = 1.0e9
    expected = min(C + mu * P, tau_max)
    res = compute_yield_stress(
        pressure=P,
        yield_stress_c=C,
        yield_stress_mu=mu,
        yield_stress_max=tau_max,
        xp=np,
    )
    assert res == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_jax_effective_viscosity_export_contract():
    """compute_effective_viscosity exported from aragog.jax in lid and local modes."""
    eta_d = 1.0e21
    tau_d = 1.0e6
    tau_y = 5.0e5
    v_i = 1.0e-9
    delta_rh = 1.0e4
    eta_i = 1.0e20
    res_lid = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_d=tau_d,
        tau_y_lid=tau_y,
        v_i=v_i,
        delta_rh=delta_rh,
        eta_i=eta_i,
        stress_closure_mode='lid',
        w_lid=1.0,
        xp=np,
    )
    assert np.isfinite(res_lid)
    assert res_lid < eta_d

    res_local = compute_effective_viscosity(
        eta_diff=np.array([eta_d]),
        stress_closure_mode='local',
        unyielded_velocity=np.array([v_i]),
        mixing_length=np.array([delta_rh]),
        tau_y_profile=np.array([tau_y]),
        xp=np,
    )
    assert np.all(np.isfinite(res_local))
    assert res_local[0] < eta_d


@pytest.mark.unit
def test_apply_surface_bc_requires_phase_stag_and_mesh_for_types_5_and_6():
    """_apply_surface_bc raises ValueError if phase_stag or mesh is None for types 5 and 6."""
    bc_type5 = _make_bc(outer_bc_type=5)
    bc_type6 = _make_bc(outer_bc_type=6)
    dummy_flux = jnp.zeros(5)
    dummy_T = jnp.zeros(5)

    with pytest.raises(
        ValueError, match='phase_stag and mesh are required for outer_bc_type 5 and 6'
    ):
        _apply_surface_bc(dummy_flux, bc_type5, dummy_T, phase_stag=None, mesh=None)

    with pytest.raises(
        ValueError, match='phase_stag and mesh are required for outer_bc_type 5 and 6'
    ):
        _apply_surface_bc(dummy_flux, bc_type6, dummy_T, phase_stag=None, mesh=None)


@pytest.mark.unit
def test_boundary_params_rejects_unsupported_outer_bc_type():
    """BoundaryParams.__init__ raises ValueError on invalid outer_bc_type."""
    with pytest.raises(
        ValueError, match='Unsupported outer_bc_type: 99; expected 1, 4, 5, or 6'
    ):
        _make_bc(outer_bc_type=99)


@pytest.mark.unit
def test_apply_surface_bc_rejects_unsupported_outer_bc_type():
    """_apply_surface_bc raises ValueError if outer_bc_type is invalid."""
    bc_bad = SimpleNamespace(
        outer_bc_type=99,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        param_utbl=False,
        table_edge_cutoff=False,
    )
    dummy_flux = jnp.zeros(5)
    dummy_T = jnp.zeros(5)

    with pytest.raises(ValueError, match='Unsupported outer_bc_type: 99'):
        _apply_surface_bc(dummy_flux, bc_bad, dummy_T)


@pytest.mark.unit
def test_apply_cmb_bc_with_cmb_flux_law():
    """_apply_cmb_bc exercises JAX CMB flux law under inner_bc_type 1 and 3."""
    n = 5
    radii = jnp.linspace(3.48e6, 6.371e6, n)
    volume = 4.0 / 3.0 * jnp.pi * (radii[1:] ** 3 - radii[:-1] ** 3)
    area = 4.0 * jnp.pi * radii**2
    gravity = jnp.full(n, 9.81)
    P_basic = jnp.linspace(1.3e11, 0.0, n)
    mesh = SimpleNamespace(
        radii_basic=radii,
        volume=volume,
        area=area,
        gravity=gravity,
        P_basic=P_basic,
    )

    phase_stag_rho = jnp.full(n - 1, 4000.0)
    phase_stag_Cp = jnp.full(n - 1, 1200.0)
    phase_stag_T = jnp.full(n - 1, 3500.0)
    phase_stag_k = jnp.full(n - 1, 4.0)

    class MockEOS:
        def temperature(self, P, S):
            return jnp.full(P.shape, 3000.0)

    phase_stag = SimpleNamespace(
        density=phase_stag_rho,
        heat_capacity=phase_stag_Cp,
        thermal_conductivity=phase_stag_k,
        thermal_expansivity=jnp.full(n - 1, 3e-5),
        viscosity=jnp.full(n - 1, 1.0e21),
        temperature=phase_stag_T,
    )
    S = jnp.full(n - 1, 2000.0)
    T_top = 1800.0
    law_inputs = (MockEOS(), phase_stag, S, T_top)

    heat_flux = jnp.zeros(n)
    bc1 = _make_bc(
        inner_bc_type=1,
        cmb_flux_law=True,
        cmb_law_interior=jnp.ones(n - 1),
    )
    hf_out1 = _apply_cmb_bc(
        heat_flux,
        bc1,
        mesh,
        phase_stag_rho,
        phase_stag_Cp,
        heating_first=0.0,
        phase_stag_T=phase_stag_T,
        law_inputs=law_inputs,
    )
    assert jnp.isfinite(hf_out1[0])
    assert jnp.isfinite(hf_out1[1])

    bc3 = _make_bc(
        inner_bc_type=3,
        inner_bc_value=3600.0,
        cmb_flux_law=True,
        cmb_law_interior=jnp.ones(n - 1),
    )
    hf_out3 = _apply_cmb_bc(
        heat_flux,
        bc3,
        mesh,
        phase_stag_rho,
        phase_stag_Cp,
        heating_first=0.0,
        phase_stag_T=phase_stag_T,
        law_inputs=law_inputs,
    )
    assert jnp.isfinite(hf_out3[0])
