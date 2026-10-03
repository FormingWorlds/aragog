"""Tests for JAX compute_mlt rheology and boundary layer branches.

Exercises disabled rheology, local stress closure, q-factor velocity modulation,
and unyielded flux returns in ``aragog.jax.phase.compute_mlt``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt


def _make_mesh_and_phase(n_basic: int = 21) -> tuple[MeshArrays, PhaseProperties]:
    """Helper to create minimal valid MeshArrays and PhaseProperties."""
    r = np.linspace(3.48e6, 6.371e6, n_basic)
    ml = np.maximum(np.minimum(r - r[0], r[-1] - r), 1.0)
    mesh = MeshArrays(
        d_dr_matrix=jnp.zeros((n_basic, n_basic - 1)),
        quantity_matrix=jnp.zeros((n_basic, n_basic - 1)),
        area=jnp.ones(n_basic),
        volume=jnp.ones(n_basic),
        radii_basic=jnp.asarray(r),
        radii_stag=jnp.asarray(0.5 * (r[1:] + r[:-1])),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.zeros(n_basic - 1),
        P_basic=jnp.zeros(n_basic),
        dP_dr_basic=jnp.zeros(n_basic),
        gravity=jnp.full(n_basic, 9.81),
    )
    ones = jnp.ones(n_basic)
    phase = PhaseProperties(
        temperature=ones * 2000.0,
        density=ones * 4000.0,
        heat_capacity=ones * 1200.0,
        thermal_expansivity=ones * 3e-5,
        dTdPs=ones,
        melt_fraction=ones * 0.0,
        viscosity=ones * 1.0e21,
        kinematic_viscosity=ones * 1.0e21 / 4000.0,
        thermal_conductivity=ones * 4.0,
        latent_heat=ones,
        capacitance=ones * 4000.0 * 2000.0,
        eta_diff=ones * 1.0e21,
        tau_y=ones * 1.0e30,  # Unyielded by default
        visc_solid_weight=ones * 1.0,
    )
    return mesh, phase


@pytest.mark.unit
def test_jax_compute_mlt_rheology_disabled_contract():
    """compute_mlt with enabled=False executes unyielded kinematic viscosity branch."""
    mesh, phase = _make_mesh_and_phase()
    grad = jnp.full(len(mesh.radii_basic), -1.0e-6)
    params = PhaseParams(enabled=False, kappah_floor=0.0)

    kh, kc = compute_mlt(grad, phase, mesh, params)
    assert np.all(np.isfinite(kh))
    assert np.all(kh[1:] > 0.0)
    assert np.all(kc >= 0.0)


@pytest.mark.unit
def test_jax_compute_mlt_local_stress_mode_contract():
    """compute_mlt with stress_closure_mode='local' applies local yielding."""
    mesh, phase_noyield = _make_mesh_and_phase()
    ones = jnp.ones(len(mesh.radii_basic))
    phase_yield = PhaseProperties(
        temperature=phase_noyield.temperature,
        density=phase_noyield.density,
        heat_capacity=phase_noyield.heat_capacity,
        thermal_expansivity=phase_noyield.thermal_expansivity,
        dTdPs=phase_noyield.dTdPs,
        melt_fraction=phase_noyield.melt_fraction,
        viscosity=phase_noyield.viscosity,
        kinematic_viscosity=phase_noyield.kinematic_viscosity,
        thermal_conductivity=phase_noyield.thermal_conductivity,
        latent_heat=phase_noyield.latent_heat,
        capacitance=phase_noyield.capacitance,
        eta_diff=phase_noyield.eta_diff,
        tau_y=ones * 1.0,  # Low yield stress induces yielding
        visc_solid_weight=phase_noyield.visc_solid_weight,
    )
    grad = jnp.full(len(mesh.radii_basic), -1.0e-6)
    params = PhaseParams(
        enabled=True,
        stress_closure_mode='local',
        kappah_floor=0.0,
    )

    kh_yield, _ = compute_mlt(grad, phase_yield, mesh, params)
    kh_noyield, _ = compute_mlt(grad, phase_noyield, mesh, params)

    assert np.all(np.isfinite(kh_yield))
    assert np.all(np.isfinite(kh_noyield))
    # Plastic softening under local yielding lowers viscosity and increases kappa_h
    mid = len(mesh.radii_basic) // 2
    assert kh_yield[mid] > kh_noyield[mid]


@pytest.mark.unit
def test_jax_compute_mlt_q_factor_scaling_contract():
    """compute_mlt scales viscous velocity by q factor when slopes are below 1."""
    mesh, phase = _make_mesh_and_phase()
    grad = jnp.full(len(mesh.radii_basic), -1.0e-6)
    params_full = PhaseParams(
        enabled=True,
        stress_closure_mode='local',
        mlt_top_slope=1.0,
        mlt_bottom_slope=1.0,
        kappah_floor=0.0,
    )
    params_tapered = PhaseParams(
        enabled=True,
        stress_closure_mode='local',
        mlt_top_slope=0.5,
        mlt_bottom_slope=1.0,
        kappah_floor=0.0,
    )

    kh_full, _ = compute_mlt(grad, phase, mesh, params_full)
    kh_tapered, _ = compute_mlt(grad, phase, mesh, params_tapered)

    # Near surface (top cell), tapered top slope must reduce kappa_h
    top = -2
    assert kh_tapered[top] < kh_full[top]


@pytest.mark.unit
def test_jax_compute_mlt_unyielded_flux_contract():
    """compute_mlt returns unyielded flux profile in local and disabled modes."""
    mesh, phase = _make_mesh_and_phase()
    grad = jnp.full(len(mesh.radii_basic), -1.0e-6)

    # 1. Local mode with return_unyielded=True
    params_local = PhaseParams(
        enabled=True,
        stress_closure_mode='local',
        kappah_floor=0.0,
    )
    kh, kc, f_unyielded = compute_mlt(grad, phase, mesh, params_local, return_unyielded=True)
    assert np.all(np.isfinite(f_unyielded))
    mid = len(mesh.radii_basic) // 2
    assert f_unyielded[mid] > 0.0

    # 2. Disabled mode with return_unyielded=True
    params_disabled = PhaseParams(enabled=False, kappah_floor=0.0)
    kh_d, kc_d, f_unyielded_d = compute_mlt(
        grad, phase, mesh, params_disabled, return_unyielded=True
    )
    assert np.all(np.isfinite(f_unyielded_d))
    assert f_unyielded_d[mid] > 0.0
