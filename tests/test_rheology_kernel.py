"""Unit tests for the shared array-agnostic rheology kernel.

Validates asymptotic physical limits, monotonicity, continuity,
and autodiff properties of the kernel in aragog.rheology.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_enthalpy,
    compute_effective_viscosity,
    compute_stagnant_lid_state,
)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_zero_driving_stress():
    """Verify effective viscosity equals diffusion creep when tau_d is zero."""
    eta_diff = 1.0e22
    tau_y_lid = 1.0e7
    v_i = 1.0e-9
    delta_rh = 5.0e4
    eta_i = 1.0e20

    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_d=0.0,
        tau_y_lid=tau_y_lid,
        v_i=v_i,
        delta_rh=delta_rh,
        eta_i=eta_i,
        stress_closure_mode='lid',
    )
    assert eta_eff == pytest.approx(eta_diff, rel=1.0e-8)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_infinite_yield_stress():
    """Verify effective viscosity equals diffusion creep when tau_y is infinite."""
    eta_diff = 1.0e22
    tau_d = 5.0e6
    tau_y_lid = float('inf')
    v_i = 1.0e-9
    delta_rh = 5.0e4
    eta_i = 1.0e20

    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        v_i=v_i,
        delta_rh=delta_rh,
        eta_i=eta_i,
        stress_closure_mode='lid',
    )
    assert eta_eff == pytest.approx(eta_diff, rel=1.0e-10)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_zero_activation_volume():
    """Verify activation enthalpy is constant when activation volume is zero."""
    e_a = 300.0e3
    p_values = np.linspace(0.0, 1.0e11, 20)
    enthalpy = compute_arrhenius_enthalpy(
        p_values,
        activation_energy=e_a,
        activation_volume=0.0,
        activation_volume_decay_pressure=float('inf'),
        xp=np,
    )
    np.testing.assert_allclose(enthalpy, e_a, rtol=1.0e-14)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_infinite_decay_pressure():
    """Verify activation enthalpy is linear in pressure when decay pressure is infinite."""
    e_a = 300.0e3
    v_0 = 5.0e-6
    p_values = np.linspace(0.0, 1.0e11, 25)
    enthalpy = compute_arrhenius_enthalpy(
        p_values,
        activation_energy=e_a,
        activation_volume=v_0,
        activation_volume_decay_pressure=float('inf'),
        xp=np,
    )
    expected = e_a + p_values * v_0
    np.testing.assert_allclose(enthalpy, expected, rtol=1.0e-14)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_extreme_yielding():
    """Verify yielded branch asymptotically governs at large driving stress."""
    eta_diff = 1.0e25
    tau_y_lid = 50.0e6
    v_i = 1.0e-9
    delta_rh = 1.0e4
    eta_i = 1.0e20
    tau_d = 100.0 * tau_y_lid

    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        v_i=v_i,
        delta_rh=delta_rh,
        eta_i=eta_i,
        yield_switch_width=0.05,
        stress_closure_mode='lid',
    )
    expected_yielded = min((tau_y_lid * delta_rh) / v_i, eta_diff)
    assert eta_eff == pytest.approx(expected_yielded, rel=1.0e-3)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_exact_yield_match():
    """Verify both branches equal eta_i at exact yield when tau_d = tau_y_lid."""
    eta_diff = 1.0e24
    tau_y_lid = 20.0e6
    v_i = 1.0e-9
    delta_rh = 1.0e4
    eta_i = (tau_y_lid * delta_rh) / v_i  # 2.0e20 Pa s
    tau_d = tau_y_lid

    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        v_i=v_i,
        delta_rh=delta_rh,
        eta_i=eta_i,
        yield_switch_width=0.1,
        stress_closure_mode='lid',
    )
    assert eta_eff == pytest.approx(eta_i, rel=1.0e-6)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_monotonicity_and_continuity_across_yield():
    """Verify effective viscosity is monotonic non-increasing and continuous."""
    n_pts = 10000
    tau_y_lid = 50.0e6
    v_i = 1.0e-9
    delta_rh = 2.0e4
    eta_i = 1.0e21
    eta_diff = 1.0e25

    stress_ratios = np.linspace(0.0, 10.0, n_pts)
    tau_d_values = stress_ratios * tau_y_lid

    eta_eff_vals = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_d=tau_d_values,
        tau_y_lid=tau_y_lid,
        v_i=v_i,
        delta_rh=delta_rh,
        eta_i=eta_i,
        yield_switch_width=0.1,
        stress_closure_mode='lid',
        xp=np,
    )

    diffs = np.diff(eta_eff_vals)
    assert np.all(diffs <= 1.0e-12), 'Effective viscosity must not increase with stress'

    log_diffs = np.abs(np.diff(np.log10(eta_eff_vals)))
    assert np.all(log_diffs < 0.5), 'Effective viscosity in log-space must be continuous'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_no_convection_lid_diagnostics():
    """Verify inactive convection suppresses lid diagnostics."""
    n_nodes = 30
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    temperature = np.linspace(3500.0, 1600.0, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    conv_flux = np.zeros(n_nodes)
    total_flux = np.full(n_nodes, 0.05)
    melt_frac = np.zeros(n_nodes)
    params = SolidRheologyParams(enabled=True, stress_closure_mode='lid')

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.zeros(n_nodes),
        xp=np,
    )

    assert state['w_active'] < 0.02
    assert state['tau_d'] == pytest.approx(0.0, abs=1.0e-10)
    assert state['d_lid'] < 0.01 * (radii[-1] - radii[0])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_whole_mantle_mush_clips_lid():
    """Verify lid thickness is zero when mantle is entirely mushy."""
    n_nodes = 30
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    temperature = np.linspace(3500.0, 1600.0, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    conv_flux = np.full(n_nodes, 100.0)
    total_flux = np.full(n_nodes, 100.0)
    melt_frac = np.full(n_nodes, 0.8)
    params = SolidRheologyParams(enabled=True, stress_closure_mode='lid', phi_visc_single=0.5)

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        xp=np,
    )

    assert state['d_lid'] == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.unit
def test_jax_autodiff_infinite_decay_pressure():
    """Verify JAX jacrev is finite at P_decay = inf."""
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')

    jax.config.update('jax_enable_x64', True)

    def _eval(p):
        return compute_arrhenius_enthalpy(
            p,
            activation_energy=300e3,
            activation_volume=5e-6,
            activation_volume_decay_pressure=jnp.inf,
            xp=jnp,
        )

    grad_fn = jax.grad(_eval)
    grad_val = grad_fn(10.0e9)
    assert jnp.isfinite(grad_val)
    assert float(grad_val) == pytest.approx(5.0e-6, rel=1.0e-10)
