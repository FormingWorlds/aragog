"""Parity tests between NumPy and JAX rheology kernel implementations.

Validates machine-precision agreement across physical regimes and ensures
autodiff reversibility without ConcretizationTypeError.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_viscosity,
    compute_effective_viscosity,
    compute_stagnant_lid_state,
    compute_yield_stress,
)

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

jax.config.update('jax_enable_x64', True)

from aragog.jax import rheology as jax_rheo  # noqa: E402


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_arrhenius_viscosity_parity_float64():
    """Verify Arrhenius viscosity evaluates identically in NumPy and JAX to 1e-12."""
    t_vals = np.linspace(1400.0, 3000.0, 50)
    p_vals = np.linspace(0.0, 135.0e9, 50)

    eta_np = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        activation_energy=300.0e3,
        activation_volume=5.0e-6,
        activation_volume_decay_pressure=100.0e9,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )

    eta_jx = jax_rheo.compute_arrhenius_viscosity(
        jnp.array(t_vals),
        jnp.array(p_vals),
        viscosity_solid=1.0e21,
        activation_energy=300.0e3,
        activation_volume=5.0e-6,
        activation_volume_decay_pressure=100.0e9,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
    )

    np.testing.assert_allclose(np.array(eta_jx), eta_np, rtol=1.0e-12, atol=1.0e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_yield_stress_parity_float64():
    """Verify yield stress evaluates identically in NumPy and JAX to 1e-12."""
    p_vals = np.linspace(0.0, 135.0e9, 50)

    tau_np = compute_yield_stress(
        p_vals,
        yield_stress_c=50.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
        xp=np,
    )

    tau_jx = jax_rheo.compute_yield_stress(
        jnp.array(p_vals),
        yield_stress_c=50.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
    )

    np.testing.assert_allclose(np.array(tau_jx), tau_np, rtol=1.0e-12, atol=1.0e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'regime',
    ['unyielded', 'yielding', 'mobile', 'no_lid', 'mushy'],
)
def test_lid_state_and_effective_viscosity_parity(regime: str):
    """Verify lid diagnostics and effective viscosity match between NumPy and JAX."""
    n_nodes = 40
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)

    if regime == 'unyielded':
        temperature = np.linspace(3000.0, 1800.0, n_nodes)
        conv_flux = np.linspace(0.0, 50.0, n_nodes)
        v_unyielded = np.linspace(0.0, 1.0e-10, n_nodes)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 500.0e6
    elif regime == 'yielding':
        temperature = np.linspace(3200.0, 1600.0, n_nodes)
        conv_flux = np.linspace(0.0, 200.0, n_nodes)
        v_unyielded = np.linspace(0.0, 1.0e-9, n_nodes)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 50.0e6
    elif regime == 'mobile':
        temperature = np.linspace(3400.0, 1400.0, n_nodes)
        conv_flux = np.linspace(0.0, 1000.0, n_nodes)
        v_unyielded = np.linspace(0.0, 1.0e-8, n_nodes)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 10.0e6
    elif regime == 'no_lid':
        temperature = np.full(n_nodes, 2200.0)
        conv_flux = np.full(n_nodes, 100.0)
        v_unyielded = np.full(n_nodes, 1.0e-9)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 100.0e6
    elif regime == 'mushy':
        temperature = np.linspace(3200.0, 2000.0, n_nodes)
        conv_flux = np.linspace(10.0, 500.0, n_nodes)
        v_unyielded = np.linspace(1.0e-11, 1.0e-9, n_nodes)
        melt_frac = np.linspace(0.6, 0.1, n_nodes)
        tau_y_max = 50.0e6
    else:
        raise ValueError(f'Unknown regime: {regime}')

    tot_flux = conv_flux + 10.0
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_max=tau_y_max,
    )

    state_np = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=tot_flux,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=v_unyielded,
        viscosity_solid=1.0e21,
        xp=np,
    )

    state_jx = jax_rheo.compute_stagnant_lid_state(
        radii=jnp.array(radii),
        temperature=jnp.array(temperature),
        pressure=jnp.array(pressure),
        convective_flux=jnp.array(conv_flux),
        total_flux=jnp.array(tot_flux),
        solidus_temperature=None,
        melt_fraction=jnp.array(melt_frac),
        params=params,
        unyielded_velocity=jnp.array(v_unyielded),
        viscosity_solid=1.0e21,
    )

    # Compare diagnostic scalars
    for key in [
        'T_i',
        'P_T_i',
        'dT_rh',
        'theta',
        'd_lid',
        'v_i',
        'tau_d',
        'tau_y_lid',
        'w_active',
    ]:
        val_np = float(state_np[key])
        val_jx = float(state_jx[key])
        np.testing.assert_allclose(
            val_jx,
            val_np,
            rtol=1.0e-9,
            atol=1.0e-12,
            err_msg=f'Mismatch in diagnostic {key} for regime {regime}',
        )

    # Compare w_lid profile
    np.testing.assert_allclose(
        np.array(state_jx['w_lid']),
        state_np['w_lid'],
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg=f'Mismatch in w_lid for regime {regime}',
    )

    # Compute effective viscosity
    eta_diff_np = compute_arrhenius_viscosity(
        temperature,
        pressure,
        viscosity_solid=1.0e21,
        activation_energy=params.activation_energy,
        activation_volume=params.activation_volume,
        arrhenius_t_ref=params.arrhenius_t_ref,
        xp=np,
    )
    eta_diff_jx = jnp.array(eta_diff_np)

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        tau_d=state_np['tau_d'],
        tau_y_lid=state_np['tau_y_lid'],
        v_i=state_np['v_i'],
        delta_rh=state_np['delta_rh'],
        eta_i=state_np['eta_i'],
        yield_switch_width=params.yield_switch_width,
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )

    eta_eff_jx = jax_rheo.compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        tau_d=state_jx['tau_d'],
        tau_y_lid=state_jx['tau_y_lid'],
        v_i=state_jx['v_i'],
        delta_rh=state_jx['delta_rh'],
        eta_i=state_jx['eta_i'],
        yield_switch_width=params.yield_switch_width,
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
    )

    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg=f'Mismatch in eta_eff for regime {regime}',
    )


@pytest.mark.unit
def test_jax_jacrev_closure_differentiability():
    """Verify JAX reverse-mode autodiff through the full closure without concretization error."""
    n_nodes = 20
    radii = jnp.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = jnp.linspace(135.0e9, 1.0e5, n_nodes)
    v_unyielded = jnp.linspace(1.0e-11, 1.0e-9, n_nodes)
    conv_flux = jnp.linspace(0.0, 200.0, n_nodes)
    tot_flux = conv_flux + 10.0
    melt_frac = jnp.zeros(n_nodes)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
    )

    def _loss(T_profile):
        lid_st = jax_rheo.compute_stagnant_lid_state(
            radii=radii,
            temperature=T_profile,
            pressure=pressure,
            convective_flux=conv_flux,
            total_flux=tot_flux,
            solidus_temperature=None,
            melt_fraction=melt_frac,
            params=params,
            unyielded_velocity=v_unyielded,
            viscosity_solid=1.0e21,
        )
        eta_d = jax_rheo.compute_arrhenius_viscosity(
            T_profile,
            pressure,
            viscosity_solid=1.0e21,
            activation_energy=params.activation_energy,
            activation_volume=params.activation_volume,
            arrhenius_t_ref=params.arrhenius_t_ref,
        )
        eta_eff = jax_rheo.compute_effective_viscosity(
            eta_diff=eta_d,
            tau_d=lid_st['tau_d'],
            tau_y_lid=lid_st['tau_y_lid'],
            v_i=lid_st['v_i'],
            delta_rh=lid_st['delta_rh'],
            eta_i=lid_st['eta_i'],
            yield_switch_width=params.yield_switch_width,
            stress_closure_mode='lid',
            w_lid=lid_st['w_lid'],
        )
        return jnp.sum(jnp.log10(eta_eff))

    T_init = jnp.linspace(3200.0, 1600.0, n_nodes)
    grad_fn = jax.grad(_loss)
    grad_val = grad_fn(T_init)

    assert grad_val.shape == (n_nodes,)
    assert jnp.all(jnp.isfinite(grad_val))
