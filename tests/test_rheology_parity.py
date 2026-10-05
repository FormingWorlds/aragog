"""Parity tests between NumPy and JAX rheology kernel implementations.

Validates machine-precision agreement across physical regimes and ensures
autodiff reversibility without ConcretizationTypeError.
"""

from __future__ import annotations

import os

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
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(0.0, 1.0e-10, n_nodes)
        v_unyielded[-8:] = 0.0
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 500.0e6
    elif regime == 'yielding':
        temperature = np.linspace(3200.0, 1600.0, n_nodes)
        conv_flux = np.linspace(0.0, 200.0, n_nodes)
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(0.0, 1.0e-9, n_nodes)
        v_unyielded[-8:] = 0.0
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 50.0e6
    elif regime == 'mobile':
        temperature = np.linspace(3400.0, 1400.0, n_nodes)
        conv_flux = np.linspace(0.0, 1000.0, n_nodes)
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(0.0, 1.0e-8, n_nodes)
        v_unyielded[-8:] = 0.0
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
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(1.0e-11, 1.0e-9, n_nodes)
        v_unyielded[-8:] = 0.0
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
        'v_m',
        'tau_d',
        'tau_buoy',
        'tau_d_over_tau_buoy',
        'Ra_eff',
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

    if regime in ('unyielded', 'yielding', 'mobile'):
        assert state_np['d_lid'] > 1.0e4, f'{regime} must form stagnant lid > 10 km'
        assert state_np['w_active'] > 0.5, f'{regime} must have active lid closure'

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
            stress_closure_mode='lid',
            w_lid=lid_st['w_lid'],
        )
        return jnp.sum(jnp.log10(eta_eff))

    T_init = jnp.linspace(3200.0, 1600.0, n_nodes)
    grad_fn = jax.grad(_loss)
    grad_val = grad_fn(T_init)

    assert grad_val.shape == (n_nodes,)
    assert jnp.all(jnp.isfinite(grad_val))


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_checkpoint_profile_parity():
    """Verify stagnant lid state on 175 kyr checkpoint profile matches between NumPy and JAX.

    Validates that both NumPy and JAX evaluate identical reference solid viscosity
    on a realistic profile, preventing the 10^19 mismatch caused by referencing
    melt const_log10visc in the NumPy lid path.
    """
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    ckpt_path = os.path.join(
        os.path.dirname(__file__), 'reference', 'checkpoint_profile_175kyr.npz'
    )
    if not os.path.exists(ckpt_path):
        pytest.skip('Reference checkpoint profile fixture not found')

    data = np.load(ckpt_path)
    r_basic = data['r_basic']
    T_basic = data['T_basic']
    P_basic = data['P_basic']
    F_conv = data['F_conv']
    F_tot = data['F_tot']
    phi_basic = data['phi_basic']
    v_unyielded = data['v_unyielded']

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_max=500.0e6,
    )

    evaluator = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=9.81,
        const_properties=True,
        const_log10visc=2.0,
        viscosity_solid=1.0e21,
        rheology=params,
    )

    # Reference viscosity must come from viscosity_solid, falling back to legacy const_log10visc
    visc_solid = getattr(evaluator, 'viscosity_solid', None)
    if visc_solid is None:
        visc_solid = 10.0 ** getattr(evaluator, 'const_log10visc', 21.0)

    state_np = compute_stagnant_lid_state(
        radii=r_basic,
        temperature=T_basic,
        pressure=P_basic,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=None,
        melt_fraction=phi_basic,
        params=params,
        unyielded_velocity=v_unyielded,
        viscosity_solid=visc_solid,
        xp=np,
    )

    state_jx = jax_rheo.compute_stagnant_lid_state(
        radii=jnp.array(r_basic),
        temperature=jnp.array(T_basic),
        pressure=jnp.array(P_basic),
        convective_flux=jnp.array(F_conv),
        total_flux=jnp.array(F_tot),
        solidus_temperature=None,
        melt_fraction=jnp.array(phi_basic),
        params=params,
        unyielded_velocity=jnp.array(v_unyielded),
        viscosity_solid=1.0e21,
    )

    for key in [
        'eta_i',
        'mu_i',
        'tau_d',
        'd_lid',
        'theta',
        'v_i',
        'v_m',
        'tau_buoy',
        'tau_d_over_tau_buoy',
        'Ra_eff',
        'delta_rh',
    ]:
        np.testing.assert_allclose(
            float(state_np[key]),
            float(state_jx[key]),
            rtol=1.0e-9,
            atol=1.0e-12,
            err_msg=f'Mismatch in diagnostic {key} on checkpoint profile',
        )

    # Assert stagnant lid formation on checkpoint profile
    assert np.max(state_np['w_lid']) > 0.5, (
        'NumPy state must form stagnant lid with w_lid > 0.5'
    )
    assert np.max(state_jx['w_lid']) > 0.5, 'JAX state must form stagnant lid with w_lid > 0.5'
    np.testing.assert_allclose(
        np.array(state_jx['w_lid']),
        state_np['w_lid'],
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in w_lid on checkpoint profile',
    )

    eta_diff_np = compute_arrhenius_viscosity(
        T_basic,
        P_basic,
        viscosity_solid=visc_solid,
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
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
    )
    assert np.any(eta_eff_np < eta_diff_np), (
        'Effective viscosity must fall below diffusion creep in yielding lid'
    )
    assert np.any(eta_eff_jx < eta_diff_jx), (
        'JAX effective viscosity must fall below diffusion creep in yielding lid'
    )
    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in eta_eff on checkpoint profile',
    )


def test_convecting_layer_viscosity_mixture_parity():
    """Verify mu_i evaluated from mixture viscosity matches between NumPy and JAX."""
    params = SolidRheologyParams()
    r = np.linspace(5.371e6, 6.371e6, 30)
    T = np.linspace(3500.0, 1400.0, 30)
    P = np.linspace(1.3e11, 1.0e5, 30)
    conv_flux = np.ones(30) * 1.0e5
    conv_flux[-4:] = 0.0
    tot_flux = np.ones(30) * 1.0e5
    phi = np.zeros(30)
    phi[:20] = 0.35

    # Viscosity profile representing mush in convective interior and cold solid in lid
    visc_mix_np = np.full(30, 1.0e11)
    visc_mix_np[-4:] = 1.0e23
    visc_mix_jx = jnp.array(visc_mix_np)

    # 1. Fallback when viscosity_mixture is None: mu_i == arrhenius(T_i, P_T_i)
    st_none_np = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=conv_flux,
        total_flux=tot_flux,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
        viscosity_mixture=None,
        xp=np,
    )
    st_none_jx = jax_rheo.compute_stagnant_lid_state(
        radii=jnp.array(r),
        temperature=jnp.array(T),
        pressure=jnp.array(P),
        convective_flux=jnp.array(conv_flux),
        total_flux=jnp.array(tot_flux),
        solidus_temperature=None,
        melt_fraction=jnp.array(phi),
        params=params,
        viscosity_solid=1.0e21,
        viscosity_mixture=None,
    )
    np.testing.assert_allclose(float(st_none_np['mu_i']), float(st_none_jx['mu_i']), rtol=1e-12)
    assert st_none_np['mu_i'] == st_none_np['eta_i']

    # 2. When viscosity_mixture is provided: mu_i is evaluated from the mixture viscosity
    st_mix_np = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=conv_flux,
        total_flux=tot_flux,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
        viscosity_mixture=visc_mix_np,
        xp=np,
    )
    st_mix_jx = jax_rheo.compute_stagnant_lid_state(
        radii=jnp.array(r),
        temperature=jnp.array(T),
        pressure=jnp.array(P),
        convective_flux=jnp.array(conv_flux),
        total_flux=jnp.array(tot_flux),
        solidus_temperature=None,
        melt_fraction=jnp.array(phi),
        params=params,
        viscosity_solid=1.0e21,
        viscosity_mixture=visc_mix_jx,
    )
    np.testing.assert_allclose(float(st_mix_np['mu_i']), float(st_mix_jx['mu_i']), rtol=1e-12)
    np.testing.assert_allclose(float(st_mix_np['tau_d']), float(st_mix_jx['tau_d']), rtol=1e-12)
    np.testing.assert_allclose(
        float(st_mix_np['Ra_eff']), float(st_mix_jx['Ra_eff']), rtol=1e-12
    )
    assert st_mix_np['mu_i'] < st_none_np['mu_i']
