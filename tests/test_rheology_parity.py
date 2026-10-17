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


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_arrhenius_viscosity_parity_float64():
    """Verify homologous Arrhenius viscosity matches analytical values and JAX parity to 1e-12."""
    t_vals = np.linspace(1400.0, 3000.0, 50)
    p_vals = np.linspace(0.0, 135.0e9, 50)
    t_sol_vals = 1400.0 + 2.4e-8 * p_vals - 5.0e-20 * p_vals**2

    eta_np = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )

    eta_jx = compute_arrhenius_viscosity(
        jnp.array(t_vals),
        jnp.array(p_vals),
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=jnp.array(t_sol_vals),
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=jnp,
    )

    np.testing.assert_allclose(np.array(eta_jx), eta_np, rtol=1.0e-12, atol=1.0e-30)

    # Verify t_sol_0=None infers low-pressure solidus endpoint for ascending and descending P:
    eta_auto_asc = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=None,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    np.testing.assert_allclose(eta_auto_asc, eta_np, rtol=1.0e-12, atol=1.0e-30)

    eta_auto_desc_jx = compute_arrhenius_viscosity(
        jnp.array(t_vals[::-1]),
        jnp.array(p_vals[::-1]),
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=jnp.array(t_sol_vals[::-1]),
        t_sol_0=None,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=jnp,
    )
    np.testing.assert_allclose(
        np.array(eta_auto_desc_jx), eta_np[::-1], rtol=1.0e-12, atol=1.0e-30
    )

    # Exact hand-computed analytical verification:
    expected = np.minimum(
        1.0e21 * np.exp(25.0 * (t_sol_vals / t_vals - 1400.0 / 1600.0)),
        10.0**40.0,
    )
    np.testing.assert_allclose(eta_np, expected, rtol=1.0e-12, atol=1.0e-30)

    # Canary 1: sign flip on g_homologous must fail rtol=1.0e-12, atol=1.0e-30.
    eta_sign_flip = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=-25.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eta_sign_flip, eta_np, rtol=1.0e-12, atol=1.0e-30)

    # Canary 2: dropping T_sol(0) / T_ref (equivalent to t_sol_0 = 0.0) must fail.
    eta_no_ref = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=0.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eta_no_ref, eta_np, rtol=1.0e-12, atol=1.0e-30)

    # Canary 3: ignoring solidus_temperature and falling back to linear solidus must fail.
    eta_linear_fallback = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=None,
        t_sol_0=1400.0,
        dt_sol_dp=2.4e-8,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eta_linear_fallback, eta_np, rtol=1.0e-12, atol=1.0e-30)


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

    tau_jx = compute_yield_stress(
        jnp.array(p_vals),
        yield_stress_c=50.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
        xp=jnp,
    )

    np.testing.assert_allclose(np.array(tau_jx), tau_np, rtol=1.0e-12, atol=1.0e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'regime',
    ['unyielded', 'yielding', 'mobile', 'no_lid', 'mushy', 'inactive_convection'],
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
    elif regime == 'inactive_convection':
        temperature = np.linspace(3000.0, 1800.0, n_nodes)
        conv_flux = np.zeros(n_nodes)
        v_unyielded = np.zeros(n_nodes)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 500.0e6
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

    state_jx = compute_stagnant_lid_state(
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
        xp=jnp,
    )

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
    elif regime in ('no_lid', 'mushy'):
        assert state_np['d_lid'] == 0.0, f'{regime} must suppress stagnant lid formation'
        assert state_np['lid_regime'] == 0.0, f'{regime} must have lid_regime == 0'
        assert state_np['tau_d'] == 0.0, f'{regime} must have tau_d == 0'
        assert state_np['v_m'] == 0.0, f'{regime} must have v_m == 0'
    elif regime == 'inactive_convection':
        assert state_np['d_lid'] == 0.0
        assert state_np['tau_d'] == 0.0

    np.testing.assert_allclose(
        np.array(state_jx['w_lid']),
        state_np['w_lid'],
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg=f'Mismatch in w_lid for regime {regime}',
    )

    eta_diff_np = compute_arrhenius_viscosity(
        temperature,
        pressure,
        viscosity_solid=1.0e21,
        g_homologous=params.g_homologous,
        t_sol_0=params.t_sol_0,
        dt_sol_dp=params.dt_sol_dp,
        arrhenius_t_ref=params.arrhenius_t_ref,
        xp=np,
    )
    eta_diff_jx = jnp.array(eta_diff_np)

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        eta_i=state_np['eta_i'],
        tau_d=state_np['tau_d'],
        tau_y_lid=state_np['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )

    eta_eff_jx = compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        eta_i=state_jx['eta_i'],
        tau_d=state_jx['tau_d'],
        tau_y_lid=state_jx['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
        xp=jnp,
    )

    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg=f'Mismatch in eta_eff for regime {regime}',
    )

    if regime == 'unyielded':
        t_sol_1d = 1400.0 + 2.4e-8 * pressure - 5.0e-20 * pressure**2
        st_sol_np = compute_stagnant_lid_state(
            radii=radii,
            temperature=temperature,
            pressure=pressure,
            convective_flux=conv_flux,
            total_flux=tot_flux,
            solidus_temperature=t_sol_1d,
            melt_fraction=melt_frac,
            params=params,
            unyielded_velocity=v_unyielded,
            viscosity_solid=1.0e21,
            xp=np,
        )
        st_sol_jx = compute_stagnant_lid_state(
            radii=jnp.array(radii),
            temperature=jnp.array(temperature),
            pressure=jnp.array(pressure),
            convective_flux=jnp.array(conv_flux),
            total_flux=jnp.array(tot_flux),
            solidus_temperature=jnp.array(t_sol_1d[:, None]),
            melt_fraction=jnp.array(melt_frac),
            params=params,
            unyielded_velocity=jnp.array(v_unyielded),
            viscosity_solid=1.0e21,
            xp=jnp,
        )
        for key in ['theta', 'dT_rh', 'd_lid', 'tau_d', 'eta_i']:
            np.testing.assert_allclose(
                float(st_sol_jx[key]),
                float(st_sol_np[key]),
                rtol=1.0e-9,
                atol=1.0e-12,
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
        lid_st = compute_stagnant_lid_state(
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
            xp=jnp,
        )
        eta_d = compute_arrhenius_viscosity(
            T_profile,
            pressure,
            viscosity_solid=1.0e21,
            g_homologous=params.g_homologous,
            t_sol_0=params.t_sol_0,
            dt_sol_dp=params.dt_sol_dp,
            arrhenius_t_ref=params.arrhenius_t_ref,
            xp=jnp,
        )
        eta_eff = compute_effective_viscosity(
            eta_diff=eta_d,
            eta_i=lid_st['eta_i'],
            tau_d=lid_st['tau_d'],
            tau_y_lid=lid_st['tau_y_lid'],
            stress_closure_mode='lid',
            w_lid=lid_st['w_lid'],
            xp=jnp,
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
    """Verify stagnant lid state on 175 kyr checkpoint profile matches between NumPy and JAX."""
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
    visc_solid = 1.0e21

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

    state_jx = compute_stagnant_lid_state(
        radii=jnp.array(r_basic),
        temperature=jnp.array(T_basic),
        pressure=jnp.array(P_basic),
        convective_flux=jnp.array(F_conv),
        total_flux=jnp.array(F_tot),
        solidus_temperature=None,
        melt_fraction=jnp.array(phi_basic),
        params=params,
        unyielded_velocity=jnp.array(v_unyielded),
        viscosity_solid=visc_solid,
        xp=jnp,
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
        g_homologous=params.g_homologous,
        t_sol_0=params.t_sol_0,
        dt_sol_dp=params.dt_sol_dp,
        arrhenius_t_ref=params.arrhenius_t_ref,
        xp=np,
    )
    eta_diff_jx = jnp.array(eta_diff_np)

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        eta_i=state_np['eta_i'],
        tau_d=state_np['tau_d'],
        tau_y_lid=state_np['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )
    eta_eff_jx = compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        eta_i=state_jx['eta_i'],
        tau_d=state_jx['tau_d'],
        tau_y_lid=state_jx['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
        xp=jnp,
    )
    np.testing.assert_array_equal(
        eta_eff_np,
        eta_diff_np,
        err_msg='Effective viscosity must match diffusion creep in sub-yield lid',
    )
    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in eta_eff on checkpoint profile (sub-yield)',
    )

    eta_eff_np_yield = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        eta_i=state_np['eta_i'],
        tau_d=state_np['tau_d'],
        tau_y_lid=50.0e6,
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )
    eta_eff_jx_yield = compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        eta_i=state_jx['eta_i'],
        tau_d=state_jx['tau_d'],
        tau_y_lid=50.0e6,
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
        xp=jnp,
    )
    assert np.any(eta_eff_np_yield < eta_diff_np), (
        'Effective viscosity must fall below diffusion creep in yielding lid'
    )
    np.testing.assert_allclose(
        np.array(eta_eff_jx_yield),
        eta_eff_np_yield,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in eta_eff on checkpoint profile (yielding)',
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
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

    visc_mix_np = np.full(30, 1.0e11)
    visc_mix_np[-4:] = 1.0e23
    visc_mix_jx = jnp.array(visc_mix_np)

    # 1. Fallback when viscosity_mixture is None: mu_i == arrhenius(T_i, P_T_i).
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
    st_none_jx = compute_stagnant_lid_state(
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
        xp=jnp,
    )
    np.testing.assert_allclose(float(st_none_np['mu_i']), float(st_none_jx['mu_i']), rtol=1e-12)
    assert st_none_np['mu_i'] == st_none_np['eta_i']

    # 2. When viscosity_mixture is provided: mu_i is evaluated from the mixture viscosity.
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
    st_mix_jx = compute_stagnant_lid_state(
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
        xp=jnp,
    )
    np.testing.assert_allclose(float(st_mix_np['mu_i']), float(st_mix_jx['mu_i']), rtol=1e-12)
    np.testing.assert_allclose(float(st_mix_np['tau_d']), float(st_mix_jx['tau_d']), rtol=1e-12)
    np.testing.assert_allclose(
        float(st_mix_np['Ra_eff']), float(st_mix_jx['Ra_eff']), rtol=1e-12
    )
    np.testing.assert_allclose(float(st_mix_np['mu_i']), 2.5234344760627e16, rtol=1e-8)
    assert not np.isclose(st_mix_np['mu_i'], st_none_np['mu_i'])
    assert st_none_np['mu_i'] / st_mix_np['mu_i'] > 1.0e5
    assert st_none_jx['mu_i'] / st_mix_jx['mu_i'] > 1.0e5

    # 3. Fallback when convective layer is inactive (zero convective flux).
    st_inactive_np = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=np.zeros(30),
        total_flux=tot_flux,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
        viscosity_mixture=visc_mix_np,
        xp=np,
    )
    st_inactive_jx = compute_stagnant_lid_state(
        radii=jnp.array(r),
        temperature=jnp.array(T),
        pressure=jnp.array(P),
        convective_flux=jnp.zeros(30),
        total_flux=jnp.array(tot_flux),
        solidus_temperature=None,
        melt_fraction=jnp.array(phi),
        params=params,
        viscosity_solid=1.0e21,
        viscosity_mixture=visc_mix_jx,
        xp=jnp,
    )
    np.testing.assert_allclose(
        float(st_inactive_np['mu_i']), float(st_inactive_jx['mu_i']), rtol=1e-12
    )
    np.testing.assert_allclose(
        float(st_inactive_np['mu_i']), float(st_inactive_np['eta_i']), rtol=1e-12
    )


@pytest.mark.unit
def test_local_stress_closure_min_form_parity():
    """Verify local mode minimum closure parity between NumPy and JAX."""
    eta_diff = np.array([1.0e21, 1.0e22, 1.0e23, 1.0e24])
    tau_y = np.array([1.0e7, 1.0e7, 1.0e7, 1.0e7])
    sr = np.array([1.0e-18, 1.0e7 / (2.0 * 1.0e22), 1.0e-10, 0.0])

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_y=tau_y,
        strain_rate=sr,
        stress_closure_mode='local',
        xp=np,
    )
    eta_eff_jx = compute_effective_viscosity(
        eta_diff=jnp.array(eta_diff),
        tau_y=jnp.array(tau_y),
        strain_rate=jnp.array(sr),
        stress_closure_mode='local',
        xp=jnp,
    )

    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-12,
        err_msg='Mismatch in local mode eta_eff between NumPy and JAX',
    )
    assert eta_eff_np[0] == eta_diff[0]
    assert eta_eff_np[1] == pytest.approx(eta_diff[1], rel=1.0e-12)
    expected_eta_y = tau_y[2] / (2.0 * sr[2])
    assert eta_eff_np[2] == pytest.approx(expected_eta_y, rel=1.0e-12)
    assert eta_eff_np[3] == eta_diff[3]
