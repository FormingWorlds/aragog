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
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )

    eta_eff_jx = jax_rheo.compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        tau_d=state_jx['tau_d'],
        tau_y_lid=state_jx['tau_y_lid'],
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
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )
    eta_eff_jx = jax_rheo.compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        tau_d=state_jx['tau_d'],
        tau_y_lid=state_jx['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
    )
    # Sub-yield branch (tau_d ~ 72.9 MPa < tau_y_lid = 500 MPa):
    # Under Moresi & Solomatov (1998) min closure, eta_eff == eta_diff exactly
    np.testing.assert_array_equal(
        eta_eff_np,
        eta_diff_np,
        err_msg='Effective viscosity must match diffusion creep exactly in sub-yield lid',
    )
    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in eta_eff on checkpoint profile (sub-yield)',
    )

    # Yielding branch (tau_y_lid = 50 MPa < tau_d ~ 72.9 MPa):
    # Effective viscosity must soften below diffusion creep in yielding lid
    eta_eff_np_yield = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        tau_d=state_np['tau_d'],
        tau_y_lid=50.0e6,
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )
    eta_eff_jx_yield = jax_rheo.compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        tau_d=state_jx['tau_d'],
        tau_y_lid=50.0e6,
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
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
    # Discriminating assertions: must match expected log-mean and discriminate from Arrhenius
    np.testing.assert_allclose(float(st_mix_np['mu_i']), 2.5234344760627e16, rtol=1e-8)
    assert not np.isclose(st_mix_np['mu_i'], st_none_np['mu_i'])
    assert st_none_np['mu_i'] / st_mix_np['mu_i'] > 1.0e5
    assert st_none_jx['mu_i'] / st_mix_jx['mu_i'] > 1.0e5

    # 3. Fallback when convective layer is inactive (zero convective flux)
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
    st_inactive_jx = jax_rheo.compute_stagnant_lid_state(
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
    # strain rates: sub-yield (low), exact switch, yielding (high), and zero strain rate
    sr = np.array([1.0e-18, 1.0e7 / (2.0 * 1.0e22), 1.0e-10, 0.0])

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_y=tau_y,
        strain_rate=sr,
        stress_closure_mode='local',
        xp=np,
    )
    eta_eff_jx = jax_rheo.compute_effective_viscosity(
        eta_diff=jnp.array(eta_diff),
        tau_y=jnp.array(tau_y),
        strain_rate=jnp.array(sr),
        stress_closure_mode='local',
    )

    # 1. Parity between NumPy and JAX
    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-12,
        err_msg='Mismatch in local mode eta_eff between NumPy and JAX',
    )

    # 2. Below yield (sr[0]): eta_eff == eta_d exactly
    assert eta_eff_np[0] == eta_diff[0]

    # 3. Exact switch (sr[1]): eta_eff == eta_d exactly
    assert eta_eff_np[1] == pytest.approx(eta_diff[1], rel=1.0e-12)

    # 4. Above yield (sr[2]): eta_eff == eta_y exactly
    expected_eta_y = tau_y[2] / (2.0 * sr[2])
    assert eta_eff_np[2] == pytest.approx(expected_eta_y, rel=1.0e-12)

    # 5. Zero strain rate (sr[3] == 0.0): eta_eff == eta_d exactly
    assert eta_eff_np[3] == eta_diff[3]


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_jax_phase_convecting_layer_viscosity_mixture_wiring_parity():
    """Verify JAX compute_mlt wires mixture viscosity to mu_i with NumPy parity."""
    from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt

    n = 30
    r = np.linspace(3.48e6, 6.371e6, n)
    ml = np.maximum(np.minimum(r - r[0], r[-1] - r), 1.0)
    mesh = MeshArrays(
        d_dr_matrix=jnp.zeros((n, n - 1)),
        quantity_matrix=jnp.zeros((n, n - 1)),
        area=jnp.ones(n),
        volume=jnp.ones(n),
        radii_basic=jnp.asarray(r),
        radii_stag=jnp.asarray(0.5 * (r[1:] + r[:-1])),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.linspace(1.3e11, 1e5, n - 1),
        P_basic=jnp.linspace(1.3e11, 1e5, n),
        dP_dr_basic=jnp.full(n, -9.81 * 4000.0),
        gravity=jnp.full(n, 9.81),
    )

    T = np.linspace(3000.0, 1400.0, n)
    visc = np.full(n, 1.0e23)
    visc[:25] = 1.0e12
    phi = np.zeros(n)
    phi[:25] = 0.35

    phase = PhaseProperties(
        temperature=jnp.asarray(T),
        density=jnp.full(n, 4000.0),
        heat_capacity=jnp.full(n, 1200.0),
        thermal_expansivity=jnp.full(n, 3e-5),
        dTdPs=jnp.full(n, 1e-8),
        melt_fraction=jnp.asarray(phi),
        viscosity=jnp.asarray(visc),
        kinematic_viscosity=jnp.asarray(visc / 4000.0),
        thermal_conductivity=jnp.full(n, 4.0),
        latent_heat=jnp.full(n, 4e5),
        capacitance=jnp.full(n, 4000.0 * 1200.0),
        eta_diff=jnp.full(n, 1.0e22),
        tau_y=jnp.full(n, 1.0e4),
        visc_solid_weight=jnp.ones(n),
    )

    params = PhaseParams(
        enabled=True,
        conduction=1.0,
        convection=1.0,
        kappah_floor=0.0,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=1800.0,
        yield_stress_c=1.0e4,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e4,
        activation_energy=300e3,
        activation_volume=1.5e-6,
        viscosity_solid=1.0e21,
    )

    # Convective interior with stable lid (nodes 27..29)
    grad = jnp.full(n, -1.0e-5).at[-3:].set(1.0e-5)
    kh, _, f_un = compute_mlt(grad, phase, mesh, params, return_unyielded=True)

    # 1. Parity between NumPy and JAX on the mush profile
    st_np = compute_stagnant_lid_state(
        radii=np.array(mesh.radii_basic),
        temperature=np.array(phase.temperature),
        pressure=np.array(mesh.P_basic),
        convective_flux=np.array(f_un),
        total_flux=np.array(jnp.maximum(f_un, 1e3)),
        solidus_temperature=None,
        melt_fraction=np.array(phase.melt_fraction),
        params=params,
        viscosity_solid=params.viscosity_solid,
        viscosity_mixture=np.array(phase.viscosity),
        xp=np,
    )
    st_jx = jax_rheo.compute_stagnant_lid_state(
        radii=mesh.radii_basic,
        temperature=phase.temperature,
        pressure=mesh.P_basic,
        convective_flux=f_un,
        total_flux=jnp.maximum(f_un, 1e3),
        solidus_temperature=None,
        melt_fraction=phase.melt_fraction,
        params=params,
        viscosity_solid=params.viscosity_solid,
        viscosity_mixture=phase.viscosity,
    )
    np.testing.assert_allclose(float(st_np['mu_i']), float(st_jx['mu_i']), rtol=1e-12)

    # 2. Discriminate against unmixed Arrhenius fallback (mu_i must reflect mush viscosity)
    st_unmixed = jax_rheo.compute_stagnant_lid_state(
        radii=mesh.radii_basic,
        temperature=phase.temperature,
        pressure=mesh.P_basic,
        convective_flux=f_un,
        total_flux=jnp.maximum(f_un, 1e3),
        solidus_temperature=None,
        melt_fraction=phase.melt_fraction,
        params=params,
        viscosity_solid=params.viscosity_solid,
        viscosity_mixture=None,
    )
    assert float(st_unmixed['mu_i']) / float(st_jx['mu_i']) > 1.0e4

    # 3. Verify compute_mlt output in the yielding lid transition nodes (nodes 25..26)
    # If viscosity_mixture is omitted in jax/phase.py, kh in these cells differs by >30x.
    assert np.all(kh[25:27] > 0.0)
    assert kh[25] < 1.0e-8
