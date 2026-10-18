"""Unit tests for the shared array-agnostic rheology kernel.

Validates asymptotic physical limits, monotonicity, continuity,
and autodiff properties of the kernel in aragog.rheology.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import (
    R_GAS,
    SolidRheologyParams,
    compute_arrhenius_enthalpy,
    compute_arrhenius_viscosity,
    compute_effective_viscosity,
    compute_mlt_top_slope,
    compute_stagnant_lid_state,
    compute_strain_rate_local,
)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_zero_driving_stress():
    """Verify effective viscosity equals diffusion creep when tau_d is zero."""
    eta_diff = 1.0e22
    tau_y_lid = 1.0e7

    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_diff,
        tau_d=0.0,
        tau_y_lid=tau_y_lid,
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

    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_diff,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        stress_closure_mode='lid',
    )
    assert eta_eff == pytest.approx(eta_diff, rel=1.0e-10)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_zero_solidus_slope():
    """Verify activation enthalpy is constant in pressure when dt_sol_dp is zero."""
    g_hom = 25.0
    t_sol_0 = 1400.0
    p_values = np.linspace(0.0, 1.0e11, 20)
    enthalpy = compute_arrhenius_enthalpy(
        p_values,
        g_homologous=g_hom,
        t_sol_0=t_sol_0,
        dt_sol_dp=0.0,
        r_gas=R_GAS,
        xp=np,
    )
    expected = g_hom * R_GAS * t_sol_0
    np.testing.assert_allclose(enthalpy, expected, rtol=1.0e-14)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_homologous_enthalpy_and_viscosity_invariants():
    """Verify homologous enthalpy H(P) = g_hom R T_sol(P) and constant-ratio viscosity invariant."""
    # 1. Linear constant-property fallback H(P) = g_hom * R_GAS * (t_sol_0 + dt_sol_dp * P).
    g_hom = 25.0
    t_sol_0 = 1400.0
    dt_sol_dp = 2.4e-8
    p_values = np.linspace(0.0, 1.0e11, 25)
    enthalpy = compute_arrhenius_enthalpy(
        p_values,
        g_homologous=g_hom,
        t_sol_0=t_sol_0,
        dt_sol_dp=dt_sol_dp,
        r_gas=R_GAS,
        xp=np,
    )
    expected = g_hom * R_GAS * (t_sol_0 + dt_sol_dp * p_values)
    np.testing.assert_allclose(enthalpy, expected, rtol=1.0e-14)

    # 2. Explicit solidus curve at P = {0, 25, 50, 100, 135} GPa and g_hom = {14, 25}.
    p_grid = np.array([0.0, 25.0e9, 50.0e9, 100.0e9, 135.0e9])
    t_sol_grid = np.array([1380.6038, 2475.9161, 2875.9474, 3673.7136, 4232.1499])
    for g_val in (14.0, 25.0):
        h_grid = compute_arrhenius_enthalpy(
            p_grid,
            g_homologous=g_val,
            solidus_temperature=t_sol_grid,
            r_gas=R_GAS,
            xp=np,
        )
        np.testing.assert_allclose(
            h_grid, g_val * R_GAS * t_sol_grid, rtol=1.0e-12, atol=1.0e-30
        )

        eta_by_frac = dict()
        # Evaluate eta_diff at T in {0.5, 0.7, 0.9, 1.0} * T_sol(P) with high cap so unclipped.
        for frac in (0.5, 0.7, 0.9, 1.0):
            t_eval = frac * t_sol_grid
            eta_eval = compute_arrhenius_viscosity(
                t_eval,
                p_grid,
                viscosity_solid=1.0e21,
                g_homologous=g_val,
                solidus_temperature=t_sol_grid,
                t_sol_0=t_sol_grid[0],
                arrhenius_t_ref=1600.0,
                r_gas=R_GAS,
                viscosity_max_log10=80.0,
                xp=np,
            )
            expected_eta = 1.0e21 * np.exp(g_val * (1.0 / frac - t_sol_grid[0] / 1600.0))
            np.testing.assert_allclose(eta_eval, expected_eta, rtol=1.0e-12, atol=1.0e-30)
            eta_by_frac[frac] = eta_eval

        # Homologous invariant: eta(0.9 * T_sol(P), P) / eta(T_sol(P), P) == exp(g_hom / 9).
        np.testing.assert_allclose(
            eta_by_frac[0.9] / eta_by_frac[1.0], np.exp(g_val / 9.0), rtol=1.0e-12, atol=1.0e-30
        )

    # 3. Verify compute_mlt_top_slope across theta in {0, 5, 10, 20, 40}.
    theta_grid = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
    s_top_vals = compute_mlt_top_slope(theta_grid, s_top_inf=0.22, c_theta=0.15, xp=np)
    expected_s_top = 0.22 + 0.78 * np.exp(-0.15 * theta_grid)
    np.testing.assert_allclose(s_top_vals, expected_s_top, rtol=1.0e-12, atol=1.0e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_extreme_yielding():
    """Verify yielded branch asymptotically governs at large driving stress."""
    eta_diff = 1.0e25
    tau_y_lid = 50.0e6
    tau_d = 100.0 * tau_y_lid
    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_diff,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        stress_closure_mode='lid',
    )
    expected_yielded = min((tau_y_lid * eta_diff) / tau_d, eta_diff)
    assert eta_eff == pytest.approx(expected_yielded, rel=1.0e-3)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_exact_yield_match():
    """Verify minimum closure yields eta_d when eta_y = eta_d.

    References
    ----------
    Moresi & Solomatov (1998), p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x
    Tackley (2000), eq. 8, p. 4, doi:10.1029/2000GC000036
    """
    eta_diff = 1.0e22
    tau_y = 1.0e8

    # Local mode: strain_rate chosen so eta_y = tau_y / (2 * strain_rate) == eta_diff.
    strain_rate_match = tau_y / (2.0 * eta_diff)
    eta_eff_local = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_y=tau_y,
        strain_rate=strain_rate_match,
        stress_closure_mode='local',
    )
    assert eta_eff_local == pytest.approx(eta_diff, rel=1.0e-12)
    eta_eff_local_2x = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_y=tau_y,
        strain_rate=2.0 * strain_rate_match,
        stress_closure_mode='local',
    )
    assert eta_eff_local_2x == pytest.approx(0.5 * eta_diff, rel=1.0e-12)

    # Stagnant lid mode: under min closure, tau_d == tau_y yields eta_eff == eta_diff.
    eta_eff_lid = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_diff,
        tau_d=tau_y,
        tau_y_lid=tau_y,
        stress_closure_mode='lid',
    )
    assert eta_eff_lid == pytest.approx(eta_diff, rel=1.0e-12)
    eta_eff_lid_2x = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_diff,
        tau_d=2.0 * tau_y,
        tau_y_lid=tau_y,
        stress_closure_mode='lid',
    )
    assert eta_eff_lid_2x == pytest.approx(0.5 * eta_diff, rel=1.0e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_min_closure_local_asymptotic_limits():
    """Verify local min closure limits: eta_y >= eta_d gives eta_d and eta_y < eta_d gives eta_y."""
    eta_d = 1.0e22
    tau_y = 1.0e8

    # Limit 1: eta_y >> eta_d (low strain rate, sub-yield).
    sr_low = 1.0e-18
    eta_y_high = tau_y / (2.0 * sr_low)
    assert eta_y_high > 100.0 * eta_d
    eta_eff_high = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_low,
        stress_closure_mode='local',
    )
    assert eta_eff_high == pytest.approx(eta_d, rel=1.0e-12)

    # Limit 2: eta_y << eta_d (high strain rate, yielding).
    sr_high = 5.0e-11
    eta_y_low = tau_y / (2.0 * sr_high)
    eta_eff_low = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_high,
        stress_closure_mode='local',
    )
    assert eta_eff_low == pytest.approx(eta_y_low, rel=1.0e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_monotonicity_and_continuity_across_yield():
    """Verify effective viscosity is monotonic non-increasing and continuous."""
    n_pts = 1000
    tau_y = 50.0e6
    eta_diff = 1.0e25

    strain_rates = np.logspace(-18.0, -10.0, n_pts)
    eta_eff_vals = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_y=tau_y,
        strain_rate=strain_rates,
        stress_closure_mode='local',
        xp=np,
    )

    assert eta_eff_vals[0] == pytest.approx(eta_diff, rel=1.0e-12)
    assert eta_eff_vals[-1] == pytest.approx(tau_y / (2.0 * strain_rates[-1]), rel=1.0e-12)
    assert eta_eff_vals[-1] < 1.0e-5 * eta_eff_vals[0]

    diffs = np.diff(eta_eff_vals)
    assert np.all(diffs <= 0.0), 'Effective viscosity must not increase with strain rate'

    log_diffs = np.abs(np.diff(np.log10(eta_eff_vals)))
    assert np.all(log_diffs < 0.1), 'Effective viscosity in log-space must be continuous'


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
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.zeros(n_nodes),
        xp=np,
    )

    assert state['w_active'] < 0.02
    # if_lid sets inactive-convection diagnostics to literal 0.0.
    assert float(state['tau_d']) == 0.0
    assert float(state['tau_d_lid']) == 0.0
    assert float(state['tau_buoy']) == 0.0
    assert float(state['v_m']) == 0.0
    assert float(state['Ra_eff']) == 0.0
    assert float(state['d_lid']) == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_inactive_convection_yield_viscosity_unyielded():
    """Verify effective viscosity remains unyielded when convection is inactive."""
    n_nodes = 30
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    temperature = np.linspace(3500.0, 1600.0, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    conv_flux = np.zeros(n_nodes)
    total_flux = np.full(n_nodes, 0.05)
    params = SolidRheologyParams(enabled=True, stress_closure_mode='lid', yield_stress_c=1.0e4)

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=np.zeros(n_nodes),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    # Exact by construction: if_lid returns literal 0.0 when has_lid is False.
    assert float(state['tau_d']) == 0.0
    eta_diff = np.full(n_nodes, 1.0e24)
    eta_eff = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=state['eta_i'],
        tau_d=state['tau_d'],
        tau_y_lid=state['tau_y_lid'],
        stress_closure_mode='lid',
        xp=np,
    )
    np.testing.assert_allclose(eta_eff, eta_diff)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_whole_mantle_mush_clips_lid():
    """Verify lid thickness is positive when solid and zero when mantle is entirely mushy."""
    n_nodes = 30
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    temperature = np.linspace(3500.0, 1600.0, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    conv_flux = np.full(n_nodes, 100.0)
    conv_flux[-6:] = 0.0
    total_flux = np.full(n_nodes, 100.0)
    params = SolidRheologyParams(enabled=True, stress_closure_mode='lid', phi_visc_single=0.5)

    state_solid = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=np.zeros(n_nodes),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        xp=np,
    )
    assert float(state_solid['d_lid']) > 1.0e5

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=np.full(n_nodes, 0.8),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        xp=np,
    )

    assert state['d_lid'] == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.unit
def test_jax_autodiff_linear_solidus_enthalpy_gradient():
    """Verify JAX dH/dP matches NumPy analytical derivative g_hom * R * dT_sol/dP."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    dt_sol_dp = 5.0e-6 / (25.0 * R_GAS)

    def _eval(p):
        return compute_arrhenius_enthalpy(
            p,
            g_homologous=25.0,
            t_sol_0=1400.0,
            dt_sol_dp=dt_sol_dp,
            r_gas=R_GAS,
            xp=jnp,
        )

    grad_fn = jax.grad(_eval)
    grad_val = grad_fn(10.0e9)
    assert jnp.isfinite(grad_val)
    assert float(grad_val) == pytest.approx(5.0e-6, rel=1.0e-10)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_strain_rate_local_eps_floor_at_zero_mixing_length():
    """Verify compute_strain_rate_local applies eps floor when mixing length is zero."""
    v = 1.0e-9
    # At boundary nodes, mixing_length is zero; eps prevents division by zero.
    sr = compute_strain_rate_local(v, 0.0, eps=1.0e-15)
    expected = 1.0e-9 / (2.0 * 1.0e-15)
    assert sr == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_molten_column_with_solidus_table_has_no_lid_across_mesh_resolutions():
    """Verify a molten column with a solidus table reports no lid at N = 50, 100, and 200."""
    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0)
    for n_nodes in (50, 100, 200):
        radii = np.linspace(3.48e6, 6.37e6, n_nodes)
        pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
        t_sol = 1400.0 + 2.4e-8 * pressure
        conv_flux = np.full(n_nodes, 99.0)
        tot_flux = np.full(n_nodes, 100.0)
        for dt_above in (100.0, 150.0, 200.0, 250.0):
            t_prof = t_sol + dt_above
            for phi_val in (0.41, 0.50):
                state = compute_stagnant_lid_state(
                    radii=radii,
                    temperature=t_prof,
                    pressure=pressure,
                    convective_flux=conv_flux,
                    total_flux=tot_flux,
                    solidus_temperature=t_sol,
                    t_sol_0=1400.0,
                    melt_fraction=np.full(n_nodes, phi_val),
                    params=params,
                    viscosity_solid=1.0e21,
                    phi_rheo=0.4,
                    phi_width=0.15,
                    xp=np,
                )
                assert bool(state['has_lid']) is False
                assert float(state['d_lid']) == 0.0
                assert float(radii[-1] - state['r_lid_base']) < 1.0
                assert float(state['tau_d']) == 0.0
                assert float(state['lid_regime']) == 0.0
                np.testing.assert_allclose(state['w_lid'], 0.0, atol=1.0e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_melt_fraction_ramp_phi_rheo_and_phi_width_below_solidus_threshold():
    """Verify w_hot_phi ramp governs w_lid when phi_rheo is below SOLIDUS_MELT_FRACTION_THRESHOLD."""
    n_nodes = 40
    radii = np.linspace(3.48e6, 6.37e6, n_nodes)
    pressure = np.linspace(135.0e9, 0.0, n_nodes)
    t_sol = 1400.0 + 2.4e-8 * pressure
    temperature = np.full(n_nodes, 1600.0)
    temperature[-8:] = np.linspace(1200.0, 300.0, 8)
    conv_flux = np.full(n_nodes, 100.0)
    conv_flux[-8:] = 0.0
    tot_flux = np.full(n_nodes, 100.0)
    phi = np.zeros(n_nodes)
    phi[-6:-1] = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0)

    phi_rheo = 0.002
    phi_width = 0.001
    zero_phi = 0.5 * (1.0 + np.tanh(-phi_rheo / phi_width))
    raw_phi = 0.5 * (1.0 + np.tanh((phi - phi_rheo) / phi_width))
    expected_w_hot_phi = np.clip((raw_phi - zero_phi) / (1.0 - zero_phi), 0.0, 1.0)

    for sol_arg in (None, t_sol):
        state = compute_stagnant_lid_state(
            radii=radii,
            temperature=temperature,
            pressure=pressure,
            convective_flux=conv_flux,
            total_flux=tot_flux,
            solidus_temperature=sol_arg,
            t_sol_0=1400.0,
            melt_fraction=phi,
            params=params,
            viscosity_solid=1.0e21,
            phi_rheo=phi_rheo,
            phi_width=phi_width,
            xp=np,
        )
        # At nodes -5..-2, T <= 943 K is far below T_lid_iso and T_sol, and phi <= 0.005
        # is well below SOLIDUS_MELT_FRACTION_THRESHOLD (0.01), so w_hot_phi dominates.
        expected_w_lid_slice = (1.0 - expected_w_hot_phi[-5:-1]) * float(state['w_active'])
        np.testing.assert_allclose(
            state['w_lid'][-5:-1], expected_w_lid_slice, rtol=1.0e-2, atol=7.0e-3
        )


@pytest.mark.unit
def test_arrhenius_viscosity_and_enthalpy_reject_invalid_viscosity_solid_and_g_homologous():
    """Verify viscosity_solid <= 0 and g_homologous < 0 raise ValueError on np and return NaN on JAX."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    for bad_g in (-25.0, -1.0e-6, np.inf, -np.inf):
        with pytest.raises(ValueError, match='g_homologous'):
            compute_arrhenius_enthalpy(1.0e9, g_homologous=bad_g, xp=np)
        with pytest.raises(ValueError, match='g_homologous'):
            compute_arrhenius_viscosity(1600.0, 1.0e9, g_homologous=bad_g, xp=np)
        h_jax = compute_arrhenius_enthalpy(jnp.asarray(1.0e9), g_homologous=bad_g, xp=jnp)
        assert bool(jnp.isnan(h_jax))
        eta_jax = compute_arrhenius_viscosity(
            jnp.asarray([1600.0, 2000.0]),
            jnp.asarray([0.0, 1.0e9]),
            g_homologous=bad_g,
            xp=jnp,
        )
        assert bool(jnp.all(jnp.isnan(eta_jax)))

    for bad_vs in (0.0, -1.0e21, np.inf, -np.inf):
        with pytest.raises(ValueError, match='viscosity_solid'):
            compute_arrhenius_viscosity(1600.0, 0.0, viscosity_solid=bad_vs, xp=np)
        eta_jax_vs = compute_arrhenius_viscosity(
            jnp.asarray([1600.0, 2000.0]),
            jnp.asarray([0.0, 1.0e9]),
            viscosity_solid=bad_vs,
            xp=jnp,
        )
        assert bool(jnp.all(jnp.isnan(eta_jax_vs)))

    assert compute_arrhenius_enthalpy(1.0e9, g_homologous=0.0, xp=np) == 0.0
    assert compute_arrhenius_viscosity(
        1600.0, 0.0, viscosity_solid=1.0e21, g_homologous=0.0
    ) == (pytest.approx(1.0e21, rel=1.0e-12))
    assert np.isnan(compute_arrhenius_enthalpy(1.0e9, g_homologous=np.nan, xp=np))
    assert np.isnan(compute_arrhenius_viscosity(1600.0, 0.0, viscosity_solid=np.nan, xp=np))
    assert np.isnan(compute_arrhenius_viscosity(1600.0, 0.0, g_homologous=np.nan, xp=np))


@pytest.mark.unit
def test_stagnant_lid_state_rejects_inf_in_array_and_profile_inputs():
    """Verify inf in any array or profile input raises ValueError on np and gives has_lid=False and NaN on JAX."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    n_nodes = 12
    base_kw = {
        'radii': np.linspace(3.48e6, 6.37e6, n_nodes),
        'temperature': np.linspace(2600.0, 300.0, n_nodes),
        'pressure': np.linspace(135.0e9, 0.0, n_nodes),
        'convective_flux': np.full(n_nodes, 80.0),
        'total_flux': np.full(n_nodes, 100.0),
        'solidus_temperature': np.linspace(4200.0, 1400.0, n_nodes),
        't_sol_0': 1400.0,
        'melt_fraction': np.zeros(n_nodes),
        'unyielded_velocity': np.full(n_nodes, 1.0e-9),
        'viscosity_mixture': np.full(n_nodes, 1.0e20),
        'density': np.full(n_nodes, 3300.0),
        'gravity': np.full(n_nodes, 9.81),
        'thermal_expansivity': np.full(n_nodes, 3.0e-5),
        'thermal_diffusivity': np.full(n_nodes, 1.0e-6),
        'params': SolidRheologyParams(enabled=True, t_sol_0=1400.0),
        'viscosity_solid': 1.0e21,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
    }

    for field_name in (
        'temperature',
        'pressure',
        'convective_flux',
        'total_flux',
        'melt_fraction',
        'solidus_temperature',
        'unyielded_velocity',
        'viscosity_mixture',
        'density',
        'gravity',
        'thermal_expansivity',
        'thermal_diffusivity',
    ):
        for sign in (1.0, -1.0):
            bad_arr = base_kw[field_name].copy()
            bad_arr[3] = sign * np.inf
            with pytest.raises(ValueError, match=field_name):
                compute_stagnant_lid_state(**{**base_kw, field_name: bad_arr, 'xp': np})

            j_kw = {
                k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
                for k, v in base_kw.items()
            }
            j_kw[field_name] = jnp.asarray(bad_arr)
            st_jax = compute_stagnant_lid_state(**j_kw, xp=jnp)
            assert bool(st_jax['has_lid']) is False
            assert bool(jnp.all(jnp.isnan(st_jax['w_lid'])))
            for k in ('T_i', 'theta', 'eta_i', 'mu_i', 'd_lid', 'v_m', 'tau_d', 'lid_regime'):
                assert bool(jnp.isnan(st_jax[k]))

    for scalar_inf in (np.inf, -np.inf):
        with pytest.raises(ValueError, match='density'):
            compute_stagnant_lid_state(**{**base_kw, 'density': scalar_inf, 'xp': np})
        j_kw_sc = {
            k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in base_kw.items()
        }
        j_kw_sc['density'] = scalar_inf
        st_jax_sc = compute_stagnant_lid_state(**j_kw_sc, xp=jnp)
        assert bool(st_jax_sc['has_lid']) is False
        assert bool(jnp.isnan(st_jax_sc['tau_d']))
        assert bool(jnp.all(jnp.isnan(st_jax_sc['w_lid'])))
