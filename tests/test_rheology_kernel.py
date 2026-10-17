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
    t_sol_grid = np.array([1428.9478, 2969.0408, 3668.0717, 4270.4757, 4595.2113])
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

    # Stagnant lid mode: under min closure, tau_d == tau_y yields eta_eff == eta_diff.
    eta_eff_lid = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_diff,
        tau_d=tau_y,
        tau_y_lid=tau_y,
        stress_closure_mode='lid',
    )
    assert eta_eff_lid == pytest.approx(eta_diff, rel=1.0e-12)


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
    assert eta_eff_high == eta_d

    # Limit 2: eta_y << eta_d (high strain rate, yielding).
    sr_high = 5.0e-11
    eta_y_low = tau_y / (2.0 * sr_high)
    eta_eff_low = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_high,
        stress_closure_mode='local',
    )
    assert eta_eff_low == eta_y_low


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
    jnp = pytest.importorskip('jax.numpy')

    jax.config.update('jax_enable_x64', True)

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
