"""Unit tests for the shared array-agnostic rheology kernel.

Validates asymptotic physical limits, monotonicity, continuity,
and autodiff properties of the kernel in aragog.rheology.
"""

from __future__ import annotations

from decimal import Decimal

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
    compute_yield_stress,
)
from aragog.rheology_lid import (
    SOLIDUS_MELT_FRACTION_THRESHOLD,
    SOLIDUS_MELT_FRACTION_WIDTH,
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
def test_wolf_bower_2018_solidus_table_interpolation_matches_pinned_grid():
    """Verify solidus.dat interpolation matches pinned test grid and docs table."""
    import os
    from pathlib import Path

    fwl_data = os.environ.get('FWL_DATA')
    if not fwl_data:
        pytest.skip('FWL_DATA not set')
    table_path = (
        Path(fwl_data) / 'interior/melting_curves/wolf_bower_2018/r15728072/solidus.dat'
    )
    if not table_path.is_file():
        pytest.skip(f'Solidus table not found at {table_path}')

    raw = np.loadtxt(table_path, comments='#')
    p_table, t_table = raw[:, 0], raw[:, 1]
    p_grid = np.array([0.0, 25.0e9, 50.0e9, 100.0e9, 135.0e9])
    t_sol_interp = np.interp(p_grid, p_table, t_table)

    pinned = np.array([1380.6038, 2475.9161, 2875.9474, 3673.7136, 4232.1499])
    np.testing.assert_allclose(t_sol_interp, pinned, atol=1.0e-4)

    docs_1dec = np.array([1380.6, 2475.9, 2875.9, 3673.7, 4232.1])
    np.testing.assert_allclose(np.round(t_sol_interp, 1), docs_1dec, atol=1.0e-12)


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
    """Verify phi >= 0.03 has no lid at N in {3, 10, 50, 100, 200} with/without table."""
    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0)
    spans = [(3.48e6, 6.37e6), (3.0e6, 12.0e6)]
    for r_in, r_out in spans:
        for n_nodes in (3, 10, 50, 100, 200):
            radii = np.linspace(r_in, r_out, n_nodes)
            pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
            t_sol = 1400.0 + 2.4e-8 * pressure
            conv_flux = np.full(n_nodes, 99.0)
            tot_flux = np.full(n_nodes, 100.0)

            def run(temperature, convective_flux, phi_val, sol_arg):
                return compute_stagnant_lid_state(
                    radii=radii,
                    temperature=temperature,
                    pressure=pressure,
                    convective_flux=convective_flux,
                    total_flux=tot_flux,
                    solidus_temperature=sol_arg,
                    t_sol_0=1400.0,
                    melt_fraction=np.full(n_nodes, phi_val),
                    params=params,
                    viscosity_solid=1.0e21,
                    phi_rheo=0.4,
                    phi_width=0.15,
                    xp=np,
                )

            for sol_arg in (t_sol, None):
                for dt_above in (100.0, 200.0):
                    t_prof = t_sol + dt_above
                    for phi_val in (0.03, 0.05, 0.10, 0.41, 0.50):
                        state = run(t_prof, conv_flux, phi_val, sol_arg)
                        assert bool(state['has_lid']) is False
                        assert float(state['d_lid']) == 0.0
                        assert float(radii[-1] - state['r_lid_base']) < 1.0
                        assert float(state['tau_d']) == 0.0
                        assert float(state['lid_regime']) == 0.0
                        atol = 1.0e-8 if phi_val == 0.03 else 1.0e-12
                        np.testing.assert_allclose(state['w_lid'], 0.0, atol=atol)

            # At phi = 0.005 (< SOLIDUS_MELT_FRACTION_THRESHOLD = 0.01) with a cold surface lid and a table,
            # the lid remains active, pinning the solidus melt threshold from both sides.
            if n_nodes >= 10:
                t_lid_prof = t_sol + 200.0
                n_lid_cold = min(6, n_nodes)
                t_lid_prof[-n_lid_cold:] = np.linspace(1600.0, 300.0, n_lid_cold)
                conv_lid = conv_flux.copy()
                conv_lid[-n_lid_cold:] = 0.0
                st_sub = run(t_lid_prof, conv_lid, 0.005, t_sol)
                assert bool(st_sub['has_lid']) is True
                assert float(st_sub['d_lid']) > 10.0e3
                assert float(st_sub['tau_d']) > 0.0

    # A molten interior (phi = 0.05) with a cold solid surface node (phi[-1] = 0.0) has a
    # surface lid bounded by sum_k (1 - w_sol_phi[k+1]) * dr_k, exceeding the k-indexed sum.
    r_12 = np.linspace(3.48e6, 6.37e6, 12)
    p_12 = np.linspace(135.0e9, 1.0e5, 12)
    t_12 = np.full(12, 3000.0)
    t_12[-1] = 300.0
    f_conv_12 = np.full(12, 80.0)
    f_conv_12[-1] = 0.0
    phi_12 = np.full(12, 0.05)
    phi_12[-1] = 0.0
    st_cold_surf = compute_stagnant_lid_state(
        radii=r_12,
        temperature=t_12,
        pressure=p_12,
        convective_flux=f_conv_12,
        total_flux=np.full(12, 100.0),
        solidus_temperature=None,
        t_sol_0=1400.0,
        melt_fraction=phi_12,
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    zero_sol = 0.5 * (
        1.0 + np.tanh(-SOLIDUS_MELT_FRACTION_THRESHOLD / SOLIDUS_MELT_FRACTION_WIDTH)
    )
    raw_sol = 0.5 * (
        1.0 + np.tanh((phi_12 - SOLIDUS_MELT_FRACTION_THRESHOLD) / SOLIDUS_MELT_FRACTION_WIDTH)
    )
    w_sol_phi_12 = np.clip((raw_sol - zero_sol) / (1.0 - zero_sol), 0.0, 1.0)
    dr_12 = np.diff(r_12)
    bound_upper_node = float(np.sum((1.0 - w_sol_phi_12[1:]) * dr_12))
    bound_lower_node = float(np.sum((1.0 - w_sol_phi_12[:-1]) * dr_12))
    assert bool(st_cold_surf['has_lid']) is True
    assert 0.0 < float(st_cold_surf['d_lid']) <= bound_upper_node
    assert float(st_cold_surf['d_lid']) > bound_lower_node


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

    zero_sol = 0.5 * (
        1.0 + np.tanh(-SOLIDUS_MELT_FRACTION_THRESHOLD / SOLIDUS_MELT_FRACTION_WIDTH)
    )
    raw_sol = 0.5 * (
        1.0 + np.tanh((phi - SOLIDUS_MELT_FRACTION_THRESHOLD) / SOLIDUS_MELT_FRACTION_WIDTH)
    )
    expected_w_sol_phi = np.clip((raw_sol - zero_sol) / (1.0 - zero_sol), 0.0, 1.0)
    expected_w_int = np.maximum(expected_w_hot_phi[-5:-1], expected_w_sol_phi[-5:-1])

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
        # is below SOLIDUS_MELT_FRACTION_THRESHOLD (0.01), so w_hot_phi dominates with w_sol_phi.
        expected_w_lid_slice = (1.0 - expected_w_int) * float(state['w_active'])
        np.testing.assert_allclose(
            state['w_lid'][-5:-1], expected_w_lid_slice, rtol=1.0e-12, atol=1.0e-14
        )


def _reaches_jit_kernel(val) -> bool:
    """Return True if JAX traces val into the jitted function instead of rejecting at trace entry."""
    if isinstance(val, (list, tuple)):
        return all(_reaches_jit_kernel(x) for x in val)
    if isinstance(val, np.ndarray):
        return val.dtype.kind in ('b', 'i', 'u', 'f', 'c')
    return val is None or isinstance(val, (bool, int, float, complex, np.generic))


def _assert_invalid_raises_np_and_nan_jit(
    fn,
    np_kw: dict,
    jax_kw: dict,
    jit_fn,
    field_name: str,
    bad_val,
    *,
    static_error: bool = False,
    is_lid_state: bool = False,
) -> None:
    """Assert ValueError on eager NumPy and NaN (or ValueError/TypeError for static errors) under jax.jit."""
    jnp = pytest.importorskip('jax.numpy')
    with pytest.raises(ValueError, match=field_name):
        fn(**{**np_kw, field_name: bad_val, 'xp': np})

    if static_error:
        # JAX rejects str, Decimal, and object leaves with TypeError before entering jit_fn.
        expected_exc = ValueError if _reaches_jit_kernel(bad_val) else TypeError
        with pytest.raises(expected_exc, match=field_name):
            jit_fn({**jax_kw, field_name: bad_val})
        return

    j_val = jnp.asarray(bad_val, dtype=float)
    res_jit = jit_fn({**jax_kw, field_name: j_val})
    if is_lid_state:
        assert bool(res_jit['has_lid']) is False
        assert bool(jnp.all(jnp.isnan(res_jit['w_lid'])))
        for k in ('T_i', 'theta', 'eta_i', 'mu_i', 'd_lid', 'v_m', 'tau_d', 'lid_regime'):
            assert bool(jnp.isnan(res_jit[k]))
    else:
        assert bool(jnp.all(jnp.isnan(res_jit)))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('field_name', 'bad_val', 'static_error'),
    [
        ('pressure', -1.0, False),
        ('pressure', np.inf, False),
        ('pressure', -np.inf, False),
        ('pressure', True, True),
        ('pressure', [True, True], True),
        ('pressure', 1.0 + 2.0j, True),
        ('pressure', np.array([1.0 + 2.0j, 2.0 + 3.0j]), True),
        ('pressure', ['a', 'b'], True),
        ('pressure', np.array(['a', 'b']), True),
        ('pressure', [1.0, None], True),
        ('pressure', np.array([1.0, object()], dtype=object), True),
        ('g_homologous', -25.0, False),
        ('g_homologous', -1.0e-6, False),
        ('g_homologous', np.inf, False),
        ('g_homologous', -np.inf, False),
        ('g_homologous', np.array([25.0, 14.0]), True),
        ('g_homologous', True, True),
        ('g_homologous', 1.0 + 2.0j, True),
        ('g_homologous', 'bad', True),
        ('g_homologous', object(), True),
        ('g_homologous', Decimal('25.0'), True),
        ('solidus_temperature', 0.0, False),
        ('solidus_temperature', -100.0, False),
        ('solidus_temperature', np.inf, False),
        ('solidus_temperature', -np.inf, False),
        ('solidus_temperature', True, True),
        ('t_sol_0', 0.0, False),
        ('t_sol_0', -100.0, False),
        ('t_sol_0', np.inf, False),
        ('t_sol_0', -np.inf, False),
        ('t_sol_0', np.array([1400.0, 1500.0]), True),
        ('t_sol_0', True, True),
        ('dt_sol_dp', -1.0e-8, False),
        ('dt_sol_dp', np.inf, False),
        ('dt_sol_dp', -np.inf, False),
        ('dt_sol_dp', True, True),
    ],
)
def test_arrhenius_enthalpy_validation_table(field_name: str, bad_val, static_error: bool):
    """Verify compute_arrhenius_enthalpy rejects invalid inputs on np and returns NaN under jax.jit."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    np_kw = {
        'pressure': 1.0e9,
        'g_homologous': 25.0,
        'solidus_temperature': None,
        't_sol_0': 1400.0,
        'dt_sol_dp': 2.4e-8,
    }
    jax_kw = {**np_kw, 'pressure': jnp.asarray(1.0e9)}
    jit_fn = jax.jit(lambda kw: compute_arrhenius_enthalpy(**kw, xp=jnp))
    _assert_invalid_raises_np_and_nan_jit(
        compute_arrhenius_enthalpy,
        np_kw,
        jax_kw,
        jit_fn,
        field_name,
        bad_val,
        static_error=static_error,
    )
    assert compute_arrhenius_enthalpy(1.0e9, g_homologous=0.0, xp=np) == 0.0
    assert np.isnan(compute_arrhenius_enthalpy(1.0e9, g_homologous=np.nan, xp=np))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('field_name', 'bad_val', 'static_error'),
    [
        ('temperature', 0.0, False),
        ('temperature', -50.0, False),
        ('temperature', np.inf, False),
        ('temperature', -np.inf, False),
        ('temperature', True, True),
        ('temperature', [True, True], True),
        ('temperature', 1600.0 + 1.0j, True),
        ('temperature', np.array([1600.0 + 1.0j, 1500.0 + 2.0j]), True),
        ('temperature', ['1600', '1500'], True),
        ('temperature', np.array(['1600', '1500']), True),
        ('temperature', [1600.0, None], True),
        ('temperature', np.array([1600.0, object()], dtype=object), True),
        ('pressure', -1.0, False),
        ('pressure', np.inf, False),
        ('pressure', -np.inf, False),
        ('pressure', False, True),
        ('pressure', [True, True], True),
        ('viscosity_solid', 0.0, False),
        ('viscosity_solid', -1.0e21, False),
        ('viscosity_solid', np.inf, False),
        ('viscosity_solid', -np.inf, False),
        ('viscosity_solid', np.array([1.0e21, 2.0e21]), True),
        ('viscosity_solid', True, True),
        ('viscosity_solid', 1.0e21 + 1.0j, True),
        ('viscosity_solid', '1e21', True),
        ('viscosity_solid', object(), True),
        ('viscosity_solid', Decimal('1e21'), True),
        ('g_homologous', -25.0, False),
        ('g_homologous', -1.0e-6, False),
        ('g_homologous', np.inf, False),
        ('g_homologous', -np.inf, False),
        ('g_homologous', np.array([25.0, 14.0]), True),
        ('g_homologous', np.bool_(True), True),
        ('solidus_temperature', 0.0, False),
        ('solidus_temperature', -500.0, False),
        ('solidus_temperature', np.inf, False),
        ('solidus_temperature', -np.inf, False),
        ('solidus_temperature', True, True),
        ('t_sol_0', 0.0, False),
        ('t_sol_0', -100.0, False),
        ('t_sol_0', np.inf, False),
        ('t_sol_0', -np.inf, False),
        ('t_sol_0', True, True),
        ('dt_sol_dp', -1.0e-8, False),
        ('dt_sol_dp', np.inf, False),
        ('dt_sol_dp', -np.inf, False),
        ('dt_sol_dp', True, True),
        ('arrhenius_t_ref', 0.0, False),
        ('arrhenius_t_ref', -100.0, False),
        ('arrhenius_t_ref', np.inf, False),
        ('arrhenius_t_ref', -np.inf, False),
        ('arrhenius_t_ref', True, True),
        ('water_prefactor', 0.0, False),
        ('water_prefactor', -0.5, False),
        ('water_prefactor', np.inf, False),
        ('water_prefactor', -np.inf, False),
        ('water_prefactor', True, True),
        ('viscosity_max_log10', np.nan, False),
        ('viscosity_max_log10', np.inf, False),
        ('viscosity_max_log10', -np.inf, False),
        ('viscosity_max_log10', -400.0, False),
        ('viscosity_max_log10', 0.0, False),
        ('viscosity_max_log10', 15.0, False),
        ('viscosity_max_log10', 20.0, False),
        ('viscosity_max_log10', 400.0, False),
        ('viscosity_max_log10', True, True),
        ('viscosity_max_log10', [40.0, 40.0], True),
        ('viscosity_max_log10', 40.0 + 1.0j, True),
        ('viscosity_max_log10', '40.0', True),
        ('viscosity_max_log10', object(), True),
        ('viscosity_max_log10', Decimal('40.0'), True),
    ],
)
def test_arrhenius_viscosity_validation_table(field_name: str, bad_val, static_error: bool):
    """Verify compute_arrhenius_viscosity rejects invalid inputs on np and returns NaN under jax.jit."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    np_kw = {
        'temperature': 1600.0,
        'pressure': 1.0e9,
        'viscosity_solid': 1.0e21,
        'g_homologous': 25.0,
        'solidus_temperature': None,
        't_sol_0': 1400.0,
        'dt_sol_dp': 2.4e-8,
        'arrhenius_t_ref': 1600.0,
        'water_prefactor': 1.0,
        'viscosity_max_log10': 40.0,
    }
    jax_kw = {
        **np_kw,
        'temperature': jnp.asarray(1600.0),
        'pressure': jnp.asarray(1.0e9),
    }
    jit_fn = jax.jit(lambda kw: compute_arrhenius_viscosity(**kw, xp=jnp))
    _assert_invalid_raises_np_and_nan_jit(
        compute_arrhenius_viscosity,
        np_kw,
        jax_kw,
        jit_fn,
        field_name,
        bad_val,
        static_error=static_error,
    )
    assert compute_arrhenius_viscosity(
        1600.0, 0.0, viscosity_solid=1.0e21, g_homologous=0.0
    ) == pytest.approx(1.0e21, rel=1.0e-12)
    assert np.isfinite(
        compute_arrhenius_viscosity(
            800.0, 1.0e9, viscosity_solid=1.0e21, g_homologous=25.0, viscosity_max_log10=300.0
        )
    )
    assert np.isnan(compute_arrhenius_viscosity(1600.0, 0.0, viscosity_solid=np.nan, xp=np))
    assert np.isnan(compute_arrhenius_viscosity(1600.0, 0.0, g_homologous=np.nan, xp=np))


_RADII_DUP = np.linspace(3.48e6, 6.37e6, 12)
_RADII_DUP[3] = _RADII_DUP[2]


@pytest.mark.unit
@pytest.mark.parametrize(
    ('field_name', 'bad_entry', 'as_scalar', 'static_error'),
    [
        ('radii', 0.0, False, False),
        ('radii', _RADII_DUP, True, False),
        ('radii', np.linspace(6.37e6, 3.48e6, 12), True, False),
        ('radii', np.full(12, 3.48e6 + 1.0j), True, True),
        ('radii', ['3.48e6'] * 12, True, True),
        ('radii', [3.48e6] * 11 + [None], True, True),
        ('radii', np.array([3.48e6] * 11 + [object()], dtype=object), True, True),
        ('radii', [Decimal('3.48e6')] * 12, True, True),
        ('temperature', np.inf, False, False),
        ('temperature', -np.inf, False, False),
        ('temperature', 0.0, False, False),
        ('temperature', -50.0, False, False),
        ('temperature', True, False, True),
        ('temperature', [True] * 12, True, True),
        ('temperature', np.full(12, 1600.0 + 1.0j), True, True),
        ('temperature', ['1600.0'] * 12, True, True),
        ('temperature', np.array(['1600.0'] * 12), True, True),
        ('temperature', [1600.0] * 11 + [None], True, True),
        ('temperature', np.array([1600.0] * 11 + [object()], dtype=object), True, True),
        ('temperature', [Decimal('1600.0')] * 12, True, True),
        ('pressure', np.inf, False, False),
        ('pressure', -np.inf, False, False),
        ('pressure', -1.0e5, False, False),
        ('convective_flux', np.inf, False, False),
        ('convective_flux', -np.inf, False, False),
        ('total_flux', np.inf, False, False),
        ('total_flux', -np.inf, False, False),
        ('melt_fraction', np.inf, False, False),
        ('melt_fraction', -np.inf, False, False),
        ('melt_fraction', -0.01, False, False),
        ('melt_fraction', 1.01, False, False),
        ('solidus_temperature', np.inf, False, False),
        ('solidus_temperature', -np.inf, False, False),
        ('solidus_temperature', 0.0, False, False),
        ('solidus_temperature', -100.0, True, False),
        ('unyielded_velocity', np.inf, False, False),
        ('unyielded_velocity', -np.inf, False, False),
        ('viscosity_mixture', np.inf, False, False),
        ('viscosity_mixture', -np.inf, False, False),
        ('viscosity_mixture', 0.0, False, False),
        ('viscosity_mixture', -1.0e20, True, False),
        ('density', np.inf, False, False),
        ('density', -np.inf, True, False),
        ('density', 0.0, False, False),
        ('density', -3300.0, True, False),
        ('density', True, True, True),
        ('density', 3300.0 + 1.0j, True, True),
        ('density', '3300.0', True, True),
        ('density', object(), True, True),
        ('density', Decimal('3300.0'), True, True),
        ('gravity', np.inf, False, False),
        ('gravity', -np.inf, False, False),
        ('gravity', 0.0, True, False),
        ('gravity', -9.81, False, False),
        ('thermal_expansivity', np.inf, False, False),
        ('thermal_expansivity', -np.inf, False, False),
        ('thermal_expansivity', 0.0, True, False),
        ('thermal_expansivity', -3.0e-5, False, False),
        ('thermal_diffusivity', np.inf, False, False),
        ('thermal_diffusivity', -np.inf, False, False),
        ('thermal_diffusivity', 0.0, True, False),
        ('thermal_diffusivity', -1.0e-6, False, False),
        ('t_sol_0', 0.0, True, False),
        ('t_sol_0', -100.0, True, False),
        ('t_sol_0', 1400.0 + 1.0j, True, True),
        ('t_sol_0', '1400.0', True, True),
        ('t_sol_0', object(), True, True),
        ('t_sol_0', Decimal('1400.0'), True, True),
    ],
)
def test_stagnant_lid_state_rejects_inf_in_array_and_profile_inputs(
    field_name: str, bad_entry, as_scalar: bool, static_error: bool
):
    """Verify compute_stagnant_lid_state rejects invalid array/scalar inputs on np and returns NaN under jax.jit."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    n_nodes = 12
    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0)
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
        'params': params,
        'viscosity_solid': 1.0e21,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
    }
    jax_kw = {
        k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
        for k, v in base_kw.items()
        if k != 'params'
    }
    jit_lid = jax.jit(lambda kw: compute_stagnant_lid_state(**kw, params=params, xp=jnp))

    if static_error:
        bad_val = bad_entry if as_scalar else np.ones(n_nodes, dtype=bool)
    elif as_scalar:
        bad_val = bad_entry
    else:
        bad_val = base_kw[field_name].copy()
        bad_val[3] = bad_entry

    _assert_invalid_raises_np_and_nan_jit(
        compute_stagnant_lid_state,
        base_kw,
        jax_kw,
        jit_lid,
        field_name,
        bad_val,
        static_error=static_error,
        is_lid_state=True,
    )


@pytest.mark.unit
def test_pressure_surface_roundoff_clamping_and_validation():
    """Verify near-zero negative surface pressure is clamped to zero and large negative pressure raises ValueError."""
    p_tol_val = -1.0e-4
    h_clamped = compute_arrhenius_enthalpy(p_tol_val, xp=np)
    h_zero = compute_arrhenius_enthalpy(0.0, xp=np)
    assert h_clamped == pytest.approx(h_zero, abs=1e-12)

    eta_clamped = compute_arrhenius_viscosity(1600.0, p_tol_val, xp=np)
    eta_zero = compute_arrhenius_viscosity(1600.0, 0.0, xp=np)
    assert eta_clamped == pytest.approx(eta_zero, rel=1e-12)

    tau_y_clamped = compute_yield_stress(p_tol_val, xp=np)
    tau_y_zero = compute_yield_stress(0.0, xp=np)
    assert tau_y_clamped == pytest.approx(tau_y_zero, rel=1e-12)

    n_nodes = 12
    params = SolidRheologyParams(enabled=True)
    radii = np.linspace(3.48e6, 6.37e6, n_nodes)
    temp = np.linspace(2600.0, 300.0, n_nodes)
    press_zero = np.linspace(135.0e9, 0.0, n_nodes)
    press_roundoff = press_zero.copy()
    press_roundoff[-1] = p_tol_val

    st_roundoff = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp,
        pressure=press_roundoff,
        convective_flux=np.full(n_nodes, 80.0),
        total_flux=np.full(n_nodes, 100.0),
        solidus_temperature=None,
        params=params,
        melt_fraction=np.zeros(n_nodes),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        xp=np,
    )
    st_zero = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp,
        pressure=press_zero,
        convective_flux=np.full(n_nodes, 80.0),
        total_flux=np.full(n_nodes, 100.0),
        solidus_temperature=None,
        params=params,
        melt_fraction=np.zeros(n_nodes),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        xp=np,
    )
    assert float(st_roundoff['d_lid']) == pytest.approx(float(st_zero['d_lid']), rel=1e-12)

    with pytest.raises(ValueError, match='pressure'):
        compute_arrhenius_enthalpy(-10.0, xp=np)

    with pytest.raises(ValueError, match='pressure'):
        compute_arrhenius_viscosity(1600.0, -10.0, xp=np)

    with pytest.raises(ValueError, match='pressure'):
        compute_yield_stress(-10.0, xp=np)

    press_bad = press_zero.copy()
    press_bad[-1] = -10.0
    with pytest.raises(ValueError, match='pressure'):
        compute_stagnant_lid_state(
            radii=radii,
            temperature=temp,
            pressure=press_bad,
            convective_flux=np.full(n_nodes, 80.0),
            total_flux=np.full(n_nodes, 100.0),
            solidus_temperature=None,
            params=params,
            melt_fraction=np.zeros(n_nodes),
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            unyielded_velocity=np.full(n_nodes, 1.0e-9),
            xp=np,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ('mode', 'field_name', 'bad_val', 'static_error'),
    [
        ('lid', 'eta_diff', 0.0, False),
        ('lid', 'eta_diff', -1.0e22, False),
        ('lid', 'eta_diff', np.inf, False),
        ('lid', 'eta_diff', -np.inf, False),
        ('lid', 'eta_diff', True, True),
        ('lid', 'eta_diff', [True, True], True),
        ('lid', 'eta_diff', 1.0e22 + 1.0j, True),
        ('lid', 'eta_diff', np.array([1.0e22 + 1.0j, 1.0e22 + 2.0j]), True),
        ('lid', 'eta_diff', ['a', 'b'], True),
        ('lid', 'eta_diff', np.array(['a', 'b']), True),
        ('lid', 'eta_diff', [1.0e22, None], True),
        ('lid', 'eta_diff', np.array([1.0e22, object()], dtype=object), True),
        ('lid', 'eta_diff', Decimal('1e22'), True),
        ('lid', 'eta_i', 0.0, False),
        ('lid', 'eta_i', -1.0e20, False),
        ('lid', 'eta_i', np.inf, False),
        ('lid', 'eta_i', -np.inf, False),
        ('lid', 'eta_i', True, True),
        ('lid', 'tau_d', np.inf, False),
        ('lid', 'tau_d', -np.inf, False),
        ('lid', 'tau_d', True, True),
        ('lid', 'tau_y_lid', -5.0, False),
        ('lid', 'tau_y_lid', -1.0e-6, False),
        ('lid', 'tau_y_lid', -np.inf, False),
        ('lid', 'tau_y_lid', True, True),
        ('lid', 'tau_y_lid', 1.0e8 + 1.0j, True),
        ('lid', 'tau_y_lid', '1e8', True),
        ('lid', 'tau_y_lid', object(), True),
        ('lid', 'tau_y_lid', Decimal('1e8'), True),
        ('lid', 'w_lid', -0.1, False),
        ('lid', 'w_lid', 1.1, False),
        ('lid', 'w_lid', np.inf, False),
        ('lid', 'w_lid', -np.inf, False),
        ('lid', 'w_lid', True, True),
        ('local', 'eta_diff', 0.0, False),
        ('local', 'eta_diff', -1.0e22, False),
        ('local', 'eta_diff', np.inf, False),
        ('local', 'eta_diff', -np.inf, False),
        ('local', 'eta_diff', True, True),
        ('local', 'eta_diff', [True, True], True),
        ('local', 'eta_diff', 1.0e22 + 1.0j, True),
        ('local', 'eta_diff', [1.0e22, None], True),
        ('local', 'strain_rate', -1.0e-15, False),
        ('local', 'strain_rate', np.inf, False),
        ('local', 'strain_rate', -np.inf, False),
        ('local', 'strain_rate', True, True),
        ('local', 'tau_y', -5.0, False),
        ('local', 'tau_y', -1.0e-6, False),
        ('local', 'tau_y', np.array([-5.0, -1.0e-6]), False),
        ('local', 'tau_y', -np.inf, False),
        ('local', 'tau_y', True, True),
        ('local', 'tau_y', 1.0e8 + 1.0j, True),
        ('local', 'tau_y', '1e8', True),
        ('local', 'tau_y', object(), True),
        ('local', 'tau_y', Decimal('1e8'), True),
        ('local', 'mixing_length', -1.0, False),
        ('local', 'mixing_length', np.inf, False),
        ('local', 'mixing_length', -np.inf, False),
        ('local', 'mixing_length', True, True),
        ('local', 'unyielded_velocity', np.inf, False),
        ('local', 'unyielded_velocity', -np.inf, False),
        ('local', 'unyielded_velocity', True, True),
        ('local', 'tau_y_profile', -5.0, False),
        ('local', 'tau_y_profile', -1.0e-6, False),
        ('local', 'tau_y_profile', np.array([-5.0, -1.0e-6]), False),
        ('local', 'tau_y_profile', -np.inf, False),
        ('local', 'tau_y_profile', True, True),
        ('local', 'tau_y_profile', [1.0e8, None], True),
        ('local', 'tau_y_profile', np.array([1.0e8 + 1.0j, 1.0e8 + 2.0j]), True),
    ],
)
def test_compute_effective_viscosity_rejects_invalid_inputs(
    mode: str, field_name: str, bad_val, static_error: bool
):
    """Verify compute_effective_viscosity rejects invalid inputs on np and returns NaN under jit."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    lid_np = {
        'eta_diff': 1.0e22,
        'eta_i': 1.0e20,
        'tau_d': 1.0e6,
        'tau_y_lid': 1.0e8,
        'w_lid': 1.0,
        'stress_closure_mode': 'lid',
    }
    local_np = {
        'eta_diff': 1.0e22,
        'strain_rate': 1.0e-15,
        'tau_y': 1.0e8,
        'stress_closure_mode': 'local',
    }
    local_vel_np = {
        'eta_diff': 1.0e22,
        'unyielded_velocity': 1.0e-9,
        'mixing_length': 1.0e5,
        'tau_y_profile': 1.0e8,
        'stress_closure_mode': 'local',
    }
    if mode == 'lid':
        np_kw = lid_np
    elif field_name in ('unyielded_velocity', 'mixing_length', 'tau_y_profile'):
        np_kw = local_vel_np
    else:
        np_kw = local_np
    jax_kw = {
        k: (v if k == 'stress_closure_mode' else jnp.asarray(v, dtype=float))
        for k, v in np_kw.items()
    }
    jit_eta = jax.jit(
        lambda m, kw: compute_effective_viscosity(
            **{**kw, 'stress_closure_mode': m, 'xp': jnp}
        ),
        static_argnums=0,
    )
    _assert_invalid_raises_np_and_nan_jit(
        compute_effective_viscosity,
        np_kw,
        {k: v for k, v in jax_kw.items() if k != 'stress_closure_mode'},
        lambda kw: jit_eta(mode, kw),
        field_name,
        bad_val,
        static_error=static_error,
    )


@pytest.mark.unit
def test_compute_effective_viscosity_accepts_infinite_tau_y_lid():
    """Verify compute_effective_viscosity accepts +inf yield stress with finite jax.grad."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    jit_eta = jax.jit(
        lambda kw: compute_effective_viscosity(**kw, stress_closure_mode='lid', xp=jnp)
    )
    for td_finite in (0.0, 1.0e6, 1.0e12, 1.0e31):
        for wl_val in (None, 0.0, 0.5, 1.0):
            res_np = compute_effective_viscosity(
                eta_diff=1.0e22,
                eta_i=1.0e20,
                tau_d=td_finite,
                tau_y_lid=np.inf,
                w_lid=wl_val,
                stress_closure_mode='lid',
                xp=np,
            )
            assert float(res_np) == pytest.approx(1.0e22, rel=1.0e-12)
        res_jit_inf = jit_eta(
            {
                'eta_diff': jnp.asarray(1.0e22),
                'eta_i': jnp.asarray(1.0e20),
                'tau_d': jnp.asarray(td_finite),
                'tau_y_lid': jnp.asarray(np.inf),
                'w_lid': jnp.asarray(1.0),
            }
        )
        assert float(res_jit_inf) == pytest.approx(1.0e22, rel=1.0e-12)

    res_large_finite = compute_effective_viscosity(
        eta_diff=1.0e22,
        eta_i=1.0e20,
        tau_d=5.0e30,
        tau_y_lid=1.0e31,
        w_lid=1.0,
        stress_closure_mode='lid',
        xp=np,
    )
    assert float(res_large_finite) == pytest.approx(1.0e22, rel=1.0e-12)

    for wl_j in (None, jnp.asarray(0.5), jnp.asarray(1.0)):
        grads_lid = jax.grad(
            lambda ei, td, ed, w=wl_j: compute_effective_viscosity(
                eta_diff=ed,
                eta_i=ei,
                tau_d=td,
                tau_y_lid=jnp.asarray(jnp.inf),
                w_lid=w,
                stress_closure_mode='lid',
                xp=jnp,
            ),
            argnums=(0, 1, 2),
        )(jnp.asarray(1.0e20), jnp.asarray(1.0e6), jnp.asarray(1.0e22))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads_lid)
        assert float(grads_lid[0]) == 0.0
        assert float(grads_lid[1]) == 0.0
        assert float(grads_lid[2]) == pytest.approx(1.0, rel=1.0e-12)

    for ed_val, sr_val in ((1.0e22, 1.0e-15), (1.0e40, 1.0e-10)):
        res_loc = compute_effective_viscosity(
            eta_diff=ed_val,
            strain_rate=sr_val,
            tau_y=np.inf,
            stress_closure_mode='local',
            xp=np,
        )
        assert float(res_loc) == pytest.approx(ed_val, rel=1.0e-12)
        grads_local = jax.grad(
            lambda ed, sr: compute_effective_viscosity(
                eta_diff=ed,
                strain_rate=sr,
                tau_y=jnp.asarray(jnp.inf),
                stress_closure_mode='local',
                xp=jnp,
            ),
            argnums=(0, 1),
        )(jnp.asarray(ed_val), jnp.asarray(sr_val))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads_local)
        assert float(grads_local[0]) == pytest.approx(1.0, rel=1.0e-12)
        assert float(grads_local[1]) == 0.0

    # Include a cold surface node (eta_diff = 1e40 Pa s, mixing_length = 0.0 m).
    ed_prof = jnp.array([1.0e22, 1.0e40])
    v_prof = jnp.array([1.0e-9, 1.0e-20])
    ml_prof = jnp.array([1.0e5, 0.0])
    res_prof = compute_effective_viscosity(
        eta_diff=ed_prof,
        unyielded_velocity=v_prof,
        mixing_length=ml_prof,
        tau_y_profile=jnp.array([jnp.inf, jnp.inf]),
        stress_closure_mode='local',
        xp=jnp,
    )
    np.testing.assert_allclose(res_prof, [1.0e22, 1.0e40], rtol=1.0e-12)
    grads_prof = jax.grad(
        lambda ed, v, ml: jnp.sum(
            compute_effective_viscosity(
                eta_diff=ed,
                unyielded_velocity=v,
                mixing_length=ml,
                tau_y_profile=jnp.array([jnp.inf, jnp.inf]),
                stress_closure_mode='local',
                xp=jnp,
            )
        ),
        argnums=(0, 1, 2),
    )(ed_prof, v_prof, ml_prof)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads_prof)
    np.testing.assert_allclose(grads_prof[0], [1.0, 1.0], rtol=1.0e-12)
    np.testing.assert_allclose(grads_prof[1], [0.0, 0.0], atol=1.0e-30)
    np.testing.assert_allclose(grads_prof[2], [0.0, 0.0], atol=1.0e-30)


@pytest.mark.unit
def test_stagnant_lid_state_unyielded_velocity_list_input():
    """Verify compute_stagnant_lid_state accepts a Python list for unyielded_velocity on numpy and jax.numpy."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    radii = [3.48e6, 4.10e6, 4.80e6, 5.60e6, 6.37e6]
    temperature = [3000.0, 2600.0, 2200.0, 1800.0, 300.0]
    pressure = [1.35e11, 9.0e10, 5.0e10, 2.0e10, 0.0]
    convective_flux = [0.0, 0.5, 0.5, 0.5, 0.0]
    total_flux = [0.5, 0.5, 0.5, 0.5, 0.5]
    melt_fraction = [0.0, 0.0, 0.0, 0.0, 0.0]
    vel_list = [1.0e-9, -2.0e-9, 3.0e-9, -4.0e-9, 1.0e-12]

    ref_np = compute_stagnant_lid_state(
        radii=np.asarray(radii),
        temperature=np.asarray(temperature),
        pressure=np.asarray(pressure),
        convective_flux=np.asarray(convective_flux),
        total_flux=np.asarray(total_flux),
        melt_fraction=np.asarray(melt_fraction),
        solidus_temperature=None,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.asarray(vel_list),
        xp=np,
    )
    state_np = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=convective_flux,
        total_flux=total_flux,
        melt_fraction=melt_fraction,
        solidus_temperature=None,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=vel_list,
        xp=np,
    )
    state_jnp = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=convective_flux,
        total_flux=total_flux,
        melt_fraction=melt_fraction,
        solidus_temperature=None,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=vel_list,
        xp=jnp,
    )
    assert float(ref_np['v_i']) > 0.0
    assert float(state_np['v_i']) == pytest.approx(float(ref_np['v_i']), rel=1.0e-12, abs=0.0)
    assert float(state_jnp['v_i']) == pytest.approx(float(ref_np['v_i']), rel=1.0e-12, abs=0.0)
    np.testing.assert_allclose(state_np['w_upper'], ref_np['w_upper'], rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(
        state_jnp['w_upper'], ref_np['w_upper'], rtol=1.0e-12, atol=1.0e-15
    )
