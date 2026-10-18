"""Tests for Foley & Bercovici (2014) stagnant lid driving stress closure.

References
----------
Foley & Bercovici (2014), GJI 199, pp. 580-603, doi:10.1093/gji/ggu275
    Eq. 26 (p. 586-587): v_m = (kappa / d) * C4 * (Ra_eff * a_rh / theta)**(2/3)
    Eq. 28 (p. 588): tau_xz = 2 * mu_i * v_m / d
    Sec. 4.2 (p. 589): C4 = 0.125, a_rh = 1.3
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from aragog.rheology import SolidRheologyParams
from aragog.rheology_lid import compute_stagnant_lid_state


def _make_earth_reference_inputs() -> dict[str, Any]:
    n_nodes = 50
    radii = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    temperature = np.full(n_nodes, 1600.0)
    temperature[-8:] = np.linspace(1600.0, 300.0, 8)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    convective_flux = np.full(n_nodes, 100.0)
    convective_flux[-8:] = 0.0
    total_flux = np.full(n_nodes, 100.0)
    melt_frac = np.zeros(n_nodes)

    g_hom_earth = 300.0e3 / (8.314462618 * 1600.0)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        g_homologous=g_hom_earth,
        t_sol_0=1600.0,
        dt_sol_dp=0.0,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
        lid_contrast_coeff=2.2,
    )
    return {
        'radii': radii,
        'temperature': temperature,
        'pressure': pressure,
        'convective_flux': convective_flux,
        'total_flux': total_flux,
        'solidus_temperature': None,
        'melt_fraction': melt_frac,
        'params': params,
        'viscosity_solid': 1.0e20,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
        'density': 3300.0,
        'gravity': 9.81,
        'thermal_expansivity': 3.0e-5,
        'thermal_diffusivity': 1.0e-6,
        'xp': np,
    }


@pytest.mark.unit
def test_fb2014_earth_reference_state():
    """Verify pinned convective driving stress and diagnostics for Earth reference state.

    Parameters for Earth reference case (solid_state_convection.md):
    - rho = 3300.0 kg/m^3
    - g = 9.81 m/s^2
    - alpha = 3.0e-5 1/K
    - T_i = 1600.0 K
    - T_s = 300.0 K -> DeltaT = 1300.0 K
    - d = 2.89e6 m (mantle thickness)
    - kappa = 1.0e-6 m^2/s
    - H = 300.0e3 J/mol
    - R = 8.31446261815324 J/(mol K)
    - eta_i = 1.0e20 Pa s
    - a_rh = 1.3 (FB2014 p. 589)
    - C4 = 0.125 (FB2014 p. 586-587, 589)

    Derived values:
    - theta = 18.322742
    - Ra_eff = 3.047482e8
    - Ra_rh = 2.162191e7
    - v_m = 3.356924e-9 m/s (0.106 m/yr)
    - tau_d = 2.323131e5 Pa (0.232313 MPa)
    - delta_rh = 30791.06 m (30.8 km)
    - tau_buoy = 2.121689e6 Pa (2.12 MPa)
    - tau_d / tau_buoy = 0.109494
    """
    state = compute_stagnant_lid_state(**_make_earth_reference_inputs())

    t_i = float(state['T_i'])
    # Conductive lid nodes at f = 0 have zero weight under w_Ti (f_0 = 0.9, width = 0.02).
    assert t_i == pytest.approx(1600.0, rel=1.0e-12)

    r_gas = 8.314462618
    h_act = 300.0e3
    rho, g, alpha, kappa, d_conv = 3300.0, 9.81, 3.0e-5, 1.0e-6, 2.89e6
    a_rh, c4 = 1.3, 0.125
    eta_i = 1.0e20 * np.exp((h_act / r_gas) * (1.0 / t_i - 1.0 / 1600.0))
    expected_dt_rh = (r_gas * t_i**2) / h_act
    expected_theta = (t_i - 300.0) / expected_dt_rh
    expected_ra_eff = (rho * g * alpha * (t_i - 300.0) * d_conv**3) / (kappa * eta_i)
    expected_ra_rh = (expected_ra_eff * a_rh) / expected_theta
    expected_vm = c4 * (kappa / d_conv) * (expected_ra_rh ** (2.0 / 3.0))
    expected_tau_d = 2.0 * eta_i * expected_vm / d_conv
    # Discretisation regression value on the 50-node mesh; converged value is about 22.5 km.
    expected_delta_rh = 30791.0646
    expected_tau_buoy = rho * g * alpha * expected_dt_rh * expected_delta_rh
    expected_ratio = expected_tau_d / expected_tau_buoy
    # Formula relation: tau_d_lid = 2 * eta_i * v_m / d_lid
    expected_tau_d_lid = 2.0 * eta_i * expected_vm / state['d_lid']
    # Derivation of eta_contrast:
    # At T_surf = 300 K, Arrhenius viscosity exceeds 10^40 Pa s and is clamped to 10^viscosity_max_log10 = 1.0e40 Pa s.
    # At T_i = arrhenius_t_ref = 1600 K, eta_i evaluates to 1.0e20 Pa s.
    # Therefore eta_contrast = eta_surf / eta_i = 1.0e40 / 1.0e20 = 1.0e20.
    expected_eta_surf = 1.0e40  # 10^viscosity_max_log10 with default 40.0
    expected_eta_contrast = expected_eta_surf / eta_i

    assert state['theta'] == pytest.approx(expected_theta, rel=1.0e-6)
    assert state['dT_rh'] == pytest.approx(expected_dt_rh, rel=1.0e-6)
    assert state['Ra_eff'] == pytest.approx(expected_ra_eff, rel=1.0e-6)
    assert state['Ra_rh'] == pytest.approx(expected_ra_rh, rel=1.0e-6)
    assert state['delta_rh'] == pytest.approx(expected_delta_rh, rel=1.0e-6)
    assert state['tau_d'] == pytest.approx(expected_tau_d, rel=1.0e-6)
    assert state['v_m'] == pytest.approx(expected_vm, rel=1.0e-6)
    assert state['tau_buoy'] == pytest.approx(expected_tau_buoy, rel=1.0e-6)
    assert state['tau_d_over_tau_buoy'] == pytest.approx(expected_ratio, rel=1.0e-6)
    assert state['tau_d_lid'] == pytest.approx(expected_tau_d_lid, rel=1.0e-6)
    # Regression pin from 6223e3f: on 50 nodes d_lid = 325494.33 m, giving tau_d_lid ~ 1.774 MPa.
    assert state['tau_d_lid'] == pytest.approx(1774312.943961, rel=1.0e-6)
    assert state['eta_contrast'] == pytest.approx(expected_eta_contrast, rel=1.0e-6)


@pytest.mark.unit
def test_fb2014_monotonicity_in_interior_viscosity():
    """Verify tau_d is strictly monotonically increasing with interior viscosity mu_i.

    In stagnant lid convection with 2/3 velocity scaling, v_m ~ mu_i^(-2/3), so
    tau_d = 2 * mu_i * v_m / d ~ mu_i^(1/3). Monotonicity requires d(tau_d)/d(mu_i) > 0.
    """
    n_nodes = 30
    radii = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    temperature = np.full(n_nodes, 1600.0)
    temperature[-5:] = np.linspace(1600.0, 300.0, 5)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    convective_flux = np.full(n_nodes, 100.0)
    convective_flux[-5:] = 0.0
    total_flux = np.full(n_nodes, 100.0)
    melt_frac = np.zeros(n_nodes)

    viscosities = [1.0e18, 1.0e19, 1.0e20, 1.0e21, 1.0e22]
    tau_d_values = []
    g_hom_earth = 300.0e3 / (8.314462618 * 1600.0)

    for visc in viscosities:
        params = SolidRheologyParams(
            enabled=True,
            stress_closure_mode='lid',
            g_homologous=g_hom_earth,
            t_sol_0=1600.0,
            dt_sol_dp=0.0,
            arrhenius_t_ref=1600.0,
            interior_flux_fraction=0.05,
        )
        st = compute_stagnant_lid_state(
            radii=radii,
            temperature=temperature,
            pressure=pressure,
            convective_flux=convective_flux,
            total_flux=total_flux,
            solidus_temperature=None,
            melt_fraction=melt_frac,
            params=params,
            viscosity_solid=visc,
            phi_rheo=0.4,
            phi_width=0.15,
            density=3300.0,
            gravity=9.81,
            thermal_expansivity=3.0e-5,
            thermal_diffusivity=1.0e-6,
            xp=np,
        )
        tau_d_values.append(float(st['tau_d']))

    for i in range(len(tau_d_values) - 1):
        assert tau_d_values[i] < tau_d_values[i + 1], (
            f'tau_d must be strictly increasing with mu_i: {tau_d_values[i]} >= {tau_d_values[i + 1]}'
        )
        # Each decade of viscosity increases tau_d by 10^(1/3) ~ 2.1544.
        ratio = tau_d_values[i + 1] / tau_d_values[i]
        assert ratio == pytest.approx(10.0 ** (1.0 / 3.0), rel=1.0e-2)


@pytest.mark.unit
def test_interior_selection_first_lid_node_and_resolution_invariance():
    """Verify fixed-width f selector keeps zero-flux lid nodes out of T_i and converges across N."""
    kwargs = _make_earth_reference_inputs()
    st_base = compute_stagnant_lid_state(**kwargs)
    assert abs(float(st_base['T_i']) - 1600.0) < 1.0

    # Setting the first zero-flux lid node (index -8) to 1500 K moves T_i by < 1 K:
    t_perturbed = kwargs['temperature'].copy()
    t_perturbed[-8] = 1500.0
    st_perturbed = compute_stagnant_lid_state(**{**kwargs, 'temperature': t_perturbed})
    assert abs(float(st_perturbed['T_i']) - float(st_base['T_i'])) < 1.0
    assert abs(float(st_perturbed['T_i']) - 1600.0) < 1.0

    # Resolution invariance from N = 200 to N = 400 on the same physical profile (< 2 % change):
    def _eval_at_n(n_nodes: int) -> dict[str, Any]:
        r_cmb, r_surf = 3480.0e3, 6370.0e3
        d_lid_true = 413.0e3
        r_lid = r_surf - d_lid_true
        r = np.linspace(r_cmb, r_surf, n_nodes)
        p = np.linspace(135.0e9, 1.0e5, n_nodes)
        t = np.where(r <= r_lid, 1600.0, 1600.0 - 1300.0 * (r - r_lid) / d_lid_true)
        f_conv = np.where(r <= r_lid, 100.0, 0.0)
        f_tot = np.full(n_nodes, 100.0)
        phi = np.zeros(n_nodes)
        return compute_stagnant_lid_state(
            **{
                **kwargs,
                'radii': r,
                'temperature': t,
                'pressure': p,
                'convective_flux': f_conv,
                'total_flux': f_tot,
                'melt_fraction': phi,
            }
        )

    st_200 = _eval_at_n(200)
    st_400 = _eval_at_n(400)
    rel_diff_ti = abs(float(st_400['T_i']) - float(st_200['T_i'])) / float(st_200['T_i'])
    rel_diff_taud = abs(float(st_400['tau_d']) - float(st_200['tau_d'])) / float(
        st_200['tau_d']
    )
    assert rel_diff_ti < 0.02
    assert rel_diff_taud < 0.02


def _assert_w_conv_fixed_width_pins_w_active_and_w_upper(kwargs: dict[str, Any]) -> None:
    """Pin w_active and w_upper on a smooth f profile rising across interior_flux_fraction."""
    n_nodes = 20
    r = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    p = np.linspace(135.0e9, 1.0e5, n_nodes)
    t = np.linspace(1600.0, 300.0, n_nodes)
    # Smooth convective fraction rising across interior_flux_fraction = 0.05 over several nodes.
    f_smooth = np.linspace(0.035, 0.060, n_nodes)
    f_tot = np.full(n_nodes, 100.0)
    f_conv = f_smooth * f_tot
    v_uny = np.full(n_nodes, 1.0e-9)
    params = kwargs['params']
    st_smooth = compute_stagnant_lid_state(
        **{
            **kwargs,
            'radii': r,
            'temperature': t,
            'pressure': p,
            'convective_flux': f_conv,
            'total_flux': f_tot,
            'melt_fraction': np.zeros(n_nodes),
            'unyielded_velocity': v_uny,
        }
    )
    w_conv_expected = 0.5 * (1.0 + np.tanh((f_smooth - params.interior_flux_fraction) / 0.01))
    f_int_max_expected = float(np.max(f_smooth[1:-1]))
    w_active_expected = 0.5 * (
        1.0 + np.tanh((f_int_max_expected - params.interior_flux_fraction) / 0.01)
    )
    r_i = float(st_smooth['r_i'])
    r_mid = 0.5 * (r_i + r[0])
    dr_scale = max(float(np.mean(np.abs(r[1:] - r[:-1]))), 1.0)
    w_above_mid = 0.5 * (1.0 + np.tanh((r - r_mid) / dr_scale))
    w_below_top = 0.5 * (1.0 + np.tanh((r_i - r) / dr_scale))
    w_upper_expected = w_above_mid * w_below_top * w_conv_expected
    assert float(st_smooth['w_active']) == pytest.approx(float(w_active_expected), rel=1.0e-12)
    assert np.allclose(st_smooth['w_upper'], w_upper_expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.unit
def test_w_conv_fixed_width_pins_w_active_and_w_upper():
    """Verify w_active and w_upper match the width-0.01 w_conv formula on a smooth f profile."""
    _assert_w_conv_fixed_width_pins_w_active_and_w_upper(_make_earth_reference_inputs())


def _assert_hot_convecting_cmb_node_excluded(
    g_hom: float, t_m: float = 1600.0, t_s: float = 300.0
) -> None:
    """Verify a hot convecting CMB node r[0] at 4000 K with f=1 moves T_i by < 1 K."""
    from math import erf

    n_nodes = 25
    r = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    z = r[-1] - r
    delta = 1800.0e3
    t_base = t_s + (t_m - t_s) * np.array([erf(x) for x in z / delta])
    t_base[1:-5] = t_m
    k_cond = 4.0
    f_cond = (
        k_cond * (t_m - t_s) * (2.0 / (np.sqrt(np.pi) * delta)) * np.exp(-((z / delta) ** 2))
    )
    f_tot = np.full(n_nodes, f_cond[-1])
    f_conv = f_tot - f_cond
    f_conv[0] = f_tot[0]
    p = np.linspace(135.0e9, 1.0e5, n_nodes)
    params = SolidRheologyParams(
        enabled=True,
        g_homologous=g_hom,
        t_sol_0=t_m,
        dt_sol_dp=0.0,
        arrhenius_t_ref=t_m,
        interior_flux_fraction=0.05,
    )
    t_hot_cmb = t_base.copy()
    t_hot_cmb[0] = 4000.0
    t_i = [
        float(
            compute_stagnant_lid_state(
                radii=r,
                temperature=t_prof,
                pressure=p,
                convective_flux=f_conv,
                total_flux=f_tot,
                solidus_temperature=None,
                melt_fraction=np.zeros(n_nodes),
                params=params,
                viscosity_solid=1.0e20,
                phi_rheo=0.4,
                phi_width=0.15,
            )['T_i']
        )
        for t_prof in (t_base, t_hot_cmb)
    ]
    assert abs(t_i[1] - t_i[0]) <= 1.0


@pytest.mark.unit
def test_hot_convecting_cmb_node_excluded_from_ti():
    """Verify boundary node r[0] at 4000 K with f=1 is excluded from T_i weights."""
    g_hom = 300.0e3 / (8.314462618 * 1600.0)
    _assert_hot_convecting_cmb_node_excluded(g_hom)


@pytest.mark.unit
def test_erf_boundary_layer_ti_and_175kyr_default_ti():
    """Verify T_i on an erf boundary layer across N in {100, 200, 400, 800} and on the 175 kyr fixture."""
    import os
    from math import erf

    with pytest.raises(ValueError, match='interior_flux_fraction'):
        SolidRheologyParams(interior_flux_fraction=0.04)
    with pytest.raises(ValueError, match='interior_flux_fraction'):
        SolidRheologyParams(interior_flux_fraction=0.51)
    assert SolidRheologyParams(
        interior_flux_fraction=0.05
    ).interior_flux_fraction == pytest.approx(0.05)
    assert SolidRheologyParams(
        interior_flux_fraction=0.50
    ).interior_flux_fraction == pytest.approx(0.50)

    r_gas = 8.314462618
    t_m, t_s = 1600.0, 300.0
    h_act = 300.0e3
    g_hom = h_act / (r_gas * t_m)
    dt_rh_true = (r_gas * t_m**2) / h_act

    for delta in (100.0e3, 200.0e3):
        ti_by_n: dict[int, float] = {}
        for n_nodes in (100, 200, 400, 800):
            r = np.linspace(3480.0e3, 6370.0e3, n_nodes)
            z = r[-1] - r
            t = t_s + (t_m - t_s) * np.array([erf(x) for x in z / delta])
            k_cond = 4.0
            f_cond = (
                k_cond
                * (t_m - t_s)
                * (2.0 / (np.sqrt(np.pi) * delta))
                * np.exp(-((z / delta) ** 2))
            )
            f_tot = np.full(n_nodes, f_cond[-1])
            f_conv = f_tot - f_cond
            p = np.linspace(135.0e9, 1.0e5, n_nodes)
            params = SolidRheologyParams(
                enabled=True,
                g_homologous=g_hom,
                t_sol_0=t_m,
                dt_sol_dp=0.0,
                arrhenius_t_ref=t_m,
                interior_flux_fraction=0.05,
            )
            st = compute_stagnant_lid_state(
                radii=r,
                temperature=t,
                pressure=p,
                convective_flux=f_conv,
                total_flux=f_tot,
                solidus_temperature=None,
                melt_fraction=np.zeros(n_nodes),
                params=params,
                viscosity_solid=1.0e20,
                phi_rheo=0.4,
                phi_width=0.15,
            )
            ti_val = float(st['T_i'])
            ti_by_n[n_nodes] = ti_val
            assert abs(ti_val - t_m) <= 2.0 * dt_rh_true
        assert abs(ti_by_n[800] - ti_by_n[100]) <= 0.5 * dt_rh_true

    _assert_hot_convecting_cmb_node_excluded(g_hom, t_m=t_m, t_s=t_s)

    # Evaluates under linear solidus fallback (table-free) with solidus_temperature=None.
    ckpt_path = os.path.join(
        os.path.dirname(__file__), 'reference', 'checkpoint_profile_175kyr.npz'
    )
    data = np.load(ckpt_path)
    f_ckpt = np.clip(
        np.maximum(data['F_conv'], 0.0) / np.maximum(np.abs(data['F_tot']), 1.0e-12),
        0.0,
        1.0,
    )
    idx_f09 = int(np.where(f_ckpt[:-1] >= 0.9)[0][-1])
    t_at_f09 = float(data['T_basic'][idx_f09])
    st_ckpt = compute_stagnant_lid_state(
        radii=data['r_basic'],
        temperature=data['T_basic'],
        pressure=data['P_basic'],
        convective_flux=data['F_conv'],
        total_flux=data['F_tot'],
        solidus_temperature=None,
        melt_fraction=data['phi_basic'],
        params=SolidRheologyParams(enabled=True),
        unyielded_velocity=data['v_unyielded'],
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    assert abs(float(st_ckpt['T_i']) - t_at_f09) <= 1.0
    assert float(st_ckpt['T_i']) == pytest.approx(1423.012466223234, rel=1.0e-9)
    assert float(st_ckpt['tau_d']) == pytest.approx(1299378.5915329107, rel=1.0e-9)
    assert float(st_ckpt['eta_i']) == pytest.approx(2.760821023551699e22, rel=1.0e-9)


def _eval_adiabat_lid_profile(f_conv_arr: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate compute_stagnant_lid_state on a 50-node 1600 K interior adiabat under a 7-node lid."""
    n_nodes = 50
    r = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    p = np.linspace(135.0e9, 1.0e5, n_nodes)
    t_adiabat = np.empty(n_nodes)
    t_adiabat[:43] = np.linspace(2200.0, 1600.0, 43)
    t_adiabat[43:] = np.linspace(1500.0, 300.0, 7)
    visc_mix = np.full(n_nodes, 1.0e20)
    visc_mix[43:] = 1.0e40
    g_hom = 300.0e3 / (8.314462618 * 1600.0)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        g_homologous=g_hom,
        t_sol_0=1600.0,
        dt_sol_dp=0.0,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
    )
    st = compute_stagnant_lid_state(
        radii=r,
        temperature=t_adiabat,
        pressure=p,
        convective_flux=f_conv_arr * 100.0,
        total_flux=np.full(n_nodes, 100.0),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_nodes),
        params=params,
        viscosity_solid=1.0e20,
        viscosity_mixture=visc_mix,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    return st, t_adiabat


@pytest.mark.unit
def test_relative_ti_threshold_and_no_convection_fallback():
    """Verify T_i relative threshold (0.9 * max interior f) and no-convection fallback."""
    st_earth = compute_stagnant_lid_state(**_make_earth_reference_inputs())
    assert float(st_earth['T_i']) == pytest.approx(1600.0, rel=1.0e-9)

    for f_val in (1.0, 0.85, 0.70):
        f_arr = np.zeros(50)
        f_arr[:43] = f_val
        st_f, t_ad = _eval_adiabat_lid_profile(f_arr)
        assert abs(float(st_f['T_i']) - float(t_ad[42])) <= float(st_f['dT_rh'])
        assert float(st_f['mu_i']) == pytest.approx(1.0e20, rel=1.0e-6)

    f_mid = np.zeros(50)
    f_mid[25] = 0.10
    st_mid, t_ad = _eval_adiabat_lid_profile(f_mid)
    assert float(st_mid['T_i']) == pytest.approx(float(t_ad[25]), abs=1.0)

    f_surf_only = np.zeros(50)
    f_surf_only[-1] = 1.0
    st_surf, _ = _eval_adiabat_lid_profile(f_surf_only)
    assert float(st_surf['T_i']) == pytest.approx(float(t_ad[1]), rel=1.0e-6)

    st_no_conv, _ = _eval_adiabat_lid_profile(np.zeros(50))
    assert float(st_no_conv['T_i']) == pytest.approx(float(t_ad[1]), rel=1.0e-6)
    assert abs(float(st_no_conv['T_i']) - float(np.mean(t_ad))) > 100.0


@pytest.mark.unit
def test_ti_continuity_across_flux_threshold():
    """Verify T_i is continuous across interior f = 0.05 for surface f = 0.0 and 0.08."""
    # Fixed rheological temperature scale at T_i = 1600 K: a_rh_fb * T_i / g_hom = 92.235105 K.
    dt_rh_fixed = 1.3 * 1600.0 / (300.0e3 / (8.314462618 * 1600.0))
    for f_surf in (0.0, 0.08):
        f_lo = np.zeros(50)
        f_lo[:43] = 0.0499
        f_lo[-1] = f_surf
        f_hi = np.zeros(50)
        f_hi[:43] = 0.0501
        f_hi[-1] = f_surf
        st_lo, t_ad = _eval_adiabat_lid_profile(f_lo)
        st_hi, _ = _eval_adiabat_lid_profile(f_hi)
        assert abs(float(st_hi['T_i']) - float(st_lo['T_i'])) <= 5.0
        assert abs(float(st_lo['T_i']) - float(t_ad[42])) <= dt_rh_fixed

        f_col_lo = np.zeros(50)
        f_col_lo[1:-1] = 0.0499
        f_col_lo[0] = 0.0499
        f_col_lo[-1] = f_surf
        f_col_hi = np.zeros(50)
        f_col_hi[1:-1] = 0.0501
        f_col_hi[0] = 0.0501
        f_col_hi[-1] = f_surf
        st_col_lo, _ = _eval_adiabat_lid_profile(f_col_lo)
        st_col_hi, _ = _eval_adiabat_lid_profile(f_col_hi)
        assert abs(float(st_col_hi['T_i']) - float(st_col_lo['T_i'])) <= 15.0


@pytest.mark.unit
def test_ti_linear_flux_profile_scale_invariance():
    """Verify linear f profile scaled by 0.06, 0.1, 0.2, 1.0 gives T_i within 1 dT_rh."""
    dt_rh_fixed = 1.3 * 1600.0 / (300.0e3 / (8.314462618 * 1600.0))
    f_ref = np.zeros(50)
    f_ref[:43] = np.linspace(1.0, 0.5, 43)
    st_ref, _ = _eval_adiabat_lid_profile(f_ref)
    ti_ref = float(st_ref['T_i'])

    for pk in (0.06, 0.1, 0.2, 1.0):
        f_pk = np.zeros(50)
        f_pk[:43] = np.linspace(pk, 0.5 * pk, 43)
        st_pk, _ = _eval_adiabat_lid_profile(f_pk)
        assert abs(float(st_pk['T_i']) - ti_ref) <= dt_rh_fixed
        assert abs(float(st_pk['T_i']) - ti_ref) <= float(st_ref['dT_rh'])


@pytest.mark.unit
def test_w_active_mesh_independence_and_low_flux_solver_continuity():
    """Verify mesh-independent w_active at f=0.03 and f=0.01 and continuity across f in 0..1e-3."""
    from aragog.rheology_lid import compute_effective_viscosity

    g_hom = 300.0e3 / (8.314462618 * 1600.0)
    params = SolidRheologyParams(
        enabled=True,
        g_homologous=g_hom,
        t_sol_0=1600.0,
        dt_sol_dp=0.0,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
    )
    w_act_by_n_003: list[float] = []
    for n_nodes in (50, 100, 200, 400):
        r = np.linspace(3480.0e3, 6370.0e3, n_nodes)
        p = np.linspace(135.0e9, 1.0e5, n_nodes)
        t = np.linspace(2200.0, 300.0, n_nodes)
        for f_val, expected_w in ((0.03, 0.0179862), (0.01, 3.3535e-4)):
            st = compute_stagnant_lid_state(
                radii=r,
                temperature=t,
                pressure=p,
                convective_flux=np.full(n_nodes, f_val * 100.0),
                total_flux=np.full(n_nodes, 100.0),
                solidus_temperature=None,
                melt_fraction=np.zeros(n_nodes),
                params=params,
                viscosity_solid=1.0e20,
                phi_rheo=0.4,
                phi_width=0.15,
            )
            rel_tol = 1.0e-6 if f_val == 0.03 else 1.0e-4
            assert float(st['w_active']) == pytest.approx(expected_w, rel=rel_tol)
            assert bool(st['has_lid']) is False
            assert float(st['d_lid']) == 0.0
            assert float(st['tau_d']) == 0.0
            assert abs(float(st['T_i']) - float(t[0])) > 1.0
            if f_val == 0.03:
                w_act_by_n_003.append(float(st['w_active']))
    for w_val in w_act_by_n_003[1:]:
        assert w_val == pytest.approx(w_act_by_n_003[0], rel=1.0e-12)

    prev_st: dict[str, Any] | None = None
    prev_log_eta: np.ndarray | None = None
    eta_diff = np.full(50, 1.0e24)
    for fv in np.linspace(0.0, 1.0e-3, 11):
        f_arr = np.zeros(50)
        f_arr[:43] = fv
        st_step, _ = _eval_adiabat_lid_profile(f_arr)
        log_eta = np.log10(
            compute_effective_viscosity(
                eta_diff=eta_diff,
                eta_i=1.0e20,
                tau_d=2.0e6,
                tau_y_lid=1.0e6,
                w_lid=st_step['w_lid'],
                stress_closure_mode='lid',
            )
        )
        if prev_st is not None and prev_log_eta is not None:
            for k in ('tau_d', 'w_y', 'lid_regime'):
                assert abs(float(st_step[k]) - float(prev_st[k])) <= 1.0e-3
            assert bool(st_step['has_lid']) == bool(prev_st['has_lid'])
            assert float(np.max(np.abs(st_step['w_lid'] - prev_st['w_lid']))) <= 1.0e-3
            assert float(np.max(np.abs(log_eta - prev_log_eta))) / 4.30103 <= 1.0e-3
        prev_st = st_step
        prev_log_eta = log_eta


@pytest.mark.unit
def test_fb2014_v_i_unyielded_velocity_and_w_active_scaling():
    """Verify v_i unyielded_velocity LogSumExp selector and scaling with w_active."""
    inputs = _make_earth_reference_inputs()
    n_nodes = len(inputs['radii'])
    # Linear unyielded_velocity across geological velocity scale [1e-10, 1e-9] m/s.
    inputs['unyielded_velocity'] = np.linspace(1.0e-10, 1.0e-9, n_nodes)
    st = compute_stagnant_lid_state(**inputs)

    # Formula derivation: upper-mantle LogSumExp selector between r_mid and r_i with T_v = 3.17e-12 m/s:
    r = inputs['radii']
    dr_scale = np.maximum(np.mean(np.abs(r[1:] - r[:-1])), 1.0)
    r_i = float(st['r_i'])
    r_mid = 0.5 * (r_i + r[0])
    w_conv = 0.5 * (
        1.0 + np.tanh((inputs['convective_flux'] / inputs['total_flux'] - 0.05) / 0.01)
    )
    w_above_mid = 0.5 * (1.0 + np.tanh((r - r_mid) / dr_scale))
    w_below_top = 0.5 * (1.0 + np.tanh((r_i - r) / dr_scale))
    w_upper = w_above_mid * w_below_top * w_conv
    n_upper = np.sum(w_upper) + 1e-12
    t_v = 3.17e-12
    v_abs = np.abs(inputs['unyielded_velocity'])
    u = w_upper * v_abs
    u_shift = np.max(u)
    sum_w_exp = np.sum(w_upper * np.exp((u - u_shift) / t_v))
    expected_v_i = float(
        u_shift + t_v * (np.log(np.maximum(sum_w_exp, 1e-300)) - np.log(n_upper))
    )

    assert float(st['v_i']) == pytest.approx(expected_v_i, rel=1.0e-9)
    # 6223e3f regression pin: on 50 nodes with linear unyielded velocity, v_i ~ 7.925e-10 m/s.
    assert float(st['v_i']) == pytest.approx(7.9253365e-10, rel=1.0e-6)
    # Formula relation: v_i is bounded between unyielded_velocity at r_mid and r_i.
    idx_mid = len(r) // 2
    assert (
        inputs['unyielded_velocity'][idx_mid]
        < float(st['v_i'])
        <= inputs['unyielded_velocity'][-1]
    )

    # Weak convection where w_active < 1: v_i must scale with w_active.
    inputs_weak = _make_earth_reference_inputs()
    inputs_weak['convective_flux'] = np.zeros(n_nodes)
    inputs_weak['convective_flux'][20] = 5.5
    inputs_weak['unyielded_velocity'] = np.full(n_nodes, 1.0e-3)
    st_weak = compute_stagnant_lid_state(**inputs_weak)

    # Formula derivation: w_active = 0.5 * (1 + tanh((f_int_max - 0.05) / 0.01)) and v_i = v_i_raw * w_active.
    f_weak = inputs_weak['convective_flux'] / inputs_weak['total_flux']
    w_conv_weak = 0.5 * (1.0 + np.tanh((f_weak - 0.05) / 0.01))
    f_int_max_weak = float(np.max(f_weak[1:-1]))
    expected_w_active = float(0.5 * (1.0 + np.tanh((f_int_max_weak - 0.05) / 0.01)))
    r_i_weak = float(st_weak['r_i'])
    r_mid_weak = 0.5 * (r_i_weak + r[0])
    w_above_mid_weak = 0.5 * (1.0 + np.tanh((r - r_mid_weak) / dr_scale))
    w_below_top_weak = 0.5 * (1.0 + np.tanh((r_i_weak - r) / dr_scale))
    w_upper_weak = w_above_mid_weak * w_below_top_weak * w_conv_weak
    n_upper_weak = np.sum(w_upper_weak) + 1e-12
    u_weak = w_upper_weak * 1.0e-3
    u_shift_weak = np.max(u_weak)
    sum_w_exp_weak = np.sum(w_upper_weak * np.exp((u_weak - u_shift_weak) / t_v))
    v_i_raw_weak = u_shift_weak + t_v * (
        np.log(np.maximum(sum_w_exp_weak, 1e-300)) - np.log(n_upper_weak)
    )
    expected_v_i_weak = float(v_i_raw_weak * expected_w_active)

    assert float(st_weak['w_active']) == pytest.approx(expected_w_active, rel=1.0e-9)
    assert float(st_weak['v_i']) == pytest.approx(expected_v_i_weak, rel=1.0e-9)
    # Regression pin: weak convection at node 20 (f_int_max = 0.055) gives w_active ~ 0.731059 and v_i ~ 2.672e-4 m/s.
    assert float(st_weak['w_active']) == pytest.approx(0.731059, rel=1.0e-4)
    assert float(st_weak['v_i']) == pytest.approx(2.672233e-04, rel=1.0e-6)
    # Formula relation: weak convection scales v_i below full unyielded velocity 1e-3 m/s by w_active.
    assert float(st_weak['v_i']) <= expected_w_active * 1.0e-3


@pytest.mark.unit
def test_lid_mask_width_cells_sensitivity():
    """Verify lid_mask_width_cells scales transition width and changes lid thickness."""
    inputs1 = _make_earth_reference_inputs()
    st1 = compute_stagnant_lid_state(**inputs1)

    inputs2 = _make_earth_reference_inputs()
    inputs2['params'] = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        g_homologous=inputs1['params'].g_homologous,
        t_sol_0=1600.0,
        dt_sol_dp=0.0,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
        lid_contrast_coeff=2.2,
        lid_mask_width_cells=2.0,
    )
    st2 = compute_stagnant_lid_state(**inputs2)
    # Formula relation: d_lid = r[-1] - r_lid_base, and doubling cell width widens transition, narrowing lid.
    assert float(st2['d_lid']) == pytest.approx(
        float(inputs2['radii'][-1] - st2['r_lid_base']), rel=1.0e-9
    )
    assert float(st2['d_lid']) < float(st1['d_lid'])
    assert abs(float(st2['d_lid']) - float(st1['d_lid'])) > 3.0e4
    # 6223e3f regression pin: doubling width shifts r_lid_base down by ~15.6 km, d_lid to ~341.1 km.
    assert float(st2['d_lid']) == pytest.approx(341105.3726, rel=1.0e-6)


@pytest.mark.unit
def test_isothermal_column_delta_rh_capped_at_mantle_thickness():
    """Verify delta_rh is capped at total mantle thickness on an isothermal column."""
    inputs = _make_earth_reference_inputs()
    inputs['temperature'] = np.full(len(inputs['radii']), 1600.0)
    st = compute_stagnant_lid_state(**inputs)
    d_conv = inputs['radii'][-1] - inputs['radii'][0]
    # Without the min(delta_rh, r[-1] - r[0]) cap, dT_rh / dTdr_lid_base exceeds d_conv.
    assert float(st['delta_rh']) == pytest.approx(d_conv, rel=1.0e-12)


@pytest.mark.unit
def test_non_uniform_profile_weighting_pins_ra_eff():
    """Verify _weighted_mean weights near-lid properties instead of column average."""
    inputs = _make_earth_reference_inputs()
    n_nodes = len(inputs['radii'])
    d_conv = inputs['radii'][-1] - inputs['radii'][0]
    delta_t = 1600.0 - 300.0
    alpha, kappa, eta_i = 3.0e-5, 1.0e-6, 1.0e20

    inputs['gravity'] = np.linspace(10.0, 4.0, n_nodes)
    st_g = compute_stagnant_lid_state(**inputs)

    # Formula relation: Ra_eff = (rho * g_i * alpha * DeltaT * d_conv^3) / (kappa * eta_i).
    # Real weighted mean evaluates near r_i (g ~ 4.7 m/s^2), yielding Ra_eff ~ 1.547e8.
    # Column mean g = 7.0 m/s^2 would yield 2.279e8 (+47%), while surface g = 4.0 m/s^2 yields 1.302e8 (-16%).
    ra_mean_g = (3300.0 * np.mean(inputs['gravity']) * alpha * delta_t * d_conv**3) / (
        kappa * eta_i
    )
    ra_surf_g = (3300.0 * inputs['gravity'][-1] * alpha * delta_t * d_conv**3) / (kappa * eta_i)
    assert ra_surf_g < float(st_g['Ra_eff']) < ra_mean_g
    assert abs(float(st_g['Ra_eff']) - ra_mean_g) / ra_mean_g > 0.25
    # 6223e3f regression pin: near-lid weighted g yields Ra_eff ~ 1.5469e8.
    assert float(st_g['Ra_eff']) == pytest.approx(154691448.57, rel=1.0e-4)

    inputs['density'] = np.linspace(4000.0, 3000.0, n_nodes)
    st_grho = compute_stagnant_lid_state(**inputs)
    # Near-lid density rho_i evaluated near r_i (rho ~ 3100 kg/m^3) bounds Ra_eff below column mean rho = 3500.
    ra_mean_rho = (
        np.mean(inputs['density'])
        * float(st_g['Ra_eff'] * kappa * eta_i / (3300.0 * alpha * delta_t * d_conv**3))
        * alpha
        * delta_t
        * d_conv**3
    ) / (kappa * eta_i)
    assert float(st_grho['Ra_eff']) < ra_mean_rho
    # 6223e3f regression pin: near-lid weighted rho and g yield Ra_eff ~ 1.4828e8.
    assert float(st_grho['Ra_eff']) == pytest.approx(148281889.62, rel=1.0e-4)


@pytest.mark.unit
def test_depth_varying_solidus_pins_theta_and_eta_i():
    """Verify depth-varying solidus table weights T_sol at r_i rather than boundary nodes."""
    from dataclasses import replace

    inputs = _make_earth_reference_inputs()
    n_nodes = len(inputs['radii'])
    inputs['solidus_temperature'] = np.linspace(2200.0, 1400.0, n_nodes)
    inputs['t_sol_0'] = 1400.0
    inputs['params'] = replace(inputs['params'], t_sol_0=1400.0)
    st = compute_stagnant_lid_state(**inputs)

    # Formula relation: theta = g_hom * T_sol_i * (T_i - T_surf) / T_i^2, where T_sol_i is weighted at r_i.
    t_sol_cmb = inputs['solidus_temperature'][0]
    t_sol_surf = inputs['solidus_temperature'][-1]
    g_hom = inputs['params'].g_homologous
    t_i = float(st['T_i'])
    theta_cmb = g_hom * t_sol_cmb * (t_i - 300.0) / (t_i**2)
    theta_surf = g_hom * t_sol_surf * (t_i - 300.0) / (t_i**2)
    assert theta_surf < float(st['theta']) < theta_cmb
    assert abs(float(st['theta']) - theta_cmb) > 5.0
    # 6223e3f regression pin: near-lid T_sol_i ~ 1528 K yields theta ~ 17.528.
    assert float(st['theta']) == pytest.approx(17.528141, rel=1.0e-4)

    # Formula relation: T_lid lies in the boundary layer between T_lid_iso and T_i.
    assert float(st['T_lid_iso']) < float(st['T_lid']) < t_i
    # 6223e3f regression pin: lid boundary weights yield T_lid ~ 1482.7 K.
    assert float(st['T_lid']) == pytest.approx(1482.7168, rel=1.0e-4)


@pytest.mark.unit
def test_surface_node_excluded_from_interior_ti_selector():
    """Verify surface node is excluded from interior_node_mask when f_surf exceeds interior."""
    inputs = _make_earth_reference_inputs()
    n_nodes = len(inputs['radii'])
    f_conv = np.zeros(n_nodes)
    f_conv[:42] = 50.0  # interior convective fraction f = 0.5
    f_conv[-1] = 100.0  # surface boundary node f = 1.0
    inputs['convective_flux'] = f_conv
    inputs['total_flux'] = np.full(n_nodes, 100.0)
    st = compute_stagnant_lid_state(**inputs)

    # Excluding surface node keeps T_i at interior value (~1600 K), avoiding surface 300 K.
    assert abs(float(st['T_i']) - 1600.0) <= float(st['dT_rh'])
    # Formula relation: tau_d = 2 * eta_i * v_m / d_conv.
    d_conv = float(inputs['radii'][-1] - inputs['radii'][0])
    expected_tau_d = 2.0 * float(st['eta_i']) * float(st['v_m']) / d_conv
    assert float(st['tau_d']) == pytest.approx(expected_tau_d, rel=1.0e-6)
    # 6223e3f regression pin: interior convective driving stress is ~232.3 kPa (surface 300 K would give ~1e11 Pa).
    assert float(st['tau_d']) == pytest.approx(232313.068898, rel=1.0e-4)
    assert float(st['tau_d']) < 1.0e6


@pytest.mark.unit
def test_depth_varying_solidus_surface_temperature_pins_eta_contrast():
    """Verify surface viscosity contrast uses surface solidus T_sol[-1], not CMB T_sol[0]."""
    from dataclasses import replace

    inputs = _make_earth_reference_inputs()
    n_nodes = len(inputs['radii'])
    inputs['solidus_temperature'] = np.linspace(2200.0, 1400.0, n_nodes)
    inputs['t_sol_0'] = 1400.0
    inputs['params'] = replace(inputs['params'], t_sol_0=1400.0)
    # Warm surface temperature 1300 K keeps surface viscosity unclipped below 10^40.
    inputs['temperature'][-8:] = np.linspace(1600.0, 1300.0, 8)
    st = compute_stagnant_lid_state(**inputs)

    # Formula derivation: eta_surf = arrhenius(T_surf, P_surf, solidus_temperature=T_sol[-1])
    # T_sol[-1] = 1400 K gives H ~ 291 kJ/mol and eta_contrast ~ 15.1.
    # Selecting CMB solidus T_sol[0] = 2200 K gives H ~ 457 kJ/mol and eta_contrast ~ 269.7.
    t_sol_surf = float(inputs['solidus_temperature'][-1])
    h_surf = inputs['params'].g_homologous * 8.314462618 * t_sol_surf
    eta_surf_expected = 1.0e20 * np.exp((h_surf / 8.314462618) * (1.0 / 1300.0 - 1.0 / 1600.0))
    expected_contrast = eta_surf_expected / float(st['eta_i'])

    assert float(st['eta_contrast']) == pytest.approx(expected_contrast, rel=1.0e-6)
    assert float(st['eta_contrast']) < 50.0
    # 6223e3f regression pin: unclipped surface viscosity yields eta_contrast ~ 15.07.
    assert float(st['eta_contrast']) == pytest.approx(15.069411, rel=1.0e-4)
