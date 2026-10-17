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
    # Zero-flux lid nodes retain weight 0.5 * (1 + tanh(-0.05 / 0.01)) = 4.54e-5 under fixed width 0.01.
    assert abs(t_i - 1600.0) < 0.5

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
    expected_delta_rh = 30791.06
    expected_tau_buoy = rho * g * alpha * expected_dt_rh * expected_delta_rh
    expected_ratio = expected_tau_d / expected_tau_buoy

    assert state['theta'] == pytest.approx(expected_theta, rel=1.0e-5)
    assert state['dT_rh'] == pytest.approx(expected_dt_rh, rel=1.0e-5)
    assert state['Ra_eff'] == pytest.approx(expected_ra_eff, rel=1.0e-3)
    assert state['Ra_rh'] == pytest.approx(expected_ra_rh, rel=1.0e-3)
    assert state['delta_rh'] == pytest.approx(expected_delta_rh, rel=1.0e-3)
    assert state['tau_d'] == pytest.approx(expected_tau_d, rel=1.0e-3)
    assert state['v_m'] == pytest.approx(expected_vm, rel=1.0e-3)
    assert state['tau_buoy'] == pytest.approx(expected_tau_buoy, rel=1.0e-3)
    assert state['tau_d_over_tau_buoy'] == pytest.approx(expected_ratio, rel=1.0e-3)


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
    g_hom_earth = 300.0e3 / (8.31446261815324 * 1600.0)
    kwargs['params'] = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        g_homologous=g_hom_earth,
        t_sol_0=1600.0,
        dt_sol_dp=0.0,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
        lid_contrast_coeff=2.2,
    )
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
            radii=r,
            temperature=t,
            pressure=p,
            convective_flux=f_conv,
            total_flux=f_tot,
            solidus_temperature=None,
            melt_fraction=phi,
            params=kwargs['params'],
            viscosity_solid=1.0e20,
            phi_rheo=0.4,
            phi_width=0.15,
            density=3300.0,
            gravity=9.81,
            thermal_expansivity=3.0e-5,
            thermal_diffusivity=1.0e-6,
            xp=np,
        )

    st_200 = _eval_at_n(200)
    st_400 = _eval_at_n(400)
    rel_diff_ti = abs(float(st_400['T_i']) - float(st_200['T_i'])) / float(st_200['T_i'])
    rel_diff_taud = abs(float(st_400['tau_d']) - float(st_200['tau_d'])) / float(
        st_200['tau_d']
    )
    assert rel_diff_ti < 0.02
    assert rel_diff_taud < 0.02
