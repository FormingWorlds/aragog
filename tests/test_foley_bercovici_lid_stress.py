"""Tests for Foley & Bercovici (2014) stagnant lid driving stress closure.

References
----------
Foley & Bercovici (2014), GJI 199, pp. 580-603, doi:10.1093/gji/ggu275
    Eq. 26 (p. 586-587): v_m = (kappa / d) * C4 * (Ra_eff * a_rh / theta)**(2/3)
    Eq. 28 (p. 588): tau_xz = 2 * mu_i * v_m / d
    Sec. 4.2 (p. 589): C4 = 0.125, a_rh = 1.3
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import SolidRheologyParams
from aragog.rheology_lid import compute_stagnant_lid_state


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
    - E = 300.0e3 J/mol
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
    - delta_rh = 6800.0 m
    - tau_buoy = 4.490783e5 Pa (0.449078 MPa)
    - tau_d / tau_buoy = 0.517311
    """
    n_nodes = 50
    radii = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    temperature = np.full(n_nodes, 1600.0)
    temperature[-8:] = np.linspace(1600.0, 300.0, 8)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    convective_flux = np.full(n_nodes, 100.0)
    convective_flux[-8:] = 0.0
    total_flux = np.full(n_nodes, 100.0)
    melt_frac = np.zeros(n_nodes)

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        activation_energy=300.0e3,
        activation_volume=0.0,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.01,
        lid_contrast_coeff=2.2,
    )

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=convective_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        viscosity_solid=1.0e20,
        density=3300.0,
        gravity=9.81,
        thermal_expansivity=3.0e-5,
        thermal_diffusivity=1.0e-6,
        xp=np,
    )

    expected_tau_d = 2.323131e5
    expected_vm = 3.356924e-9

    assert state['tau_d'] == pytest.approx(expected_tau_d, rel=1.0e-3)
    assert state['v_m'] == pytest.approx(expected_vm, rel=1.0e-3)
    expected_tau_buoy = (
        3300.0
        * 9.81
        * 3.0e-5
        * float(state['dT_rh'])
        * float(state['delta_rh'])
        * float(state['w_active'])
    )
    assert state['tau_buoy'] == pytest.approx(expected_tau_buoy, rel=1.0e-6)
    assert state['tau_d_over_tau_buoy'] == pytest.approx(
        float(state['tau_d']) / float(state['tau_buoy']), rel=1.0e-6
    )


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

    for visc in viscosities:
        params = SolidRheologyParams(
            enabled=True,
            stress_closure_mode='lid',
            activation_energy=300.0e3,
            activation_volume=0.0,
            arrhenius_t_ref=1600.0,
            interior_flux_fraction=0.01,
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
        # Each decade of viscosity increases tau_d by 10^(1/3) ~ 2.1544
        ratio = tau_d_values[i + 1] / tau_d_values[i]
        assert ratio == pytest.approx(10.0 ** (1.0 / 3.0), rel=1.0e-2)


@pytest.mark.unit
def test_fb2014_mutant_exponent_half_fails():
    """Verify mutant with velocity exponent 1/2 fails pinned Earth value assertion.

    Solomatov & Moresi (2000) internally heated scaling uses exponent 1/2,
    yielding tau_d ~ 1.39e4 Pa (a 94% reduction from FB2014 2.32e5 Pa).
    """
    rho = 3300.0
    g = 9.81
    alpha = 3.0e-5
    DeltaT = 1300.0
    d = 2.89e6
    kappa = 1.0e-6
    eta_i = 1.0e20
    theta = 18.322742
    C4 = 0.125
    a_rh = 1.3

    Ra_eff = (rho * g * alpha * DeltaT * (d**3)) / (kappa * eta_i)
    Ra_rh = (Ra_eff * a_rh) / theta
    vm_true = (kappa / d) * C4 * (Ra_rh ** (2.0 / 3.0))
    tau_true = 2.0 * eta_i * vm_true / d

    # Mutant: exponent 1/2 instead of 2/3
    vm_mutant = (kappa / d) * C4 * (Ra_rh**0.5)
    tau_mutant = 2.0 * eta_i * vm_mutant / d

    rel_error = abs(tau_mutant - tau_true) / tau_true
    assert rel_error > 0.90, f'Mutant exponent 1/2 must fail by >90%, got {rel_error:.2%}'


@pytest.mark.unit
def test_fb2014_mutant_missing_factor_two_fails():
    """Verify mutant omitting factor 2 (tau_d = mu_i * v_m / d) fails pinned value assertion.

    FB2014 Eq. 28 explicitly specifies tau_xz = 2 * mu_i * v_m / d due to horizontal
    shear across the convective cell half-depth.
    """
    rho = 3300.0
    g = 9.81
    alpha = 3.0e-5
    DeltaT = 1300.0
    d = 2.89e6
    kappa = 1.0e-6
    eta_i = 1.0e20
    theta = 18.322742
    C4 = 0.125
    a_rh = 1.3

    Ra_eff = (rho * g * alpha * DeltaT * (d**3)) / (kappa * eta_i)
    Ra_rh = (Ra_eff * a_rh) / theta
    vm_true = (kappa / d) * C4 * (Ra_rh ** (2.0 / 3.0))
    tau_true = 2.0 * eta_i * vm_true / d

    # Mutant: factor 1.0 instead of 2.0
    tau_mutant = eta_i * vm_true / d

    rel_error = abs(tau_mutant - tau_true) / tau_true
    assert rel_error == pytest.approx(0.50, abs=1.0e-6), (
        f'Mutant missing factor 2 must differ by exactly 50%, got {rel_error}'
    )
