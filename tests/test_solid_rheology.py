"""Unit tests for solid-state mantle rheology formulations.

Validates Arrhenius temperature- and pressure-dependent viscosity (eta_diff),
pseudoplastic Byerlee yielding and effective viscosity capping (eta_eff),
and 1D stress closure strain-rate formulations (local MLT and global BLT modes).
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import (
    compute_strain_rate_local,
    compute_yield_stress,
    eta_diff,
    eta_eff,
    stress_closure,
)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_arrhenius_viscosity_scaling():
    r"""Assert Arrhenius viscosity (eta_diff) scales correctly with T and P.

    Physics invariants:
    1. Monotonic temperature dependence: d(eta)/dT < 0 (higher T gives lower viscosity).
    2. Monotonic pressure dependence: d(eta)/dP > 0 (higher P gives higher viscosity).
    3. Exact reference condition pin: eta(T_ref, P=0) == eta_0.

    Reference:
    Karato & Wu (1993), Science 260, 771-778;
    Solomatov (1995), Phys. Fluids 7, 266-274.
    """
    eta_0 = 1.0e21
    e_a = 300.0e3
    v_a = 5.0e-6
    t_ref = 1600.0
    r_gas = 8.314

    # 1. Reference condition pin: T = T_ref, P = 0 Pa -> eta == eta_0
    eta_ref = eta_diff(
        temperature=t_ref,
        pressure=0.0,
        viscosity_solid=eta_0,
        activation_energy=e_a,
        activation_volume=v_a,
        t_ref=t_ref,
        r_gas=r_gas,
    )
    assert eta_ref == pytest.approx(eta_0, rel=1.0e-12)

    # 2. Temperature scaling at constant pressure (P = 10 GPa)
    p_const = 10.0e9
    t_array = np.linspace(1200.0, 2400.0, 25)
    eta_t = eta_diff(
        temperature=t_array,
        pressure=p_const,
        viscosity_solid=eta_0,
        activation_energy=e_a,
        activation_volume=v_a,
        t_ref=t_ref,
        r_gas=r_gas,
    )
    # Monotonicity check: strictly decreasing with T
    diffs_t = np.diff(eta_t)
    assert np.all(diffs_t < 0.0), 'Viscosity must strictly decrease with increasing T'
    # Quantitative ratio check between T=1400 K and T=1800 K
    t1, t2 = 1400.0, 1800.0
    eta_t1 = eta_diff(t1, p_const, eta_0, e_a, v_a, t_ref, r_gas)
    eta_t2 = eta_diff(t2, p_const, eta_0, e_a, v_a, t_ref, r_gas)
    expected_ratio = np.exp(((e_a + p_const * v_a) / r_gas) * (1.0 / t1 - 1.0 / t2))
    assert (eta_t1 / eta_t2) == pytest.approx(expected_ratio, rel=1.0e-6)
    # Discrimination guard: opposite temperature sign would invert ratio
    assert eta_t1 / eta_t2 > 1.0

    # 3. Pressure scaling at constant temperature (T = 1600 K)
    t_const = 1600.0
    p_array = np.linspace(0.0, 100.0e9, 25)
    eta_p = eta_diff(
        temperature=t_const,
        pressure=p_array,
        viscosity_solid=eta_0,
        activation_energy=e_a,
        activation_volume=v_a,
        t_ref=t_ref,
        r_gas=r_gas,
    )
    # Monotonicity check: strictly increasing with P
    diffs_p = np.diff(eta_p)
    assert np.all(diffs_p > 0.0), 'Viscosity must strictly increase with increasing P'
    # Quantitative ratio check between P=0 and P=20 GPa
    p1, p2 = 0.0, 20.0e9
    eta_p1 = eta_diff(t_const, p1, eta_0, e_a, v_a, t_ref, r_gas)
    eta_p2 = eta_diff(t_const, p2, eta_0, e_a, v_a, t_ref, r_gas)
    expected_p_ratio = np.exp((v_a / (r_gas * t_const)) * (p2 - p1))
    assert (eta_p2 / eta_p1) == pytest.approx(expected_p_ratio, rel=1.0e-6)
    assert eta_p2 > eta_p1

    # 4. Invariant: strictly positive everywhere
    assert np.all(eta_t > 0.0)
    assert np.all(eta_p > 0.0)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_effective_viscosity_yield_capping():
    r"""Assert effective viscosity caps at the yield limit at high strain rate.

    Physics invariants:
    1. Low strain-rate limit: eta_eff -> eta_diff (ductile regime).
    2. High strain-rate limit: eta_eff -> tau_y / (2 * strain_rate) (plastic regime).
    3. Deviatoric stress cap: tau = 2 * eta_eff * strain_rate <= tau_y everywhere.
    4. Capping inequality: eta_eff <= eta_diff everywhere.

    Reference:
    Moresi & Solomatov (1998), Geophys. J. Int. 133, 669-682;
    Tackley (2000), Science 288, 2002-2007.
    """
    # Mock parameters: stiff lid with high Arrhenius viscosity
    visc_diff = 1.0e26  # Pa s (cold lithosphere)
    p_litho = 1.0e8  # 100 MPa
    tau_y = compute_yield_stress(p_litho, yield_stress_c=50.0e6, yield_stress_mu=0.6)
    expected_tau_y = 50.0e6 + 0.6 * 1.0e8  # 110 MPa
    assert tau_y == pytest.approx(expected_tau_y, rel=1.0e-10)

    # 1. Low strain rate (1.0e-24 1/s) -> ductile regime, no yielding
    sr_low = 1.0e-24
    eta_eff_low = eta_eff(visc_diff, tau_y, sr_low, smooth=True)
    assert eta_eff_low == pytest.approx(visc_diff, rel=1.0e-4)
    assert eta_eff_low <= visc_diff

    # 2. Artificially high strain rate (1.0e-10 1/s to 1.0 1/s) -> plastic yield limit
    sr_high = 1.0e-10
    eta_yield_expected = tau_y / (2.0 * sr_high)  # 110e6 / (2e-10) = 5.5e17 Pa s
    eta_eff_high = eta_eff(visc_diff, tau_y, sr_high, smooth=True)

    # Must be heavily capped below ductile viscosity: eta_eff << visc_diff
    assert eta_eff_high < 1.0e19
    assert eta_eff_high < visc_diff * 1.0e-6
    # For smooth harmonic mean, eta_eff = eta_d * eta_y / (eta_d + eta_y)
    assert eta_eff_high == pytest.approx(eta_yield_expected, rel=1.0e-5)

    # 3. Stress cap invariant: 2 * eta_eff * strain_rate <= tau_y for all strain rates
    strain_rates = np.logspace(-22, 2, 100)
    eta_eff_arr = eta_eff(visc_diff, tau_y, strain_rates, smooth=True)
    stresses = 2.0 * eta_eff_arr * strain_rates

    # Include machine precision tolerance (1e-6 Pa on 110 MPa)
    assert np.all(stresses <= tau_y + 1.0e-6), (
        'Deviatoric stress must never exceed yield stress'
    )
    assert np.all(eta_eff_arr <= visc_diff), 'Effective viscosity must never exceed eta_diff'
    # At extreme strain rate, stress should approach tau_y asymptotically
    assert stresses[-1] == pytest.approx(tau_y, rel=1.0e-4)

    # 4. Sharp cutoff comparison (smooth=False)
    eta_sharp = eta_eff(visc_diff, tau_y, strain_rates, smooth=False)
    # Harmonic mean is strictly smaller than or equal to sharp min
    assert np.all(eta_eff_arr <= eta_sharp + 1.0e-10)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_stress_closure_modes():
    r"""Assert stress closure correctly handles local and global closure modes.

    Physics invariants:
    1. 'local' mode: strain rate scales inversely with mixing length and linearly with velocity.
    2. 'global' mode: strain rate scales as v_int / d_lid based on lithospheric lid depth.
    3. Input validation: raises ValueError on unknown mode or missing required mode arguments.

    Reference:
    Solomatov (2007), Treatise on Geophysics 9, 91-119;
    Bower et al. (2018), PEPI 274, 49-62.
    """
    # ── Test local mode ──
    v_local = 1.0e-8  # m/s
    l_mix = 1.0e5  # 100 km
    sr_local = stress_closure('local', viscous_velocity=v_local, mixing_length=l_mix)
    assert sr_local == pytest.approx(v_local / l_mix, rel=1.0e-12)

    # Scaling: doubling velocity doubles strain rate
    sr_double_v = stress_closure('local', viscous_velocity=2.0 * v_local, mixing_length=l_mix)
    assert sr_double_v == pytest.approx(2.0 * sr_local, rel=1.0e-12)

    # Scaling: doubling mixing length halves strain rate
    sr_double_l = stress_closure('local', viscous_velocity=v_local, mixing_length=2.0 * l_mix)
    assert sr_double_l == pytest.approx(0.5 * sr_local, rel=1.0e-12)

    # Local mode with zero mixing length does not raise ZeroDivisionError
    sr_zero_l = compute_strain_rate_local(viscous_velocity=v_local, mixing_length=0.0)
    assert np.isfinite(sr_zero_l)
    assert sr_zero_l > 0.0

    # ── Test global mode ──
    # Synthetic planet: R_cmb = 3480 km, R_surf = 6371 km
    r_grid = np.linspace(3480.0e3, 6371.0e3, 50)
    # Cold lid down to 6000 km (depth = 371 km), interior at 1800 K
    t_profile = np.where(r_grid > 6000.0e3, 1000.0, 1800.0)
    # Convective velocity: zero in lid, 0.03 m/yr (~1.0e-9 m/s) in interior
    v_profile = np.where(r_grid > 6000.0e3, 0.0, 1.0e-9)

    sr_global = stress_closure(
        'global',
        viscous_velocity=v_profile,
        radius=r_grid,
        temperature=t_profile,
        t_lid_base=1400.0,
    )
    # Expected lid thickness: 6371 km - 6000 km = 371 km
    expected_d_lid = 6371.0e3 - 6000.0e3
    expected_v_int = 1.0e-9
    assert sr_global == pytest.approx(expected_v_int / expected_d_lid, rel=0.05)

    # ── Error contract validation ──
    # Missing mixing_length in local mode
    with pytest.raises(ValueError, match='mixing_length must be provided'):
        stress_closure('local', viscous_velocity=v_local)

    # Missing radius/temperature in global mode
    with pytest.raises(ValueError, match='radius and temperature must be provided'):
        stress_closure('global', viscous_velocity=v_profile)

    # Unrecognized mode
    with pytest.raises(ValueError, match='Unknown stress closure mode'):
        stress_closure('unsupported_mode', viscous_velocity=v_local)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_full_profile_mock_and_effective_viscosity_invariants():
    r"""Assert effective viscosity across a mocked 1D mantle column satisfies all physical limits.

    Validates:
    1. Surface lid yields: high Arrhenius viscosity (10^28+ Pa s) is capped to yield branch.
    2. Deep interior is ductile: Arrhenius viscosity governs; yielding is inactive.
    3. Effective viscosity is bounded, positive, and finite at every radial cell.

    Reference:
    Stacey & Davis (2008), Physics of the Earth, 4th ed.;
    Tosi et al. (2015), J. Geophys. Res. Planets 120, 714-730.
    """
    n_nodes = 80
    r_cmb = 3480.0e3
    r_surf = 6371.0e3
    r = np.linspace(r_cmb, r_surf, n_nodes)

    # Hydrostatic pressure profile: 135 GPa at CMB, 0 GPa at surface
    p = 135.0e9 * (r_surf - r) / (r_surf - r_cmb)

    # Temperature profile: 300 K at surface, conductive lid to 1500 K,
    # adiabatic mantle to ~2600 K, CMB thermal boundary layer up to 3500 K
    norm_r = (r - r_cmb) / (r_surf - r_cmb)
    t = 1600.0 + 1000.0 * (1.0 - norm_r) - 1200.0 * (norm_r**8)
    t = np.clip(t, 300.0, 4000.0)

    # Convective velocity and mixing length
    l_mix = np.maximum(0.1 * (r_surf - r_cmb), 1.0e4) * np.ones_like(r)
    v_visc = 1.0e-9 * np.sin(np.pi * norm_r)  # zero at boundaries, peak in mid-mantle

    # 1. Compute Arrhenius diffusion creep viscosity with mantle activation volume
    eta_d = eta_diff(t, p, viscosity_solid=1.0e21, activation_volume=1.5e-6)
    assert np.all(eta_d > 0.0)
    assert np.all(np.isfinite(eta_d))

    # At cold surface, Arrhenius viscosity should be astronomically large
    assert eta_d[-1] > 1.0e28

    # 2. Compute Byerlee yield stress
    tau_y = compute_yield_stress(p, yield_stress_c=50.0e6, yield_stress_mu=0.6)
    assert np.all(tau_y > 0.0)

    # 3. Compute strain rate via local closure
    strain_rate = stress_closure('local', v_visc, mixing_length=l_mix)

    # 4. Compute effective viscosity
    eta_effective = eta_eff(eta_d, tau_y, strain_rate, smooth=True)

    # Invariants
    assert np.all(eta_effective > 0.0)
    assert np.all(np.isfinite(eta_effective))
    assert np.all(eta_effective <= eta_d)

    # Check yielding activation:
    # In mid-mantle (warm, ductile), yielding is inactive (eta_eff ~= eta_d)
    mid_idx = n_nodes // 2
    assert eta_effective[mid_idx] == pytest.approx(
        eta_d[mid_idx], rel=0.05
    )  # Harmonic mean reduces it slightly

    # In cold near-surface lid (nodes 73 to 78), yielding caps the viscosity
    # by multiple orders of magnitude down to physically realistic lithospheric values
    lid_nodes = slice(73, 78)
    assert np.all(eta_effective[lid_nodes] < 1.0e25)
    assert np.all(eta_effective[lid_nodes] < eta_d[lid_nodes] * 1.0e-2)
