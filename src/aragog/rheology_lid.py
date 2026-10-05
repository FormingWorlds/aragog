"""Stagnant lid boundary layer indicators and effective viscosity closures."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from aragog.rheology import (
    R_GAS,
    FloatOrArray,
    compute_arrhenius_enthalpy,
    compute_arrhenius_viscosity,
    compute_strain_rate_local,
    compute_yield_stress,
)

__all__ = [
    'FB2014_STRESS_FACTOR',
    'FB2014_VELOCITY_EXPONENT',
    'LID_REGIME_LABEL_WIDTH',
    'SOLIDUS_MELT_FRACTION_THRESHOLD',
    'SOLIDUS_MELT_FRACTION_WIDTH',
    'compute_stagnant_lid_state',
    'compute_effective_viscosity',
    'stress_closure',
]

# Foley & Bercovici (2014) scaling parameters (Eq. 26 and Eq. 28).
FB2014_VELOCITY_EXPONENT: float = 2.0 / 3.0
FB2014_STRESS_FACTOR: float = 2.0

# Transition smoothing width that sets the lid_regime diagnostic label only.
LID_REGIME_LABEL_WIDTH: float = 0.1

# Melt fraction threshold marking solidus crossing / surface crust solidification.
# This 1% threshold is distinct from phi_rheo (~0.4), the rheological transition.
SOLIDUS_MELT_FRACTION_THRESHOLD: float = 0.01
SOLIDUS_MELT_FRACTION_WIDTH: float = 0.002


def compute_stagnant_lid_state(
    radii: FloatOrArray,
    temperature: FloatOrArray,
    pressure: FloatOrArray,
    convective_flux: FloatOrArray,
    total_flux: FloatOrArray,
    solidus_temperature: FloatOrArray | None,
    melt_fraction: FloatOrArray,
    params: Any,
    unyielded_velocity: FloatOrArray | None = None,
    viscosity_solid: float | None = None,
    viscosity_mixture: FloatOrArray | None = None,
    phi_rheo: float | None = None,
    phi_width: float | None = None,
    density: FloatOrArray = 3300.0,
    gravity: FloatOrArray = 9.81,
    thermal_expansivity: FloatOrArray = 3.0e-5,
    thermal_diffusivity: FloatOrArray = 1.0e-6,
    a_rh_fb: float = 1.3,
    C4: float = 0.125,
    xp: Any = np,
) -> dict[str, Any]:
    r"""Compute stagnant lid boundary-layer indicators and convective driving stress.

    Parameters
    ----------
    radii : float or array-like
        Radial grid coordinates r [m] from core to surface.
    temperature : float or array-like
        Radial temperature profile T [K].
    pressure : float or array-like
        Radial pressure profile P [Pa].
    convective_flux : float or array-like
        Convective heat flux profile F_conv [W/m^2].
    total_flux : float or array-like
        Total heat flux profile F_tot [W/m^2].
    solidus_temperature : float or array-like or None
        Solidus temperature profile T_sol [K]. When None, solidus crossing is
        evaluated using SOLIDUS_MELT_FRACTION_THRESHOLD (0.01 melt fraction).
    melt_fraction : float or array-like
        Melt mass fraction profile phi [-].
    params : SolidRheologyParams
        Rheology parameter set.
    unyielded_velocity : float or array-like, optional
        Unyielded convective velocity profile v [m/s].
    viscosity_solid : float, optional
        Reference solid mantle viscosity [Pa s]. Defaults to 1e21 if unspecified.
    viscosity_mixture : float or array-like, optional
        Effective mixture viscosity profile [Pa s] of the convecting layer (in mush,
        the blended two-phase viscosity used by the solver). When provided, mu_i
        is evaluated as the W_tilde-weighted log mean using the same boundary layer
        weights as T_i; otherwise falls back to pure solid Arrhenius viscosity.
    phi_rheo : float, optional
        Critical melt fraction for rheological transition. Defaults to 0.4.
    phi_width : float, optional
        Transition width for rheological melt fraction indicator. Defaults to 0.15.
    density : float or array-like, default 3300.0
        Mantle density rho [kg/m^3].
    gravity : float or array-like, default 9.81
        Gravitational acceleration g [m/s^2].
    thermal_expansivity : float or array-like, default 3.0e-5
        Thermal expansivity alpha [1/K].
    thermal_diffusivity : float or array-like, default 1.0e-6
        Thermal diffusivity kappa [m^2/s].
    a_rh_fb : float, default 1.3
        Rheological sublayer parameter fitted with C4 in FB2014 (p. 589).
    C4 : float, default 0.125
        Convective velocity prefactor in FB2014 (Eq. 26, p. 586-587; p. 589).
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    dict[str, Any]
        Dictionary of boundary layer state diagnostics.

    Notes
    -----
    The Frank-Kamenetskii parameter follows Foley & Bercovici (2014, p. 582):
    ``theta = params.activation_energy * (T_i - T_surf) / (R_GAS * T_i**2)``,
    using activation energy E rather than enthalpy H = E + P V.

    When convective layer weights place interior temperature T_i below lid base
    temperature T_lid (for example in cold_top_lid where T_i < T_lid), the solver
    evaluates theta and driving stress directly from the convective layer state T_i
    without clamping T_i to T_lid.

    When ``solidus_temperature`` is None, the solidus crossing indicator ``w_hot_sol``
    is evaluated from melt fraction using ``SOLIDUS_MELT_FRACTION_THRESHOLD`` (0.01).
    Because ``w_interior`` takes the maximum of isothermal, solidus, and rheological
    indicators, layers with melt fraction exceeding 0.01 are treated as non-lid interior
    by ``w_hot_sol``, rendering the lid boundary position independent of ``phi_rheo``.
    """
    r = xp.asarray(radii, dtype=float)
    T = xp.asarray(temperature, dtype=float)
    P = xp.asarray(pressure, dtype=float)
    F_conv = xp.asarray(convective_flux, dtype=float)
    F_tot = xp.asarray(total_flux, dtype=float)
    phi = xp.asarray(melt_fraction, dtype=float)

    F_tot_safe = xp.where(xp.abs(F_tot) > 1e-12, F_tot, 1e-12)
    f = xp.clip(xp.maximum(F_conv, 0.0) / xp.abs(F_tot_safe), 0.0, 1.0)

    f_conv_min = params.interior_flux_fraction
    df = xp.maximum(xp.abs(xp.gradient(f)), 1e-4)
    w_conv = 0.5 * (1.0 + xp.tanh((f - f_conv_min) / df))

    # Surface to CMB scan
    w_from_surf = w_conv[::-1]
    omw = 1.0 - w_from_surf
    ones = xp.ones(1, dtype=w_conv.dtype)
    omw_padded = xp.concatenate([ones, omw[:-1]])
    prefix = xp.cumprod(omw_padded)
    W_from_surf = w_from_surf * prefix
    W = W_from_surf[::-1]
    W_sum = xp.sum(W)
    W_safe = xp.maximum(W_sum, 1e-12)
    W_uniform = xp.ones_like(W) / float(T.shape[0])
    W_tilde = xp.where(W_sum > 1e-6, W / W_safe, W_uniform)

    T_i = xp.sum(W_tilde * T)
    P_T_i = xp.sum(W_tilde * P)
    r_i = xp.sum(W_tilde * r)

    w_active = 0.5 * (1.0 + xp.tanh((xp.sum(w_conv) - 1.0) / 0.5))
    is_convecting = (W_sum > 1e-6) & (w_active > 0.05)

    H_T_i = compute_arrhenius_enthalpy(
        P_T_i,
        activation_energy=params.activation_energy,
        activation_volume=params.activation_volume,
        activation_volume_decay_pressure=params.activation_volume_decay_pressure,
        xp=xp,
    )
    H_safe = xp.maximum(H_T_i, 1e-6)
    dT_rh = (R_GAS * T_i**2) / H_safe
    T_surf = T[-1]
    P_surf = P[-1]
    # Foley & Bercovici (2014) p. 582: theta = E * DeltaT / (R * T_i**2).
    theta = (params.activation_energy * (T_i - T_surf)) / (R_GAS * T_i**2)

    eta_0 = viscosity_solid
    if eta_0 is None:
        eta_0 = getattr(params, 'viscosity_solid', 1.0e21)

    arrhenius = partial(
        compute_arrhenius_viscosity,
        viscosity_solid=eta_0,
        activation_energy=params.activation_energy,
        activation_volume=params.activation_volume,
        activation_volume_decay_pressure=params.activation_volume_decay_pressure,
        arrhenius_t_ref=params.arrhenius_t_ref,
        r_gas=R_GAS,
        viscosity_max_log10=params.viscosity_max_log10,
        xp=xp,
    )
    eta_surf = arrhenius(T_surf, P_surf)
    eta_arr_i = arrhenius(T_i, P_T_i)
    if viscosity_mixture is not None:
        visc_mix = xp.asarray(viscosity_mixture, dtype=float)
        if xp.ndim(visc_mix) == 0:
            mu_i = visc_mix
        else:
            log_visc = xp.log10(xp.maximum(xp.ravel(visc_mix), 1e-30))
            mu_i_mix = 10.0 ** xp.sum(W_tilde * log_visc)
            # Uniform weights corrupt mu_i with cold surface viscosity when convection is inactive.
            mu_i = xp.where(W_sum > 1e-6, mu_i_mix, eta_arr_i)
    else:
        mu_i = eta_arr_i
    eta_i = mu_i
    eta_contrast = eta_surf / xp.maximum(eta_i, 1e-30)

    if params.lid_base_mode == 'fixed':
        T_lid_iso = params.lid_base_temperature
    else:
        T_lid_iso = T_i - params.lid_contrast_coeff * dT_rh

    dT_cell = xp.maximum(xp.abs(xp.gradient(T)), 1.0)
    w_mask = params.lid_mask_width_cells
    scale = w_mask * dT_cell

    phi_rheo_val = phi_rheo if phi_rheo is not None else getattr(params, 'phi_rheo', 0.4)
    phi_width_val = phi_width if phi_width is not None else getattr(params, 'phi_width', 0.15)

    w_hot_iso = 0.5 * (1.0 + xp.tanh((T - T_lid_iso) / scale))
    if solidus_temperature is not None:
        T_sol = xp.asarray(solidus_temperature, dtype=float)
        w_hot_sol = 0.5 * (1.0 + xp.tanh((T - T_sol) / scale))
    else:
        w_raw_sol = 0.5 * (
            1.0 + xp.tanh((phi - SOLIDUS_MELT_FRACTION_THRESHOLD) / SOLIDUS_MELT_FRACTION_WIDTH)
        )
        w_zero_sol = 0.5 * (
            1.0 + xp.tanh(-SOLIDUS_MELT_FRACTION_THRESHOLD / SOLIDUS_MELT_FRACTION_WIDTH)
        )
        w_hot_sol = xp.clip((w_raw_sol - w_zero_sol) / (1.0 - w_zero_sol), 0.0, 1.0)

    w_raw_phi = 0.5 * (1.0 + xp.tanh((phi - phi_rheo_val) / phi_width_val))
    w_zero_phi = 0.5 * (1.0 + xp.tanh(-phi_rheo_val / phi_width_val))
    w_hot_phi = xp.clip((w_raw_phi - w_zero_phi) / (1.0 - w_zero_phi), 0.0, 1.0)

    w_interior = xp.maximum(xp.maximum(w_hot_iso, w_hot_sol), w_hot_phi)
    w_lid = (1.0 - w_interior) * w_active

    w_int_surf = w_interior[::-1]
    om_int = 1.0 - w_int_surf
    om_int_padded = xp.concatenate([ones, om_int[:-1]])
    prefix_int = xp.cumprod(om_int_padded)
    W_base_surf = w_int_surf * prefix_int
    W_base = W_base_surf[::-1]
    W_base_sum = xp.sum(W_base)
    W_base_safe = xp.maximum(W_base_sum, 1e-12)
    W_base_uniform = xp.ones_like(W_base) / float(T.shape[0])
    W_base_tilde = xp.where(W_base_sum > 1e-6, W_base / W_base_safe, W_base_uniform)

    r_lid_base = xp.sum(W_base_tilde * r)
    P_lid_base = xp.sum(W_base_tilde * P)
    T_lid = xp.sum(W_base_tilde * T)
    d_lid = xp.where(is_convecting, xp.maximum(r[-1] - r_lid_base, 0.0) * w_active, 0.0)

    dT_dr = xp.abs(xp.gradient(T, r))
    dr_min = xp.min(xp.abs(r[1:] - r[:-1]))
    dTdr_floor = 1.0 / xp.maximum(dr_min, 1.0)
    dTdr_safe = xp.maximum(dT_dr, dTdr_floor)
    dTdr_lid_base = xp.sum(W_base_tilde * dTdr_safe)
    delta_rh = dT_rh / xp.maximum(dTdr_lid_base, 1e-12)
    delta_rh = xp.minimum(delta_rh, r[-1] - r[0])

    if unyielded_velocity is not None:
        T_v = 3.17e-12
        r_mid = 0.5 * (r_i + r[0])
        dr_scale = xp.maximum(xp.mean(xp.abs(r[1:] - r[:-1])), 1.0)
        w_above_mid = 0.5 * (1.0 + xp.tanh((r - r_mid) / dr_scale))
        w_below_top = 0.5 * (1.0 + xp.tanh((r_i - r) / dr_scale))
        w_upper = w_above_mid * w_below_top * w_conv
        N_upper = xp.sum(w_upper) + 1e-12

        v_abs = xp.abs(unyielded_velocity)
        u = w_upper * v_abs
        u_shift = xp.max(u)
        sum_w_exp = xp.sum(w_upper * xp.exp((u - u_shift) / T_v))
        v_i_raw = u_shift + T_v * (xp.log(xp.maximum(sum_w_exp, 1e-300)) - xp.log(N_upper))
        v_i = xp.where(is_convecting, xp.maximum(v_i_raw, 0.0) * w_active, 0.0)
    else:
        v_i = 0.0
        w_upper = xp.zeros_like(r)

    d_conv = xp.maximum(r[-1] - r[0], 1.0)

    def _weighted_mean(val):
        arr = xp.asarray(val, dtype=float)
        if xp.ndim(arr) == 0:
            return arr
        return xp.sum(W_tilde * arr)

    rho_i = _weighted_mean(density)
    g_i = _weighted_mean(gravity)
    alpha_i = _weighted_mean(thermal_expansivity)
    kappa_i = _weighted_mean(thermal_diffusivity)
    DeltaT_conv = xp.maximum(T_i - T[-1], 1e-6)
    eta_i_safe = xp.maximum(eta_i, 1e-30)

    # Convective interior velocity and lid driving stress (Foley & Bercovici 2014, eqs. 26, 28).
    Ra_eff = xp.where(
        is_convecting,
        (rho_i * g_i * alpha_i * DeltaT_conv * (d_conv**3)) / (kappa_i * eta_i_safe),
        0.0,
    )
    theta_safe = xp.maximum(theta, 1e-6)
    Ra_rh = xp.where(is_convecting, (Ra_eff * a_rh_fb) / theta_safe, 0.0)
    v_m = xp.where(
        is_convecting,
        (kappa_i / d_conv) * C4 * (xp.maximum(Ra_rh, 0.0) ** FB2014_VELOCITY_EXPONENT),
        0.0,
    )

    tau_d = xp.where(
        is_convecting,
        (FB2014_STRESS_FACTOR * eta_i * v_m / d_conv) * w_active,
        0.0,
    )
    tau_d_lid = xp.where(
        is_convecting,
        (FB2014_STRESS_FACTOR * eta_i * v_m) / xp.maximum(d_lid, 1e-6),
        0.0,
    )
    tau_buoy = xp.where(
        is_convecting,
        rho_i * g_i * alpha_i * dT_rh * delta_rh * w_active,
        0.0,
    )
    tau_d_over_tau_buoy = xp.where(
        is_convecting,
        tau_d / xp.maximum(tau_buoy, 1e-30),
        0.0,
    )

    tau_y_lid = compute_yield_stress(
        P_lid_base,
        yield_stress_c=params.yield_stress_c,
        yield_stress_mu=params.yield_stress_mu,
        yield_stress_max=params.yield_stress_max,
        xp=xp,
    )

    ratio = tau_d / xp.maximum(tau_y_lid, 1e-10)
    w_y = 0.5 * (1.0 + xp.tanh((ratio - 1.0) / LID_REGIME_LABEL_WIDTH))
    lid_cell_count = xp.sum(w_lid)
    w_solid_surf = 0.5 * (
        1.0 - xp.tanh((phi[-1] - SOLIDUS_MELT_FRACTION_THRESHOLD) / SOLIDUS_MELT_FRACTION_WIDTH)
    )
    w_has_lid = 0.5 * (1.0 + xp.tanh((lid_cell_count - 0.5) / 0.1)) * w_solid_surf
    lid_regime = xp.where(is_convecting, w_active * w_has_lid * (1.0 + w_y), 0.0)

    return {
        'T_i': T_i,
        'P_T_i': P_T_i,
        'r_i': r_i,
        'dT_rh': dT_rh,
        'theta': theta,
        'eta_contrast': eta_contrast,
        'T_lid': T_lid,
        'T_lid_iso': T_lid_iso,
        'd_lid': d_lid,
        'r_lid_base': r_lid_base,
        'P_lid_base': P_lid_base,
        'delta_rh': delta_rh,
        'v_i': v_i,
        'v_m': v_m,
        'Ra_eff': Ra_eff,
        'Ra_rh': Ra_rh,
        'w_upper': w_upper,
        'tau_d': tau_d,
        'tau_d_lid': tau_d_lid,
        'tau_buoy': tau_buoy,
        'tau_d_over_tau_buoy': tau_d_over_tau_buoy,
        'tau_y_lid': tau_y_lid,
        'w_lid': w_lid,
        'w_y': w_y,
        'w_active': w_active,
        'lid_regime': lid_regime,
        'lid_cell_count': lid_cell_count,
        'eta_i': eta_i,
        'mu_i': mu_i,
        'eta_solid_arrhenius': eta_arr_i,
    }


def compute_effective_viscosity(
    eta_diff: FloatOrArray,
    tau_d: FloatOrArray | None = None,
    tau_y_lid: FloatOrArray | None = None,
    v_i: FloatOrArray | None = None,
    delta_rh: FloatOrArray | None = None,
    eta_i: FloatOrArray | None = None,
    stress_closure_mode: str = 'lid',
    unyielded_velocity: FloatOrArray | None = None,
    mixing_length: FloatOrArray | None = None,
    tau_y_profile: FloatOrArray | None = None,
    w_lid: FloatOrArray | None = None,
    strain_rate: FloatOrArray | None = None,
    tau_y: FloatOrArray | None = None,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute effective dynamic viscosity with boundary-layer or local stress closure.

    Both stagnant lid mode and local stress closure mode use the minimum closure
    of diffusion creep and plastic yield viscosity per Moresi & Solomatov (1998):
        eta_eff = min(eta_diff, eta_y)

    In stagnant lid mode:
        eta_eff = eta_diff * min(1, tau_y / tau_d)
    where edot_lid = tau_d / (2 * eta_diff) and eta_y = tau_y * eta_diff / tau_d.

    In local stress closure mode:
        eta_eff = min(eta_diff, eta_y)
        eta_y = tau_y / (2 * strain_rate)

    References
    ----------
    Moresi & Solomatov (1998), p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x
        (viscosity switch / minimum form eta_eff = min(eta_creep, eta_yield))
    Tackley (2000), eq. 8, p. 4, doi:10.1029/2000GC000036
        (viscosity cap / minimum form eta_eff = min[eta, sigma_y / (2 e_dot)])
    Foley & Becker (2009), eqs. 7-8, p. 3, doi:10.1029/2009GC002378
        (alternative harmonic mean formulation)
    """
    if stress_closure_mode == 'global':
        raise ValueError(
            "Unknown stress_closure_mode 'global'; expected 'lid' or 'local'. "
            "'global' mode is unsupported; use 'lid'."
        )
    if stress_closure_mode == 'local' or strain_rate is not None or tau_y is not None:
        if strain_rate is None:
            if unyielded_velocity is None or mixing_length is None:
                raise ValueError(
                    "mixing_length and unyielded_velocity required for 'local' mode"
                )
            sr = compute_strain_rate_local(unyielded_velocity, mixing_length, xp=xp)
        else:
            sr = xp.asarray(strain_rate, dtype=float)
        ty = xp.asarray(tau_y if tau_y is not None else tau_y_profile, dtype=float)
        eta_d = xp.asarray(eta_diff, dtype=float)
        safe_ty = xp.maximum(ty, 0.0)
        sr_safe = xp.where(sr > 0.0, sr, 1.0)
        tau_y_term = safe_ty / (2.0 * sr_safe)
        is_unyielded = (sr <= 0.0) | xp.isinf(tau_y_term) | xp.isnan(tau_y_term)
        safe_ty_term = xp.where(is_unyielded, eta_d, tau_y_term)
        res = xp.minimum(eta_d, safe_ty_term)
        return xp.where(is_unyielded, eta_d, res)

    eta_d = xp.asarray(eta_diff, dtype=float)

    # Moresi & Solomatov (1998) Eq. 13-14 min closure:
    # eta_eff = min(eta_diff, eta_y) = eta_diff * min(1, tau_y / tau_d)
    if tau_d is not None and tau_y_lid is not None:
        td = xp.asarray(tau_d, dtype=float)
        ty_lid = xp.asarray(tau_y_lid, dtype=float)
        ratio = xp.where(td > 0.0, ty_lid / xp.maximum(td, 1e-30), 1.0)
        ratio = xp.where(xp.isnan(ratio) | xp.isinf(ty_lid), 1.0, ratio)
        ratio_eff = xp.clip(ratio, 0.0, 1.0)
        eta_eff_val = xp.where(ratio_eff >= 1.0, eta_d, eta_d * ratio_eff)
    else:
        eta_eff_val = eta_d

    if w_lid is not None:
        wl = xp.asarray(w_lid, dtype=float)
        log_diff = xp.log10(xp.maximum(eta_d, 1e-30))
        log_eff = xp.log10(xp.maximum(eta_eff_val, 1e-30))
        log_solid = wl * log_eff + (1.0 - wl) * log_diff
        res = 10.0**log_solid
        return xp.where((wl == 0.0) | (eta_eff_val == eta_d), eta_d, res)

    return eta_eff_val


def stress_closure(
    mode: str,
    viscous_velocity: FloatOrArray,
    mixing_length: FloatOrArray | None = None,
    eps: float = 1.0e-15,
) -> FloatOrArray:
    r"""Calculate strain rate dispatching on stress closure mode."""
    if mode == 'local':
        if mixing_length is None:
            raise ValueError("mixing_length must be provided when mode='local'")
        return compute_strain_rate_local(viscous_velocity, mixing_length, eps=eps)
    elif mode == 'global':
        raise ValueError(
            "Unknown stress closure mode 'global'; expected 'lid' or 'local'. "
            "'global' mode is unsupported; use 'lid'."
        )
    elif mode == 'lid':
        raise ValueError(
            "Mode 'lid' uses boundary-layer stress closure via compute_stagnant_lid_state"
        )
    else:
        raise ValueError(f"Unknown stress closure mode {mode!r}; expected 'lid' or 'local'")
