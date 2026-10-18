"""Stagnant lid boundary layer indicators and effective viscosity closures."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from aragog.rheology import (
    COHESION_FLOOR,
    R_GAS,
    FloatOrArray,
    _require_x64_if_jax,
    compute_arrhenius_enthalpy,
    compute_arrhenius_viscosity,
    compute_strain_rate_local,
    compute_yield_stress,
)

__all__ = [
    'COHESION_FLOOR',
    'FB2014_STRESS_FACTOR',
    'FB2014_VELOCITY_EXPONENT',
    'LID_REGIME_LABEL_WIDTH',
    'NO_LID_DT_THRESHOLD',
    'NO_LID_THICKNESS_THRESHOLD',
    'SOFTMAX_VELOCITY_SCALE',
    'SOLIDUS_MELT_FRACTION_THRESHOLD',
    'SOLIDUS_MELT_FRACTION_WIDTH',
    'compute_stagnant_lid_state',
    'compute_effective_viscosity',
    'stress_closure',
]

# Foley & Bercovici (2014) scaling parameters (Eq. 26 and Eq. 28).
FB2014_VELOCITY_EXPONENT: float = 2.0 / 3.0
FB2014_STRESS_FACTOR: float = 2.0

# Minimum thermal contrast [K] and thickness [m] required for a physical conductive lid.
NO_LID_DT_THRESHOLD: float = 1.0e-3
NO_LID_THICKNESS_THRESHOLD: float = 1.0

# Transition smoothing width that sets the lid_regime diagnostic label only.
LID_REGIME_LABEL_WIDTH: float = 0.1

# Melt fraction threshold marking solidus crossing / surface crust solidification.
# This 1% threshold is distinct from phi_rheo (~0.4), the rheological transition.
SOLIDUS_MELT_FRACTION_THRESHOLD: float = 0.01
SOLIDUS_MELT_FRACTION_WIDTH: float = 0.002

# Soft-maximum smoothing scale [m/s] (~0.1 mm/yr) for the upper-mantle velocity diagnostic v_i.
SOFTMAX_VELOCITY_SCALE: float = 3.17e-12


def _boundary_weights(w: Any, xp: Any, *, fallback: Any | None = None) -> tuple[Any, Any, Any]:
    """Smooth weights of the shallowest crossing of ``w``, scanned from the surface down.

    Evaluates ``W_k = w_k * prod_{j > k} (1 - w_j)`` from the outermost node ``r[-1]``
    inward and returns ``(W_tilde, W_sum, w_weight)``; ``W_tilde`` relaxes to ``fallback``
    (or to the bottom node at index 0 when ``fallback`` is None) when ``max(w)`` vanishes.
    """
    w_surf = w[::-1]
    omw = xp.concatenate([xp.ones(1, dtype=w.dtype), 1.0 - w_surf[:-1]])
    W = (w_surf * xp.cumprod(omw))[::-1]
    W_sum = xp.sum(W)
    w_weight = xp.tanh(xp.max(w) / 1.0e-6)
    n = w.shape[0]
    W_fallback = xp.where(xp.arange(n) == 0, 1.0, 0.0) if fallback is None else fallback
    W_tilde = w_weight * (W / xp.maximum(W_sum, 1e-12)) + (1.0 - w_weight) * W_fallback
    return W_tilde, W_sum, w_weight


def _scalar_or_nan(val: Any, name: str) -> float:
    """Return ``float(val)`` on numeric scalars, ``nan`` on traced scalars; reject arrays, bools, and strings."""
    if (
        isinstance(val, (bool, np.bool_, str, bytes))
        or getattr(val, 'ndim', 0) > 0
        or (not hasattr(val, 'ndim') and np.ndim(val) > 0)
    ):
        raise ValueError(f'{name} must be a scalar')
    try:
        if np.asarray(val).dtype.kind not in 'iuf':
            raise ValueError(f'{name} must be a scalar')
        return float(val)
    except TypeError as exc:
        if 'Tracer' in type(val).__name__ or 'Concretization' in type(exc).__name__:
            return np.nan
        raise ValueError(f'{name} must be a real scalar, got {val!r}') from exc


def compute_stagnant_lid_state(
    radii: FloatOrArray,
    temperature: FloatOrArray,
    pressure: FloatOrArray,
    convective_flux: FloatOrArray,
    total_flux: FloatOrArray,
    solidus_temperature: FloatOrArray | None,
    melt_fraction: FloatOrArray,
    params: Any,
    viscosity_solid: float,
    phi_rheo: float,
    phi_width: float,
    unyielded_velocity: FloatOrArray | None = None,
    viscosity_mixture: FloatOrArray | None = None,
    density: FloatOrArray = 3300.0,
    gravity: FloatOrArray = 9.81,
    thermal_expansivity: FloatOrArray = 3.0e-5,
    thermal_diffusivity: FloatOrArray = 1.0e-6,
    a_rh_fb: float = 1.3,
    C4: float = 0.125,
    t_sol_0: float | None = None,
    xp: Any = np,
) -> dict[str, Any]:
    r"""Compute stagnant lid boundary-layer indicators and convective driving stress.

    Parameters
    ----------
    radii : array-like
        Radial grid coordinates r [m] from core to surface.
    temperature : array-like
        Radial temperature profile T [K].
    pressure : array-like
        Radial pressure profile P [Pa].
    convective_flux : array-like
        Convective heat flux profile F_conv [W/m^2].
    total_flux : array-like
        Total heat flux profile F_tot [W/m^2].
    solidus_temperature : float or array-like or None
        Solidus temperature profile T_sol [K]. When None, solidus crossing is
        evaluated using SOLIDUS_MELT_FRACTION_THRESHOLD (0.01 melt fraction).
    melt_fraction : array-like
        Melt mass fraction profile phi [-].
    params : SolidRheologyParams
        Rheology parameter set.
    viscosity_solid : float
        Reference solid mantle viscosity [Pa s], equal to ``eta_diff`` at
        ``(params.arrhenius_t_ref, P = 0)`` for the linear solidus; when a
        ``solidus_temperature`` table is provided, it is referenced to
        ``params.t_sol_0`` and differs from ``eta_diff`` at the table's
        ``T_sol(0)`` by ``exp(g_hom * (T_sol(0) - params.t_sol_0) / T)``
        (1.57x at 1600 K for 1429 K against 1400 K).
    phi_rheo : float
        Critical melt fraction for rheological transition [-].
    phi_width : float
        Transition width for rheological melt fraction indicator [-].
    unyielded_velocity : float or array-like, optional
        Unyielded convective velocity profile v [m/s].
    viscosity_mixture : float or array-like, optional
        Effective mixture viscosity profile [Pa s] of the convecting layer (in mush,
        the blended two-phase viscosity used by the solver). When provided, mu_i
        blends smoothly via ``w_weight = tanh(max_k w_conv_int / 1e-6)`` between the
        W_tilde-weighted log mean of ``viscosity_mixture`` and the solid Arrhenius
        viscosity ``eta_arr_i``; otherwise falls back to ``eta_arr_i``.
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
    t_sol_0 : float, optional
        Zero-pressure reference solidus temperature T_sol(0) [K]. Required when
        ``solidus_temperature`` is provided, and must match ``params.t_sol_0``
        when given; defaults to ``params.t_sol_0`` for the linear fallback.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    dict[str, Any]
        Dictionary of boundary layer state diagnostics. ``has_lid`` is False when
        any input is NaN while numeric diagnostics propagate NaN.

    Raises
    ------
    ValueError
        If scalar parameters are None, non-scalar, infinite, or out of bounds on
        the eager path, if ``t_sol_0`` is missing when ``solidus_temperature`` is
        given or does not match ``params.t_sol_0``, or if radial and profile
        arrays fail shape or monotonicity checks. Under ``jax.jit``, traced
        scalar value violations propagate NaN through ``zero_or_nan`` instead of
        raising ``ValueError``.

    Notes
    -----
    The Frank-Kamenetskii parameter uses the homologous activation enthalpy at the
    interior pressure ``P_i``:
    ``theta = H(P_i) * (T_i - T_surf) / (R_GAS * T_i**2)`` with
    ``H(P_i) = params.g_homologous * R_GAS * T_sol(P_i)``; when ``solidus_temperature``
    is None, ``T_sol`` is the linear fallback ``t_sol_0 + dt_sol_dp * P``.

    When convective layer weights place interior temperature T_i below lid base
    temperature T_lid (for example in cold_top_lid where T_i < T_lid), the solver
    evaluates theta and driving stress directly from the convective layer state T_i
    without clamping T_i to T_lid.

    When convection is inactive or no physical conductive lid forms
    (``d_lid_raw <= NO_LID_THICKNESS_THRESHOLD`` of 1.0 m or
    ``T_i - T_surf <= NO_LID_DT_THRESHOLD`` of 1.0e-3 K, where the ``w_active > 0.05``
    threshold sits at ``f_int_max = params.interior_flux_fraction + 0.01 * atanh(-0.9) = 0.035278``
    at the default ``interior_flux_fraction = 0.05``), boundary-layer diagnostics
    (``d_lid``, ``Ra_eff``, ``Ra_rh``, ``v_m``, ``tau_d``, ``tau_d_lid``,
    ``tau_buoy``, ``tau_d_over_tau_buoy``, ``w_y``, ``lid_regime``) are zeroed and
    ``has_lid`` is False, while ``T_i``, ``eta_i``, ``mu_i``, and ``theta`` serve as
    interior fallback diagnostics (selecting the shallowest interior node near
    ``0.9 * f_int_max`` when ``max_k w_conv_int >> 1e-6``, such as
    ``1e-4 <= f_int_max < 0.035278``, and relaxing to the interior fallback node
    ``T[1]`` as ``max_k w_conv_int -> 0`` at ``f = 0``) and all solver-facing outputs
    vary continuously across uniform convective flux fractions ``f`` in ``[0, 1e-3]``.

    The lid-base locator ``W_base_tilde = _boundary_weights(w_interior, xp)[0]`` scans
    ``w_interior = max(w_hot_iso, w_hot_sol, w_hot_phi)`` (smoothed over the local cell
    temperature step ``scale_cell = lid_mask_width_cells * max(|grad T|, 1.0 K)``) from
    the outermost node ``r[-1]`` inward, whereas the nodal lid mask
    ``w_lid = (1 - w_interior_mask) * w_active`` uses the capped width
    ``scale_mask = min(scale_cell, lid_mask_width_cells * max(dT_rh, 1.0 K))`` via
    ``w_interior_mask = max(w_hot_iso_mask, w_hot_sol_mask, w_hot_phi)``.
    When ``solidus_temperature`` is None, the solidus crossing indicator ``w_hot_sol``
    is evaluated from melt fraction using ``SOLIDUS_MELT_FRACTION_THRESHOLD`` (0.01).
    Because ``w_interior`` and ``w_interior_mask`` take the maximum of isothermal,
    solidus, and rheological indicators, layers with melt fraction exceeding 0.01 are
    treated as non-lid interior, rendering the lid boundary position independent of
    ``phi_rheo``.
    """
    _require_x64_if_jax(xp)
    if viscosity_solid is None or phi_rheo is None or phi_width is None:
        raise ValueError('viscosity_solid, phi_rheo, and phi_width must be provided')
    pos_fin = lambda v: (0.0 < v) & (v < np.inf)  # noqa: E731
    scalar_bounds = (
        ('viscosity_solid', viscosity_solid, pos_fin, 'positive and finite'),
        ('phi_width', phi_width, pos_fin, 'positive and finite'),
        ('phi_rheo', phi_rheo, lambda v: (0.0 <= v) & (v <= 1.0), 'finite and in [0, 1]'),
        ('a_rh_fb', a_rh_fb, pos_fin, 'positive and finite'),
        ('C4', C4, pos_fin, 'positive and finite'),
    )
    for s_name, s_raw, ok, msg in scalar_bounds:
        v = _scalar_or_nan(s_raw, s_name)
        if not np.isnan(v) and not ok(v):
            raise ValueError(f'{s_name} must be {msg}, got {s_raw}')
    if solidus_temperature is not None and t_sol_0 is None:
        raise ValueError('t_sol_0 must be provided when solidus_temperature is given')
    if t_sol_0 is not None:
        ts0 = _scalar_or_nan(t_sol_0, 't_sol_0')
        if not np.isnan(ts0) and ts0 != params.t_sol_0:
            raise ValueError(
                f't_sol_0 ({t_sol_0}) must be finite and match params.t_sol_0 ({params.t_sol_0})'
            )
    t_sol_0_eff = params.t_sol_0 if t_sol_0 is None else t_sol_0
    r = xp.asarray(radii, dtype=float)
    if r.ndim != 1 or r.shape[0] < 3:
        raise ValueError('radii must be a 1-D array with at least 3 nodes')
    try:
        r_np = np.asarray(r, dtype=float)
    except Exception:
        r_np = None
    if r_np is not None:
        dr_np = r_np[1:] - r_np[:-1]
        if np.any(np.isinf(r_np)) or np.any(np.isinf(dr_np) | (dr_np <= 0.0)):
            raise ValueError('radii must be strictly increasing from core to surface')
    T = xp.asarray(temperature, dtype=float)
    P = xp.asarray(pressure, dtype=float)
    F_conv = xp.asarray(convective_flux, dtype=float)
    F_tot = xp.asarray(total_flux, dtype=float)
    phi = xp.asarray(melt_fraction, dtype=float)
    for name, val, allow_scalar in (
        ('temperature', T, False),
        ('pressure', P, False),
        ('convective_flux', F_conv, False),
        ('total_flux', F_tot, False),
        ('melt_fraction', phi, False),
        ('solidus_temperature', solidus_temperature, True),
        ('unyielded_velocity', unyielded_velocity, True),
        ('viscosity_mixture', viscosity_mixture, True),
        ('density', density, True),
        ('gravity', gravity, True),
        ('thermal_expansivity', thermal_expansivity, True),
        ('thermal_diffusivity', thermal_diffusivity, True),
    ):
        if val is None:
            continue
        a = xp.asarray(val, dtype=float)
        if (allow_scalar and a.ndim == 0) or (a.ndim == 1 and a.shape[0] == r.shape[0]):
            continue
        raise ValueError(f'{name} must have the same 1-D length as radii ({r.shape[0]})')

    F_tot_safe = xp.where(xp.abs(F_tot) > 1e-12, F_tot, 1e-12)
    f = xp.clip(xp.maximum(F_conv, 0.0) / xp.abs(F_tot_safe), 0.0, 1.0)
    opt_arrays = (solidus_temperature, unyielded_velocity, viscosity_mixture)
    nan_inputs = (
        r,
        T,
        P,
        F_conv,
        F_tot,
        f,
        phi,
        density,
        gravity,
        thermal_expansivity,
        thermal_diffusivity,
        *(a for a in opt_arrays if a is not None),
    )
    any_input_nan = xp.any(
        xp.isnan(xp.concatenate([xp.ravel(xp.asarray(a, dtype=float)) for a in nan_inputs]))
    )
    invalid_scalar = xp.asarray(t_sol_0_eff, dtype=float) != params.t_sol_0
    for _, s_raw, ok, _ in scalar_bounds:
        invalid_scalar = invalid_scalar | ~ok(xp.asarray(s_raw, dtype=float))
    zero_or_nan = xp.where(any_input_nan | invalid_scalar, xp.nan, 0.0)

    w_conv = 0.5 * (1.0 + xp.tanh((f - params.interior_flux_fraction) / 0.01)) + zero_or_nan
    idx = xp.arange(r.shape[0])
    interior_node_mask = xp.where((idx > 0) & (idx < r.shape[0] - 1), 1.0, 0.0)
    w_conv_0 = 0.5 * (1.0 + xp.tanh(-params.interior_flux_fraction / 0.01))
    w_conv_int = xp.maximum(w_conv - w_conv_0, 0.0) * interior_node_mask + zero_or_nan
    interior_fallback = xp.where(idx == 1, 1.0, 0.0)
    W_conv_tilde, W_sum, w_weight = _boundary_weights(
        w_conv_int, xp, fallback=interior_fallback
    )

    f_int_max = xp.max(f * interior_node_mask)
    f_int_scale = xp.maximum(f_int_max, 1.0e-12)
    f_Ti_thresh = 0.9 * f_int_max
    w_Ti_width = 0.02 * f_int_scale
    w_Ti = (
        0.5 * (1.0 + xp.tanh((f - f_Ti_thresh) / w_Ti_width)) * interior_node_mask + zero_or_nan
    )
    W_Ti_tilde = _boundary_weights(w_Ti, xp, fallback=W_conv_tilde)[0]
    W_tilde = w_weight * W_Ti_tilde + (1.0 - w_weight) * W_conv_tilde

    def _weighted_mean(val):
        arr = xp.asarray(val, dtype=float)
        return arr if xp.ndim(arr) == 0 else xp.sum(W_tilde * xp.ravel(arr))

    T_i = xp.sum(W_tilde * T)
    P_T_i = xp.sum(W_tilde * P)
    r_i = xp.sum(W_tilde * r)

    w_active = (
        0.5 * (1.0 + xp.tanh((f_int_max - params.interior_flux_fraction) / 0.01)) + zero_or_nan
    )
    is_convecting = (W_sum > 1e-6) & (w_active > 0.05)

    T_sol_arr = T_sol_i = T_sol_surf = None
    if solidus_temperature is not None:
        T_sol_arr = xp.asarray(solidus_temperature, dtype=float)
        if xp.ndim(T_sol_arr) > 0:
            T_sol_arr = xp.ravel(T_sol_arr)
            T_sol_i = xp.sum(W_tilde * T_sol_arr)
            T_sol_surf = T_sol_arr[-1]
        else:
            T_sol_i = T_sol_surf = T_sol_arr

    H_T_i = compute_arrhenius_enthalpy(
        P_T_i,
        g_homologous=params.g_homologous,
        solidus_temperature=T_sol_i,
        t_sol_0=t_sol_0_eff,
        dt_sol_dp=params.dt_sol_dp,
        r_gas=R_GAS,
        xp=xp,
    )
    H_safe = xp.maximum(H_T_i, 1e-6)
    T_i_safe = xp.maximum(T_i, 1.0)
    dT_rh = (R_GAS * T_i_safe**2) / H_safe
    T_surf = T[-1]
    P_surf = P[-1]
    # Homologous Frank-Kamenetskii contrast: theta = H(P_i) * (T_i - T_surf) / (R * T_i**2).
    theta = (H_T_i * (T_i - T_surf)) / (R_GAS * T_i_safe**2)

    arrhenius = partial(
        compute_arrhenius_viscosity,
        viscosity_solid=viscosity_solid,
        g_homologous=params.g_homologous,
        t_sol_0=t_sol_0_eff,
        dt_sol_dp=params.dt_sol_dp,
        arrhenius_t_ref=params.arrhenius_t_ref,
        r_gas=R_GAS,
        viscosity_max_log10=params.viscosity_max_log10,
        water_prefactor=params.water_prefactor,
        xp=xp,
    )
    eta_surf = arrhenius(T_surf, P_surf, solidus_temperature=T_sol_surf)
    eta_arr_i = arrhenius(T_i, P_T_i, solidus_temperature=T_sol_i)
    if viscosity_mixture is not None:
        log_eta_arr = xp.log10(xp.maximum(eta_arr_i, 1e-30))
        log_mu_mix = _weighted_mean(
            xp.log10(xp.maximum(xp.asarray(viscosity_mixture, dtype=float), 1e-30))
        )
        # Smooth log-blend from eta_arr_i when convection is inactive to mu_i_mix when active.
        eta_i = 10.0 ** (w_weight * log_mu_mix + (1.0 - w_weight) * log_eta_arr)
    else:
        eta_i = eta_arr_i
    eta_contrast = eta_surf / xp.maximum(eta_i, 1e-30)

    if params.lid_base_mode == 'fixed':
        T_lid_iso = params.lid_base_temperature + zero_or_nan
    else:
        T_lid_iso = T_i - params.lid_contrast_coeff * dT_rh

    scale = params.lid_mask_width_cells * xp.maximum(xp.abs(xp.gradient(T)), 1.0)
    scale_mask = xp.minimum(scale, params.lid_mask_width_cells * xp.maximum(dT_rh, 1.0))

    def _melt_ramp(centre, width):
        """Hot-side tanh indicator of ``phi``, rescaled to 0 at ``phi = 0``."""
        zero = 0.5 * (1.0 + xp.tanh(-centre / width))
        raw = 0.5 * (1.0 + xp.tanh((phi - centre) / width))
        return xp.clip((raw - zero) / (1.0 - zero), 0.0, 1.0)

    w_hot_iso = 0.5 * (1.0 + xp.tanh((T - T_lid_iso) / scale))
    w_hot_iso_mask = 0.5 * (1.0 + xp.tanh((T - T_lid_iso) / scale_mask))
    if T_sol_arr is not None:
        w_hot_sol = 0.5 * (1.0 + xp.tanh((T - T_sol_arr) / scale))
        w_hot_sol_mask = 0.5 * (1.0 + xp.tanh((T - T_sol_arr) / scale_mask))
    else:
        w_hot_sol = _melt_ramp(SOLIDUS_MELT_FRACTION_THRESHOLD, SOLIDUS_MELT_FRACTION_WIDTH)
        w_hot_sol_mask = w_hot_sol
    w_hot_phi = _melt_ramp(phi_rheo, phi_width)

    w_interior = xp.maximum(xp.maximum(w_hot_iso, w_hot_sol), w_hot_phi) + zero_or_nan
    w_interior_mask = (
        xp.maximum(xp.maximum(w_hot_iso_mask, w_hot_sol_mask), w_hot_phi) + zero_or_nan
    )
    w_lid = (1.0 - w_interior_mask) * w_active

    W_base_tilde = _boundary_weights(w_interior, xp)[0]

    r_lid_base = xp.sum(W_base_tilde * r)
    P_lid_base = xp.sum(W_base_tilde * P)
    T_lid = xp.sum(W_base_tilde * T)
    d_geom = xp.maximum(r[-1] - r_lid_base, 1.0e-6)
    d_lid_raw = xp.maximum(r[-1] - r_lid_base, 0.0) * w_active
    has_lid = (
        is_convecting
        & ((T_i - T_surf) > NO_LID_DT_THRESHOLD)
        & (d_lid_raw > NO_LID_THICKNESS_THRESHOLD)
    )

    def if_lid(val):
        return xp.where(has_lid, val, zero_or_nan)

    d_lid = if_lid(d_lid_raw)

    dT_dr = xp.abs(xp.gradient(T, r))
    dr_min = xp.min(xp.abs(r[1:] - r[:-1]))
    dTdr_floor = 1.0 / xp.maximum(dr_min, 1.0)
    dTdr_safe = xp.maximum(dT_dr, dTdr_floor)
    dTdr_lid_base = xp.sum(W_base_tilde * dTdr_safe)
    delta_rh = dT_rh / xp.maximum(dTdr_lid_base, 1e-12)
    delta_rh = xp.minimum(delta_rh, r[-1] - r[0])

    if unyielded_velocity is not None:
        T_v = SOFTMAX_VELOCITY_SCALE
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
        v_i = xp.where(is_convecting, xp.maximum(v_i_raw, 0.0) * w_active, zero_or_nan)
    else:
        v_i = zero_or_nan
        w_upper = xp.zeros_like(r) + zero_or_nan

    d_conv = xp.maximum(r[-1] - r[0], 1.0)

    rho_i = _weighted_mean(density)
    g_i = _weighted_mean(gravity)
    alpha_i = _weighted_mean(thermal_expansivity)
    kappa_i = _weighted_mean(thermal_diffusivity)
    DeltaT_conv = xp.maximum(T_i - T_surf, 1e-6)

    # Convective interior velocity and lid driving stress (Foley & Bercovici 2014, eqs. 26, 28).
    Ra_eff = if_lid(
        (rho_i * g_i * alpha_i * DeltaT_conv * (d_conv**3))
        / (kappa_i * xp.maximum(eta_i, 1e-30))
    )
    Ra_rh = if_lid((Ra_eff * a_rh_fb) / xp.maximum(theta, 1e-6))
    v_m = xp.where(
        xp.isnan(Ra_rh),
        xp.nan,
        xp.where(
            Ra_rh > 0.0,
            (kappa_i / d_conv) * C4 * (xp.maximum(Ra_rh, 1.0e-30) ** FB2014_VELOCITY_EXPONENT),
            zero_or_nan,
        ),
    )

    tau_d = if_lid((FB2014_STRESS_FACTOR * eta_i * v_m / d_conv) * w_active)
    tau_d_lid = if_lid((FB2014_STRESS_FACTOR * eta_i * v_m / d_geom) * w_active)
    tau_buoy = if_lid(rho_i * g_i * alpha_i * dT_rh * delta_rh * w_active)
    tau_d_over_tau_buoy = if_lid(tau_d / xp.maximum(tau_buoy, 1e-30))

    tau_y_lid = compute_yield_stress(
        P_lid_base,
        yield_stress_c=params.yield_stress_c,
        yield_stress_mu=params.yield_stress_mu,
        yield_stress_max=params.yield_stress_max,
        xp=xp,
    )

    ratio = tau_d / xp.maximum(tau_y_lid, COHESION_FLOOR)
    w_y = if_lid(0.5 * (1.0 + xp.tanh((ratio - 1.0) / LID_REGIME_LABEL_WIDTH)))
    lid_cell_count = xp.sum(w_lid)
    w_solid_surf = 0.5 * (
        1.0 - xp.tanh((phi[-1] - SOLIDUS_MELT_FRACTION_THRESHOLD) / SOLIDUS_MELT_FRACTION_WIDTH)
    )
    w_has_lid = 0.5 * (1.0 + xp.tanh((lid_cell_count - 0.5) / 0.1)) * w_solid_surf
    lid_regime = if_lid(w_active * w_has_lid * (1.0 + w_y))

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
        'has_lid': has_lid,
        'lid_regime': lid_regime,
        'lid_cell_count': lid_cell_count,
        'eta_i': eta_i,
        'mu_i': eta_i,
        'eta_solid_arrhenius': eta_arr_i,
    }


def compute_effective_viscosity(
    eta_diff: FloatOrArray,
    tau_d: FloatOrArray | None = None,
    tau_y_lid: FloatOrArray | None = None,
    stress_closure_mode: str = 'lid',
    unyielded_velocity: FloatOrArray | None = None,
    mixing_length: FloatOrArray | None = None,
    tau_y_profile: FloatOrArray | None = None,
    w_lid: FloatOrArray | None = None,
    strain_rate: FloatOrArray | None = None,
    tau_y: FloatOrArray | None = None,
    eta_i: FloatOrArray | None = None,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute effective dynamic viscosity with boundary-layer or local stress closure.

    Both boundary-layer lid mode and local stress closure mode use the minimum closure
    of diffusion creep and plastic yield viscosity per Moresi & Solomatov (1998):
        eta_eff = min(eta_diff, eta_y)

    In boundary-layer lid mode, plastic yield viscosity is evaluated at the interior
    convective strain-rate invariant scale D_conv = 2 * v_m / d_conv = tau_d / eta_i:
        eta_y = eta_i * (tau_y_lid / tau_d)
        eta_eff = eta_diff                              for tau_d <= tau_y_lid
        eta_eff = min(eta_diff, eta_i * tau_y_lid / tau_d)  for tau_d > tau_y_lid
    When ``eta_i < eta_diff``, ``eta_eff`` jumps at ``tau_d = tau_y_lid`` from ``eta_diff``
    for ``tau_d <= tau_y_lid`` to ``eta_i * (tau_y_lid / tau_d)`` for ``tau_d > tau_y_lid``.
    ``eta_i``, ``tau_d``, and ``tau_y_lid`` are required in ``'lid'`` mode.

    In local stress closure mode:
        eta_eff = min(eta_diff, eta_y)
        eta_y = max(tau_y, COHESION_FLOOR) / (2 * strain_rate)

    References
    ----------
    Moresi & Solomatov (1998), p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x
        (viscosity switch / minimum form eta_eff = min(eta_creep, eta_yield))
    Foley & Bercovici (2014), p. 588, eq. 28, doi:10.1093/gji/ggu275
        (convective driving shear stress tau_d = 2 * mu_i * v_m / d)
    Tackley (2000), eq. 8, p. 4, doi:10.1029/2000GC000036
        (viscosity cap / minimum form eta_eff = min[eta, sigma_y / (2 e_dot)])
    Foley & Becker (2009), eqs. 7-8, p. 3, doi:10.1029/2009GC002378
        (alternative harmonic mean formulation)
    """
    _require_x64_if_jax(xp)
    if stress_closure_mode not in ('lid', 'local'):
        hint = (
            ". 'global' mode is unsupported; use 'lid'."
            if stress_closure_mode == 'global'
            else ''
        )
        raise ValueError(
            f"Unknown stress_closure_mode {stress_closure_mode!r}; expected 'lid' or 'local'{hint}"
        )
    if stress_closure_mode == 'local':
        if any(x is not None for x in (tau_d, tau_y_lid, w_lid, eta_i)):
            raise ValueError(
                "tau_d, tau_y_lid, w_lid, and eta_i are not allowed when stress_closure_mode='local'"
            )
        if strain_rate is None:
            if unyielded_velocity is None or mixing_length is None:
                raise ValueError(
                    "mixing_length and unyielded_velocity required for 'local' mode"
                )
            sr = compute_strain_rate_local(unyielded_velocity, mixing_length, xp=xp)
        else:
            sr = xp.asarray(strain_rate, dtype=float)
        if tau_y is not None and tau_y_profile is not None:
            raise ValueError("Provide either tau_y or tau_y_profile in 'local' mode, not both")
        ty_raw = tau_y if tau_y is not None else tau_y_profile
        if ty_raw is None:
            raise ValueError("tau_y or tau_y_profile required for 'local' mode")
        ty = xp.asarray(ty_raw, dtype=float)
        eta_d = xp.asarray(eta_diff, dtype=float)
        safe_ty = xp.maximum(ty, COHESION_FLOOR)
        tau_y_term = safe_ty / (2.0 * xp.where(sr > 0.0, sr, 1.0))
        is_unyielded = (sr <= 0.0) | xp.isnan(tau_y_term)
        res = xp.minimum(eta_d, tau_y_term)
        has_nan_local = xp.isnan(eta_d) | xp.isnan(ty) | xp.isnan(sr)
        return xp.where(has_nan_local, xp.nan, xp.where(is_unyielded, eta_d, res))

    if any(
        x is not None
        for x in (strain_rate, tau_y, tau_y_profile, unyielded_velocity, mixing_length)
    ):
        raise ValueError(
            'strain_rate, tau_y, tau_y_profile, unyielded_velocity, and mixing_length '
            "are not allowed when stress_closure_mode='lid'"
        )
    if eta_i is None:
        raise ValueError("eta_i is required when stress_closure_mode='lid'")
    if tau_d is None or tau_y_lid is None:
        raise ValueError("tau_d and tau_y_lid are required when stress_closure_mode='lid'")

    eta_d = xp.asarray(eta_diff, dtype=float)
    eta_i_arr = xp.asarray(eta_i, dtype=float)

    # Moresi & Solomatov (1998) Eq. 13-14 min closure at convective strain rate D_conv = tau_d / eta_i:
    td = xp.asarray(tau_d, dtype=float)
    ty_lid = xp.maximum(xp.asarray(tau_y_lid, dtype=float), COHESION_FLOOR)
    ratio = xp.where(td > 0.0, ty_lid / xp.maximum(td, 1e-30), 1.0)
    ratio = xp.where(xp.isinf(ty_lid), 1.0, ratio)
    eta_eff_val = xp.where(ratio >= 1.0, eta_d, xp.minimum(eta_d, eta_i_arr * ratio))
    has_nan_lid = xp.isnan(eta_d) | xp.isnan(eta_i_arr) | xp.isnan(td) | xp.isnan(ty_lid)
    eta_eff_val = xp.where(has_nan_lid, xp.nan, eta_eff_val)

    if w_lid is not None:
        wl = xp.asarray(w_lid, dtype=float)
        try:
            wl_np = np.asarray(wl, dtype=float)
        except Exception:
            wl_np = None
        if wl_np is not None and np.any(np.isinf(wl_np) | (wl_np < 0.0) | (wl_np > 1.0)):
            raise ValueError('w_lid must be in [0, 1]')
        log_diff = xp.log10(xp.maximum(eta_d, 1e-30))
        log_eff = xp.log10(xp.maximum(eta_eff_val, 1e-30))
        log_solid = wl * log_eff + (1.0 - wl) * xp.where(wl == 1.0, 0.0, log_diff)
        res = 10.0**log_solid
        blended = xp.where((wl == 0.0) | (eta_eff_val == eta_d), eta_d, res)
        return xp.where(has_nan_lid | xp.isnan(wl) | (wl < 0.0) | (wl > 1.0), xp.nan, blended)

    return eta_eff_val


def stress_closure(
    mode: str,
    viscous_velocity: FloatOrArray,
    mixing_length: FloatOrArray | None = None,
    eps: float = 1.0e-15,
    xp: Any = np,
) -> FloatOrArray:
    r"""Calculate strain rate dispatching on stress closure mode."""
    _require_x64_if_jax(xp)
    if mode == 'local':
        if mixing_length is None:
            raise ValueError("mixing_length must be provided when mode='local'")
        return compute_strain_rate_local(viscous_velocity, mixing_length, eps=eps, xp=xp)
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
