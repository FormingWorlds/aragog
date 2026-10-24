"""Stagnant lid boundary layer indicators and effective viscosity closures.

Intentional non-smooth switches retained for boolean diagnostics and physical bounds
include ``is_convecting``, ``has_lid``, the ``is_convecting`` gate on ``v_i``, the
``[0, 1]`` clip on ``f``, ``f_int_max = max(f * interior_node_mask)``,
``max(T_i, 1.0)``, ``max(H_T_i, 1.0e-6)``, ``DeltaT_conv = max(T_i - T_surf, 1.0e-6)``,
``max(theta, 1.0e-6)``, ``max(Ra_rh_raw, 1.0e-30)``, the elementwise maxima in
``w_interior`` and ``w_hot_sol``, ``dTdr_floor`` with ``max(dr_min, 1.0)``, the
``r[-1] - r[0]`` cap on ``delta_rh``, the ``COHESION_FLOOR`` yield-stress clamp,
``if_lid`` (``where(w_lid_present >= 1 - 1e-12)``), ``d_geom``, the max in ``d_lid_raw``,
``max(w_conv - w_conv_0, 0)``, ``f_int_scale``, the ``v_m_raw`` branches, and the
``1.0e-30`` denominators.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from aragog.rheology import (
    COHESION_FLOOR,
    R_GAS,
    FloatOrArray,
    _clamp_pressure,
    _is_non_real,
    _require_x64_if_jax,
    _softmax2,
    _softmin2,
    _validate_inputs,
    compute_arrhenius_enthalpy,
    compute_arrhenius_viscosity,
    compute_strain_rate_local,
    compute_yield_stress,
)

__all__ = [
    'COHESION_FLOOR',
    'CONV_ACTIVE_GATE_WIDTH',
    'CONV_SUM_GATE_WIDTH',
    'FB2014_STRESS_FACTOR',
    'FB2014_VELOCITY_EXPONENT',
    'F_TOT_FLOOR_FRAC',
    'LID_REGIME_LABEL_WIDTH',
    'LOCAL_CAP_LOG10_WIDTH',
    'LOCAL_STRAIN_RATE_FLOOR',
    'NO_LID_DT_THRESHOLD',
    'NO_LID_DT_WIDTH',
    'NO_LID_THICKNESS_THRESHOLD',
    'NO_LID_THICKNESS_WIDTH',
    'SCALE_FLOOR_WIDTH',
    'SCALE_MASK_REL_WIDTH',
    'SMOOTH_MAX_REL_WIDTH',
    'SOFTMAX_VELOCITY_SCALE',
    'SOLIDUS_MELT_FRACTION_THRESHOLD',
    'SOLIDUS_MELT_FRACTION_WIDTH',
    'YIELD_BLEND_RATIO_WIDTH',
    'compute_stagnant_lid_state',
    'compute_effective_viscosity',
    'stress_closure',
]

# Foley & Bercovici (2014) scaling parameters (Eq. 26 and Eq. 28).
FB2014_VELOCITY_EXPONENT: float = 2.0 / 3.0
FB2014_STRESS_FACTOR: float = 2.0

# Minimum thermal contrast [K] and thickness [m] required for a physical conductive lid.
NO_LID_DT_THRESHOLD: float = 1.0e-3
NO_LID_DT_WIDTH: float = 2.5e-4
NO_LID_THICKNESS_THRESHOLD: float = 1.0
NO_LID_THICKNESS_WIDTH: float = 0.25

# Smooth transition widths for convective activity and boundary-weight gates.
CONV_ACTIVE_GATE_WIDTH: float = 0.0025
CONV_SUM_GATE_WIDTH: float = 2.5e-7

# Smooth softplus floor and cap widths for cell/rheological temperature scales.
SCALE_FLOOR_WIDTH: float = 0.005
SCALE_MASK_REL_WIDTH: float = 0.005

# Relative smoothing width for Boltzmann-weighted smooth maximum and softplus total-flux floor.
SMOOTH_MAX_REL_WIDTH: float = 0.02
F_TOT_FLOOR_FRAC: float = 0.1

# Smooth yield transition widths and local strain-rate regularization floor.
YIELD_BLEND_RATIO_WIDTH: float = 0.025
LOCAL_STRAIN_RATE_FLOOR: float = 1.0e-100
LOCAL_CAP_LOG10_WIDTH: float = 0.01

# Transition smoothing width that sets the lid_regime diagnostic label and w_has_lid.
LID_REGIME_LABEL_WIDTH: float = 0.1

# Melt fraction threshold marking solidus crossing / surface crust solidification.
# This 1% threshold is distinct from phi_rheo (~0.4), the rheological transition.
SOLIDUS_MELT_FRACTION_THRESHOLD: float = 0.01
SOLIDUS_MELT_FRACTION_WIDTH: float = 0.002

# Soft-maximum smoothing scale [m/s] (~0.1 mm/yr) for the upper-mantle velocity diagnostic v_i.
SOFTMAX_VELOCITY_SCALE: float = 3.17e-12


def _smooth_max(v: Any, xp: Any = np, *, mask: Any | None = None) -> Any:
    """Boltzmann-weighted smooth maximum with scale ``SMOOTH_MAX_REL_WIDTH * rms(v)``."""
    v_arr = xp.asarray(v, dtype=float)
    v_fin = xp.where(xp.isfinite(v_arr), v_arr, 0.0)
    if mask is None:
        m = xp.ones_like(v_fin)
    else:
        m = xp.where(xp.asarray(mask, dtype=float) > 0.0, 1.0, 0.0)
    m_count = xp.maximum(xp.sum(m), 1.0)
    v_rms = xp.sqrt(xp.sum(m * (v_fin**2)) / m_count + 1.0e-300)
    sigma = xp.maximum(SMOOTH_MAX_REL_WIDTH * v_rms, 1.0e-150)
    v_masked = xp.where(m > 0.0, v_fin, -xp.inf)
    v_max = xp.max(v_masked)
    v_max_safe = xp.where(xp.isfinite(v_max), v_max, 0.0)
    diff = v_fin - v_max_safe
    u = xp.maximum(diff / sigma, -500.0)
    weights = m * xp.exp(u)
    smax = v_max_safe + xp.sum(weights * diff) / xp.maximum(xp.sum(weights), 1.0e-300)
    return xp.where(xp.any(~xp.isfinite(v_arr)), xp.max(v_arr), smax)


def _melt_ramp(
    phi: Any, centre: Any, width: Any, xp: Any = np, *, compact: bool = False
) -> Any:
    """Evaluate melt-fraction indicator ramp (C^2 quintic when compact=True, else baseline-subtracted tanh)."""
    phi_arr = xp.asarray(phi, dtype=float)
    if compact:
        t = xp.clip((phi_arr - (centre - width)) / (2.0 * width), 0.0, 1.0)
        return (t**3) * (10.0 - 15.0 * t + 6.0 * t**2)
    w_raw = 0.5 * (1.0 + xp.tanh((phi_arr - centre) / width))
    w_0 = 0.5 * (1.0 + xp.tanh(-centre / width))
    return xp.clip((w_raw - w_0) / xp.maximum(1.0 - w_0, 1.0e-12), 0.0, 1.0)


def _boundary_weights(
    w: Any,
    xp: Any,
    *,
    fallback: Any | None = None,
    mask: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Smooth weights of the shallowest crossing of ``w``, scanned from the surface down.

    Evaluates ``W_k = w_k * prod_{j > k} (1 - w_j)`` from the outermost node ``r[-1]``
    inward and returns ``(W_tilde, W_sum, w_base_weight)``; ``W_tilde`` relaxes to
    ``fallback`` (or to the bottom node at index 0 when ``fallback`` is None) when
    ``_smooth_max(w)`` vanishes.
    """
    w_surf = w[::-1]
    omw = xp.concatenate([xp.ones(1, dtype=w.dtype), 1.0 - w_surf[:-1]])
    W = (w_surf * xp.cumprod(omw))[::-1]
    W_sum = xp.sum(W)
    w_base_weight = xp.tanh(_smooth_max(w, xp, mask=mask) / 1.0e-6)
    n = w.shape[0]
    W_fallback = xp.where(xp.arange(n) == 0, 1.0, 0.0) if fallback is None else fallback
    W_tilde = (
        w_base_weight * (W / xp.maximum(W_sum, 1.0e-12)) + (1.0 - w_base_weight) * W_fallback
    )
    return W_tilde, W_sum, w_base_weight


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
        Radial pressure profile P [Pa] (non-negative; negative round-off
        excursions down to ``-PRESSURE_ROUNDOFF_TOLERANCE`` = -1.0 Pa are
        clamped to 0.0 Pa).
    convective_flux : array-like
        Convective heat flux profile F_conv [W/m^2].
    total_flux : array-like
        Total heat flux profile F_tot [W/m^2].
    solidus_temperature : float or array-like or None
        Solidus temperature profile T_sol [K]. When provided, the solidus
        crossing indicators ``w_hot_sol`` and ``w_hot_sol_mask`` take the
        elementwise maximum of the thermal ``tanh((T - T_sol) / scale)`` ramp
        and the melt-fraction ramp centered at
        ``SOLIDUS_MELT_FRACTION_THRESHOLD`` (0.01). When None, they use the
        melt-fraction ramp alone.
    melt_fraction : array-like
        Melt mass fraction profile phi [-].
    params : SolidRheologyParams
        Rheology parameter set.
    viscosity_solid : float
        Reference solid mantle viscosity [Pa s] such that
        ``viscosity_solid * f_water`` equals ``eta_diff`` at
        ``(params.arrhenius_t_ref, P = 0)`` for the linear solidus; when a
        ``solidus_temperature`` table is provided, ``viscosity_solid`` is
        referenced to ``params.t_sol_0``, so the table ``eta_diff(T, 0)``
        scales relative to the linear-solidus ``eta_diff(T, 0)`` at the same
        ``T`` by ``exp(g_hom * (T_sol(0) - params.t_sol_0) / T)`` (a ratio
        of 2.752, or 2.75x, at 1600 K for 1464.8 K against 1400 K).
    phi_rheo : float
        Critical melt fraction for rheological transition [-].
    phi_width : float
        Transition width for rheological melt fraction indicator [-].
    unyielded_velocity : float or array-like, optional
        Unyielded convective velocity profile v [m/s].
    viscosity_mixture : float or array-like, optional
        Effective mixture viscosity profile [Pa s] of the convecting layer (in mush,
        the blended two-phase viscosity used by the solver). When provided, mu_i
        blends smoothly via ``w_weight = tanh(_smooth_max(w_conv_int) / 1e-6)``
        between the W_tilde-weighted log mean of ``viscosity_mixture`` and the
        solid Arrhenius viscosity ``eta_arr_i``; otherwise falls back to ``eta_arr_i``.
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
        Dictionary of boundary layer state diagnostics, including
        ``tau_d = w_lid_present * w_active * 2 * eta_i * v_m_raw / d = w_active * 2 * eta_i * v_m / d``
        and the lid-thickness-scaled stress diagnostic
        ``tau_d_lid = w_lid_present * w_active * 2 * eta_i * v_m_raw / max(r_top - r_lid_base, 1e-6 m)``
        (gated smoothly by ``w_lid_present = w_d * w_T * w_act_gate * w_sum_gate * w_weight``;
        because ``w_d == 0`` at ``d_lid_raw <= 0`` and ``has_lid`` turns on at
        ``d_lid_raw > NO_LID_THICKNESS_THRESHOLD`` of 1.0 m, the 1e-6 m floor is
        inactive whenever ``has_lid`` is True, returns 0.0 in non-lid columns,
        and ``tau_d_lid / tau_d = d / d_geom`` scales as ``1 / d_lid`` down to 1 m).
        ``has_lid`` is False when any input is NaN while numeric diagnostics propagate NaN.

    Raises
    ------
    ValueError
        If ``viscosity_solid``, ``phi_rheo``, or ``phi_width`` is None; if
        scalar parameters (``viscosity_solid``, ``phi_rheo``, ``phi_width``,
        ``a_rh_fb``, ``C4``) are non-scalar; if ``radii``
        is not a 1-D array of at least 3 nodes; if any profile input does not
        match ``radii`` in 1-D length; if any input is non-real (such as a
        boolean, string, or complex value); or, on the eager NumPy path
        (``xp is np``), if ``radii`` is not strictly increasing or has ``inf``
        values (``NaN`` propagates), if any input is infinite, if
        ``viscosity_solid``, ``phi_width``, ``a_rh_fb``, ``C4``,
        ``temperature``, ``solidus_temperature``,
        ``viscosity_mixture``, ``density``, ``gravity``,
        ``thermal_expansivity``, or ``thermal_diffusivity`` is non-positive, if
        ``pressure`` is negative, or if ``phi_rheo`` or ``melt_fraction`` lies
        outside ``[0, 1]``. Under ``jax.jit``, traced out-of-range numeric
        values (including non-increasing ``radii``) propagate ``NaN`` through
        ``zero_or_nan``.

    Notes
    -----
    See ``docs/Explanations/solid_state_convection.md`` for model formulation
    and derivations.

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
    ``tau_buoy``, ``tau_d_over_tau_buoy``, ``w_y``, ``lid_regime``) are smoothly
    gated by ``w_lid_present = w_d * w_T * w_act_gate * w_sum_gate * w_weight``
    (where ``w_d`` and ``w_sum_gate`` are $C^0$ with a one-sided slope at the
    domain edge ``d_lid_raw >= 0`` and ``W_sum >= 0``, and ``w_T`` uses the $C^1$
    positive part ``where(dT_lid > 0, dT_lid * tanh(dT_lid / (0.3 * NO_LID_DT_WIDTH)), 0)``
    so ``w_lid_present`` vanishes identically when ``d_lid_raw <= 0``,
    ``T_i - T_surf <= 0``, or ``W_sum <= 0``) and ``has_lid`` is False, while
    ``T_i``, ``eta_i``, ``mu_i``, and ``theta`` serve as interior fallback
    diagnostics and all solver-facing outputs vary continuously across uniform
    convective flux fractions ``f`` in ``[0, 1e-3]``. Intentional non-smooth
    switches retained for boolean diagnostics and physical bounds include
    ``is_convecting``, ``has_lid``, the ``is_convecting`` gate on the diagnostic
    ``v_i``, the ``[0, 1]`` clip on ``f``, ``f_int_max = max(f * interior_node_mask)``,
    ``max(T_i, 1.0)``, ``max(H_T_i, 1.0e-6)``,
    ``DeltaT_conv = max(T_i - T_surf, 1.0e-6)``, ``max(theta, 1.0e-6)``,
    ``max(Ra_rh_raw, 1.0e-30)``, the elementwise maxima in ``w_interior`` and
    ``w_hot_sol``, the temperature-gradient floor ``dTdr_floor`` with
    ``max(dr_min, 1.0)`` in ``delta_rh``, the domain-thickness cap ``r[-1] - r[0]``
    on ``delta_rh``, the ``COHESION_FLOOR`` clamp on yield stress,
    ``if_lid`` (``where(w_lid_present >= 1 - 1e-12)``), ``d_geom``, the max in
    ``d_lid_raw``, ``max(w_conv - w_conv_0, 0)``, ``f_int_scale``, the
    ``v_m_raw`` branches, and the ``1.0e-30`` denominators (while the
    5-node local interior RMS ``|F_tot|`` floor
    ``F_tot_floor = _softmax2(F_TOT_FLOOR_FRAC * F_loc_rms, 1.0e-12, 0.01 * 1.0e-12)``
    replaces boundary nodes ``k = 0`` and ``k = N - 1`` by their adjacent interior
    nodes ``F_tot[1]`` and ``F_tot[N - 2]`` before 2-node edge replication, is
    continuous in every ``F_k``, and differs from a scalar ``1.0e-12`` floor only at
    nodes where ``F_conv > 0`` and ``|F_tot| < 0.14 * F_loc_rms`` when
    ``0.1 * F_loc_rms >> 1.0e-12 W/m2``).

    The lid-base locator ``W_base_tilde = _boundary_weights(w_interior, xp)[0]`` scans
    ``w_interior`` (smoothed over the local cell temperature step
    ``scale = lid_mask_width_cells * _softmax2(|grad T|, 1.0 K, SCALE_FLOOR_WIDTH)``) from
    the outermost node ``r[-1]`` inward, whereas the nodal lid mask
    ``w_lid = (1 - w_interior_mask) * w_active`` uses the smooth capped width
    ``scale_mask = _softmin2(scale, scale_cap, SCALE_MASK_REL_WIDTH * scale_cap)`` with
    ``scale_cap = lid_mask_width_cells * _softmax2(dT_rh, 1.0 K, SCALE_FLOOR_WIDTH)``.
    The solidus crossing indicators ``w_hot_sol`` and ``w_hot_sol_mask`` evaluate the
    compactly supported $C^2$ quintic melt-fraction ramp ``w_sol_phi`` on
    ``[SOLIDUS_MELT_FRACTION_THRESHOLD - SOLIDUS_MELT_FRACTION_WIDTH,
    SOLIDUS_MELT_FRACTION_THRESHOLD + SOLIDUS_MELT_FRACTION_WIDTH] = [0.008, 0.012]``,
    so that ``w_sol_phi == 0`` for ``phi <= 0.008`` and ``w_sol_phi == 1`` for
    ``phi >= 0.012``, giving ``has_lid = False`` and ``d_lid = 0`` for any column
    with ``phi >= 0.012`` at every node across all mesh resolutions ``N``.
    """
    _require_x64_if_jax(xp)
    if viscosity_solid is None or phi_rheo is None or phi_width is None:
        raise ValueError('viscosity_solid, phi_rheo, and phi_width must be provided')
    if _is_non_real(radii):
        raise ValueError(f'radii must be a real number or array, got {radii!r}')
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
    for name, val, allow_scalar in (
        ('temperature', temperature, False),
        ('pressure', pressure, False),
        ('convective_flux', convective_flux, False),
        ('total_flux', total_flux, False),
        ('melt_fraction', melt_fraction, False),
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
        if _is_non_real(val):
            raise ValueError(f'{name} must be a real number or array, got {val!r}')
        a = xp.asarray(val, dtype=float)
        if not ((allow_scalar and a.ndim == 0) or (a.ndim == 1 and a.shape[0] == r.shape[0])):
            raise ValueError(f'{name} must have the same 1-D length as radii ({r.shape[0]})')

    invalid_inputs = _validate_inputs(
        (
            ('viscosity_solid', viscosity_solid, 'pos', True),
            ('phi_width', phi_width, 'pos', True),
            ('phi_rheo', phi_rheo, 'unit', True),
            ('a_rh_fb', a_rh_fb, 'pos', True),
            ('C4', C4, 'pos', True),
            ('radii', radii, 'finite', False),
            ('temperature', temperature, 'pos', False),
            ('pressure', pressure, 'pressure', False),
            ('convective_flux', convective_flux, 'finite', False),
            ('total_flux', total_flux, 'finite', False),
            ('melt_fraction', melt_fraction, 'unit', False),
            ('solidus_temperature', solidus_temperature, 'pos', False),
            ('unyielded_velocity', unyielded_velocity, 'finite', False),
            ('viscosity_mixture', viscosity_mixture, 'pos', False),
            ('density', density, 'pos', False),
            ('gravity', gravity, 'pos', False),
            ('thermal_expansivity', thermal_expansivity, 'pos', False),
            ('thermal_diffusivity', thermal_diffusivity, 'pos', False),
        ),
        xp,
        reduce_any=True,
    )
    T = xp.asarray(temperature, dtype=float)
    P = _clamp_pressure(pressure, xp)
    F_conv = xp.asarray(convective_flux, dtype=float)
    F_tot = xp.asarray(total_flux, dtype=float)
    phi = xp.asarray(melt_fraction, dtype=float)

    # 5-node local interior RMS floor: replace boundary nodes (0, N-1) with (1, N-2)
    # and replicate 2 edge nodes; differs from 1e-12 only where F_conv > 0 and |F_tot| < 0.14 * F_loc_rms.
    n_nodes = r.shape[0]
    F_pad = xp.concatenate([F_tot[1:2], F_tot[1:-1], F_tot[-2:-1]]) if n_nodes >= 3 else F_tot
    F_ext = xp.concatenate([F_pad[:1], F_pad[:1], F_pad, F_pad[-1:], F_pad[-1:]])
    F_loc_rms = xp.sqrt(
        0.2
        * (
            F_ext[:-4] ** 2
            + F_ext[1:-3] ** 2
            + F_ext[2:-2] ** 2
            + F_ext[3:-1] ** 2
            + F_ext[4:] ** 2
        )
        + 1.0e-300
    )
    F_tot_floor = _softmax2(F_TOT_FLOOR_FRAC * F_loc_rms, 1.0e-12, 0.01 * 1.0e-12, xp)
    F_tot_abs = xp.sqrt(F_tot**2 + 1.0e-60)
    F_tot_safe = _softmax2(F_tot_abs, F_tot_floor, 0.01 * F_tot_floor, xp)
    f = xp.clip(xp.maximum(F_conv, 0.0) / F_tot_safe, 0.0, 1.0)
    invalid_scalar = xp.any((r[1:] - r[:-1]) <= 0.0)
    zero_or_nan = xp.where(
        invalid_inputs | invalid_scalar | xp.any(~xp.isfinite(f)), xp.nan, 0.0
    )

    w_conv = 0.5 * (1.0 + xp.tanh((f - params.interior_flux_fraction) / 0.01)) + zero_or_nan
    idx = xp.arange(r.shape[0])
    interior_node_mask = xp.where((idx > 0) & (idx < r.shape[0] - 1), 1.0, 0.0)
    w_conv_0 = 0.5 * (1.0 + xp.tanh(-params.interior_flux_fraction / 0.01))
    w_conv_int = xp.maximum(w_conv - w_conv_0, 0.0) * interior_node_mask + zero_or_nan
    interior_fallback = xp.where(idx == 1, 1.0, 0.0)
    W_conv_tilde, W_sum, w_weight = _boundary_weights(
        w_conv_int, xp, fallback=interior_fallback, mask=interior_node_mask
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
        t_sol_0=params.t_sol_0,
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
        t_sol_0=params.t_sol_0,
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
    eta_i_fin = xp.where(xp.isfinite(eta_i), eta_i, 1.0)
    eta_contrast = xp.where(
        xp.isfinite(eta_surf) & xp.isfinite(eta_i),
        eta_surf / xp.maximum(eta_i_fin, 1e-30),
        xp.nan,
    )

    if params.lid_base_mode == 'fixed':
        T_lid_iso = params.lid_base_temperature + zero_or_nan
    else:
        T_lid_iso = T_i - params.lid_contrast_coeff * dT_rh

    gradT_abs = xp.sqrt(xp.gradient(T) ** 2 + 1.0e-30)
    scale = params.lid_mask_width_cells * _softmax2(gradT_abs, 1.0, SCALE_FLOOR_WIDTH, xp)
    scale_cap = params.lid_mask_width_cells * _softmax2(dT_rh, 1.0, SCALE_FLOOR_WIDTH, xp)
    scale_mask = _softmin2(scale, scale_cap, SCALE_MASK_REL_WIDTH * scale_cap, xp)

    w_hot_iso = 0.5 * (1.0 + xp.tanh((T - T_lid_iso) / scale))
    w_hot_iso_mask = 0.5 * (1.0 + xp.tanh((T - T_lid_iso) / scale_mask))
    w_sol_phi = _melt_ramp(
        phi,
        SOLIDUS_MELT_FRACTION_THRESHOLD,
        SOLIDUS_MELT_FRACTION_WIDTH,
        xp,
        compact=True,
    )
    if T_sol_arr is not None:
        w_sol_T = 0.5 * (1.0 + xp.tanh((T - T_sol_arr) / scale))
        w_sol_T_mask = 0.5 * (1.0 + xp.tanh((T - T_sol_arr) / scale_mask))
        w_hot_sol = xp.maximum(w_sol_T, w_sol_phi)
        w_hot_sol_mask = xp.maximum(w_sol_T_mask, w_sol_phi)
    else:
        w_hot_sol = w_sol_phi
        w_hot_sol_mask = w_sol_phi
    w_hot_phi = _melt_ramp(phi, phi_rheo, phi_width, xp, compact=False)

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
    dT_lid = xp.sum(W_tilde * (T - T_surf))
    dT_pos = xp.where(
        dT_lid > 0.0,
        dT_lid * xp.tanh(dT_lid / (0.3 * NO_LID_DT_WIDTH)),
        0.0,
    )
    w_d = _melt_ramp(
        d_lid_raw, NO_LID_THICKNESS_THRESHOLD, NO_LID_THICKNESS_WIDTH, xp, compact=False
    )
    w_T = _melt_ramp(dT_pos, NO_LID_DT_THRESHOLD, NO_LID_DT_WIDTH, xp, compact=False)
    w_act_gate = 0.5 * (1.0 + xp.tanh((w_active - 0.05) / CONV_ACTIVE_GATE_WIDTH))
    w_sum_gate = _melt_ramp(W_sum, 1.0e-6, CONV_SUM_GATE_WIDTH, xp, compact=False)
    w_lid_present = w_d * w_T * w_act_gate * w_sum_gate * w_weight

    def if_lid(val):
        return xp.where(w_lid_present >= 1.0 - 1.0e-12, val, w_lid_present * val) + zero_or_nan

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

        v_abs = xp.abs(xp.asarray(unyielded_velocity, dtype=float))
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
    DeltaT_conv = xp.maximum(T_i - T_surf, 1.0e-6)

    # Convective interior velocity and lid driving stress (Foley & Bercovici 2014, eqs. 26, 28).
    Ra_eff_raw = (rho_i * g_i * alpha_i * DeltaT_conv * (d_conv**3)) / (
        kappa_i * xp.maximum(eta_i, 1e-30)
    )
    Ra_rh_raw = (Ra_eff_raw * a_rh_fb) / xp.maximum(theta, 1.0e-6)
    Ra_eff = if_lid(Ra_eff_raw)
    Ra_rh = if_lid(Ra_rh_raw)
    v_m_raw = xp.where(
        xp.isnan(Ra_rh_raw),
        xp.nan,
        xp.where(
            Ra_rh_raw > 0.0,
            (kappa_i / d_conv)
            * C4
            * (xp.maximum(Ra_rh_raw, 1.0e-30) ** FB2014_VELOCITY_EXPONENT),
            zero_or_nan,
        ),
    )
    v_m = if_lid(v_m_raw)

    tau_d_raw = (FB2014_STRESS_FACTOR * eta_i_fin * v_m_raw / d_conv) * w_active
    tau_d_lid_raw = (FB2014_STRESS_FACTOR * eta_i_fin * v_m_raw / d_geom) * w_active
    tau_buoy_raw = rho_i * g_i * alpha_i * dT_rh * delta_rh * w_active
    tau_d = if_lid(tau_d_raw)
    tau_d_lid = if_lid(tau_d_lid_raw)
    tau_buoy = if_lid(tau_buoy_raw)
    tau_d_over_tau_buoy = if_lid(tau_d_raw / xp.maximum(tau_buoy_raw, 1e-30))

    tau_y_lid = compute_yield_stress(
        P_lid_base,
        yield_stress_c=params.yield_stress_c,
        yield_stress_mu=params.yield_stress_mu,
        yield_stress_max=params.yield_stress_max,
        xp=xp,
    )

    ratio = tau_d_raw / xp.maximum(tau_y_lid, COHESION_FLOOR)
    w_y = if_lid(0.5 * (1.0 + xp.tanh((ratio - 1.0) / LID_REGIME_LABEL_WIDTH)))
    lid_cell_count = xp.sum(w_lid)
    w_solid_surf = 1.0 - _melt_ramp(
        phi[-1],
        SOLIDUS_MELT_FRACTION_THRESHOLD,
        SOLIDUS_MELT_FRACTION_WIDTH,
        xp,
        compact=True,
    )
    w_has_lid = (
        0.5 * (1.0 + xp.tanh((lid_cell_count - 0.5) / LID_REGIME_LABEL_WIDTH)) * w_solid_surf
    )
    lid_regime = if_lid(w_active * w_has_lid * (1.0 + w_y))

    if xp is np and not bool(np.any(np.isnan(zero_or_nan))):
        assert xp.all((w_lid_present >= 0.0) & (w_lid_present <= 1.0))
        assert xp.all(xp.isfinite(eta_i))

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

    Both boundary-layer lid mode and local stress closure mode use a smooth
    minimum closure of diffusion creep and plastic yield viscosity per
    Moresi & Solomatov (1998):

    In boundary-layer lid mode, plastic yield viscosity is evaluated at the interior
    convective strain-rate invariant scale D_conv = 2 * v_m / d_conv = tau_d / eta_i:
        log10(eta_y) = _softmin2(log10(eta_diff), log10(eta_i * (tau_y_lid / tau_d)), LOCAL_CAP_LOG10_WIDTH)
        w_b = 0.5 * (1 + tanh(((tau_d / tau_y_lid) - 1) / YIELD_BLEND_RATIO_WIDTH))
        log10(eta_eff_lid) = (1 - w_b) * log10(eta_diff) + w_b * log10(eta_y)
    with linear stress-ratio width ``YIELD_BLEND_RATIO_WIDTH`` (0.025) and
    yield-branch log10 cap width ``LOCAL_CAP_LOG10_WIDTH`` (0.01). ``eta_i``,
    ``tau_d``, and ``tau_y_lid`` are required in ``'lid'`` mode.

    In local stress closure mode, ``eta_y = max(tau_y, COHESION_FLOOR) / (2 * sr_safe)``
    with ``sr_safe = sqrt(strain_rate**2 + LOCAL_STRAIN_RATE_FLOOR**2)`` and
    smooth log10 softplus cap of width ``LOCAL_CAP_LOG10_WIDTH`` (0.01).

    Raises
    ------
    ValueError
        If ``stress_closure_mode`` is not ``'lid'`` or ``'local'``, if required
        mode arguments are missing or cross-mode arguments are passed, if any
        input is non-real (such as a boolean, string, or complex value), or, on
        the eager NumPy path (``xp is np``), if ``eta_diff`` or ``eta_i`` is
        non-positive or infinite, if ``strain_rate`` or ``mixing_length`` is
        negative or infinite, if ``tau_d`` or ``unyielded_velocity`` is
        infinite, if ``w_lid`` lies outside ``[0, 1]``, or if ``tau_y_lid``,
        ``tau_y``, or ``tau_y_profile`` is negative (including ``-inf``;
        ``+inf`` is allowed). On the JAX path, out-of-range numeric values
        return ``NaN``.

    Notes
    -----
    Setting ``tau_y_lid = +inf`` in ``'lid'`` mode (or ``tau_y = +inf`` in
    ``'local'`` mode) represents infinite yield strength (no plastic yielding);
    a double-``where`` substitution replaces ``+inf`` with a finite placeholder
    before the ratio so ``jax.grad`` remains finite while returning ``eta_diff``
    in the outer ``where`` without capping finite yield stresses or high
    diffusion viscosities.

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
        if strain_rate is None and (unyielded_velocity is None or mixing_length is None):
            raise ValueError("mixing_length and unyielded_velocity required for 'local' mode")
        if tau_y is not None and tau_y_profile is not None:
            raise ValueError("Provide either tau_y or tau_y_profile in 'local' mode, not both")
        ty_raw = tau_y if tau_y is not None else tau_y_profile
        if ty_raw is None:
            raise ValueError("tau_y or tau_y_profile required for 'local' mode")
        invalid_local = _validate_inputs(
            (
                ('eta_diff', eta_diff, 'pos', False),
                ('strain_rate', strain_rate, 'nonneg', False),
                ('unyielded_velocity', unyielded_velocity, 'finite', False),
                ('mixing_length', mixing_length, 'nonneg', False),
                ('tau_y', tau_y, 'nonneg_or_posinf', False),
                ('tau_y_profile', tau_y_profile, 'nonneg_or_posinf', False),
            ),
            xp,
        )
        sr = (
            compute_strain_rate_local(unyielded_velocity, mixing_length, xp=xp)
            if strain_rate is None
            else xp.asarray(strain_rate, dtype=float)
        )
        ty = xp.asarray(ty_raw, dtype=float)
        eta_d = xp.asarray(eta_diff, dtype=float)
        inf_ty = xp.isinf(ty) & (ty > 0.0)
        ty_fin = xp.where(inf_ty, 1.0, xp.maximum(ty, COHESION_FLOOR))
        sr_safe = xp.sqrt(sr**2 + LOCAL_STRAIN_RATE_FLOOR**2)
        tau_y_term = ty_fin / (2.0 * sr_safe)
        is_unyielded = inf_ty | (sr <= 0.0) | xp.isnan(tau_y_term)
        log10_d = xp.log10(xp.maximum(eta_d, 1.0e-300))
        log10_y = xp.log10(xp.maximum(tau_y_term, 1.0e-300))
        d_loc = (log10_d - log10_y) / LOCAL_CAP_LOG10_WIDTH
        log10_res = _softmin2(log10_d, log10_y, LOCAL_CAP_LOG10_WIDTH, xp)
        res = xp.where(
            d_loc > 40.0, tau_y_term, xp.where(d_loc < -40.0, eta_d, 10.0**log10_res)
        )
        return xp.where(
            invalid_local | xp.isnan(sr), xp.nan, xp.where(is_unyielded, eta_d, res)
        )

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

    invalid_lid = _validate_inputs(
        (
            ('eta_diff', eta_diff, 'pos', False),
            ('eta_i', eta_i, 'pos', False),
            ('tau_d', tau_d, 'nonneg', False),
            ('tau_y_lid', tau_y_lid, 'nonneg_or_posinf', False),
            ('w_lid', w_lid, 'unit', False),
        ),
        xp,
    )
    eta_d = xp.asarray(eta_diff, dtype=float)
    eta_i_arr = xp.asarray(eta_i, dtype=float)
    td = xp.asarray(tau_d, dtype=float)

    # Smooth log10 tanh blend at convective strain rate D_conv = tau_d / eta_i:
    ty_raw_lid = xp.asarray(tau_y_lid, dtype=float)
    inf_ty = xp.isinf(ty_raw_lid) & (ty_raw_lid > 0.0)
    ty_lid = xp.where(inf_ty, 1.0, xp.maximum(ty_raw_lid, COHESION_FLOOR))
    unyielded = (td <= 0.0) | inf_ty
    ratio = xp.where(unyielded, 1.0, ty_lid / xp.maximum(td, 1.0e-30))
    x_ratio = xp.where(unyielded, 0.0, td / xp.maximum(ty_lid, 1.0e-30))
    w_b = 0.5 * (1.0 + xp.tanh((x_ratio - 1.0) / YIELD_BLEND_RATIO_WIDTH))
    log10_d_lid = xp.log10(xp.maximum(eta_d, 1.0e-300))
    log10_y_raw = xp.log10(xp.maximum(eta_i_arr * ratio, 1.0e-300))
    log10_y_lid = _softmin2(log10_d_lid, log10_y_raw, LOCAL_CAP_LOG10_WIDTH, xp)
    d_y_lid = (log10_d_lid - log10_y_raw) / LOCAL_CAP_LOG10_WIDTH
    eta_y_branch = xp.where(
        d_y_lid > 40.0,
        eta_i_arr * ratio,
        xp.where(d_y_lid < -40.0, eta_d, 10.0**log10_y_lid),
    )
    log10_eff_val = (1.0 - w_b) * log10_d_lid + w_b * log10_y_lid
    eta_eff_val = xp.where(
        w_b <= 1.0e-15,
        eta_d,
        xp.where(w_b >= 1.0 - 1.0e-15, eta_y_branch, 10.0**log10_eff_val),
    )

    if w_lid is not None:
        wl = xp.asarray(w_lid, dtype=float)
        log_diff = xp.log10(xp.maximum(eta_d, 1e-30))
        log_eff = xp.log10(xp.maximum(eta_eff_val, 1e-30))
        log_solid = wl * log_eff + (1.0 - wl) * log_diff
        res = 10.0**log_solid
        eta_eff_val = xp.where((wl == 0.0) | (eta_eff_val == eta_d), eta_d, res)

    return xp.where(invalid_lid, xp.nan, eta_eff_val)


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
