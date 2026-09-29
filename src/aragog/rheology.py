"""Solid-state mantle rheology formulations.

Provides temperature- and pressure-dependent Arrhenius diffusion creep viscosity,
Byerlee yield stress, boundary-layer stress closure, effective viscosity capping,
and two-branch regime switching for stagnant and mobile convective lids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

R_GAS = 8.314462618

FloatOrArray = float | npt.NDArray[np.floating]


@dataclass(frozen=True, eq=True)
class SolidRheologyParams:
    """Parameters for solid-state mantle rheology and stagnant lid scaling.

    This class is the single source of truth for rheology parameter defaults
    and bounds validation across Aragog and PROTEUS.
    """

    enabled: bool = False
    activation_energy: float = 300e3
    activation_volume: float = 5e-6
    activation_volume_decay_pressure: float = float('inf')
    arrhenius_t_ref: float = 1600.0
    viscosity_max_log10: float = 40.0
    water_prefactor: float = 1.0
    yield_stress_c: float = 50e6
    yield_stress_mu: float = 0.6
    yield_stress_max: float = 500e6
    yield_switch_width: float = 0.1
    stress_closure_mode: str = 'lid'
    interior_flux_fraction: float = 0.05
    lid_base_mode: str = 'rheological'
    lid_base_temperature: float = 1400.0
    lid_contrast_coeff: float = 2.2
    lid_mask_width_cells: float = 1.0
    phi_visc_single: float = 0.5
    # Slopes of the viscous-branch mixing length at the top and bottom boundary,
    # l_v = min(bottom (r - r_in), top (r_out - r)); 1 and 1 give the Abe profile.
    # The top slope is fit jointly to Korenaga (2009) and Deschamps and Vilella (2021).
    mlt_top_slope: float = 0.22
    mlt_bottom_slope: float = 1.0

    def __post_init__(self) -> None:
        for name in ('mlt_top_slope', 'mlt_bottom_slope'):
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f'{name} must be positive and finite, got {val}')
        if self.stress_closure_mode not in ('lid', 'local'):
            raise ValueError(
                f'Unknown stress_closure_mode {self.stress_closure_mode!r}; '
                f"expected 'lid' or 'local'. Note that 'global' mode is unsupported; use 'lid'."
            )
        if self.lid_base_mode not in ('fixed', 'rheological'):
            raise ValueError(
                f'Unknown lid_base_mode {self.lid_base_mode!r}; '
                f"expected 'fixed' or 'rheological'"
            )
        if not math.isfinite(self.arrhenius_t_ref) or self.arrhenius_t_ref <= 0.0:
            raise ValueError(f'arrhenius_t_ref must be positive, got {self.arrhenius_t_ref}')
        if not math.isfinite(self.activation_energy) or self.activation_energy < 0.0:
            raise ValueError(
                f'activation_energy must be non-negative, got {self.activation_energy}'
            )
        if not math.isfinite(self.activation_volume) or self.activation_volume < 0.0:
            raise ValueError(
                f'activation_volume must be non-negative, got {self.activation_volume}'
            )
        if math.isnan(self.activation_volume_decay_pressure) or (
            self.activation_volume_decay_pressure <= 0.0
            and not math.isinf(self.activation_volume_decay_pressure)
        ):
            raise ValueError(
                'activation_volume_decay_pressure must be positive or inf, '
                f'got {self.activation_volume_decay_pressure}'
            )
        if not math.isfinite(self.viscosity_max_log10) or self.viscosity_max_log10 <= 20.0:
            raise ValueError(
                f'viscosity_max_log10 must be > 20, got {self.viscosity_max_log10}'
            )
        if not math.isfinite(self.water_prefactor) or self.water_prefactor <= 0.0:
            raise ValueError(f'water_prefactor must be positive, got {self.water_prefactor}')
        if not math.isfinite(self.yield_stress_c) or self.yield_stress_c < 0.0:
            raise ValueError(f'yield_stress_c must be non-negative, got {self.yield_stress_c}')
        if not math.isfinite(self.yield_stress_mu) or self.yield_stress_mu < 0.0:
            raise ValueError(
                f'yield_stress_mu must be non-negative, got {self.yield_stress_mu}'
            )
        if not math.isfinite(self.yield_stress_max) or self.yield_stress_max <= 0.0:
            raise ValueError(f'yield_stress_max must be positive, got {self.yield_stress_max}')
        if not math.isfinite(self.yield_switch_width) or self.yield_switch_width <= 0.0:
            raise ValueError(
                f'yield_switch_width must be positive, got {self.yield_switch_width}'
            )
        if not math.isfinite(self.interior_flux_fraction) or not (
            0.0 < self.interior_flux_fraction < 1.0
        ):
            raise ValueError(
                f'interior_flux_fraction must be in (0, 1), got {self.interior_flux_fraction}'
            )
        if not math.isfinite(self.lid_base_temperature) or self.lid_base_temperature <= 0.0:
            raise ValueError(
                f'lid_base_temperature must be positive, got {self.lid_base_temperature}'
            )
        if not math.isfinite(self.lid_contrast_coeff) or self.lid_contrast_coeff <= 0.0:
            raise ValueError(
                f'lid_contrast_coeff must be positive, got {self.lid_contrast_coeff}'
            )
        if not math.isfinite(self.lid_mask_width_cells) or self.lid_mask_width_cells <= 0.0:
            raise ValueError(
                f'lid_mask_width_cells must be positive, got {self.lid_mask_width_cells}'
            )
        if not math.isfinite(self.phi_visc_single) or not (0.0 < self.phi_visc_single < 1.0):
            raise ValueError(f'phi_visc_single must be in (0, 1), got {self.phi_visc_single}')
        if (
            self.enabled
            and self.lid_base_mode == 'rheological'
            and self.activation_energy <= 0.0
        ):
            raise ValueError(
                f'Invalid combination: lid_base_mode={self.lid_base_mode!r} requires non-zero '
                f'activation_energy, but activation_energy={self.activation_energy}'
            )


def viscous_mixing_length_factor(
    radii, r_in, r_out, mixing_length, top_slope, bottom_slope, xp=np
):
    """Factor on the viscous MLT velocity that replaces the Abe length by the calibrated one.

    The viscous eddy diffusivity scales as ``l^4``, so with
    ``l_v = min(bottom_slope (r - r_in), top_slope (r_out - r))`` (Wagner et al.
    2019, eq. 11, written with the two boundary slopes) it is multiplied by
    ``(l_v / l)^4``; the inviscid branch keeps ``l``.

    Parameters
    ----------
    radii : array
        Basic-node radii [m].
    r_in, r_out : float
        Inner and outer radius of the mantle [m].
    mixing_length : array
        Abe mixing length ``l`` at the basic nodes [m].
    top_slope, bottom_slope : float
        Slopes of ``l_v`` at the surface and at the CMB.
    xp : module
        Array namespace.

    Returns
    -------
    array
        ``(l_v / l)^4``, and 1 where ``l`` is 0 (the boundary nodes).
    """
    l_v = xp.minimum(bottom_slope * (radii - r_in), top_slope * (r_out - radii))
    positive = mixing_length > 0.0
    return xp.where(positive, (l_v / xp.where(positive, mixing_length, 1.0)) ** 4, 1.0)


def compute_arrhenius_enthalpy(
    pressure: FloatOrArray,
    activation_energy: float = 300.0e3,
    activation_volume: float = 5.0e-6,
    activation_volume_decay_pressure: float = float('inf'),
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute saturating activation enthalpy H(P) [J/mol].

    .. math::
        H(P) = E_a + V_0 P_\mathrm{decay} \left(1 - \exp\left(-\frac{P}{P_\mathrm{decay}}\right)\right)

    Parameters
    ----------
    pressure : float or array-like
        Pressure P [Pa].
    activation_energy : float, default 300.0e3
        Molar activation energy E_a [J/mol].
    activation_volume : float, default 5.0e-6
        Zero-pressure molar activation volume V_0 [m^3/mol].
    activation_volume_decay_pressure : float, default inf
        Characteristic pressure scale P_decay [Pa].
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Activation enthalpy H(P) [J/mol].
    """
    p = xp.asarray(pressure, dtype=float)
    is_inf = xp.isinf(activation_volume_decay_pressure)
    p_safe = xp.where(is_inf, 1.0, activation_volume_decay_pressure)
    p_term = xp.where(is_inf, p, -p_safe * xp.expm1(-p / p_safe))
    h = activation_energy + activation_volume * p_term
    if xp is np and np.ndim(pressure) == 0:
        return float(h.item())
    return h


def compute_arrhenius_viscosity(
    temperature: FloatOrArray,
    pressure: FloatOrArray,
    viscosity_solid: float = 1.0e21,
    activation_energy: float = 300.0e3,
    activation_volume: float = 5.0e-6,
    activation_volume_decay_pressure: float = float('inf'),
    arrhenius_t_ref: float = 1600.0,
    r_gas: float = R_GAS,
    viscosity_max_log10: float = 40.0,
    water_prefactor: FloatOrArray = 1.0,
    xp: Any = np,
    t_ref: float | None = None,
    T_ref: float | None = None,
    R: float | None = None,
) -> FloatOrArray:
    r"""Compute temperature- and pressure-dependent Arrhenius viscosity.

    .. math::
        \eta_\mathrm{diff}(T, P) = \eta_0 f_\mathrm{water} \exp\left(
            \frac{H(P)}{R T} - \frac{E_a}{R T_\mathrm{ref}}
        \right)

    Parameters
    ----------
    temperature : float or array-like
        Temperature T [K].
    pressure : float or array-like
        Pressure P [Pa].
    viscosity_solid : float, default 1.0e21
        Reference solid-state dynamic viscosity eta_0 [Pa s] at (T_ref, P=0).
    activation_energy : float, default 300.0e3
        Molar activation energy E_a [J/mol].
    activation_volume : float, default 5.0e-6
        Zero-pressure molar activation volume V_0 [m^3/mol].
    activation_volume_decay_pressure : float, default inf
        Characteristic pressure scale P_decay [Pa].
    arrhenius_t_ref : float, default 1600.0
        Reference temperature T_ref [K].
    r_gas : float, default 8.314462618
        Universal gas constant R [J/(mol K)].
    viscosity_max_log10 : float, default 40.0
        Upper bound on log10 dynamic viscosity [log10(Pa s)].
    water_prefactor : float or array-like, default 1.0
        Multiplicative prefactor for hydration weakening. Must be positive.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).
    t_ref : float, optional
        Alias for arrhenius_t_ref.
    T_ref : float, optional
        Alias for arrhenius_t_ref.
    R : float, optional
        Alias for r_gas.

    Returns
    -------
    float or array-like
        Dynamic diffusion creep viscosity eta_diff [Pa s].
    """
    if t_ref is not None:
        arrhenius_t_ref = t_ref
    if T_ref is not None:
        arrhenius_t_ref = T_ref
    if R is not None:
        r_gas = R

    if arrhenius_t_ref <= 0.0:
        raise ValueError(f'arrhenius_t_ref must be positive, got {arrhenius_t_ref}')

    wp = xp.asarray(water_prefactor, dtype=float)
    if xp is np and np.any(wp <= 0.0):
        raise ValueError(f'water_prefactor must be positive, got {water_prefactor}')

    t = xp.asarray(temperature, dtype=float)
    p = xp.asarray(pressure, dtype=float)

    h_p = compute_arrhenius_enthalpy(
        pressure=p,
        activation_energy=activation_energy,
        activation_volume=activation_volume,
        activation_volume_decay_pressure=activation_volume_decay_pressure,
        xp=xp,
    )
    h_0 = activation_energy

    t_safe = xp.where(t > 0.0, t, 1.0)
    exponent = h_p / (r_gas * t_safe) - h_0 / (r_gas * arrhenius_t_ref)

    max_exponent = (
        viscosity_max_log10 - xp.log10(xp.maximum(viscosity_solid, 1e-300))
    ) * math.log(10.0)
    clipped_exp = xp.clip(exponent, -700.0, max_exponent)
    result = viscosity_solid * wp * xp.exp(clipped_exp)
    result = xp.minimum(result, 10.0**viscosity_max_log10)
    if (
        xp is np
        and np.ndim(temperature) == 0
        and np.ndim(pressure) == 0
        and np.ndim(water_prefactor) == 0
    ):
        return float(result.item())
    return result


def eta_diff(
    temperature: FloatOrArray,
    pressure: FloatOrArray,
    viscosity_solid: float = 1.0e21,
    activation_energy: float = 300.0e3,
    activation_volume: float = 5.0e-6,
    t_ref: float = 1600.0,
    r_gas: float = R_GAS,
    viscosity_max_log10: float = 40.0,
    water_prefactor: FloatOrArray = 1.0,
    activation_volume_decay_pressure: float = float('inf'),
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute temperature- and pressure-dependent Arrhenius viscosity."""
    return compute_arrhenius_viscosity(
        temperature=temperature,
        pressure=pressure,
        viscosity_solid=viscosity_solid,
        activation_energy=activation_energy,
        activation_volume=activation_volume,
        activation_volume_decay_pressure=activation_volume_decay_pressure,
        arrhenius_t_ref=t_ref,
        r_gas=r_gas,
        viscosity_max_log10=viscosity_max_log10,
        water_prefactor=water_prefactor,
        xp=xp,
    )


def compute_yield_stress(
    pressure: FloatOrArray,
    yield_stress_c: float = 50.0e6,
    yield_stress_mu: float = 0.6,
    yield_stress_max: float = 500.0e6,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute Byerlee frictional yield stress.

    .. math::
        \tau_y(P) = \min(c + \mu P, \tau_\mathrm{max})

    Parameters
    ----------
    pressure : float or array-like
        Pressure P [Pa].
    yield_stress_c : float, default 50.0e6
        Cohesion c [Pa].
    yield_stress_mu : float, default 0.6
        Friction coefficient mu [-].
    yield_stress_max : float, default 500.0e6
        Maximum yield stress ceiling [Pa].
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Yield stress tau_y [Pa].
    """
    p = xp.asarray(pressure, dtype=float)
    tau_y = xp.minimum(yield_stress_c + yield_stress_mu * p, yield_stress_max)
    if xp is np and np.ndim(pressure) == 0:
        return float(tau_y.item())
    return tau_y


def eta_eff(
    visc_diff: FloatOrArray,
    tau_y: FloatOrArray,
    strain_rate: FloatOrArray,
    smooth: bool = True,
    eps: float = 1.0e-30,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute effective viscosity with plastic yielding cap."""
    eta_d = xp.asarray(visc_diff, dtype=float)
    ty = xp.asarray(tau_y, dtype=float)
    sr = xp.maximum(xp.asarray(strain_rate, dtype=float), eps)

    tau_y_term = ty / (2.0 * sr)
    if smooth:
        is_inf = xp.isinf(tau_y_term)
        safe_ty_term = xp.where(is_inf, 1.0, tau_y_term)
        result = (eta_d * safe_ty_term) / (eta_d + safe_ty_term)
        result = xp.where(is_inf, eta_d, result)
    else:
        result = xp.minimum(eta_d, tau_y_term)

    if (
        xp is np
        and np.ndim(visc_diff) == 0
        and np.ndim(tau_y) == 0
        and np.ndim(strain_rate) == 0
    ):
        return float(result.item())
    return result


def compute_strain_rate_local(
    viscous_velocity: FloatOrArray,
    mixing_length: FloatOrArray,
    eps: float = 1.0e-15,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute local convective strain rate from MLT state variables.

    .. math::
        \dot{\epsilon}_\mathrm{local} = \frac{|v|}{2 \max(l_\mathrm{mix}, \epsilon)}

    Parameters
    ----------
    viscous_velocity : float or array-like
        Convective / viscous velocity v [m/s].
    mixing_length : float or array-like
        Mixing length l_mix [m].
    eps : float, default 1.0e-15
        Denominator floor to prevent division by zero.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Local strain rate [1/s].
    """
    v = xp.asarray(viscous_velocity, dtype=float)
    l_mix = xp.asarray(mixing_length, dtype=float)
    denom = 2.0 * xp.maximum(l_mix, eps)
    res = xp.abs(v) / denom
    if xp is np and np.ndim(viscous_velocity) == 0 and np.ndim(mixing_length) == 0:
        return float(res.item())
    return res


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
        Solidus temperature profile T_sol [K].
    melt_fraction : float or array-like
        Melt mass fraction profile phi [-].
    params : SolidRheologyParams
        Rheology parameter set.
    unyielded_velocity : float or array-like, optional
        Unyielded convective velocity profile v [m/s].
    viscosity_solid : float, optional
        Reference solid mantle viscosity [Pa s]. Defaults to 1e21 if unspecified.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    dict[str, Any]
        Dictionary of boundary layer state diagnostics.
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
    theta = (H_T_i * (T_i - T_surf)) / (R_GAS * T_i**2)

    eta_0 = viscosity_solid
    if eta_0 is None:
        eta_0 = getattr(params, 'viscosity_solid', None)
    if eta_0 is None and hasattr(params, 'solid_log10visc'):
        eta_0 = 10.0**params.solid_log10visc
    if eta_0 is None and hasattr(params, 'log10_visc_solid'):
        eta_0 = 10.0**params.log10_visc_solid
    if eta_0 is None:
        eta_0 = 1.0e21

    eta_surf = compute_arrhenius_viscosity(
        T_surf,
        P_surf,
        viscosity_solid=eta_0,
        activation_energy=params.activation_energy,
        activation_volume=params.activation_volume,
        activation_volume_decay_pressure=params.activation_volume_decay_pressure,
        arrhenius_t_ref=params.arrhenius_t_ref,
        r_gas=R_GAS,
        viscosity_max_log10=params.viscosity_max_log10,
        xp=xp,
    )
    eta_i = compute_arrhenius_viscosity(
        T_i,
        P_T_i,
        viscosity_solid=eta_0,
        activation_energy=params.activation_energy,
        activation_volume=params.activation_volume,
        activation_volume_decay_pressure=params.activation_volume_decay_pressure,
        arrhenius_t_ref=params.arrhenius_t_ref,
        r_gas=R_GAS,
        viscosity_max_log10=params.viscosity_max_log10,
        xp=xp,
    )
    eta_contrast = eta_surf / xp.maximum(eta_i, 1e-30)

    if params.lid_base_mode == 'fixed':
        T_lid_iso = params.lid_base_temperature
    else:
        T_lid_iso = T_i - params.lid_contrast_coeff * dT_rh

    dT_cell = xp.maximum(xp.abs(xp.gradient(T)), 1.0)
    w_mask = params.lid_mask_width_cells
    scale = w_mask * dT_cell

    w_hot_iso = 0.5 * (1.0 + xp.tanh((T - T_lid_iso) / scale))
    if solidus_temperature is not None:
        T_sol = xp.asarray(solidus_temperature, dtype=float)
        w_hot_sol = 0.5 * (1.0 + xp.tanh((T - T_sol) / scale))
    else:
        w_hot_sol = 0.5 * (1.0 + xp.tanh((phi - 0.01) / 0.002))
    w_hot_phi = 0.5 * (1.0 + xp.tanh((phi - 0.4) / 0.05))

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
    d_lid = xp.maximum(r[-1] - r_lid_base, 0.0) * w_active

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
        v_max = xp.max(xp.where(w_upper > 0.05, v_abs, 0.0))
        v_diff = xp.where(w_upper > 0.001, (v_abs - v_max) / T_v, -100.0)
        exp_term = xp.exp(xp.clip(v_diff, -100.0, 50.0))
        sum_exp = xp.sum(w_upper * exp_term)
        v_i_raw = v_max + T_v * xp.log(xp.maximum(sum_exp / N_upper, 1e-300))
        v_i = xp.maximum(v_i_raw, 0.0) * w_active
    else:
        v_i = 0.0

    tau_d = (eta_i * v_i) / xp.maximum(delta_rh, 1e-6)
    tau_d = tau_d * w_active
    tau_d_lid = (eta_i * v_i) / xp.maximum(d_lid, 1e-6)

    tau_y_lid = compute_yield_stress(
        P_lid_base,
        yield_stress_c=params.yield_stress_c,
        yield_stress_mu=params.yield_stress_mu,
        yield_stress_max=params.yield_stress_max,
        xp=xp,
    )

    ratio = tau_d / xp.maximum(tau_y_lid, 1e-10)
    w_y = 0.5 * (1.0 + xp.tanh((ratio - 1.0) / params.yield_switch_width))
    lid_cell_count = xp.sum(w_lid)
    w_solid_surf = 0.5 * (1.0 - xp.tanh((phi[-1] - 0.01) / 0.002))
    w_has_lid = 0.5 * (1.0 + xp.tanh((lid_cell_count - 0.5) / 0.1)) * w_solid_surf
    lid_regime = w_active * w_has_lid * (1.0 + w_y)

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
        'tau_d': tau_d,
        'tau_d_lid': tau_d_lid,
        'tau_y_lid': tau_y_lid,
        'w_lid': w_lid,
        'w_y': w_y,
        'w_active': w_active,
        'lid_regime': lid_regime,
        'lid_cell_count': lid_cell_count,
        'eta_i': eta_i,
    }


def compute_effective_viscosity(
    eta_diff: FloatOrArray,
    tau_d: FloatOrArray | None = None,
    tau_y_lid: FloatOrArray | None = None,
    v_i: FloatOrArray | None = None,
    delta_rh: FloatOrArray | None = None,
    eta_i: FloatOrArray | None = None,
    yield_switch_width: float = 0.1,
    stress_closure_mode: str = 'lid',
    unyielded_velocity: FloatOrArray | None = None,
    mixing_length: FloatOrArray | None = None,
    tau_y_profile: FloatOrArray | None = None,
    w_lid: FloatOrArray | None = None,
    strain_rate: FloatOrArray | None = None,
    tau_y: FloatOrArray | None = None,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute effective dynamic viscosity with boundary-layer or local stress closure."""
    # Support 3-argument positional calling: compute_effective_viscosity(eta_diff, tau_y, strain_rate)
    if v_i is None and delta_rh is None and tau_d is not None and tau_y_lid is not None:
        tau_y = tau_d
        strain_rate = tau_y_lid
        tau_d = None
        tau_y_lid = None

    if stress_closure_mode == 'global':
        raise ValueError(
            "Unknown stress_closure_mode 'global'; expected 'lid' or 'local'. "
            "'global' mode is unsupported; use 'lid'."
        )
    if stress_closure_mode == 'local' or strain_rate is not None:
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
        sr_safe = xp.maximum(sr, 1e-30)
        tau_y_term = ty / (2.0 * sr_safe)
        is_inf = xp.isinf(tau_y_term)
        safe_ty_term = xp.where(is_inf, 1.0, tau_y_term)
        res = (eta_d * safe_ty_term) / (eta_d + safe_ty_term)
        return xp.where(is_inf, eta_d, res)

    eta_d = xp.asarray(eta_diff, dtype=float)
    td = xp.asarray(tau_d, dtype=float)
    ty_lid = xp.asarray(tau_y_lid, dtype=float)
    vi = xp.asarray(v_i, dtype=float)
    drh = xp.asarray(delta_rh, dtype=float)

    eta_lid = (ty_lid * drh) / xp.maximum(vi, 1e-30)
    eta_y = xp.minimum(eta_lid, eta_d)

    ratio = td / xp.maximum(ty_lid, 1e-10)
    eta_below = xp.maximum(eta_d * (1.0 - ratio), eta_y)
    eta_yielded = eta_y

    w_y = 0.5 * (1.0 + xp.tanh((ratio - 1.0) / yield_switch_width))
    log_below = xp.log10(xp.maximum(eta_below, 1e-30))
    log_yielded = xp.log10(xp.maximum(eta_yielded, 1e-30))
    log_eff = (1.0 - w_y) * log_below + w_y * log_yielded
    eta_eff_val = 10.0**log_eff

    if w_lid is not None:
        wl = xp.asarray(w_lid, dtype=float)
        log_diff = xp.log10(xp.maximum(eta_d, 1e-30))
        log_solid = wl * log_eff + (1.0 - wl) * log_diff
        return 10.0**log_solid

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
