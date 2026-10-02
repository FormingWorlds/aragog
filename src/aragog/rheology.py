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


def __getattr__(name: str) -> Any:
    if name in ('compute_stagnant_lid_state', 'compute_effective_viscosity', 'stress_closure'):
        import aragog.rheology_lid as _lid

        return getattr(_lid, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(
        list(globals().keys())
        + ['compute_stagnant_lid_state', 'compute_effective_viscosity', 'stress_closure']
    )
