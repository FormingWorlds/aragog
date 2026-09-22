"""Solid-state mantle rheology formulations.

Provides temperature- and pressure-dependent Arrhenius diffusion creep viscosity,
Byerlee yield stress, effective viscosity capping, and 1D stress closure strain-rate
models (local MLT and global boundary-layer closures).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

R_GAS = 8.314462618

FloatOrArray = float | npt.NDArray[np.floating]


def compute_t_lid_base(
    t_m: float,
    p_lid: float,
    e_a: float = 300.0e3,
    v_a: float = 5.0e-6,
    lid_contrast_coeff: float = 2.2,
    r_gas: float = R_GAS,
) -> float:
    """Compute the stagnant lid base temperature (Solomatov & Moresi 2000).

    The rheological temperature scale is dT_rh = R T_m^2 / (E_a + P V_a).
    The lid base is defined as t_m - lid_contrast_coeff * dT_rh.
    """
    e_eff = max(float(e_a + p_lid * v_a), 1e-6)
    dt_rh = r_gas * t_m**2 / e_eff
    return t_m - lid_contrast_coeff * dt_rh


def eta_diff(
    temperature: FloatOrArray,
    pressure: FloatOrArray,
    viscosity_solid: float = 1.0e21,
    activation_energy: float = 300.0e3,
    activation_volume: float = 5.0e-6,
    t_ref: float = 1600.0,
    r_gas: float = R_GAS,
    viscosity_max_log10: float = 40.0,
) -> FloatOrArray:
    r"""Compute temperature- and pressure-dependent Arrhenius viscosity.

    Implements the solid-state diffusion creep viscosity law:
    .. math::
        \eta_\mathrm{diff}(T, P) = \eta_0 \exp\left(
            \frac{E_a + P V_a}{R T} - \frac{E_a}{R T_\mathrm{ref}}
        \right)

    Parameters
    ----------
        temperature : float or numpy.ndarray
            Temperature :math:`T` [K].
        pressure : float or numpy.ndarray
            Pressure :math:`P` [Pa].
        viscosity_solid : float, default 1.0e21
        viscosity_max_log10 : float, default 40.0
            Reference solid-state dynamic viscosity :math:`\eta_0` [Pa s] at
            reference conditions :math:`(T_\mathrm{ref}, P=0)`.
        activation_energy : float, default 300.0e3
            Molar activation energy :math:`E_a` [J/mol].
        activation_volume : float, default 5.0e-6
            Molar activation volume :math:`V_a` [m^3/mol].
        t_ref : float, default 1600.0
            Reference temperature :math:`T_\mathrm{ref}` [K].
        r_gas : float, default 8.314462618
            Universal gas constant :math:`R` [J/(mol K)].

        Returns
        -------
        float or numpy.ndarray
            Dynamic diffusion creep viscosity :math:`\eta_\mathrm{diff}` [Pa s].
    """
    if t_ref <= 0.0:
        raise ValueError(f't_ref must be positive, got {t_ref}')
    t = np.asarray(temperature, dtype=float)
    p = np.asarray(pressure, dtype=float)
    exponent = (activation_energy + p * activation_volume) / (r_gas * t) - activation_energy / (
        r_gas * t_ref
    )
    max_exponent = (
        viscosity_max_log10 - np.log10(np.maximum(viscosity_solid, 1e-300))
    ) * np.log(10.0)
    clipped_exp = np.clip(exponent, -700.0, max_exponent)
    result = viscosity_solid * np.exp(clipped_exp)
    if np.ndim(temperature) == 0 and np.ndim(pressure) == 0:
        return float(result.item())
    return result


def compute_yield_stress(
    pressure: FloatOrArray,
    yield_stress_c: float = 50.0e6,
    yield_stress_mu: float = 0.6,
    yield_stress_max: float = 500.0e6,
) -> FloatOrArray:
    r"""Compute Byerlee frictional yield stress.

    .. math::
        \tau_y(P) = c + \mu P

    Parameters
    ----------
    pressure : float or numpy.ndarray
        Pressure :math:`P` [Pa].
    yield_stress_c : float, default 50.0e6
        Cohesion :math:`c` [Pa].
    yield_stress_mu : float, default 0.6
        Friction coefficient :math:`\mu` [-].

    Returns
    -------
    float or numpy.ndarray
        Yield stress :math:`\tau_y` [Pa].
    """
    p = np.asarray(pressure, dtype=float)
    tau_y = np.minimum(yield_stress_c + yield_stress_mu * p, yield_stress_max)
    if np.ndim(pressure) == 0:
        return float(tau_y.item())
    return tau_y


def eta_eff(
    visc_diff: FloatOrArray,
    tau_y: FloatOrArray,
    strain_rate: FloatOrArray,
    smooth: bool = True,
    eps: float = 1.0e-30,
) -> FloatOrArray:
    r"""Compute effective viscosity with plastic yielding cap.

    The plastic yielding branch limits the deviatoric stress to :math:`\tau_y`:
    .. math::
        \eta_\mathrm{yield} = \frac{\tau_y}{2 \dot{\epsilon}}

    Under smooth harmonic blending (continuous pseudoplastic formulation):
    .. math::
        \eta_\mathrm{eff} = \frac{\eta_\mathrm{diff} \eta_\mathrm{yield}}{\eta_\mathrm{diff} + \eta_\mathrm{yield}}
        = \left( \frac{1}{\eta_\mathrm{diff}} + \frac{2 \dot{\epsilon}}{\tau_y} \right)^{-1}

    Parameters
    ----------
    visc_diff : float or numpy.ndarray
        Ductile / diffusion creep viscosity :math:`\eta_\mathrm{diff}` [Pa s].
    tau_y : float or numpy.ndarray
        Yield stress :math:`\tau_y` [Pa].
    strain_rate : float or numpy.ndarray
        Second invariant of deviatoric strain rate :math:`\dot{\epsilon}` [1/s].
    smooth : bool, default True
        If True, use harmonic mean smooth minimum. If False, use sharp minimum.
    eps : float, default 1.0e-30
        Small regularization to prevent division by zero at zero strain rate.

    Returns
    -------
    float or numpy.ndarray
        Effective dynamic viscosity :math:`\eta_\mathrm{eff}` [Pa s].
    """
    eta_d = np.asarray(visc_diff, dtype=float)
    ty = np.asarray(tau_y, dtype=float)
    sr = np.maximum(np.asarray(strain_rate, dtype=float), eps)

    tau_y_term = ty / (2.0 * sr)
    if smooth:
        is_inf = np.isinf(tau_y_term)
        safe_ty_term = np.where(is_inf, 1.0, tau_y_term)
        result = (eta_d * safe_ty_term) / (eta_d + safe_ty_term)
        result = np.where(is_inf, eta_d, result)
    else:
        result = np.minimum(eta_d, tau_y_term)

    if np.ndim(visc_diff) == 0 and np.ndim(tau_y) == 0 and np.ndim(strain_rate) == 0:
        return float(result.item())
    return result


def compute_strain_rate_local(
    viscous_velocity: FloatOrArray,
    mixing_length: FloatOrArray,
    eps: float = 1.0e-15,
) -> FloatOrArray:
    r"""Compute local convective strain rate from MLT state variables.

    .. math::
        \dot{\epsilon}_\mathrm{local} = \frac{v}{\max(l_\mathrm{mix}, \epsilon)}

    Parameters
    ----------
    viscous_velocity : float or numpy.ndarray
        Convective / viscous velocity :math:`v` [m/s].
    mixing_length : float or numpy.ndarray
        Mixing length :math:`l_\mathrm{mix}` [m].
    eps : float, default 1.0e-15
        Denominator floor to prevent division by zero.

    Returns
    -------
    float or numpy.ndarray
        Local strain rate :math:`\dot{\epsilon}` [1/s].
    """
    v = np.asarray(viscous_velocity, dtype=float)
    l_mix = np.asarray(mixing_length, dtype=float)
    denom = np.maximum(l_mix, eps)
    res = np.abs(v) / denom
    if np.ndim(viscous_velocity) == 0 and np.ndim(mixing_length) == 0:
        return float(res.item())
    return res


def compute_strain_rate_global(
    radius: npt.NDArray[np.floating],
    temperature: npt.NDArray[np.floating],
    viscous_velocity: npt.NDArray[np.floating],
    t_lid_base: float = 1400.0,
    eps: float = 1.0e-15,
) -> FloatOrArray:
    r"""Compute global boundary-layer lithospheric strain rate proxy.

    Locates the conductive lid base where temperature drops below `t_lid_base`,
    evaluates lid thickness :math:`d_\mathrm{lid} = R_\mathrm{surf} - r_\mathrm{base}`,
    and uses the peak convective velocity below the lid as interior velocity :math:`v_\mathrm{int}`:
    .. math::
        \dot{\epsilon}_\mathrm{global} = \frac{v_\mathrm{int}}{\max(d_\mathrm{lid}, \epsilon)}

    Parameters
    ----------
    radius : numpy.ndarray
        Radial grid coordinates :math:`r` [m] from core to surface.
    temperature : numpy.ndarray
        Radial temperature profile :math:`T(r)` [K].
    viscous_velocity : numpy.ndarray
        Convective velocity profile :math:`v(r)` [m/s].
    t_lid_base : float, default 1400.0
        Threshold temperature defining the base of the stagnant lid [K].
    eps : float, default 1.0e-15
        Denominator floor [m].

    Returns
    -------
    float or numpy.ndarray
        Global lithospheric strain rate :math:`\dot{\epsilon}_\mathrm{global}` [1/s].
    """
    r = np.asarray(radius, dtype=float)
    t = np.asarray(temperature, dtype=float)
    v = np.asarray(viscous_velocity, dtype=float)

    is_colder = t <= t_lid_base
    if not np.any(is_colder):
        return 0.0

    r_surf = r[-1]
    is_hot = t > t_lid_base
    if np.any(is_hot):
        r_lid_base = float(np.max(r[is_hot]))
        d_lid = r_surf - r_lid_base
    else:
        d_lid = r_surf - r[0]

    dr_min = r[-1] - r[-2] if len(r) > 1 else eps
    d_lid = max(d_lid, dr_min)

    interior_mask = r <= (r_surf - d_lid)
    if np.any(interior_mask):
        v_int = float(np.max(np.abs(v[interior_mask])))
    else:
        v_int = float(np.max(np.abs(v)))

    return v_int / d_lid


def stress_closure(
    mode: str,
    viscous_velocity: FloatOrArray,
    mixing_length: FloatOrArray | None = None,
    radius: npt.NDArray[np.floating] | None = None,
    temperature: npt.NDArray[np.floating] | None = None,
    t_lid_base: float = 1400.0,
    eps: float = 1.0e-15,
) -> FloatOrArray:
    r"""Calculate strain rate dispatching on stress closure mode.

    Parameters
    ----------
    mode : {'local', 'global'}
        Stress closure mode selector.
    viscous_velocity : float or numpy.ndarray
        Convective velocity [m/s].
    mixing_length : float or numpy.ndarray, optional
        Mixing length [m] (required for 'local' mode).
    radius : numpy.ndarray, optional
        Radial coordinates [m] (required for 'global' mode).
    temperature : numpy.ndarray, optional
        Temperature profile [K] (required for 'global' mode).
    t_lid_base : float, default 1400.0
        Rheological lid base temperature [K].
    eps : float, default 1.0e-15
        Regularization floor.

    Returns
    -------
    float or numpy.ndarray
        Calculated strain rate :math:`\dot{\epsilon}` [1/s].

    Raises
    ------
    ValueError
        If `mode` is not 'local' or 'global', or if required inputs are missing.
    """
    if mode == 'local':
        if mixing_length is None:
            raise ValueError("mixing_length must be provided when mode='local'")
        return compute_strain_rate_local(viscous_velocity, mixing_length, eps=eps)
    elif mode == 'global':
        if radius is None or temperature is None:
            raise ValueError("radius and temperature must be provided when mode='global'")
        return compute_strain_rate_global(
            radius, temperature, np.asarray(viscous_velocity), t_lid_base=t_lid_base, eps=eps
        )
    else:
        raise ValueError(f"Unknown stress closure mode {mode!r}; expected 'local' or 'global'")
