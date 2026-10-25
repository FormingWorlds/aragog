"""Standalone 0-D parameterized stagnant-lid reference model for verification tests.

This module implements the independent 0-D stagnant-lid parameterized convection
and secular-cooling ODE model formulated from:

- Moresi & Solomatov (1998), Geophys. J. Int., 133, 669-682 (Eqs. 5-7, 9, 13-15, 20-26).
- Solomatov & Moresi (2000), J. Geophys. Res., 105(B9), 21795-21817 (Eqs. 8-12).
- Foley & Bercovici (2014), Geophys. J. Int., 199, 580-603 (Eqs. 17-18, 26-28, Sec. 4.2).
- Thiriet et al. (2019), Phys. Earth Planet. Inter., 286, 138-153 (Eqs. 1, 3, 6, 10-12, 15, 18).

It imports no ``aragog`` modules; all physical constants and closure parameters are
passed in via :class:`StagnantLid0DParams`.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

# Reference physical and model defaults for the 0-D stagnant lid reference model
R_GAS_DEFAULT: float = 8.314462618
COHESION_FLOOR_DEFAULT: float = 1.0e-6
VISCOSITY_CEILING_LOG10_WIDTH_DEFAULT: float = 0.05

# SolidRheologyParams solver defaults
G_HOMOLOGOUS_DEFAULT: float = 25.0
T_SOL_0_DEFAULT: float = 1400.0
DT_SOL_DP_DEFAULT: float = 2.4e-8
ARRHENIUS_T_REF_DEFAULT: float = 1600.0
VISCOSITY_MAX_LOG10_DEFAULT: float = 40.0
WATER_PREFACTOR_DEFAULT: float = 1.0
ACTIVATION_ENERGY_DEFAULT: float | None = None
ACTIVATION_VOLUME_DEFAULT: float = 0.0
ACTIVATION_VOLUME_DECAY_PRESSURE_DEFAULT: float = float('inf')
LID_BASE_MODE_DEFAULT: str = 'rheological'
LID_CONTRAST_COEFF_DEFAULT: float = 2.2
LID_BASE_TEMPERATURE_DEFAULT: float = 1400.0
YIELD_STRESS_C_DEFAULT: float = 50.0e6
YIELD_STRESS_MU_DEFAULT: float = 0.6
YIELD_STRESS_MAX_DEFAULT: float = 500.0e6
INTERIOR_FLUX_FRACTION_DEFAULT: float = 0.05


_POSITIVE_PARAM_FIELDS: tuple[str, ...] = (
    'r_cmb',
    'rho',
    'c_p',
    'alpha',
    'k_cond',
    'gravity',
    't_surf',
    'viscosity_solid',
    'r_gas',
    'g_homologous',
    't_sol_0',
    'arrhenius_t_ref',
    'viscosity_ceiling_width',
    'water_prefactor',
    'lid_contrast_coeff',
    'lid_base_temperature',
    'a_rh_fb',
    'c4_fb',
    'a_nu',
    'yield_stress_max',
    'yield_stress_floor',
    'yield_blend_ratio_width',
)

_NONNEG_PARAM_FIELDS: tuple[str, ...] = (
    'p_i_ref',
    'dt_sol_dp',
    'activation_volume',
    'yield_stress_c',
    'yield_stress_mu',
)


def _require_real_scalar(value: Any, name: str, *, allow_inf: bool = False) -> float:
    """Validate that ``value`` is a finite (or infinite when allowed) non-boolean real scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a real scalar, got {value!r}')
    fval = float(value)
    if math.isnan(fval) or (math.isinf(fval) and not allow_inf):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return fval


def _require_positive_scalar(value: Any, name: str, *, allow_pos_inf: bool = False) -> float:
    """Validate that ``value`` is a strictly positive real scalar."""
    fval = _require_real_scalar(value, name, allow_inf=allow_pos_inf)
    if fval <= 0.0:
        raise ValueError(f'{name} must be > 0, got {value!r}')
    return fval


def _require_nonneg_scalar(value: Any, name: str) -> float:
    """Validate that ``value`` is a non-negative finite real scalar."""
    fval = _require_real_scalar(value, name, allow_inf=False)
    if fval < 0.0:
        raise ValueError(f'{name} must be >= 0, got {value!r}')
    return fval


@dataclass(frozen=True, slots=True)
class StagnantLid0DParams:
    """Complete parameter set for the standalone 0-D stagnant-lid model.

    Parameters
    ----------
    r_surf : float
        Outer planetary radius R_surf [m] (must exceed ``r_cmb``).
    r_cmb : float
        Core-mantle boundary radius R_CMB [m] (> 0).
    rho : float
        Solid mantle density rho [kg/m^3] (> 0).
    c_p : float
        Solid mantle specific heat capacity c_p [J/(kg K)] (> 0).
    alpha : float
        Thermal expansivity alpha [1/K] (> 0).
    k_cond : float
        Solid thermal conductivity k [W/(m K)] (> 0).
    gravity : float
        Gravitational acceleration g [m/s^2] (> 0).
    t_surf : float
        Fixed surface temperature T_surf [K] (> 0).
    viscosity_solid : float
        Reference solid viscosity eta_solid at (arrhenius_t_ref, P=0) [Pa s] (> 0).
    p_i_ref : float
        Representative sub-lid interior pressure P_i [Pa] (>= 0).
    r_gas : float
        Universal gas constant R [J/(mol K)] (> 0).
    g_homologous : float
        Homologous enthalpy prefactor g_hom [-] (> 0).
    t_sol_0 : float
        Zero-pressure solidus temperature intercept T_sol,0 [K] (> 0).
    dt_sol_dp : float
        Linear solidus pressure slope dT_sol/dP [K/Pa] (>= 0).
    arrhenius_t_ref : float
        Reference Arrhenius temperature T_ref [K] (> 0).
    viscosity_max_log10 : float
        Softplus log10 viscosity ceiling log10(eta_max) [-] (> log10(viscosity_solid)).
    viscosity_ceiling_width : float
        Softmin2 smoothing width w_ceil in log10 space [-] (> 0).
    water_prefactor : float
        Hydration viscosity prefactor f_water [-] (> 0).
    activation_energy : float or None
        Explicit activation energy E_a [J/mol] (> 0), or None for homologous law.
    activation_volume : float
        Explicit zero-pressure activation volume V_0 [m^3/mol] (>= 0).
    activation_volume_decay_p : float
        Activation-volume exponential decay pressure P_decay [Pa] (> 0 or +inf).
    lid_base_mode : str
        Lid-base isotherm mode, either ``'rheological'`` or ``'fixed'``.
    lid_contrast_coeff : float
        Rheological lid-base temperature drop coefficient a_lid [-] (> 0).
    lid_base_temperature : float
        Fixed lid-base isotherm temperature T_base [K] (> t_surf in ``'fixed'`` mode).
    a_rh_fb : float
        Foley & Bercovici (2014) rheological sublayer constant a_rh,FB [-] (> 0).
    c4_fb : float
        Foley & Bercovici (2014) convective velocity prefactor C_4 [-] (> 0).
    a_nu : float
        Stagnant-lid Nusselt-Rayleigh scaling prefactor a_Nu [-] (> 0).
    yield_stress_c : float
        Byerlee lithospheric cohesion c [Pa] (>= 0).
    yield_stress_mu : float
        Byerlee friction coefficient mu [-] (>= 0).
    yield_stress_max : float
        Ductile yield-stress ceiling tau_max [Pa] (> 0).
    yield_stress_floor : float
        Positive stress floor in stress_ratio denominator [Pa] (> 0).
    yield_blend_ratio_width : float
        Hyperbolic-tangent transition width w_ratio for yield mobilisation [-] (> 0).
    """

    r_surf: float
    r_cmb: float
    rho: float
    c_p: float
    alpha: float
    k_cond: float
    gravity: float
    t_surf: float
    viscosity_solid: float
    p_i_ref: float
    r_gas: float
    g_homologous: float
    t_sol_0: float
    dt_sol_dp: float
    arrhenius_t_ref: float
    viscosity_max_log10: float
    viscosity_ceiling_width: float
    water_prefactor: float
    activation_energy: float | None
    activation_volume: float
    activation_volume_decay_p: float
    lid_base_mode: str
    lid_contrast_coeff: float
    lid_base_temperature: float
    a_rh_fb: float
    c4_fb: float
    a_nu: float
    yield_stress_c: float
    yield_stress_mu: float
    yield_stress_max: float
    yield_stress_floor: float
    yield_blend_ratio_width: float

    def __post_init__(self) -> None:
        """Validate all physical and closure parameters on construction."""
        for fname in _POSITIVE_PARAM_FIELDS:
            _require_positive_scalar(getattr(self, fname), fname)
        for fname in _NONNEG_PARAM_FIELDS:
            _require_nonneg_scalar(getattr(self, fname), fname)

        r_surf_val = _require_real_scalar(self.r_surf, 'r_surf')
        if r_surf_val <= self.r_cmb:
            raise ValueError(f'r_surf ({r_surf_val}) must exceed r_cmb ({self.r_cmb})')

        vmax_val = _require_real_scalar(self.viscosity_max_log10, 'viscosity_max_log10')
        if vmax_val <= math.log10(self.viscosity_solid) or vmax_val > 300.0:
            raise ValueError(
                f'viscosity_max_log10 ({vmax_val}) must be in (log10(viscosity_solid), 300]'
            )

        if self.activation_energy is not None:
            _require_positive_scalar(self.activation_energy, 'activation_energy')
        _require_positive_scalar(
            self.activation_volume_decay_p,
            'activation_volume_decay_p',
            allow_pos_inf=True,
        )

        if self.lid_base_mode not in ('rheological', 'fixed'):
            raise ValueError(
                f"lid_base_mode must be 'rheological' or 'fixed', got {self.lid_base_mode!r}"
            )
        if self.lid_base_mode == 'fixed' and self.lid_base_temperature <= self.t_surf:
            raise ValueError(
                f'lid_base_temperature ({self.lid_base_temperature}) must exceed t_surf ({self.t_surf})'
            )


def load_aragog_0d_params(
    *,
    rheology_py: Path,
    abe_solid_toml: Path,
    viscosity_solid: float,
    t_surf: float,
    p_i_depth_m: float,
    a_rh_fb: float,
    c4_fb: float,
    a_nu: float,
    yield_blend_ratio_width: float,
) -> StagnantLid0DParams:
    """Construct :class:`StagnantLid0DParams` from explicit solver defaults and configuration.

    Parameters
    ----------
    rheology_py : Path
        Unused path kept for backwards compatibility with existing test signatures.
    abe_solid_toml : Path
        Path to ``src/aragog/cfg/abe_solid.toml`` read via ``tomllib``.
    viscosity_solid : float
        Reference solid viscosity eta_solid [Pa s].
    t_surf : float
        Surface temperature T_surf [K].
    p_i_depth_m : float
        Representative sub-lid depth [m] used to compute ``P_i = rho * g * depth``.
    a_rh_fb : float
        Foley & Bercovici (2014) rheological sublayer constant a_rh,FB [-].
    c4_fb : float
        Foley & Bercovici (2014) velocity constant C_4 [-].
    a_nu : float
        Stagnant-lid Nusselt prefactor a_Nu [-].
    yield_blend_ratio_width : float
        Hyperbolic-tangent transition width w_ratio for yield mobilisation [-].

    Returns
    -------
    StagnantLid0DParams
        Validated parameter bundle for the 0-D model.
    """
    _ = rheology_py
    cfg = tomllib.loads(abe_solid_toml.read_text(encoding='utf-8'))
    rho = float(cfg['phase_solid']['density'])
    gravity = float(cfg['mesh']['gravitational_acceleration'])
    p_i_ref = rho * gravity * float(p_i_depth_m)

    return StagnantLid0DParams(
        r_surf=float(cfg['mesh']['outer_radius']),
        r_cmb=float(cfg['mesh']['inner_radius']),
        rho=rho,
        c_p=float(cfg['phase_solid']['heat_capacity']),
        alpha=float(cfg['phase_solid']['thermal_expansivity']),
        k_cond=float(cfg['phase_solid']['thermal_conductivity']),
        gravity=gravity,
        t_surf=float(t_surf),
        viscosity_solid=float(viscosity_solid),
        p_i_ref=p_i_ref,
        r_gas=R_GAS_DEFAULT,
        g_homologous=G_HOMOLOGOUS_DEFAULT,
        t_sol_0=T_SOL_0_DEFAULT,
        dt_sol_dp=DT_SOL_DP_DEFAULT,
        arrhenius_t_ref=ARRHENIUS_T_REF_DEFAULT,
        viscosity_max_log10=VISCOSITY_MAX_LOG10_DEFAULT,
        viscosity_ceiling_width=VISCOSITY_CEILING_LOG10_WIDTH_DEFAULT,
        water_prefactor=WATER_PREFACTOR_DEFAULT,
        activation_energy=ACTIVATION_ENERGY_DEFAULT,
        activation_volume=ACTIVATION_VOLUME_DEFAULT,
        activation_volume_decay_p=ACTIVATION_VOLUME_DECAY_PRESSURE_DEFAULT,
        lid_base_mode=LID_BASE_MODE_DEFAULT,
        lid_contrast_coeff=LID_CONTRAST_COEFF_DEFAULT,
        lid_base_temperature=LID_BASE_TEMPERATURE_DEFAULT,
        a_rh_fb=float(a_rh_fb),
        c4_fb=float(c4_fb),
        a_nu=float(a_nu),
        yield_stress_c=YIELD_STRESS_C_DEFAULT,
        yield_stress_mu=YIELD_STRESS_MU_DEFAULT,
        yield_stress_max=YIELD_STRESS_MAX_DEFAULT,
        yield_stress_floor=COHESION_FLOOR_DEFAULT,
        yield_blend_ratio_width=float(yield_blend_ratio_width),
    )


def softmin2_scalar(a: float, b: float, width: float) -> float:
    """Evaluate smooth minimum ``min(a, b) - width * log(1 + exp(-|a - b| / width))``.

    Parameters
    ----------
    a : float
        First scalar operand.
    b : float
        Second scalar operand (upper ceiling).
    width : float
        Positive smoothing width (> 0).

    Returns
    -------
    float
        Smooth minimum of ``a`` and ``b``.
    """
    a_val = _require_real_scalar(a, 'a')
    b_val = _require_real_scalar(b, 'b')
    w_val = _require_positive_scalar(width, 'width')
    return min(a_val, b_val) - w_val * math.log1p(math.exp(-abs(a_val - b_val) / w_val))


def compute_0d_activation_enthalpy(pressure: float, params: StagnantLid0DParams) -> float:
    """Compute activation enthalpy H(P) [J/mol] from the homologous or saturating law.

    Parameters
    ----------
    pressure : float
        Pressure P [Pa] (>= 0).
    params : StagnantLid0DParams
        Model parameter bundle.

    Returns
    -------
    float
        Activation enthalpy H(P) [J/mol].
    """
    p_val = _require_nonneg_scalar(pressure, 'pressure')
    if params.activation_energy is None:
        t_sol_p = params.t_sol_0 + params.dt_sol_dp * p_val
        return params.g_homologous * params.r_gas * t_sol_p

    if math.isinf(params.activation_volume_decay_p):
        return params.activation_energy + params.activation_volume * p_val

    p_dec = params.activation_volume_decay_p
    return params.activation_energy + params.activation_volume * p_dec * (
        -math.expm1(-p_val / p_dec)
    )


def compute_0d_interior_viscosity(
    t_i: float,
    p_i: float,
    params: StagnantLid0DParams,
) -> tuple[float, float]:
    """Compute Arrhenius diffusion-creep interior viscosity eta_i(T_i, P_i) via Eq. (4).

    Parameters
    ----------
    t_i : float
        Convecting interior temperature T_i [K] (> ``params.t_surf``).
    p_i : float
        Representative sub-lid pressure P_i [Pa] (>= 0).
    params : StagnantLid0DParams
        Model parameter bundle.

    Returns
    -------
    tuple[float, float]
        ``(eta_i_Pa_s, log10_eta_i)``.
    """
    t_val = _require_real_scalar(t_i, 't_i')
    if t_val <= params.t_surf:
        raise ValueError(f't_i ({t_val}) must exceed t_surf ({params.t_surf})')
    p_val = _require_nonneg_scalar(p_i, 'p_i')

    h_pi = compute_0d_activation_enthalpy(p_val, params)
    h_0 = compute_0d_activation_enthalpy(0.0, params)
    exponent = (
        h_pi / (params.r_gas * t_val)
        - h_0 / (params.r_gas * params.arrhenius_t_ref)
        + math.log(params.water_prefactor)
    )
    log10_raw = math.log10(params.viscosity_solid) + exponent / math.log(10.0)
    log10_eta_i = softmin2_scalar(
        log10_raw,
        params.viscosity_max_log10,
        params.viscosity_ceiling_width,
    )
    return 10.0**log10_eta_i, log10_eta_i


def evaluate_stagnant_lid_0d(
    t_i: float,
    params: StagnantLid0DParams,
    *,
    p_i: float | None = None,
    p_lid: float | None = None,
) -> dict[str, float]:
    """Evaluate all algebraic stagnant-lid diagnostics at interior temperature ``t_i``.

    Implements the 0-D stagnant-lid boundary-layer and yield-mobilisation closure
    from Moresi & Solomatov (1998), Foley & Bercovici (2014), and Thiriet et al. (2019).

    Parameters
    ----------
    t_i : float
        Convecting interior temperature T_i [K] (> ``params.t_surf``).
    params : StagnantLid0DParams
        Model parameter bundle.
    p_i : float or None, optional
        Override for sub-lid interior pressure P_i [Pa] (>= 0); defaults to ``params.p_i_ref``.
    p_lid : float or None, optional
        Override for lid-base pressure P_lid [Pa] (>= 0); defaults to ``rho * g * d_lid_unyielded``.

    Returns
    -------
    dict[str, float]
        Dictionary of dimensional 0-D stagnant-lid state diagnostics.
    """
    p_i_in = params.p_i_ref if p_i is None else p_i
    eta_i, log10_eta_i = compute_0d_interior_viscosity(t_i, p_i_in, params)
    t_val = float(t_i)
    p_i_val = float(p_i_in)
    if p_lid is not None:
        _require_nonneg_scalar(p_lid, 'p_lid')

    d_mantle = params.r_surf - params.r_cmb
    kappa = params.k_cond / (params.rho * params.c_p)

    h_pi = compute_0d_activation_enthalpy(p_i_val, params)
    h_0 = compute_0d_activation_enthalpy(0.0, params)

    dt_conv = t_val - params.t_surf
    dt_rh = (params.r_gas * (t_val**2)) / h_pi
    theta = dt_conv / dt_rh

    if params.lid_base_mode == 'rheological':
        t_lid_iso = t_val - params.lid_contrast_coeff * dt_rh
        lid_frac = max(1.0 - params.lid_contrast_coeff / theta, 0.0)
    else:
        t_lid_iso = params.lid_base_temperature
        lid_frac = min((t_lid_iso - params.t_surf) / dt_conv, 1.0)

    ra_i = (params.rho * params.gravity * params.alpha * dt_conv * (d_mantle**3)) / (
        kappa * eta_i
    )
    ra_rh = ra_i * params.a_rh_fb / theta

    nu_unyielded = params.a_nu * (theta ** (-4.0 / 3.0)) * (ra_i ** (1.0 / 3.0))
    delta_l_unyielded = min(d_mantle / nu_unyielded, d_mantle)
    d_lid_unyielded = lid_frac * delta_l_unyielded
    delta_rh = (params.a_rh_fb / theta) * delta_l_unyielded

    v_m = (kappa / d_mantle) * params.c4_fb * (ra_rh ** (2.0 / 3.0))
    tau_d = 2.0 * eta_i * v_m / d_mantle
    d_geom = max(d_lid_unyielded, delta_rh)
    tau_d_lid = 2.0 * eta_i * v_m / d_geom

    p_lid_val = params.rho * params.gravity * d_lid_unyielded if p_lid is None else float(p_lid)
    tau_y_lid = min(
        params.yield_stress_c + params.yield_stress_mu * p_lid_val,
        params.yield_stress_max,
    )

    tau_y_eff = max(tau_y_lid, params.yield_stress_floor)
    stress_ratio = tau_d_lid / tau_y_eff
    w_b = 0.5 * (1.0 + math.tanh((stress_ratio - 1.0) / params.yield_blend_ratio_width))
    thin_factor = min(1.0, (tau_y_eff / tau_d_lid) ** (1.0 / 3.0))
    blend_mult = (1.0 - w_b) + w_b * thin_factor
    d_lid_eff = d_lid_unyielded * blend_mult
    delta_l_eff = delta_l_unyielded * blend_mult

    f_top_planar = params.k_cond * dt_conv / delta_l_eff
    f_top_spherical = f_top_planar * ((params.r_surf - d_lid_eff) / params.r_surf)

    return {
        'T_i_K': t_val,
        'T_surf_K': params.t_surf,
        'P_i_Pa': p_i_val,
        'H_Pi_J_mol': h_pi,
        'H_0_J_mol': h_0,
        'eta_i_Pa_s': eta_i,
        'log10_eta_i': log10_eta_i,
        'dT_rh_K': dt_rh,
        'theta': theta,
        'T_lid_iso_K': t_lid_iso,
        'Ra_i': ra_i,
        'Ra_rh': ra_rh,
        'Nu_unyielded': nu_unyielded,
        'delta_l_unyielded_m': delta_l_unyielded,
        'lid_frac': lid_frac,
        'd_lid_unyielded_m': d_lid_unyielded,
        'delta_rh_m': delta_rh,
        'v_m_m_s': v_m,
        'tau_d_Pa': tau_d,
        'tau_d_lid_Pa': tau_d_lid,
        'P_lid_Pa': p_lid_val,
        'tau_y_lid_Pa': tau_y_lid,
        'stress_ratio': stress_ratio,
        'w_b_yield': w_b,
        'thin_factor': thin_factor,
        'd_lid_eff_m': d_lid_eff,
        'delta_l_eff_m': delta_l_eff,
        'F_top_planar_W_m2': f_top_planar,
        'F_top_spherical_W_m2': f_top_spherical,
    }


def integrate_stagnant_lid_0d(
    t_i_init: float,
    t_eval_s: Sequence[float] | np.ndarray,
    params: StagnantLid0DParams,
    *,
    f_bot: float = 0.0,
    h_rad: float = 0.0,
    conduction_geometry: str = 'spherical',
    heat_capacity_mode: str = 'isothermal',
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Integrate the 0-D mantle thermal evolution ODE Eq. (1) over ``t_eval_s``.

    Parameters
    ----------
    t_i_init : float
        Initial interior temperature T_i(t_0) [K] (> ``params.t_surf``).
    t_eval_s : Sequence[float] or np.ndarray
        Strictly increasing 1-D array of evaluation times [s] (>= 0, length >= 2).
    params : StagnantLid0DParams
        Model parameter bundle.
    f_bot : float, default 0.0
        Prescribed core-mantle boundary heat flux F_bot [W/m^2] (>= 0).
    h_rad : float, default 0.0
        Volumetric/specific internal heating rate H_rad [W/kg] (>= 0).
    conduction_geometry : str, default ``'spherical'``
        Surface flux conduction geometry, either ``'spherical'`` or ``'planar'``.
    heat_capacity_mode : str, default ``'isothermal'``
        Mantle effective heat capacity mode, either ``'isothermal'`` or ``'adiabatic'``.
    rtol : float, default 1.0e-10
        Relative tolerance for ``scipy.integrate.solve_ivp`` (> 0).
    atol : float, default 1.0e-12
        Absolute tolerance for ``scipy.integrate.solve_ivp`` (> 0).

    Returns
    -------
    dict[str, Any]
        Trajectory arrays and energy-balance diagnostics over ``t_eval_s``.
    """
    t0_val = _require_real_scalar(t_i_init, 't_i_init')
    if t0_val <= params.t_surf:
        raise ValueError(f't_i_init ({t0_val}) must exceed t_surf ({params.t_surf})')

    t_arr = np.asarray(t_eval_s, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 2 or not np.all(np.isfinite(t_arr)):
        raise ValueError('t_eval_s must be a 1-D finite sequence of at least 2 times')
    if t_arr[0] < 0.0:
        raise ValueError(f't_eval_s[0] must be >= 0, got {t_arr[0]}')
    if not np.all(np.diff(t_arr) > 0.0):
        raise ValueError('t_eval_s must be strictly increasing')

    f_bot_val = _require_nonneg_scalar(f_bot, 'f_bot')
    h_rad_val = _require_nonneg_scalar(h_rad, 'h_rad')
    if conduction_geometry not in ('spherical', 'planar'):
        raise ValueError(
            f"conduction_geometry must be 'spherical' or 'planar', got {conduction_geometry!r}"
        )
    if heat_capacity_mode not in ('isothermal', 'adiabatic'):
        raise ValueError(
            f"heat_capacity_mode must be 'isothermal' or 'adiabatic', got {heat_capacity_mode!r}"
        )
    rtol_val = _require_positive_scalar(rtol, 'rtol')
    atol_val = _require_positive_scalar(atol, 'atol')

    v_mantle = (4.0 / 3.0) * math.pi * (params.r_surf**3 - params.r_cmb**3)
    a_surf = 4.0 * math.pi * (params.r_surf**2)
    a_cmb = 4.0 * math.pi * (params.r_cmb**2)
    m_mantle = params.rho * v_mantle

    if heat_capacity_mode == 'adiabatic':
        r_grid = np.linspace(params.r_cmb, params.r_surf, 500)
        t_ad_ratio = np.exp(
            params.alpha * params.gravity * (params.r_surf - r_grid) / params.c_p
        )
        bar_ratio = float(
            np.trapezoid(t_ad_ratio * (r_grid**2), r_grid) / np.trapezoid(r_grid**2, r_grid)
        )
    else:
        bar_ratio = 1.0
    c_eff = m_mantle * params.c_p * bar_ratio

    flux_key = (
        'F_top_spherical_W_m2' if conduction_geometry == 'spherical' else 'F_top_planar_W_m2'
    )
    q_in_const = a_cmb * f_bot_val + m_mantle * h_rad_val

    def _rhs(_t: float, y: list[float]) -> list[float]:
        st = evaluate_stagnant_lid_0d(float(y[0]), params)
        q_out = a_surf * st[flux_key]
        return [(q_in_const - q_out) / c_eff]

    sol = solve_ivp(
        _rhs,
        (float(t_arr[0]), float(t_arr[-1])),
        [t0_val],
        t_eval=t_arr,
        method='RK45',
        rtol=rtol_val,
        atol=atol_val,
    )
    if not sol.success:
        raise RuntimeError(f'solve_ivp integration failed: {sol.message}')

    states = [evaluate_stagnant_lid_0d(float(tv), params) for tv in sol.y[0]]
    f_top_arr = np.array([st[flux_key] for st in states], dtype=float)
    q_out_arr = a_surf * f_top_arr
    q_in_arr = np.full_like(q_out_arr, q_in_const)
    dt_dt_arr = (q_in_arr - q_out_arr) / c_eff

    return {
        't_s': np.array(sol.t, dtype=float),
        'T_i_K': np.array(sol.y[0], dtype=float),
        'd_lid_eff_m': np.array([st['d_lid_eff_m'] for st in states], dtype=float),
        'delta_l_eff_m': np.array([st['delta_l_eff_m'] for st in states], dtype=float),
        'F_top_W_m2': f_top_arr,
        'Q_out_W': q_out_arr,
        'Q_in_W': q_in_arr,
        'dT_i_dt_K_s': dt_dt_arr,
        'C_eff_J_K': c_eff,
        'A_surf_m2': a_surf,
        'A_cmb_m2': a_cmb,
        'M_mantle_kg': m_mantle,
        'states': states,
    }
