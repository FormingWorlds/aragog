"""Solid-state mantle rheology formulations.

Provides temperature- and pressure-dependent Arrhenius diffusion creep viscosity,
Byerlee yield stress, boundary-layer stress closure, effective viscosity capping,
and viscosity-contrast-dependent mixing-length scaling for solid-state convection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import numpy.typing as npt

R_GAS = 8.314462618
COHESION_FLOOR = 1.0e-6
DEFAULT_VISCOSITY_SOLID = 1.0e21

FloatOrArray = float | npt.NDArray[np.floating]


def _positive(v: float) -> bool:
    return v > 0.0


def _non_negative(v: float) -> bool:
    return v >= 0.0


# (field, in-range test, expectation text) for the finite-and-bounded numeric fields.
_BOUNDS = (
    *((n, lambda v: 0.0 < v <= 1.0, 'in (0, 1]') for n in ('mlt_bottom_slope', 's_top_inf')),
    ('c_theta', _non_negative, 'non-negative and finite'),
    ('g_homologous', _non_negative, 'non-negative and finite'),
    ('t_sol_0', _positive, 'positive and finite'),
    ('dt_sol_dp', _non_negative, 'non-negative and finite'),
    ('arrhenius_t_ref', _positive, 'positive'),
    ('viscosity_max_log10', lambda v: 20.0 < v <= 300.0, 'in (20, 300]'),
    ('water_prefactor', _positive, 'positive'),
    ('yield_stress_c', _non_negative, 'non-negative'),
    ('yield_stress_mu', _non_negative, 'non-negative'),
    ('yield_stress_max', _positive, 'positive'),
    ('interior_flux_fraction', lambda v: 0.05 <= v <= 0.5, 'in [0.05, 0.5]'),
    ('lid_base_temperature', _positive, 'positive'),
    ('lid_contrast_coeff', _positive, 'positive'),
    ('lid_mask_width_cells', _positive, 'positive'),
    ('phi_visc_single', lambda v: 0.0 < v < 1.0, 'in (0, 1)'),
)


@dataclass(frozen=True, eq=True)
class SolidRheologyParams:
    """Parameters for solid-state mantle rheology and stagnant lid scaling.

    This class is the single source of truth for rheology parameter defaults
    and bounds validation across Aragog and PROTEUS.

    Notes
    -----
    At the default ``g_homologous = 25.0`` and ``t_sol_0 = 1400.0`` K, the
    zero-pressure activation enthalpy is :math:`H(0) = 291` kJ/mol, close to a
    standard olivine diffusion-creep activation energy :math:`E_a = 300` kJ/mol
    (Karato & Wu 1993). The linear solidus fallback defaults ``t_sol_0 = 1400.0`` K
    and ``dt_sol_dp = 2.4e-8`` K/Pa (24 K/GPa) approximate the whole-mantle secant
    of the SPIDER phase-boundary solidus table read by Aragog (1429 K at 0 GPa to
    4595 K at 135 GPa, slope 23.5 K/GPa). Because the physical solidus is concave
    downward, the linear form is about 30 % low at 25 GPa (2000 K versus 2969 K)
    and serves table-free constant-property benchmarks only.
    """

    enabled: bool = False
    g_homologous: float = 25.0
    t_sol_0: float = 1400.0
    dt_sol_dp: float = 2.4e-8
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
    # Viscous-branch mixing-length boundary slopes and theta-dependent top-slope parameters.
    mlt_bottom_slope: float = 1.0
    s_top_inf: float = 0.22
    c_theta: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f'enabled must be a bool, got {self.enabled!r}')
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
        for name, in_range, expected in _BOUNDS:
            val = getattr(self, name)
            if isinstance(val, (bool, np.bool_)):
                raise TypeError(f'{name} must be a real number, got {val!r}')
            if not math.isfinite(val) or not in_range(val):
                raise ValueError(f'{name} must be {expected}, got {val}')
        if self.enabled and self.g_homologous <= 0.0:
            raise ValueError(
                'Invalid combination: enabled=True requires positive '
                f'g_homologous, got g_homologous={self.g_homologous}'
            )


def merge_rheology(
    rheology: SolidRheologyParams | dict[str, Any] | None, overrides: dict[str, Any]
) -> SolidRheologyParams:
    """Return ``rheology`` (default parameters when None) with field ``overrides`` applied."""
    if isinstance(rheology, dict):
        base = SolidRheologyParams(**rheology)
    elif rheology is None:
        base = SolidRheologyParams()
    else:
        base = rheology
    return replace(base, **overrides) if overrides else base


def delegate_rheology(cls):
    """Expose SolidRheologyParams fields as read-only properties reading self.rheology."""
    for f in fields(SolidRheologyParams):
        setattr(cls, f.name, property(lambda self, n=f.name: getattr(self.rheology, n)))
    return cls


def _require_x64_if_jax(xp: Any) -> None:
    """Raise RuntimeError when evaluating on the JAX path with 64-bit precision disabled."""
    if xp is not np:
        import jax

        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                'aragog.rheology requires JAX 64-bit precision (jax_enable_x64=True); '
                "enable it with jax.config.update('jax_enable_x64', True)."
            )


def _unwrap_scalar(result: Any, xp: Any, *inputs: Any) -> Any:
    """Return a Python float for NumPy results when every input is a scalar."""
    if xp is np and all(np.ndim(x) == 0 for x in inputs):
        return float(result.item())
    return result


def compute_mlt_top_slope(
    theta: FloatOrArray,
    s_top_inf: float | None = None,
    c_theta: float | None = None,
    xp: Any = np,
) -> FloatOrArray:
    r"""Evaluate viscosity-contrast-dependent upper mixing-length slope.

    .. math::
        s_\mathrm{top}(\theta) = s_{\mathrm{top},\infty}
            + (1 - s_{\mathrm{top},\infty}) \exp(-c_\theta \max(\theta, 0))

    Parameters
    ----------
    theta : float or array-like
        Frank-Kamenetskii viscosity contrast parameter [-].
    s_top_inf : float, optional
        Asymptotic upper mixing-length slope in the stagnant-lid limit [-].
        Defaults to ``SolidRheologyParams.s_top_inf``.
    c_theta : float, optional
        Exponential decay rate with respect to theta [-].
        Defaults to ``SolidRheologyParams.c_theta``.
    xp : module, default np
        Array namespace (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Upper mixing-length slope s_top(theta) [-].
    """
    _require_x64_if_jax(xp)
    if s_top_inf is None:
        s_top_inf = SolidRheologyParams.s_top_inf
    if c_theta is None:
        c_theta = SolidRheologyParams.c_theta
    theta_pos = xp.maximum(xp.asarray(theta, dtype=float), 0.0)
    s_top = s_top_inf + (1.0 - s_top_inf) * xp.exp(-c_theta * theta_pos)
    return _unwrap_scalar(s_top, xp, theta)


def viscous_mixing_length_factor(
    radii: Any,
    r_in: Any,
    r_out: Any,
    mixing_length: Any,
    s_top_inf: float | None = None,
    bottom_slope: float | None = None,
    theta: Any = None,
    c_theta: float | None = None,
    *,
    xp: Any = np,
) -> Any:
    r"""Factor on the viscous MLT velocity replacing the Abe length by the calibrated one.

    The viscous eddy diffusivity scales as ``l^4``, so with
    ``l_v = min(bottom_slope (r - r_in), s_top(theta) (r_out - r))`` (Wagner et al.
    2019, eq. 11, written with the two boundary slopes) it is multiplied by
    ``(l_v / l)^4``; the inviscid branch keeps ``l``. When ``theta`` is provided,
    the upper slope is evaluated via :func:`compute_mlt_top_slope`.

    Parameters
    ----------
    radii : array
        Basic-node radii [m].
    r_in, r_out : float
        Inner and outer radius of the mantle [m].
    mixing_length : array
        Abe mixing length ``l`` at the basic nodes [m].
    s_top_inf : float, optional
        Upper mixing-length slope at the surface when ``theta`` is None, or asymptotic
        upper slope in the stagnant-lid limit when ``theta`` is provided [-].
        Defaults to ``SolidRheologyParams.s_top_inf`` when None.
    bottom_slope : float, optional
        Lower mixing-length slope at the CMB [-]. Defaults to
        ``SolidRheologyParams.mlt_bottom_slope`` when None.
    theta : float or array-like, optional
        Frank-Kamenetskii viscosity contrast parameter [-]. When provided, evaluates
        the upper slope via :func:`compute_mlt_top_slope`.
    c_theta : float, optional
        Exponential decay rate with respect to ``theta`` [-]. Defaults to
        ``SolidRheologyParams.c_theta`` when None.
    xp : module, default np
        Array namespace (keyword-only).

    Returns
    -------
    array
        ``(l_v / l)^4``, and 1 where ``l`` is 0 (the boundary nodes).
    """
    _require_x64_if_jax(xp)
    if s_top_inf is None:
        s_top_inf = SolidRheologyParams.s_top_inf
    if bottom_slope is None:
        bottom_slope = SolidRheologyParams.mlt_bottom_slope
    eff_top_slope = s_top_inf
    if theta is not None:
        eff_top_slope = compute_mlt_top_slope(
            theta, s_top_inf=s_top_inf, c_theta=c_theta, xp=xp
        )
    l_v = xp.minimum(bottom_slope * (radii - r_in), eff_top_slope * (r_out - radii))
    positive = mixing_length > 0.0
    return xp.where(
        xp.isnan(mixing_length) | xp.isnan(l_v),
        xp.nan,
        xp.where(positive, (l_v / xp.where(positive, mixing_length, 1.0)) ** 4, 1.0),
    )


def compute_arrhenius_enthalpy(
    pressure: FloatOrArray,
    g_homologous: float | None = None,
    solidus_temperature: FloatOrArray | None = None,
    t_sol_0: float | None = None,
    dt_sol_dp: float | None = None,
    r_gas: float = R_GAS,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute homologous-temperature activation enthalpy H(P) [J/mol].

    .. math::
        H(P) = g_\mathrm{hom} R T_\mathrm{sol}(P)

    When ``solidus_temperature`` is not provided, the linear fallback
    :math:`T_\mathrm{sol}(P) = T_{\mathrm{sol},0} + (dT_\mathrm{sol}/dP) P` is
    used; it serves table-free benchmarks only (see ``SolidRheologyParams``).

    Parameters
    ----------
    pressure : float or array-like
        Pressure P [Pa].
    g_homologous : float, optional
        Dimensionless homologous activation parameter g_hom [-]. Defaults to
        ``SolidRheologyParams.g_homologous`` when None.
    solidus_temperature : float or array-like, optional
        Local solidus temperature T_sol(P) [K].
    t_sol_0 : float, optional
        Zero-pressure solidus T_sol(0) [K] for the linear fallback
        (``SolidRheologyParams.t_sol_0`` when None).
    dt_sol_dp : float, optional
        Linear solidus slope dT_sol/dP [K/Pa] for the linear fallback. Defaults to
        ``SolidRheologyParams.dt_sol_dp`` when None.
    r_gas : float, default 8.314462618
        Universal gas constant R [J/(mol K)].
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Activation enthalpy H(P) [J/mol].
    """
    _require_x64_if_jax(xp)
    if g_homologous is None:
        g_homologous = SolidRheologyParams.g_homologous
    if dt_sol_dp is None:
        dt_sol_dp = SolidRheologyParams.dt_sol_dp
    p = xp.asarray(pressure, dtype=float)
    if solidus_temperature is not None:
        t_sol = xp.asarray(solidus_temperature, dtype=float) + xp.zeros_like(p)
    else:
        t_sol = (SolidRheologyParams.t_sol_0 if t_sol_0 is None else t_sol_0) + dt_sol_dp * p
    h = g_homologous * r_gas * t_sol
    return _unwrap_scalar(h, xp, pressure, solidus_temperature)


def compute_arrhenius_viscosity(
    temperature: FloatOrArray,
    pressure: FloatOrArray,
    viscosity_solid: float | None = None,
    g_homologous: float | None = None,
    solidus_temperature: FloatOrArray | None = None,
    t_sol_0: float | None = None,
    dt_sol_dp: float | None = None,
    arrhenius_t_ref: float | None = None,
    r_gas: float = R_GAS,
    viscosity_max_log10: float | None = None,
    water_prefactor: FloatOrArray | None = None,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute temperature- and pressure-dependent homologous Arrhenius viscosity.

    .. math::
        \eta_\mathrm{diff}(T, P) = \eta_0 f_\mathrm{water} \exp\left[
            g_\mathrm{hom} \left(
                \frac{T_\mathrm{sol}(P)}{T} - \frac{T_\mathrm{sol}(0)}{T_\mathrm{ref}}
            \right)
        \right]

    Parameters
    ----------
    temperature : float or array-like
        Temperature T [K].
    pressure : float or array-like
        Pressure P [Pa].
    viscosity_solid : float, optional
        Reference solid-state dynamic viscosity eta_0 [Pa s] at (T_ref, P=0).
        Defaults to ``DEFAULT_VISCOSITY_SOLID`` (1.0e21) when None.
    g_homologous : float, optional
        Dimensionless homologous activation parameter g_hom [-]. Defaults to
        ``SolidRheologyParams.g_homologous`` when None.
    solidus_temperature : float or array-like, optional
        Local solidus temperature T_sol(P) [K]. When None, uses the linear fallback
        ``t_sol_0 + dt_sol_dp * P``.
    t_sol_0 : float, optional
        Zero-pressure reference solidus temperature T_sol(0) [K]. Required when
        ``solidus_temperature`` is provided; defaults to
        ``SolidRheologyParams.t_sol_0`` for the linear fallback.
    dt_sol_dp : float, optional
        Linear solidus slope dT_sol/dP [K/Pa] for the linear fallback. Defaults to
        ``SolidRheologyParams.dt_sol_dp`` when None.
    arrhenius_t_ref : float, optional
        Reference temperature T_ref [K]. Defaults to
        ``SolidRheologyParams.arrhenius_t_ref`` when None.
    r_gas : float, default 8.314462618
        Universal gas constant R [J/(mol K)].
    viscosity_max_log10 : float, optional
        Upper bound on log10 dynamic viscosity [log10(Pa s)]. Defaults to
        ``SolidRheologyParams.viscosity_max_log10`` when None.
    water_prefactor : float or array-like, optional
        Multiplicative prefactor for hydration weakening. Defaults to
        ``SolidRheologyParams.water_prefactor`` when None. Must be positive;
        values ``<= 0`` raise ``ValueError`` when ``xp is np`` and return ``NaN``
        on the JAX path for JIT compatibility.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Dynamic diffusion creep viscosity eta_diff [Pa s].
    """
    _require_x64_if_jax(xp)
    if viscosity_solid is None:
        viscosity_solid = DEFAULT_VISCOSITY_SOLID
    if g_homologous is None:
        g_homologous = SolidRheologyParams.g_homologous
    if dt_sol_dp is None:
        dt_sol_dp = SolidRheologyParams.dt_sol_dp
    if arrhenius_t_ref is None:
        arrhenius_t_ref = SolidRheologyParams.arrhenius_t_ref
    if viscosity_max_log10 is None:
        viscosity_max_log10 = SolidRheologyParams.viscosity_max_log10
    if water_prefactor is None:
        water_prefactor = SolidRheologyParams.water_prefactor
    if arrhenius_t_ref <= 0.0:
        raise ValueError(f'arrhenius_t_ref must be positive, got {arrhenius_t_ref}')

    wp = xp.asarray(water_prefactor, dtype=float)
    if xp is np and np.any(wp <= 0.0):
        raise ValueError(f'water_prefactor must be positive, got {water_prefactor}')

    if solidus_temperature is not None and t_sol_0 is None:
        raise ValueError('t_sol_0 must be provided when solidus_temperature is given')

    t = xp.asarray(temperature, dtype=float)
    p = xp.asarray(pressure, dtype=float)

    t_sol_ref_0 = SolidRheologyParams.t_sol_0 if t_sol_0 is None else t_sol_0

    h_p = compute_arrhenius_enthalpy(
        pressure=p,
        g_homologous=g_homologous,
        solidus_temperature=solidus_temperature,
        t_sol_0=t_sol_ref_0,
        dt_sol_dp=dt_sol_dp,
        r_gas=r_gas,
        xp=xp,
    )
    h_0 = g_homologous * r_gas * t_sol_ref_0

    t_safe = xp.where(t > 0.0, t, 1.0)
    wp_pos = xp.where(wp > 0.0, wp, 1.0)
    exponent = h_p / (r_gas * t_safe) - h_0 / (r_gas * arrhenius_t_ref) + xp.log(wp_pos)

    max_exponent = (
        viscosity_max_log10 - xp.log10(xp.maximum(viscosity_solid, 1e-300))
    ) * math.log(10.0)
    clipped_exp = xp.clip(exponent, -700.0, max_exponent)
    result = viscosity_solid * xp.exp(clipped_exp)
    result = xp.minimum(result, 10.0**viscosity_max_log10)
    invalid = xp.isnan(t) | xp.isnan(p) | xp.isnan(wp) | (t <= 0.0) | ~(wp > 0.0)
    result = xp.where(invalid, xp.nan, result)
    return _unwrap_scalar(
        result, xp, temperature, pressure, water_prefactor, solidus_temperature
    )


def compute_yield_stress(
    pressure: FloatOrArray,
    yield_stress_c: float | None = None,
    yield_stress_mu: float | None = None,
    yield_stress_max: float | None = None,
    xp: Any = np,
) -> FloatOrArray:
    r"""Compute Byerlee frictional yield stress.

    .. math::
        \tau_y(P) = \max(0, \min(c + \mu P, \tau_\mathrm{max}))

    Parameters
    ----------
    pressure : float or array-like
        Pressure P [Pa].
    yield_stress_c : float, optional
        Cohesion c [Pa]. Defaults to ``SolidRheologyParams.yield_stress_c`` when None.
    yield_stress_mu : float, optional
        Friction coefficient mu [-]. Defaults to
        ``SolidRheologyParams.yield_stress_mu`` when None.
    yield_stress_max : float, optional
        Maximum yield stress ceiling [Pa]. Defaults to
        ``SolidRheologyParams.yield_stress_max`` when None.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).

    Returns
    -------
    float or array-like
        Yield stress tau_y [Pa].
    """
    _require_x64_if_jax(xp)
    if yield_stress_c is None:
        yield_stress_c = SolidRheologyParams.yield_stress_c
    if yield_stress_mu is None:
        yield_stress_mu = SolidRheologyParams.yield_stress_mu
    if yield_stress_max is None:
        yield_stress_max = SolidRheologyParams.yield_stress_max
    p = xp.asarray(pressure, dtype=float)
    tau_y = xp.maximum(0.0, xp.minimum(yield_stress_c + yield_stress_mu * p, yield_stress_max))
    return _unwrap_scalar(tau_y, xp, pressure)


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
    _require_x64_if_jax(xp)
    v = xp.asarray(viscous_velocity, dtype=float)
    l_mix = xp.asarray(mixing_length, dtype=float)
    denom = 2.0 * xp.maximum(l_mix, eps)
    return _unwrap_scalar(xp.abs(v) / denom, xp, viscous_velocity, mixing_length)


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
