"""Solid-state mantle rheology formulations.

Provides temperature- and pressure-dependent Arrhenius diffusion creep viscosity,
Byerlee yield stress, boundary-layer stress closure, effective viscosity capping,
and viscosity-contrast-dependent mixing-length scaling for solid-state convection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import numpy.typing as npt

R_GAS = 8.314462618
COHESION_FLOOR = 1.0e-6
DEFAULT_VISCOSITY_SOLID = 1.0e21
PRESSURE_ROUNDOFF_TOLERANCE = 1.0
_MIN_ARRHENIUS_EXPONENT_BOUND = math.log(np.finfo(float).tiny) + 1.0e-5
VISCOSITY_CEILING_LOG10_WIDTH = 0.05

FloatOrArray = float | npt.NDArray[np.floating]


def _positive(v: float) -> bool:
    return math.isfinite(v) and v > 0.0


def _non_negative(v: float) -> bool:
    return math.isfinite(v) and v >= 0.0


def _in_vmax_range(v: Any) -> Any:
    return (v > 20.0) & (v <= 300.0)


# (field, in-range test, expectation text) for the bounded numeric fields.
_BOUNDS = (
    *((n, lambda v: 0.0 < v <= 1.0, 'in (0, 1]') for n in ('mlt_bottom_slope', 's_top_inf')),
    ('c_theta', _non_negative, 'non-negative and finite'),
    ('g_homologous', _non_negative, 'non-negative and finite'),
    ('t_sol_0', _positive, 'positive and finite'),
    ('dt_sol_dp', _non_negative, 'non-negative and finite'),
    ('arrhenius_t_ref', _positive, 'positive'),
    ('viscosity_max_log10', _in_vmax_range, 'in (20, 300]'),
    ('water_prefactor', _positive, 'positive'),
    ('yield_stress_c', _non_negative, 'non-negative'),
    ('yield_stress_mu', _non_negative, 'non-negative'),
    ('yield_stress_max', _positive, 'positive'),
    ('interior_flux_fraction', lambda v: 0.05 <= v <= 0.5, 'in [0.05, 0.5]'),
    ('lid_base_temperature', _positive, 'positive'),
    ('lid_contrast_coeff', _positive, 'positive'),
    ('lid_mask_width_cells', _positive, 'positive'),
    ('phi_visc_single', lambda v: 0.0 < v < 1.0, 'in (0, 1)'),
    ('activation_volume', _non_negative, 'non-negative and finite'),
    ('activation_volume_decay_pressure', lambda v: v > 0.0, 'positive (finite or +inf)'),
)


@dataclass(frozen=True, eq=True)
class SolidRheologyParams:
    r"""Parameters for solid-state mantle rheology and stagnant lid scaling.

    This class is the single source of truth for rheology parameter defaults
    and bounds validation across Aragog and PROTEUS.

    Notes
    -----
    Coupled and tabulated PALEOS runs evaluate the phase-boundary solidus
    :math:`T_\mathrm{sol}(P) = 0.8\,T_\mathrm{liq}(P)` from the PALEOS
    :math:`\mathrm{MgSiO_3}` melting curve. In both table-solidus and
    linear-fallback modes, ``t_sol_0`` sets the zero-pressure reference
    activation enthalpy ``h_0 = g_homologous * R * t_sol_0``, so
    ``viscosity_solid`` is the viscosity at ``(arrhenius_t_ref, P=0)`` on the
    linear solidus ``t_sol_0``; with a table solidus,
    ``eta(T_ref, 0) = viscosity_solid * exp(g_homologous * (T_sol,table(0) - t_sol_0) / T_ref)``
    (``1.8459 * viscosity_solid`` for the CI test table with
    ``T_sol,table(0) = 1439.230032 K`` and ``t_sol_0 = 1400 K``). When no
    solidus table reaches a call, the linear fallback defaults
    ``t_sol_0 = 1400.0`` K and ``dt_sol_dp = 2.4e-8`` K/Pa (24 K/GPa) provide a
    table-free benchmark form whose largest deviation from the PALEOS solidus
    over 0 to 135 GPa is 1108.1 K (see
    ``docs/Explanations/solid_state_convection.md`` Section 1).
    When ``activation_energy`` is set, activation enthalpy follows the saturating
    integral law :math:`H(P) = E_a + V_0 P_\mathrm{decay} (1 - \exp(-P / P_\mathrm{decay}))`
    (reducing to :math:`E_a + V_0 P` when :math:`P_\mathrm{decay} = \infty`); the
    published secant-form parameter sets of Tackley et al. (2013, Table 2) for
    :math:`E_0 + P V_0 \exp(-P / P_\mathrm{decay})` are not parameter sets of the
    integral law. See ``docs/Explanations/solid_state_convection.md`` for model
    formulation and derivations.
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
    # Optional saturating integral activation enthalpy law H(P) = E_a + V_0 P_decay (1 - exp(-P/P_decay)).
    activation_energy: float | None = None
    activation_volume: float = 0.0
    activation_volume_decay_pressure: float = float('inf')

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
            if math.isnan(val) or not in_range(val):
                raise ValueError(f'{name} must be {expected}, got {val}')
        if self.activation_energy is None:
            if self.activation_volume != 0.0:
                raise ValueError('activation_volume requires activation_energy to be set')
            if math.isfinite(float(self.activation_volume_decay_pressure)):
                raise ValueError(
                    'activation_volume_decay_pressure requires activation_energy to be set'
                )
            if self.enabled and self.g_homologous <= 0.0:
                raise ValueError(
                    'Invalid combination: enabled=True requires positive '
                    f'g_homologous, got g_homologous={self.g_homologous}'
                )
        else:
            ea = self.activation_energy
            if isinstance(ea, (bool, np.bool_)) or _is_non_real(ea) or np.ndim(ea) > 0:
                raise TypeError(f'activation_energy must be a real number, got {ea!r}')
            if not math.isfinite(float(ea)) or float(ea) < 0.0:
                raise ValueError(f'activation_energy must be non-negative and finite, got {ea}')
            if self.enabled and float(ea) <= 0.0:
                raise ValueError(
                    'Invalid combination: enabled=True requires positive '
                    f'activation_energy, got activation_energy={ea}'
                )


def merge_rheology(
    rheology: SolidRheologyParams | Mapping[str, Any] | None, overrides: Mapping[str, Any]
) -> SolidRheologyParams:
    """Return ``rheology`` (default parameters when None) with field ``overrides`` applied.

    Parameters
    ----------
    rheology : SolidRheologyParams or Mapping[str, Any] or None
        Base rheology parameter set, mapping of field values, or ``None`` for
        default :class:`SolidRheologyParams`.
    overrides : Mapping[str, Any]
        Mapping of :class:`SolidRheologyParams` field names to replacement values
        within the valid parameter bounds of :class:`SolidRheologyParams`.

    Returns
    -------
    SolidRheologyParams
        Validated rheology parameter set with ``overrides`` applied.

    Raises
    ------
    TypeError, ValueError
        If any field in ``rheology`` or ``overrides`` violates the type or range
        constraints of :class:`SolidRheologyParams`.
    """
    if isinstance(rheology, Mapping):
        base = SolidRheologyParams(**rheology)
    elif rheology is None:
        base = SolidRheologyParams()
    else:
        base = rheology
    return replace(base, **overrides) if overrides else base


def check_viscosity_ceiling(viscosity: Any, rheology: SolidRheologyParams) -> None:
    """Raise ValueError when a numeric ``viscosity`` is not below ``10**viscosity_max_log10``.

    Parameters
    ----------
    viscosity : Any
        Reference solid dynamic viscosity [Pa s]; finite positive scalar values
        in ``(0, 10**viscosity_max_log10)`` are accepted, while non-scalar or
        boolean values are ignored by this scalar ceiling check.
    rheology : SolidRheologyParams
        Rheology parameter set providing ``viscosity_max_log10`` (dimensionless,
        valid range ``(20, 300]``).

    Returns
    -------
    None
        Returns ``None`` when ``viscosity`` satisfies the ceiling.

    Raises
    ------
    ValueError
        If ``viscosity`` overflows ``float`` or if ``viscosity > 0`` is finite
        and ``rheology.viscosity_max_log10 <= log10(viscosity)``.
    """
    if isinstance(viscosity, (bool, np.bool_)):
        return
    is_num_scalar = isinstance(viscosity, (int, float, np.number)) or (
        getattr(viscosity, 'ndim', None) == 0
        and getattr(getattr(viscosity, 'dtype', None), 'kind', '') in 'iuf'
    )
    if is_num_scalar:
        try:
            v = float(viscosity)
        except OverflowError as exc:
            raise ValueError('viscosity must be finite') from exc
        if math.isfinite(v) and v > 0.0 and rheology.viscosity_max_log10 <= math.log10(v):
            raise ValueError(
                f'viscosity_max_log10 ({rheology.viscosity_max_log10}) must be > '
                f'log10(viscosity) ({math.log10(v)})'
            )


def delegate_rheology(cls):
    """Expose :class:`SolidRheologyParams` fields as read-only properties reading ``self.rheology``.

    Parameters
    ----------
    cls : type
        Target class holding a ``rheology`` attribute of type
        :class:`SolidRheologyParams`.

    Returns
    -------
    type
        The decorated class ``cls`` with one read-only property per field of
        :class:`SolidRheologyParams`.
    """
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


def _is_non_real(val: Any) -> bool:
    """Return True if ``val`` is or contains a boolean, string, complex, or non-numeric object."""
    if isinstance(val, (bool, np.bool_, str, bytes)):
        return True
    if isinstance(val, (int, float)):
        return False
    dtype = getattr(val, 'dtype', None)
    if dtype is not None:
        return dtype.kind not in 'iuf'
    if isinstance(val, (list, tuple)):
        return any(_is_non_real(x) for x in val)
    return True


def _require_scalar(val: Any, name: str) -> None:
    """Reject arrays, collections, and non-real types for scalar inputs."""
    if _is_non_real(val) or isinstance(val, (list, tuple)) or np.ndim(val) > 0:
        raise ValueError(f'{name} must be a scalar')


_RULES: dict[str, tuple[Any, str]] = {
    'pos': (lambda a: (a > 0.0) & (a < np.inf), 'positive and finite'),
    'nonneg': (lambda a: (a >= 0.0) & (a < np.inf), 'non-negative and finite'),
    'pressure': (
        lambda a: (a >= -PRESSURE_ROUNDOFF_TOLERANCE) & (a < np.inf),
        'non-negative and finite',
    ),
    'unit': (lambda a: (a >= 0.0) & (a <= 1.0), 'in [0, 1]'),
    'half_open_unit': (lambda a: (a > 0.0) & (a <= 1.0), 'in (0, 1]'),
    'finite': (lambda a: (a > -np.inf) & (a < np.inf), 'finite (no inf values)'),
    'vmax_scalar': (_in_vmax_range, 'in (20, 300]'),
    'nonneg_or_posinf': (lambda a: a >= 0.0, 'non-negative (or +inf)'),
    'pos_or_inf': (lambda a: a > 0.0, 'positive (finite or +inf)'),
}


def _clamp_pressure(pressure: Any, xp: Any) -> Any:
    """Clamp small negative pressures in [-PRESSURE_ROUNDOFF_TOLERANCE, 0.0) to 0.0."""
    p = xp.asarray(pressure, dtype=float)
    return xp.where((p < 0.0) & (p >= -PRESSURE_ROUNDOFF_TOLERANCE), 0.0, p)


def _softmax2(a: Any, b: Any, width: Any, xp: Any) -> Any:
    """Smooth softplus maximum ``a + width * log(1 + exp((b - a) / width))``."""
    a_arr = xp.asarray(a, dtype=float)
    b_arr = xp.asarray(b, dtype=float)
    w_arr = xp.asarray(width, dtype=float)
    both_fin = xp.isfinite(a_arr) & xp.isfinite(b_arr) & xp.isfinite(w_arr)
    s = xp.where(both_fin, xp.maximum(w_arr, 1.0e-30), 1.0)
    a_fin = xp.where(both_fin, a_arr, 0.0)
    b_fin = xp.where(both_fin, b_arr, 0.0)
    d = (b_fin - a_fin) / s
    soft = xp.where(
        d > 40.0,
        b_fin,
        xp.where(d < -40.0, a_fin, a_fin + s * xp.logaddexp(0.0, xp.clip(d, -500.0, 40.0))),
    )
    return xp.where(both_fin, soft, xp.maximum(a_arr, b_arr) + 0.0 * w_arr)


def _softmin2(a: Any, b: Any, width: Any, xp: Any) -> Any:
    """Smooth softplus minimum ``a - width * log(1 + exp((a - b) / width))``."""
    return -_softmax2(-xp.asarray(a, dtype=float), -xp.asarray(b, dtype=float), width, xp)


def _vmax_not_above_log10_viscosity(viscosity_solid: Any, vmax_log10: Any, xp: Any) -> Any:
    """Return the mask where ``vmax_log10 <= log10(viscosity_solid)``; raise on eager NumPy input."""
    vs = xp.asarray(viscosity_solid, dtype=float)
    ok_vs = (vs > 0.0) & xp.isfinite(vs)
    bad = ok_vs & xp.isfinite(xp.asarray(vmax_log10, dtype=float))
    bad = bad & (xp.asarray(vmax_log10, dtype=float) <= xp.log10(xp.where(ok_vs, vs, 1.0)))
    if xp is np and bool(np.any(bad)):
        v_bad = float(np.asarray(vs)[bad].flat[0])
        raise ValueError(
            f'viscosity_max_log10 ({vmax_log10}) must be > '
            f'log10(viscosity_solid) ({math.log10(v_bad)})'
        )
    return bad


def _validate_inputs(checks: Any, xp: Any, *, reduce_any: bool = False) -> Any:
    """Validate ``(name, value, rule, is_scalar)`` entries; raise ``ValueError`` on ``np`` or return a ``NaN`` mask."""
    invalid: Any = xp.asarray(False)
    for name, val, rule, is_scalar in checks:
        if val is None:
            continue
        if is_scalar:
            _require_scalar(val, name)
        elif _is_non_real(val):
            raise ValueError(f'{name} must be a real number or array, got {val!r}')
        ok, msg = _RULES[rule]
        try:
            arr = xp.asarray(val, dtype=float)
        except OverflowError as exc:
            raise ValueError(f'{name} must be {msg}, got {val}') from exc
        bad = ~ok(arr)
        if xp is np and np.any((~np.isnan(arr) if rule != 'vmax_scalar' else True) & bad):
            raise ValueError(f'{name} must be {msg}, got {val}')
        invalid = invalid | (xp.any(bad) if reduce_any else bad)
    return invalid


def _compute_mlt_top_slope_core(
    theta: FloatOrArray,
    s_top_inf: float | None = None,
    c_theta: float | None = None,
    xp: Any = np,
) -> Any:
    """Unchecked core evaluation of ``s_top(theta)``."""
    s_top_inf_eff = SolidRheologyParams.s_top_inf if s_top_inf is None else s_top_inf
    c_theta_eff = SolidRheologyParams.c_theta if c_theta is None else c_theta
    theta_pos = xp.maximum(xp.asarray(theta, dtype=float), 0.0)
    s_top_arr = xp.asarray(s_top_inf_eff, dtype=float)
    c_th_arr = xp.asarray(c_theta_eff, dtype=float)
    return s_top_arr + (1.0 - s_top_arr) * xp.exp(-c_th_arr * theta_pos)


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

    Raises
    ------
    ValueError
        If ``theta`` is None.
        If ``s_top_inf`` is not in ``(0, 1]`` or ``c_theta`` is negative on NumPy.
    """
    _require_x64_if_jax(xp)
    if theta is None:
        raise ValueError('theta must be a real number or array, got None')
    s_top_inf_eff = SolidRheologyParams.s_top_inf if s_top_inf is None else s_top_inf
    c_theta_eff = SolidRheologyParams.c_theta if c_theta is None else c_theta
    invalid = _validate_inputs(
        (
            ('theta', theta, 'finite', False),
            ('s_top_inf', s_top_inf_eff, 'half_open_unit', True),
            ('c_theta', c_theta_eff, 'nonneg', True),
        ),
        xp,
    )
    s_top = _compute_mlt_top_slope_core(
        theta, s_top_inf=s_top_inf_eff, c_theta=c_theta_eff, xp=xp
    )
    s_top = xp.where(invalid, xp.nan, s_top)
    return _unwrap_scalar(s_top, xp, theta)


def _viscous_mixing_length_factor_core(
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
    """Unchecked core evaluation of ``(l_v / l)^4``."""
    s_top_inf_eff = SolidRheologyParams.s_top_inf if s_top_inf is None else s_top_inf
    bottom_slope_eff = (
        SolidRheologyParams.mlt_bottom_slope if bottom_slope is None else bottom_slope
    )
    if theta is not None:
        eff_top_slope = _compute_mlt_top_slope_core(
            theta, s_top_inf=s_top_inf_eff, c_theta=c_theta, xp=xp
        )
    else:
        eff_top_slope = s_top_inf_eff
    r_arr = xp.asarray(radii, dtype=float)
    l_v = xp.minimum(bottom_slope_eff * (r_arr - r_in), eff_top_slope * (r_out - r_arr))
    ml_arr = xp.asarray(mixing_length, dtype=float)
    positive = ml_arr > 0.0
    return xp.where(
        xp.isnan(ml_arr) | xp.isnan(l_v),
        xp.nan,
        xp.where(positive, (l_v / xp.where(positive, ml_arr, 1.0)) ** 4, 1.0),
    )


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
    r"""Factor ``(l_v / l)^4`` on the viscous MLT eddy diffusivity for the calibrated length.

    The viscous eddy diffusivity scales as ``l^4`` and the viscous convective
    velocity scales as ``l^3``, so with
    ``l_v = min(bottom_slope (r - r_in), s_top(theta) (r_out - r))`` (Wagner et al.
    2019, eq. 11, written with the two boundary slopes) the diffusivity is
    multiplied by ``(l_v / l)^4`` and the viscous velocity by ``(l_v / l)^3``;
    the inviscid branch keeps ``l``. When ``theta`` is provided, the upper slope
    is evaluated via :func:`compute_mlt_top_slope`.

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

    Raises
    ------
    ValueError
        If ``radii``, ``r_in``, ``r_out``, or ``mixing_length`` is None.
        If ``r_in >= r_out`` or any finite radius lies outside ``[r_in, r_out]``
        beyond tolerance ``max(1 m, 1e-9 * (r_out - r_in))`` on NumPy.
        If ``c_theta`` is negative.
        On JAX (``xp is jnp``), domain violations evaluate the profile to NaN.
    """
    _require_x64_if_jax(xp)
    if radii is None:
        raise ValueError('radii must be a real number or array, got None')
    if r_in is None:
        raise ValueError('r_in must be a real number or array, got None')
    if r_out is None:
        raise ValueError('r_out must be a real number or array, got None')
    if mixing_length is None:
        raise ValueError('mixing_length must be a real number or array, got None')
    s_top_inf_eff = SolidRheologyParams.s_top_inf if s_top_inf is None else s_top_inf
    bottom_slope_eff = (
        SolidRheologyParams.mlt_bottom_slope if bottom_slope is None else bottom_slope
    )
    c_theta_eff = SolidRheologyParams.c_theta if c_theta is None else c_theta
    invalid = _validate_inputs(
        (
            ('radii', radii, 'finite', False),
            ('r_in', r_in, 'pos', True),
            ('r_out', r_out, 'pos', True),
            ('mixing_length', mixing_length, 'nonneg', False),
            ('s_top_inf', s_top_inf_eff, 'half_open_unit', True),
            ('bottom_slope', bottom_slope_eff, 'half_open_unit', True),
            ('c_theta', c_theta_eff, 'nonneg', True),
        ),
        xp,
    )
    r_arr = xp.asarray(radii, dtype=float)
    rin_val = xp.asarray(r_in, dtype=float)
    rout_val = xp.asarray(r_out, dtype=float)
    if xp is np:
        rin_finite = np.isfinite(rin_val)
        rout_finite = np.isfinite(rout_val)
        if rin_finite and rout_finite and float(rin_val) >= float(rout_val):
            raise ValueError(
                f'r_in must be strictly less than r_out, got r_in={r_in}, r_out={r_out}'
            )
        if rin_finite and rout_finite:
            tol = max(1.0, 1.0e-9 * (float(rout_val) - float(rin_val)))
            if np.any((r_arr < float(rin_val) - tol) | (r_arr > float(rout_val) + tol)):
                raise ValueError(
                    f'radii must lie within [r_in, r_out], got radii outside [{r_in}, {r_out}]'
                )
    else:
        tol = xp.maximum(1.0, 1.0e-9 * (rout_val - rin_val))
        domain_bad = (rin_val >= rout_val) | xp.any(
            (r_arr < rin_val - tol) | (r_arr > rout_val + tol)
        )
        invalid = invalid | domain_bad

    if theta is not None:
        eff_top_slope = compute_mlt_top_slope(
            theta, s_top_inf=s_top_inf_eff, c_theta=c_theta_eff, xp=xp
        )
        invalid = invalid | xp.isnan(eff_top_slope)
    else:
        eff_top_slope = s_top_inf_eff
    res = _viscous_mixing_length_factor_core(
        radii,
        r_in,
        r_out,
        mixing_length,
        s_top_inf=eff_top_slope,
        bottom_slope=bottom_slope_eff,
        theta=None,
        c_theta=c_theta_eff,
        xp=xp,
    )
    return xp.where(invalid, xp.nan, res)


def _compute_arrhenius_enthalpy_core(
    pressure: FloatOrArray,
    g_homologous: float | None = None,
    solidus_temperature: FloatOrArray | None = None,
    t_sol_0: float | None = None,
    dt_sol_dp: float | None = None,
    r_gas: float = R_GAS,
    xp: Any = np,
    *,
    activation_energy: float | None = None,
    activation_volume: float | None = None,
    activation_volume_decay_pressure: float | None = None,
) -> Any:
    """Unchecked core evaluation of activation enthalpy ``H(P)``."""
    p = _clamp_pressure(pressure, xp)
    p_safe = xp.where(xp.isfinite(p) & (p >= 0.0), p, 0.0)
    if activation_energy is not None:
        v_eff = (
            SolidRheologyParams.activation_volume
            if activation_volume is None
            else activation_volume
        )
        p_dec_eff = (
            SolidRheologyParams.activation_volume_decay_pressure
            if activation_volume_decay_pressure is None
            else activation_volume_decay_pressure
        )
        e_arr = xp.asarray(activation_energy, dtype=float)
        v_arr = xp.asarray(v_eff, dtype=float)
        p_dec_arr = xp.asarray(p_dec_eff, dtype=float)
        p_dec_safe = xp.where(p_dec_arr > 0.0, p_dec_arr, 1.0)
        p_dec_fin = xp.where(xp.isfinite(p_dec_safe), p_dec_safe, 1.0)
        x = p_safe / p_dec_safe
        x_fin = xp.where(x < 1.0e-8, 1.0, x)
        eff_p = xp.where(
            x < 1.0e-8,
            p_safe * (1.0 - 0.5 * x),
            -p_dec_fin * xp.expm1(-x_fin),
        )
        return e_arr + v_arr * eff_p + xp.zeros_like(p_safe)
    g_eff = SolidRheologyParams.g_homologous if g_homologous is None else g_homologous
    dt_eff = SolidRheologyParams.dt_sol_dp if dt_sol_dp is None else dt_sol_dp
    t_sol_0_eff = SolidRheologyParams.t_sol_0 if t_sol_0 is None else t_sol_0
    if solidus_temperature is not None:
        ts_arr = xp.asarray(solidus_temperature, dtype=float)
        t_sol = xp.where((ts_arr > 0.0) & xp.isfinite(ts_arr), ts_arr, 1.0) + xp.zeros_like(
            p_safe
        )
    else:
        t_sol = t_sol_0_eff + dt_eff * p_safe
    g_arr = xp.asarray(g_eff, dtype=float)
    return g_arr * r_gas * t_sol


def compute_arrhenius_enthalpy(
    pressure: FloatOrArray,
    g_homologous: float | None = None,
    solidus_temperature: FloatOrArray | None = None,
    t_sol_0: float | None = None,
    dt_sol_dp: float | None = None,
    r_gas: float = R_GAS,
    xp: Any = np,
    *,
    activation_energy: float | None = None,
    activation_volume: float | None = None,
    activation_volume_decay_pressure: float | None = None,
) -> FloatOrArray:
    r"""Compute homologous-temperature or saturating-integral activation enthalpy H(P) [J/mol].

    When ``activation_energy`` is ``None`` (the default), evaluates the homologous law:

    .. math::
        H(P) = g_\mathrm{hom} R T_\mathrm{sol}(P)

    When ``activation_energy`` is provided, evaluates the saturating integral law:

    .. math::
        H(P) = E_a + \int_0^P V_0 \exp(-p' / P_\mathrm{decay}) \, dp'
             = E_a + V_0 P_\mathrm{decay} \left(1 - \exp(-P / P_\mathrm{decay})\right)

    which reduces to :math:`H(P) = E_a + V_0 P` when :math:`P_\mathrm{decay} = \infty`
    and to :math:`H(P) = E_a` when :math:`V_0 = 0`. Note that the published secant-form
    parameter sets of Tackley et al. (2013, Table 2) for
    :math:`E_0 + P V_0 \exp(-P / P_\mathrm{decay})` are not parameter sets of the
    integral law.

    Parameters
    ----------
    pressure : float or array-like
        Pressure P [Pa] (non-negative; negative round-off excursions down to
        ``-PRESSURE_ROUNDOFF_TOLERANCE`` = -1.0 Pa are clamped to 0.0 Pa).
    g_homologous : float, optional
        Dimensionless homologous activation parameter g_hom [-]. Defaults to
        ``SolidRheologyParams.g_homologous`` when None. Not used when
        ``activation_energy`` is set.
    solidus_temperature : float or array-like, optional
        Local solidus temperature T_sol(P) [K]. Not used when
        ``activation_energy`` is set.
    t_sol_0 : float, optional
        Zero-pressure solidus T_sol(0) [K] for the linear fallback
        (``SolidRheologyParams.t_sol_0`` when None). Not used when
        ``activation_energy`` is set.
    dt_sol_dp : float, optional
        Linear solidus slope dT_sol/dP [K/Pa] for the linear fallback. Defaults to
        ``SolidRheologyParams.dt_sol_dp`` when None. Not used when
        ``activation_energy`` is set.
    r_gas : float, default 8.314462618
        Universal gas constant R [J/(mol K)].
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).
    activation_energy : float, optional
        Zero-pressure activation energy E_a [J/mol] for the saturating integral
        law. When None (default), uses the homologous law.
    activation_volume : float, optional
        Zero-pressure activation volume V_0 [m^3/mol] for the saturating integral
        law. Requires ``activation_energy`` to be set when non-zero.
    activation_volume_decay_pressure : float, optional
        Characteristic decay pressure P_decay [Pa] (> 0, finite or +inf) for the
        saturating integral law. Requires ``activation_energy`` to be set when finite.

    Returns
    -------
    float or array-like
        Activation enthalpy H(P) [J/mol].

    Raises
    ------
    ValueError
        If scalar arguments are non-scalar; if any input is non-real; if
        ``activation_volume`` or finite ``activation_volume_decay_pressure`` is
        given when ``activation_energy`` is None; or, on the eager NumPy path,
        if any parameter lies outside its valid domain. On the JAX path,
        out-of-range numeric values return ``NaN``.
    """
    _require_x64_if_jax(xp)
    if pressure is None:
        raise ValueError('pressure must be a real number or array, got None')
    if activation_energy is None:
        if activation_volume is not None and activation_volume != 0.0:
            raise ValueError('activation_volume requires activation_energy to be set')
        if activation_volume_decay_pressure is not None:
            if (
                _is_non_real(activation_volume_decay_pressure)
                or np.ndim(activation_volume_decay_pressure) > 0
                or not (
                    math.isinf(float(activation_volume_decay_pressure))
                    and float(activation_volume_decay_pressure) > 0.0
                )
            ):
                raise ValueError(
                    'activation_volume_decay_pressure requires activation_energy to be set'
                )
        g_hom_eff = SolidRheologyParams.g_homologous if g_homologous is None else g_homologous
        dt_sol_dp_eff = SolidRheologyParams.dt_sol_dp if dt_sol_dp is None else dt_sol_dp
        t_sol_0_eff = SolidRheologyParams.t_sol_0 if t_sol_0 is None else t_sol_0
        invalid = _validate_inputs(
            (
                ('pressure', pressure, 'pressure', False),
                ('g_homologous', g_hom_eff, 'nonneg', True),
                ('solidus_temperature', solidus_temperature, 'pos', False),
                ('t_sol_0', t_sol_0_eff, 'pos', True),
                ('dt_sol_dp', dt_sol_dp_eff, 'nonneg', True),
                ('r_gas', r_gas, 'pos', True),
            ),
            xp,
        )
        h = _compute_arrhenius_enthalpy_core(
            pressure=pressure,
            g_homologous=g_hom_eff,
            solidus_temperature=solidus_temperature,
            t_sol_0=t_sol_0_eff,
            dt_sol_dp=dt_sol_dp_eff,
            r_gas=r_gas,
            xp=xp,
        )
        return _unwrap_scalar(xp.where(invalid, xp.nan, h), xp, pressure, solidus_temperature)

    v_eff = (
        SolidRheologyParams.activation_volume
        if activation_volume is None
        else activation_volume
    )
    p_dec_eff = (
        SolidRheologyParams.activation_volume_decay_pressure
        if activation_volume_decay_pressure is None
        else activation_volume_decay_pressure
    )
    invalid = _validate_inputs(
        (
            ('pressure', pressure, 'pressure', False),
            ('activation_energy', activation_energy, 'nonneg', True),
            ('activation_volume', v_eff, 'nonneg', True),
            ('activation_volume_decay_pressure', p_dec_eff, 'pos_or_inf', True),
            ('solidus_temperature', solidus_temperature, 'pos', False),
            ('r_gas', r_gas, 'pos', True),
        ),
        xp,
    )
    h = _compute_arrhenius_enthalpy_core(
        pressure=pressure,
        r_gas=r_gas,
        xp=xp,
        activation_energy=activation_energy,
        activation_volume=v_eff,
        activation_volume_decay_pressure=p_dec_eff,
    )
    return _unwrap_scalar(xp.where(invalid, xp.nan, h), xp, pressure, solidus_temperature)


def _compute_arrhenius_viscosity_core(
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
    *,
    activation_energy: float | None = None,
    activation_volume: float | None = None,
    activation_volume_decay_pressure: float | None = None,
    _precomputed_enthalpy: Any = None,
) -> Any:
    """Unchecked core evaluation of homologous or saturating Arrhenius diffusion-creep viscosity."""
    vs_eff = DEFAULT_VISCOSITY_SOLID if viscosity_solid is None else viscosity_solid
    g_eff = SolidRheologyParams.g_homologous if g_homologous is None else g_homologous
    dt_eff = SolidRheologyParams.dt_sol_dp if dt_sol_dp is None else dt_sol_dp
    t_ref_eff = (
        SolidRheologyParams.arrhenius_t_ref if arrhenius_t_ref is None else arrhenius_t_ref
    )
    vmax_eff = (
        SolidRheologyParams.viscosity_max_log10
        if viscosity_max_log10 is None
        else viscosity_max_log10
    )
    wp_eff = SolidRheologyParams.water_prefactor if water_prefactor is None else water_prefactor
    t_sol_ref_0 = SolidRheologyParams.t_sol_0 if t_sol_0 is None else t_sol_0
    h_p = (
        _compute_arrhenius_enthalpy_core(
            pressure=pressure,
            g_homologous=g_eff,
            solidus_temperature=solidus_temperature,
            t_sol_0=t_sol_ref_0,
            dt_sol_dp=dt_eff,
            r_gas=r_gas,
            xp=xp,
            activation_energy=activation_energy,
            activation_volume=activation_volume,
            activation_volume_decay_pressure=activation_volume_decay_pressure,
        )
        if _precomputed_enthalpy is None
        else _precomputed_enthalpy
    )
    t = xp.asarray(temperature, dtype=float)
    wp = xp.asarray(wp_eff, dtype=float)
    vs = xp.asarray(vs_eff, dtype=float)
    g_arr = xp.asarray(g_eff, dtype=float)
    t_ref_arr = xp.asarray(t_ref_eff, dtype=float)
    vmax_log10 = xp.asarray(vmax_eff, dtype=float)
    h_0 = (
        xp.asarray(activation_energy, dtype=float)
        if activation_energy is not None
        else g_arr * r_gas * t_sol_ref_0
    )

    # Safe operands keep jax.grad finite at nodes the invalid mask sets to NaN.
    t_safe = xp.where((t > 0.0) & xp.isfinite(t), t, 1.0)
    t_ref_safe = xp.where((t_ref_arr > 0.0) & xp.isfinite(t_ref_arr), t_ref_arr, 1.0)
    wp_pos = xp.where((wp > 0.0) & xp.isfinite(wp), wp, 1.0)
    h_p_safe = xp.where(xp.isnan(h_p), 0.0, h_p)
    exponent = h_p_safe / (r_gas * t_safe) - h_0 / (r_gas * t_ref_safe) + xp.log(wp_pos)

    vs_pos = xp.where((vs > 0.0) & xp.isfinite(vs), vs, 1.0)
    # Prevent float64 underflow to zero when viscosity_solid is small.
    min_exponent = xp.maximum(
        -700.0, _MIN_ARRHENIUS_EXPONENT_BOUND - xp.log(xp.maximum(vs_pos, 1.0e-300))
    )
    log10_raw = xp.log10(xp.maximum(vs_pos, 1.0e-300)) + exponent / math.log(10.0)
    w_cap = VISCOSITY_CEILING_LOG10_WIDTH
    d_cap = (log10_raw - vmax_log10) / w_cap
    log10_soft = _softmin2(log10_raw, vmax_log10, w_cap, xp)
    max_exponent = xp.minimum(
        (vmax_log10 - xp.log10(xp.maximum(vs_pos, 1.0e-300))) * math.log(10.0), 700.0
    )
    log10_clipped = xp.maximum(log10_soft, -300.0)
    # Above 700 the clip on exp would cut the value; the log10 branch is exact there.
    result = xp.where(
        (d_cap < -40.0) & (exponent <= max_exponent),
        vs * xp.exp(xp.clip(exponent, min_exponent, max_exponent)),
        10.0**log10_clipped,
    )
    return xp.where(xp.isnan(h_p), xp.nan, result)


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
    *,
    activation_energy: float | None = None,
    activation_volume: float | None = None,
    activation_volume_decay_pressure: float | None = None,
) -> FloatOrArray:
    r"""Compute temperature- and pressure-dependent Arrhenius diffusion-creep viscosity.

    .. math::
        \eta_\mathrm{diff}(T, P) = \eta_0 f_\mathrm{water} \exp\left(
            \frac{H(P)}{R T} - \frac{H(0)}{R T_\mathrm{ref}}
        \right)

    where :math:`H(P) = g_\mathrm{hom} R T_\mathrm{sol}(P)` when
    ``activation_energy`` is ``None`` (default) and
    :math:`H(P) = E_a + V_0 P_\mathrm{decay} (1 - \exp(-P / P_\mathrm{decay}))`
    when ``activation_energy`` is set. Note that the published secant-form
    parameter sets of Tackley et al. (2013, Table 2) for
    :math:`E_0 + P V_0 \exp(-P / P_\mathrm{decay})` are not parameter sets of
    the integral law.

    Parameters
    ----------
    temperature : float or array-like
        Temperature T [K].
    pressure : float or array-like
        Pressure P [Pa] (non-negative; negative round-off excursions down to
        ``-PRESSURE_ROUNDOFF_TOLERANCE`` = -1.0 Pa are clamped to 0.0 Pa).
    viscosity_solid : float, optional
        Reference solid-state dynamic viscosity eta_0 [Pa s] at ``(T_ref, P=0)``
        on the linear solidus ``t_sol_0`` (since ``h_0 = g_hom * R * t_sol_0``;
        with a table solidus ``T_sol,table(P)``,
        ``eta_diff(T_ref, 0) = viscosity_solid * f_water * exp(g_hom * (T_sol,table(0) - t_sol_0) / T_ref)``,
        which is ``1.8459 * viscosity_solid`` for the CI test table with
        ``T_sol,table(0) = 1439.230032 K`` and ``t_sol_0 = 1400 K``).
        Defaults to ``DEFAULT_VISCOSITY_SOLID`` (1.0e21) when None.
    g_homologous : float, optional
        Dimensionless homologous activation parameter g_hom [-]. Defaults to
        ``SolidRheologyParams.g_homologous`` when None. Not used when
        ``activation_energy`` is set.
    solidus_temperature : float or array-like, optional
        Local solidus temperature T_sol(P) [K]. When None, uses the linear fallback
        ``t_sol_0 + dt_sol_dp * P``. Not used when ``activation_energy`` is set.
    t_sol_0 : float, optional
        Zero-pressure reference solidus temperature ``T_sol,0`` [K] setting the
        reference activation enthalpy ``h_0 = g_hom * R * t_sol_0`` in both
        table-solidus and linear-fallback modes (and the zero-pressure intercept
        of the linear fallback when ``solidus_temperature`` is None). Required
        when ``solidus_temperature`` is provided and ``activation_energy`` is None;
        defaults to ``SolidRheologyParams.t_sol_0`` for the linear fallback. Not
        used when ``activation_energy`` is set.
    dt_sol_dp : float, optional
        Linear solidus slope dT_sol/dP [K/Pa] for the linear fallback. Defaults to
        ``SolidRheologyParams.dt_sol_dp`` when None. Not used when
        ``activation_energy`` is set.
    arrhenius_t_ref : float, optional
        Reference temperature T_ref [K]. Defaults to
        ``SolidRheologyParams.arrhenius_t_ref`` when None.
    r_gas : float, default 8.314462618
        Universal gas constant R [J/(mol K)].
    viscosity_max_log10 : float, optional
        Smooth log10 softplus ceiling on dynamic viscosity [log10(Pa s)] with
        width ``VISCOSITY_CEILING_LOG10_WIDTH`` (0.05). Defaults to
        ``SolidRheologyParams.viscosity_max_log10`` when None.
    water_prefactor : float or array-like, optional
        Multiplicative prefactor for hydration weakening. Defaults to
        ``SolidRheologyParams.water_prefactor`` when None. Must be positive;
        values ``<= 0`` raise ``ValueError`` when ``xp is np`` and return ``NaN``
        on the JAX path for JIT compatibility.
    xp : array module, default np
        Array namespace module (numpy or jax.numpy).
    activation_energy : float, optional
        Zero-pressure activation energy E_a [J/mol] for the saturating integral law.
    activation_volume : float, optional
        Zero-pressure activation volume V_0 [m^3/mol] for the saturating integral law.
    activation_volume_decay_pressure : float, optional
        Characteristic decay pressure P_decay [Pa] (> 0, finite or +inf) for the
        saturating integral law.

    Returns
    -------
    float or array-like
        Dynamic diffusion creep viscosity eta_diff [Pa s].

    Raises
    ------
    ValueError
        If ``solidus_temperature`` is provided without ``t_sol_0`` when
        ``activation_energy`` is None; if scalar parameters are non-scalar; if
        any input is non-real; or, on the eager NumPy path (``xp is np``), if
        any parameter lies outside its valid domain. On the JAX path,
        out-of-range numeric values return ``NaN``.
    """
    _require_x64_if_jax(xp)
    if temperature is None:
        raise ValueError('temperature must be a real number or array, got None')
    if pressure is None:
        raise ValueError('pressure must be a real number or array, got None')
    vs_eff = DEFAULT_VISCOSITY_SOLID if viscosity_solid is None else viscosity_solid
    g_hom_eff = SolidRheologyParams.g_homologous if g_homologous is None else g_homologous
    dt_sol_dp_eff = SolidRheologyParams.dt_sol_dp if dt_sol_dp is None else dt_sol_dp
    t_ref_eff = (
        SolidRheologyParams.arrhenius_t_ref if arrhenius_t_ref is None else arrhenius_t_ref
    )
    vmax_eff = (
        SolidRheologyParams.viscosity_max_log10
        if viscosity_max_log10 is None
        else viscosity_max_log10
    )
    wp_eff = SolidRheologyParams.water_prefactor if water_prefactor is None else water_prefactor
    if activation_energy is None and solidus_temperature is not None and t_sol_0 is None:
        raise ValueError('t_sol_0 must be provided when solidus_temperature is given')

    t_sol_ref_0 = SolidRheologyParams.t_sol_0 if t_sol_0 is None else t_sol_0
    invalid = _validate_inputs(
        (
            ('temperature', temperature, 'pos', False),
            ('viscosity_solid', vs_eff, 'pos', True),
            ('arrhenius_t_ref', t_ref_eff, 'pos', True),
            ('viscosity_max_log10', vmax_eff, 'vmax_scalar', True),
            ('water_prefactor', wp_eff, 'pos', False),
        ),
        xp,
    )
    invalid = invalid | _vmax_not_above_log10_viscosity(vs_eff, vmax_eff, xp)
    h_p = compute_arrhenius_enthalpy(
        pressure=pressure,
        g_homologous=g_hom_eff,
        solidus_temperature=solidus_temperature,
        t_sol_0=t_sol_ref_0,
        dt_sol_dp=dt_sol_dp_eff,
        r_gas=r_gas,
        xp=xp,
        activation_energy=activation_energy,
        activation_volume=activation_volume,
        activation_volume_decay_pressure=activation_volume_decay_pressure,
    )
    result = _compute_arrhenius_viscosity_core(
        temperature=temperature,
        pressure=pressure,
        viscosity_solid=vs_eff,
        g_homologous=g_hom_eff,
        solidus_temperature=solidus_temperature,
        t_sol_0=t_sol_ref_0,
        dt_sol_dp=dt_sol_dp_eff,
        arrhenius_t_ref=t_ref_eff,
        r_gas=r_gas,
        viscosity_max_log10=vmax_eff,
        water_prefactor=wp_eff,
        xp=xp,
        activation_energy=activation_energy,
        activation_volume=activation_volume,
        activation_volume_decay_pressure=activation_volume_decay_pressure,
        _precomputed_enthalpy=h_p,
    )
    result = xp.where(invalid | xp.isnan(h_p), xp.nan, result)
    return _unwrap_scalar(
        result, xp, temperature, pressure, water_prefactor, solidus_temperature
    )


def _compute_yield_stress_core(
    pressure: FloatOrArray,
    yield_stress_c: float | None = None,
    yield_stress_mu: float | None = None,
    yield_stress_max: float | None = None,
    xp: Any = np,
) -> Any:
    """Unchecked core evaluation of Byerlee yield stress."""
    c_eff = SolidRheologyParams.yield_stress_c if yield_stress_c is None else yield_stress_c
    mu_eff = SolidRheologyParams.yield_stress_mu if yield_stress_mu is None else yield_stress_mu
    max_eff = (
        SolidRheologyParams.yield_stress_max if yield_stress_max is None else yield_stress_max
    )
    p = _clamp_pressure(pressure, xp)
    return xp.maximum(0.0, xp.minimum(c_eff + mu_eff * p, max_eff))


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
        Pressure P [Pa] (non-negative; negative round-off excursions down to
        ``-PRESSURE_ROUNDOFF_TOLERANCE`` = -1.0 Pa are clamped to 0.0 Pa).
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

    Raises
    ------
    ValueError
        If ``yield_stress_c``, ``yield_stress_mu``, or ``yield_stress_max`` is
        not a scalar; if any input is non-real; or, on the eager NumPy path
        (``xp is np``), if ``pressure`` is below ``-PRESSURE_ROUNDOFF_TOLERANCE``
        (-1.0 Pa) or infinite, if ``yield_stress_c`` or ``yield_stress_mu`` is
        negative or infinite, or if ``yield_stress_max`` is non-positive or
        infinite. On the JAX path, out-of-range numeric values return ``NaN``.
    """
    _require_x64_if_jax(xp)
    if pressure is None:
        raise ValueError('pressure must be a real number or array, got None')
    c_eff = SolidRheologyParams.yield_stress_c if yield_stress_c is None else yield_stress_c
    mu_eff = SolidRheologyParams.yield_stress_mu if yield_stress_mu is None else yield_stress_mu
    max_eff = (
        SolidRheologyParams.yield_stress_max if yield_stress_max is None else yield_stress_max
    )
    invalid = _validate_inputs(
        (
            ('pressure', pressure, 'pressure', False),
            ('yield_stress_c', c_eff, 'nonneg', True),
            ('yield_stress_mu', mu_eff, 'nonneg', True),
            ('yield_stress_max', max_eff, 'pos', True),
        ),
        xp,
    )
    tau_y = _compute_yield_stress_core(
        pressure,
        yield_stress_c=c_eff,
        yield_stress_mu=mu_eff,
        yield_stress_max=max_eff,
        xp=xp,
    )
    return _unwrap_scalar(xp.where(invalid, xp.nan, tau_y), xp, pressure)


def _compute_strain_rate_local_core(
    viscous_velocity: FloatOrArray,
    mixing_length: FloatOrArray,
    eps: float = 1.0e-15,
    xp: Any = np,
) -> Any:
    """Unchecked core evaluation of local convective strain rate."""
    v = xp.asarray(viscous_velocity, dtype=float)
    l_mix = xp.asarray(mixing_length, dtype=float)
    denom = 2.0 * xp.maximum(l_mix, eps)
    return xp.abs(v) / denom


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

    Raises
    ------
    ValueError
        If ``viscous_velocity`` or ``mixing_length`` is None.
        If ``mixing_length`` is negative or infinite on NumPy.
    """
    _require_x64_if_jax(xp)
    if viscous_velocity is None:
        raise ValueError('viscous_velocity must be a real number or array, got None')
    if mixing_length is None:
        raise ValueError('mixing_length must be a real number or array, got None')
    invalid = _validate_inputs(
        (
            ('viscous_velocity', viscous_velocity, 'finite', False),
            ('mixing_length', mixing_length, 'nonneg', False),
            ('eps', eps, 'pos', True),
        ),
        xp,
    )
    res = _compute_strain_rate_local_core(viscous_velocity, mixing_length, eps=eps, xp=xp)
    res = xp.where(invalid, xp.nan, res)
    return _unwrap_scalar(res, xp, viscous_velocity, mixing_length)


_LID_EXPORTS = (
    'compute_stagnant_lid_state',
    'compute_effective_viscosity',
    'stress_closure',
    '_compute_stagnant_lid_state_core',
    '_compute_effective_viscosity_core',
)


def __getattr__(name: str) -> Any:
    if name in _LID_EXPORTS:
        import aragog.rheology_lid as _lid

        return getattr(_lid, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LID_EXPORTS))
