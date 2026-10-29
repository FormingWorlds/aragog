"""JAX functions of the sub-grid surface lid (see ``aragog.surface_lid`` for the model).

All functions are JIT-compilable and differentiable.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

DAWSON_H: float = 0.25
DAWSON_WINDOW: float = 8.0
DAWSON_TERMS: int = math.ceil(DAWSON_WINDOW / DAWSON_H) + 1


def dawson(x: jax.Array) -> jax.Array:
    """Dawson's integral F(x) = exp(-x^2) int_0^x exp(t^2) dt.

    Rybicki's sampling sum (Press et al. 2007, Numerical Recipes, 3rd ed., section 6.9),
    F(x) = lim_{h->0} pi^(-1/2) sum_{n odd} exp(-(x - n h)^2) / n, with the terms of n and -n
    paired as exp(-(x - n h)^2) (1 - exp(-4 n h x)) / n so that it has no cancellation near 0
    and no overflow at large x. With h = 0.25 and the terms with |x - n h| <= 8, the relative
    error against ``scipy.special.dawsn`` is below 1e-13 for |x| <= 75.

    Parameters
    ----------
    x : jax.Array
        Argument [-], any real value.

    Returns
    -------
    jax.Array
        F(x), odd in x, of the shape of ``x``.
    """
    x = jnp.asarray(x, dtype=float)
    sgn = jnp.where(x < 0.0, -1.0, 1.0)
    a = (sgn * x)[..., None]
    n_lo = jnp.maximum(1.0, 2.0 * jnp.floor((a - DAWSON_WINDOW) / (2.0 * DAWSON_H)) + 1.0)
    n = n_lo + 2.0 * jnp.arange(DAWSON_TERMS)
    terms = jnp.exp(-((a - n * DAWSON_H) ** 2)) * -jnp.expm1(-4.0 * DAWSON_H * n * a) / n
    return sgn * terms.sum(-1) / math.sqrt(math.pi)


def shape(s: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Slopes at the top and the base, and the mean, of the similarity profile f_s.

    The JAX form of :func:`aragog.surface_lid.shape`: ``(f'(0), f'(1), m)`` for the shape
    parameter ``s`` [-], with the series below ``|s| = 1e-6`` and Dawson's integral for
    ``s < 0``. Each branch is evaluated on a safe argument, so the gradient has no NaN.
    """
    from aragog.surface_lid import _S_SERIES

    s = jnp.asarray(s, dtype=float)
    series = (1.0 + s / 3.0, 1.0 - 2.0 * s / 3.0, 0.5 + s / 12.0)
    sp = jnp.where(s >= _S_SERIES, s, 1.0)
    mu = jnp.sqrt(sp)
    g = mu * math.sqrt(math.pi) * jax.scipy.special.erf(mu)
    grow = (2.0 * sp / g, 2.0 * sp * jnp.exp(-sp) / g, 1.0 + jnp.expm1(-sp) / g)
    sn = jnp.where(s <= -_S_SERIES, s, -1.0)
    nu = jnp.sqrt(-sn)
    dw = dawson(nu)
    thin = (nu * jnp.exp(sn) / dw, nu / dw, 1.0 + jnp.expm1(sn) / (2.0 * nu * dw))
    branch = jnp.where(s >= _S_SERIES, 1, jnp.where(s <= -_S_SERIES, 2, 0))
    return tuple(
        jnp.select([branch == 1, branch == 2], [a, b], c) for a, b, c in zip(grow, thin, series)
    )


def grey_flux(T_s: jax.Array, emissivity: jax.Array, T_eq: jax.Array) -> jax.Array:
    """Grey-body flux [W/m^2] of a surface at ``T_s`` [K] against the equilibrium ``T_eq`` [K]."""
    from scipy.constants import Stefan_Boltzmann

    return emissivity * Stefan_Boltzmann * (jnp.maximum(T_s, 0.0) ** 4 - T_eq**4)


def profile(s, x):
    """The similarity profile f_s(x), as :func:`aragog.surface_lid.profile`."""
    from aragog.surface_lid import _S_SERIES

    s, x = jnp.asarray(s, dtype=float), jnp.asarray(x, dtype=float)
    sp = jnp.where(s >= _S_SERIES, s, 1.0)
    mu = jnp.sqrt(sp)
    grow = jax.scipy.special.erf(mu * x) / jax.scipy.special.erf(mu)
    sn = jnp.where(s <= -_S_SERIES, s, -1.0)
    nu = jnp.sqrt(-sn)
    thin = jnp.exp(sn * (1.0 - x * x)) * dawson(nu * x) / dawson(nu)
    series = x + s * x * (1.0 - x * x) / 3.0
    return jnp.where(s >= _S_SERIES, grow, jnp.where(s <= -_S_SERIES, thin, series))


def profile_nodes(s):
    """Quadrature nodes and weights on [0, 1], as :func:`aragog.surface_lid.profile_nodes`."""
    from aragog.surface_lid import _GL16, PROFILE_PANELS

    g, w = (jnp.asarray(v) for v in _GL16)
    s = jnp.asarray(s, dtype=float)
    c_grow = jnp.minimum(0.5, 6.0 / jnp.sqrt(jnp.where(s > 0.0, s, 1.0)))
    c_thin = 1.0 - jnp.minimum(0.5, -15.0 / jnp.where(s < 0.0, s, -1.0))
    c = jnp.where(s > 0.0, c_grow, jnp.where(s < 0.0, c_thin, 0.5))
    u = jnp.linspace(0.0, 1.0, PROFILE_PANELS + 1)
    edges = jnp.concatenate([c * u, c + (1.0 - c) * u[1:]])
    half = 0.5 * jnp.diff(edges)[:, None]
    return (edges[:-1, None] + half * (g + 1.0)).ravel(), (half * w).ravel()


class Enthalpy(NamedTuple):
    """Heat content per volume H(T) as cubic pieces on the knots ``T`` (coefficients ``c``,
    highest power first) with linear continuation, and the density on the same knots."""

    T: jax.Array
    c: jax.Array
    slopes: jax.Array
    rho: jax.Array


def enthalpy(hv) -> Enthalpy:
    """The JAX form of a :class:`aragog.surface_lid.LidEnthalpy`."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    kind, *v = hv.data
    if kind == 'linear':
        rho, cp, T_b, h_b = v
        T = np.array([T_b - 1.0e4, T_b + 1.0e4])
        H0 = rho * (h_b + cp * (T[0] - T_b))
        c = np.array([[0.0], [0.0], [rho * cp], [H0]])
        return Enthalpy(*(jnp.asarray(a) for a in (T, c, [rho * cp, rho * cp], [rho, rho])))
    T, H, rho = v
    spline = PchipInterpolator(T, H)
    d = spline.derivative()
    slopes = [float(d(T[0])), float(d(T[-1]))]
    return Enthalpy(*(jnp.asarray(a) for a in (T, spline.c, slopes, rho)))


def enthalpy_H(e: Enthalpy, T):
    """H(T) [J/m^3] of a JAX enthalpy table, linear outside its knots."""
    T = jnp.asarray(T, dtype=float)
    i = jnp.clip(jnp.searchsorted(e.T, T, side='right') - 1, 0, e.T.size - 2)
    dx = jnp.clip(T, e.T[0], e.T[-1]) - e.T[i]
    inside = ((e.c[0, i] * dx + e.c[1, i]) * dx + e.c[2, i]) * dx + e.c[3, i]
    H_lo, H_hi = e.c[3, 0], _H_end(e)
    below = H_lo + e.slopes[0] * (T - e.T[0])
    above = H_hi + e.slopes[1] * (T - e.T[-1])
    return jnp.where(T < e.T[0], below, jnp.where(T > e.T[-1], above, inside))


def _H_end(e: Enthalpy):
    dx = e.T[-1] - e.T[-2]
    return ((e.c[0, -1] * dx + e.c[1, -1]) * dx + e.c[2, -1]) * dx + e.c[3, -1]


def profile_heat(e: Enthalpy, s, T_s, T_b):
    """Mean heat content per volume [J/m^3] of a lid profile, as ``LidEnthalpy.profile_heat``."""
    x, w = profile_nodes(s)
    return jnp.dot(w, enthalpy_H(e, T_s + (T_b - T_s) * profile(s, x)))


def profile_density(e: Enthalpy, s, T_s, T_b):
    """Mean density [kg/m^3] of a lid profile, as ``LidEnthalpy.profile_density``."""
    x, w = profile_nodes(s)
    return jnp.dot(w, jnp.interp(T_s + (T_b - T_s) * profile(s, x), e.T, e.rho))


def _implicit_root(root, residual):
    """``root`` in value, with the implicit derivative of ``residual(root) = 0`` in the
    parameters that ``residual`` closes over."""
    root = jax.lax.stop_gradient(root)
    slope = jax.lax.stop_gradient(jax.grad(residual)(root))
    return root - (residual(root) - jax.lax.stop_gradient(residual(root))) / slope


def grey_surface_temperature(f0, d, T_b, k, emissivity, T_eq):
    """Surface temperature [K] of the grey-body balance, as
    :func:`aragog.surface_lid.grey_surface_temperature` (40 Newton steps from ``T_b``)."""
    from scipy.constants import Stefan_Boltzmann

    c = k * f0 / d

    def g(T):
        return grey_flux(T, emissivity, T_eq) - c * (T_b - T)

    def step(_, T):
        return T - g(T) / (4.0 * emissivity * Stefan_Boltzmann * T**3 + c)

    T = jax.lax.fori_loop(0, 40, step, jax.lax.stop_gradient(T_b))
    T = jnp.maximum(_implicit_root(T, g), T_eq)
    return jnp.where(T_b <= T_eq, T_b, T)


def _bracket_root(f, lo, hi, f_lo, f_hi):
    """Root of an increasing ``f`` with ``f_lo < 0 < f_hi`` at ``[lo, hi]``: 12 bisection steps,
    then 10 Illinois steps (regula falsi that halves the value of an end kept twice in a row).
    On the lid states of the tests it agrees with 64 bisection steps to relative 4.1e-13."""

    def bisect(_, c):
        lo, hi, f_lo, f_hi = c
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        below = f_mid < 0.0
        return (
            jnp.where(below, mid, lo),
            jnp.where(below, hi, mid),
            jnp.where(below, f_mid, f_lo),
            jnp.where(below, f_hi, f_mid),
        )

    def secant(lo, hi, f_lo, f_hi):
        den = f_hi - f_lo
        x = (lo * f_hi - hi * f_lo) / jnp.where(den > 0.0, den, 1.0)
        return jnp.where(den > 0.0, jnp.clip(x, lo, hi), 0.5 * (lo + hi))

    def illinois(_, c):
        lo, hi, f_lo, f_hi, kept = c
        x = secant(lo, hi, f_lo, f_hi)
        f_x = f(x)
        below = f_x < 0.0
        f_lo = jnp.where(below, f_x, jnp.where(kept == -1, 0.5 * f_lo, f_lo))
        f_hi = jnp.where(below, jnp.where(kept == 1, 0.5 * f_hi, f_hi), f_x)
        lo, hi = jnp.where(below, x, lo), jnp.where(below, hi, x)
        return lo, hi, f_lo, f_hi, jnp.where(below, 1, -1)

    c = jax.lax.fori_loop(0, 12, bisect, (lo, hi, f_lo, f_hi))
    c = jax.lax.fori_loop(0, 10, illinois, (*c, jnp.asarray(0)))
    return secant(*c[:4])


def surface_state(delta, E, T_b, k, emissivity, T_eq, delta_floor, e: Enthalpy):
    """Grey-body surface of a lid with energy ``E`` [J/m^2], as
    :func:`aragog.surface_lid.surface_state`.

    The root in ``s`` of ``Q(s) = E / delta`` comes from :func:`_bracket_root` over the shape
    bounds and carries the implicit derivative; at a clipped bound ``s`` is constant.

    Returns
    -------
    tuple of jax.Array
        ``(T_s, F_s, q_b, s, clipped, T_m)`` in the units and meaning of ``LidSurface``.
    """
    from aragog.surface_lid import LID_SHAPE_MAX, LID_SHAPE_MIN

    d = jnp.hypot(delta, delta_floor)
    q = E / jnp.maximum(delta, delta_floor)

    def T_s_of(s):
        return grey_surface_temperature(shape(s)[0], d, T_b, k, emissivity, T_eq)

    def residual(s):
        return profile_heat(e, s, T_s_of(s), T_b) - q

    lo, hi = jnp.asarray(LID_SHAPE_MIN), jnp.asarray(LID_SHAPE_MAX)
    r_min, r_max = (jax.lax.stop_gradient(residual(v)) for v in (lo, hi))
    root = _bracket_root(lambda v: jax.lax.stop_gradient(residual(v)), lo, hi, r_min, r_max)
    s_in = _implicit_root(root, residual)
    clipped = jnp.where(r_min >= 0.0, -1, jnp.where(r_max <= 0.0, 1, 0))
    s = jnp.select([clipped == -1, clipped == 1], [LID_SHAPE_MIN, LID_SHAPE_MAX], s_in)
    f0, f1, m = shape(s)
    T_s = T_s_of(s)
    dT = T_b - T_s
    return T_s, k * dT * f0 / d, k * dT * f1 / d, s, clipped, T_s + m * dT


def convective_base_flux(dT, k, rho, cp, alpha, g, eta, ra_crit):
    """Boundary-layer flux [W/m^2], as :func:`aragog.surface_lid.convective_base_flux`."""
    pos = dT > 0.0
    dT_safe = jnp.where(pos, dT, 1.0)
    kappa = k / (rho * cp)
    q = k * dT_safe ** (4.0 / 3.0) * (alpha * rho * g / (ra_crit * kappa * eta)) ** (1.0 / 3.0)
    return jnp.where(pos, q, 0.0)


def stefan_rate(q_b, F_b, rho, dh, delta, h_top, delta_lo):
    """Base growth rate [m/s] and stop factor, as :func:`aragog.surface_lid.stefan_rate`."""
    from aragog.surface_lid import LID_CAP_FRAC, LID_CAP_WIDTH, LID_DH_MIN

    r = (q_b - F_b) / (rho * jnp.maximum(dh, LID_DH_MIN))
    grow = jnp.clip((LID_CAP_FRAC - delta / h_top) / LID_CAP_WIDTH, 0.0, 1.0)
    melt = jnp.clip((delta - delta_lo) / delta_lo, 0.0, 1.0)
    f = jnp.where(r > 0.0, grow, jnp.where(r < 0.0, melt, 1.0))
    return r * f, f


def existence_weight(F_b, T_b, k, emissivity, T_eq, delta_on):
    """Weight [0, 1] of the lid, as :func:`aragog.surface_lid.existence_weight`."""
    from scipy.constants import Stefan_Boltzmann

    from aragog.surface_lid import LID_EXIST_WIDTH, LID_WEIGHT_FLOOR

    F_rad = grey_flux(T_b, emissivity, T_eq)
    inside = (F_b > 0.0) & (F_rad > 0.0) & (F_b < F_rad)
    F = jnp.where(inside, F_b, 0.5 * jnp.maximum(F_rad, 1.0))
    T_s = (F / (emissivity * Stefan_Boltzmann) + T_eq**4) ** 0.25
    d_eq = jnp.maximum(k * (T_b - T_s) / F, 1e-300)
    w = 0.5 * (1.0 + jnp.tanh(jnp.log10(d_eq / delta_on) / LID_EXIST_WIDTH))
    w = jnp.where(w > LID_WEIGHT_FLOOR, w, 0.0)
    return jnp.where(F_b <= 0.0, 1.0, jnp.where(inside, w, 0.0))


def lid_rates(
    *, delta, E_lid, surf, F_b, F_grid, w, rho, H_f, T_b, H_b, e, h_top, delta_lo, heating
):
    """Lid rates and the fluid flux, as :func:`aragog.surface_lid.lid_rates`; ``surf`` is the
    tuple of :func:`surface_state`.

    Returns
    -------
    tuple of jax.Array
        ``(ddelta, dE, F_top, F_fluid, w, stop, H_lid)`` as ``LidRates``.
    """
    from aragog.surface_lid import LID_TAU_HIDDEN

    T_s, F_s, q_b, s = surf[0], surf[1], surf[2], surf[3]
    r, stop = stefan_rate(q_b, F_b, rho, (H_f - H_b) / rho, delta, h_top, delta_lo)
    E_ph = delta_lo * profile_heat(e, 0.0, T_s, T_b)
    relax = (1.0 - w) / LID_TAU_HIDDEN
    ddelta = w * r - relax * (delta - delta_lo)
    H_lid = heating * delta * profile_density(e, s, T_s, T_b)
    dE = w * (q_b - F_s + H_b * r) - relax * (E_lid - E_ph) + H_lid
    F_top = w * F_s + (1.0 - w) * F_grid
    F_fluid = F_top + dE - H_lid - H_f * ddelta
    return ddelta, dE, F_top, F_fluid, w, stop, H_lid


class LidParams(NamedTuple):
    """Sub-grid lid data of one solver for the JAX right-hand side (SI units).

    ``e`` is the heat content per volume of the solid in temperature and ``hs`` the same in
    entropy, both at the top-cell pressure ``P``; ``rheo``, ``closure_lid`` and ``fixed`` are
    0/1 flags of the solid rheology, its 'lid' stress closure and its 'fixed' lid-base mode.
    """

    e: Enthalpy
    hs: Enthalpy
    P: float
    S_sol: float
    S_liq: float
    T_sol: float
    k: float
    emissivity: float
    T_eq: float
    delta_on: float
    delta_lo: float
    h_top: float
    area: float
    volume_top: float
    ra_crit: float
    rheo: float
    closure_lid: float
    fixed: float
    T_fixed: float
    contrast: float
    H_arr: float
    phi_rheo: float
    phi_width: float


def lid_base(lp: LidParams, eos, S_f, T_f, phi_f, T_lid_iso):
    """Lid-base entropy, temperature and density, as the NumPy ``_lid_base_state``: the solidus above the rheological transition, below it the
    rheological lid-base temperature on the EOS, capped at the solidus, blended in entropy."""
    from aragog.rheology import R_GAS
    from aragog.rheology_lid import _melt_ramp

    w_phi = _melt_ramp(phi_f, lp.phi_rheo, lp.phi_width, jnp)
    arrhenius = T_f - lp.contrast * R_GAS * T_f**2 / lp.H_arr
    T_rh = jnp.where(
        lp.closure_lid > 0.5, T_lid_iso, jnp.where(lp.fixed > 0.5, lp.T_fixed, arrhenius)
    )
    T_rh = jnp.maximum(T_rh, 1.0)
    capped = T_rh >= lp.T_sol
    S_rh = eos.entropy_at_temperature(lp.P, jnp.where(capped, lp.T_sol - 1.0, T_rh))
    S_rh = jnp.minimum(jnp.where(capped, lp.S_sol, S_rh), lp.S_sol)
    S_b = jnp.where(
        (lp.rheo > 0.5) & (w_phi < 1.0), w_phi * lp.S_sol + (1.0 - w_phi) * S_rh, lp.S_sol
    )
    return dict(
        S_b=S_b,
        T_b=eos.temperature(lp.P, S_b),
        rho=eos.density(lp.P, S_b),
    )


def lid_top(lp: LidParams, eos, y_lid, S_f, top: dict, F_in, F_grid, T_lid_iso, heating):
    """The lid at the top cell of one RHS evaluation, as the NumPy ``_lid_evaluate``.

    ``top`` holds the top-cell temperature, density, heat capacity, conductivity, expansivity,
    gravity, viscosity and melt fraction. ``F_in`` is the flux into the top cell from below and
    ``F_grid`` the surface flux of the mesh [W/m^2].

    Returns
    -------
    tuple
        ``(F_top, F_fluid, V_fluid, ddelta, dE, ledger)``: the surface loss and the fluid flux
        [W/m^2], the fluid volume of the top cell [m^3], the lid rates [m/s, W/m^2], and the
        ledger terms ``(H_f, m_lid)``: the fluid heat content per volume [J/m^3] and the lid
        mass per area [kg/m^2].
    """
    from aragog.surface_lid import LID_DELTA_FLOOR

    delta, E = y_lid[0], y_lid[1]
    base = lid_base(lp, eos, S_f, top['T'], top['phi'], T_lid_iso)
    H_f = enthalpy_H(lp.hs, S_f)
    q_conv = convective_base_flux(
        top['T'] - base['T_b'],
        top['k'],
        top['rho'],
        top['cp'],
        top['alpha'],
        jnp.abs(top['g']),
        top['eta'],
        lp.ra_crit,
    )
    F_b = jnp.maximum(jnp.maximum(q_conv, F_in), 0.0)
    w = existence_weight(F_b, base['T_b'], lp.k, lp.emissivity, lp.T_eq, lp.delta_on)
    surf = surface_state(
        delta, E, base['T_b'], lp.k, lp.emissivity, lp.T_eq, LID_DELTA_FLOOR, lp.e
    )
    ddelta, dE, F_top, F_fluid, *_ = lid_rates(
        delta=delta,
        E_lid=E,
        surf=surf,
        F_b=F_b,
        F_grid=F_grid,
        w=w,
        rho=base['rho'],
        H_f=H_f,
        T_b=base['T_b'],
        H_b=enthalpy_H(lp.hs, base['S_b']),
        e=lp.e,
        h_top=lp.h_top,
        delta_lo=lp.delta_lo,
        heating=heating,
    )
    m_lid = delta * profile_density(lp.e, surf[3], surf[0], base['T_b'])
    ledger = (H_f, m_lid)
    return F_top, F_fluid, lp.volume_top - lp.area * delta, ddelta, dE, ledger
