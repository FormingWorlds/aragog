"""Sub-grid conductive lid at the surface of the top cell.

A crust thinner than the top cell cannot be resolved by the mesh. The lid model carries it as
two states per unit area: the thickness ``delta`` [m] and the energy ``E_lid`` [J/m^2]. The
temperature profile across the lid is taken from the similarity family of the one-phase Stefan
problem (Neumann solution),

    T(z) = T_s + (T_b - T_s) f_s(z / delta),
    f_s(x) = erf(sqrt(s) x) / erf(sqrt(s))        (s > 0),
             x                                    (s = 0),
             erfi(sqrt(-s) x) / erfi(sqrt(-s))    (s < 0),

with the shape parameter ``s`` and the surface temperature ``T_s`` fixed by the lid energy,
``E_lid = delta int_0^1 H(T(x)) dx`` with ``H`` the heat content per volume of the lid material
(:class:`LidEnthalpy`), and by the grey-body balance at the surface. The family holds the
Neumann growth exactly (``s = lambda^2``) and reduces to the linear profile in steady state.
Material joins the lid at its base by the Stefan condition ``(W(S_f) - W(S_b)) d(delta)/dt =
q_b - F_b``, where ``q_b`` is the conduction into the lid at its base, ``F_b`` the flux the
fluid below delivers and ``W(S) = int rho T dS`` the heat content per volume at the top-cell
pressure, the measure the mesh and the lid store heat in, at the fluid entropy ``S_f`` and the
lid-base entropy ``S_b``.

The functions here are pure: the solver evaluates the material properties, passes them in and
assembles the rates. All quantities are SI (K, m, W/m^2, J/kg, J/m^2, kg/m^3, s).

References
----------
Turcotte, D. L. and Schubert, G. (2014), Geodynamics, 3rd ed., Cambridge University Press:
section 4.18, the Stefan problem (eqs. 4.136 and 4.141, pp. 197-198), and section 4.15, the
cooling of a half-space (eq. 4.116, p. 189).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.constants import Stefan_Boltzmann
from scipy.optimize import brentq
from scipy.special import dawsn, erf

# Growth stop: the lid stops growing over [CAP_FRAC - CAP_WIDTH, CAP_FRAC] of the top-cell width.
LID_CAP_FRAC: float = 0.8
LID_CAP_WIDTH: float = 0.02
# A lid weight below this value is set to 0 (the lid is hidden).
LID_WEIGHT_FLOOR: float = 1.0e-10
# Width [decades] of the existence ramp in log10(delta_eq / lid_delta_on).
LID_EXIST_WIDTH: float = 0.1
# Floor [J/kg] of the enthalpy jump in the Stefan rate.
LID_DH_MIN: float = 1.0e3
# Relaxation time [s] of a hidden lid to its phantom state.
LID_TAU_HIDDEN: float = 1.0e10
# Bounds of the mean ratio m = (T_m - T_s) / (T_b - T_s) the shape inverse accepts.
LID_MEAN_RATIO_MIN: float = 1.0e-4
LID_MEAN_RATIO_MAX: float = 0.99
# Smallest temperature contrast [K] across the lid.
LID_DT_MIN: float = 1.0e-6
# Conduction floor [m] of the lid thickness.
LID_DELTA_FLOOR: float = 1.0e-5

_S_SERIES: float = 1.0e-6


def shape(s: float) -> tuple[float, float, float]:
    """Slopes at the top and the base, and the mean, of the similarity profile f_s.

    Parameters
    ----------
    s : float
        Shape parameter [-]; ``s > 0`` for a growing lid, ``s < 0`` for a thinning lid.

    Returns
    -------
    tuple of float
        ``(f'(0), f'(1), m)``: the dimensionless surface and base slopes and the mean of f on
        [0, 1]. ``s = 0`` gives ``(1, 1, 1/2)``.
    """
    s = float(s)
    if abs(s) < _S_SERIES:
        return 1.0 + s / 3.0, 1.0 - 2.0 * s / 3.0, 0.5 + s / 12.0
    if s > 0.0:
        mu = np.sqrt(s)
        g = mu * np.sqrt(np.pi) * erf(mu)
        return 2.0 * s / g, 2.0 * s * np.exp(-s) / g, 1.0 + np.expm1(-s) / g
    nu = np.sqrt(-s)
    dw = dawsn(nu)
    return nu * np.exp(s) / dw, nu / dw, 1.0 + np.expm1(s) / (2.0 * nu * dw)


def _shape_of_mean(m: float) -> float:
    return brentq(lambda s: shape(s)[2] - m, -1.0e5, 1.0e5, xtol=1e-12, rtol=1e-14)


LID_SHAPE_MIN: float = _shape_of_mean(LID_MEAN_RATIO_MIN)
LID_SHAPE_MAX: float = _shape_of_mean(LID_MEAN_RATIO_MAX)


def profile(s: float, x: npt.ArrayLike) -> npt.NDArray:
    """The similarity profile f_s(x) on [0, 1] (series below ``|s| = 1e-6``)."""
    x = np.asarray(x, dtype=float)
    if abs(s) < _S_SERIES:
        return x + s * x * (1.0 - x * x) / 3.0
    if s > 0.0:
        mu = np.sqrt(s)
        return erf(mu * x) / erf(mu)
    nu = np.sqrt(-s)
    return np.exp(s * (1.0 - x * x)) * dawsn(nu * x) / dawsn(nu)


_GL16 = np.polynomial.legendre.leggauss(16)
PROFILE_PANELS: int = 16


def profile_nodes(s: float) -> tuple[npt.NDArray, npt.NDArray]:
    """Quadrature nodes and weights on [0, 1] for integrals over the profile f_s.

    Two segments split at ``c(s)``, which follows the steep part of the profile:
    ``min(1/2, 6 / sqrt(s))`` for ``s > 0`` (erf layer at the surface) and
    ``1 - min(1/2, 15 / |s|)`` for ``s < 0`` (exponential layer at the base); ``c`` is
    continuous in ``s``. Each segment holds ``PROFILE_PANELS`` 16-point Gauss-Legendre panels:
    the table enthalpy has kinks at the EOS nodes, so the panels, not the order, set the error
    (about 3e-6 relative at worst on the test tables).
    """
    g, w = _GL16
    if s > 0.0:
        c = min(0.5, 6.0 / np.sqrt(s))
    elif s < 0.0:
        c = 1.0 - min(0.5, -15.0 / s)
    else:
        c = 0.5
    edges = np.concatenate(
        [np.linspace(0.0, c, PROFILE_PANELS + 1), np.linspace(c, 1.0, PROFILE_PANELS + 1)[1:]]
    )
    half = 0.5 * np.diff(edges)[:, None]
    x = (edges[:-1, None] + half * (g + 1.0)).ravel()
    return x, (half * w).ravel()


@dataclass(frozen=True)
class LidEnthalpy:
    """Heat content per volume ``H(T)`` [J/m^3] and density ``rho(T)`` [kg/m^3] of the lid
    material at the top-cell pressure, both vectorised in ``T`` [K]."""

    H: Callable[[npt.ArrayLike], npt.NDArray]
    rho: Callable[[npt.ArrayLike], npt.NDArray]
    data: tuple = ()  # ('linear', rho, cp, T_b, h_b) or ('table', T, H, rho), for the JAX form

    @classmethod
    def linear(cls, rho: float, cp: float, T_b: float, h_b: float) -> LidEnthalpy:
        """Constant properties: ``H(T) = rho (h_b + cp (T - T_b))``."""
        return cls(
            H=lambda T: rho * (h_b + cp * (np.asarray(T, dtype=float) - T_b)),
            rho=lambda T: np.full(np.shape(T), rho),
            data=('linear', rho, cp, T_b, h_b),
        )

    @classmethod
    def table(cls, T: npt.ArrayLike, H: npt.ArrayLike, rho: npt.ArrayLike) -> LidEnthalpy:
        """From a table on increasing ``T``: monotone cubic ``H``, linear ``rho``, both continued
        linearly outside the table with the end slopes of ``H`` and constant ``rho``."""
        from scipy.interpolate import PchipInterpolator

        T, H, rho = (np.asarray(v, dtype=float) for v in (T, H, rho))
        spline = PchipInterpolator(T, H, extrapolate=False)
        slope = spline.derivative()
        lo, hi = (T[0], H[0], float(slope(T[0]))), (T[-1], H[-1], float(slope(T[-1])))

        def H_of(t):
            t = np.asarray(t, dtype=float)
            inside = np.clip(t, lo[0], hi[0])
            return np.where(
                t < lo[0],
                lo[1] + lo[2] * (t - lo[0]),
                np.where(t > hi[0], hi[1] + hi[2] * (t - hi[0]), spline(inside)),
            )

        return cls(H=H_of, rho=lambda t: np.interp(t, T, rho), data=('table', T, H, rho))

    def profile_heat(self, s: float, T_s: float, T_b: float) -> float:
        """Mean heat content per volume [J/m^3] of a lid with shape ``s`` from ``T_s`` to ``T_b``."""
        x, w = profile_nodes(s)
        return float(np.dot(w, self.H(T_s + (T_b - T_s) * profile(s, x))))

    def profile_density(self, s: float, T_s: float, T_b: float) -> float:
        """Mean density [kg/m^3] of a lid with shape ``s`` from ``T_s`` to ``T_b``."""
        x, w = profile_nodes(s)
        return float(np.dot(w, self.rho(T_s + (T_b - T_s) * profile(s, x))))


def grey_surface_temperature(
    f0: float, d: float, T_b: float, k: float, emissivity: float, T_eq: float
) -> float:
    """Surface temperature [K] where ``grey(T_s) = k (T_b - T_s) f0 / d``.

    Newton from ``T_b``: the residual is convex and increasing in ``T_s`` and non-negative at
    ``T_b``, so the iterates fall monotonically to the root. ``T_b`` when ``T_b <= T_eq``.
    """
    if T_b <= T_eq:
        return T_b
    c = k * f0 / d
    T = T_b
    for _ in range(100):
        g = grey_flux(T, emissivity, T_eq) - c * (T_b - T)
        step = g / (4.0 * emissivity * Stefan_Boltzmann * T**3 + c)
        T -= step
        if abs(step) <= 1e-13 * T:
            break
    return max(T, T_eq)


def grey_flux(T_s: float, emissivity: float, T_eq: float) -> float:
    """Grey-body flux [W/m^2] of a surface at ``T_s`` [K] against the equilibrium ``T_eq`` [K]."""
    return emissivity * Stefan_Boltzmann * (max(T_s, 0.0) ** 4 - T_eq**4)


@dataclass(frozen=True)
class LidSurface:
    """Surface solution of the lid at one state.

    Attributes
    ----------
    T_s : float
        Surface temperature [K].
    F_s : float
        Surface flux [W/m^2], positive out of the planet.
    q_b : float
        Conduction into the lid at its base [W/m^2], positive upward.
    s : float
        Shape parameter of the profile [-].
    clipped : int
        0 when the root lies inside the shape bounds; +1 when the lid energy is too high for
        any profile (``s`` at its upper bound), -1 when it is too low (``s`` at its lower bound).
    T_m : float
        Mean temperature of the profile [K].
    """

    T_s: float
    F_s: float
    q_b: float
    s: float
    clipped: int
    T_m: float


def _surface(s: float, T_s: float, T_b: float, k: float, d: float, clipped: int) -> LidSurface:
    f0, f1, m = shape(s)
    dT = T_b - T_s
    return LidSurface(
        T_s=T_s,
        F_s=k * dT * f0 / d,
        q_b=k * dT * f1 / d,
        s=s,
        clipped=clipped,
        T_m=T_s + m * dT,
    )


def _shape_of_energy(q: float, heat: Callable[[float], float]) -> tuple[float, int]:
    """Root in ``s`` of ``heat(s) = q`` (``heat`` increasing), clipped at the shape bounds."""
    if heat(LID_SHAPE_MIN) >= q:
        return LID_SHAPE_MIN, -1
    if heat(LID_SHAPE_MAX) <= q:
        return LID_SHAPE_MAX, 1
    s = brentq(lambda v: heat(v) - q, LID_SHAPE_MIN, LID_SHAPE_MAX, xtol=1e-13, rtol=1e-14)
    return s, 0


def surface_state(
    delta: float,
    E: float,
    T_b: float,
    k: float,
    emissivity: float,
    T_eq: float,
    delta_floor: float,
    hv: LidEnthalpy,
) -> LidSurface:
    """Solve the grey-body surface of a lid with energy ``E`` [J/m^2].

    For a shape ``s`` the grey-body balance ``grey(T_s) = k (T_b - T_s) f'(0) / d`` fixes the
    surface temperature, and with it the lid heat content ``Q(s) = int_0^1 H(T(x)) dx``. ``Q``
    rises with ``s``, so ``Q(s) = E / delta`` has one root in the shape bounds.

    Parameters
    ----------
    delta : float
        Lid thickness [m].
    E : float
        Lid energy [J/m^2].
    T_b : float
        Lid-base temperature [K].
    k : float
        Lid thermal conductivity [W/m/K].
    emissivity : float
        Surface emissivity [-].
    T_eq : float
        Equilibrium temperature of the radiation balance [K].
    delta_floor : float
        Conduction floor [m]: the flux uses ``hypot(delta, delta_floor)``; the heat content
        per volume uses ``max(delta, delta_floor)``.
    hv : LidEnthalpy
        Heat content and density of the lid material.

    Returns
    -------
    LidSurface
    """
    d = float(np.hypot(delta, delta_floor))
    q = E / max(delta, delta_floor)

    def T_s_of(v):
        return grey_surface_temperature(shape(v)[0], d, T_b, k, emissivity, T_eq)

    s, clipped = _shape_of_energy(q, lambda v: hv.profile_heat(v, T_s_of(v), T_b))
    return _surface(s, T_s_of(s), T_b, k, d, clipped)


def fixed_surface_state(
    delta: float,
    E: float,
    T_b: float,
    k: float,
    T_s: float,
    delta_floor: float,
    hv: LidEnthalpy,
) -> LidSurface:
    """Lid with a fixed surface temperature ``T_s`` [K]: the shape follows from
    ``Q(s) = E / delta`` at that surface temperature, held inside the shape bounds.

    Parameters and the result as :func:`surface_state`.
    """
    d = float(np.hypot(delta, delta_floor))
    q = E / max(delta, delta_floor)
    s, clipped = _shape_of_energy(q, lambda v: hv.profile_heat(v, T_s, T_b))
    return _surface(s, T_s, T_b, k, d, clipped)


def linear_lid_surface(
    delta: float, T_b: float, k: float, emissivity: float, T_eq: float
) -> float:
    """Surface temperature [K] of a linear (steady) lid: ``grey(T_s) = k (T_b - T_s) / delta``."""
    if T_b <= T_eq:
        return T_b
    return brentq(
        lambda T: grey_flux(T, emissivity, T_eq) - k * (T_b - T) / delta, T_eq, T_b, xtol=1e-12
    )


def convective_base_flux(
    dT: float,
    k: float,
    rho: float,
    cp: float,
    alpha: float,
    g: float,
    eta: float,
    ra_crit: float,
) -> float:
    """Boundary-layer flux [W/m^2] of a fluid ``dT`` [K] hotter than the lid base.

    ``q = k dT^(4/3) (alpha rho g / (Ra_crit kappa eta))^(1/3)`` with ``kappa = k / (rho cp)``:
    the flux of a boundary layer at the critical Rayleigh number. Zero for ``dT <= 0``.
    """
    if dT <= 0.0:
        return 0.0
    kappa = k / (rho * cp)
    return k * dT ** (4.0 / 3.0) * (alpha * rho * g / (ra_crit * kappa * eta)) ** (1.0 / 3.0)


def stefan_rate(
    q_b: float, F_b: float, rho: float, dh: float, delta: float, h_top: float, delta_lo: float
) -> tuple[float, float]:
    """Base growth rate [m/s] of the lid and the factor of the stop that acts.

    ``r = (q_b - F_b) / (rho max(dh, LID_DH_MIN))``. Growth stops linearly over
    ``[LID_CAP_FRAC - LID_CAP_WIDTH, LID_CAP_FRAC] h_top``; melt-back stops linearly over
    ``[delta_lo, 2 delta_lo]``.

    Returns
    -------
    tuple of float
        ``(r, factor)``: the rate after the stop and the stop factor in [0, 1].
    """
    r = (q_b - F_b) / (rho * max(dh, LID_DH_MIN))
    if r > 0.0:
        f = float(np.clip((LID_CAP_FRAC - delta / h_top) / LID_CAP_WIDTH, 0.0, 1.0))
    elif r < 0.0:
        f = float(np.clip((delta - delta_lo) / delta_lo, 0.0, 1.0))
    else:
        f = 1.0
    return r * f, f


def existence_weight(
    F_b: float, T_b: float, k: float, emissivity: float, T_eq: float, delta_on: float
) -> float:
    """Weight [0, 1] of the lid from the quasi-steady crust thickness.

    A crust in steady state carries ``F_b`` by conduction, ``delta_eq = k (T_b - T_s) / F_b``,
    with ``grey(T_s) = F_b``. The lid acts where ``delta_eq`` exceeds ``delta_on``:
    ``w = (1 + tanh(log10(delta_eq / delta_on) / LID_EXIST_WIDTH)) / 2``, set to 0 below
    ``LID_WEIGHT_FLOOR``. A fluid that delivers more than the grey flux at ``T_b`` carries no lid.
    """
    F_rad = grey_flux(T_b, emissivity, T_eq)
    if F_b <= 0.0:
        return 1.0
    if F_rad <= 0.0 or F_b >= F_rad:
        return 0.0
    T_s = (F_b / (emissivity * Stefan_Boltzmann) + T_eq**4) ** 0.25
    d_eq = k * (T_b - T_s) / F_b
    w = 0.5 * (1.0 + np.tanh(np.log10(d_eq / delta_on) / LID_EXIST_WIDTH))
    return float(w) if w > LID_WEIGHT_FLOOR else 0.0


@dataclass(frozen=True)
class LidRates:
    """Rates and fluxes of the lid at one state.

    Attributes
    ----------
    ddelta : float
        d(delta)/dt [m/s].
    dE : float
        dE_lid/dt [W/m^2], internal heating of the lid included.
    F_top : float
        Flux the planet loses at the surface [W/m^2].
    F_fluid : float
        Flux out of the fluid at the top of the top cell [W/m^2], from energy conservation.
    w : float
        Lid weight [0, 1].
    stop : float
        Factor of the growth or melt-back stop [0, 1].
    H_lid : float
        Internal heating of the lid [W/m^2].
    """

    ddelta: float
    dE: float
    F_top: float
    F_fluid: float
    w: float
    stop: float
    H_lid: float


def lid_rates(
    *,
    delta: float,
    E_lid: float,
    surf: LidSurface,
    F_b: float,
    F_grid: float,
    w: float,
    rho: float,
    H_f: float,
    T_b: float,
    H_b: float,
    hv: LidEnthalpy,
    h_top: float,
    delta_lo: float,
    heating: float,
) -> LidRates:
    """Assemble the lid rates and the fluid flux at one state.

    Fluid of heat content ``H_f`` [J/m^3] per volume joins the lid at ``T_b`` with heat content
    ``H_b`` [J/m^3] per volume, so the Stefan rate releases ``H_f - H_b`` per volume; ``rho`` is
    the density at the lid base, which scales the floor of that jump. The lid acts with weight
    ``w``; the hidden part ``1 - w`` relaxes to the phantom lid (thickness ``delta_lo``, linear
    profile) over ``LID_TAU_HIDDEN``, and the surface then sees the grid flux ``F_grid``. The
    fluid flux follows from the conservation of the energy of the fluid and the lid:
    ``F_fluid = F_top + dE - H_lid - H_f d(delta)/dt``. Exact for any ``w`` and any stop.

    Parameters
    ----------
    heating : float
        Internal heating per mass [W/kg] of the top cell, applied to the lid mass.
    """
    r, stop = stefan_rate(surf.q_b, F_b, rho, (H_f - H_b) / rho, delta, h_top, delta_lo)
    E_ph = delta_lo * hv.profile_heat(0.0, surf.T_s, T_b)
    relax = (1.0 - w) / LID_TAU_HIDDEN
    ddelta = w * r - relax * (delta - delta_lo)
    H_lid = heating * delta * hv.profile_density(surf.s, surf.T_s, T_b)
    dE = w * (surf.q_b - surf.F_s + H_b * r) - relax * (E_lid - E_ph) + H_lid
    F_top = w * surf.F_s + (1.0 - w) * F_grid
    F_fluid = F_top + dE - H_lid - H_f * ddelta
    return LidRates(
        ddelta=ddelta, dE=dE, F_top=F_top, F_fluid=F_fluid, w=w, stop=stop, H_lid=H_lid
    )
