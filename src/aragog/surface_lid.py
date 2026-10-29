"""Sub-grid conductive lid at the surface of the top cell.

A crust thinner than the top cell cannot be resolved by the mesh. The lid model carries it as
two states per unit area: the thickness ``delta`` [m] and the energy ``E_lid`` [J/m^2]. The
temperature profile across the lid is taken from the similarity family of the one-phase Stefan
problem (Neumann solution),

    T(z) = T_s + (T_b - T_s) f_s(z / delta),
    f_s(x) = erf(sqrt(s) x) / erf(sqrt(s))        (s > 0),
             x                                    (s = 0),
             erfi(sqrt(-s) x) / erfi(sqrt(-s))    (s < 0),

with the shape parameter ``s`` fixed by the mean lid temperature. The family holds the Neumann
growth exactly (``s = lambda^2``) and reduces to the linear profile in steady state. Material
joins the lid at its base by the Stefan condition ``rho dh d(delta)/dt = q_b - F_b``, where
``q_b`` is the conduction into the lid at its base, ``F_b`` the flux the fluid below delivers
and ``dh`` the enthalpy released per kilogram.

The functions here are pure: the solver evaluates the material properties, passes them in and
assembles the rates. All quantities are SI (K, m, W/m^2, J/kg, J/m^2, kg/m^3, s).

References
----------
Carslaw, H. S. and Jaeger, J. C. (1959), Conduction of Heat in Solids, 2nd ed., Oxford.
Turcotte, D. L. and Schubert, G. (2014), Geodynamics, 3rd ed., Cambridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
        0 when the root lies inside the shape bounds; +1 when the lid mean is too hot for any
        profile (the steepest profile, coldest surface, is used), -1 when it is too cold (the
        flattest profile, hottest surface).
    """

    T_s: float
    F_s: float
    q_b: float
    s: float
    clipped: int


def surface_state(
    delta: float,
    T_m: float,
    T_b: float,
    k: float,
    emissivity: float,
    T_eq: float,
    delta_floor: float,
) -> LidSurface:
    """Solve the grey-body surface of a lid with mean temperature ``T_m``.

    The mean fixes the surface temperature for a given shape, ``T_s = (T_m - m T_b) / (1 - m)``,
    so the balance ``grey(T_s) = k (T_b - T_s) f'(0) / delta`` is one equation in ``s``. Its
    left side falls and its right side rises with ``s``, so the root is unique in the bounds.

    Parameters
    ----------
    delta : float
        Lid thickness [m].
    T_m : float
        Mean lid temperature [K]; held inside ``(T_eq, T_b)``.
    T_b : float
        Lid-base temperature [K].
    k : float
        Lid thermal conductivity [W/m/K].
    emissivity : float
        Surface emissivity [-].
    T_eq : float
        Equilibrium temperature of the radiation balance [K].
    delta_floor : float
        Conduction floor [m]: the flux uses ``hypot(delta, delta_floor)``.

    Returns
    -------
    LidSurface
    """
    d = float(np.hypot(delta, delta_floor))
    T_m = float(np.clip(T_m, T_eq + LID_DT_MIN, T_b - LID_DT_MIN))
    D = T_b - T_m

    def ts_of(s):
        f0, f1, m = shape(s)
        return (T_m - m * T_b) / (1.0 - m), f0, f1, m

    def balance(s):
        T_s, f0, _, m = ts_of(s)
        return grey_flux(T_s, emissivity, T_eq) - k * D * f0 / ((1.0 - m) * d)

    clipped = 0
    if balance(LID_SHAPE_MIN) <= 0.0:
        s, clipped = LID_SHAPE_MIN, -1
    elif balance(LID_SHAPE_MAX) >= 0.0:
        s, clipped = LID_SHAPE_MAX, 1
    else:
        s = brentq(balance, LID_SHAPE_MIN, LID_SHAPE_MAX, xtol=1e-13, rtol=1e-14)
    T_s, f0, f1, m = ts_of(s)
    dT = D / (1.0 - m)
    return LidSurface(T_s=T_s, F_s=k * dT * f0 / d, q_b=k * dT * f1 / d, s=s, clipped=clipped)


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


def lid_energy(
    rho: float, delta: float, h_b: float, cp: float, T_b: float, T_m: float
) -> float:
    """Lid energy [J/m^2] of a lid with mean temperature ``T_m``: ``rho delta (h_b - cp (T_b - T_m))``."""
    return rho * delta * (h_b - cp * (T_b - T_m))


def lid_mean_temperature(
    E_lid: float, rho: float, delta: float, h_b: float, cp: float, T_b: float
) -> float:
    """Mean lid temperature [K], the inverse of :func:`lid_energy`."""
    return T_b - (h_b - E_lid / (rho * delta)) / cp


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
    """

    ddelta: float
    dE: float
    F_top: float
    F_fluid: float
    w: float
    stop: float


def lid_rates(
    *,
    delta: float,
    E_lid: float,
    surf: LidSurface,
    F_b: float,
    F_grid: float,
    w: float,
    rho: float,
    cp: float,
    h_b: float,
    h_f: float,
    T_b: float,
    h_top: float,
    delta_lo: float,
    heating: float,
) -> LidRates:
    """Assemble the lid rates and the fluid flux at one state.

    The lid acts with weight ``w``; the hidden part ``1 - w`` relaxes to the phantom lid
    (thickness ``delta_lo``, linear profile) over ``LID_TAU_HIDDEN``, and the surface then sees
    the grid flux ``F_grid``. The fluid flux follows from the conservation of the energy of the
    fluid and the lid: ``F_fluid = F_top + dE - H_lid - rho h_f d(delta)/dt``, with
    ``H_lid = heating rho delta``. Exact for any ``w`` and any stop.

    Parameters
    ----------
    heating : float
        Internal heating per mass [W/kg] of the top cell, applied to the lid mass.
    """
    r, stop = stefan_rate(surf.q_b, F_b, rho, h_f - h_b, delta, h_top, delta_lo)
    T_ph = 0.5 * (T_b + surf.T_s)
    E_ph = lid_energy(rho, delta_lo, h_b, cp, T_b, T_ph)
    relax = (1.0 - w) / LID_TAU_HIDDEN
    ddelta = w * r - relax * (delta - delta_lo)
    H_lid = heating * rho * delta
    dE = w * (surf.q_b - surf.F_s + rho * h_b * r) - relax * (E_lid - E_ph) + H_lid
    F_top = w * surf.F_s + (1.0 - w) * F_grid
    F_fluid = F_top + dE - H_lid - rho * h_f * ddelta
    return LidRates(ddelta=ddelta, dE=dE, F_top=F_top, F_fluid=F_fluid, w=w, stop=stop)
