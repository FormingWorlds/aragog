"""Lower thermal boundary layer law for the CMB heat flux.

Deschamps and Sotin (2000, GJI 143, 204-218), eqs. 32 and 33: the Rayleigh number
of the lower thermal boundary layer, ``Ra_delta = rho g alpha dT_c delta^3 /
(kappa eta)``, equals ``0.28 Ra^0.21``, with the core Rayleigh number
``Ra = rho g alpha (T_c - T_s) D^3 / (kappa eta)``. Both use the viscosity of the
well-mixed interior. With ``delta = k dT_c / q`` this gives the CMB flux ``q`` from
the CMB temperature ``T_c`` and the interior temperature ``T_m`` above the layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RA_CRIT_PREFACTOR = 0.28
RA_CRIT_EXPONENT = 0.21
CMB_FLUX_LAWS = ('none', 'deschamps_sotin_2000')


def cmb_flux(
    T_c: Any,
    T_m: Any,
    T_s: Any,
    depth: Any,
    rho: Any,
    g: Any,
    alpha: Any,
    kappa: Any,
    k: Any,
    eta: Any,
    xp: Any = np,
) -> Any:
    """Heat flux out of the core through the lower thermal boundary layer [W/m^2].

    Parameters
    ----------
    T_c, T_m, T_s : float or array
        CMB temperature, interior temperature above the layer, and surface
        temperature [K]; ``T_c > T_s``.
    depth : float
        Mantle depth [m].
    rho, g, alpha, kappa, k : float or array
        Density [kg/m^3], gravity [m/s^2], thermal expansivity [1/K], thermal
        diffusivity [m^2/s] and conductivity [W/m/K] of the layer.
    eta : float or array
        Viscosity of the well-mixed interior [Pa s].
    xp : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    float or array
        ``q = k dT_c / delta`` with ``delta = (Ra_crit kappa eta / (rho g alpha
        |dT_c|))^(1/3)``, ``dT_c = T_c - T_m``; positive out of the core, odd in
        ``dT_c`` and differentiable at ``dT_c = 0``.
    """
    buoyancy = rho * g * alpha / (kappa * eta)
    ra_crit = RA_CRIT_PREFACTOR * (buoyancy * (T_c - T_s) * depth**3) ** RA_CRIT_EXPONENT
    dT = T_c - T_m
    return k * xp.sign(dT) * xp.abs(dT) ** (4.0 / 3.0) * (buoyancy / ra_crit) ** (1.0 / 3.0)
