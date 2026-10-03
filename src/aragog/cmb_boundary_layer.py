"""Lower thermal boundary layer law for the CMB heat flux.

Deschamps and Sotin (2000, GJI 143, 204-218, doi:10.1046/j.1365-246x.2000.00228.x): the
Rayleigh number of the lower thermal boundary layer, ``Ra_delta = rho g alpha dT_c delta^3 /
(kappa eta)``, scales as ``0.28 Ra^0.21``, with the mantle Rayleigh number
``Ra = rho g alpha (T_c - T_s) D^3 / (kappa eta)``. Both use the viscosity of the
well-mixed interior. Convective onset follows Chandrasekhar (1961, Hydrodynamic
and Hydromagnetic Stability, Oxford: Clarendon Press) with free-free boundary conditions
giving critical Rayleigh number ``Ra_c = 27 pi^4 / 4``. For dT_c <= 0 (core at or below the
mantle base) the bottom layer is stable: Nu = 1, q = k dT_c / D (conductive for a stable
layer, into the core). For dT_c > 0 the boundary-layer law is extrapolated with Nu >= 1
rather than switching to pure conduction across depth D; at onset
Nu(Ra_c) = (Ra_l / Ra_dc(Ra_c))^(1/3) (bounded by approximately 8.44 when Ra_l = Ra_c),
and the absolute heat flux remains small because it scales with the small temperature
contrast dT_c = T_c - T_m.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RA_CRIT_PREFACTOR = 0.28
RA_CRIT_EXPONENT = 0.21
RA_C = 27.0 * np.pi**4 / 4.0
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
        temperature [K].
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
        Heat flux out of the core [W/m^2].
    """
    buoyancy = rho * g * alpha / (kappa * eta)
    ra = buoyancy * (T_c - T_s) * depth**3
    ra_eff = xp.maximum(ra, RA_C)
    ra_dc = RA_CRIT_PREFACTOR * ra_eff**RA_CRIT_EXPONENT
    dT_c = T_c - T_m
    pos_dT_c = xp.maximum(dT_c, 0.0)
    ra_l = buoyancy * pos_dT_c * depth**3
    safe_ra_l = xp.maximum(ra_l, ra_dc)
    nu_conv = (safe_ra_l / ra_dc) ** (1.0 / 3.0)
    nu = xp.where(dT_c > 0.0, nu_conv, 1.0)
    q = (k * dT_c / depth) * nu
    if xp is np and np.ndim(T_c) == 0 and np.ndim(T_s) == 0 and np.ndim(T_m) == 0:
        return float(q)
    return q
