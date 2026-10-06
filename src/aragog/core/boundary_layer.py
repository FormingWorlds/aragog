"""Core-mantle boundary heat flux from a mantle-side thermal boundary layer.

The flux follows the parameterised-convection closure of Foley & Driscoll
(2016, G3 17, 1885, eq. 20), ``q = k dT / delta``, with the layer thickness
set by a critical boundary-layer Rayleigh number (Thiriet et al. 2019, PEPI
286, 138, eqs. 13-14 with ``beta = 1/3``):

    delta_BL = (Ra_crit kappa eta / (alpha rho g dT))^(1/3),  dT = T_core - T_m.

``RA_CRIT_CMB_DEFAULT`` is the theoretical value Thiriet et al. take for the
upper boundary layer (their Table 2); their lower-layer law (Deschamps & Sotin
2000) is not adopted. Two choices are this module's own, not the papers': the
flux never falls below conduction across the half cell between the CMB and
the bottom staggered node, and a core colder than the mantle above it (a
stably stratified layer) receives heat by that conduction only. The flux
therefore has the sign of ``dT`` and is continuous at zero.
"""

from __future__ import annotations

import jax.numpy as jnp

RA_CRIT_CMB_DEFAULT = 450.0


def cmb_boundary_layer_flux(
    t_core,
    t_mantle,
    *,
    conductivity,
    density,
    heat_capacity,
    expansivity,
    viscosity,
    gravity,
    dr_half,
    ra_crit=RA_CRIT_CMB_DEFAULT,
):
    """Heat flux out of the core across the CMB [W/m^2].

    Parameters
    ----------
    t_core : float or jax.Array
        Core temperature at the CMB [K].
    t_mantle : float or jax.Array
        Mantle temperature at the CMB [K]: the bottom cell's entropy
        evaluated at the CMB pressure in the solver.
    conductivity, density, heat_capacity, expansivity, viscosity : float
        Bottom-cell thermal conductivity [W/m/K], density [kg/m^3], heat
        capacity [J/kg/K], thermal expansivity [1/K] and dynamic viscosity
        [Pa s]. The thermal diffusivity is ``k / (rho c_p)``, so ``c_p`` is
        the material heat capacity without a latent contribution.
    gravity : float
        Gravitational acceleration at the CMB [m/s^2].
    dr_half : float
        Distance from the CMB to the bottom staggered node [m], positive.
    ra_crit : float
        Critical Rayleigh number of the boundary layer.

    Returns
    -------
    jax.Array
        ``max(q_cond, q_conv)`` for ``T_core > T_mantle``, else ``q_cond``,
        with ``q_cond = k dT / dr_half``. Positive cools the core.
    """
    dT = t_core - t_mantle
    q_cond = conductivity * dT / dr_half
    # Safe base keeps the cube root and its derivative finite where the branch is unused.
    dT_pos = jnp.where(dT > 0.0, dT, 1.0)
    kappa = conductivity / (density * heat_capacity)
    # Floor (not zero) so a non-buoyant layer gives q_conv ~ 0 with a finite derivative.
    buoyancy = jnp.maximum(expansivity * density * gravity, 1e-300)
    q_conv = (
        conductivity
        * dT_pos ** (4.0 / 3.0)
        * (buoyancy / (ra_crit * kappa * viscosity)) ** (1.0 / 3.0)
    )
    return jnp.where(dT > 0.0, jnp.maximum(q_cond, q_conv), q_cond)


def cmb_node_gradient(
    t_core, s_bottom, temperature_at_cmb, dr_offset, width=2.0e4, n_bisect=52, ds=1e-2
):
    """Entropy gradient at the CMB basic node that puts the node at ``t_core``.

    ``temperature_at_cmb(S)`` rises with S, so bisection on ``[S0 - width, S0 + width]``
    brackets the root of ``T(S) = t_core`` wherever the slope is small or the table
    is flat, which a Newton step from S0 does not. One Newton step from the bracketed
    root, clipped to ``ds``, carries the derivative ``dS/dT_core = 1/T'(S)`` that the
    JAX Jacobian needs. The gradient is taken over the basic-to-staggered offset
    ``dr_offset`` (negative: the node lies below the cell). A ``t_core`` outside the
    bracket's temperature range returns its nearest end.
    """
    lo, hi = s_bottom - width, s_bottom + width
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        above = temperature_at_cmb(mid) > t_core
        lo, hi = jnp.where(above, lo, mid), jnp.where(above, mid, hi)
    s = 0.5 * (lo + hi)
    slope = (temperature_at_cmb(s + ds) - temperature_at_cmb(s - ds)) / (2.0 * ds)
    step = (temperature_at_cmb(s) - t_core) / jnp.maximum(slope, 1e-6)
    s = s - jnp.clip(step, -ds, ds)
    return (s - s_bottom) / dr_offset
