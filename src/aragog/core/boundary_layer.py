"""Core-mantle boundary heat flux from a mantle-side thermal boundary layer.

For a core hotter than the mantle above it the flux is the boundary-layer law of
Thiriet et al. (2019)[^cite-thiriet2019], eqs. 13-14 with ``beta = 1/3``,
``q = k dT / delta`` with

    delta_BL = (Ra_crit kappa eta / (alpha rho g dT))^(1/3),  dT = T_core - T_m,

and the layer properties at the mean of the two temperatures. The critical Rayleigh
number is their eq. 16 (from Deschamps and Sotin 2000[^cite-deschamps2000]),
``Ra_crit = 0.28 Ra_i^0.21``, with the internal Rayleigh number of their eq. 3,
``Ra_i = alpha rho g dT_shell D^3 / (kappa eta)``, unless a fixed value is given
(450 is the value Thiriet et al. take for the upper layer). The flux does not depend
on the mesh. Two choices are this module's own, not the papers': a core colder than
the mantle above it (a stably stratified layer) receives heat by conduction across
the half cell between the CMB and the bottom staggered node, and so does a layer
that is not buoyant (``alpha <= 0``), where no boundary layer forms. The flux has
the sign of ``dT`` and is continuous at zero.
"""

from __future__ import annotations

import math
import numbers

import jax.numpy as jnp

# None selects Thiriet et al. (2019) eq. 16; a number fixes the critical Rayleigh number.
RA_CRIT_CMB_DEFAULT = None
RA_CRIT_UPPER_LAYER = 450.0  # Thiriet et al. (2019), Table 2, the upper boundary layer


def check_ra_crit(ra_crit) -> float | None:
    """Return the CMB critical Rayleigh number: None (eq. 16) or a float.

    Raises
    ------
    ValueError
        When ``ra_crit`` is not None and not a real number (a boolean or a string
        included), not positive or not finite.
    """
    if ra_crit is None:
        return None
    real = isinstance(ra_crit, numbers.Real) and not isinstance(ra_crit, bool)
    if not (real and 0.0 < float(ra_crit) < math.inf):
        raise ValueError(f'ra_crit_cmb must be positive and finite, got {ra_crit!r}')
    return float(ra_crit)


def critical_rayleigh_eq16(
    *,
    conductivity,
    density,
    heat_capacity,
    expansivity,
    viscosity,
    gravity,
    shell_thickness,
    shell_contrast,
):
    """Thiriet et al. (2019) eq. 16, ``0.28 Ra_i^0.21``, with ``Ra_i`` of their eq. 3.

    ``shell_thickness`` [m] is the mantle thickness and ``shell_contrast`` [K] the
    temperature difference across the shell; the other arguments are the layer properties
    as in ``cmb_boundary_layer_flux``. A non-positive ``Ra_i`` is held at a tiny positive
    value so the power stays finite (the caller conducts there).
    """
    kappa = conductivity / (density * heat_capacity)
    ra_i = (
        expansivity
        * density
        * gravity
        * shell_contrast
        * shell_thickness**3
        / (kappa * viscosity)
    )
    return 0.28 * jnp.maximum(ra_i, 1e-300) ** 0.21


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
    shell_thickness=None,
    shell_contrast=None,
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
        [Pa s]. The thermal diffusivity is ``k / (rho c_p)``; ``c_p`` and the
        expansivity are the material values, without a phase-change contribution.
    gravity : float
        Gravitational acceleration at the CMB [m/s^2].
    dr_half : float
        Distance from the CMB to the bottom staggered node [m], positive.
    ra_crit : float or None
        Critical Rayleigh number of the boundary layer; None (or NaN inside a JAX trace)
        selects Thiriet et al. (2019) eq. 16.
    shell_thickness, shell_contrast : float, optional
        Mantle thickness [m] and temperature difference across the shell [K] for the
        internal Rayleigh number of eq. 16; required when ``ra_crit`` is None.

    Returns
    -------
    jax.Array
        ``q_conv = k dT / delta_BL`` for ``T_core > T_mantle`` over a buoyant layer, else
        ``q_cond = k dT / dr_half``. Positive cools the core.

    Raises
    ------
    ValueError
        When ``ra_crit`` is None and the shell thickness or contrast is missing.
    """
    if ra_crit is None:
        if shell_thickness is None or shell_contrast is None:
            raise ValueError(
                'ra_crit=None (Thiriet eq. 16) needs shell_thickness and shell_contrast'
            )
        ra_crit = jnp.nan
    dT = t_core - t_mantle
    q_cond = conductivity * dT / dr_half
    # Safe base keeps the cube root and its derivative finite where the branch is unused.
    dT_pos = jnp.where(dT > 0.0, dT, 1.0)
    kappa = conductivity / (density * heat_capacity)
    buoyant = expansivity * density * gravity > 0.0
    buoyancy = jnp.where(buoyant, expansivity * density * gravity, 1.0)
    if shell_thickness is not None and shell_contrast is not None:
        eq16 = critical_rayleigh_eq16(
            conductivity=conductivity,
            density=density,
            heat_capacity=heat_capacity,
            expansivity=jnp.where(buoyant, expansivity, 1.0),
            viscosity=viscosity,
            gravity=gravity,
            shell_thickness=shell_thickness,
            # The shell spans the CMB layer: its contrast is at least the CMB contrast.
            shell_contrast=jnp.maximum(shell_contrast, dT_pos),
        )
        ra_crit = jnp.where(jnp.isnan(ra_crit), eq16, ra_crit)
    q_conv = (
        conductivity
        * dT_pos ** (4.0 / 3.0)
        * (buoyancy / (ra_crit * kappa * viscosity)) ** (1.0 / 3.0)
    )
    return jnp.where((dT > 0.0) & buoyant, q_conv, q_cond)
