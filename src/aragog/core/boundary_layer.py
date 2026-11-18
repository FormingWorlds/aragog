"""Core-mantle boundary heat flux from a mantle-side thermal boundary layer.

For a core hotter than the mantle above it the flux is the boundary-layer law of
Thiriet et al. (2019)[^cite-thiriet2019], eqs. 13-14 with ``beta = 1/3``,
``q = k dT / delta`` with

    delta_BL = (Ra_crit kappa eta / (alpha rho g dT))^(1/3),  dT = T_core - T_m,

and the layer properties at the mean of the two temperatures. The critical Rayleigh
number is their eq. 16 (from Deschamps and Sotin 2000[^cite-deschamps2000]),
``Ra_crit = 0.28 Ra_i^0.21``, with the internal Rayleigh number of their eq. 3,
``Ra_i = alpha rho g dT_shell D^3 / (kappa eta_i)`` with the viscosity ``eta_i`` of the
interior at ``T_m``, unless a fixed value is given
(450 is the value Thiriet et al. take for the upper layer). The selector ``'phase'``
takes 450 while the bottom cell is liquid and eq. 16 while it is solid, the switch of
Miyazaki and Korenaga (2019)[^cite-miyazaki2019b] at their critical melt fraction, blended
across the rheological transition (``liquid_weight``) in the logarithm of ``Ra_crit``, so the
flux is continuous in the melt fraction. The flux does not depend
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

# None selects Thiriet et al. (2019) eq. 16; a number fixes the critical Rayleigh number;
# 'phase' blends 450 (liquid bottom cell) and eq. 16 (solid) across the rheological transition.
RA_CRIT_CMB_DEFAULT = None
RA_CRIT_UPPER_LAYER = 450.0  # Thiriet et al. (2019), Table 2, the upper boundary layer
RA_CRIT_PHASE = 'phase'


def check_ra_crit(ra_crit) -> float | str | None:
    """Return the CMB critical Rayleigh number: None (eq. 16), ``'phase'`` or a float.

    Raises
    ------
    ValueError
        When ``ra_crit`` is not None, not ``'phase'`` and not a real number (a boolean or
        another string included), not positive or not finite.
    """
    if ra_crit is None or ra_crit == RA_CRIT_PHASE:
        return ra_crit
    real = isinstance(ra_crit, numbers.Real) and not isinstance(ra_crit, bool)
    if not (real and 0.0 < float(ra_crit) < math.inf):
        raise ValueError(
            f"ra_crit_cmb must be positive and finite (or 'phase'), got {ra_crit!r}"
        )
    return float(ra_crit)


def liquid_weight(melt_fraction, loc, width):
    """Weight of the liquid law (``Ra_crit = 450``) in the CMB law by phase.

    A C1 smoothstep ``3u^2 - 2u^3`` of ``u = (phi - (loc - width/2)) / width`` clipped to
    [0, 1]: 0 at and below ``loc - width/2``, 1 at and above ``loc + width/2``, 1/2 at the
    rheological transition ``loc``.
    """
    u = jnp.clip((melt_fraction - (loc - 0.5 * width)) / width, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def check_liquid_band(loc, width) -> None:
    """Refuse a transition band of the phase law that leaves [0, 1] or has no width.

    Raises
    ------
    ValueError
        When ``loc`` or ``width`` is not finite, ``width <= 0``, or the band
        ``[loc - width/2, loc + width/2]`` leaves [0, 1].
    """
    lo, hi = float(loc) - 0.5 * float(width), float(loc) + 0.5 * float(width)
    if not (math.isfinite(lo) and math.isfinite(hi) and float(width) > 0.0):
        raise ValueError(f'phase law band needs a finite centre and width > 0, got {loc}, {width}')
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"ra_crit_cmb = 'phase' needs the rheological transition band [{lo:g}, {hi:g}] "
            '(centre -+ width/2) inside [0, 1]'
        )


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

    ``shell_thickness`` [m] is the mantle thickness, ``shell_contrast`` [K] the
    temperature difference across the shell and ``viscosity`` [Pa s] that of the interior
    at ``T_m``; the other arguments are the layer properties as in
    ``cmb_boundary_layer_flux``. A non-positive ``Ra_i`` is held at a tiny positive
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
    interior_viscosity=None,
    melt_fraction=None,
    liquid_band=None,
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
    interior_viscosity : float, optional
        Viscosity of the interior at ``t_mantle`` [Pa s], the viscosity of the internal
        Rayleigh number of eq. 16; required when ``ra_crit`` is None. The layer law
        (eq. 14) keeps ``viscosity``.
    melt_fraction : float, optional
        Melt fraction of the bottom cell's own state at the CMB pressure; required with
        ``liquid_band``.
    liquid_band : tuple of float, optional
        ``(loc, width)`` of the rheological transition for the law by phase:
        ``ln Ra_crit = s ln 450 + (1 - s) ln Ra_crit,16`` with
        ``s = liquid_weight(melt_fraction, loc, width)``, exactly 450 at ``s = 1`` and exactly
        eq. 16 at ``s = 0``. It needs the eq. 16 inputs and replaces ``ra_crit``; a NaN ``loc``
        (the 'off' value inside a JAX trace) keeps ``ra_crit``.

    Returns
    -------
    jax.Array
        ``q_conv = k dT / delta_BL`` for ``T_core > T_mantle`` over a buoyant layer, else
        ``q_cond = k dT / dr_half``. Positive cools the core.

    Raises
    ------
    ValueError
        When ``ra_crit`` is None and the shell thickness or contrast or the interior
        viscosity is missing, or when ``liquid_band`` is given without ``melt_fraction``
        or without those inputs.
    """
    eq16_inputs = (shell_thickness, shell_contrast, interior_viscosity)
    if liquid_band is not None and (
        melt_fraction is None or any(x is None for x in eq16_inputs)
    ):
        raise ValueError(
            'liquid_band (the CMB law by phase) needs melt_fraction, shell_thickness, '
            'shell_contrast and interior_viscosity'
        )
    if ra_crit is None:
        if any(x is None for x in eq16_inputs):
            raise ValueError(
                'ra_crit=None (Thiriet eq. 16) needs shell_thickness, shell_contrast and '
                'interior_viscosity'
            )
        ra_crit = jnp.nan
    dT = t_core - t_mantle
    q_cond = conductivity * dT / dr_half
    # Safe base keeps the cube root and its derivative finite where the branch is unused.
    dT_pos = jnp.where(dT > 0.0, dT, 1.0)
    kappa = conductivity / (density * heat_capacity)
    buoyant = expansivity * density * gravity > 0.0
    buoyancy = jnp.where(buoyant, expansivity * density * gravity, 1.0)
    if all(x is not None for x in eq16_inputs):
        eq16 = critical_rayleigh_eq16(
            conductivity=conductivity,
            density=density,
            heat_capacity=heat_capacity,
            expansivity=jnp.where(buoyant, expansivity, 1.0),
            viscosity=interior_viscosity,
            gravity=gravity,
            shell_thickness=shell_thickness,
            # The shell spans the CMB layer: its contrast is at least the CMB contrast.
            shell_contrast=jnp.maximum(shell_contrast, dT_pos),
        )
        ra_crit = jnp.where(jnp.isnan(ra_crit), eq16, ra_crit)
        if liquid_band is not None:
            loc, width = liquid_band
            off = jnp.isnan(loc)
            # Safe band where it is off, so the unused blend keeps finite derivatives.
            w = liquid_weight(
                melt_fraction, jnp.where(off, 0.5, loc), jnp.where(off, 1.0, width)
            )
            blend = jnp.exp(w * math.log(RA_CRIT_UPPER_LAYER) + (1.0 - w) * jnp.log(eq16))
            blend = jnp.where(w >= 1.0, RA_CRIT_UPPER_LAYER, jnp.where(w <= 0.0, eq16, blend))
            ra_crit = jnp.where(off, ra_crit, blend)
    q_conv = (
        conductivity
        * dT_pos ** (4.0 / 3.0)
        * (buoyancy / (ra_crit * kappa * viscosity)) ** (1.0 / 3.0)
    )
    return jnp.where((dT > 0.0) & buoyant, q_conv, q_cond)
