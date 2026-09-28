"""Conductive skin at the surface: a grey surface over the top half cell.

The top staggered cell (temperature ``T_top``, conductivity ``k_top``) sits
``dr_half`` below the surface. Its heat reaches the surface by conduction and
leaves by grey-body radiation, so the surface temperature ``T_s`` solves

    emissivity * sigma * (T_s**4 - T_eq**4) = G * (T_top - T_s),  G = k_top / dr_half.

In melt the half-cell conduction is not the transport, so the flux blends from
the plain grey body (liquid top cell) to the skin flux (solid top cell) with a
weight that is exactly 0 or 1 outside the rheological transition. Every
function takes the array namespace ``xp`` (``numpy`` or ``jax.numpy``).
"""

from __future__ import annotations

import numpy as np

SKIN_NEWTON_ITERATIONS = 12
"""Newton steps for the skin temperature; the start is above the root."""

TABLE_EDGE_OFFSET = 100.0
"""Entropy above the table edge [J/kg/K] where the O2 cutoff is at half strength."""

TABLE_EDGE_WIDTH = 30.0
"""Entropy width [J/kg/K] of the O2 cutoff."""


def solid_weight(phi, phi_rheo: float, half_width: float = 0.1, xp=np):
    """Weight of the solid skin, 1 below and 0 above the rheological transition.

    Parameters
    ----------
    phi : float or array
        Melt fraction of the top staggered cell.
    phi_rheo : float
        Rheological transition melt fraction (centre of the ramp).
    half_width : float
        Half width of the ramp in melt fraction.
    xp : module
        Array namespace.

    Returns
    -------
    float or array
        ``1 - q(x)`` with ``q`` the quintic smoothstep of
        ``x = (phi - phi_rheo + half_width) / (2 half_width)`` clipped to [0, 1];
        exactly 1 for ``phi <= phi_rheo - half_width``, exactly 0 for
        ``phi >= phi_rheo + half_width``, twice continuously differentiable.
    """
    x = xp.clip((phi - phi_rheo + half_width) / (2.0 * half_width), 0.0, 1.0)
    return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def skin_temperature(T_top, G, emissivity, T_eq, sigma: float, xp=np):
    """Surface temperature of a grey surface fed by conduction across a layer.

    Parameters
    ----------
    T_top : float or array
        Temperature below the layer [K].
    G : float or array
        Conductance of the layer, ``k / thickness`` [W/m^2/K], positive.
    emissivity : float
        Grey-body emissivity.
    T_eq : float
        Radiative equilibrium temperature [K].
    sigma : float
        Stefan-Boltzmann constant [W/m^2/K^4].
    xp : module
        Array namespace.

    Returns
    -------
    float or array
        The root ``T_s`` of ``f(T) = emissivity sigma (T^4 - T_eq^4) - G (T_top - T)``.

    Notes
    -----
    ``f`` is convex and increasing for ``T > 0``, so Newton converges
    monotonically from any start where ``f >= 0``. For ``T_top >= T_eq`` the
    start is ``min(T_top, T_u)`` with
    ``T_u = (T_eq^4 + G (T_top - T_eq) / (emissivity sigma))^(1/4)``, an upper
    bound because ``T_s >= T_eq``; ``T_u`` is close to the root when radiation
    limits the flux. For ``T_top < T_eq`` the start is ``T_eq``.
    """
    es = emissivity * sigma
    T_u = (T_eq**4 + G * xp.maximum(T_top - T_eq, 0.0) / es) ** 0.25
    T = xp.where(T_top >= T_eq, xp.minimum(T_top, T_u), T_eq + 0.0 * T_top)
    for _ in range(SKIN_NEWTON_ITERATIONS):
        f = es * (T**4 - T_eq**4) - G * (T_top - T)
        T = T - f / (4.0 * es * T**3 + G)
    return T


def table_edge_factor(S, S_edge: float, xp=np):
    """Factor on the outgoing surface flux that holds the top cell on the table.

    Parameters
    ----------
    S : float or array
        Entropy of the top staggered cell [J/kg/K].
    S_edge : float
        Lower entropy edge of the solid table [J/kg/K].
    xp : module
        Array namespace.

    Returns
    -------
    float or array
        ``0.5 (1 + tanh((S - S_edge - TABLE_EDGE_OFFSET) / TABLE_EDGE_WIDTH))``.
    """
    return 0.5 * (1.0 + xp.tanh((S - S_edge - TABLE_EDGE_OFFSET) / TABLE_EDGE_WIDTH))
