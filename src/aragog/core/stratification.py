"""Thermal stratification at the top of the core.

When the CMB heat flow drops below what conduction carries along the
adiabat, the top of the core stops convecting and a stably stratified
layer grows. Two ODE-cost quantities describe the state. The adiabatic
ratio ``ADR = Q_cmb / Q_k`` (the actual-to-adiabatic CMB gradient ratio,
as defined in the Leeds ``thermal_history`` stable-layer model) is the
onset criterion: below one, stratification grows. The equilibrium stratification depth is the layer thickness
``r_cmb - r_s``, where ``r_s`` is the radius where the adiabatic
conducted flow ``Q_ad(r) = 4 pi r^2 k |dT_a/dr|`` matches the CMB heat
flow: above it conduction alone carries the load, below it convection
must. The quasi-static depth is the limit a steady layer reaches; the
time-dependent layer itself is resolved by :class:`aragog.core.layer.CoreShell`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from aragog.core.entropy import CoreEntropyBudget

jax.config.update('jax_enable_x64', True)

_BISECT_ITERS = 60


def _q_ad(profiles, k_core, r, t_cmb):
    """Heat conducted along the adiabat through radius ``r`` [W]."""
    return -4.0 * jnp.pi * r**2 * k_core * profiles.adiabat_gradient(r, t_cmb)


def _thickness_primal(profiles, k_core, t_cmb, q_cmb):
    """Equilibrium layer thickness [m] by bisection; see the wrapper."""
    p = profiles

    # The layer base sits on the rising branch of Q_ad, so the search runs on the monotone
    # [0, min(r_peak, r_cmb)], the whole core for Earth-scale cores.
    r_peak = p.r_peak
    upper = jnp.minimum(r_peak, p.r_cmb)

    def body(_, bracket):
        lo, hi = bracket
        mid = (lo + hi) / 2.0
        conducts_less = _q_ad(p, k_core, mid, t_cmb) < q_cmb
        return jnp.where(conducts_less, mid, lo), jnp.where(conducts_less, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0,
        _BISECT_ITERS,
        body,
        (jnp.zeros_like(t_cmb * 1.0), jnp.full_like(t_cmb * 1.0, upper)),
    )
    r_s = (lo + hi) / 2.0
    thickness = p.r_cmb - r_s
    # A core larger than the peak radius: the thin-layer estimate ends at
    # the peak, so a layer that would reach past it is reported clamped.
    thickness = jnp.where(r_peak < p.r_cmb, jnp.minimum(thickness, p.r_cmb - r_peak), thickness)
    # Superadiabatic CMB: no stratification. Non-positive flow: fully stratified.
    thickness = jnp.where(q_cmb >= _q_ad(p, k_core, p.r_cmb, t_cmb), 0.0, thickness)
    return jnp.where(q_cmb <= 0.0, p.r_cmb, thickness)


def adiabatic_ratio(entropy: 'CoreEntropyBudget', t_cmb, q_cmb):
    """ADR = Q_cmb / Q_k: below one the top of the core is subadiabatic."""
    return q_cmb / entropy.adiabatic_heat_flow(t_cmb)


def stratification_depth(entropy: 'CoreEntropyBudget', t_cmb, q_cmb):
    """Equilibrium thickness [m] of the stably stratified sub-CMB layer.

    Solves ``Q_ad(r_s) = Q_cmb`` for the layer base ``r_s`` by fixed
    bisection on the inner rising branch of ``Q_ad(r) = 4 pi r^2 k |dT_a/dr|``,
    which peaks at ``profiles.r_peak`` (``D sqrt(3/2)`` on the small-radius adiabat) and
    decreases beyond it; the thickness is ``r_cmb - r_s``. For Earth-scale cores
    the peak sits outside the CMB and the branch spans the whole core;
    for larger cores the search is bracketed at the peak, and a layer
    reaching the peak is reported clamped there, since the thin-layer
    conductive-matching estimate has no meaning deeper. Zero when the
    flow is superadiabatic at the CMB (``ADR >= 1``); the whole core
    when the flow is non-positive (``q_cmb <= 0``).
    """
    return _thickness_primal(entropy.budget.profiles, entropy.k_core, t_cmb, q_cmb)
