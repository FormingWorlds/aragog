"""Thermal stratification diagnostics at the top of the core.

When the CMB heat flow drops below what conduction carries along the
adiabat, the top of the core stops convecting and a stably stratified
layer grows. Two ODE-cost quantities describe the state. The adiabatic
ratio ``ADR = Q_cmb / Q_k`` (the actual-to-adiabatic CMB gradient ratio,
as defined in the Leeds ``thermal_history`` stable-layer model) is the
onset criterion: below one, stratification grows. The equilibrium
stratification depth is the radius where the adiabatic conducted flow
``Q_ad(r) = 4 pi r^2 k |dT_a/dr|`` matches the CMB heat flow: above it
conduction alone carries the load, below it convection must. The full
time-dependent layer (diffusive profile, entrainment) is a
partial-differential problem outside this stage's cost budget, and the
reduction of the convecting volume in the energy and entropy integrals is
the coupling step that builds on these diagnostics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aragog.core.entropy import CoreEntropyBudget

jax.config.update('jax_enable_x64', True)

_BISECT_ITERS = 60


def adiabatic_ratio(entropy: CoreEntropyBudget, t_cmb, q_cmb):
    """ADR = Q_cmb / Q_k: below one the top of the core is subadiabatic."""
    return q_cmb / entropy.adiabatic_heat_flow(t_cmb)


def stratification_depth(entropy: CoreEntropyBudget, t_cmb, q_cmb):
    """Equilibrium thickness [m] of the stably stratified sub-CMB layer.

    Solves ``Q_ad(r_s) = Q_cmb`` for the layer base ``r_s`` by fixed
    bisection; the thickness is ``r_cmb - r_s``. ``Q_ad(r) = (8 pi k /
    D^2) r^3 T_a(r)`` increases monotonically outward through the core,
    so the root is unique. Zero when the flow is superadiabatic at the
    CMB (``ADR >= 1``); the whole core when even the centre cannot carry
    the flow conductively (``q_cmb <= 0``).
    """
    p = entropy.budget.profiles

    def q_ad(r):
        grad = 2.0 * r * p.adiabat(r, t_cmb) / p.d_scale**2
        return 4.0 * jnp.pi * r**2 * entropy.k_core * grad

    def body(_, bracket):
        lo, hi = bracket
        mid = (lo + hi) / 2.0
        conducts_less = q_ad(mid) < q_cmb
        return jnp.where(conducts_less, mid, lo), jnp.where(conducts_less, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0,
        _BISECT_ITERS,
        body,
        (jnp.zeros_like(t_cmb * 1.0), jnp.full_like(t_cmb * 1.0, p.r_cmb)),
    )
    r_s = (lo + hi) / 2.0
    thickness = p.r_cmb - r_s
    # Superadiabatic CMB: no stratification. Non-positive flow: fully stratified.
    thickness = jnp.where(q_cmb >= q_ad(p.r_cmb), 0.0, thickness)
    return jnp.where(q_cmb <= 0.0, p.r_cmb, thickness)
