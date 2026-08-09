"""Thermal stratification at the top of the core.

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
partial-differential problem outside this stage's cost budget; the
budgets couple to these diagnostics through
``CoreEnergyBudget.convecting_radius``, which reduces the convecting
volume in the energy and entropy integrals when stratification is
enabled.
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
    grad = 2.0 * r * profiles.adiabat(r, t_cmb) / profiles.d_scale**2
    return 4.0 * jnp.pi * r**2 * k_core * grad


def _thickness_primal(profiles, k_core, t_cmb, q_cmb):
    """Equilibrium layer thickness [m] by bisection; see the wrapper."""
    p = profiles

    # Q_ad rises from the centre to its peak at D sqrt(3/2) and falls
    # beyond; the layer base always sits on the rising branch, so the
    # search runs on [0, min(r_peak, r_cmb)] where the profile is
    # monotone. For Earth-scale cores the peak lies outside the CMB and
    # this is the whole core.
    r_peak = p.d_scale * jnp.sqrt(1.5)
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


def make_thickness_fn(profiles, k_core: float):
    """Build the stratified-thickness solve with its sensitivity rule.

    The bisection is comparison-driven, so autodiff sees only the
    converged constant and reads zero derivatives through it; without the
    rule the analytic-Jacobian path would miss the layer's response to
    the state entirely. The rule is the implicit function theorem on
    ``Q_ad(r_s, T) = q``:

        dr_s/dq = 1 / (dQ_ad/dr),  dr_s/dT = -(dQ_ad/dT) / (dQ_ad/dr),

    with the thickness tangent the negative of the base's, pinned to zero
    in the clamped regimes (superadiabatic, fully stratified, past-peak)
    where the primal does not move.
    """

    @jax.custom_jvp
    def thickness(t_cmb, q_cmb):
        return _thickness_primal(profiles, k_core, t_cmb, q_cmb)

    @thickness.defjvp
    def thickness_jvp(primals, tangents):
        t_cmb, q_cmb = primals
        t_dot, q_dot = tangents
        p = profiles
        value = _thickness_primal(profiles, k_core, t_cmb, q_cmb)
        r_s = p.r_cmb - value
        dq_dr = jax.grad(lambda r: _q_ad(p, k_core, r, t_cmb))(r_s)
        dq_dt = jax.grad(lambda t: _q_ad(p, k_core, r_s, t))(t_cmb)
        safe = jnp.where(jnp.abs(dq_dr) > 0.0, dq_dr, 1.0)
        dr_s = (q_dot - dq_dt * t_dot) / safe
        # The thickness moves only while the base sits strictly inside
        # the open rising branch; every clamped regime (no layer, full
        # stratification, past-peak clamp) has zero sensitivity.
        r_peak = p.d_scale * jnp.sqrt(1.5)
        moving = (
            (value > 0.0)
            & (value < p.r_cmb)
            & (r_s > 0.0)
            & (r_s < jnp.minimum(r_peak, p.r_cmb))
        )
        return value, jnp.where(moving, -dr_s, 0.0)

    return thickness


def adiabatic_ratio(entropy: 'CoreEntropyBudget', t_cmb, q_cmb):
    """ADR = Q_cmb / Q_k: below one the top of the core is subadiabatic."""
    return q_cmb / entropy.adiabatic_heat_flow(t_cmb)


def stratification_depth(entropy: 'CoreEntropyBudget', t_cmb, q_cmb):
    """Equilibrium thickness [m] of the stably stratified sub-CMB layer.

    Solves ``Q_ad(r_s) = Q_cmb`` for the layer base ``r_s`` by fixed
    bisection on the outer branch of ``Q_ad(r) = (8 pi k / D^2) r^3
    T_a(r)``, which peaks at ``r_peak = D sqrt(3/2)`` and decreases
    beyond it; the thickness is ``r_cmb - r_s``. For Earth-scale cores
    the peak sits outside the CMB and the branch spans the whole core;
    for larger cores the search is bracketed at the peak, and a layer
    reaching the peak is reported clamped there, since the thin-layer
    conductive-matching estimate has no meaning deeper. Zero when the
    flow is superadiabatic at the CMB (``ADR >= 1``); the whole core
    when the flow is non-positive (``q_cmb <= 0``).
    """
    return _thickness_primal(entropy.budget.profiles, entropy.k_core, t_cmb, q_cmb)
