"""Crystallization-regime diagnostics of the core.

Where a core crystallizes is set by where the adiabat sits against the
melting curve: solid at the centre only (bottom-up, the Earth case), solid
against the CMB with liquid below (top-down), or one or more interior
solid shells (snow zones). Published evolution models hard-code one
scenario per body from the local slope comparison (the taxonomy of Breuer,
Rueckriemen & Spohn 2015); here the regime is read off the superheat
profile itself on a fixed radial grid and reported as a diagnostic flag,
so a run announces when it leaves the bottom-up regime the budget's
boundary terms assume. Classification only; the multi-zone energetics of
non-bottom-up regimes is deliberately out of scope for this stage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aragog.core.budget import CoreEnergyBudget

jax.config.update('jax_enable_x64', True)

# Regime codes: plain ints so the flag is trace-safe and storable in
# output tables. Names are for logs and docs.
REGIME_FULLY_LIQUID = 0
REGIME_BOTTOM_UP = 1
REGIME_TOP_DOWN = 2
REGIME_SNOW = 3
REGIME_FULLY_FROZEN = 4

REGIME_NAMES = {
    REGIME_FULLY_LIQUID: 'fully_liquid',
    REGIME_BOTTOM_UP: 'bottom_up',
    REGIME_TOP_DOWN: 'top_down',
    REGIME_SNOW: 'snow',
    REGIME_FULLY_FROZEN: 'fully_frozen',
}

_N_GRID = 512  # fixed sampling of the superheat profile; jit-safe


def crystallization_regime(budget: CoreEnergyBudget, t_cmb):
    """Regime code (see ``REGIME_NAMES``) at ``t_cmb``.

    The superheat ``T_a - T_m`` is sampled on a fixed radial grid; its
    sign at the centre and the CMB plus the number of sign changes
    classify the state:

    * no solid anywhere: ``fully_liquid``
    * no liquid anywhere: ``fully_frozen``
    * one interface, solid centre: ``bottom_up``
    * one interface, solid top: ``top_down``
    * more than one interface: ``snow`` (at least one interior solid or
      liquid shell)

    The grid resolves shells wider than ``r_cmb / 512``; a thinner shell
    than that reads as its surrounding regime.
    """
    p = budget.profiles
    r = jnp.linspace(0.0, p.r_cmb, _N_GRID)
    superheat = p.adiabat(r, t_cmb) - budget.melting_curve.t_melt(p.pressure(r))
    solid = superheat < 0.0
    changes = jnp.sum(jnp.abs(jnp.diff(solid.astype(jnp.int32))))
    centre_solid = solid[0]
    any_solid = jnp.any(solid)
    all_solid = jnp.all(solid)

    return jnp.where(
        ~any_solid,
        REGIME_FULLY_LIQUID,
        jnp.where(
            all_solid,
            REGIME_FULLY_FROZEN,
            jnp.where(
                changes > 1,
                REGIME_SNOW,
                jnp.where(centre_solid, REGIME_BOTTOM_UP, REGIME_TOP_DOWN),
            ),
        ),
    )


def regime_name(code) -> str:
    """Human-readable name for a regime code (eager helper for logs)."""
    return REGIME_NAMES[int(code)]
