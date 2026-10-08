"""Crystallization-regime diagnostics of the core.

Where a core crystallizes is set by where the adiabat sits against the
melting curve: solid at the centre only (bottom-up, the Earth case), solid
against the CMB with liquid below (top-down), or one or more interior
solid shells (snow zones). Published evolution models hard-code one
scenario per body from the local slope comparison (the taxonomy of Breuer,
Rueckriemen & Spohn 2015); here the regime is read off the superheat
profile itself on a fixed radial grid. The budget's boundary terms assume
bottom-up growth, so the solver refuses a call that enters any other regime
(``refuse_unmodelled_regime``); multi-zone energetics are out of scope.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
    r = jnp.linspace(0.0, budget.profiles.r_cmb, _N_GRID)
    superheat = budget.superheat(r, t_cmb)
    solid = superheat < 0.0
    changes = jnp.sum(jnp.abs(jnp.diff(solid.astype(jnp.int32))))
    # jnp.select takes the first true condition, the order of the nested checks.
    return jnp.select(
        [~jnp.any(solid), jnp.all(solid), changes > 1, solid[0]],
        [REGIME_FULLY_LIQUID, REGIME_FULLY_FROZEN, REGIME_SNOW, REGIME_BOTTOM_UP],
        REGIME_TOP_DOWN,
    )


def refuse_unmodelled_regime(budget: CoreEnergyBudget, t_cmb) -> None:
    """Refuse core states whose latent and gravitational heat the budget does not book.

    The budget books them only for an inner core that grows from the centre and for its
    freeze-out, so every regime other than fully liquid and bottom-up is refused, and so is a
    fully frozen core whose CMB freezes before its centre.

    Parameters
    ----------
    budget : CoreEnergyBudget
        The core budget; its ``regime_batch``, ``t_onset`` and ``t_freeze`` are used.
    t_cmb : float or array_like
        CMB temperatures [K] of the states to check.

    Raises
    ------
    ValueError
        At the first refused state, naming its regime and temperature.
    """
    t_cmb = np.atleast_1d(np.asarray(t_cmb, dtype=float))
    codes = np.asarray(budget.regime_batch(t_cmb))
    frozen_ok = budget.t_freeze <= budget.t_onset
    bad = (codes > REGIME_BOTTOM_UP) & ((codes != REGIME_FULLY_FROZEN) | (not frozen_ok))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(
            f'core_module: the core crystallizes {regime_name(codes[i])} at T_core = '
            f'{t_cmb[i]:.1f} K; only bottom-up growth is modelled, and the budget books no '
            'latent or gravitational heat in this regime'
        )


def regime_name(code) -> str:
    """Human-readable name for a regime code (eager helper for logs)."""
    return REGIME_NAMES[int(code)]
