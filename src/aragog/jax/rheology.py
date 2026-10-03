"""JAX bindings for solid-state mantle rheology evaluators.

This module binds the array namespace ``xp`` of the shared formulations in
``aragog.rheology`` to ``jax.numpy`` and defines zero physics.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp

from aragog import rheology

compute_arrhenius_enthalpy = partial(rheology.compute_arrhenius_enthalpy, xp=jnp)
compute_arrhenius_viscosity = partial(rheology.compute_arrhenius_viscosity, xp=jnp)
compute_yield_stress = partial(rheology.compute_yield_stress, xp=jnp)
compute_strain_rate_local = partial(rheology.compute_strain_rate_local, xp=jnp)
compute_stagnant_lid_state = partial(rheology.compute_stagnant_lid_state, xp=jnp)
compute_effective_viscosity = partial(rheology.compute_effective_viscosity, xp=jnp)
