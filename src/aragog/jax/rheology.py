"""JAX bindings for solid-state mantle rheology evaluators.

This module provides JAX-compatible bindings to the shared array-agnostic
rheology formulations defined in ``aragog.rheology``. It binds the array
namespace parameter ``xp`` to ``jax.numpy`` and defines zero physics.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from aragog import rheology


def compute_arrhenius_enthalpy(*args: Any, **kwargs: Any) -> Any:
    """Compute saturating activation enthalpy using jax.numpy."""
    return rheology.compute_arrhenius_enthalpy(*args, xp=jnp, **kwargs)


def compute_arrhenius_viscosity(*args: Any, **kwargs: Any) -> Any:
    """Compute temperature- and pressure-dependent Arrhenius viscosity using jax.numpy."""
    return rheology.compute_arrhenius_viscosity(*args, xp=jnp, **kwargs)


def compute_yield_stress(*args: Any, **kwargs: Any) -> Any:
    """Compute Byerlee frictional yield stress using jax.numpy."""
    return rheology.compute_yield_stress(*args, xp=jnp, **kwargs)


def compute_strain_rate_local(*args: Any, **kwargs: Any) -> Any:
    """Compute local strain rate using jax.numpy."""
    return rheology.compute_strain_rate_local(*args, xp=jnp, **kwargs)


def compute_stagnant_lid_state(*args: Any, **kwargs: Any) -> Any:
    """Compute stagnant lid state diagnostics using jax.numpy."""
    return rheology.compute_stagnant_lid_state(*args, xp=jnp, **kwargs)


def compute_effective_viscosity(*args: Any, **kwargs: Any) -> Any:
    """Compute effective viscosity with plastic yielding cap using jax.numpy."""
    return rheology.compute_effective_viscosity(*args, xp=jnp, **kwargs)
