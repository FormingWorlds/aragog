"""JAX functions of the sub-grid surface lid (see ``aragog.surface_lid`` for the model).

All functions are JIT-compilable and differentiable.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

DAWSON_H: float = 0.25
DAWSON_WINDOW: float = 8.0
DAWSON_TERMS: int = math.ceil(DAWSON_WINDOW / DAWSON_H) + 1


def dawson(x: jax.Array) -> jax.Array:
    """Dawson's integral F(x) = exp(-x^2) int_0^x exp(t^2) dt.

    Rybicki's sampling sum (Press et al. 2007, Numerical Recipes, 3rd ed., section 6.9),
    F(x) = lim_{h->0} pi^(-1/2) sum_{n odd} exp(-(x - n h)^2) / n, with the terms of n and -n
    paired as exp(-(x - n h)^2) (1 - exp(-4 n h x)) / n so that it has no cancellation near 0
    and no overflow at large x. With h = 0.25 and the terms with |x - n h| <= 8, the relative
    error against ``scipy.special.dawsn`` is below 1e-13 for |x| <= 75.

    Parameters
    ----------
    x : jax.Array
        Argument [-], any real value.

    Returns
    -------
    jax.Array
        F(x), odd in x, of the shape of ``x``.
    """
    x = jnp.asarray(x, dtype=float)
    sgn = jnp.where(x < 0.0, -1.0, 1.0)
    a = (sgn * x)[..., None]
    n_lo = jnp.maximum(1.0, 2.0 * jnp.floor((a - DAWSON_WINDOW) / (2.0 * DAWSON_H)) + 1.0)
    n = n_lo + 2.0 * jnp.arange(DAWSON_TERMS)
    terms = jnp.exp(-((a - n * DAWSON_H) ** 2)) * -jnp.expm1(-4.0 * DAWSON_H * n * a) / n
    return sgn * terms.sum(-1) / math.sqrt(math.pi)
