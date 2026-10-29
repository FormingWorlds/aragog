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


def shape(s: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Slopes at the top and the base, and the mean, of the similarity profile f_s.

    The JAX form of :func:`aragog.surface_lid.shape`: ``(f'(0), f'(1), m)`` for the shape
    parameter ``s`` [-], with the series below ``|s| = 1e-6`` and Dawson's integral for
    ``s < 0``. Each branch is evaluated on a safe argument, so the gradient has no NaN.
    """
    from aragog.surface_lid import _S_SERIES

    s = jnp.asarray(s, dtype=float)
    series = (1.0 + s / 3.0, 1.0 - 2.0 * s / 3.0, 0.5 + s / 12.0)
    sp = jnp.where(s >= _S_SERIES, s, 1.0)
    mu = jnp.sqrt(sp)
    g = mu * math.sqrt(math.pi) * jax.scipy.special.erf(mu)
    grow = (2.0 * sp / g, 2.0 * sp * jnp.exp(-sp) / g, 1.0 + jnp.expm1(-sp) / g)
    sn = jnp.where(s <= -_S_SERIES, s, -1.0)
    nu = jnp.sqrt(-sn)
    dw = dawson(nu)
    thin = (nu * jnp.exp(sn) / dw, nu / dw, 1.0 + jnp.expm1(sn) / (2.0 * nu * dw))
    branch = jnp.where(s >= _S_SERIES, 1, jnp.where(s <= -_S_SERIES, 2, 0))
    return tuple(
        jnp.select([branch == 1, branch == 2], [a, b], c) for a, b, c in zip(grow, thin, series)
    )
