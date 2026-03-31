"""JAX-based Aragog solver components.

This subpackage provides JIT-compilable, differentiable versions of
the Aragog entropy solver's core components. All modules use JAX arrays
and are compatible with jax.jit, jax.grad, and jax.vmap.

Dependencies: jax, equinox (both already in PROTEUS ecosystem via atmodeller).
No new packages required for Phases 1-2.
"""
