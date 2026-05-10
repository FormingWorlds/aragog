"""JAX-based Aragog solver components.

This subpackage provides JIT-compilable, differentiable versions of
the Aragog entropy solver's core components. All modules use JAX
arrays and are compatible with ``jax.jit``, ``jax.grad``,
``jax.jacrev``, and ``jax.vmap``.

It is loaded only when ``solver.use_jax_jacobian = true`` (the
production default). The numpy path in
``aragog.solver.entropy_state`` remains the reference implementation
and is exercised by the standalone tests.

Dependencies: ``jax``, ``equinox`` (both already in the PROTEUS
ecosystem via Atmodeller).
"""

from __future__ import annotations

from aragog.jax.eos import EntropyEOS_JAX, PhaseState
from aragog.jax.nondim import NonDimScales
from aragog.jax.phase import (
    FluxOutput,
    MeshArrays,
    PhaseParams,
    PhaseProperties,
    compute_fluxes,
    compute_mlt,
    evaluate_phase,
)
from aragog.jax.solver import BoundaryParams, SolveResult

__all__ = [
    'EntropyEOS_JAX',
    'PhaseState',
    'NonDimScales',
    'FluxOutput',
    'MeshArrays',
    'PhaseParams',
    'PhaseProperties',
    'compute_fluxes',
    'compute_mlt',
    'evaluate_phase',
    'BoundaryParams',
    'SolveResult',
]
