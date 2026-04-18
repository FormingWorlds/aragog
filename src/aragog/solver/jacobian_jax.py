"""Analytic Jacobian for CVODE via JAX autodiff (option Z).

Wires JAX-computed Jacobian into scikits.odes CVODE through the
`jacfn` parameter, replacing CVODE's default finite-difference
Jacobian with an exact analytic one.

Status: PROTOTYPE. See module-level NOTES for the design and the
known blockers that need resolving before this can replace the FD
Jacobian in production.

NOTES
-----
1. State vector layout
   The scikits.odes RHS in entropy_solver.py operates on a
   nondimensionalised state vector. In `core_bc='quasi_steady'` mode
   the state is N entropy values; in `core_bc='energy_balance'` mode
   it is N+1 (entropy + dSdr_cmb). The JAX `dSdt` in
   ``aragog.jax.solver`` operates on N entropy values only.

   For energy_balance support, the JAX Jacobian must be extended to
   the N+1 state vector. The dSdr_cmb closure equation needs an
   analytic derivative w.r.t. the entropy block.

2. Physics parity
   The numpy and JAX RHS implementations both compute the entropy
   equation, but via different code paths. The Jacobian provided to
   CVODE MUST match CVODE's actual RHS (the numpy one). If the JAX
   dSdt(t, S) differs from the numpy dSdt(t, S) by more than a few
   ULP, CVODE's Newton iteration will fail to converge.

   Verification step needed: run both RHS implementations on the
   same state, check |numpy - JAX| < 1e-10 * |numpy| for all output
   components. If they don't match, port the necessary numpy code
   to JAX before using this Jacobian.

3. JIT compilation cost
   `jax.jit(jax.jacrev(dSdt, argnums=1))` requires JAX to trace the
   computation, which takes 5-30 seconds on first call. Subsequent
   calls reuse the compiled artifact and are fast (<1 ms typically).

   The compiled Jacobian must be cached across PROTEUS coupling
   steps; the existing `EntropySolver._jit_jac` slot should hold it
   for the lifetime of the solver instance.

4. scikits.odes jacfn signature
   def jacfn(t, y, fy, J, user_data=None):
       J[...] = jax_jac_fn(t, y)
       return 0
   The output J must be a 2D numpy array (n, n), filled in-place.
   For the banded linear solver, the layout is jac_packed[i-j+lband, j].
   We currently use 'dense' for energy_balance, which expects the
   plain (n, n) layout — simpler.

5. Performance expectation
   FD Jacobian for n=80 dense: 81 RHS evaluations per setup.
   Analytic Jacobian: 1 jax.jacrev call (which internally is ~n
   reverse-mode passes for a square Jacobian, but JIT-compiled to
   roughly 1 RHS-equivalent worth of compute).

   Net speedup per Jacobian rebuild: ~50-80x.
   With CVODE rebuilding ~5% of steps, this saves ~50% of total
   "setup" time, which is roughly half of total CVODE time.
   Expected per-coupling-step Aragog speedup: 2-5x.
   Per coupling step where AGNI dominates ~80%: net wall-time gain
   of 5-15%.

   The bigger win is robustness: analytic Jacobian has no FD
   truncation noise, which is the underlying cause of CVODE's
   Newton convergence failures at phase boundaries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

logger = logging.getLogger(__name__)


def make_jacobian_fn_quasi_steady(
    rhs_jax: Callable,
    state_scale: np.ndarray,
    rhs_scale: np.ndarray,
    t_ref: float,
):
    """Build a CVODE-compatible Jacobian function from a JAX RHS.

    Only supports quasi_steady core_bc (state vector = N entropy
    values). For energy_balance, see make_jacobian_fn_energy_balance
    (TODO).

    Parameters
    ----------
    rhs_jax : callable
        JAX-traceable RHS: rhs_jax(t_phys, S_phys) -> dS/dt_phys.
        Must accept JAX arrays and return a JAX array.
    state_scale : array, shape (n,)
        Per-component scale factor: y_phys = y_nd * state_scale.
    rhs_scale : array, shape (n,)
        Per-component scale factor: dy/dt_nd = dy/dt_phys * rhs_scale.
    t_ref : float
        Time scale: t_phys = t_nd * t_ref.

    Returns
    -------
    jacfn : callable
        Function with scikits.odes signature
        ``jacfn(t_nd, y_nd, fy_nd, J_nd, user_data=None) -> int``.
        Fills J_nd in place with d(dy_nd/dt_nd)/dy_nd evaluated at
        the nondimensionalised state.

    NOTES
    -----
    The Jacobian transforms under nondimensionalisation as:

        J_nd[i,j] = d(dy_nd[i]/dt_nd) / dy_nd[j]
                  = d(dy_phys[i]/dt_phys * rhs_scale[i] * t_ref) / d(y_phys[j] / state_scale[j])
                  = J_phys[i,j] * rhs_scale[i] * t_ref * state_scale[j]

    where J_phys is the physical Jacobian computed by JAX.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            'Option Z (analytic Jacobian) requires JAX. '
            f'Install jax to enable. Original error: {exc}'
        ) from exc

    # Build the JIT-compiled physical Jacobian function
    def _rhs_phys(t_phys, S_phys):
        return rhs_jax(t_phys, S_phys)

    jac_phys = jax.jit(jax.jacrev(_rhs_phys, argnums=1))

    state_scale_arr = jnp.asarray(state_scale)
    rhs_scale_arr = jnp.asarray(rhs_scale)

    # Pre-compute the nondim transformation matrix
    # J_nd[i,j] = J_phys[i,j] * (rhs_scale[i] * t_ref) * state_scale[j]
    nondim_pre = rhs_scale_arr * t_ref          # shape (n,), broadcasts on row
    nondim_post = state_scale_arr               # shape (n,), broadcasts on col

    def jacfn(t_nd, y_nd, fy_nd, J, user_data=None):
        """scikits.odes Jacobian function."""
        try:
            t_phys = float(t_nd) * t_ref
            S_phys = jnp.asarray(y_nd) * state_scale_arr
            jac = jac_phys(t_phys, S_phys)
            # Apply nondim transformation:
            # J_nd[i,j] = jac[i,j] * nondim_pre[i] * nondim_post[j]
            jac_nd = jac * nondim_pre[:, None] * nondim_post[None, :]
            J[...] = np.asarray(jac_nd)
            return 0
        except Exception as exc:
            logger.error('JAX Jacobian failed: %s; falling back to FD', exc)
            return 1  # signal CVODE to use FD

    return jacfn


def verify_rhs_match(
    rhs_numpy: Callable,
    rhs_jax: Callable,
    t_test: float,
    S_test: np.ndarray,
    rtol: float = 1e-10,
):
    """Verify that the JAX RHS matches the numpy RHS to FP precision.

    Required before using the JAX Jacobian: if the JAX path produces
    a different RHS than the numpy path that CVODE actually evaluates,
    the Jacobian will be inconsistent and CVODE's Newton iteration
    will fail.

    Parameters
    ----------
    rhs_numpy : callable
        rhs_numpy(t, S) -> dS/dt using the numpy code path.
    rhs_jax : callable
        rhs_jax(t, S) -> dS/dt using the JAX code path.
    t_test, S_test : float, array
        Test point.
    rtol : float
        Maximum tolerated relative difference.

    Returns
    -------
    matched : bool
        True if RHS implementations match to within rtol.
    info : dict
        Diagnostic info with max abs/rel error per component.
    """
    import jax.numpy as jnp

    f_np = np.asarray(rhs_numpy(t_test, S_test))
    f_jax = np.asarray(rhs_jax(t_test, jnp.asarray(S_test)))
    abs_err = np.abs(f_np - f_jax)
    rel_err = abs_err / np.maximum(np.abs(f_np), 1e-30)
    matched = bool(np.all(rel_err < rtol))
    info = {
        'matched': matched,
        'max_abs_err': float(abs_err.max()),
        'max_rel_err': float(rel_err.max()),
        'argmax_rel': int(rel_err.argmax()),
        'rtol': rtol,
    }
    return matched, info


# TODO: make_jacobian_fn_energy_balance for N+1 state vector with
# dSdr_cmb closure equation. Requires analytic derivative of the
# energy_balance BC w.r.t. the entropy block.
