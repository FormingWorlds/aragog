"""CVODE solver with JAX RHS and analytic Jacobian (option Z).

Combines the existing JAX physics RHS (``aragog.jax.solver.dSdt``)
with SUNDIALS CVODE via scikits.odes. The Jacobian is computed
analytically via ``jax.jacrev`` instead of CVODE's default finite-
difference approximation.

Benefits over pure scipy/CVODE (numpy RHS):
- JIT-compiled RHS via XLA: 2-5x faster per RHS call typically
- Analytic Jacobian: no FD truncation noise → better Newton
  convergence at phase boundaries (the underlying mechanism for
  the marginal-stability bifurcation)
- Single source of truth for physics (JAX), no risk of numpy/JAX
  divergence

Limitations (current prototype):
- Only supports quasi_steady core_bc (state vector = N entropy values)
- The energy_balance and bower2018 modes use an extended state
  vector (N+1 with dSdr_cmb at end) that the JAX RHS does not
  currently model. Extending requires deriving an analytic dSdr_cmb
  closure equation in JAX.

Status: PROTOTYPE.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def build_jax_rhs_and_jacobian(
    eos_jax,
    phase_params,
    mesh_arrays,
    boundary_params,
    heating_array,
    state_scale,
    rhs_scale,
    t_ref: float,
    core_bc_mode: str = 'quasi_steady',
):
    """Build CVODE-compatible RHS and Jacobian functions backed by JAX.

    Parameters
    ----------
    eos_jax : EntropyEOS_JAX
        JAX EOS tables.
    phase_params : PhaseParams
        Material parameters as a JAX pytree.
    mesh_arrays : MeshArrays
        Mesh geometry as a JAX pytree.
    boundary_params : BoundaryParams
        Boundary conditions as a JAX pytree.
    heating_array : ndarray, shape (n,)
        Internal heating per cell [W/kg]. Will be cast to a JAX array.
    state_scale : ndarray, shape (n,)
        Per-component nondim scale factor: y_phys = y_nd * state_scale.
        Currently expected to be [S_ref] * n (uniform entropy scale).
    rhs_scale : ndarray, shape (n,)
        Per-component nondim scale factor: dy/dt_nd = dy/dt_phys * rhs_scale.
    t_ref : float
        Time scale: t_phys = t_nd * t_ref.
    core_bc_mode : str, default 'quasi_steady'
        Which JAX RHS to wrap: 'quasi_steady' uses ``jax.solver.dSdt``
        (N-state), 'energy_balance' uses ``jax.solver.dSdt_energy_balance``
        (N+1 state with dSdr_cmb closure equation as the (N+1)-th
        component). The latter matches the production CHILI Earth code
        path (PROTEUS d55726c5+, aragog 718202a+).

    Returns
    -------
    rhs_fn : callable
        scikits.odes RHS signature ``rhs_fn(t_nd, y_nd, ydot_nd) -> int``.
    jacfn : callable
        scikits.odes Jacobian signature
        ``jacfn(t_nd, y_nd, fy_nd, J, user_data=None) -> int``.
        Fills ``J`` in-place with the nondim Jacobian matrix.
    info : dict
        Diagnostic info dict (counters, JIT compile times) populated
        on first call.
    """
    try:
        import jax
        import jax.numpy as jnp
        from aragog.jax.solver import dSdt as jax_dsdt
        from aragog.jax.solver import dSdt_energy_balance as jax_dsdt_eb
    except ImportError as exc:
        raise RuntimeError(
            'Option Z (JAX RHS + Jacobian) requires JAX and the '
            f'aragog.jax module. Original error: {exc}'
        ) from exc

    if core_bc_mode == 'quasi_steady':
        _rhs_jax = jax_dsdt
    elif core_bc_mode == 'energy_balance':
        _rhs_jax = jax_dsdt_eb
    else:
        raise ValueError(
            f"core_bc_mode must be 'quasi_steady' or 'energy_balance', "
            f"got {core_bc_mode!r}"
        )

    state_scale_jax = jnp.asarray(state_scale)
    rhs_scale_jax = jnp.asarray(rhs_scale)
    heating_jax = jnp.asarray(heating_array)
    args_tuple = (eos_jax, phase_params, mesh_arrays, boundary_params,
                  heating_jax)

    # Wrap RHS as a function of (t_phys, S_phys) only
    def _rhs_phys(t_phys, S_phys):
        return _rhs_jax(t_phys, S_phys, args_tuple)

    # The "nondim wrapper" applied to JAX RHS and used by both the
    # solver RHS callback and the Jacobian autodiff. Defined as a
    # JAX-traceable function so jacrev can differentiate through it.
    def _rhs_nondim(t_nd, y_nd):
        t_phys = t_nd * t_ref
        S_phys = y_nd * state_scale_jax
        dydt_phys = _rhs_phys(t_phys, S_phys)
        return dydt_phys * rhs_scale_jax

    # JIT-compile both the RHS and its Jacobian. Compilation happens
    # on first call; subsequent calls reuse the compiled artifact.
    rhs_jit = jax.jit(_rhs_nondim)
    # jacrev is fine for square Jacobians; jacfwd would also work
    jac_jit = jax.jit(jax.jacrev(_rhs_nondim, argnums=1))

    info = {
        'rhs_calls': 0,
        'jac_calls': 0,
        'first_rhs_compile_done': False,
        'first_jac_compile_done': False,
    }

    def rhs_fn(t_nd, y_nd, ydot_nd):
        """scikits.odes RHS function: fills ydot in-place."""
        try:
            result = rhs_jit(float(t_nd), jnp.asarray(y_nd))
            ydot_nd[:] = np.asarray(result)
            info['rhs_calls'] += 1
            if not info['first_rhs_compile_done']:
                info['first_rhs_compile_done'] = True
                logger.info('JAX RHS first call (JIT compile complete)')
            return 0
        except Exception as exc:
            logger.error('JAX RHS failed: %s', exc)
            return 1

    def jacfn(t_nd, y_nd, fy_nd, J, user_data=None):
        """scikits.odes Jacobian function: fills J in-place."""
        try:
            jac = jac_jit(float(t_nd), jnp.asarray(y_nd))
            J[...] = np.asarray(jac)
            info['jac_calls'] += 1
            if not info['first_jac_compile_done']:
                info['first_jac_compile_done'] = True
                logger.info('JAX Jacobian first call (JIT compile complete)')
            return 0
        except Exception as exc:
            logger.error('JAX Jacobian failed: %s; CVODE will fall back to FD',
                         exc)
            return 1

    return rhs_fn, jacfn, info


def verify_jax_vs_numpy_rhs(
    rhs_numpy: Callable,
    rhs_jax_phys: Callable,
    t_test: float,
    S_test: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> tuple[bool, dict]:
    """Compare JAX physical RHS against numpy physical RHS.

    The JAX RHS used in CVODE must match numpy's RHS to within
    integrator tolerance, otherwise CVODE's Newton iteration with
    the (analytic JAX) Jacobian will fail to converge against the
    (numpy) RHS. This is a pre-flight check before enabling Z.

    Returns
    -------
    matched : bool
        True if max relative error < rtol.
    info : dict
        Diagnostic with max errors per component.
    """
    import jax.numpy as jnp

    f_np = np.asarray(rhs_numpy(t_test, S_test)).ravel()
    f_jax = np.asarray(rhs_jax_phys(t_test, jnp.asarray(S_test))).ravel()
    abs_err = np.abs(f_np - f_jax)
    denom = np.maximum(np.maximum(np.abs(f_np), np.abs(f_jax)), atol)
    rel_err = abs_err / denom
    matched = bool(np.all(rel_err < rtol))
    info = {
        'matched': matched,
        'max_abs_err': float(abs_err.max()),
        'max_rel_err': float(rel_err.max()),
        'argmax_rel': int(rel_err.argmax()),
        'rtol': rtol,
        'atol': atol,
        'n_components': int(f_np.size),
    }
    return matched, info
