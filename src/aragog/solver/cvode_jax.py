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

Supported ``core_bc_mode`` values:
- ``quasi_steady``: state vector is N entropy values; RHS is
  ``jax.solver.dSdt``.
- ``energy_balance``: state vector is N+1 (entropy + dSdr_cmb);
  RHS is ``jax.solver.dSdt_energy_balance``. This is the
  production CHILI Earth path under PROTEUS d55726c5+.

Unsupported (factory raises ``ValueError`` and the calling solver
falls back to numpy RHS + FD Jacobian after logging a warning):
- ``bower2018``: extended state with absolute T_core. No JAX
  closure for the core thermal balance has been implemented.
- ``gradient``: extended state with both boundary entropies. No
  JAX implementation.

Status: PROTOTYPE for the supported modes; fallback for the rest.
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
    scales,
    core_bc_mode: str = 'quasi_steady',
    radio_isotope_params: tuple = (),
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
    scales : NonDimScales
        Single source of truth for the nondim scaling (state_scale,
        rhs_scale, t_ref). Constructed by EntropySolver via
        ``_build_nondim_scales``; the contract
        ``rhs_scale = t_ref / state_scale`` is enforced inside
        ``NonDimScales.__post_init__`` (OQ3 option C). The factory only
        validates the per-call shape against ``heating_array`` and
        ``core_bc_mode``.
    core_bc_mode : str, default 'quasi_steady'
        Which JAX RHS to wrap: 'quasi_steady' uses ``jax.solver.dSdt``
        (N-state), 'energy_balance' uses ``jax.solver.dSdt_energy_balance``
        (N+1 state with dSdr_cmb closure equation as the (N+1)-th
        component). The latter matches the production CHILI Earth code
        path (PROTEUS d55726c5+, aragog 718202a+).
    radio_isotope_params : tuple, default ()
        Optional 5-tuple ``(heat_prod, abundance, concentration,
        t0_years, half_life_years)`` of 1D arrays, one entry per
        radionuclide. When non-empty, the JAX RHS evaluates the
        Soucasse §1.2 radio sum at the live integrator time
        ``t_phys`` instead of freezing the value at the coupling-step
        start (A2). The empty default reproduces the pre-A2
        behaviour (no radio).

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
        from aragog.jax.solver import (
            _no_radio,
            dSdt as jax_dsdt,
            dSdt_energy_balance as jax_dsdt_eb,
            make_radio_heating_fn,
        )
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
        # A4: explicit warning + clear error message so the
        # calling solver's catch-all fallback (entropy_solver.py
        # ~1704) logs an informative reason. Previously the ValueError
        # said only what was got and what was expected, leaving the
        # user to guess whether a JAX install was missing or a
        # core_bc_mode mismatch caused the FD-Jacobian fallback.
        logger.warning(
            'JAX CVODE factory: core_bc_mode=%r is not implemented '
            'in the JAX RHS; only quasi_steady and energy_balance '
            'are supported. Falling back to numpy RHS + FD Jacobian.',
            core_bc_mode,
        )
        raise ValueError(
            f'core_bc_mode={core_bc_mode!r} is not supported by the '
            f"JAX CVODE factory. Supported modes: 'quasi_steady', "
            f"'energy_balance'. To use 'bower2018' or 'gradient', "
            f'set ``use_jax_jacobian = false`` in the config.'
        )

    # ── OQ3 option C: NonDimScales is the single source of truth ──
    #
    # The internal nondim contract ``rhs_scale = t_ref / state_scale``
    # is enforced inside ``NonDimScales.__post_init__``, so we no
    # longer recheck it here. The factory only validates the per-call
    # shape of ``scales`` against ``heating_array`` and ``core_bc_mode``.
    from aragog.jax.nondim import NonDimScales

    if not isinstance(scales, NonDimScales):
        raise TypeError(
            'scales must be an aragog.jax.nondim.NonDimScales instance; '
            f'got {type(scales).__name__}. The legacy '
            '(state_scale, rhs_scale, t_ref) positional API was removed '
            'in OQ3 option C; build NonDimScales(state_scale=..., '
            't_ref=...) and let it derive rhs_scale.'
        )
    heating_np = np.asarray(heating_array)
    expected_size = (
        heating_np.size if core_bc_mode == 'quasi_steady'
        else heating_np.size + 1
    )
    if scales.n != expected_size:
        raise ValueError(
            f'state_scale length {scales.n} is incompatible '
            f"with core_bc_mode={core_bc_mode!r} and heating_array length "
            f'{heating_np.size}; expected {expected_size}.'
        )

    state_scale_jax = jnp.asarray(scales.state_scale)
    rhs_scale_jax = jnp.asarray(scales.rhs_scale)
    t_ref = float(scales.t_ref)
    heating_jax = jnp.asarray(heating_array)

    # ── A2: per-step radio heating ──
    # When the caller supplies a non-empty radio_isotope_params tuple,
    # build a JAX-traceable H_radio(t_yr) callable that the dSdt /
    # dSdt_energy_balance RHS evaluates at the live integrator time.
    # Empty tuple -> use the no-op callable, reproducing the pre-A2
    # frozen-at-t_start behaviour (no time dependence).
    if radio_isotope_params:
        if len(radio_isotope_params) != 5:
            raise ValueError(
                'radio_isotope_params must be a 5-tuple '
                '(heat_prod, abundance, concentration, t0_years, '
                f'half_life_years); got length {len(radio_isotope_params)}'
            )
        H_radio_fn = make_radio_heating_fn(*radio_isotope_params)
    else:
        H_radio_fn = _no_radio

    args_tuple = (eos_jax, phase_params, mesh_arrays, boundary_params,
                  heating_jax, H_radio_fn)

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
