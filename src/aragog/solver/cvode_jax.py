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
  production PROTEUS path.
- ``core_module``: state vector is N+2 (entropy + dSdr_cmb +
  T_core); RHS is ``jax.solver.dSdt_core_module``, closed by the
  staged core-evolution budget passed as ``core_module_budget``.

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

logger = logging.getLogger('fwl.' + __name__)


def build_jax_rhs_and_jacobian(
    eos_jax,
    phase_params,
    mesh_arrays,
    boundary_params,
    heating_array,
    scales,
    core_bc_mode: str = 'quasi_steady',
    radio_isotope_params: tuple = (),
    core_module_budget=None,
    core_module_q_radio: float = 0.0,
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
        ``NonDimScales.__post_init__``. The factory validates the
        per-call shape against ``heating_array`` and ``core_bc_mode``.
    core_bc_mode : str, default 'quasi_steady'
        Which JAX RHS to wrap: 'quasi_steady' uses ``jax.solver.dSdt``
        (N-state), 'energy_balance' uses ``jax.solver.dSdt_energy_balance``
        (N+1 state with the dSdr_cmb closure equation), and
        'core_module' uses ``jax.solver.dSdt_core_module`` (N+2 state
        with dSdr_cmb and T_core; requires ``core_module_budget``).
    radio_isotope_params : tuple, default ()
        Optional 5-tuple ``(heat_prod, abundance, concentration,
        t0_years, half_life_years)`` of 1D arrays, one entry per
        radionuclide. When non-empty, the JAX RHS evaluates the
        radiogenic source at the live integrator time ``t_phys`` so
        the heating reflects in-step decay. Empty default disables
        radio heating.
    core_module_budget : CoreEnergyBudget, optional
        The staged core-evolution budget whose ``dtcmb_dt`` closes the
        boundary for ``core_bc_mode='core_module'``; required in that
        mode, ignored otherwise. Its methods are pure JAX, so the
        Jacobian differentiates through it (the boundary solve carries
        a custom JVP).
    core_module_q_radio : float, default 0.0
        Constant core internal source power [W] for the core_module
        closure.

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
            make_radio_heating_fn,
        )
        from aragog.jax.solver import (
            dSdt as jax_dsdt,
        )
        from aragog.jax.solver import (
            dSdt_core_module as jax_dsdt_cm,
        )
        from aragog.jax.solver import (
            dSdt_energy_balance as jax_dsdt_eb,
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
    elif core_bc_mode == 'core_module':
        if core_module_budget is None:
            raise ValueError(
                "core_bc_mode='core_module' requires core_module_budget "
                '(the CoreEnergyBudget the solver built from its config); '
                'got None.'
            )
        _rhs_jax = jax_dsdt_cm
    else:
        # Explicit warning + clear error message so the calling
        # solver's catch-all fallback in entropy_solver.solve logs
        # an informative reason for the FD-Jacobian fallback rather
        # than a bare ValueError.
        logger.warning(
            'JAX CVODE factory: core_bc_mode=%r is not implemented '
            'in the JAX RHS; only quasi_steady, energy_balance, and '
            'core_module are supported. Falling back to numpy RHS + '
            'FD Jacobian.',
            core_bc_mode,
        )
        raise ValueError(
            f'core_bc_mode={core_bc_mode!r} is not supported by the '
            f"JAX CVODE factory. Supported modes: 'quasi_steady', "
            f"'energy_balance', 'core_module'. To use any other mode "
            f'(including {core_bc_mode!r}), set ``use_jax_jacobian = '
            f'false`` in the config, or leave it true to get the '
            f'automatic FD-Jacobian fallback.'
        )

    # NonDimScales enforces the internal nondim contract
    # rhs_scale = t_ref / state_scale in __post_init__. The factory
    # validates only the per-call shape compatibility with
    # ``heating_array`` and ``core_bc_mode``.
    from aragog.jax.nondim import NonDimScales

    if not isinstance(scales, NonDimScales):
        raise TypeError(
            'scales must be an aragog.jax.nondim.NonDimScales instance; '
            f'got {type(scales).__name__}. Build NonDimScales('
            'state_scale=..., t_ref=...) and let it derive rhs_scale.'
        )
    heating_np = np.asarray(heating_array)
    # Boundary states per mode, plus the 2 trailing quadrature states
    # every supported mode carries (EXTRA_STATE_SLOTS is the authority).
    n_extra = {'quasi_steady': 2, 'energy_balance': 3, 'core_module': 4}[core_bc_mode]
    expected_size = heating_np.size + n_extra
    if scales.n != expected_size:
        raise ValueError(
            f'state_scale length {scales.n} is incompatible '
            f'with core_bc_mode={core_bc_mode!r} and heating_array length '
            f'{heating_np.size}; expected {expected_size}.'
        )

    state_scale_jax = jnp.asarray(scales.state_scale)
    rhs_scale_jax = jnp.asarray(scales.rhs_scale)
    t_ref = float(scales.t_ref)
    heating_jax = jnp.asarray(heating_array)

    # Per-step radio heating evaluation. When the caller supplies
    # a non-empty radio_isotope_params tuple, build a JAX-traceable
    # H_radio(t_yr) callable that the dSdt / dSdt_energy_balance RHS
    # evaluates at the live integrator time. Empty tuple -> no-op
    # callable returning zero heating.
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

    args_tuple = (eos_jax, phase_params, mesh_arrays, boundary_params, heating_jax, H_radio_fn)
    if core_bc_mode == 'core_module':
        # The budget rides in the closure; its methods are pure JAX and
        # its parameters are Python floats, so jit treats it as static.
        args_tuple = args_tuple + (core_module_budget, float(core_module_q_radio))

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
            logger.error('JAX Jacobian failed: %s; CVODE will fall back to FD', exc)
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
