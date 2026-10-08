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
  core evolution budget passed as ``core_module_budget``.

Unsupported (factory raises ``ValueError`` and the calling solver
falls back to numpy RHS + FD Jacobian after logging a warning):
- ``bower2018``: extended state with absolute T_core. No JAX
  closure for the core thermal balance has been implemented.
- ``gradient``: extended state with both boundary entropies. No
  JAX implementation.

Status: PROTOTYPE for the supported modes; fallback for the rest.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger('fwl.' + __name__)

_CACHE_MAXSIZE = 8

# Module-level counters for JIT tracing events.
_TRACE_COUNTERS = {'rhs': 0, 'jac': 0}


@dataclass
class _JitCacheEntry:
    rhs_jit: Any
    jac_jit: Any
    phase_params: Any
    eos_jax: Any
    budget: Any = None


_JIT_CACHE: OrderedDict[tuple, _JitCacheEntry] = OrderedDict()


def clear_jit_cache() -> None:
    """Clear the module-level JIT cache and reset trace counters."""
    _JIT_CACHE.clear()
    _TRACE_COUNTERS['rhs'] = 0
    _TRACE_COUNTERS['jac'] = 0


def _make_jitted_rhs_and_jacobian(
    core_bc_mode: str,
    use_radio: bool,
    phase_params: Any,
    eos_jax: Any,
    budget: Any = None,
):
    """Build JIT-compiled RHS and Jacobian functions.

    PhaseParams, eos_jax and the core_module budget are closure constants. All remaining
    parameters are passed as a pytree in data; for core_module it ends with the core source
    power and the critical Rayleigh number of the CMB boundary layer.
    """
    import jax

    from aragog.jax import solver as js

    _rhs_jax = {
        'quasi_steady': js.dSdt,
        'energy_balance': js.dSdt_energy_balance,
        'core_module': js.dSdt_core_module,
    }[core_bc_mode]

    def _eval_core(t_nd, y_nd, data):
        (
            mesh_arrays,
            boundary_params,
            heating_jax,
            radio_arrays,
            state_scale_jax,
            rhs_scale_jax,
            t_ref_jax,
            offset_jax,
        ) = data[:8]
        t_phys = t_nd * t_ref_jax
        S_phys = y_nd * state_scale_jax + offset_jax
        H_radio_fn = (
            functools.partial(js.compute_radio_heating, radio_arrays=radio_arrays)
            if use_radio
            else js._no_radio
        )
        args_tuple = (
            eos_jax,
            phase_params,
            mesh_arrays,
            boundary_params,
            heating_jax,
            H_radio_fn,
        )
        if budget is not None:
            args_tuple = args_tuple + (budget, *data[8:])
        dydt_phys = _rhs_jax(t_phys, S_phys, args_tuple)
        return dydt_phys * rhs_scale_jax

    def _make_wrapper(kind: str):
        def _wrapper(t_nd, y_nd, data):
            _TRACE_COUNTERS[kind] += 1
            return _eval_core(t_nd, y_nd, data)

        return _wrapper

    rhs_jit = jax.jit(_make_wrapper('rhs'))
    jac_jit = jax.jit(jax.jacrev(_make_wrapper('jac'), argnums=1))
    return rhs_jit, jac_jit


def _get_or_create_jitted(
    core_bc_mode: str,
    use_radio: bool,
    phase_params: Any,
    eos_jax: Any,
    budget: Any = None,
) -> tuple[Any, Any, bool]:
    """Retrieve cached jitted functions or compile new ones.

    Returns
    -------
    tuple
        (rhs_jit, jac_jit, is_cache_hit)
    """
    key = (core_bc_mode, use_radio, id(phase_params), id(eos_jax))
    key += () if budget is None else (id(budget),)
    entry = _JIT_CACHE.pop(key, None)
    hit = (
        entry is not None
        and entry.phase_params is phase_params
        and entry.eos_jax is eos_jax
        and entry.budget is budget
    )
    if not hit:
        rhs_jit, jac_jit = _make_jitted_rhs_and_jacobian(
            core_bc_mode, use_radio, phase_params, eos_jax, budget
        )
        entry = _JitCacheEntry(rhs_jit, jac_jit, phase_params, eos_jax, budget)
    _JIT_CACHE[key] = entry
    if len(_JIT_CACHE) > _CACHE_MAXSIZE:
        _JIT_CACHE.popitem(last=False)
    return entry.rhs_jit, entry.jac_jit, hit


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
    core_module_ra_crit_cmb: float | None = None,
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
        The core evolution budget whose ``dtcmb_dt`` closes the
        boundary for ``core_bc_mode='core_module'``; required in that
        mode, ignored otherwise. Its methods are pure JAX, so the
        Jacobian differentiates through it (the boundary solve carries
        a custom JVP).
    core_module_q_radio : float, default 0.0
        Constant core internal source power [W] for the core_module
        closure.
    core_module_ra_crit_cmb : float
        Critical Rayleigh number of the CMB boundary layer for the
        core_module flux (``aragog.core.cmb_boundary_layer_flux``),
        the solver's ``_core_module_ra_crit_cmb``; required for
        core_module and checked with ``aragog.core.check_ra_crit``.

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
        import jax.numpy as jnp

        import aragog.jax.solver  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Option Z (JAX RHS + Jacobian) requires JAX and the '
            f'aragog.jax module. Original error: {exc}'
        ) from exc

    if core_bc_mode == 'core_module' and core_module_budget is None:
        raise ValueError(
            "core_bc_mode='core_module' requires core_module_budget "
            '(the CoreEnergyBudget the solver built from its config); '
            'got None.'
        )
    if core_bc_mode not in ('quasi_steady', 'energy_balance', 'core_module'):
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
    n_extra = {'quasi_steady': 0, 'energy_balance': 1, 'core_module': 2}[core_bc_mode]
    shell = (
        getattr(core_module_budget, 'shell', None) if core_bc_mode == 'core_module' else None
    )
    n_extra += 0 if shell is None else shell.n_cells
    expected_size = heating_np.size + n_extra
    if scales.n != expected_size:
        raise ValueError(
            f'state_scale length {scales.n} is incompatible '
            f'with core_bc_mode={core_bc_mode!r} and heating_array length '
            f'{heating_np.size}; expected {expected_size}.'
        )

    state_scale_jax = jnp.asarray(scales.state_scale)
    rhs_scale_jax = jnp.asarray(scales.rhs_scale)
    t_ref_jax = jnp.asarray(float(scales.t_ref), dtype=jnp.float64)
    heating_jax = jnp.asarray(heating_array)

    radio_arrays = ()
    if radio_isotope_params:
        if len(radio_isotope_params) != 5:
            raise ValueError(
                'radio_isotope_params must be a 5-tuple '
                '(heat_prod, abundance, concentration, t0_years, '
                f'half_life_years); got length {len(radio_isotope_params)}'
            )
        shapes = [np.shape(a) for a in radio_isotope_params]
        if len(set(shapes)) > 1:
            raise ValueError(
                f'All radio_isotope_params arrays must have identical shapes, got {shapes}'
            )
        if np.size(radio_isotope_params[0]) > 0:
            radio_arrays = tuple(
                jnp.asarray(a, dtype=jnp.float64) for a in radio_isotope_params
            )
    use_radio = bool(radio_arrays)

    data = (
        mesh_arrays,
        boundary_params,
        heating_jax,
        radio_arrays,
        state_scale_jax,
        rhs_scale_jax,
        t_ref_jax,
        jnp.asarray(scales.state_offset),
    )
    if core_bc_mode == 'core_module':
        from aragog.core import check_ra_crit

        q_radio, ra_crit = core_module_q_radio, check_ra_crit(core_module_ra_crit_cmb)
        data = data + (jnp.float64(q_radio), jnp.float64(ra_crit))

    budget = core_module_budget if core_bc_mode == 'core_module' else None
    rhs_jit, jac_jit, is_cache_hit = _get_or_create_jitted(
        core_bc_mode, use_radio, phase_params, eos_jax, budget
    )

    info = {
        'rhs_calls': 0,
        'jac_calls': 0,
        'first_rhs_compile_done': False,
        'first_jac_compile_done': False,
    }

    def rhs_fn(t_nd, y_nd, ydot_nd):
        """scikits.odes RHS function: fills ydot in-place."""
        try:
            result = rhs_jit(float(t_nd), jnp.asarray(y_nd), data)
            ydot_nd[:] = np.asarray(result)
            info['rhs_calls'] += 1
            if not info['first_rhs_compile_done']:
                info['first_rhs_compile_done'] = True
                if is_cache_hit:
                    logger.debug('JAX RHS cache hit')
                else:
                    logger.info('JAX RHS first call (JIT compile complete)')
            return 0
        except Exception as exc:
            logger.error('JAX RHS failed: %s', exc)
            return 1

    def jacfn(t_nd, y_nd, fy_nd, J, user_data=None):
        """scikits.odes Jacobian function: fills J in-place."""
        try:
            jac = jac_jit(float(t_nd), jnp.asarray(y_nd), data)
            J[...] = np.asarray(jac)
            info['jac_calls'] += 1
            if not info['first_jac_compile_done']:
                info['first_jac_compile_done'] = True
                if is_cache_hit:
                    logger.debug('JAX Jacobian cache hit')
                else:
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
