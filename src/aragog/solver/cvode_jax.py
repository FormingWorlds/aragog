"""CVODE solver with JAX RHS and analytic Jacobian (option Z).

Combines the existing JAX physics RHS (``aragog.jax.solver.dSdt``) with
SUNDIALS CVODE via scikits.odes. The Jacobian is computed analytically
via ``jax.jacrev`` instead of CVODE's default finite-difference
approximation.
"""

from __future__ import annotations

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
):
    """Build JIT-compiled RHS and Jacobian functions.

    PhaseParams and eos_jax are closure constants.
    All remaining parameters are passed as a pytree in data.
    """
    import jax

    from aragog.jax.solver import (
        _no_radio,
        compute_radio_heating,
    )
    from aragog.jax.solver import (
        dSdt as jax_dsdt,
    )
    from aragog.jax.solver import (
        dSdt_energy_balance as jax_dsdt_eb,
    )

    _rhs_jax = jax_dsdt if core_bc_mode == 'quasi_steady' else jax_dsdt_eb

    def _eval_core(t_nd, y_nd, data):
        (
            mesh_arrays,
            boundary_params,
            heating_jax,
            radio_arrays,
            state_scale_jax,
            rhs_scale_jax,
            t_ref_jax,
        ) = data
        t_phys = t_nd * t_ref_jax
        S_phys = y_nd * state_scale_jax
        if use_radio:

            def H_radio_fn(t):
                return compute_radio_heating(t, radio_arrays)

        else:
            H_radio_fn = _no_radio
        args_tuple = (
            eos_jax,
            phase_params,
            mesh_arrays,
            boundary_params,
            heating_jax,
            H_radio_fn,
        )
        dydt_phys = _rhs_jax(t_phys, S_phys, args_tuple)
        return dydt_phys * rhs_scale_jax

    def _rhs_nondim(t_nd, y_nd, data):
        _TRACE_COUNTERS['rhs'] += 1
        return _eval_core(t_nd, y_nd, data)

    def _jac_nondim(t_nd, y_nd, data):
        _TRACE_COUNTERS['jac'] += 1
        return _eval_core(t_nd, y_nd, data)

    rhs_jit = jax.jit(_rhs_nondim)
    jac_jit = jax.jit(jax.jacrev(_jac_nondim, argnums=1))
    return rhs_jit, jac_jit


def _get_or_create_jitted(
    core_bc_mode: str,
    use_radio: bool,
    phase_params: Any,
    eos_jax: Any,
) -> tuple[Any, Any, bool]:
    """Retrieve cached jitted functions or compile new ones.

    Returns
    -------
    tuple
        (rhs_jit, jac_jit, is_cache_hit)
    """
    key = (core_bc_mode, use_radio, id(phase_params), id(eos_jax))
    if key in _JIT_CACHE:
        entry = _JIT_CACHE[key]
        if entry.phase_params is phase_params and entry.eos_jax is eos_jax:
            _JIT_CACHE.move_to_end(key)
            return entry.rhs_jit, entry.jac_jit, True
        del _JIT_CACHE[key]

    rhs_jit, jac_jit = _make_jitted_rhs_and_jacobian(
        core_bc_mode, use_radio, phase_params, eos_jax
    )
    _JIT_CACHE[key] = _JitCacheEntry(rhs_jit, jac_jit, phase_params, eos_jax)
    if len(_JIT_CACHE) > _CACHE_MAXSIZE:
        _JIT_CACHE.popitem(last=False)
    return rhs_jit, jac_jit, False


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
        Internal heating per cell [W/kg]. Cast to a JAX array.
    scales : NonDimScales
        Source of truth for nondim scaling (state_scale, rhs_scale, t_ref).
    core_bc_mode : str, default 'quasi_steady'
        Which JAX RHS to wrap: 'quasi_steady' or 'energy_balance'.
    radio_isotope_params : tuple, default ()
        Optional 5-tuple of radionuclide arrays.

    Returns
    -------
    rhs_fn : callable
        scikits.odes RHS signature ``rhs_fn(t_nd, y_nd, ydot_nd) -> int``.
    jacfn : callable
        scikits.odes Jacobian signature
        ``jacfn(t_nd, y_nd, fy_nd, J, user_data=None) -> int``.
    info : dict
        Diagnostic info dict populated on first call.
    """
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            'Option Z (JAX RHS + Jacobian) requires JAX and the '
            f'aragog.jax module. Original error: {exc}'
        ) from exc

    if core_bc_mode not in ('quasi_steady', 'energy_balance'):
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

    from aragog.jax.nondim import NonDimScales

    if not isinstance(scales, NonDimScales):
        raise TypeError(
            'scales must be an aragog.jax.nondim.NonDimScales instance; '
            f'got {type(scales).__name__}. Build NonDimScales('
            'state_scale=..., t_ref=...) and let it derive rhs_scale.'
        )
    heating_np = np.asarray(heating_array)
    expected_size = heating_np.size if core_bc_mode == 'quasi_steady' else heating_np.size + 1
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

    use_radio = False
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
        if any(np.asarray(a).size > 0 for a in radio_isotope_params):
            use_radio = True
            radio_arrays = tuple(
                jnp.asarray(a, dtype=jnp.float64) for a in radio_isotope_params
            )

    data = (
        mesh_arrays,
        boundary_params,
        heating_jax,
        radio_arrays,
        state_scale_jax,
        rhs_scale_jax,
        t_ref_jax,
    )

    rhs_jit, jac_jit, is_cache_hit = _get_or_create_jitted(
        core_bc_mode, use_radio, phase_params, eos_jax
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
