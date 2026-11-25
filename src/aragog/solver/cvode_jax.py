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
  T_core) plus the shell temperatures of a stratified core; RHS
  is ``jax.solver.dSdt_core_module``, closed by the core evolution
  budget passed as ``core_module_budget``.

Unsupported (factory raises ``UnsupportedCoreBCError`` and the calling
solver falls back to numpy RHS + FD Jacobian after logging a warning):
- ``bower2018``: extended state with absolute T_core. No JAX
  closure for the core thermal balance has been implemented.
- ``gradient``: extended state with both boundary entropies. No
  JAX implementation.

Status: PROTOTYPE for the supported modes; fallback for the rest.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

import numpy as np

from aragog.config.boundary import CORE_BC_MODES

logger = logging.getLogger('fwl.' + __name__)

# Fixed chunk length for batched per-call energy power evaluation.
C: int = 512

# The core_bc modes with a JAX RHS, each with its count of state entries after the entropies.
_JAX_N_EXTRA = {'quasi_steady': 0, 'energy_balance': 1, 'core_module': 2}

# Module-level counters for JIT tracing events.
_TRACE_COUNTERS = {'rhs': 0, 'jac': 0, 'powers': 0}


class UnsupportedCoreBCError(ValueError):
    """A config ``core_bc_mode`` with no JAX RHS (bower2018, gradient); solve() uses NumPy."""


class _ByValue:
    """A pytree held as a static program argument: its leaves are compile-time constants, and
    two holders are equal when their treedefs and the bytes of their leaves are equal.

    The leaves are read-only host copies, so a later in-place change of the source cannot reach
    a compiled program. The hash is taken once; equality is identity first.
    """

    __slots__ = ('tree', '_leaves', '_treedef', '_hash')

    def __init__(self, tree) -> None:
        import jax

        leaves, treedef = jax.tree.flatten(tree)
        held = []
        for x in leaves:
            a = np.array(x)
            a.flags.writeable = False
            held.append(a)
        for name, value in (
            ('tree', jax.tree.unflatten(treedef, held)),
            ('_leaves', tuple(held)),
            ('_treedef', treedef),
            (
                '_hash',
                hash((treedef, tuple((a.tobytes(), a.dtype.str, a.shape) for a in held))),
            ),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value) -> None:
        raise AttributeError('a _ByValue cannot change after it is built')

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, _ByValue) or self._hash != other._hash:
            return False
        return self._treedef == other._treedef and all(
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            for a, b in zip(self._leaves, other._leaves, strict=True)
        )


# One static argument tuple per value, so a program's jit cache matches builds by identity.
_HELD: dict = {}

# Core budget leaves held as program constants, by attribute path: the config constants. The
# profile fit (rho_cen, length_scale, d_scale) and the CMB follow the structure and stay traced.
_BUDGET_CONSTANTS = frozenset(
    {
        ('alpha_c',),
        ('c_light',),
        ('ds_fusion',),
        ('icn_width',),
        ('latent_heat',),
        ('melting_curve', 'depression'),
        ('melting_curve', 'light_element_fraction'),
        ('melting_curve', 't_m0'),
        ('melting_curve', 't_m1'),
        ('melting_curve', 't_m2'),
        ('profiles', 'alpha'),
        ('profiles', 'c_p'),
    }
)


def _is_constant(path) -> bool:
    """Whether the leaf at ``path`` of ``(data, phase_params, eos, budget)`` is held as a
    program constant: every phase parameter and EOS table value, and the budget config
    constants."""
    part = path[0].idx
    if part == 3:
        return tuple(getattr(k, 'name', None) for k in path[1:]) in _BUDGET_CONSTANTS
    return part in (1, 2)


# At most this many input forms keep compiled CVODE programs; a new form beyond it clears them.
MAX_FORMS = 8
_FORMS: set = set()


def clear_jit_cache() -> None:
    """Drop the compiled CVODE programs, the record of their input forms, the interned static
    arguments and the trace counters."""
    if _programs.cache_info().currsize:
        for program in _programs():
            program.clear_cache()
    _programs.cache_clear()
    _FORMS.clear()
    _HELD.clear()
    for key in _TRACE_COUNTERS:
        _TRACE_COUNTERS[key] = 0


def _bound_forms(core_bc_mode, use_radio, data, consts) -> None:
    """Record the input form of a factory build; a new form beyond ``MAX_FORMS`` first clears the
    compiled programs.

    Notes
    -----
    The bound covers the 4 CVODE programs of ``_programs`` (RHS, Jacobian, powers, lid weight),
    by the count of input forms. It counts builds: a factory built before a clear and called
    again compiles once more, and its static arguments then match those of a later equal build
    by value, not identity. Outside the bound:

    - the budget programs of ``aragog.core.budget``: their batch calls are padded to a power-of-2
      length, so they compile once per length class and budget form; a tabulated core whose grid
      length follows ``r_cmb`` is a new budget form, so it compiles them again at each structure
      update;
    - the executables JAX keeps for eager array operations, 1 per operation and input shape, so a
      new mesh size adds some.

    A tabulated core with the default ``t_cmb_range`` sizes its CMB temperature grid from the
    core pressures, so a structure update that changes its length is also a new form of the RHS
    and Jacobian.
    """
    import jax

    leaves, treedef = jax.tree_util.tree_flatten((data, consts))
    form = (
        core_bc_mode,
        use_radio,
        treedef,
        tuple(
            x if isinstance(x, _ByValue) else (np.shape(x), getattr(x, 'dtype', type(x)))
            for x in leaves
        ),
    )
    if form in _FORMS:
        return
    if len(_FORMS) >= MAX_FORMS:
        logger.debug('JAX CVODE programs: %d input forms held, clearing them', len(_FORMS))
        clear_jit_cache()
    _FORMS.add(form)


def _args_from_data(
    data, phase_params: Any, eos_jax: Any, use_radio: bool, budget: Any = None
) -> tuple:
    """Build the physical RHS argument tuple from the program arguments ``data``, the phase
    parameters and the EOS; for core_module it ends with the budget and the data entries after
    the state offset."""
    from aragog.jax import solver as js

    mesh_arrays, boundary_params, heating_jax, radio_arrays = data[:4]
    H_radio_fn = (
        functools.partial(js.compute_radio_heating, radio_arrays=radio_arrays)
        if use_radio
        else js._no_radio
    )
    args = (eos_jax, phase_params, mesh_arrays, boundary_params, heating_jax, H_radio_fn)
    return args if budget is None else args + (budget, *data[8:])


def _eval_core(core_bc_mode, use_radio, t_nd, y_nd, data, consts):
    """Nondimensional RHS of the state ``y_nd`` at time ``t_nd``; ``consts`` holds the phase
    parameters, the EOS and the core_module budget (None in the other modes)."""
    from aragog.jax import solver as js

    rhs = {
        'quasi_steady': js.dSdt,
        'energy_balance': js.dSdt_energy_balance,
        'core_module': js.dSdt_core_module,
    }[core_bc_mode]
    phase_params, eos_jax, budget = consts
    state_scale_jax, rhs_scale_jax, t_ref_jax, offset_jax = data[4:8]
    args_tuple = _args_from_data(data, phase_params, eos_jax, use_radio, budget)
    dydt_phys = rhs(t_nd * t_ref_jax, y_nd * state_scale_jax + offset_jax, args_tuple)
    return dydt_phys * rhs_scale_jax


def _flat_core(core_bc_mode, use_radio, held, layout, t_nd, y_nd, leaves):
    """``_eval_core`` with the constant leaves from the static holder ``held`` and the other
    leaves flat; ``layout`` is the treedef of ``(data, phase_params, eos, budget)`` and its mask
    of constant leaves."""
    import jax

    treedef, mask = layout
    const, traced = iter(jax.tree.map(jax.numpy.asarray, held.tree)), iter(leaves)
    data, *consts = jax.tree.unflatten(treedef, [next(const if m else traced) for m in mask])
    return _eval_core(core_bc_mode, use_radio, t_nd, y_nd, data, consts)


def _rhs_program(core_bc_mode, use_radio, held, layout, t_nd, y_nd, *leaves):
    _TRACE_COUNTERS['rhs'] += 1
    return _flat_core(core_bc_mode, use_radio, held, layout, t_nd, y_nd, leaves)


def _jac_program(core_bc_mode, use_radio, held, layout, t_nd, y_nd, *leaves):
    _TRACE_COUNTERS['jac'] += 1
    return _flat_core(core_bc_mode, use_radio, held, layout, t_nd, y_nd, leaves)


def _powers_program(core_bc_mode, use_radio, t_chunk, Y_chunk, data, aux, consts):
    import jax

    from aragog.jax import solver as js

    _TRACE_COUNTERS['powers'] += 1
    phase_params, eos_jax, budget = consts

    def _eval_node(t_phys, y_phys):
        args_tuple = _args_from_data(data, phase_params, eos_jax, use_radio, budget)
        return js.step_powers(t_phys, y_phys, args_tuple, core_bc_mode, aux)

    return jax.vmap(_eval_node, in_axes=(0, 1))(t_chunk, Y_chunk)


def _lid_weight_program(core_bc_mode, use_radio, t_chunk, Y_chunk, data, consts):
    import jax

    from aragog.jax import solver as js

    phase_params, eos_jax, budget = consts

    def _eval_node(t_phys, y_phys):
        args_tuple = _args_from_data(data, phase_params, eos_jax, use_radio, budget)
        return js.lid_weight(t_phys, y_phys, args_tuple, core_bc_mode)

    return jax.vmap(_eval_node, in_axes=(0, 1))(t_chunk, Y_chunk)


@functools.cache
def _programs():
    """The RHS, Jacobian, step-power and lid-weight programs, with the core_bc mode and the
    radiogenic switch static. The RHS and Jacobian also take as static arguments the by-value
    holder of the constant leaves (the phase parameters, the EOS tables and the budget config
    constants) and the layout (treedef and mask), and every other leaf flat; the step-power
    and lid-weight programs take all inputs as traced arguments."""
    import jax

    return (
        jax.jit(_rhs_program, static_argnums=(0, 1, 2, 3)),
        jax.jit(jax.jacrev(_jac_program, argnums=5), static_argnums=(0, 1, 2, 3)),
        jax.jit(_powers_program, static_argnums=(0, 1)),
        jax.jit(_lid_weight_program, static_argnums=(0, 1)),
    )


def _strong(tree):
    """``tree`` with its Python and NumPy float leaves as strong float64 scalars and its weak JAX
    arrays as strong arrays, so that every build passes the same argument types."""
    import jax

    def leaf(x):
        if isinstance(x, jax.Array) and x.weak_type:
            return x.astype(x.dtype)
        return np.float64(x) if isinstance(x, (float, np.floating)) else x

    return jax.tree_util.tree_map(leaf, tree)


class LidFactoryMismatch(ValueError):
    """The JAX factory and the solver disagree on the sub-grid lid."""


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
    core_module_ra_crit_cmb: float | str | None = None,
    start_time: float | None = None,
    *,
    lid_params=None,
    core_module_surface_temperature: float | None = None,
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
        t0_years, half_life_years)`` of 1D arrays of one shape, one entry per
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
    core_module_ra_crit_cmb : float, 'phase' or None
        Critical Rayleigh number of the CMB boundary layer for the
        core_module flux (``aragog.core.cmb_boundary_layer_flux``),
        the solver's ``_core_module_ra_crit_cmb``; None selects Thiriet
        et al. (2019) eq. 16; 'phase' blends 450 and eq. 16 by the bottom-cell melt fraction
        across the rheological transition of ``phase_params``; checked with
        ``aragog.core.check_ra_crit``.
    core_module_surface_temperature : float or None
        Surface temperature [K] of the shell contrast of eq. 16, the solver's
        ``cmb_shell_surface_temperature()``; None or NaN: the top cell's temperature.
    lid_params : aragog.jax.surface_lid.LidParams, optional
        The sub-grid lid of the solver (``EntropySolver.jax_lid_params()``). With it the state
        ends with the lid thickness and energy, and ``rhs_fn.lid`` and ``info['lid']`` are
        True; None, the default, builds the RHS without the lid. It decides the lid of the
        factory, also over a ``lid`` in ``boundary_params``.
    start_time : float, optional
        Time [yr] at which the radiogenic heating of each isotope of ``radio_isotope_params``
        must be finite (``aragog.parser.validate_radionuclide``); None skips the check.
        ``EntropySolver.solve`` passes no start time to the factory, and no caller in
        aragog or PROTEUS passes one.

    Returns
    -------
    rhs_fn : callable
        scikits.odes RHS signature ``rhs_fn(t_nd, y_nd, ydot_nd) -> int``.
        Carries ``step_powers(t_nodes, Y_nodes, aux) -> ndarray (n, 7)``
        for batched per-call energy power evaluation in physical units.
    jacfn : callable
        scikits.odes Jacobian signature
        ``jacfn(t_nd, y_nd, fy_nd, J, user_data=None) -> int``.
        Fills ``J`` in-place with the nondim Jacobian matrix.
    info : dict
        Diagnostic info dict (counters, JIT compile times) populated
        on first call.

    Raises
    ------
    RuntimeError
        Without JAX or the ``aragog.jax`` module, for a supported ``core_bc_mode``.
    TypeError
        For ``scales`` that is not a ``NonDimScales``.
    UnsupportedCoreBCError
        For a ``core_bc_mode`` of ``aragog.config.boundary.CORE_BC_MODES`` with no JAX RHS
        ('bower2018', 'gradient').
    ValueError
        For a missing ``core_module_budget``, a ``core_bc_mode`` outside ``CORE_BC_MODES``,
        inputs of the wrong size or shape, a type 5 setup that
        ``aragog.jax.solver.check_surface_temperature`` refuses, and radionuclide arrays
        that ``aragog.jax.solver.check_radio_arrays`` refuses.

    Notes
    -----
    The RHS and Jacobian programs compile once per input form: the mode, the radiogenic
    switch, the shape and dtype of each traced leaf, and the values of the compile-time
    constants. Those are every leaf of ``phase_params`` and of ``eos_jax``, the budget leaves
    named in ``_BUDGET_CONSTANTS``, and every static field of the inputs, which the treedef
    holds: the EOS pressure and entropy grid axes, the boundary condition types and
    ``param_utbl``, the mesh flag ``top_limited``, the static phase fields
    ``matprop_smooth_width`` and ``rheology``, and the non-array budget attributes. The mesh
    and boundary arrays and every other budget leaf are traced, so a structure update with new
    mesh, boundary and core profile values compiles nothing; a new EOS table value compiles
    once.
    """
    if core_bc_mode == 'core_module' and core_module_budget is None:
        raise ValueError(
            "core_bc_mode='core_module' requires core_module_budget "
            '(the CoreEnergyBudget the solver built from its config); '
            'got None.'
        )
    if core_bc_mode not in _JAX_N_EXTRA:
        err = UnsupportedCoreBCError if core_bc_mode in CORE_BC_MODES else ValueError
        raise err(
            f'core_bc_mode={core_bc_mode!r} is not supported by the JAX CVODE factory. '
            f'Supported modes: {tuple(_JAX_N_EXTRA)}. Set ``use_jax_jacobian = false`` for '
            'the NumPy RHS; with it true, the other config modes use the NumPy RHS after a '
            'warning.'
        )

    try:
        import jax
        import jax.numpy as jnp

        import aragog.jax.solver
    except ImportError as exc:
        raise RuntimeError(
            'Option Z (JAX RHS + Jacobian) requires JAX and the '
            f'aragog.jax module. Original error: {exc}'
        ) from exc

    aragog.jax.solver.check_surface_temperature(
        eos_jax, phase_params, mesh_arrays, boundary_params
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
    n_extra = _JAX_N_EXTRA[core_bc_mode]
    shell = (
        getattr(core_module_budget, 'shell', None) if core_bc_mode == 'core_module' else None
    )
    n_extra += 0 if shell is None else shell.n_cells
    n_extra += 0 if lid_params is None else 2
    if getattr(boundary_params, 'lid', None) is not lid_params:
        import equinox as eqx

        boundary_params = eqx.tree_at(
            lambda b: b.lid, boundary_params, lid_params, is_leaf=lambda x: x is None
        )
    expected_size = heating_np.size + n_extra
    if scales.n != expected_size:
        msg = (
            f'state_scale length {scales.n} is incompatible '
            f'with core_bc_mode={core_bc_mode!r} and heating_array length '
            f'{heating_np.size}; expected {expected_size}.'
        )
        if lid_params is None and scales.n == expected_size + 2:
            raise LidFactoryMismatch(
                f'{msg} If the solver runs the sub-grid lid (2 more states), pass '
                'lid_params=solver.jax_lid_params() to build_jax_rhs_and_jacobian.'
            )
        if lid_params is not None and scales.n == expected_size - 2:
            raise LidFactoryMismatch(
                f'{msg} lid_params is given but the solver runs without the sub-grid lid; '
                'pass lid_params=None.'
            )
        raise ValueError(msg)

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
        try:
            shapes = [np.shape(a) for a in radio_isotope_params]
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Radionuclide arrays: no 5 arrays of numbers ({exc})') from exc
        if len(set(shapes)) > 1:
            raise ValueError(
                f'All radio_isotope_params arrays must have identical shapes, got {shapes}'
            )
        checked = aragog.jax.solver.check_radio_arrays(
            *radio_isotope_params, start_time=start_time
        )
        radio_arrays = checked if checked[0].size > 0 else ()
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
        from aragog.core import RA_CRIT_PHASE, check_ra_crit

        q_radio, ra_crit = core_module_q_radio, check_ra_crit(core_module_ra_crit_cmb)
        phase_law = ra_crit == RA_CRIT_PHASE
        # NaN carries 'eq. 16' (ra_crit None, also the base of the law by phase) through the
        # traced arguments; the last entry switches the law by phase on.
        data = data + (
            jnp.float64(q_radio),
            jnp.float64(jnp.nan if ra_crit is None or phase_law else ra_crit),
            jnp.float64(
                jnp.nan
                if core_module_surface_temperature is None
                else core_module_surface_temperature
            ),
            jnp.float64(1.0 if phase_law else 0.0),
        )

    budget = core_module_budget if core_bc_mode == 'core_module' else None
    data, consts = _strong(data), _strong((phase_params, eos_jax, budget))
    paths, treedef = jax.tree_util.tree_flatten_with_path((data, *consts))
    mask = tuple(_is_constant(p) for p, _ in paths)
    held = _ByValue(tuple(x for (_, x), m in zip(paths, mask, strict=True) if m))
    _bound_forms(core_bc_mode, use_radio, data, (held, consts))
    rhs_program, jac_program, powers_program, weights_program = _programs()

    # flattened once per build, on the device from the first call on, not at every call
    leaves = [x for (_, x), m in zip(paths, mask, strict=True) if not m]
    static_args = (core_bc_mode, use_radio, held, (treedef, mask))
    # after the bound, which can clear _HELD
    static_args = _HELD.setdefault(static_args, static_args)

    rhs_jit = functools.partial(rhs_program, *static_args)
    jac_jit = functools.partial(jac_program, *static_args)

    def first_call(kind, label, program, t_nd, y_nd):
        """Put the traced leaves on the device, run ``program`` once and log whether it
        compiled a program or reused one."""
        leaves[:] = [jnp.asarray(x) for x in leaves]
        before = _TRACE_COUNTERS[kind]
        out = program(t_nd, y_nd, *leaves)
        compiled = _TRACE_COUNTERS[kind] > before
        how = 'compiled for new inputs' if compiled else 'reused'
        logger.log(
            logging.INFO if compiled else logging.DEBUG,
            'JAX %s first call: program %s',
            label,
            how,
        )
        return out

    info = {
        'lid': lid_params is not None,
        'rhs_calls': 0,
        'jac_calls': 0,
        'first_rhs_compile_done': False,
        'first_jac_compile_done': False,
    }

    def rhs_fn(t_nd, y_nd, ydot_nd):
        """scikits.odes RHS function: fills ydot in-place."""
        try:
            t, y = float(t_nd), jnp.asarray(y_nd)
            if info['first_rhs_compile_done']:
                ydot_nd[:] = np.asarray(rhs_jit(t, y, *leaves))
            else:
                ydot_nd[:] = np.asarray(first_call('rhs', 'RHS', rhs_jit, t, y))
                info['first_rhs_compile_done'] = True
            info['rhs_calls'] += 1
            return 0
        except Exception as exc:
            logger.error('JAX RHS failed: %s', exc)
            return 1

    def jacfn(t_nd, y_nd, fy_nd, J, user_data=None):
        """scikits.odes Jacobian function: fills J in-place."""
        try:
            t, y = float(t_nd), jnp.asarray(y_nd)
            if info['first_jac_compile_done']:
                J[...] = np.asarray(jac_jit(t, y, *leaves))
            else:
                J[...] = np.asarray(first_call('jac', 'Jacobian', jac_jit, t, y))
                info['first_jac_compile_done'] = True
            info['jac_calls'] += 1
            return 0
        except Exception as exc:
            logger.error('JAX Jacobian failed: %s; CVODE will fall back to FD', exc)
            return 1

    from aragog.jax.solver import _PARTS_REGISTRY

    if core_bc_mode in _PARTS_REGISTRY:

        def step_powers(t_nodes, Y_nodes, aux):
            """Batched per-node energy powers [W] for accepted trajectory nodes.

            Parameters
            ----------
            t_nodes : array_like, shape (n,)
                Times at evaluation nodes [yr].
            Y_nodes : array_like, shape (dim, n)
                State at evaluation nodes in physical units (entropy [J/kg/K],
                plus the extra slots of energy_balance and core_module).
            aux : StepPowersAux
                Per-solve geometry and mass structure.

            Returns
            -------
            ndarray, shape (n, 7)
                Powers in the order:
                [-F_int*A_int, F_cmb*A_cmb, Q_radio, Q_tidal,
                 Q_radio_cons, Q_tidal_cons, residual]
            """
            t_arr = np.asarray(t_nodes, dtype=np.float64)
            if t_arr.ndim > 1:
                raise ValueError(f't_nodes must be 1D; got ndim={t_arr.ndim}')
            t_arr = t_arr.ravel()
            n = t_arr.size

            Y_arr = np.asarray(Y_nodes, dtype=np.float64)
            if Y_arr.ndim != 2:
                raise ValueError(f'Y_nodes must be 2D of shape (dim, n); got ndim={Y_arr.ndim}')
            if Y_arr.shape[1] != n:
                raise ValueError(
                    f'Y_nodes second dimension {Y_arr.shape[1]} does not match t_nodes length {n}'
                )
            if Y_arr.shape[0] != expected_size:
                raise ValueError(
                    f'Y_nodes first dimension {Y_arr.shape[0]} does not match '
                    f'expected dimension {expected_size}'
                )

            if n == 0:
                return np.empty((0, 7), dtype=np.float64)

            pad_len = -n % C
            t_pad = np.pad(t_arr, (0, pad_len), mode='edge')
            Y_pad = np.pad(Y_arr, ((0, 0), (0, pad_len)), mode='edge')
            chunks = [
                powers_program(
                    core_bc_mode,
                    use_radio,
                    t_pad[k : k + C],
                    Y_pad[:, k : k + C],
                    data,
                    aux,
                    consts,
                )
                for k in range(0, t_pad.size, C)
            ]
            return np.concatenate(chunks)[:n]

        rhs_fn.step_powers = step_powers

        if lid_params is not None:

            def lid_weights(t_nodes, Y_nodes):
                """Lid weight at each accepted node, shape (n,), states in physical units."""
                t_arr = np.asarray(t_nodes, dtype=np.float64).ravel()
                Y_arr = np.asarray(Y_nodes, dtype=np.float64)
                if Y_arr.shape != (expected_size, t_arr.size):
                    raise ValueError(
                        f'Y_nodes shape {Y_arr.shape}, expected ({expected_size}, {t_arr.size})'
                    )
                if t_arr.size == 0:
                    return np.empty(0, dtype=np.float64)
                pad_len = -t_arr.size % C
                t_pad = np.pad(t_arr, (0, pad_len), mode='edge')
                Y_pad = np.pad(Y_arr, ((0, 0), (0, pad_len)), mode='edge')
                chunks = [
                    np.asarray(
                        weights_program(
                            core_bc_mode,
                            use_radio,
                            t_pad[k : k + C],
                            Y_pad[:, k : k + C],
                            data,
                            consts,
                        )
                    )
                    for k in range(0, t_pad.size, C)
                ]
                return np.concatenate(chunks)[: t_arr.size]

            rhs_fn.lid_weights = lid_weights

    if getattr(getattr(phase_params, 'rheology', None), 'enabled', False) and hasattr(
        mesh_arrays, 'validate_for_rheology'
    ):
        mesh_arrays.validate_for_rheology()
    rhs_fn.phase_params = phase_params
    rhs_fn.rheology = getattr(phase_params, 'rheology', None)
    rhs_fn.lid = lid_params is not None
    rhs_fn.program_args = (data, consts)

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
