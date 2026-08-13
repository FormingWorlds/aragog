"""Tests for ``aragog.jax.solver.dSdt_core_module`` and its factory branch.

The core_module JAX RHS (extended state ``[S, dSdr_cmb, T_core]`` of
length N+2) mirrors the numpy ``EntropySolver._dSdt_single`` closure for
``core_bc='core_module'``: the energy_balance boundary-gradient balance
with the staged core-evolution budget's effective heat capacity in place
of the isothermal-reservoir factor. The contract clauses exercised here:

- the factory refuses the mode without a budget and enforces the N+2
  nondim-scale shape;
- the JAX RHS reproduces the numpy RHS on a real-EOS driven state to
  integrator precision, component by component, including both boundary
  slots (the check that keeps CVODE's Newton iteration consistent);
- the analytic Jacobian is finite and carries the boundary couplings
  the FD path resolves (the custom JVP through the inner-core bisection
  must survive ``jacrev``).

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)

# Module tier: the real-EOS parity and Jacobian solves are smoke; the
# two factory-contract tests additionally carry the unit marker so the
# PR lane still runs them (they need no external data).
pytestmark = [pytest.mark.smoke, pytest.mark.timeout(300)]


def _tiny_budget(r_cmb: float = 3.48e6):
    from aragog.core import build_core_module_budget

    return build_core_module_budget(
        {'melting_curve': 'iron', 'light_element_fraction': 0.1, 'depression': 1.2},
        r_cmb=r_cmb,
        p_cmb_fallback=135e9,
    )


@pytest.mark.unit
def test_factory_requires_budget_for_core_module():
    """The factory refuses core_bc_mode='core_module' without a budget:
    a silent None would surface later as an AttributeError inside the
    first jitted RHS call, after CVODE is already wired up."""
    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    n = 4
    scales = NonDimScales(state_scale=np.full(n + 2, 1.0), t_ref=1.0)
    with pytest.raises(ValueError, match='requires core_module_budget'):
        build_jax_rhs_and_jacobian(
            eos_jax=None,
            phase_params=None,
            mesh_arrays=None,
            boundary_params=None,
            heating_array=np.zeros(n),
            scales=scales,
            core_bc_mode='core_module',
        )


@pytest.mark.unit
def test_factory_shape_contract_core_module():
    """core_module expects N+4 nondim scales: N+3 (the energy_balance
    length, the most plausible off-by-one) raises the incompatibility
    error, and N+4 with a budget builds callable rhs/jac functions."""
    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    n = 4
    budget = _tiny_budget()
    bad_scales = NonDimScales(state_scale=np.full(n + 3, 1.0), t_ref=1.0)
    with pytest.raises(ValueError, match='incompatible'):
        build_jax_rhs_and_jacobian(
            eos_jax=None,
            phase_params=None,
            mesh_arrays=None,
            boundary_params=None,
            heating_array=np.zeros(n),
            scales=bad_scales,
            core_bc_mode='core_module',
            core_module_budget=budget,
        )

    good_scales = NonDimScales(state_scale=np.full(n + 4, 1.0), t_ref=1.0)
    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
        eos_jax=None,
        phase_params=None,
        mesh_arrays=None,
        boundary_params=None,
        heating_array=np.zeros(n),
        scales=good_scales,
        core_bc_mode='core_module',
        core_module_budget=budget,
    )
    assert callable(rhs_fn) and callable(jac_fn)
    assert info['rhs_calls'] == 0


def _build_numpy_solver(shared_eos):
    """The driven core_module solver from the smoke harness, radau-free.

    Kept in sync with tests/test_entropy_solver_core_module_smoke.py's
    ``_build`` so the parity state matches a configuration the solve
    tests actually integrate.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

    return _build('core_module', shared_eos, CORE_MODULE_PARAMS, s_init='driven')


def _build_jax_pieces(solver):
    """JAX pytrees mirroring the numpy solver's mesh, phases, and BC."""
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams, _no_radio

    eos_jax = EntropyEOS_JAX(EOS_DIR)
    mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
    pm = solver.parameters.phase_mixed
    pl = solver.parameters.phase_liquid
    ps = solver.parameters.phase_solid
    en = solver.parameters.energy
    params_jax = PhaseParams(
        phi_rheo=float(pm.rheological_transition_melt_fraction),
        phi_width=float(pm.rheological_transition_width),
        viscosity_solid=float(ps.viscosity),
        viscosity_liquid=float(pl.viscosity),
        grain_size=float(pm.grain_size),
        k_solid=float(ps.thermal_conductivity),
        k_liquid=float(pl.thermal_conductivity),
        matprop_smooth_width=float(getattr(pm, 'matprop_smooth_width', 0.0)),
        conduction=bool(en.conduction),
        convection=bool(en.convection),
        grav_sep=bool(en.gravitational_separation),
        mixing=bool(en.mixing),
        eddy_diff_thermal=float(en.eddy_diffusivity_thermal),
        eddy_diff_chemical=float(en.eddy_diffusivity_chemical),
        kappah_floor=float(en.kappah_floor),
        phase_smoothing=str(getattr(pm, 'phase_smoothing', 'tanh')),
        phase_smoothing_width=float(pm.phase_transition_width),
    )
    bc_cfg = solver.parameters.boundary_conditions
    bc_jax = BoundaryParams(
        outer_bc_type=1,  # grey body, as in the numpy harness
        outer_bc_value=0.0,
        emissivity=float(bc_cfg.emissivity),
        T_eq=float(bc_cfg.equilibrium_temperature),
        inner_bc_type=1,  # unused by dSdt_core_module; flux is state-derived
        inner_bc_value=0.0,
        core_density=float(solver._core_density),
        core_heat_capacity=float(solver._core_cp),
        tfac_core_avg=float(solver._core_tfac),
        cmb_area=float(solver._cmb_area),
        core_M=float(solver._core_M),
        cmb_dr_cmb=float(solver._cmb_dr_cmb),
    )
    n_stag = solver._n_stag
    args = (
        eos_jax,
        params_jax,
        mesh_jax,
        bc_jax,
        jnp.zeros(n_stag),
        _no_radio,
        solver._core_module_budget,
        float(solver._core_module_q_radio),
    )
    return args


@needs_eos
def test_rhs_parity_with_numpy_on_driven_state():
    """The JAX RHS matches the numpy RHS component-by-component on the
    driven real-EOS state, including the dSdr_cmb and T_core slots.

    This is the contract that keeps CVODE's Newton iteration coherent
    when the analytic Jacobian is active. The comparison runs at the
    cold-start state AND at a perturbed state (dSdr_cmb doubled, T_core
    +150 K into the nucleation band) so agreement is not an artifact of
    the FD cold start's near-zero gradient; the perturbed T_core also
    drags C_eff away from its secular value, exercising the capacity
    swap on both sides.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.solver import dSdt_core_module

    solver = _build_numpy_solver(EntropyEOS(EOS_DIR))
    args = _build_jax_pieces(solver)
    n_stag = solver._n_stag

    y0 = np.asarray(solver._S0, dtype=float)
    states = [y0]
    y1 = y0.copy()
    y1[n_stag] *= 2.0
    y1[n_stag + 1] += 150.0
    states.append(y1)

    for k, y in enumerate(states):
        f_np = np.asarray(solver.dSdt(0.0, y)).ravel()
        f_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y), args)).ravel()
        assert f_np.shape == f_jax.shape == (n_stag + 2,)
        denom = np.maximum(np.maximum(np.abs(f_np), np.abs(f_jax)), 1e-12)
        rel = np.abs(f_np - f_jax) / denom
        # The two boundary slots are the physics this mode adds; they
        # must match to integrator precision.
        assert rel[n_stag] < 1e-8 and rel[n_stag + 1] < 1e-8, (
            f'state {k}: boundary-slot parity {rel[n_stag]:.3e} / {rel[n_stag + 1]:.3e}'
        )
        # Interior entropy nodes: 4e-4 at two mid-mantle nodes is the
        # pre-existing numpy-vs-JAX difference, measured identically on
        # the production energy_balance RHS at this exact state, so the
        # bound documents the shared flux assembly rather than this
        # mode. Tightening it below 1e-3 requires fixing that shared
        # difference first.
        assert rel.max() < 1e-3, (
            f'state {k}: max rel err {rel.max():.3e} at component {rel.argmax()} '
            f'(numpy {f_np[rel.argmax()]:.6e} vs jax {f_jax[rel.argmax()]:.6e})'
        )
        # The boundary slots must be active, not vacuously matching zeros:
        # the driven profile cools the core through a real flux.
        assert f_np[n_stag + 1] < 0.0

    # q_radio path: both sides receive the same nonzero core source and
    # must still agree on the boundary slots; the offset must shift the
    # cooling rate in the warming direction by a resolvable amount
    # (edge case: a source comparable to the CMB heat flow).
    f_jax_base = np.asarray(dSdt_core_module(0.0, jnp.asarray(y0), args)).ravel()
    q_radio = 5.0e12
    solver._core_module_q_radio = q_radio
    args_r = args[:7] + (q_radio,)
    try:
        f_np_r = np.asarray(solver.dSdt(0.0, y0)).ravel()
    finally:
        solver._core_module_q_radio = 0.0
    f_jax_r = np.asarray(dSdt_core_module(0.0, jnp.asarray(y0), args_r)).ravel()
    for slot in (n_stag, n_stag + 1):
        denom = max(abs(f_np_r[slot]), abs(f_jax_r[slot]), 1e-12)
        assert abs(f_np_r[slot] - f_jax_r[slot]) / denom < 1e-8
    assert f_jax_r[n_stag + 1] > f_jax_base[n_stag + 1], (
        'a positive core source must warm the cooling rate'
    )


@needs_eos
def test_jacobian_carries_boundary_couplings():
    """``jacrev`` through the RHS (budget custom-JVP included) yields a
    finite Jacobian whose T_core row and column carry the couplings the
    sparsity pattern promises: dT_core_dt responds to dSdr_cmb (through
    the flux) and the basal entropy responds to T_core (through the
    boundary balance). A zero cross-coupling here means the custom JVP
    was lost and CVODE's Newton would iterate on wrong derivatives.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.solver import dSdt_core_module

    solver = _build_numpy_solver(EntropyEOS(EOS_DIR))
    args = _build_jax_pieces(solver)
    n_stag = solver._n_stag
    y0 = np.asarray(solver._S0, dtype=float)

    # Pin T_core inside the nucleation-active band, found by scanning the
    # budget itself: outside the band the latent term is exactly zero and
    # d(C_eff)/dT_core is a TRUE zero, so the self-coupling assertion
    # below would fail regardless of the JVP rule. The EOS tables set the
    # default T_core, so without this pin the test's premise depends on
    # which table cache the environment resolves.
    budget = solver._core_module_budget
    secular = float(budget.secular_capacity())
    scan = np.linspace(3200.0, 6000.0, 281)
    active = [t for t in scan if float(budget.latent_capacity(t)) > 0.01 * secular]
    assert active, 'no nucleation-active band in scan range; params drifted'
    y0[n_stag + 1] = float(np.median(active))
    y0 = jnp.asarray(y0)

    J = np.asarray(jax.jacrev(lambda y: dSdt_core_module(0.0, y, args))(y0))
    assert J.shape == (n_stag + 2, n_stag + 2)
    assert np.all(np.isfinite(J))
    # dT_core/dt depends on the flux, which depends on the gradient state.
    assert abs(J[n_stag + 1, n_stag]) > 0.0
    # dT_core/dt depends on T_core through C_eff(T_core): this element
    # is nonzero only because reverse-mode survives the budget's custom
    # JVP through the inner-core bisection; a lost rule reads zero here.
    assert abs(J[n_stag + 1, n_stag + 1]) > 0.0
    # The gradient equation couples to itself (flux feedback) and to
    # T_core (the boundary balance rides on the cooling rate).
    assert abs(J[n_stag, n_stag]) > 0.0
    assert abs(J[n_stag, n_stag + 1]) > 0.0
    # The basal ENTROPY equation does NOT couple to T_core directly:
    # the core reaches the mantle only through the integrated gradient
    # state, never instantaneously. The sparsity pattern is a superset;
    # this pins the physics.
    assert J[0, n_stag + 1] == 0.0
    # ... while the boundary-gradient state drives it strongly.
    assert abs(J[0, n_stag]) > 0.0


@needs_eos
def test_stratified_budget_parity_and_jacobian_through_the_full_rhs():
    """The stratified reduction reaches the coupled RHS on both paths:
    with stratification on and a subadiabatic state (the uniform
    isentrope drives a near-zero flux, deeply subadiabatic for k=130),
    numpy and JAX still agree on the boundary slots to integrator
    precision, the cooling rate differs from the unstratified twin by
    the reduced thermal inertia, and ``jacrev`` through the full RHS
    stays finite with the thickness solve's sensitivity composed inside
    (the regime where a lost or mis-signed layer JVP would corrupt the
    analytic Jacobian without failing any standalone gradient test)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.solver import dSdt_core_module

    eos = EntropyEOS(EOS_DIR)
    strat_params = dict(CORE_MODULE_PARAMS) | {'stratification': True, 'k_core': 130.0}
    strat = _build('core_module', eos, strat_params)  # uniform isentrope
    plain = _build('core_module', eos, CORE_MODULE_PARAMS)
    n_stag = strat._n_stag
    y0 = np.asarray(strat._S0, dtype=float)

    f_np = np.asarray(strat.dSdt(0.0, y0)).ravel()
    args = _build_jax_pieces(strat)
    f_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y0), args)).ravel()
    for slot in (n_stag, n_stag + 1):
        denom = max(abs(f_np[slot]), abs(f_jax[slot]), 1e-15)
        assert abs(f_np[slot] - f_jax[slot]) / denom < 1e-8

    # The reduced convecting volume amplifies the cooling response: the
    # stratified T_core rate exceeds the unstratified twin's at the same
    # state by well over the parity tolerance.
    f_plain = np.asarray(plain.dSdt(0.0, np.asarray(plain._S0, dtype=float))).ravel()
    assert abs(f_np[n_stag + 1]) > 2.0 * abs(f_plain[n_stag + 1])

    J = np.asarray(jax.jacrev(lambda y: dSdt_core_module(0.0, y, args))(jnp.asarray(y0)))
    assert J.shape == (n_stag + 2, n_stag + 2)
    assert np.all(np.isfinite(J))
