"""Smoke coverage for the ``core_bc='core_module'`` solver coupling.

The staged core-evolution budget joins the entropy solver as two appended
ODE states, ``[S, dSdr_cmb, T_core]``. The checks here are the coupling
contract: the state vector grows by two, the CMB flux is the state-derived
physical flux (not a conduction-only estimate), the reported core
temperature is the integrated boundary state rather than the basal node's
EOS read-off, the core-side energy booking closes against the budget's
effective capacity, and the legacy-capacity limit reproduces the
``energy_balance`` mantle trajectory on the same setup.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import entropy_eos_copy, needs_eos

pytestmark = [pytest.mark.smoke, needs_eos]

CORE_MODULE_PARAMS = {
    'rho_cen': 12500.0,
    'length_scale': 7200e3,
    'alpha': 1.35e-5,
    'c_p': 840.0,
    'melting_curve': 'iron',
    'light_element_fraction': 0.1,
    'depression': 1.2,
    'ds_fusion': 170.0,
    'icn_width': 10.0,
    'q_radio': 0.0,
}


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy()


def _driven_s_profile(n_stag: int):
    """Convectively unstable S(r) on the staggered nodes: hot below, cold above.

    A uniform isentrope gives dS/dr = 0 everywhere, so every boundary
    flux is near zero and any two flux closures agree trivially. The
    solve-level guards in this file need the CMB to actually carry
    heat, which requires a finite negative entropy gradient.
    """
    return np.linspace(2950.0, 2600.0, n_stag)


def _build(
    core_bc: str,
    shared_eos,
    core_module_params=None,
    n_nodes: int = 10,
    end_time: float = 1.0,
    solver_method: str = 'radau',
    s_init=None,
    use_jax_jacobian: bool = False,
):
    from aragog.parser import (
        Parameters,
        _BoundaryConditionsParameters,
        _EnergyParameters,
        _InitialConditionParameters,
        _MeshParameters,
        _PhaseMixedParameters,
        _PhaseParameters,
        _SolverParameters,
    )
    from aragog.solver import EntropySolver

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=1,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc=core_bc,
        core_module_params=core_module_params,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method=solver_method,
        use_jax_jacobian=use_jax_jacobian,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(
        start_time=0.0,
        end_time=end_time,
        # CVODE at the Aragog default 1e-8; a looser rtol can lock it at dS/dr = 0.
        atol=1.0e-8 if solver_method == 'cvode' else 1.0e-6,
        rtol=1.0e-8 if solver_method == 'cvode' else 1.0e-6,
        tsurf_poststep_change=30.0,
    )
    params = Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        solver=sv,
    )
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()
    if s_init is None:
        solver.set_initial_entropy(2700.0)
    elif isinstance(s_init, str) and s_init == 'driven':
        solver.set_initial_entropy(_driven_s_profile(solver._n_stag))
    else:
        solver.set_initial_entropy(s_init)
    return solver


def test_core_module_state_extension_and_integrated_t_core(shared_eos):
    """The core_module solve carries the two extra states, reports the
    integrated T_core (finite, near the EOS-derived start), keeps the
    boundary-gradient state finite, and the budget object was built with
    the mesh's own CMB radius."""
    solver = _build('core_module', shared_eos, CORE_MODULE_PARAMS, s_init='driven')
    assert solver._state_is_extended
    n_stag = solver._n_stag
    assert len(solver._S0) == n_stag + 2  # entropy block, dSdr_cmb, T_cmb
    assert solver._core_module_budget.profiles.r_cmb == pytest.approx(3.480e6)
    solver.solve()
    out = solver.get_state()
    y = solver._solution.y
    assert y.shape[0] == n_stag + 2
    dsdr_path = y[n_stag]
    t_core_path = y[n_stag + 1]
    assert np.all(np.isfinite(dsdr_path))
    assert np.all(np.isfinite(t_core_path))
    # Reported T_core is the ODE state's endpoint, not the basal node.
    assert out.T_core == pytest.approx(float(t_core_path[-1]), rel=1e-12)
    # Physical bounds: positive, and within a few hundred K of the start
    # over one year of integration.
    assert 0.0 < out.T_core
    assert abs(float(t_core_path[-1]) - float(t_core_path[0])) < 300.0
    # Sub-step smoothness: no single internal jump exceeds 50 K over this
    # short, transient-free integration.
    if t_core_path.size > 2:
        assert np.max(np.abs(np.diff(t_core_path))) < 50.0


def test_core_module_core_cools_through_the_state_derived_flux(shared_eos):
    """The core actually loses heat through the mantle-side transport,
    verified against a quantity the core equation never sees.

    Three guards on a driven (convectively unstable) profile:

    1. Flux continuity: the state-derived CMB flux sits within a factor
       of the mantle transport one node above (measured ratio 1.14; a
       conduction-only closure sits five orders below the node-1 flux).
    2. Independent cooling magnitude: the core temperature change
       matches the prediction built from the NODE-1 flux, which does
       not enter the core equation, so this cannot be satisfied by the
       internal consistency of the T_core ODE alone. An insulated core
       fails this by three-plus orders of magnitude.
    3. Booking consistency: the trapezoid-booked CMB energy equals
       C_eff times the temperature change (guards the energy
       diagnostics, not the flux law, which guards 1 and 2 carry).

    The uniform-isentrope variant of this test is worthless: with
    dS/dr = 0 every closure produces near-zero flux and they all agree
    trivially, which is why the driven profile is load-carrying here.
    """
    solver = _build(
        'core_module', shared_eos, CORE_MODULE_PARAMS, end_time=5.0, s_init='driven'
    )
    solver.solve()
    out = solver.get_state()
    y = solver._solution.y
    n_stag = solver._n_stag
    t0, t1 = float(y[n_stag + 1, 0]), float(y[n_stag + 1, -1])

    # Guard 1: flux continuity across the two lowest basic nodes.
    F_node1 = float(out.heat_flux[1])
    assert abs(F_node1) > 1.0  # the driven profile must actually drive
    assert 0.2 < abs(out.F_cmb) / abs(F_node1) < 5.0

    # Guard 2: independent cooling prediction from the node-1 flux.
    c_eff = float(solver._core_module_budget.effective_capacity(0.5 * (t0 + t1)))
    span_s = float(out.dt_actual) * 3.15576e7
    area = 4.0 * np.pi * 3.480e6**2
    dT_pred = -abs(F_node1) * area * span_s / c_eff
    assert t1 - t0 < 0.0  # outgoing flux cools the core (second law)
    assert 0.2 < (t1 - t0) / dT_pred < 5.0
    # Absolute insulation catch: the measured change is far above the
    # sub-microkelvin an insulated core produces on this window.
    assert abs(t1 - t0) > 1.0e-4

    # Guard 3: booking consistency.
    dE_from_T = -(t1 - t0) * c_eff
    dE_booked = float(out.step_dE_F_cmb_J)
    assert dE_booked > 0.0
    assert dE_from_T == pytest.approx(dE_booked, rel=0.10)
    assert out.step_dE_core_J < 0.0
    assert abs(out.step_dE_core_J + out.step_dE_F_cmb_J) / out.step_dE_F_cmb_J < 0.05


def test_core_module_legacy_capacity_matches_energy_balance(shared_eos):
    """With the budget in legacy capacity mode wired to the reservoir
    constants (rho = mesh core_density, cp = bc core_heat_capacity,
    tfac = 1.147), the mantle trajectory reproduces energy_balance: the
    boundary-gradient equations are then algebraically identical, and
    T_core is a passive record of the reservoir drain. Discriminates a
    silent flux-path divergence between the two modes."""
    legacy_capacity_params = {
        'c_p': 880.0,  # = bc core_heat_capacity in _build
        'melting_curve': 'iron',
        'light_element_fraction': 0.1,
        'depression': 1.2,
        'capacity_mode': 'legacy',
        'legacy_rho_core': 10500.0,  # = mesh core_density in _build
        'legacy_tfac': 1.147,  # solver default tfac_core_avg
    }
    eb = _build('energy_balance', shared_eos, None, end_time=2.0, s_init='driven')
    eb.solve()
    S_eb = eb._solution.y[: eb._n_stag, -1]
    dsdr_eb = float(eb._solution.y[eb._n_stag, -1])

    cm = _build(
        'core_module', shared_eos, legacy_capacity_params, end_time=2.0, s_init='driven'
    )
    cm.solve()
    S_cm = cm._solution.y[: cm._n_stag, -1]
    dsdr_cm = float(cm._solution.y[cm._n_stag, -1])

    # Same S trajectory to solver tolerance (rtol 1e-6 solves; the
    # extra passive state perturbs step selection slightly).
    np.testing.assert_allclose(S_cm, S_eb, rtol=1e-4)
    assert dsdr_cm == pytest.approx(dsdr_eb, rel=1e-2, abs=1e-9)


def test_core_module_against_quasi_steady_baseline(shared_eos):
    """Cross-mode sanity on the same driven setup: both core temperatures
    are finite, the module's integrated state cools under the outgoing
    flux, and it sits within 5 K of the quasi_steady CMB basic node. The
    quasi_steady T_core is read at the bottom staggered cell, half a cell
    above the CMB node, so the two differ by 86 K on this mesh; the 100 K
    bracket only catches catastrophic divergence (initialisation or unit
    errors), and the closure discrimination lives in the flux-continuity test."""
    legacy = _build('quasi_steady', shared_eos, s_init='driven')
    legacy.solve()
    t_legacy = legacy.get_state().T_core

    module = _build('core_module', shared_eos, CORE_MODULE_PARAMS, s_init='driven')
    module.solve()
    out = module.get_state()
    y = module._solution.y
    n_stag = module._n_stag
    t_module = out.T_core

    assert np.isfinite(t_legacy) and np.isfinite(t_module)
    # Outgoing driven flux: the integrated core state must cool.
    assert float(y[n_stag + 1, -1]) < float(y[n_stag + 1, 0])
    t_legacy_cmb = float(legacy.state.phase_basic.temperature()[0])
    assert abs(t_module - t_legacy_cmb) < 5.0
    assert abs(t_module - t_legacy) < 100.0


def test_core_module_missing_params_still_builds_with_defaults(shared_eos):
    """An empty params dict builds the budget entirely from defaults plus
    the mesh geometry and the EOS-derived CMB pressure fallback; the
    solve still runs. A wrong key fails at solver construction with the
    factory's message."""
    solver = _build('core_module', shared_eos, {})
    assert float(solver._core_module_budget.profiles.p_cmb) > 0.0
    solver.solve()
    assert np.isfinite(solver.get_state().T_core)

    with pytest.raises(ValueError, match='unrecognised'):
        _build('core_module', shared_eos, {'not_a_key': 1.0})


def test_core_module_solves_through_cvode(shared_eos):
    """core_module completes a driven solve through the CVODE production
    integrator (FD Jacobian; no analytic-Jacobian factory is registered
    here) and lands on the Radau twin's answer. Guards the
    production path PROTEUS actually runs, which the scipy-only tests
    never touch, including the N+2 sparsity and nondim scales under
    CVODE."""
    pytest.importorskip('scikits_odes_sundials')
    cv = _build(
        'core_module',
        shared_eos,
        CORE_MODULE_PARAMS,
        end_time=2.0,
        solver_method='cvode',
        s_init='driven',
    )
    cv.solve()
    out_cv = cv.get_state()
    y_cv = cv._solution.y
    n_stag = cv._n_stag
    assert y_cv.shape[0] == n_stag + 2
    assert np.all(np.isfinite(y_cv))

    rd = _build(
        'core_module',
        shared_eos,
        CORE_MODULE_PARAMS,
        end_time=2.0,
        solver_method='radau',
        s_init='driven',
    )
    rd.solve()
    out_rd = rd.get_state()
    # Same physics through both integrators: fluxes to 1%, the core
    # temperature drop to 10% (both integrate the same smooth ODE; the
    # bands absorb step-selection differences only).
    assert out_cv.F_cmb == pytest.approx(out_rd.F_cmb, rel=1e-2)
    dT_cv = float(y_cv[n_stag + 1, -1] - y_cv[n_stag + 1, 0])
    dT_rd = float(rd._solution.y[n_stag + 1, -1] - rd._solution.y[n_stag + 1, 0])
    assert dT_cv == pytest.approx(dT_rd, rel=0.10)
    assert dT_cv < 0.0


def test_core_module_solve_with_nucleation_active(shared_eos):
    """A driven solve started inside the inner-core nucleation band
    exercises the latent and gravitational capacity terms in the coupled
    ODE (not just on the standalone budget): C_eff exceeds the secular
    capacity at the initial state, the FD-Jacobian solve completes, and
    the T_core path stays finite and smooth through the band."""
    probe = _build('core_module', shared_eos, CORE_MODULE_PARAMS)
    budget = probe._core_module_budget
    secular = float(budget.secular_capacity())
    # Scan for the nucleation-active band: where the latent term
    # contributes at least 1% of the secular capacity.
    T_scan = np.linspace(3200.0, 6000.0, 281)
    active = [t for t in T_scan if float(budget.latent_capacity(t)) > 0.01 * secular]
    assert active, 'no nucleation-active band in scan range; params drifted'
    t_start = float(np.median(active))

    solver = _build(
        'core_module', shared_eos, CORE_MODULE_PARAMS, end_time=2.0, s_init='driven'
    )
    solver.set_initial_core_temperature(t_start)
    solver.set_initial_entropy(_driven_s_profile(solver._n_stag))
    assert float(budget.effective_capacity(t_start)) > 1.01 * secular
    solver.solve()
    y = solver._solution.y
    n_stag = solver._n_stag
    t_path = y[n_stag + 1]
    assert np.all(np.isfinite(t_path))
    assert np.all(t_path > 0.0)
    if t_path.size > 2:
        # Latent buffering makes the effective capacity LARGER, so the
        # per-step motion must stay below the capacity-free estimate.
        assert np.max(np.abs(np.diff(t_path))) < 1.0


def test_core_module_solves_through_cvode_with_jax_jacobian(shared_eos):
    """Option Z end-to-end for core_module: the registered JAX factory is
    consumed (not silently bypassed for the FD fallback), the solve
    completes, and the trajectory lands on the FD twin's answer. This is
    the production path that removes the FD-Jacobian wall-clock penalty;
    a factory rejection would still pass a bare completes-check via the
    fallback, which is why the call counter is asserted."""
    pytest.importorskip('scikits_odes_sundials')
    pytest.importorskip('equinox')
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_jax_dsdt_core_module import _build_jax_pieces

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    zsolver = _build(
        'core_module',
        shared_eos,
        CORE_MODULE_PARAMS,
        end_time=2.0,
        solver_method='cvode',
        s_init='driven',
        use_jax_jacobian=True,
    )
    args = _build_jax_pieces(zsolver)
    calls = {'n': 0}

    def factory(scales, core_bc_mode):
        calls['n'] += 1
        rhs_fn, jac_fn, _info = build_jax_rhs_and_jacobian(
            eos_jax=args[0],
            phase_params=args[1],
            mesh_arrays=args[2],
            boundary_params=args[3],
            heating_array=np.zeros(zsolver._n_stag),
            scales=scales,
            core_bc_mode=core_bc_mode,
            core_module_budget=args[6],
            core_module_q_radio=args[7],
        )
        return rhs_fn, jac_fn

    zsolver.set_jax_cvode_factory(factory)
    zsolver.solve()
    assert calls['n'] > 0, 'the JAX factory was never consumed; Option Z did not engage'
    sol = zsolver._solution
    assert sol.status == 0
    n_stag = zsolver._n_stag
    y_z = sol.y[:, -1]
    assert np.all(np.isfinite(y_z))

    fd = _build(
        'core_module',
        shared_eos,
        CORE_MODULE_PARAMS,
        end_time=2.0,
        solver_method='cvode',
        s_init='driven',
        use_jax_jacobian=False,
    )
    fd.solve()
    y_fd = fd._solution.y[:, -1]
    # Same physics through both Jacobian paths. The interior parity
    # carries the pre-existing numpy-vs-JAX RHS difference (~4e-4 at
    # two mid-mantle nodes), so the trajectories agree at that level,
    # not at integrator tolerance.
    np.testing.assert_allclose(y_z[:n_stag], y_fd[:n_stag], rtol=5e-3)
    dT_z = float(y_z[n_stag + 1] - sol.y[n_stag + 1, 0])
    dT_fd = float(y_fd[n_stag + 1] - fd._solution.y[n_stag + 1, 0])
    assert dT_z < 0.0  # the core cools under the driven flux
    assert dT_z == pytest.approx(dT_fd, rel=0.05)


def test_nucleation_temperature_independent_of_mesh_resolution(shared_eos):
    """Core profile and initial core temperature anchor at the CMB basic node.

    Asserts that the core hydrostatic pressure and inner-core nucleation
    temperature do not vary with mantle mesh resolution, and that default
    initial core temperature matches the CMB basic node temperature.
    """
    s10 = _build('core_module', shared_eos, CORE_MODULE_PARAMS, n_nodes=10)
    s10.initialize()
    s60 = _build('core_module', shared_eos, CORE_MODULE_PARAMS, n_nodes=60)
    s60.initialize()

    b10 = s10._core_module_budget
    b60 = s60._core_module_budget

    p10 = b10.profiles
    p60 = b60.profiles

    t_nuc_10 = float(b10.melting_curve.t_melt(p10.pressure(0.0)) / p10.adiabat(0.0, 1.0))
    t_nuc_60 = float(b60.melting_curve.t_melt(p60.pressure(0.0)) / p60.adiabat(0.0, 1.0))

    assert abs(t_nuc_10 - t_nuc_60) < 1.0

    s_init_10 = np.linspace(2950.0, 2600.0, s10._n_stag)
    s_init_60 = np.linspace(2950.0, 2600.0, s60._n_stag)
    s10.set_initial_entropy(s_init_10)
    s60.set_initial_entropy(s_init_60)

    t_core_10 = float(s10._S0[s10._n_stag + 1])
    t_core_60 = float(s60._S0[s60._n_stag + 1])

    s10.state.update(s_init_10, 0.0, dSdr_cmb=float(s10._S0[s10._n_stag]))
    s60.state.update(s_init_60, 0.0, dSdr_cmb=float(s60._S0[s60._n_stag]))

    t_cmb_node_10 = float(s10.state.phase_basic.temperature()[0])
    t_cmb_node_60 = float(s60.state.phase_basic.temperature()[0])

    assert t_core_10 == pytest.approx(t_cmb_node_10, abs=1e-6)
    assert t_core_60 == pytest.approx(t_cmb_node_60, abs=1e-6)
