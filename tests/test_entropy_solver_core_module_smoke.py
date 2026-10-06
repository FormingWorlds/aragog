"""Smoke coverage for the ``core_bc='core_module'`` solver coupling.

The staged core-evolution budget joins the entropy solver as two appended
ODE states, ``[S, dSdr_cmb, T_core]``. The checks here are the coupling
contract: the state vector grows by two, the CMB flux is the boundary-layer
law of T_core against the bottom cell, the reported core temperature is the
integrated boundary state rather than the basal node's EOS read-off, and the
heat the core loses is the heat booked into the mantle and the budget's own
content change.
"""

from __future__ import annotations

import logging

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
    core_offset: float | None = None,
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
    if core_offset is not None:
        # Start the core core_offset above the bottom cell at the CMB pressure.
        S = np.array(solver._S0[: solver._n_stag])
        t_m = float(np.asarray(shared_eos.temperature(solver._P_basic_flat[:1], S[:1])).item())
        solver.set_initial_core_temperature(t_m + core_offset)
        solver.set_initial_entropy(S)
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


@pytest.mark.parametrize(
    ('d_core', 'regime'),
    [(1000.0, 'boundary_layer'), (-1000.0, 'conduction')],
    ids=['hot_core_boundary_layer', 'cold_core_conduction'],
)
def test_core_module_cmb_flux_follows_the_core_mantle_contrast(shared_eos, d_core, regime):
    """The solve applies the boundary-layer CMB flux of T_core against the mantle.

    1. Wiring: the end-of-step CMB flux equals ``cmb_boundary_layer_flux`` rebuilt
       from the end state alone (T_core, the bottom entropy at the CMB pressure, the
       bottom cell's properties), so the RHS, the output and the law agree; the hot
       core is in the boundary-layer branch (above conduction across the half cell),
       the cold core on the conduction branch. The mantle-gradient flux fails both.
    2. Second law over the solve: a core 1000 K hotter than the mantle loses heat
       (positive flux and booked CMB energy), a core 1000 K colder gains it.

    The driven profile's base is mushy (phi 0.19), so the flux is small (about 0.3 W/m^2
    for the hot core) and the core temperature change over the window is below the
    integrator tolerance; the sign of the booked flux integral is the solve-level check.
    """
    from aragog.core import cmb_boundary_layer_flux

    solver = _build('core_module', shared_eos, CORE_MODULE_PARAMS, end_time=5.0)
    S = _driven_s_profile(solver._n_stag)
    p_cmb = float(solver._P_basic_flat[0])
    t_m0 = float(np.asarray(shared_eos.temperature(np.array([p_cmb]), S[:1])).item())
    solver.set_initial_core_temperature(t_m0 + d_core)
    solver.set_initial_entropy(S)
    solver.solve()
    out = solver.get_state()
    y_end = solver._solution.y[:, -1]
    n_stag = solver._n_stag

    solver.dSdt(float(solver._solution.t[-1]), y_end)
    ph = solver.state.phase_staggered

    def first(values):
        return float(np.asarray(values).flat[0])

    t_core, k = float(y_end[n_stag + 1]), first(ph.thermal_conductivity())
    t_m = first(shared_eos.temperature(np.array([p_cmb]), y_end[:1]))
    dr_half = 0.5 * float(solver._r_basic_flat[1] - solver._r_basic_flat[0])
    expected = float(
        cmb_boundary_layer_flux(
            t_core,
            t_m,
            conductivity=k,
            density=first(ph.density()),
            heat_capacity=first(shared_eos.heat_capacity(solver._P_stag_flat[:1], y_end[:1])),
            expansivity=first(ph.thermal_expansivity()),
            viscosity=first(ph.viscosity()),
            gravity=first(solver.state.phase_basic.gravitational_acceleration()),
            dr_half=dr_half,
        )
    )
    assert float(out.heat_flux[0]) == pytest.approx(expected, rel=1e-10)
    q_cond = k * (t_core - t_m) / dr_half
    if regime == 'conduction':
        assert expected == pytest.approx(q_cond, rel=1e-12)
    else:
        assert expected > 2.0 * q_cond
    # Node 0 of the output is the bottom cell at the CMB pressure, carrying the applied flux
    # as conduction; the gradient slot does not enter it.
    assert float(out.T_basic[0]) == pytest.approx(t_m, rel=1e-10)
    assert float(out.jcond_b[0]) == pytest.approx(expected, rel=1e-10)
    assert [out.jconv_b[0], out.jgrav_b[0], out.jmix_b[0]] == [0.0, 0.0, 0.0]
    solver._solution.y[n_stag, -1] = 1.5 * y_end[n_stag] + 1e-4
    again = solver.get_state()
    for name in ('T_basic', 'phi_basic', 'cp_basic', 'rho_basic', 'porosity_b', 'dSdr_b'):
        assert float(getattr(again, name)[0]) == pytest.approx(
            float(getattr(out, name)[0]), rel=1e-12
        )
    assert np.sign(out.F_cmb) == np.sign(d_core)
    assert np.sign(out.step_dE_F_cmb_J) == np.sign(d_core)
    if regime == 'boundary_layer':
        phi = float(shared_eos.melt_fraction(np.array([p_cmb]), y_end[:1])[0])
        assert phi == pytest.approx(0.19, abs=0.01)
        assert 0.1 < out.F_cmb < 1.0


def test_energy_balance_output_keeps_the_gradient_node_diagnostics(shared_eos):
    """Outside core_module the CMB node of the output is the state's own diagnostic: the
    temperature of the gradient-extrapolated basic node and the flux components the
    state computed."""
    solver = _build('energy_balance', shared_eos, s_init='driven')
    solver.solve()
    out = solver.get_state()
    p_cmb = float(solver._P_basic_flat[0])
    t_m = float(np.asarray(shared_eos.temperature(np.array([p_cmb]), out.S_final[:1])).item())
    state = solver.state
    assert float(out.T_basic[0]) == pytest.approx(float(state.T_basic_diag[0]), rel=1e-15)
    components = ('jcond_b', 'jconv_b', 'jgrav_b', 'jmix_b')
    for name, attr in zip(components, ('jcond', 'jconv', 'jgrav_heat', 'jmix_heat')):
        assert float(getattr(out, name)[0]) == pytest.approx(
            float(getattr(state, attr)[0]), rel=1e-15
        )
    assert abs(float(out.T_basic[0]) - t_m) > 1.0e-3


@pytest.mark.physics_invariant
def test_core_module_core_cools_through_the_boundary_layer_and_closes_its_energy(shared_eos):
    """A core 300 K above a liquid base (phi 0.67 to 0.65, eta 33 to 48 Pa s) loses heat through the
    boundary layer fast enough to cool by about 1 K in 4 yr. The heat it loses is the heat
    booked into the mantle (q_radio = 0), and it equals the budget's content change between
    the start and end core temperatures (secular only, the core stays above nucleation)."""
    solver = _build('core_module', shared_eos, CORE_MODULE_PARAMS, end_time=4.0)
    S = np.linspace(7000.0, 6700.0, solver._n_stag)
    p_cmb = float(solver._P_basic_flat[0])
    t_m0 = float(np.asarray(shared_eos.temperature(np.array([p_cmb]), S[:1])).item())
    solver.set_initial_core_temperature(t_m0 + 300.0)
    solver.set_initial_entropy(S)
    solver.solve()
    out = solver.get_state()
    t_core = solver._solution.y[solver._n_stag + 1]
    budget = solver._core_module_budget
    content = budget.heat_content(float(t_core[-1])) - budget.heat_content(float(t_core[0]))
    t_m = float(np.asarray(shared_eos.temperature(np.array([p_cmb]), out.S_final[:1])).item())
    phi = [
        float(shared_eos.melt_fraction(np.array([p_cmb]), x[:1])[0]) for x in (S, out.S_final)
    ]
    # The numbers core_bc.md quotes for this case.
    assert 0.2 < float(t_core[0] - t_core[-1]) / 4.0 < 0.4
    assert 5.0e4 < out.F_cmb < 2.0e5
    assert 320.0 < float(t_core[-1]) - t_m < 340.0
    assert phi == pytest.approx([0.67, 0.65], abs=0.01)
    assert out.step_dE_core_J == pytest.approx(-out.step_dE_F_cmb_J, rel=1e-5)
    assert out.step_dE_core_J == pytest.approx(content, rel=1e-6)


@pytest.mark.physics_invariant
def test_core_module_cvode_solve_crosses_the_inner_core_onset(shared_eos):
    """A core 2 K above the inner-core onset over a liquid base cools through it under CVODE
    (a quadratic curve at 1.5 times Nimmo's t_m0 puts the onset above the base). The core
    heat change across the square-root cusp equals the heat_content difference and the CMB
    heat to the integrator's precision (2.6e-5 at the default rtol, falling with rtol)."""
    params = {
        k: v
        for k, v in CORE_MODULE_PARAMS.items()
        if k not in ('light_element_fraction', 'depression')
    }
    params.update(melting_curve='quadratic', t_m0=4015.5, t_m1=2.95e-12, t_m2=8.37e-25)
    solver = _build('core_module', shared_eos, params, end_time=4.0, solver_method='cvode')
    budget = solver._core_module_budget
    S = np.linspace(7000.0, 6700.0, solver._n_stag)
    p_cmb = float(solver._P_basic_flat[0])
    t_m = float(np.asarray(shared_eos.temperature(np.array([p_cmb]), S[:1])).item())
    t_onset = float(budget.t_onset)
    assert t_onset > t_m + 500.0
    solver.set_initial_core_temperature(t_onset + 2.0)
    solver.set_initial_entropy(S)
    solver.solve()
    out = solver.get_state()
    t0, t1 = (float(x) for x in solver._solution.y[solver._n_stag + 1, [0, -1]])
    assert t0 > t_onset > t1
    assert float(budget.r_icb(t1)) > 0.0
    content = float(budget.heat_content(t1) - budget.heat_content(t0))
    assert out.step_dE_core_J == pytest.approx(content, rel=1e-9)
    assert out.step_dE_core_J == pytest.approx(-out.step_dE_F_cmb_J, rel=1e-4)


STRATIFIED_PARAMS = {**CORE_MODULE_PARAMS, 'stratification': True, 'k_core': 130.0}
FLOOR_WARNING = 'convecting-radius floor'


@pytest.mark.physics_invariant
def test_a_stratified_default_start_keeps_the_core_temperature(shared_eos, caplog):
    """With stratification on, the default start (T_core at the mantle side of the CMB, so
    q = 0) sits on the convecting-radius floor, but the flow stays far below 1e-3 Q_k: the
    core temperature does not move, nothing warns, and the core gains the heat the mantle
    loses through the CMB."""
    solver = _build('core_module', shared_eos, STRATIFIED_PARAMS, end_time=5.0)
    solver.set_initial_entropy(_driven_s_profile(solver._n_stag))
    with caplog.at_level(logging.WARNING):
        solver.solve()
    out = solver.get_state()
    t_core = solver._solution.y[solver._n_stag + 1]
    assert abs(float(t_core[-1] - t_core[0])) < 1.0e-6
    assert not any(FLOOR_WARNING in r.message for r in caplog.records)
    assert out.step_dE_core_J == pytest.approx(-out.step_dE_F_cmb_J, rel=1.0e-2)


@pytest.mark.physics_invariant
def test_a_cold_stratified_core_warns_once_on_the_floor(shared_eos, caplog):
    """A core 300 K below the mantle (above the inner-core onset) gains heat through the CMB,
    so the whole core is subadiabatic and the budget uses the floor capacity: the heat gained
    over the change in T_core is C_eff on the floor, about 700 times below the full core's
    (core_bc.md).
    The solver warns once over two calls, and the core heat closes against the CMB heat."""
    solver = _build('core_module', shared_eos, STRATIFIED_PARAMS, end_time=5.0)
    budget = solver._core_module_budget
    S = _driven_s_profile(solver._n_stag)
    p_cmb = float(solver._P_basic_flat[0])
    t_m0 = float(np.asarray(shared_eos.temperature(np.array([p_cmb]), S[:1])).item())
    solver.set_initial_core_temperature(t_m0 - 300.0)
    solver.set_initial_entropy(S)
    with caplog.at_level(logging.WARNING):
        solver.solve()
        out = solver.get_state()
        t0, t1 = (float(x) for x in solver._solution.y[solver._n_stag + 1, [0, -1]])
        solver.solve()
    assert t1 > t0 > float(budget.t_onset)
    assert sum(FLOOR_WARNING in r.message for r in caplog.records) == 1
    assert out.step_dE_F_cmb_J < 0.0
    assert out.step_dE_core_J == pytest.approx(-out.step_dE_F_cmb_J, rel=1.0e-6)
    c_floor = float(budget.effective_capacity(t0, -1.0e12))
    assert out.step_dE_core_J / (t1 - t0) == pytest.approx(c_floor, rel=1.0e-6)
    assert 600.0 < float(budget.effective_capacity(t0, 1.0e14)) / c_floor < 800.0


@pytest.mark.parametrize('ra_crit', [0.0, -450.0, float('nan'), float('inf')])
def test_core_module_refuses_a_non_physical_critical_rayleigh_number(shared_eos, ra_crit):
    """ra_crit_cmb must be positive and finite; the solver refuses anything else when it
    builds the core budget, before any RHS call."""
    with pytest.raises(ValueError, match='ra_crit_cmb must be positive and finite'):
        _build('core_module', shared_eos, dict(CORE_MODULE_PARAMS, ra_crit_cmb=ra_crit))


def test_core_module_against_quasi_steady_baseline(shared_eos):
    """Cross-mode sanity on the same driven setup: both core temperatures
    are finite, the module's core, started 50 K above the mantle, cools
    through the CMB, and its start sits within 5 K of the quasi_steady CMB
    basic node plus that offset. The
    quasi_steady T_core is read at the bottom staggered cell, half a cell
    above the CMB node, so the two differ by 86 K on this mesh plus the 50 K
    start offset; the 150 K bracket only catches catastrophic divergence
    (initialisation or unit errors), and the flux-law discrimination lives in
    the contrast test."""
    legacy = _build('quasi_steady', shared_eos, s_init='driven')
    legacy.solve()
    t_legacy = legacy.get_state().T_core

    module = _build(
        'core_module', shared_eos, CORE_MODULE_PARAMS, s_init='driven', core_offset=50.0
    )
    module.solve()
    out = module.get_state()
    y = module._solution.y
    n_stag = module._n_stag
    t_module = out.T_core

    assert np.isfinite(t_legacy) and np.isfinite(t_module)
    assert float(y[n_stag + 1, -1]) < float(y[n_stag + 1, 0])
    t_legacy_cmb = float(legacy.state.phase_basic.temperature()[0])
    assert abs(float(y[n_stag + 1, 0]) - 50.0 - t_legacy_cmb) < 5.0
    assert abs(t_module - t_legacy) < 150.0


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
    """core_module completes a driven solve, core 50 K above the mantle, through the CVODE production
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
        core_offset=50.0,
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
        core_offset=50.0,
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
        core_offset=50.0,
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
            core_module_ra_crit_cmb=zsolver._core_module_ra_crit_cmb,
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
        core_offset=50.0,
    )
    fd.solve()
    y_fd = fd._solution.y[:, -1]
    # Same physics through both Jacobian paths, to the shared numpy-vs-JAX RHS difference
    # (about 4e-4 at two mid-mantle nodes), not to integrator tolerance.
    np.testing.assert_allclose(y_z[:n_stag], y_fd[:n_stag], rtol=5e-3)
    dT_z = float(y_z[n_stag + 1] - sol.y[n_stag + 1, 0])
    dT_fd = float(y_fd[n_stag + 1] - fd._solution.y[n_stag + 1, 0])
    assert dT_z < 0.0  # the core, 50 K above the mantle, cools
    assert dT_z == pytest.approx(dT_fd, rel=0.05)


def test_nucleation_temperature_independent_of_mesh_resolution(shared_eos):
    """Core profile and initial core temperature anchor at the CMB.

    Asserts that the core hydrostatic pressure and inner-core nucleation
    temperature do not vary with mantle mesh resolution, and that the default
    initial core temperature is the bottom cell at the CMB pressure, so a
    default start has no CMB flux.
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

    for solver, s_init, t_core in ((s10, s_init_10, t_core_10), (s60, s_init_60, t_core_60)):
        p_cmb = solver._P_basic_flat[:1]
        t_m = float(np.asarray(shared_eos.temperature(p_cmb, s_init[:1])).item())
        assert t_core == pytest.approx(t_m, rel=1e-12)
        solver.dSdt(0.0, solver._S0)
        assert float(solver.state.heat_flux[0]) == 0.0
