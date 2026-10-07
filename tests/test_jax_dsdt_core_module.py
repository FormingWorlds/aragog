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
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

from aragog.jax.solver import dSdt_core_module  # noqa: E402
from tests.conftest import entropy_eos_copy, entropy_eos_jax, needs_eos  # noqa: E402
from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build  # noqa: E402

# Module tier: the real-EOS parity and Jacobian solves are smoke and slow;
# the two factory-contract tests carry the unit marker so the PR lane still runs them.
pytestmark = [pytest.mark.timeout(300)]


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
    """core_module expects N+2 nondim scales: N+1 (the energy_balance
    length, the most plausible off-by-one) raises the incompatibility
    error, and N+2 with a budget builds callable rhs/jac functions."""
    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    n = 4
    budget = _tiny_budget()
    bad_scales = NonDimScales(state_scale=np.full(n + 1, 1.0), t_ref=1.0)
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

    good_scales = NonDimScales(state_scale=np.full(n + 2, 1.0), t_ref=1.0)
    kwargs = dict(
        eos_jax=None,
        phase_params=None,
        mesh_arrays=None,
        boundary_params=None,
        heating_array=np.zeros(n),
        scales=good_scales,
        core_bc_mode='core_module',
        core_module_budget=budget,
    )
    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(**kwargs, core_module_ra_crit_cmb=450.0)
    assert callable(rhs_fn) and callable(jac_fn)
    assert info['rhs_calls'] == 0
    # The critical Rayleigh number must come from the caller (the solver's value), and be
    # positive and finite.
    for bad in (None, 0.0, -450.0, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='ra_crit_cmb must be positive and finite'):
            build_jax_rhs_and_jacobian(**kwargs, core_module_ra_crit_cmb=bad)
    # A stratified budget needs the layer onset, which enters the compiled functions.
    kwargs['core_module_budget'] = budget.__class__(
        budget.profiles,
        budget.melting_curve,
        ds_fusion=170.0,
        icn_width=10.0,
        stratification=True,
        k_core=130.0,
    )
    with pytest.raises(ValueError, match='needs core_module_layer_start'):
        build_jax_rhs_and_jacobian(**kwargs, core_module_ra_crit_cmb=450.0)
    build_jax_rhs_and_jacobian(
        **kwargs, core_module_ra_crit_cmb=450.0, core_module_layer_start=0.0
    )


def _build_numpy_solver(shared_eos):
    """The driven core_module solver from the smoke harness, radau-free.

    Kept in sync with tests/test_entropy_solver_core_module_smoke.py's
    ``_build`` so the parity state matches a configuration the solve
    tests actually integrate.
    """
    return _build('core_module', shared_eos, CORE_MODULE_PARAMS, s_init='driven')


def _build_jax_pieces(solver):
    """JAX pytrees mirroring the numpy solver's mesh, phases, and BC."""
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams, _no_radio

    eos_jax = entropy_eos_jax()
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
        inner_bc_type=1,  # unused by dSdt_core_module, whose CMB flux is the boundary-layer law
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
        float(solver._core_module_ra_crit_cmb),
    )
    return args


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
def test_dsdt_core_module_direct_call_bounds_and_transient_excursion():
    """Direct evaluation of dSdt_core_module asserts shape, bounds, and excursion safety.

    Exercises:
    1. Shape contract: d(state_ext)/dt has shape (N+2,).
    2. Cooling direction: when CMB heat flux is positive and radioactive
       heating is absent, the core cooling rate dT_core/dt is strictly negative.
    3. Transient excursion protection: when an integrator excursion drives
       T_core non-positive (e.g. -50 K), the 1.0 K floor inside dSdt_core_module
       prevents passing non-positive temperatures to the budget, ensuring the
       RHS evaluates at the 1 K floor value and remains finite.
    4. Flux law: the gradient slot does not change the core or the bottom
       cell, and the sign of the core rate follows T_core against the
       mantle (a colder core is heated).
    """

    solver = _build_numpy_solver(entropy_eos_copy())
    args = _build_jax_pieces(solver)
    n_stag = solver._n_stag
    y0 = np.asarray(solver._S0, dtype=float)
    y0[n_stag + 1] += 50.0  # a core 50 K above the mantle

    # Base evaluation
    f0 = np.asarray(dSdt_core_module(0.0, jnp.asarray(y0), args)).ravel()
    assert f0.shape == (n_stag + 2,)
    assert np.all(np.isfinite(f0))
    # Core cooling rate must be negative under positive CMB heat flow
    assert f0[n_stag + 1] < 0.0

    # Excursion test: stub budget returns NaN for T < 1 K to verify the floor guard
    class GuardedBudget:
        def __init__(self, inner):
            self.inner = inner

        def dtcmb_dt(self, t_cmb, q_cmb, q_sources=0.0, t_layer=None):
            return jnp.where(
                t_cmb < 1.0,
                jnp.nan,
                self.inner.dtcmb_dt(t_cmb, q_cmb, q_sources=q_sources),
            )

    args_guarded = (*args[:6], GuardedBudget(args[6]), *args[7:])
    y_floor = y0.copy()
    y_floor[n_stag + 1] = 1.0
    f_floor = np.asarray(dSdt_core_module(0.0, jnp.asarray(y_floor), args_guarded)).ravel()
    assert np.all(np.isfinite(f_floor))

    y_excursion = y0.copy()
    y_excursion[n_stag + 1] = -50.0
    f_exc = np.asarray(dSdt_core_module(0.0, jnp.asarray(y_excursion), args_guarded)).ravel()
    assert f_exc.shape == (n_stag + 2,)
    assert np.all(np.isfinite(f_exc))
    np.testing.assert_allclose(np.asarray(f_exc), np.asarray(f_floor), rtol=1e-12)

    # The gradient slot does not set the flux: perturbing it leaves the core and the
    # bottom cell unchanged.
    y_slot = y0.copy()
    y_slot[n_stag] += 1e-5
    f_slot = np.asarray(dSdt_core_module(0.0, jnp.asarray(y_slot), args)).ravel()
    assert f_slot[n_stag + 1] == f0[n_stag + 1]
    assert f_slot[0] == f0[0]

    # A hotter core loses heat faster and heats the bottom cell; a core colder than the
    # mantle is heated and cools the bottom cell (second law at the CMB).
    for d_core, core_sign in ((10.0, -1.0), (-1000.0, 1.0)):
        y_core = y0.copy()
        y_core[n_stag + 1] += d_core
        f_core = np.asarray(dSdt_core_module(0.0, jnp.asarray(y_core), args)).ravel()
        assert np.sign(f_core[n_stag + 1]) == core_sign
        assert np.sign(f_core[0] - f0[0]) == -core_sign


@pytest.mark.slow
@pytest.mark.reference_pinned
@needs_eos
def test_rhs_parity_with_numpy_on_driven_state(monkeypatch):
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

    solver = _build_numpy_solver(entropy_eos_copy())
    args = _build_jax_pieces(solver)
    n_stag = solver._n_stag

    y0 = np.asarray(solver._S0, dtype=float)
    y0[n_stag + 1] += 50.0  # a core 50 K above the mantle, so the CMB carries heat
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
        # Interior nodes: 4e-4 at two mid-mantle nodes is the numpy-vs-JAX difference of the
        # shared flux assembly, the same on the energy_balance RHS at this state; a bound
        # below 1e-3 needs that difference fixed first.
        assert rel.max() < 1e-3, (
            f'state {k}: max rel err {rel.max():.3e} at component {rel.argmax()} '
            f'(numpy {f_np[rel.argmax()]:.6e} vs jax {f_jax[rel.argmax()]:.6e})'
        )
        # The boundary slots must be active, not vacuously matching zeros:
        # the driven profile cools the core through a real flux.
        assert f_np[n_stag + 1] < 0.0

    # q_radio path: with the same nonzero core source both sides agree on the boundary
    # slots, and a source comparable to the CMB heat flow warms the cooling rate.
    f_jax_base = np.asarray(dSdt_core_module(0.0, jnp.asarray(y0), args)).ravel()
    q_radio = 5.0e12
    solver._core_module_q_radio = q_radio
    args_r = args[:7] + (q_radio,) + args[8:]
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

    # ra_crit_cmb reaches both RHS: with a core 1000 K above the base the flux is on the
    # boundary-layer branch, so a 4x larger Ra_crit thickens the layer and slows the cooling.
    y_hot = y0.copy()
    y_hot[n_stag + 1] += 1000.0
    f_hot = np.asarray(solver.dSdt(0.0, y_hot)).ravel()
    monkeypatch.setattr(solver, '_core_module_ra_crit_cmb', 1800.0)
    f_np_c = np.asarray(solver.dSdt(0.0, y_hot)).ravel()
    f_jax_c = np.asarray(
        dSdt_core_module(0.0, jnp.asarray(y_hot), args[:8] + (1800.0,))
    ).ravel()
    np.testing.assert_allclose(f_np_c[n_stag:], f_jax_c[n_stag:], rtol=1e-8)
    assert f_hot[n_stag + 1] < f_np_c[n_stag + 1] < 0.0


@pytest.mark.slow
@needs_eos
def test_jacobian_carries_boundary_couplings():
    """``jacrev`` through the RHS (budget custom-JVP included) yields a
    finite Jacobian whose T_core row and column carry the couplings the
    sparsity pattern promises: dT_core_dt responds to the bottom cell and
    to T_core (through the boundary-layer flux), and the bottom cell
    responds to T_core (it receives the same flux). A zero cross-coupling
    here means the custom JVP was lost and CVODE's Newton would iterate on
    wrong derivatives.
    """
    solver = _build_numpy_solver(entropy_eos_copy())
    args = _build_jax_pieces(solver)
    n_stag = solver._n_stag
    y0 = np.asarray(solver._S0, dtype=float)

    # Pin T_core inside the nucleation-active band (scanned from the budget): outside it
    # d(C_eff)/dT_core is exactly zero and the self-coupling assertion would fail for any
    # JVP rule; the default T_core depends on the EOS tables in use.
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
    # dT_core/dt depends on the flux, which depends on T_core and on the bottom cell's
    # entropy, but not on the gradient slot.
    assert abs(J[n_stag + 1, 0]) > 0.0
    assert J[n_stag + 1, n_stag] == 0.0
    # dT_core/dt depends on T_core through the flux and C_eff(T_core); the C_eff part
    # survives only because reverse-mode keeps the budget's custom JVP.
    assert abs(J[n_stag + 1, n_stag + 1]) > 0.0
    # The gradient slot rides on the cooling rate.
    assert abs(J[n_stag, n_stag + 1]) > 0.0
    # The bottom cell receives the CMB flux directly, so it couples to T_core and no
    # longer to the gradient slot.
    assert abs(J[0, n_stag + 1]) > 0.0
    # The gradient slot is inert: its column is zero outside its own row.
    np.testing.assert_array_equal(np.delete(J[:, n_stag], n_stag), 0.0)


@pytest.mark.slow
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
    eos = entropy_eos_copy()
    strat_params = dict(CORE_MODULE_PARAMS) | {'stratification': True, 'k_core': 130.0}
    strat = _build('core_module', eos, strat_params)  # uniform isentrope
    plain = _build('core_module', eos, CORE_MODULE_PARAMS)
    n_stag = strat._n_stag
    # The isentrope starts T_core at the mantle CMB temperature (zero flux); 1 K hotter
    # drives a small, deeply subadiabatic flux (5.8e-5 W/m^2 through the mushy base).
    y0 = np.asarray(strat._S0, dtype=float)
    y0[n_stag + 1] += 1.0

    f_np = np.asarray(strat.dSdt(0.0, y0)).ravel()
    args = _build_jax_pieces(strat)
    f_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y0), args)).ravel()
    for slot in (n_stag, n_stag + 1):
        denom = max(abs(f_np[slot]), abs(f_jax[slot]), 1e-15)
        assert abs(f_np[slot] - f_jax[slot]) / denom < 1e-8

    # The reduced convecting volume amplifies the cooling response: the
    # stratified T_core rate exceeds the unstratified twin's at the same
    # state by well over the parity tolerance.
    y_plain = np.asarray(plain._S0, dtype=float)
    y_plain[n_stag + 1] += 1.0
    f_plain = np.asarray(plain.dSdt(0.0, y_plain)).ravel()
    assert abs(f_np[n_stag + 1]) > 2.0 * abs(f_plain[n_stag + 1])

    J = np.asarray(jax.jacrev(lambda y: dSdt_core_module(0.0, y, args))(jnp.asarray(y0)))
    assert J.shape == (n_stag + 2, n_stag + 2)
    assert np.all(np.isfinite(J))


@pytest.mark.unit
@pytest.mark.physics_invariant
@needs_eos
def test_boundary_slots_match_numpy_on_a_five_node_mesh():
    """The production JAX RHS and the numpy RHS agree on every component, both boundary
    slots included, on a 5-node mesh with the core 50 K above the mantle."""
    solver = _build(
        'core_module', entropy_eos_copy(), CORE_MODULE_PARAMS, s_init='driven', n_nodes=5
    )
    n_stag = solver._n_stag
    y = np.asarray(solver._S0, dtype=float)
    y[n_stag + 1] += 50.0
    f_np = np.asarray(solver.dSdt(0.0, y)).ravel()
    f_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y), _build_jax_pieces(solver))).ravel()
    assert f_np.shape == f_jax.shape == (n_stag + 2,)
    np.testing.assert_allclose(f_jax, f_np, rtol=1e-10)
    assert f_np[n_stag + 1] < 0.0


@pytest.mark.slow
@pytest.mark.physics_invariant
@needs_eos
@pytest.mark.parametrize(
    'state', ['nucleating', 'above_onset', 'stratified', 'stratified_capped']
)
def test_jacobian_core_column_matches_central_differences(state):
    """``jacrev`` of the core_module RHS agrees with a central difference in T_core for the
    gradient-slot and T_core rows, on the driven (non-uniform) profile: inside the
    nucleation band, above the onset, stratified above the convecting-radius floor, and with a
    10 Myr layer capped by its diffusion length, where JAX also matches NumPy."""
    params = dict(CORE_MODULE_PARAMS)
    if state.startswith('stratified'):
        params |= {'stratification': True, 'k_core': 130.0}
    solver = _build('core_module', entropy_eos_copy(), params, s_init='driven')
    budget, n = solver._core_module_budget, solver._n_stag
    y = np.asarray(solver._S0, dtype=float)
    if state == 'nucleating':
        scan = np.linspace(3200.0, 6000.0, 281)
        secular = float(budget.secular_capacity())
        latent = np.asarray(jax.vmap(budget.latent_capacity)(jnp.asarray(scan)))
        y[n + 1] = np.median(scan[latent > 0.01 * secular])
    elif state == 'above_onset':
        y[n + 1] = float(budget.t_onset) + 100.0
    else:
        y[n + 1] += 50.0
        solver.dSdt(0.0, y)
        q = float(solver.state.heat_flux[0]) * solver._cmb_area
        r_conv = float(budget.convecting_radius(y[n + 1], q)) / budget.profiles.r_cmb
        assert 0.2 < r_conv < 0.9
    args = _build_jax_pieces(solver)
    if state == 'stratified_capped':
        solver.set_initial_layer_start(-1.0e7)
        solver._resolve_layer_start(0.0)
        args += (-1.0e7,)
        assert (
            float(budget.convecting_radius(y[n + 1], q, 1e7 * 3.15576e7))
            / budget.profiles.r_cmb
            > 0.9
        )

    def rhs(v):
        return np.asarray(dSdt_core_module(0.0, jnp.asarray(v), args))

    np.testing.assert_allclose(rhs(y), np.asarray(solver.dSdt(0.0, y), dtype=float), rtol=1e-10)
    if state == 'stratified_capped':  # the factory passes the onset into its compiled RHS
        from aragog.jax.nondim import NonDimScales
        from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

        scales = NonDimScales(state_scale=np.ones(n + 2), t_ref=1.0)
        kw = dict(
            core_module_budget=budget, core_module_q_radio=args[7], core_module_layer_start=-1e7
        )
        rhs_fn, _, _ = build_jax_rhs_and_jacobian(
            *args[:5], scales, 'core_module', core_module_ra_crit_cmb=args[8], **kw
        )
        out = np.empty(n + 2)
        rhs_fn(0.0, y, out)
        np.testing.assert_allclose(out, rhs(y), rtol=1e-12)

    J = np.asarray(jax.jacrev(lambda v: dSdt_core_module(0.0, v, args))(jnp.asarray(y)))
    h, up, down = 0.1, y.copy(), y.copy()
    up[n + 1] += h
    down[n + 1] -= h
    fd = (rhs(up) - rhs(down)) / (2.0 * h)
    for row in (n, n + 1):
        assert fd[row] != 0.0
        assert J[row, n + 1] == pytest.approx(fd[row], rel=1e-6)
