"""Unit tests for ``EntropySolver`` helper functions and accessors.

The helper-function and small-method coverage in
``src/aragog/solver/entropy_solver.py`` lags the bulk of the file
because most existing tests go through the integration path
(``solve()`` → ``get_state()``). These targeted unit tests exercise:

* ``_phase_prop_float`` — the float / .eval() / default fallback used
  to coerce viscosity and conductivity strings from legacy .cfg
  configs into floats. Lines 141-147.
* ``_PhiCapRootFunction.evaluate`` — the ``mass_total <= 0`` branch
  (line 226) and the exception-swallow branch (227-235).
* ``_phi_cap_event_factory._event`` — same edge cases (lines 265,
  267-268).
* ``EntropySolver.set_jax_cvode_factory`` — registers / clears the
  factory (line 670).
* ``EntropySolver.get_current_dSdr_cmb`` — None-return branches when
  no solution exists or the state vector lacks the dSdr_cmb slot
  (lines 1325-1335).
* ``EntropySolver._state_is_extended`` — branches across all four
  core_bc modes.

All tests are pure unit (no EOS, no solve) so they cost <100 ms each.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
#                        _phase_prop_float
# ──────────────────────────────────────────────────────────────────────


def test_phase_prop_float_converts_plain_float():
    """Numeric input round-trips through ``float()``."""
    from aragog.solver.entropy_solver import _phase_prop_float

    assert _phase_prop_float(1.234, 99.0) == pytest.approx(1.234)
    assert _phase_prop_float(2, 99.0) == 2.0


def test_phase_prop_float_evaluates_legacy_string_expression():
    """Object with ``.eval()`` returning a float is coerced via that
    method (legacy .cfg parser path).
    """
    from aragog.solver.entropy_solver import _phase_prop_float

    class _StrExpr:
        def eval(self):
            return 1.0e21

    assert _phase_prop_float(_StrExpr(), 99.0) == pytest.approx(1.0e21)


def test_phase_prop_float_returns_default_when_neither_works():
    """When neither ``float()`` nor ``.eval()`` succeed, the supplied
    default is returned.

    Edge case: passing ``None`` as the default must yield ``None`` so
    the caller's "thermal_conductivity is None" guard fires (the
    EntropyPhaseEvaluator default is then used).
    """
    from aragog.solver.entropy_solver import _phase_prop_float

    # An object that neither floats nor has .eval(); we expect default.
    bad = object()
    assert _phase_prop_float(bad, 1.0) == 1.0
    assert _phase_prop_float(bad, None) is None


# ──────────────────────────────────────────────────────────────────────
#                _PhiCapRootFunction zero-mass / exception branches
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not __import__(
        'aragog.solver.entropy_solver', fromlist=['_CV_ROOTFN_AVAILABLE']
    )._CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_phi_cap_rootfn_zero_total_mass_falls_back_to_anchor_phi():
    """When the EOS-derived per-cell mass sums to zero (degenerate
    state), the rootfn must fall back to ``phi0`` rather than divide
    by zero.

    Discriminator: ``g[0] = cap - |phi_global - phi0|``. If the fallback
    set ``phi_global = 0`` instead of ``phi0``, ``g[0]`` would equal
    ``cap - phi0`` and the cap would fire spuriously at IC. Setting
    ``phi_global = phi0`` keeps ``g[0] = cap`` (the cap is fully
    available, no fire).
    """
    from aragog.solver.entropy_solver import _PhiCapRootFunction

    n_stag = 5
    eos = MagicMock()
    eos.density.return_value = np.zeros(n_stag)  # zero mass
    eos.melt_fraction.return_value = np.full(n_stag, 0.7)
    P_stag = np.full(n_stag, 5.0e10)
    volume = np.full(n_stag, 1.0e18)
    state_scale = np.ones(n_stag)
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=0.42,
        cap=0.05,
        state_scale=state_scale,
    )
    g = np.zeros(1)
    rc = rootfn.evaluate(0.0, np.zeros(n_stag), g)
    assert rc == 0
    assert float(g[0]) == pytest.approx(0.05, rel=1e-12), (
        f'g[0]={float(g[0]):.3e} != cap=0.05; mass_total<=0 fallback is broken'
    )


def test_phi_cap_event_factory_zero_total_mass_falls_back_to_anchor_phi():
    """Same fallback for the scipy ``solve_ivp`` event factory."""
    from aragog.solver.entropy_solver import _phi_cap_event_factory

    n_stag = 4
    eos = MagicMock()
    eos.density.return_value = np.zeros(n_stag)
    eos.melt_fraction.return_value = np.full(n_stag, 0.7)
    event = _phi_cap_event_factory(
        eos=eos,
        P_stag=np.full(n_stag, 5.0e10),
        volume=np.full(n_stag, 1.0e18),
        n_stag=n_stag,
        phi0_global=0.30,
        cap=0.10,
        state_scale=np.ones(n_stag),
    )
    val = event(0.0, np.zeros(n_stag))
    assert val == pytest.approx(0.10), (
        f'event value {val:.3e} != cap=0.10; mass_total<=0 fallback returns wrong value'
    )
    # The event must be flagged terminal with negative direction so
    # solve_ivp only fires on the cap-crossing direction.
    assert event.terminal is True
    assert event.direction == pytest.approx(-1.0)


def test_phi_cap_event_factory_eos_raises_falls_back_to_cap():
    """Exceptions inside the event callback must be swallowed, returning
    ``cap`` so the integrator continues without firing.

    Discriminator: a regression that re-raised would crash the
    integrator with a stale callback partway through a long solve.
    """
    from aragog.solver.entropy_solver import _phi_cap_event_factory

    eos = MagicMock()
    eos.density.side_effect = RuntimeError('intentional EOS failure')
    event = _phi_cap_event_factory(
        eos=eos,
        P_stag=np.full(3, 5.0e10),
        volume=np.full(3, 1.0e18),
        n_stag=3,
        phi0_global=0.25,
        cap=0.07,
        state_scale=np.ones(3),
    )
    val = event(0.0, np.zeros(3))
    assert val == pytest.approx(0.07), (
        f'event swallowed exception but returned {val:.3e}, expected cap=0.07'
    )


# ──────────────────────────────────────────────────────────────────────
#                  EntropySolver lightweight accessors
# ──────────────────────────────────────────────────────────────────────


def _build_minimal_solver(*, core_bc: str = 'energy_balance'):
    """Construct an ``EntropySolver`` whose Parameters are minimal-
    enough to instantiate but whose mesh / EOS / phase machinery is
    NOT initialised. Keeps the construction cost <50 ms.
    """
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
    from aragog.solver.entropy_solver import EntropySolver

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc=core_bc,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='radau',
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=10,
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
        end_time=1.0,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
    parameters = Parameters(
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
    return EntropySolver(parameters, entropy_eos=None)


def test_set_jax_cvode_factory_registers_and_clears():
    """``set_jax_cvode_factory`` registers a callable that ``solve()``
    later picks up; ``None`` clears it.

    Discriminator: a regression that ignored the argument would leave
    ``self._jax_cvode_factory`` permanently None, breaking the
    Option Z dispatch.
    """
    solver = _build_minimal_solver()
    assert solver._jax_cvode_factory is None  # default after __init__

    def _factory(scales, mode):
        return (None, None)

    solver.set_jax_cvode_factory(_factory)
    assert solver._jax_cvode_factory is _factory

    solver.set_jax_cvode_factory(None)
    assert solver._jax_cvode_factory is None


@pytest.mark.parametrize(
    'core_bc, expected',
    [
        ('quasi_steady', False),
        ('energy_balance', True),
        ('bower2018', True),
        ('gradient', True),
    ],
)
def test_state_is_extended_dispatches_per_core_bc_mode(core_bc, expected):
    """``_state_is_extended`` returns True for any non-quasi_steady mode.

    Discriminator: this property is consumed inside ``_dSdt_single``,
    ``solve``, ``get_state``, and ``_compute_step_energy_integrals``.
    A regression that hard-coded True would mishandle quasi_steady
    state-vector slicing; one that hard-coded False would crash the
    extended modes on the very first integrator call.
    """
    solver = _build_minimal_solver(core_bc=core_bc)
    # Skip initialize() — _state_is_extended only reads ``self._core_bc``.
    solver._core_bc = core_bc
    assert solver._state_is_extended is expected


def test_get_current_dsdr_cmb_returns_none_when_no_solution_exists():
    """Before ``solve()`` runs there is no ``_solution``; the accessor
    must return None (used by PROTEUS's retry-ladder snapshot).

    Discriminator: a regression that raised AttributeError instead of
    returning None would break the retry logic on the first coupling
    step.
    """
    solver = _build_minimal_solver(core_bc='energy_balance')
    # ``_n_stag`` is normally set by initialize(); fake it.
    solver._n_stag = 10
    solver._solution = None
    assert solver.get_current_dSdr_cmb() is None


def test_get_current_dsdr_cmb_returns_none_for_quasi_steady_state_shape():
    """When the solution state vector is length N (quasi_steady) rather
    than N+1 (energy_balance), the accessor must return None — the
    ``dSdr_cmb`` slot does not exist in this layout.
    """
    solver = _build_minimal_solver(core_bc='quasi_steady')
    solver._n_stag = 10

    fake_sol = MagicMock()
    fake_sol.y = np.zeros((10, 3))  # length N, not N+1
    solver._solution = fake_sol
    assert solver.get_current_dSdr_cmb() is None


def test_get_current_dsdr_cmb_returns_last_column_value_when_state_extended():
    """When the state has shape (N+1, K), the accessor returns the
    final-time dSdr_cmb (``y[N, -1]``).

    Discriminator: a regression that returned ``y[-1, -1]`` (the
    last STATE element) instead of ``y[N, -1]`` (the dSdr_cmb slot)
    would still pass for energy_balance (since N is the last slot)
    BUT would fail for any future extension of the state vector.
    Using a sentinel value at row N catches the off-by-one.
    """
    solver = _build_minimal_solver(core_bc='energy_balance')
    n_stag = 7
    solver._n_stag = n_stag
    sentinel = 1.234e-5
    fake_sol = MagicMock()
    y = np.zeros((n_stag + 1, 4))
    y[n_stag, -1] = sentinel
    fake_sol.y = y
    solver._solution = fake_sol
    val = solver.get_current_dSdr_cmb()
    assert val == pytest.approx(sentinel, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────
#                    _step_heat_content (entropy-transported heat)
# ──────────────────────────────────────────────────────────────────────


def test_step_heat_content_matches_analytic_integral():
    """``_step_heat_content`` integrates ``rho T dS`` per cell to the
    analytic value, not an endpoint estimate.

    With a constant EOS density and a temperature linear in entropy,
    ``integral rho T dS`` has the closed form ``rho (a dS + b (Sf^2 - S0^2)/2)``
    per cell, which is computed here independently of the trapezoidal
    quadrature in the method. The discrimination guard asserts the result
    differs from the naive endpoint estimate ``rho T(Sf) dS`` by more than
    rounding, so a regression to an endpoint formula is caught.
    """
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    # Curved T(S) = a + b S + c S^2 so the integrand rho*T is genuinely
    # nonlinear in S: the trapezoidal rule is then only approximate and the
    # test actually exercises quadrature resolution (a linear T would be
    # integrated exactly at any node count and would not catch a crippled
    # n_quad).
    rho0, a, b, c = 4000.0, 500.0, 0.5, 2.0e-4
    eos = MagicMock()
    eos.density.side_effect = lambda P, S: np.full(np.asarray(S, float).shape, rho0)
    eos.temperature.side_effect = lambda P, S: (
        a + b * np.asarray(S, float) + c * np.asarray(S, float) ** 2
    )

    P = np.array([1.0e10, 5.0e10, 1.0e11])
    V = np.array([1.0e18, 2.0e18, 1.5e18])
    S0 = np.array([3000.0, 2800.0, 2600.0])
    Sf = np.array([2500.0, 2700.0, 2000.0])  # cooling: Sf < S0
    fake = SimpleNamespace(entropy_eos=eos, _P_stag_flat=P, _volume_flat=V)

    got = EntropySolver._step_heat_content(fake, S0, Sf, n_quad=16)

    dS = Sf - S0
    analytic = float(
        np.sum(rho0 * (a * dS + 0.5 * b * (Sf**2 - S0**2) + c / 3.0 * (Sf**3 - S0**3)) * V)
    )
    # 16-point trapezoid on a quadratic integrand: close but not exact.
    assert got == pytest.approx(analytic, rel=1e-3)

    # Cooling must lower the heat content.
    assert got < 0.0

    # Discrimination 1: a crippled 2-point quadrature must differ from the
    # 16-point result by more than the tolerance, so dropping resolution is
    # caught (a linear T would make these identical).
    coarse = EntropySolver._step_heat_content(fake, S0, Sf, n_quad=2)
    assert abs(coarse - got) > 1e-3 * abs(got)
    # ...and 16 points must be much closer to the analytic value than 2.
    assert abs(got - analytic) < abs(coarse - analytic)

    # Discrimination 2: the endpoint estimate rho T(Sf) dS is a different
    # number, so the test fails if the integral degrades to an endpoint read.
    endpoint = float(np.sum(rho0 * (a + b * Sf + c * Sf**2) * dS * V))
    assert abs(got - endpoint) > 1e-3 * abs(got)


def test_step_heat_content_zero_when_no_eos():
    """Returns 0.0 when no EOS is attached (non-EOS interior backends)."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    fake = SimpleNamespace(
        entropy_eos=None,
        _P_stag_flat=np.ones(3),
        _volume_flat=np.ones(3),
    )
    assert EntropySolver._step_heat_content(fake, np.ones(3), np.zeros(3)) == 0.0


def test_remap_entropy_handles_missing_xi_pre_resolve():
    """The per-parcel entropy remap must not raise when the pre-resolve mass
    grid was never cached.

    ``_remap_entropy_to_current_mesh`` reads ``self._xi_pre_resolve``. A solver
    constructed outside the normal init/reset path (a stub, or an alternate
    constructor) has no such attribute; the guard must treat that as 'no remap
    cached', return the entropy unchanged, and leave the cache cleared rather
    than raising AttributeError.
    """
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    fake = SimpleNamespace()  # deliberately lacks _xi_pre_resolve
    S = np.array([3000.0, 2900.0, 2800.0])
    out = EntropySolver._remap_entropy_to_current_mesh(fake, S)
    np.testing.assert_array_equal(out, S)
    assert fake._xi_pre_resolve is None


@pytest.mark.physics_invariant
def test_remap_conserves_mass_weighted_mean_entropy():
    """The per-parcel entropy remap conserves the mass-weighted mean specific
    entropy across a structure re-solve.

    Sbar = sum(m_i S_i) / sum(m_i) is the discrete total mantle entropy per unit
    mass. A parcel-conservative remap must leave it unchanged when the mesh
    deforms but the material does not. The mass weight of a node is its
    cumulative-mass-fraction width, so Sbar is the trapezoidal integral of S over
    the mass fraction f. For an entropy field linear in f the trapezoid rule is
    exact, so remapping on f conserves Sbar to floating point; interpolating on
    the raw length coordinate xi does not, because xi is nonlinear in enclosed
    mass, so it reconstructs a curved field whose mass-weighted mean drifts.

    Scenario: a genuine re-solve shrinks r_surf 3.3 percent and r_core 0.6
    percent (matching a measured coupled Zalmoxis+Aragog step) and reshapes the
    interior mass distribution, over a steep solidification entropy gradient
    (1000 -> 3000 J/kg/K) representative of the mush phase where the raw-xi error
    is largest.

    Discrimination guard: under the identical deformation the raw-xi
    interpolation drifts Sbar by more than 100x the mass-fraction remap, so the
    invariance assertion distinguishes the two formulas rather than passing for
    both.
    """
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver, _mantle_mass_fraction

    def _mass_weighted_mean(s_vals, frac):
        # Trapezoidal integral of specific entropy over cumulative mass fraction,
        # normalised by the fraction span: the discrete mass-weighted mean.
        weight = np.abs(np.diff(frac))
        seg = 0.5 * (s_vals[1:] + s_vals[:-1])
        return float(np.sum(seg * weight) / np.sum(weight))

    n = 64
    u = np.linspace(0.0, 1.0, n)
    # Old mesh: staggered mass-radius nodes, mildly bottom-heavy density profile
    # (mass-fraction spacing non-uniform in radius).
    r_core_old, r_surf_old = 3.4149e6, 7.0670e6
    xi_old = np.cbrt(r_core_old**3 + (r_surf_old**3 - r_core_old**3) * u**1.15)
    # New mesh: endpoints move by different factors and the interior profile
    # reshapes, so each parcel's cumulative mass fraction shifts.
    r_core_new, r_surf_new = r_core_old * 0.994, r_surf_old * 0.967
    xi_new = np.cbrt(r_core_new**3 + (r_surf_new**3 - r_core_new**3) * u**1.30)

    f_old = _mantle_mass_fraction(xi_old)
    f_new = _mantle_mass_fraction(xi_new)
    # Steep entropy field, linear in mass fraction (a mush-phase gradient).
    s_bot, s_top = 1000.0, 3000.0
    s_old = s_bot + (s_top - s_bot) * f_old

    # Branch B (shipped): remap on mass fraction via the real method.
    fake = SimpleNamespace(
        _xi_pre_resolve=xi_old.copy(),
        staggered_mass_coordinates=xi_new,
    )
    s_new_b = EntropySolver._remap_entropy_to_current_mesh(fake, s_old.copy())
    # Branch A (superseded formula): raw-xi interpolation of the same field.
    s_new_a = np.interp(xi_new, xi_old, s_old)

    sbar_old = _mass_weighted_mean(s_old, f_old)
    sbar_new_b = _mass_weighted_mean(s_new_b, f_new)
    sbar_new_a = _mass_weighted_mean(s_new_a, f_new)

    drift_b = abs(sbar_new_b - sbar_old) / sbar_old
    drift_a = abs(sbar_new_a - sbar_old) / sbar_old

    # Invariance under the shipped mass-fraction remap.
    assert drift_b < 1e-9, f'mass-fraction remap must conserve Sbar, got {drift_b:.2e}'
    # Discrimination: the raw-xi formula is materially non-conservative here.
    assert drift_a > 100.0 * max(drift_b, 1e-15), (
        f'raw-xi interp (drift {drift_a:.2e}) must move Sbar far more than the '
        f'mass-fraction remap (drift {drift_b:.2e}); else the test is vacuous'
    )
    assert drift_a > 1e-6, (
        f'the steep-gradient scenario must make raw-xi non-conservation '
        f'resolvable, got {drift_a:.2e}'
    )
    # The remap consumes the cached pre-resolve grid (one-shot).
    assert fake._xi_pre_resolve is None


# ──────────────────────────────────────────────────────────────────────
#                 _phase_boundary_max_step_clamp
# ──────────────────────────────────────────────────────────────────────
#
# The 1-yr max_step clamp resolves the stiff RHS a cell sees while
# crossing the solidus/liquidus. It must fire near/inside the two-phase
# window and must NOT fire for a fully-frozen mantle far below the
# solidus (no phase-boundary stiffness there), otherwise CVODE is pinned
# to 1-yr steps for the entire post-solidus thermal history. The CMB
# liquidus-margin term is therefore two-sided (abs); a one-sided
# ``margin < 200`` test is unconditionally true for a sub-liquidus CMB
# cell and is the failure mode these tests guard against.


def test_max_step_clamp_off_for_deep_cold_solid():
    """A fully-frozen mantle far below the solidus must NOT trip the clamp.

    No cell is near or inside the two-phase window, and the CMB cell sits
    well below the solidus, so both CMB margins are large and negative.
    The clamp must return False so ``max_step`` stays at ``np.inf`` in the
    deep-solid regime. Discrimination: the CMB liquidus margin here is
    -1000 J/kg/K; a one-sided ``margin < 200`` test would return True
    (regression), whereas the two-sided ``abs(margin) < 200`` returns
    False. The gap between the two verdicts is unambiguous.
    """
    from aragog.solver.entropy_solver import _phase_boundary_max_step_clamp

    clamp = _phase_boundary_max_step_clamp(
        near_liq=False,
        near_sol=False,
        in_mushy=False,
        cmb_margin_to_liq=-1000.0,  # CMB entropy 1000 J/kg/K below liquidus
        cmb_margin_to_sol=-800.0,  # and 800 below solidus (fully frozen)
    )
    assert clamp is False
    # A one-sided margin<200 test (the pre-fix regression) would be True
    # here; assert the correct two-sided verdict differs from it.
    assert (-1000.0 < 200.0) is True  # the wrong-formula result
    assert clamp != (-1000.0 < 200.0)


def test_max_step_clamp_on_when_cmb_near_liquidus():
    """A CMB cell just below the liquidus (within 200 J/kg/K) trips the clamp.

    ``cmb_margin_to_liq = -50`` is inside the 200 J/kg/K band on the
    sub-liquidus side, so the two-sided test fires even though no other
    cell flags proximity. Boundary check: a margin of exactly -50 is well
    inside the band, and the complementary deep-solid case above (-1000)
    is well outside, so the 200 J/kg/K threshold is resolved on both sides.
    """
    from aragog.solver.entropy_solver import _phase_boundary_max_step_clamp

    clamp = _phase_boundary_max_step_clamp(
        near_liq=False,
        near_sol=False,
        in_mushy=False,
        cmb_margin_to_liq=-50.0,
        cmb_margin_to_sol=+150.0,
    )
    assert clamp is True


def test_max_step_clamp_on_when_cmb_in_mushy_band():
    """A CMB cell inside the two-phase window trips the clamp via the mushy term.

    Here ``cmb_margin_to_liq`` is -400 (below liquidus, outside the 200
    band, so the near-liquidus term is False) but ``cmb_margin_to_sol`` is
    +300 (above solidus): the CMB cell is mushy. The dedicated
    below-liquidus-and-above-solidus term must catch this even though the
    two-sided near-liquidus band does not.
    """
    from aragog.solver.entropy_solver import _phase_boundary_max_step_clamp

    clamp = _phase_boundary_max_step_clamp(
        near_liq=False,
        near_sol=False,
        in_mushy=False,
        cmb_margin_to_liq=-400.0,  # outside the 200 band, below liquidus
        cmb_margin_to_sol=+300.0,  # above solidus -> mushy CMB cell
    )
    assert clamp is True


def test_max_step_clamp_on_when_any_interior_cell_near_boundary():
    """An interior cell flagged near a boundary trips the clamp regardless of CMB.

    With the CMB cell far below the solidus (both margins large-negative,
    as in the deep-solid case), the clamp is driven purely by an interior
    ``near_sol`` flag. This proves the interior-cell terms are OR-combined
    with the CMB backstop rather than gated behind it: a crystallisation
    front deep in an otherwise-frozen mantle still tightens ``max_step``.
    """
    from aragog.solver.entropy_solver import _phase_boundary_max_step_clamp

    clamp = _phase_boundary_max_step_clamp(
        near_liq=False,
        near_sol=True,  # an interior cell within 200 J/kg/K of the solidus
        in_mushy=False,
        cmb_margin_to_liq=-1000.0,  # CMB itself deep-solid
        cmb_margin_to_sol=-800.0,
    )
    assert clamp is True
