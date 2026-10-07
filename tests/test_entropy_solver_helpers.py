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


def _build_minimal_solver(*, core_bc: str = 'energy_balance', cvode_output_points: int = 65):
    """Construct an ``EntropySolver`` whose Parameters are minimal-
    enough to instantiate but whose mesh / EOS / phase machinery is
    NOT initialised. Keeps the construction cost <50 ms.

    ``cvode_output_points`` passes through to the ``[solver]`` config so a
    test can check that the value reaches ``_cvode_output_points``.
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
        cvode_output_points=cvode_output_points,
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


def test_cvode_output_points_config_reaches_solver_attribute():
    """A non-default ``[solver].cvode_output_points`` reaches the solver.

    Drives the value through the real construction path
    (``_SolverParameters`` -> ``Parameters`` -> ``EntropySolver.__init__``)
    and reads back ``_cvode_output_points``. The non-default value is the
    mutation guard: a solver that hardcoded 65 instead of reading the
    config would leave ``_cvode_output_points`` at 65 and fail this
    assertion.
    """
    solver = _build_minimal_solver(cvode_output_points=33)
    assert solver._cvode_output_points == 33


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
    # The pressure term pins the pairing of each cell's pressure with its entropy path.
    eos.temperature.side_effect = lambda P, S: (
        a + b * np.asarray(S, float) + c * np.asarray(S, float) ** 2 + 1.0e-9 * np.asarray(P)
    )

    P = np.array([1.0e10, 5.0e10, 1.0e11])
    V = np.array([1.0e18, 2.0e18, 1.5e18])
    S0 = np.array([3000.0, 2800.0, 2600.0])
    Sf = np.array([2500.0, 2700.0, 2000.0])  # cooling: Sf < S0
    fake = SimpleNamespace(entropy_eos=eos, _P_stag_flat=P, _volume_flat=V)

    got = EntropySolver._step_heat_content(fake, S0, Sf, n_quad=16)

    dS = Sf - S0
    analytic = float(
        np.sum(
            rho0
            * ((a + 1.0e-9 * P) * dS + 0.5 * b * (Sf**2 - S0**2) + c / 3.0 * (Sf**3 - S0**3))
            * V
        )
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
    endpoint = float(np.sum(rho0 * (a + b * Sf + c * Sf**2 + 1.0e-9 * P) * dS * V))
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


@pytest.mark.parametrize(('core_bc', 'n_quad'), [('core_module', 512), ('energy_balance', 16)])
def test_step_heat_content_default_quadrature_points(core_bc, n_quad):
    """The default rule has 512 points for core_module and 16 for the other modes: the result
    equals an explicit call with that count and differs from the other count."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    eos = MagicMock()
    eos.density.side_effect = lambda P, S: np.full(np.asarray(S).shape, 4000.0)
    eos.temperature.side_effect = lambda P, S: 500.0 + 2.0e-4 * np.asarray(S, float) ** 2
    fake = SimpleNamespace(
        entropy_eos=eos, _P_stag_flat=np.ones(2), _volume_flat=np.ones(2), _core_bc=core_bc
    )
    S0, Sf = np.array([3000.0, 2800.0]), np.array([2500.0, 2700.0])
    got = EntropySolver._step_heat_content(fake, S0, Sf)
    assert got == EntropySolver._step_heat_content(fake, S0, Sf, n_quad=n_quad)
    assert got != EntropySolver._step_heat_content(fake, S0, Sf, n_quad=528 - n_quad)


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
    entropy across a structure re-solve, for a field that is CURVED in mass
    fraction so conservation is a genuine property, not linear-interp exactness.

    Sbar = sum(m_i S_i) / sum(m_i) is the discrete total mantle entropy per unit
    mass; the mass weight of a node is its mass-fraction width, so Sbar is the
    trapezoidal integral of S over the mass fraction f. Moving parcels at fixed
    specific entropy leaves the continuum integral unchanged, so a
    parcel-conservative remap conserves Sbar up to quadrature error that shrinks
    as O(h^2) with resolution. Interpolating on the raw length coordinate xi
    mis-locates parcels (xi is nonlinear in enclosed mass) and injects a real,
    resolution-independent Sbar drift.

    The entropy field here is quadratic in f (s = s_bot + dS * f**2), a mush
    profile that is NOT linear in f, so the mass-fraction remap does not conserve
    Sbar by trivial linear-interp exactness. Scenario: a genuine re-solve shrinks
    r_surf 3.3 percent and r_core 0.6 percent (matching a measured coupled
    Zalmoxis+Aragog step) and reshapes the interior mass distribution, over a
    steep 1000 -> 3000 J/kg/K gradient. Measured on this scenario at n=256:
    mass-fraction drift 3.7e-6 (quadrature-limited), raw-xi drift 8.2e-2
    (physical mislocation), a ratio near 2.2e4.

    Discrimination guard: the raw-xi interpolation drifts Sbar by more than 1e-2,
    two orders of magnitude past the 1e-4 conservation bar the mass-fraction
    remap clears, so the superseded raw-xi formula would FAIL this test. The
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

    # Fine grid so the mass-fraction remap's quadrature error is well resolved
    # and cleanly separated from the raw-xi mislocation error.
    n = 256
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
    # Steep entropy field, QUADRATIC in mass fraction (curved mush-phase
    # gradient): linear interpolation of this field is not exact, so conserving
    # Sbar is a real invariant rather than an artefact of the field shape.
    s_bot, s_top = 1000.0, 3000.0
    s_old = s_bot + (s_top - s_bot) * f_old**2

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

    # Invariance under the shipped mass-fraction remap: conserved to quadrature
    # (measured 3.7e-6 at n=256; the 1e-4 bar leaves >20x headroom).
    assert drift_b < 1e-4, f'mass-fraction remap must conserve Sbar, got {drift_b:.2e}'
    # Discrimination: the superseded raw-xi formula violates the same 1e-2 bar
    # by two orders of magnitude, so it would fail this conservation test.
    assert drift_a > 1e-2, (
        f'raw-xi interp (drift {drift_a:.2e}) must break Sbar conservation on '
        f'the curved field; else the test does not discriminate the formulas'
    )
    # Explicit separation guard between the two formulas.
    assert drift_a > 100.0 * drift_b, (
        f'raw-xi drift {drift_a:.2e} must exceed mass-fraction drift {drift_b:.2e} by >100x'
    )
    # The remap consumes the cached pre-resolve grid (one-shot).
    assert fake._xi_pre_resolve is None


@pytest.mark.physics_invariant
def test_mantle_mass_fraction_uses_cube_exponent():
    """_mantle_mass_fraction normalises xi**3 (affine in enclosed mass), not
    xi**2, xi**2.5, or xi**4.

    Build a staggered array whose nodes sit at KNOWN, non-uniform enclosed-mass
    fractions via the defining relation xi = cbrt(r_core**3 + (r_surf**3 -
    r_core**3) * m_frac). The cube-normalised coordinate must recover those
    fractions to floating point. Any other exponent is a different, wrong mass
    measure: the anchor pins the physical cube law rather than passing for a
    family of exponents.

    Discrimination guard: the xi**2, xi**2.5, and xi**4 normalisations each
    deviate from the true fraction by more than 1e-2 at interior nodes (measured
    4e-2 to 8e-2), far past the 1e-12 recovery tolerance, so a swapped exponent
    would fail this test.
    """
    from aragog.solver.entropy_solver import _mantle_mass_fraction

    r_core, r_surf = 3.4149e6, 7.0670e6
    # Non-uniform known mass fractions (not symmetric, so exponent errors show).
    m_frac = np.linspace(0.0, 1.0, 48) ** 1.15
    xi = np.cbrt(r_core**3 + (r_surf**3 - r_core**3) * m_frac)

    f_cube = _mantle_mass_fraction(xi)
    # The cube law recovers the enclosed-mass fractions exactly (endpoints are
    # 0 and 1 by construction; interior to machine epsilon).
    np.testing.assert_allclose(f_cube, m_frac, rtol=0.0, atol=1e-12)

    # Wrong exponents: none reproduce the true mass fraction.
    for p in (2.0, 2.5, 4.0):
        xip = xi**p
        lo, hi = min(xip[0], xip[-1]), max(xip[0], xip[-1])
        f_wrong = (xip - lo) / (hi - lo)
        max_dev = float(np.max(np.abs(f_wrong - m_frac)))
        assert max_dev > 1e-2, (
            f'xi**{p} normalisation deviates only {max_dev:.2e} from the true '
            f'mass fraction; the cube anchor would not discriminate it'
        )


@pytest.mark.physics_invariant
def test_remap_uniform_scaling_is_a_no_op():
    """A pure uniform radial rescale of the mesh must carry the entropy field
    through unchanged, because the mass coordinate is scale-invariant.

    Under xi -> alpha * xi every cube scales by alpha**3, so
    (xi**3 - xi[0]**3) / (xi[-1]**3 - xi[0]**3) is unchanged: each parcel keeps
    its mass fraction, so the same material sits at the same mass coordinate and
    its entropy must not move. The shipped mass-fraction remap returns the field
    to floating point.

    Discrimination guard: raw-xi interpolation of the same rescaled mesh shifts
    the field by hundreds of J/kg/K (it treats the compressed radii as new
    positions), so the no-op property is specific to the mass-fraction mapping,
    not shared by the superseded formula.
    """
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver, _mantle_mass_fraction

    u = np.linspace(0.0, 1.0, 48)
    r_core, r_surf = 3.4149e6, 7.0670e6
    xi_old = np.cbrt(r_core**3 + (r_surf**3 - r_core**3) * u**1.15)
    xi_new = 0.97 * xi_old  # uniform 3 percent contraction
    # Curved field so a mislocating remap would visibly move it.
    s_old = 1000.0 + 2000.0 * _mantle_mass_fraction(xi_old) ** 2

    fake = SimpleNamespace(
        _xi_pre_resolve=xi_old.copy(),
        staggered_mass_coordinates=xi_new,
    )
    s_new = EntropySolver._remap_entropy_to_current_mesh(fake, s_old.copy())
    # Mass-fraction remap is an exact no-op under uniform scaling.
    np.testing.assert_allclose(s_new, s_old, rtol=0.0, atol=1e-9)

    # Discrimination: raw-xi interpolation is NOT a no-op here.
    s_raw = np.interp(xi_new, xi_old, s_old)
    max_shift = float(np.max(np.abs(s_raw - s_old)))
    assert max_shift > 1.0, (
        f'raw-xi interp should move the field under uniform scaling, got '
        f'{max_shift:.2e} J/kg/K; the no-op is specific to the mass mapping'
    )
    assert fake._xi_pre_resolve is None


@pytest.mark.physics_invariant
def test_remap_descending_order_matches_ascending():
    """The remap gives the same physical result whether the mesh is stored
    CMB-to-surface (ascending xi) or surface-to-CMB (descending xi).

    np.interp needs an ascending abscissa; the descending branch reverses the
    mass coordinate, interpolates, and reverses back. Feeding the reversed mesh
    and reversed field must return exactly the reverse of the ascending-order
    result. This exercises the f_old[0] > f_old[-1] branch and pins its
    correctness, not merely that it runs.
    """
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver, _mantle_mass_fraction

    u = np.linspace(0.0, 1.0, 40)
    r_core, r_surf = 3.4149e6, 7.0670e6
    xi_old = np.cbrt(r_core**3 + (r_surf**3 - r_core**3) * u**1.15)
    xi_new = np.cbrt(
        (r_core * 0.994) ** 3 + ((r_surf * 0.967) ** 3 - (r_core * 0.994) ** 3) * u**1.30
    )
    s_old = 1000.0 + 2000.0 * _mantle_mass_fraction(xi_old) ** 2

    fake_asc = SimpleNamespace(_xi_pre_resolve=xi_old.copy(), staggered_mass_coordinates=xi_new)
    out_asc = EntropySolver._remap_entropy_to_current_mesh(fake_asc, s_old.copy())

    fake_desc = SimpleNamespace(
        _xi_pre_resolve=xi_old[::-1].copy(),
        staggered_mass_coordinates=xi_new[::-1].copy(),
    )
    out_desc = EntropySolver._remap_entropy_to_current_mesh(fake_desc, s_old[::-1].copy())

    # Descending-order result is the exact reverse of the ascending-order one.
    np.testing.assert_allclose(out_desc, out_asc[::-1], rtol=0.0, atol=1e-9)
    # And the remap is non-trivial (the field actually moved), so the equality
    # is not the vacuous 'both returned the input'.
    assert float(np.max(np.abs(out_asc - s_old))) > 1.0


def test_remap_degenerate_mass_span_carries_field_unchanged(caplog):
    """A degenerate cached grid (all nodes coincident, zero mass span) must fail
    safe: carry the entropy through unchanged and warn, never collapse it.

    _mantle_mass_fraction returns all zeros when every node coincides. Feeding
    those to np.interp would collapse the whole field to a single value. The
    remap detects the degenerate span and returns the input untouched instead,
    logging a warning so the anomaly is visible.
    """
    import logging
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    xi_deg = np.full(6, 4.0e6)  # coincident nodes: zero mass span
    u = np.linspace(0.0, 1.0, 6)
    xi_new = np.cbrt(3.4149e6**3 + (7.0670e6**3 - 3.4149e6**3) * u**1.1)
    s_in = np.array([3000.0, 2900.0, 2800.0, 2700.0, 2600.0, 2500.0])

    fake = SimpleNamespace(_xi_pre_resolve=xi_deg.copy(), staggered_mass_coordinates=xi_new)
    with caplog.at_level(logging.WARNING):
        out = EntropySolver._remap_entropy_to_current_mesh(fake, s_in.copy())

    # Field is carried through unchanged, NOT collapsed to a single value.
    np.testing.assert_array_equal(out, s_in)
    assert len(np.unique(out)) == 6, 'degenerate remap must not collapse the field'
    assert any('degenerate mass span' in r.message for r in caplog.records)
    assert fake._xi_pre_resolve is None


def test_remap_shape_mismatch_passes_through_and_warns(caplog):
    """A cached pre-resolve grid whose length does not match the entropy array
    is a coupling inconsistency: the remap carries the field through unchanged
    and warns, rather than silently masking the mismatch.

    This is not the normal no-cache path (that returns silently); a length
    mismatch means the cached grid and the entropy array came from different
    mesh sizes, which should surface in the log.
    """
    import logging
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    xi_old = np.linspace(3.4e6, 7.0e6, 5)  # length 5
    xi_new = np.linspace(3.4e6, 7.0e6, 5)
    s_in = np.array([3000.0, 2900.0, 2800.0, 2700.0])  # length 4: mismatch

    fake = SimpleNamespace(_xi_pre_resolve=xi_old.copy(), staggered_mass_coordinates=xi_new)
    with caplog.at_level(logging.WARNING):
        out = EntropySolver._remap_entropy_to_current_mesh(fake, s_in.copy())

    np.testing.assert_array_equal(out, s_in)
    assert any('shape mismatch' in r.message for r in caplog.records)
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
        entropy_margin=200.0,
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
        entropy_margin=200.0,
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
        entropy_margin=200.0,
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
        entropy_margin=200.0,
    )
    assert clamp is True


def test_max_step_clamp_entropy_margin_is_configurable():
    """The near-liquidus proximity band scales with ``entropy_margin``.

    A CMB cell 300 J/kg/K below the liquidus and below the solidus (so not
    mushy, and with no interior-cell flags) sits outside a 200 J/kg/K band,
    so the clamp is off at that margin. Widening the margin to 400 J/kg/K
    brings the same cell inside the band and the clamp fires. The two verdicts
    straddle the change, proving the threshold is driven by the parameter
    rather than a hard-coded 200, and that the 300 J/kg/K offset is resolved
    decisively on both sides. ``entropy_margin`` is a required argument, so
    production can never fall back to a stale signature default.
    """
    import inspect

    from aragog.solver.entropy_solver import _phase_boundary_max_step_clamp

    common = dict(
        near_liq=False,
        near_sol=False,
        in_mushy=False,
        cmb_margin_to_liq=-300.0,  # 300 below liquidus
        cmb_margin_to_sol=-100.0,  # and below solidus -> not mushy
    )
    assert _phase_boundary_max_step_clamp(**common, entropy_margin=200.0) is False
    assert _phase_boundary_max_step_clamp(**common, entropy_margin=400.0) is True
    # The margin has no default: the production caller always resolves and
    # passes it, so a signature default would be dead code that a maintainer
    # could silently drift away from the config default it must track.
    sig = inspect.signature(_phase_boundary_max_step_clamp)
    assert sig.parameters['entropy_margin'].default is inspect.Parameter.empty


def test_set_initial_entropy_gradient_mode():
    """Initial entropy in gradient mode sets S0 with gradients and surface entropy."""
    s = _build_minimal_solver(core_bc='gradient')
    s.initialize()
    s.set_initial_entropy(3000.0)
    assert hasattr(s, '_S0')
    assert len(s._S0) == s.parameters.mesh.number_of_nodes + 1


def test_set_initial_entropy_core_module_with_and_without_mesh():
    """Initial entropy resolution for core_module exercises offset with and without mesh."""
    from types import SimpleNamespace

    mock_eos = MagicMock()
    mock_eos.temperature.return_value = np.array([4500.0])

    # Case A: with mesh basic radii
    s_a = _build_minimal_solver(core_bc='core_module')
    s_a.initialize()
    s_a.entropy_eos = mock_eos
    s_a._P_basic_flat = np.array([136e9])
    s_a._P_stag_flat = np.array([135e9])
    mesh_mock = SimpleNamespace(basic=SimpleNamespace(radii=np.linspace(3.48e6, 6.371e6, 10)))
    s_a.evaluator = SimpleNamespace(mesh=mesh_mock)
    s_a.set_initial_entropy(3000.0)
    assert s_a._S0[10] == pytest.approx(4500.0)

    # Case B: without mesh (evaluator is None)
    s_b = _build_minimal_solver(core_bc='core_module')
    s_b.initialize()
    s_b.entropy_eos = mock_eos
    s_b._P_basic_flat = np.array([136e9])
    s_b.evaluator = None
    s_b._dSdr_cmb_init = 0.0
    s_b.set_initial_entropy(3000.0)
    assert s_b._S0[10] == pytest.approx(4500.0)

    # Case C: bower2018 with no initial core temperature (resolves from EOS at P_stag)
    s_c = _build_minimal_solver(core_bc='bower2018')
    s_c.initialize()
    s_c.entropy_eos = mock_eos
    s_c._P_stag_flat = np.array([135e9])
    s_c._T_core_init = None
    s_c.set_initial_entropy(3000.0)
    assert s_c._S0[9] == pytest.approx(4500.0)


def test_step_energy_stratified_core_module_and_fallback():
    """A stratified core books the convecting core's trapezoid of C_eff dT_c plus the change of
    the shell's heat, on the compiled path and on the fallback."""
    from types import SimpleNamespace

    from scipy.optimize import OptimizeResult

    import aragog.solver.entropy_solver as es
    from aragog.core import CoreEnergyBudget, GaussianCoreProfiles, QuadraticMeltingCurve

    profiles = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=139e9,
        alpha=1.25e-5,
        c_p=840.0,
    )
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    budget = CoreEnergyBudget(
        profiles,
        curve,
        ds_fusion=170.0,
        icn_width=10.0,
        stratification=True,
        k_core=130.0,
        layer={'n_cells': 4},
    )
    shell = budget.shell
    t_shell = np.column_stack(
        [shell.adiabatic_profile(4500.0), shell.adiabatic_profile(4400.0)]
    )
    sol = OptimizeResult(
        t=np.array([0.0, 1.0]), y=np.vstack([[2000.0, 2000.0], [4500.0, 4400.0], t_shell])
    )
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._solution, s.entropy_eos = sol, object()
    s._r_basic_flat = np.array([1.0, 2.0])
    s.state = SimpleNamespace(_pb_cache_hits=0, _pb_cache_misses=0)
    s._stag_entropy = lambda y: y
    s._step_heat_content = lambda a, b: 0.0
    s._step_powers = lambda t, y: np.array([0.0, 1e12, 0.0, 0.0, 0.0, 0.0, 0.0])
    s._core_bc, s._n_stag, s._core_module_budget = 'core_module', 0, budget
    capacity = [
        float(budget.effective_capacity(t, shell.layer_base(t_shell[:, i], t)))
        for i, t in enumerate((4500.0, 4400.0))
    ]
    content = [float(shell.heat_content(t_shell[:, i])) for i in (0, 1)]
    expected = 0.5 * sum(capacity) * -100.0 + content[1] - content[0]
    assert s._compute_step_energy_integrals()['core'] == pytest.approx(expected, rel=1e-10)

    def fail_vmap(*args):
        raise RuntimeError('simulated vmap failure')

    budget._vmap_effective_capacity = fail_vmap
    assert s._compute_step_energy_integrals()['core'] == pytest.approx(expected, rel=1e-10)


def test_solver_output_to_netcdf_step_dE_core_J(tmp_path):
    """SolverOutput.to_netcdf writes step_dE_core_J to netCDF."""
    import netCDF4 as nc

    from tests.test_solver_output_netcdf import _make_output

    out = _make_output()
    p = tmp_path / 'test.nc'
    out.to_netcdf(p)
    with nc.Dataset(p) as ds:
        assert float(ds['step_dE_core_J'][...]) == pytest.approx(out.step_dE_core_J)


def test_get_current_core_temperature():
    """get_current_core_temperature returns None or the last T_core state value."""
    from scipy.optimize import OptimizeResult

    s = _build_minimal_solver(core_bc='core_module')
    s.initialize()

    # None cases: no solution, shape mismatch, or core_bc without T_core
    assert s.get_current_core_temperature() is None

    s._solution = OptimizeResult(y=np.array([]))
    assert s.get_current_core_temperature() is None

    # Shape mismatch (e.g. wrong number of rows)
    s._solution = OptimizeResult(y=np.ones((5, 2)))
    assert s.get_current_core_temperature() is None

    # core_bc with no T_core slot
    s._core_bc = 'gradient'
    assert s.get_current_core_temperature() is None

    # Valid core_module case: n_stag + 2 slots (dSdr_cmb, T_core)
    s._core_bc = 'core_module'
    n_stag = s._n_stag
    y = np.zeros((n_stag + 2, 3))
    y[n_stag + 1, -1] = 5234.5
    s._solution = OptimizeResult(y=y)
    assert s.get_current_core_temperature() == pytest.approx(5234.5)

    # Fallback to parameters.boundary_conditions.core_bc when _core_bc is None
    s._core_bc = None
    assert s.get_current_core_temperature() == pytest.approx(5234.5)


def test_set_initial_entropy_warm_restart_and_warnings(caplog):
    """Warm restart preserves T_core and dSdr_cmb without a warning; a supplied start
    with a large offset warns once per solver."""
    import logging

    from scipy.optimize import OptimizeResult

    s = _build_minimal_solver(core_bc='core_module')
    s.initialize()
    n_stag = s._n_stag

    # Mock previous solution for warm restart
    y_prev = np.zeros((n_stag + 2, 4))
    y_prev[n_stag, -1] = 1.5e-4
    y_prev[n_stag + 1, -1] = 3000.0
    s._solution = OptimizeResult(y=y_prev)

    mock_eos = MagicMock()
    mock_eos.temperature.return_value = np.array([5000.0])
    s.entropy_eos = mock_eos
    s._P_basic_flat = np.array([136e9])

    with caplog.at_level(logging.WARNING):
        s.set_initial_entropy(3000.0)

    # Solution preserved from previous solve
    assert s._S0[n_stag] == pytest.approx(1.5e-4)
    assert s._S0[n_stag + 1] == pytest.approx(3000.0)
    msg = 'differs from the mantle temperature at the CMB'
    assert not any(msg in r.message for r in caplog.records)

    s.set_initial_core_temperature(3000.0)
    with caplog.at_level(logging.WARNING):
        s.set_initial_entropy(3000.0)
    assert s._S0[n_stag + 1] == pytest.approx(3000.0)
    assert sum(msg in r.message for r in caplog.records) == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        s.set_initial_entropy(3000.0)
    assert not any(msg in r.message for r in caplog.records)

    # Test n_stag < 2 fallback for dSdr_cmb_init
    s2 = _build_minimal_solver(core_bc='core_module')
    s2.initialize()
    s2._n_stag = 1
    assert s2._resolve_dSdr_cmb_init(np.array([3000.0]), 1) == 0.0


def test_step_dE_core_heating_and_exception_fallback():
    """Unstratified step_dE_core supports heating across onset and handles jit fallback."""
    from types import SimpleNamespace

    from scipy.optimize import OptimizeResult

    import aragog.solver.entropy_solver as es

    # Heating: T_start=5000 < T_end=5300 crossing onset=5150
    sol = OptimizeResult(
        t=np.array([0.0, 1.0]),
        y=np.array([[2000.0, 2000.0], [5000.0, 5300.0]]),
    )
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._solution = sol
    s.entropy_eos = object()
    s._r_basic_flat = np.array([1.0, 2.0])
    s.state = SimpleNamespace(_pb_cache_hits=0, _pb_cache_misses=0)
    s._stag_entropy = lambda y: y
    s._step_heat_content = lambda a, b: 0.0
    s._step_powers = lambda t, y: np.array([0.0, 1e12, 0.0, 0.0, 0.0, 0.0, 0.0])
    s._core_bc = 'core_module'
    s._n_stag = 0

    mock_budget = SimpleNamespace(
        stratification=False,
        t_onset=5150.0,
        t_freeze=4000.0,
        effective_capacity=lambda t: 2.0e27,
    )
    s._core_module_budget = mock_budget

    out = s._compute_step_energy_integrals()
    assert out['core'] == pytest.approx(2.0e27 * (5300.0 - 5000.0), rel=1e-5)

    # Exception fallback branch
    def fail_vmap(*args):
        raise RuntimeError('simulated vmap failure')

    mock_budget._vmap_effective_capacity = fail_vmap
    out2 = s._compute_step_energy_integrals()
    assert out2['core'] == pytest.approx(2.0e27 * (5300.0 - 5000.0), rel=1e-5)


def test_step_dE_core_bower2018():
    """Bower2018 core energy change integrates constant core capacity."""
    from types import SimpleNamespace

    from scipy.optimize import OptimizeResult

    import aragog.solver.entropy_solver as es

    sol = OptimizeResult(
        t=np.array([0.0, 1.0]),
        y=np.array([[5000.0, 4800.0]]),
    )
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._solution = sol
    s.entropy_eos = object()
    s._r_basic_flat = np.array([3.48e6, 4.0e6])
    s.state = SimpleNamespace(_pb_cache_hits=0, _pb_cache_misses=0)
    s._stag_entropy = lambda y: y
    s._step_heat_content = lambda a, b: 0.0
    s._step_powers = lambda t, y: np.array([0.0, 1e12, 0.0, 0.0, 0.0, 0.0, 0.0])
    s._core_bc = 'bower2018'
    s._n_stag = 0
    s.parameters = SimpleNamespace(
        mesh=SimpleNamespace(core_density=7000.0),
        boundary_conditions=SimpleNamespace(core_heat_capacity=800.0),
    )

    out = s._compute_step_energy_integrals()
    vol_c = 4.0 / 3.0 * np.pi * (3.48e6**3)
    c_core = vol_c * 7000.0 * 800.0
    assert out['core'] == pytest.approx(c_core * (4800.0 - 5000.0))


@pytest.mark.parametrize('mode', ['bower2018', 'core_module'])
def test_set_initial_entropy_no_eos_and_no_override_raises(mode):
    """Raise ValueError when no entropy EOS and no core temperature override exist.

    Setting initial entropy without an EOS and without an explicit core temperature
    cannot determine the core temperature and must raise ValueError rather than
    substituting the raw entropy value in Kelvin.
    """
    s = _build_minimal_solver(core_bc=mode)
    s.initialize()
    s.entropy_eos = None
    s._T_core_init = None
    with pytest.raises(ValueError, match='entropy EOS is not available'):
        s.set_initial_entropy(2900.0)


def test_the_conducted_adiabatic_flow_needs_a_core_conductivity():
    """An unstratified budget has no k_core, so the conducted flow is refused by name."""
    from aragog.core import build_core_module_budget

    budget = build_core_module_budget({}, r_cmb=3.48e6, p_cmb_fallback=136e9)
    with pytest.raises(ValueError, match='needs k_core'):
        budget.conducted_adiabatic_flow(3.48e6, 4500.0)
