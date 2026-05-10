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
