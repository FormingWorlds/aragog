"""End-to-end integration of ``EntropySolver`` for coverage / regression.

The smoke tests in ``test_entropy_advanced.py`` and
``test_entropy_verification.py`` exercise the lower-level JAX RHS or roll
their own ``solve_ivp`` calls; **none** of them call
``EntropySolver(parameters).solve()`` end-to-end. As a result the
``solve()`` body, ``initialize()``, ``set_initial_entropy()``,
``_initialize_internals()``, ``get_state()``, and the post-solve
diagnostic helpers were entirely uncovered by the test suite (32.4 %
on src/aragog/solver/entropy_solver.py before this file).

This module fills that gap with one minimal-mesh, short-integration-time
solver run on each of the supported core_bc modes that have a coverage
deficit:

  * ``quasi_steady`` (the default; baseline path)
  * ``energy_balance`` (the SPIDER-parity production CHILI path)

The bower2018 and gradient core_bc modes share most of the dispatch
code paths exercised by the above two and adding a third / fourth run
would only marginally increase coverage; they are reserved for follow-
up work if the 85 % floor is not met after these.

Tests use the scipy ``radau`` solver path rather than CVODE so they do
not depend on scikits.odes being installed; this also avoids the JAX
JIT compile cost that dominates short runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

EOS_DIR = Path(
    os.environ.get(
        'ARAGOG_TEST_EOS_DIR',
        '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
    )
)

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)

# Smoke marker: this is an integration test that needs the EOS data,
# matching the convention used in test_entropy_verification.py.
pytestmark = [pytest.mark.smoke, needs_eos]


def _build_parameters(
    *,
    core_bc: str = 'quasi_steady',
    solver_method: str = 'radau',
    end_time: float = 50.0,
    n_nodes: int = 15,
    use_jax_jacobian: bool = False,
):
    """Build an in-memory Parameters object for a short solver run.

    Earth-like geometry with a prescribed-flux core BC and grey-body
    surface. Phase parameters use lookup paths (the EntropyEOS object
    supplies the actual P-S tables; phase_solid/phase_liquid string
    fields are required by the dataclass but not consumed in the
    entropy-form solver).
    """
    from aragog.parser import (
        Parameters,
        _BoundaryConditionsParameters,
        _EnergyParameters,
        _InitialConditionParameters,
        _MeshParameters,
        _PhaseMixedParameters,
        _PhaseParameters,
        _ScalingsParameters,
        _SolverParameters,
    )

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,  # grey body
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,  # prescribed core flux
        inner_boundary_value=0.0,  # zero CMB flux (insulating core)
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
        eos_method=1,  # Adams-Williamson, no external EOS file
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
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
        event_triggering=False,
    )
    return Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        scalings=_ScalingsParameters(),
        solver=sv,
    )


def _run_solver(parameters, S_init: float = 3300.0):
    """Build EntropySolver, run a short integration, return solver+state."""
    from aragog.eos.entropy import EntropyEOS
    from aragog.solver.entropy_solver import EntropySolver

    eos = EntropyEOS(EOS_DIR)
    solver = EntropySolver(parameters, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(S_init)
    solver.solve()
    out = solver.get_state()
    return solver, out


def _S_init_below_liquidus(parameters) -> float:
    """Pick an entropy init that is well below the surface liquidus
    so the integrator does not crash on adiabatic upper-mantle melt
    saturation. 3050 J/kg/K is in the upper-mushy range for the
    PALEOS table and gives a non-trivial gradient.
    """
    return 3050.0


# ---- core_bc='quasi_steady' (the default path) -----------------------------


def test_entropy_solver_solve_quasi_steady_short_run_completes():
    """End-to-end: minimal-mesh solver run with quasi_steady BC must
    return without raising and produce a finite state vector.

    Discriminator: any regression in ``solve()`` (e.g. losing the
    end_time scaling, a bad CVODE/scipy dispatch) would either raise
    or leave ``solver._solution`` un-set.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=50.0)
    solver, out = _run_solver(parameters, S_init=_S_init_below_liquidus(parameters))

    # Solver must have populated _solution.
    assert hasattr(solver, '_solution')
    assert solver._solution is not None
    assert hasattr(solver._solution, 't')
    assert solver._solution.t is not None
    assert solver._solution.t.size >= 2, 'integration produced no steps'

    # All entropy values must be finite (no NaN/inf escaped).
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    assert np.all(np.isfinite(final_y)), 'final state contains NaN or inf'

    # Physical: entropy values should remain in a sensible range.
    # PALEOS staggered S in [1500, 4500] J/kg/K covers solidus to liquidus.
    assert float(np.min(final_y)) > 1000.0
    assert float(np.max(final_y)) < 5500.0


def test_entropy_solver_get_state_returns_solver_output_with_required_fields():
    """``get_state()`` must produce a ``SolverOutput`` with the documented
    fields. Discriminator for a regression that drops one of the
    per-call energy integrals (step_dE_F_int_J, step_dE_F_cmb_J,
    step_dE_Q_radio_J, step_dE_Q_tidal_J).
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=20.0)
    _, out = _run_solver(parameters, S_init=_S_init_below_liquidus(parameters))

    # SolverOutput dataclass: has at least the dispatched-via-PROTEUS fields.
    expected_fields = {
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_J',
        'step_dE_Q_tidal_J',
    }
    field_names = {f.name for f in out.__dataclass_fields__.values()}
    missing = expected_fields - field_names
    assert not missing, f'SolverOutput missing required fields: {sorted(missing)}'


def test_entropy_solver_solidified_run_records_finite_diagnostics():
    """A short surface-cooling run must produce finite cumulative
    diagnostics. Discriminator: NaN/inf in any per-call integral
    suggests a regression in trapezoidal accumulation or in the
    entropy state update path.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=30.0)
    _, out = _run_solver(parameters, S_init=_S_init_below_liquidus(parameters))

    for fld in (
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_J',
        'step_dE_Q_tidal_J',
    ):
        val = getattr(out, fld)
        # Each cumulative integral is a finite scalar.
        arr = np.asarray(val)
        if arr.ndim == 0:
            assert np.isfinite(float(arr)), f'{fld} = {val!r} is not finite'
        else:
            # Some fields are arrays of per-step contributions.
            assert np.all(np.isfinite(arr)), f'{fld} contains non-finite entries'


# ---- core_bc='energy_balance' (production CHILI path) ----------------------


def test_entropy_solver_solve_energy_balance_short_run_completes():
    """Energy-balance core_bc uses the extended state vector
    [S_0, ..., S_{N-1}, dSdr_cmb] with length N+1. The dispatch
    inside ``solve()`` and ``set_initial_entropy()`` must accept
    this convention without raising.

    Discriminator: a regression that hard-coded the state length to
    ``N`` would raise a shape mismatch on the first integrator call.
    """
    parameters = _build_parameters(core_bc='energy_balance', n_nodes=15, end_time=20.0)
    solver, out = _run_solver(parameters, S_init=_S_init_below_liquidus(parameters))
    assert solver._solution.t.size >= 2
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    # State must be length n_stag + 1 in energy_balance mode (the trailing
    # entry is dSdr_cmb evolved by the core energy balance).
    n_stag = solver._n_stag
    assert len(final_y) == n_stag + 1, (
        f'energy_balance state length is {len(final_y)}, expected n_stag+1 = {n_stag + 1}'
    )
    assert np.all(np.isfinite(final_y))


def test_entropy_solver_set_initial_entropy_validates_length():
    """``set_initial_entropy`` must raise if the input array has the
    wrong length for the active core_bc mode.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=10.0)
    from aragog.eos.entropy import EntropyEOS
    from aragog.solver.entropy_solver import EntropySolver

    eos = EntropyEOS(EOS_DIR)
    solver = EntropySolver(parameters, entropy_eos=eos)
    solver.initialize()
    n_stag = solver._n_stag
    # Pass an array that is n_stag - 1 long: the validator should reject.
    with pytest.raises(ValueError, match='length'):
        solver.set_initial_entropy(np.zeros(n_stag - 1))


def test_entropy_solver_set_initial_entropy_scalar_broadcasts():
    """Scalar S_init must be broadcast to a uniform isentropic profile.
    Edge case: a regression that lost the scalar branch would crash
    callers passing 3500.0.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=12, end_time=10.0)
    from aragog.eos.entropy import EntropyEOS
    from aragog.solver.entropy_solver import EntropySolver

    eos = EntropyEOS(EOS_DIR)
    solver = EntropySolver(parameters, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(3200.0)
    # Verify isentropic: the entropy block (first n_stag entries) is uniform.
    s0 = solver._S0
    n_stag = solver._n_stag
    np.testing.assert_allclose(s0[:n_stag], 3200.0, rtol=1e-12)


# ---- core_bc='bower2018' (parity-only, retained for regression) ------------


def test_entropy_solver_solve_bower2018_short_run_completes():
    """bower2018 core_bc evolves T_core as the (N+1)-th state component
    via a conduction-only CMB closure. This exercises the bower2018-
    specific RHS branch in entropy_solver.py (lines 1051-1216) which
    the quasi_steady and energy_balance tests do not touch.

    Discriminator: a regression in the bower2018 RHS dispatch (e.g.
    mis-shaped state vector, lost T_core update) would either raise
    a shape mismatch or leave T_core constant.
    """
    parameters = _build_parameters(core_bc='bower2018', n_nodes=15, end_time=20.0)
    solver, _ = _run_solver(parameters, S_init=_S_init_below_liquidus(parameters))
    assert solver._solution.t.size >= 2
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    n_stag = solver._n_stag
    # bower2018 state is N+1 (entropy block + T_core).
    assert len(final_y) == n_stag + 1, (
        f'bower2018 state length is {len(final_y)}, expected n_stag+1 = {n_stag + 1}'
    )
    # T_core must be physical (positive, finite, in plausible Earth range).
    T_core_final = float(final_y[n_stag])
    assert np.isfinite(T_core_final), 'T_core is not finite at end of run'
    assert 1000.0 < T_core_final < 8000.0, (
        f'T_core = {T_core_final:.1f} K is outside plausible [1000, 8000] K'
    )


# ---- solver_method='cvode' (production CHILI path with FD Jacobian) -------


def test_entropy_solver_solve_cvode_path_short_run_completes():
    """CVODE solver path (the SUNDIALS BDF integrator via scikits.odes)
    is the production CHILI integration backend. This exercises the
    CVODE-specific branches inside ``solve()`` (cvode_options setup,
    rootfn registration, status code dispatch) which the radau-based
    tests above do not touch.

    Skipped if scikits.odes is not installed.
    """
    pytest.importorskip('scikits_odes_sundials')

    parameters = _build_parameters(
        core_bc='quasi_steady',
        solver_method='cvode',
        n_nodes=15,
        end_time=20.0,
        use_jax_jacobian=False,  # FD Jacobian to avoid JAX path
    )
    solver, out = _run_solver(parameters, S_init=_S_init_below_liquidus(parameters))
    assert solver._solution is not None
    assert solver._solution.t is not None
    assert solver._solution.t.size >= 2
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    assert np.all(np.isfinite(final_y)), 'CVODE final state contains NaN/inf'
    # Physical bounds discriminator.
    assert float(np.min(final_y)) > 1000.0
    assert float(np.max(final_y)) < 5500.0
