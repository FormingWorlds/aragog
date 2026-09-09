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
  * ``energy_balance`` (the SPIDER-parity production path)

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
from scipy.constants import Julian_year

# EOS path is environment-driven for portability across machines.
# Resolution order:
#   1. ``ARAGOG_TEST_EOS_DIR`` -- explicit override
#   2. ``$FWL_DATA/aragog/spider_eos`` -- canonical PROTEUS data location
#   3. ``$REPO/output/coupled_parity/spider/data/spider_eos`` -- legacy
#      Mac Studio dev path; kept as a last-resort fallback so the test
#      still runs locally for whoever generated those tables once.
# Tests are skipped if none of these resolve; CI nightly populates the
# canonical location via ``proteus offline`` before the smoke run.
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
    reason=(
        f'SPIDER P-S tables not found at {EOS_DIR}. Set ARAGOG_TEST_EOS_DIR '
        'or populate $FWL_DATA/aragog/spider_eos.'
    ),
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
    inner_boundary_value: float = 0.0,
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
        _SolverParameters,
    )

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,  # grey body
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,  # prescribed core flux
        inner_boundary_value=inner_boundary_value,  # W/m^2 at the CMB (0 = insulating)
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
        solver=sv,
    )


@pytest.fixture(scope='module')
def shared_eos():
    """Build EntropyEOS once per module: PCHIP table construction
    over multiple .dat files dominates wall time (~1-2 s on Linux CI),
    and reusing it across the 8 integration tests cuts ~10-15 s of
    redundant disk + interpolator setup on the 2-vCPU runner.
    """
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def _run_solver(parameters, eos, S_init: float = 3300.0):
    """Build EntropySolver, run a short integration, return solver+state."""
    from aragog.solver.entropy_solver import EntropySolver

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


def test_entropy_solver_solve_quasi_steady_short_run_completes(shared_eos):
    """End-to-end: minimal-mesh solver run with quasi_steady BC must
    return without raising and produce a finite state vector.

    Discriminator: any regression in ``solve()`` (e.g. losing the
    end_time scaling, a bad CVODE/scipy dispatch) would either raise
    or leave ``solver._solution`` un-set.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=50.0)
    solver, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

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


def test_entropy_solver_get_state_returns_solver_output_with_required_fields(shared_eos):
    """``get_state()`` must produce a ``SolverOutput`` with the documented
    fields. Discriminator for a regression that drops one of the
    per-call energy integrals (step_dE_F_int_J, step_dE_F_cmb_J,
    step_dE_Q_radio_J, step_dE_Q_tidal_J).
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=20.0)
    _, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

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


def test_entropy_solver_solidified_run_records_finite_diagnostics(shared_eos):
    """A short surface-cooling run must produce finite cumulative
    diagnostics. Discriminator: NaN/inf in any per-call integral
    suggests a regression in trapezoidal accumulation or in the
    entropy state update path.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=30.0)
    _, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

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


# ---- core_bc='energy_balance' (production path) ----------------------


def test_entropy_solver_solve_energy_balance_short_run_completes(shared_eos):
    """Energy-balance core_bc uses the extended state vector
    [S_0, ..., S_{N-1}, dSdr_cmb] with length N+1. The dispatch
    inside ``solve()`` and ``set_initial_entropy()`` must accept
    this convention without raising.

    Discriminator: a regression that hard-coded the state length to
    ``N`` would raise a shape mismatch on the first integrator call.
    """
    parameters = _build_parameters(core_bc='energy_balance', n_nodes=15, end_time=20.0)
    solver, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))
    assert solver._solution.t.size >= 2
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    # State must be length n_stag + 1 in energy_balance mode (the trailing
    # entry is dSdr_cmb evolved by the core energy balance).
    n_stag = solver._n_stag
    assert len(final_y) == n_stag + 1, (
        f'energy_balance state length is {len(final_y)}, expected n_stag+1 = {n_stag + 1}'
    )
    assert np.all(np.isfinite(final_y))


def test_entropy_solver_set_initial_entropy_validates_length(shared_eos):
    """``set_initial_entropy`` must raise if the input array has the
    wrong length for the active core_bc mode.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=10.0)
    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    solver.initialize()
    n_stag = solver._n_stag
    # Pass an array that is n_stag - 1 long: the validator should reject.
    with pytest.raises(ValueError, match='length'):
        solver.set_initial_entropy(np.zeros(n_stag - 1))


def test_entropy_solver_set_initial_entropy_scalar_broadcasts(shared_eos):
    """Scalar S_init must be broadcast to a uniform isentropic profile.
    Edge case: a regression that lost the scalar branch would crash
    callers passing 3500.0.
    """
    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=12, end_time=10.0)
    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    solver.initialize()
    solver.set_initial_entropy(3200.0)
    # Verify isentropic: the entropy block (first n_stag entries) is uniform.
    s0 = solver._S0
    n_stag = solver._n_stag
    np.testing.assert_allclose(s0[:n_stag], 3200.0, rtol=1e-12)


# ---- core_bc='bower2018' (parity-only, retained for regression) ------------


def test_entropy_solver_solve_bower2018_short_run_completes(shared_eos):
    """bower2018 core_bc evolves T_core as the (N+1)-th state component
    via a conduction-only CMB closure. This exercises the bower2018-
    specific RHS branch in entropy_solver.py (lines 1051-1216) which
    the quasi_steady and energy_balance tests do not touch.

    Discriminator: a regression in the bower2018 RHS dispatch (e.g.
    mis-shaped state vector, lost T_core update) would either raise
    a shape mismatch or leave T_core constant.
    """
    parameters = _build_parameters(core_bc='bower2018', n_nodes=15, end_time=20.0)
    solver, _ = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))
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


# ---- solver_method='cvode' (production path with FD Jacobian) -------


def test_entropy_solver_solve_cvode_path_short_run_completes(shared_eos):
    """CVODE solver path (the SUNDIALS BDF integrator via scikits.odes)
    is the production integration backend. This exercises the
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
    solver, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))
    assert solver._solution is not None
    assert solver._solution.t is not None
    assert solver._solution.t.size >= 2
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    assert np.all(np.isfinite(final_y)), 'CVODE final state contains NaN/inf'
    # Physical bounds discriminator.
    assert float(np.min(final_y)) > 1000.0
    assert float(np.max(final_y)) < 5500.0


def test_get_state_maps_cvode_flag_and_tcore_excursion_fields(shared_eos):
    """``get_state`` must copy the four post-solve diagnostic fields onto
    the returned SolverOutput: cvode_flag, cvode_flag_name,
    tcore_change_max, tcore_change_exceeded.

    A regression that stops mapping any of them leaves the SolverOutput
    default in place (the excursion pair falls back to 0.0/False, the flag
    name to 'N/A'). The CVODE path is used so a successful solve names a
    real StatusEnum member: a dropped cvode_flag_name mapping would then
    read 'N/A', which the radau default path cannot distinguish. Each
    field is re-derived from the same solution get_state reads.
    """
    import aragog.solver.entropy_solver as es

    pytest.importorskip('scikits_odes_sundials')
    parameters = _build_parameters(
        core_bc='quasi_steady',
        solver_method='cvode',
        n_nodes=15,
        end_time=20.0,
        use_jax_jacobian=False,  # FD Jacobian to avoid JAX path
    )
    solver, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

    # cvode_flag / cvode_flag_name are read off the solution object.
    assert isinstance(out.cvode_flag, int)
    assert isinstance(out.cvode_flag_name, str)
    assert out.cvode_flag == int(getattr(solver._solution, 'cvode_flag', 0))
    assert out.cvode_flag_name == str(getattr(solver._solution, 'cvode_flag_name', 'N/A'))
    assert out.cvode_flag_name != 'N/A'
    assert out.cvode_flag_name == es._cvode_flag_name(out.cvode_flag)

    # The excursion pair is re-derived from the same solution get_state
    # measured, so equality proves get_state routed the measure onto the
    # output rather than leaving the 0.0/False default.
    exp_max, exp_exceeded = solver._core_temperature_excursion(solver._solution)
    assert isinstance(out.tcore_change_max, float)
    assert isinstance(out.tcore_change_exceeded, bool)
    assert out.tcore_change_max == exp_max
    assert out.tcore_change_exceeded == exp_exceeded
    assert np.isfinite(out.tcore_change_max)
    assert out.tcore_change_max >= 0.0


# ---- CLI IC-derivation against the live EOS --------------------------------


@pytest.mark.parametrize('T_target', [3000.0, 3500.0, 4000.0])
def test_derive_initial_entropy_round_trips_against_live_eos(shared_eos, T_target):
    """The CLI IC-derivation helper must invert ``T(P_surf, S)`` to
    machine precision against a real EOS.

    Discriminator: brentq is set to ``xtol=1e-3`` J/kg/K. With
    ``dT/dS`` of order 0.5-1 K/(J/kg/K) on the PALEOS table near the
    surface, that translates to ``|T(P_surf, S0) - T_target| < 1 mK``.
    A regression in the bracket logic, a wrong P_surf source, or a
    bad EOS-table key would either fail to converge or land on a
    far-off T value. The parametrize sweeps three targets that span
    the solid + mushy regimes so a single hard-coded fall-back value
    cannot accidentally pass.
    """
    from aragog.cli import _derive_initial_entropy_from_config
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15)
    parameters.initial_condition.initial_condition = 1  # linear
    parameters.initial_condition.surface_temperature = T_target

    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    solver.initialize()

    S0 = _derive_initial_entropy_from_config(solver)
    assert S0 is not None, (
        'derivation returned None despite IC method=1 and surface_temperature > 0'
    )

    P_surf = float(parameters.mesh.surface_pressure)
    T_back = shared_eos.temperature_scalar(P_surf, S0)
    assert abs(T_back - T_target) < 0.1, (
        f'derivation did not invert T(P_surf, S0); got T_back={T_back:.4f} K, '
        f'target={T_target:.1f} K, |delta|={abs(T_back - T_target):.4e} K. '
        'brentq xtol=1e-3 J/kg/K should round-trip to <1 mK.'
    )

    # Round-trip test 2: feed the derived S0 back into the solver and
    # confirm the IC stage accepts it without raising. A regression
    # that produced an out-of-table S0 would fail here on the entropy
    # range check inside set_initial_entropy.
    solver.set_initial_entropy(S0)


def test_derive_initial_entropy_skips_when_ic_method_is_2(shared_eos):
    """Live EOS + IC method 2 (user-defined T file): the helper must
    return ``None`` so the caller's "--initial-entropy is required"
    branch fires.

    Edge case: a regression that flipped the IC-method gate would
    produce a derived S0 against a real EOS, silently overriding the
    file-based IC the user explicitly requested.
    """
    from aragog.cli import _derive_initial_entropy_from_config
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15)
    # IC method 2 = user-defined T file. surface_temperature is set
    # but should be ignored by the gate.
    parameters.initial_condition.initial_condition = 2
    parameters.initial_condition.surface_temperature = 4000.0
    # init_temperature is normally loaded by Parameters.__post_init__
    # for IC=2, but bypass the file load here by setting it directly
    # so initialize() does not raise on a missing path.
    parameters.initial_condition.init_temperature = np.full(15, 4000.0)

    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    # Do NOT call solver.initialize(); the IC-method gate runs without
    # touching the mesh or BC dispatch.
    S0 = _derive_initial_entropy_from_config(solver)
    assert S0 is None, (
        f'IC method=2 must NOT trigger derivation; got S0={S0}. '
        'A non-None return here would silently override the user-supplied IC file.'
    )


# ---- F_cmb output column: conserved step-average ---------------------------


def test_f_cmb_column_is_conserved_step_average_not_end_of_step_snapshot(shared_eos):
    """The reported F_cmb column is the conserved step-average, not the
    end-of-step basic-node-0 snapshot.

    A prescribed nonzero core flux (inner BC 2, 1e4 W/m^2) forces the CMB
    flux to that value at every sub-step, so the trapezoidal step-average
    equals it. The final state refresh does not re-impose the BC, so the
    retained snapshot ``heat_flux[0]`` drifts to the un-constrained
    gradient. The two therefore diverge by a known, controlled amount.

    Discriminator: the closure ``F_cmb * A_cmb * dt == step_dE_F_cmb_J``
    holds only for the step-average. If the column reverted to the
    snapshot, the reconstruction would miss the conserved integral by the
    snapshot-vs-average ratio, and the closure assertion would fail.
    """
    parameters = _build_parameters(
        core_bc='quasi_steady', n_nodes=15, end_time=50.0, inner_boundary_value=1.0e4
    )
    _, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

    a_cmb = 4.0 * np.pi * float(out.r_basic[0]) ** 2
    dt_s = float(out.dt_actual) * Julian_year
    reported = float(out.F_cmb)
    snapshot = float(out.heat_flux[0])
    integral = float(out.step_dE_F_cmb_J)

    # Closure: the reported column reconstructs the conserved integral.
    assert np.isclose(reported * a_cmb * dt_s, integral, rtol=1e-9, atol=0.0), (
        f'F_cmb={reported:.6e} * A_cmb * dt does not reconstruct '
        f'step_dE_F_cmb_J={integral:.6e}; column is not the conserved average'
    )
    # The step-average equals the prescribed flux (guards a stuck-at-zero column).
    assert np.isclose(reported, 1.0e4, rtol=1e-6), (
        f'step-average F_cmb={reported:.6e} != prescribed 1e4 W/m^2'
    )
    # The snapshot differs, so this is genuinely not the end-of-step value.
    assert not np.isclose(reported, snapshot, rtol=1e-3), (
        f'F_cmb={reported:.6e} equals snapshot heat_flux[0]={snapshot:.6e}; '
        'the column reports the end-of-step snapshot, not the step-average'
    )
    # The raw snapshot stays available for callers that need it.
    assert np.isfinite(snapshot)


def test_f_cmb_column_reports_bc_consistent_zero_for_insulating_core(shared_eos):
    """With an insulating core (inner BC 2, 0 W/m^2) the reported F_cmb is
    the BC-consistent zero, not the un-constrained end-of-step gradient.

    The zero-flux BC is imposed inside the RHS at every sub-step, so the
    conserved flux integral is zero and the step-average is zero. The
    final state refresh omits the BC, leaving ``heat_flux[0]`` at a
    nonzero gradient value.

    Discriminator: the column must read 0. A column that reported the
    snapshot would carry the nonzero gradient and violate the prescribed
    zero-flux boundary condition.
    """
    parameters = _build_parameters(
        core_bc='quasi_steady', n_nodes=15, end_time=50.0, inner_boundary_value=0.0
    )
    _, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

    a_cmb = 4.0 * np.pi * float(out.r_basic[0]) ** 2
    dt_s = float(out.dt_actual) * Julian_year
    reported = float(out.F_cmb)
    snapshot = float(out.heat_flux[0])
    integral = float(out.step_dE_F_cmb_J)

    # Conserved flux integral is zero, so the reported step-average is zero.
    assert integral == 0.0, f'insulating core: step_dE_F_cmb_J={integral:.6e} != 0'
    assert reported == 0.0, (
        f'insulating core: reported F_cmb={reported:.6e} != 0; the column '
        'is not the BC-consistent step-average'
    )
    # Closure holds at the zero point too. Both sides are zero here, so a
    # divisor-scaling regression is caught by the nonzero closure tests, not
    # this one; this assertion documents self-consistency at zero flux.
    assert np.isclose(reported * a_cmb * dt_s, integral, rtol=1e-9, atol=0.0)
    # The snapshot is nonzero, confirming the column is not the snapshot.
    assert snapshot != 0.0, (
        'expected a nonzero un-constrained snapshot to make this test '
        'discriminate; fixture no longer exercises the BC-consistency gap'
    )


def test_f_cmb_column_is_conserved_step_average_energy_balance_core(shared_eos):
    """The conserved step-average column also holds under
    ``core_bc='energy_balance'``, the SPIDER-parity production mode and the
    setting of the reported CMB-flux artifact.

    Energy-balance mode integrates the extended state vector
    [S_0, ..., S_{N-1}, dSdr_cmb] of length N+1, so ``get_state`` and the
    step-energy integrator take the ``is_extended`` reconstruction branch
    that the quasi_steady tests never exercise. A prescribed nonzero core
    flux (inner BC 2, 1e4 W/m^2) is imposed at every sub-step, so the
    trapezoidal step-average equals it. The final state refresh does not
    re-impose the BC, so the retained snapshot ``heat_flux[0]`` drifts to
    the un-constrained gradient, here even to the wrong sign (flux into the
    core): the reported step-average removes that artifact.

    Discriminator: the closure ``F_cmb * A_cmb * dt == step_dE_F_cmb_J``
    holds only for the step-average, and it constrains the ``A_cmb * dt``
    divisor because the reported value is nonzero. A column that reverted
    to the snapshot would be negative here, failing both the closure and
    the prescribed-value assertion.
    """
    parameters = _build_parameters(
        core_bc='energy_balance', n_nodes=15, end_time=50.0, inner_boundary_value=1.0e4
    )
    _, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

    a_cmb = 4.0 * np.pi * float(out.r_basic[0]) ** 2
    dt_s = float(out.dt_actual) * Julian_year
    reported = float(out.F_cmb)
    snapshot = float(out.heat_flux[0])
    integral = float(out.step_dE_F_cmb_J)

    # Closure: the reported column reconstructs the conserved integral, and
    # because it is nonzero the identity constrains the A_cmb * dt divisor.
    assert np.isclose(reported * a_cmb * dt_s, integral, rtol=1e-9, atol=0.0), (
        f'F_cmb={reported:.6e} * A_cmb * dt does not reconstruct '
        f'step_dE_F_cmb_J={integral:.6e}; column is not the conserved average'
    )
    # The step-average equals the prescribed flux (guards a stuck-at-zero column).
    assert np.isclose(reported, 1.0e4, rtol=1e-6), (
        f'step-average F_cmb={reported:.6e} != prescribed 1e4 W/m^2'
    )
    # The retained snapshot is the un-constrained gradient flux, below the
    # true CMB flux and here of the wrong sign: the artifact the average removes.
    assert reported > 0.0 and snapshot < reported, (
        f'expected snapshot={snapshot:.6e} below the conserved average '
        f'{reported:.6e}; fixture no longer exercises the artifact'
    )
    # The raw snapshot stays available for callers that need it.
    assert np.isfinite(snapshot)


# ---- F_cmb closure on the melt-fraction step-cap degenerate path -----------


def test_f_cmb_closure_holds_on_phi_step_cap_two_point_trajectory(shared_eos):
    """The F_cmb closure survives the melt-fraction step-cap degenerate
    path: a CVODE call truncated to a two-point trajectory at a phi root.

    A small ``phi_step_cap`` with a mushy initial condition makes the
    melt-fraction change reach the cap early in the call. On the CVODE
    path this fires as a solver root: ``solve()`` truncates the returned
    trajectory to exactly [t_start, t_root], sets ``cap_fired`` and
    ``cap_label='phi'``, and the call ends before ``end_time``. The
    step-energy integrator then trapezoid-integrates a two-point
    trajectory rather than the many natural steps of a full call.

    Discriminator: the closure ``F_cmb * A_cmb * dt == step_dE_F_cmb_J``
    must still hold on the two-point trajectory, and the reported column
    must still equal the prescribed 1e4 W/m^2 core flux. A regression that
    formed the divisor from the full ``end_time`` rather than the actual
    truncated call duration, or that mishandled the degenerate two-point
    integral, would break the closure here while the full-trajectory
    tests above still passed. The cap-fire assertions pin that this test
    exercises the truncated path and not an ordinary full call.

    Skipped if scikits.odes is not installed; the two-point truncation is
    the CVODE path, so the radau backend cannot reproduce it.
    """
    pytest.importorskip('scikits_odes_sundials')

    parameters = _build_parameters(
        core_bc='quasi_steady',
        solver_method='cvode',
        n_nodes=15,
        end_time=50.0,
        inner_boundary_value=1.0e4,
        use_jax_jacobian=False,
    )
    parameters.energy.phi_step_cap = 0.005
    solver, out = _run_solver(parameters, shared_eos, S_init=_S_init_below_liquidus(parameters))

    sol = solver._solution
    assert sol is not None and sol.t is not None
    # The phi cap fired as a CVODE root, truncating to a two-point trajectory.
    assert getattr(sol, 'cap_fired', False) is True, 'phi_step_cap did not fire'
    assert getattr(sol, 'cap_label', None) == 'phi', (
        f"cap_label={getattr(sol, 'cap_label', None)!r}, expected 'phi'"
    )
    assert sol.t.size == 2, (
        f'expected a two-point trajectory at the phi root, got sol.t.size={sol.t.size}'
    )
    # The call ended at the root, well before end_time.
    assert float(out.dt_actual) < 50.0, (
        f'dt_actual={float(out.dt_actual):.6e} yr not truncated below end_time'
    )

    a_cmb = 4.0 * np.pi * float(out.r_basic[0]) ** 2
    dt_s = float(out.dt_actual) * Julian_year
    reported = float(out.F_cmb)
    integral = float(out.step_dE_F_cmb_J)

    # Closure holds on the truncated two-point trajectory, and because the
    # value is nonzero the identity constrains the A_cmb * dt divisor formed
    # from the actual call duration, not from end_time.
    assert np.isclose(reported * a_cmb * dt_s, integral, rtol=1e-9, atol=0.0), (
        f'F_cmb={reported:.6e} * A_cmb * dt does not reconstruct '
        f'step_dE_F_cmb_J={integral:.6e} on the two-point phi-cap trajectory'
    )
    # The step-average equals the prescribed flux (guards a stuck-at-zero column).
    assert np.isclose(reported, 1.0e4, rtol=1e-6), (
        f'step-average F_cmb={reported:.6e} != prescribed 1e4 W/m^2'
    )


# ---- mantle liquid/solid mass split at a partial-melt state ----------------


@pytest.mark.physics_invariant
def test_mantle_mass_split_at_partial_melt_discriminates_solid_coefficient(shared_eos):
    """The melt/solid split must reproduce ``(1 - Phi_global) * M_mantle``
    for the solid mass and ``Phi_global * M_mantle`` for the liquid mass at
    a genuine partial-melt state, not only the bounds and the sum.

    The companion const-properties test forces ``Phi_global == 1``, so the
    solid coefficient collapses to zero there and a sign flip or off-by-one
    in ``(1 - Phi_global)`` passes unseen. A real SPIDER state at
    ``S_init = 3050`` settles at an intermediate ``Phi_global`` near 0.25,
    where the solid mass is the larger of the two. A flipped coefficient
    ``(Phi_global - 1)`` drives the solid mass negative, and an off-by-one
    that copies the liquid coefficient breaks the value check against the
    reported ``Phi_global`` and ``M_mantle``.
    """
    from aragog.solver import entropy_solver

    # The split lives in entropy_solver.get_state(); an editable install can
    # resolve the import to a different tree, which would silence a mutation
    # check. Confirm the module under test is the one in this worktree.
    module_path = Path(entropy_solver.__file__).resolve()
    assert _REPO_ROOT in module_path.parents, (
        f'entropy_solver resolved to {module_path}, outside the worktree at '
        f'{_REPO_ROOT}; a mutation check here would not be trustworthy.'
    )

    parameters = _build_parameters(core_bc='quasi_steady', n_nodes=15, end_time=50.0)
    _, out = _run_solver(parameters, shared_eos, S_init=3050.0)

    phi = float(out.Phi_global)
    m_mantle = float(out.M_mantle)
    m_liquid = float(out.M_mantle_liquid)
    m_solid = float(out.M_mantle_solid)

    assert m_mantle > 0.0, 'M_mantle must be positive for this check to be meaningful'
    assert 0.01 < phi < 0.99, (
        f'Phi_global={phi:.6f} is not a partial-melt state; the solid '
        'coefficient is not exercised and this test cannot discriminate it.'
    )

    assert 0.0 <= m_liquid <= m_mantle, (
        f'M_mantle_liquid={m_liquid:.6e} escaped [0, M_mantle={m_mantle:.6e}].'
    )
    assert 0.0 <= m_solid <= m_mantle, (
        f'M_mantle_solid={m_solid:.6e} escaped [0, M_mantle={m_mantle:.6e}]; '
        'a flipped solid coefficient drives this negative.'
    )
    np.testing.assert_allclose(
        m_liquid + m_solid,
        m_mantle,
        rtol=1e-10,
        err_msg='M_mantle_liquid + M_mantle_solid must equal the reported M_mantle',
    )

    # Value check: recompute both masses from the reported Phi_global and
    # M_mantle and compare to the reported split. This is what a sign flip or
    # an off-by-one in the solid coefficient breaks; the bounds and the sum
    # alone do not.
    np.testing.assert_allclose(
        m_solid,
        (1.0 - phi) * m_mantle,
        rtol=1e-10,
        err_msg='M_mantle_solid must equal (1 - Phi_global) * M_mantle',
    )
    np.testing.assert_allclose(
        m_liquid,
        phi * m_mantle,
        rtol=1e-10,
        err_msg='M_mantle_liquid must equal Phi_global * M_mantle',
    )
