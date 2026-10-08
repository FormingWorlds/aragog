"""Tests for batched per-call energy power evaluation in Aragog.

Verifies:
- Per-node power parity between JAX batch and numpy step powers
  across 8 combinations of mode, radio heating, and tidal heating for n > 512.
- Compile counter verifies exactly one trace across varying trace lengths
  and one additional trace for a second mode.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from aragog.jax.nondim import NonDimScales  # noqa: E402
from aragog.jax.phase import MeshArrays, PhaseParams  # noqa: E402
from aragog.jax.solver import BoundaryParams, StepPowersAux  # noqa: E402
from aragog.parser import Parameters, _Radionuclide  # noqa: E402
from aragog.solver import entropy_solver as es  # noqa: E402
from aragog.solver.cvode_jax import (  # noqa: E402
    _TRACE_COUNTERS,
    build_jax_rhs_and_jacobian,
    clear_jit_cache,
)
from tests.conftest import EOS_DIR, entropy_eos_copy, entropy_eos_jax  # noqa: E402
from tests.test_entropy_solver_integration import _build_parameters  # noqa: E402

jax.config.update('jax_enable_x64', True)

pytestmark = [
    pytest.mark.skipif(
        not EOS_DIR.exists(),
        reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
    ),
    pytest.mark.skipif(
        es._scikits_cvode is None,
        reason='scikits-odes-sundials not installed',
    ),
]


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_jit_cache()


@pytest.fixture(scope='module')
def eos_np():
    """Return numpy EntropyEOS with alpha derived from thermodynamic identity."""
    eos = entropy_eos_copy(EOS_DIR)
    eos._has_alpha_tables = False
    eos._tables.pop('thermal_exp_solid', None)
    eos._tables.pop('thermal_exp_melt', None)
    assert not eos._has_alpha_tables
    return eos


@pytest.fixture(scope='module')
def eos_jax():
    """Return JAX EntropyEOS."""
    return entropy_eos_jax(EOS_DIR)


def _make_radionuclide() -> _Radionuclide:
    return _Radionuclide(
        name='Al26',
        t0_years=0.0,
        abundance=1.0,
        concentration=1.0,
        heat_production=3.5e-12,
        half_life_years=7.17e5,
    )


def _make_radio_tuple(radionuclides: list[_Radionuclide]) -> tuple:
    if not radionuclides:
        return ()
    return (
        np.array([r.heat_production for r in radionuclides]),
        np.array([r.abundance for r in radionuclides]),
        np.array([r.concentration for r in radionuclides]),
        np.array([r.t0_years for r in radionuclides]),
        np.array([r.half_life_years for r in radionuclides]),
    )


def _make_boundary_params(mesh_arr: MeshArrays, mode: str, inner_val: float) -> BoundaryParams:
    inner_type = 5 if mode == 'energy_balance' else 2
    r_cmb = float(mesh_arr.radii_basic[0])
    r_next = float(mesh_arr.radii_basic[1])
    return BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=1500.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=inner_type,
        inner_bc_value=inner_val,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=(4.0 / 3.0) * np.pi * r_cmb**3 * 10500.0,
        cmb_dr_cmb=r_next - r_cmb,
        param_utbl=False,
        param_utbl_const=0.0,
    )


def _set_initial_entropy(solver, s_base: float = 3050.0, s_span: float = 150.0) -> None:
    r = np.asarray(solver._r_stag_flat)
    solver.set_initial_entropy(s_base + s_span * (r[-1] - r) / (r[-1] - r[0]))


def _spy_step_powers(solver) -> list[int]:
    call_count = [0]
    orig_sp = solver._step_powers

    def spy(t, y):
        call_count[0] += 1
        return orig_sp(t, y)

    solver._step_powers = spy
    return call_count


def _make_step_powers_aux(mesh) -> StepPowersAux:
    vol = np.asarray(mesh.basic.volume).ravel()
    r_b = np.asarray(mesh.basic.radii).ravel()
    mass_struct = np.asarray(mesh.staggered_effective_density).ravel() * vol
    return StepPowersAux(
        A_int=4.0 * np.pi * float(r_b[-1]) ** 2,
        A_cmb=4.0 * np.pi * float(r_b[0]) ** 2,
        volume=vol,
        mass_struct=mass_struct,
        P_stag=np.asarray(mesh.staggered_pressure).ravel(),
    )


def _assert_powers_match(P_jax: np.ndarray, P_np: np.ndarray) -> None:
    for col in range(6):
        colmax = float(np.max(np.abs(P_np[:, col])))
        np.testing.assert_allclose(
            P_jax[:, col],
            P_np[:, col],
            rtol=1e-12,
            atol=1e-12 * colmax,
        )
    resid_atol = 1e-12 * (float(np.max(np.abs(P_np[:, 0]))) + float(np.max(np.abs(P_np[:, 1]))))
    np.testing.assert_allclose(
        P_jax[:, 6],
        P_np[:, 6],
        atol=resid_atol,
    )


def _make_phase_params(params: Parameters) -> PhaseParams:
    return PhaseParams(
        phi_rheo=float(params.phase_mixed.rheological_transition_melt_fraction),
        phi_width=float(params.phase_mixed.rheological_transition_width),
        viscosity_solid=float(params.phase_solid.viscosity),
        viscosity_liquid=float(params.phase_liquid.viscosity),
        grain_size=float(params.phase_mixed.grain_size),
        k_solid=float(params.phase_solid.thermal_conductivity),
        k_liquid=float(params.phase_liquid.thermal_conductivity),
        matprop_smooth_width=float(params.phase_mixed.matprop_smooth_width),
        conduction=params.energy.conduction,
        convection=params.energy.convection,
        grav_sep=params.energy.gravitational_separation,
        mixing=params.energy.mixing,
        eddy_diff_thermal=float(params.energy.eddy_diffusivity_thermal),
        eddy_diff_chemical=float(params.energy.eddy_diffusivity_chemical),
        kappah_floor=float(params.energy.kappah_floor),
        bottom_up_grav_sep=params.energy.bottom_up_grav_sep,
        phase_smoothing=params.energy.phase_smoothing,
        separation_viscosity=params.phase_mixed.separation_viscosity,
    )


def _make_cvode_jax_factory(solver, eos_jax, params, mode, tidal_arr, radio_params, inner_val):
    mesh = solver.evaluator.mesh
    mesh_arr = MeshArrays.from_numpy_mesh(mesh)
    bc = _make_boundary_params(mesh_arr, mode, inner_val)
    phase_params = _make_phase_params(params)

    def factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=phase_params,
            mesh_arrays=mesh_arr,
            boundary_params=bc,
            heating_array=tidal_arr,
            scales=scales,
            core_bc_mode=core_bc_mode,
            radio_isotope_params=radio_params,
        )
        return rhs_fn, jac_fn

    return factory


@pytest.mark.smoke
@pytest.mark.parametrize('mode', ['quasi_steady', 'energy_balance'])
@pytest.mark.parametrize('use_radio', [False, True])
@pytest.mark.parametrize('use_tidal', [False, True])
def test_step_powers_per_node_parity(
    eos_np, eos_jax, mode: str, use_radio: bool, use_tidal: bool
):
    """Verify per-node power parity between JAX batch and numpy step powers.

    Checks columns 0 to 5 relative differences <= 1e-12 against numpy with
    derived alpha, and residual consistency on column 6.
    """
    assert not eos_np._has_alpha_tables

    n_nodes = 15
    n_stag = n_nodes - 1
    inner_bc = 1 if mode == 'energy_balance' else 2
    inner_val = 0.05 if inner_bc == 2 else 0.0
    params = _build_parameters(
        core_bc=mode,
        solver_method='cvode',
        end_time=20.0,
        n_nodes=n_nodes,
        use_jax_jacobian=False,
        inner_boundary_condition=inner_bc,
        inner_boundary_value=inner_val,
    )

    params.phase_mixed.phase_transition_width = 0.0
    params.solver.cvode_output_points = 550

    if use_radio:
        r26 = _make_radionuclide()
        params.energy.radionuclides = True
        params.radionuclides = [r26]
    else:
        params.energy.radionuclides = False
        params.radionuclides = []

    if use_tidal:
        params.energy.tidal = True
        tidal_arr = np.linspace(1e-12, 5e-12, n_stag)
        params.energy.tidal_array = list(tidal_arr)
    else:
        params.energy.tidal = False
        params.energy.tidal_array = [0.0]
        tidal_arr = np.zeros(n_stag)

    solver = es.EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()
    _set_initial_entropy(solver, s_base=3050.0, s_span=150.0)
    solver.solve()
    sol = solver._solution
    assert sol is not None
    t_pts, y_pts = sol.energy_trace
    assert len(t_pts) > 512, f'Expected > 512 nodes, got {len(t_pts)}'

    P_np = np.array([solver._step_powers(float(t), y) for t, y in zip(t_pts, y_pts.T)])

    mesh = solver.evaluator.mesh
    mesh_arr = MeshArrays.from_numpy_mesh(mesh)
    bc = _make_boundary_params(mesh_arr, mode, inner_val)
    radio_params = _make_radio_tuple(params.radionuclides)
    phase_params = _make_phase_params(params)

    if mode == 'energy_balance':
        scale_vec = np.empty(n_stag + 1)
        scale_vec[:n_stag] = 3000.0
        scale_vec[-1] = 1e-6
    else:
        scale_vec = np.full(n_stag, 3000.0)
    scales = NonDimScales(state_scale=scale_vec, t_ref=100.0)

    rhs_fn, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=phase_params,
        mesh_arrays=mesh_arr,
        boundary_params=bc,
        heating_array=tidal_arr,
        scales=scales,
        core_bc_mode=mode,
        radio_isotope_params=radio_params,
    )

    aux = _make_step_powers_aux(mesh)
    P_jax = rhs_fn.step_powers(t_pts, y_pts, aux=aux)
    assert P_jax.shape == P_np.shape

    _assert_powers_match(P_jax, P_np)


@pytest.mark.smoke
@pytest.mark.parametrize('mode', ['quasi_steady', 'energy_balance'])
def test_step_powers_per_node_parity_smooth_width(eos_np, eos_jax, mode: str):
    """Verify per-node power parity when matprop_smooth_width is active.

    Uses matprop_smooth_width = 0.01 on both sides and an initial state with
    melt fraction strictly between 0 and 1 at >= 2 nodes.
    """
    assert not eos_np._has_alpha_tables

    n_nodes = 15
    n_stag = n_nodes - 1
    inner_bc = 1 if mode == 'energy_balance' else 2
    inner_val = 0.05 if inner_bc == 2 else 0.0
    params = _build_parameters(
        core_bc=mode,
        solver_method='cvode',
        end_time=5.0,
        n_nodes=n_nodes,
        use_jax_jacobian=False,
        inner_boundary_condition=inner_bc,
        inner_boundary_value=inner_val,
    )
    params.phase_mixed.matprop_smooth_width = 0.01
    params.phase_mixed.phase_transition_width = 0.0
    params.solver.cvode_output_points = 60

    r26 = _make_radionuclide()
    params.energy.radionuclides = True
    params.radionuclides = [r26]

    params.energy.tidal = True
    tidal_arr = np.linspace(1e-12, 5e-12, n_stag)
    params.energy.tidal_array = list(tidal_arr)

    solver = es.EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()
    _set_initial_entropy(solver, s_base=3195.0, s_span=10.0)

    r = np.asarray(solver._r_stag_flat)
    S0 = 3195.0 + 10.0 * (r[-1] - r) / (r[-1] - r[0])
    if mode == 'energy_balance':
        solver._dSdt_single(0.0, np.append(S0, 0.0))
    else:
        solver._dSdt_single(0.0, S0)
    phi0 = np.asarray(solver.state.phase_staggered.melt_fraction()).ravel()
    mixed_nodes = np.where((phi0 > 0.0) & (phi0 < 1.0))[0]
    assert len(mixed_nodes) >= 2, f'Expected >= 2 mixed nodes, got {len(mixed_nodes)}'

    solver.solve()
    sol = solver._solution
    assert sol is not None
    t_pts, y_pts = sol.energy_trace

    P_np = np.array([solver._step_powers(float(t), y) for t, y in zip(t_pts, y_pts.T)])

    mesh = solver.evaluator.mesh
    mesh_arr = MeshArrays.from_numpy_mesh(mesh)
    bc = _make_boundary_params(mesh_arr, mode, inner_val)
    phase_params = _make_phase_params(params)
    if mode == 'energy_balance':
        scale_vec = np.empty(n_stag + 1)
        scale_vec[:n_stag] = 3000.0
        scale_vec[-1] = 1e-6
    else:
        scale_vec = np.full(n_stag, 3000.0)
    scales = NonDimScales(state_scale=scale_vec, t_ref=100.0)

    radio_params = _make_radio_tuple(params.radionuclides)
    rhs_fn, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=phase_params,
        mesh_arrays=mesh_arr,
        boundary_params=bc,
        heating_array=tidal_arr,
        scales=scales,
        core_bc_mode=mode,
        radio_isotope_params=radio_params,
    )
    aux = _make_step_powers_aux(mesh)
    P_jax = rhs_fn.step_powers(t_pts, y_pts, aux=aux)

    _assert_powers_match(P_jax, P_np)


@pytest.mark.unit
def test_step_powers_compile_once(eos_jax):
    """Verify that powers JIT is compiled once per cache key."""
    from tests.conftest import make_mesh as _make_mesh
    from tests.test_cvode_jax_cache import _make_bc

    clear_jit_cache()
    assert _TRACE_COUNTERS['powers'] == 0

    params = PhaseParams()
    mesh10 = _make_mesh(N=10)
    bc1 = _make_bc(mesh10, outer_type=4, inner_type=2)
    scales1 = NonDimScales(state_scale=np.full(10, 3000.0), t_ref=100.0)

    r26 = _make_radionuclide()
    radio_params = _make_radio_tuple([r26])

    rhs_fn1, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc1,
        heating_array=np.zeros(10),
        scales=scales1,
        core_bc_mode='quasi_steady',
        radio_isotope_params=radio_params,
    )

    aux1 = StepPowersAux(
        A_int=float(mesh10.area[-1]),
        A_cmb=float(mesh10.area[0]),
        volume=mesh10.volume,
        mass_struct=np.asarray(mesh10.volume * 3300.0),
        P_stag=mesh10.P_stag,
    )

    # 5 solves with different trace lengths, one above C = 512
    lengths = [50, 100, 512, 600, 1000]
    out_1000 = None
    t_1000 = None
    Y_1000 = None
    for n in lengths:
        t_nodes = np.linspace(0.0, 10.0, n)
        Y_nodes = 3000.0 + 200.0 * np.linspace(0.0, 1.0, n)[None, :] * np.ones((10, 1))
        out = rhs_fn1.step_powers(t_nodes, Y_nodes, aux=aux1)
        assert out.shape == (n, 7)
        assert _TRACE_COUNTERS['powers'] == 1, (
            f'Expected 1 trace, got {_TRACE_COUNTERS["powers"]} at length {n}'
        )
        if n == 1000:
            out_1000 = out
            t_1000 = t_nodes
            Y_1000 = Y_nodes

    # Re-invoking factory with identical cache key reuses powers_jit with 0 new traces
    rhs_fn1_repeat, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc1,
        heating_array=np.zeros(10),
        scales=scales1,
        core_bc_mode='quasi_steady',
        radio_isotope_params=radio_params,
    )
    out_repeat = rhs_fn1_repeat.step_powers(
        np.linspace(0.0, 10.0, 50),
        3000.0 + 200.0 * np.linspace(0.0, 1.0, 50)[None, :] * np.ones((10, 1)),
        aux=aux1,
    )
    assert out_repeat.shape == (50, 7)
    assert _TRACE_COUNTERS['powers'] == 1, (
        'Expected 0 additional traces on repeated factory call'
    )

    # Compare n=1000 chunked output row by row against n=1 calls at boundary indices
    check_indices = [0, 510, 511, 512, 513, 999]
    for idx in check_indices:
        single_out = rhs_fn1.step_powers(
            np.array([t_1000[idx]]),
            Y_1000[:, idx : idx + 1],
            aux=aux1,
        )
        colmax = float(np.max(np.abs(single_out[0])))
        np.testing.assert_allclose(
            out_1000[idx],
            single_out[0],
            rtol=1e-12,
            atol=1e-12 * max(colmax, 1.0),
        )

    # Mode 2 with a new cache key must trace exactly once more
    bc2 = _make_bc(mesh10, outer_type=4, inner_type=5)
    scale_vec = np.empty(11)
    scale_vec[:10] = 3000.0
    scale_vec[-1] = 1e-6
    scales2 = NonDimScales(state_scale=scale_vec, t_ref=100.0)

    rhs_fn2, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc2,
        heating_array=np.zeros(10),
        scales=scales2,
        core_bc_mode='energy_balance',
        radio_isotope_params=radio_params,
    )

    t_nodes2 = np.linspace(0.0, 10.0, 600)
    Y_nodes2 = 3000.0 + 200.0 * np.linspace(0.0, 1.0, 600)[None, :] * np.ones((11, 1))
    out2 = rhs_fn2.step_powers(t_nodes2, Y_nodes2, aux=aux1)
    assert out2.shape == (600, 7)
    assert _TRACE_COUNTERS['powers'] == 2, (
        f'Expected 2 traces total, got {_TRACE_COUNTERS["powers"]}'
    )


@pytest.mark.unit
def test_step_powers_input_validation(eos_jax):
    """Verify input validation and error branches in step_powers."""
    from tests.conftest import make_mesh as _make_mesh
    from tests.test_cvode_jax_cache import _make_bc

    params = PhaseParams()
    mesh10 = _make_mesh(N=10)
    bc = _make_bc(mesh10, outer_type=4, inner_type=2)
    scales = NonDimScales(state_scale=np.full(10, 3000.0), t_ref=100.0)

    rhs_fn, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc,
        heating_array=np.zeros(10),
        scales=scales,
        core_bc_mode='quasi_steady',
    )
    aux = StepPowersAux(
        A_int=float(mesh10.area[-1]),
        A_cmb=float(mesh10.area[0]),
        volume=mesh10.volume,
        mass_struct=np.asarray(mesh10.volume * 3300.0),
        P_stag=mesh10.P_stag,
    )

    # Missing required aux raises TypeError
    with pytest.raises(TypeError):
        rhs_fn.step_powers([0.0], np.full((10, 1), 3000.0))

    # Multidimensional t_nodes
    with pytest.raises(ValueError, match='t_nodes must be 1D'):
        rhs_fn.step_powers(np.zeros((2, 2)), np.zeros((10, 4)), aux=aux)

    # Empty nodes
    empty_out = rhs_fn.step_powers([], np.empty((10, 0)), aux=aux)
    assert empty_out.shape == (0, 7)

    # Empty nodes with non-2D Y_nodes
    with pytest.raises(ValueError, match='Y_nodes must be 2D'):
        rhs_fn.step_powers([], np.zeros(10), aux=aux)

    # Non-2D Y_nodes
    with pytest.raises(ValueError, match='Y_nodes must be 2D'):
        rhs_fn.step_powers([0.0], np.zeros(10), aux=aux)

    # Length mismatch
    with pytest.raises(ValueError, match='does not match t_nodes length'):
        rhs_fn.step_powers([0.0, 1.0], np.zeros((10, 3)), aux=aux)

    # Dimension mismatch
    with pytest.raises(ValueError, match='does not match expected dimension'):
        rhs_fn.step_powers([0.0], np.zeros((9, 1)), aux=aux)

    # Valid aux call
    out_aux = rhs_fn.step_powers([0.0], np.full((10, 1), 3000.0), aux=aux)
    assert out_aux.shape == (1, 7)

    # Unknown mode in step_powers
    from aragog.jax.solver import step_powers

    with pytest.raises(ValueError, match='is not supported by step_powers'):
        step_powers(
            0.0,
            np.full(10, 3000.0),
            (None, None, None, None, None, None),
            'unsupported_mode',
            aux,
        )


@pytest.mark.unit
def test_step_powers_radio_uses_table_density(eos_jax):
    """Verify Q_radio integrates raw table density, not smoothed phase density."""
    from aragog.jax.phase import evaluate_phase
    from tests.conftest import make_mesh as _make_mesh
    from tests.test_cvode_jax_cache import _make_bc

    params = PhaseParams(matprop_smooth_width=0.05)
    mesh10 = _make_mesh(N=10)
    P_stag = np.asarray(mesh10.P_stag)
    S_test = np.full((10, 1), 3153.599)

    # First assert phase density and table density differ by > 1e-6 relative
    phase = evaluate_phase(eos_jax, params, P_stag, S_test.ravel())
    rho_phase = np.asarray(phase.density)
    rho_table = np.asarray(eos_jax.density(P_stag, S_test.ravel()))
    density_rel_diff = float(np.max(np.abs(rho_phase - rho_table) / rho_table))
    assert density_rel_diff > 1e-6, (
        f'Phase and table density must differ by > 1e-6, got {density_rel_diff:.4e}'
    )

    bc = _make_bc(mesh10, outer_type=4, inner_type=2)
    scales = NonDimScales(state_scale=np.full(10, 3000.0), t_ref=100.0)

    r26 = _make_radionuclide()
    radio_params = _make_radio_tuple([r26])

    rhs_fn, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc,
        heating_array=np.zeros(10),
        scales=scales,
        core_bc_mode='quasi_steady',
        radio_isotope_params=radio_params,
    )

    P_stag = np.asarray(mesh10.P_stag)
    S_test = np.full((10, 1), 3153.599)
    aux = StepPowersAux(
        A_int=float(mesh10.area[-1]),
        A_cmb=float(mesh10.area[0]),
        volume=mesh10.volume,
        mass_struct=np.asarray(mesh10.volume * 3300.0),
        P_stag=P_stag,
    )

    out = rhs_fn.step_powers([0.0], S_test, aux=aux)
    Q_radio = float(out[0, 2])

    # Reference Q_radio from table density
    h_radio = r26.heat_production * r26.abundance * r26.concentration
    mass_ref = eos_jax.density(P_stag, S_test.ravel()) * mesh10.volume
    Q_radio_ref = float(h_radio * np.sum(mass_ref))

    rel_diff = abs(Q_radio - Q_radio_ref) / Q_radio_ref
    assert rel_diff <= 1e-12, f'Q_radio relative difference {rel_diff:.4e} exceeds 1e-12'


@pytest.mark.smoke
@pytest.mark.parametrize(
    'mode,output_pts',
    [
        ('quasi_steady', 65),
        ('energy_balance', 65),
        ('quasi_steady', 550),
    ],
)
def test_step_powers_per_call_integrals(eos_np, eos_jax, mode: str, output_pts: int, caplog):
    """Verify per-call energy integrals match reference within tolerance."""
    assert not eos_np._has_alpha_tables

    n_nodes = 15
    n_stag = n_nodes - 1
    inner_bc = 1 if mode == 'energy_balance' else 2
    inner_val = 0.05 if inner_bc == 2 else 0.0
    params = _build_parameters(
        core_bc=mode,
        solver_method='cvode',
        end_time=20.0,
        n_nodes=n_nodes,
        use_jax_jacobian=True,
        inner_boundary_condition=inner_bc,
        inner_boundary_value=inner_val,
    )
    params.solver.cvode_output_points = output_pts
    params.phase_mixed.phase_transition_width = 0.0

    r26 = _make_radionuclide()
    params.energy.radionuclides = True
    params.radionuclides = [r26]
    radio_params = _make_radio_tuple([r26])

    params.energy.tidal = True
    tidal_arr = np.linspace(1e-12, 5e-12, n_stag)
    params.energy.tidal_array = list(tidal_arr)

    solver = es.EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()

    factory = _make_cvode_jax_factory(
        solver, eos_jax, params, mode, tidal_arr, radio_params, inner_val
    )
    solver.set_jax_cvode_factory(factory)
    _set_initial_entropy(solver, s_base=3050.0, s_span=150.0)
    solver.solve()

    sol = solver._solution
    assert sol is not None
    batch_integrals = dict(sol.energy_integrals)

    # Reference evaluation with forced numpy loop
    batch_callable = solver._cvode_step_powers_batch
    solver._cvode_step_powers_batch = None
    ref_integrals = solver._compute_step_energy_integrals()
    solver._cvode_step_powers_batch = batch_callable

    for key in ['F_int', 'F_cmb', 'Q_radio', 'Q_tidal', 'Q_radio_cons', 'Q_tidal_cons']:
        val_batch = batch_integrals[key]
        val_ref = ref_integrals[key]
        denom = abs(val_ref)
        if denom > 0.0:
            rel = abs(val_batch - val_ref) / denom
            assert rel <= 1e-12, f'{key} relative difference {rel:.4e} exceeds 1e-12'
        else:
            assert abs(val_batch - val_ref) <= 1e-12

    if ref_integrals['F_cmb_step_avg'] is not None:
        denom = abs(ref_integrals['F_cmb_step_avg'])
        if denom > 0.0:
            rel = (
                abs(batch_integrals['F_cmb_step_avg'] - ref_integrals['F_cmb_step_avg']) / denom
            )
            assert rel <= 1e-12, f'F_cmb_step_avg relative difference {rel:.4e} exceeds 1e-12'
        else:
            assert (
                abs(batch_integrals['F_cmb_step_avg'] - ref_integrals['F_cmb_step_avg'])
                <= 1e-12
            )
    else:
        assert batch_integrals['F_cmb_step_avg'] is None

    # Residual consistency: absolute value <= 1e-12 * sum of abs boundary and source integrals
    scale = (
        abs(batch_integrals['F_int'])
        + abs(batch_integrals['F_cmb'])
        + abs(batch_integrals['Q_radio'])
        + abs(batch_integrals['Q_tidal'])
    )
    assert abs(batch_integrals['solver_residual']) <= 1e-12 * scale

    # State heat is bit-equal
    assert batch_integrals['state_heat'] == ref_integrals['state_heat']
    assert not any(
        'batch step powers evaluation failed' in rec.message for rec in caplog.records
    ), 'Batch step powers evaluation fell back unexpectedly'


@pytest.mark.smoke
def test_step_powers_solver_dispatch(eos_np, eos_jax):
    """Verify solver dispatch routes to batch or numpy path as configured."""
    assert not eos_np._has_alpha_tables

    for mode in ['quasi_steady', 'energy_balance']:
        n_nodes = 15
        inner_bc = 1 if mode == 'energy_balance' else 2
        inner_val = 0.05 if inner_bc == 2 else 0.0
        params = _build_parameters(
            core_bc=mode,
            solver_method='cvode',
            end_time=10.0,
            n_nodes=n_nodes,
            use_jax_jacobian=True,
            inner_boundary_condition=inner_bc,
            inner_boundary_value=inner_val,
        )
        solver = es.EntropySolver(params, entropy_eos=eos_np)
        solver.initialize()
        factory = _make_cvode_jax_factory(
            solver, eos_jax, params, mode, np.zeros(n_nodes - 1), (), inner_val
        )
        solver.set_jax_cvode_factory(factory)
        _set_initial_entropy(solver, s_base=3050.0, s_span=150.0)
        call_count = _spy_step_powers(solver)
        solver.solve()
        assert solver._cvode_step_powers_batch is not None
        assert call_count[0] == 0, (
            f'Expected 0 numpy calls on CVODE batch path, got {call_count[0]}'
        )

    params_scipy = _build_parameters(
        core_bc='quasi_steady', solver_method='radau', end_time=10.0
    )
    solver_scipy = es.EntropySolver(params_scipy, entropy_eos=eos_np)
    solver_scipy.initialize()
    _set_initial_entropy(solver_scipy, s_base=3050.0, s_span=150.0)
    call_count_scipy = _spy_step_powers(solver_scipy)
    solver_scipy.solve()
    assert solver_scipy._cvode_step_powers_batch is None
    assert call_count_scipy[0] > 0, 'Expected numpy calls on scipy path'

    params_nofac = _build_parameters(
        core_bc='quasi_steady',
        solver_method='cvode',
        end_time=10.0,
        use_jax_jacobian=False,
    )
    solver_nofac = es.EntropySolver(params_nofac, entropy_eos=eos_np)
    solver_nofac.initialize()
    _set_initial_entropy(solver_nofac, s_base=3050.0, s_span=150.0)
    call_count_nofac = _spy_step_powers(solver_nofac)
    solver_nofac.solve()
    assert solver_nofac._cvode_step_powers_batch is None
    assert call_count_nofac[0] > 0, 'Expected numpy calls without factory'

    for unsupported_mode in ['gradient', 'bower2018']:
        params_unsup = _build_parameters(
            core_bc=unsupported_mode,
            solver_method='cvode',
            end_time=10.0,
            use_jax_jacobian=True,
        )
        solver_unsup = es.EntropySolver(params_unsup, entropy_eos=eos_np)
        solver_unsup.initialize()
        factory_unsup = _make_cvode_jax_factory(
            solver_unsup, eos_jax, params_unsup, unsupported_mode, np.zeros(14), (), 0.0
        )
        solver_unsup.set_jax_cvode_factory(factory_unsup)
        if unsupported_mode == 'gradient':
            n_basic = solver_unsup._n_stag + 1
            y0 = np.zeros(n_basic + 1)
            y0[:n_basic] = 0.0
            y0[n_basic] = 3000.0
            solver_unsup._S0 = y0
        else:
            _set_initial_entropy(solver_unsup, s_base=3050.0, s_span=150.0)
        call_count_unsup = _spy_step_powers(solver_unsup)
        solver_unsup.solve()
        assert solver_unsup._cvode_step_powers_batch is None
        assert call_count_unsup[0] > 0, f'Expected numpy calls in {unsupported_mode} mode'

    params_noattr = _build_parameters(
        core_bc='quasi_steady',
        solver_method='cvode',
        end_time=10.0,
        use_jax_jacobian=True,
    )
    solver_noattr = es.EntropySolver(params_noattr, entropy_eos=eos_np)
    solver_noattr.initialize()
    factory_raw = _make_cvode_jax_factory(
        solver_noattr, eos_jax, params_noattr, 'quasi_steady', np.zeros(14), (), 0.0
    )

    def factory_no_attr(scales, core_bc_mode):
        rhs_fn, jac_fn = factory_raw(scales, core_bc_mode)

        def raw_rhs(t, y, ydot):
            return rhs_fn(t, y, ydot)

        return raw_rhs, jac_fn

    solver_noattr.set_jax_cvode_factory(factory_no_attr)
    _set_initial_entropy(solver_noattr, s_base=3050.0, s_span=150.0)
    call_count_noattr = _spy_step_powers(solver_noattr)
    solver_noattr.solve()
    assert solver_noattr._cvode_step_powers_batch is None
    assert call_count_noattr[0] > 0, 'Expected numpy calls when rhs_fn lacks step_powers'


@pytest.mark.smoke
@pytest.mark.parametrize('mode', ['quasi_steady', 'energy_balance'])
def test_step_powers_state_refresh(eos_np, eos_jax, mode: str, caplog):
    """Verify that solver state after batch powers matches numpy state."""
    assert not eos_np._has_alpha_tables

    n_nodes = 15
    inner_bc = 1 if mode == 'energy_balance' else 2
    inner_val = 0.05 if inner_bc == 2 else 0.0
    params = _build_parameters(
        core_bc=mode,
        solver_method='cvode',
        end_time=20.0,
        n_nodes=n_nodes,
        use_jax_jacobian=True,
        inner_boundary_condition=inner_bc,
        inner_boundary_value=inner_val,
    )
    params.phase_mixed.phase_transition_width = 0.0

    r26 = _make_radionuclide()
    params.energy.radionuclides = True
    params.radionuclides = [r26]
    radio_params = _make_radio_tuple([r26])
    params.energy.tidal = True
    tidal_arr = np.linspace(1e-12, 5e-12, n_nodes - 1)
    params.energy.tidal_array = list(tidal_arr)

    solver = es.EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()
    factory = _make_cvode_jax_factory(
        solver, eos_jax, params, mode, tidal_arr, radio_params, inner_val
    )
    solver.set_jax_cvode_factory(factory)
    _set_initial_entropy(solver, s_base=3050.0, s_span=150.0)
    solver.solve()

    state_batch_flux = solver.state._heat_flux.copy()
    state_batch_radio = np.asarray(solver.state.heating_radio).copy()
    state_batch_tidal = np.asarray(solver.state.heating_tidal).copy()
    state_batch_rho = np.asarray(solver.state.phase_staggered.density()).copy()
    state_batch_T = np.asarray(solver.state.phase_staggered.temperature()).copy()
    state_batch_cap = np.asarray(solver.state.capacitance_staggered()).copy()
    hits_batch = solver.state._pb_cache_hits
    misses_batch = solver.state._pb_cache_misses

    # Computing integrals on batch path must preserve cache counters
    solver._compute_step_energy_integrals()
    assert solver.state._pb_cache_hits == hits_batch
    assert solver.state._pb_cache_misses == misses_batch

    # Replay on numpy path must preserve cache counters and match state arrays
    batch_callable = solver._cvode_step_powers_batch
    solver._cvode_step_powers_batch = None
    solver._compute_step_energy_integrals()
    solver._cvode_step_powers_batch = batch_callable

    state_np_flux = solver.state._heat_flux.copy()
    state_np_radio = np.asarray(solver.state.heating_radio).copy()
    state_np_tidal = np.asarray(solver.state.heating_tidal).copy()
    state_np_rho = np.asarray(solver.state.phase_staggered.density()).copy()
    state_np_T = np.asarray(solver.state.phase_staggered.temperature()).copy()
    state_np_cap = np.asarray(solver.state.capacitance_staggered()).copy()
    hits_np = solver.state._pb_cache_hits
    misses_np = solver.state._pb_cache_misses

    np.testing.assert_array_equal(state_batch_flux, state_np_flux)
    np.testing.assert_array_equal(state_batch_radio, state_np_radio)
    np.testing.assert_array_equal(state_batch_tidal, state_np_tidal)
    np.testing.assert_array_equal(state_batch_rho, state_np_rho)
    np.testing.assert_array_equal(state_batch_T, state_np_T)
    np.testing.assert_array_equal(state_batch_cap, state_np_cap)
    assert hits_batch == hits_np
    assert misses_batch == misses_np
    assert not any(
        'batch step powers evaluation failed' in rec.message for rec in caplog.records
    ), 'Batch step powers evaluation fell back unexpectedly'


@pytest.mark.unit
def test_step_powers_batch_dispatch_unit(caplog):
    """Verify batch dispatch, exception fallback, and shape fallback without solve()."""
    s = es.EntropySolver.__new__(es.EntropySolver)
    s._warned = set()
    s.entropy_eos = object()
    t_nodes = np.array([0.0, 1.0])
    s._solution = OptimizeResult(
        t=t_nodes,
        y=np.zeros((2, 2)),
    )
    s._r_basic_flat = np.array([1.0, 2.0])
    s._volume_flat = np.array([1.0, 1.0])
    s._P_stag_flat = np.array([1e9, 2e9])
    s.evaluator = SimpleNamespace(
        mesh=SimpleNamespace(staggered_effective_density=np.array([3000.0, 3000.0]))
    )
    s.state = SimpleNamespace(_pb_cache_hits=5, _pb_cache_misses=6)
    s._stag_entropy = lambda y: y
    s._step_heat_content = lambda a, b: 0.0
    s._dSdt_single = lambda t, y: y

    numpy_called = []

    def stub_step_powers(t, y):
        numpy_called.append(float(t))
        return np.full(7, 10.0)

    s._step_powers = stub_step_powers

    # 1. Success uses batch output without calling numpy loop
    batch_called = []

    def stub_batch_success(t_n, y_n, aux):
        batch_called.append(len(t_n))
        assert isinstance(aux, StepPowersAux)
        return np.full((len(t_n), 7), 20.0)

    s._cvode_step_powers_batch = stub_batch_success
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        out_success = s._compute_step_energy_integrals()

    assert batch_called == [2]
    assert len(numpy_called) == 0
    assert not any(
        'batch step powers evaluation failed' in rec.message for rec in caplog.records
    )
    expected_integral = 20.0 * 1.0 * es.SECS_PER_YEAR
    np.testing.assert_allclose(out_success['F_int'], expected_integral, rtol=1e-12)

    # 2. Exception falls back to numpy loop with one warning
    def stub_batch_raise(t_n, y_n, aux):
        raise RuntimeError('simulated batch failure')

    s._cvode_step_powers_batch = stub_batch_raise
    s._warned.clear()
    caplog.clear()
    numpy_called.clear()
    with caplog.at_level(logging.WARNING):
        out_raise = s._compute_step_energy_integrals()

    assert len(numpy_called) == 2
    expected_np_integral = 10.0 * 1.0 * es.SECS_PER_YEAR
    np.testing.assert_allclose(out_raise['F_int'], expected_np_integral, rtol=1e-12)
    warn_raise = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warn_raise) == 1
    assert 'batch step powers evaluation failed' in warn_raise[0].message
    assert 'simulated batch failure' in warn_raise[0].message

    # Repeated call emits no second warning
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        s._compute_step_energy_integrals()
    assert len([rec for rec in caplog.records if rec.levelno == logging.WARNING]) == 0

    # 3. Wrong shape falls back to numpy loop with one warning
    def stub_batch_wrong_shape(t_n, y_n, aux):
        return np.zeros((len(t_n), 5))

    s._cvode_step_powers_batch = stub_batch_wrong_shape
    s._warned.clear()
    caplog.clear()
    numpy_called.clear()
    with caplog.at_level(logging.WARNING):
        out_shape = s._compute_step_energy_integrals()

    assert len(numpy_called) == 2
    np.testing.assert_allclose(out_shape['F_int'], expected_np_integral, rtol=1e-12)
    warn_shape = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warn_shape) == 1
    assert 'Expected batch step powers shape' in warn_shape[0].message


@pytest.mark.unit
def test_step_powers_batch_interface(eos_np):
    """Verify EntropySolver._step_powers_batch signature handling and error contract."""
    params = _build_parameters(core_bc='quasi_steady')
    solver = es.EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()

    n_dim = len(solver._r_basic_flat)

    # When no batch callable is registered, raises RuntimeError
    solver._cvode_step_powers_batch = None
    with pytest.raises(RuntimeError, match='No batch step powers callable available'):
        solver._step_powers_batch(np.array([0.0]), np.zeros((n_dim, 1)))

    # When batch callable is registered, passes aux keyword argument
    called_with_aux = []

    def mock_batch(t_nodes, Y_nodes, aux):
        called_with_aux.append(aux is not None)
        return np.ones((len(t_nodes), 7))

    solver._cvode_step_powers_batch = mock_batch
    out = solver._step_powers_batch(np.array([0.0, 1.0]), np.zeros((n_dim, 2)))
    assert out.shape == (2, 7)
    assert called_with_aux == [True]
