"""Tests for batched per-call energy power evaluation in Aragog.

Verifies:
- Per-node power parity between JAX batch and numpy step powers
  across 8 combinations of mode, radio heating, and tidal heating for n > 512.
- Compile counter verifies exactly one trace across varying trace lengths
  and one additional trace for a second mode.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    pytest.mark.unit,
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


def _make_phase_params(params: Parameters) -> PhaseParams:
    return PhaseParams(
        matprop_smooth_width=float(params.phase_mixed.phase_transition_width),
        kappah_floor=float(params.energy.kappah_floor),
        k_solid=float(params.phase_solid.thermal_conductivity),
        k_liquid=float(params.phase_liquid.thermal_conductivity),
        phi_rheo=float(params.phase_mixed.rheological_transition_melt_fraction),
        phi_width=float(params.phase_mixed.rheological_transition_width),
        grain_size=float(params.phase_mixed.grain_size),
        conduction=params.energy.conduction,
        convection=params.energy.convection,
    )


@pytest.mark.parametrize('mode', ['quasi_steady', 'energy_balance'])
@pytest.mark.parametrize('use_radio', [False, True])
@pytest.mark.parametrize('use_tidal', [False, True])
def test_step_powers_per_node_parity(eos_np, eos_jax, mode: str, use_radio: bool, use_tidal: bool):
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
    r = np.asarray(solver._r_stag_flat)
    solver.set_initial_entropy(3050.0 + 150.0 * (r[-1] - r) / (r[-1] - r[0]))
    solver.solve()
    sol = solver._solution
    assert sol is not None
    t_pts, y_pts = sol.energy_trace
    assert len(t_pts) > 512, f'Expected > 512 nodes, got {len(t_pts)}'

    P_np = np.array([solver._step_powers(float(t), y) for t, y in zip(t_pts, y_pts.T)])

    mesh = solver.evaluator.mesh
    mesh_arr = MeshArrays.from_numpy_mesh(mesh)
    inner_type = 5 if mode == 'energy_balance' else 2
    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=1500.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=inner_type,
        inner_bc_value=inner_val,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_area=4.0 * np.pi * float(mesh_arr.radii_basic[0]) ** 2,
        core_M=(4.0 / 3.0) * np.pi * float(mesh_arr.radii_basic[0]) ** 3 * 10500.0,
        cmb_dr_cmb=float(mesh_arr.radii_basic[1] - mesh_arr.radii_basic[0]),
        param_utbl=False,
        param_utbl_const=0.0,
    )
    if use_radio:
        radio_params = (
            np.array([r.heat_production for r in params.radionuclides]),
            np.array([r.abundance for r in params.radionuclides]),
            np.array([r.concentration for r in params.radionuclides]),
            np.array([r.t0_years for r in params.radionuclides]),
            np.array([r.half_life_years for r in params.radionuclides]),
        )
    else:
        radio_params = ()

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

    vol = np.asarray(mesh.basic.volume).ravel()
    r_b = np.asarray(mesh.basic.radii).ravel()
    mass_struct = np.asarray(mesh.staggered_effective_density).ravel() * vol
    aux = StepPowersAux(
        A_int=4.0 * np.pi * float(r_b[-1]) ** 2,
        A_cmb=4.0 * np.pi * float(r_b[0]) ** 2,
        volume=vol,
        mass_struct=mass_struct,
        P_stag=np.asarray(mesh.staggered_pressure).ravel(),
    )

    P_jax = rhs_fn.step_powers(t_pts, y_pts, aux=aux)
    assert P_jax.shape == P_np.shape

    for col in range(6):
        denom = np.max(np.abs(P_np[:, col]))
        if denom == 0.0:
            np.testing.assert_allclose(P_jax[:, col], 0.0, atol=1e-12)
        else:
            rel_diff = np.max(np.abs(P_jax[:, col] - P_np[:, col])) / denom
            assert rel_diff <= 1e-12, (
                f'Col {col} rel diff {rel_diff:.4e} exceeds 1e-12 for mode={mode}'
            )

    scale_np = np.sum(np.abs(P_np[:, :4]), axis=1)
    scale_jax = np.sum(np.abs(P_jax[:, :4]), axis=1)
    np_resid_ratio = np.max(np.abs(P_np[:, 6]) / np.maximum(scale_np, 1.0))
    jax_resid_ratio = np.max(np.abs(P_jax[:, 6]) / np.maximum(scale_jax, 1.0))
    assert np_resid_ratio <= 1e-12, f'NP residual {np_resid_ratio:.4e} exceeds 1e-12'
    assert jax_resid_ratio <= 1e-12, f'JAX residual {jax_resid_ratio:.4e} exceeds 1e-12'


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

    rhs_fn1, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc1,
        heating_array=np.zeros(10),
        scales=scales1,
        core_bc_mode='quasi_steady',
    )

    # 5 solves with different trace lengths, one above C = 512
    lengths = [50, 100, 512, 600, 1000]
    for n in lengths:
        t_nodes = np.linspace(0.0, 10.0, n)
        Y_nodes = np.full((10, n), 3000.0)
        out = rhs_fn1.step_powers(t_nodes, Y_nodes)
        assert out.shape == (n, 7)
        assert _TRACE_COUNTERS['powers'] == 1, (
            f'Expected 1 trace, got {_TRACE_COUNTERS["powers"]} at length {n}'
        )

    # Re-invoking factory with identical cache key reuses powers_jit with 0 new traces
    rhs_fn1_repeat, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc1,
        heating_array=np.zeros(10),
        scales=scales1,
        core_bc_mode='quasi_steady',
    )
    out_repeat = rhs_fn1_repeat.step_powers(
        np.linspace(0.0, 10.0, 50), np.full((10, 50), 3000.0)
    )
    assert out_repeat.shape == (50, 7)
    assert _TRACE_COUNTERS['powers'] == 1, (
        'Expected 0 additional traces on repeated factory call'
    )

    # Verify numerical consistency across chunk boundaries (C = 512)
    np.testing.assert_allclose(out[0], out[-1], atol=1e-12)
    np.testing.assert_allclose(out[511], out[512], atol=1e-12)

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
    )

    t_nodes2 = np.linspace(0.0, 10.0, 600)
    Y_nodes2 = np.full((11, 600), 3000.0)
    out2 = rhs_fn2.step_powers(t_nodes2, Y_nodes2)
    assert out2.shape == (600, 7)
    assert _TRACE_COUNTERS['powers'] == 2, (
        f'Expected 2 traces total, got {_TRACE_COUNTERS["powers"]}'
    )


def test_step_powers_input_validation(eos_jax):
    """Verify input validation and error branches in step_powers."""
    import pytest

    from aragog.jax.solver import StepPowersAux
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

    # Multidimensional t_nodes
    with pytest.raises(ValueError, match='t_nodes must be 1D'):
        rhs_fn.step_powers(np.zeros((2, 2)), np.zeros((10, 4)))

    # Empty nodes
    empty_out = rhs_fn.step_powers([], np.empty((10, 0)))
    assert empty_out.shape == (0, 7)

    # Empty nodes with non-2D Y_nodes
    with pytest.raises(ValueError, match='Y_nodes must be 2D'):
        rhs_fn.step_powers([], np.zeros(10))

    # Non-2D Y_nodes
    with pytest.raises(ValueError, match='Y_nodes must be 2D'):
        rhs_fn.step_powers([0.0], np.zeros(10))

    # Length mismatch
    with pytest.raises(ValueError, match='does not match t_nodes length'):
        rhs_fn.step_powers([0.0, 1.0], np.zeros((10, 3)))

    # Dimension mismatch
    with pytest.raises(ValueError, match='does not match expected dimension'):
        rhs_fn.step_powers([0.0], np.zeros((9, 1)))

    # Explicit aux
    aux = StepPowersAux(
        A_int=float(mesh10.area[-1]),
        A_cmb=float(mesh10.area[0]),
        volume=mesh10.volume,
        mass_struct=np.asarray(mesh10.volume * 3300.0),
        P_stag=mesh10.P_stag,
    )
    out_aux = rhs_fn.step_powers([0.0], np.full((10, 1), 3000.0), aux=aux)
    assert out_aux.shape == (1, 7)

    # Factory with mesh_arrays=None and no aux provided
    rhs_no_mesh, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=None,
        boundary_params=bc,
        heating_array=np.zeros(10),
        scales=scales,
        core_bc_mode='quasi_steady',
    )
    with pytest.raises(ValueError, match='aux must be supplied'):
        rhs_no_mesh.step_powers([0.0], np.full((10, 1), 3000.0))

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


def test_step_powers_radio_uses_table_density(eos_jax):
    """Verify Q_radio integrates raw table density, not smoothed phase density."""
    from aragog.jax.solver import StepPowersAux
    from tests.conftest import make_mesh as _make_mesh
    from tests.test_cvode_jax_cache import _make_bc

    params = PhaseParams(matprop_smooth_width=0.05)
    mesh10 = _make_mesh(N=10)
    bc = _make_bc(mesh10, outer_type=4, inner_type=2)
    scales = NonDimScales(state_scale=np.full(10, 3000.0), t_ref=100.0)

    r26 = _make_radionuclide()
    radio_params = (
        np.array([r26.heat_production]),
        np.array([r26.abundance]),
        np.array([r26.concentration]),
        np.array([r26.t0_years]),
        np.array([r26.half_life_years]),
    )

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
