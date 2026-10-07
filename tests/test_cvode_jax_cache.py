"""Tests for JAX CVODE JIT caching and data pytree parameterization.

Verifies:
(a) Compile counter: 1 RHS and 1 Jacobian trace over repeated factory calls with
    varying parameters; trace increments on mesh shape change and static BC mode change;
(b) Parity with pre-change factory across quasi_steady and energy_balance, radio on/off;
(c) No stale values: perturbation of each pytree leaf alters the output and matches
    an independent reference build;
(d) ID-reuse guard: object identity check protects against recycled object addresses.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pytest

from aragog.jax.nondim import NonDimScales
from aragog.jax.phase import PhaseParams
from aragog.solver.cvode_jax import (
    _CACHE_MAXSIZE,
    _JIT_CACHE,
    _TRACE_COUNTERS,
    _get_or_create_jitted,
    _JitCacheEntry,
    build_jax_rhs_and_jacobian,
    clear_jit_cache,
)
from tests.conftest import EOS_DIR, entropy_eos_jax
from tests.conftest import make_mesh as _make_mesh


def needs_eos(fn):
    """Require SPIDER P-S tables; fail in CI when tables are missing."""
    if os.environ.get('CI') and not EOS_DIR.exists():

        @functools.wraps(fn)
        def _ci_missing_eos_fail(*args, **kwargs):
            pytest.fail(f'CI requires SPIDER P-S tables, but {EOS_DIR} not found.')

        return _ci_missing_eos_fail
    return pytest.mark.skipif(
        not EOS_DIR.exists(),
        reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
    )(fn)


jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
eqx = pytest.importorskip('equinox')


jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_jit_cache()


@pytest.fixture(scope='module')
def shared_eos():
    if not EOS_DIR.exists():
        if os.environ.get('CI'):
            pytest.fail(f'CI requires SPIDER P-S tables, but {EOS_DIR} not found.')
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}.')
    return entropy_eos_jax(EOS_DIR)


def _make_bc(
    mesh,
    outer_type: int = 4,
    inner_type: int = 2,
    outer_val: float = 0.0,
    param_utbl: bool = False,
    param_utbl_const: float = 0.0,
):
    from aragog.jax.solver import BoundaryParams

    r_cmb = float(mesh.radii_basic[0])
    r_above = float(mesh.radii_basic[1])
    return BoundaryParams(
        outer_bc_type=outer_type,
        outer_bc_value=outer_val,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=inner_type,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=(4.0 / 3.0) * np.pi * r_cmb**3 * 10500.0,
        cmb_dr_cmb=r_above - r_cmb,
        param_utbl=param_utbl,
        param_utbl_const=param_utbl_const,
    )


def _make_radio_tuple():
    hp = np.array([1.0e-8, 2.0e-8])
    ab = np.array([1.0, 1.0])
    cn = np.array([1.0, 1.0])
    t0 = np.array([0.0, 0.0])
    hl = np.array([50.0, 100.0])
    return (hp, ab, cn, t0, hl)


_LOG_TWO: float = 0.6931471805599453


def _reference_make_radio_heating_fn(
    heat_prod, abundance, concentration, t0_years, half_life_years
):
    """Test-only independent copy of the abc6101 make_radio_heating_fn implementation."""
    hp = jnp.asarray(heat_prod, dtype=jnp.float64)
    ab = jnp.asarray(abundance, dtype=jnp.float64)
    cn = jnp.asarray(concentration, dtype=jnp.float64)
    t0 = jnp.asarray(t0_years, dtype=jnp.float64)
    hl = jnp.asarray(half_life_years, dtype=jnp.float64)
    amp = hp * ab * cn

    def _h(t_yr):
        arg = _LOG_TWO * (t0 - t_yr) / jnp.maximum(hl, 1e-10)
        per_iso = amp * jnp.exp(jnp.where(amp != 0.0, arg, 0.0))
        return jnp.sum(per_iso)

    return _h


def _build_reference_factory(
    eos_jax,
    phase_params,
    mesh_arrays,
    boundary_params,
    heating_array,
    scales,
    core_bc_mode='quasi_steady',
    radio_isotope_params=(),
):
    """Independent reference implementation of the pre-change factory."""
    from aragog.jax.solver import (
        _no_radio,
    )
    from aragog.jax.solver import (
        dSdt as jax_dsdt,
    )
    from aragog.jax.solver import (
        dSdt_energy_balance as jax_dsdt_eb,
    )

    _rhs_jax = jax_dsdt if core_bc_mode == 'quasi_steady' else jax_dsdt_eb

    state_scale_jax = jnp.asarray(scales.state_scale)
    rhs_scale_jax = jnp.asarray(scales.rhs_scale)
    t_ref = float(scales.t_ref)
    heating_jax = jnp.asarray(heating_array)

    if radio_isotope_params:
        H_radio_fn = _reference_make_radio_heating_fn(*radio_isotope_params)
    else:
        H_radio_fn = _no_radio

    args_tuple = (
        eos_jax,
        phase_params,
        mesh_arrays,
        boundary_params,
        heating_jax,
        H_radio_fn,
    )

    def _rhs_phys(t_phys, S_phys):
        return _rhs_jax(t_phys, S_phys, args_tuple)

    def _rhs_nondim(t_nd, y_nd):
        t_phys = t_nd * t_ref
        S_phys = y_nd * state_scale_jax
        dydt_phys = _rhs_phys(t_phys, S_phys)
        return dydt_phys * rhs_scale_jax

    rhs_jit = jax.jit(_rhs_nondim)
    jac_jit = jax.jit(jax.jacrev(_rhs_nondim, argnums=1))

    def rhs_fn(t_nd, y_nd, ydot_nd):
        res = rhs_jit(float(t_nd), jnp.asarray(y_nd))
        ydot_nd[:] = np.asarray(res)
        return 0

    def jacfn(t_nd, y_nd, fy_nd, J, user_data=None):
        jac = jac_jit(float(t_nd), jnp.asarray(y_nd))
        J[...] = np.asarray(jac)
        return 0

    return rhs_fn, jacfn, {}


@needs_eos
def test_compile_counter_a(shared_eos):
    """Verify trace counters increment exactly once over repeated calls,
    once more on shape change, and once more on static BC type changes.
    """
    eos = shared_eos
    params = PhaseParams()
    mesh10 = _make_mesh(N=10)
    n_stag = 10

    # 1. Five factory calls with varying parameters but same EOS & PhaseParams
    scales = NonDimScales(state_scale=np.full(n_stag, 3.0e3), t_ref=1.0)
    y_nd = np.full(n_stag, 3050.0 / 3.0e3)
    ydot = np.zeros(n_stag)
    J = np.zeros((n_stag, n_stag))

    for i in range(5):
        mesh_i = _make_mesh(N=10, scale_p=1.0 + 0.01 * i)
        bc_i = _make_bc(mesh_i, outer_type=4, inner_type=2, outer_val=10.0 * i)
        heating_i = np.full(n_stag, 1e-12 * i)
        scales_i = NonDimScales(
            state_scale=np.full(n_stag, 3.0e3 + 10.0 * i), t_ref=1.0 + 0.1 * i
        )

        rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
            eos_jax=eos,
            phase_params=params,
            mesh_arrays=mesh_i,
            boundary_params=bc_i,
            heating_array=heating_i,
            scales=scales_i,
            core_bc_mode='quasi_steady',
        )
        rhs_fn(0.0, y_nd, ydot)
        jac_fn(0.0, y_nd, None, J)

    assert _TRACE_COUNTERS['rhs'] == 1
    assert _TRACE_COUNTERS['jac'] == 1

    # 2. Changed node count (N=12): triggers shape-based retrace
    mesh12 = _make_mesh(N=12)
    scales12 = NonDimScales(state_scale=np.full(12, 3.0e3), t_ref=1.0)
    y_nd12 = np.full(12, 3050.0 / 3.0e3)
    ydot12 = np.zeros(12)
    J12 = np.zeros((12, 12))

    rhs12, jac12, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params,
        mesh_arrays=mesh12,
        boundary_params=_make_bc(mesh12),
        heating_array=np.zeros(12),
        scales=scales12,
        core_bc_mode='quasi_steady',
    )
    rhs12(0.0, y_nd12, ydot12)
    jac12(0.0, y_nd12, None, J12)

    assert _TRACE_COUNTERS['rhs'] == 2
    assert _TRACE_COUNTERS['jac'] == 2

    # 3. Changed outer_bc_type (4 -> 1)
    bc_outer1 = _make_bc(mesh10, outer_type=1, inner_type=2)
    rhs_o1, jac_o1, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc_outer1,
        heating_array=np.zeros(10),
        scales=scales,
        core_bc_mode='quasi_steady',
    )
    rhs_o1(0.0, y_nd, ydot)
    jac_o1(0.0, y_nd, None, J)
    assert _TRACE_COUNTERS['rhs'] == 3
    assert _TRACE_COUNTERS['jac'] == 3

    # Check parity with reference on outer_bc_type=1
    r_rhs_ref, r_jac_ref, _ = _build_reference_factory(
        eos, params, mesh10, bc_outer1, np.zeros(10), scales, 'quasi_steady'
    )
    ydot_ref = np.zeros(10)
    J_ref = np.zeros((10, 10))
    r_rhs_ref(0.0, y_nd, ydot_ref)
    r_jac_ref(0.0, y_nd, None, J_ref)
    assert np.max(np.abs(ydot - ydot_ref)) <= 1e-14 * np.max(np.abs(ydot_ref))
    assert np.max(np.abs(J - J_ref)) <= 2e-12 * np.max(np.abs(J_ref))

    # 4. Changed inner_bc_type (2 -> 0)
    bc_inner0 = _make_bc(mesh10, outer_type=4, inner_type=0)
    rhs_i0, jac_i0, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc_inner0,
        heating_array=np.zeros(10),
        scales=scales,
        core_bc_mode='quasi_steady',
    )
    rhs_i0(0.0, y_nd, ydot)
    jac_i0(0.0, y_nd, None, J)
    assert _TRACE_COUNTERS['rhs'] == 4
    assert _TRACE_COUNTERS['jac'] == 4

    # Check parity with reference on inner_bc_type=0
    r_rhs_ref_i0, r_jac_ref_i0, _ = _build_reference_factory(
        eos, params, mesh10, bc_inner0, np.zeros(10), scales, 'quasi_steady'
    )
    ydot_ref_i0 = np.zeros(10)
    J_ref_i0 = np.zeros((10, 10))
    r_rhs_ref_i0(0.0, y_nd, ydot_ref_i0)
    r_jac_ref_i0(0.0, y_nd, None, J_ref_i0)
    assert np.max(np.abs(ydot - ydot_ref_i0)) <= 1e-14 * np.max(np.abs(ydot_ref_i0))
    assert np.max(np.abs(J - J_ref_i0)) <= 2e-12 * np.max(np.abs(J_ref_i0))

    # 5. Changed param_utbl (False -> True)
    bc_utbl = _make_bc(
        mesh10, outer_type=1, inner_type=2, param_utbl=True, param_utbl_const=1e-7
    )
    rhs_u, jac_u, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params,
        mesh_arrays=mesh10,
        boundary_params=bc_utbl,
        heating_array=np.zeros(10),
        scales=scales,
        core_bc_mode='quasi_steady',
    )
    rhs_u(0.0, y_nd, ydot)
    jac_u(0.0, y_nd, None, J)
    assert _TRACE_COUNTERS['rhs'] == 5
    assert _TRACE_COUNTERS['jac'] == 5

    # Check parity with reference on param_utbl=True
    r_rhs_ref_u, r_jac_ref_u, _ = _build_reference_factory(
        eos, params, mesh10, bc_utbl, np.zeros(10), scales, 'quasi_steady'
    )
    ydot_ref_u = np.zeros(10)
    J_ref_u = np.zeros((10, 10))
    r_rhs_ref_u(0.0, y_nd, ydot_ref_u)
    r_jac_ref_u(0.0, y_nd, None, J_ref_u)
    assert np.max(np.abs(ydot - ydot_ref_u)) <= 1e-14 * np.max(np.abs(ydot_ref_u))
    assert np.max(np.abs(J - J_ref_u)) <= 2e-12 * np.max(np.abs(J_ref_u))


@needs_eos
@pytest.mark.parametrize('core_bc_mode', ['quasi_steady', 'energy_balance'])
@pytest.mark.parametrize('use_radio', [False, True])
@pytest.mark.parametrize('outer_case', ['grey_body', 'utbl'])
def test_parity_with_pre_change_factory_b(
    shared_eos, core_bc_mode: str, use_radio: bool, outer_case: str
):
    """Verify numeric parity between cached factory and reference factory.

    Bounds are RHS 1e-14 max|f| and J 2e-12 max|J|. Across three table sets
    and random states the measured maximum differences are RHS 2.54e-15 and
    J 3.97e-13, with a Jacobian round-off floor up to 8.2e-13.
    """
    eos = shared_eos
    params = PhaseParams()
    mesh = _make_mesh(N=8)
    inner_type = 5 if core_bc_mode == 'energy_balance' else 2
    if outer_case == 'grey_body':
        bc = _make_bc(mesh, outer_type=4, inner_type=inner_type)
    else:
        bc = _make_bc(
            mesh, outer_type=1, inner_type=inner_type, param_utbl=True, param_utbl_const=1e-7
        )
    n_stag = 8
    n_state = n_stag if core_bc_mode == 'quasi_steady' else n_stag + 1

    heating = np.full(n_stag, 1e-12)
    radio_params = _make_radio_tuple() if use_radio else ()

    state_scale = np.full(n_state, 3.0e3)
    if core_bc_mode == 'energy_balance':
        state_scale[-1] = 1e-6
    scales = NonDimScales(state_scale=state_scale, t_ref=100.0)

    # 1. Build cached factory
    rhs_cached, jac_cached, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params,
        mesh_arrays=mesh,
        boundary_params=bc,
        heating_array=heating,
        scales=scales,
        core_bc_mode=core_bc_mode,
        radio_isotope_params=radio_params,
    )

    # 2. Build reference pre-change factory
    rhs_ref, jac_ref, _ = _build_reference_factory(
        eos_jax=eos,
        phase_params=params,
        mesh_arrays=mesh,
        boundary_params=bc,
        heating_array=heating,
        scales=scales,
        core_bc_mode=core_bc_mode,
        radio_isotope_params=radio_params,
    )

    test_states = [
        np.full(n_stag, 2200.0 / 3.0e3),
        np.full(n_stag, 3050.0 / 3.0e3),
        np.linspace(2400.0, 3600.0, n_stag) / 3.0e3,
    ]
    t_nd = 0.5

    for y_stag in test_states:
        if core_bc_mode == 'energy_balance':
            y_nd = np.concatenate([y_stag, [1.0e-7 / 1e-6]])
        else:
            y_nd = y_stag

        ydot_cached = np.zeros(n_state)
        J_cached = np.zeros((n_state, n_state))
        assert rhs_cached(t_nd, y_nd, ydot_cached) == 0
        assert jac_cached(t_nd, y_nd, None, J_cached) == 0

        ydot_ref = np.zeros(n_state)
        J_ref = np.zeros((n_state, n_state))
        assert rhs_ref(t_nd, y_nd, ydot_ref) == 0
        assert jac_ref(t_nd, y_nd, None, J_ref) == 0

        max_f = np.max(np.abs(ydot_ref))
        assert np.isfinite(max_f) and max_f > 0.0
        diff_f = np.max(np.abs(ydot_cached - ydot_ref))
        assert diff_f <= 1e-14 * max_f, f'RHS diff {diff_f} > 1e-14 * {max_f}'

        max_J = np.max(np.abs(J_ref))
        assert np.isfinite(max_J) and max_J > 0.0
        diff_J = np.max(np.abs(J_cached - J_ref))
        assert diff_J <= 2e-12 * max_J, f'Jacobian diff {diff_J} > 2e-12 * {max_J}'


@needs_eos
def test_no_stale_values_c(shared_eos):
    """Verify that perturbing each pytree leaf alters the output and matches reference."""
    eos = shared_eos
    params = PhaseParams()
    mesh = _make_mesh(N=8)
    bc = _make_bc(mesh)
    n_stag = 8
    heating = np.full(n_stag, 1e-12)
    radio = _make_radio_tuple()
    scales = NonDimScales(state_scale=np.full(n_stag, 3.0e3), t_ref=100.0)
    y_nd = np.full(n_stag, 3050.0 / 3.0e3)
    t_nd = 0.5

    # Base evaluation
    rhs_base, _, _ = build_jax_rhs_and_jacobian(
        eos, params, mesh, bc, heating, scales, 'quasi_steady', radio
    )
    ydot_base = np.zeros(n_stag)
    rhs_base(t_nd, y_nd, ydot_base)

    # Reference base evaluation
    ref_base, _, _ = _build_reference_factory(
        eos, params, mesh, bc, heating, scales, 'quasi_steady', radio
    )
    ydot_ref_base = np.zeros(n_stag)
    ret_ref_base = ref_base(t_nd, y_nd, ydot_ref_base)
    assert ret_ref_base == 0
    max_f_base = np.max(np.abs(ydot_ref_base))
    assert np.isfinite(max_f_base) and max_f_base > 0.0

    # Leaves to perturb
    perturbations = [
        ('mesh.P_stag', lambda: (_make_mesh(N=8, scale_p=1.1), bc, heating, scales, radio)),
        (
            'bc.outer_bc_value',
            lambda: (mesh, _make_bc(mesh, outer_val=500.0), heating, scales, radio),
        ),
        ('heating', lambda: (mesh, bc, heating + 1e-10, scales, radio)),
        ('radio', lambda: (mesh, bc, heating, scales, (radio[0] * 2.0, *radio[1:]))),
        (
            'state_scale',
            lambda: (
                mesh,
                bc,
                heating,
                NonDimScales(state_scale=scales.state_scale + 50.0, t_ref=scales.t_ref),
                radio,
            ),
        ),
        (
            't_ref',
            lambda: (
                mesh,
                bc,
                heating,
                NonDimScales(state_scale=scales.state_scale, t_ref=scales.t_ref + 20.0),
                radio,
            ),
        ),
    ]

    for label, perturber in perturbations:
        p_mesh, p_bc, p_heating, p_scales, p_radio = perturber()

        # 1. Perturbation must change reference output by > 1e-6 * max|f|
        ref_rhs, _, _ = _build_reference_factory(
            eos, params, p_mesh, p_bc, p_heating, p_scales, 'quasi_steady', p_radio
        )
        ydot_ref = np.zeros(n_stag)
        ret_ref = ref_rhs(t_nd, y_nd, ydot_ref)
        assert ret_ref == 0
        ref_diff = np.max(np.abs(ydot_ref - ydot_ref_base))
        assert ref_diff > 1e-6 * max_f_base, (
            f'Perturbation {label} does not change reference output enough: {ref_diff} <= 1e-6 * {max_f_base}'
        )

        # 2. Cached implementation under test
        p_rhs, _, _ = build_jax_rhs_and_jacobian(
            eos, params, p_mesh, p_bc, p_heating, p_scales, 'quasi_steady', p_radio
        )
        ydot_pert = np.zeros(n_stag)
        ret_pert = p_rhs(t_nd, y_nd, ydot_pert)
        assert ret_pert == 0

        # 3. Output must differ from base (no stale value)
        diff_from_base = np.max(np.abs(ydot_pert - ydot_base))
        assert diff_from_base > 1e-6 * np.max(np.abs(ydot_base)), (
            f'Leaf {label} produced stale value'
        )

        # 4. Output must match independent reference build
        diff_from_ref = np.max(np.abs(ydot_pert - ydot_ref))
        assert diff_from_ref <= 1e-14 * np.max(np.abs(ydot_ref)), (
            f'Leaf {label} diverged from reference'
        )


def test_id_reuse_guard_d():
    """Verify that if an object address is reused without identity, cache misses."""
    from unittest.mock import patch

    class Dummy:
        pass

    obj1 = Dummy()
    obj2 = Dummy()
    eos = Dummy()

    # Pre-populate cache with obj1
    key = ('quasi_steady', False, id(obj1), id(eos))
    _JIT_CACHE[key] = _JitCacheEntry('dummy_rhs', 'dummy_jac', obj1, eos)

    # Look up with obj1 -> HIT
    r, j, hit = _get_or_create_jitted('quasi_steady', False, obj1, eos)
    assert hit is True
    assert r == 'dummy_rhs'

    # Simulate address collision: obj2 with same key as obj1 -> MISS
    _JIT_CACHE[('quasi_steady', False, id(obj2), id(eos))] = _JitCacheEntry(
        'stale_rhs', 'stale_jac', obj1, eos
    )
    with patch(
        'aragog.solver.cvode_jax._make_jitted_rhs_and_jacobian',
        return_value=('fresh_rhs', 'fresh_jac'),
    ):
        r2, j2, hit2 = _get_or_create_jitted('quasi_steady', False, obj2, eos)
        assert hit2 is False
        assert r2 == 'fresh_rhs'


@needs_eos
def test_radio_params_shape_mismatch_and_empty(shared_eos):
    """Verify radio parameter shape consistency check and empty array handling."""
    n = 10
    mesh_arrays = _make_mesh(n)
    bp = _make_bc(mesh_arrays)
    heating = np.zeros(n)
    scales = NonDimScales(state_scale=np.ones(n), rhs_scale=np.ones(n), t_ref=1.0)
    pp = PhaseParams()
    eos = shared_eos

    # Mismatched shapes: 2 isotopes vs 3 isotopes
    mismatched = (
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match='identical shapes'):
        build_jax_rhs_and_jacobian(
            eos, pp, mesh_arrays, bp, heating, scales, radio_isotope_params=mismatched
        )

    # Empty arrays: length 5 tuple of 0-element arrays
    empty_arrays = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )
    # Should succeed with use_radio=False
    rhs_fn, jacfn, info = build_jax_rhs_and_jacobian(
        eos, pp, mesh_arrays, bp, heating, scales, radio_isotope_params=empty_arrays
    )
    assert rhs_fn is not None
    y = np.ones(n)
    ydot = np.zeros(n)
    assert rhs_fn(0.0, y, ydot) == 0
    assert np.all(np.isfinite(ydot))


@needs_eos
def test_lru_eviction_when_cache_exceeds_maxsize(shared_eos):
    """Verify 9 distinct keys evict the first key; 10th call traces again with identical output."""
    eos = shared_eos

    # Maintain references to 9 distinct PhaseParams instances so addresses are not recycled
    params_list = [PhaseParams() for _ in range(9)]
    n = 10
    mesh = _make_mesh(n)
    bp = _make_bc(mesh)
    heating = np.zeros(n)
    scales = NonDimScales(state_scale=np.ones(n), rhs_scale=np.ones(n), t_ref=1.0)
    y_nd = np.full(n, 1.0)
    ydot = np.zeros(n)

    # 1. First key call
    rhs_0, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params_list[0],
        mesh_arrays=mesh,
        boundary_params=bp,
        heating_array=heating,
        scales=scales,
    )
    rhs_0(0.0, y_nd, ydot)
    ydot_0 = ydot.copy()
    trace_rhs_after_first = _TRACE_COUNTERS['rhs']
    assert trace_rhs_after_first >= 1

    # 2. Calls 2 through 9 with distinct PhaseParams (total 9 keys inserted)
    for i in range(1, 9):
        rhs_i, _, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos,
            phase_params=params_list[i],
            mesh_arrays=mesh,
            boundary_params=bp,
            heating_array=heating,
            scales=scales,
        )
        rhs_i(0.0, y_nd, ydot)

    # Verify cache size is capped at _CACHE_MAXSIZE (8) and first key was evicted
    assert len(_JIT_CACHE) == _CACHE_MAXSIZE
    key_0 = ('quasi_steady', False, id(params_list[0]), id(eos))
    assert key_0 not in _JIT_CACHE

    trace_rhs_before_10th = _TRACE_COUNTERS['rhs']

    # 3. 10th call requesting first key again: must trace anew
    rhs_10, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=params_list[0],
        mesh_arrays=mesh,
        boundary_params=bp,
        heating_array=heating,
        scales=scales,
    )
    rhs_10(0.0, y_nd, ydot)

    assert _TRACE_COUNTERS['rhs'] == trace_rhs_before_10th + 1

    # Verify results match reference build
    rhs_ref, _, _ = _build_reference_factory(
        eos_jax=eos,
        phase_params=params_list[0],
        mesh_arrays=mesh,
        boundary_params=bp,
        heating_array=heating,
        scales=scales,
    )
    ydot_ref = np.zeros(n)
    rhs_ref(0.0, y_nd, ydot_ref)
    assert np.allclose(ydot, ydot_ref, atol=1e-12)
    assert np.allclose(ydot, ydot_0, atol=1e-12)


@needs_eos
def test_lru_cache_access_order_move_to_end(shared_eos):
    """Verify accessing an earlier key moves it to most-recent, evicting the next oldest."""
    eos = shared_eos
    params_list = [PhaseParams() for _ in range(9)]

    # 1. Insert 8 keys (keys 0 through 7)
    for i in range(8):
        _get_or_create_jitted('quasi_steady', False, params_list[i], eos)

    assert len(_JIT_CACHE) == _CACHE_MAXSIZE
    key_0 = ('quasi_steady', False, id(params_list[0]), id(eos))
    key_1 = ('quasi_steady', False, id(params_list[1]), id(eos))
    assert key_0 in _JIT_CACHE
    assert key_1 in _JIT_CACHE

    # 2. Access key 0: move_to_end should move key 0 to the MRU position
    _, _, hit = _get_or_create_jitted('quasi_steady', False, params_list[0], eos)
    assert hit is True

    # 3. Insert 9th key (key 8): evicts the LRU item, which must now be key 1, NOT key 0
    key_8 = ('quasi_steady', False, id(params_list[8]), id(eos))
    _get_or_create_jitted('quasi_steady', False, params_list[8], eos)

    assert len(_JIT_CACHE) == _CACHE_MAXSIZE
    assert key_0 in _JIT_CACHE
    assert key_1 not in _JIT_CACHE
    assert key_8 in _JIT_CACHE


@needs_eos
def test_eos_jax_identity_guard_misses_on_forged_colliding_key(shared_eos):
    """Verify cache miss when phase_params matches but eos_jax object differs."""
    import copy

    eos1 = shared_eos
    eos2 = copy.copy(eos1)
    assert eos1 is not eos2
    params = PhaseParams()

    # Populate cache for eos1
    rhs_1, jac_1, hit_1 = _get_or_create_jitted('quasi_steady', False, params, eos1)
    assert hit_1 is False

    # Forge a cache collision entry keyed on id(eos2) but pointing to eos1
    colliding_key = ('quasi_steady', False, id(params), id(eos2))
    _JIT_CACHE[colliding_key] = _JitCacheEntry(rhs_1, jac_1, params, eos1)

    # Calling with eos2 must detect that entry.eos_jax is not eos2, evict, and recompile
    rhs_2, jac_2, hit_2 = _get_or_create_jitted('quasi_steady', False, params, eos2)
    assert hit_2 is False
    assert _JIT_CACHE[colliding_key].eos_jax is eos2


def test_compute_radio_heating_closed_form():
    """Verify compute_radio_heating matches analytical sum and Jacobian is finite for amp=0."""
    from aragog.jax.solver import compute_radio_heating

    # 3 isotopes of different half-lives; 3rd has zero amplitude (abundance=0)
    hp = np.array([4.38e-11, 5.68e-11, 1.0e-10])
    ab = np.array([1.0, 1.0, 0.0])
    cn = np.array([3.1e-8, 1.24e-7, 1.0e-6])
    t0 = np.array([4.55, 4.55, 4.55])
    hl = np.array([7.17e5, 2.6e6, 1.0e5])
    radio_arrays = (
        jnp.asarray(hp),
        jnp.asarray(ab),
        jnp.asarray(cn),
        jnp.asarray(t0),
        jnp.asarray(hl),
    )

    t_val = 1.5e6  # t != t0
    # Closed-form analytical formula
    expected_sum = np.sum(hp * ab * cn * np.exp(np.log(2.0) * (t0 - t_val) / hl))

    calc_val = float(compute_radio_heating(t_val, radio_arrays))
    assert np.isclose(calc_val, expected_sum, rtol=1e-14, atol=1e-25)

    # Check Jacobian with respect to t is finite and non-NaN
    grad_fn = jax.grad(compute_radio_heating, argnums=0)
    dt_val = float(grad_fn(t_val, radio_arrays))
    assert np.isfinite(dt_val)
    assert not np.isnan(dt_val)

    expected_dt = np.sum(
        hp * ab * cn * (-np.log(2.0) / hl) * np.exp(np.log(2.0) * (t0 - t_val) / hl)
    )
    assert np.isclose(dt_val, expected_dt, rtol=1e-14, atol=1e-25)
