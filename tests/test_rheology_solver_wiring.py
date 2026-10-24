"""Solver-wiring tests for the solid-state rheology and stagnant-lid closure.

Covers:
- Row 7 viscous mixing-length scaling exponents in both NumPy and JAX paths
  (``u_visc ~ q^(3/4)``, ``Re ~ q^1``, ``kappa_h ~ q^1``).
- Two-pass evaluation order in ``EntropyState.update`` (NumPy) and
  ``compute_fluxes`` / ``dSdt`` (JAX), plus non-local Jacobian coupling
  through the surface stagnant-lid diagnostic.
- Per-RHS call-count contracts from Ruling 29 / Row 11 (zero ``_validate_inputs``
  calls in the RHS, zero extra EOS solidus lookups, and exact ``_core`` kernel
  invocation counts).
- ``_check_rheology_invariants`` positive and negative checks on ``EntropySolver``.
- ``A2-2`` NumPy vs JAX right-hand-side parity across 100 states of an
  ``abe_solid.toml`` cooling trajectory with ``rheology.enabled = True``.
"""

from __future__ import annotations

import functools
import importlib.resources
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aragog.rheology as rheo_mod
import aragog.rheology_lid as rheo_lid_mod
from aragog.jax.phase import (
    MeshArrays,
    PhaseParams,
    compute_fluxes,
    compute_mlt,
    entropy_at_basic_nodes,
    evaluate_phase,
)
from aragog.jax.solver import (
    BoundaryParams,
    compute_radio_heating,
    dSdt,
)
from aragog.parser import Parameters
from aragog.rheology import SolidRheologyParams
from aragog.solver import EntropySolver
from tests.conftest import EOS_DIR, entropy_eos_copy, entropy_eos_jax

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


@functools.cache
def _resolve_eos_dir() -> Path:
    """Resolve SPIDER P-S table directory or generate a synthetic fallback."""
    candidates = [
        EOS_DIR,
        Path('/Users/timlichtenberg/work/aragog-test-data-dev3set/spider_eos'),
        Path('/Users/timlichtenberg/work/fwl_data-dev-5/aragog/spider_eos'),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand

    tmp_dir = Path(tempfile.mkdtemp(prefix='aragog_rheo_eos_'))
    p_grid = np.linspace(0.0, 150.0e9, 16)
    s_grid = np.linspace(500.0, 4000.0, 16)
    np_len, ns_len = len(p_grid), len(s_grid)

    def _write_2d(fname: str, fn) -> None:
        lines = [f'# 2 {np_len} {ns_len}', '# 1.0 1.0 1.0']
        for s_val in s_grid:
            for p_val in p_grid:
                lines.append(f'{p_val:.8e} {s_val:.8e} {fn(p_val, s_val):.8e}')
        (tmp_dir / fname).write_text('\n'.join(lines) + '\n', encoding='utf-8')

    _write_2d(
        'temperature_solid.dat',
        lambda p, s: 300.0 + 0.85 * s + 8.0e-9 * p,
    )
    _write_2d(
        'temperature_melt.dat',
        lambda p, s: 500.0 + 0.90 * s + 1.0e-8 * p,
    )
    _write_2d('density_solid.dat', lambda p, s: 4200.0 + 8.0e-9 * p - 0.05 * s)
    _write_2d('density_melt.dat', lambda p, s: 3800.0 + 9.0e-9 * p - 0.06 * s)
    _write_2d('heat_capacity_solid.dat', lambda p, s: 1200.0 + 0.01 * s)
    _write_2d('heat_capacity_melt.dat', lambda p, s: 1400.0 + 0.01 * s)
    _write_2d('adiabat_temp_grad_solid.dat', lambda p, s: 1.2e-8 + 1.0e-13 * s)
    _write_2d('adiabat_temp_grad_melt.dat', lambda p, s: 1.5e-8 + 1.0e-13 * s)

    sol_lines = [f'# 2 {np_len}', '# 1.0 1.0']
    liq_lines = [f'# 2 {np_len}', '# 1.0 1.0']
    for p_val in p_grid:
        s_sol = 1800.0 + 2.0e-9 * p_val
        s_liq = 2500.0 + 2.5e-9 * p_val
        sol_lines.append(f'{p_val:.8e} {s_sol:.8e}')
        liq_lines.append(f'{p_val:.8e} {s_liq:.8e}')
    (tmp_dir / 'solidus_P-S.dat').write_text('\n'.join(sol_lines) + '\n', encoding='utf-8')
    (tmp_dir / 'liquidus_P-S.dat').write_text('\n'.join(liq_lines) + '\n', encoding='utf-8')
    return tmp_dir


def _make_abe_solid_params(
    rheology: SolidRheologyParams,
    *,
    number_of_nodes: int = 25,
) -> Parameters:
    """Load ``abe_solid.toml`` parameters with an overridden rheology block."""
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        params = Parameters.from_file(p)
    return replace(
        params,
        mesh=replace(params.mesh, number_of_nodes=number_of_nodes),
        phase_solid=replace(params.phase_solid, rheology=rheology),
        phase_mixed=replace(params.phase_mixed, matprop_smooth_width=0.01),
    )


def _make_solver(params: Parameters) -> EntropySolver:
    """Build and initialize an EntropySolver using the SPIDER P-S tables."""
    eos_np = entropy_eos_copy(_resolve_eos_dir(), identity_alpha=True)
    solver = EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()
    return solver


def _build_jax_bundle(solver: EntropySolver):
    """Build JAX (mesh_jax, eos_jax, params_jax, dSdt_args) matching ``solver``."""
    params = solver.parameters
    eos_jax = entropy_eos_jax(_resolve_eos_dir())
    mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
    params_jax = PhaseParams(
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
        rheology=params.phase_solid.rheology,
    )
    r_cmb = float(mesh_jax.radii_basic[0])
    r_next = float(mesh_jax.radii_basic[1])
    bc_jax = BoundaryParams(
        outer_bc_type=int(params.boundary_conditions.outer_boundary_condition),
        outer_bc_value=float(params.boundary_conditions.outer_boundary_value),
        emissivity=float(params.boundary_conditions.emissivity),
        T_eq=float(params.boundary_conditions.equilibrium_temperature),
        inner_bc_type=int(params.boundary_conditions.inner_boundary_condition),
        inner_bc_value=float(params.boundary_conditions.inner_boundary_value),
        core_density=float(params.mesh.core_density),
        core_heat_capacity=float(params.boundary_conditions.core_heat_capacity),
        tfac_core_avg=float(solver._core_tfac),
        cmb_area=4.0 * np.pi * r_cmb**2,
        core_M=float(solver._core_cap / params.boundary_conditions.core_heat_capacity),
        cmb_dr_cmb=r_next - r_cmb,
        param_utbl=bool(params.boundary_conditions.SOT_Look_Up),
        param_utbl_const=float(params.boundary_conditions.surface_temperature_Look_Up),
    )
    if params.energy.radionuclides and params.radionuclides:
        radio_arrays = (
            jnp.asarray([r.heat_production for r in params.radionuclides], dtype=jnp.float64),
            jnp.asarray([r.abundance for r in params.radionuclides], dtype=jnp.float64),
            jnp.asarray([r.concentration for r in params.radionuclides], dtype=jnp.float64),
            jnp.asarray([r.t0_years for r in params.radionuclides], dtype=jnp.float64),
            jnp.asarray([r.half_life_years for r in params.radionuclides], dtype=jnp.float64),
        )
        H_radio_fn = functools.partial(compute_radio_heating, radio_arrays=radio_arrays)
    else:
        H_radio_fn = None
    H_internal = jnp.zeros(solver._n_stag, dtype=jnp.float64)
    args = (eos_jax, params_jax, mesh_jax, bc_jax, H_internal)
    if H_radio_fn is not None:
        args = args + (H_radio_fn,)
    return mesh_jax, eos_jax, params_jax, args


def _make_solid_profile(
    solver: EntropySolver, s_low: float = 1100.0, s_high: float = 1900.0
) -> np.ndarray:
    """Construct a super-adiabatic solid/mushy entropy profile on staggered nodes."""
    return np.linspace(s_high, s_low, solver._n_stag)


# ============================================================================
# 1. Row 7 Exponents in Both Paths (u_visc ~ q^(3/4), Re ~ q, kappa_h ~ q)
# ============================================================================


def test_row7_scaling_exponents_numpy() -> None:
    """In EntropyState.update, scaling the viscous mixing-length factor by q
    scales u_visc as q^(3/4) and Re, kappa_h as q^1 in the viscous limit."""
    rheo = SolidRheologyParams(
        enabled=True,
        yield_stress_c=1.0e30,
        yield_stress_max=1.0e30,
        viscosity_max_log10=28.0,
    )
    params = _make_abe_solid_params(rheo, number_of_nodes=25)
    params = replace(
        params,
        phase_solid=replace(params.phase_solid, viscosity=1.0e22, rheology=rheo),
    )
    solver = _make_solver(params)

    s_stag = _make_solid_profile(solver, s_low=900.0, s_high=1400.0)
    orig_mlt_core = rheo_mod._viscous_mixing_length_factor_core

    def _eval_at_scale(q_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _scaled_mlt_core(*args, **kwargs):
            return q_scale * orig_mlt_core(*args, **kwargs)

        with (
            patch.object(
                rheo_mod, '_viscous_mixing_length_factor_core', side_effect=_scaled_mlt_core
            ),
            patch(
                'aragog.solver.entropy_state._viscous_mixing_length_factor_core',
                side_effect=_scaled_mlt_core,
            ),
        ):
            solver.state.update(s_stag, 0.0)
            u_visc = solver.state._last_viscous_velocity.copy()
            re_num = solver.state._last_reynolds.copy()
            kh_raw = solver.state._eddy_diffusivity.copy()
        return u_visc, re_num, kh_raw

    u1, re1, kh1 = _eval_at_scale(1.0)
    interior = slice(2, -2)
    assert np.all(u1[interior] > 0.0)
    assert np.all(re1[interior] > 0.0)
    assert np.all(kh1[interior] > 1.0e-8)

    for q in (0.25, 0.5, 2.0, 4.0):
        uq, req, khq = _eval_at_scale(q)
        exp_u = np.log(uq[interior] / u1[interior]) / np.log(q)
        exp_re = np.log(req[interior] / re1[interior]) / np.log(q)
        exp_kh = np.log(khq[interior] / kh1[interior]) / np.log(q)
        np.testing.assert_allclose(exp_u, 0.75, rtol=1.0e-11, atol=1.0e-11)
        np.testing.assert_allclose(exp_re, 1.0, rtol=1.0e-11, atol=1.0e-11)
        np.testing.assert_allclose(exp_kh, 1.0, rtol=1.0e-11, atol=1.0e-11)


def test_row7_scaling_exponents_jax() -> None:
    """In JAX compute_mlt, scaling the viscous mixing-length factor by q
    scales u_visc as q^(3/4) and Re, kappa_h as q^1 in the viscous limit."""
    rheo = SolidRheologyParams(
        enabled=True,
        yield_stress_c=1.0e30,
        yield_stress_max=1.0e30,
        viscosity_max_log10=28.0,
    )
    params = _make_abe_solid_params(rheo, number_of_nodes=25)
    params = replace(
        params,
        phase_solid=replace(params.phase_solid, viscosity=1.0e22, rheology=rheo),
    )
    solver = _make_solver(params)
    mesh_jax, eos_jax, params_jax, _ = _build_jax_bundle(solver)

    s_stag = jnp.asarray(_make_solid_profile(solver, s_low=900.0, s_high=1400.0))
    s_basic = entropy_at_basic_nodes(mesh_jax, s_stag)
    dSdr = mesh_jax.d_dr_matrix @ s_stag
    phase_basic = evaluate_phase(eos_jax, params_jax, mesh_jax.P_basic, s_basic)

    orig_mlt_core = rheo_mod._viscous_mixing_length_factor_core

    def _eval_jax_at_scale(q_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _scaled_mlt_core(*args, **kwargs):
            return q_scale * orig_mlt_core(*args, **kwargs)

        with patch(
            'aragog.jax.phase._viscous_mixing_length_factor_core', side_effect=_scaled_mlt_core
        ):
            kh, _, re_num, _ = compute_mlt(dSdr, phase_basic, mesh_jax, params_jax)
        re_arr = np.asarray(re_num)
        kh_arr = np.asarray(kh)
        u_visc = (
            re_arr
            * np.asarray(phase_basic.kinematic_viscosity)
            / np.asarray(mesh_jax.mixing_length)
        )
        return u_visc, re_arr, kh_arr

    u1, re1, kh1 = _eval_jax_at_scale(1.0)
    interior = slice(2, -2)
    assert np.all(u1[interior] > 0.0)

    for q in (0.25, 0.5, 2.0, 4.0):
        uq, req, khq = _eval_jax_at_scale(q)
        exp_u = np.log(uq[interior] / u1[interior]) / np.log(q)
        exp_re = np.log(req[interior] / re1[interior]) / np.log(q)
        exp_kh = np.log(khq[interior] / kh1[interior]) / np.log(q)
        np.testing.assert_allclose(exp_u, 0.75, rtol=1.0e-11, atol=1.0e-11)
        np.testing.assert_allclose(exp_re, 1.0, rtol=1.0e-11, atol=1.0e-11)
        np.testing.assert_allclose(exp_kh, 1.0, rtol=1.0e-11, atol=1.0e-11)


# ============================================================================
# 2. Two-Pass Order and Non-Local Jacobian Coupling
# ============================================================================


def test_two_pass_order_numpy() -> None:
    """Pass 1 unyielded fluxes feed _compute_stagnant_lid_state_core, and Pass 2
    uses eta_eff and theta in EntropyState.update."""
    rheo = SolidRheologyParams(
        enabled=True,
        yield_stress_c=5.0e6,
        yield_stress_mu=0.05,
        yield_stress_max=5.0e7,
    )
    params = _make_abe_solid_params(rheo, number_of_nodes=25)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver, s_low=1000.0, s_high=1850.0)

    events: list[tuple[str, dict]] = []
    orig_mlt = rheo_mod._viscous_mixing_length_factor_core
    orig_lid = rheo_lid_mod._compute_stagnant_lid_state_core
    orig_eff = rheo_lid_mod._compute_effective_viscosity_core

    def _spy_mlt(*args, **kwargs):
        res = orig_mlt(*args, **kwargs)
        events.append(('mlt', {'theta': kwargs.get('theta')}))
        return res

    def _spy_lid(*args, **kwargs):
        res = orig_lid(*args, **kwargs)
        events.append(
            (
                'lid',
                {
                    'convective_flux': np.asarray(kwargs['convective_flux']).copy(),
                    'total_flux': np.asarray(kwargs['total_flux']).copy(),
                    'theta': float(res['theta']),
                },
            )
        )
        return res

    def _spy_eff(*args, **kwargs):
        res = orig_eff(*args, **kwargs)
        events.append(
            (
                'eff',
                {
                    'tau_d': float(kwargs['tau_d']),
                    'tau_y_lid': float(kwargs['tau_y_lid']),
                    'eta_eff': np.asarray(res).copy(),
                },
            )
        )
        return res

    with (
        patch(
            'aragog.solver.entropy_state._viscous_mixing_length_factor_core',
            side_effect=_spy_mlt,
        ),
        patch(
            'aragog.solver.entropy_state._compute_stagnant_lid_state_core', side_effect=_spy_lid
        ),
        patch(
            'aragog.solver.entropy_state._compute_effective_viscosity_core',
            side_effect=_spy_eff,
        ),
    ):
        solver.state.update(s_stag, 0.0)

    names = [e[0] for e in events]
    assert names == ['mlt', 'lid', 'eff', 'mlt']
    assert events[0][1]['theta'] is None
    assert events[3][1]['theta'] is not None
    assert np.isclose(float(events[3][1]['theta']), events[1][1]['theta'], rtol=1.0e-15)
    jconv_pass1 = events[1][1]['convective_flux']
    jconv_pass2 = solver.state.jconv
    assert not np.allclose(jconv_pass1, jconv_pass2)


def test_two_pass_order_and_nonlocal_jacobian_jax() -> None:
    """In JAX, compute_fluxes executes the 2-pass lid closure and dSdt exhibits
    non-zero off-tridiagonal Jacobian entries d(dSdt_j)/dS_{N-1} when enabled."""
    rheo_on = SolidRheologyParams(
        enabled=True,
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    rheo_off = SolidRheologyParams(enabled=False)

    params_on = _make_abe_solid_params(rheo_on, number_of_nodes=20)
    solver_on = _make_solver(params_on)
    _, _, _, args_on = _build_jax_bundle(solver_on)

    params_off = _make_abe_solid_params(rheo_off, number_of_nodes=20)
    solver_off = _make_solver(params_off)
    _, _, _, args_off = _build_jax_bundle(solver_off)

    s_stag = jnp.asarray(_make_solid_profile(solver_on, s_low=1050.0, s_high=1850.0))

    jac_off = np.asarray(jax.jacfwd(lambda s: dSdt(0.0, s, args_off))(s_stag))
    jac_on = np.asarray(jax.jacfwd(lambda s: dSdt(0.0, s, args_on))(s_stag))

    assert np.all(np.isfinite(jac_on))
    assert jac_off[3, -1] == 0.0
    assert abs(jac_on[3, -1]) > 0.0


# ============================================================================
# 3. Ruling 29 / Row 11 Call-Count Contracts
# ============================================================================


def _measure_numpy_rhs_counts(rheology: SolidRheologyParams) -> dict[str, int]:
    """Measure exact per-RHS call counts on EntropySolver.dSdt."""
    params = _make_abe_solid_params(rheology, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver, s_low=1050.0, s_high=1850.0)

    counts = {
        'validate_inputs': 0,
        'arrhenius_viscosity_core': 0,
        'yield_stress_core': 0,
        'stagnant_lid_core': 0,
        'effective_viscosity_core': 0,
        'viscous_mlt_core': 0,
        'solid_temp_lookup': 0,
    }

    orig_val = rheo_mod._validate_inputs
    orig_arr = rheo_mod._compute_arrhenius_viscosity_core
    orig_yld = rheo_mod._compute_yield_stress_core
    orig_lid = rheo_lid_mod._compute_stagnant_lid_state_core
    orig_eff = rheo_lid_mod._compute_effective_viscosity_core
    orig_mlt = rheo_mod._viscous_mixing_length_factor_core
    orig_lookup = solver.state.phase_basic._lookup

    def _count_val(*a, **kw):
        counts['validate_inputs'] += 1
        return orig_val(*a, **kw)

    def _count_arr(*a, **kw):
        counts['arrhenius_viscosity_core'] += 1
        return orig_arr(*a, **kw)

    def _count_yld(*a, **kw):
        counts['yield_stress_core'] += 1
        return orig_yld(*a, **kw)

    def _count_lid(*a, **kw):
        counts['stagnant_lid_core'] += 1
        return orig_lid(*a, **kw)

    def _count_eff(*a, **kw):
        counts['effective_viscosity_core'] += 1
        return orig_eff(*a, **kw)

    def _count_mlt(*a, **kw):
        counts['viscous_mlt_core'] += 1
        return orig_mlt(*a, **kw)

    def _count_lookup(prop: str, P_arr, phase: str):
        if prop == 'temperature' and phase == 'solid':
            counts['solid_temp_lookup'] += 1
        return orig_lookup(prop, P_arr, phase)

    with (
        patch.object(rheo_mod, '_validate_inputs', side_effect=_count_val),
        patch.object(rheo_lid_mod, '_validate_inputs', side_effect=_count_val),
        patch.object(rheo_mod, '_compute_arrhenius_viscosity_core', side_effect=_count_arr),
        patch.object(rheo_lid_mod, '_compute_arrhenius_viscosity_core', side_effect=_count_arr),
        patch(
            'aragog.eos.entropy_phase._compute_arrhenius_viscosity_core', side_effect=_count_arr
        ),
        patch.object(rheo_mod, '_compute_yield_stress_core', side_effect=_count_yld),
        patch.object(rheo_lid_mod, '_compute_yield_stress_core', side_effect=_count_yld),
        patch('aragog.eos.entropy_phase._compute_yield_stress_core', side_effect=_count_yld),
        patch.object(rheo_lid_mod, '_compute_stagnant_lid_state_core', side_effect=_count_lid),
        patch(
            'aragog.solver.entropy_state._compute_stagnant_lid_state_core',
            side_effect=_count_lid,
        ),
        patch.object(rheo_lid_mod, '_compute_effective_viscosity_core', side_effect=_count_eff),
        patch(
            'aragog.solver.entropy_state._compute_effective_viscosity_core',
            side_effect=_count_eff,
        ),
        patch.object(rheo_mod, '_viscous_mixing_length_factor_core', side_effect=_count_mlt),
        patch(
            'aragog.solver.entropy_state._viscous_mixing_length_factor_core',
            side_effect=_count_mlt,
        ),
        patch.object(solver.state.phase_basic, '_lookup', side_effect=_count_lookup),
    ):
        solver.dSdt(0.0, s_stag)

    return counts


def test_rhs_zero_validation_calls() -> None:
    """R11-RHS-VALIDATE: _validate_inputs is called 0 times during RHS evaluation."""
    counts = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts['stagnant_lid_core'] == 1
    assert counts['validate_inputs'] == 0


def test_rhs_validate_inputs_zero_calls() -> None:
    """R11-VALIDATE-CALLS: both NumPy and JAX RHS paths invoke _validate_inputs 0 times."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    assert counts_np['stagnant_lid_core'] == 1
    assert counts_np['validate_inputs'] == 0

    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    _, _, params_jax, args = _build_jax_bundle(solver)
    assert params_jax.rheology.enabled is True
    s_stag = jnp.asarray(_make_solid_profile(solver, s_low=1050.0, s_high=1850.0))

    val_calls_jax = 0
    orig_val = rheo_mod._validate_inputs

    def _spy_val(*a, **kw):
        nonlocal val_calls_jax
        val_calls_jax += 1
        return orig_val(*a, **kw)

    with (
        patch.object(rheo_mod, '_validate_inputs', side_effect=_spy_val),
        patch.object(rheo_lid_mod, '_validate_inputs', side_effect=_spy_val),
    ):
        _ = dSdt(0.0, s_stag, args)
    assert val_calls_jax == 0


def test_rhs_zero_extra_eos_lookups() -> None:
    """R11-EXTRA-EOS: enabling rheology adds 0 extra EOS solidus temperature lookups."""
    counts_off = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=False))
    counts_on = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts_on['stagnant_lid_core'] == 1
    extra_np = counts_on['solid_temp_lookup'] - counts_off['solid_temp_lookup']
    assert extra_np == 0

    params_off = _make_abe_solid_params(SolidRheologyParams(enabled=False), number_of_nodes=20)
    solver_off = _make_solver(params_off)
    mesh_jax, eos_jax, params_jax_off, _ = _build_jax_bundle(solver_off)

    params_on = _make_abe_solid_params(SolidRheologyParams(enabled=True), number_of_nodes=20)
    solver_on = _make_solver(params_on)
    _, _, params_jax_on, _ = _build_jax_bundle(solver_on)
    assert params_jax_on.rheology.enabled is True
    s_stag = jnp.asarray(_make_solid_profile(solver_on, s_low=1050.0, s_high=1850.0))
    H_zero = jnp.zeros(solver_on._n_stag, dtype=jnp.float64)

    jax_lookups = {'off': 0, 'on': 0}
    orig_jax_lookup = eos_jax._lookup_at_phase_boundary

    def _make_jax_spy(key: str):
        def _spy(prop_name: str, P, phase: str):
            if prop_name == 'temperature' and phase == 'solid':
                jax_lookups[key] += 1
            return orig_jax_lookup(prop_name, P, phase)

        return _spy

    with patch.object(eos_jax, '_lookup_at_phase_boundary', side_effect=_make_jax_spy('off')):
        _ = compute_fluxes(s_stag, 0.0, eos_jax, params_jax_off, mesh_jax, H_zero)
    with patch.object(eos_jax, '_lookup_at_phase_boundary', side_effect=_make_jax_spy('on')):
        _ = compute_fluxes(s_stag, 0.0, eos_jax, params_jax_on, mesh_jax, H_zero)
    assert jax_lookups['on'] - jax_lookups['off'] == 0


def test_rhs_arrhenius_viscosity_core_call_counts() -> None:
    """R11-ARRHENIUS-CALLS: _compute_arrhenius_viscosity_core is called 2 times per RHS."""
    counts = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts['arrhenius_viscosity_core'] == 2


def test_rhs_yield_stress_core_call_counts() -> None:
    """R11-YIELD-CALLS: _compute_yield_stress_core is called 2 times per RHS."""
    counts = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts['yield_stress_core'] == 2


def test_rhs_stagnant_lid_core_call_counts() -> None:
    """R11-LID-CALLS: _compute_stagnant_lid_state_core is called 1 time per RHS."""
    counts = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts['stagnant_lid_core'] == 1


def test_rhs_effective_viscosity_core_call_counts() -> None:
    """R11-EFFVISC-CALLS: _compute_effective_viscosity_core is called 1 time per RHS."""
    counts = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts['effective_viscosity_core'] == 1


def test_rhs_viscous_mlt_core_call_counts() -> None:
    """R11-MLT-CALLS: _viscous_mixing_length_factor_core is called 2 times per RHS."""
    counts = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    assert counts['viscous_mlt_core'] == 2


def test_rheology_off_bitwise_parity_numpy_and_jax() -> None:
    """When rheology.enabled is False, 0 rheology _core calls occur and outputs
    differ from rheology.enabled=True on both NumPy and JAX paths."""
    counts_off = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=False))
    assert counts_off['arrhenius_viscosity_core'] == 0
    assert counts_off['yield_stress_core'] == 0
    assert counts_off['stagnant_lid_core'] == 0
    assert counts_off['effective_viscosity_core'] == 0
    assert counts_off['viscous_mlt_core'] == 0

    params_off = _make_abe_solid_params(SolidRheologyParams(enabled=False), number_of_nodes=20)
    solver_off = _make_solver(params_off)
    _, _, _, args_off = _build_jax_bundle(solver_off)

    params_on = _make_abe_solid_params(SolidRheologyParams(enabled=True), number_of_nodes=20)
    solver_on = _make_solver(params_on)
    _, _, _, args_on = _build_jax_bundle(solver_on)

    s_stag = _make_solid_profile(solver_off, s_low=1050.0, s_high=1850.0)
    rhs_np_off = solver_off.dSdt(0.0, s_stag)
    rhs_np_on = solver_on.dSdt(0.0, s_stag)
    assert not np.allclose(rhs_np_off, rhs_np_on)

    rhs_jax_off = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_off))
    rhs_jax_on = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_on))
    assert not np.allclose(rhs_jax_off, rhs_jax_on)


# ============================================================================
# 4. Invariant Checks on Accepted Steps (_check_rheology_invariants)
# ============================================================================


def test_check_rheology_invariants_passes_and_catches_violations() -> None:
    """_check_rheology_invariants passes on valid states and raises RuntimeError on
    eta_eff > eta_diff, non-finite eta_eff, or ceiling exceedance."""
    rheo = SolidRheologyParams(enabled=True, viscosity_max_log10=25.0)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver, s_low=1050.0, s_high=1850.0)
    solver.state.update(s_stag, 0.0)

    solver._check_rheology_invariants()

    orig_eta_eff = solver.state._eta_eff_basic.copy()
    eta_diff = solver.state.phase_basic.eta_diff().copy()
    solver.state._eta_eff_basic = eta_diff * 1.05
    with pytest.raises(RuntimeError, match='eta_eff'):
        solver._check_rheology_invariants()

    solver.state._eta_eff_basic = orig_eta_eff.copy()
    solver.state._eta_eff_basic[3] = np.nan
    with pytest.raises(RuntimeError, match='finite'):
        solver._check_rheology_invariants()

    solver.state._eta_eff_basic = orig_eta_eff.copy()
    solver.state._eta_eff_basic[3] = 10.0 ** (rheo.viscosity_max_log10 + 1.0)
    with pytest.raises(RuntimeError, match='ceiling|viscosity_max_log10'):
        solver._check_rheology_invariants()

    solver.state._eta_eff_basic = orig_eta_eff


# ============================================================================
# 5. A2-2: NumPy vs JAX RHS at 100 States of abe_solid.toml (rel diff <= 1e-6)
# ============================================================================


@pytest.mark.parametrize('closure_mode', ['lid', 'local'])
def test_a2_2_numpy_vs_jax_rhs_100_states_abe_solid(closure_mode: str) -> None:
    """A2-2: NumPy and JAX right-hand sides agree to relative difference <= 1e-6
    across 100 physical states spanning the abe_solid.toml trajectory with
    rheology enabled."""
    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode=closure_mode,
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    params = _make_abe_solid_params(rheo, number_of_nodes=30)
    solver = _make_solver(params)
    mesh_jax, eos_jax, params_jax, args = _build_jax_bundle(solver)
    assert solver.state.phase_basic._rheology.enabled is True
    assert params_jax.rheology.enabled is True

    H_zero = jnp.zeros(solver._n_stag, dtype=jnp.float64)
    jit_dSdt = jax.jit(lambda t, s: dSdt(t, s, args))
    jit_fluxes = jax.jit(
        lambda t, s: compute_fluxes(s, t, eos_jax, params_jax, mesh_jax, H_zero).heat_flux
    )

    n_stag = solver._n_stag
    s_bot_grid = np.linspace(3100.0, 1450.0, 100)
    delta_s_grid = np.linspace(50.0, 450.0, 100)
    xi = np.linspace(0.0, 1.0, n_stag)

    for idx in range(100):
        s_bot = s_bot_grid[idx]
        ds = delta_s_grid[idx]
        s_profile = s_bot - ds * (0.7 * xi + 0.3 * xi**2)
        t_yr = float(idx) * 1.0e4

        rhs_np = solver.dSdt(t_yr, s_profile)
        flux_np = solver.state.heat_flux.copy()

        s_jax = jnp.asarray(s_profile)
        rhs_jax = np.asarray(jit_dSdt(t_yr, s_jax))
        flux_jax = np.asarray(jit_fluxes(t_yr, s_jax))

        assert np.all(np.isfinite(rhs_np))
        assert np.all(np.isfinite(rhs_jax))

        denom_rhs = np.maximum(np.max(np.abs(rhs_np)), 1.0e-30)
        rel_rhs = float(np.max(np.abs(rhs_np - rhs_jax)) / denom_rhs)
        denom_flux = np.maximum(np.max(np.abs(flux_np[1:-1])), 1.0e-30)
        rel_flux = float(np.max(np.abs(flux_np[1:-1] - flux_jax[1:-1])) / denom_flux)

        assert rel_rhs <= 1.0e-6, (
            f'State {idx} ({closure_mode}): RHS rel_diff={rel_rhs:.3e} > 1e-6'
        )
        assert rel_flux <= 1.0e-6, (
            f'State {idx} ({closure_mode}): Flux rel_diff={rel_flux:.3e} > 1e-6'
        )
