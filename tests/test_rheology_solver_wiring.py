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
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aragog.jax.phase as jax_phase_mod
import aragog.rheology as rheo_mod
import aragog.rheology_lid as rheo_lid_mod
import aragog.solver.entropy_state as entropy_state_mod
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
    _no_radio,
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
        Path('/Users/timlichtenberg/work/ssc-verify-task6/test-data/spider_eos'),
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
        boundary_conditions=replace(
            params.boundary_conditions,
            core_bc='quasi_steady',
            inner_boundary_condition=1,
        ),
        energy=replace(params.energy, kappah_floor=0.0),
        phase_solid=replace(params.phase_solid, rheology=rheology),
        phase_mixed=replace(params.phase_mixed, matprop_smooth_width=0.01),
    )


def _make_solver(params: Parameters, *, identity_alpha: bool = True) -> EntropySolver:
    """Build and initialize an EntropySolver using the SPIDER P-S tables."""
    eos_np = entropy_eos_copy(_resolve_eos_dir(), identity_alpha=identity_alpha)
    solver = EntropySolver(params, entropy_eos=eos_np)
    solver.initialize()
    return solver


def _build_jax_bundle(
    solver: EntropySolver,
    rheology_override: SolidRheologyParams | None = None,
):
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
        rheology=(
            rheology_override if rheology_override is not None else params.phase_solid.rheology
        ),
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
        param_utbl=bool(params.boundary_conditions.param_utbl),
        param_utbl_const=float(params.boundary_conditions.param_utbl_const),
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
        H_radio_fn = _no_radio
    H_internal = jnp.zeros(solver._n_stag, dtype=jnp.float64)
    args = (eos_jax, params_jax, mesh_jax, bc_jax, H_internal, H_radio_fn)
    return mesh_jax, eos_jax, params_jax, args


def _make_solid_profile(
    solver: EntropySolver,
    s_low: float | None = None,
    s_high: float | None = None,
) -> np.ndarray:
    """Construct a super-adiabatic solid entropy profile on staggered nodes."""
    if s_low is None or s_high is None:
        eos = solver.state.phase_basic._eos
        p_b = np.asarray(solver.state.phase_basic.pressure, dtype=float)
        s_sol = np.asarray(eos.solidus_entropy(p_b), dtype=float)
        if s_low is None:
            s_low = float(s_sol[-1]) - 120.0
        if s_high is None:
            s_high = float(s_sol[0]) - 65.0
    return np.linspace(s_high, s_low, solver._n_stag)


# ============================================================================
# 1. Row 7 Exponents in Both Paths (u_visc ~ q^(3/4), Re ~ q, kappa_h ~ q)
# ============================================================================


def test_row7_scaling_exponents_numpy() -> None:
    """In EntropyState.update, scaling the viscous mixing-length factor by q
    scales u_visc as q^(3/4) and Re, kappa_h as q^1 in the viscous limit."""
    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='local',
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

    s_stag = _make_solid_profile(solver)
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
    assert np.all(kh1[interior] > 0.0)

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

    s_stag = jnp.asarray(_make_solid_profile(solver))
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
            kh_2, kc_2 = compute_mlt(dSdr, phase_basic, mesh_jax, params_jax)
            kh, kc, re_num, u_visc_jax = compute_mlt(
                dSdr, phase_basic, mesh_jax, params_jax, return_aux=True
            )
            np.testing.assert_array_equal(np.asarray(kh_2), np.asarray(kh))
            np.testing.assert_array_equal(np.asarray(kc_2), np.asarray(kc))
        re_arr = np.asarray(re_num)
        kh_arr = np.asarray(kh)
        u_visc = np.asarray(u_visc_jax)
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
        yield_stress_c=1.0e4,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e4,
    )
    params = _make_abe_solid_params(rheo, number_of_nodes=25)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)

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
    non-zero off-tridiagonal Jacobian entries when enabled."""
    rheo_on = SolidRheologyParams(
        enabled=True,
        yield_stress_c=1.0e4,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e4,
    )
    rheo_off = SolidRheologyParams(enabled=False)

    params_on = _make_abe_solid_params(rheo_on, number_of_nodes=20)
    solver_on = _make_solver(params_on)
    _, _, _, args_on = _build_jax_bundle(solver_on)

    params_off = _make_abe_solid_params(rheo_off, number_of_nodes=20)
    solver_off = _make_solver(params_off)
    _, _, _, args_off = _build_jax_bundle(solver_off)

    s_stag = jnp.asarray(_make_solid_profile(solver_on))

    jac_off = np.asarray(jax.jacfwd(lambda s: dSdt(0.0, s, args_off))(s_stag))
    jac_on = np.asarray(jax.jacfwd(lambda s: dSdt(0.0, s, args_on))(s_stag))

    assert np.all(np.isfinite(jac_on))
    assert jac_off[-4, -1] == 0.0
    assert jac_off[-5, -1] == 0.0
    assert abs(jac_on[-4, -1]) > 0.0
    assert abs(jac_on[-5, -1]) > 0.0


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
    orig_lookup = solver.state.phase_basic._eos._lookup_at_phase_boundary

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
        patch.object(
            solver.state.phase_basic._eos,
            '_lookup_at_phase_boundary',
            side_effect=_count_lookup,
        ),
    ):
        solver.dSdt(0.0, s_stag)

    return counts


def _measure_jax_rhs_counts(rheology: SolidRheologyParams) -> dict[str, int]:
    """Count per-RHS calls on the JAX dSdt path."""
    params = _make_abe_solid_params(rheology, number_of_nodes=20)
    solver = _make_solver(params)
    _, _, _, args = _build_jax_bundle(solver)
    s_stag = jnp.asarray(_make_solid_profile(solver, s_low=1050.0, s_high=1850.0))

    counts = {
        'validate_inputs': 0,
        'arrhenius_viscosity_core': 0,
        'yield_stress_core': 0,
        'stagnant_lid_core': 0,
        'effective_viscosity_core': 0,
        'viscous_mlt_core': 0,
    }
    orig_val = rheo_mod._validate_inputs
    orig_arr = rheo_mod._compute_arrhenius_viscosity_core
    orig_yld = rheo_mod._compute_yield_stress_core
    orig_lid = rheo_lid_mod._compute_stagnant_lid_state_core
    orig_eff = rheo_lid_mod._compute_effective_viscosity_core
    orig_mlt = rheo_mod._viscous_mixing_length_factor_core

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

    with (
        patch.object(rheo_mod, '_validate_inputs', side_effect=_count_val),
        patch.object(rheo_lid_mod, '_validate_inputs', side_effect=_count_val),
        patch.object(rheo_mod, '_compute_arrhenius_viscosity_core', side_effect=_count_arr),
        patch.object(rheo_lid_mod, '_compute_arrhenius_viscosity_core', side_effect=_count_arr),
        patch.object(
            jax_phase_mod, '_compute_arrhenius_viscosity_core', side_effect=_count_arr
        ),
        patch.object(rheo_mod, '_compute_yield_stress_core', side_effect=_count_yld),
        patch.object(rheo_lid_mod, '_compute_yield_stress_core', side_effect=_count_yld),
        patch.object(jax_phase_mod, '_compute_yield_stress_core', side_effect=_count_yld),
        patch.object(rheo_lid_mod, '_compute_stagnant_lid_state_core', side_effect=_count_lid),
        patch.object(jax_phase_mod, '_compute_stagnant_lid_state_core', side_effect=_count_lid),
        patch.object(rheo_lid_mod, '_compute_effective_viscosity_core', side_effect=_count_eff),
        patch.object(
            jax_phase_mod, '_compute_effective_viscosity_core', side_effect=_count_eff
        ),
        patch.object(rheo_mod, '_viscous_mixing_length_factor_core', side_effect=_count_mlt),
        patch.object(
            jax_phase_mod, '_viscous_mixing_length_factor_core', side_effect=_count_mlt
        ),
    ):
        _ = dSdt(0.0, s_stag, args)

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
    orig_jax_lookup = type(eos_jax)._lookup_at_phase_boundary

    def _make_jax_spy(key: str):
        def _spy(self_eos, prop_name: str, P, phase: str):
            if prop_name == 'temperature' and phase == 'solid':
                jax_lookups[key] += 1
            return orig_jax_lookup(self_eos, prop_name, P, phase)

        return _spy

    with patch.object(
        type(eos_jax),
        '_lookup_at_phase_boundary',
        autospec=True,
        side_effect=_make_jax_spy('off'),
    ):
        _ = compute_fluxes(s_stag, 0.0, eos_jax, params_jax_off, mesh_jax, H_zero)
    with patch.object(
        type(eos_jax),
        '_lookup_at_phase_boundary',
        autospec=True,
        side_effect=_make_jax_spy('on'),
    ):
        _ = compute_fluxes(s_stag, 0.0, eos_jax, params_jax_on, mesh_jax, H_zero)
    assert jax_lookups['on'] - jax_lookups['off'] == 0


def test_rhs_arrhenius_viscosity_core_call_counts() -> None:
    """R11-ARRHENIUS-CALLS: _compute_arrhenius_viscosity_core is called 2 times per RHS."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['arrhenius_viscosity_core'] == 2
    assert counts_jax['arrhenius_viscosity_core'] == 2


def test_rhs_yield_stress_core_call_counts() -> None:
    """R11-YIELD-CALLS: _compute_yield_stress_core is called 2 times per RHS."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['yield_stress_core'] == 2
    assert counts_jax['yield_stress_core'] == 2


def test_rhs_stagnant_lid_core_call_counts() -> None:
    """R11-LID-CALLS: _compute_stagnant_lid_state_core is called 1 time per RHS in 'lid' mode
    and 0 times per RHS in 'local' mode (Ruling 58d 2.3, Ruling 58e)."""
    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['stagnant_lid_core'] == 1
    assert counts_jax['stagnant_lid_core'] == 1

    rheo_local = SolidRheologyParams(enabled=True, stress_closure_mode='local')
    counts_np_local = _measure_numpy_rhs_counts(rheo_local)
    counts_jax_local = _measure_jax_rhs_counts(rheo_local)
    assert counts_np_local['stagnant_lid_core'] == 0
    assert counts_jax_local['stagnant_lid_core'] == 0
    assert counts_np_local['arrhenius_viscosity_core'] == 1
    assert counts_jax_local['arrhenius_viscosity_core'] == 1
    assert counts_np_local['yield_stress_core'] == 1
    assert counts_jax_local['yield_stress_core'] == 1
    assert counts_np_local['effective_viscosity_core'] == 1
    assert counts_jax_local['effective_viscosity_core'] == 1
    assert counts_np_local['viscous_mlt_core'] == 2
    assert counts_jax_local['viscous_mlt_core'] == 2

    solver_local = _make_solver(_make_abe_solid_params(rheo_local, number_of_nodes=20))
    s_stag = _make_solid_profile(solver_local)
    solver_local.state.update(s_stag, 0.0)
    assert solver_local.state._lid_state is None


def test_rhs_effective_viscosity_core_call_counts() -> None:
    """R11-EFFVISC-CALLS: _compute_effective_viscosity_core is called 1 time per RHS."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['effective_viscosity_core'] == 1
    assert counts_jax['effective_viscosity_core'] == 1


def test_rhs_viscous_mlt_core_call_counts(monkeypatch) -> None:
    """R11-MLT-CALLS: _viscous_mixing_length_factor_core is called 2 times per RHS,
    and local mode reuses q_025_1 * mixing_length in _compute_effective_viscosity_core."""
    for mode in ('lid', 'local'):
        rheo = SolidRheologyParams(enabled=True, stress_closure_mode=mode)
        counts_np = _measure_numpy_rhs_counts(rheo)
        counts_jax = _measure_jax_rhs_counts(rheo)
        assert counts_np['viscous_mlt_core'] == 2
        assert counts_jax['viscous_mlt_core'] == 2

    # Verify JAX local mode reuses q_025_1 * mesh.mixing_length and matches NumPy on yielding state
    rheo_loc_yld = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='local',
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
    )
    base_loc = _make_abe_solid_params(rheo_loc_yld, number_of_nodes=20)
    params_loc = replace(
        base_loc,
        phase_solid=replace(base_loc.phase_solid, viscosity=1.0e15),
    )
    solver_loc = _make_solver(params_loc)
    _, _, _, args_loc = _build_jax_bundle(solver_loc)
    s_stag = _make_solid_profile(solver_loc)

    captured_np: list[np.ndarray] = []
    captured_jax: list[np.ndarray] = []
    orig_eff_np = entropy_state_mod._compute_effective_viscosity_core
    orig_eff_jax = jax_phase_mod._compute_effective_viscosity_core

    def _spy_np(*a, **kw):
        if 'mixing_length' in kw and kw['mixing_length'] is not None:
            captured_np.append(np.asarray(kw['mixing_length'], dtype=float).copy())
        return orig_eff_np(*a, **kw)

    def _spy_jax(*a, **kw):
        if 'mixing_length' in kw and kw['mixing_length'] is not None:
            captured_jax.append(np.asarray(kw['mixing_length'], dtype=float).copy())
        return orig_eff_jax(*a, **kw)

    monkeypatch.setattr(entropy_state_mod, '_compute_effective_viscosity_core', _spy_np)
    monkeypatch.setattr(jax_phase_mod, '_compute_effective_viscosity_core', _spy_jax)

    rhs_np = solver_loc.dSdt(0.0, s_stag)
    rhs_jax = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_loc))
    assert len(captured_np) == 1
    assert len(captured_jax) == 1
    raw_ml = np.asarray(solver_loc.state._mixing_length, dtype=float)
    np.testing.assert_allclose(captured_jax[0], captured_np[0], rtol=1.0e-14, atol=0.0)
    assert not np.allclose(captured_jax[0], raw_ml, rtol=1.0e-3)
    denom = float(np.max(np.abs(rhs_np)))
    assert float(np.max(np.abs(rhs_np - rhs_jax)) / denom) <= 1.0e-6


def test_signed_negative_gravity_rheology_boundary_parity_numpy_and_jax() -> None:
    """Passing a signed (inward, negative) gravity at the evaluator produces identical
    stagnant-lid and effective-viscosity outputs to |g| on both NumPy and JAX paths."""
    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
    )
    base_params = _make_abe_solid_params(rheo, number_of_nodes=20)
    params = replace(
        base_params,
        phase_solid=replace(base_params.phase_solid, viscosity=1.0e15),
    )

    solver_pos = _make_solver(params)
    solver_neg = _make_solver(params)
    s_stag = _make_solid_profile(solver_pos)

    g_pos = np.abs(np.asarray(solver_pos.state.phase_basic.gravitational_acceleration())).copy()
    solver_pos.state.phase_basic._g = g_pos.copy()
    solver_neg.state.phase_basic._g = -g_pos.copy()

    rhs_np_pos = solver_pos.dSdt(0.0, s_stag)
    rhs_np_neg = solver_neg.dSdt(0.0, s_stag)
    lid_np_pos = solver_pos.state._lid_state
    lid_np_neg = solver_neg.state._lid_state
    assert lid_np_pos is not None and lid_np_neg is not None
    assert float(lid_np_neg['Ra_eff']) > 0.0
    assert float(lid_np_neg['tau_d']) > 0.0
    assert float(lid_np_neg['v_m']) > 0.0
    for key in ('Ra_eff', 'Ra_rh', 'v_m', 'tau_d', 'tau_d_lid', 'tau_buoy', 'theta', 'd_lid'):
        np.testing.assert_allclose(
            np.asarray(lid_np_neg[key], dtype=float),
            np.asarray(lid_np_pos[key], dtype=float),
            rtol=1.0e-14,
            atol=0.0,
        )
    np.testing.assert_allclose(
        solver_neg.state._eta_eff_basic,
        solver_pos.state._eta_eff_basic,
        rtol=1.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(rhs_np_neg, rhs_np_pos, rtol=1.0e-14, atol=0.0)

    import copy

    mesh_pos, eos_jax, params_jax, args_pos = _build_jax_bundle(solver_pos)
    mesh_neg = copy.copy(mesh_pos)
    object.__setattr__(mesh_neg, 'gravity', -jnp.abs(mesh_pos.gravity))
    args_neg = (args_pos[0], args_pos[1], mesh_neg, *args_pos[3:])

    rhs_jax_pos = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_pos))
    rhs_jax_neg = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_neg))
    np.testing.assert_allclose(rhs_jax_neg, rhs_jax_pos, rtol=1.0e-14, atol=0.0)
    denom = float(np.max(np.abs(rhs_np_pos)))
    assert float(np.max(np.abs(rhs_np_neg - rhs_jax_neg)) / denom) <= 1.0e-6


def test_rheology_off_bitwise_parity_numpy_and_jax() -> None:
    """When rheology.enabled is False, 0 rheology _core calls occur and outputs
    differ from rheology.enabled=True on both NumPy and JAX paths."""
    counts_off = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=False))
    assert counts_off['arrhenius_viscosity_core'] == 0
    assert counts_off['yield_stress_core'] == 0
    assert counts_off['stagnant_lid_core'] == 0
    assert counts_off['effective_viscosity_core'] == 0
    assert counts_off['viscous_mlt_core'] == 0

    counts_jax_off = _measure_jax_rhs_counts(SolidRheologyParams(enabled=False))
    assert counts_jax_off['arrhenius_viscosity_core'] == 0
    assert counts_jax_off['yield_stress_core'] == 0
    assert counts_jax_off['stagnant_lid_core'] == 0
    assert counts_jax_off['effective_viscosity_core'] == 0
    assert counts_jax_off['viscous_mlt_core'] == 0

    params_off = _make_abe_solid_params(SolidRheologyParams(enabled=False), number_of_nodes=20)
    solver_off = _make_solver(params_off)
    _, _, _, args_off = _build_jax_bundle(solver_off)

    params_on = _make_abe_solid_params(SolidRheologyParams(enabled=True), number_of_nodes=20)
    solver_on = _make_solver(params_on)
    _, _, _, args_on = _build_jax_bundle(solver_on)

    s_stag = _make_solid_profile(solver_off)
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
    rheo = SolidRheologyParams(enabled=True, viscosity_max_log10=26.0)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)
    solver.state.update(s_stag, 0.0)

    solver._check_rheology_invariants()

    orig_eta_eff = solver.state._eta_eff_basic.copy()
    eta_diff = solver.state.phase_basic.eta_diff().copy()
    solver.state._eta_eff_basic = orig_eta_eff.copy()
    solver.state._eta_eff_basic[0] = eta_diff[0] * 1.05
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

    assert solver._build_jac_sparsity() is None
    solver_local = _make_solver(
        _make_abe_solid_params(
            SolidRheologyParams(enabled=True, stress_closure_mode='local'),
            number_of_nodes=20,
        )
    )
    assert solver_local._build_jac_sparsity() is not None
    solver_off = _make_solver(
        _make_abe_solid_params(SolidRheologyParams(enabled=False), number_of_nodes=20)
    )
    assert solver_off._build_jac_sparsity() is not None


def test_validate_rheology_setup_rejects_invalid_params_and_mesh() -> None:
    """_validate_rheology_setup re-runs SolidRheologyParams.__post_init__ and validates
    basic-node mesh and phase invariants at solver setup."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    solver._validate_rheology_setup()

    # 1. Mutate frozen rheology param in-place to verify rheo.__post_init__() is called
    bad_rheo = SolidRheologyParams(enabled=True)
    object.__setattr__(bad_rheo, 'g_homologous', -5.0)
    bad_params = _make_abe_solid_params(bad_rheo, number_of_nodes=20)
    with pytest.raises(ValueError, match='g_homologous'):
        _make_solver(bad_params)

    # 2. Non-increasing basic-node radii
    orig_r = solver._r_basic_flat.copy()
    solver._r_basic_flat = orig_r[::-1].copy()
    with pytest.raises(ValueError, match='radii'):
        solver._validate_rheology_setup()
    solver._r_basic_flat = orig_r

    # 3. Negative mixing length
    orig_ml = solver.state._mixing_length.copy()
    solver.state._mixing_length = orig_ml.copy()
    solver.state._mixing_length[1] = -1.0
    with pytest.raises(ValueError, match='mixing_length'):
        solver._validate_rheology_setup()
    solver.state._mixing_length = orig_ml

    # 4. Non-positive viscosity_solid
    orig_vs = solver.state.phase_basic._visc_solid
    solver.state.phase_basic._visc_solid = 0.0
    with pytest.raises(ValueError, match='viscosity_solid'):
        solver._validate_rheology_setup()
    solver.state.phase_basic._visc_solid = orig_vs

    # 5. Non-positive inner basic-node radius r_b[0] <= 0 (Ruling 58d 1.2)
    orig_r = solver._r_basic_flat.copy()
    solver._r_basic_flat = orig_r.copy()
    solver._r_basic_flat[0] = 0.0
    with pytest.raises(ValueError, match='radii'):
        solver._validate_rheology_setup()
    solver._r_basic_flat = orig_r

    # 6. Staggered-node pressure < -1.0 Pa (Ruling 58d 1.2)
    orig_ps = solver.state.phase_staggered.pressure.copy()
    solver.state.phase_staggered.set_pressure(np.full_like(orig_ps, -5.0))
    with pytest.raises(ValueError, match='pressure'):
        solver._validate_rheology_setup()
    solver.state.phase_staggered.set_pressure(orig_ps)

    # 7. Non-positive basic or staggered gravity (Ruling 58d 1.2)
    orig_gb = solver.state.phase_basic._g.copy()
    solver.state.phase_basic._g = np.zeros_like(orig_gb)
    with pytest.raises(ValueError, match='gravitational_acceleration'):
        solver._validate_rheology_setup()
    solver.state.phase_basic._g = orig_gb

    orig_gs = solver.state.phase_staggered._g.copy()
    solver.state.phase_staggered._g = -np.ones_like(orig_gs)
    with pytest.raises(ValueError, match='gravitational_acceleration'):
        solver._validate_rheology_setup()
    solver.state.phase_staggered._g = orig_gs

    # 8. log10(viscosity_solid) >= viscosity_max_log10 (Ruling 58d 1.2)
    orig_vs = solver.state.phase_basic._visc_solid
    solver.state.phase_basic._visc_solid = 10.0 ** float(rheo.viscosity_max_log10)
    with pytest.raises(ValueError, match='viscosity_max_log10'):
        solver._validate_rheology_setup()
    solver.state.phase_basic._visc_solid = orig_vs


def test_check_rheology_invariants_checks_all_trace_columns_and_lid_none(monkeypatch) -> None:
    """Ruling 58d (1.1, 1.3): _check_rheology_invariants(trace_y, trace_t) checks every
    accepted step column in trace_y (not only the final column) and raises RuntimeError
    in 'lid' mode if lid_state is None."""
    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_good = _make_solid_profile(solver)

    # 1. lid_state is None in 'lid' mode must raise RuntimeError (Lead 1.3)
    solver.state.update(s_good, 0.0)
    solver.state._lid_state = None
    with pytest.raises(RuntimeError, match='lid_state'):
        solver._check_rheology_invariants()

    # 2. Intermediate step violation in trace_y column 1 must raise RuntimeError naming step 1
    trace_y = np.column_stack([s_good, s_good * 1.001, s_good])
    trace_t = np.array([0.0, 0.5, 1.0])
    orig_dSdt_single = solver._dSdt_single

    def _corrupt_step_1(t_val: float, y_col: np.ndarray):
        res = orig_dSdt_single(t_val, y_col)
        if abs(t_val - 0.5) < 1.0e-9:
            solver.state._eta_eff_basic = solver.state.phase_basic.eta_diff() * 10.0
        return res

    monkeypatch.setattr(solver, '_dSdt_single', _corrupt_step_1)
    with pytest.raises(RuntimeError, match='step 1'):
        solver._check_rheology_invariants(trace_y, trace_t)


def test_entropy_state_resets_lid_state_when_rheology_disabled() -> None:
    """Ruling 58d (2.1): EntropyState.update resets self._lid_state = None when
    rheology.enabled is False so stale lid diagnostics are never retained."""
    rheo_on = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo_on, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)

    solver.state.update(s_stag, 0.0)
    assert solver.state._lid_state is not None

    solver.state.phase_basic._rheology = SolidRheologyParams(enabled=False)
    solver.state.update(s_stag, 0.0)
    assert solver.state._lid_state is None


def test_solver_passes_abs_gravity_to_stagnant_lid_core() -> None:
    """Ruling 58d (2.2): Both EntropyState.update and aragog.jax.phase.compute_fluxes
    pass xp.abs(gravity) at the solver-to-rheology boundary."""
    rheo_on = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo_on, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)

    solver.state.update(s_stag, 0.0)
    lid_pos = dict(solver.state._lid_state)
    kappah_pos = solver.state._eddy_diffusivity.copy()

    orig_g = solver.state.phase_basic._g.copy()
    try:
        solver.state.phase_basic._g = -np.abs(orig_g)
        solver.state.update(s_stag, 0.0)
        lid_neg = solver.state._lid_state
        assert lid_neg is not None
        np.testing.assert_allclose(lid_neg['tau_d'], lid_pos['tau_d'], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(
            solver.state._eddy_diffusivity, kappah_pos, rtol=1e-14, atol=0.0
        )
    finally:
        solver.state.phase_basic._g = orig_g

    import equinox as eqx

    mesh_jax, eos_jax, params_jax, _ = _build_jax_bundle(solver)
    H_zero = jnp.zeros(solver._n_stag, dtype=jnp.float64)
    flux_pos = compute_fluxes(jnp.asarray(s_stag), 0.0, eos_jax, params_jax, mesh_jax, H_zero)
    mesh_neg = eqx.tree_at(lambda m: m.gravity, mesh_jax, -jnp.abs(mesh_jax.gravity))
    flux_neg = compute_fluxes(jnp.asarray(s_stag), 0.0, eos_jax, params_jax, mesh_neg, H_zero)
    np.testing.assert_allclose(
        np.asarray(flux_neg.eddy_diffusivity),
        np.asarray(flux_pos.eddy_diffusivity),
        rtol=1e-14,
        atol=0.0,
    )


def test_entropy_phase_const_mode_none_solidus_and_dirty_flag() -> None:
    """Ruling 58d (3.1, 3.2, 3.3):
    - (3.1) _solidus_temperature = None uses linear solidus fallback (finite eta_diff, no NaN);
    - (3.2) const_properties=True with rheology.enabled=True evaluates Arrhenius eta_diff and tau_y;
    - (3.3) Pass 2 clears _rheology_viscosity_dirty so viscosity() returns yielded mu_pass2."""
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e5,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e5,
    )
    # (3.2) const_properties=True with rheology.enabled=True
    ev_const = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=9.81,
        const_properties=True,
        viscosity_solid=1.0e21,
        viscosity_liquid=1.0e-1,
        rheology=rheo,
    )
    ev_const.set_pressure(np.array([0.0, 1.0e9, 5.0e9]))
    ev_const.set_entropy(np.array([1500.0, 2000.0, 2500.0]))
    ev_const.update()
    assert ev_const.solidus_temperature() is None
    eta_d = np.asarray(ev_const.eta_diff(), dtype=float)
    tau_y = np.asarray(ev_const.tau_y(), dtype=float)
    assert np.all(np.isfinite(eta_d))
    assert not np.allclose(eta_d, 1.0e21)
    np.testing.assert_allclose(tau_y, 1.0e5, rtol=1e-14, atol=0.0)

    # (3.1) _solidus_temperature = None on EOS evaluator uses linear fallback without NaN
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)
    solver.state.update(s_stag, 0.0)
    solver.state.phase_basic._solidus_temperature = None
    solver.state.phase_basic._compute_rheology_viscosity()
    assert np.all(np.isfinite(solver.state.phase_basic.eta_diff()))

    # (3.3) Pass 2 yielded viscosity is not overwritten when _defer_rheology_viscosity is True
    solver.state.phase_basic._defer_rheology_viscosity = True
    solver.state.update(s_stag, 0.0)
    mu_before = np.asarray(solver.state.phase_basic._viscosity_val, dtype=float).copy()
    assert solver.state.phase_basic._rheology_viscosity_dirty is False
    mu_after = np.asarray(solver.state.phase_basic.viscosity(), dtype=float)
    np.testing.assert_array_equal(mu_after, mu_before)


def test_jax_phase_params_dict_validation_smoothing_weight_and_core_bl() -> None:
    """Ruling 58d (4.1, 4.2, 4.3, 4.4, 5.1):
    - (4.1) PhaseParams accepts a dict for rheology via coercion to SolidRheologyParams;
    - (4.2) PhaseParams and MeshArrays validate static rheology and mesh invariants;
    - (4.3) spider_get_smoothing and phase_boundary_smoothing with smooth_width=0.0 are finite;
    - (4.4) PhaseProperties.visc_solid_weight and EntropyPhaseEvaluator.visc_solid_weight() match eq. 77;
    - (5.1) dSdt_core_module evaluates Arrhenius viscosity on the CMB boundary layer."""
    from aragog.jax.phase import (
        MeshArrays,
        PhaseParams,
        phase_boundary_smoothing,
        spider_get_smoothing,
    )

    # (4.1) Dict coercion on PhaseParams
    pp = PhaseParams(rheology={'enabled': True, 'g_homologous': 28.0})
    assert isinstance(pp.rheology, SolidRheologyParams)
    assert pp.rheology.enabled is True
    assert pp.rheology.g_homologous == 28.0

    # (4.2) PhaseParams and MeshArrays validation
    with pytest.raises(ValueError, match='viscosity_max_log10'):
        PhaseParams(
            viscosity_solid=1.0e35,
            rheology=SolidRheologyParams(enabled=True, viscosity_max_log10=30.0),
        )
    with pytest.raises(ValueError, match='radii'):
        MeshArrays(
            d_dr_matrix=jnp.zeros((3, 2)),
            quantity_matrix=jnp.zeros((3, 2)),
            area=jnp.ones(3),
            volume=jnp.ones(2),
            radii_basic=jnp.array([3.0e6, 2.0e6, 1.0e6]),
            mixing_length=jnp.ones(3),
            mixing_length_sq=jnp.ones(3),
            mixing_length_cu=jnp.ones(3),
            radii_stag=jnp.array([2.5e6, 1.5e6]),
            P_stag=jnp.ones(2),
            P_basic=jnp.ones(3),
            gravity=jnp.ones(3) * 9.8,
        )

    # (4.3) smooth_width == 0.0 in cubic_hermite mode does not produce 0.0 * NaN
    gphi = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_tanh = spider_get_smoothing(gphi, 0.0)
    assert bool(jnp.all(jnp.isfinite(s_tanh)))
    pp_cubic = PhaseParams(phase_smoothing='cubic_hermite', phase_smoothing_width=0.0)
    smth = phase_boundary_smoothing(gphi, pp_cubic)
    assert bool(jnp.all(jnp.isfinite(smth)))

    # (4.4) visc_solid_weight on both NumPy and JAX paths
    rheo = SolidRheologyParams(enabled=True, phi_visc_single=0.35)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)
    solver.state.update(s_stag, 0.0)
    w_np = np.asarray(solver.state.phase_basic.visc_solid_weight(), dtype=float)
    assert w_np.shape == (solver._n_stag + 1,)
    assert np.all((w_np >= 0.0) & (w_np <= 1.0))

    _, eos_jax, params_jax, args = _build_jax_bundle(solver)
    mesh_jax = args[2]
    s_basic = entropy_at_basic_nodes(mesh_jax, jnp.asarray(s_stag))
    props_jax = evaluate_phase(eos_jax, params_jax, mesh_jax.P_basic, s_basic)
    w_jax = np.asarray(props_jax.visc_solid_weight, dtype=float)
    np.testing.assert_allclose(w_np, w_jax, rtol=1e-14, atol=1e-15)

    # (5.1) CMB boundary-layer evaluation in dSdt_core_module includes rheology
    s_bl = eos_jax.entropy_at_temperature(mesh_jax.P_basic[0], jnp.asarray(1800.0))
    bl_rheo = evaluate_phase(eos_jax, params_jax, mesh_jax.P_basic[:1], jnp.reshape(s_bl, (1,)))
    bl_no_rheo = evaluate_phase(
        eos_jax,
        params_jax,
        mesh_jax.P_basic[:1],
        jnp.reshape(s_bl, (1,)),
        include_rheology=False,
    )
    assert bl_rheo.eta_diff is not None
    assert not np.allclose(np.asarray(bl_rheo.viscosity), np.asarray(bl_no_rheo.viscosity))
    solver._bl_phase.set_entropy(np.array([float(s_bl)]))
    solver._bl_phase.update()
    np.testing.assert_allclose(
        float(np.asarray(solver._bl_phase.viscosity()).flat[0]),
        float(bl_rheo.viscosity[0]),
        rtol=1e-12,
        atol=0.0,
    )


def test_public_effective_viscosity_local_delegates_to_compute_strain_rate_local(
    monkeypatch,
) -> None:
    """Public compute_effective_viscosity in 'local' mode delegates strain-rate
    computation to public compute_strain_rate_local when strain_rate is None."""
    called = {'count': 0}
    orig_fn = rheo_lid_mod.compute_strain_rate_local

    def _spy(*a, **kw):
        called['count'] += 1
        return orig_fn(*a, **kw)

    monkeypatch.setattr(rheo_lid_mod, 'compute_strain_rate_local', _spy)
    eta_diff = np.array([1.0e21, 1.0e21, 1.0e21])
    u_visc = np.array([1.0e-8, 2.0e-8, 3.0e-8])
    l_mix = np.array([1.0e5, 1.0e5, 1.0e5])
    tau_y = np.array([1.0e7, 1.0e7, 1.0e7])
    out = rheo_lid_mod.compute_effective_viscosity(
        eta_diff,
        stress_closure_mode='local',
        unyielded_velocity=u_visc,
        mixing_length=l_mix,
        tau_y=tau_y,
    )
    assert called['count'] == 1
    assert np.all(out <= eta_diff)


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


# ============================================================================
# 6. Reference-Pinned Validation Benchmarks (A2-2, A2-3, A2-4)
# ============================================================================

_BENCHMARK_JSON = Path(__file__).parent / 'reference' / 'ssc_wp2_validation_benchmarks.json'


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a2_2_rhs_numpy_jax_parity_rheology_on(validation_plot) -> None:
    """A2-2: NumPy and JAX right-hand sides agree to relative difference <= 1e-6 across
    100 cooling states on:
    1. the active-solid variant of abe_solid.toml (s_init=2500, kappah_floor=0,
       viscosity_solid=1e18), where Arrhenius strengthening shifts the RHS by >= 1e-4
       at every state and a JAX-only g_homologous perturbation fails 1e-6 by > 8000x;
    2. the yield-active lid state set (s_init=2150, s_bot in [1950, 2150], delta_s in [1620, 1820],
       kappah_floor=0, viscosity_solid=3e17, yield_stress_c=150 Pa, yield_stress_mu=0),
       where plastic yielding acts in 9..13 nodes across all 100/100 states, changes the RHS
       relative to yield-off (tau_y -> infinity) by >= 1e-4 at every state, agrees between NumPy
       and JAX to <= 1.95e-7 <= 1e-6, and a JAX-only yield-stress perturbation fails 1e-6 by > 230x;
    3. the default abe_solid.toml (s_init=2900, kappah_floor=10, viscosity_solid=1e21),
       which remains an inert-path check with RHS_on == RHS_off."""
    data = json.loads(_BENCHMARK_JSON.read_text(encoding='utf-8'))
    a2_2_block = data['a2_2_rhs_parity_and_activity']
    var_rec = a2_2_block['abe_solid_kappah_floor_0']
    yld_rec = a2_2_block['abe_solid_yield_active']
    def_rec = a2_2_block['abe_solid_default']
    diag_59b = a2_2_block['ruling_59b_flux_vs_rhs_diagnostic']

    # Verify recorded 100-state benchmark invariants for all 3 state sets
    rel_np_jax = np.asarray(var_rec['rel_rhs_numpy_vs_jax_per_state'], dtype=float)
    rel_np_jax_harm = np.asarray(
        var_rec['rel_rhs_numpy_vs_jax_harmonised_alpha_per_state'], dtype=float
    )
    rel_on_off = np.asarray(var_rec['rel_rhs_on_vs_off_per_state'], dtype=float)
    rel_canary = np.asarray(var_rec['canary_jax_broken_rel_rhs_per_state'], dtype=float)

    yld_np_jax = np.asarray(yld_rec['rel_rhs_numpy_vs_jax_per_state'], dtype=float)
    yld_np_jax_harm = np.asarray(
        yld_rec['rel_rhs_numpy_vs_jax_harmonised_alpha_per_state'], dtype=float
    )
    yld_on_yoff = np.asarray(yld_rec['rel_rhs_yield_on_vs_yield_off_per_state'], dtype=float)
    yld_canary = np.asarray(yld_rec['canary_jax_yield_broken_rel_rhs_per_state'], dtype=float)
    yld_nodes = np.asarray(yld_rec['n_nodes_eta_eff_ne_eta_diff_per_state'], dtype=int)

    def_np_jax = np.asarray(def_rec['rel_rhs_numpy_vs_jax_per_state'], dtype=float)
    def_np_jax_harm = np.asarray(
        def_rec['rel_rhs_numpy_vs_jax_harmonised_alpha_per_state'], dtype=float
    )
    def_on_off = np.asarray(def_rec['rel_rhs_on_vs_off_per_state'], dtype=float)

    assert rel_np_jax.shape == (100,)
    assert rel_np_jax_harm.shape == (100,)
    assert float(np.max(rel_np_jax)) <= 1.0e-6
    assert float(np.max(rel_np_jax_harm)) <= 1.0e-14
    assert float(np.min(rel_on_off)) >= 100.0 * 1.0e-6
    assert float(np.min(rel_canary)) > 100.0 * 1.0e-6
    assert var_rec['n_nodes_visc_on_ne_visc_off_min'] == 100
    assert var_rec['n_nodes_kappah_on_ne_kappah_off_min'] >= 90

    # Ruling 59 (a) & 59b (c): yield-active state set assertions (both unharmonised and harmonised alpha)
    assert yld_np_jax.shape == (100,)
    assert yld_np_jax_harm.shape == (100,)
    assert int(np.sum(yld_nodes > 0)) == 100
    assert yld_rec['n_states_eta_eff_ne_eta_diff_pos'] == 100
    assert yld_rec['n_nodes_eta_eff_ne_eta_diff_min'] == 9
    assert yld_rec['n_nodes_eta_eff_ne_eta_diff_max'] == 13
    assert float(np.max(yld_np_jax)) <= 1.0e-6
    assert float(np.max(yld_np_jax_harm)) <= 1.0e-14
    assert float(np.min(yld_on_yoff)) >= 1.0e-4
    assert float(np.min(yld_canary)) >= 1.0e-4

    assert def_np_jax.shape == (100,)
    assert def_np_jax_harm.shape == (100,)
    assert float(np.max(def_np_jax)) <= 1.0e-14
    assert float(np.max(def_np_jax_harm)) <= 1.0e-14

    # Ruling 59 (b): flux vs RHS diagnostic assertions on abe_solid_kappah_floor_0
    assert diag_59b['worst_state_index'] == 4
    assert diag_59b['worst_basic_node_index'] == 64
    assert diag_59b['worst_staggered_cell_index'] == 63
    assert diag_59b['max_rel_flux_numpy_vs_jax'] == pytest.approx(
        var_rec['max_rel_flux_numpy_vs_jax'], rel=1.0e-12
    )
    assert diag_59b['max_rel_rhs_numpy_vs_jax'] == pytest.approx(
        var_rec['max_rel_rhs_numpy_vs_jax'], rel=1.0e-12
    )

    assert float(np.max(def_on_off)) == 0.0
    assert def_rec['n_nodes_visc_on_ne_visc_off_min'] == 100
    assert def_rec['n_nodes_kappah_on_ne_kappah_off_max'] == 0

    # Live solver verification on the active-solid, yield-active, and default configs + canaries
    rheo_on = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    rheo_off = SolidRheologyParams(enabled=False)

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        base_params = Parameters.from_file(p)

    def _build_pair(
        kappah_floor: float,
        viscosity_solid: float,
        rheo_a: SolidRheologyParams = rheo_on,
        rheo_b: SolidRheologyParams = rheo_off,
        *,
        identity_alpha: bool = True,
    ):
        p_a = replace(
            base_params,
            boundary_conditions=replace(
                base_params.boundary_conditions,
                core_bc='quasi_steady',
                inner_boundary_condition=1,
            ),
            energy=replace(base_params.energy, kappah_floor=kappah_floor),
            phase_solid=replace(
                base_params.phase_solid, viscosity=viscosity_solid, rheology=rheo_a
            ),
        )
        p_b = replace(
            p_a,
            phase_solid=replace(
                base_params.phase_solid, viscosity=viscosity_solid, rheology=rheo_b
            ),
        )
        return _make_solver(p_a, identity_alpha=identity_alpha), _make_solver(
            p_b, identity_alpha=identity_alpha
        )

    sol_var_on, sol_var_off = _build_pair(0.0, 1.0e18, identity_alpha=True)
    sol_var_on_unharm, _ = _build_pair(0.0, 1.0e18, identity_alpha=False)
    sol_def_on, sol_def_off = _build_pair(10.0, 1.0e21)

    _, _, _, args_var_on = _build_jax_bundle(sol_var_on)
    rheo_broken = replace(rheo_on, g_homologous=0.1)
    _, _, _, args_var_broken = _build_jax_bundle(sol_var_on, rheology_override=rheo_broken)

    xi_var = np.linspace(0.0, 1.0, sol_var_on._n_stag)
    s_var = 2300.0 - 600.0 * (0.2 * xi_var + 0.8 * xi_var**4)
    rhs_var_np_on = sol_var_on.dSdt(0.0, s_var)
    rhs_var_np_on_unharm = sol_var_on_unharm.dSdt(0.0, s_var)
    rhs_var_np_off = sol_var_off.dSdt(0.0, s_var)
    rhs_var_jax_on = np.asarray(dSdt(0.0, jnp.asarray(s_var), args_var_on))
    rhs_var_jax_broken = np.asarray(dSdt(0.0, jnp.asarray(s_var), args_var_broken))

    denom_var_on = float(np.max(np.abs(rhs_var_np_on)))
    denom_var_on_unharm = float(np.max(np.abs(rhs_var_np_on_unharm)))
    denom_var_off = float(np.max(np.abs(rhs_var_np_off)))
    assert float(np.max(np.abs(rhs_var_np_on - rhs_var_jax_on)) / denom_var_on) <= 1.0e-13
    assert (
        float(np.max(np.abs(rhs_var_np_on_unharm - rhs_var_jax_on)) / denom_var_on_unharm)
        <= 1.0e-6
    )
    assert float(np.max(np.abs(rhs_var_np_on - rhs_var_np_off)) / denom_var_off) >= 1.0e-4
    assert float(np.max(np.abs(rhs_var_np_on - rhs_var_jax_broken)) / denom_var_on) > 1.0e-3

    # Live solver verification of the yield-active state set + JAX-only yield canary (Ruling 59a & 59b)
    rheo_yld_on = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
    )
    rheo_yld_off = replace(rheo_yld_on, yield_stress_c=1.0e30, yield_stress_max=1.0e30)
    sol_yld_on, sol_yld_off = _build_pair(
        0.0, 3.0e17, rheo_yld_on, rheo_yld_off, identity_alpha=True
    )
    sol_yld_on_unharm, _ = _build_pair(
        0.0, 3.0e17, rheo_yld_on, rheo_yld_off, identity_alpha=False
    )
    _, _, _, args_yld_on = _build_jax_bundle(sol_yld_on)
    _, _, _, args_yld_broken = _build_jax_bundle(sol_yld_on, rheology_override=rheo_yld_off)

    s_yld = 2050.0 - 1720.0 * (0.2 * xi_var + 0.8 * xi_var**4)
    rhs_yld_np_on = sol_yld_on.dSdt(0.0, s_yld)
    rhs_yld_np_on_unharm = sol_yld_on_unharm.dSdt(0.0, s_yld)
    eta_eff_yld = sol_yld_on.state._eta_eff_basic.copy()
    eta_diff_yld = sol_yld_on.state.phase_basic.eta_diff().copy()
    rhs_yld_np_yoff = sol_yld_off.dSdt(0.0, s_yld)
    rhs_yld_jax_on = np.asarray(dSdt(0.0, jnp.asarray(s_yld), args_yld_on))
    rhs_yld_jax_broken = np.asarray(dSdt(0.0, jnp.asarray(s_yld), args_yld_broken))

    denom_yld_on = float(np.max(np.abs(rhs_yld_np_on)))
    denom_yld_on_unharm = float(np.max(np.abs(rhs_yld_np_on_unharm)))
    denom_yld_yoff = float(np.max(np.abs(rhs_yld_np_yoff)))
    assert int(np.sum(eta_eff_yld != eta_diff_yld)) >= 8
    assert float(np.max(np.abs(rhs_yld_np_on - rhs_yld_jax_on)) / denom_yld_on) <= 1.0e-13
    assert (
        float(np.max(np.abs(rhs_yld_np_on_unharm - rhs_yld_jax_on)) / denom_yld_on_unharm)
        <= 1.0e-6
    )
    assert float(np.max(np.abs(rhs_yld_np_on - rhs_yld_np_yoff)) / denom_yld_yoff) >= 1.0e-4
    assert float(np.max(np.abs(rhs_yld_np_on - rhs_yld_jax_broken)) / denom_yld_on) >= 1.0e-4

    s_def = 2700.0 - 600.0 * (0.2 * xi_var + 0.8 * xi_var**4)
    rhs_def_np_on = sol_def_on.dSdt(0.0, s_def)
    rhs_def_np_off = sol_def_off.dSdt(0.0, s_def)
    _, _, _, args_def_on = _build_jax_bundle(sol_def_on)
    _, _, _, args_def_off = _build_jax_bundle(sol_def_off)
    rhs_def_jax_on = np.asarray(dSdt(0.0, jnp.asarray(s_def), args_def_on))
    rhs_def_jax_off = np.asarray(dSdt(0.0, jnp.asarray(s_def), args_def_off))
    denom_def_on = float(np.max(np.abs(rhs_def_np_on)))
    denom_def_off = float(np.max(np.abs(rhs_def_np_off)))
    assert float(np.max(np.abs(rhs_def_np_on - rhs_def_jax_on)) / denom_def_on) <= 1.0e-13
    assert float(np.max(np.abs(rhs_def_np_off - rhs_def_jax_off)) / denom_def_off) <= 1.0e-13
    assert float(np.max(np.abs(rhs_def_np_on - rhs_def_np_off))) > 1.0e-4

    def figure():
        import matplotlib.pyplot as plt

        states = np.arange(100, dtype=float)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(
            states,
            np.log10(np.maximum(rel_np_jax_harm, 1.0e-17)),
            label='NP vs JAX (Arrh, harm alpha)',
        )
        ax.plot(
            states,
            np.log10(np.maximum(yld_np_jax_harm, 1.0e-17)),
            label='NP vs JAX (yield, harm alpha)',
        )
        ax.plot(
            states,
            np.log10(np.maximum(yld_np_jax, 1.0e-17)),
            ':',
            label='NP vs JAX (yield, table alpha)',
        )
        ax.plot(states, np.log10(rel_on_off), '--', label='on vs off (Arrh)')
        ax.plot(states, np.log10(yld_on_yoff), '-.', label='yield on vs off')
        ax.axhline(-6.0, color=plt.rcParams['text.color'], ls=':')
        ax.axhline(-4.0, color=plt.rcParams['text.color'], ls=':')
        ax.set(xlabel='State index [-]', ylabel='log10 rel diff [-]')
        ax.legend(ncol=2, fontsize=7.5)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'ssc_a2_2_rhs_parity')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a2_3_perturbed_starts_spread(validation_plot) -> None:
    """A2-3: 5 runs of abe_solid.toml from perturbed initial entropy profiles
    S_0(r)(1 + k * 1e-12) for k in {-2, -1, 0, +1, +2} over 100 yr satisfy
    Delta T_on(r) <= max(2 * Delta T_off(r), 0.05 K) across the full radial profile
    on both the active-solid variant (s_init=2500, kappah_floor=0, viscosity_solid=1e18)
    and the default inert configuration (s_init=2900, kappah_floor=10, viscosity_solid=1e21)."""
    data = json.loads(_BENCHMARK_JSON.read_text(encoding='utf-8'))
    epsilons = data['a2_3_perturbed_starts']['epsilons']
    assert epsilons == [-2.0e-12, -1.0e-12, 0.0, 1.0e-12, 2.0e-12]

    var_suite = data['a2_3_perturbed_starts']['abe_solid_kappah_floor_0']
    def_suite = data['a2_3_perturbed_starts']['abe_solid_default']

    r_km = np.asarray(var_suite['on']['r_basic_m'], dtype=float) / 1.0e3
    dt_var_on = np.asarray(var_suite['on']['delta_T_basic_profile_K'], dtype=float)
    dt_var_off = np.asarray(var_suite['off']['delta_T_basic_profile_K'], dtype=float)
    dt_def_on = np.asarray(def_suite['on']['delta_T_basic_profile_K'], dtype=float)
    dt_def_off = np.asarray(def_suite['off']['delta_T_basic_profile_K'], dtype=float)

    assert np.all(dt_var_on <= np.maximum(2.0 * dt_var_off, 0.05))
    assert np.all(dt_def_on <= np.maximum(2.0 * dt_def_off, 0.05))
    assert float(np.max(dt_var_on)) <= 0.05
    assert float(np.max(dt_var_off)) <= 0.05
    assert np.allclose(dt_def_on, dt_def_off, rtol=0.0, atol=1.0e-15)

    # Ruling 58a Condition (1): variant on-vs-off difference >= 100 * 1e-6, default == 0
    var_act = var_suite['on_vs_off_activity']
    def_act = def_suite['on_vs_off_activity']
    assert var_act['max_rel_diff_T_basic_on_vs_off'] >= 1.0e-4
    assert var_act['max_rel_rhs_on_vs_off_at_t_end'] >= 1.0e-4
    assert def_act['identical_final_T_basic'] is True
    assert def_act['max_rel_diff_T_basic_on_vs_off'] == 0.0

    t_var_on_k0 = np.asarray(var_suite['on']['runs'][2]['T_basic_K'], dtype=float)
    t_var_off_k0 = np.asarray(var_suite['off']['runs'][2]['T_basic_K'], dtype=float)
    assert float(np.max(np.abs(t_var_on_k0 - t_var_off_k0)) / np.max(t_var_off_k0)) >= 1.0e-4

    def figure():
        import matplotlib.pyplot as plt

        fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(5.6, 4.8))
        top.plot(r_km, t_var_on_k0, label='on')
        top.plot(r_km, t_var_off_k0, '--', label='off')
        top.set(ylabel='T [K]')
        top.legend()

        bottom.plot(r_km, np.log10(np.maximum(dt_var_on, 1.0e-12)), label='on')
        bottom.plot(r_km, np.log10(np.maximum(dt_var_off, 1.0e-12)), '--', label='off')
        bottom.plot(r_km, np.log10(np.maximum(dt_def_on, 1.0e-12)), ':', label='default')
        bottom.axhline(np.log10(0.05), color=plt.rcParams['text.color'], ls='-.')
        bottom.set(xlabel='r [km]', ylabel='log10 dT [K]')
        bottom.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'ssc_a2_3_perturbed_spread')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a2_4_rhs_call_counts_and_cost(validation_plot) -> None:
    """A2-4: Enabling two-pass solid rheology adds 0 extra EOS solidus lookups,
    +2 Arrhenius viscosity calls, +2 yield stress calls, +1 stagnant-lid state call,
    +1 effective viscosity call, and +2 viscous MLT factor calls per RHS on both
    NumPy and JAX, with JIT-compiled JAX per-RHS wall-time ratio <= 1.25."""
    data = json.loads(_BENCHMARK_JSON.read_text(encoding='utf-8'))
    a2_4 = data['a2_4_rhs_wall_time']
    ratios = a2_4['ratios']
    table = a2_4['six_row_call_count_table']

    assert ratios['jax_ratio_on_lid_over_off'] <= 1.25
    assert ratios['jax_ratio_on_local_over_off'] <= 1.25

    # Verify live per-RHS call counts on both NumPy and JAX against the 6-row table
    counts_np_off = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=False))
    counts_np_on = _measure_numpy_rhs_counts(SolidRheologyParams(enabled=True))
    counts_jax_off = _measure_jax_rhs_counts(SolidRheologyParams(enabled=False))
    counts_jax_on = _measure_jax_rhs_counts(SolidRheologyParams(enabled=True))

    measured_deltas_np = [
        counts_np_on['solid_temp_lookup'] - counts_np_off['solid_temp_lookup'],
        counts_np_on['arrhenius_viscosity_core'] - counts_np_off['arrhenius_viscosity_core'],
        counts_np_on['yield_stress_core'] - counts_np_off['yield_stress_core'],
        counts_np_on['stagnant_lid_core'] - counts_np_off['stagnant_lid_core'],
        counts_np_on['effective_viscosity_core'] - counts_np_off['effective_viscosity_core'],
        counts_np_on['viscous_mlt_core'] - counts_np_off['viscous_mlt_core'],
    ]
    measured_deltas_jax = [
        0,
        counts_jax_on['arrhenius_viscosity_core'] - counts_jax_off['arrhenius_viscosity_core'],
        counts_jax_on['yield_stress_core'] - counts_jax_off['yield_stress_core'],
        counts_jax_on['stagnant_lid_core'] - counts_jax_off['stagnant_lid_core'],
        counts_jax_on['effective_viscosity_core'] - counts_jax_off['effective_viscosity_core'],
        counts_jax_on['viscous_mlt_core'] - counts_jax_off['viscous_mlt_core'],
    ]
    assert measured_deltas_np == [0, 2, 2, 1, 1, 2]
    assert measured_deltas_jax == [0, 2, 2, 1, 1, 2]
    assert [row['measured_numpy_delta'] for row in table] == [0, 2, 2, 1, 1, 2]
    assert [row['measured_jax_delta'] for row in table] == [0, 2, 2, 1, 1, 2]

    counts_np_local = _measure_numpy_rhs_counts(
        SolidRheologyParams(enabled=True, stress_closure_mode='local')
    )
    counts_jax_local = _measure_jax_rhs_counts(
        SolidRheologyParams(enabled=True, stress_closure_mode='local')
    )
    measured_local_np = [
        counts_np_local['solid_temp_lookup'] - counts_np_off['solid_temp_lookup'],
        counts_np_local['arrhenius_viscosity_core'] - counts_np_off['arrhenius_viscosity_core'],
        counts_np_local['yield_stress_core'] - counts_np_off['yield_stress_core'],
        counts_np_local['stagnant_lid_core'] - counts_np_off['stagnant_lid_core'],
        counts_np_local['effective_viscosity_core'] - counts_np_off['effective_viscosity_core'],
        counts_np_local['viscous_mlt_core'] - counts_np_off['viscous_mlt_core'],
    ]
    measured_local_jax = [
        0,
        counts_jax_local['arrhenius_viscosity_core']
        - counts_jax_off['arrhenius_viscosity_core'],
        counts_jax_local['yield_stress_core'] - counts_jax_off['yield_stress_core'],
        counts_jax_local['stagnant_lid_core'] - counts_jax_off['stagnant_lid_core'],
        counts_jax_local['effective_viscosity_core']
        - counts_jax_off['effective_viscosity_core'],
        counts_jax_local['viscous_mlt_core'] - counts_jax_off['viscous_mlt_core'],
    ]
    assert measured_local_np == [0, 1, 1, 0, 1, 2]
    assert measured_local_jax == [0, 1, 1, 0, 1, 2]

    def figure():
        import matplotlib.pyplot as plt

        x_idx = np.arange(6, dtype=float)
        plan_deltas = np.array([row['plan_a2_4_delta'] for row in table], dtype=float)
        meas_np = np.array(measured_deltas_np, dtype=float)
        meas_jax = np.array(measured_deltas_jax, dtype=float)

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 4.8))
        top.plot(x_idx - 0.12, plan_deltas, 'o', ms=4, label='plan')
        top.plot(x_idx, meas_np, 'o', ms=4, label='NumPy')
        top.plot(x_idx + 0.12, meas_jax, 'o', ms=4, label='JAX')
        top.set(xlabel='Item index [-]', ylabel='Calls [-]')
        top.legend()

        ratio_x = np.array([0.0, 1.0, 2.0, 3.0])
        ratio_y = np.array(
            [
                ratios['jax_ratio_on_lid_over_off'],
                ratios['jax_ratio_on_local_over_off'],
                ratios['numpy_ratio_on_lid_over_off'],
                ratios['numpy_ratio_on_local_over_off'],
            ],
            dtype=float,
        )
        bottom.plot(ratio_x[:2], ratio_y[:2], '-o', ms=4, label='JAX')
        bottom.plot(ratio_x[2:], ratio_y[2:], '--o', ms=4, label='NumPy')
        bottom.axhline(1.25, color=plt.rcParams['text.color'], ls=':')
        bottom.set(xlabel='Mode index [-]', ylabel='Ratio [-]')
        bottom.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'ssc_a2_4_rhs_cost')


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_wp4_solver_tau_d_lid_and_zero_anchored_eddy_floor_and_invariants():
    """Verify solver passes tau_d_lid in lid mode, zero-anchors eddy floor at widths 0.15 and 0.20, and checks lid invariants."""
    from aragog.utilities import tanh_weight

    # 1. Zero-anchored eddy-diffusivity floor at widths 0.15 (Aragog default) and 0.20 (PROTEUS rfront_wid default).
    for width in (0.15, 0.20):
        rheo_on = SolidRheologyParams(
            enabled=True,
            stress_closure_mode='lid',
            yield_stress_c=1.0e9,
            yield_stress_mu=0.6,
            yield_stress_max=1.0e10,
        )
        params_on = _make_abe_solid_params(rheo_on, number_of_nodes=20)
        params_on.phase_mixed.rheological_transition_width = width
        params_on.energy.kappah_floor = 100.0
        solver_on = _make_solver(params_on)
        s_solid = _make_solid_profile(solver_on)
        solver_on.state.update(s_solid, 0.0)
        phi_basic = np.asarray(solver_on.state.phase_basic.melt_fraction(), dtype=float)
        assert np.all(phi_basic == 0.0)

        # Compare against kappah_floor = 0.0: at phi == 0, zero-anchored floor adds 0.0.
        params_no_floor = _make_abe_solid_params(rheo_on, number_of_nodes=20)
        params_no_floor.phase_mixed.rheological_transition_width = width
        params_no_floor.energy.kappah_floor = 0.0
        solver_no_floor = _make_solver(params_no_floor)
        solver_no_floor.state.update(s_solid, 0.0)
        np.testing.assert_allclose(
            solver_on.state._eddy_diffusivity,
            solver_no_floor.state._eddy_diffusivity,
            rtol=1.0e-14,
            atol=0.0,
        )

        # JAX compute_fluxes parity at phi == 0 with floor = 100 vs floor = 0.
        mesh_on, eos_on, p_jax_on, _ = _build_jax_bundle(solver_on)
        mesh_nf, eos_nf, p_jax_nf, _ = _build_jax_bundle(solver_no_floor)
        h_zero = jnp.zeros(solver_on._n_stag, dtype=jnp.float64)
        fl_on = compute_fluxes(jnp.asarray(s_solid), 0.0, eos_on, p_jax_on, mesh_on, h_zero)
        fl_nf = compute_fluxes(jnp.asarray(s_solid), 0.0, eos_nf, p_jax_nf, mesh_nf, h_zero)
        np.testing.assert_allclose(
            np.asarray(fl_on.eddy_diffusivity),
            np.asarray(fl_nf.eddy_diffusivity),
            rtol=1.0e-14,
            atol=0.0,
        )

        # Disabled mode preserves the un-anchored tanh_weight(0.0, 0.4, width) > 0 floor.
        f_unanchored_0 = float(tanh_weight(0.0, 0.4, width))
        assert f_unanchored_0 > 0.004

    # 2. Solver passes tau_d_lid (not tau_d) to _compute_effective_viscosity_core in lid mode.
    rheo_base = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e9,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e9,
    )
    solver_base = _make_solver(_make_abe_solid_params(rheo_base, number_of_nodes=20))
    s_prof = _make_solid_profile(solver_base)
    solver_base.state.update(s_prof, 0.0)
    lid_base = solver_base.state._lid_state
    assert lid_base is not None
    tau_d_val = float(lid_base['tau_d'])
    tau_d_lid_val = float(lid_base['tau_d_lid'])
    assert tau_d_lid_val > tau_d_val * 1.3

    # Choose tau_y strictly between tau_d and tau_d_lid: yielding must activate under tau_d_lid!
    tau_y_mid = 0.5 * (tau_d_val + tau_d_lid_val)
    rheo_mid = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=tau_y_mid,
        yield_stress_mu=0.0,
        yield_stress_max=tau_y_mid,
    )
    solver_mid = _make_solver(_make_abe_solid_params(rheo_mid, number_of_nodes=20))
    solver_mid.state.update(s_prof, 0.0)
    eta_diff = np.asarray(solver_mid.state.phase_basic.eta_diff(), dtype=float)
    eta_eff = np.asarray(solver_mid.state._eta_eff_basic, dtype=float)
    assert np.any(eta_eff < eta_diff * 0.99)

    # 3. _check_rheology_invariants enforces 0 <= d_lid <= mantle_thickness.
    solver_mid._check_rheology_invariants()
    solver_mid.state._lid_state['d_lid'] = -1.0
    with pytest.raises(RuntimeError, match='d_lid'):
        solver_mid._check_rheology_invariants()
    solver_mid.state._lid_state['d_lid'] = 1.0e9
    with pytest.raises(RuntimeError, match='d_lid'):
        solver_mid._check_rheology_invariants()
