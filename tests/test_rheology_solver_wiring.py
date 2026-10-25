"""Solver-wiring tests for the solid-state rheology and stagnant-lid closure.

Covers:
- Viscous mixing-length scaling exponents in both NumPy and JAX paths
  (``u_visc ~ q^(3/4)``, ``Re ~ q^1``, ``kappa_h ~ q^1``).
- Two-pass evaluation order in ``EntropyState.update`` (NumPy) and
  ``compute_fluxes`` / ``dSdt`` (JAX), plus non-local Jacobian coupling
  through the surface stagnant-lid diagnostic.
- Per-RHS call-count contracts (_validate_inputs called 0 times, zero extra
  EOS solidus lookups, and exact ``_core`` kernel invocation counts).
- ``_check_rheology_invariants`` positive and negative checks on ``EntropySolver``.
- NumPy vs JAX right-hand-side parity across 100 cooling states of an
  ``abe_solid.toml`` trajectory with ``rheology.enabled = True``.
"""

from __future__ import annotations

import copy
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
from aragog.config import Config
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.jax.phase import (
    MeshArrays,
    PhaseParams,
    blend_viscosity_with_solid_jax,
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

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_BENCHMARK_JSON = Path(__file__).parent / 'reference' / 'ssc_wp2_validation_benchmarks.json'


@functools.cache
def _resolve_eos_dir() -> Path:
    """Resolve SPIDER P-S table directory or generate a synthetic fallback."""
    if EOS_DIR.is_dir():
        return EOS_DIR

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
    overrides = {'rheology': rheology_override} if rheology_override is not None else {}
    params_jax = PhaseParams.from_parameters(params, **overrides)
    mesh_jax = MeshArrays.from_numpy_mesh(
        solver.evaluator.mesh,
        validate_rheology=bool(params_jax.rheology.enabled),
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
# 1. Viscous MLT Exponents in Both Paths (u_visc ~ q^(3/4), Re ~ q, kappa_h ~ q)
# ============================================================================


def test_viscous_mlt_scaling_exponents_numpy() -> None:
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


def test_rhs_arrhenius_viscosity_core_call_counts() -> None:
    """_compute_arrhenius_viscosity_core is called 2 times per RHS."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['arrhenius_viscosity_core'] == 2
    assert counts_jax['arrhenius_viscosity_core'] == 2


def test_rhs_yield_stress_core_call_counts() -> None:
    """_compute_yield_stress_core is called 2 times per RHS."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['yield_stress_core'] == 2
    assert counts_jax['yield_stress_core'] == 2


def test_rhs_stagnant_lid_core_call_counts() -> None:
    """_compute_stagnant_lid_state_core is called 1 time per RHS in 'lid' mode
    and 0 times per RHS in 'local' mode."""
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
    """_compute_effective_viscosity_core is called 1 time per RHS."""
    rheo = SolidRheologyParams(enabled=True)
    counts_np = _measure_numpy_rhs_counts(rheo)
    counts_jax = _measure_jax_rhs_counts(rheo)
    assert counts_np['effective_viscosity_core'] == 1
    assert counts_jax['effective_viscosity_core'] == 1


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

    # 5. Non-positive inner basic-node radius r_b[0] <= 0
    orig_r = solver._r_basic_flat.copy()
    solver._r_basic_flat = orig_r.copy()
    solver._r_basic_flat[0] = 0.0
    with pytest.raises(ValueError, match='radii'):
        solver._validate_rheology_setup()
    solver._r_basic_flat = orig_r

    # 6. Staggered-node pressure < -1.0 Pa
    orig_ps = solver.state.phase_staggered.pressure.copy()
    solver.state.phase_staggered.set_pressure(np.full_like(orig_ps, -5.0))
    with pytest.raises(ValueError, match='pressure'):
        solver._validate_rheology_setup()
    solver.state.phase_staggered.set_pressure(orig_ps)

    # 7. Non-positive basic or staggered gravity
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

    # 8. log10(viscosity_solid) >= viscosity_max_log10
    orig_vs = solver.state.phase_basic._visc_solid
    solver.state.phase_basic._visc_solid = 10.0 ** float(rheo.viscosity_max_log10)
    with pytest.raises(ValueError, match='viscosity_max_log10'):
        solver._validate_rheology_setup()
    solver.state.phase_basic._visc_solid = orig_vs


def test_check_rheology_invariants_checks_all_trace_columns_and_lid_none(monkeypatch) -> None:
    """_check_rheology_invariants(trace_y, trace_t) checks every
    accepted step column in trace_y (not only the final column) and raises RuntimeError
    in 'lid' mode if lid_state is None."""
    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_good = _make_solid_profile(solver)

    # 1. lid_state is None in 'lid' mode must raise RuntimeError
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
    """EntropyState.update resets self._lid_state = None when
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


def test_entropy_phase_const_mode_none_solidus_and_dirty_flag() -> None:
    """EntropyPhaseEvaluator verifies solidus fallback, constant properties with rheology,
    and yielded viscosity retention:
    - _solidus_temperature = None uses linear solidus fallback (finite eta_diff, no NaN);
    - const_properties=True with rheology.enabled=True evaluates Arrhenius eta_diff and tau_y;
    - Pass 2 clears _rheology_viscosity_dirty so viscosity() returns yielded mu_pass2."""
    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e5,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e5,
    )
    # const_properties=True with rheology.enabled=True
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

    # _solidus_temperature = None on EOS evaluator uses linear fallback without NaN
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)
    solver.state.update(s_stag, 0.0)
    solver.state.phase_basic._solidus_temperature = None
    solver.state.phase_basic._compute_rheology_viscosity()
    assert np.all(np.isfinite(solver.state.phase_basic.eta_diff()))

    # Pass 2 yielded viscosity is not overwritten when _defer_rheology_viscosity is True
    solver.state.phase_basic._defer_rheology_viscosity = True
    solver.state.update(s_stag, 0.0)
    mu_before = np.asarray(solver.state.phase_basic._viscosity_val, dtype=float).copy()
    assert solver.state.phase_basic._rheology_viscosity_dirty is False
    mu_after = np.asarray(solver.state.phase_basic.viscosity(), dtype=float)
    np.testing.assert_array_equal(mu_after, mu_before)
    eta_diff_val = np.asarray(solver.state.phase_basic.eta_diff(), dtype=float)
    assert np.any(mu_after < eta_diff_val)
    assert not np.allclose(mu_after, eta_diff_val)


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
    expected_out = np.array([1.0e20, 5.0e19, 1.0e20 / 3.0])
    np.testing.assert_allclose(out, expected_out, rtol=1.0e-14, atol=0.0)


# ============================================================================
# 5. NumPy vs JAX RHS Parity Across 100 Cooling States
# ============================================================================


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a2_3_perturbed_starts_spread(validation_plot) -> None:
    """Five runs of abe_solid.toml from perturbed initial entropy profiles
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
    assert float(np.max(dt_def_on)) == pytest.approx(5.557178610615665e-07, rel=1.0e-9)
    assert float(np.max(dt_def_off)) == pytest.approx(1.034986325976206e-06, rel=1.0e-9)
    assert float(np.max(dt_def_on)) <= 0.05
    assert float(np.max(dt_def_off)) <= 0.05

    var_act = var_suite['on_vs_off_activity']
    def_act = def_suite['on_vs_off_activity']
    assert var_act['max_rel_diff_T_basic_on_vs_off'] >= 1.0e-4
    assert var_act['max_rel_rhs_on_vs_off_at_t_end'] >= 1.0e-4
    assert def_act['max_rel_diff_T_basic_on_vs_off'] == pytest.approx(
        5.27953830828029e-05, rel=1.0e-9
    )
    assert def_act['max_abs_diff_T_basic_on_vs_off_K'] == pytest.approx(
        0.1938837410866654, rel=1.0e-9
    )
    assert def_act['max_abs_diff_T_basic_on_vs_off_K'] > float(np.max(dt_def_on))
    assert def_act['max_abs_diff_T_basic_on_vs_off_K'] > float(np.max(dt_def_off))

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
    """Enabling two-pass solid rheology adds 0 extra EOS solidus lookups,
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


def test_visc_single_threshold_governs_viscosity_blend_numpy_and_jax() -> None:
    """visc_single threshold governs the solid-liquid transition in both
    NumPy and JAX blending paths."""
    rheo = SolidRheologyParams(enabled=True, phi_visc_single=0.35)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)
    solver.state.update(s_stag, 0.0)

    # In NumPy evaluator: blend_viscosity_with_solid at phi = 0.42 > 0.35
    solver.state.phase_basic._melt_fraction = np.array([0.42])
    solver.state.phase_basic._smth_val = np.array([0.0])
    eta_blend_np = float(
        np.asarray(solver.state.phase_basic.blend_viscosity_with_solid(1.0e21)).flat[0]
    )
    assert eta_blend_np == pytest.approx(100.0)

    # In JAX: blend_viscosity_with_solid_jax at phi = 0.42 > 0.35
    pp = PhaseParams(
        rheology=rheo,
        viscosity_solid=1.0e21,
        viscosity_liquid=100.0,
    )
    eta_blend_jax = float(
        blend_viscosity_with_solid_jax(
            jnp.array([0.42]),
            jnp.array([0.0]),
            jnp.array([21.0]),
            pp,
        )[0]
    )
    assert eta_blend_jax == pytest.approx(100.0)


def test_phase_params_phi_rheo_and_phi_width_validation() -> None:
    """PhaseParams validates phi_rheo in [0, 1] and phi_width > 0."""
    rheo = SolidRheologyParams(enabled=True)
    with pytest.raises(ValueError, match='phi_rheo'):
        PhaseParams(phi_rheo=-0.1, rheology=rheo)
    with pytest.raises(ValueError, match='phi_rheo'):
        PhaseParams(phi_rheo=1.1, rheology=rheo)
    with pytest.raises(ValueError, match='phi_width'):
        PhaseParams(phi_width=-0.05, rheology=rheo)


def test_mesh_arrays_invariant_validation() -> None:
    """MeshArrays accepts rheology-off inputs on main by default and validates
    node count, non-negative mixing length, pressure, and positive gravity when
    validate_rheology=True or validate_for_rheology() is called."""

    def _valid_kwargs():
        return dict(
            d_dr_matrix=jnp.zeros((3, 2)),
            quantity_matrix=jnp.zeros((3, 2)),
            area=jnp.ones(3),
            volume=jnp.ones(2),
            radii_basic=jnp.array([1.0e6, 2.0e6, 3.0e6]),
            mixing_length=jnp.ones(3),
            mixing_length_sq=jnp.ones(3),
            mixing_length_cu=jnp.ones(3),
            radii_stag=jnp.array([1.5e6, 2.5e6]),
            P_stag=jnp.ones(2),
            P_basic=jnp.ones(3),
            gravity=jnp.ones(3) * 9.8,
        )

    # Node count < 3: accepted when validate_rheology=False, rejected when True
    kw_size = _valid_kwargs()
    kw_size['radii_basic'] = jnp.array([1.0e6, 2.0e6])
    kw_size['P_basic'] = jnp.ones(2)
    m_size = MeshArrays(**kw_size)
    with pytest.raises(ValueError, match='radii'):
        m_size.validate_for_rheology()
    with pytest.raises(ValueError, match='radii'):
        MeshArrays(**kw_size, validate_rheology=True)

    # r_0 = 0: accepted when validate_rheology=False, rejected when True
    kw_r0 = _valid_kwargs()
    kw_r0['radii_basic'] = jnp.array([0.0, 2.0e6, 3.0e6])
    m_r0 = MeshArrays(**kw_r0)
    with pytest.raises(ValueError, match='radii'):
        m_r0.validate_for_rheology()
    with pytest.raises(ValueError, match='radii'):
        MeshArrays(**kw_r0, validate_rheology=True)

    # Negative mixing length: accepted when validate_rheology=False, rejected when True
    kw_ml = _valid_kwargs()
    kw_ml['mixing_length'] = jnp.array([-1.0, 1.0, 1.0])
    m_ml = MeshArrays(**kw_ml)
    with pytest.raises(ValueError, match='mixing_length'):
        m_ml.validate_for_rheology()
    with pytest.raises(ValueError, match='mixing_length'):
        MeshArrays(**kw_ml, validate_rheology=True)

    # Negative pressure (< -1.0 Pa): accepted when validate_rheology=False, rejected when True
    kw_p = _valid_kwargs()
    kw_p['P_basic'] = jnp.array([-5.0, 1.0e5, 1.0e5])
    m_p = MeshArrays(**kw_p)
    with pytest.raises(ValueError, match='pressure'):
        m_p.validate_for_rheology()
    with pytest.raises(ValueError, match='pressure'):
        MeshArrays(**kw_p, validate_rheology=True)

    # Non-positive / negative gravity: accepted when validate_rheology=False, rejected when True
    kw_g = _valid_kwargs()
    kw_g['gravity'] = jnp.array([-9.8, -9.8, -9.8])
    m_g = MeshArrays(**kw_g)
    with pytest.raises(ValueError, match='gravity'):
        m_g.validate_for_rheology()
    with pytest.raises(ValueError, match='gravity'):
        MeshArrays(**kw_g, validate_rheology=True)

    # Traced P_basic under jax.jit with validate_rheology=True does not raise TracerArrayConversionError
    @jax.jit
    def _build_traced(p_b):
        kw_t = _valid_kwargs()
        kw_t['P_basic'] = p_b
        kw_t['dP_dr_basic'] = jnp.zeros(3)
        m = MeshArrays(**kw_t, validate_rheology=True)
        return jnp.sum(m.P_basic)

    assert float(_build_traced(jnp.array([1.0e5, 2.0e5, 3.0e5]))) == pytest.approx(6.0e5)


def test_cvode_dense_linear_solver_selected_for_nonlocal_lid_mode() -> None:
    """CVODE dense linear solver is selected when non-local lid closure is active."""
    captured_options: dict = {}

    def _spy_cvode(rhs, **kwargs):
        captured_options.update(kwargs)
        raise RuntimeError('stop early')

    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)

    with patch('aragog.solver.entropy_solver._scikits_cvode', side_effect=_spy_cvode):
        with pytest.raises(RuntimeError, match='stop early'):
            solver._solve_cvode(0.0, 1.0, s_stag, 1.0e-4, 1.0e-4, 1.0)
    assert captured_options.get('linsolver') == 'dense'

    captured_options.clear()
    rheo_off = SolidRheologyParams(enabled=False)
    solver_off = _make_solver(_make_abe_solid_params(rheo_off, number_of_nodes=20))
    s_stag_off = _make_solid_profile(solver_off)

    with patch('aragog.solver.entropy_solver._scikits_cvode', side_effect=_spy_cvode):
        with pytest.raises(RuntimeError, match='stop early'):
            solver_off._solve_cvode(0.0, 1.0, s_stag_off, 1.0e-4, 1.0e-4, 1.0)
    assert captured_options.get('linsolver') == 'band'
    assert captured_options.get('lband') == 2
    assert captured_options.get('uband') == 2


def test_solver_setup_validation_pressure_phi_rheo_phi_width() -> None:
    """Solver setup validates basic pressure, phi_rheo in [0, 1], and phi_width > 0."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)

    # Negative pressure guard
    orig_p = solver._P_basic_flat.copy()
    solver._P_basic_flat = np.full_like(orig_p, -10.0)
    with pytest.raises(ValueError, match='basic-node pressure'):
        solver._validate_rheology_setup()
    solver._P_basic_flat = orig_p

    # phi_rheo bounds
    orig_pr = solver.state.phase_basic._phi_rheo
    solver.state.phase_basic._phi_rheo = 1.5
    with pytest.raises(ValueError, match='rheological_transition_melt_fraction'):
        solver._validate_rheology_setup()
    solver.state.phase_basic._phi_rheo = orig_pr

    # phi_width bounds
    orig_pw = solver.state.phase_basic._phi_width
    solver.state.phase_basic._phi_width = -0.1
    with pytest.raises(ValueError, match='rheological_transition_width'):
        solver._validate_rheology_setup()
    solver.state.phase_basic._phi_width = orig_pw


def test_check_rheology_invariants_lid_guards() -> None:
    """_check_rheology_invariants validates lid thickness, active weight, and finite values."""
    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_stag = _make_solid_profile(solver)
    solver.state.update(s_stag, 0.0)
    orig_lid = dict(solver.state._lid_state)

    # Active weight outside [0, 1]
    bad_w = dict(orig_lid)
    bad_w['w_y'] = 1.5
    solver.state._lid_state = bad_w
    with pytest.raises(RuntimeError, match='w_y'):
        solver._check_rheology_invariants()

    # Lid thickness negative
    bad_d = dict(orig_lid)
    bad_d['d_lid'] = -100.0
    solver.state._lid_state = bad_d
    with pytest.raises(RuntimeError, match='d_lid'):
        solver._check_rheology_invariants()

    # Non-finite key in lid_state
    bad_theta = dict(orig_lid)
    bad_theta['theta'] = np.nan
    solver.state._lid_state = bad_theta
    with pytest.raises(RuntimeError, match='non-finite'):
        solver._check_rheology_invariants()

    solver.state._lid_state = orig_lid


def test_phase_params_post_init_validation() -> None:
    """PhaseParams executes rheology.__post_init__ on the supplied rheology object."""
    bad_rheo = SolidRheologyParams(enabled=True)
    object.__setattr__(bad_rheo, 'g_homologous', -1.0)
    with pytest.raises(ValueError, match='g_homologous'):
        PhaseParams(rheology=bad_rheo)


def test_phase_liquid_rheology_nondefault_rejected_across_all_config_paths(tmp_path) -> None:
    """Non-default [phase_liquid.rheology] is rejected with ValueError on TOML, INI, and
    Config.from_dict paths, while default phase_liquid.rheology is accepted."""
    toml_base = importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml').read_text()
    ini_base = importlib.resources.files('aragog').joinpath('cfg/abe_solid.cfg').read_text()

    toml_bad = tmp_path / 'bad_liq.toml'
    toml_bad.write_text(
        toml_base + '\n[phase_liquid.rheology]\nenabled = true\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='phase_liquid'):
        Parameters.from_file(toml_bad)
    with pytest.raises(ValueError, match='phase_liquid'):
        Config.from_toml(str(toml_bad))
    with pytest.raises(ValueError, match='phase_liquid'):
        Config.from_file(str(toml_bad))

    ini_bad = tmp_path / 'bad_liq.cfg'
    ini_bad.write_text(
        ini_base + '\n[phase_liquid.rheology]\nenabled = True\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='phase_liquid'):
        Parameters.from_file(ini_bad)
    with pytest.raises(ValueError, match='phase_liquid'):
        Config.from_file(str(ini_bad))

    import tomllib

    raw_dict = tomllib.loads(toml_base)
    d_nested = copy.deepcopy(raw_dict)
    d_nested['phase_liquid']['rheology'] = {'enabled': True}
    with pytest.raises(ValueError, match='phase_liquid'):
        Config.from_dict(d_nested)

    d_dotted = copy.deepcopy(raw_dict)
    d_dotted['phase_liquid.rheology'] = {'g_homologous': 30.0}
    with pytest.raises(ValueError, match='phase_liquid'):
        Config.from_dict(d_dotted)

    d_default = copy.deepcopy(raw_dict)
    d_default['phase_liquid']['rheology'] = {'enabled': False}
    loaded = Config.from_dict(d_default)
    assert loaded.phase_liquid.rheology == SolidRheologyParams()


def test_arrhenius_reference_viscosity_ci_table_vs_linear_solidus() -> None:
    """At (T_ref = 1600 K, P = 0 Pa) with viscosity_solid = 1e21 Pa s, g_homologous = 25,
    and t_sol_0 = 1400 K, the linear solidus gives 1.0e21 Pa s and the CI EOS solidus
    table (T_sol(0) = 1439.230032 K) gives 1.8459e21 Pa s."""
    import math

    from aragog.rheology import compute_arrhenius_viscosity

    eta_linear = float(
        compute_arrhenius_viscosity(
            temperature=1600.0,
            pressure=0.0,
            viscosity_solid=1.0e21,
            g_homologous=25.0,
            solidus_temperature=None,
            t_sol_0=1400.0,
            dt_sol_dp=2.4e-8,
            arrhenius_t_ref=1600.0,
        )
    )
    assert eta_linear == pytest.approx(1.0e21, rel=1.0e-15, abs=0.0)

    eos_np = entropy_eos_copy(_resolve_eos_dir(), identity_alpha=True)
    t_sol_0_ci = float(
        eos_np._lookup_at_phase_boundary('temperature', np.array([0.0]), 'solid')[0]
    )
    assert t_sol_0_ci == pytest.approx(1439.2298575750483, rel=1.0e-12, abs=0.0)
    assert t_sol_0_ci == pytest.approx(1439.230032, rel=2.0e-7, abs=0.0)

    expected_ci = 1.0e21 * math.exp(25.0 * (t_sol_0_ci - 1400.0) / 1600.0)
    assert expected_ci == pytest.approx(1.8459e21, rel=1.0e-4, abs=0.0)

    eta_ci = float(
        compute_arrhenius_viscosity(
            temperature=1600.0,
            pressure=0.0,
            viscosity_solid=1.0e21,
            g_homologous=25.0,
            solidus_temperature=t_sol_0_ci,
            t_sol_0=1400.0,
            dt_sol_dp=2.4e-8,
            arrhenius_t_ref=1600.0,
        )
    )
    assert eta_ci == pytest.approx(expected_ci, rel=1.0e-14, abs=0.0)
    assert eta_ci == pytest.approx(1.8459e21, rel=1.0e-4, abs=0.0)


def test_check_jax_factory_rheology_requires_cand_rheology_when_enabled() -> None:
    """Verify _check_jax_factory_rheology raises ValueError when solver rheology is enabled
    and the candidate factory does not expose PhaseParams.rheology."""
    from types import SimpleNamespace

    solver_rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(solver_rheo, number_of_nodes=20)
    solver = _make_solver(params)
    bare_factory = SimpleNamespace(rhs_fn=lambda t, y: y)

    with pytest.raises(
        ValueError,
        match='JAX CVODE factory rhs_fn does not expose PhaseParams.rheology',
    ):
        solver._check_jax_factory_rheology(bare_factory, require_when_enabled=True)


def test_compute_stagnant_lid_state_fixed_mode_melt_ramp_transition() -> None:
    """Verify compute_stagnant_lid_state in fixed mode uses a finite melt ramp width."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    n = 20
    r = np.linspace(3.48e6, 6.371e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.full(n, 1400.0)
    t[-4:] = np.linspace(1400.0, 300.0, 4)
    ft = np.ones(n)
    f_thresh = 0.10
    fc = np.full(n, 0.5)
    fc[10] = f_thresh * ft[10]

    st = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=np.zeros(n),
        params=SolidRheologyParams(
            enabled=True,
            lid_base_mode='fixed',
            lid_base_temperature=2000.0,
            interior_flux_fraction=f_thresh,
        ),
        viscosity_solid=1.0e20,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    w_lid = np.asarray(st['w_lid'])
    assert np.all(np.isfinite(w_lid))
    assert not np.isnan(w_lid[10])


# Targeted tests covering nightly selection coverage gaps


def test_jax_phase_params_rejects_non_positive_viscosity_solid() -> None:
    """PhaseParams raises ValueError when viscosity_solid is non-positive or non-finite."""
    with pytest.raises(ValueError, match='viscosity_solid must be finite and positive'):
        PhaseParams(
            rheology=SolidRheologyParams(enabled=True),
            viscosity_solid=0.0,
            phi_rheo=0.5,
            phi_width=0.05,
        )
    with pytest.raises(ValueError, match='viscosity_solid must be finite and positive'):
        PhaseParams(
            rheology=SolidRheologyParams(enabled=True),
            viscosity_solid=-1.0e20,
            phi_rheo=0.5,
            phi_width=0.05,
        )


def test_mesh_arrays_rejects_negative_staggered_pressure() -> None:
    """MeshArrays raises ValueError when P_stag contains negative pressure below -1.0 Pa."""
    kw = dict(
        d_dr_matrix=jnp.zeros((3, 2)),
        quantity_matrix=jnp.zeros((3, 2)),
        area=jnp.ones(3),
        volume=jnp.ones(2),
        radii_basic=jnp.array([1.0e6, 2.0e6, 3.0e6]),
        mixing_length=jnp.ones(3),
        mixing_length_sq=jnp.ones(3),
        mixing_length_cu=jnp.ones(3),
        radii_stag=jnp.array([1.5e6, 2.5e6]),
        P_stag=jnp.array([-5.0, 1.0e5]),
        P_basic=jnp.ones(3),
        gravity=jnp.ones(3) * 9.8,
    )
    with pytest.raises(ValueError, match='pressure at basic and staggered nodes'):
        MeshArrays(**kw, validate_rheology=True)


def test_check_jax_factory_rheology_returns_when_solver_rheo_is_none() -> None:
    """_check_jax_factory_rheology returns immediately when solver has no phase_solid rheology."""
    rheo = SolidRheologyParams(enabled=False)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    solver.parameters.phase_solid = None
    solver._check_jax_factory_rheology(candidate=None)


def test_check_jax_factory_rheology_rejects_mismatched_candidate_rheology() -> None:
    """_check_jax_factory_rheology raises ValueError when candidate rheology differs from solver."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    bad_cand = type('Cand', (), {'rheology': SolidRheologyParams(enabled=False)})()
    with pytest.raises(ValueError, match='does not match parameters.phase_solid.rheology'):
        solver._check_jax_factory_rheology(candidate=bad_cand)


def test_validate_rheology_setup_rejects_non_default_liquid_rheology() -> None:
    """_validate_rheology_setup raises ValueError when phase_liquid.rheology is non-default."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    solver.parameters.phase_liquid.rheology = SolidRheologyParams(enabled=True)
    with pytest.raises(ValueError, match='phase_liquid.rheology must keep default'):
        solver._validate_rheology_setup()


def test_validate_rheology_setup_rejects_fewer_than_three_basic_nodes() -> None:
    """_validate_rheology_setup raises ValueError when basic-node array has fewer than 3 nodes."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    solver._r_basic_flat = np.array([1.0, 2.0])
    with pytest.raises(
        ValueError, match='basic-node radii must be a 1-D array with at least 3 nodes'
    ):
        solver._validate_rheology_setup()


def test_check_rheology_invariants_1d_trace_y_reshapes_and_returns() -> None:
    """_check_rheology_invariants reshapes 1-D trace_y and checks invariants without raising."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_profile = _make_solid_profile(solver)
    solver._check_rheology_invariants(trace_t=0.0, trace_y=s_profile)


def test_check_rheology_step_invariants_raises_on_out_of_bounds_w_lid() -> None:
    """_check_rheology_step_invariants raises RuntimeError when w_lid is outside [0, 1]."""
    rheo = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver = _make_solver(params)
    s_profile = _make_solid_profile(solver)
    solver.state.update(s_profile, 0.0)
    bad_lid = dict(solver.state._lid_state)
    bad_lid['w_lid'] = np.array([-0.5] * len(bad_lid['w_lid']))
    solver.state._lid_state = bad_lid
    with pytest.raises(
        RuntimeError,
        match=r'rheology invariant violated at step 0, node 0: w_lid=.*outside \[0, 1\]',
    ):
        solver._check_rheology_step_invariants(rheo, step_idx=0)
