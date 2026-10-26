"""Smoke tests for solid-state rheology right-hand side and JAX factory wiring."""

from __future__ import annotations

import copy
import importlib.resources
import json
from dataclasses import replace
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
    dSdt,
    dSdt_core_module,
    dSdt_energy_balance,
)
from aragog.parser import (
    Parameters,
)
from aragog.rheology import (
    SolidRheologyParams,
)
from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
from tests.conftest import EOS_DIR, entropy_eos_copy
from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build
from tests.test_jax_dsdt_core_module import _build_jax_pieces
from tests.test_rheology_solver_wiring import (
    _BENCHMARK_JSON,
    _build_jax_bundle,
    _make_abe_solid_params,
    _make_solid_profile,
    _make_solver,
    _measure_jax_rhs_counts,
    _measure_numpy_rhs_counts,
)

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


def test_viscous_mlt_scaling_exponents_jax() -> None:
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

    # Verify non-local Jacobian coupling against central finite differences
    eps = 1.0e-5
    s_p = np.array(s_stag, copy=True)
    s_m = np.array(s_stag, copy=True)
    s_p[-1] += eps
    s_m[-1] -= eps
    fp = np.asarray(dSdt(0.0, jnp.asarray(s_p), args_on))
    fm = np.asarray(dSdt(0.0, jnp.asarray(s_m), args_on))
    fd_col = (fp - fm) / (2.0 * eps)
    np.testing.assert_allclose(jac_on[[-5, -4], -1], fd_col[[-5, -4]], rtol=1.0e-5)


# ============================================================================
# 3. Call-Count Contracts
# ============================================================================


def test_rhs_validate_inputs_zero_calls() -> None:
    """Both NumPy and JAX RHS paths invoke _validate_inputs 0 times."""
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
    """Enabling rheology adds 0 extra EOS solidus temperature lookups."""
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


def test_rhs_viscous_mlt_core_call_counts(monkeypatch) -> None:
    """_viscous_mixing_length_factor_core is called 2 times per RHS,
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
    assert (
        int(np.sum(solver_neg.state._eta_eff_basic != solver_neg.state.phase_basic.eta_diff()))
        >= 8
    )
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

    mesh_pos, eos_jax, params_jax, args_pos = _build_jax_bundle(solver_pos)
    mesh_neg = copy.copy(mesh_pos)
    object.__setattr__(mesh_neg, 'gravity', -jnp.abs(mesh_pos.gravity))
    args_neg = (args_pos[0], args_pos[1], mesh_neg, *args_pos[3:])

    rhs_jax_pos = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_pos))
    rhs_jax_neg = np.asarray(dSdt(0.0, jnp.asarray(s_stag), args_neg))
    np.testing.assert_allclose(rhs_jax_neg, rhs_jax_pos, rtol=1.0e-14, atol=0.0)
    denom = float(np.max(np.abs(rhs_np_pos)))
    assert float(np.max(np.abs(rhs_np_neg - rhs_jax_neg)) / denom) <= 1.0e-6


def test_solver_passes_abs_gravity_to_stagnant_lid_core() -> None:
    """Both EntropyState.update and aragog.jax.phase.compute_fluxes
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


def test_jax_phase_params_dict_validation_smoothing_weight_and_core_bl() -> None:
    """PhaseParams, MeshArrays, and core module boundary layer verify:
    - PhaseParams accepts a dict for rheology via coercion to SolidRheologyParams;
    - PhaseParams and MeshArrays validate static rheology and mesh invariants;
    - spider_get_smoothing and phase_boundary_smoothing with smooth_width=0.0 are finite;
    - PhaseProperties.visc_solid_weight and EntropyPhaseEvaluator.visc_solid_weight() match eq. 77;
    - dSdt_core_module evaluates Arrhenius viscosity on the CMB boundary layer."""
    from aragog.jax.phase import (
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
            validate_rheology=True,
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


@pytest.mark.parametrize('closure_mode', ['lid', 'local'])
def test_rhs_numpy_vs_jax_parity_100_cooling_states(closure_mode: str) -> None:
    """NumPy and JAX right-hand sides agree to relative difference <= 1e-6
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
# 6. Reference-Pinned Validation Benchmarks
# ============================================================================


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_rhs_numpy_jax_parity_rheology_on(validation_plot) -> None:
    """NumPy and JAX right-hand sides agree to relative difference <= 1e-6 across
    100 cooling states on:
    1. the active-solid variant of abe_solid.toml (s_init=2500, kappah_floor=0,
       viscosity_solid=1e18), where Arrhenius strengthening shifts the RHS by >= 1e-4
       at every state and a JAX-only g_homologous perturbation fails 1e-6 by > 8000x;
    2. the yield-active lid state set (s_init=2150, kappah_floor=0, viscosity_solid=3e17,
       yield_stress_c=150 Pa, yield_stress_mu=0), where plastic yielding acts in 9..13
       nodes across all 100/100 states, changes the RHS relative to yield-off
       (tau_y -> infinity) by >= 2.33e-4 >= 1e-4 at every state, agrees between NumPy and JAX to
       <= 1.95e-7 <= 1e-6 (and <= 6.11e-16 with harmonised alpha), and a JAX-only yield-stress
       perturbation fails 1e-6 by > 120x;
    3. the default abe_solid.toml (s_init=2900, kappah_floor=10, viscosity_solid=1e21),
       which remains an inert-path check with RHS_on == RHS_off."""
    data = json.loads(_BENCHMARK_JSON.read_text(encoding='utf-8'))
    parity_block = data['a2_2_rhs_parity_and_activity']
    var_rec = parity_block['abe_solid_kappah_floor_0']
    yld_rec = parity_block['abe_solid_yield_active']
    def_rec = parity_block['abe_solid_default']
    flux_diag = parity_block['flux_vs_rhs_diagnostic']

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

    # Yield-active state set assertions (both unharmonised and harmonised alpha)
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

    # Flux vs RHS diagnostic assertions on abe_solid_kappah_floor_0
    assert flux_diag['worst_state_index'] == 4
    assert flux_diag['worst_basic_node_index'] == 64
    assert flux_diag['worst_staggered_cell_index'] == 63
    assert flux_diag['max_rel_flux_numpy_vs_jax'] == pytest.approx(
        var_rec['max_rel_flux_numpy_vs_jax'], rel=1.0e-12
    )
    assert flux_diag['max_rel_rhs_numpy_vs_jax'] == pytest.approx(
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

    # Live solver verification of the yield-active state set + JAX-only yield canary
    rheo_yld_on = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
    )
    rheo_yld_off = replace(rheo_yld_on, yield_stress_c=1.0e30, yield_stress_max=1.0e30)
    sol_yld_on, sol_yld_off = _build_pair(
        0.0, 1.0e17, rheo_yld_on, rheo_yld_off, identity_alpha=True
    )
    sol_yld_on_unharm, _ = _build_pair(
        0.0, 1.0e17, rheo_yld_on, rheo_yld_off, identity_alpha=False
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
    np.testing.assert_allclose(rhs_def_np_on, rhs_def_np_off, rtol=1.0e-14, atol=0.0)
    _, _, _, args_def_on = _build_jax_bundle(sol_def_on)
    _, _, _, args_def_off = _build_jax_bundle(sol_def_off)
    rhs_def_jax_on = np.asarray(dSdt(0.0, jnp.asarray(s_def), args_def_on))
    rhs_def_jax_off = np.asarray(dSdt(0.0, jnp.asarray(s_def), args_def_off))
    np.testing.assert_allclose(rhs_def_jax_on, rhs_def_jax_off, rtol=1.0e-14, atol=0.0)

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

    validation_plot(figure, 'ssc_rhs_parity')


def test_local_closure_mode_positive_kappah_floor_entropy_state_update() -> None:
    """EntropyState.update in 'local' stress closure mode with positive kappah_floor
    evaluates basic melt fraction before the floor calculation, enforces the
    phase-modulated eddy-diffusivity floor, and agrees with the JAX right-hand side."""
    rheo_local = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='local',
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    base_params = _make_abe_solid_params(rheo_local, number_of_nodes=25)
    params = replace(base_params, energy=replace(base_params.energy, kappah_floor=10.0))
    solver = _make_solver(params, identity_alpha=True)
    _, _, _, args_jax = _build_jax_bundle(solver)

    n_stag = solver._n_stag
    xi = np.linspace(0.0, 1.0, n_stag)
    s_profile = 2500.0 - 400.0 * (0.3 * xi + 0.7 * xi**3)

    rhs_np = solver.dSdt(0.0, s_profile)
    assert np.all(np.isfinite(rhs_np))
    assert solver.state._lid_state is None
    assert np.all(np.isfinite(solver.state.eddy_diffusivity))
    assert np.all(np.isfinite(solver.state._eta_eff_basic))

    phi_b = np.asarray(solver.state.phase_basic.melt_fraction()).ravel()
    phi_rheo = float(getattr(solver.state.phase_basic, '_phi_rheo', 0.4))
    phi_width = float(getattr(solver.state.phase_basic, '_phi_width', 0.15))
    f_floor = 0.5 * (1.0 + np.tanh((phi_b - phi_rheo) / phi_width))
    floor_mask = solver.state._dSdr < solver.state._convective_threshold
    kh_expected_min = 10.0 * f_floor * floor_mask
    assert np.all(solver.state.eddy_diffusivity[1:] >= kh_expected_min[1:] - 1.0e-12)
    assert solver.state.eddy_diffusivity[0] == solver.state.eddy_diffusivity[1]

    rhs_jax = np.asarray(dSdt(0.0, jnp.asarray(s_profile), args_jax))
    denom = float(np.max(np.abs(rhs_np)))
    assert float(np.max(np.abs(rhs_np - rhs_jax)) / denom) <= 1.0e-13


def test_rhs_parity_rheology_on_energy_balance_mode() -> None:
    """NumPy and JAX right-hand sides agree with rheology in energy_balance mode."""
    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        yield_stress_c=1.0e7,
        yield_stress_mu=0.1,
        yield_stress_max=1.0e8,
    )
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        base_params = Parameters.from_file(p)

    params_eb = replace(
        base_params,
        mesh=replace(base_params.mesh, number_of_nodes=20),
        boundary_conditions=replace(
            base_params.boundary_conditions,
            core_bc='energy_balance',
            inner_boundary_condition=1,
        ),
        phase_solid=replace(base_params.phase_solid, rheology=rheo),
    )
    solver_eb = _make_solver(params_eb, identity_alpha=True)
    _, _, _, args_eb = _build_jax_bundle(solver_eb)

    s_stag = _make_solid_profile(solver_eb)
    y_eb = np.concatenate([s_stag, [-1.0e-4]])

    rhs_np = solver_eb.dSdt(0.0, y_eb)
    rhs_jax = np.asarray(dSdt_energy_balance(0.0, jnp.asarray(y_eb), args_eb))

    denom = float(np.max(np.abs(rhs_np)))
    rel_diff = float(np.max(np.abs(rhs_np - rhs_jax)) / denom)
    assert rel_diff <= 1.0e-10


def test_rhs_parity_rheology_on_core_module_mode() -> None:
    """NumPy and JAX right-hand sides agree with rheology in core_module mode."""
    eos = entropy_eos_copy(EOS_DIR, identity_alpha=True)
    solver = _build('core_module', eos, CORE_MODULE_PARAMS, n_nodes=20)
    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    solver.parameters.phase_solid.rheology = rheo
    solver.state.phase_basic._rheology = rheo

    args = _build_jax_pieces(solver)

    s_stag = np.linspace(2950.0, 2600.0, solver._n_stag)
    y_core = np.concatenate([s_stag, [0.0, 5000.0]])

    rhs_np = solver.dSdt(0.0, y_core)
    rhs_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y_core), args))

    denom = float(np.max(np.abs(rhs_np)))
    rel_diff = float(np.max(np.abs(rhs_np - rhs_jax)) / denom)
    assert rel_diff <= 1.0e-10


def test_build_jax_rhs_and_jacobian_validates_mesh_arrays_for_rheology() -> None:
    """Verify build_jax_rhs_and_jacobian calls validate_for_rheology on mesh_arrays
    when phase_params.rheology is enabled."""
    rheo_on = SolidRheologyParams(enabled=True)
    params = _make_abe_solid_params(rheo_on, number_of_nodes=12)
    pp = PhaseParams.from_parameters(params)
    solver = _make_solver(params)
    solver.set_initial_entropy(2500.0)
    mesh_jax, eos_jax, _, dSdt_args = _build_jax_bundle(solver)
    bc_jax = dSdt_args[3]
    heating_arr = np.zeros(solver._n_stag, dtype=float)

    bad_radii = mesh_jax.radii_basic.at[0].set(0.0)
    bad_mesh = replace(mesh_jax, radii_basic=bad_radii)

    # Valid call exposes phase_params and rheology on rhs_fn
    rhs_fn, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=pp,
        mesh_arrays=mesh_jax,
        boundary_params=bc_jax,
        heating_array=heating_arr,
        scales=solver._build_nondim_scales(),
        core_bc_mode='quasi_steady',
    )
    assert getattr(rhs_fn, 'phase_params', None) is not None
    assert getattr(rhs_fn, 'rheology', None) == pp.rheology

    with pytest.raises(ValueError, match='basic-node radii'):
        build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=pp,
            mesh_arrays=bad_mesh,
            boundary_params=bc_jax,
            heating_array=heating_arr,
            scales=solver._build_nondim_scales(),
            core_bc_mode='quasi_steady',
        )
