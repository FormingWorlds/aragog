"""Integration tests for solid-state rheology and stagnant-lid solver execution."""

from __future__ import annotations

import copy
import importlib.resources
import logging
import subprocess
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.jax.phase import (
    MeshArrays,
    PhaseParams,
)
from aragog.jax.solver import (
    BoundaryParams,
    _no_radio,
    dSdt,
    dSdt_energy_balance,
)
from aragog.parser import (
    Parameters,
)
from aragog.rheology import (
    SolidRheologyParams,
)
from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
from aragog.solver.entropy_solver import (
    EntropySolver,
)
from tests.conftest import EOS_DIR, entropy_eos_copy, entropy_eos_jax
from tests.test_rheology_solver_wiring import (
    _build_jax_bundle,
    _make_abe_solid_params,
    _make_solid_profile,
    _make_solver,
    _measure_jax_rhs_counts,
    _measure_numpy_rhs_counts,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# Round-off bounds for the stored main reference (recorded on macOS, read on Linux CI too):
# single RHS evaluations agree to 1e-12; a solve accumulates round-off (Linux: 3.3e-11 on S_final).
MAIN_REF_RTOL = 1.0e-12
MAIN_REF_SOLVE_RTOL = 1.0e-9


def _assert_matches_main_reference(
    actual, expected, key: str, rtol: float = MAIN_REF_RTOL
) -> None:
    """Assert ``actual`` equals the stored main-branch reference ``expected`` within round-off."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    diff = np.abs(a - e)
    scale = float(np.max(np.abs(e))) if e.size else 0.0
    rel = diff / np.maximum(np.abs(e), 1e-300)
    np.testing.assert_allclose(
        a,
        e,
        rtol=rtol,
        atol=rtol * scale,
        err_msg=f'{key}: max abs diff {diff.max():.3e}, max rel diff {rel.max():.3e}',
    )


def test_rheology_off_preserves_main_rhs_and_solve(caplog) -> None:
    """When rheology is disabled, zero rheology core functions are invoked,
    and RHS and solve outputs match the pinned main reference across all 54 arrays
    (tolerance 1e-12 per RHS and 1e-9 per solve, relative to each value or to the
    largest value of the array). The JAX solve, here and in the reference, runs on the
    JAX RHS with the analytic Jacobian (option Z)."""
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

    ref_path = Path(__file__).parent / 'reference' / 'rheology_off_main_reference.npz'
    ref = np.load(ref_path)

    assert 'source_commit' in ref, 'rheology_off_main_reference.npz must record source_commit'
    source_commit = str(ref['source_commit']).strip()
    assert len(source_commit) == 40, f'Invalid commit SHA: {source_commit}'

    proc_ancestor = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', source_commit, 'origin/main'],
        cwd=Path(__file__).parent,
        capture_output=True,
    )
    if proc_ancestor.returncode != 0:
        proc_mb = subprocess.run(
            ['git', 'merge-base', 'HEAD', 'origin/main'],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        assert proc_mb.returncode == 0 and proc_mb.stdout.strip() == source_commit, (
            f'Stored commit {source_commit} is not an ancestor of origin/main or equal to merge base'
        )

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        raw_params = Parameters.from_file(p)

    eos_dir = EOS_DIR

    # 1. 42 quasi-steady RHS arrays
    configs = {
        'default': {},
        'floor0': {'kappah_floor': 0.0},
        'gravsep': {'gravitational_separation': True},
        'mixing': {'mixing': True},
        'gravmix': {'gravitational_separation': True, 'mixing': True},
        'nocond': {'conduction': False},
        'noconv': {'convection': False},
    }

    for cfg_name, en in configs.items():
        params = replace(
            raw_params,
            mesh=replace(raw_params.mesh, number_of_nodes=20),
            boundary_conditions=replace(
                raw_params.boundary_conditions,
                core_bc='quasi_steady',
                inner_boundary_condition=1,
            ),
            energy=replace(raw_params.energy, **en),
        )
        solver = EntropySolver(
            params, entropy_eos=entropy_eos_copy(eos_dir, identity_alpha=True)
        )
        solver.initialize()
        eos = solver.state.phase_basic._eos
        ps = np.asarray(solver.state.phase_staggered.pressure, float)
        ssol = np.asarray(eos.solidus_entropy(ps), float)
        sliq = np.asarray(eos.liquidus_entropy(ps), float)
        n = solver._n_stag
        profs = dict(
            solid=np.linspace(ssol[0] - 65.0, ssol[-1] - 120.0, n),
            mushy=ssol + (sliq - ssol) * np.linspace(0.1, 0.9, n),
            melt=sliq + np.linspace(50.0, 300.0, n),
        )
        mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
        pj = PhaseParams(
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
        r_cmb = float(mesh_jax.radii_basic[0])
        r_next = float(mesh_jax.radii_basic[1])
        bc = BoundaryParams(
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
        H_int = jnp.zeros(solver._n_stag, dtype=jnp.float64)
        args_jax = (entropy_eos_jax(eos_dir), pj, mesh_jax, bc, H_int, _no_radio)

        for prof_name, s_profile in profs.items():
            rhs_np = solver.dSdt(0.0, s_profile)
            rhs_jax = np.asarray(dSdt(0.0, jnp.asarray(s_profile), args_jax))
            _assert_matches_main_reference(
                rhs_np, ref[f'rhs/{cfg_name}/{prof_name}/np'], f'rhs/{cfg_name}/{prof_name}/np'
            )
            _assert_matches_main_reference(
                rhs_jax,
                ref[f'rhs/{cfg_name}/{prof_name}/jax'],
                f'rhs/{cfg_name}/{prof_name}/jax',
            )

    # 2. 6 energy_balance RHS arrays
    params_eb = replace(
        raw_params,
        mesh=replace(raw_params.mesh, number_of_nodes=20),
        boundary_conditions=replace(
            raw_params.boundary_conditions,
            core_bc='energy_balance',
            inner_boundary_condition=1,
        ),
    )
    solver_eb = EntropySolver(
        params_eb, entropy_eos=entropy_eos_copy(eos_dir, identity_alpha=True)
    )
    solver_eb.initialize()
    eos_eb = solver_eb.state.phase_basic._eos
    ps_eb = np.asarray(solver_eb.state.phase_staggered.pressure, float)
    ssol_eb = np.asarray(eos_eb.solidus_entropy(ps_eb), float)
    sliq_eb = np.asarray(eos_eb.liquidus_entropy(ps_eb), float)
    n_eb = solver_eb._n_stag
    profs_eb = dict(
        solid=np.concatenate(
            [np.linspace(ssol_eb[0] - 65.0, ssol_eb[-1] - 120.0, n_eb), [-1e-4]]
        ),
        mushy=np.concatenate(
            [ssol_eb + (sliq_eb - ssol_eb) * np.linspace(0.1, 0.9, n_eb), [-1e-4]]
        ),
        melt=np.concatenate([sliq_eb + np.linspace(50.0, 300.0, n_eb), [-1e-4]]),
    )
    mesh_jax_eb = MeshArrays.from_numpy_mesh(solver_eb.evaluator.mesh)
    pj_eb = PhaseParams(
        phi_rheo=float(params_eb.phase_mixed.rheological_transition_melt_fraction),
        phi_width=float(params_eb.phase_mixed.rheological_transition_width),
        viscosity_solid=float(params_eb.phase_solid.viscosity),
        viscosity_liquid=float(params_eb.phase_liquid.viscosity),
        grain_size=float(params_eb.phase_mixed.grain_size),
        k_solid=float(params_eb.phase_solid.thermal_conductivity),
        k_liquid=float(params_eb.phase_liquid.thermal_conductivity),
        matprop_smooth_width=float(params_eb.phase_mixed.matprop_smooth_width),
        conduction=params_eb.energy.conduction,
        convection=params_eb.energy.convection,
        grav_sep=params_eb.energy.gravitational_separation,
        mixing=params_eb.energy.mixing,
        eddy_diff_thermal=float(params_eb.energy.eddy_diffusivity_thermal),
        eddy_diff_chemical=float(params_eb.energy.eddy_diffusivity_chemical),
        kappah_floor=float(params_eb.energy.kappah_floor),
        bottom_up_grav_sep=params_eb.energy.bottom_up_grav_sep,
        phase_smoothing=params_eb.energy.phase_smoothing,
        separation_viscosity=params_eb.phase_mixed.separation_viscosity,
    )
    r_cmb_eb = float(mesh_jax_eb.radii_basic[0])
    r_next_eb = float(mesh_jax_eb.radii_basic[1])
    bc_eb = BoundaryParams(
        outer_bc_type=int(params_eb.boundary_conditions.outer_boundary_condition),
        outer_bc_value=float(params_eb.boundary_conditions.outer_boundary_value),
        emissivity=float(params_eb.boundary_conditions.emissivity),
        T_eq=float(params_eb.boundary_conditions.equilibrium_temperature),
        inner_bc_type=int(params_eb.boundary_conditions.inner_boundary_condition),
        inner_bc_value=float(params_eb.boundary_conditions.inner_boundary_value),
        core_density=float(params_eb.mesh.core_density),
        core_heat_capacity=float(params_eb.boundary_conditions.core_heat_capacity),
        tfac_core_avg=float(solver_eb._core_tfac),
        cmb_area=4.0 * np.pi * r_cmb_eb**2,
        core_M=float(solver_eb._core_cap / params_eb.boundary_conditions.core_heat_capacity),
        cmb_dr_cmb=r_next_eb - r_cmb_eb,
        param_utbl=bool(params_eb.boundary_conditions.param_utbl),
        param_utbl_const=float(params_eb.boundary_conditions.param_utbl_const),
    )
    args_eb = (
        entropy_eos_jax(eos_dir),
        pj_eb,
        mesh_jax_eb,
        bc_eb,
        jnp.zeros(solver_eb._n_stag),
        _no_radio,
    )
    for name, y in profs_eb.items():
        rhs_np = solver_eb.dSdt(0.0, y)
        rhs_jax = np.asarray(dSdt_energy_balance(0.0, jnp.asarray(y), args_eb))
        _assert_matches_main_reference(
            rhs_np,
            ref['rhs/energy_balance/' + name + '/np'],
            'rhs/energy_balance/' + name + '/np',
        )
        _assert_matches_main_reference(
            rhs_jax,
            ref['rhs/energy_balance/' + name + '/jax'],
            'rhs/energy_balance/' + name + '/jax',
        )

    # 3. 6 short solve() trajectory arrays (100 yr, 20 nodes)
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        c_solve = Config.from_file(str(p))
    c_solve.mesh.number_of_nodes = 20
    c_solve.solver.end_time = 100.0
    c_solve.solver.cvode_max_steps = 100
    eos_inst = EntropyEOS(str(eos_dir))

    s_np = EntropySolver(c_solve, entropy_eos=eos_inst)
    s_np.initialize()
    s_np.set_initial_entropy(2500.0)
    s_np.solve()
    out_np = s_np.get_state()
    _assert_matches_main_reference(
        np.asarray(out_np.S_final),
        ref['solve/np/S_final'],
        'solve/np/S_final',
        rtol=MAIN_REF_SOLVE_RTOL,
    )
    _assert_matches_main_reference(
        np.asarray(out_np.T_stag),
        ref['solve/np/T_stag'],
        'solve/np/T_stag',
        rtol=MAIN_REF_SOLVE_RTOL,
    )
    _assert_matches_main_reference(
        np.asarray(out_np.heat_flux),
        ref['solve/np/heat_flux'],
        'solve/np/heat_flux',
        rtol=MAIN_REF_SOLVE_RTOL,
    )

    s_jax = EntropySolver(c_solve, entropy_eos=eos_inst)
    s_jax.initialize()
    s_jax.set_initial_entropy(2500.0)
    mesh_jx, eos_jx, pp_jx, args_jx = _build_jax_bundle(s_jax)
    built = []

    def _solve_factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jx,
            phase_params=pp_jx,
            mesh_arrays=mesh_jx,
            boundary_params=args_jx[3],
            heating_array=np.zeros(s_jax._n_stag),
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        built.append(core_bc_mode)
        return rhs_fn, jac_fn

    s_jax.set_jax_cvode_factory(_solve_factory)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.aragog.solver.entropy_solver'):
        s_jax.solve()
    # solve() catches a factory error and continues on the NumPy RHS.
    assert 'JAX CVODE factory failed' not in caplog.text, caplog.text
    assert 'option Z active' in caplog.text, 'the JAX solve did not run on the JAX RHS'
    assert built, 'solve() never built the JAX callbacks'
    assert bool(ref['solve/jax/option_z_active']), 'the stored JAX solve ran on the NumPy RHS'
    out_jax = s_jax.get_state()
    _assert_matches_main_reference(
        np.asarray(out_jax.S_final),
        ref['solve/jax/S_final'],
        'solve/jax/S_final',
        rtol=MAIN_REF_SOLVE_RTOL,
    )
    _assert_matches_main_reference(
        np.asarray(out_jax.T_stag),
        ref['solve/jax/T_stag'],
        'solve/jax/T_stag',
        rtol=MAIN_REF_SOLVE_RTOL,
    )
    _assert_matches_main_reference(
        np.asarray(out_jax.heat_flux),
        ref['solve/jax/heat_flux'],
        'solve/jax/heat_flux',
        rtol=MAIN_REF_SOLVE_RTOL,
    )

    # Verify that turning rheology on shifts RHS relative to rheology off
    params_on = _make_abe_solid_params(SolidRheologyParams(enabled=True), number_of_nodes=20)
    solver_on = _make_solver(params_on)
    _, _, _, args_on = _build_jax_bundle(solver_on)
    s_stag_ref = profs['solid']
    rhs_np_off = solver.dSdt(0.0, s_stag_ref)
    rhs_np_on = solver_on.dSdt(0.0, s_stag_ref)
    assert not np.allclose(rhs_np_off, rhs_np_on)

    rhs_jax_off = np.asarray(dSdt(0.0, jnp.asarray(s_stag_ref), args_jax))
    rhs_jax_on = np.asarray(dSdt(0.0, jnp.asarray(s_stag_ref), args_on))
    assert not np.allclose(rhs_jax_off, rhs_jax_on)


# ============================================================================
# 4. Invariant Checks on Accepted Steps (_check_rheology_invariants)
# ============================================================================


def test_solve_short_trajectory_rheology_on_lid_and_local() -> None:
    """Short solve() trajectory completes successfully with rheology in both
    'lid' and 'local' modes."""
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        base_cfg = Config.from_file(str(p))

    base_cfg.mesh.number_of_nodes = 20
    base_cfg.solver.end_time = 100.0
    base_cfg.solver.cvode_max_steps = 100
    eos_inst = EntropyEOS(str(EOS_DIR))

    for mode in ('lid', 'local'):
        cfg = copy.deepcopy(base_cfg)
        cfg.phase_solid.rheology = replace(
            cfg.phase_solid.rheology, enabled=True, stress_closure_mode=mode
        )
        solver = EntropySolver(cfg, entropy_eos=eos_inst)
        solver.initialize()
        solver.set_initial_entropy(2500.0)
        solver.solve()
        out = solver.get_state()
        assert np.all(np.isfinite(out.S_final))
        assert np.all(np.isfinite(out.T_stag))
        assert np.all(np.isfinite(out.heat_flux))


def test_const_properties_with_rheology_enabled_sets_phi_zero_arrhenius_viscosity_and_logs(
    caplog,
) -> None:
    """When const_properties=True and rheology.enabled=True, setup logs once,
    melt_fraction is 0 (solid), and viscosity is the Arrhenius solid viscosity
    rather than 10**const_log10visc."""
    import logging

    from aragog.rheology import compute_arrhenius_viscosity

    rheo = SolidRheologyParams(
        enabled=True,
        g_homologous=25.0,
        t_sol_0=1400.0,
        dt_sol_dp=2.4e-8,
        arrhenius_t_ref=1600.0,
    )
    base_params = _make_abe_solid_params(rheo, number_of_nodes=12)
    params = replace(
        base_params,
        phase_solid=replace(base_params.phase_solid, viscosity=1.0e20),
        phase_mixed=replace(
            base_params.phase_mixed,
            const_properties=True,
            const_log10visc=2.0,
            const_T_ref=1600.0,
            const_S_ref=2500.0,
            const_Cp=1200.0,
            const_rho=4000.0,
            const_alpha=3.0e-5,
            const_cond=4.0,
        ),
    )
    solver = EntropySolver(params, entropy_eos=None)
    with caplog.at_level(logging.INFO, logger='fwl.aragog.solver.entropy_solver'):
        solver.initialize()
    log_msgs = [r.message for r in caplog.records if 'const_properties' in r.message]
    assert len(log_msgs) == 1
    assert 'phi=0' in log_msgs[0]
    assert 'const_log10visc' in log_msgs[0]

    s_stag = np.full(solver._n_stag, 2500.0, dtype=float)
    solver.dSdt(0.0, s_stag)
    phi_b = np.asarray(solver.state.phase_basic.melt_fraction(), dtype=float)
    phi_s = np.asarray(solver.state.phase_staggered.melt_fraction(), dtype=float)
    assert np.all(phi_b == 0.0)
    assert np.all(phi_s == 0.0)

    t_b = np.asarray(solver.state.phase_basic.temperature(), dtype=float)
    p_b = np.asarray(solver.state.phase_basic.pressure, dtype=float)
    eta_expected = np.asarray(
        compute_arrhenius_viscosity(
            temperature=t_b,
            pressure=p_b,
            viscosity_solid=1.0e20,
            g_homologous=25.0,
            solidus_temperature=None,
            t_sol_0=1400.0,
            dt_sol_dp=2.4e-8,
            arrhenius_t_ref=1600.0,
        ),
        dtype=float,
    )
    eta_diff_b = np.asarray(solver.state.phase_basic.eta_diff(), dtype=float)
    np.testing.assert_allclose(eta_diff_b, eta_expected, rtol=1.0e-14, atol=0.0)
    assert not np.any(np.isclose(eta_diff_b, 100.0))

    params_solve = replace(
        params,
        solver=replace(params.solver, end_time=10.0, max_steps=20),
    )
    solver_on = EntropySolver(params_solve, entropy_eos=None)
    solver_on.initialize()
    solver_on.set_initial_entropy(2500.0)
    solver_on.solve()
    out_on = solver_on.get_state()
    assert float(out_on.Phi_global) == 0.0
    assert float(out_on.Phi_global_vol) == 0.0
    assert np.all(np.asarray(out_on.phi_stag) == 0.0)
    assert np.all(np.asarray(out_on.porosity_b) == 0.0)

    params_off = replace(
        params_solve,
        phase_solid=replace(params_solve.phase_solid, rheology=SolidRheologyParams()),
    )
    solver_off = EntropySolver(params_off, entropy_eos=None)
    solver_off.initialize()
    solver_off.set_initial_entropy(2500.0)
    solver_off.solve()
    out_off = solver_off.get_state()
    assert float(out_off.Phi_global) == pytest.approx(1.0)
    assert float(out_off.Phi_global_vol) == pytest.approx(1.0)
    assert np.all(np.asarray(out_off.phi_stag) == 1.0)
    assert np.all(np.asarray(out_off.porosity_b) == 1.0)


def test_solve_rheology_invariants_checked_inside_step_energy_integrals_without_double_replay(
    monkeypatch,
) -> None:
    """solve() checks rheology step invariants inside _compute_step_energy_integrals()
    without a duplicate NumPy _dSdt_single replay loop per step."""
    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo, number_of_nodes=12)
    params = replace(
        params,
        solver=replace(params.solver, end_time=20.0, max_steps=50),
    )
    solver = _make_solver(params)
    solver.set_initial_entropy(2500.0)

    step_invariant_indices: list[int] = []
    orig_step_check = solver._check_rheology_step_invariants

    def _spy_step_check(rheo_arg, step_idx: int = 0, time_yr: float | None = None) -> None:
        step_invariant_indices.append(step_idx)
        orig_step_check(rheo_arg, step_idx=step_idx, time_yr=time_yr)

    monkeypatch.setattr(solver, '_check_rheology_step_invariants', _spy_step_check)

    orig_step_powers = solver._step_powers
    step_powers_calls = [0]

    def _spy_step_powers(t: float, y: np.ndarray):
        step_powers_calls[0] += 1
        return orig_step_powers(t, y)

    monkeypatch.setattr(solver, '_step_powers', _spy_step_powers)
    solver.solve()

    assert step_powers_calls[0] >= 2
    assert step_invariant_indices == list(range(step_powers_calls[0]))


def test_phase_params_from_parameters_and_jax_factory_rheology_mismatch_guard() -> None:
    """PhaseParams.from_parameters mirrors all fields of Parameters, and solve() /
    set_jax_cvode_factory raise ValueError when factory PhaseParams.rheology mismatches
    parameters.phase_solid.rheology."""
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    rheo_on = SolidRheologyParams(enabled=True, g_homologous=27.5, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo_on, number_of_nodes=12)
    params = replace(
        params,
        energy=replace(params.energy, use_jax_jacobian=True),
        solver=replace(params.solver, end_time=10.0, max_steps=30),
    )
    pp = PhaseParams.from_parameters(params)
    assert pp.rheology == rheo_on
    assert pp.phi_rheo == float(params.phase_mixed.rheological_transition_melt_fraction)
    assert pp.phi_width == float(params.phase_mixed.rheological_transition_width)
    assert float(pp.log10_visc_solid) == pytest.approx(np.log10(params.phase_solid.viscosity))
    assert float(pp.log10_visc_liquid) == pytest.approx(np.log10(params.phase_liquid.viscosity))
    assert pp.kappah_floor == float(params.energy.kappah_floor)
    assert pp.matprop_smooth_width == float(params.phase_mixed.matprop_smooth_width)
    assert pp.phase_smoothing_width == (
        float(params.phase_mixed.matprop_smooth_width) or 1.0e-2
    )

    solver = _make_solver(params)
    solver.set_initial_entropy(2500.0)
    mesh_jax, eos_jax, _, dSdt_args = _build_jax_bundle(solver)
    bc_jax = dSdt_args[3]
    heating_arr = np.zeros(solver._n_stag, dtype=float)

    pp_mismatch = PhaseParams.from_parameters(
        params, rheology=SolidRheologyParams(enabled=False)
    )
    with pytest.raises(ValueError, match='rheology'):
        solver.set_jax_cvode_factory(
            lambda scales, core_bc: (None, None), phase_params=pp_mismatch
        )

    def _mismatched_factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=pp_mismatch,
            mesh_arrays=mesh_jax,
            boundary_params=bc_jax,
            heating_array=heating_arr,
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs_fn, jac_fn

    solver.set_jax_cvode_factory(_mismatched_factory)
    with pytest.raises(ValueError, match='rheology'):
        solver.solve()

    def _matched_factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=pp,
            mesh_arrays=mesh_jax,
            boundary_params=bc_jax,
            heating_array=heating_arr,
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs_fn, jac_fn

    solver.set_jax_cvode_factory(_matched_factory)
    solver.solve()
    assert np.all(np.isfinite(solver.get_state().S_final))


def test_check_rheology_invariants_enforces_eos_entropy_range_and_stops_table_excursion() -> (
    None
):
    """_check_rheology_invariants requires every accepted state's entropy S to lie inside
    [S_min, S_max] of the loaded EOS table (raising RuntimeError naming node, time, and S),
    and stops the radiative cooling case (abe_solid.toml, T_eqm = 273 K) at 17 to 18 kyr."""
    import re

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params_20 = _make_abe_solid_params(rheo, number_of_nodes=20)
    solver_20 = _make_solver(params_20)
    eos = solver_20.entropy_eos
    assert eos is not None
    s_min = float(eos.S_min)
    s_max = float(eos.S_max)

    s_good = _make_solid_profile(solver_20)
    solver_20.state.update(s_good, 100.0)
    solver_20._check_rheology_invariants()

    # 1. Direct state check below S_min names node, time, and S
    s_below = s_good.copy()
    s_below[14] = s_min - 10.0
    solver_20.state.update(s_below, 1234.5)
    with pytest.raises(RuntimeError) as exc_below:
        solver_20._check_rheology_invariants()
    msg_below = str(exc_below.value)
    assert 'node 14' in msg_below
    assert '1.234500e+03' in msg_below
    assert f'{s_min - 10.0:.6f}' in msg_below

    # 2. Direct state check above S_max names node, time, and S
    s_above = s_good.copy()
    s_above[7] = s_max + 25.0
    solver_20.state.update(s_above, 5678.0)
    with pytest.raises(RuntimeError) as exc_above:
        solver_20._check_rheology_invariants()
    msg_above = str(exc_above.value)
    assert 'node 7' in msg_above
    assert '5.678000e+03' in msg_above
    assert f'{s_max + 25.0:.6f}' in msg_above

    # 3. Trace check catches an intermediate accepted column outside [S_min, S_max]
    trace_y = np.column_stack([s_good, s_good, s_good])
    trace_y[11, 1] = s_min - 5.0
    trace_t = np.array([0.0, 42.0, 100.0])
    with pytest.raises(RuntimeError) as exc_trace:
        solver_20._check_rheology_invariants(trace_y, trace_t)
    msg_trace = str(exc_trace.value)
    assert 'step 1' in msg_trace
    assert 'node 11' in msg_trace
    assert '4.200000e+01' in msg_trace

    # Bypass when entropy_eos is None or const_properties=True
    params_const = replace(
        params_20,
        phase_mixed=replace(params_20.phase_mixed, const_properties=True),
    )
    solver_const = EntropySolver(params_const, entropy_eos=None)
    solver_const.initialize()
    s_extreme = np.full(solver_const._n_stag, -5.0e5, dtype=float)
    solver_const.state.update(s_extreme, 10.0)
    solver_const._check_rheology_invariants()

    if not EOS_DIR.is_dir():
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}.')

    # 4. Radiative cooling case (cfg/abe_solid.toml at (0.25, 0.25), T_eqm = 273 K)
    # stops at ~17 to 18 kyr when top staggered node 98 crosses below S_min.
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        base_abe = Parameters.from_file(p)
    rheo_cool = SolidRheologyParams(enabled=True, s_top_inf=0.25, c_theta=0.25)
    params_cool = replace(
        base_abe,
        energy=replace(base_abe.energy, use_jax_jacobian=True),
        solver=replace(base_abe.solver, end_time=1.0e6),
        phase_solid=replace(base_abe.phase_solid, rheology=rheo_cool),
    )
    solver_cool = _make_solver(params_cool, identity_alpha=False)
    solver_cool.set_initial_entropy(2900.0)
    mesh_jax, eos_jax, pp, dSdt_args = _build_jax_bundle(solver_cool)
    bc_jax = dSdt_args[3]
    heating_arr = np.zeros(solver_cool._n_stag, dtype=float)

    def _factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=pp,
            mesh_arrays=mesh_jax,
            boundary_params=bc_jax,
            heating_array=heating_arr,
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs_fn, jac_fn

    solver_cool.set_jax_cvode_factory(_factory, phase_params=pp)
    with pytest.raises(RuntimeError) as exc_cool:
        solver_cool.solve()
    msg_cool = str(exc_cool.value)
    assert 'node 98' in msg_cool
    m_time = re.search(r't=([0-9.eE+-]+)\s*yr', msg_cool)
    assert m_time is not None, f'Expected time in error message, got: {msg_cool}'
    t_fail_yr = float(m_time.group(1))
    assert 1.7e4 <= t_fail_yr <= 1.85e4
    m_s = re.search(r'S=([0-9.eE+-]+)\s*J/\(kg K\)', msg_cool)
    assert m_s is not None, f'Expected entropy S in error message, got: {msg_cool}'
    s_fail = float(m_s.group(1))
    assert s_fail < s_min
    assert 1.7e4 <= float(solver_cool._solution.t[-1]) <= 1.85e4


def test_step_energy_integrals_checks_eos_entropy_range_on_every_step_both_branches(
    caplog,
) -> None:
    """_compute_step_energy_integrals checks the EOS entropy-range invariant on every column
    of y_pts on both the JAX batch branch and the NumPy loop branch, raising RuntimeError when
    an intermediate accepted step leaves [S_min, S_max] even if the final step is in bounds."""
    import logging

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    rheo = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params = _make_abe_solid_params(rheo, number_of_nodes=20)
    params = replace(
        params,
        energy=replace(params.energy, use_jax_jacobian=True),
        solver=replace(params.solver, end_time=20.0, max_steps=60),
    )
    solver = _make_solver(params, identity_alpha=False)
    solver.set_initial_entropy(2500.0)
    eos = solver.entropy_eos
    assert eos is not None
    s_min = float(eos.S_min)
    s_max = float(eos.S_max)

    mesh_jax, eos_jax, pp, dSdt_args = _build_jax_bundle(solver)
    bc_jax = dSdt_args[3]
    heating_arr = np.zeros(solver._n_stag, dtype=float)

    def _factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=pp,
            mesh_arrays=mesh_jax,
            boundary_params=bc_jax,
            heating_array=heating_arr,
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs_fn, jac_fn

    solver.set_jax_cvode_factory(_factory, phase_params=pp)
    with caplog.at_level(logging.INFO, logger='fwl.aragog.solver.entropy_solver'):
        solver.solve()
    assert solver._batch_powers_calls == 1
    assert any('via JAX batch path' in r.message for r in caplog.records)
    batch_fn = solver._cvode_step_powers_batch
    assert batch_fn is not None

    y_end = np.asarray(solver._solution.y[:, -1], dtype=float).copy()
    t_trace = np.array([0.0, 10.0, 20.0], dtype=float)

    # Intermediate step 1 below S_min; step 0 and final step 2 are valid in-bounds states.
    y_below = np.column_stack([y_end, y_end, y_end])
    y_below[12, 1] = s_min - 20.0
    solver._solution['energy_trace'] = (t_trace, y_below)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.aragog.solver.entropy_solver'):
        with pytest.raises(RuntimeError) as exc_batch_below:
            solver._compute_step_energy_integrals()
    msg_batch_below = str(exc_batch_below.value)
    assert any('via JAX batch path' in r.message for r in caplog.records)
    assert 'step 1' in msg_batch_below
    assert 'node 12' in msg_batch_below
    assert '1.000000e+01' in msg_batch_below
    assert f'{s_min - 20.0:.6f}' in msg_batch_below

    # Intermediate step 1 above S_max on the JAX batch branch.
    y_above = np.column_stack([y_end, y_end, y_end])
    y_above[5, 1] = s_max + 20.0
    solver._solution['energy_trace'] = (t_trace, y_above)
    with pytest.raises(RuntimeError) as exc_batch_above:
        solver._compute_step_energy_integrals()
    msg_batch_above = str(exc_batch_above.value)
    assert 'step 1' in msg_batch_above
    assert 'node 5' in msg_batch_above
    assert '1.000000e+01' in msg_batch_above
    assert f'{s_max + 20.0:.6f}' in msg_batch_above

    # NumPy loop fallback branch also catches the intermediate step 1 excursion.
    solver._cvode_step_powers_batch = None
    solver._solution['energy_trace'] = (t_trace, y_below)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.aragog.solver.entropy_solver'):
        with pytest.raises(RuntimeError) as exc_numpy_below:
            solver._compute_step_energy_integrals()
    msg_numpy_below = str(exc_numpy_below.value)
    assert any('via numpy loop fallback' in r.message for r in caplog.records)
    assert 'step 1' in msg_numpy_below
    assert 'node 12' in msg_numpy_below
    assert '1.000000e+01' in msg_numpy_below
    assert f'{s_min - 20.0:.6f}' in msg_numpy_below
