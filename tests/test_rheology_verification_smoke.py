"""Smoke-tier verification tests for 0-D stagnant-lid cooling and yield thinning."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aragog.rheology import SolidRheologyParams

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(120)]

_R = 8.314462618


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_0d_stagnant_lid_reference_and_yield_thinning(validation_plot) -> None:
    """Compare 1-D stagnant-lid cooling and yield thinning against the 0-D reference model.

    The comparison fails the derived 0-D reference bounds on both the 4.5 Gyr secular
    cooling trajectory (5 % on T_i, 22 % on d_lid) and the steady-state yield-ceiling
    thinning ratio (11 % at 0.5 * tau_d_lid against 0.5**(1/3) = 0.7937):
    1. Over 4.5 Gyr of spherical constant-property secular cooling from T_init = 1800 K
       at (s_top_inf, c_theta) = (0.25, 0.25), the 1-D interior temperature T_i(t)
       deviates from the 0-D ODE by up to 6.67 % (0.066718 at 4.50 Gyr, exceeding 5 %
       after 3.32 Gyr; linear crossing at 3.324 Gyr, first grid sample above 5 % at
       3.384 Gyr, and 5.34 % maximum error within the strict 0-D Nu > 1 validity window
       t < 3.632 Gyr before the 0-D Nu = max(Nu_unyielded, 1.0) clamp activates), and the
       post-relaxation (t >= 25 Myr) isotherm lid thickness d_lid(t) deviates by up to
       47.32 % (0.473228 at 3.63 Gyr, crossing 22 % at 408.4 Myr; 1.000000 at t = 0 yr
       before boundary-layer relaxation).
    2. In the steady-state planar internal-heating yield-ceiling sweep (E300, N = 100,
       tau_d_lid = 1.956 MPa), lowering surface cohesion to 0.5, 0.2, and 0.1 * tau_d_lid
       gives Pass-1 isotherm lid-thickness ratios d_lid(tau_y) / d_lid(unyielded) of
       1.5259, 1.7397, and 1.9118 at fixed internal heating and 1.0466, 1.0551, and 1.0581
       at matched T_i = 1666.42 K (against the 0-D target 0.7937, 0.5848, and 0.4642),
       because Pass-2 effective-viscosity reduction mobilizes convective heat transport
       below ~20.2 km depth (2 radial node spacings on N = 100) while steepening the
       conductive temperature gradient in the top 20.2 km, leaving the Pass-1 rheological
       isotherm T_i - 2.2 * dT_rh unthinned.

    This test verifies the Korenaga (2009) Table 1 a_nu provenance and published
    Solomatov & Moresi (2000) a_nu = 0.53 reference trajectory, runs a live 25 Myr 1-D
    trajectory integration against the 5-start round-off spread tolerance, and evaluates
    live 1-D yielded states (E300 N = 100 steady states and the 1601.47 K spherical
    cooling state at 0.5 * tau_d_lid) so loss of lid yielding is caught directly.
    """
    import importlib.resources
    import json
    from pathlib import Path

    from aragog.parser import (
        Parameters,
        _BoundaryConditionsParameters,
        _EnergyParameters,
        _InitialConditionParameters,
        _MeshParameters,
        _PhaseMixedParameters,
        _PhaseParameters,
        _Radionuclide,
        _SolverParameters,
    )
    from aragog.solver.entropy_solver import EntropySolver
    from tests.reference.stagnant_lid_0d import (
        evaluate_stagnant_lid_0d,
        integrate_stagnant_lid_0d,
        load_aragog_0d_params,
    )

    ref_path = Path(__file__).parent / 'reference' / 'ssc_0d_reference_benchmarks.json'
    data = json.loads(ref_path.read_text(encoding='utf-8'))
    tols = data['metadata']['tolerances']
    rel_pin = float(tols['round_off_spread_rel_tol'])
    assert rel_pin == pytest.approx(1.0e-7, rel=1.0e-12)
    assert float(tols['tol_rel_T_i']) == pytest.approx(0.05, rel=1.0e-12)
    assert float(tols['tol_rel_d_lid']) == pytest.approx(0.22, rel=1.0e-12)
    assert float(tols['tol_rel_R_thin']) == pytest.approx(0.11, rel=1.0e-12)

    # Verify Korenaga (2009) Table 1 provenance of a_nu = 0.5954345503010264
    prov = data['a_nu_provenance']
    assert bool(prov['fitted_to_1d_aragog_solver']) is False
    kor_rows = prov['korenaga_2009_table1_rows']
    assert len(kor_rows) == 12
    a_nu_rows = np.array(
        [
            float(r['Nu_ref'])
            * (float(r['theta']) ** (4.0 / 3.0))
            * (float(r['Ra_i']) ** (-1.0 / 3.0))
            for r in kor_rows
        ]
    )
    assert float(np.mean(a_nu_rows)) == pytest.approx(0.5954345503010264, rel=1.0e-15)
    assert float(prov['a_nu_value']) == pytest.approx(0.5954345503010264, rel=1.0e-15)

    repo_root = Path(__file__).resolve().parents[1]
    p0d = load_aragog_0d_params(
        abe_solid_toml=repo_root / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml',
        viscosity_solid=1.0e18,
        t_surf=300.0,
        p_i_depth_m=100.0e3,
        a_rh_fb=1.3,
        c4_fb=0.125,
        a_nu=0.5954345503010264,
        yield_blend_ratio_width=0.025,
    )

    # 1. Live 0-D ODE verification over the 4.5 Gyr time grid against pinned trajectory
    tr_4p5 = data['trajectory_calibrated_4p5Gyr']
    traj = tr_4p5['trajectory']
    t_yr = np.asarray(traj['t_yr'], dtype=float)
    ti_1d = np.asarray(traj['T_i_1d_K'], dtype=float)
    ti_0d_pinned = np.asarray(traj['T_i_0d_K'], dtype=float)
    dlid_1d = np.asarray(traj['d_lid_1d_m'], dtype=float)
    dlid_0d_pinned = np.asarray(traj['d_lid_0d_m'], dtype=float)
    ftop_1d = np.asarray(traj['F_top_1d_W_m2'], dtype=float)
    ftop_0d_pinned = np.asarray(traj['F_top_0d_W_m2'], dtype=float)

    secs_per_yr = 31557600.0
    ode_live = integrate_stagnant_lid_0d(
        float(tr_4p5['t_init_K']),
        t_yr * secs_per_yr,
        p0d,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(ode_live['T_i_K'], ti_0d_pinned, rtol=rel_pin, atol=0.0)
    np.testing.assert_allclose(ode_live['d_lid_eff_m'], dlid_0d_pinned, rtol=rel_pin, atol=0.0)
    np.testing.assert_allclose(ode_live['F_top_W_m2'], ftop_0d_pinned, rtol=rel_pin, atol=0.0)

    # Published Solomatov & Moresi (2000, Table 6 / Eq. 23) prefactor a_nu = 0.53 reference case
    p0d_sm00 = dataclasses.replace(p0d, a_nu=0.53)
    ode_sm00 = integrate_stagnant_lid_0d(
        float(tr_4p5['t_init_K']),
        t_yr * secs_per_yr,
        p0d_sm00,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    sm00_ref = prov['prefactor_comparisons_4p5Gyr']['solomatov_moresi_2000_table6_0p53']
    err_ti_sm00 = np.abs(ti_1d - ode_sm00['T_i_K']) / ode_sm00['T_i_K']
    err_dlid_sm00 = np.abs(dlid_1d - ode_sm00['d_lid_eff_m']) / ode_sm00['d_lid_eff_m']
    assert float(np.max(err_ti_sm00)) == pytest.approx(
        float(sm00_ref['max_rel_err_T_i_4p5Gyr']), rel=rel_pin
    )
    assert float(np.max(err_dlid_sm00[t_yr >= float(tr_4p5['t_relax_yr'])])) == pytest.approx(
        float(sm00_ref['max_rel_err_d_lid_post_relax']), rel=rel_pin
    )

    # 2. Pinned 4.5 Gyr trajectory deviations (exceeding 5 % T_i and 22 % d_lid tolerances)
    rel_err_ti = np.abs(ti_1d - ode_live['T_i_K']) / ode_live['T_i_K']
    rel_err_dlid = np.abs(dlid_1d - ode_live['d_lid_eff_m']) / ode_live['d_lid_eff_m']
    mask_post_relax = t_yr >= float(tr_4p5['t_relax_yr'])
    mask_25_400myr = (t_yr >= 2.5e7) & (t_yr <= 4.0e8)
    mask_valid_nu = t_yr < float(tr_4p5['t_nu_0d_clamp_1_yr'])
    mask_to_clamp = t_yr <= float(tr_4p5['t_nu_0d_clamp_1_yr'])

    max_err_ti_4p5 = float(np.max(rel_err_ti))
    max_err_dlid_post = float(np.max(rel_err_dlid[mask_post_relax]))
    max_err_dlid_all = float(np.max(rel_err_dlid))
    assert max_err_ti_4p5 == pytest.approx(0.06671792832545187, rel=rel_pin)
    assert float(tr_4p5['max_rel_err_T_i_valid_nu_ge_1']) == pytest.approx(
        float(np.max(rel_err_ti[mask_valid_nu])), rel=rel_pin
    )
    assert float(tr_4p5['max_rel_err_T_i_valid_nu_ge_1']) == pytest.approx(
        0.05339733875471053, rel=rel_pin
    )
    assert float(np.max(rel_err_ti[mask_to_clamp])) == pytest.approx(
        0.05430943733844024, rel=rel_pin
    )
    assert float(tr_4p5['t_cross_rel_err_T_i_0p05_yr']) == pytest.approx(
        3323877118.8670225, rel=rel_pin
    )
    assert max_err_dlid_post == pytest.approx(0.4732278443438256, rel=rel_pin)
    assert max_err_dlid_all == pytest.approx(1.0, rel=rel_pin)
    assert float(np.max(rel_err_ti[mask_25_400myr])) == pytest.approx(
        0.03601152242691059, rel=rel_pin
    )
    assert float(np.max(rel_err_dlid[mask_25_400myr])) == pytest.approx(
        0.21720708687458298, rel=rel_pin
    )
    assert float(tr_4p5['t_cross_rel_err_d_lid_0p22_yr']) == pytest.approx(
        408352329.28442734, rel=rel_pin
    )
    assert float(ftop_1d[-1] / ode_live['F_top_W_m2'][-1]) == pytest.approx(
        2.170113464492712, rel=rel_pin
    )

    tr_def_4p5 = data['trajectory_default_4p5Gyr']
    assert float(tr_def_4p5['max_rel_err_T_i_all']) == pytest.approx(
        0.05861057437471298, rel=rel_pin
    )
    assert float(tr_def_4p5['max_rel_err_d_lid_post_relax']) == pytest.approx(
        0.45685389428887524, rel=rel_pin
    )

    # 3. Live 1-D EntropySolver RHS at 4.5 Gyr and live 25 Myr trajectory integration
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p_cfg:
        base_params = Parameters.from_file(p_cfg)

    rheo_cal = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        s_top_inf=0.25,
        c_theta=0.25,
        yield_stress_c=500.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
    )
    params_1d = dataclasses.replace(
        base_params,
        boundary_conditions=dataclasses.replace(
            base_params.boundary_conditions,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            core_bc='quasi_steady',
            outer_boundary_condition=1,
            equilibrium_temperature=p0d.t_surf,
            emissivity=1.0,
        ),
        solver=dataclasses.replace(
            base_params.solver,
            start_time=0.0,
            end_time=4.5e9,
            atol=1.0e-8,
            rtol=1.0e-8,
            tsurf_poststep_change=100.0,
        ),
        mesh=dataclasses.replace(base_params.mesh, number_of_nodes=100, mass_coordinates=False),
        energy=dataclasses.replace(base_params.energy, kappah_floor=0.0),
        phase_solid=dataclasses.replace(
            base_params.phase_solid, viscosity=p0d.viscosity_solid, rheology=rheo_cal
        ),
        phase_mixed=dataclasses.replace(
            base_params.phase_mixed,
            const_properties=True,
            const_rho=p0d.rho,
            const_Cp=float(base_params.phase_solid.heat_capacity),
            const_alpha=float(base_params.phase_solid.thermal_expansivity),
            const_cond=float(base_params.phase_solid.thermal_conductivity),
        ),
    )
    sol_live = EntropySolver(params_1d, entropy_eos=None)
    sol_live.initialize()
    for ph in (sol_live.state.phase_basic, sol_live.state.phase_staggered):
        ph.pressure[:] = p0d.p_i_ref

    s_last_4p5 = np.asarray(tr_4p5['solver_last_S'], dtype=float)
    rhs_live = sol_live._dSdt_single(4.5e9, s_last_4p5)
    jconv_max_live = float(np.max(sol_live.state._jconv))
    snap_128 = tr_4p5['states_1d_snapshots']['128']
    assert float(sol_live.state._lid_state['T_i']) == pytest.approx(
        float(snap_128['T_i_K']), rel=rel_pin
    )
    assert float(sol_live.state._lid_state['d_lid']) == pytest.approx(
        float(snap_128['d_lid_pass1_m']), rel=rel_pin
    )
    assert float(sol_live.state._heat_flux[-1]) == pytest.approx(
        float(snap_128['F_top_W_m2']), rel=rel_pin
    )
    live_rel_err_ti_end = abs(
        float(sol_live.state._lid_state['T_i']) - float(ode_live['T_i_K'][-1])
    ) / float(ode_live['T_i_K'][-1])
    assert live_rel_err_ti_end == pytest.approx(0.06671792832545187, rel=rel_pin)

    # Short live 1-D integration over the first 25 Myr (17 output points) vs pinned trajectory
    seg_spread = data['segment_25Myr_spread_5starts']
    tol_seg = float(seg_spread['tolerance_rel_used'])
    assert float(seg_spread['max_rel_T_i']) < tol_seg
    assert float(seg_spread['max_rel_d_lid_post0']) < tol_seg
    assert float(seg_spread['max_rel_F_top']) < tol_seg

    params_25m = dataclasses.replace(
        params_1d,
        solver=dataclasses.replace(params_1d.solver, start_time=0.0, end_time=25.0e6),
    )
    sol_25m = EntropySolver(params_25m, entropy_eos=None)
    sol_25m._cvode_output_points = int(seg_spread['n_output_points'])
    sol_25m.initialize()
    for ph in (sol_25m.state.phase_basic, sol_25m.state.phase_staggered):
        ph.pressure[:] = p0d.p_i_ref
    sol_25m.set_initial_entropy(float(3000.0 + p0d.c_p * np.log(1800.0 / 3500.0)))
    orig_cv_25m = sol_25m._solve_cvode
    sol_25m._solve_cvode = lambda *a, **kw: orig_cv_25m(*a, **{**kw, 'max_step': np.inf})
    sol_25m.solve()
    assert sol_25m.solution.energy_integrals is not None
    t_25m = np.asarray(sol_25m.solution.t, dtype=float)
    y_25m = np.asarray(sol_25m.solution.y, dtype=float)
    ti_25m, dlid_25m, ftop_25m = [], [], []
    for j in range(len(t_25m)):
        sol_25m._dSdt_single(float(t_25m[j]), y_25m[:, j])
        ti_25m.append(float(sol_25m.state._lid_state['T_i']))
        dlid_25m.append(float(sol_25m.state._lid_state['d_lid']))
        ftop_25m.append(float(sol_25m.state._heat_flux[-1]))
    tr_400 = data['trajectory_calibrated_400Myr']['trajectory']
    np.testing.assert_allclose(
        t_25m, np.asarray(tr_400['t_yr'][:17], dtype=float), rtol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(ti_25m), np.asarray(tr_400['T_i_1d_K'][:17], dtype=float), rtol=tol_seg
    )
    np.testing.assert_allclose(
        np.asarray(dlid_25m[1:]),
        np.asarray(tr_400['d_lid_1d_m'][1:17], dtype=float),
        rtol=tol_seg,
    )
    np.testing.assert_allclose(
        np.asarray(ftop_25m),
        np.asarray(tr_400['F_top_1d_W_m2'][:17], dtype=float),
        rtol=tol_seg,
    )

    # 4. Pinned and live (E300, N = 100) steady-state yield-ceiling thinning ratios at (0.25, 0.25)
    steady_thinning = data['steady_yield_thinning']
    fixed_h_cases = steady_thinning['e300_n100_fixed_h']['cases']
    matched_ti_cases = steady_thinning['e300_n100_matched_ti']['cases']
    tau_d_uny_1d = float(steady_thinning['e300_n100_fixed_h']['tau_d_lid_unyielded_Pa'])
    assert tau_d_uny_1d == pytest.approx(1956123.3776120436, rel=rel_pin)
    assert float(steady_thinning['e300_n100_matched_ti']['T_i_lid_target_K']) == pytest.approx(
        1666.4195750648337, rel=rel_pin
    )

    # Live 1-D yielded state evaluations on E300 (N = 100) and s_ref_51 (1601.47 K)
    live_vecs = data['live_yield_states']['e300_n100_steady_vectors']

    def _eval_live_e300_state(
        tau_y_pa: float, h_scale: float, f_top_bc: float, s_vec_in: list[float]
    ) -> dict[str, float]:
        d_m = 1.0e6
        r_out_m = 100.0 * d_m
        r_in_m = r_out_m - d_m
        rho_m, cp_m, k_m, alpha_m, g_m = 4000.0, 100000.0, 400.0, 3.0e-5, 10.0
        t_s_m, dt_m = 273.0, 1544.0
        ti_nom_m = t_s_m + dt_m
        e_act_m = 300.0e3
        ra_m, nu_ref_m = 1.49e7, 3.40
        kappa_m = k_m / (rho_m * cp_m)
        eta_i_m = alpha_m * rho_m * g_m * dt_m * (d_m**3) / (kappa_m * ra_m)
        h_heat_m = float((nu_ref_m * dt_m * k_m / (rho_m * (d_m**2))) * h_scale)
        g_hom_m = e_act_m / (_R * ti_nom_m)
        rheo_m = SolidRheologyParams(
            enabled=True,
            g_homologous=g_hom_m,
            t_sol_0=ti_nom_m,
            dt_sol_dp=0.0,
            arrhenius_t_ref=ti_nom_m,
            s_top_inf=0.25,
            c_theta=0.25,
            mlt_bottom_slope=1.0,
            yield_stress_c=float(tau_y_pa),
            yield_stress_mu=0.0,
            yield_stress_max=float(tau_y_pa),
            stress_closure_mode='lid',
        )
        p_e300 = Parameters(
            boundary_conditions=_BoundaryConditionsParameters(
                outer_boundary_condition=4,
                outer_boundary_value=float(f_top_bc),
                inner_boundary_condition=2,
                inner_boundary_value=0.0,
                emissivity=1.0,
                equilibrium_temperature=t_s_m,
                core_heat_capacity=880.0,
                core_bc='quasi_steady',
            ),
            energy=_EnergyParameters(
                conduction=True,
                convection=True,
                gravitational_separation=False,
                mixing=False,
                radionuclides=True,
                tidal=False,
                solver_method='cvode',
                use_jax_jacobian=False,
                kappah_floor=0.0,
            ),
            initial_condition=_InitialConditionParameters(
                initial_condition=1, surface_temperature=ti_nom_m, basal_temperature=ti_nom_m
            ),
            mesh=_MeshParameters(
                outer_radius=r_out_m,
                inner_radius=r_in_m,
                number_of_nodes=100,
                mixing_length_profile='nearest_boundary',
                core_density=rho_m,
                eos_method=1,
                surface_density=rho_m,
                gravitational_acceleration=g_m,
                mass_coordinates=False,
            ),
            phase_liquid=_PhaseParameters(
                melt_fraction=1.0,
                viscosity=100.0,
                density=rho_m,
                heat_capacity=cp_m,
                thermal_conductivity=k_m,
                thermal_expansivity=alpha_m,
            ),
            phase_solid=_PhaseParameters(
                melt_fraction=0.0,
                viscosity=eta_i_m,
                rheology=rheo_m,
                density=rho_m,
                heat_capacity=cp_m,
                thermal_conductivity=k_m,
                thermal_expansivity=alpha_m,
            ),
            phase_mixed=_PhaseMixedParameters(
                latent_heat_of_fusion=4.0e5,
                rheological_transition_melt_fraction=0.4,
                rheological_transition_width=0.15,
                solidus='solidus.dat',
                liquidus='liquidus.dat',
                phase='mixed',
                phase_transition_width=0.01,
                grain_size=1.0e-3,
                matprop_smooth_width=0.01,
                const_properties=True,
                const_rho=rho_m,
                const_Cp=cp_m,
                const_alpha=alpha_m,
                const_cond=k_m,
                const_log10visc=float(np.log10(eta_i_m)),
                const_T_ref=ti_nom_m,
                const_S_ref=3000.0,
            ),
            radionuclides=[
                _Radionuclide(
                    name='const',
                    t0_years=0.0,
                    abundance=1.0,
                    concentration=1.0e6,
                    heat_production=h_heat_m,
                    half_life_years=1.0e30,
                )
            ],
            solver=_SolverParameters(
                start_time=0.0,
                end_time=4.0e10,
                atol=1.0e-7,
                rtol=1.0e-7,
                tsurf_poststep_change=30.0,
            ),
        )
        sol_e = EntropySolver(p_e300, entropy_eos=None)
        sol_e.initialize()
        rhs_e = np.asarray(
            sol_e._dSdt_single(4.0e10, np.asarray(s_vec_in, dtype=float)), dtype=float
        )
        st_e = sol_e.state
        lid_e = st_e._lid_state
        w_lid_e = np.asarray(lid_e['w_lid'], dtype=float).ravel()
        eta_eff_e = np.asarray(st_e._eta_eff_basic, dtype=float).ravel()
        eta_diff_e = np.asarray(st_e.phase_basic.eta_diff(), dtype=float).ravel()
        eta_eff_lid_min_e = float(np.min(eta_eff_e[w_lid_e >= 0.5]))
        eta_i_lid_e = float(lid_e['eta_i'])
        n_yield_e = int(np.sum(eta_eff_e < 0.99 * eta_diff_e))
        t_cell_top = float(np.asarray(st_e.phase_staggered.temperature()).ravel()[-1])
        r_b_e = np.asarray(sol_e.evaluator.mesh.basic.radii).ravel()
        f_top_e = float(k_m * (t_cell_top - t_s_m) / (0.5 * float(r_b_e[-1] - r_b_e[-2])))
        theta_e = float(lid_e['theta'])
        ti_e = float(lid_e['T_i'])
        lid_frac_e = max(1.0 - 2.2 / theta_e, 0.0)
        d_lid_cond_e = float(lid_frac_e * k_m * (ti_e - t_s_m) / f_top_e)
        return {
            'd_lid_pass1_m': float(lid_e['d_lid']),
            'd_lid_cond_lid_m': d_lid_cond_e,
            'eta_eff_lid_over_eta_i': eta_eff_lid_min_e / eta_i_lid_e,
            'n_yielding_basic_nodes': float(n_yield_e),
            'max_abs_rhs': float(np.max(np.abs(rhs_e))),
        }

    live_u = _eval_live_e300_state(
        500.0e6,
        1.0,
        float(live_vecs['unyielded_500MPa']['F_top_W_m2']),
        live_vecs['unyielded_500MPa']['S_steady_J_kg_K'],
    )
    live_fh05 = _eval_live_e300_state(
        0.5 * tau_d_uny_1d,
        1.0,
        float(live_vecs['fixed_h_below_0p5x_tau_d_lid']['F_top_W_m2']),
        live_vecs['fixed_h_below_0p5x_tau_d_lid']['S_steady_J_kg_K'],
    )
    live_mt05 = _eval_live_e300_state(
        0.5 * tau_d_uny_1d,
        float(live_vecs['matched_ti_below_0p5x_tau_d_lid']['h_scale']),
        float(live_vecs['matched_ti_below_0p5x_tau_d_lid']['F_top_W_m2']),
        live_vecs['matched_ti_below_0p5x_tau_d_lid']['S_steady_J_kg_K'],
    )
    assert live_u['n_yielding_basic_nodes'] == 0.0
    assert live_fh05['n_yielding_basic_nodes'] >= 40.0
    assert live_mt05['n_yielding_basic_nodes'] >= 30.0
    assert live_u['eta_eff_lid_over_eta_i'] == pytest.approx(15.6470507167325, rel=rel_pin)
    assert live_fh05['eta_eff_lid_over_eta_i'] == pytest.approx(0.5029920741267555, rel=rel_pin)
    assert live_mt05['eta_eff_lid_over_eta_i'] == pytest.approx(0.5232816199771473, rel=rel_pin)
    assert (live_fh05['d_lid_pass1_m'] / live_u['d_lid_pass1_m']) == pytest.approx(
        1.5259308431784937, rel=rel_pin
    )
    assert (live_fh05['d_lid_cond_lid_m'] / live_u['d_lid_cond_lid_m']) == pytest.approx(
        0.9281772785559389, rel=rel_pin
    )
    assert (live_mt05['d_lid_pass1_m'] / live_u['d_lid_pass1_m']) == pytest.approx(
        1.0466365056333917, rel=rel_pin
    )
    assert (live_mt05['d_lid_cond_lid_m'] / live_u['d_lid_cond_lid_m']) == pytest.approx(
        0.609581859649797, rel=rel_pin
    )
    # Single-RHS steady-state residual bound set at 100x the measured 5-start maximum
    # (max_over_5starts = 2.595e-12, 5.754e-13, 2.225e-15 J/(kg K yr); <= 1.0e-9 J/(kg K yr)).
    s5_e300 = steady_thinning['e300_n100_five_start_spread']['five_start_steady_spread']
    assert (
        live_u['max_abs_rhs']
        <= 100.0 * float(s5_e300['unyielded_500MPa']['max_over_5starts_max_abs_rhs_J_kg_K_yr'])
        <= 1.0e-9
    )
    assert (
        live_fh05['max_abs_rhs']
        <= 100.0
        * float(
            s5_e300['fixed_h_below_0p5x_tau_d_lid']['max_over_5starts_max_abs_rhs_J_kg_K_yr']
        )
        <= 1.0e-9
    )
    assert (
        live_mt05['max_abs_rhs']
        <= 100.0
        * float(
            s5_e300['matched_ti_below_0p5x_tau_d_lid']['max_over_5starts_max_abs_rhs_J_kg_K_yr']
        )
        <= 1.0e-9
    )

    # Also verify live spherical cooling state s_ref_51 (T_i = 1601.47 K) at 0.5 * tau_d_lid
    s_ref_51 = np.asarray(data['live_yield_states']['s_ref_51']['S_ref_J_kg_K'], dtype=float)
    t_ref_51 = float(data['live_yield_states']['s_ref_51']['t_yr'])
    tau_d_ref_51 = float(data['yield_sweep_reference_state']['tau_d_lid_Pa'])
    sol_y51 = EntropySolver(
        dataclasses.replace(
            params_1d,
            phase_solid=dataclasses.replace(
                params_1d.phase_solid,
                rheology=dataclasses.replace(
                    rheo_cal,
                    yield_stress_c=0.5 * tau_d_ref_51,
                    yield_stress_mu=0.0,
                    yield_stress_max=0.5 * tau_d_ref_51,
                ),
            ),
        ),
        entropy_eos=None,
    )
    sol_y51.initialize()
    for ph in (sol_y51.state.phase_basic, sol_y51.state.phase_staggered):
        ph.pressure[:] = p0d.p_i_ref
    sol_y51._dSdt_single(t_ref_51, s_ref_51)
    w_lid_51 = np.asarray(sol_y51.state._lid_state['w_lid'], dtype=float).ravel()
    eta_eff_51 = np.asarray(sol_y51.state._eta_eff_basic, dtype=float).ravel()
    eta_diff_51 = np.asarray(sol_y51.state.phase_basic.eta_diff(), dtype=float).ravel()
    eta_i_51 = float(sol_y51.state._lid_state['eta_i'])
    assert int(np.sum(eta_eff_51 < 0.99 * eta_diff_51)) == 11
    assert int(np.sum((eta_eff_51 < 0.99 * eta_diff_51) & (w_lid_51 >= 0.5))) == 9
    assert float(np.min(eta_eff_51[w_lid_51 >= 0.5]) / eta_i_51) == pytest.approx(
        0.5, rel=1.0e-5
    )

    # Live 0-D thinning check across the 5 yield multipliers at T_i = 1666.419575 K
    ev_uny_0d = evaluate_stagnant_lid_0d(1666.4195750648337, p0d)
    tau_d_0d_uny = float(ev_uny_0d['tau_d_lid_Pa'])
    for key, mult, exp_r0d in (
        ('above_2p0x_tau_d_lid', 2.0, 1.0),
        ('below_0p5x_tau_d_lid', 0.5, 0.7937005259840998),
        ('below_0p2x_tau_d_lid', 0.2, 0.5848035476425732),
        ('below_0p1x_tau_d_lid', 0.1, 0.4641588833612779),
    ):
        p0d_y = dataclasses.replace(
            p0d,
            yield_stress_c=mult * tau_d_0d_uny,
            yield_stress_mu=0.0,
            yield_stress_max=mult * tau_d_0d_uny,
        )
        ev_y = evaluate_stagnant_lid_0d(1666.4195750648337, p0d_y)
        r_live_0d = float(ev_y['d_lid_eff_m'] / ev_uny_0d['d_lid_eff_m'])
        assert r_live_0d == pytest.approx(exp_r0d, rel=1.0e-12)
        assert float(fixed_h_cases[key]['r_thin_0d_fixed_Ti']) == pytest.approx(
            exp_r0d, rel=rel_pin
        )

    # Fixed internal heating ratios: Pass-1 isotherm thickens (1.5259 / 1.7397 / 1.9118);
    # post-hoc conductive diagnostic thins weakly (0.9282 / 0.9022 / 0.8809) as interior cools
    for key, exp_p1, exp_cond, exp_ti in (
        ('unyielded_500MPa', 1.0, 1.0, 1666.4195750648337),
        ('above_2p0x_tau_d_lid', 1.0000000840446674, 1.00000007265864, 1666.4195526054918),
        ('below_0p5x_tau_d_lid', 1.5259308431784937, 0.9281772785559389, 1557.004207586273),
        ('below_0p2x_tau_d_lid', 1.739741462006333, 0.9021715533274383, 1517.8287881051435),
        ('below_0p1x_tau_d_lid', 1.9118197399314252, 0.8809058564042608, 1485.9637158182675),
    ):
        row = fixed_h_cases[key]
        assert float(row['R_thin_pass1']) == pytest.approx(exp_p1, rel=rel_pin)
        assert float(row['R_thin_cond_lid']) == pytest.approx(exp_cond, rel=rel_pin)
        assert float(row['T_i_lid_K']) == pytest.approx(exp_ti, rel=rel_pin)

    # Matched T_i = 1666.42 K ratios: Pass-1 isotherm stays flat (1.0466 / 1.0551 / 1.0581);
    # post-hoc conductive diagnostic thins (0.6096 / 0.4980 / 0.4197)
    for key, exp_p1_m, exp_cond_m, exp_ftop_m in (
        ('unyielded_500MPa', 1.0, 1.0, 2078.9117759903147),
        ('above_2p0x_tau_d_lid', 1.0000000840446674, 1.00000007265864, 2078.9115946414036),
        ('below_0p5x_tau_d_lid', 1.0466365056333917, 0.6095818596497969, 3408.95295983677),
        ('below_0p2x_tau_d_lid', 1.0551011113837658, 0.49796454730272127, 4171.834353271644),
        ('below_0p1x_tau_d_lid', 1.0580583212570414, 0.41972455470298903, 4947.857676673418),
    ):
        row_m = matched_ti_cases[key]
        assert float(row_m['R_thin_pass1']) == pytest.approx(exp_p1_m, rel=rel_pin)
        assert float(row_m['R_thin_cond_lid']) == pytest.approx(exp_cond_m, rel=rel_pin)
        assert float(row_m['F_top_mW_m2']) == pytest.approx(exp_ftop_m, rel=rel_pin)

    # 5. Canaries: asserting the reference bounds or perturbing either code path fails
    with pytest.raises(AssertionError):
        assert max_err_ti_4p5 <= float(tols['tol_rel_T_i'])
    with pytest.raises(AssertionError):
        assert max_err_dlid_post <= float(tols['tol_rel_d_lid'])
    with pytest.raises(AssertionError):
        assert float(
            fixed_h_cases['below_0p5x_tau_d_lid']['rel_err_pass1_vs_0d_fixed_Ti']
        ) <= float(tols['tol_rel_R_thin'])
    with pytest.raises(AssertionError):
        assert float(
            matched_ti_cases['below_0p5x_tau_d_lid']['rel_err_pass1_vs_0d_fixed_Ti']
        ) <= float(tols['tol_rel_R_thin'])

    p0d_perturbed = dataclasses.replace(p0d, a_nu=p0d.a_nu * 1.01)
    ode_perturbed = integrate_stagnant_lid_0d(
        float(tr_4p5['t_init_K']),
        t_yr * secs_per_yr,
        p0d_perturbed,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(ode_perturbed['T_i_K'], ti_0d_pinned, rtol=rel_pin, atol=0.0)

    sol_perturbed = EntropySolver(
        dataclasses.replace(
            params_1d,
            phase_solid=dataclasses.replace(
                params_1d.phase_solid,
                rheology=dataclasses.replace(rheo_cal, s_top_inf=0.26, lid_contrast_coeff=2.3),
            ),
        ),
        entropy_eos=None,
    )
    sol_perturbed.initialize()
    for ph in (sol_perturbed.state.phase_basic, sol_perturbed.state.phase_staggered):
        ph.pressure[:] = p0d.p_i_ref
    rhs_perturbed = sol_perturbed._dSdt_single(4.5e9, s_last_4p5)
    with pytest.raises(AssertionError):
        assert float(sol_perturbed.state._lid_state['d_lid']) == pytest.approx(
            float(snap_128['d_lid_pass1_m']), rel=rel_pin
        )
    with pytest.raises(AssertionError):
        assert float(np.max(sol_perturbed.state._jconv)) == pytest.approx(
            jconv_max_live, rel=rel_pin
        )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(rhs_perturbed, rhs_live, rtol=rel_pin, atol=0.0)

    def figure():
        import matplotlib.pyplot as plt

        idx = np.unique(np.append(np.arange(0, len(t_yr), 16), len(t_yr) - 1))
        t_gyr = t_yr[idx] / 1.0e9
        order = [
            'below_0p1x_tau_d_lid',
            'below_0p2x_tau_d_lid',
            'below_0p5x_tau_d_lid',
            'above_2p0x_tau_d_lid',
        ]
        log_mult = np.log10(np.array([0.1, 0.2, 0.5, 2.0]))
        r_p1_fh = np.array([float(fixed_h_cases[k]['R_thin_pass1']) for k in order])
        r_cd_fh = np.array([float(fixed_h_cases[k]['R_thin_cond_lid']) for k in order])
        r_p1_mt = np.array([float(matched_ti_cases[k]['R_thin_pass1']) for k in order])
        r_cd_mt = np.array([float(matched_ti_cases[k]['R_thin_cond_lid']) for k in order])
        r_0d = np.array([float(matched_ti_cases[k]['r_thin_0d_fixed_Ti']) for k in order])

        plt.rcParams.update(
            {
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 6.5,
                'legend.title_fontsize': 6.5,
                'axes.grid': False,
            }
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.9))
        ax1b = ax1.twinx()
        ax1b.spines['right'].set_visible(True)
        cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']

        l1 = ax1.plot(t_gyr, ti_1d[idx], '-', color=cyc[0], label='1D T_i')
        l2 = ax1.plot(t_gyr, ti_0d_pinned[idx], '--', color=cyc[1], label='0D T_i')
        l3 = ax1b.plot(t_gyr, dlid_1d[idx] / 1.0e3, '-.', color=cyc[4], label='1D d_lid')
        l4 = ax1b.plot(t_gyr, dlid_0d_pinned[idx] / 1.0e3, ':', color=cyc[2], label='0D d_lid')
        ax1.set(
            xlabel='(a) t [Gyr]',
            ylabel='T_i [K]',
            xlim=(0.0, 4.5),
            ylim=(1200, 1850),
            xticks=[0, 2, 4],
            yticks=[1200, 1800],
        )
        ax1b.set(
            ylabel='d_lid [km]',
            ylim=(0, 950),
            yticks=[0, 400, 800],
        )
        lines = l1 + l2 + l3 + l4
        ax1.legend(
            lines,
            [ln.get_label() for ln in lines],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.24),
            ncol=2,
        )

        ax2.plot(log_mult, r_p1_fh, '-', label='isotherm lid, fixed heating')
        ax2.plot(log_mult, r_p1_mt, '--', label='isotherm lid, matched T_i')
        ax2.plot(log_mult, r_cd_fh, '-.', label='conductive lid*, fixed heating')
        ax2.plot(log_mult, r_cd_mt, ':', label='conductive lid*, matched T_i')
        ax2.plot(log_mult, r_0d, '-', label='0D cube-root law')
        ax2.set(
            xlabel='(b) log10(tau_y/tau_d)',
            ylabel='R_thin',
            ylim=(0.35, 2.05),
            xticks=[-1, 0],
            yticks=[0.5, 1, 2],
        )
        ax2.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.24),
            title='* diagnostic chosen after the result',
        )

        fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.42, wspace=0.42)
        return fig

    validation_plot(figure, 'ssc_0d_reference_comparison')
