"""Generate and verify reference benchmarks for internal-heating stagnant-lid scaling tests.

Generates or checks ``tests/reference/ssc_wp6_a6_2_benchmarks.json``.

Execution modes and per-block coverage:
- ``--full`` (~50 s on 1 CPU core): recomputes the steady-state trajectories, snapshots,
  conduction floors, and OLS fits using ``EntropySolver`` (temporarily replacing
  ``EntropySolver._solve_cvode`` with an unbounded-step wrapper ``max_step = np.inf`` to
  integrate to the multi-Gyr steady states without hitting the 100 yr phase-boundary
  step cap; this override is removed once the step-cap fix lands on ``main``):
  1. ``metadata.fixed_parameters`` (~0.01 s): derives the analytical and discrete
     ``N = 100`` staggered-grid pure-conduction Nusselt floors (parabola peak ``2.0``,
     analytical Korenaga ``1 + sqrt(6)/2 = 2.224745``, planar ``N = 100`` ``2.227638``,
     and spherical-shell ``r_in/r_out = 0.99``, ``N = 100`` ``2.247151``).
  2. ``cases_results`` and ``steady_state_snapshots`` (~50 s): integrates all 12
     internally heated stagnant-lid cases (``case_01``..``case_12``) to statistical
     steady state in 8-chunk doublings (``t_end`` up to ``320 Gyr``) and extracts the
     12 steady-state result rows and 5 full steady-state profile snapshots
     (``case_01``, ``case_03``, ``case_06``, ``case_07``, ``case_12``).
  3. ``primary_fit_all_12_nu_korenaga`` and ``realized_state_fits`` (~0.01 s): computes
     the ordinary least-squares multilinear and 1-D sweep power-law fits on both design
     parameters ``(Ra_i, theta)`` and realized steady-state parameters
     ``(Ra_i_realized_kor, theta_realized_kor)``.
- ``--check`` (~0.25 s on 1 CPU core): verifies expected schema keys and the no-table
  constant-property record (``eos_tables``), and recomputes a stated fast live subset
  across every block:
  1. ``metadata.fixed_parameters``: recomputes ``compute_discrete_conduction_floor_n100()``
     and verifies agreement within ``1e-12``.
  2. ``cases_results`` and ``steady_state_snapshots``: evaluates live 1-D
     ``EntropySolver._dSdt_single`` and ``korenaga_ti_fixed_point`` on all 5 stored
     steady-state snapshots (``case_01``, ``case_03``, ``case_06``, ``case_07``,
     ``case_12``), verifying ``max_abs_rhs_J_kg_K_yr <= 1e-9`` and agreement of
     ``rhs_final``, ``F_top_W_m2``, ``jconv_max_W_m2``, ``T_i_korenaga_K``,
     ``Nu_korenaga``, and ``_lid_state`` observables within ``1e-11``.
  3. ``primary_fit_all_12_nu_korenaga`` and ``realized_state_fits``: recomputes the
     design-parameter and realized-state OLS fits from ``cases_results`` and verifies
     agreement within ``1e-12``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

from aragog.parser import (  # noqa: E402
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
from aragog.rheology import R_GAS, SolidRheologyParams, compute_mlt_top_slope  # noqa: E402
from aragog.solver import entropy_solver as es_mod  # noqa: E402
from aragog.solver import entropy_state as estate_mod  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402

DEFAULT_BENCHMARK_PATH = REPO_ROOT / 'tests' / 'reference' / 'ssc_wp6_a6_2_benchmarks.json'
NO_TABLE_RECORD = (
    'no table (constant properties: eos_method=1, const_properties=True, entropy_eos=None)'
)
REQUIRED_SNAPSHOT_IDS = ('case_01', 'case_03', 'case_06', 'case_07', 'case_12')
STEADY_RESIDUAL_BOUND = 1.0e-9

EXPECTED_TOP_KEYS = (
    'metadata',
    'cases_spec',
    'cases_results',
    'primary_fit_all_12_nu_korenaga',
    'diagnostic_fits',
    'steady_state_snapshots',
    'realized_state_fits',
)
EXPECTED_META_KEYS = (
    'git_commit',
    'eos_tables',
    'rheology_calibration',
    'acceptance_window',
    'fixed_parameters',
)
EXPECTED_SPEC_KEYS = (
    'case_id',
    'case_num',
    'name',
    'sweep',
    'target_Ra_i',
    'target_theta',
    'E_activation_J_per_mol',
    'eta_i_Pa_s',
    'H_W_per_kg',
    'Nu_estimated',
    'T_surf_K',
    'T_i_nominal_K',
    'delta_T_nominal_K',
    'D_m',
    'R_out_m',
    'R_in_m',
    'rho_kg_per_m3',
    'g_m_per_s2',
    'alpha_thermal_per_K',
    'k_thermal_W_per_m_K',
    'cp_J_per_kg_K',
    'kappa_m2_per_s',
    'cp_scale_F',
    'inner_boundary_condition',
    'outer_boundary_condition',
)
EXPECTED_DIAG_FIT_KEYS = (
    'all_12_cases_nu_flux_kor',
    'all_12_cases_nu_flux_tmax',
    'all_12_cases_nu_dth_lid',
    'convecting_7_cases_nu_korenaga',
    'convecting_7_cases_nu_flux_tmax',
    'convecting_7_cases_nu_dth_lid',
)
EXPECTED_REALIZED_FIT_KEYS = (
    'regressor_definition',
    'all_12_nu_korenaga_on_realized_kor',
    'convecting_7_nu_korenaga_on_realized_kor',
)


def korenaga_ti_fixed_point(
    t_stag: np.ndarray, z_norm: np.ndarray, t_surf: float, dth: float
) -> float:
    """Solve Korenaga (2009) Eq. 20 fixed point for sub-lid interior temperature ``T_i``."""
    order = np.argsort(z_norm)
    z_s, t_s_arr = z_norm[order], t_stag[order]
    ti = float(t_s_arr.max())
    for _ in range(100):
        dl = (ti - t_surf) / dth
        z_above = z_s[z_s > dl]
        if z_above.size == 0:
            raise RuntimeError(
                'Korenaga fixed-point iteration did not converge: lid exceeds domain'
            )
        zz = np.concatenate([[dl], z_above])
        if zz[-1] <= dl:
            raise RuntimeError(
                'Korenaga fixed-point iteration did not converge: zero sub-lid span'
            )
        tt = np.interp(zz, z_s, t_s_arr)
        ti_new = float(np.trapezoid(tt, zz) / (zz[-1] - dl))
        if abs(ti_new - ti) < 1.0e-9:
            return ti_new
        ti = ti_new
    raise RuntimeError('Korenaga fixed-point iteration did not converge within 100 steps')


def _ols(x_cols: list[np.ndarray], y_vec: np.ndarray) -> dict[str, Any]:
    """Fit ``y = c_0 + sum_j c_j x_j`` by ordinary least squares with standard errors."""
    n = len(y_vec)
    x_mat = np.column_stack([np.ones(n), *x_cols])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    resid = y_vec - x_mat @ coeffs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    dof = max(n - x_mat.shape[1], 1)
    cov = (ss_res / dof) * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    if len(x_cols) == 2:
        ln_a, alpha, beta = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return {
            'a': float(np.exp(ln_a)),
            'ln_a': ln_a,
            'alpha': alpha,
            'beta': beta,
            'std_err_ln_a': float(se[0]),
            'std_err_alpha': float(se[1]),
            'std_err_beta': float(se[2]),
            'r_squared': r2,
            'n_cases': int(n),
            'dof': int(dof),
        }
    return {
        'slope': float(coeffs[1]),
        'std_err_slope': float(se[1]),
        'intercept_log10': float(coeffs[0]),
        'r_squared': r2,
        'n_cases': int(n),
        'dof': int(dof),
    }


def compute_discrete_conduction_floor_n100() -> dict[str, float]:
    """Compute analytical, planar grid, and spherical-shell N=100 pure-conduction floors."""
    n_stag = 99
    z_stag = (np.arange(n_stag, dtype=float) + 0.5) / float(n_stag)
    z_desc = z_stag[::-1]
    t_norm = z_desc * (1.0 - 0.5 * z_desc)

    def _kor_fixed(t_arr: np.ndarray, z_arr: np.ndarray) -> float:
        order = np.argsort(z_arr)
        z_s, t_s = z_arr[order], t_arr[order]
        ti = float(t_s.max())
        for _ in range(100):
            dl = ti
            zz = np.concatenate([[dl], z_s[z_s > dl]])
            tt = np.interp(zz, z_s, t_s)
            ti_new = float(np.trapezoid(tt, zz) / (zz[-1] - dl))
            if abs(ti_new - ti) < 1.0e-14:
                return ti_new
            ti = ti_new
        raise RuntimeError('Conduction floor fixed point did not converge')

    ti_planar_n100 = _kor_fixed(t_norm, z_desc)

    r_b = np.linspace(99.0, 100.0, 100)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])
    dr = 1.0 / 99.0
    f_top_norm = (100.0**3 - 99.0**3) / (3.0 * 100.0**2)
    t_sph = np.zeros(99)
    t_sph[-1] = 0.5 * dr * f_top_norm
    for i in range(98, 0, -1):
        f_i = (r_b[i] ** 3 - 99.0**3) / (3.0 * r_b[i] ** 2)
        t_sph[i - 1] = t_sph[i] + dr * f_i
    ti_sph_n100 = _kor_fixed(t_sph, 100.0 - r_s)

    dl_analytic = float(np.sqrt(6.0) - 2.0)
    return {
        'Nu_analytic_parabola_peak': 2.0,
        'Nu_analytic_volume_mean': 3.0,
        'dl_korenaga_analytic_planar': dl_analytic,
        'Nu_korenaga_analytic_planar': float(1.0 + 0.5 * np.sqrt(6.0)),
        'T_i_korenaga_norm_planar_N100': ti_planar_n100,
        'Nu_korenaga_conduction_floor_planar_N100': float(1.0 / ti_planar_n100),
        'T_i_korenaga_norm_spherical_N100': ti_sph_n100,
        'Nu_korenaga_conduction_floor_spherical_N100': float(1.0 / ti_sph_n100),
    }


def build_scaling_case_params(spec: dict[str, Any]) -> Parameters:
    """Construct ``Parameters`` for a single planar internal-heating scaling case."""
    rho = float(spec.get('rho_kg_per_m3', 4000.0))
    cp = float(spec.get('cp_J_per_kg_K', 100000.0))
    k_cond = float(spec.get('k_thermal_W_per_m_K', 400.0))
    alpha = float(spec.get('alpha_thermal_per_K', 3.0e-5))
    grav = float(spec.get('g_m_per_s2', 10.0))
    t_surf = float(spec['T_surf_K'])
    t_i_nom = float(spec['T_i_nominal_K'])
    eta_i = float(spec['eta_i_Pa_s'])
    e_act = float(spec['E_activation_J_per_mol'])
    h_heat = float(spec['H_W_per_kg'])
    r_out = float(spec['R_out_m'])
    r_in = float(spec['R_in_m'])

    rheo = SolidRheologyParams(
        enabled=True,
        s_top_inf=0.25,
        c_theta=0.25,
        mlt_bottom_slope=1.0,
        activation_energy=e_act,
        activation_volume=0.0,
        arrhenius_t_ref=t_i_nom,
        yield_stress_c=1.0e30,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e30,
        viscosity_max_log10=40.0,
        stress_closure_mode='lid',
    )
    common = dict(
        density=rho, heat_capacity=cp, thermal_conductivity=k_cond, thermal_expansivity=alpha
    )
    return Parameters(
        boundary_conditions=_BoundaryConditionsParameters(
            outer_boundary_condition=5,
            outer_boundary_value=t_surf,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=t_surf,
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
            initial_condition=1, surface_temperature=t_i_nom, basal_temperature=t_i_nom
        ),
        mesh=_MeshParameters(
            outer_radius=r_out,
            inner_radius=r_in,
            number_of_nodes=100,
            mixing_length_profile='nearest_boundary',
            core_density=rho,
            eos_method=1,
            surface_density=rho,
            gravitational_acceleration=grav,
            mass_coordinates=False,
        ),
        phase_solid=_PhaseParameters(
            melt_fraction=0.0, viscosity=eta_i, rheology=rheo, **common
        ),
        phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=100.0, **common),
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
            const_rho=rho,
            const_Cp=cp,
            const_alpha=alpha,
            const_cond=k_cond,
            const_log10visc=float(np.log10(eta_i)),
            const_T_ref=t_i_nom,
            const_S_ref=3000.0,
        ),
        radionuclides=[
            _Radionuclide(
                name='const',
                t0_years=0.0,
                abundance=1.0,
                concentration=1.0e6,
                heat_production=h_heat,
                half_life_years=1.0e30,
            )
        ],
        solver=_SolverParameters(
            start_time=0.0,
            end_time=4.0e10,
            atol=1.0e-6,
            rtol=1.0e-6,
            tsurf_poststep_change=30.0,
        ),
    )


def evaluate_live_snapshot(
    spec: dict[str, Any], s_final: np.ndarray, t_end_yr: float
) -> dict[str, Any]:
    """Evaluate 1-D ``EntropySolver._dSdt_single`` on a steady-state entropy profile."""
    params = build_scaling_case_params(spec)
    orig_update = estate_mod.EntropyState.update

    def _half_cell_update(
        self_st: Any, entropy: Any, time_val: Any, *a: Any, **kw: Any
    ) -> None:
        orig_update(self_st, entropy, time_val, *a, **kw)
        bc_set = self_st._evaluator.boundary_conditions._settings
        if int(bc_set.outer_boundary_condition) == 5:
            k_surf = float(np.asarray(self_st.phase_basic.thermal_conductivity()).ravel()[-1])
            t_cell = float(np.asarray(self_st.phase_staggered.temperature()).ravel()[-1])
            r_b = np.asarray(self_st._evaluator.mesh.basic.radii).ravel()
            dr_half = 0.5 * float(r_b[-1] - r_b[-2])
            self_st._heat_flux[-1] = (
                k_surf * (t_cell - float(bc_set.outer_boundary_value)) / dr_half
            )

    estate_mod.EntropyState.update = _half_cell_update
    try:
        solver = EntropySolver(params, entropy_eos=None)
        solver.initialize()
        rhs = np.asarray(solver._dSdt_single(float(t_end_yr), s_final), dtype=float)
        t_stag = np.asarray(solver.state.phase_staggered.temperature(), dtype=float).ravel()
        rb = np.asarray(solver.evaluator.mesh.basic.radii, dtype=float).ravel()
        rs = np.asarray(solver.evaluator.mesh.staggered.radii, dtype=float).ravel()
        k_cond = float(spec.get('k_thermal_W_per_m_K', 400.0))
        rho = float(spec.get('rho_kg_per_m3', 4000.0))
        d_m = float(spec['D_m'])
        t_surf = float(spec['T_surf_K'])
        h_heat = float(spec['H_W_per_kg'])
        dth = rho * h_heat * d_m**2 / k_cond
        f_top = float(k_cond * (t_stag[-1] - t_surf) / (rb[-1] - rs[-1]))
        ti_kor = korenaga_ti_fixed_point(t_stag, (rb[-1] - rs) / d_m, t_surf, dth)
        nu_kor = float(dth / (ti_kor - t_surf))
        lid = solver.state._lid_state
        return {
            'rhs_final': rhs,
            'max_abs_rhs_J_kg_K_yr': float(np.max(np.abs(rhs))),
            'T_stag_final': t_stag,
            'F_top_W_m2': f_top,
            'jconv_max_W_m2': float(np.max(solver.state._jconv)),
            'T_i_korenaga_K': ti_kor,
            'Nu_korenaga': nu_kor,
            'lid_state_T_i_K': float(lid['T_i']),
            'lid_state_theta': float(lid['theta']),
            'lid_state_d_lid_m': float(lid['d_lid']),
            'lid_state_has_lid': bool(lid['has_lid']),
        }
    finally:
        estate_mod.EntropyState.update = orig_update


def run_single_steady_case(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Integrate a single scaling case to statistical steady state in 8-chunk doublings."""
    cid = str(spec['case_id'])
    d_m = float(spec['D_m'])
    k_cond = float(spec.get('k_thermal_W_per_m_K', 400.0))
    rho = float(spec.get('rho_kg_per_m3', 4000.0))
    cp = float(spec.get('cp_J_per_kg_K', 100000.0))
    alpha = float(spec.get('alpha_thermal_per_K', 3.0e-5))
    grav = float(spec.get('g_m_per_s2', 10.0))
    h_heat = float(spec['H_W_per_kg'])
    t_surf = float(spec['T_surf_K'])
    t_ref = float(spec['T_i_nominal_K'])
    e_act = float(spec['E_activation_J_per_mol'])
    eta_ref = float(spec['eta_i_Pa_s'])
    kappa = k_cond / (rho * cp)
    dth = rho * h_heat * d_m**2 / k_cond
    f_geom = float(spec['R_in_m']) / float(spec['R_out_m'])
    f_top_expected = rho * h_heat * d_m * (1.0 + f_geom + f_geom**2) / 3.0

    orig_update = estate_mod.EntropyState.update
    orig_cvode = es_mod.EntropySolver._solve_cvode

    def _half_cell_update(
        self_st: Any, entropy: Any, time_val: Any, *a: Any, **kw: Any
    ) -> None:
        orig_update(self_st, entropy, time_val, *a, **kw)
        bc_set = self_st._evaluator.boundary_conditions._settings
        if int(bc_set.outer_boundary_condition) == 5:
            k_surf = float(np.asarray(self_st.phase_basic.thermal_conductivity()).ravel()[-1])
            t_cell = float(np.asarray(self_st.phase_staggered.temperature()).ravel()[-1])
            r_b = np.asarray(self_st._evaluator.mesh.basic.radii).ravel()
            dr_half = 0.5 * float(r_b[-1] - r_b[-2])
            self_st._heat_flux[-1] = (
                k_surf * (t_cell - float(bc_set.outer_boundary_value)) / dr_half
            )

    def _unbounded_cvode(self_s: Any, *a: Any, **kw: Any) -> Any:
        kw['max_step'] = np.inf
        return orig_cvode(self_s, *a, **kw)

    estate_mod.EntropyState.update = _half_cell_update
    es_mod.EntropySolver._solve_cvode = _unbounded_cvode
    try:
        chunks = 8
        for t_end in (4.0e10, 8.0e10, 1.6e11, 3.2e11):
            params = build_scaling_case_params(spec)
            solver = EntropySolver(params, entropy_eos=None)
            solver.initialize()
            solver.set_initial_entropy(3000.0)
            edges = np.linspace(0.0, t_end, chunks + 1)
            hist: list[tuple[float, float, float]] = []
            st = None
            for chunk_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                params.solver.start_time = float(lo)
                params.solver.end_time = float(hi)
                solver.solve()
                if chunk_idx < chunks - 2:
                    solver.set_initial_entropy(
                        np.asarray(solver._solution.y[: solver._n_stag, -1], dtype=float)
                    )
                    continue
                st = solver.get_state()
                solver.set_initial_entropy(st.S_final)
                t_arr = np.asarray(st.T_stag, dtype=float).ravel()
                rb = np.asarray(st.r_basic, dtype=float).ravel()
                rs = np.asarray(st.r_stag, dtype=float).ravel()
                ti_chunk = korenaga_ti_fixed_point(t_arr, (rb[-1] - rs) / d_m, t_surf, dth)
                nu_chunk = dth / (ti_chunk - t_surf)
                hist.append((float(hi), float(ti_chunk), float(nu_chunk)))
            if abs(hist[-1][2] - hist[-2][2]) < 1.0e-5 * hist[-1][2]:
                break
        else:
            raise RuntimeError(f'Case {cid} did not converge by t_end={t_end:.2e} yr')

        s_final = np.asarray(solver._S0, dtype=float).copy()
        rhs_final = np.asarray(solver._dSdt_single(float(t_end), s_final), dtype=float)
        lid = solver.state._lid_state
        t_stag = np.asarray(st.T_stag, dtype=float).ravel()
        rb = np.asarray(st.r_basic, dtype=float).ravel()
        rs = np.asarray(st.r_stag, dtype=float).ravel()
        f_top = float(k_cond * (t_stag[-1] - t_surf) / (rb[-1] - rs[-1]))
        f_bot = float(solver.state._heat_flux[0])
        ti_kor = float(hist[-1][1])
        t_max = float(np.max(t_stag))
        ti_lid = float(lid['T_i'])
        theta_lid = float(lid['theta'])
        d_lid_p1 = float(lid['d_lid'])
        nu_kor = float(dth / (ti_kor - t_surf))
        dt_real = ti_kor - t_surf
        theta_real = float(e_act * dt_real / (R_GAS * ti_kor**2))
        eta_real = float(eta_ref * np.exp((e_act / R_GAS) * (1.0 / ti_kor - 1.0 / t_ref)))
        ra_real = float(alpha * rho * grav * dt_real * d_m**3 / (kappa * eta_real))

        row = {
            'case_id': cid,
            'sweep': str(spec['sweep']),
            'sweep_type': str(spec['sweep']),
            'Ra_i': float(spec['target_Ra_i']),
            'theta': float(spec['target_theta']),
            'Nu': nu_kor,
            'Nu_korenaga': nu_kor,
            'Nu_flux_kor': float(f_top * d_m / (k_cond * (ti_kor - t_surf))),
            'Nu_flux_tmax': float(f_top * d_m / (k_cond * (t_max - t_surf))),
            'Nu_dth_lid': float(dth / (ti_lid - t_surf)),
            'Nu_est': float(spec['Nu_estimated']),
            'T_i_korenaga_K': ti_kor,
            'T_max_K': t_max,
            'T_i_lid_K': ti_lid,
            'T_surf_K': t_surf,
            'delta_T_korenaga_K': dt_real,
            'theta_lid': theta_lid,
            'theta_realized_kor': theta_real,
            'Ra_i_realized_kor': ra_real,
            'd_lid_pass1_m': d_lid_p1,
            's_top_nominal': float(
                compute_mlt_top_slope(float(spec['target_theta']), s_top_inf=0.25, c_theta=0.25)
            ),
            's_top_live': float(compute_mlt_top_slope(theta_lid, s_top_inf=0.25, c_theta=0.25)),
            'F_top_W_m2': f_top,
            'F_top_expected_W_m2': f_top_expected,
            'F_bal': float(f_top / f_top_expected),
            'F_bot_W_m2': f_bot,
            'rel_dNu_last': float(abs(hist[-1][2] - hist[-2][2]) / hist[-1][2]),
            'abs_dTi_last_K': float(abs(hist[-1][1] - hist[-2][1])),
            't_end_yr': float(t_end),
            'status': int(st.status),
        }
        snap = {
            'case_id': cid,
            'Ra_i': float(spec['target_Ra_i']),
            'theta': float(spec['target_theta']),
            'H_W_per_kg': h_heat,
            'E_activation_J_per_mol': e_act,
            'dTH_K': dth,
            'T_surf_K': t_surf,
            't_end_yr': float(t_end),
            'S_final': s_final.tolist(),
            'rhs_final': rhs_final.tolist(),
            'T_stag_final': t_stag.tolist(),
            'F_top_W_m2': f_top,
            'jconv_max_W_m2': float(np.max(solver.state._jconv)),
            'T_i_korenaga_K': ti_kor,
            'Nu_korenaga': nu_kor,
            'lid_state_T_i_K': ti_lid,
            'lid_state_theta': theta_lid,
            'lid_state_d_lid_m': d_lid_p1,
            'lid_state_has_lid': bool(lid['has_lid']),
            'max_abs_rhs_J_kg_K_yr': float(np.max(np.abs(rhs_final))),
        }
        return row, snap
    finally:
        estate_mod.EntropyState.update = orig_update
        es_mod.EntropySolver._solve_cvode = orig_cvode


def compute_primary_fit_all_12_nu_korenaga(
    cases: list[dict[str, Any]],
    win: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute design-parameter multilinear and 1-D sweep OLS fits on ``Nu_korenaga``."""
    ra_des = np.array([float(c['Ra_i']) for c in cases])
    th_des = np.array([float(c['theta']) for c in cases])
    nu_kor = np.array([float(c['Nu_korenaga']) for c in cases])
    mask_ra = np.array([c['sweep'] == 'Ra_i' for c in cases])
    mask_th = np.array([c['sweep'] == 'theta' for c in cases])

    ml_raw = _ols([-np.log(th_des), np.log(ra_des)], np.log(nu_kor))
    x_mat = np.column_stack([np.ones(len(cases)), -np.log(th_des), np.log(ra_des)])
    coeffs = np.array([ml_raw['ln_a'], ml_raw['alpha'], ml_raw['beta']], dtype=float)
    residuals = (np.log(nu_kor) - x_mat @ coeffs).tolist()

    b_raw = _ols([np.log(ra_des[mask_ra])], np.log(nu_kor[mask_ra]))
    a_raw = _ols([-np.log(th_des[mask_th])], np.log(nu_kor[mask_th]))

    b_min = float(win['beta_min']) if isinstance(win, dict) and 'beta_min' in win else 0.294
    b_max = float(win['beta_max']) if isinstance(win, dict) and 'beta_max' in win else 0.350
    a_min = float(win['alpha_min']) if isinstance(win, dict) and 'alpha_min' in win else 1.294
    a_max = float(win['alpha_max']) if isinstance(win, dict) and 'alpha_max' in win else 1.350

    b_in = bool(b_min <= ml_raw['beta'] <= b_max)
    a_in = bool(a_min <= ml_raw['alpha'] <= a_max)
    bs_in = bool(b_min <= b_raw['slope'] <= b_max)
    as_in = bool(a_min <= a_raw['slope'] <= a_max)
    all_in = bool(b_in and a_in)
    return {
        'multilinear': {
            'a': ml_raw['a'],
            'ln_a': ml_raw['ln_a'],
            'alpha': ml_raw['alpha'],
            'beta': ml_raw['beta'],
            'r_squared': ml_raw['r_squared'],
            'std_err_alpha': ml_raw['std_err_alpha'],
            'std_err_beta': ml_raw['std_err_beta'],
            'std_err_ln_a': ml_raw['std_err_ln_a'],
            'residuals': residuals,
        },
        'beta_sweep': {
            'slope': b_raw['slope'],
            'intercept': b_raw['intercept_log10'],
            'r_squared': b_raw['r_squared'],
            'std_err_slope': b_raw['std_err_slope'],
        },
        'alpha_sweep': {
            'slope': a_raw['slope'],
            'intercept': a_raw['intercept_log10'],
            'r_squared': a_raw['r_squared'],
            'std_err_slope': a_raw['std_err_slope'],
        },
        'beta_in_window': b_in,
        'alpha_in_window': a_in,
        'all_in_window': all_in,
        'beta_sweep_in_window': bs_in,
        'alpha_sweep_in_window': as_in,
        'all_sweep_in_window': bool(bs_in and as_in),
        'verdict': 'PASS' if all_in else 'FAIL',
    }


def compute_realized_state_fits(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute multilinear and 1-D OLS fits on realized Korenaga parameters."""
    ra_real = np.array([float(c['Ra_i_realized_kor']) for c in cases])
    th_real = np.array([float(c['theta_realized_kor']) for c in cases])
    nu_kor = np.array([float(c['Nu_korenaga']) for c in cases])
    nu_est = np.array([float(c['Nu_est']) for c in cases])
    mask_ra = np.array([c['sweep'] == 'Ra_i' for c in cases])
    mask_th = np.array([c['sweep'] == 'theta' for c in cases])
    mask_c7 = nu_est >= 2.8

    return {
        'regressor_definition': (
            'Ra_i_realized_kor = rho * g * alpha * (T_i_korenaga - T_surf) * D^3 / '
            '(kappa * eta(T_i_korenaga)); '
            'theta_realized_kor = E_a * (T_i_korenaga - T_surf) / (R * T_i_korenaga^2)'
        ),
        'all_12_nu_korenaga_on_realized_kor': {
            'multilinear': _ols([-np.log(th_real), np.log(ra_real)], np.log(nu_kor)),
            'ra_sweep_1d': _ols([np.log10(ra_real[mask_ra])], np.log10(nu_kor[mask_ra])),
            'theta_sweep_1d': _ols([-np.log10(th_real[mask_th])], np.log10(nu_kor[mask_th])),
        },
        'convecting_7_nu_korenaga_on_realized_kor': {
            'case_ids': [str(c['case_id']) for c, m in zip(cases, mask_c7) if m],
            'multilinear': _ols(
                [-np.log(th_real[mask_c7]), np.log(ra_real[mask_c7])], np.log(nu_kor[mask_c7])
            ),
            'ra_sweep_1d': _ols(
                [np.log10(ra_real[mask_ra & mask_c7])], np.log10(nu_kor[mask_ra & mask_c7])
            ),
            'theta_sweep_1d': _ols(
                [-np.log10(th_real[mask_th & mask_c7])], np.log10(nu_kor[mask_th & mask_c7])
            ),
        },
    }


def _is_commit_on_origin(sha: str, repo_root: Path = REPO_ROOT) -> bool:
    """Return True if ``sha`` is a 40-hex git commit reachable from any ``origin/*`` branch."""
    if not isinstance(sha, str) or len(sha) != 40 or sha == 'uncommitted':
        return False
    try:
        obj_type = subprocess.check_output(
            ['git', '-C', str(repo_root), 'cat-file', '-t', sha],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if obj_type != 'commit':
            return False
        branches_out = subprocess.check_output(
            ['git', '-C', str(repo_root), 'branch', '-r', '--contains', sha],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return any(line.strip().startswith('origin/') for line in branches_out.splitlines())


def _resolve_origin_head_commit(repo_root: Path = REPO_ROOT) -> str:
    """Return ``HEAD`` SHA only when the tracked tree is clean and ``HEAD`` is on ``origin``."""
    status_out = subprocess.check_output(
        ['git', '-C', str(repo_root), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if status_out:
        raise RuntimeError(
            'Working tree has uncommitted tracked changes; commit and push before generating benchmarks'
        )
    head_sha = subprocess.check_output(
        ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if not _is_commit_on_origin(head_sha, repo_root):
        raise RuntimeError(
            f'HEAD {head_sha} is not reachable from any origin/* branch; push before generating benchmarks'
        )
    return head_sha


def verify_benchmarks(data: dict[str, Any]) -> list[str]:
    """Verify schema keys, no-table metadata record, and recompute the live fast subset."""
    failures: list[str] = []

    if tuple(data.keys()) != EXPECTED_TOP_KEYS:
        failures.append(f'Top-level keys {tuple(data.keys())} != {EXPECTED_TOP_KEYS}')

    meta = data.get('metadata', {})
    if not isinstance(meta, dict) or tuple(meta.keys()) != EXPECTED_META_KEYS:
        failures.append(f'metadata keys {tuple(meta.keys())} != {EXPECTED_META_KEYS}')
    elif not _is_commit_on_origin(str(meta.get('git_commit'))):
        failures.append(
            f"metadata['git_commit'] {meta.get('git_commit')!r} is not a commit reachable from origin/*"
        )
    elif meta.get('eos_tables') != NO_TABLE_RECORD:
        failures.append(
            f"metadata['eos_tables'] {meta.get('eos_tables')!r} != {NO_TABLE_RECORD!r}"
        )

    specs_list = data.get('cases_spec', [])
    if not isinstance(specs_list, list) or len(specs_list) != 12:
        failures.append('cases_spec must be a 12-element list')
    elif any(
        not isinstance(s, dict) or tuple(s.keys()) != EXPECTED_SPEC_KEYS for s in specs_list
    ):
        failures.append(f'cases_spec entry keys != {EXPECTED_SPEC_KEYS}')

    diag = data.get('diagnostic_fits', {})
    if not isinstance(diag, dict) or tuple(diag.keys()) != EXPECTED_DIAG_FIT_KEYS:
        failures.append(f'diagnostic_fits keys != {EXPECTED_DIAG_FIT_KEYS}')

    real_fits = data.get('realized_state_fits', {})
    if not isinstance(real_fits, dict) or tuple(real_fits.keys()) != EXPECTED_REALIZED_FIT_KEYS:
        failures.append(f'realized_state_fits keys != {EXPECTED_REALIZED_FIT_KEYS}')

    snaps = data.get('steady_state_snapshots', {})
    if not isinstance(snaps, dict) or tuple(snaps.keys()) != REQUIRED_SNAPSHOT_IDS:
        failures.append(f'steady_state_snapshots keys != {REQUIRED_SNAPSHOT_IDS}')
    if failures:
        return failures

    # 1. Block 1 live check: pure-conduction floor derivations
    floor_live = compute_discrete_conduction_floor_n100()
    fp = meta['fixed_parameters']
    if (
        abs(
            floor_live['Nu_korenaga_conduction_floor_planar_N100']
            - float(fp['conduction_floor_Nu_korenaga_planar_N100'])
        )
        > 1.0e-12
    ):
        failures.append('conduction_floor_Nu_korenaga_planar_N100 mismatch')
    if (
        abs(
            floor_live['Nu_korenaga_conduction_floor_spherical_N100']
            - float(fp['conduction_floor_Nu_korenaga_spherical_N100'])
        )
        > 1.0e-12
    ):
        failures.append('conduction_floor_Nu_korenaga_spherical_N100 mismatch')

    # 2. Block 2 live check: 5 steady-state snapshots through EntropySolver._dSdt_single
    specs_by_id = {str(s['case_id']): s for s in data['cases_spec']}
    results_by_id = {str(r['case_id']): r for r in data['cases_results']}
    for cid in REQUIRED_SNAPSHOT_IDS:
        snap = snaps[cid]
        row = results_by_id[cid]
        live_obs = evaluate_live_snapshot(
            specs_by_id[cid],
            np.asarray(snap['S_final'], dtype=float),
            float(snap['t_end_yr']),
        )
        if live_obs['max_abs_rhs_J_kg_K_yr'] > STEADY_RESIDUAL_BOUND:
            failures.append(
                f'{cid} live max_abs_rhs {live_obs["max_abs_rhs_J_kg_K_yr"]:.3e} > {STEADY_RESIDUAL_BOUND}'
            )
        rhs_diff = float(
            np.max(np.abs(live_obs['rhs_final'] - np.asarray(snap['rhs_final'], dtype=float)))
        )
        if rhs_diff > 1.0e-13:
            failures.append(f'{cid} snapshot rhs_final live mismatch: {rhs_diff:.3e}')
        for key in (
            'F_top_W_m2',
            'jconv_max_W_m2',
            'T_i_korenaga_K',
            'Nu_korenaga',
            'lid_state_T_i_K',
            'lid_state_theta',
            'lid_state_d_lid_m',
        ):
            if (
                abs(live_obs[key] - float(snap[key])) / max(abs(float(snap[key])), 1.0e-30)
                > 1.0e-11
            ):
                failures.append(f'{cid} snapshot {key} live mismatch')
        for key in ('F_top_W_m2', 'T_i_korenaga_K', 'Nu_korenaga'):
            if abs(live_obs[key] - float(row[key])) / abs(float(row[key])) > 1.0e-11:
                failures.append(f'{cid} cases_results {key} live mismatch')

    # 3. Block 3 live check: design-parameter and realized-state OLS fits
    cases = data['cases_results']
    ml_des = compute_primary_fit_all_12_nu_korenaga(cases, meta.get('acceptance_window'))[
        'multilinear'
    ]
    ml_stored = data['primary_fit_all_12_nu_korenaga']['multilinear']
    for k_fit in ('alpha', 'beta', 'r_squared'):
        if abs(ml_des[k_fit] - float(ml_stored[k_fit])) > 1.0e-12:
            failures.append(f'primary_fit_all_12_nu_korenaga {k_fit} mismatch')

    live_real = compute_realized_state_fits(cases)
    for grp in (
        'all_12_nu_korenaga_on_realized_kor',
        'convecting_7_nu_korenaga_on_realized_kor',
    ):
        for sub in ('multilinear', 'ra_sweep_1d', 'theta_sweep_1d'):
            for k_fit, val in live_real[grp][sub].items():
                if isinstance(val, float):
                    if abs(val - float(real_fits[grp][sub][k_fit])) > 1.0e-12:
                        failures.append(f'realized_state_fits {grp}.{sub}.{k_fit} mismatch')

    return failures


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate or check stagnant-lid scaling reference benchmarks.'
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--full', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_arguments(argv)
    output_path = args.output.resolve()

    if args.check:
        if not output_path.is_file():
            sys.stderr.write(f'Benchmark file not found: {output_path}\n')
            return 1
        data = json.loads(output_path.read_text(encoding='utf-8'))
        failures = verify_benchmarks(data)
        if failures:
            sys.stderr.write(f'Verification failed with {len(failures)} issue(s):\n')
            for f in failures:
                sys.stderr.write(f'  - {f}\n')
            return 1
        print(f'Verification passed: {output_path}')
        return 0

    if not output_path.is_file():
        sys.stderr.write(f'Baseline benchmark file not found: {output_path}\n')
        return 1

    try:
        head_sha = _resolve_origin_head_commit(REPO_ROOT)
    except RuntimeError as exc:
        sys.stderr.write(f'{exc}\n')
        return 1

    data = json.loads(output_path.read_text(encoding='utf-8'))
    if isinstance(data.get('cases_spec'), list):
        data['cases_spec'] = [
            {k: s[k] for k in EXPECTED_SPEC_KEYS if k in s}
            for s in data['cases_spec']
            if isinstance(s, dict)
        ]
    if args.full:
        new_rows = []
        new_snaps: dict[str, Any] = {}
        for spec in data['cases_spec']:
            row, snap = run_single_steady_case(spec)
            new_rows.append(row)
            if row['case_id'] in REQUIRED_SNAPSHOT_IDS:
                new_snaps[row['case_id']] = snap
        data['cases_results'] = new_rows
        data['steady_state_snapshots'] = new_snaps

    floor_info = compute_discrete_conduction_floor_n100()
    raw_meta = data.get('metadata', {})
    fp = dict(raw_meta.get('fixed_parameters', {}))
    fp['conduction_floor_Nu_analytic_peak'] = floor_info['Nu_analytic_parabola_peak']
    fp['conduction_floor_Nu_korenaga_analytic_planar'] = floor_info[
        'Nu_korenaga_analytic_planar'
    ]
    fp['conduction_floor_Nu_korenaga_planar_N100'] = floor_info[
        'Nu_korenaga_conduction_floor_planar_N100'
    ]
    fp['conduction_floor_Nu_korenaga_spherical_N100'] = floor_info[
        'Nu_korenaga_conduction_floor_spherical_N100'
    ]
    fp['steady_residual_bound_J_kg_K_yr'] = STEADY_RESIDUAL_BOUND
    acc_win = raw_meta.get('acceptance_window', {})
    data['metadata'] = {
        'git_commit': head_sha,
        'eos_tables': NO_TABLE_RECORD,
        'rheology_calibration': raw_meta.get('rheology_calibration', {}),
        'acceptance_window': acc_win,
        'fixed_parameters': fp,
    }
    data['primary_fit_all_12_nu_korenaga'] = compute_primary_fit_all_12_nu_korenaga(
        data['cases_results'], acc_win
    )
    data['realized_state_fits'] = compute_realized_state_fits(data['cases_results'])

    failures = verify_benchmarks(data)
    if failures:
        sys.stderr.write(f'Generated benchmark failed verification ({len(failures)} issues):\n')
        for f in failures:
            sys.stderr.write(f'  - {f}\n')
        return 1

    output_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f'Benchmarks written to {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
