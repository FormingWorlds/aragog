"""Generate and verify reference benchmarks for internal-heating stagnant-lid scaling tests.

Generates or checks ``tests/reference/ssc_scaling_exponents_benchmarks.json``.

Prescribed surface temperature boundary condition (``outer_boundary_condition = 5``):
- In the entropy solver, ``outer_boundary_condition = 5`` is not branched:
  ``EntropySolver._dSdt_single`` (``src/aragog/solver/entropy_solver.py``)
  branches only on ``self._outer_bc_kind == 1`` and ``self._outer_bc_kind == 4``, and
  ``EntropyState.update`` (``src/aragog/solver/entropy_state.py``) copies the
  surface entropy gradient from the adjacent interior node via ``dSdxi[-1] = dSdxi[-2]``.
- For the 12 internally heated steady-state stagnant-lid cases with prescribed surface
  temperature ``T_surf = 273.0 K``, this tool applies the verification-only context manager
  ``half_cell_surface_flux`` (``tools/verification/_ssc_helpers.py``), which sets the top
  conductive heat flux to the temperature-solver BC 5 half-cell formula
  ``F_top = k * (T_cell - T_surf) / (dr / 2)``.
- Because ``half_cell_surface_flux`` sets only ``_heat_flux[-1]`` after ``EntropyState.update``
  returns, the stagnant-lid closure (``_compute_stagnant_lid_state_core`` in
  ``src/aragog/rheology_lid.py``, called inside ``EntropyState.update`` in
  ``src/aragog/solver/entropy_state.py``) reads the extrapolated surface
  temperature ``T_basic[-1]`` and the unpatched surface flux, so the stored states are
  steady for a closure fed the extrapolated surface temperature and flux.

Execution modes and per-block coverage:
- ``--full``: recomputes the steady-state trajectories, snapshots, conduction floors, and
  OLS fits using ``EntropySolver`` (temporarily replacing ``EntropySolver._solve_cvode``
  with an unbounded-step wrapper ``max_step = np.inf`` because the solver caps steps at
  100 yr after solidification):
  1. ``metadata.fixed_parameters``: derives the analytical and discrete ``N = 100``
     staggered-grid pure-conduction Nusselt floors (parabola peak ``2.0``, analytical
     Korenaga ``1 + sqrt(6)/2 = 2.224745``, planar ``N = 100`` ``2.227638``, and
     spherical-shell ``r_in/r_out = 0.99``, ``N = 100`` ``2.247151``).
  2. ``cases_results`` and ``steady_state_snapshots``: integrates all 12 internally heated
     stagnant-lid cases (``case_01``..``case_12``) to statistical steady state in 8-chunk
     doublings (``t_end`` up to ``320 Gyr``) and extracts the 12 steady-state result rows
     and 5 full steady-state profile snapshots (``case_01``, ``case_03``, ``case_06``,
     ``case_07``, ``case_12``).
  3. ``primary_fit_all_12_nu_korenaga``, ``diagnostic_fits``, and ``realized_state_fits``:
     computes the ordinary least-squares multilinear and 1-D sweep power-law fits on design
     parameters ``(Ra_i, theta)``, alternative Nusselt diagnostics, and realized steady-state
     parameters ``(Ra_i_realized_kor, theta_realized_kor)``.
- ``--check``: verifies expected schema keys, the no-table constant-property record
  (``eos_tables``), and compares every stored field across all blocks, combining live 1-D
  evaluations on the 5 stored snapshots with algebraic and regression checks on the
  remaining 7 stored rows:
  1. ``metadata``: verifies ``git_commit`` as a 40-hex commit SHA and
     ``generator_input_sha256``, ``eos_tables``,
     ``rheology_calibration``, ``acceptance_window``, and all ``fixed_parameters``
     conduction-floor values within ``1e-12``.
  2. ``cases_results`` and ``steady_state_snapshots``: evaluates live 1-D
     ``EntropySolver._dSdt_single`` and ``korenaga_ti_fixed_point`` on the 5 stored
     steady-state snapshots (``case_01``, ``case_03``, ``case_06``, ``case_07``,
     ``case_12``), verifying ``max_abs_rhs_J_kg_K_yr <= 1e-9``, elementwise relative
     agreement of ``rhs_final`` (``rtol=1e-4, atol=1e-18``) and ``T_stag_final``
     (``rtol=1e-11``), and agreement of ``F_top_W_m2``, ``jconv_max_W_m2``,
     ``T_i_korenaga_K``, ``Nu_korenaga``, and ``_lid_state`` observables within ``1e-11``;
     for all 12 rows (including the 7 cases without stored profile snapshots), verifies
     ``cases_results`` against ``cases_spec`` and internal Nusselt/Rayleigh/Frank-Kamenetskii
     identities within ``1e-11``.
  3. ``primary_fit_all_12_nu_korenaga``, ``diagnostic_fits``, and ``realized_state_fits``:
     recomputes all design-parameter, diagnostic, and realized-state multilinear and 1-D
     sweep OLS fits (slopes, intercepts, standard errors, ``R^2``, residuals, window
     flags, and ``verdict``) from ``cases_results`` and verifies agreement within ``1e-12``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402
from tools.verification._ssc_helpers import (  # noqa: E402
    REPO_ROOT,
    _check_generator_input_fingerprint,
    _resolve_origin_head_commit,
    compute_generator_input_fingerprint,
    evaluate_bc5_surface_patch_state,
    half_cell_surface_flux,
    korenaga_ti_fixed_point,
    unbounded_cvode_step,
)
from tools.verification._ssc_helpers import (  # noqa: E402
    ols_fit as _ols,
)

DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / 'tests' / 'reference' / 'ssc_scaling_exponents_benchmarks.json'
)
SSC_SCALING_INPUT_PATHS: tuple[str, ...] = (
    'src/aragog',
    'tools/verification/_ssc_helpers.py',
    'tools/verification/generate_ssc_scaling_exponents_benchmarks.py',
)
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
    'bc5_surface_patch_diagnostics',
)
EXPECTED_META_KEYS = (
    'git_commit',
    'generator_input_sha256',
    'generator_input_files',
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
    with half_cell_surface_flux():
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

    with half_cell_surface_flux(), unbounded_cvode_step():
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


def _compute_nu_fit_bundle(
    cases: list[dict[str, Any]],
    nu_key: str,
    win: dict[str, Any] | None = None,
    *,
    include_meta: bool = False,
) -> dict[str, Any]:
    """Compute multilinear and 1-D sweep OLS fits on ``nu_key`` for ``cases``."""
    ra_des = np.array([float(c['Ra_i']) for c in cases])
    th_des = np.array([float(c['theta']) for c in cases])
    nu_vec = np.array([float(c[nu_key]) for c in cases])
    mask_ra = np.array([c['sweep'] == 'Ra_i' for c in cases])
    mask_th = np.array([c['sweep'] == 'theta' for c in cases])

    ml_raw = _ols([-np.log(th_des), np.log(ra_des)], np.log(nu_vec))
    x_mat = np.column_stack([np.ones(len(cases)), -np.log(th_des), np.log(ra_des)])
    coeffs = np.array([ml_raw['ln_a'], ml_raw['alpha'], ml_raw['beta']], dtype=float)
    residuals = (np.log(nu_vec) - x_mat @ coeffs).tolist()

    b_raw = _ols([np.log(ra_des[mask_ra])], np.log(nu_vec[mask_ra]))
    a_raw = _ols([-np.log(th_des[mask_th])], np.log(nu_vec[mask_th]))

    b_min = float(win['beta_min']) if isinstance(win, dict) and 'beta_min' in win else 0.294
    b_max = float(win['beta_max']) if isinstance(win, dict) and 'beta_max' in win else 0.350
    a_min = float(win['alpha_min']) if isinstance(win, dict) and 'alpha_min' in win else 1.294
    a_max = float(win['alpha_max']) if isinstance(win, dict) and 'alpha_max' in win else 1.350

    b_in = bool(b_min <= ml_raw['beta'] <= b_max)
    a_in = bool(a_min <= ml_raw['alpha'] <= a_max)
    bs_in = bool(b_min <= b_raw['slope'] <= b_max)
    as_in = bool(a_min <= a_raw['slope'] <= a_max)
    all_in = bool(b_in and a_in)
    out: dict[str, Any] = {}
    if include_meta:
        out.update(
            {
                'nu_key': nu_key,
                'ra_key': 'Ra_i',
                'th_key': 'theta',
                'n_cases': int(len(cases)),
                'n_ra_sweep': int(np.sum(mask_ra)),
                'n_theta_sweep': int(np.sum(mask_th)),
            }
        )
    out.update(
        {
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
                'intercept': b_raw['intercept'],
                'r_squared': b_raw['r_squared'],
                'std_err_slope': b_raw['std_err_slope'],
            },
            'alpha_sweep': {
                'slope': a_raw['slope'],
                'intercept': a_raw['intercept'],
                'r_squared': a_raw['r_squared'],
                'std_err_slope': a_raw['std_err_slope'],
            },
            'beta_in_window': b_in,
            'alpha_in_window': a_in,
            'all_in_window': all_in,
            'beta_sweep_in_window': bs_in,
            'alpha_sweep_in_window': as_in,
            'all_sweep_in_window': bool(bs_in and as_in),
        }
    )
    if not include_meta:
        out['verdict'] = 'PASS' if all_in else 'FAIL'
    return out


def compute_primary_fit_all_12_nu_korenaga(
    cases: list[dict[str, Any]],
    win: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute design-parameter multilinear and 1-D sweep OLS fits on ``Nu_korenaga``."""
    return _compute_nu_fit_bundle(cases, 'Nu_korenaga', win, include_meta=False)


def compute_diagnostic_fits(
    cases: list[dict[str, Any]],
    win: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the 6 diagnostic OLS fit bundles across all 12 cases and the 7 convecting cases."""
    conv_cases = [c for c in cases if float(c['Nu_est']) >= 2.8]
    return {
        'all_12_cases_nu_flux_kor': _compute_nu_fit_bundle(
            cases, 'Nu_flux_kor', win, include_meta=True
        ),
        'all_12_cases_nu_flux_tmax': _compute_nu_fit_bundle(
            cases, 'Nu_flux_tmax', win, include_meta=True
        ),
        'all_12_cases_nu_dth_lid': _compute_nu_fit_bundle(
            cases, 'Nu_dth_lid', win, include_meta=True
        ),
        'convecting_7_cases_nu_korenaga': _compute_nu_fit_bundle(
            conv_cases, 'Nu_korenaga', win, include_meta=True
        ),
        'convecting_7_cases_nu_flux_tmax': _compute_nu_fit_bundle(
            conv_cases, 'Nu_flux_tmax', win, include_meta=True
        ),
        'convecting_7_cases_nu_dth_lid': _compute_nu_fit_bundle(
            conv_cases, 'Nu_dth_lid', win, include_meta=True
        ),
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


def compute_scaling_bc5_surface_patch_diagnostics(
    specs_list: list[dict[str, Any]],
    snaps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute BC 5 half-cell patch vs unpatched and surface-consistent closure diagnostics on the 5 snapshots."""
    specs_by_id = {str(s['case_id']): s for s in specs_list}
    snaps_out: dict[str, dict[str, Any]] = {}
    for cid in REQUIRED_SNAPSHOT_IDS:
        params = build_scaling_case_params(specs_by_id[cid])
        s_vec = np.asarray(snaps[cid]['S_final'], dtype=float)
        t_yr = float(snaps[cid]['t_end_yr'])
        snaps_out[cid] = evaluate_bc5_surface_patch_state(params, s_vec, t_yr)

    t_ext_vals = [float(st['T_basic_extrapolated_K']) for st in snaps_out.values()]
    jcond_vals = [float(st['jcond_unpatched_surface_W_m2']) for st in snaps_out.values()]
    ftop_vals = [float(st['with_patch_F_top_W_m2']) for st in snaps_out.values()]
    theta_diffs = [float(st['theta_rel_diff']) for st in snaps_out.values()]
    stored_rhs = [float(st['with_patch_max_abs_rhs_J_kg_K_yr']) for st in snaps_out.values()]
    return {
        'snapshots': snaps_out,
        'summary_ranges': {
            'T_basic_extrapolated_min_K': min(t_ext_vals),
            'T_basic_extrapolated_max_K': max(t_ext_vals),
            'jcond_unpatched_min_W_m2': min(jcond_vals),
            'jcond_unpatched_max_W_m2': max(jcond_vals),
            'F_top_patched_min_W_m2': min(ftop_vals),
            'F_top_patched_max_W_m2': max(ftop_vals),
            'd_lid_rel_diff_case_06': float(snaps_out['case_06']['d_lid_pass1_rel_diff']),
            'd_lid_abs_rel_diff_case_06': float(
                snaps_out['case_06']['d_lid_pass1_abs_rel_diff']
            ),
            'd_lid_rel_diff_case_07': float(snaps_out['case_07']['d_lid_pass1_rel_diff']),
            'd_lid_cond_lid_rel_diff_case_12': float(
                snaps_out['case_12']['d_lid_cond_lid_rel_diff']
            ),
            'd_lid_cond_lid_rel_diff_case_01': float(
                snaps_out['case_01']['d_lid_cond_lid_rel_diff']
            ),
            'd_lid_cond_lid_rel_diff_case_03': float(
                snaps_out['case_03']['d_lid_cond_lid_rel_diff']
            ),
            'd_lid_cond_lid_rel_diff_case_07': float(
                snaps_out['case_07']['d_lid_cond_lid_rel_diff']
            ),
            'd_lid_cond_lid_rel_diff_case_06': float(
                snaps_out['case_06']['d_lid_cond_lid_rel_diff']
            ),
            'theta_rel_diff_min': min(theta_diffs),
            'theta_rel_diff_max': max(theta_diffs),
            'stored_max_abs_rhs_min': min(stored_rhs),
            'stored_max_abs_rhs_max': max(stored_rhs),
            'consistent_max_abs_rhs_min_case_12': float(
                snaps_out['case_12']['consistent_max_abs_rhs_J_kg_K_yr']
            ),
            'consistent_max_abs_rhs_case_01': float(
                snaps_out['case_01']['consistent_max_abs_rhs_J_kg_K_yr']
            ),
            'consistent_max_abs_rhs_max_case_06': float(
                snaps_out['case_06']['consistent_max_abs_rhs_J_kg_K_yr']
            ),
        },
    }


def _compare_fit_bundle(
    live_bundle: dict[str, Any],
    stored_bundle: dict[str, Any],
    prefix: str,
    failures: list[str],
) -> None:
    """Compare all fields of an OLS fit bundle against stored JSON values."""
    if not isinstance(stored_bundle, dict):
        failures.append(f'{prefix} is not a dict')
        return
    for k_top, v_live in live_bundle.items():
        if k_top not in stored_bundle:
            failures.append(f'{prefix} missing key {k_top!r}')
            continue
        v_stored = stored_bundle[k_top]
        if isinstance(v_live, bool):
            if v_stored is not v_live:
                failures.append(f'{prefix}.{k_top} {v_stored!r} != {v_live!r}')
        elif isinstance(v_live, (str, int)):
            if v_stored != v_live:
                failures.append(f'{prefix}.{k_top} {v_stored!r} != {v_live!r}')
        elif isinstance(v_live, float):
            if abs(v_live - float(v_stored)) > 1.0e-12:
                failures.append(f'{prefix}.{k_top} mismatch: {v_stored} vs {v_live}')
        elif isinstance(v_live, dict):
            if not isinstance(v_stored, dict):
                failures.append(f'{prefix}.{k_top} is not a dict')
                continue
            for sk, sv_live in v_live.items():
                if sk not in v_stored:
                    failures.append(f'{prefix}.{k_top} missing key {sk!r}')
                    continue
                sv_stored = v_stored[sk]
                if isinstance(sv_live, list):
                    if not isinstance(sv_stored, list) or len(sv_stored) != len(sv_live):
                        failures.append(f'{prefix}.{k_top}.{sk} length mismatch')
                    elif not np.allclose(
                        np.asarray(sv_live, dtype=float),
                        np.asarray(sv_stored, dtype=float),
                        rtol=1.0e-11,
                        atol=1.0e-12,
                    ):
                        failures.append(f'{prefix}.{k_top}.{sk} array mismatch')
                elif isinstance(sv_live, int):
                    if sv_stored != sv_live:
                        failures.append(f'{prefix}.{k_top}.{sk} {sv_stored!r} != {sv_live!r}')
                elif isinstance(sv_live, float):
                    if abs(sv_live - float(sv_stored)) > 1.0e-12:
                        failures.append(
                            f'{prefix}.{k_top}.{sk} mismatch: {sv_stored} vs {sv_live}'
                        )


def verify_benchmarks(data: dict[str, Any]) -> list[str]:
    """Verify schema keys, no-table metadata record, and recompute every quoted field."""
    failures: list[str] = []

    if tuple(data.keys()) != EXPECTED_TOP_KEYS:
        failures.append(f'Top-level keys {tuple(data.keys())} != {EXPECTED_TOP_KEYS}')

    meta = data.get('metadata', {})
    if not isinstance(meta, dict) or tuple(meta.keys()) != EXPECTED_META_KEYS:
        failures.append(f'metadata keys {tuple(meta.keys())} != {EXPECTED_META_KEYS}')
    else:
        failures.extend(
            _check_generator_input_fingerprint(
                meta.get('git_commit'),
                meta.get('generator_input_sha256'),
                SSC_SCALING_INPUT_PATHS,
                recorded_files=meta.get('generator_input_files'),
                repo_root=REPO_ROOT,
            )
        )
        if meta.get('eos_tables') != NO_TABLE_RECORD:
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

    cases_list = data.get('cases_results', [])
    if not isinstance(cases_list, list) or len(cases_list) != 12:
        failures.append('cases_results must be a 12-element list')

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

    # 1. Block 1 live check: metadata calibration, acceptance window, and conduction floors
    rcal = meta.get('rheology_calibration', {})
    if rcal.get('enabled') is not True:
        failures.append('metadata.rheology_calibration.enabled is not True')
    for k_cal, v_exp in (
        ('s_top_inf', 0.25),
        ('c_theta', 0.25),
    ):
        if abs(float(rcal.get(k_cal, -1.0)) - v_exp) > 1.0e-12:
            failures.append(f'metadata.rheology_calibration.{k_cal} != {v_exp}')

    awin = meta.get('acceptance_window', {})
    for k_win, v_exp in (
        ('beta_min', 0.294),
        ('beta_max', 0.350),
        ('alpha_min', 1.294),
        ('alpha_max', 1.350),
    ):
        if abs(float(awin.get(k_win, -1.0)) - v_exp) > 1.0e-12:
            failures.append(f'metadata.acceptance_window.{k_win} != {v_exp}')

    floor_live = compute_discrete_conduction_floor_n100()
    fp = meta['fixed_parameters']
    for fp_key, fl_key in (
        ('conduction_floor_Nu_analytic_peak', 'Nu_analytic_parabola_peak'),
        ('conduction_floor_Nu_korenaga_analytic_planar', 'Nu_korenaga_analytic_planar'),
        (
            'conduction_floor_Nu_korenaga_planar_N100',
            'Nu_korenaga_conduction_floor_planar_N100',
        ),
        (
            'conduction_floor_Nu_korenaga_spherical_N100',
            'Nu_korenaga_conduction_floor_spherical_N100',
        ),
    ):
        if abs(floor_live[fl_key] - float(fp.get(fp_key, -1.0))) > 1.0e-12:
            failures.append(f'{fp_key} mismatch')
    if (
        abs(float(fp.get('steady_residual_bound_J_kg_K_yr', -1.0)) - STEADY_RESIDUAL_BOUND)
        > 1.0e-18
    ):
        failures.append('steady_residual_bound_J_kg_K_yr mismatch')

    # 2. Block 2 check: all 12 rows of cases_results + 5 live steady-state snapshots
    specs_by_id = {str(s['case_id']): s for s in specs_list}
    results_by_id: dict[str, dict[str, Any]] = {}
    for row in cases_list:
        if not isinstance(row, dict) or 'case_id' not in row:
            failures.append('Invalid row in cases_results')
            continue
        cid = str(row['case_id'])
        if cid not in specs_by_id:
            failures.append(f'Unknown case_id {cid} in cases_results')
            continue
        results_by_id[cid] = row
        spec = specs_by_id[cid]
        d_m = float(spec['D_m'])
        k_cond = float(spec['k_thermal_W_per_m_K'])
        rho = float(spec['rho_kg_per_m3'])
        cp = float(spec['cp_J_per_kg_K'])
        alpha = float(spec['alpha_thermal_per_K'])
        grav = float(spec['g_m_per_s2'])
        h_heat = float(spec['H_W_per_kg'])
        t_surf = float(spec['T_surf_K'])
        t_ref = float(spec['T_i_nominal_K'])
        e_act = float(spec['E_activation_J_per_mol'])
        eta_ref = float(spec['eta_i_Pa_s'])
        kappa = k_cond / (rho * cp)
        dth = rho * h_heat * d_m**2 / k_cond
        f_geom = float(spec['R_in_m']) / float(spec['R_out_m'])
        f_top_exp = rho * h_heat * d_m * (1.0 + f_geom + f_geom**2) / 3.0

        ti_kor = float(row['T_i_korenaga_K'])
        t_max = float(row['T_max_K'])
        ti_lid = float(row['T_i_lid_K'])
        theta_lid = float(row['theta_lid'])
        f_top = float(row['F_top_W_m2'])
        dt_real = ti_kor - t_surf
        nu_kor = dth / dt_real
        theta_real = e_act * dt_real / (R_GAS * ti_kor**2)
        eta_real = eta_ref * np.exp((e_act / R_GAS) * (1.0 / ti_kor - 1.0 / t_ref))
        ra_real = alpha * rho * grav * dt_real * d_m**3 / (kappa * eta_real)

        expected_pairs = (
            ('Ra_i', float(spec['target_Ra_i'])),
            ('theta', float(spec['target_theta'])),
            ('Nu', nu_kor),
            ('Nu_korenaga', nu_kor),
            ('Nu_flux_kor', f_top * d_m / (k_cond * dt_real)),
            ('Nu_flux_tmax', f_top * d_m / (k_cond * (t_max - t_surf))),
            ('Nu_dth_lid', dth / (ti_lid - t_surf)),
            ('Nu_est', float(spec['Nu_estimated'])),
            ('T_surf_K', t_surf),
            ('delta_T_korenaga_K', dt_real),
            ('theta_realized_kor', theta_real),
            ('Ra_i_realized_kor', ra_real),
            (
                's_top_nominal',
                float(
                    compute_mlt_top_slope(
                        float(spec['target_theta']), s_top_inf=0.25, c_theta=0.25
                    )
                ),
            ),
            (
                's_top_live',
                float(compute_mlt_top_slope(theta_lid, s_top_inf=0.25, c_theta=0.25)),
            ),
            ('F_top_expected_W_m2', f_top_exp),
            ('F_bal', f_top / f_top_exp),
        )
        for rk, r_exp in expected_pairs:
            if abs(float(row.get(rk, -1.0e99)) - r_exp) / max(abs(r_exp), 1.0e-30) > 1.0e-11:
                failures.append(f'cases_results {cid} {rk} mismatch')
        if row.get('sweep') != spec['sweep'] or row.get('sweep_type') != spec['sweep']:
            failures.append(f'cases_results {cid} sweep mismatch')
        if int(row.get('status', -1)) != 0 or float(row.get('rel_dNu_last', 1.0)) >= 1.0e-5:
            failures.append(f'cases_results {cid} convergence/status check failed')

    if len(results_by_id) != 12:
        failures.append(f'Expected 12 unique cases in cases_results, got {len(results_by_id)}')
        return failures

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
        if not np.isclose(
            live_obs['max_abs_rhs_J_kg_K_yr'],
            float(snap['max_abs_rhs_J_kg_K_yr']),
            rtol=1.0e-4,
            atol=1.0e-18,
        ):
            failures.append(f'{cid} snapshot max_abs_rhs_J_kg_K_yr mismatch')
        if not np.allclose(
            live_obs['rhs_final'],
            np.asarray(snap['rhs_final'], dtype=float),
            rtol=1.0e-4,
            atol=1.0e-18,
        ):
            failures.append(f'{cid} snapshot rhs_final live mismatch')
        if not np.allclose(
            live_obs['T_stag_final'],
            np.asarray(snap['T_stag_final'], dtype=float),
            rtol=1.0e-11,
            atol=0.0,
        ):
            failures.append(f'{cid} snapshot T_stag_final live mismatch')
        if bool(live_obs['lid_state_has_lid']) is not bool(snap.get('lid_state_has_lid')):
            failures.append(f'{cid} snapshot lid_state_has_lid mismatch')
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
        for snap_k, row_k in (
            ('F_top_W_m2', 'F_top_W_m2'),
            ('T_i_korenaga_K', 'T_i_korenaga_K'),
            ('Nu_korenaga', 'Nu_korenaga'),
            ('lid_state_T_i_K', 'T_i_lid_K'),
            ('lid_state_theta', 'theta_lid'),
            ('lid_state_d_lid_m', 'd_lid_pass1_m'),
        ):
            if (
                abs(live_obs[snap_k] - float(row[row_k])) / max(abs(float(row[row_k])), 1.0e-30)
                > 1.0e-11
            ):
                failures.append(f'{cid} cases_results {row_k} live mismatch')

    # 3. Block 3 live check: primary, diagnostic, and realized-state OLS fits
    cases = data['cases_results']
    live_prim = compute_primary_fit_all_12_nu_korenaga(cases, awin)
    _compare_fit_bundle(
        live_prim,
        data['primary_fit_all_12_nu_korenaga'],
        'primary_fit_all_12_nu_korenaga',
        failures,
    )

    live_diag = compute_diagnostic_fits(cases, awin)
    for d_key, d_bundle in live_diag.items():
        _compare_fit_bundle(d_bundle, diag.get(d_key, {}), f'diagnostic_fits.{d_key}', failures)

    live_real = compute_realized_state_fits(cases)
    if real_fits.get('regressor_definition') != live_real['regressor_definition']:
        failures.append('realized_state_fits.regressor_definition mismatch')
    for grp in (
        'all_12_nu_korenaga_on_realized_kor',
        'convecting_7_nu_korenaga_on_realized_kor',
    ):
        if 'case_ids' in live_real[grp]:
            if real_fits.get(grp, {}).get('case_ids') != live_real[grp]['case_ids']:
                failures.append(f'realized_state_fits.{grp}.case_ids mismatch')
        _compare_fit_bundle(
            {k: v for k, v in live_real[grp].items() if k != 'case_ids'},
            real_fits.get(grp, {}),
            f'realized_state_fits.{grp}',
            failures,
        )

    live_bc5 = compute_scaling_bc5_surface_patch_diagnostics(specs_list, snaps)
    stored_bc5 = data.get('bc5_surface_patch_diagnostics', {})
    if not isinstance(stored_bc5, dict):
        failures.append('bc5_surface_patch_diagnostics is not a dict')
    else:
        st_snaps = stored_bc5.get('snapshots', {})
        for cid, l_st in live_bc5['snapshots'].items():
            s_st = st_snaps.get(cid) if isinstance(st_snaps, dict) else None
            if not isinstance(s_st, dict):
                failures.append(f'bc5_surface_patch_diagnostics.snapshots missing {cid!r}')
                continue
            for fk, fv in l_st.items():
                sv = s_st.get(fk, '__missing__')
                if fv is None:
                    if sv is not None:
                        failures.append(
                            f'bc5_surface_patch_diagnostics.{cid}.{fk} expected None'
                        )
                elif isinstance(fv, int):
                    if int(sv) != fv:
                        failures.append(f'bc5_surface_patch_diagnostics.{cid}.{fk} mismatch')
                else:
                    tol_f = 1.0e-4 if 'max_abs_rhs' in fk else 1.0e-9
                    if not np.isclose(float(sv), float(fv), rtol=tol_f, atol=1.0e-15):
                        failures.append(f'bc5_surface_patch_diagnostics.{cid}.{fk} mismatch')
        st_sum = stored_bc5.get('summary_ranges', {})
        for sk, sv_live in live_bc5['summary_ranges'].items():
            if not isinstance(st_sum, dict) or sk not in st_sum:
                failures.append(f'bc5_surface_patch_diagnostics.summary_ranges missing {sk!r}')
                continue
            tol_s = 1.0e-4 if 'max_abs_rhs' in sk else 1.0e-9
            if not np.isclose(float(st_sum[sk]), float(sv_live), rtol=tol_s, atol=1.0e-15):
                failures.append(f'bc5_surface_patch_diagnostics.summary_ranges.{sk} mismatch')

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
    input_sha, input_files = compute_generator_input_fingerprint(
        SSC_SCALING_INPUT_PATHS, repo_root=REPO_ROOT
    )
    data['metadata'] = {
        'git_commit': head_sha,
        'generator_input_sha256': input_sha,
        'generator_input_files': input_files,
        'eos_tables': NO_TABLE_RECORD,
        'rheology_calibration': raw_meta.get('rheology_calibration', {}),
        'acceptance_window': acc_win,
        'fixed_parameters': fp,
    }
    data['primary_fit_all_12_nu_korenaga'] = compute_primary_fit_all_12_nu_korenaga(
        data['cases_results'], acc_win
    )
    data['diagnostic_fits'] = compute_diagnostic_fits(data['cases_results'], acc_win)
    data['realized_state_fits'] = compute_realized_state_fits(data['cases_results'])
    data['bc5_surface_patch_diagnostics'] = compute_scaling_bc5_surface_patch_diagnostics(
        data['cases_spec'], data['steady_state_snapshots']
    )

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
