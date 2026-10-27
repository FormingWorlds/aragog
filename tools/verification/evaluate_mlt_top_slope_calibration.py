"""Evaluate and verify the viscous mixing-length top-slope calibration benchmarks.

This script evaluates the 36 published 2-D stagnant-lid convection benchmark cases
used to assess the viscosity-contrast-dependent upper mixing-length slope
``s_top(theta) = s_top_inf + (1 - s_top_inf) * exp(-c_theta * max(theta, 0))``:

1. 12 planar internally heated Newtonian cases from Korenaga (2009, Table 1),
   evaluated against a 10% relative Nusselt-number tolerance (12 normalised
   residuals).
2. 24 spherical internally and mixed heated cases from Reese et al. (2005) as
   tabulated by Deschamps and Vilella (2021, Table 1), evaluated against a 0.03
   absolute interior-temperature (T_m) tolerance, a 15% relative top-heat-flow
   (Phi_top) tolerance, and a 20% relative lid-thickness (d_lid) tolerance (72
   normalised residuals, 84 normalised residuals in total across the 36 cases).

In default / ``--full`` mode, the script runs all 36 steady-state solves at the
chosen pair ``(s_top_inf, c_theta) = (0.25, 0.25)`` and at the
``SolidRheologyParams`` default pair ``(0.22, 0.15)`` using
``stress_closure_mode = 'local'``, ``s_top`` frozen at each table's nominal
``theta`` via ``c_theta = 0.0``, a half-cell conductive boundary-flux patch on
inner BC 3 via ``_patched_state_update``, and a
Frank-Kamenetskii viscosity override during steady-state integration on the
spherical cases, verifies the generated payload in memory, and writes
``tests/reference/ssc_mlt_top_slope_calibration.json``.

In ``--check`` mode, the script validates the 40-hex provenance commit SHA, the
SHA-256 generator-input fingerprint (``generator_input_sha256``), and
tolerance/default metadata fields, recomputes every per-row normalised residual,
top-slope evaluation, and summary statistic (maxima, worst-case row names,
exceedance counts, and mean-square normalised residuals) from the stored row
data for both parameter pairs (34 of 36 cases verified from stored rows), and
runs live ``EntropySolver`` steady-state checks on the worst Korenaga (2009)
case (``E340``) and on a Deschamps and Vilella (2021) spherical case
(``f60_H4``) at both ``(0.25, 0.25)`` and ``(0.22, 0.15)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aragog import rheology as rheo_mod  # noqa: E402
from aragog import rheology_lid as lid_mod  # noqa: E402
from aragog.eos import entropy_phase as ep_mod  # noqa: E402
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
from tools.verification._ssc_helpers import (  # noqa: E402
    _check_generator_input_fingerprint,
    _is_valid_commit_sha,
    _resolve_origin_head_commit,
    compute_generator_input_fingerprint,
    korenaga_ti_fixed_point,
)

DEFAULT_OUT = REPO_ROOT / 'tests' / 'reference' / 'ssc_mlt_top_slope_calibration.json'
MLT_CALIBRATION_INPUT_PATHS: tuple[str, ...] = (
    'src/aragog',
    'tools/verification/_ssc_helpers.py',
    'tools/verification/evaluate_mlt_top_slope_calibration.py',
)

KORENAGA_NU_REL_TOL = 0.10
DV2021_TM_ABS_TOL = 0.03
DV2021_PHI_REL_TOL = 0.15
DV2021_DLID_REL_TOL = 0.20

KORENAGA_2009_ROWS: dict[str, tuple[float, float, float, float, float]] = {
    'E230a': (230.0, 1531.0, 13.0, 2.41e6, 2.94),
    'E340': (340.0, 1531.0, 19.2, 1.24e7, 3.04),
    'E270': (270.0, 1546.0, 15.2, 7.18e6, 3.20),
    'E300': (300.0, 1544.0, 16.9, 1.49e7, 3.40),
    'E210': (210.0, 1561.0, 11.7, 8.36e6, 3.84),
    'E250b': (250.0, 1564.0, 13.9, 1.89e7, 4.03),
    'E280a': (280.0, 1522.0, 15.9, 5.13e6, 2.96),
    'E250a': (250.0, 1549.0, 14.0, 7.52e6, 3.39),
    'E230b': (230.0, 1528.0, 13.0, 6.38e6, 3.44),
    'E320': (320.0, 1537.0, 18.1, 1.36e7, 3.22),
    'E280b': (280.0, 1546.0, 15.7, 1.57e7, 3.59),
    'E260': (260.0, 1561.0, 14.5, 1.72e7, 3.84),
}

DESCHAMPS_VILELLA_2021_ROWS: dict[
    str, tuple[float, float, float, float, float, float, float, float]
] = {
    'f60_H4': (16.0, 0.60, 1e6, 4.0, 0.932, 3.970, 3.767, 0.221),
    'f80_H2': (10.0, 0.80, 1e6, 2.0, 0.938, 3.617, 3.107, 0.242),
    'f80_H4': (10.0, 0.80, 1e6, 4.0, 0.977, 4.267, 1.580, 0.213),
    'f80_H8': (10.0, 0.80, 1e6, 8.0, 1.051, 6.039, -0.729, 0.168),
    'f85_H4': (10.0, 0.85, 1e6, 4.0, 0.983, 4.403, 1.345, 0.215),
    'f70_1e7_H2': (10.0, 0.70, 1e7, 2.0, 0.907, 4.914, 7.046, 0.171),
    'f75_1e7_H4': (10.0, 0.75, 1e7, 4.0, 0.945, 5.834, 4.883, 0.155),
    'f80_1e8_H4': (1.0, 0.80, 1e8, 4.0, 0.966, 5.266, 3.157, 0.181),
    'f60_1e6_H10': (16.0, 0.60, 1e6, 10.0, 1.034, 6.313, -0.607, 0.155),
    'f60_1e7_H4': (5.1, 0.60, 1e7, 4.0, 0.929, 4.389, 4.928, 0.204),
    'f70_1e6_H8': (10.0, 0.70, 1e6, 8.0, 1.035, 5.596, -0.498, 0.177),
    'f70_1e7_H8': (3.2, 0.70, 1e7, 8.0, 1.011, 5.833, -0.011, 0.167),
    'f70_15p8_1e7_H3': (15.8, 0.70, 1e7, 3.0, 0.917, 5.845, 7.461, 0.149),
    'f75_3p2_1e7_H4': (3.2, 0.75, 1e7, 4.0, 0.964, 4.530, 2.568, 0.206),
    'f75_3p2_1e7_H8': (3.2, 0.75, 1e7, 8.0, 1.017, 6.049, -0.208, 0.162),
    'f75_3p2_1e7_H10': (3.2, 0.75, 1e7, 10.0, 1.054, 7.188, -0.923, 0.141),
    'f75_10_1e7_H10': (10.0, 0.75, 1e7, 10.0, 1.005, 7.817, 0.192, 0.123),
    'f80_10_1e6_H10': (10.0, 0.80, 1e6, 10.0, 1.098, 7.289, -1.315, 0.144),
    'f80_32_1e6_H4': (32.0, 0.80, 1e6, 4.0, 0.952, 5.412, 3.368, 0.167),
    'f80_3p2_1e7_H4': (3.2, 0.80, 1e7, 4.0, 0.972, 4.666, 2.205, 0.201),
    'f80_3p2_1e7_H8': (3.2, 0.80, 1e7, 8.0, 1.025, 6.253, -0.395, 0.159),
    'f80_3p2_1e7_H10': (3.2, 0.80, 1e7, 10.0, 1.058, 7.503, -0.979, 0.134),
    'f80_3p2_1e8_H3': (3.2, 0.80, 1e8, 3.0, 0.940, 6.482, 6.387, 0.144),
    'f85_10_1e6_H8': (10.0, 0.85, 1e6, 8.0, 1.058, 6.253, -0.838, 0.164),
}

_orig_solve_cvode = es_mod.EntropySolver._solve_cvode
_orig_state_update = estate_mod.EntropyState.update
_orig_check_rheo_inv = es_mod.EntropySolver._check_rheology_invariants
_orig_energy_integrals = es_mod.EntropySolver._compute_step_energy_integrals
_orig_vmlf_core = rheo_mod._viscous_mixing_length_factor_core
_orig_arr_pub = rheo_mod.compute_arrhenius_viscosity
_orig_arr_core = rheo_mod._compute_arrhenius_viscosity_core
_FK_STATE: dict[str, float] = {}


def _patched_solve_cvode(self: Any, *a: Any, **k: Any) -> Any:
    k['max_step'] = np.inf
    return _orig_solve_cvode(self, *a, **k)


def _patched_vmlf_core(
    radii: Any,
    r_in: Any,
    r_out: Any,
    mixing_length: Any,
    s_top_inf: float | None = None,
    bottom_slope: float | None = None,
    theta: Any = None,
    c_theta: float | None = None,
    *,
    xp: Any = np,
) -> Any:
    zero_c = c_theta is not None and float(c_theta) == 0.0
    return _orig_vmlf_core(
        radii,
        r_in,
        r_out,
        mixing_length,
        s_top_inf=s_top_inf,
        bottom_slope=bottom_slope,
        theta=None if zero_c else theta,
        c_theta=0.0 if zero_c else c_theta,
        xp=xp,
    )


def _patched_state_update(self: Any, entropy: Any, time_val: Any, *a: Any, **k: Any) -> None:
    _orig_state_update(self, entropy, time_val, *a, **k)
    bc_settings = self._evaluator.boundary_conditions._settings
    if int(bc_settings.inner_boundary_condition) == 3:
        k_first = float(np.asarray(self.phase_staggered.thermal_conductivity()).ravel()[0])
        t_first = float(np.asarray(self.phase_staggered.temperature()).ravel()[0])
        r_b = np.asarray(self._evaluator.mesh.basic.radii).ravel()
        dr_cmb_half = 0.5 * float(r_b[1] - r_b[0])
        self._heat_flux[0] = (
            k_first * (float(bc_settings.inner_boundary_value) - t_first) / dr_cmb_half
        )


def _fast_check_rheo_inv(
    self: Any, trace_y: Any = None, trace_t: Any = None, *args: Any, **kwargs: Any
) -> None:
    if trace_y is not None:
        y_pts = np.asarray(trace_y, dtype=float)
        if y_pts.ndim == 2:
            y_pts = y_pts[:, -1:]
        t_pts = np.asarray(trace_t, dtype=float).ravel()[-1:] if trace_t is not None else None
        return _orig_check_rheo_inv(self, y_pts, t_pts, *args, **kwargs)
    return _orig_check_rheo_inv(self, None, None, *args, **kwargs)


def _make_fk_viscosity(orig_fn: Any) -> Any:
    def _fk_viscosity(temperature: Any, pressure: Any, *a: Any, xp: Any = np, **k: Any) -> Any:
        if not _FK_STATE:
            return orig_fn(temperature, pressure, *a, xp=xp, **k)
        res = _FK_STATE['eta0'] * xp.exp(
            -_FK_STATE['gamma'] * (xp.asarray(temperature, dtype=float) - _FK_STATE['T0'])
        ) + 0.0 * xp.asarray(pressure, dtype=float)
        if xp is np and np.ndim(temperature) == 0 and np.ndim(pressure) == 0:
            return float(res)
        return res

    return _fk_viscosity


_fk_viscosity_pub = _make_fk_viscosity(_orig_arr_pub)
_fk_viscosity_core = _make_fk_viscosity(_orig_arr_core)


def _install_benchmark_patches() -> None:
    es_mod.EntropySolver._solve_cvode = _patched_solve_cvode
    estate_mod.EntropyState.update = _patched_state_update
    es_mod.EntropySolver._check_rheology_invariants = _fast_check_rheo_inv
    es_mod.EntropySolver._compute_step_energy_integrals = lambda self: dict(
        es_mod._ZERO_ENERGY_INTEGRALS
    )
    rheo_mod._viscous_mixing_length_factor_core = _patched_vmlf_core
    estate_mod._viscous_mixing_length_factor_core = _patched_vmlf_core
    rheo_mod.compute_arrhenius_viscosity = _fk_viscosity_pub
    rheo_mod._compute_arrhenius_viscosity_core = _fk_viscosity_core
    ep_mod.compute_arrhenius_viscosity = _fk_viscosity_pub
    ep_mod._compute_arrhenius_viscosity_core = _fk_viscosity_core
    lid_mod.compute_arrhenius_viscosity = _fk_viscosity_pub
    lid_mod._compute_arrhenius_viscosity_core = _fk_viscosity_core


def _restore_benchmark_patches() -> None:
    _FK_STATE.clear()
    es_mod.EntropySolver._solve_cvode = _orig_solve_cvode
    estate_mod.EntropyState.update = _orig_state_update
    es_mod.EntropySolver._check_rheology_invariants = _orig_check_rheo_inv
    es_mod.EntropySolver._compute_step_energy_integrals = _orig_energy_integrals
    rheo_mod._viscous_mixing_length_factor_core = _orig_vmlf_core
    estate_mod._viscous_mixing_length_factor_core = _orig_vmlf_core
    rheo_mod.compute_arrhenius_viscosity = _orig_arr_pub
    rheo_mod._compute_arrhenius_viscosity_core = _orig_arr_core
    ep_mod.compute_arrhenius_viscosity = _orig_arr_pub
    ep_mod._compute_arrhenius_viscosity_core = _orig_arr_core
    lid_mod.compute_arrhenius_viscosity = _orig_arr_pub
    lid_mod._compute_arrhenius_viscosity_core = _orig_arr_core


def get_git_commit() -> str:
    """Return current git HEAD SHA of the repository after checking clean tree and origin."""
    return _resolve_origin_head_commit(REPO_ROOT)


def lid_depth_tangent(jconv: np.ndarray, z: np.ndarray) -> float:
    """Evaluate the Davaille and Jaupart tangent construction for stagnant-lid depth."""
    i0 = int(np.argmax(jconv))
    if i0 < 2 or jconv[i0] <= 0.0:
        return float('nan')
    sub = slice(0, i0 + 1)
    zz, jj = z[sub], jconv[sub]
    slope = np.gradient(jj, zz)
    m = int(np.argmax(slope))
    return float(zz[m] - jj[m] / slope[m]) if slope[m] > 0.0 else float('nan')


def _row_s_top(theta_val: float, s_top_inf: float, c_theta: float) -> float:
    if float(c_theta) == 0.0:
        return float(s_top_inf)
    return float(compute_mlt_top_slope(theta_val, s_top_inf=s_top_inf, c_theta=c_theta))


def build_korenaga_params(
    row_name: str,
    n_nodes: int,
    s_top_inf: float,
    c_theta: float,
) -> tuple[Parameters, dict[str, Any]]:
    """Build Parameters for a Korenaga (2009, Table 1) planar internally heated case."""
    e_kj, dt_k, theta_nom, ra_i, nu_ref = KORENAGA_2009_ROWS[row_name]
    d_shell = 1.0e6
    r_out = 100.0 * d_shell
    scale_f = 100.0
    rho, cp, k_cond, alpha, g = 4000.0, 1000.0 * scale_f, 4.0 * scale_f, 3.0e-5, 10.0
    t_s = 273.0
    t_i = t_s + dt_k
    e_act = e_kj * 1.0e3
    kappa = k_cond / (rho * cp)
    eta_i = alpha * rho * g * dt_k * d_shell**3 / (kappa * ra_i)
    h_heat = nu_ref * dt_k * k_cond / (rho * d_shell**2)
    g_hom = e_act / (R_GAS * t_i)

    s_top_used = _row_s_top(theta_nom, s_top_inf, c_theta)
    rheo = SolidRheologyParams(
        enabled=True,
        g_homologous=g_hom,
        t_sol_0=t_i,
        dt_sol_dp=0.0,
        arrhenius_t_ref=t_i,
        s_top_inf=s_top_used,
        c_theta=0.0,
        mlt_bottom_slope=1.0,
        yield_stress_c=1.0e30,
        yield_stress_max=1.0e30,
        stress_closure_mode='local',
    )
    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=5,
        outer_boundary_value=t_s,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=t_s,
        core_heat_capacity=880.0,
        core_bc='quasi_steady',
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=True,
        tidal=False,
        solver_method='cvode',
        use_jax_jacobian=False,
        kappah_floor=0.0,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=t_i, basal_temperature=t_i
    )
    mesh = _MeshParameters(
        outer_radius=r_out,
        inner_radius=r_out - d_shell,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=rho,
        eos_method=1,
        surface_density=rho,
        gravitational_acceleration=g,
        mass_coordinates=False,
    )
    common = dict(
        density=rho, heat_capacity=cp, thermal_conductivity=k_cond, thermal_expansivity=alpha
    )
    pl = _PhaseParameters(melt_fraction=1.0, viscosity=1.0e2, **common)
    ps = _PhaseParameters(melt_fraction=0.0, viscosity=eta_i, rheology=rheo, **common)
    pm = _PhaseMixedParameters(
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
        const_T_ref=t_i,
        const_S_ref=3000.0,
    )
    sv = _SolverParameters(start_time=0.0, end_time=4.0e10, atol=1.0e-6, rtol=1.0e-6)
    rn = _Radionuclide(
        name='const',
        t0_years=0.0,
        abundance=1.0,
        concentration=1.0e6,
        heat_production=h_heat,
        half_life_years=1.0e30,
    )
    p = Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[rn],
        solver=sv,
    )
    meta = dict(
        D=d_shell,
        F=scale_f,
        eta_i=eta_i,
        H=h_heat,
        kappa=kappa,
        T_s=t_s,
        dT=dt_k,
        E=e_act,
        theta=theta_nom,
        Ra_i=ra_i,
        Nu_ref=nu_ref,
        s_top_used=s_top_used,
    )
    return p, meta


def run_korenaga_row(
    row_name: str,
    n_nodes: int,
    s_top_inf: float,
    c_theta: float,
) -> dict[str, Any]:
    """Run one Korenaga (2009, Table 1) row to steady state and return Nu and residual."""
    _install_benchmark_patches()
    _FK_STATE.clear()
    try:
        t0 = time.time()
        chunks = 8
        for t_end in (4.0e10, 8.0e10, 1.6e11, 3.2e11):
            p, meta = build_korenaga_params(row_name, n_nodes, s_top_inf, c_theta)
            s = EntropySolver(p, entropy_eos=None)
            s.initialize()
            s.set_initial_entropy(3000.0)
            edges = np.linspace(0.0, t_end, chunks + 1)
            hist: list[tuple[float, float, float, float, int]] = []
            for chunk_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                p.solver.start_time, p.solver.end_time = float(lo), float(hi)
                s.solve()
                if chunk_idx < chunks - 2:
                    s.set_initial_entropy(
                        np.asarray(s._solution.y[: s._n_stag, -1], dtype=float)
                    )
                    continue
                st = s.get_state()
                s.set_initial_entropy(st.S_final)
                t_stag = np.asarray(st.T_stag, float)
                rb, rs = np.asarray(st.r_basic, float), np.asarray(st.r_stag, float)
                k_cond = 4.0 * meta['F']
                f_top = k_cond * (t_stag[-1] - meta['T_s']) / (rb[-1] - rs[-1])
                dth = 4000.0 * meta['H'] * meta['D'] ** 2 / k_cond
                ti_val = korenaga_ti_fixed_point(
                    t_stag, (rb[-1] - rs) / meta['D'], meta['T_s'], dth, tol_k=1.0e-9
                )
                hist.append(
                    (
                        float(hi),
                        float(ti_val),
                        float(dth / (ti_val - meta['T_s'])),
                        float(f_top / (4000.0 * meta['H'] * meta['D'])),
                        int(st.status),
                    )
                )
            if abs(hist[-1][2] - hist[-2][2]) < 1.0e-5 * hist[-1][2]:
                break
        else:
            raise RuntimeError(f'Korenaga (2009) row {row_name} did not reach steady state')
    finally:
        _restore_benchmark_patches()

    wall_s = time.time() - t0
    nu_model = float(hist[-1][2])
    nu_ref = float(meta['Nu_ref'])
    rel_err = float(nu_model / nu_ref - 1.0)
    r_norm = float(rel_err / KORENAGA_NU_REL_TOL)
    e_kj, dt_k, th_val, rai_val, _ = KORENAGA_2009_ROWS[row_name]
    return {
        'row': row_name,
        'E_kJ_mol': float(e_kj),
        'dT_K': float(dt_k),
        'theta': float(th_val),
        'Ra_i': float(rai_val),
        's_top_used': float(meta['s_top_used']),
        'Nu': nu_model,
        'Nu_ref': nu_ref,
        'Ti': float(hist[-1][1]),
        'F_balance': float(hist[-1][3]),
        'rel_err': rel_err,
        'r_norm': r_norm,
        'abs_r_norm': abs(r_norm),
        'exceeds_1': bool(abs(r_norm) > 1.0),
        'wall_s': wall_s,
    }


def run_deschamps_vilella_row(
    row_name: str,
    n_nodes: int,
    s_top_inf: float,
    c_theta: float,
) -> dict[str, Any]:
    """Run one Deschamps and Vilella (2021, Table 1) spherical row to steady state."""
    _install_benchmark_patches()
    try:
        ra_s, f_core, deta, h_dimless, tm_ref, phi_top_ref, phi_bot_ref, d_lid_ref = (
            DESCHAMPS_VILELLA_2021_ROWS[row_name]
        )
        d_shell = 1.0e6
        r_out = d_shell / (1.0 - f_core)
        rho, cp, k_cond, alpha, g = 4000.0, 1000.0, 4.0, 3.0e-5, 10.0
        t_s, dt_k = 273.0, 1000.0
        t_bot = t_s + dt_k
        kappa = k_cond / (rho * cp)
        eta0 = alpha * rho * g * dt_k * d_shell**3 / (kappa * ra_s)
        gamma = float(np.log(deta) / dt_k)
        theta = float(np.log(deta))
        h_heat = h_dimless * k_cond * dt_k / (rho * d_shell**2)

        _FK_STATE.clear()
        _FK_STATE.update(eta0=eta0, gamma=gamma, T0=t_s)

        s_top_used = _row_s_top(theta, s_top_inf, c_theta)
        g_hom = gamma * t_bot
        rheo = SolidRheologyParams(
            enabled=True,
            g_homologous=g_hom,
            t_sol_0=t_bot,
            dt_sol_dp=0.0,
            arrhenius_t_ref=t_bot,
            s_top_inf=s_top_used,
            c_theta=0.0,
            mlt_bottom_slope=1.0,
            yield_stress_c=1.0e30,
            yield_stress_max=1.0e30,
            stress_closure_mode='local',
        )
        bc = _BoundaryConditionsParameters(
            outer_boundary_condition=5,
            outer_boundary_value=t_s,
            inner_boundary_condition=3,
            inner_boundary_value=t_bot,
            emissivity=1.0,
            equilibrium_temperature=t_s,
            core_heat_capacity=880.0,
            core_bc='quasi_steady',
        )
        en = _EnergyParameters(
            conduction=True,
            convection=True,
            gravitational_separation=False,
            mixing=False,
            radionuclides=True,
            tidal=False,
            solver_method='cvode',
            use_jax_jacobian=False,
            kappah_floor=0.0,
        )
        ic = _InitialConditionParameters(
            initial_condition=1,
            surface_temperature=t_s + 0.9 * dt_k,
            basal_temperature=t_s + 0.9 * dt_k,
        )
        mesh = _MeshParameters(
            outer_radius=r_out,
            inner_radius=r_out - d_shell,
            number_of_nodes=n_nodes,
            mixing_length_profile='nearest_boundary',
            core_density=rho,
            eos_method=1,
            surface_density=rho,
            gravitational_acceleration=g,
            mass_coordinates=False,
        )
        common = dict(
            density=rho,
            heat_capacity=cp,
            thermal_conductivity=k_cond,
            thermal_expansivity=alpha,
        )
        pl = _PhaseParameters(melt_fraction=1.0, viscosity=1.0e2, **common)
        ps = _PhaseParameters(melt_fraction=0.0, viscosity=eta0, rheology=rheo, **common)
        pm = _PhaseMixedParameters(
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
            const_log10visc=float(np.log10(eta0)),
            const_T_ref=t_bot,
            const_S_ref=3000.0,
        )
        t_end = 2.0e11
        chunks = 40
        sv = _SolverParameters(start_time=0.0, end_time=t_end, atol=1.0e-6, rtol=1.0e-6)
        rn = _Radionuclide(
            name='const',
            t0_years=0.0,
            abundance=1.0,
            concentration=1.0e6,
            heat_production=h_heat,
            half_life_years=1.0e30,
        )
        p = Parameters(
            boundary_conditions=bc,
            energy=en,
            initial_condition=ic,
            mesh=mesh,
            phase_solid=ps,
            phase_liquid=pl,
            phase_mixed=pm,
            radionuclides=[rn],
            solver=sv,
        )
        t0 = time.time()
        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        s.set_initial_entropy(3000.0)
        edges = np.linspace(0.0, t_end, chunks + 1)
        for chunk_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            p.solver.start_time, p.solver.end_time = float(lo), float(hi)
            s.solve()
            if chunk_idx < chunks - 1:
                s.set_initial_entropy(np.asarray(s._solution.y[: s._n_stag, -1], dtype=float))
            else:
                st = s.get_state()
                s.set_initial_entropy(st.S_final)

        t_stag = np.asarray(st.T_stag, float)
        rb, rs = np.asarray(st.r_basic, float), np.asarray(st.r_stag, float)
        f_top = k_cond * (t_stag[-1] - t_s) / (rb[-1] - rs[-1])
        s.dSdt(float(t_end), np.asarray(st.S_final, float).ravel())
        f_bot = float(np.asarray(s.state.heat_flux, float).ravel()[0])
    finally:
        _restore_benchmark_patches()
    z = (rb[-1] - rs) / d_shell
    vol = np.diff(rb**3)
    sel = (z > 0.5) & (z < 0.9)
    tm_dim = float(np.sum(t_stag[sel] * vol[sel]) / np.sum(vol[sel]))
    tm_val = float((tm_dim - t_s) / dt_k)
    phi_top_val = float(f_top * d_shell / (k_cond * dt_k))
    phi_bot_val = float(f_bot * d_shell / (k_cond * dt_k))
    jc = np.asarray(st.jconv_b, float)
    zb = (rb[-1] - rb) / d_shell
    order = np.argsort(zb)
    d_lid_val = lid_depth_tangent(jc[order], zb[order])
    wall_s = time.time() - t0

    d_tm = float(tm_val - tm_ref)
    r_tm = float(d_tm / DV2021_TM_ABS_TOL)
    rel_phi = float(phi_top_val / phi_top_ref - 1.0)
    r_phi = float(rel_phi / DV2021_PHI_REL_TOL)
    if np.isfinite(d_lid_val):
        rel_d = float(d_lid_val / d_lid_ref - 1.0)
        r_d = float(rel_d / DV2021_DLID_REL_TOL)
    else:
        rel_d = 2.0
        r_d = 10.0

    abs_tm, abs_phi, abs_d = abs(r_tm), abs(r_phi), abs(r_d)
    max_abs_row = max(abs_tm, abs_phi, abs_d)
    n_res_gt_1 = int(abs_tm > 1.0) + int(abs_phi > 1.0) + int(abs_d > 1.0)
    return {
        'row': row_name,
        'Ra_s': float(ra_s),
        'f': float(f_core),
        'delta_eta': float(deta),
        'H': float(h_dimless),
        'theta': theta,
        's_top_used': s_top_used,
        'Tm': tm_val,
        'Tm_ref': float(tm_ref),
        'dTm': d_tm,
        'r_Tm': r_tm,
        'abs_r_Tm': abs_tm,
        'Phi_top': phi_top_val,
        'Phi_top_ref': float(phi_top_ref),
        'rel_Phi': rel_phi,
        'r_Phi': r_phi,
        'abs_r_Phi': abs_phi,
        'Phi_bot': phi_bot_val,
        'Phi_bot_ref': float(phi_bot_ref),
        'd_lid': d_lid_val,
        'd_lid_ref': float(d_lid_ref),
        'rel_d': rel_d,
        'r_d': r_d,
        'abs_r_d': abs_d,
        'max_abs_r_row': max_abs_row,
        'n_residuals_exceed_1': n_res_gt_1,
        'exceeds_1': bool(max_abs_row > 1.0),
        'wall_s': wall_s,
    }


def _eval_worker(task: tuple[str, str, int, float, float]) -> tuple[str, str, dict[str, Any]]:
    suite, name, n_nodes, s_top_inf, c_theta = task
    if suite == 'korenaga_2009':
        return suite, name, run_korenaga_row(name, n_nodes, s_top_inf, c_theta)
    return suite, name, run_deschamps_vilella_row(name, n_nodes, s_top_inf, c_theta)


def compute_point_summary(
    korenaga_rows: dict[str, dict[str, Any]],
    dv_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics from the 12 Korenaga and 24 Deschamps-Vilella rows."""
    k_list = [korenaga_rows[k] for k in KORENAGA_2009_ROWS]
    dv_list = [dv_rows[k] for k in DESCHAMPS_VILELLA_2021_ROWS]

    k_ranked = sorted(k_list, key=lambda r: abs(float(r['r_norm'])), reverse=True)
    dv_by_tm = sorted(dv_list, key=lambda r: abs(float(r['r_Tm'])), reverse=True)
    dv_by_phi = sorted(dv_list, key=lambda r: abs(float(r['r_Phi'])), reverse=True)
    dv_by_d = sorted(dv_list, key=lambda r: abs(float(r['r_d'])), reverse=True)
    dv_ranked = sorted(dv_list, key=lambda r: float(r['max_abs_r_row']), reverse=True)

    k_sq = [float(r['r_norm']) ** 2 for r in k_list]
    dv_sq: list[float] = []
    n_dv_res_gt_1 = 0
    n_dv_rows_gt_1 = 0
    for r in dv_list:
        rtm, rphi, rd = float(r['r_Tm']), float(r['r_Phi']), float(r['r_d'])
        dv_sq.extend([rtm**2, rphi**2, rd**2])
        n_gt = int(abs(rtm) > 1.0) + int(abs(rphi) > 1.0) + int(abs(rd) > 1.0)
        n_dv_res_gt_1 += n_gt
        if max(abs(rtm), abs(rphi), abs(rd)) > 1.0:
            n_dv_rows_gt_1 += 1

    j_korenaga = float(sum(k_sq) / len(k_sq))
    j_dv = float(sum(dv_sq) / len(dv_sq))
    all_sq = k_sq + dv_sq
    j_mean_sq_84 = float(sum(all_sq) / len(all_sq))

    n_k_gt_1 = int(sum(1 for r in k_list if abs(float(r['r_norm'])) > 1.0))
    max_r_k = float(abs(k_ranked[0]['r_norm']))
    max_r_dv = float(dv_ranked[0]['max_abs_r_row'])

    return {
        'n_korenaga_rows': len(k_list),
        'n_korenaga_residuals': len(k_list),
        'n_korenaga_pass': len(k_list) - n_k_gt_1,
        'n_exceed_1_korenaga': n_k_gt_1,
        'max_abs_r_korenaga': max_r_k,
        'worst_korenaga_row': str(k_ranked[0]['row']),
        'worst_korenaga_r_norm': float(k_ranked[0]['r_norm']),
        'J_korenaga': j_korenaga,
        'n_deschamps_vilella_rows': len(dv_list),
        'n_deschamps_vilella_residuals': 3 * len(dv_list),
        'n_exceed_1_deschamps_vilella': n_dv_res_gt_1,
        'n_exceed_1_deschamps_vilella_rows': n_dv_rows_gt_1,
        'max_abs_r_deschamps_vilella': max_r_dv,
        'max_abs_r_Tm': float(abs(dv_by_tm[0]['r_Tm'])),
        'worst_Tm_row': str(dv_by_tm[0]['row']),
        'worst_Tm_r': float(dv_by_tm[0]['r_Tm']),
        'max_abs_r_Phi': float(abs(dv_by_phi[0]['r_Phi'])),
        'worst_Phi_row': str(dv_by_phi[0]['row']),
        'worst_Phi_r': float(dv_by_phi[0]['r_Phi']),
        'max_abs_r_d': float(abs(dv_by_d[0]['r_d'])),
        'worst_d_row': str(dv_by_d[0]['row']),
        'worst_d_r': float(dv_by_d[0]['r_d']),
        'J_deschamps_vilella': j_dv,
        'n_total_rows': len(k_list) + len(dv_list),
        'n_total_residuals': len(all_sq),
        'n_exceed_1_all': n_k_gt_1 + n_dv_res_gt_1,
        'max_abs_r_all': max(max_r_k, max_r_dv),
        'J_mean_sq_84': j_mean_sq_84,
    }


def evaluate_calibration_point(
    label: str,
    s_top_inf: float,
    c_theta: float,
    n_nodes: int = 100,
) -> dict[str, Any]:
    """Evaluate all 36 benchmark rows at ``(s_top_inf, c_theta)``."""
    tasks: list[tuple[str, str, int, float, float]] = [
        ('korenaga_2009', name, n_nodes, float(s_top_inf), float(c_theta))
        for name in KORENAGA_2009_ROWS
    ] + [
        ('deschamps_vilella_2021', name, n_nodes, float(s_top_inf), float(c_theta))
        for name in DESCHAMPS_VILELLA_2021_ROWS
    ]
    k_rows: dict[str, dict[str, Any]] = {}
    dv_rows: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for idx, task in enumerate(tasks, 1):
        suite, name, res = _eval_worker(task)
        if suite == 'korenaga_2009':
            k_rows[name] = res
        else:
            dv_rows[name] = res
        if idx % 12 == 0:
            print(
                f'  [{label}] {idx}/{len(tasks)} rows complete ({time.time() - t0:.1f}s)',
                flush=True,
            )
    wall_s = time.time() - t0
    summary = compute_point_summary(k_rows, dv_rows)
    return {
        'label': label,
        'git_commit': get_git_commit(),
        's_top_inf': float(s_top_inf),
        'c_theta': float(c_theta),
        'n_nodes': int(n_nodes),
        'wall_total_s': float(wall_s),
        'summary': summary,
        'korenaga_2009_rows': k_rows,
        'deschamps_vilella_2021_rows': dv_rows,
    }


def verify_calibration_json(
    data: dict[str, Any],
    *,
    run_live_spot_check: bool = True,
    repo_root: Path = REPO_ROOT,
    strict_fingerprint: bool = False,
) -> list[str]:
    """Verify every stored field, recomputed summary, and live spot-check in ``data``."""
    failures: list[str] = []
    meta = data.get('metadata', {})
    if not isinstance(meta, dict):
        return ['Missing metadata block']
    failures.extend(
        _check_generator_input_fingerprint(
            meta.get('git_commit'),
            meta.get('generator_input_sha256'),
            MLT_CALIBRATION_INPUT_PATHS,
            recorded_files=meta.get('generator_input_files'),
            repo_root=repo_root,
            regenerate_cmd=(
                'python tools/verification/evaluate_mlt_top_slope_calibration.py --full'
            ),
            strict=strict_fingerprint,
        )
    )

    tols = meta.get('tolerances', {})
    expected_tols = {
        'korenaga_2009_nu_rel_tol': KORENAGA_NU_REL_TOL,
        'korenaga_2009_nu_rel_tol_pct': 10.0,
        'deschamps_vilella_2021_tm_abs_tol': DV2021_TM_ABS_TOL,
        'deschamps_vilella_2021_phi_top_rel_tol': DV2021_PHI_REL_TOL,
        'deschamps_vilella_2021_phi_top_rel_tol_pct': 15.0,
        'deschamps_vilella_2021_d_lid_rel_tol': DV2021_DLID_REL_TOL,
        'deschamps_vilella_2021_d_lid_rel_tol_pct': 20.0,
        'normalised_residual_threshold': 1.0,
    }
    for k, exp_v in expected_tols.items():
        if k not in tols or not math.isclose(
            float(tols[k]), exp_v, rel_tol=1e-12, abs_tol=1e-15
        ):
            failures.append(
                f'metadata.tolerances.{k}: stored={tols.get(k)} != expected={exp_v}'
            )

    srp = SolidRheologyParams()
    defaults_meta = meta.get('parameter_defaults', {})
    if not math.isclose(
        float(defaults_meta.get('s_top_inf', -1.0)), float(srp.s_top_inf), rel_tol=1e-12
    ):
        failures.append('metadata.parameter_defaults.s_top_inf mismatch')
    if not math.isclose(
        float(defaults_meta.get('c_theta', -1.0)), float(srp.c_theta), rel_tol=1e-12
    ):
        failures.append('metadata.parameter_defaults.c_theta mismatch')

    pts = data.get('evaluated_points', {})
    for pt_key, (exp_s, exp_c) in (
        ('minimax_0p25_0p25', (0.25, 0.25)),
        ('default_0p22_0p15', (0.22, 0.15)),
    ):
        pt = pts.get(pt_key)
        if not isinstance(pt, dict):
            failures.append(f'Missing evaluated_points.{pt_key}')
            continue
        if not _is_valid_commit_sha(pt.get('git_commit')):
            failures.append(f'{pt_key}.git_commit invalid format: {pt.get("git_commit")!r}')
        if not math.isclose(float(pt.get('s_top_inf', -1.0)), exp_s, rel_tol=1e-12):
            failures.append(f'{pt_key}.s_top_inf: {pt.get("s_top_inf")} != {exp_s}')
        if not math.isclose(float(pt.get('c_theta', -1.0)), exp_c, rel_tol=1e-12):
            failures.append(f'{pt_key}.c_theta: {pt.get("c_theta")} != {exp_c}')

        k_rows = pt.get('korenaga_2009_rows', {})
        dv_rows = pt.get('deschamps_vilella_2021_rows', {})
        if set(k_rows.keys()) != set(KORENAGA_2009_ROWS.keys()):
            failures.append(f'{pt_key}.korenaga_2009_rows keys mismatch')
            continue
        if set(dv_rows.keys()) != set(DESCHAMPS_VILELLA_2021_ROWS.keys()):
            failures.append(f'{pt_key}.deschamps_vilella_2021_rows keys mismatch')
            continue

        for rname, (e_kj, dt_k, th_nom, rai, nu_ref) in KORENAGA_2009_ROWS.items():
            row = k_rows[rname]
            exp_stop = _row_s_top(th_nom, exp_s, exp_c)
            if not math.isclose(float(row.get('s_top_used', -1.0)), exp_stop, rel_tol=1e-12):
                failures.append(f'{pt_key}.{rname}.s_top_used mismatch')
            if not math.isclose(float(row.get('Nu_ref', -1.0)), nu_ref, rel_tol=1e-12):
                failures.append(f'{pt_key}.{rname}.Nu_ref mismatch')
            nu_val = float(row.get('Nu', -1.0))
            rel_err_calc = nu_val / nu_ref - 1.0
            r_norm_calc = rel_err_calc / KORENAGA_NU_REL_TOL
            if not math.isclose(
                float(row.get('rel_err', -999.0)), rel_err_calc, rel_tol=1e-11, abs_tol=1e-14
            ):
                failures.append(f'{pt_key}.{rname}.rel_err mismatch')
            if not math.isclose(
                float(row.get('r_norm', -999.0)), r_norm_calc, rel_tol=1e-11, abs_tol=1e-14
            ):
                failures.append(f'{pt_key}.{rname}.r_norm mismatch')

        for rname, (
            ra_s,
            f_c,
            deta,
            h_d,
            tm_ref,
            phi_ref,
            _,
            dlid_ref,
        ) in DESCHAMPS_VILELLA_2021_ROWS.items():
            row = dv_rows[rname]
            th_val = float(np.log(deta))
            exp_stop = _row_s_top(th_val, exp_s, exp_c)
            if not math.isclose(float(row.get('s_top_used', -1.0)), exp_stop, rel_tol=1e-12):
                failures.append(f'{pt_key}.{rname}.s_top_used mismatch')
            tm_val = float(row.get('Tm', -999.0))
            phi_val = float(row.get('Phi_top', -999.0))
            dlid_val = float(row.get('d_lid', -999.0))
            r_tm_calc = (tm_val - tm_ref) / DV2021_TM_ABS_TOL
            r_phi_calc = (phi_val / phi_ref - 1.0) / DV2021_PHI_REL_TOL
            r_d_calc = (dlid_val / dlid_ref - 1.0) / DV2021_DLID_REL_TOL
            if not math.isclose(
                float(row.get('r_Tm', -999.0)), r_tm_calc, rel_tol=1e-11, abs_tol=1e-14
            ):
                failures.append(f'{pt_key}.{rname}.r_Tm mismatch')
            if not math.isclose(
                float(row.get('r_Phi', -999.0)), r_phi_calc, rel_tol=1e-11, abs_tol=1e-14
            ):
                failures.append(f'{pt_key}.{rname}.r_Phi mismatch')
            if not math.isclose(
                float(row.get('r_d', -999.0)), r_d_calc, rel_tol=1e-11, abs_tol=1e-14
            ):
                failures.append(f'{pt_key}.{rname}.r_d mismatch')

        recomp_sum = compute_point_summary(k_rows, dv_rows)
        stored_sum = pt.get('summary', {})
        for sk, exp_sv in recomp_sum.items():
            if sk not in stored_sum:
                failures.append(f'{pt_key}.summary missing {sk}')
                continue
            st_v = stored_sum[sk]
            if isinstance(exp_sv, str):
                if str(st_v) != exp_sv:
                    failures.append(f'{pt_key}.summary.{sk}: stored={st_v!r} != {exp_sv!r}')
            elif isinstance(exp_sv, int):
                if int(st_v) != exp_sv:
                    failures.append(f'{pt_key}.summary.{sk}: stored={st_v} != {exp_sv}')
            else:
                if not math.isclose(float(st_v), float(exp_sv), rel_tol=1e-11, abs_tol=1e-14):
                    failures.append(f'{pt_key}.summary.{sk}: stored={st_v} != {exp_sv}')

        if run_live_spot_check:
            n_nodes_pt = int(pt.get('n_nodes', 100))
            live_e340 = run_korenaga_row('E340', n_nodes_pt, exp_s, exp_c)
            st_e340 = k_rows['E340']
            if not math.isclose(float(st_e340['Nu']), float(live_e340['Nu']), rel_tol=1e-6):
                failures.append(
                    f'{pt_key}.E340 live Nu mismatch: stored={st_e340["Nu"]} vs live={live_e340["Nu"]}'
                )
            if not math.isclose(
                float(st_e340['r_norm']), float(live_e340['r_norm']), rel_tol=1e-5
            ):
                failures.append(
                    f'{pt_key}.E340 live r_norm mismatch: stored={st_e340["r_norm"]} vs live={live_e340["r_norm"]}'
                )

            live_dv = run_deschamps_vilella_row('f60_H4', n_nodes_pt, exp_s, exp_c)
            st_dv = dv_rows['f60_H4']
            for dv_fld in ('Tm', 'Phi_top', 'd_lid', 'r_Tm', 'r_Phi', 'r_d'):
                if not math.isclose(
                    float(st_dv[dv_fld]),
                    float(live_dv[dv_fld]),
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                ):
                    failures.append(
                        f'{pt_key}.f60_H4 live {dv_fld} mismatch: stored={st_dv[dv_fld]} vs live={live_dv[dv_fld]}'
                    )

    return failures


def build_full_payload(n_nodes: int = 100) -> dict[str, Any]:
    """Generate the full calibration reference payload for (0.25, 0.25) and (0.22, 0.15)."""
    srp = SolidRheologyParams()
    pt_minimax = evaluate_calibration_point('minimax_0p25_0p25', 0.25, 0.25, n_nodes=n_nodes)
    pt_default = evaluate_calibration_point(
        'default_0p22_0p15',
        float(srp.s_top_inf),
        float(srp.c_theta),
        n_nodes=n_nodes,
    )
    input_sha, input_files = compute_generator_input_fingerprint(
        MLT_CALIBRATION_INPUT_PATHS, repo_root=REPO_ROOT
    )
    return {
        'metadata': {
            'description': (
                'Viscous mixing-length top-slope s_top(theta) benchmark evaluation across '
                '12 planar internally heated Newtonian cases (Korenaga 2009, Table 1) and '
                '24 spherical internally and mixed heated cases (Deschamps and Vilella 2021, Table 1).'
            ),
            'generator': 'tools/verification/evaluate_mlt_top_slope_calibration.py',
            'git_commit': get_git_commit(),
            'generator_input_sha256': input_sha,
            'generator_input_files': input_files,
            'parameter_defaults': {
                's_top_inf': float(srp.s_top_inf),
                'c_theta': float(srp.c_theta),
                'mlt_bottom_slope': float(srp.mlt_bottom_slope),
            },
            'tolerances': {
                'korenaga_2009_nu_rel_tol': KORENAGA_NU_REL_TOL,
                'korenaga_2009_nu_rel_tol_pct': 10.0,
                'deschamps_vilella_2021_tm_abs_tol': DV2021_TM_ABS_TOL,
                'deschamps_vilella_2021_phi_top_rel_tol': DV2021_PHI_REL_TOL,
                'deschamps_vilella_2021_phi_top_rel_tol_pct': 15.0,
                'deschamps_vilella_2021_d_lid_rel_tol': DV2021_DLID_REL_TOL,
                'deschamps_vilella_2021_d_lid_rel_tol_pct': 20.0,
                'normalised_residual_threshold': 1.0,
            },
        },
        'evaluated_points': {
            'minimax_0p25_0p25': pt_minimax,
            'default_0p22_0p15': pt_default,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUT,
        help='Path to ssc_mlt_top_slope_calibration.json.',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify existing JSON against recomputed summaries and live E340 solves.',
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Evaluate all 36 benchmark rows at (0.25, 0.25) and (0.22, 0.15) and write JSON.',
    )
    parser.add_argument('--n-nodes', type=int, default=100)
    args = parser.parse_args(argv)

    if args.check:
        if not args.output.is_file():
            print(f'ERROR: benchmark file not found: {args.output}', file=sys.stderr)
            return 1
        data = json.loads(args.output.read_text(encoding='utf-8'))
        failures = verify_calibration_json(data, run_live_spot_check=True)
        if failures:
            for msg in failures:
                print(f'CHECK FAILED: {msg}', file=sys.stderr)
            return 1
        print(f'CHECK PASSED: {args.output} verified.')
        return 0

    try:
        _resolve_origin_head_commit(REPO_ROOT)
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    if not args.full and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding='utf-8'))
        existing_meta = existing.get('metadata', {})
        fp_errs = _check_generator_input_fingerprint(
            existing_meta.get('git_commit'),
            existing_meta.get('generator_input_sha256'),
            MLT_CALIBRATION_INPUT_PATHS,
            recorded_files=existing_meta.get('generator_input_files'),
            repo_root=REPO_ROOT,
            regenerate_cmd=(
                'python tools/verification/evaluate_mlt_top_slope_calibration.py --full'
            ),
            strict=True,
        )
        if fp_errs:
            for err in fp_errs:
                print(err, file=sys.stderr)
            return 1

    payload = build_full_payload(n_nodes=args.n_nodes)
    failures = verify_calibration_json(
        payload, run_live_spot_check=False, strict_fingerprint=True
    )
    if failures:
        for msg in failures:
            print(f'INTERNAL VERIFICATION FAILED: {msg}', file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp_path, args.output)
    print(f'Wrote verified calibration benchmarks to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
