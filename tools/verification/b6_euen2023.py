"""Verification harness for Benchmark B6: Euen et al. (2023) thermal convection.

This module evaluates the 1-D radial mixing-length convection solver aragog
against the 3-D spherical shell benchmark solutions of Euen et al. (2023),
Geosci. Model Dev., 16, 3221-3239, doi:10.5194/gmd-16-3221-2023.

Benchmark cases evaluated:
- Case A1: Ra = 7e3, Delta_eta = 1 (isoviscous)
- Case A3: Ra = 7e3, Delta_eta = 20
- Case A7: Ra = 7e3, Delta_eta = 1e5 (stagnant lid)
- Case C1: Ra = 1e5, Delta_eta = 1 (isoviscous)
- Case C2: Ra = 1e5, Delta_eta = 10
- Case C3: Ra = 1e5, Delta_eta = 30
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import aragog

_REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path(aragog.__file__).resolve().is_relative_to(_REPO_ROOT), (
    f'aragog imported from unexpected location: {aragog.__file__}'
)

from aragog import rheology  # noqa: E402
from aragog.parser import (  # noqa: E402
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _SolverParameters,
)
from aragog.solver import SECS_PER_YEAR  # noqa: E402
from aragog.solver import entropy_solver as es_mod  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402

_ORIG_CVODE = es_mod.EntropySolver._solve_cvode
_ORIG_ARRHENIUS = rheology.compute_arrhenius_viscosity

FK: dict[str, Any] = {}


def _install_fk() -> None:
    """Install Frank-Kamenetskii viscosity benchmark patch."""
    if getattr(rheology.compute_arrhenius_viscosity, '_is_fk_patch', False):
        return

    def fk(temperature: Any, pressure: Any, *a: Any, xp: Any = np, **k: Any) -> Any:
        if not FK:
            return _ORIG_ARRHENIUS(temperature, pressure, *a, xp=xp, **k)
        res = FK['eta0'] * xp.exp(
            -FK['gamma'] * (xp.asarray(temperature, dtype=float) - FK['T0'])
        ) + 0.0 * xp.asarray(pressure, dtype=float)
        if xp is np and np.ndim(temperature) == 0 and np.ndim(pressure) == 0:
            return float(res)
        return res

    fk._is_fk_patch = True
    rheology.compute_arrhenius_viscosity = fk


def restore_fk() -> None:
    """Restore the unpatched Arrhenius viscosity function."""
    rheology.compute_arrhenius_viscosity = _ORIG_ARRHENIUS
    FK.clear()


def lift_step_cap(max_step_yr: float) -> None:
    """Lift the CVODE step cap in constant-properties mode."""
    restore_step_cap()

    def patched(self: Any, *a: Any, **k: Any) -> Any:
        k['max_step'] = max_step_yr / self._t_ref_yr if np.isfinite(max_step_yr) else np.inf
        return _ORIG_CVODE(self, *a, **k)

    patched._is_lifted_patch = True
    es_mod.EntropySolver._solve_cvode = patched


def restore_step_cap() -> None:
    """Restore the unpatched EntropySolver._solve_cvode method."""
    es_mod.EntropySolver._solve_cvode = _ORIG_CVODE


# Benchmark reference values from Euen et al. (2023) Tables 5, 7, 8, 9, 10
BENCHMARK_CASES = {
    'A1': {
        'Ra': 7000.0,
        'E': 0.0,
        'aspect': {'mean_T': 0.215594, 'Nu_top': 3.496530, 'Nu_bottom': 3.496210},
        'citcoms': {'mean_T': 0.215568, 'Nu_top': 3.498330, 'Nu_bottom': 3.497110},
    },
    'A3': {
        'Ra': 7000.0,
        'E': math.log(20.0),
        'aspect': {'mean_T': 0.241552, 'Nu_top': 3.154130, 'Nu_bottom': 3.153510},
        'citcoms': {'mean_T': 0.241432, 'Nu_top': 3.154510, 'Nu_bottom': 3.153400},
    },
    'A7': {
        'Ra': 7000.0,
        'E': math.log(1.0e5),
        'aspect': {'mean_T': 0.515379, 'Nu_top': 2.826770, 'Nu_bottom': 2.833820},
        'citcoms': {'mean_T': 0.511768, 'Nu_top': 2.800620, 'Nu_bottom': 2.797030},
    },
    'C1': {
        'Ra': 100000.0,
        'E': 0.0,
        'aspect': {'mean_T': 0.171155, 'Nu_top': 7.814170, 'Nu_bottom': 7.801320},
        'citcoms': {'mean_T': 0.171024, 'Nu_top': 7.838360, 'Nu_bottom': 7.800320},
    },
    'C2': {
        'Ra': 100000.0,
        'E': math.log(10.0),
        'aspect': {'mean_T': 0.188114, 'Nu_top': 7.131370, 'Nu_bottom': 7.063790},
        'citcoms': {'mean_T': 0.195129, 'Nu_top': 6.871260, 'Nu_bottom': 6.841830},
    },
    'C3': {
        'Ra': 100000.0,
        'E': math.log(30.0),
        'aspect': {'mean_T': 0.198035, 'Nu_top': 6.786910, 'Nu_bottom': 6.722170},
        'citcoms': {'mean_T': 0.197531, 'Nu_top': 6.716270, 'Nu_bottom': 6.703680},
    },
}

# Physical scale parameters
SHELL_D = 1.0e6  # 1000 km thickness
RB_ND = 0.55
RT_ND = 1.0
D_ND = RT_ND - RB_ND  # 0.45
R_OUT = (RT_ND / D_ND) * SHELL_D
R_IN = (RB_ND / D_ND) * SHELL_D

RHO = 4000.0
CP = 1000.0
K_COND = 4.0
ALPHA = 3.0e-5
GRAV = 0.001  # chosen so adiabatic variation dT_ad / dT <= 1e-3
KAPPA = K_COND / (RHO * CP)  # 1.0e-6 m^2/s

T_SURF = 500.0  # T = 0 non-dimensional
T_CMB = 1500.0  # T = 1 non-dimensional
DELTA_T = T_CMB - T_SURF  # 1000 K
T_MID = T_SURF + 0.5 * DELTA_T  # 1000 K


def compute_ra_ds2000(solver: EntropySolver) -> float:
    """Compute Deschamps and Sotin (2000) interior Rayleigh number."""
    ps = solver.state.phase_staggered
    rho = np.asarray(ps.density()).ravel()
    r_basic = np.asarray(solver._r_basic_flat)
    r_stag = np.asarray(solver._r_stag_flat)
    depth_frac = (r_basic[-1] - r_stag) / (r_basic[-1] - r_basic[0])
    mask = (depth_frac >= 0.25) & (depth_frac <= 0.75)
    weight = np.asarray(solver._volume_flat)[mask] * rho[mask]
    visc_arr = np.asarray(ps.viscosity()).ravel()[mask]
    eta_int = float(np.exp(np.sum(weight * np.log(visc_arr)) / np.sum(weight)))

    k = float(np.asarray(ps.thermal_conductivity()).ravel()[0])
    cp = float(np.asarray(ps.heat_capacity()).ravel()[0])
    alpha = float(np.asarray(ps.thermal_expansivity()).ravel()[0])
    kappa = k / (rho[0] * cp)
    d = float(r_basic[-1] - r_basic[0])
    return float(rho[0] * GRAV * alpha * DELTA_T * d**3 / (kappa * eta_int))


def build_parameters(
    case_name: str,
    n_nodes: int = 100,
    law_on: bool = True,
    convection_on: bool = True,
    t_end_yr: float = 2.0e11,
    ra_override: float | None = None,
    alpha_override: float | None = None,
) -> tuple[Parameters, float]:
    """Build aragog configuration parameters for a benchmark case."""
    info = BENCHMARK_CASES[case_name]
    ra = ra_override if ra_override is not None else info['Ra']
    e_val = info['E']
    alpha_val = alpha_override if alpha_override is not None else ALPHA

    # Viscosity at T=0.5
    eta_mid = (RHO * alpha_val * GRAV * DELTA_T * SHELL_D**3) / (KAPPA * ra)
    FK.clear()
    FK.update(eta0=eta_mid, gamma=e_val / DELTA_T, T0=T_MID)

    cmb_law = 'deschamps_sotin_2000' if law_on else 'none'
    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=5,
        outer_boundary_value=T_SURF,
        inner_boundary_condition=3,
        inner_boundary_value=T_CMB,
        emissivity=1.0,
        equilibrium_temperature=T_SURF,
        core_heat_capacity=880.0,
        core_bc='quasi_steady',
        cmb_flux_law=cmb_law,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=convection_on,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='cvode',
        use_jax_jacobian=False,
        kappah_floor=0.0,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=T_MID, basal_temperature=T_MID
    )
    mesh = _MeshParameters(
        outer_radius=R_OUT,
        inner_radius=R_IN,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=RHO,
        eos_method=1,
        surface_density=RHO,
        gravitational_acceleration=GRAV,
        mass_coordinates=False,
    )
    common = dict(
        density=RHO,
        heat_capacity=CP,
        thermal_conductivity=K_COND,
        thermal_expansivity=alpha_val,
    )
    pl = _PhaseParameters(melt_fraction=1.0, viscosity=1.0e2, **common)
    ps = _PhaseParameters(
        melt_fraction=0.0,
        viscosity=eta_mid,
        enabled=True,
        activation_energy=300.0e3,  # required positive value for parameter validation
        activation_volume=0.0,
        arrhenius_t_ref=T_MID,
        yield_stress_c=1.0e30,
        yield_stress_max=1.0e30,
        **common,
    )
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
        const_rho=RHO,
        const_Cp=CP,
        const_alpha=alpha_val,
        const_cond=K_COND,
        const_log10visc=float(np.log10(eta_mid)),
        const_T_ref=T_MID,
        const_S_ref=3000.0,
    )
    sv = _SolverParameters(start_time=0.0, end_time=t_end_yr, atol=1.0e-6, rtol=1.0e-6)
    p = Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        solver=sv,
    )
    p.phase_solid.rheology = dataclasses.replace(
        p.phase_solid.rheology, mlt_top_slope=0.22, mlt_bottom_slope=1.0
    )
    return p, eta_mid


def run_simulation(
    case_name: str,
    n_nodes: int = 100,
    law_on: bool = True,
    convection_on: bool = True,
    t_end_yr: float = 2.0e11,
    n_chunks: int = 10,
    ra_override: float | None = None,
    alpha_override: float | None = None,
) -> dict[str, Any]:
    """Run a single benchmark case to steady state with chunked progression."""
    _install_fk()
    lift_step_cap(np.inf)

    p, eta_mid = build_parameters(
        case_name,
        n_nodes=n_nodes,
        law_on=law_on,
        convection_on=convection_on,
        t_end_yr=t_end_yr,
        ra_override=ra_override,
        alpha_override=alpha_override,
    )
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    s.set_initial_entropy(3000.0)

    t0 = time.time()
    edges = np.linspace(0.0, t_end_yr, n_chunks + 1)
    mean_t_history = []
    total_steps = 0

    for lo, hi in zip(edges[:-1], edges[1:]):
        p.solver.start_time, p.solver.end_time = float(lo), float(hi)
        s.solve()
        st = s.get_state()
        s.set_initial_entropy(st.S_final)
        total_steps += int(s._solution.get('cvode_nst', 0))

        ts = np.asarray(st.T_stag, float)
        rb = np.asarray(st.r_basic, float)
        vols = np.diff(rb**3)
        mean_t = float(np.sum(ts * vols) / np.sum(vols))
        mean_t_history.append(mean_t)

    wall_s = time.time() - t0
    final_st = s.get_state()

    # Steady-state check 1: relative change of nondimensional mean T over last 10%
    mean_t_nd_history = [(t - T_SURF) / DELTA_T for t in mean_t_history]
    rel_dt_last10 = abs(mean_t_nd_history[-1] - mean_t_nd_history[-2]) / mean_t_nd_history[-1]

    # Non-dimensional mean temperature
    mean_t_nd = mean_t_nd_history[-1]

    # Rheology verification: fk_check
    vs = np.asarray(final_st.visc_stag, float)
    fk_ref = FK['eta0'] * np.exp(-FK['gamma'] * (ts - FK['T0']))
    fk_check = float(np.max(np.abs(np.log(vs / fk_ref))))

    # Boundary heat fluxes
    # (a) Instantaneous from dSdt at final state
    s._dSdt_single(float(s._solution.t[-1]), np.asarray(s._solution.y[:, -1], dtype=float))
    f_top_inst = float(s.state.heat_flux[-1])
    f_bot_inst = float(s.state.heat_flux[0])

    # Conductive flux normalizations
    f_cond_top = (K_COND * DELTA_T / SHELL_D) * (RB_ND / RT_ND)
    f_cond_bot = (K_COND * DELTA_T / SHELL_D) * (RT_ND / RB_ND)

    nu_top_inst = f_top_inst / f_cond_top
    nu_bot_inst = f_bot_inst / f_cond_bot

    # (b) Time-integrated flux over final chunk trajectory via dual-flux energy integrals
    sol = s._solution
    rb = np.asarray(final_st.r_basic, float)
    a_int = 4.0 * np.pi * float(rb[-1]) ** 2
    a_cmb = 4.0 * np.pi * float(rb[0]) ** 2
    dt_chunk_s = (
        float(sol.t[-1] - sol.t[0]) * SECS_PER_YEAR
        if sol is not None and sol.t is not None and len(sol.t) >= 2
        else 0.0
    )
    if dt_chunk_s > 0.0 and final_st.step_dE_F_int_J != 0.0:
        f_top_avg = -float(final_st.step_dE_F_int_J) / (a_int * dt_chunk_s)
        f_bot_avg = float(final_st.step_dE_F_cmb_J) / (a_cmb * dt_chunk_s)
        nu_top_int = f_top_avg / f_cond_top
        nu_bot_int = f_bot_avg / f_cond_bot
    else:
        f_top_avg = f_top_inst
        f_bot_avg = f_bot_inst
        nu_top_int = nu_top_inst
        nu_bot_int = nu_bot_inst

    dual_flux_error_top = abs(nu_top_int - nu_top_inst) / nu_top_inst
    dual_flux_error_bot = abs(nu_bot_int - nu_bot_inst) / nu_bot_inst

    # Steady-state check 2: energy balance
    balance_error = abs(nu_top_inst - nu_bot_inst) / nu_top_inst

    converged = (rel_dt_last10 < 1.0e-6) and (balance_error <= 1.0e-3)
    status_str = 'CONVERGED' if converged else 'NOT CONVERGED'

    # Boussinesq metric
    dt_ad = ALPHA * GRAV * T_MID * SHELL_D / CP
    dt_ad_ratio = dt_ad / DELTA_T

    # DS2000 Rayleigh number
    ra_ds2000 = compute_ra_ds2000(s) if convection_on else 0.0

    return {
        'case': case_name,
        'n_nodes': n_nodes,
        'law_on': law_on,
        'convection_on': convection_on,
        'mean_T_nd': mean_t_nd,
        'mean_T_K': mean_t_history[-1],
        'f_top_inst': f_top_inst,
        'f_bot_inst': f_bot_inst,
        'f_top_avg': f_top_avg,
        'f_bot_avg': f_bot_avg,
        'Nu_top': nu_top_inst,
        'Nu_bottom': nu_bot_inst,
        'Nu_top_integral': nu_top_int,
        'Nu_bottom_integral': nu_bot_int,
        'dual_flux_error_top': dual_flux_error_top,
        'dual_flux_error_bot': dual_flux_error_bot,
        'balance_error': balance_error,
        'rel_dT_last10': rel_dt_last10,
        'status': status_str,
        'wall_s': wall_s,
        'steps': total_steps,
        'fk_check': fk_check,
        'dT_ad_ratio': dt_ad_ratio,
        'Ra_ds2000': ra_ds2000,
    }


def run_canaries(out_dir: Path) -> dict[str, Any]:
    """Execute verification canaries and save b6_canary.json."""
    print('=== Running Verification Canaries ===')

    # 1. Conduction limit canary: sub-critical buoyancy with MLT on recovers pure conduction
    print('1. Conduction limit canary (small buoyancy limit with convection ON)...')
    res_cond = run_simulation(
        'A1', n_nodes=100, law_on=False, convection_on=True, t_end_yr=2.0e11, ra_override=1.0
    )
    analytical_mean_t = 77.0 / 247.0
    mean_t_diff = abs(res_cond['mean_T_nd'] - analytical_mean_t)
    cond_pass = (
        abs(res_cond['Nu_top'] - 1.0) <= 1.0e-3
        and abs(res_cond['Nu_bottom'] - 1.0) <= 1.0e-3
        and mean_t_diff <= 1.0e-4
    )
    print(
        f'   Conduction Nu_top={res_cond["Nu_top"]:.6f}, Nu_bot={res_cond["Nu_bottom"]:.6f}, '
        f'<T>_nd={res_cond["mean_T_nd"]:.6f} (exact {analytical_mean_t:.6f}, diff={mean_t_diff:.2e}) -> '
        f'{"PASS" if cond_pass else "FAIL"}'
    )

    # 2. Negative canary: planar flux formula on actual spherical conduction fluxes
    print('2. Negative canary (planar flux normalization on simulation fluxes)...')
    f_planar = K_COND * DELTA_T / SHELL_D
    nu_top_planar = res_cond['f_top_inst'] / f_planar
    nu_bot_planar = res_cond['f_bot_inst'] / f_planar
    planar_failed = (abs(nu_top_planar - 1.0) > 0.1) and (abs(nu_bot_planar - 1.0) > 0.1)
    print(
        f'   Planar Nu_top={nu_top_planar:.4f}, Nu_bot={nu_bot_planar:.4f} '
        f'(expected mismatch detected: {planar_failed}) -> {"PASS" if planar_failed else "FAIL"}'
    )

    # 3. Step-cap lift canary
    print('3. Step-cap lift canary (step count discrimination and solution invariance)...')

    def run_interval(lift: bool) -> tuple[int, float]:
        if lift:
            lift_step_cap(np.inf)
        else:
            restore_step_cap()
        try:
            p, _ = build_parameters(
                'A1', n_nodes=50, law_on=True, convection_on=True, t_end_yr=20000.0
            )
            s = EntropySolver(p, entropy_eos=None)
            s.initialize()
            s.set_initial_entropy(3000.0)
            s.solve()
            st = s.get_state()
            steps = int(s._solution.get('cvode_nst', 0))
            ts = np.asarray(st.T_stag, float)
            rb = np.asarray(st.r_basic, float)
            vols = np.diff(rb**3)
            mean_t = float(np.sum(ts * vols) / np.sum(vols))
            return steps, mean_t
        finally:
            restore_step_cap()

    steps_lifted, tm_lifted = run_interval(True)
    steps_default, tm_default = run_interval(False)
    lift_rel_diff = abs(tm_lifted - tm_default) / tm_default
    step_reduction_pass = steps_default > steps_lifted
    lift_pass = (lift_rel_diff <= 1.0e-4) and step_reduction_pass
    print(
        f'   Lifted Tm={tm_lifted:.8f} K ({steps_lifted} steps), '
        f'Default Tm={tm_default:.8f} K ({steps_default} steps), '
        f'rel_diff={lift_rel_diff:.2e} -> {"PASS" if lift_pass else "FAIL"}'
    )

    canary_data = {
        'conduction_canary': {
            'status': 'PASS' if cond_pass else 'FAIL',
            'mean_T_nd': res_cond['mean_T_nd'],
            'analytical_mean_T_nd': analytical_mean_t,
            'mean_T_diff': mean_t_diff,
            'Nu_top': res_cond['Nu_top'],
            'Nu_bottom': res_cond['Nu_bottom'],
            'balance_error': res_cond['balance_error'],
        },
        'planar_negative_canary': {
            'status': 'PASS' if planar_failed else 'FAIL',
            'Nu_top_planar': nu_top_planar,
            'Nu_bottom_planar': nu_bot_planar,
            'expected_mismatch_detected': planar_failed,
        },
        'step_cap_lift_canary': {
            'status': 'PASS' if lift_pass else 'FAIL',
            'steps_lifted': steps_lifted,
            'steps_default': steps_default,
            'tm_lifted_K': tm_lifted,
            'tm_default_K': tm_default,
            'rel_diff': lift_rel_diff,
            'limit': 1.0e-4,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'b6_canary.json', 'w') as f:
        json.dump(canary_data, f, indent=2)
    print(f'Wrote {out_dir / "b6_canary.json"}')
    restore_fk()
    restore_step_cap()
    return canary_data


def compute_richardson(values: dict[int, float]) -> tuple[float, float]:
    """Compute Richardson extrapolation estimate and empirical order.

    Takes values dictionary keyed by node count {50, 100, 200, 400}.
    """
    if set([100, 200, 400]).issubset(values.keys()):
        f1 = values[400]
        f2 = values[200]
        f3 = values[100]
        diff21 = f2 - f1
        diff32 = f3 - f2
        if diff21 != 0.0 and (diff32 / diff21) > 0.0:
            p = math.log(abs(diff32 / diff21)) / math.log(2.0)
            if 0.5 <= p <= 4.0:
                extrap = f1 + (f1 - f2) / (2.0**p - 1.0)
                return extrap, p
        # Default to second order (p = 2) if monotonic ratio is unavailable
        extrap = f1 + (f1 - f2) / 3.0
        return extrap, 2.0
    return values.get(400, float('nan')), 2.0


def run_benchmark_suite(out_dir: Path) -> None:
    """Execute all benchmark runs, compute convergence, and write tables."""
    run_canaries(out_dir)

    print('\n=== Running B6 Benchmark Matrix ===')
    cases = ['A1', 'A3', 'A7', 'C1', 'C2', 'C3']
    node_counts = [50, 100, 200, 400]

    runs_records: list[dict[str, Any]] = []

    # Primary runs: law ON at n = 50, 100, 200, 400
    for case in cases:
        for n in node_counts:
            print(f'Running case {case} (law ON, n={n})...', flush=True)
            res = run_simulation(case, n_nodes=n, law_on=True, convection_on=True)
            runs_records.append(res)
            print(
                f'   status={res["status"]} Tm_nd={res["mean_T_nd"]:.4f} '
                f'Nu_top={res["Nu_top"]:.3f} Nu_bot={res["Nu_bottom"]:.3f} '
                f'bal={res["balance_error"]:.2e} wall={res["wall_s"]:.1f}s'
            )

    # Sensitivity runs: law OFF at n = 100
    for case in cases:
        print(f'Running case {case} sensitivity (law OFF, n=100)...', flush=True)
        res = run_simulation(case, n_nodes=100, law_on=False, convection_on=True)
        runs_records.append(res)
        print(
            f'   status={res["status"]} Tm_nd={res["mean_T_nd"]:.4f} '
            f'Nu_top={res["Nu_top"]:.3f} Nu_bot={res["Nu_bottom"]:.3f} '
            f'bal={res["balance_error"]:.2e} wall={res["wall_s"]:.1f}s'
        )

    # Write b6_runs.csv
    csv_path = out_dir / 'b6_runs.csv'
    fieldnames = [
        'case',
        'n_nodes',
        'law_on',
        'convection_on',
        'mean_T_nd',
        'mean_T_K',
        'Nu_top',
        'Nu_bottom',
        'Nu_top_integral',
        'Nu_bottom_integral',
        'dual_flux_error_top',
        'dual_flux_error_bot',
        'balance_error',
        'rel_dT_last10',
        'status',
        'wall_s',
        'steps',
        'fk_check',
        'dT_ad_ratio',
        'Ra_ds2000',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in runs_records:
            writer.writerow(r)
    print(f'\nWrote {csv_path}')

    # Compute convergence and Richardson extrapolation
    print('\n=== Computing Richardson Extrapolation and Convergence ===')
    conv_records: list[dict[str, Any]] = []

    for case in cases:
        case_runs = [r for r in runs_records if r['case'] == case and r['law_on']]
        t_vals = {r['n_nodes']: r['mean_T_nd'] for r in case_runs}
        nut_vals = {r['n_nodes']: r['Nu_top'] for r in case_runs}
        nub_vals = {r['n_nodes']: r['Nu_bottom'] for r in case_runs}

        t_extrap, t_order = compute_richardson(t_vals)
        nut_extrap, nut_order = compute_richardson(nut_vals)
        nub_extrap, nub_order = compute_richardson(nub_vals)

        ref_aspect = BENCHMARK_CASES[case]['aspect']
        ref_citcoms = BENCHMARK_CASES[case]['citcoms']

        # Production values at n=100
        p_run = next(r for r in case_runs if r['n_nodes'] == 100)

        # Verification tolerances: mean_T within 0.03 absolute, Nu within 15%
        dt_aspect = abs(p_run['mean_T_nd'] - ref_aspect['mean_T'])
        dnut_aspect = abs(p_run['Nu_top'] - ref_aspect['Nu_top']) / ref_aspect['Nu_top']
        dnub_aspect = (
            abs(p_run['Nu_bottom'] - ref_aspect['Nu_bottom']) / ref_aspect['Nu_bottom']
        )

        pass_case = (dt_aspect <= 0.03) and (dnut_aspect <= 0.15) and (dnub_aspect <= 0.15)
        status = 'PASS' if pass_case else 'FAIL'

        conv_records.append(
            {
                'case': case,
                'status': status,
                'mean_T_n50': t_vals.get(50),
                'mean_T_n100': t_vals.get(100),
                'mean_T_n200': t_vals.get(200),
                'mean_T_n400': t_vals.get(400),
                'mean_T_extrap': t_extrap,
                'mean_T_aspect': ref_aspect['mean_T'],
                'mean_T_citcoms': ref_citcoms['mean_T'],
                'mean_T_diff': dt_aspect,
                'Nu_top_n50': nut_vals.get(50),
                'Nu_top_n100': nut_vals.get(100),
                'Nu_top_n200': nut_vals.get(200),
                'Nu_top_n400': nut_vals.get(400),
                'Nu_top_extrap': nut_extrap,
                'Nu_top_aspect': ref_aspect['Nu_top'],
                'Nu_top_citcoms': ref_citcoms['Nu_top'],
                'Nu_top_diff_pct': dnut_aspect * 100.0,
                'Nu_bottom_n50': nub_vals.get(50),
                'Nu_bottom_n100': nub_vals.get(100),
                'Nu_bottom_n200': nub_vals.get(200),
                'Nu_bottom_n400': nub_vals.get(400),
                'Nu_bottom_extrap': nub_extrap,
                'Nu_bottom_aspect': ref_aspect['Nu_bottom'],
                'Nu_bottom_citcoms': ref_citcoms['Nu_bottom'],
                'Nu_bottom_diff_pct': dnub_aspect * 100.0,
            }
        )
        print(
            f'Case {case}: Status={status} | dT={dt_aspect:.4f} (lim 0.03), '
            f'dNu_top={dnut_aspect * 100:.1f}%, dNu_bot={dnub_aspect * 100:.1f}% (lim 15%)'
        )

    conv_path = out_dir / 'b6_convergence.csv'
    conv_fields = list(conv_records[0].keys())
    with open(conv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=conv_fields)
        writer.writeheader()
        for r in conv_records:
            writer.writerow(r)
    print(f'Wrote {conv_path}')
    restore_fk()
    restore_step_cap()


def main() -> None:
    """Parse arguments and dispatch benchmark tasks."""
    parser = argparse.ArgumentParser(description='Benchmark B6 verification harness')
    parser.add_argument(
        '--canary-only', action='store_true', help='Run verification canaries only'
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('/Users/timlichtenberg/work/ssc-dev-task6/data/WP2'),
        help='Target output directory',
    )
    args = parser.parse_args()

    if args.canary_only:
        run_canaries(args.out_dir)
    else:
        run_benchmark_suite(args.out_dir)


if __name__ == '__main__':
    main()
