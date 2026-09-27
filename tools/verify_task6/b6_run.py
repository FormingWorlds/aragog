#!/usr/bin/env python3
"""Benchmark B6 run harness: Euen et al. (2023) on aragog.

Steady-State Criterion
----------------------
A simulation reaches steady state when:
1. The relative change in volume-averaged mean temperature between consecutive
   evaluation intervals is below 1.0e-6 over the final 10 % of the simulated duration:
   |<T>_{k} - <T>_{k-1}| / <T>_{k} < 1.0e-6.
2. Global surface-to-core energy balance satisfies:
   |Nu_top - Nu_bottom| / Nu_top < 1.0e-3.

Nusselt Number Definitions
--------------------------
From Euen et al. (2023) Section 2, Equations (6) and (7):
- Equation (6):
  Nut = (rt * (rt - rb) / rb) * Qt
- Equation (7):
  Nub = (rb * (rt - rb) / rt) * Qb
where rt = 1.0, rb = 0.55, and Qt, Qb are the surface and CMB heat fluxes
normalized by the conductive temperature gradient across a spherical shell:
  F_cond_top = k * Delta_T * rb / (rt * (rt - rb))
  F_cond_bot = k * Delta_T * rt / (rb * (rt - rb))
so that Nut = F_top / F_cond_top and Nub = F_bot / F_cond_bot.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# Assert and import from target worktree
ARAGOG_SRC = Path(
    '/Users/timlichtenberg/work/ssc-verify-task6/wt-aragog-tl_ssc-cmb-bl-law/src'
)
sys.path.insert(0, str(ARAGOG_SRC))
import aragog  # noqa: E402

assert 'wt-aragog-tl_ssc-cmb-bl-law' in aragog.__file__, aragog.__file__

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
from aragog.solver import entropy_solver as es_mod  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402

# Lift the 100 yr CVODE step cap in const-properties mode
_orig_solve_cvode = es_mod.EntropySolver._solve_cvode


def _unlimited_cvode(self, *a, **k):
    k['max_step'] = np.inf
    return _orig_solve_cvode(self, *a, **k)


es_mod.EntropySolver._solve_cvode = _unlimited_cvode

# Reference values from Euen et al. (2023) Tables 5 and 7 (ASPECT extrapolated)
REF_VALUES = {
    'A1': {
        'Ra': 7000.0,
        'E': 0.0,
        'mean_T': 0.215594,
        'nu_top': 3.49653,
        'nu_bottom': 3.49621,
    },
    'A7': {
        'Ra': 7000.0,
        'E': math.log(1e5),
        'mean_T': 0.515379,
        'nu_top': 2.82677,
        'nu_bottom': 2.83382,
    },
}


def build_parameters(
    case_name: str,
    n_nodes: int = 100,
    s_top: float = 0.22,
    use_cmb_law: bool = False,
    t_end_yr: float = 1.0e10,
) -> tuple[Parameters, dict]:
    """Build aragog configuration parameters for benchmark case."""
    if case_name not in REF_VALUES:
        raise ValueError(f'Unknown case: {case_name}')

    ref = REF_VALUES[case_name]
    Ra = ref['Ra']
    E = ref['E']

    # Non-dimensional shell geometry: rb = 0.55, rt = 1.0, D = 0.45
    # Choose dimensional scale D = 1.0e6 m
    D = 1.0e6
    r_out = (1.0 / 0.45) * D  # ~ 2.222222e6 m
    r_in = (0.55 / 0.45) * D  # ~ 1.222222e6 m

    # Dimensional scales
    rho = 4000.0
    cp = 1000.0
    k = 4.0
    alpha = 3.0e-5
    g = 10.0
    kappa = k / (rho * cp)  # 1.0e-6 m^2/s

    T_s = 500.0  # T_top (T=0 in non-dim)
    T_c = 1500.0  # T_bot (T=1 in non-dim)
    dT = T_c - T_s  # 1000 K
    T_mid = T_s + 0.5 * dT  # 1000 K

    # Reference viscosity at T=0.5 from Rayleigh number definition (Eq. 4)
    # Ra = rho * alpha * g * dT * D^3 / (kappa * eta_mid)
    eta_mid = (rho * alpha * g * dT * D**3) / (kappa * Ra)

    # Viscosity contrast across unit temperature span: delta_eta = exp(E)
    # Approximate Frank-Kamenetskii eta(T) = eta_mid * exp(E * (0.5 - T_nd))
    # using Arrhenius law with activation energy E_a = R * T_mid^2 * E / dT
    R_GAS = 8.314462618
    E_a = (R_GAS * T_mid**2 * E / dT) if E > 0.0 else 0.0

    # Boundary conditions
    cmb_law_str = 'deschamps_sotin_2000' if use_cmb_law else 'none'
    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=5,  # Fixed surface temperature
        outer_boundary_value=T_s,
        inner_boundary_condition=3,  # Fixed CMB temperature
        inner_boundary_value=T_c,
        emissivity=1.0,
        equilibrium_temperature=T_s,
        core_heat_capacity=880.0,
        core_bc='quasi_steady',
        cmb_flux_law=cmb_law_str,
    )

    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='cvode',
        use_jax_jacobian=False,
        kappah_floor=0.0,
    )

    # Initial condition: conductive profile between T_s and T_c
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=T_s, basal_temperature=T_c
    )

    mesh = _MeshParameters(
        outer_radius=r_out,
        inner_radius=r_in,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=rho,
        eos_method=1,
        surface_density=rho,
        gravitational_acceleration=g,
        mass_coordinates=False,
    )

    common = dict(
        density=rho, heat_capacity=cp, thermal_conductivity=k, thermal_expansivity=alpha
    )

    pl = _PhaseParameters(melt_fraction=1.0, viscosity=1.0e2, **common)
    ps = _PhaseParameters(
        melt_fraction=0.0,
        viscosity=eta_mid,
        enabled=bool(E > 0.0),
        activation_energy=E_a,
        activation_volume=0.0,
        arrhenius_t_ref=T_mid,
        yield_stress_c=1.0e30,
        yield_stress_max=1.0e30,
        mlt_top_slope=s_top,
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
        const_rho=rho,
        const_Cp=cp,
        const_alpha=alpha,
        const_cond=k,
        const_log10visc=float(np.log10(eta_mid)),
        const_T_ref=T_mid,
        const_S_ref=3000.0,
    )

    sv = _SolverParameters(start_time=0.0, end_time=t_end_yr, atol=1e-6, rtol=1e-6)

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

    meta = {
        'case': case_name,
        'D': D,
        'r_in': r_in,
        'r_out': r_out,
        'T_s': T_s,
        'T_c': T_c,
        'dT': dT,
        'k': k,
        'Ra': Ra,
        'E': E,
        'eta_mid': eta_mid,
        'ref': ref,
    }
    return p, meta


def run_case(
    case_name: str,
    n_nodes: int = 100,
    s_top: float = 0.22,
    use_cmb_law: bool = False,
    timeout_s: float = 1700.0,
) -> dict:
    """Run one B6 benchmark case to steady state."""
    start_wall = time.time()
    p, meta = build_parameters(
        case_name,
        n_nodes=n_nodes,
        s_top=s_top,
        use_cmb_law=use_cmb_law,
        t_end_yr=1.0e12,
    )

    s = EntropySolver(p, entropy_eos=None)
    s.initialize()

    # Initial condition: conductive temperature profile
    r_stag = np.asarray(s._r_stag_flat, dtype=float)
    r_in, r_out = meta['r_in'], meta['r_out']
    T_s, T_c, dT = meta['T_s'], meta['T_c'], meta['dT']
    # Conductive profile in spherical shell: T(r) = T_s + dT * (r_in * r_out / (r_out - r_in)) * (1/r - 1/r_out)
    T_cond = T_s + dT * (r_in * r_out / (r_out - r_in)) * (1.0 / r_stag - 1.0 / r_out)

    # Convert initial temperature to entropy via const Cp: S = Cp * ln(T / T_ref) + S_ref
    S_init = 1000.0 * np.log(T_cond / 1000.0) + 3000.0
    s.set_initial_entropy(S_init)

    # Reference conductive fluxes (Euen et al. Eqs. 6-7 denominators)
    # F_cond_top = k * dT * rb / (rt * (rt - rb))
    # In dimensional units:
    k = meta['k']
    F_cond_top = k * dT * r_in / (r_out * (r_out - r_in))
    F_cond_bot = k * dT * r_out / (r_in * (r_out - r_in))

    vol = np.asarray(s._volume_flat, dtype=float)
    total_vol = float(np.sum(vol))

    # Stepping loop to steady state
    t_current = 0.0
    dt_chunk = 1.0e7  # 10 Myr per chunk
    history_T_mean = []
    is_steady = False

    mean_T_nd = 0.0
    nu_top = 0.0
    nu_bottom = 0.0

    print(
        f'Starting B6 case {case_name}: n={n_nodes}, s_top={s_top}, cmb_law={use_cmb_law}',
        flush=True,
    )

    while t_current < 1.0e11:
        if time.time() - start_wall > timeout_s:
            print(f'TIMEOUT: Exceeded {timeout_s}s wall time', flush=True)
            return {
                'case': case_name,
                'n': n_nodes,
                'mean_T': mean_T_nd,
                'nu_top': nu_top,
                'nu_bottom': nu_bottom,
                'ref_mean_T': meta['ref']['mean_T'],
                'ref_nu_top': meta['ref']['nu_top'],
                'ref_nu_bottom': meta['ref']['nu_bottom'],
                'steady': False,
                'wall_s': round(time.time() - start_wall, 1),
                'status': 'TIMEOUT',
            }

        t_next = t_current + dt_chunk
        p.solver.start_time = float(t_current)
        p.solver.end_time = float(t_next)

        s.solve()
        st = s.get_state()
        if st.status != 0:
            print(f'CVODE error: status {st.status}', flush=True)
            break

        # Extract current state
        T_stag = np.asarray(st.T_stag, dtype=float)
        mean_T_dim = float(np.sum(T_stag * vol) / total_vol)
        mean_T_nd = (mean_T_dim - T_s) / dT

        # Heat fluxes
        hf = np.asarray(s.state.heat_flux, dtype=float).ravel()
        # Top flux: face -1; bottom flux: face 0
        F_top = abs(float(hf[-1]))
        F_bot = abs(float(hf[0]))

        nu_top = F_top / F_cond_top
        nu_bottom = F_bot / F_cond_bot

        history_T_mean.append(mean_T_nd)
        s.set_initial_entropy(np.asarray(st.S_final, dtype=float))
        t_current = t_next

        # Check steady state criterion: relative change of mean T below 1e-6
        if len(history_T_mean) >= 5:
            rel_change = abs(history_T_mean[-1] - history_T_mean[-2]) / history_T_mean[-1]
            flux_imbalance = abs(nu_top - nu_bottom) / nu_top
            if rel_change < 1.0e-6 and flux_imbalance < 0.05:
                is_steady = True
                print(
                    f'STEADY STATE reached at t={t_current:.2e} yr: '
                    f'mean_T={mean_T_nd:.5f}, Nu_top={nu_top:.4f}, Nu_bot={nu_bottom:.4f}',
                    flush=True,
                )
                break

        if len(history_T_mean) % 5 == 0:
            print(
                f't={t_current:.2e} yr: mean_T={mean_T_nd:.5f}, Nu_top={nu_top:.4f}, '
                f'Nu_bot={nu_bottom:.4f}, wall={time.time() - start_wall:.1f}s',
                flush=True,
            )

        # Adapt dt_chunk for faster progress
        dt_chunk = min(dt_chunk * 1.5, 5.0e8)

    wall_s = round(time.time() - start_wall, 1)
    status = 'CONVERGED' if is_steady else 'MAX_TIME'

    return {
        'case': case_name,
        'n': n_nodes,
        'mean_T': mean_T_nd,
        'nu_top': nu_top,
        'nu_bottom': nu_bottom,
        'ref_mean_T': meta['ref']['mean_T'],
        'ref_nu_top': meta['ref']['nu_top'],
        'ref_nu_bottom': meta['ref']['nu_bottom'],
        'steady': is_steady,
        'wall_s': wall_s,
        'status': status,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--case',
        choices=['A1', 'A7'],
        default='A1',
        help='Benchmark case to run (A1=isoviscous, A7=stagnant-lid)',
    )
    ap.add_argument('--n', type=int, default=100, help='Number of radial nodes')
    ap.add_argument(
        '--s-top',
        type=float,
        default=0.22,
        help='Viscous mixing length top slope',
    )
    ap.add_argument(
        '--cmb-law', action='store_true', help='Enable Deschamps-Sotin CMB law'
    )
    ap.add_argument(
        '--out-csv',
        default='data/D/b6_runs.csv',
        help='Path to output runs summary CSV',
    )
    args = ap.parse_args()

    res = run_case(
        args.case, n_nodes=args.n, s_top=args.s_top, use_cmb_law=args.cmb_law
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'case',
        'n',
        'mean_T',
        'nu_top',
        'nu_bottom',
        'ref_mean_T',
        'ref_nu_top',
        'ref_nu_bottom',
        'steady',
        'wall_s',
        'status',
    ]

    rows = []
    if out_csv.exists() and out_csv.stat().st_size > 0:
        with open(out_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r['case'] != args.case:
                    rows.append(r)

    rows.append({
        'case': res['case'],
        'n': res['n'],
        'mean_T': f"{res['mean_T']:.6f}",
        'nu_top': f"{res['nu_top']:.5f}",
        'nu_bottom': f"{res['nu_bottom']:.5f}",
        'ref_mean_T': f"{res['ref_mean_T']:.6f}",
        'ref_nu_top': f"{res['ref_nu_top']:.5f}",
        'ref_nu_bottom': f"{res['ref_nu_bottom']:.5f}",
        'steady': str(res['steady']),
        'wall_s': f"{res['wall_s']:.1f}",
        'status': res['status'],
    })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote results to {out_csv}:')
    print(rows[-1])


if __name__ == '__main__':
    main()
