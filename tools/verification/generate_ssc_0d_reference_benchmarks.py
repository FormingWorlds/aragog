"""Generate and verify reference benchmarks for 0-D stagnant-lid and yield-thinning tests.

Generates or checks ``tests/reference/ssc_0d_reference_benchmarks.json``.

Prescribed surface temperature boundary condition (``outer_boundary_condition = 5``):
- The shipped entropy solver ignores ``outer_boundary_condition = 5`` today:
  ``EntropySolver._dSdt_single`` (``src/aragog/solver/entropy_solver.py``)
  branches only on ``self._outer_bc_kind == 1`` and ``self._outer_bc_kind == 4``, and
  ``EntropyState.update`` (``src/aragog/solver/entropy_state.py``) copies the
  surface entropy gradient from the adjacent interior node via ``dSdxi[-1] = dSdxi[-2]``.
- For steady-state internally heated benchmarks with prescribed surface temperature
  ``T_surf = 273.0 K`` (Korenaga 2009 Table 1 ``E300`` steady states), this tool applies the
  verification-only context manager ``half_cell_surface_flux``
  (``tools/verification/_ssc_helpers.py``), which sets the top conductive heat flux to the
  temperature-solver BC 5 half-cell formula ``F_top = k * (T_cell - T_surf) / (dr / 2)``
  (whereas the 400 Myr and 4.5 Gyr secular-cooling trajectories use ``outer_boundary_condition = 4``
  with the 0-D surface heat flux updated between segments).
- Because ``half_cell_surface_flux`` sets only ``_heat_flux[-1]`` after ``EntropyState.update``
  returns, the stagnant-lid closure (``_compute_stagnant_lid_state_core`` in
  ``src/aragog/rheology_lid.py``, called inside ``EntropyState.update`` in
  ``src/aragog/solver/entropy_state.py``) reads the extrapolated surface
  temperature ``T_basic[-1]`` (276.20 to 280.41 K on the 3 ``E300`` states) and
  unpatched surface flux (1.75 to 2.66 W/m^2 vs patched 2.08 to
  3.41 W/m^2), so the stored ``E300`` states are steady for a closure fed the
  extrapolated surface temperature and flux. Recomputing the closure with ``T_surf = 273.0 K``
  and the patched flux leaves ``T_i`` unchanged (0.0), leaves the isotherm ``d_lid``
  unchanged (0.0 to -1.412e-12 relative; up to 3.8e-5 / -3.816e-05 on
  internal-heating scaling ``case_06``), shifts the conductive-equivalent lid thickness by
  +3.185e-04 to +7.401e-04 (+5.5e-5 to +3.7e-3 / +5.470e-05 on ``case_12`` and
  +8.505e-05 on ``case_01`` to +3.726e-03 on ``case_06`` on the scaling snapshots),
  shifts ``theta`` by +2.302e-03 to +5.347e-03 (up to +2.3e-2 /
  +2.282e-02 on ``case_06``), and gives a consistent-evaluation residual
  ``max|dS/dt|`` of 2.265e-07 to 9.609e-07 J/(kg K yr) (up to 3.4e-5 /
  3.426e-05 J/(kg K yr) on ``case_06``).

Execution modes and per-block coverage:
- ``--full`` (~45 s on 1 CPU core): recomputes the secular-cooling trajectories and 0-D
  sweeps using the standalone 0-D stagnant-lid model (``tests.reference.stagnant_lid_0d``)
  and the 1-D ``EntropySolver`` (temporarily replacing ``EntropySolver._solve_cvode``
  with an unbounded-step wrapper ``max_step = np.inf`` to integrate the multi-Gyr
  secular-cooling runs without hitting the 100 yr phase-boundary step cap; this override
  is removed once the step-cap fix lands on ``main``), and then verifies all blocks:
  1. ``a_nu_provenance`` (~0.02 s): computes empirical ``a_nu = Nu * theta**(4/3) * Ra_i**(-1/3)``
     across all 12 Newtonian Arrhenius rows of Korenaga (2009) Table 1 (mean ``0.5954``,
     sample standard deviation ``0.0651``) and integrates the 4.5 Gyr 0-D ODE across
     ``a_nu in {0.5954, 0.57, 0.53}``, computing each prefactor's own ``Nu_unyielded < 1``
     grid crossing ``t_nu_clamp_1_yr`` and strict pre-clamp validity window ``t < t_clamp``.
  2. ``trajectory_calibrated_400Myr`` and ``trajectory_default_400Myr`` (~18 s):
     integrates the 400 Myr 1-D ``EntropySolver`` secular-cooling trajectories at
     ``(s_top_inf, c_theta) = (0.25, 0.25)`` and ``(0.22, 0.15)`` and the 400 Myr 0-D
     ODE trajectories, updating the stored 1-D snapshot states and window error metrics.
  3. ``trajectory_calibrated_4p5Gyr`` and ``trajectory_default_4p5Gyr`` (~25 s):
     integrates the 4.5 Gyr 1-D ``EntropySolver`` and 0-D ODE trajectories (129 output
     points), updates the stored 1-D snapshot states, computes the linear crossing times
     ``t_cross_rel_err_d_lid_0p22_yr`` and ``t_cross_rel_err_T_i_0p05_yr``, and records
     ``t_nu_0d_clamp_1_yr``, ``max_rel_err_T_i_valid_nu_ge_1``, and
     ``max_rel_err_d_lid_valid_nu_ge_1_post_relax`` on the strict window ``t < t_clamp``.
  4. ``yield_sweep_0d_at_1600K`` (~0.01 s) and ``acceptance_summary``: evaluates the
     5-case 0-D yield-ceiling sweep across ``tau_y / tau_d_lid in {500 MPa, 2.0, 0.5, 0.2, 0.1}``
     at ``T_i = 1600 K`` and updates ``acceptance_summary``.
  5. Fixed-input 1-D blocks (``steady_yield_thinning``, ``live_yield_states``,
     ``yield_sweep_reference_state``, ``yield_sweep_1d_by_dt``,
     ``yield_sweep_1d_equal_Ti_from_S_ref``, ``yield_sweep_1d_equal_Ti_1600K_from_init``,
     ``segment_25Myr_spread_5starts``): refreshes 0-D fixed-``T_i`` and fixed-``H``
     steady-state ratios and stamps ``source_git_commit`` and ``eos_tables`` on each
     fixed-input block.
- ``--check`` (~1.8 s on 1 CPU core): verifies expected schema keys, the no-table
  constant-property record (``eos_tables``), and recomputes and compares every stored
  field across all blocks:
  1. ``a_nu_provenance``: recomputes all 12 Korenaga (2009) Table 1 ``a_nu`` values,
     mean, standard deviation, and all three 4.5 Gyr 0-D ODE trajectories and per-prefactor
     ``t_nu_clamp_1_yr`` and pre-clamp window metrics within ``1e-12``.
  2. ``trajectory_calibrated_400Myr``, ``trajectory_default_400Myr``,
     ``trajectory_calibrated_4p5Gyr``, ``trajectory_default_4p5Gyr``,
     ``segment_25Myr_spread_5starts``, and ``acceptance_summary``: recomputes all four
     0-D ODE trajectories, all error series, all window maxima, all linear crossing
     times, all snapshot entries, and ``acceptance_summary`` within ``1e-12``, integrates
     the live 1-D ``EntropySolver`` over ``[0, 25 Myr]`` (17 points) within ``1e-7``, and
     evaluates live 1-D ``EntropySolver._dSdt_single`` on all four ``solver_last_S``
     endpoint profiles within ``1e-9``.
  3. ``yield_sweep_0d_at_1600K``, ``yield_sweep_reference_state``,
     ``yield_sweep_1d_by_dt``, ``yield_sweep_1d_equal_Ti_from_S_ref``, and
     ``yield_sweep_1d_equal_Ti_1600K_from_init``: recomputes all 5 0-D yield cases at
     1600 K, verifies all 1-D thinning ratios across all horizons within ``1e-12``, and
     evaluates live 1-D ``EntropySolver._dSdt_single`` on ``s_ref_51`` across all 5 cases
     of ``0Myr`` within ``1e-11``.
  4. ``steady_yield_thinning`` and ``live_yield_states``: recomputes the 0-D fixed-``T_i``
     and fixed-``H`` steady-state ratios and relative errors across all 7 steady-state
     sweeps within ``1e-11`` and evaluates live 1-D ``EntropySolver._dSdt_single`` on all
     3 stored ``E300`` (``N=100``) steady-state profiles within ``1e-11``, verifying
     ``max_abs_rhs <= 100 * max_5start_rhs``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

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
from aragog.rheology import R_GAS, SolidRheologyParams  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402
from tests.reference.stagnant_lid_0d import (  # noqa: E402
    evaluate_stagnant_lid_0d,
    integrate_stagnant_lid_0d,
    load_aragog_0d_params,
)
from tools.verification._ssc_helpers import (  # noqa: E402
    half_cell_surface_flux,
    unbounded_cvode_step,
)

SECS_PER_YR = 31557600.0

DEFAULT_BENCHMARK_PATH = REPO_ROOT / 'tests' / 'reference' / 'ssc_0d_reference_benchmarks.json'
NO_TABLE_RECORD = (
    'no table (constant properties: eos_method=1, const_properties=True, entropy_eos=None)'
)

EXPECTED_TOP_KEYS = (
    'metadata',
    'trajectory_calibrated_400Myr',
    'trajectory_default_400Myr',
    'trajectory_calibrated_4p5Gyr',
    'trajectory_default_4p5Gyr',
    'yield_sweep_reference_state',
    'yield_sweep_0d_at_1600K',
    'yield_sweep_1d_by_dt',
    'yield_sweep_1d_equal_Ti_from_S_ref',
    'yield_sweep_1d_equal_Ti_1600K_from_init',
    'acceptance_summary',
    'steady_yield_thinning',
    'segment_25Myr_spread_5starts',
    'live_yield_states',
    'a_nu_provenance',
)
EXPECTED_STEADY_KEYS = (
    'calibration_point',
    'e300_n100_fixed_h',
    'e300_n50_fixed_h',
    'e300_n100_fixed_h_self_consistent',
    'e340_n100_fixed_h',
    'e270_n100_fixed_h',
    'e300_n100_matched_ti',
    'e340_n100_matched_ti',
    'e300_n100_five_start_spread',
)
EXPECTED_LIVE_YIELD_KEYS = (
    'e300_n100_steady_vectors',
    's_ref_51',
)
EXPECTED_PROVENANCE_KEYS = (
    'a_nu_value',
    'a_nu_std',
    'fitted_to_1d_aragog_solver',
    'source',
    'korenaga_2009_table1_rows',
    'prefactor_comparisons_4p5Gyr',
    'citation',
)
FIXED_INPUT_1D_BLOCKS = (
    'yield_sweep_reference_state',
    'yield_sweep_1d_by_dt',
    'yield_sweep_1d_equal_Ti_from_S_ref',
    'yield_sweep_1d_equal_Ti_1600K_from_init',
    'steady_yield_thinning',
    'segment_25Myr_spread_5starts',
    'live_yield_states',
)

KORENAGA_TABLE1_ROWS: tuple[tuple[str, float, float, float, float, float], ...] = (
    ('E230a', 230.0, 1531.0, 13.0, 2.41e6, 2.94),
    ('E340', 340.0, 1531.0, 19.2, 1.24e7, 3.04),
    ('E270', 270.0, 1546.0, 15.2, 7.18e6, 3.20),
    ('E300', 300.0, 1544.0, 16.9, 1.49e7, 3.40),
    ('E210', 210.0, 1561.0, 11.7, 8.36e6, 3.84),
    ('E250b', 250.0, 1564.0, 13.9, 1.89e7, 4.03),
    ('E280a', 280.0, 1522.0, 15.9, 5.13e6, 2.96),
    ('E250a', 250.0, 1549.0, 14.0, 7.52e6, 3.39),
    ('E230b', 230.0, 1528.0, 13.0, 6.38e6, 3.44),
    ('E320', 320.0, 1537.0, 18.1, 1.36e7, 3.22),
    ('E280b', 280.0, 1546.0, 15.7, 1.57e7, 3.59),
    ('E260', 260.0, 1561.0, 14.5, 1.72e7, 3.84),
)

YIELD_MULTIPLIERS: tuple[tuple[str, float | None], ...] = (
    ('unyielded_500MPa', None),
    ('above_2p0x_tau_d_lid', 2.0),
    ('below_0p5x_tau_d_lid', 0.5),
    ('below_0p2x_tau_d_lid', 0.2),
    ('below_0p1x_tau_d_lid', 0.1),
)


def compute_crossing_time_linear(
    t_yr: np.ndarray, series: np.ndarray, threshold: float
) -> float | None:
    """Compute linear-interpolated first crossing time where ``series`` crosses above ``threshold``."""
    idx = np.where(series > threshold)[0]
    if idx.size == 0:
        return None
    i1 = int(idx[0])
    if i1 == 0:
        return None
    i0 = i1 - 1
    y0, y1 = float(series[i0]), float(series[i1])
    frac = (threshold - y0) / (y1 - y0)
    return float(t_yr[i0] + frac * (t_yr[i1] - t_yr[i0]))


def build_default_0d_params() -> Any:
    """Load the canonical 0-D stagnant-lid parameter bundle for ``abe_solid.toml``."""
    return load_aragog_0d_params(
        abe_solid_toml=REPO_ROOT / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml',
        viscosity_solid=1.0e18,
        t_surf=300.0,
        p_i_depth_m=100.0e3,
        a_rh_fb=1.3,
        c4_fb=0.125,
        a_nu=0.5954345503010264,
        yield_blend_ratio_width=0.025,
    )


def build_const_prop_1d_solver(
    base_params: Parameters,
    rheo: SolidRheologyParams,
    p0d: Any,
    *,
    t_end_yr: float,
    n_output_points: int = 65,
) -> EntropySolver:
    """Construct and initialize a 1-D constant-property ``EntropySolver`` on ``abe_solid.toml``."""
    params = replace(
        base_params,
        boundary_conditions=replace(
            base_params.boundary_conditions,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            core_bc='quasi_steady',
            outer_boundary_condition=1,
            equilibrium_temperature=p0d.t_surf,
            emissivity=1.0,
        ),
        solver=replace(
            base_params.solver,
            start_time=0.0,
            end_time=t_end_yr,
            atol=1.0e-8,
            rtol=1.0e-8,
            tsurf_poststep_change=100.0,
        ),
        mesh=replace(base_params.mesh, number_of_nodes=100, mass_coordinates=False),
        energy=replace(base_params.energy, kappah_floor=0.0),
        phase_solid=replace(
            base_params.phase_solid, viscosity=p0d.viscosity_solid, rheology=rheo
        ),
        phase_mixed=replace(
            base_params.phase_mixed,
            const_properties=True,
            const_rho=p0d.rho,
            const_Cp=float(base_params.phase_solid.heat_capacity),
            const_alpha=float(base_params.phase_solid.thermal_expansivity),
            const_cond=float(base_params.phase_solid.thermal_conductivity),
        ),
    )
    solver = EntropySolver(params, entropy_eos=None)
    solver._cvode_output_points = int(n_output_points)
    solver.initialize()
    for ph in (solver.state.phase_basic, solver.state.phase_staggered):
        ph.pressure[:] = p0d.p_i_ref
    return solver


def compute_a_nu_provenance_block(
    p0d: Any,
    t_4p5_yr: np.ndarray,
    t_i_1d_4p5: np.ndarray,
    d_lid_1d_4p5: np.ndarray,
    f_top_1d_4p5: np.ndarray,
    _t_nu_clamp_yr_unused: float | None = None,
) -> dict[str, Any]:
    """Compute Korenaga (2009) Table 1 ``a_nu`` statistics and 4.5 Gyr 0-D comparisons."""
    rows_out = []
    a_vals = []
    for name, e_kj, dt_k, theta, ra_i, nu_ref in KORENAGA_TABLE1_ROWS:
        a_emp = float(nu_ref * (theta ** (4.0 / 3.0)) * (ra_i ** (-1.0 / 3.0)))
        a_vals.append(a_emp)
        rows_out.append(
            {
                'case': name,
                'E_kJ_mol': e_kj,
                'dT_K': dt_k,
                'theta': theta,
                'Ra_i': ra_i,
                'Nu_ref': nu_ref,
                'a_nu_empirical': a_emp,
            }
        )
    a_arr = np.asarray(a_vals, dtype=float)
    a_mean = float(np.mean(a_arr))
    a_std = float(np.std(a_arr, ddof=1))

    mask_post_25 = t_4p5_yr >= 2.5e7
    comp: dict[str, Any] = {}
    for label, a_val in (
        ('korenaga_2009_table1_mean_0p5954', a_mean),
        ('solomatov_moresi_2000_table6_0p53', 0.53),
        ('korenaga_2009_eq29_fig4a_0p57', 0.57),
    ):
        p_sub = replace(p0d, a_nu=float(a_val))
        ode = integrate_stagnant_lid_0d(
            1800.0,
            t_4p5_yr * SECS_PER_YR,
            p_sub,
            conduction_geometry='spherical',
            heat_capacity_mode='isothermal',
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        ti_0d = np.asarray(ode['T_i_K'], dtype=float)
        dl_0d = np.asarray(ode['d_lid_eff_m'], dtype=float)
        ft_0d = np.asarray(ode['F_top_W_m2'], dtype=float)
        nu_unyielded_sub = np.array(
            [
                float(evaluate_stagnant_lid_0d(float(ti_val), p_sub)['Nu_unyielded'])
                for ti_val in ti_0d
            ],
            dtype=float,
        )
        clamp_idx_sub = np.where(nu_unyielded_sub < 1.0)[0]
        t_nu_clamp_sub = (
            float(t_4p5_yr[int(clamp_idx_sub[0])])
            if clamp_idx_sub.size > 0
            else float(t_4p5_yr[-1])
        )
        mask_valid_nu = t_4p5_yr < t_nu_clamp_sub
        mask_valid_post = mask_post_25 & mask_valid_nu
        err_ti = np.abs(t_i_1d_4p5 - ti_0d) / ti_0d
        err_dl = np.abs(d_lid_1d_4p5 - dl_0d) / dl_0d
        comp[label] = {
            'a_nu': float(a_val),
            'T_i_0d_4p5Gyr_K': float(ti_0d[-1]),
            'd_lid_0d_4p5Gyr_m': float(dl_0d[-1]),
            'F_top_0d_4p5Gyr_W_m2': float(ft_0d[-1]),
            'max_rel_err_T_i_4p5Gyr': float(np.max(err_ti)),
            'max_rel_err_T_i_valid_nu_ge_1': float(np.max(err_ti[mask_valid_nu])),
            'max_rel_err_d_lid_post_relax': float(np.max(err_dl[mask_post_25])),
            'max_rel_err_d_lid_valid_nu_ge_1_post_relax': float(
                np.max(err_dl[mask_valid_post])
            ),
            't_nu_clamp_1_yr': float(t_nu_clamp_sub),
            'F_top_ratio_1d_over_0d_4p5Gyr': float(f_top_1d_4p5[-1] / ft_0d[-1]),
        }

    return {
        'a_nu_value': a_mean,
        'a_nu_std': a_std,
        'fitted_to_1d_aragog_solver': False,
        'source': (
            'Arithmetic mean of Nu_ref * theta^(4/3) * Ra_i^(-1/3) across all 12 '
            'Newtonian (n = 1) Arrhenius 2-D finite-element runs in Korenaga (2009) '
            'Table 1 (p. 157).'
        ),
        'korenaga_2009_table1_rows': rows_out,
        'prefactor_comparisons_4p5Gyr': comp,
        'citation': 'Korenaga (2009), Geophys. J. Int., 179, 154-170, Table 1 (p. 157)',
    }


def compute_yield_sweep_0d_at_1600k(p0d: Any) -> dict[str, Any]:
    """Evaluate the 5-case 0-D stagnant-lid yield sweep at ``T_i = 1600 K``."""
    base_0d = evaluate_stagnant_lid_0d(1600.0, p0d)
    tau_d_0d = float(base_0d['tau_d_lid_Pa'])
    out: dict[str, Any] = {}
    for label, mult in YIELD_MULTIPLIERS:
        if mult is None:
            p_case = p0d
        else:
            tau_ovr = float(mult * tau_d_0d)
            p_case = replace(
                p0d, yield_stress_c=tau_ovr, yield_stress_mu=0.0, yield_stress_max=tau_ovr
            )
        ev = evaluate_stagnant_lid_0d(1600.0, p_case)
        out[label] = {
            'multiplier': mult,
            'tau_y_lid_Pa': float(ev['tau_y_lid_Pa']),
            'tau_d_lid_Pa': float(ev['tau_d_lid_Pa']),
            'w_b': float(ev['w_b_yield']),
            'thin_factor': float(ev['thin_factor']),
            'd_lid_eff_m': float(ev['d_lid_eff_m']),
            'F_top_spherical_W_m2': float(ev['F_top_spherical_W_m2']),
            'R_thin_0d': float(ev['d_lid_eff_m'] / base_0d['d_lid_eff_m']),
        }
    return out


def solve_0d_fixed_h_steady_ratio(
    row_name: str, t_i_uny_k: float, mult: float | None
) -> dict[str, float]:
    """Compute 0-D steady-state lid-thinning ratios under fixed ``T_i`` and fixed ``H``."""
    row_map = {r[0]: r for r in KORENAGA_TABLE1_ROWS}
    _, e_kj, _, _, _, _ = row_map[row_name]
    t_s = 273.0
    e_act = e_kj * 1.0e3
    a_lid = 2.2
    r_thin_0d_fixed_ti = 1.0 if (mult is None or mult >= 1.0) else float(mult ** (1.0 / 3.0))

    def _log_flux_factor(t_val: float) -> float:
        return (
            -(1.0 / 3.0) * np.log(t_val - t_s)
            + (8.0 / 3.0) * np.log(t_val)
            - e_act / (3.0 * R_GAS * t_val)
        )

    target = _log_flux_factor(t_i_uny_k) + np.log(r_thin_0d_fixed_ti)
    t_i_0d_fixed_h = float(
        brentq(
            lambda t_val: _log_flux_factor(t_val) - target,
            t_s + 100.0,
            t_i_uny_k + 500.0,
            xtol=1.0e-9,
        )
    )
    theta_uny = e_act * (t_i_uny_k - t_s) / (R_GAS * t_i_uny_k**2)
    theta_new = e_act * (t_i_0d_fixed_h - t_s) / (R_GAS * t_i_0d_fixed_h**2)
    r_thin_dl_0d_fixed_h = (t_i_0d_fixed_h - t_s) / (t_i_uny_k - t_s)
    r_thin_dlid_0d_fixed_h = ((1.0 - a_lid / theta_new) * (t_i_0d_fixed_h - t_s)) / (
        (1.0 - a_lid / theta_uny) * (t_i_uny_k - t_s)
    )
    return {
        'r_thin_0d_fixed_Ti': float(r_thin_0d_fixed_ti),
        'T_i_0d_fixed_H_K': float(t_i_0d_fixed_h),
        'r_thin_delta_l_0d_fixed_H': float(r_thin_dl_0d_fixed_h),
        'r_thin_dlid_0d_fixed_H': float(r_thin_dlid_0d_fixed_h),
    }


def evaluate_live_e300_steady_vector(
    tau_y_pa: float, h_scale: float, s_vec: np.ndarray
) -> dict[str, float]:
    """Evaluate 1-D ``EntropySolver._dSdt_single`` on a stored ``E300`` steady-state vector."""
    e_kj, dt_k, _, ra_i, nu_ref = 300.0, 1544.0, 16.9, 1.49e7, 3.40
    d_m = 1.0e6
    r_out = 100.0 * d_m
    r_in = r_out - d_m
    f_scale = 100.0
    rho, cp, k_cond, alpha, grav = 4000.0, 1000.0 * f_scale, 4.0 * f_scale, 3.0e-5, 10.0
    t_s = 273.0
    t_i_nom = t_s + dt_k
    e_act = e_kj * 1.0e3
    kappa = k_cond / (rho * cp)
    eta_i = alpha * rho * grav * dt_k * d_m**3 / (kappa * ra_i)
    h_heat = float((nu_ref * dt_k * k_cond / (rho * d_m**2)) * h_scale)
    g_hom = e_act / (R_GAS * t_i_nom)
    rheo = SolidRheologyParams(
        enabled=True,
        g_homologous=g_hom,
        t_sol_0=t_i_nom,
        dt_sol_dp=0.0,
        arrhenius_t_ref=t_i_nom,
        s_top_inf=0.25,
        c_theta=0.25,
        mlt_bottom_slope=1.0,
        yield_stress_c=float(tau_y_pa),
        yield_stress_mu=0.0,
        yield_stress_max=float(tau_y_pa),
        stress_closure_mode='lid',
    )
    common = dict(
        density=rho, heat_capacity=cp, thermal_conductivity=k_cond, thermal_expansivity=alpha
    )
    params = Parameters(
        boundary_conditions=_BoundaryConditionsParameters(
            outer_boundary_condition=5,
            outer_boundary_value=t_s,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=t_s,
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
        phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=1.0e2, **common),
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
        solver=_SolverParameters(start_time=0.0, end_time=1.0, atol=1.0e-6, rtol=1.0e-6),
    )
    with half_cell_surface_flux():
        solver = EntropySolver(params, entropy_eos=None)
        solver.initialize()
        rhs = np.asarray(solver._dSdt_single(0.0, s_vec), dtype=float)
        lid = solver.state._lid_state
        f_top = float(solver.state.heat_flux[-1])
        t_i_lid = float(lid['T_i'])
        theta_lid = float(lid['theta'])
        delta_l_cond = k_cond * (t_i_lid - t_s) / f_top
        d_lid_cond = (1.0 - float(rheo.lid_contrast_coeff) / theta_lid) * delta_l_cond
        eta_diff_b = np.asarray(solver.state.phase_basic.eta_diff(), dtype=float).ravel()
        eta_eff_b = np.asarray(solver.state._eta_eff_basic, dtype=float).ravel()
        w_lid_b = np.asarray(lid['w_lid'], dtype=float).ravel()
        eta_eff_lid_min = float(np.min(eta_eff_b[w_lid_b > 0.5]))
        return {
            'max_abs_rhs': float(np.max(np.abs(rhs))),
            'd_lid_pass1_m': float(lid['d_lid']),
            'd_lid_cond_lid_m': float(d_lid_cond),
            'eta_eff_lid_over_eta_i': float(eta_eff_lid_min / float(lid['eta_i'])),
            'T_i_lid_K': t_i_lid,
            'F_top_W_m2': f_top,
            'n_yielding_basic_nodes': int(np.sum(eta_eff_b < 0.999 * eta_diff_b)),
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


def _compute_trajectory_metrics_from_arrays(
    t_1d: np.ndarray,
    ti_1d: np.ndarray,
    dl_1d: np.ndarray,
    dl_cond_1d: np.ndarray,
    ft_1d: np.ndarray,
    p0d: Any,
    *,
    is_4p5gyr: bool,
) -> dict[str, Any]:
    """Recompute 0-D ODE trajectory, error series, window maxima, and crossing times."""
    ode_0d = integrate_stagnant_lid_0d(
        1800.0,
        t_1d * SECS_PER_YR,
        p0d,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    ti_0d = np.asarray(ode_0d['T_i_K'], dtype=float)
    dl_0d = np.asarray(ode_0d['d_lid_eff_m'], dtype=float)
    ft_0d = np.asarray(ode_0d['F_top_W_m2'], dtype=float)
    s_err_ti = (ti_1d - ti_0d) / ti_0d
    s_err_dl = (dl_1d - dl_0d) / dl_0d
    err_ti = np.abs(s_err_ti)
    err_dl = np.abs(s_err_dl)
    err_dl_cond = np.abs(dl_cond_1d - dl_0d) / dl_0d

    mask_post = t_1d >= 2.5e7
    mask_10m = t_1d >= 1.0e7
    mask_16p5m = t_1d >= 1.65e7
    mask_25_400 = (t_1d >= 2.5e7) & (t_1d <= 4.0e8)
    mask_1600_1800 = (ti_0d >= 1600.0) & (ti_0d <= 1800.0) & mask_post

    metrics: dict[str, Any] = {
        'ti_0d': ti_0d,
        'dl_0d': dl_0d,
        'ft_0d': ft_0d,
        's_err_ti': s_err_ti,
        's_err_dl': s_err_dl,
        'err_ti': err_ti,
        'err_dl': err_dl,
        'max_rel_err_T_i_all': float(np.max(err_ti)),
        'max_rel_err_T_i_post_relax': float(np.max(err_ti[mask_post])),
        'max_rel_err_T_i_window_25_400Myr': float(np.max(err_ti[mask_25_400])),
        'max_rel_err_T_i_window_1600_1800': float(np.max(err_ti[mask_1600_1800])),
        'max_rel_err_d_lid_all': float(np.max(err_dl)),
        'max_rel_err_d_lid_post_10Myr': float(np.max(err_dl[mask_10m])),
        'max_rel_err_d_lid_post_16p5Myr': float(np.max(err_dl[mask_16p5m])),
        'max_rel_err_d_lid_post_relax': float(np.max(err_dl[mask_post])),
        'max_rel_err_d_lid_window_25_400Myr': float(np.max(err_dl[mask_25_400])),
        'max_rel_err_d_lid_window_1600_1800': float(np.max(err_dl[mask_1600_1800])),
        'max_rel_err_d_lid_cond_post_relax': float(np.max(err_dl_cond[mask_post])),
        't_cross_rel_err_d_lid_0p22_yr': compute_crossing_time_linear(
            t_1d[mask_post], err_dl[mask_post], 0.22
        ),
    }
    if is_4p5gyr:
        metrics['t_cross_rel_err_T_i_0p05_yr'] = compute_crossing_time_linear(
            t_1d, err_ti, 0.05
        )
        nu_unyielded_grid = np.array(
            [
                float(evaluate_stagnant_lid_0d(float(ti_val), p0d)['Nu_unyielded'])
                for ti_val in ti_0d
            ],
            dtype=float,
        )
        clamp_idx = np.where(nu_unyielded_grid < 1.0)[0]
        t_nu_clamp = float(t_1d[int(clamp_idx[0])]) if clamp_idx.size > 0 else float(t_1d[-1])
        mask_valid_nu = t_1d < t_nu_clamp
        metrics['t_nu_0d_clamp_1_yr'] = t_nu_clamp
        metrics['max_rel_err_T_i_valid_nu_ge_1'] = float(np.max(err_ti[mask_valid_nu]))
        metrics['max_rel_err_d_lid_valid_nu_ge_1_post_relax'] = float(
            np.max(err_dl[mask_post & mask_valid_nu])
        )
    return metrics


def compute_acceptance_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Compute ``acceptance_summary`` from the trajectory and yield-sweep blocks."""
    r0d_ref = float(data['yield_sweep_0d_at_1600K']['below_0p5x_tau_d_lid']['R_thin_0d'])
    by_dt = data['yield_sweep_1d_by_dt']
    eq_sref = data['yield_sweep_1d_equal_Ti_from_S_ref']['below_0p5x_tau_d_lid']
    eq_init = data['yield_sweep_1d_equal_Ti_1600K_from_init']['below_0p5x_tau_d_lid']

    p1_10m = float(by_dt['10Myr']['below_0p5x_tau_d_lid']['R_thin_d_lid_pass1'])
    p1_sref = float(eq_sref['R_thin_d_lid_pass1'])
    p1_init = float(eq_init['R_thin_d_lid_pass1'])
    tol_r = 0.11

    tr400 = data['trajectory_calibrated_400Myr']
    tr4p5 = data['trajectory_calibrated_4p5Gyr']
    return {
        'R_thin_0p5x_tau_d_lid': {
            'r_thin_0d_ref': r0d_ref,
            'tol_rel': tol_r,
            'pass1_isotherm_d_lid_at_dt0': float(
                by_dt['0Myr']['below_0p5x_tau_d_lid']['R_thin_d_lid_pass1']
            ),
            'pass1_isotherm_d_lid_at_10Myr': p1_10m,
            'pass1_isotherm_rel_err_at_10Myr': abs(p1_10m - r0d_ref) / r0d_ref,
            'pass1_isotherm_d_lid_at_equal_Ti_from_S_ref': p1_sref,
            'pass1_isotherm_rel_err_at_equal_Ti_from_S_ref': abs(p1_sref - r0d_ref) / r0d_ref,
            'pass1_isotherm_d_lid_at_equal_Ti_1600K_from_init': p1_init,
            'pass1_isotherm_rel_err_at_equal_Ti_1600K_from_init': abs(p1_init - r0d_ref)
            / r0d_ref,
            'pass2_fixed_d_lid_at_10Myr': float(
                by_dt['10Myr']['below_0p5x_tau_d_lid']['R_thin_d_lid_pass2_fixed']
            ),
            'pass2_fixed_d_lid_at_equal_Ti_from_S_ref': float(
                eq_sref['R_thin_d_lid_pass2_fixed']
            ),
            'conductive_d_lid_planar_at_5Myr': float(
                by_dt['5Myr']['below_0p5x_tau_d_lid']['R_thin_d_lid_cond_planar']
            ),
            'conductive_d_lid_planar_at_10Myr': float(
                by_dt['10Myr']['below_0p5x_tau_d_lid']['R_thin_d_lid_cond_planar']
            ),
            'conductive_d_lid_planar_at_equal_Ti_from_S_ref': float(
                eq_sref['R_thin_d_lid_cond_planar']
            ),
            'conductive_d_lid_planar_at_equal_Ti_1600K_from_init': float(
                eq_init['R_thin_d_lid_cond_planar']
            ),
            'pass1_isotherm_within_tolerance': bool(abs(p1_10m - r0d_ref) / r0d_ref <= tol_r),
        },
        'T_i_trajectory_calibrated': {
            'max_rel_err_full_400Myr': float(tr400['max_rel_err_T_i_all']),
            'max_rel_err_window_25_400Myr': float(tr400['max_rel_err_T_i_window_25_400Myr']),
            'max_rel_err_window_1600_1800': float(tr400['max_rel_err_T_i_window_1600_1800']),
            'max_rel_err_full_4p5Gyr': float(tr4p5['max_rel_err_T_i_all']),
            'max_rel_err_post_relax_4p5Gyr': float(tr4p5['max_rel_err_T_i_post_relax']),
            'tol_rel': 0.05,
        },
        'd_lid_trajectory_calibrated': {
            'max_rel_err_full_400Myr': float(tr400['max_rel_err_d_lid_all']),
            'max_rel_err_window_25_400Myr': float(tr400['max_rel_err_d_lid_window_25_400Myr']),
            'max_rel_err_window_1600_1800': float(tr400['max_rel_err_d_lid_window_1600_1800']),
            'max_rel_err_full_4p5Gyr': float(tr4p5['max_rel_err_d_lid_all']),
            'max_rel_err_post_relax_4p5Gyr': float(tr4p5['max_rel_err_d_lid_post_relax']),
            't_cross_0p22_yr': float(tr4p5['t_cross_rel_err_d_lid_0p22_yr']),
            'tol_rel': 0.22,
        },
    }


def verify_benchmarks(data: dict[str, Any]) -> list[str]:
    """Verify schema keys, no-table metadata record, and recompute every quoted field."""
    failures: list[str] = []

    if tuple(data.keys()) != EXPECTED_TOP_KEYS:
        failures.append(f'Top-level keys {tuple(data.keys())} != {EXPECTED_TOP_KEYS}')

    meta = data.get('metadata', {})
    if not isinstance(meta, dict) or not meta.get('git_commit') or 'params_0d' not in meta:
        failures.append("Missing 'git_commit' or 'params_0d' in metadata")
    elif not _is_commit_on_origin(str(meta.get('git_commit'))):
        failures.append(
            f"metadata['git_commit'] {meta.get('git_commit')!r} is not a commit reachable from origin/*"
        )
    elif meta.get('eos_tables') != NO_TABLE_RECORD or 'eos_md5' in meta:
        failures.append(
            f"metadata['eos_tables'] {meta.get('eos_tables')!r} != {NO_TABLE_RECORD!r} or 'eos_md5' present"
        )

    steady = data.get('steady_yield_thinning', {})
    steady_keys = tuple(
        k for k in steady.keys() if k not in ('source_git_commit', 'eos_tables')
    )
    if not isinstance(steady, dict) or steady_keys != EXPECTED_STEADY_KEYS:
        failures.append(f'steady_yield_thinning keys {steady_keys} != {EXPECTED_STEADY_KEYS}')

    live_states = data.get('live_yield_states', {})
    live_keys = tuple(
        k for k in live_states.keys() if k not in ('source_git_commit', 'eos_tables')
    )
    if not isinstance(live_states, dict) or live_keys != EXPECTED_LIVE_YIELD_KEYS:
        failures.append(f'live_yield_states keys {live_keys} != {EXPECTED_LIVE_YIELD_KEYS}')

    prov = data.get('a_nu_provenance', {})
    if not isinstance(prov, dict) or tuple(prov.keys()) != EXPECTED_PROVENANCE_KEYS:
        failures.append(f'a_nu_provenance keys != {EXPECTED_PROVENANCE_KEYS}')
    if failures:
        return failures

    p0d = build_default_0d_params()
    p0d_dict = asdict(p0d)
    stored_p0d = meta.get('params_0d', {})
    if not isinstance(stored_p0d, dict):
        failures.append("metadata['params_0d'] is not a dict")
    else:
        for pk, pv in p0d_dict.items():
            if pk not in stored_p0d:
                failures.append(f"metadata['params_0d'] missing {pk!r}")
            elif isinstance(pv, float) and abs(pv - float(stored_p0d[pk])) > 1.0e-12:
                failures.append(f"metadata['params_0d'][{pk!r}] mismatch")

    base_params = Parameters.from_file(REPO_ROOT / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml')

    # 1. Block 1 live check: a_nu_provenance (all 12 rows, mean, std, and all 3 prefactor comparisons)
    tr_4p5 = data['trajectory_calibrated_4p5Gyr']['trajectory']
    t_4p5_yr = np.asarray(tr_4p5['t_yr'], dtype=float)
    ti_1d_4p5 = np.asarray(tr_4p5['T_i_1d_K'], dtype=float)
    dl_1d_4p5 = np.asarray(tr_4p5['d_lid_1d_m'], dtype=float)
    ft_1d_4p5 = np.asarray(tr_4p5['F_top_1d_W_m2'], dtype=float)
    live_prov = compute_a_nu_provenance_block(p0d, t_4p5_yr, ti_1d_4p5, dl_1d_4p5, ft_1d_4p5)
    if abs(live_prov['a_nu_value'] - float(prov['a_nu_value'])) > 1.0e-15:
        failures.append('a_nu_provenance mean mismatch')
    if abs(live_prov['a_nu_std'] - float(prov['a_nu_std'])) > 1.0e-15:
        failures.append('a_nu_provenance std mismatch')
    if prov.get('fitted_to_1d_aragog_solver') is not False:
        failures.append('a_nu_provenance fitted_to_1d_aragog_solver != False')
    stored_rows = prov.get('korenaga_2009_table1_rows', [])
    if not isinstance(stored_rows, list) or len(stored_rows) != len(
        live_prov['korenaga_2009_table1_rows']
    ):
        failures.append('a_nu_provenance korenaga_2009_table1_rows length mismatch')
    else:
        for r_live, r_st in zip(live_prov['korenaga_2009_table1_rows'], stored_rows):
            if r_st.get('case') != r_live['case']:
                failures.append(
                    f'a_nu_provenance row case {r_st.get("case")} != {r_live["case"]}'
                )
            for rk in ('E_kJ_mol', 'dT_K', 'theta', 'Ra_i', 'Nu_ref', 'a_nu_empirical'):
                if (
                    abs(float(r_st.get(rk, -1.0)) - float(r_live[rk]))
                    / max(abs(float(r_live[rk])), 1.0e-30)
                    > 1.0e-12
                ):
                    failures.append(f'a_nu_provenance row {r_live["case"]} {rk} mismatch')

    stored_pref = prov.get('prefactor_comparisons_4p5Gyr', {})
    if not isinstance(stored_pref, dict) or tuple(stored_pref.keys()) != tuple(
        live_prov['prefactor_comparisons_4p5Gyr'].keys()
    ):
        failures.append('a_nu_provenance prefactor_comparisons_4p5Gyr keys mismatch')
    else:
        for k_sub, v_sub in live_prov['prefactor_comparisons_4p5Gyr'].items():
            stored_sub = stored_pref[k_sub]
            for metric, m_val in v_sub.items():
                if metric not in stored_sub:
                    failures.append(f'a_nu_provenance {k_sub} missing {metric!r}')
                elif (
                    abs(m_val - float(stored_sub[metric])) / max(abs(m_val), 1.0e-30) > 1.0e-12
                ):
                    failures.append(f'a_nu_provenance {k_sub} {metric} mismatch')

    # 2. Block 2 & 3 check: all 4 trajectory blocks, live 25 Myr segment, live 4.5 Gyr endpoints, and acceptance_summary
    with unbounded_cvode_step():
        for blk_key, s_inf_exp, c_th_exp, is_4p5 in (
            ('trajectory_calibrated_400Myr', 0.25, 0.25, False),
            ('trajectory_default_400Myr', 0.22, 0.15, False),
            ('trajectory_calibrated_4p5Gyr', 0.25, 0.25, True),
            ('trajectory_default_4p5Gyr', 0.22, 0.15, True),
        ):
            blk = data.get(blk_key, {})
            if not isinstance(blk, dict) or 'trajectory' not in blk:
                failures.append(f'{blk_key} missing trajectory')
                continue
            if (
                abs(float(blk.get('s_top_inf', -1.0)) - s_inf_exp) > 1.0e-12
                or abs(float(blk.get('c_theta', -1.0)) - c_th_exp) > 1.0e-12
            ):
                failures.append(f'{blk_key} s_top_inf/c_theta mismatch')
            tr = blk['trajectory']
            t_arr = np.asarray(tr['t_yr'], dtype=float)
            ti_1d = np.asarray(tr['T_i_1d_K'], dtype=float)
            dl_1d = np.asarray(tr['d_lid_1d_m'], dtype=float)
            dlc_1d = np.asarray(tr['d_lid_cond_1d_m'], dtype=float)
            ft_1d = np.asarray(tr['F_top_1d_W_m2'], dtype=float)
            m_live = _compute_trajectory_metrics_from_arrays(
                t_arr, ti_1d, dl_1d, dlc_1d, ft_1d, p0d, is_4p5gyr=is_4p5
            )
            for arr_k, live_arr in (
                ('T_i_0d_K', m_live['ti_0d']),
                ('d_lid_0d_m', m_live['dl_0d']),
                ('F_top_0d_W_m2', m_live['ft_0d']),
                ('signed_rel_err_T_i', m_live['s_err_ti']),
                ('signed_rel_err_d_lid', m_live['s_err_dl']),
                ('rel_err_T_i', m_live['err_ti']),
                ('rel_err_d_lid', m_live['err_dl']),
            ):
                if not np.allclose(
                    np.asarray(tr.get(arr_k, []), dtype=float),
                    live_arr,
                    rtol=1.0e-12,
                    atol=1.0e-14,
                ):
                    failures.append(f'{blk_key}.trajectory.{arr_k} mismatch')

            scalar_keys = [
                'max_rel_err_T_i_all',
                'max_rel_err_T_i_post_relax',
                'max_rel_err_T_i_window_25_400Myr',
                'max_rel_err_T_i_window_1600_1800',
                'max_rel_err_d_lid_all',
                'max_rel_err_d_lid_post_10Myr',
                'max_rel_err_d_lid_post_16p5Myr',
                'max_rel_err_d_lid_post_relax',
                'max_rel_err_d_lid_window_25_400Myr',
                'max_rel_err_d_lid_window_1600_1800',
                'max_rel_err_d_lid_cond_post_relax',
                't_cross_rel_err_d_lid_0p22_yr',
            ]
            if is_4p5:
                for k_4p5 in (
                    't_cross_rel_err_T_i_0p05_yr',
                    't_nu_0d_clamp_1_yr',
                    'max_rel_err_T_i_valid_nu_ge_1',
                    'max_rel_err_d_lid_valid_nu_ge_1_post_relax',
                ):
                    if k_4p5 in blk or k_4p5 in (
                        't_nu_0d_clamp_1_yr',
                        'max_rel_err_T_i_valid_nu_ge_1',
                        'max_rel_err_d_lid_valid_nu_ge_1_post_relax',
                    ):
                        scalar_keys.append(k_4p5)
            for sk in scalar_keys:
                v_exp = m_live[sk]
                v_st = blk.get(sk, '__missing__')
                if v_exp is None:
                    if v_st is not None:
                        failures.append(f'{blk_key}.{sk} expected None, got {v_st!r}')
                elif v_st is None or v_st == '__missing__':
                    failures.append(f'{blk_key}.{sk} missing or None')
                elif (
                    abs(float(v_st) - float(v_exp)) / max(abs(float(v_exp)), 1.0e-30) > 1.0e-12
                ):
                    failures.append(f'{blk_key}.{sk} mismatch: {v_st} vs {v_exp}')

            snaps_1d = blk.get('states_1d_snapshots', {})
            if isinstance(snaps_1d, dict):
                for idx_str, snap_v in snaps_1d.items():
                    idx_i = int(idx_str)
                    if not isinstance(snap_v, dict):
                        failures.append(f'{blk_key}.states_1d_snapshots[{idx_str}] not a dict')
                        continue
                    for sk_snap, arr_ref in (
                        ('T_i_K', ti_1d),
                        ('d_lid_pass1_m', dl_1d),
                        ('d_lid_cond_m', dlc_1d),
                        ('F_top_W_m2', ft_1d),
                    ):
                        if sk_snap in snap_v and (
                            abs(float(snap_v[sk_snap]) - float(arr_ref[idx_i]))
                            / max(abs(float(arr_ref[idx_i])), 1.0e-30)
                            > 1.0e-12
                        ):
                            failures.append(
                                f'{blk_key}.states_1d_snapshots[{idx_str}].{sk_snap} mismatch'
                            )

            rheo_blk = SolidRheologyParams(
                enabled=True,
                stress_closure_mode='lid',
                lid_base_mode='rheological',
                s_top_inf=s_inf_exp,
                c_theta=c_th_exp,
                yield_stress_c=500.0e6,
                yield_stress_mu=0.6,
                yield_stress_max=500.0e6,
            )
            solver_ep = build_const_prop_1d_solver(
                base_params, rheo_blk, p0d, t_end_yr=float(t_arr[-1]), n_output_points=2
            )
            s_last = np.asarray(blk['solver_last_S'], dtype=float)
            solver_ep._dSdt_single(float(t_arr[-1]), s_last)
            lid_ep = solver_ep.state._lid_state
            ti_ep = float(lid_ep['T_i'])
            dl_ep = float(lid_ep['d_lid'])
            ft_ep = float(solver_ep.state._heat_flux[-1])
            if abs(ti_ep - float(ti_1d[-1])) / ti_ep > 1.0e-9:
                failures.append(f'{blk_key} endpoint T_i mismatch')
            if abs(dl_ep - float(dl_1d[-1])) / dl_ep > 1.0e-9:
                failures.append(f'{blk_key} endpoint d_lid mismatch')
            if abs(ft_ep - float(ft_1d[-1])) / ft_ep > 1.0e-9:
                failures.append(f'{blk_key} endpoint F_top mismatch')

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
        solver_25 = build_const_prop_1d_solver(
            base_params, rheo_cal, p0d, t_end_yr=2.5e7, n_output_points=17
        )
        s_init = float(3000.0 + p0d.c_p * np.log(1800.0 / 3500.0))
        solver_25.set_initial_entropy(s_init)
        solver_25.solve()
        y_25 = np.asarray(solver_25.solution.y, dtype=float)
        t_25 = np.asarray(solver_25.solution.t, dtype=float)
        tr_400 = data['trajectory_calibrated_400Myr']['trajectory']
        for idx_25 in range(len(t_25)):
            solver_25._dSdt_single(float(t_25[idx_25]), y_25[:, idx_25])
            lid_25 = solver_25.state._lid_state
            ti_25 = float(lid_25['T_i'])
            dl_25 = float(lid_25['d_lid'])
            ft_25 = float(solver_25.state._heat_flux[-1])
            if abs(ti_25 - float(tr_400['T_i_1d_K'][idx_25])) / ti_25 > 1.0e-7:
                failures.append(f'25 Myr step {idx_25} live T_i mismatch')
            if idx_25 > 0 and abs(dl_25 - float(tr_400['d_lid_1d_m'][idx_25])) / dl_25 > 1.0e-7:
                failures.append(f'25 Myr step {idx_25} live d_lid mismatch')
            if abs(ft_25 - float(tr_400['F_top_1d_W_m2'][idx_25])) / ft_25 > 1.0e-7:
                failures.append(f'25 Myr step {idx_25} live F_top mismatch')

        seg_25 = data.get('segment_25Myr_spread_5starts', {})
        if (
            int(seg_25.get('n_starts', 0)) != 5
            or float(seg_25.get('perturbation_rel', 0.0)) != 1.0e-12
            or float(seg_25.get('tolerance_rel_used', 0.0)) != 1.0e-7
            or float(seg_25.get('max_rel_T_i', 1.0)) >= 1.0e-7
            or float(seg_25.get('max_rel_d_lid_post0', 1.0)) >= 1.0e-7
            or float(seg_25.get('max_rel_F_top', 1.0)) >= 1.0e-7
        ):
            failures.append('segment_25Myr_spread_5starts bounds/metadata check failed')

        # 3. Block 4 check: 0-D yield sweep at 1600 K, live s_ref_51 across all 5 cases of 0Myr, and all 1-D yield horizons
        live_0d_sweep = compute_yield_sweep_0d_at_1600k(p0d)
        stored_0d_sweep = data.get('yield_sweep_0d_at_1600K', {})
        if not isinstance(stored_0d_sweep, dict) or tuple(stored_0d_sweep.keys()) != tuple(
            live_0d_sweep.keys()
        ):
            failures.append('yield_sweep_0d_at_1600K keys mismatch')
        else:
            for k_case, v_case in live_0d_sweep.items():
                stored_c = stored_0d_sweep[k_case]
                for fk, fv in v_case.items():
                    if fk == 'multiplier':
                        if stored_c.get('multiplier') != fv:
                            failures.append(
                                f'yield_sweep_0d_at_1600K {k_case} multiplier mismatch'
                            )
                    elif (
                        abs(fv - float(stored_c.get(fk, -1.0))) / max(abs(fv), 1.0e-30)
                        > 1.0e-12
                    ):
                        failures.append(f'yield_sweep_0d_at_1600K {k_case} {fk} mismatch')

        s_ref_51 = np.asarray(live_states['s_ref_51']['S_ref_J_kg_K'], dtype=float)
        t_ref_51 = float(live_states['s_ref_51']['t_yr'])
        ref_state_blk = data.get('yield_sweep_reference_state', {})
        by_dt_0 = data.get('yield_sweep_1d_by_dt', {}).get('0Myr', {})
        for k_case, _mult in YIELD_MULTIPLIERS:
            c_0myr = by_dt_0.get(k_case, {})
            tau_y_set = float(c_0myr.get('tau_y_set_Pa', 500.0e6))
            rheo_c = (
                rheo_cal
                if k_case == 'unyielded_500MPa'
                else replace(
                    rheo_cal,
                    yield_stress_c=tau_y_set,
                    yield_stress_mu=0.0,
                    yield_stress_max=tau_y_set,
                )
            )
            solver_c = build_const_prop_1d_solver(base_params, rheo_c, p0d, t_end_yr=1.0)
            solver_c._dSdt_single(t_ref_51, s_ref_51)
            lid_c = solver_c.state._lid_state
            ti_c = float(lid_c['T_i'])
            dl_c = float(lid_c['d_lid'])
            th_c = float(lid_c['theta'])
            ft_c = float(solver_c.state._heat_flux[-1])
            td_c = float(lid_c['tau_d_lid'])
            wy_c = float(lid_c['w_y'])
            ts_c = float(np.asarray(solver_c.state.phase_basic.temperature()).ravel()[-1])
            dlc_pl = (1.0 - p0d.lid_contrast_coeff / th_c) * (p0d.k_cond * (ti_c - ts_c) / ft_c)
            for mk, mv in (
                ('T_i_K', ti_c),
                ('d_lid_pass1_m', dl_c),
                ('d_lid_cond_planar_m', dlc_pl),
                ('F_top_W_m2', ft_c),
                ('tau_d_lid_Pa', td_c),
                ('w_y', wy_c),
            ):
                if abs(mv - float(c_0myr.get(mk, -1.0))) / max(abs(mv), 1.0e-30) > 1.0e-11:
                    failures.append(f'yield_sweep_1d_by_dt.0Myr.{k_case}.{mk} live mismatch')
            if k_case == 'unyielded_500MPa':
                for mk in (
                    'T_i_K',
                    'd_lid_pass1_m',
                    'd_lid_cond_planar_m',
                    'F_top_W_m2',
                    'tau_d_lid_Pa',
                ):
                    if (
                        abs(float(ref_state_blk.get(mk, -1.0)) - float(c_0myr[mk]))
                        / max(abs(float(c_0myr[mk])), 1.0e-30)
                        > 1.0e-12
                    ):
                        failures.append(f'yield_sweep_reference_state.{mk} mismatch')

        for grp_label, grp_dict in (
            *(
                (f'yield_sweep_1d_by_dt.{hk}', hv)
                for hk, hv in data.get('yield_sweep_1d_by_dt', {}).items()
            ),
            (
                'yield_sweep_1d_equal_Ti_from_S_ref',
                data.get('yield_sweep_1d_equal_Ti_from_S_ref', {}),
            ),
            (
                'yield_sweep_1d_equal_Ti_1600K_from_init',
                data.get('yield_sweep_1d_equal_Ti_1600K_from_init', {}),
            ),
        ):
            uny_g = grp_dict.get('unyielded_500MPa', {})
            for k_case, _ in YIELD_MULTIPLIERS:
                cg = grp_dict.get(k_case, {})
                for r_key, num_key in (
                    ('R_thin_d_lid_pass1', 'd_lid_pass1_m'),
                    ('R_thin_d_lid_pass2_fixed', 'd_lid_pass2_fixed_m'),
                    ('R_thin_d_lid_cond_planar', 'd_lid_cond_planar_m'),
                    ('R_thin_d_lid_cond_sph', 'd_lid_cond_sph_m'),
                    ('F_top_ratio_to_unyielded', 'F_top_W_m2'),
                ):
                    r_exp = float(cg.get(num_key, -1.0)) / float(uny_g.get(num_key, 1.0))
                    if (
                        abs(float(cg.get(r_key, -1.0)) - r_exp) / max(abs(r_exp), 1.0e-30)
                        > 1.0e-12
                    ):
                        failures.append(f'{grp_label}.{k_case}.{r_key} ratio mismatch')

    # Check acceptance_summary
    live_acc = compute_acceptance_summary(data)
    stored_acc = data.get('acceptance_summary', {})
    for ak, av in live_acc.items():
        st_av = stored_acc.get(ak, {})
        if not isinstance(st_av, dict):
            failures.append(f'acceptance_summary.{ak} is not a dict')
            continue
        for sk, sv in av.items():
            if sk not in st_av:
                failures.append(f'acceptance_summary.{ak} missing {sk!r}')
            elif isinstance(sv, bool):
                if st_av[sk] is not sv:
                    failures.append(f'acceptance_summary.{ak}.{sk} {st_av[sk]!r} != {sv!r}')
            elif abs(sv - float(st_av[sk])) / max(abs(sv), 1.0e-30) > 1.0e-12:
                failures.append(f'acceptance_summary.{ak}.{sk} mismatch')

    # 4. Block 5 live check: all 7 steady-state yield sweeps + 3 live E300 steady-state profiles
    for sweep_key in (
        'e300_n100_fixed_h',
        'e300_n50_fixed_h',
        'e300_n100_fixed_h_self_consistent',
        'e340_n100_fixed_h',
        'e270_n100_fixed_h',
    ):
        blk = steady[sweep_key]
        row_name = str(blk['row_name'])
        ti_uny = float(blk['T_i_lid_unyielded_K'])
        uny_c = blk['cases']['unyielded_500MPa']
        for c_name, mult in YIELD_MULTIPLIERS:
            pred = solve_0d_fixed_h_steady_ratio(row_name, ti_uny, mult)
            c_stored = blk['cases'][c_name]
            for pk in ('r_thin_0d_fixed_Ti', 'T_i_0d_fixed_H_K', 'r_thin_dlid_0d_fixed_H'):
                if (
                    abs(pred[pk] - float(c_stored.get(pk, -1.0))) / max(abs(pred[pk]), 1.0e-30)
                    > 1.0e-11
                ):
                    failures.append(f'{sweep_key}.{c_name}.{pk} mismatch')
            r_p1 = float(c_stored['d_lid_pass1_m']) / float(uny_c['d_lid_pass1_m'])
            r_cl = float(c_stored['d_lid_cond_lid_m']) / float(uny_c['d_lid_cond_lid_m'])
            r_p2 = float(c_stored['d_lid_pass2_fixed_m']) / float(uny_c['d_lid_pass2_fixed_m'])
            r0_ti = pred['r_thin_0d_fixed_Ti']
            r0_fh = pred['r_thin_dlid_0d_fixed_H']
            for rk, r_exp in (
                ('R_thin_pass1', r_p1),
                ('rel_err_pass1_vs_0d_fixed_Ti', abs(r_p1 - r0_ti) / r0_ti),
                ('R_thin_cond_lid', r_cl),
                ('rel_err_cond_lid_vs_0d_fixed_Ti', abs(r_cl - r0_ti) / r0_ti),
                ('rel_err_cond_lid_vs_0d_fixed_H', abs(r_cl - r0_fh) / r0_fh),
                ('R_thin_pass2_fixed', r_p2),
            ):
                if (
                    abs(float(c_stored.get(rk, -1.0)) - r_exp) / max(abs(r_exp), 1.0e-30)
                    > 1.0e-11
                ):
                    failures.append(f'{sweep_key}.{c_name}.{rk} mismatch')

    for sweep_key, base_fixed_key in (
        ('e300_n100_matched_ti', 'e300_n100_fixed_h'),
        ('e340_n100_matched_ti', 'e340_n100_fixed_h'),
    ):
        blk = steady[sweep_key]
        uny_c = steady[base_fixed_key]['cases']['unyielded_500MPa']
        for c_name, mult in YIELD_MULTIPLIERS:
            if c_name not in blk.get('cases', {}):
                continue
            c_stored = blk['cases'][c_name]
            r0_ti = 1.0 if (mult is None or mult >= 1.0) else float(mult ** (1.0 / 3.0))
            r_p1 = float(c_stored['d_lid_pass1_m']) / float(uny_c['d_lid_pass1_m'])
            r_cl = float(c_stored['d_lid_cond_lid_m']) / float(uny_c['d_lid_cond_lid_m'])
            r_p2 = float(c_stored['d_lid_pass2_fixed_m']) / float(uny_c['d_lid_pass2_fixed_m'])
            for rk, r_exp in (
                ('r_thin_0d_fixed_Ti', r0_ti),
                ('R_thin_pass1', r_p1),
                ('rel_err_pass1_vs_0d_fixed_Ti', abs(r_p1 - r0_ti) / r0_ti),
                ('R_thin_cond_lid', r_cl),
                ('rel_err_cond_lid_vs_0d_fixed_Ti', abs(r_cl - r0_ti) / r0_ti),
                ('R_thin_pass2_fixed', r_p2),
            ):
                if (
                    abs(float(c_stored.get(rk, -1.0)) - r_exp) / max(abs(r_exp), 1.0e-30)
                    > 1.0e-11
                ):
                    failures.append(f'{sweep_key}.{c_name}.{rk} mismatch')

    vecs = live_states['e300_n100_steady_vectors']
    spread_5 = steady['e300_n100_five_start_spread']['five_start_steady_spread']
    e300_fh = steady['e300_n100_fixed_h']['cases']
    e300_mt = steady['e300_n100_matched_ti']['cases']
    live_specs = (
        ('unyielded_500MPa', e300_fh['unyielded_500MPa']),
        ('fixed_h_below_0p5x_tau_d_lid', e300_fh['below_0p5x_tau_d_lid']),
        ('matched_ti_below_0p5x_tau_d_lid', e300_mt['below_0p5x_tau_d_lid']),
    )
    for vec_key, ref_case in live_specs:
        obs = evaluate_live_e300_steady_vector(
            float(ref_case['tau_y_lid_Pa']),
            float(ref_case['h_scale']),
            np.asarray(vecs[vec_key]['S_steady_J_kg_K'], dtype=float),
        )
        max_5start_rhs = float(spread_5[vec_key]['max_over_5starts_max_abs_rhs_J_kg_K_yr'])
        if obs['max_abs_rhs'] > 100.0 * max_5start_rhs:
            failures.append(
                f'{vec_key} max_abs_rhs {obs["max_abs_rhs"]:.3e} > 100 * {max_5start_rhs:.3e}'
            )
        if int(obs['n_yielding_basic_nodes']) != int(ref_case['n_yielding_basic_nodes']) or int(
            obs['n_yielding_basic_nodes']
        ) != int(vecs[vec_key]['n_yielding_basic_nodes']):
            failures.append(f'{vec_key} n_yielding_basic_nodes mismatch')
        for m_key in (
            'd_lid_pass1_m',
            'd_lid_cond_lid_m',
            'eta_eff_lid_over_eta_i',
            'T_i_lid_K',
            'F_top_W_m2',
        ):
            if abs(obs[m_key] - float(ref_case[m_key])) / abs(float(ref_case[m_key])) > 1.0e-11:
                failures.append(f'{vec_key} {m_key} live vs steady_yield_thinning mismatch')
            if m_key in vecs[vec_key] and (
                abs(obs[m_key] - float(vecs[vec_key][m_key])) / abs(float(vecs[vec_key][m_key]))
                > 1.0e-11
            ):
                failures.append(f'{vec_key} {m_key} live vs live_yield_states mismatch')

    return failures


def _recompute_trajectory_block(
    base_params: Parameters,
    p0d: Any,
    existing_block: dict[str, Any],
    *,
    s_top_inf: float,
    c_theta: float,
    t_end_yr: float,
    n_output_points: int,
) -> dict[str, Any]:
    """Integrate the 1-D ``EntropySolver`` and 0-D ODE over ``[0, t_end_yr]``."""
    rheo = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        s_top_inf=s_top_inf,
        c_theta=c_theta,
        yield_stress_c=500.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
    )
    with unbounded_cvode_step():
        solver = build_const_prop_1d_solver(
            base_params, rheo, p0d, t_end_yr=t_end_yr, n_output_points=n_output_points
        )
        s_init = float(3000.0 + p0d.c_p * np.log(1800.0 / 3500.0))
        solver.set_initial_entropy(s_init)
        solver.solve()
        t_1d = np.asarray(solver.solution.t, dtype=float)
        y_1d = np.asarray(solver.solution.y, dtype=float)
        ti_1d, dl_1d, dl_cond_1d, ft_1d = [], [], [], []
        for k in range(len(t_1d)):
            solver._dSdt_single(float(t_1d[k]), y_1d[:, k])
            lid = solver.state._lid_state
            ti = float(lid['T_i'])
            theta = float(lid['theta'])
            ft = float(solver.state._heat_flux[-1])
            dl_p = p0d.k_cond * (ti - p0d.t_surf) / max(ft, 1.0e-30)
            ti_1d.append(ti)
            dl_1d.append(float(lid['d_lid']))
            dl_cond_1d.append(
                max(1.0 - p0d.lid_contrast_coeff / max(theta, 1.0e-30), 0.0) * dl_p
            )
            ft_1d.append(ft)

    is_4p5 = t_end_yr > 1.0e9
    m_live = _compute_trajectory_metrics_from_arrays(
        t_1d,
        np.asarray(ti_1d, dtype=float),
        np.asarray(dl_1d, dtype=float),
        np.asarray(dl_cond_1d, dtype=float),
        np.asarray(ft_1d, dtype=float),
        p0d,
        is_4p5gyr=is_4p5,
    )

    out = dict(existing_block)
    for mk in (
        'max_rel_err_T_i_all',
        'max_rel_err_T_i_post_relax',
        'max_rel_err_T_i_window_25_400Myr',
        'max_rel_err_T_i_window_1600_1800',
        'max_rel_err_d_lid_all',
        'max_rel_err_d_lid_post_10Myr',
        'max_rel_err_d_lid_post_16p5Myr',
        'max_rel_err_d_lid_post_relax',
        'max_rel_err_d_lid_window_25_400Myr',
        'max_rel_err_d_lid_window_1600_1800',
        'max_rel_err_d_lid_cond_post_relax',
        't_cross_rel_err_d_lid_0p22_yr',
    ):
        out[mk] = m_live[mk]
    if is_4p5:
        out['t_cross_rel_err_T_i_0p05_yr'] = m_live['t_cross_rel_err_T_i_0p05_yr']
        out['t_nu_0d_clamp_1_yr'] = m_live['t_nu_0d_clamp_1_yr']
        out['max_rel_err_T_i_valid_nu_ge_1'] = m_live['max_rel_err_T_i_valid_nu_ge_1']
        out['max_rel_err_d_lid_valid_nu_ge_1_post_relax'] = m_live[
            'max_rel_err_d_lid_valid_nu_ge_1_post_relax'
        ]
    out['solver_last_S'] = y_1d[:, -1].tolist()
    if isinstance(out.get('states_1d_snapshots'), dict):
        snaps_updated = dict(out['states_1d_snapshots'])
        for k_str, snap_val in snaps_updated.items():
            idx_k = int(k_str)
            if 0 <= idx_k < len(t_1d) and isinstance(snap_val, dict):
                snap_copy = dict(snap_val)
                snap_copy['T_i_K'] = float(ti_1d[idx_k])
                snap_copy['d_lid_pass1_m'] = float(dl_1d[idx_k])
                snap_copy['d_lid_cond_m'] = float(dl_cond_1d[idx_k])
                snap_copy['F_top_W_m2'] = float(ft_1d[idx_k])
                snaps_updated[k_str] = snap_copy
        out['states_1d_snapshots'] = snaps_updated
    out['trajectory'] = {
        't_yr': t_1d.tolist(),
        'T_i_1d_K': ti_1d,
        'T_i_0d_K': m_live['ti_0d'].tolist(),
        'd_lid_1d_m': dl_1d,
        'd_lid_cond_1d_m': dl_cond_1d,
        'd_lid_0d_m': m_live['dl_0d'].tolist(),
        'F_top_1d_W_m2': ft_1d,
        'F_top_0d_W_m2': m_live['ft_0d'].tolist(),
        'signed_rel_err_T_i': m_live['s_err_ti'].tolist(),
        'signed_rel_err_d_lid': m_live['s_err_dl'].tolist(),
        'rel_err_T_i': m_live['err_ti'].tolist(),
        'rel_err_d_lid': m_live['err_dl'].tolist(),
    }
    return out


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate or check 0-D stagnant-lid and yield-thinning reference benchmarks.'
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
    p0d = build_default_0d_params()
    meta = data.setdefault('metadata', {})
    meta.pop('eos_tables_release', None)
    meta.pop('eos_md5', None)
    meta['eos_tables'] = NO_TABLE_RECORD
    meta['params_0d'] = asdict(p0d)
    meta['git_commit'] = head_sha

    if args.full:
        base_params = Parameters.from_file(
            REPO_ROOT / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'
        )
        data['trajectory_calibrated_400Myr'] = _recompute_trajectory_block(
            base_params,
            p0d,
            data['trajectory_calibrated_400Myr'],
            s_top_inf=0.25,
            c_theta=0.25,
            t_end_yr=4.0e8,
            n_output_points=65,
        )
        data['trajectory_default_400Myr'] = _recompute_trajectory_block(
            base_params,
            p0d,
            data['trajectory_default_400Myr'],
            s_top_inf=0.22,
            c_theta=0.15,
            t_end_yr=4.0e8,
            n_output_points=65,
        )
        data['trajectory_calibrated_4p5Gyr'] = _recompute_trajectory_block(
            base_params,
            p0d,
            data['trajectory_calibrated_4p5Gyr'],
            s_top_inf=0.25,
            c_theta=0.25,
            t_end_yr=4.5e9,
            n_output_points=129,
        )
        data['trajectory_default_4p5Gyr'] = _recompute_trajectory_block(
            base_params,
            p0d,
            data['trajectory_default_4p5Gyr'],
            s_top_inf=0.22,
            c_theta=0.15,
            t_end_yr=4.5e9,
            n_output_points=129,
        )
    else:
        for blk_key, is_4p5 in (
            ('trajectory_calibrated_400Myr', False),
            ('trajectory_default_400Myr', False),
            ('trajectory_calibrated_4p5Gyr', True),
            ('trajectory_default_4p5Gyr', True),
        ):
            blk = data[blk_key]
            tr = blk['trajectory']
            m_live = _compute_trajectory_metrics_from_arrays(
                np.asarray(tr['t_yr'], dtype=float),
                np.asarray(tr['T_i_1d_K'], dtype=float),
                np.asarray(tr['d_lid_1d_m'], dtype=float),
                np.asarray(tr['d_lid_cond_1d_m'], dtype=float),
                np.asarray(tr['F_top_1d_W_m2'], dtype=float),
                p0d,
                is_4p5gyr=is_4p5,
            )
            if is_4p5:
                blk['t_cross_rel_err_T_i_0p05_yr'] = m_live['t_cross_rel_err_T_i_0p05_yr']
                blk['t_nu_0d_clamp_1_yr'] = m_live['t_nu_0d_clamp_1_yr']
                blk['max_rel_err_T_i_valid_nu_ge_1'] = m_live['max_rel_err_T_i_valid_nu_ge_1']
                blk['max_rel_err_d_lid_valid_nu_ge_1_post_relax'] = m_live[
                    'max_rel_err_d_lid_valid_nu_ge_1_post_relax'
                ]

    tr_4p5 = data['trajectory_calibrated_4p5Gyr']['trajectory']
    t_4p5_yr = np.asarray(tr_4p5['t_yr'], dtype=float)
    ti_1d_4p5 = np.asarray(tr_4p5['T_i_1d_K'], dtype=float)
    dl_1d_4p5 = np.asarray(tr_4p5['d_lid_1d_m'], dtype=float)
    ft_1d_4p5 = np.asarray(tr_4p5['F_top_1d_W_m2'], dtype=float)
    data['a_nu_provenance'] = compute_a_nu_provenance_block(
        p0d, t_4p5_yr, ti_1d_4p5, dl_1d_4p5, ft_1d_4p5
    )
    data['yield_sweep_0d_at_1600K'] = compute_yield_sweep_0d_at_1600k(p0d)
    data['acceptance_summary'] = compute_acceptance_summary(data)

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
