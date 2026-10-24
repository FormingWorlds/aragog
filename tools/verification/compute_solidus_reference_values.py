"""Compute solidus reference values from the PALEOS MgSiO3 solidus or a table file.

Evaluates the PALEOS MgSiO3 solidus ``T_sol(P) = 0.8 * T_liq(P)`` (or reads a
two-column ``(P [Pa], T_sol [K])`` ASCII table when ``--table`` is passed) and
prints the solidus temperatures, homologous activation enthalpies, secant and
high-pressure slopes, linear-fallback deviations and crossings over 0..135 GPa,
zero-pressure viscosity ratio at 1600 K, and the 175 kyr checkpoint profile
solidus-indicator comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from aragog.rheology import R_GAS, SolidRheologyParams  # noqa: E402
from aragog.rheology_lid import (  # noqa: E402
    SOLIDUS_MELT_FRACTION_THRESHOLD,
    SOLIDUS_MELT_FRACTION_WIDTH,
    compute_stagnant_lid_state,
)

RELATIVE_SOLIDUS_PATH = Path('interior/melting_curves/wolf_bower_2018/r15728072/solidus.dat')
EXPECTED_MD5: str | None = None
_PALEOS_P0_GPA = 2.551686137257537
PALEOS_MUSHY_ZONE_FACTOR = 0.8


def paleos_liquidus_temperature(pressure_pa: np.ndarray) -> np.ndarray:
    """Evaluate the PALEOS MgSiO3 liquidus temperature [K] as a function of pressure [Pa].

    Uses the piecewise Simon-Glatzel fit from Belonoshko et al. (2005) below
    ``_PALEOS_P0_GPA`` and Fei et al. (2021) above ``_PALEOS_P0_GPA``, with the
    continuous zero-pressure limit 1831.0 K at ``P = 0``.

    Parameters
    ----------
    pressure_pa : np.ndarray
        Pressure array [Pa] (non-negative).

    Returns
    -------
    np.ndarray
        Liquidus temperature array [K].
    """
    p_gpa = np.maximum(np.asarray(pressure_pa, dtype=float), 0.0) * 1.0e-9
    p_hi = np.where(p_gpa >= _PALEOS_P0_GPA, p_gpa, _PALEOS_P0_GPA)
    return np.where(
        p_gpa < _PALEOS_P0_GPA,
        1831.0 * (1.0 + p_gpa / 4.6) ** 0.33,
        6000.0 * (p_hi / 140.0) ** 0.26,
    )


def paleos_solidus_curve(
    p_max_pa: float = 135.0e9, n_points: int = 135001
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense pressure [Pa] and PALEOS solidus temperature [K] arrays.

    Parameters
    ----------
    p_max_pa : float, default 135.0e9
        Maximum pressure [Pa].
    n_points : int, default 135001
        Number of uniform pressure grid points from 0 to ``p_max_pa``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pressure array [Pa] and solidus temperature array ``0.8 * T_liq(P)`` [K].
    """
    p_table = np.linspace(0.0, p_max_pa, n_points)
    t_table = PALEOS_MUSHY_ZONE_FACTOR * paleos_liquidus_temperature(p_table)
    return p_table, t_table


def resolve_solidus_table_path(table_arg: Path | None = None) -> Path:
    """Resolve an explicit ``--table`` path or fall back to ``FWL_DATA``.

    Parameters
    ----------
    table_arg : Path | None, optional
        Explicit path to a two-column solidus table passed via ``--table``.

    Returns
    -------
    Path
        Resolved path to the solidus table file.

    Raises
    ------
    FileNotFoundError
        If the requested table file does not exist.
    """
    if table_arg is not None:
        path = Path(table_arg)
        if path.is_file():
            return path
        raise FileNotFoundError(f'Explicit --table path does not exist: {path}.')
    fwl_data = os.environ.get('FWL_DATA')
    if fwl_data:
        path = Path(fwl_data) / RELATIVE_SOLIDUS_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'Could not find {RELATIVE_SOLIDUS_PATH}: set FWL_DATA or pass --table <path>.'
    )


def load_solidus_curve(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load pressure [Pa] and solidus temperature [K] from a two-column ASCII table.

    Parameters
    ----------
    path : Path
        Path to the two-column ASCII solidus table.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, str]
        Pressure array [Pa], solidus temperature array [K], and file MD5 hex digest.
    """
    md5_hex = hashlib.md5(path.read_bytes()).hexdigest()
    raw = np.loadtxt(path, comments='#')
    return raw[:, 0], raw[:, 1], md5_hex


def find_solidus_crossings(
    p_table: np.ndarray,
    t_table: np.ndarray,
    t_sol_0: float,
    dt_sol_dp: float,
    p_max_pa: float = 135.0e9,
) -> list[float]:
    """Return all crossing pressures [GPa] of the linear and tabulated solidi up to ``p_max_pa``.

    Parameters
    ----------
    p_table : np.ndarray
        Monotonically increasing table pressures [Pa].
    t_table : np.ndarray
        Tabulated solidus temperatures [K] matching ``p_table``.
    t_sol_0 : float
        Zero-pressure linear solidus intercept [K].
    dt_sol_dp : float
        Linear solidus pressure slope [K/Pa].
    p_max_pa : float, default 135.0e9
        Upper pressure bound [Pa] of the mantle domain.

    Returns
    -------
    list[float]
        Linearly interpolated crossing pressures [GPa] in ``[0, p_max_pa * 1e-9]``,
        deduplicated when a root coincides with an interior grid node.
    """
    p_arr = np.asarray(p_table, dtype=float)
    t_arr = np.asarray(t_table, dtype=float)
    mantle_mask = p_arr <= p_max_pa
    p_mantle = p_arr[mantle_mask]
    t_mantle = t_arr[mantle_mask]
    if p_mantle.size > 0 and p_mantle[-1] < p_max_pa and p_arr[-1] >= p_max_pa:
        p_mantle = np.append(p_mantle, p_max_pa)
        t_mantle = np.append(t_mantle, float(np.interp(p_max_pa, p_arr, t_arr)))
    p_gpa = p_mantle * 1.0e-9
    dt_sol_dp_gpa = dt_sol_dp * 1.0e9
    diff_lin = (t_sol_0 + dt_sol_dp_gpa * p_gpa) - t_mantle
    crossings_gpa: list[float] = []
    for idx in range(len(diff_lin) - 1):
        d0 = float(diff_lin[idx])
        d1 = float(diff_lin[idx + 1])
        if d0 == 0.0 and d1 == 0.0:
            continue
        if d0 * d1 <= 0.0:
            p_cross = (
                float(p_gpa[idx])
                if d1 == d0
                else float(p_gpa[idx] - d0 * (p_gpa[idx + 1] - p_gpa[idx]) / (d1 - d0))
            )
            if not crossings_gpa or not math.isclose(
                p_cross, crossings_gpa[-1], rel_tol=0.0, abs_tol=1.0e-9
            ):
                crossings_gpa.append(p_cross)
    return crossings_gpa


def format_crossings_description(crossings_gpa: list[float]) -> str:
    """Format every solidus crossing pressure [GPa] without dropping third or later roots."""
    if not crossings_gpa:
        return 'no crossings over 0..135 GPa (crossings_gpa=[])'
    all_2dec = [round(c, 2) for c in crossings_gpa]
    if len(crossings_gpa) >= 2 and crossings_gpa[0] < 1.0:
        rest_1dec = ', '.join(f'{c:.1f}' for c in crossings_gpa[1:])
        return (
            f'crossings below 1 GPa ({crossings_gpa[0]:.2f} GPa) and at {rest_1dec} GPa '
            f'(all: {all_2dec} GPa)'
        )
    all_1dec = ', '.join(f'{c:.1f}' for c in crossings_gpa)
    return f'crossings at {all_1dec} GPa (all: {all_2dec} GPa)'


def main(argv: list[str] | None = None) -> None:
    """Print all solidus reference quantities and 175 kyr solidus-indicator node trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--table',
        type=Path,
        default=None,
        help='Optional path to a two-column solidus.dat table (defaults to PALEOS 0.8 * T_liq).',
    )
    parser.add_argument(
        '--allow-md5-mismatch',
        action='store_true',
        help='Continue instead of exiting non-zero when EXPECTED_MD5 is set and mismatches.',
    )
    args = parser.parse_args(argv)
    if args.table is not None:
        table_path = resolve_solidus_table_path(args.table)
        p_table, t_table, md5_hex = load_solidus_curve(table_path)
        md5_match = EXPECTED_MD5 is None or md5_hex == EXPECTED_MD5
        print(f'Table path: {table_path}')
        print(f'MD5: {md5_hex} (expected {EXPECTED_MD5}, match={md5_match})')
        if not md5_match and not args.allow_md5_mismatch:
            raise SystemExit(
                f'MD5 mismatch for {table_path}: got {md5_hex}, expected {EXPECTED_MD5} '
                '(pass --allow-md5-mismatch to override).'
            )
    else:
        p_table, t_table = paleos_solidus_curve()
        print(
            'Table source: PALEOS MgSiO3 solidus T_sol(P) = 0.8 * T_liq(P) (mushy_zone_factor=0.8)'
        )

    if p_table.size == 0 or float(p_table[-1]) < 135.0e9:
        raise ValueError(
            f'Solidus table pressure range ends at {float(p_table[-1]) if p_table.size else 0.0:.3e} Pa; '
            'must cover at least 135.0e9 Pa for whole-mantle reference metrics.'
        )

    params = SolidRheologyParams()
    p_gpa = np.array([0.0, 25.0, 50.0, 100.0, 135.0])
    p_pa = p_gpa * 1.0e9
    t_sol = np.interp(p_pa, p_table, t_table)

    print('\n--- Section 1 Table Values ---')
    print('P_GPa | T_sol_exact_K | T_sol_1dec_K | H_g14_kJ_mol | H_g25_kJ_mol | ratio_0p9_g25')
    ratio_0p9_g25 = math.exp(params.g_homologous / 9.0)
    for pg, ts in zip(p_gpa, t_sol):
        h14 = 14.0 * R_GAS * ts * 1.0e-3
        h25 = params.g_homologous * R_GAS * ts * 1.0e-3
        print(
            f'{pg:5.0f} | {ts:13.4f} | {ts:12.1f} | {h14:12.1f} | {h25:12.1f} | {ratio_0p9_g25:13.3f}'
        )

    print(
        '\nt_sol_grid for tests/test_rheology_kernel.py: '
        + '['
        + ', '.join(f'{ts:.4f}' for ts in t_sol)
        + ']'
    )

    dt_sol_dp_gpa = params.dt_sol_dp * 1.0e9
    secant_exact = (t_sol[-1] - t_sol[0]) / 135.0
    secant_1dec = (round(float(t_sol[-1]), 1) - round(float(t_sol[0]), 1)) / 135.0
    err_0_exact = (params.t_sol_0 - t_sol[0]) / t_sol[0] * 100.0
    err_0_1dec = (
        (params.t_sol_0 - round(float(t_sol[0]), 1)) / round(float(t_sol[0]), 1) * 100.0
    )
    t_lin_25 = params.t_sol_0 + dt_sol_dp_gpa * 25.0
    err_25_exact = (t_sol[1] - t_lin_25) / t_sol[1] * 100.0
    err_25_1dec = (round(float(t_sol[1]), 1) - t_lin_25) / round(float(t_sol[1]), 1) * 100.0
    t_lin_135 = params.t_sol_0 + dt_sol_dp_gpa * 135.0
    err_135_exact = (t_lin_135 - t_sol[-1]) / t_sol[-1] * 100.0
    err_135_1dec = (t_lin_135 - round(float(t_sol[-1]), 1)) / round(float(t_sol[-1]), 1) * 100.0
    slope_25_135 = (t_sol[-1] - t_sol[1]) / (135.0 - 25.0)
    crossings_gpa = find_solidus_crossings(
        p_table, t_table, params.t_sol_0, params.dt_sol_dp, p_max_pa=135.0e9
    )

    p_dense = np.linspace(0.0, 135.0e9, 135001)
    p_dense_gpa = p_dense * 1.0e-9
    t_dense = np.interp(p_dense, p_table, t_table)
    t_sol_1gpa = float(np.interp(1.0e9, p_table, t_table))
    diff_dense = (params.t_sol_0 + params.dt_sol_dp * p_dense) - t_dense
    idx_max_abs = int(np.argmax(np.abs(diff_dense)))
    max_abs_dev_k = float(np.abs(diff_dense[idx_max_abs]))
    max_signed_dev_k = float(diff_dense[idx_max_abs])
    p_max_abs_gpa = float(p_dense_gpa[idx_max_abs])
    rms_dev_k = float(np.sqrt(np.mean(diff_dense**2)))

    rel_diff_dense = diff_dense / t_dense * 100.0
    idx_max_rel = int(np.argmax(np.abs(rel_diff_dense)))
    max_signed_rel_pct = float(rel_diff_dense[idx_max_rel])
    p_max_rel_gpa = float(p_dense_gpa[idx_max_rel])

    slope_mm_pa = float((t_dense[-1] - t_dense[0]) / 135.0e9)
    slope_mm_gpa = slope_mm_pa * 1.0e9
    excess_mm = t_dense - (t_dense[0] + slope_mm_pa * p_dense)
    idx_mm_peak = int(np.argmax(np.abs(excess_mm)))
    t0_mm = float(t_dense[0] + 0.5 * (float(np.max(excess_mm)) + float(np.min(excess_mm))))
    diff_mm = (t0_mm + slope_mm_pa * p_dense) - t_dense
    max_abs_mm_k = float(np.max(np.abs(diff_mm)))
    p_mm_peak_gpa = float(p_dense_gpa[idx_mm_peak])
    rms_mm_k = float(np.sqrt(np.mean(diff_mm**2)))
    cross_mm_desc = format_crossings_description(
        find_solidus_crossings(p_table, t_table, t0_mm, slope_mm_pa, p_max_pa=135.0e9)
    )

    slope_l2_gpa_arr, t0_l2_arr = np.polyfit(p_dense_gpa, t_dense, 1)
    t0_l2 = float(t0_l2_arr)
    slope_l2_gpa = float(slope_l2_gpa_arr)
    slope_l2_pa = slope_l2_gpa * 1.0e-9
    diff_l2 = (t0_l2 + slope_l2_pa * p_dense) - t_dense
    idx_l2_max = int(np.argmax(np.abs(diff_l2)))
    max_abs_l2_k = float(np.abs(diff_l2[idx_l2_max]))
    p_l2_max_gpa = float(p_dense_gpa[idx_l2_max])
    rms_l2_k = float(np.sqrt(np.mean(diff_l2**2)))
    cross_l2_desc = format_crossings_description(
        find_solidus_crossings(p_table, t_table, t0_l2, slope_l2_pa, p_max_pa=135.0e9)
    )

    t_sol_0_1dec = round(float(t_sol[0]), 1)
    ratio_eta_exact = math.exp(
        params.g_homologous * (float(t_sol[0]) - params.t_sol_0) / params.arrhenius_t_ref
    )
    ratio_eta_1dec = math.exp(
        params.g_homologous * (t_sol_0_1dec - params.t_sol_0) / params.arrhenius_t_ref
    )

    print('\n--- Derived Slopes, Errors, and Viscosity Ratios ---')
    print(
        f'T_sol(0 GPa): {t_sol[0]:.4f} K (1-dec: {t_sol[0]:.1f} K, int: {round(float(t_sol[0])):d} K)'
    )
    print(
        f'T_sol(1 GPa): {t_sol_1gpa:.4f} K (1-dec: {t_sol_1gpa:.1f} K, int: {round(t_sol_1gpa):d} K)'
    )
    print(
        f'T_sol(135 GPa): {t_sol[-1]:.4f} K '
        f'(1-dec: {t_sol[-1]:.1f} K, int: {round(float(t_sol[-1])):d} K)'
    )
    print(
        f'Secant slope 0..135 GPa: {secant_exact:.4f} K/GPa '
        f'(from 1-dec: {secant_1dec:.4f} K/GPa -> {secant_exact:.1f} K/GPa)'
    )
    print(
        f'High-pressure slope 25..135 GPa: {slope_25_135:.4f} K/GPa '
        f'(-> {slope_25_135:.1f} K/GPa)'
    )
    print(
        f'Linear fallback at 0 GPa: {params.t_sol_0:.1f} K vs table {t_sol[0]:.1f} K '
        f'({err_0_exact:+.2f}%, 1-dec: {err_0_1dec:+.2f}% -> {err_0_exact:+.1f}%)'
    )
    err_25_dir = 'low' if err_25_exact >= 0.0 else 'high'
    print(
        f'Linear fallback at 25 GPa: {t_lin_25:.1f} K vs table {t_sol[1]:.1f} K '
        f'({abs(err_25_exact):.2f}% {err_25_dir}, 1-dec: {abs(err_25_1dec):.2f}% -> '
        f'{abs(err_25_exact):.1f}% {err_25_dir})'
    )
    cross_desc = format_crossings_description(crossings_gpa)
    print(
        f'Linear fallback at 135 GPa: {t_lin_135:.1f} K vs table {t_sol[-1]:.1f} K '
        f'({err_135_exact:+.2f}%, 1-dec: {err_135_1dec:+.2f}% -> {err_135_exact:+.1f}%; '
        f'{cross_desc})'
    )
    print(
        f'Max absolute linear deviation over 0..135 GPa: {max_abs_dev_k:.2f} K '
        f'({max_signed_dev_k:+.2f} K -> {max_signed_dev_k:+.1f} K) at {p_max_abs_gpa:.3f} GPa '
        f'({p_max_abs_gpa:.1f} GPa); RMS deviation: {rms_dev_k:.2f} K ({rms_dev_k:.1f} K)'
    )
    print(
        f'Max relative linear deviation over 0..135 GPa: {max_signed_rel_pct:+.2f}% '
        f'({max_signed_rel_pct:+.1f}%) at {p_max_rel_gpa:.3f} GPa ({p_max_rel_gpa:.1f} GPa)'
    )
    print(
        f'Chebyshev minimax line over 0..135 GPa: {t0_mm:.1f} K + {slope_mm_gpa:.2f} K/GPa * P '
        f'(exact: {t0_mm:.2f} K + {slope_mm_gpa:.4f} K/GPa * P); max abs deviation: '
        f'{max_abs_mm_k:.2f} K ({max_abs_mm_k:.1f} K) at {p_mm_peak_gpa:.3f} GPa; '
        f'RMS: {rms_mm_k:.2f} K ({rms_mm_k:.1f} K); {cross_mm_desc}'
    )
    print(
        f'Least-squares (L2) line over 0..135 GPa: {t0_l2:.1f} K + {slope_l2_gpa:.2f} K/GPa * P '
        f'(exact: {t0_l2:.2f} K + {slope_l2_gpa:.4f} K/GPa * P); max abs deviation: '
        f'{max_abs_l2_k:.2f} K ({max_abs_l2_k:.1f} K) at {p_l2_max_gpa:.1f} GPa; '
        f'RMS: {rms_l2_k:.2f} K ({rms_l2_k:.1f} K); {cross_l2_desc}'
    )
    print(
        f'Table-to-linear eta_diff({params.arrhenius_t_ref:.0f} K, 0) ratio at '
        f'g_hom={params.g_homologous:g}, t_sol_0={params.t_sol_0:g} K: '
        f'exact={ratio_eta_exact:.6f} ({ratio_eta_exact:.3f}x / {ratio_eta_exact:.2f}x), '
        f'from {t_sol_0_1dec:.1f} K={ratio_eta_1dec:.6f} ({ratio_eta_1dec:.3f}x / {ratio_eta_1dec:.2f}x)'
    )

    ckpt_path = REPO_ROOT / 'tests' / 'reference' / 'checkpoint_profile_175kyr.npz'
    if ckpt_path.is_file():
        d = np.load(ckpt_path)
        r = d['r_basic']
        t_prof = d['T_basic']
        p_prof = d['P_basic']
        phi = d['phi_basic']
        t_sol_prof = (
            PALEOS_MUSHY_ZONE_FACTOR * paleos_liquidus_temperature(p_prof)
            if args.table is None
            else np.interp(p_prof, p_table, t_table)
        )
        params_ckpt = SolidRheologyParams(enabled=True)
        st = compute_stagnant_lid_state(
            radii=r,
            temperature=t_prof,
            pressure=p_prof,
            convective_flux=d['F_conv'],
            total_flux=d['F_tot'],
            solidus_temperature=t_sol_prof,
            melt_fraction=phi,
            params=params_ckpt,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
        )
        scale_cell = params_ckpt.lid_mask_width_cells * np.maximum(
            np.abs(np.gradient(t_prof)), 1.0
        )
        scale_mask = np.minimum(
            scale_cell, params_ckpt.lid_mask_width_cells * max(float(st['dT_rh']), 1.0)
        )
        w_th = 0.5 * (1.0 + np.tanh((t_prof - t_sol_prof) / scale_cell))
        zero_sol = 0.5 * (
            1.0 + np.tanh(-SOLIDUS_MELT_FRACTION_THRESHOLD / SOLIDUS_MELT_FRACTION_WIDTH)
        )
        raw_sol = 0.5 * (
            1.0 + np.tanh((phi - SOLIDUS_MELT_FRACTION_THRESHOLD) / SOLIDUS_MELT_FRACTION_WIDTH)
        )
        w_phi = np.clip((raw_sol - zero_sol) / (1.0 - zero_sol), 0.0, 1.0)
        w_comb = np.maximum(w_th, w_phi)
        w_th_mask = 0.5 * (1.0 + np.tanh((t_prof - t_sol_prof) / scale_mask))
        w_comb_mask = np.maximum(w_th_mask, w_phi)

        print('\n--- 175 kyr Profile Nodes with dw_hot_sol > 1e-6 or Top 12 ---')
        print(
            'k  | P_GPa  | T_K     | T_sol_K | T-T_sol  | phi     | w_th     | w_comb   | dw_hot_sol | dw_mask'
        )
        for k in range(len(r)):
            if (
                (w_comb[k] - w_th[k]) > 1.0e-6
                or (w_comb_mask[k] - w_th_mask[k]) > 1.0e-6
                or k >= len(r) - 12
            ):
                print(
                    f'{k:2d} | {p_prof[k] * 1e-9:6.3f} | {t_prof[k]:7.2f} | {t_sol_prof[k]:7.2f} | '
                    f'{t_prof[k] - t_sol_prof[k]:8.2f} | {phi[k]:7.5f} | {w_th[k]:8.6f} | '
                    f'{w_comb[k]:8.6f} | {w_comb[k] - w_th[k]:10.6f} | {w_comb_mask[k] - w_th_mask[k]:8.6f}'
                )


if __name__ == '__main__':
    main()
