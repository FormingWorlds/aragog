"""Compute solidus reference values from the Wolf and Bower (2018) melting curve table.

Reads ``interior/melting_curves/wolf_bower_2018/r15728072/solidus.dat`` (Zenodo
record ``10.5281/zenodo.15728072``) from ``FWL_DATA`` and prints the solidus
temperatures, homologous activation enthalpies, secant slope, linear-fallback
comparison at 25 GPa, zero-pressure viscosity ratio at 1600 K, and the 175 kyr
checkpoint profile solidus-indicator comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np

from aragog.rheology import R_GAS, SolidRheologyParams
from aragog.rheology_lid import (
    SOLIDUS_MELT_FRACTION_THRESHOLD,
    SOLIDUS_MELT_FRACTION_WIDTH,
    compute_stagnant_lid_state,
)

RELATIVE_SOLIDUS_PATH = Path('interior/melting_curves/wolf_bower_2018/r15728072/solidus.dat')
EXPECTED_MD5 = '66b297a120c79e7e58de5761ac40f4c4'


def resolve_solidus_table_path(table_arg: Path | None = None) -> Path:
    """Resolve the Wolf and Bower (2018) solidus table path from ``--table`` or ``FWL_DATA``.

    Parameters
    ----------
    table_arg : Path | None, optional
        Explicit path to ``solidus.dat`` passed via ``--table``.

    Returns
    -------
    Path
        Path to ``solidus.dat`` in release ``r15728072``.

    Raises
    ------
    FileNotFoundError
        If neither ``--table`` nor ``FWL_DATA`` resolves to an existing ``solidus.dat`` file.
    """
    if table_arg is not None:
        path = Path(table_arg)
        if path.is_file():
            return path
        raise FileNotFoundError(
            f'Explicit --table path does not exist: {path} (expected {RELATIVE_SOLIDUS_PATH}).'
        )
    fwl_data = os.environ.get('FWL_DATA')
    if fwl_data:
        path = Path(fwl_data) / RELATIVE_SOLIDUS_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'Could not find {RELATIVE_SOLIDUS_PATH}: set FWL_DATA or pass --table <path>.'
    )


def load_solidus_curve(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load pressure [Pa] and solidus temperature [K] from ``solidus.dat``.

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


def main(argv: list[str] | None = None) -> None:
    """Print all solidus reference quantities and 175 kyr solidus-indicator node trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--table',
        type=Path,
        default=None,
        help=f'Path to solidus.dat (defaults to $FWL_DATA/{RELATIVE_SOLIDUS_PATH}).',
    )
    args = parser.parse_args(argv)
    table_path = resolve_solidus_table_path(args.table)
    p_table, t_table, md5_hex = load_solidus_curve(table_path)
    print(f'Table path: {table_path}')
    print('Release: r15728072 (Zenodo 10.5281/zenodo.15728072)')
    print(f'MD5: {md5_hex} (expected {EXPECTED_MD5}, match={md5_hex == EXPECTED_MD5})')

    p_gpa = np.array([0.0, 25.0, 50.0, 100.0, 135.0])
    p_pa = p_gpa * 1.0e9
    t_sol = np.interp(p_pa, p_table, t_table)

    print('\n--- Section 1 Table Values ---')
    print('P_GPa | T_sol_exact_K | T_sol_1dec_K | H_g14_kJ_mol | H_g25_kJ_mol | ratio_0p9_g25')
    ratio_0p9_g25 = math.exp(25.0 / 9.0)
    for pg, ts in zip(p_gpa, t_sol):
        h14 = 14.0 * R_GAS * ts * 1.0e-3
        h25 = 25.0 * R_GAS * ts * 1.0e-3
        print(
            f'{pg:5.0f} | {ts:13.4f} | {ts:12.1f} | {h14:12.1f} | {h25:12.1f} | {ratio_0p9_g25:13.3f}'
        )

    print(
        '\nt_sol_grid for tests/test_rheology_kernel.py: '
        + '['
        + ', '.join(f'{ts:.4f}' for ts in t_sol)
        + ']'
    )

    secant_exact = (t_sol[-1] - t_sol[0]) / 135.0
    secant_1dec = (round(float(t_sol[-1]), 1) - round(float(t_sol[0]), 1)) / 135.0
    t_lin_25 = 1400.0 + 24.0 * 25.0
    err_25_exact = (t_sol[1] - t_lin_25) / t_sol[1] * 100.0
    err_25_1dec = (round(float(t_sol[1]), 1) - t_lin_25) / round(float(t_sol[1]), 1) * 100.0

    params = SolidRheologyParams()
    ratio_eta_exact = math.exp(
        params.g_homologous * (float(t_sol[0]) - params.t_sol_0) / params.arrhenius_t_ref
    )
    ratio_eta_1dec = math.exp(
        params.g_homologous
        * (round(float(t_sol[0]), 1) - params.t_sol_0)
        / params.arrhenius_t_ref
    )

    print('\n--- Derived Slopes, Errors, and Viscosity Ratios ---')
    print(
        f'T_sol(0 GPa): {t_sol[0]:.4f} K (1-dec: {t_sol[0]:.1f} K, int: {round(float(t_sol[0])):d} K)'
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
        f'Linear fallback at 25 GPa: {t_lin_25:.1f} K vs table {t_sol[1]:.1f} K '
        f'({err_25_exact:.2f}% low, 1-dec: {err_25_1dec:.2f}% -> {round(float(err_25_exact)):.0f}% low)'
    )
    print(
        f'Table-to-linear eta_diff(1600 K, 0) ratio at g_hom=25, t_sol_0=1400 K: '
        f'exact={ratio_eta_exact:.6f} ({ratio_eta_exact:.3f}x / {ratio_eta_exact:.2f}x), '
        f'from 1380.6 K={ratio_eta_1dec:.6f} ({ratio_eta_1dec:.3f}x / {ratio_eta_1dec:.2f}x)'
    )

    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = repo_root / 'tests' / 'reference' / 'checkpoint_profile_175kyr.npz'
    if ckpt_path.is_file():
        d = np.load(ckpt_path)
        r = d['r_basic']
        t_prof = d['T_basic']
        p_prof = d['P_basic']
        phi = d['phi_basic']
        t_sol_prof = np.interp(p_prof, p_table, t_table)
        params_ckpt = SolidRheologyParams(enabled=True, t_sol_0=1400.0)
        st = compute_stagnant_lid_state(
            radii=r,
            temperature=t_prof,
            pressure=p_prof,
            convective_flux=d['F_conv'],
            total_flux=d['F_tot'],
            solidus_temperature=t_sol_prof,
            t_sol_0=1400.0,
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

        print(
            '\n--- 175 kyr Profile Nodes with dw_hot_sol > 1e-6 or Top 12 (r15728072 solidus.dat) ---'
        )
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
