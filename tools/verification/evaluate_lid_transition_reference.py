"""Evaluate independent stagnant-lid transition reference values from geotherm formulas.

This script evaluates interior temperature T_i, Arrhenius viscosity eta_i, and
mixture viscosity mu_i at the 7 sub-threshold transition points where
0 < w_weight < 1. It implements the closed-form physical formulas for geotherm
endpoint blending documented in Section 3 of
docs/Explanations/solid_state_convection.md using NumPy only, without importing
any aragog modules for its reference computation. Physical constants and default
parameters are parsed from the repository source via ast.parse.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import numpy as np

TRANSITION_POINTS: tuple[tuple[float, float], ...] = (
    (0.05, 6.00843e-5),
    (0.05, 1.0e-4),
    (0.10, 6.00843e-5),
    (0.10, 1.0e-4),
    (0.10, 1.0e-3),
    (0.10, 0.03),
    (0.20, 0.03),
)


def load_rheology_constants(repo_root: Path) -> dict[str, Any]:
    """Parse physical constants and parameter defaults from aragog source via AST.

    Parameters
    ----------
    repo_root : Path
        Root path of the repository.

    Returns
    -------
    dict[str, Any]
        Dictionary of extracted constants and default parameters.
    """
    rheology_py = repo_root / 'src' / 'aragog' / 'rheology.py'
    tree = ast.parse(rheology_py.read_text(encoding='utf-8'), filename=str(rheology_py))
    consts: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'R_GAS':
                    consts['R_GAS'] = ast.literal_eval(node.value)
        elif isinstance(node, ast.ClassDef) and node.name == 'SolidRheologyParams':
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.value is not None
                ):
                    if isinstance(item.value, ast.Call):
                        consts[item.target.id] = float(ast.literal_eval(item.value.args[0]))
                    else:
                        consts[item.target.id] = ast.literal_eval(item.value)
    return consts


def evaluate_lid_transition_state(
    f0_cfg: float,
    f_test: float,
    *,
    n: int = 50,
    visc_solid: float = 1.0e21,
    visc_mix_val: float = 1.0e15,
    consts: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Evaluate T_i, eta_i, and mu_i from the transition formulas with NumPy only.

    Follows 'Geotherm Endpoint Blending and Smooth Ramp Weighting' in
    docs/Explanations/solid_state_convection.md.

    Parameters
    ----------
    f0_cfg : float
        Interior convective flux fraction threshold parameter.
    f_test : float
        Uniform convective heat flux fraction across the column.
    n : int, optional
        Number of grid nodes, by default 50.
    visc_solid : float, optional
        Solid reference viscosity [Pa s], by default 1.0e21.
    visc_mix_val : float, optional
        Uniform mixture viscosity [Pa s], by default 1.0e15.
    consts : dict[str, Any] | None, optional
        Constants dictionary parsed from source, or None to use default values.

    Returns
    -------
    tuple[float, float, float]
        Evaluated (T_i, eta_i, mu_i) in SI units.
    """
    if consts is None:
        g_hom = 25.0
        t_sol_0 = 1400.0
        dt_sol_dp = 2.4e-8
        t_ref = 1600.0
    else:
        g_hom = float(consts.get('g_homologous', 25.0))
        t_sol_0 = float(consts.get('t_sol_0', 1400.0))
        dt_sol_dp = float(consts.get('dt_sol_dp', 2.4e-8))
        t_ref = float(consts.get('arrhenius_t_ref', 1600.0))

    t = np.linspace(2600.0, 1400.0, n)
    p = np.linspace(135.0e9, 0.0, n)
    visc_mix = np.full(n, visc_mix_val)

    # 1. Convective activity indicator
    w_conv_raw = 0.5 * (1.0 + np.tanh((f_test - f0_cfg) / 0.01))
    w_conv_0 = 0.5 * (1.0 + np.tanh(-f0_cfg / 0.01))
    w_conv_int_val = max(w_conv_raw - w_conv_0, 0.0)

    w_conv_int = np.zeros(n)
    w_conv_int[1:-1] = w_conv_int_val

    # 2. Surface-down scanning product for convective fallback distribution
    w_surf = w_conv_int[::-1]
    omw = np.concatenate([np.ones(1), 1.0 - w_surf[:-1]])
    w_conv_scan = (w_surf * np.cumprod(omw))[::-1]
    w_conv_sum = np.sum(w_conv_scan)
    w_weight = np.tanh(w_conv_int_val / 1.0e-6)

    fallback_0 = np.zeros(n)
    fallback_0[1] = 1.0
    w_conv_tilde = (
        w_weight * (w_conv_scan / max(w_conv_sum, 1.0e-12)) + (1.0 - w_weight) * fallback_0
    )

    # 3. Top-of-convection selector distribution
    f_int_max = f_test
    f_ti_thresh = 0.9 * f_int_max
    w_ti_width = 0.02 * f_int_max
    w_ti_val = 0.5 * (1.0 + np.tanh((f_test - f_ti_thresh) / w_ti_width))
    w_ti = np.zeros(n)
    w_ti[1:-1] = w_ti_val

    w_surf_ti = w_ti[::-1]
    omw_ti = np.concatenate([np.ones(1), 1.0 - w_surf_ti[:-1]])
    w_ti_scan = (w_surf_ti * np.cumprod(omw_ti))[::-1]
    w_ti_sum = np.sum(w_ti_scan)
    w_base_weight_ti = np.tanh(w_ti_val / 1.0e-6)
    w_ti_tilde = (
        w_base_weight_ti * (w_ti_scan / max(w_ti_sum, 1.0e-12))
        + (1.0 - w_base_weight_ti) * w_conv_tilde
    )

    # 4. Composite distribution and interior state
    w_tilde = w_weight * w_ti_tilde + (1.0 - w_weight) * w_conv_tilde

    t_0 = t[1]
    p_0 = p[1]
    t_1 = float(np.sum(w_ti_tilde * t))
    p_1 = float(np.sum(w_ti_tilde * p))
    t_i = float(np.sum(w_tilde * t))

    # 5. Geotherm endpoint blend ('Geotherm Endpoint Blending and Smooth Ramp Weighting' in docs/Explanations/solid_state_convection.md)
    w_eff = float(np.clip((t_0 - t_i) / (t_0 - t_1), 0.0, 1.0))

    def arrhenius_viscosity(t_val: float, p_val: float) -> float:
        t_sol = t_sol_0 + dt_sol_dp * p_val
        return float(visc_solid * np.exp(g_hom * (t_sol / t_val - t_sol_0 / t_ref)))

    eta_arr_0 = arrhenius_viscosity(t_0, p_0)
    eta_arr_1 = arrhenius_viscosity(t_1, p_1)

    log_eta_arr = w_eff * np.log10(eta_arr_1) + (1.0 - w_eff) * np.log10(eta_arr_0)

    log_mu_mix = float(np.sum(w_tilde * np.log10(visc_mix)))
    log_eta_i = w_weight * log_mu_mix + (1.0 - w_weight) * log_eta_arr
    eta_i = float(10.0**log_eta_i)
    mu_i = eta_i

    return t_i, eta_i, mu_i


def main() -> None:
    """CLI entry point for evaluating and verifying transition reference values."""
    parser = argparse.ArgumentParser(
        description='Evaluate independent reference values for stagnant lid transition points.'
    )
    parser.add_argument(
        '--compare-kernel',
        action='store_true',
        help='Import aragog and compute max relative difference to kernel.',
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    consts = load_rheology_constants(repo_root)

    results: dict[tuple[float, float], tuple[float, float, float]] = {}
    print('Independent evaluation of geotherm transition reference values (NumPy only):')
    print(f'{"(f0, f_test)":<22} {"T_i [K]":<26} {"eta_i [Pa s]":<28} {"mu_i [Pa s]":<28}')
    print('-' * 104)

    for f0_cfg, f_test in TRANSITION_POINTS:
        t_i, eta_i, mu_i = evaluate_lid_transition_state(f0_cfg, f_test, consts=consts)
        results[(f0_cfg, f_test)] = (t_i, eta_i, mu_i)
        print(
            f'({f0_cfg:.2f}, {f_test:.5e}): T_i={t_i:<20.16e} eta_i={eta_i:<22.16e} mu_i={mu_i:<22.16e}'
        )

    if args.compare_kernel:
        import sys

        sys.path.insert(0, str(repo_root / 'src'))
        from aragog.rheology import SolidRheologyParams
        from aragog.rheology_lid import compute_stagnant_lid_state

        n = 50
        r_l1 = np.linspace(3.48e6, 6.371e6, n)
        t_l1 = np.linspace(2600.0, 1400.0, n)
        p_l1 = np.linspace(135.0e9, 0.0, n)
        phi_l1 = np.zeros(n)
        visc_mix_l1 = np.full(n, 1.0e15)

        max_rel_t = 0.0
        max_rel_eta = 0.0
        print('\nKernel comparison (compute_stagnant_lid_state):')
        for f0_cfg, f_test in TRANSITION_POINTS:
            t_i_indep, eta_i_indep, _ = results[(f0_cfg, f_test)]
            st = compute_stagnant_lid_state(
                radii=r_l1,
                temperature=t_l1,
                pressure=p_l1,
                convective_flux=np.full(n, f_test),
                total_flux=np.ones(n),
                solidus_temperature=None,
                melt_fraction=phi_l1,
                params=SolidRheologyParams(enabled=True, interior_flux_fraction=f0_cfg),
                viscosity_solid=1.0e21,
                viscosity_mixture=visc_mix_l1,
                phi_rheo=0.4,
                phi_width=0.15,
                xp=np,
            )
            rel_t = abs(t_i_indep - float(st['T_i'])) / float(st['T_i'])
            rel_eta = abs(eta_i_indep - float(st['eta_i'])) / float(st['eta_i'])
            max_rel_t = max(max_rel_t, rel_t)
            max_rel_eta = max(max_rel_eta, rel_eta)
            print(
                f'({f0_cfg:.2f}, {f_test:.5e}): rel diff T_i = {rel_t:.2e}, rel diff eta_i = {rel_eta:.2e}'
            )

        print('\nMaximum relative difference to kernel across 7 points:')
        print(f'  max_rel_diff(T_i)   = {max_rel_t:.2e}')
        print(f'  max_rel_diff(eta_i) = {max_rel_eta:.2e}')


if __name__ == '__main__':
    main()
