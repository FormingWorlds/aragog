"""Compute freezing front trapping diagnostics for Step 5 (PROTEUS issue 519).

Computes eta_diff_b, tau_s = eta_diff_b / (delta_rho * g * L_front),
tau_s ratio, and w_rel at three points along the freezing front
with solid rheology enabled vs disabled.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import Parameters
from aragog.rheology import SolidRheologyParams


def compute_front_trapping_table() -> None:
    """Compute and display the freezing front trapping table."""
    fwl_data = os.environ.get('FWL_DATA')
    eos_dir_env = os.environ.get('ARAGOG_TEST_EOS_DIR')
    candidates = [
        eos_dir_env,
        f'{fwl_data}/aragog/spider_eos' if fwl_data else None,
        '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
    ]
    eos_path = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if eos_path is None:
        raise RuntimeError('SPIDER P-S tables not found.')

    eos = EntropyEOS(eos_path)
    cfg_path = Path(__file__).resolve().parent / 'configs' / 'ssc_earth_4p5gyr.toml'
    params = Parameters.from_file(cfg_path)

    p_test = np.array([120.0e9, 60.0e9, 15.0e9])  # Pa
    phi_rheo = 0.40
    g = 9.81
    l_front = 100.0e3  # 100 km front thickness

    print(
        f'{"P [GPa]":>8} | {"T [K]":>7} | {"eta_off [Pa s]":>14} | {"eta_on [Pa s]":>14} | '
        f'{"tau_s_off [s]":>13} | {"tau_s_on [s]":>13} | {"tau_ratio":>11} | '
        f'{"w_rel_off [m/s]":>15} | {"w_rel_on [m/s]":>15}'
    )
    print('-' * 125)

    for p in p_test:
        p_arr = np.array([p])
        s_sol = float(eos.solidus_entropy(p_arr)[0])
        s_liq = float(eos.liquidus_entropy(p_arr)[0])
        s_front = s_sol + phi_rheo * (s_liq - s_sol)
        s_arr = np.array([s_front])
        t_front = float(eos.temperature(p_arr, s_arr)[0])
        rho_s = float(eos._lookup_at_phase_boundary('density', p_arr, 'solid')[0])
        rho_l = float(eos._lookup_at_phase_boundary('density', p_arr, 'melt')[0])
        drho = abs(rho_s - rho_l)

        # 1. Switch OFF
        rheo_off = SolidRheologyParams(enabled=False)
        ev_off = EntropyPhaseEvaluator(
            entropy_eos=eos,
            gravitational_acceleration=g,
            viscosity_solid=params.phase_solid.viscosity,
            viscosity_liquid=params.phase_liquid.viscosity,
            grain_size=params.phase_mixed.grain_size,
            separation_viscosity='mixture',
            rheology=rheo_off,
        )
        ev_off.pressure = p_arr
        ev_off.entropy = s_arr
        ev_off.update()
        eta_off = (
            float(ev_off.eta_diff[0])
            if ev_off.eta_diff is not None
            else params.phase_solid.viscosity
        )
        w_rel_off = float(ev_off.relative_velocity()[0])
        tau_s_off = eta_off / (drho * g * l_front)

        # 2. Switch ON
        rheo_on = params.phase_solid.rheology
        ev_on = EntropyPhaseEvaluator(
            entropy_eos=eos,
            gravitational_acceleration=g,
            viscosity_solid=params.phase_solid.viscosity,
            viscosity_liquid=params.phase_liquid.viscosity,
            grain_size=params.phase_mixed.grain_size,
            separation_viscosity='mixture',
            rheology=rheo_on,
        )
        ev_on.pressure = p_arr
        ev_on.entropy = s_arr
        ev_on.update()
        eta_on = float(ev_on.eta_diff[0])
        w_rel_on = float(ev_on.relative_velocity()[0])
        tau_s_on = eta_on / (drho * g * l_front)

        ratio = tau_s_on / tau_s_off
        print(
            f'{p / 1e9:8.1f} | {t_front:7.1f} | {eta_off:14.3e} | {eta_on:14.3e} | '
            f'{tau_s_off:13.3e} | {tau_s_on:13.3e} | {ratio:11.3e} | '
            f'{w_rel_off:15.3e} | {w_rel_on:15.3e}'
        )


if __name__ == '__main__':
    compute_front_trapping_table()
