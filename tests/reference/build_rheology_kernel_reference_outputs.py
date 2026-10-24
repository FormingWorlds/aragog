"""Build and verify ``tests/reference/rheology_kernel_reference_outputs.npz``.

Evaluates the reference input grids for the 7 public rheology kernels
(``compute_arrhenius_enthalpy``, ``compute_arrhenius_viscosity``,
``compute_yield_stress``, ``compute_strain_rate_local``,
``compute_mlt_top_slope``, ``viscous_mixing_length_factor``,
``compute_stagnant_lid_state``, and ``compute_effective_viscosity``) and stores
their reference outputs in ``rheology_kernel_reference_outputs.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_enthalpy,
    compute_arrhenius_viscosity,
    compute_mlt_top_slope,
    compute_strain_rate_local,
    compute_yield_stress,
    viscous_mixing_length_factor,
)
from aragog.rheology_lid import compute_stagnant_lid_state

FIXTURE_PATH = Path(__file__).resolve().parent / 'rheology_kernel_reference_outputs.npz'


def compute_reference_grid_outputs() -> dict[str, np.ndarray]:
    """Compute the reference output dictionary across the 8 kernel evaluation grids."""
    out: dict[str, np.ndarray] = {}

    # 1. compute_arrhenius_enthalpy
    p_grid = np.linspace(0.0, 135e9, 25)
    out['enthalpy_arr'] = np.asarray(compute_arrhenius_enthalpy(p_grid, xp=np))
    out['enthalpy_scalar'] = np.asarray(compute_arrhenius_enthalpy(10.0e9, xp=np))

    # 2. compute_arrhenius_viscosity (unsmoothed hard cap reference for below_cap comparison)
    t_1d = np.linspace(1200.0, 2500.0, 10)
    p_1d = np.linspace(0.0, 100e9, 10)
    t_grid, p_grid_2d = np.meshgrid(t_1d, p_1d, indexing='ij')
    out['viscosity_arr'] = np.minimum(
        compute_arrhenius_viscosity(t_grid, p_grid_2d, viscosity_max_log10=200.0, xp=np),
        1.0e40,
    )
    out['viscosity_scalar'] = np.asarray(compute_arrhenius_viscosity(1600.0, 10.0e9, xp=np))

    # 3. compute_yield_stress
    out['yield_stress_arr'] = np.asarray(compute_yield_stress(p_grid, xp=np))
    out['yield_stress_scalar'] = np.asarray(compute_yield_stress(50.0e9, xp=np))

    # 4. compute_strain_rate_local
    v_arr = np.linspace(-1e-8, 1e-8, 20)
    ml_arr = np.linspace(1e3, 1e6, 20)
    out['strain_rate_arr'] = np.asarray(compute_strain_rate_local(v_arr, ml_arr, xp=np))
    out['strain_rate_scalar'] = np.asarray(compute_strain_rate_local(1e-9, 1e5, xp=np))

    # 5. compute_mlt_top_slope
    th_arr = np.linspace(-5.0, 50.0, 25)
    out['top_slope_arr'] = np.asarray(compute_mlt_top_slope(th_arr, xp=np))
    out['top_slope_scalar'] = np.asarray(compute_mlt_top_slope(15.0, xp=np))

    # 6. viscous_mixing_length_factor
    radii = np.linspace(3.48e6, 6.371e6, 30)
    r_in = float(radii[0])
    r_out = float(radii[-1])
    ml_col = np.minimum(radii - r_in, r_out - radii)
    out['vml_none'] = np.asarray(
        viscous_mixing_length_factor(radii, r_in, r_out, ml_col, xp=np)
    )
    out['vml_theta'] = np.asarray(
        viscous_mixing_length_factor(radii, r_in, r_out, ml_col, theta=15.0, xp=np)
    )

    # 7. compute_stagnant_lid_state
    temp = np.linspace(3000.0, 1500.0, 30)
    press = np.linspace(135e9, 0.0, 30)
    conv_flux = np.full(30, 100.0)
    conv_flux[-5:] = 0.0
    tot_flux = np.full(30, 100.0)
    params = SolidRheologyParams(enabled=True)
    lid_st = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp,
        pressure=press,
        convective_flux=conv_flux,
        total_flux=tot_flux,
        solidus_temperature=None,
        params=params,
        melt_fraction=np.zeros(30),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(30, 1e-9),
        xp=np,
    )
    for k, v in lid_st.items():
        out[f'lid_{k}'] = v if isinstance(v, np.ndarray) else np.asarray(v)

    # 8. compute_effective_viscosity (asymptotic hard-min reference outside transition bands)
    tau_d_arr = np.linspace(1e4, 1e8, 20)
    eta_diff = np.full(20, 1.0e22)
    w_lid = np.linspace(0.0, 1.0, 20)
    eta_y_branch = np.where(
        tau_d_arr <= 50e6, eta_diff, np.minimum(eta_diff, 1.0e21 * 50e6 / tau_d_arr)
    )
    out['eff_lid'] = eta_diff * ((eta_y_branch / eta_diff) ** w_lid)

    sr_grid = np.logspace(-18, -12, 20)
    out['eff_local'] = np.minimum(eta_diff, 50e6 / (2.0 * sr_grid))

    return out


def verify_fixture(path: Path = FIXTURE_PATH) -> None:
    """Verify ``path`` matches ``compute_reference_grid_outputs()`` within rtol=1e-12."""
    if not path.exists():
        raise FileNotFoundError(f'Missing reference fixture: {path}')
    saved = dict(np.load(path))
    expected = compute_reference_grid_outputs()
    if set(saved.keys()) != set(expected.keys()):
        raise AssertionError(
            f'Key mismatch in {path.name}: {sorted(saved.keys())} vs {sorted(expected.keys())}'
        )
    for k, exp_arr in expected.items():
        got_arr = saved[k]
        if exp_arr.dtype == bool:
            if bool(got_arr) != bool(exp_arr):
                raise AssertionError(f'Boolean mismatch for {k}: {got_arr} != {exp_arr}')
        else:
            np.testing.assert_allclose(got_arr, exp_arr, rtol=1.0e-12, atol=0.0, err_msg=k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify the existing NPZ matches computed reference outputs without writing.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=FIXTURE_PATH,
        help='Path to the reference NPZ file.',
    )
    args = parser.parse_args(argv)
    if args.check:
        verify_fixture(args.output)
        print(f'OK: {args.output} matches reference grid outputs (45 keys).')
        return 0
    outputs = compute_reference_grid_outputs()
    np.savez(args.output, **outputs)
    verify_fixture(args.output)
    print(f'Wrote and verified {args.output} (45 keys).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
