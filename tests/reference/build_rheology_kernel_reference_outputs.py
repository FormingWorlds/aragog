"""Build and verify ``tests/reference/rheology_kernel_reference_outputs.npz``.

Evaluates the reference input grids for the 8 public rheology kernels
(``compute_arrhenius_enthalpy``, ``compute_arrhenius_viscosity``,
``compute_yield_stress``, ``compute_strain_rate_local``,
``compute_mlt_top_slope``, ``viscous_mixing_length_factor``,
``compute_stagnant_lid_state``, and ``compute_effective_viscosity``) on the
``main`` baseline and stores their reference outputs in
``rheology_kernel_reference_outputs.npz``.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
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
from aragog.rheology_lid import compute_effective_viscosity, compute_stagnant_lid_state

MAIN_BASE_COMMIT = '2457a8c95099b04b766a220772d2e55bb3d02608'
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_PATH = Path(__file__).resolve().parent / 'rheology_kernel_reference_outputs.npz'


def _has_git_commit(commit: str) -> bool:
    """Return True if ``commit`` is present in the local git object database."""
    res = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'cat-file', '-e', f'{commit}^{{commit}}'],
        capture_output=True,
        check=False,
    )
    return res.returncode == 0


def _load_main_modules(commit: str = MAIN_BASE_COMMIT):
    """Load ``rheology.py`` and ``rheology_lid.py`` from ``commit`` in a temporary tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pkg_dir = tmp_path / 'aragog_main_ref'
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / '__init__.py').write_text('', encoding='utf-8')
        for rel_name in ('rheology.py', 'rheology_lid.py'):
            src = subprocess.check_output(
                ['git', '-C', str(REPO_ROOT), 'show', f'{commit}:src/aragog/{rel_name}'],
                text=True,
            )
            src = src.replace(
                'from aragog.rheology import', 'from aragog_main_ref.rheology import'
            )
            (pkg_dir / rel_name).write_text(src, encoding='utf-8')

        sys.path.insert(0, str(tmp_path))
        try:
            spec_r = importlib.util.spec_from_file_location(
                'aragog_main_ref.rheology', pkg_dir / 'rheology.py'
            )
            assert spec_r is not None and spec_r.loader is not None
            mod_r = importlib.util.module_from_spec(spec_r)
            sys.modules['aragog_main_ref.rheology'] = mod_r
            spec_r.loader.exec_module(mod_r)

            spec_l = importlib.util.spec_from_file_location(
                'aragog_main_ref.rheology_lid', pkg_dir / 'rheology_lid.py'
            )
            assert spec_l is not None and spec_l.loader is not None
            mod_l = importlib.util.module_from_spec(spec_l)
            sys.modules['aragog_main_ref.rheology_lid'] = mod_l
            spec_l.loader.exec_module(mod_l)
            return mod_r, mod_l
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop('aragog_main_ref.rheology_lid', None)
            sys.modules.pop('aragog_main_ref.rheology', None)


def _evaluate_grid_with_modules(mod_r, mod_l) -> dict[str, np.ndarray]:
    """Evaluate all 8 public rheology kernels on the reference grids using ``(mod_r, mod_l)``."""
    out: dict[str, np.ndarray] = {'main_commit': np.asarray(MAIN_BASE_COMMIT)}

    # 1. compute_arrhenius_enthalpy
    p_grid = np.linspace(0.0, 135e9, 25)
    out['enthalpy_arr'] = np.asarray(mod_r.compute_arrhenius_enthalpy(p_grid, xp=np))
    out['enthalpy_scalar'] = np.asarray(mod_r.compute_arrhenius_enthalpy(10.0e9, xp=np))

    # 2. compute_arrhenius_viscosity
    t_1d = np.linspace(1200.0, 2500.0, 10)
    p_1d = np.linspace(0.0, 100e9, 10)
    t_grid, p_grid_2d = np.meshgrid(t_1d, p_1d, indexing='ij')
    out['viscosity_arr'] = np.asarray(
        mod_r.compute_arrhenius_viscosity(t_grid, p_grid_2d, xp=np)
    )
    out['viscosity_scalar'] = np.asarray(
        mod_r.compute_arrhenius_viscosity(1600.0, 10.0e9, xp=np)
    )

    # 3. compute_yield_stress
    out['yield_stress_arr'] = np.asarray(mod_r.compute_yield_stress(p_grid, xp=np))
    out['yield_stress_scalar'] = np.asarray(mod_r.compute_yield_stress(50.0e9, xp=np))

    # 4. compute_strain_rate_local
    v_arr = np.linspace(-1e-8, 1e-8, 20)
    ml_arr = np.linspace(1e3, 1e6, 20)
    out['strain_rate_arr'] = np.asarray(mod_r.compute_strain_rate_local(v_arr, ml_arr, xp=np))
    out['strain_rate_scalar'] = np.asarray(mod_r.compute_strain_rate_local(1e-9, 1e5, xp=np))

    # 5. compute_mlt_top_slope
    th_arr = np.linspace(-5.0, 50.0, 25)
    out['top_slope_arr'] = np.asarray(mod_r.compute_mlt_top_slope(th_arr, xp=np))
    out['top_slope_scalar'] = np.asarray(mod_r.compute_mlt_top_slope(15.0, xp=np))

    # 6. viscous_mixing_length_factor
    radii = np.linspace(3.48e6, 6.371e6, 30)
    r_in = float(radii[0])
    r_out = float(radii[-1])
    ml_col = np.minimum(radii - r_in, r_out - radii)
    out['vml_none'] = np.asarray(
        mod_r.viscous_mixing_length_factor(radii, r_in, r_out, ml_col, xp=np)
    )
    out['vml_theta'] = np.asarray(
        mod_r.viscous_mixing_length_factor(radii, r_in, r_out, ml_col, theta=15.0, xp=np)
    )

    # 7. compute_stagnant_lid_state
    temp = np.linspace(3000.0, 1500.0, 30)
    press = np.linspace(135e9, 0.0, 30)
    conv_flux = np.full(30, 100.0)
    conv_flux[-5:] = 0.0
    tot_flux = np.full(30, 100.0)
    params = mod_r.SolidRheologyParams(enabled=True)
    lid_st = mod_l.compute_stagnant_lid_state(
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

    # 8. compute_effective_viscosity
    tau_d_arr = np.linspace(1e4, 1e8, 20)
    eta_diff = np.full(20, 1.0e22)
    w_lid = np.linspace(0.0, 1.0, 20)
    out['eff_lid'] = np.asarray(
        mod_l.compute_effective_viscosity(
            eta_diff=eta_diff,
            eta_i=1.0e21,
            tau_d=tau_d_arr,
            tau_y_lid=50e6,
            stress_closure_mode='lid',
            w_lid=w_lid,
            xp=np,
        )
    )

    sr_grid = np.logspace(-18, -12, 20)
    out['eff_local'] = np.asarray(
        mod_l.compute_effective_viscosity(
            eta_diff=eta_diff,
            tau_y=50e6,
            strain_rate=sr_grid,
            stress_closure_mode='local',
            xp=np,
        )
    )

    return out


def compute_main_reference_grid_outputs(
    commit: str = MAIN_BASE_COMMIT,
) -> dict[str, np.ndarray]:
    """Compute the reference output dictionary from ``commit`` on the 8 kernel grids."""
    mod_r, mod_l = _load_main_modules(commit)
    return _evaluate_grid_with_modules(mod_r, mod_l)


def verify_fixture(path: Path = FIXTURE_PATH) -> None:
    """Verify ``path`` pins ``MAIN_BASE_COMMIT`` and matches current kernels outside smooth bands."""
    if not path.exists():
        raise FileNotFoundError(f'Missing reference fixture: {path}')
    saved = dict(np.load(path))
    if 'main_commit' not in saved or str(saved['main_commit']) != MAIN_BASE_COMMIT:
        raise AssertionError(
            f'Expected main_commit={MAIN_BASE_COMMIT!r} in {path.name}, got {saved.get("main_commit")!r}'
        )

    if not _has_git_commit(MAIN_BASE_COMMIT):
        raise RuntimeError(
            f'Main baseline commit {MAIN_BASE_COMMIT} is not present in local git object database; '
            f'cannot verify against main.'
        )

    expected_main = compute_main_reference_grid_outputs(MAIN_BASE_COMMIT)
    if set(saved.keys()) != set(expected_main.keys()):
        raise AssertionError(
            f'Key mismatch in {path.name}: {sorted(saved.keys())} vs {sorted(expected_main.keys())}'
        )
    for k, exp_arr in expected_main.items():
        got_arr = saved[k]
        if k == 'main_commit':
            assert str(got_arr) == str(exp_arr)
        elif exp_arr.dtype == bool:
            if bool(got_arr) != bool(exp_arr):
                raise AssertionError(f'Boolean mismatch for {k}: {got_arr} != {exp_arr}')
        else:
            np.testing.assert_allclose(got_arr, exp_arr, rtol=1.0e-12, atol=0.0, err_msg=k)

    # Also verify current branch kernels against saved main outputs outside transition bands
    p_grid = np.linspace(0.0, 135e9, 25)
    np.testing.assert_allclose(
        compute_arrhenius_enthalpy(p_grid, xp=np), saved['enthalpy_arr'], rtol=1.0e-12, atol=0.0
    )
    t_1d = np.linspace(1200.0, 2500.0, 10)
    p_1d = np.linspace(0.0, 100e9, 10)
    t_grid, p_grid_2d = np.meshgrid(t_1d, p_1d, indexing='ij')
    visc_out = compute_arrhenius_viscosity(t_grid, p_grid_2d, xp=np)
    below_cap = saved['viscosity_arr'] < 1.0e38
    np.testing.assert_allclose(
        visc_out[below_cap], saved['viscosity_arr'][below_cap], rtol=1.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        compute_yield_stress(p_grid, xp=np), saved['yield_stress_arr'], rtol=1.0e-12, atol=0.0
    )
    v_arr = np.linspace(-1e-8, 1e-8, 20)
    ml_arr = np.linspace(1e3, 1e6, 20)
    np.testing.assert_allclose(
        compute_strain_rate_local(v_arr, ml_arr, xp=np),
        saved['strain_rate_arr'],
        rtol=1.0e-12,
        atol=0.0,
    )
    th_arr = np.linspace(-5.0, 50.0, 25)
    np.testing.assert_allclose(
        compute_mlt_top_slope(th_arr, xp=np), saved['top_slope_arr'], rtol=1.0e-12, atol=0.0
    )
    radii = np.linspace(3.48e6, 6.371e6, 30)
    r_in, r_out = float(radii[0]), float(radii[-1])
    ml_col = np.minimum(radii - r_in, r_out - radii)
    np.testing.assert_allclose(
        viscous_mixing_length_factor(radii, r_in, r_out, ml_col, theta=15.0, xp=np),
        saved['vml_theta'],
        rtol=1.0e-12,
        atol=0.0,
    )
    lid_st = compute_stagnant_lid_state(
        radii=radii,
        temperature=np.linspace(3000.0, 1500.0, 30),
        pressure=np.linspace(135e9, 0.0, 30),
        convective_flux=np.concatenate([np.full(25, 100.0), np.zeros(5)]),
        total_flux=np.full(30, 100.0),
        solidus_temperature=None,
        params=SolidRheologyParams(enabled=True),
        melt_fraction=np.zeros(30),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(30, 1e-9),
        xp=np,
    )
    exp_d_lid_raw = (radii[-1] - float(saved['lid_r_lid_base'])) * float(saved['lid_w_active'])
    w_lid_pres_main = float(saved['lid_d_lid']) / exp_d_lid_raw
    tau_d_lid_raw_main = float(saved['lid_tau_d_lid']) / w_lid_pres_main
    exp_wy_from_main = (
        w_lid_pres_main
        * 0.5
        * (1.0 + np.tanh((tau_d_lid_raw_main / float(saved['lid_tau_y_lid']) - 1.0) / 0.1))
    )
    exp_regime_from_main = float(saved['lid_w_active']) * (1.0 + exp_wy_from_main)

    for k, v in lid_st.items():
        arr = v if isinstance(v, np.ndarray) else np.asarray(v)
        if arr.dtype == bool:
            assert bool(arr) == bool(saved[f'lid_{k}'])
        elif k == 'd_lid_raw':
            assert 'lid_d_lid_raw' not in saved
            np.testing.assert_allclose(arr, exp_d_lid_raw, rtol=1.0e-12, atol=0.0)
        elif k == 'w_y':
            assert float(arr) > float(saved['lid_w_y']) * 3.8
            np.testing.assert_allclose(arr, exp_wy_from_main, rtol=1.0e-12, atol=0.0)
        elif k == 'lid_regime':
            assert float(arr) != float(saved['lid_lid_regime'])
            np.testing.assert_allclose(arr, exp_regime_from_main, rtol=1.0e-12, atol=0.0)
        else:
            np.testing.assert_allclose(arr, saved[f'lid_{k}'], rtol=1.0e-12, atol=0.0)

    tau_d_arr = np.linspace(1e4, 1e8, 20)
    eta_diff = np.full(20, 1.0e22)
    w_lid = np.linspace(0.0, 1.0, 20)
    eff_lid_out = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=1.0e21,
        tau_d=tau_d_arr,
        tau_y_lid=50e6,
        stress_closure_mode='lid',
        w_lid=w_lid,
        xp=np,
    )
    w_b_grid = 0.5 * (1.0 + np.tanh((tau_d_arr / 50e6 - 1.0) / 0.025))
    outside_b1 = (w_b_grid <= 1.0e-15) | (w_b_grid >= 1.0 - 1.0e-15)
    np.testing.assert_allclose(
        eff_lid_out[outside_b1], saved['eff_lid'][outside_b1], rtol=1.0e-12, atol=0.0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify the existing NPZ matches the main baseline without writing.',
    )
    parser.add_argument(
        '--allow-regenerate',
        action='store_true',
        help='Allow regenerating and overwriting the reference NPZ from MAIN_BASE_COMMIT.',
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
        saved = dict(np.load(args.output))
        print(f'OK: {args.output} matches main reference grid outputs ({len(saved)} keys).')
        return 0
    if args.output.exists() and not args.allow_regenerate:
        raise RuntimeError(
            f'Refusing to overwrite {args.output} without --allow-regenerate '
            f'(pinned to main commit {MAIN_BASE_COMMIT}).'
        )
    outputs = compute_main_reference_grid_outputs(MAIN_BASE_COMMIT)
    np.savez(args.output, **outputs)
    verify_fixture(args.output)
    print(f'Wrote and verified {args.output} ({len(outputs)} keys).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
