"""Generate the Leeds thermal_history reference tables for the core verification page.

Runs the ``leeds`` core model of thermal_history (Greenwood et al. 2021;
https://github.com/sam-greenwood/thermal_history, MIT licence) on the inputs of
``INPUTS`` below, a core-only Earth-like thermal history driven by a prescribed
CMB heat flow, and writes ``tools/verification/data/thermal_history_evolution.csv``
with the inputs, the thermal_history commit and the package versions in its header.
``run_core_verification.py`` reads that table and integrates aragog on the same
inputs, so the page needs thermal_history only to regenerate the table.

thermal_history needs numba and ``scipy.integrate.trapz`` (SciPy < 1.14); the
table was made in a separate environment::

    python -m venv thvenv
    thvenv/bin/pip install 'numpy==2.0.2' 'scipy==1.13.1' 'numba==0.60.0' \\
        'jax==0.4.30' 'jaxlib==0.4.30' <aragog checkout>
    git clone https://github.com/sam-greenwood/thermal_history && git -C thermal_history checkout ea9aa99
    PYTHONPATH=<aragog>/src:thermal_history thvenv/bin/python tools/verification/thermal_history_reference.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from math import factorial
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'data' / 'thermal_history_evolution.csv'
LAYER_OUT = OUT.with_name('thermal_history_stable_layer.csv')
LAYER = dict(q_cmb=[8e12, 12e12], t_end_myr=1000.0)

# Nimmo (2015, ch. 8.02) Table 2 core with a prescribed CMB heat flow; aragog uses the
# same numbers with pressure_mode='quadrature', since thermal_history integrates the
# hydrostatic pressure of its density polynomial.
INPUTS = dict(
    rho_cen=12500.0,
    length_scale=7272e3,
    r_cmb=3480e3,
    p_cmb=139e9,
    alpha=1.25e-5,
    c_p=840.0,
    k_core=130.0,
    latent_heat=750e3,
    t_m0=2677.0,
    t_m1=2.95e-12,
    t_m2=8.37e-25,
    alpha_c=1.0,
    c_light=560.0 / 12150.0,
    q_cmb=10e12,
    t_cmb_start=4400.0,
    dt_myr=1.0,
    t_end_myr=1500.0,
    n_profiles=4000,
    taylor_terms=8,
    G=6.67430e-11,
)


def _gaussian_poly(scale: float, amplitude: float, terms: int) -> list[float]:
    """Radial polynomial (increasing powers) of ``amplitude * exp(-r^2 / scale^2)``."""
    coeffs = np.zeros(2 * terms - 1)
    for k in range(terms):
        coeffs[2 * k] = amplitude * (-1) ** k / (factorial(k) * scale ** (2 * k))
    return coeffs.tolist()


def _parameter_file(inp: dict, d_scale: float, stable_layer: bool = False) -> str:
    rho = _gaussian_poly(inp['length_scale'], inp['rho_cen'], inp['taylor_terms'])
    adiabat = _gaussian_poly(d_scale, 1.0, inp['taylor_terms'])
    tm = [inp['t_m0'], inp['t_m0'] * inp['t_m1'], inp['t_m0'] * inp['t_m2']]
    lines = {
        'core': True,
        'stable_layer': stable_layer,
        'mantle': False,
        'T_cmb': inp['t_cmb_start'],
        'conc_l': [inp['c_light']],
        'core_solid_density_params': rho,
        'core_liquid_density_params': rho,
        'ri': 0.0,
        'r_cmb': inp['r_cmb'],
        'core_alpha_T_params': [inp['alpha']],
        'core_cp_params': [inp['c_p']],
        'core_conductivity_params': [inp['k_core']],
        'core_melting_params': ['AL', *tm],
        'entropy_melting_params': [0.0],
        'mm': [56, 32],
        'alpha_c': [inp['alpha_c']],
        'diffusivity_c': [1e-8],
        'use_partition_coeff': True,
        'core_h0': 0.0,
        'half_life': 0.0,
        'partition_coeff': [0.0],
        'lambda_sol': [0.0],
        'lambda_liq': [0.0],
        'dmu': [0.0],
        'n_profiles': inp['n_profiles'],
        'P_cmb': inp['p_cmb'],
        'precip_temp': 0.0,
        'Cm_mgo': 0.0,
        'alpha_c_mgo': 0.0,
        'core_adiabat_params': adiabat,
        'core_latent_heat': inp['latent_heat'],
        'include_baro_diffusion': False,
    }
    return '\n'.join(f'{k} = {v!r}' for k, v in lines.items()) + '\n'


def _versions(th_root: Path) -> dict:
    import numba
    import scipy

    commit = subprocess.run(
        ['git', '-C', str(th_root), 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return dict(
        thermal_history=commit,
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
    )


def _model(inp: dict, stable_layer: bool):
    """A thermal_history model on ``inp``, with or without the leeds_thermal stable layer."""
    import thermal_history.core_models.leeds.routines.profiles as th_profiles
    from thermal_history.model import Parameters, setup_model

    th_profiles.G = inp['G']  # the CODATA value aragog uses
    d_scale = np.sqrt(3 * inp['c_p'] / (2 * np.pi * inp['alpha'] * inp['rho_cen'] * inp['G']))
    with tempfile.TemporaryDirectory() as tmp:
        prm_file = Path(tmp) / 'core_params.py'
        prm_file.write_text(_parameter_file(inp, d_scale, stable_layer))
        return setup_model(
            Parameters(str(prm_file)),
            core_method='leeds',
            stable_layer_method='leeds_thermal' if stable_layer else None,
            verbose=False,
            log_file=str(Path(tmp) / 'thermal_history.log'),
        )


def _write(path: Path, header: dict, rows: list) -> None:
    import thermal_history

    header = {
        'source': 'thermal_history (https://github.com/sam-greenwood/thermal_history), MIT licence',
        **header,
        'versions': _versions(Path(thermal_history.__file__).resolve().parents[1]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        fh.write('# ' + json.dumps(header) + '\n')
        np.savetxt(fh, np.array(rows), delimiter=',', fmt='%.10e')
    print(f'{len(rows)} rows to {path}', file=sys.stderr)


def stable_layer_runs() -> None:
    """The leeds_thermal layer under fixed subadiabatic CMB heat flows."""
    inp = dict(INPUTS, t_end_myr=LAYER['t_end_myr'])
    rows = []
    for q in LAYER['q_cmb']:
        model = _model(dict(inp, q_cmb=q), stable_layer=True)
        ys = model.parameters.ys
        for _ in range(int(inp['t_end_myr'] / inp['dt_myr'])):
            model.mantle.Q_cmb = q
            core = model.core
            rows.append(
                [
                    q,
                    model.time / (1e6 * ys),
                    float(core.profiles['T'][-1]),
                    float(core.Tcen),
                    float(core.ri),
                    float(core.rs),
                ]
            )
            model.evolve(inp['dt_myr'] * 1e6 * ys, verbose=False)
    _write(
        LAYER_OUT,
        {
            'inputs': inp,
            'q_cmb': LAYER['q_cmb'],
            'columns': ['q_cmb', 'time_myr', 'T_cmb', 'T_cen', 'r_icb', 'r_s'],
        },
        rows,
    )


def main() -> None:
    inp = INPUTS
    model = _model(inp, stable_layer=False)
    ys = model.parameters.ys
    rows = []
    for _ in range(int(inp['t_end_myr'] / inp['dt_myr'])):
        model.mantle.Q_cmb = inp['q_cmb']
        core = model.core
        state = [
            model.time / (1e6 * ys),
            float(core.profiles['T'][-1]),
            float(core.Tcen),
            float(core.ri),
            float(core.conc_l[0]),
        ]
        # evolve() evaluates the budget on the current state, then advances it.
        model.evolve(inp['dt_myr'] * 1e6 * ys, verbose=False)
        d = core.dT_dt  # dTcen/dt: each Q_tilde = Q / dT_dt
        rows.append(
            state
            + [
                core.Qs / d,
                core.Ql / d,
                core.Qg / d,
                core.Es / d,
                core.El / d,
                core.Eg / d,
                core.Ek,
            ]
        )
    columns = [
        'time_myr',
        'T_cmb',
        'T_cen',
        'r_icb',
        'conc_l',
        'Qs_per_dTcen',
        'Ql_per_dTcen',
        'Qg_per_dTcen',
        'Es_per_dTcen',
        'El_per_dTcen',
        'Eg_per_dTcen',
        'Ek',
    ]
    _write(OUT, {'inputs': inp, 'columns': columns}, rows)
    stable_layer_runs()


if __name__ == '__main__':
    main()
