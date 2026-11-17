"""Zalmoxis result['core'] arrays of a two-layer iron core for the aragog core tests.

Run with the Zalmoxis source (branch tl/core-profile-two-phase) first on the path::

    python tools/verification/make_core_arrays.py <mass in M_E> <out dir> [--t-surf K]
        [--pure-fe] [--cache-root DIR] [--skip-paleos-check]

The Zalmoxis tree must hold its data/ folder (EOS tables); without it the mantle lookups give
no finite solve. Setup: input/default.toml (adiabatic, surface anchor, Picard, tolerance_outer
1e-3), inner_core 'PALEOS-API-2phase:iron-solid', core_liquidus 'Anzellini13-iron'. The outer
core is 'PALEOS-API-2phase:iron-liquid:0.8615+CoreAlloy:S:0.1385' with side tables on the
40-per-decade grid (1e9-3e12 Pa, 1e3-2e4 K); with ``--pure-fe`` it is pure liquid iron with
side tables on the default 2-phase iron grid (1e5-1e14 Pa, 300-1e5 K, 150 per decade).
``--cache-root`` names the side-table cache (default: that of Zalmoxis); ``--skip-paleos-check``
skips the PALEOS commit check, and the provenance then says so.

Writes zalmoxis_core_arrays[_fe]_<mass>me_<T_surf>K.npz (the arrays of result['core']) and a
.json of the same name (scalars, config, source, md5 of the .npz), and prints the file name
and md5.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import zalmoxis
from zalmoxis import get_zalmoxis_root
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    validate_config,
)
from zalmoxis.eos import paleos_api, paleos_api_cache
from zalmoxis.mixing import parse_all_layer_mixtures
from zalmoxis.solver import main

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('mass', type=float, help='planet mass [M_E]')
ap.add_argument('out', type=Path, help='output directory')
ap.add_argument('--t-surf', type=float, help='surface temperature [K] (default.toml if unset)')
ap.add_argument('--pure-fe', action='store_true', help='pure-Fe outer core, default grid')
ap.add_argument('--cache-root', type=Path, help='side-table cache directory')
ap.add_argument('--skip-paleos-check', action='store_true', help='skip the PALEOS commit check')
args = ap.parse_args()

LIQ, SOL = 'PALEOS-API-2phase:iron-liquid', 'PALEOS-API-2phase:iron-solid'
if args.pure_fe:
    grid, core_eos = paleos_api.make_default_grid_iron_2phase(), LIQ
    grid_text = (
        f'default 2-phase iron grid {grid.hash_short()}: 1e5-1e14 Pa, 300-1e5 K, 150 per decade'
    )
else:
    grid, core_eos = (
        paleos_api.make_grid_at_resolution(1e9, 3e12, 1000.0, 2.0e4, 40),
        f'{LIQ}:0.8615+CoreAlloy:S:0.1385',
    )
    grid_text = '40 per decade, 1e9-3e12 Pa, 1e3-2e4 K'
md = load_material_dictionaries()
for n in (LIQ, SOL):
    md[n] = dict(copy.deepcopy(md[n]), grid_spec=grid)
if args.cache_root is not None:
    paleos_api_cache._cache_root = lambda: args.cache_root
if args.skip_paleos_check:
    paleos_api_cache._check_paleos_once = lambda sha: 'skipped'
paleos_api_cache.resolve_paleos_api_2phase_iron(grid, n_workers=1)
for n in (LIQ, SOL):
    paleos_api_cache.resolve_registry_entry(md[n])
rd = get_zalmoxis_root()
cfg = copy.deepcopy(load_zalmoxis_config(os.path.join(rd, 'input', 'default.toml')))
cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core=core_eos, inner_core=SOL)
cfg.update(
    core_liquidus='Anzellini13-iron', tolerance_outer=1e-3, planet_mass=args.mass * 5.972e24
)
if args.t_surf is not None:
    cfg['surface_temperature'] = args.t_surf
validate_config(cfg)
melt = load_solidus_liquidus_functions(
    cfg['layer_eos_config'], cfg['rock_solidus'], cfg['rock_liquidus'], 1.0
)
t0 = time.time()
res = main(
    cfg,
    md,
    melt,
    os.path.join(rd, 'input'),
    layer_mixtures=parse_all_layer_mixtures(cfg['layer_eos_config']),
)
core = res['core']
arrays = {
    k: np.asarray(v)
    for k, v in core.items()
    if isinstance(v, (list, np.ndarray)) and k != 'w_layers'
}
scalars = {
    k: (
        float(v)
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
        else v
    )
    for k, v in core.items()
    if k not in arrays
}
name = f'zalmoxis_core_arrays{"_fe" if args.pure_fe else ""}_{args.mass:g}me_{cfg["surface_temperature"]:g}K'
npz = args.out / f'{name}.npz'
np.savez_compressed(npz, **arrays)
src = Path(zalmoxis.__file__).resolve().parents[2]


def git(*cmd):
    """Output of a git command in the Zalmoxis source tree."""
    return subprocess.run(
        ['git', '-C', str(src), *cmd], capture_output=True, text=True
    ).stdout.strip()


info = {
    'mass_M_E': args.mass,
    'zalmoxis_sha': git('rev-parse', 'HEAD'),
    'src_clean': not git('status', '--porcelain', 'src'),
    'paleos_tested_sha': paleos_api.PALEOS_TESTED_SHA,
    'paleos_installed_sha': paleos_api.paleos_installed_sha(),
    'paleos_check': 'skipped (--skip-paleos-check)' if args.skip_paleos_check else 'passed',
    'build_key_iron': paleos_api.build_key('iron'),
    'grid': grid_text,
    'config': {
        k: cfg[k]
        for k in (
            'planet_mass',
            'surface_temperature',
            'temperature_mode',
            'layer_eos_config',
            'core_liquidus',
            'core_mass_fraction',
            'tolerance_outer',
            'num_layers',
        )
    },
    'converged': bool(res['converged']),
    'accepted_on': res.get('accepted_on'),
    'padded': bool(res.get('padded')),
    'seconds': round(time.time() - t0, 1),
    'scalars': scalars,
    'array_keys': sorted(arrays),
    'nodes': int(len(arrays['r'])),
    'npz_md5': hashlib.md5(npz.read_bytes()).hexdigest(),
    'made_by': 'tools/verification/make_core_arrays.py'
    + (' --pure-fe' if args.pure_fe else '')
    + ", run on the Zalmoxis commit zalmoxis_sha; arrays of result['core']",
}
(args.out / f'{name}.json').write_text(json.dumps(info, default=str))
print(npz.name, info['npz_md5'], {k: scalars.get(k) for k in ('M_ic', 'R_icb', 'P_icb')})
