"""Zalmoxis result['core'] arrays of a dry pure-Fe two-layer core on the default side-table grid.

Run with the Zalmoxis source (branch tl/core-profile-two-phase) first on the path: python tools/verification/make_core_arrays_fe.py <mass in M_E> <out dir> [T_surf in K, default that of default.toml].
The Zalmoxis tree must hold its data/ folder (EOS tables); without it the mantle lookups give no finite solve.
Setup: input/default.toml (adiabatic, surface anchor, Picard, tolerance_outer 1e-3), core
'PALEOS-API-2phase:iron-liquid' (pure Fe, no CoreAlloy) with inner_core 'PALEOS-API-2phase:iron-solid',
core_liquidus 'Anzellini13-iron', side tables on the default grid (make_default_grid_iron_2phase: 1e5-1e14 Pa, 300-1e5 K, 150 per decade).
Writes core_arrays_<mass>me.npz (the arrays of result['core']) and core_arrays_<mass>me.json (scalars, config, source).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
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

mass, out = float(sys.argv[1]), Path(sys.argv[2])
T_surf = float(sys.argv[3]) if len(sys.argv) > 3 else None

LIQ, SOL = 'PALEOS-API-2phase:iron-liquid', 'PALEOS-API-2phase:iron-solid'
md = load_material_dictionaries()
grid = paleos_api.make_default_grid_iron_2phase()
for n in (LIQ, SOL):
    md[n] = dict(copy.deepcopy(md[n]), grid_spec=grid)
paleos_api_cache._check_paleos_once = lambda sha: 'tested'
paleos_api_cache.resolve_paleos_api_2phase_iron(grid, n_workers=1)
for n in (LIQ, SOL):
    paleos_api_cache.resolve_registry_entry(md[n])
rd = get_zalmoxis_root()
cfg = copy.deepcopy(load_zalmoxis_config(os.path.join(rd, 'input', 'default.toml')))
cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core=LIQ, inner_core=SOL)
cfg.update(core_liquidus='Anzellini13-iron', tolerance_outer=1e-3, planet_mass=mass * 5.972e24)
if T_surf is not None:
    cfg['surface_temperature'] = T_surf
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
tag = f'{mass:g}me_{cfg["surface_temperature"]:g}K'
np.savez_compressed(out / f'core_arrays_{tag}.npz', **arrays)
src = Path(zalmoxis.__file__).resolve().parents[2]
sha = subprocess.run(
    ['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True
).stdout.strip()
dirty = subprocess.run(
    ['git', '-C', str(src), 'status', '--porcelain', 'src'], capture_output=True, text=True
).stdout.strip()
info = {
    'mass_M_E': mass,
    'zalmoxis_src': str(src),
    'zalmoxis_sha': sha,
    'src_clean': not dirty,
    'paleos_tested_sha': paleos_api.PALEOS_TESTED_SHA,
    'build_key_iron': paleos_api.build_key('iron'),
    'grid': f'default 2-phase iron grid {grid.hash_short()}: 1e5-1e14 Pa, 300-1e5 K, 150 per decade',
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
}
(out / f'core_arrays_{tag}.json').write_text(json.dumps(info, indent=1, default=str))
print(
    json.dumps(
        {k: info[k] for k in ('mass_M_E', 'converged', 'accepted_on', 'seconds', 'nodes')}
    ),
    {k: scalars.get(k) for k in ('M_ic', 'R_icb', 'P_icb')},
)
