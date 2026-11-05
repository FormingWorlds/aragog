"""Surface flux and lid-base isotherm depth at the stored states of a run of ``run_surface_lid_case``.

Each state of every part goes through one right-hand side on the same column, so the series has
the time resolution of the stored ``y`` and needs no new solve. ``z_lid`` is the deepest point of
the mesh column where the temperature rises through ``T_lid_iso``; it is NaN when the mesh column
holds no such rise. When the whole mesh column lies above that isotherm under a live lid, the
isotherm lies inside the lid.

Usage, from the root of a checkout, with its ``src`` on ``PYTHONPATH``::

    python tools/verification/run_surface_lid_series.py RUN_DIR OUT.npz --eos-dir DIR --nodes N
        [--no-lid] [--surface-cell M] --eta 1e18 --heating 1e-10 --s-init S
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ssc_helpers import entropy_eos_copy  # noqa: E402
from run_surface_lid_case import RHEOLOGY, STEM  # noqa: E402


def isotherm_depth(T_top_first: np.ndarray, z: np.ndarray, T_iso: float) -> float:
    """Depth [m] of the deepest rise through ``T_iso``, linear between nodes; NaN without one."""
    rise = np.flatnonzero((T_top_first[:-1] < T_iso) & (T_top_first[1:] >= T_iso))
    if rise.size == 0:
        return float('nan')
    i = int(rise[-1])
    return float(
        z[i]
        + (T_iso - T_top_first[i]) * (z[i + 1] - z[i]) / (T_top_first[i + 1] - T_top_first[i])
    )


def main(argv=None) -> None:
    from aragog.config import Config
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--eos-dir', type=Path, required=True)
    ap.add_argument('--nodes', type=int, required=True)
    ap.add_argument('--no-lid', action='store_true')
    ap.add_argument('--surface-cell', type=float)
    ap.add_argument('--eta', type=float, required=True)
    ap.add_argument('--heating', type=float, required=True)
    ap.add_argument('--s-init', type=float, required=True)
    a = ap.parse_args(argv)
    text = Path('src/aragog/cfg/abe_solid.toml').read_text()
    if not a.no_lid:
        text = text.replace(
            '[boundary_conditions]\n', '[boundary_conditions]\nsubgrid_lid = true\n'
        )
    text = text.replace('[phase_mixed]\n', '[phase_mixed]\nmatprop_smooth_width = 0.01\n')
    cfg = Path(tempfile.mkdtemp()) / 'column.toml'
    cfg.write_text(text)
    c = Config.from_file(str(cfg))
    c.solver.end_time, c.energy.kappah_floor, c.phase_solid.viscosity = 1.0, 10.0, a.eta
    c.energy.tidal, c.energy.tidal_array = True, np.array([a.heating])
    c.phase_solid.rheology = SolidRheologyParams(**RHEOLOGY)
    c.boundary_conditions.emissivity, c.mesh.number_of_nodes = 1.0, a.nodes
    if a.surface_cell is not None:
        c.mesh.surface_cell_thickness = a.surface_cell
    s = EntropySolver(c, entropy_eos=entropy_eos_copy(a.eos_dir))
    s.initialize()
    s.set_initial_entropy(a.s_init)
    mesh = s.evaluator.mesh
    z = (float(np.ravel(mesh.basic.radii)[-1]) - np.ravel(mesh.staggered.radii))[::-1]
    rows = json.loads((a.run / f'{STEM}.json').read_text())
    t_all, flux, depth, delta = [], [], [], []
    for k in range(len(rows)):
        part = np.load(a.run / f'{STEM}_part{k}.npz')
        for t, y in zip(part['t'], part['y'].T):
            if t_all and t <= t_all[-1]:
                continue
            s._dSdt_single(float(t), y)
            T = np.asarray(s.state.phase_staggered.temperature()).ravel()[::-1]
            state = s.state.lid_state or {}
            T_iso = float(np.asarray(state.get('T_lid_iso', np.nan)))
            t_all.append(float(t))
            flux.append(float(s.state.heat_flux[-1]))
            depth.append(isotherm_depth(T, z, T_iso))
            delta.append(0.0 if a.no_lid else float(y[-2]))
    handover = [r['handover_time'] for r in rows if r.get('handover_time') is not None]
    np.savez_compressed(
        a.out,
        t=np.array(t_all),
        F_top=np.array(flux),
        z_lid=np.array(depth),
        delta=np.array(delta),
        handover_yr=np.array(max(handover) if handover else np.nan),
        part_yr=np.array([r['t_target'] for r in rows]),
        part_F_top=np.array([r['F_top'] for r in rows]),
        f_int_max=np.array([r.get('f_int_max') for r in rows], dtype=float),
        w_active=np.array([r.get('w_active') for r in rows], dtype=float),
        sha=np.array(rows[-1]['sha']),
    )


if __name__ == '__main__':
    main()
