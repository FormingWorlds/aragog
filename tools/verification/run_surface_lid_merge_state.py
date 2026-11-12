"""State on both sides of the lid hand-over of a stored run of ``run_surface_lid_case``.

The run restarts from its last stored state before the hand-over and stops a little after it. The
result holds the surface flux, the top-cell and surface-node temperatures before and after the
merge, how long the flux stays negative, and the temperature of the top node against a run without
the lid at the last stored state, with the offset ``(delta - h/2) (T_b - T_s) / delta``.

Usage, from the root of a checkout, with its ``src`` on ``PYTHONPATH``::

    python tools/verification/run_surface_lid_merge_state.py RUN REF OUT.json --eos-dir DIR
        --nodes N --ref-cell M --eta 1e18 --heating 1e-10
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
from run_surface_lid_case import (  # noqa: E402
    STEM,
    _install_factory,
    check_stored_mesh,
    stored_column,
)


def _solver(a, *, lid, nodes, cell, end):
    """A solver of the column of ``run_surface_lid_case``, with its first start set."""
    from aragog.config import Config
    from aragog.solver.entropy_solver import EntropySolver

    text = Path('src/aragog/cfg/abe_solid.toml').read_text()
    text = text.replace('[phase_mixed]\n', '[phase_mixed]\nmatprop_smooth_width = 0.01\n')
    if lid:
        text = text.replace(
            '[boundary_conditions]\n', '[boundary_conditions]\nsubgrid_lid = true\n'
        )
    cfg = Path(tempfile.mkdtemp()) / 'column.toml'
    cfg.write_text(text)
    c = Config.from_file(str(cfg))
    c.solver.end_time, c.solver.max_steps = end, 800_000
    c.energy.kappah_floor, c.phase_solid.viscosity = 10.0, a.eta
    c.energy.tidal, c.energy.tidal_array = True, np.array([a.heating])
    stored_column(c, cell)
    c.boundary_conditions.emissivity, c.mesh.number_of_nodes = 1.0, nodes
    s = EntropySolver(c, entropy_eos=entropy_eos_copy(a.eos_dir))
    s.initialize()
    s.set_initial_entropy(2437.5)
    return s


def _state(s, y):
    """Surface flux and the temperatures of the top cell and the surface node at ``y``."""
    s._dSdt_single(0.0, np.array(y))
    return dict(
        flux=float(s.state.heat_flux[-1]),
        T_cell=float(np.asarray(s.state.phase_staggered.temperature()).ravel()[-1]),
        T_node=float(s.state.top_temperature.item()),
    )


def over_the_merge(a, t0: float, y0: np.ndarray, t_ho: float) -> tuple[dict, object, float]:
    """Run the lid column from the state ``y0`` at ``t0`` to 2e4 yr after the hand-over near
    ``t_ho``; return the merge values, the solver and the top-node offset of the lid base."""
    from aragog.solver.entropy_solver import EntropySolver

    s = _solver(a, lid=True, nodes=a.nodes, cell=None, end=t_ho + 2.0e4)
    s.set_initial_lid(float(y0[-2]), float(y0[-1]))
    s.set_initial_entropy(2437.5)
    _install_factory(s, a.eos_dir)
    scales = s._build_nondim_scales()
    off = 0.0 if scales.state_offset is None else np.asarray(scales.state_offset)
    y_nd, seg = (y0 - off) / np.asarray(scales.state_scale), s._solve_cvode_segments

    def restart(**kw):
        kw.setdefault('max_segments', 50_000)
        return seg(**{**kw, 'start_time': t0 / kw['t_ref'], 'y0': y_nd})

    s._solve_cvode_segments = restart
    merged, merge = {}, EntropySolver._lid_merge

    def recording(self, y):
        y1, resid = merge(self, y)
        merged.update(before=np.array(y), after=np.array(y1), resid=float(resid))
        return y1, resid

    EntropySolver._lid_merge = recording
    try:
        s.solve()
    finally:
        EntropySolver._lid_merge = merge
    sol = s._solution
    t, y = sol.get('energy_trace') or (sol.t, sol.y)
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    t_merge = float(s._lid_handover_time)
    negative = [k for k in np.flatnonzero(t > t_merge) if _state(s, y[:, k])['flux'] < 0.0]
    delta = float(merged['before'][-2])
    before = _state(s, merged['before'])
    lid = s._lid_last
    offset = (delta - 0.5 * s._lid_h_top) * (lid.T_b - lid.surf.T_s) / delta
    values = dict(
        t_merge=t_merge,
        delta=delta,
        h_top=float(s._lid_h_top),
        merge_residual=merged['resid'],
        before=before,
        after=_state(s, merged['after']),
        negative_yr=float(t[negative[-1]] - t_merge) if negative else 0.0,
    )
    return values, s, float(offset)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path)
    ap.add_argument('ref', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--eos-dir', type=Path, required=True)
    ap.add_argument('--nodes', type=int, required=True)
    ap.add_argument('--ref-cell', type=float, required=True)
    ap.add_argument('--eta', type=float, required=True)
    ap.add_argument('--heating', type=float, required=True)
    a = ap.parse_args(argv)
    rows = json.loads((a.run / f'{STEM}.json').read_text())
    t_ho = max(r['handover_time'] for r in rows if r.get('handover_time') is not None)
    part = next(k for k, r in enumerate(rows) if r['t_target'] > t_ho)
    z = np.load(a.run / f'{STEM}_part{part}.npz')
    before = np.flatnonzero(z['t'] < t_ho * (1.0 - 1.0e-6))[-1]
    t0, y0 = float(z['t'][before]), np.asarray(z['y'][:, before], dtype=float)
    values, s, offset = over_the_merge(a, t0, y0, t_ho)
    check_stored_mesh(z, s.evaluator.mesh, f'{a.run.name}/{STEM}_part{part}.npz')

    refs = json.loads((a.ref / f'{STEM}.json').read_text())
    k_ref = next(k for k, r in enumerate(refs) if r['t_target'] > t0)
    zr = np.load(a.ref / f'{STEM}_part{k_ref}.npz')
    j = int(np.argmin(np.abs(zr['t'] - t0)))
    r = _solver(a, lid=False, nodes=400, cell=a.ref_cell, end=1.0)
    check_stored_mesh(zr, r.evaluator.mesh, f'{a.ref.name}/{STEM}_part{k_ref}.npz')
    r._dSdt_single(0.0, zr['y'][:, j])
    T_ref = np.asarray(r.state.phase_staggered.temperature()).ravel()[::-1]
    mesh = r.evaluator.mesh
    z_ref = (float(np.ravel(mesh.basic.radii)[-1]) - np.ravel(mesh.staggered.radii))[::-1]
    a.out.write_text(
        json.dumps(
            dict(
                sha=rows[-1]['sha'],
                **values,
                offset_predicted=offset,
                offset_measured=float(
                    values['before']['T_cell'] - np.interp(0.5 * s._lid_h_top, z_ref, T_ref)
                ),
            )
        )
    )


if __name__ == '__main__':
    main()
