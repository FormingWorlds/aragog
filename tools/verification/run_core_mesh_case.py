"""Run one case of the CMB mesh study: a solid mantle over a ``core_module`` core, in segments.

The run starts from one isentrope (``--solid-s0``) and a core temperature (``--t-core``) and
runs in segments: the first ``--seg1-yr`` long, then ``--seg-yr`` each. Every segment is one
``solve()`` of the same solver, started from the end entropy of the previous segment; the core
temperature carries over. The CMB flux is aragog's own law (``aragog.core.cmb_boundary_layer_flux``).

Outputs, with ``<out>`` the path given:

- ``<out>``: one CSV row per segment (CVODE counters, surface, core and lid state).
- ``<out>.cmb.csv``: one row per call of the CMB flux law: the caller (``cvode`` for the CVODE
  right-hand side, ``trace`` for the energy pass after a solve, ``other``), the time [yr], the
  core and mantle temperatures, the viscosity, the conductive and the convective flux and the
  flux the solver used [W/m^2].
- ``<out>.nodes.bin``: float64 rows ``[t_yr, y...]`` of the accepted CVODE states, one each time
  the time has advanced by ``--node-dt-yr``; ``<out>.nodes.json`` holds the row length.
- ``<out>.S.npy``: the end entropy of the last segment.
- ``<out>.toml``: the config of the run, the base config with the mesh options applied.

Run it from the root of the aragog tree under test::

    python tools/verification/run_core_mesh_case.py tools/verification/data/core_mesh_study.toml
        <out> --eos-dir DIR --solid-s0 2045.05 --t-core 2500 [--nodes N] [--cmb-cell M]
        [--surface-cell M|uniform] [--segments 2001] [--seg1-yr 2000] [--seg-yr 50000]
        [--node-dt-yr 50000]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

import aragog.core as aragog_core
from aragog.eos.entropy import EntropyEOS
from aragog.parser import Parameters
from aragog.solver.entropy_solver import EntropySolver

LID_KEYS = (
    'T_i', 'theta', 'dT_rh', 'd_lid', 'r_lid_base', 'P_lid_base', 'tau_d_lid', 'tau_y_lid',
    'w_y', 'w_active', 'has_lid', 'eta_i',
)  # fmt: skip
COLS = (
    'seg', 't_end_yr', 'wall_s', 'nst', 'nfe', 'n_out', 'last_step_nd', 'netf', 'T_surf',
    'T_top_cell', 'T_core', *LID_KEYS, 'F_lid_base', 'phi_top', 'Phi_global', 'alpha_i',
    'max_phi_outputs', 'n_mush_outputs',
)  # fmt: skip
CMB_COLS = ('src', 't_yr', 'n_nodes', 'T_core', 'T_m', 'eta', 'q_cond', 'q_conv', 'q')


MESH_KEYS = {
    'nodes': 'number_of_nodes',
    'cmb_cell': 'cmb_cell_thickness',
    'surface_cell': 'surface_cell_thickness',
}


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config', type=Path)
    ap.add_argument('out')
    ap.add_argument('--nodes', type=int, help='number_of_nodes; default: as in the config')
    ap.add_argument('--cmb-cell', type=float, help='cmb_cell_thickness [m]; 0 = uniform end')
    ap.add_argument('--surface-cell', help="surface_cell_thickness [m], 0 or 'uniform'")
    ap.add_argument('--eos-dir', type=Path, required=True, help='aragog P-S tables of the run')
    ap.add_argument('--solid-s0', type=float, required=True, help='start entropy [J/kg/K]')
    ap.add_argument('--t-core', type=float, required=True, help='start core temperature [K]')
    ap.add_argument('--segments', type=int, default=2001)
    ap.add_argument('--seg1-yr', type=float, default=2000.0)
    ap.add_argument('--seg-yr', type=float, default=50000.0)
    ap.add_argument('--node-dt-yr', type=float, default=50000.0)
    return ap.parse_args(argv)


def case_config(text: str, a: argparse.Namespace) -> str:
    """The config text with the mesh options of the run in place of the base values."""
    for opt, key in MESH_KEYS.items():
        value = getattr(a, opt)
        if value is None:
            continue
        if opt == 'surface_cell' and value != 'uniform':
            value = float(value)
        value = '"uniform"' if value == 'uniform' else repr(value)
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if line.split('=')[0].strip() == key]
        assert len(hits) == 1, f'{key}: {len(hits)} lines in the config'
        lines[hits[0]] = f'{key} = {value}'
        text = '\n'.join(lines) + '\n'
    return text


class _Stats(logging.Handler):
    """Keep the arguments of the last 'CVODE stats' log record."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.args = None

    def emit(self, record):
        if record.getMessage().startswith('CVODE stats'):
            self.args = record.args


class _Recorder:
    """Records every call of the CMB flux law and writes the accepted CVODE states."""

    def __init__(self, out: str, node_dt_yr: float):
        self.fh = open(out + '.cmb.csv', 'w', newline='')
        self.w = csv.writer(self.fh)
        self.w.writerow(CMB_COLS)
        self.nodes = open(out + '.nodes.bin', 'wb')
        self.out, self.node_dt_yr = out, node_dt_yr
        self.scales, self.last_t, self.n_y, self.n_written, self.calls = (
            None,
            -np.inf,
            None,
            0,
            0,
        )
        self.law = aragog_core.cmb_boundary_layer_flux

    def install(self):
        """Wrap the scale builder (for the time unit) and the CMB flux law."""
        build = EntropySolver._build_nondim_scales

        def scales(solver, *a, **k):
            self.scales = build(solver, *a, **k)
            return self.scales

        EntropySolver._build_nondim_scales = scales
        aragog_core.cmb_boundary_layer_flux = self.flux

    @staticmethod
    def _caller():
        """(src, t_nd, nodes) of the current call, from the Python call stack."""
        f = sys._getframe(2)
        src, t_nd, nodes = 'other', float('nan'), None
        for _ in range(40):
            if f is None:
                break
            name = f.f_code.co_name
            if name == 'rhs_fn' and src == 'other':
                src, t_nd = 'cvode', float(f.f_locals['t'])
            elif name == '_solve_cvode' and 'nodes' in f.f_locals:
                nodes = f.f_locals['nodes']
                break
            elif name == '_compute_step_energy_integrals':
                src = 'trace'
                break
            f = f.f_back
        return src, t_nd, nodes

    def flux(self, t_hot, t_m, *, conductivity, density, heat_capacity, expansivity, viscosity,
             gravity, dr_half, ra_crit, **kw):  # fmt: skip
        q = float(
            self.law(
                t_hot, t_m, conductivity=conductivity, density=density,
                heat_capacity=heat_capacity, expansivity=expansivity, viscosity=viscosity,
                gravity=gravity, dr_half=dr_half, ra_crit=ra_crit, **kw,
            )
        )  # fmt: skip
        dT = t_hot - t_m
        kappa = conductivity / (density * heat_capacity)
        q_cond = conductivity * dT / dr_half
        q_conv = float('nan')
        if dT > 0:
            buoy = expansivity * density * gravity / (ra_crit * kappa * viscosity)
            q_conv = conductivity * abs(dT) ** (4 / 3) * buoy ** (1 / 3)
        src, t_nd, nodes = self._caller()
        sc = self.scales
        t_yr = t_nd * sc.t_ref if (sc is not None and t_nd == t_nd) else float('nan')
        if nodes and sc is not None:
            t_last, y_last = nodes[-1]
            if t_last * sc.t_ref >= self.last_t + self.node_dt_yr:
                y = np.asarray(y_last, dtype=float) * sc.state_scale + sc.state_offset
                if self.n_y is None:
                    self.n_y = y.size
                assert y.size == self.n_y, (y.size, self.n_y)
                self.nodes.write(np.r_[t_last * sc.t_ref, y].astype('<f8').tobytes())
                self.last_t = t_last * sc.t_ref
                self.n_written += 1
        n_nodes = len(nodes) if nodes is not None else -1
        row = (
            src,
            t_yr,
            n_nodes,
            float(t_hot),
            float(t_m),
            float(viscosity),
            q_cond,
            q_conv,
            q,
        )
        self.w.writerow(row)
        self.calls += 1
        return q

    def close(self):
        self.fh.close()
        self.nodes.close()
        meta = {'n_y': self.n_y, 'n_written': self.n_written, 'node_dt_yr': self.node_dt_yr}
        meta['calls'] = self.calls
        Path(self.out + '.nodes.json').write_text(json.dumps(meta))


def alpha_i(eos, st, lid) -> float:
    """Thermal expansivity [1/K] of the cell nearest the interior radius of the lid state."""
    j = int(np.argmin(np.abs(np.ravel(st.r_stag) - float(np.asarray(lid['r_i'])))))
    return float(np.ravel(eos.thermal_expansivity(np.ravel(st.P_stag)[j], np.ravel(st.S_final)[j]))[0])  # fmt: skip


def mush_outputs(eos, solver, sol) -> tuple[float, int]:
    """Largest cell melt fraction over the outputs of a segment, and the number of outputs with
    a cell between the solidus and the liquidus entropy."""
    P = np.ravel(solver._P_stag_flat)
    Y = np.asarray(sol.y[: solver._n_stag], dtype=float)
    s_sol, s_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    phi = np.array([np.ravel(eos.melt_fraction(P, Y[:, j])) for j in range(Y.shape[1])])
    mush = (Y.T > s_sol) & (Y.T < s_liq)
    return float(phi.max()), int(mush.any(axis=1).sum())


def solid_start(solver, eos, s0: float) -> np.ndarray:
    """One isentrope at ``s0``; refuses a start with melt in any cell."""
    P = np.ravel(solver._P_stag_flat)
    s = np.full_like(P, s0)
    phi0 = np.ravel(eos.melt_fraction(P, s))
    assert np.all(phi0 == 0.0), f'solid start has melt: max phi {phi0.max()}'
    return s


def segment_row(k, t_end, wall, sol, stats, solver, eos) -> list:
    """The CSV row of segment ``k`` after one right-hand side call at its end state."""
    solver._dSdt_single(float(t_end), np.asarray(sol.y[:, -1], dtype=float))
    st = solver.get_state()
    lid = solver.state._lid_state
    assert lid is not None, f'segment {k}: no lid state'
    r_b, f_b = np.ravel(st.r_basic), np.ravel(st.heat_flux)
    r_lb = float(np.asarray(lid['r_lid_base']))
    f_lb = float(np.interp(r_lb, r_b, f_b)) if r_b[0] <= r_lb <= r_b[-1] else float('nan')
    sa = stats.args or ()
    return [
        k, t_end, wall, int(getattr(sol, 'cvode_nst', -1)), int(getattr(sol, 'cvode_nfe', -1)),
        len(sol.t), float(sa[5]) if len(sa) > 5 else float('nan'), int(sa[3]) if len(sa) > 3 else -1,
        float(np.ravel(st.T_basic)[-1]), float(np.ravel(st.T_stag)[-1]), float(st.T_core),
        *[float(np.asarray(lid[key])) for key in LID_KEYS], f_lb, float(np.ravel(st.phi_stag)[-1]),
        float(st.Phi_global), alpha_i(eos, st, lid), *mush_outputs(eos, solver, sol),
    ]  # fmt: skip


def main(argv=None) -> int:
    a = parse(argv)
    stats = _Stats()
    logging.getLogger('fwl').addHandler(stats)
    logging.getLogger('fwl').setLevel(logging.INFO)
    config = Path(a.out + '.toml')
    config.write_text(case_config(a.config.read_text(), a))
    rec = _Recorder(a.out, a.node_dt_yr)
    rec.install()
    params = Parameters.from_file(str(config))
    eos = EntropyEOS(str(a.eos_dir))
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_core_temperature(a.t_core)
    solver.set_initial_entropy(solid_start(solver, eos, a.solid_s0))
    # Later segments continue from the core temperature of the previous one.
    solver.set_initial_core_temperature(None)

    def t_end(k):
        return 0.0 if k <= 0 else a.seg1_yr + (k - 1) * a.seg_yr

    try:
        with open(a.out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            for k in range(1, a.segments + 1):
                params.solver.start_time, params.solver.end_time = t_end(k - 1), t_end(k)
                stats.args = None
                t0 = time.perf_counter()
                solver.solve()
                wall = time.perf_counter() - t0
                sol = solver._solution
                assert int(sol.status) >= 0, f'segment {k}: status {sol.status} {sol.message}'
                s_end = np.asarray(sol.y[: solver._n_stag, -1], dtype=float)
                assert np.all(np.isfinite(s_end)), f'segment {k}: non-finite entropy'
                w.writerow(segment_row(k, t_end(k), wall, sol, stats, solver, eos))
                fh.flush()
                np.save(a.out + '.S.npy', s_end)
                solver.set_initial_entropy(s_end)
    finally:
        rec.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
