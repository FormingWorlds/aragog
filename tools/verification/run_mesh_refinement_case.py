"""Run one case of the mesh refinement study: abe_solid.toml from a molten start, in parts.

Each part is a fresh solver started from the end state of the previous part and solved to the
next sample time. After each part the run folder gets ``za_part<k>.npz`` (stored rows ``t``
[yr], ``y``, the basic and staggered temperatures, ``S_sol`` the staggered solidus entropy and
``y_end``) and one record in ``za.json`` (time, status, melt fraction, CMB temperature, energy
integrals, CVODE counters, mesh, source commit, config hash and the md5 of every table file).
``tools/verification/run_mesh_refinement_study.py`` reads these folders. Run it from the root of
the aragog tree under test.

Usage::

    python tools/verification/run_mesh_refinement_case.py <outdir> --eos-dir DIR
        [--eos-md5 MD5] [--samples 1e2,1e3,1e4,1e5] [--nodes 100] [--surface-cell M]
        [--cmb-cell M] [--s-init S] [--core-module-params JSON] [--resume] [--max-parts K]
        [--rows OUT.npz]

With ``--rows`` nothing is solved: the run folder's solver is rebuilt from the same arguments
(the config hash must match its records) and ``OUT.npz`` gets the time ``t`` [yr] and the global
melt fraction ``Phi`` of every stored row, the input of the study tool's ``--phi-rows``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

STEM = 'za'
RHEOLOGY = dict(
    enabled=True, stress_closure_mode='lid', yield_stress_c=1.0e7, yield_stress_mu=0.1
)
RUN = dict(
    kappah_floor=10.0, viscosity=1.0e21, emissivity=1.0, max_steps=800_000, yield_max=1.0e8
)
CV_RE = re.compile(
    r'CVODE stats: nst=(-?\d+) nfe=(-?\d+) nsetups=-?\d+ netf=(-?\d+) '
    r'order=-?\d+ last_step=(\S+)'
)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('outdir', type=Path)
    ap.add_argument('--eos-dir', type=Path, required=True, help='aragog P-S tables of the run')
    ap.add_argument('--eos-md5', help='required md5 of temperature_solid.dat in --eos-dir')
    ap.add_argument('--samples', default='1e2,1e3,1e4,1e5', help='part end times [yr]')
    ap.add_argument(
        '--numpy', action='store_true', help='NumPy right-hand side, no JAX factory'
    )
    ap.add_argument('--s-init', type=float, default=9466.0, help='start entropy [J/kg/K]')
    ap.add_argument('--nodes', type=int, default=100)
    ap.add_argument('--surface-cell', type=float, default=0.0, help='top cell [m]; 0 = off')
    ap.add_argument('--cmb-cell', type=float, default=0.0, help='CMB cell [m]; 0 = off')
    ap.add_argument(
        '--core-module-params', type=Path, help='JSON: core_bc core_module with these'
    )
    ap.add_argument('--resume', action='store_true', help='continue after the last part')
    ap.add_argument(
        '--max-parts', type=int, default=0, help='stop after this many parts; 0 = all'
    )
    ap.add_argument('--rows', type=Path, help='write t and Phi of every stored row; no solve')
    return ap.parse_args(argv)


def case_config(a: argparse.Namespace, tree: Path) -> tuple[str, dict | None, str]:
    """The config text, the core module parameters and the config hash of the case."""
    from aragog.rheology import SolidRheologyParams

    text = (tree / 'src/aragog/cfg/abe_solid.toml').read_text()
    if text.count('[phase_mixed]\n') != 1 or 'matprop_smooth_width' in text:
        raise ValueError('abe_solid.toml has no single [phase_mixed] table to add the width to')
    text = text.replace('[phase_mixed]\n', '[phase_mixed]\nmatprop_smooth_width = 0.01\n')
    core = json.loads(a.core_module_params.read_text()) if a.core_module_params else None
    rheo = SolidRheologyParams(**RHEOLOGY, yield_stress_max=RUN['yield_max'])
    key = dict(kappah_floor=RUN['kappah_floor'], viscosity=RUN['viscosity'], rheo=repr(rheo))
    key |= dict(emissivity=RUN['emissivity'], nodes=a.nodes, s_init=repr(a.s_init))
    key |= dict(surface_cell=a.surface_cell, numpy=a.numpy, max_steps=RUN['max_steps'])
    key |= ({'cmb_cell': a.cmb_cell} if a.cmb_cell > 0 else {}) | (
        {'core_module': core} if core else {}
    )
    return text, core, hashlib.sha256((text + json.dumps(key)).encode()).hexdigest()[:16]


def _table_md5(eos_dir: Path) -> dict:
    """The md5 of every table file ``*.dat`` in ``eos_dir``, by file name."""
    return {
        p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(eos_dir.glob('*.dat'))
    }


class _CvodeStats(logging.Handler):
    """Sum CVODE's counters from the solver's per-call INFO line."""

    def __init__(self):
        super().__init__()
        self.cv = {}

    def emit(self, record):
        m = CV_RE.search(record.getMessage())
        if m:
            for k, v in zip(('nst', 'nfe', 'netf'), m.groups()[:3]):
                self.cv[k] += int(v)
            self.cv['calls'] += 1
            self.cv['last_step'] = float(m.group(4))


def _install_factory(s, eos_dir, counter):
    """JAX CVODE right-hand side and Jacobian on the run's tables, with a call counter."""
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    b = s.parameters.boundary_conditions
    bc = BoundaryParams(
        outer_bc_type=b.outer_boundary_condition,
        outer_bc_value=b.outer_boundary_value,
        emissivity=b.emissivity,
        T_eq=b.equilibrium_temperature,
        inner_bc_type=(5 if s._core_bc == 'energy_balance' else b.inner_boundary_condition),
        inner_bc_value=b.inner_boundary_value,
        core_density=s.parameters.mesh.core_density,
        core_heat_capacity=b.core_heat_capacity,
        tfac_core_avg=b.tfac_core_avg,
        cmb_area=float(s._cmb_area),
        core_M=float(s._core_M),
        cmb_dr_cmb=float(s._cmb_dr_cmb),
        param_utbl=bool(b.param_utbl),
        param_utbl_const=float(b.param_utbl_const),
    )
    eos = EntropyEOS_JAX(str(eos_dir))
    mesh, params = (
        MeshArrays.from_numpy_mesh(s.evaluator.mesh),
        PhaseParams.from_parameters(s.parameters),
    )

    def factory(scales, core_bc_mode):
        rhs, jac, _ = build_jax_rhs_and_jacobian(
            eos,
            params,
            mesh,
            bc,
            np.zeros(s._n_stag),
            scales,
            core_bc_mode=core_bc_mode,
            radio_isotope_params=(),
            core_module_budget=getattr(s, '_core_module_budget', None),
            core_module_q_radio=float(getattr(s, '_core_module_q_radio', 0.0)),
            core_module_ra_crit_cmb=getattr(s, '_core_module_ra_crit_cmb', None),
        )

        def counted(*args, **kw):
            counter[0] += 1
            return rhs(*args, **kw)

        for name in ('rheology', 'phase_params'):  # the solver's rheology check reads these
            if hasattr(rhs, name):
                setattr(counted, name, getattr(rhs, name))
        return counted, jac

    s.set_jax_cvode_factory(factory)


def _record(s, t_yr, y):
    """Diagnostics of the state ``y`` at ``t_yr`` and the arrays stored with the part."""
    s.dSdt(t_yr, y)
    st = s.state
    lid = st._lid_state if isinstance(st._lid_state, dict) else {}
    S = np.asarray(y, float).ravel()[: s._n_stag]
    S_sol = np.asarray(st._S_sol_stag, float)
    phi = np.asarray(st.phase_staggered.melt_fraction(), float).ravel()
    r_b = np.asarray(s.evaluator.mesh.basic.radii, float).ravel()
    m = (
        np.asarray(st.phase_staggered.density(), float).ravel()
        * 4
        / 3
        * np.pi
        * np.diff(r_b**3)
    )
    tb = np.asarray(st.phase_basic.temperature(), float).ravel()
    d = dict(
        t_yr=t_yr,
        lid={k: float(np.asarray(v)) for k, v in lid.items() if np.asarray(v).size == 1},
        F_top=float(np.ravel(st._heat_flux)[-1]),
        T_top=float(tb[-1]),
        phi_max=float(phi.max()),
        Phi_global=float((m * phi).sum() / m.sum()),
        n_melt=int((S > S_sol).sum()),
        all_subsolidus=bool(np.all(S <= S_sol)),
        melt_cells=np.flatnonzero(S > S_sol).tolist(),
    )
    ts = np.asarray(st.phase_staggered.temperature(), float).ravel()
    return d, dict(tb=tb, ts=ts, S=S, S_liq=np.asarray(st._S_liq_stag, float), S_sol=S_sol)


def _steady(prev, d):
    """Solid in two parts in a row, the lid unchanged to 1 % in depth and 1 K at its base."""
    a, b = prev['lid'], d['lid']
    if not (prev['all_subsolidus'] and d['all_subsolidus'] and a.get('d_lid') is not None):
        return False
    return abs(b['d_lid'] - a['d_lid']) <= 0.01 * max(abs(a['d_lid']), 1.0) and (
        abs(b['T_i'] - a['T_i']) <= 1.0
    )


def main(argv=None) -> int:
    a = parse(argv)
    sys.path.insert(0, str(Path(__file__).parent))
    from _ssc_helpers import entropy_eos_copy

    import aragog
    from aragog.config import Config
    from aragog.jax.solver import SECS_PER_YEAR
    from aragog.solver.entropy_solver import EntropySolver

    tree = Path.cwd()
    if not Path(aragog.__file__).resolve().is_relative_to(tree.resolve()):
        raise SystemExit(f'aragog imported from {aragog.__file__}, not from {tree}')
    md5 = hashlib.md5((a.eos_dir / 'temperature_solid.dat').read_bytes()).hexdigest()
    if a.eos_md5 and md5 != a.eos_md5:
        raise SystemExit(f'temperature_solid.dat md5 {md5}, expected {a.eos_md5}')
    tables = _table_md5(a.eos_dir)
    git = ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}']
    sha, tree_id = subprocess.run(
        git, capture_output=True, text=True, check=True
    ).stdout.split()
    text, core, config_hash = case_config(a, tree)
    cfg = Path(tempfile.mkdtemp()) / 'abe_solid_w.toml'
    cfg.write_text(text)
    rows, t_prev, y_prev, k0 = [], 0.0, None, 0
    out_json = a.outdir / f'{STEM}.json'
    if a.rows:
        rows = json.loads(out_json.read_text())
        if {r['config_hash'] for r in rows} != {config_hash} or (
            rows[0].get('table_md5', tables) != tables
        ):
            raise SystemExit(f'{out_json} comes from another config or table set')
    if a.resume and out_json.exists():
        rows = json.loads(out_json.read_text())
        if (
            {r['sha'] for r in rows} != {sha}
            or {r['config_hash'] for r in rows} != {config_hash}
            or rows[0].get('table_md5', tables) != tables
        ):
            raise SystemExit(f'{out_json} comes from another commit, config or table set')
        if not rows[-1]['reached'] or rows[-1]['status'] < 0:
            raise SystemExit(
                f'{out_json}: part {len(rows) - 1} did not reach its time; rerun it'
            )
        k0 = len(rows)
        z = np.load(a.outdir / f'{STEM}_part{k0 - 1}.npz')
        t_prev, y_prev = float(z['t'][-1]), np.asarray(z['y_end'], float)
    eos_np = entropy_eos_copy(a.eos_dir)
    stats = _CvodeStats()
    log = logging.getLogger('fwl.aragog.solver.entropy_solver')
    log.setLevel(logging.INFO)
    log.addHandler(stats)
    calls = [0]
    samples = [float(v) for v in a.samples.split(',')]
    a.outdir.mkdir(parents=True, exist_ok=True)

    def make(end_time):
        from aragog.rheology import SolidRheologyParams

        c = Config.from_file(str(cfg))
        c.solver.end_time, c.solver.max_steps = end_time, RUN['max_steps']
        c.energy.kappah_floor, c.phase_solid.viscosity = RUN['kappah_floor'], RUN['viscosity']
        c.phase_solid.rheology = SolidRheologyParams(
            **RHEOLOGY, yield_stress_max=RUN['yield_max']
        )
        c.boundary_conditions.emissivity = RUN['emissivity']
        c.mesh.number_of_nodes = a.nodes
        c.mesh.surface_cell_thickness, c.mesh.cmb_cell_thickness = a.surface_cell, a.cmb_cell
        if core:
            b = c.boundary_conditions
            b.inner_boundary_condition, b.core_module_params, b.core_bc = (
                1,
                dict(core),
                'core_module',
            )
        s = EntropySolver(c, entropy_eos=eos_np)
        s.initialize()
        s.set_initial_entropy(a.s_init)
        if not (a.numpy or a.rows):
            _install_factory(s, a.eos_dir, calls)
        return s

    if a.rows:
        s, t, phi = make(samples[-1]), [], []
        for k in range(len(rows)):
            z = np.load(a.outdir / f'{STEM}_part{k}.npz')
            for j in range(z['t'].size):
                t.append(float(z['t'][j]))
                phi.append(_record(s, t[-1], z['y'][:, j])[0]['Phi_global'])
        np.savez(a.rows, t=np.array(t), Phi=np.array(phi))
        return 0

    w_all = time.perf_counter()
    for k, t_end in enumerate(samples[k0:], start=k0):
        if a.max_parts and k >= k0 + a.max_parts:
            break
        s = make(t_end)
        if y_prev is not None:
            sc = s._build_nondim_scales()
            off = 0.0 if sc.state_offset is None else np.asarray(sc.state_offset)
            y_nd0, seg = (y_prev - off) / np.asarray(sc.state_scale), s._solve_cvode_segments

            def restart(_seg=seg, _t=t_prev, _y=y_nd0, **kw):
                kw.setdefault('max_segments', 50_000)
                return _seg(**{**kw, 'start_time': _t / kw['t_ref'], 'y0': _y})

            s._solve_cvode_segments = restart
        calls[0] = 0
        stats.cv = dict(calls=0, nst=0, nfe=0, netf=0, last_step=None)
        w0 = time.perf_counter()
        s.solve()
        wall = time.perf_counter() - w0
        tt, yy = np.asarray(s._solution.t, float), np.asarray(s._solution.y, float)
        if y_prev is not None and (
            abs(tt[0] - t_prev) > 1e-9 * t_prev
            or np.max(np.abs(yy[:, 0] - y_prev) / np.maximum(np.abs(y_prev), 1e-30)) > 1e-9
        ):
            raise RuntimeError(f'part {k} does not start from the end state of part {k - 1}')
        t_got, y_end = float(tt[-1]), yy[:, -1].copy()
        ei = {
            k2: (None if v is None else float(v))
            for k2, v in dict(s._solution.get('energy_integrals') or {}).items()
        }
        d, arrays = _record(s, t_got, y_end)
        r = np.ravel(s.evaluator.mesh.basic.radii)
        d.update(
            part=k,
            t_target=t_end,
            reached=bool(t_got >= t_end * (1 - 1e-9)),
            status=int(s._solution.status),
            n_out=int(tt.size),
            rhs_calls=calls[0],
            cvode=dict(stats.cv),
            wall_part_s=round(wall, 1),
            wall_total_s=round(time.perf_counter() - w_all, 1),
            sha=sha,
            tree=tree_id,
            factory=not a.numpy,
            nodes=a.nodes,
            s_init=a.s_init,
            surface_cell=a.surface_cell,
            cmb_cell=a.cmb_cell,
            eos_dir=str(a.eos_dir),
            eos_md5=md5,
            table_md5=tables,
            core_bc=s._core_bc,
            core_module_params=core,
            h_top=float(r[-1] - r[-2]),
            h_cmb=float(r[1] - r[0]),
            energy_integrals=ei,
            T_cmb=float(arrays['tb'][0]),
            config_hash=config_hash,
            F_top_mean_step_integral=(
                -ei['F_int'] / (4 * np.pi * r[-1] ** 2) / ((t_got - tt[0]) * SECS_PER_YEAR)
                if ei.get('F_int') is not None and t_got > tt[0]
                else None
            ),
        )
        rows.append(d)
        np.savez(a.outdir / f'{STEM}_part{k}.npz', t=tt, y=yy, y_end=y_end, **arrays)
        out_json.write_text(
            '[\n' + ',\n'.join(json.dumps(x, separators=(',', ':')) for x in rows) + '\n]\n'
        )
        print(
            json.dumps(
                {x: d[x] for x in ('part', 't_yr', 'status', 'wall_part_s', 'Phi_global')}
            ),
            flush=True,
        )
        if not d['reached'] or d['status'] < 0:
            return 3
        if len(rows) > 6 and _steady(rows[-2], d):
            break
        t_prev, y_prev = t_got, y_end
    return 0


if __name__ == '__main__':
    sys.exit(main())
