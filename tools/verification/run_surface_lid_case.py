"""Run one case of the lid base study: abe_solid.toml with the solid rheology, the 'lid' closure
and the sub-grid lid, from a molten start, in parts.

Each part is a fresh solver started from the end state of the previous part and solved to the
next sample time. After each part the run folder gets ``za_part<k>.npz`` (stored rows ``t``
[yr], ``y`` and ``y_end``) and one record in ``za.json`` (time, lid thickness and heat content,
surface flux, interior temperature of the lid closure, status, mesh, jump floor, source commit,
config hash and the md5 of every table file). Run it from the root of the aragog tree under
test. A 400-node run to 4e5 yr takes about 10 h on one core.

Usage::

    python tools/verification/run_surface_lid_case.py <outdir> --eos-dir DIR [--eos-md5 MD5]
        [--samples 1e2,1e3,...] [--nodes 100] [--s-init S] [--dh-min J_PER_KG] [--resume]
    python tools/verification/run_surface_lid_case.py --collect OUT.npz NAME=DIR [NAME=DIR ...]

``--collect`` writes the time and, per run NAME, ``<NAME>_delta`` [m], ``<NAME>_F_top`` [W/m^2]
and ``<NAME>_T_i`` [K] of every part to ``OUT.npz``; all runs must share the times, the commit
and the table folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

STEM = 'za'
SAMPLES = '1e2,1e3,3e3,1e4,3e4,1e5,2e5,3e5,4e5'
RHEOLOGY = dict(
    enabled=True,
    stress_closure_mode='lid',
    yield_stress_c=1.0e7,
    yield_stress_mu=0.1,
    yield_stress_max=1.0e8,
)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('outdir', type=Path, nargs='?')
    ap.add_argument('--eos-dir', type=Path, help='aragog P-S tables of the run')
    ap.add_argument('--eos-md5', help='required md5 of temperature_solid.dat in --eos-dir')
    ap.add_argument('--samples', default=SAMPLES, help='part end times [yr]')
    ap.add_argument('--nodes', type=int, default=100)
    ap.add_argument('--s-init', type=float, default=9466.0, help='start entropy [J/kg/K]')
    ap.add_argument('--dh-min', type=float, help='jump floor LID_DH_MIN [J/kg]; default: code')
    ap.add_argument('--resume', action='store_true', help='continue after the last part')
    ap.add_argument('--collect', nargs='+', metavar=('OUT.npz', 'NAME=DIR'))
    return ap.parse_args(argv)


def _table_md5(eos_dir: Path) -> dict:
    """The md5 of every table file ``*.dat`` in ``eos_dir``, by file name."""
    return {
        p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(eos_dir.glob('*.dat'))
    }


def _install_factory(s, eos_dir):
    """Give ``s`` the JAX CVODE right-hand side and Jacobian, with the lid, on ``eos_dir``."""
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    eos_jax = EntropyEOS_JAX(str(eos_dir))
    mesh_jax = MeshArrays.from_numpy_mesh(s.evaluator.mesh)
    params_jax = PhaseParams.from_parameters(s.parameters)
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
        tfac_core_avg=getattr(b, 'tfac_core_avg', 1.147),
        cmb_area=float(getattr(s, '_cmb_area', 0.0)),
        core_M=float(getattr(s, '_core_M', 0.0)),
        cmb_dr_cmb=float(getattr(s, '_cmb_dr_cmb', 0.0)),
        param_utbl=bool(getattr(b, 'param_utbl', False)),
        param_utbl_const=float(getattr(b, 'param_utbl_const', 0.0)),
    )

    def factory(scales, core_bc_mode):
        rhs, jac, _ = build_jax_rhs_and_jacobian(
            eos_jax,
            params_jax,
            mesh_jax,
            bc,
            np.zeros(s._n_stag),
            scales,
            core_bc_mode=core_bc_mode,
            radio_isotope_params=(),
            core_module_budget=getattr(s, '_core_module_budget', None),
            core_module_q_radio=float(getattr(s, '_core_module_q_radio', 0.0)),
            core_module_ra_crit_cmb=getattr(s, '_core_module_ra_crit_cmb', None),
            lid_params=s.jax_lid_params(),
        )
        return rhs, jac

    s.set_jax_cvode_factory(factory)


def collect(out: Path, runs: list[str]) -> None:
    """Write the per-part time, lid thickness, surface flux and T_i of each run to ``out``.

    The thickness is ``y_end[-2]`` of each part file; ``T_i`` is read from the record or from
    its ``lid`` entry, so folders of an older record layout with the same parts also load.
    """
    data, meta = {}, set()
    for spec in runs:
        name, folder = spec.split('=', 1)
        rows = json.loads((Path(folder) / f'{STEM}.json').read_text())
        meta.add(
            (tuple(r['t_yr'] for r in rows), rows[0]['sha'], Path(rows[0]['eos_dir']).name)
        )
        data[f'{name}_delta'] = np.array(
            [
                np.load(Path(folder) / f'{STEM}_part{k}.npz')['y_end'][-2]
                for k in range(len(rows))
            ]
        )
        data[f'{name}_F_top'] = np.array([r['F_top'] for r in rows])
        data[f'{name}_T_i'] = np.array(
            [r.get('T_i', r.get('lid', {}).get('T_i')) for r in rows]
        )
        data[f'{name}_dh_min'] = rows[0]['dh_min']
        data[f'{name}_nodes'] = rows[0]['nodes']
    if len(meta) != 1:
        raise SystemExit(f'the runs differ in their times, commit or table: {meta}')
    times, sha, table = meta.pop()
    np.savez(out, times_yr=np.array(times), commit=sha, table=table, **data)


def main(argv=None) -> int:
    a = parse(argv)
    if a.collect:
        collect(Path(a.collect[0]), a.collect[1:])
        return 0
    sys.path.insert(0, str(Path(__file__).parent))
    from _ssc_helpers import entropy_eos_copy

    import aragog
    import aragog.surface_lid as sl
    from aragog.config import Config
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    tree = Path.cwd()
    if not Path(aragog.__file__).resolve().is_relative_to(tree.resolve()):
        raise SystemExit(f'aragog imported from {aragog.__file__}, not from {tree}')
    md5 = hashlib.md5((a.eos_dir / 'temperature_solid.dat').read_bytes()).hexdigest()
    if a.eos_md5 and md5 != a.eos_md5:
        raise SystemExit(f'temperature_solid.dat md5 {md5}, expected {a.eos_md5}')
    if a.dh_min is not None:
        if not (np.isfinite(a.dh_min) and a.dh_min > 0.0):
            raise SystemExit(f'--dh-min must be finite and positive, got {a.dh_min}')
        sl.LID_DH_MIN = a.dh_min
    tables = _table_md5(a.eos_dir)
    git = ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}']
    sha, tree_id = subprocess.run(
        git, capture_output=True, text=True, check=True
    ).stdout.split()
    text = (tree / 'src/aragog/cfg/abe_solid.toml').read_text()
    for head in ('[phase_mixed]\n', '[boundary_conditions]\n'):
        if text.count(head) != 1:
            raise SystemExit(f'abe_solid.toml has {text.count(head)} {head.strip()} tables')
    text = text.replace(
        '[boundary_conditions]\n', '[boundary_conditions]\nsubgrid_lid = true\n'
    )
    text = text.replace('[phase_mixed]\n', '[phase_mixed]\nmatprop_smooth_width = 0.01\n')
    config_hash = hashlib.sha256(
        (
            text
            + json.dumps(
                dict(RHEOLOGY, nodes=a.nodes, s_init=repr(a.s_init), dh_min=sl.LID_DH_MIN)
            )
        ).encode()
    ).hexdigest()[:16]
    cfg = Path(tempfile.mkdtemp()) / 'abe_solid_lid.toml'
    cfg.write_text(text)
    rows, t_prev, y_prev = [], 0.0, None
    out_json = a.outdir / f'{STEM}.json'
    if a.resume and out_json.exists():
        rows = json.loads(out_json.read_text())
        if {(r['sha'], r['config_hash']) for r in rows} != {(sha, config_hash)} or (
            rows[0]['table_md5'] != tables
        ):
            raise SystemExit(f'{out_json} comes from another commit, config or table set')
        if not rows[-1]['reached'] or rows[-1]['status'] < 0:
            raise SystemExit(f'{out_json}: part {len(rows) - 1} did not reach its time')
        z = np.load(a.outdir / f'{STEM}_part{len(rows) - 1}.npz')
        t_prev, y_prev = float(z['t'][-1]), np.asarray(z['y_end'], float)
    eos_np = entropy_eos_copy(a.eos_dir)
    a.outdir.mkdir(parents=True, exist_ok=True)
    samples = [float(v) for v in a.samples.split(',')]

    for k, t_end in enumerate(samples[len(rows) :], start=len(rows)):
        c = Config.from_file(str(cfg))
        c.solver.end_time, c.solver.max_steps = t_end, 800_000
        c.energy.kappah_floor, c.phase_solid.viscosity = 10.0, 1.0e21
        c.phase_solid.rheology = SolidRheologyParams(**RHEOLOGY)
        c.boundary_conditions.emissivity = 1.0
        c.mesh.number_of_nodes = a.nodes
        s = EntropySolver(c, entropy_eos=eos_np)
        s.initialize()
        if y_prev is not None:
            s.set_initial_lid(float(y_prev[-2]), float(y_prev[-1]))
        s.set_initial_entropy(a.s_init)
        _install_factory(s, a.eos_dir)
        if y_prev is not None:
            sc = s._build_nondim_scales()
            off = 0.0 if sc.state_offset is None else np.asarray(sc.state_offset)
            y_nd0, seg = (y_prev - off) / np.asarray(sc.state_scale), s._solve_cvode_segments

            def restart(_seg=seg, _t=t_prev, _y=y_nd0, **kw):
                kw.setdefault('max_segments', 50_000)
                return _seg(**{**kw, 'start_time': _t / kw['t_ref'], 'y0': _y})

            s._solve_cvode_segments = restart
        s.solve()
        tt, yy = np.asarray(s._solution.t, float), np.asarray(s._solution.y, float)
        if y_prev is not None and (
            abs(tt[0] - t_prev) > 1e-9 * t_prev
            or np.max(np.abs(yy[:, 0] - y_prev) / np.maximum(np.abs(y_prev), 1e-30)) > 1e-9
        ):
            raise RuntimeError(f'part {k} does not start from the end state of part {k - 1}')
        t_got, y_end = float(tt[-1]), yy[:, -1].copy()
        s.dSdt(t_got, y_end)
        lid = s.state._lid_state or {}
        rows.append(
            dict(
                part=k,
                t_yr=t_got,
                t_target=t_end,
                reached=bool(t_got >= t_end * (1 - 1e-9)),
                status=int(s._solution.status),
                delta=float(y_end[-2]),
                E=float(y_end[-1]),
                F_top=float(np.ravel(s.state._heat_flux)[-1]),
                T_i=float(np.asarray(lid['T_i'])) if 'T_i' in lid else None,
                nodes=a.nodes,
                s_init=a.s_init,
                dh_min=float(sl.LID_DH_MIN),
                sha=sha,
                tree=tree_id,
                eos_dir=str(a.eos_dir),
                eos_md5=md5,
                table_md5=tables,
                config_hash=config_hash,
            )
        )
        np.savez(a.outdir / f'{STEM}_part{k}.npz', t=tt, y=yy, y_end=y_end)
        out_json.write_text('[\n' + ',\n'.join(json.dumps(r) for r in rows) + '\n]\n')
        print(
            json.dumps({x: rows[-1][x] for x in ('part', 't_yr', 'status', 'delta')}),
            flush=True,
        )
        if not rows[-1]['reached'] or rows[-1]['status'] < 0:
            return 3
        t_prev, y_prev = t_got, y_end
    return 0


if __name__ == '__main__':
    sys.exit(main())
