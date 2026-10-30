"""Validation cases of the phase-boundary step controller (``phase_boundary_cap = 'rate'``).

Computes the V3-R3, V3-R4 and V4 rows of ``docs/Validation/numerics.md``:

===========  =====  ==========  ===============================  ================================
Case         Nodes  End [yr]    Setting                          Result
===========  =====  ==========  ===============================  ================================
v3_r3        12     1.5e4       F = 20 W/m^2, 2 cells above      max|T - T_ref|, round-off
                                the stiff zone                   envelope, segment counts
v3_r4        12     2e5         F = 200 W/m^2, start            as v3_r3, plus the entropy range
                                S_sol + 5 J/(kg K)               against the table
v4_rheo_off  100    1.2e8       abe_solid.toml, T_eq = 273 K,   stall point 107.575 Myr passed,
                                S0 = 3165.90 J/(kg K)            phi_max = 0, status 0
v4_rheo_on   100    1e9         as v4_rheo_off, stagnant-lid     stall point 83.684 Myr passed,
                                rheology, JAX CVODE factory      phi_max = 0, status 0
===========  =====  ==========  ===============================  ================================

The v3 cases use ``_build_mushy_parameters`` of ``tests/test_phi_step_cap_armed_smoke.py`` and
the CI test table with ``matprop_smooth_width = 0``; their fluxes are chosen for the table's
entropy range and they test the controller only. The solver floors atol at 1e-8 and multiplies
it by 10 when the mean start melt fraction is at most 0.01. Each case runs ``'fixed'`` at rtol
1e-10 with ``solver._atol_sf`` set for a physical atol of at most 1e-10 as the reference, and
``'rate'`` at rtol = atol = 1e-8 from S0 (1 + m 1e-12), m = -2..2. The v4 cases need
the PALEOS low-temperature table set (``paleos_lowt_spider_eos_1350x281``). They stop at the
first segment end with phi_max = 0 after the stall point, or at the end time of the case.
Run from the repository root::

    python tools/verification/run_step_cap_validation.py --case v3_r3 [--eos-dir DIR] [--workers N]
    python tools/verification/run_step_cap_validation.py --case v4_rheo_on --eos-dir PALEOS_DIR

``--eos-dir`` defaults to ``ARAGOG_TEST_EOS_DIR``. The result goes to ``--output``
(default ``step_cap_<case>.json`` in the working directory).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import aragog.solver.entropy_solver as es  # noqa: E402
from aragog.eos.entropy import EntropyEOS  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402

# max_steps: ceiling on the m = 0 rate steps (224 and 1031 measured; 272 and 2669 without the
# distance floor, 2081 with the 100 yr ceiling kept); inf_cap: a sub-solidus cap must reach inf.
V3 = {
    'v3_r3': dict(end=1.5e4, flux=20.0, width=0.0, max_steps=250, inf_cap=False),
    'v3_r4': dict(end=2.0e5, flux=200.0, width=0.0, max_steps=1500, inf_cap=True),
}
V4 = {
    'v4_rheo_off': dict(rheology=False, stall=107.575e6, max_steps=100_000, end=1.2e8),
    'v4_rheo_on': dict(rheology=True, stall=83.684e6, max_steps=800_000, end=1.0e9),
}
PALEOS_LOWT_MD5 = (
    '18d6f9eb5ba654d58fe3fad30ce118f0'  # temperature_solid.dat of the 1350 x 281 set
)
STARTS = (-2, -1, 0, 1, 2)
V3_BOUND_K = 1.3e-4  # largest |dT| [K] of a v3 rate run against its reference
SLOW_RATE = 0.05  # slow-landing rate [delta per year]


def regime(S, dSdt, S_liq, S_sol, mass, delta, rate_floor=True, **_):
    """Return (slow landing, sub-solidus) of one segment start of the rate cap.

    Slow landing: the cell that sets the step (shortest time from
    ``_rate_phase_boundary_times``) is within ``alpha`` delta of a boundary (``alpha`` of
    ``_PhaseBoundarySegmentRoot``, 0.5) and moves slower than ``SLOW_RATE`` delta per year
    (rate floored at the mass-weighted mean). Sub-solidus: no cell above the solidus.
    """
    alpha = es._PhaseBoundarySegmentRoot.alpha
    t_c = es._rate_phase_boundary_times(S, dSdt, S_liq, S_sol, mass, delta, rate_floor)
    slow = False
    if t_c is not None and np.isfinite(t_c).any():
        b = int(np.nanargmin(np.where(np.isfinite(t_c), t_c, np.nan)))
        dist = min(abs(S[b] - S_liq[b]), abs(S[b] - S_sol[b]))
        # t_c = alpha delta / floored rate here, so the rate test is t_c > alpha / SLOW_RATE
        slow = bool(dist < alpha * delta and t_c[b] > alpha / SLOW_RATE)
    return slow, es._subsolidus(S, S_sol)


@contextmanager
def regime_counter():
    """Count the segment starts of each regime while the block runs.

    An error of the counter is raised again at the end of the block: the solver catches
    errors of the step rule and runs on at 1 yr.
    """
    counts = {'segments': 0, 'slow_landing': 0, 'subsolidus': 0}
    original = es._rate_phase_boundary_max_step
    errors = []

    def counted(*args, **kwargs):
        try:
            slow, sub = regime(*args[:6], **kwargs)
        except Exception as exc:
            errors.append(exc)
            raise
        counts['segments'] += 1
        counts['slow_landing'] += slow
        counts['subsolidus'] += sub
        return original(*args, **kwargs)

    es._rate_phase_boundary_max_step = counted
    try:
        yield counts
    finally:
        es._rate_phase_boundary_max_step = original
    if errors:
        raise errors[0]


def _v3_solver(eos, case, mode, tol, width=None):
    from tests.test_phi_step_cap_armed_smoke import _build_mushy_parameters

    p = _build_mushy_parameters(solver_method='cvode', n_nodes=12, end_time=case['end'])
    p.energy = dataclasses.replace(
        p.energy, phase_boundary_cap=mode, phi_step_cap=None, convective_threshold=0.0
    )
    p.boundary_conditions = dataclasses.replace(
        p.boundary_conditions,
        outer_boundary_condition=4,
        outer_boundary_value=case['flux'],
        core_bc='quasi_steady',
    )
    p.solver.rtol = p.solver.atol = tol
    p.solver.max_steps = 100_000
    p.phase_mixed.matprop_smooth_width = case['width'] if width is None else width
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    return s


def v3_start(eos, name):
    """Return the unperturbed start entropy [J/(kg K)] of a v3 case."""
    S_sol = np.ravel(eos.solidus_entropy(_v3_solver(eos, V3[name], 'rate', 1e-8)._P_stag_flat))
    if name == 'v3_r4':
        return np.full_like(S_sol, S_sol[-1] + 5.0)
    S0 = S_sol - 15.0  # off the stiff-zone edge, so a round-off start keeps dist > delta
    S0[4], S0[5] = S_sol[4] + 11.0, S_sol[5] + 12.5
    return S0


def v3_run(task):
    """Integrate one v3 run; ``task`` is (eos_dir, case name, mode, tol, m, width)."""
    eos_dir, name, mode, tol, m, width = task
    eos = EntropyEOS(eos_dir)
    s = _v3_solver(eos, V3[name], mode, tol, width)
    s.set_initial_entropy(v3_start(eos, name) * (1.0 + m * 1e-12))
    if mode == 'fixed':
        s._atol_sf = tol / 1.0e-7  # atol_base 1e-8 times atol_scale up to 10
    t0 = time.perf_counter()
    with regime_counter() as counts:
        s.solve()
    sol = s._solution
    if sol.status != 0:
        raise RuntimeError(f'{name} {mode} at {tol:g}, m = {m}: solver status {sol.status}')
    S = np.asarray(sol.y)[: s._n_stag, :]
    T_end = np.asarray(eos.temperature(s._P_stag_flat, S[:, -1])).ravel()
    caps = [seg[2] for seg in getattr(sol, 'segments', None) or []]
    return dict(
        inf_caps=sum(not np.isfinite(h) for h in caps),
        mode=mode,
        tol=tol,
        m=m,
        steps=int(sol.get('cvode_nst') or 0),
        wall_s=time.perf_counter() - t0,
        T_end=T_end.tolist(),
        S_min=float(S.min()),
        S_max=float(S.max()),
        **counts,
    )


def run_v3(eos_dir, name, workers, width=None):
    """Run the reference and the 5 rate starts of a v3 case; return the result rows."""
    tasks = [(eos_dir, name, 'fixed', 1e-10, 0, width)] + [
        (eos_dir, name, 'rate', 1e-8, m, width) for m in STARTS
    ]
    rows = []
    pool = mp.get_context('spawn').Pool(min(workers, len(tasks))) if workers > 1 else None
    with pool or nullcontext():
        for row in pool.imap(v3_run, tasks) if pool else map(v3_run, tasks):
            print(
                f'{name} {row["mode"]} tol {row["tol"]:g} m {row["m"]:+d}: {row["steps"]} steps, '
                f'{row["wall_s"]:.1f} s, slow-landing/sub-solidus segments '
                f'{row["slow_landing"]}/{row["subsolidus"]}',
                flush=True,
            )
            rows.append(row)
    ref = next(r for r in rows if r['mode'] == 'fixed')
    rate = {r['m']: r for r in rows if r['mode'] == 'rate'}
    T0 = np.asarray(rate[0]['T_end'])
    table = EntropyEOS(eos_dir)._tables['temperature_solid']['S']
    metrics = dict(
        max_dT_rate_vs_ref_K=float(np.max(np.abs(T0 - np.asarray(ref['T_end'])))),
        round_off_envelope_K=max(
            float(np.max(np.abs(np.asarray(rate[m]['T_end']) - T0))) for m in STARTS if m
        ),
        steps=rate[0]['steps'],
        inf_cap_segments=rate[0]['inf_caps'],
        slow_landing_segments=rate[0]['slow_landing'],
        subsolidus_segments=rate[0]['subsolidus'],
        S_min=min(r['S_min'] for r in rows),
        S_max=max(r['S_max'] for r in rows),
        inside_table=bool(
            min(r['S_min'] for r in rows) >= table[0]
            and max(r['S_max'] for r in rows) <= table[-1]
        ),
    )
    return dict(rows=rows, metrics=metrics)


def _v4_factory(s, eos_dir):
    """Install the JAX CVODE factory on ``s`` the way PROTEUS installs it for Aragog."""
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    bc = s.parameters.boundary_conditions
    args = (
        EntropyEOS_JAX(str(eos_dir)),
        PhaseParams.from_parameters(s.parameters),
        MeshArrays.from_numpy_mesh(s.evaluator.mesh),
        BoundaryParams(
            outer_bc_type=bc.outer_boundary_condition,
            outer_bc_value=bc.outer_boundary_value,
            emissivity=bc.emissivity,
            T_eq=bc.equilibrium_temperature,
            inner_bc_type=5 if s._core_bc == 'energy_balance' else bc.inner_boundary_condition,
            inner_bc_value=bc.inner_boundary_value,
            core_density=s.parameters.mesh.core_density,
            core_heat_capacity=bc.core_heat_capacity,
            tfac_core_avg=getattr(bc, 'tfac_core_avg', 1.147),
            cmb_area=float(getattr(s, '_cmb_area', 0.0)),
            core_M=float(getattr(s, '_core_M', 0.0)),
            cmb_dr_cmb=float(getattr(s, '_cmb_dr_cmb', 0.0)),
            param_utbl=bool(getattr(bc, 'param_utbl', False)),
            param_utbl_const=float(getattr(bc, 'param_utbl_const', 0.0)),
        ),
        np.zeros(s._n_stag),
    )

    def factory(scales, core_bc_mode):
        rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
            *args,
            scales,
            core_bc_mode=core_bc_mode,
            radio_isotope_params=(),
            core_module_budget=getattr(s, '_core_module_budget', None),
            core_module_q_radio=float(getattr(s, '_core_module_q_radio', 0.0)),
            core_module_ra_crit_cmb=getattr(s, '_core_module_ra_crit_cmb', None),
        )
        return rhs_fn, jac_fn

    s.set_jax_cvode_factory(factory)


def run_v4(eos_dir, name, end_time=None):
    """Integrate a v4 case to phi_max = 0 past its stall point; return the result."""
    from aragog.parser import Parameters

    md5 = hashlib.md5((Path(eos_dir) / 'temperature_solid.dat').read_bytes()).hexdigest()
    if md5 != PALEOS_LOWT_MD5:
        raise SystemExit(f'{name} needs the PALEOS low-T table set; {eos_dir} has md5 {md5}')
    case = V4[name]
    eos = EntropyEOS(str(eos_dir))
    p0 = Parameters.from_file(ROOT / 'src/aragog/cfg/abe_solid.toml')
    p = dataclasses.replace(
        p0,
        energy=dataclasses.replace(
            p0.energy,
            use_jax_jacobian=True,
            eos_floor_margin=0.0,
            radionuclides=False,
            tidal=False,
        ),
        boundary_conditions=dataclasses.replace(
            p0.boundary_conditions, outer_boundary_condition=1, equilibrium_temperature=273.0
        ),
        solver=dataclasses.replace(
            p0.solver,
            start_time=0.0,
            end_time=case['end'] if end_time is None else end_time,
            max_steps=case['max_steps'],
        ),
    )
    if case['rheology']:
        from aragog.parser import SolidRheologyParams  # with the solid rheology solver wiring

        rheo = SolidRheologyParams(
            enabled=True, s_top_inf=0.25, c_theta=0.25, stress_closure_mode='lid'
        )
        p = dataclasses.replace(
            p, phase_solid=dataclasses.replace(p.phase_solid, rheology=rheo)
        )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3165.90)
    if case['rheology']:
        _v4_factory(s, eos_dir)
    scales = s._build_nondim_scales()
    state = dict(segments=0, steps=0, t_yr=0.0, phi_max=1.0, S_min=np.inf, S_max=-np.inf)
    solve_cvode, t0 = s._solve_cvode, time.perf_counter()

    def monitored(*a, **k):
        res = solve_cvode(*a, **k)
        S = (np.asarray(res.y[:, -1]) * scales.state_scale + scales.state_offset)[: s._n_stag]
        state.update(
            segments=state['segments'] + 1,
            steps=state['steps'] + int(res.get('cvode_nst', 0)),
            t_yr=float(res.t[-1]) * scales.t_ref,
            phi_max=float(np.max(eos.melt_fraction(s._P_stag_flat, S))),
            S_min=min(state['S_min'], float(S.min())),
            S_max=max(state['S_max'], float(S.max())),
        )
        print(
            f'{name} segment {state["segments"]}: t {state["t_yr"]:.6e} yr, phi_max '
            f'{state["phi_max"]:.6f}, {state["steps"]} steps, {time.perf_counter() - t0:.1f} s',
            flush=True,
        )
        return stop_after_stall(res, state, case['stall'])

    s._solve_cvode = monitored
    s.solve()
    sol = s._solution
    return dict(
        status=int(sol.status),
        wall_s=time.perf_counter() - t0,
        stall_point_yr=case['stall'],
        passed_stall_point=state['t_yr'] >= case['stall'],
        solidified=state['phi_max'] == 0.0,
        inside_table=bool(state['S_min'] >= eos.S_min and state['S_max'] <= eos.S_max),
        **state,
    )


def stop_after_stall(res, state, stall):
    """End the segment loop once a solid state passes the stall point; keep a failed status."""
    if res.status == 0 and state['phi_max'] == 0.0 and state['t_yr'] >= stall:
        res.cvode_flag = 0
    return res


def check(case, result):
    """Return the failed tolerances of the docs page for one case result."""
    if case in V3:
        m, c = result['metrics'], V3[case]
        fails = [] if m['max_dT_rate_vs_ref_K'] <= V3_BOUND_K else [f'|dT| > {V3_BOUND_K} K']
        fails += [f'steps > {c["max_steps"]}'] if m['steps'] > c['max_steps'] else []
        fails += ['no inf cap'] if c['inf_cap'] and not m['inf_cap_segments'] else []
    else:
        m = result
        fails = [] if m['status'] == 0 else [f'status {m["status"]}']
        fails += [k for k in ('passed_stall_point', 'solidified') if not m[k]]
    return fails + ([] if m['inside_table'] else ['inside_table'])


def main(argv=None) -> int:
    """Run one case, write its JSON result and print the result line."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--case', required=True, choices=[*V3, *V4])
    ap.add_argument('--eos-dir', default=os.environ.get('ARAGOG_TEST_EOS_DIR'))
    ap.add_argument('--workers', type=int, default=1, help='processes for the 6 v3 runs')
    ap.add_argument('--width', type=float, default=None, help='replace the v3 width 0')
    ap.add_argument('--end-time', type=float, default=None, help='replace the v4 end time [yr]')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--check', action='store_true', help='exit 1 when a tolerance fails')
    args = ap.parse_args(argv)
    if not args.eos_dir:
        ap.error('--eos-dir or ARAGOG_TEST_EOS_DIR is required')
    os.environ['ARAGOG_TEST_EOS_DIR'] = str(args.eos_dir)
    if args.case in V3:
        result = run_v3(str(args.eos_dir), args.case, args.workers, args.width)
        print(json.dumps(result['metrics']))
    else:
        result = run_v4(args.eos_dir, args.case, args.end_time)
        print(json.dumps(result))
    out = args.output or Path(f'step_cap_{args.case}.json')
    out.write_text(json.dumps(dict(case=args.case, eos_dir=str(args.eos_dir), **result)) + '\n')
    fails = check(args.case, result)
    print(f'{args.case}: ' + ('FAIL ' + ', '.join(fails) if fails else 'PASS'))
    return 1 if args.check and fails else 0


if __name__ == '__main__':
    sys.exit(main())
