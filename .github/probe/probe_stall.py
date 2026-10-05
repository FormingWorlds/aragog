"""Probe the fixed/energy_balance _isentropic_end solve: wall time, RHS trace, CVODE stats.

Usage: python probe_stall.py [--perturb REL] [--seed K] [--limit SEC] [--out DIR] [--mode fixed]
"""

import argparse
import json
import logging
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.environ.get('ARAGOG_TESTS_PARENT', '.'))
from tests.test_phase_boundary_cap import _solver  # noqa: E402
from tests.test_phi_step_cap_armed_smoke import _pick_mushy_S  # noqa: E402,F401

ap = argparse.ArgumentParser()
ap.add_argument('--perturb', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=float, default=300.0)
ap.add_argument('--out', default='/tmp/probe/out')
ap.add_argument('--mode', default='fixed')
ap.add_argument('--tol', type=float, default=1e-6)
ap.add_argument('--core', default='energy_balance')
ap.add_argument('--variant', default='')
ap.add_argument('--jax', action='store_true')
ap.add_argument('--S', type=float, default=8182.3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = f'{a.mode}_{a.core}_S{a.S:g}_p{a.perturb:g}_s{a.seed}{a.variant}{"_jax" if a.jax else ""}'

logging.basicConfig(
    filename=f'{a.out}/{tag}.log', level=logging.INFO, format='%(asctime)s %(name)s %(message)s'
)

from aragog.eos.entropy import EntropyEOS  # noqa: E402
from tests.conftest import EOS_DIR  # noqa: E402

eos = EntropyEOS(EOS_DIR)
import aragog.solver.entropy_state as _est
if a.variant.startswith('eps'):
    _eps = float(a.variant[3:])
    _sab = _est._smooth_abs_neg
    _est._smooth_abs_neg = lambda x, eps=0.0: _sab(x, eps=_eps)
s = _solver(eos, a.mode, a.core, n_nodes=24, end_time=200.0, S=a.S, tol=a.tol)
import dataclasses
if a.jax:
    s.parameters.energy = dataclasses.replace(s.parameters.energy, use_jax_jacobian=True)
if a.perturb:
    rng = np.random.default_rng(a.seed)
    S0 = np.asarray(s._S0, float).copy()
    S0[: s._n_stag] *= 1.0 + a.perturb * rng.standard_normal(s._n_stag)
    s._S0 = S0

full = []
trace = []  # (wall, t_phys, S_min, S_max, extra)
real = s._dSdt_single
t0w = time.time()


def rec(t, y):
    out = real(t, y)
    n = s._n_stag
    trace.append((time.time() - t0w, float(t), float(y[:n].min()), float(y[:n].max()),
                  float(y[n]) if y.size > n else np.nan))
    full.append(np.concatenate(([t], y, out)))
    return out


s._dSdt_single = rec
import aragog.solver.entropy_solver as _es
_orig_cv = _es._scikits_cvode
steps = []


def _cv(rhs, **opts):
    rf = opts['rootfn']

    def rf2(t, y, g):
        steps.append(np.concatenate(([t], y)))
        return rf(t, y, g)

    opts['rootfn'] = rf2
    if a.variant == 'order5':
        opts['order'] = 5
    if a.variant in ('jacacc', 'jacpred'):
        n = len(s._S0)
        atol = np.asarray(opts['atol'], float) * np.ones(n)
        rtol = opts['rtol']
        fbuf = np.empty(n)

        def jac(t, y, fy, J):
            if a.variant == 'jacacc' and len(steps) > 1:
                t0, y0 = steps[-1][0], steps[-1][1:].copy()
            else:
                t0, y0 = t, np.array(y, float)
            f0 = np.empty(n)
            rhs(t0, y0, f0)
            for j in range(n):
                inc = 1.4901161193847656e-08 * max(abs(y0[j]), rtol * abs(y0[j]) + atol[j], 1e-30)
                yj = y0.copy(); yj[j] += inc
                rhs(t0, yj, fbuf)
                J[:, j] = (fbuf - f0) / inc
            return 0

        opts['jacfn'] = jac
        opts['linsolver'] = 'dense'
        opts.pop('lband', None); opts.pop('uband', None)
    if a.variant == 'nl3':
        opts['max_nonlin_iters'] = 3
    if a.variant == 'nonlin3':
        opts['max_nonlin_iters'] = 3
        opts['max_conv_fails'] = 10
    steps.append(np.concatenate(([np.nan], np.asarray(opts['atol'], float).ravel() * np.ones(len(s._S0)), [opts['rtol']])[: len(s._S0) + 1]))
    return _orig_cv(rhs, **opts)


_es._scikits_cvode = _cv


def dump(final):
    tr = np.array(trace) if trace else np.zeros((0, 5))
    np.save(f'{a.out}/{tag}_trace.npy', tr)
    if os.environ.get('PROBE_FULL'):
        np.save(f'{a.out}/{tag}_full.npy', np.array(full))
    np.save(f'{a.out}/{tag}_steps.npy', np.array(steps[1:]))
    np.save(f'{a.out}/{tag}_atol.npy', steps[0])
    summ = dict(tag=tag, final=final, wall=time.time() - t0w, nrhs=len(trace),
                t_last=float(tr[-1, 1]) if len(tr) else None)
    if len(tr) > 50:
        last = tr[-2000:]
        summ['t_span_last2000'] = [float(last[:, 1].min()), float(last[:, 1].max())]
        summ['state_last'] = list(map(float, tr[-1, 2:]))
    with open(f'{a.out}/{tag}_summary.json', 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ), flush=True)


def watchdog():
    dump(final=False)
    os._exit(3)


timer = threading.Timer(a.limit, watchdog)
timer.daemon = True
timer.start()
s.solve()
timer.cancel()
dump(final=True)
