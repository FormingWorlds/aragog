"""Per-call energy integrals and node checks, evaluated in bounded chunks while CVODE steps.

The quadrature nodes of a call are every accepted CVODE step, every output point and, on a
fired root, the root-search points below the root. They pass through three stages:

1. ``CallNodes`` filters the nodes of one CVODE call to increasing times up to the call end
   and brackets them by the start and end states.
2. ``SolveNodes`` joins the calls of one solve, keeps the last of each run of nodes that
   round to the same physical time, converts them to physical units and collects them in
   chunks of ``CHUNK`` nodes.
3. The consumers (``EnergyQuadrature``, ``TableGuard``, ``FloorMin``, ``RegimeCheck``) take
   one chunk at a time and keep only running sums, extremes and the last node.

So the memory of a call does not grow with its step count. The nodes, their order and the
arithmetic per node are those of a stored trace; only the trapezoid sums are added chunk by
chunk. A consumer stores its first error and raises it after the solve, never inside a CVODE
callback.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.constants import Julian_year as SECS_PER_YEAR

if TYPE_CHECKING:
    from aragog.solver.entropy_solver import EntropySolver

logger = logging.getLogger('fwl.aragog.solver.entropy_solver')

# Nodes per chunk: a multiple of the JAX batch size ``cvode_jax.C`` (512), so the batch
# powers of a chunk equal those of a whole trace.
CHUNK = 2048


class ListSink:
    """Collects nondimensional nodes; the trace of a CVODE call made outside ``solve``."""

    def __init__(self):
        self.t, self.y = [], []

    def add(self, t: float, y: npt.NDArray) -> None:
        self.t.append(t)
        self.y.append(y)

    def arrays(self) -> tuple[npt.NDArray, npt.NDArray] | None:
        """Node times and states (n_state, n), or None without nodes."""
        if not self.t:
            return None
        return np.array(self.t), np.array(self.y).T


class CallNodes:
    """Quadrature nodes of one CVODE call, passed to ``sink`` in increasing time.

    Nodes later than the call end and nodes not later than the previous kept node are
    dropped; the rest are bracketed by the start state (unless ``start`` is False, when the
    previous call's end state is the same node) and the end state. The call end is known only
    when the call returns, but a node can lie past it only within the last accepted step: so
    the nodes recorded before the latest running maximum of the time are final and are passed
    on, and the nodes from that maximum on wait for ``end``.
    """

    def __init__(self, sink, t0: float, y0: npt.NDArray, start: bool = True):
        self.sink = sink
        self.last = float(t0)
        self._t: list[float] = []
        self._y: list[npt.NDArray] = []
        self._t_max, self._i_max = -math.inf, 0
        self.count = 0  # nodes passed on
        self.recorded, self.latest = 0, None  # root-function states seen, the last of them
        if start:
            self._emit(self.last, np.array(y0, dtype=float).ravel())

    def _emit(self, t: float, y: npt.NDArray) -> None:
        self.last = t
        self.count += 1
        self.sink.add(t, y)

    def record(self, t: float, y: npt.ArrayLike) -> None:
        """Record the state the root function sees at time ``t``."""
        t = float(t)
        self.recorded += 1
        if t > self._t_max:
            self._t_max, self._i_max = t, len(self._t)
        self._t.append(t)
        self._y.append(np.array(y, dtype=float))
        self.latest = (t, self._y[-1])
        if len(self._t) >= CHUNK and self._i_max > 0:
            self._pass(self._i_max, math.inf)

    def _pass(self, n: int, t_end: float) -> None:
        for t, y in zip(self._t[:n], self._y[:n]):
            if self.last < t <= t_end:
                self._emit(t, y)
        del self._t[:n], self._y[:n]
        self._i_max -= n

    def end(self, t: npt.NDArray, y: npt.NDArray) -> None:
        """Close the call with its result times ``t`` and states ``y`` (n_state, n_t)."""
        t_end = float(t[-1])
        self._pass(len(self._t), t_end)
        if self.last < t_end:
            self._emit(t_end, np.array(y[:, -1], dtype=float))


class SolveNodes:
    """Joined nodes of one solve, in physical units, handed to the consumers in chunks.

    Of each run of nondimensional times that round to the same year the last node is kept;
    nodes at an equal nondimensional time (the two states of a lid hand-over) are all kept.
    Each state becomes ``y_nd * scale + offset``; the rows from ``change_from`` on are also
    passed as ``y_nd * scale``, their change from the call start. With a sub-grid lid the lid
    energy is set to 0 where the lid thickness is 0.

    Parameters
    ----------
    consumers : list
        Objects with ``add(t, y, change)``.
    t_ref : float
        Time scale [yr].
    scale, offset : ndarray
        State scale and offset per row.
    change_from : int
        First row of the changes.
    lid : bool
        The state ends with the lid thickness and energy.
    keep : bool
        Also keep every node (for tests); see ``kept``.
    """

    def __init__(self, consumers, t_ref, scale, offset, change_from, lid=False, keep=False):
        self.consumers = consumers
        self.t_ref = float(t_ref)
        self.scale = np.reshape(np.asarray(scale, dtype=float), (-1, 1))
        self.offset = np.reshape(np.asarray(offset, dtype=float), (-1, 1))
        self.change_from, self.lid, self.keep = change_from, lid, keep
        self._prev = None
        self._t: list[float] = []
        self._y: list[npt.NDArray] = []
        self._kept: list[tuple] = []
        self.n = 0

    def add(self, t_nd: float, y_nd: npt.NDArray) -> None:
        if self._prev is not None:
            tp = self._prev[0]
            if t_nd * self.t_ref > tp * self.t_ref or t_nd == tp:
                self._push(*self._prev)
        self._prev = (t_nd, y_nd)

    def _push(self, t_nd, y_nd) -> None:
        self._t.append(t_nd)
        self._y.append(y_nd)
        if len(self._t) == CHUNK:
            self._flush()

    def _flush(self) -> None:
        if not self._t:
            return
        t = np.array(self._t) * self.t_ref
        y_nd = np.array(self._y).T
        self._t.clear()
        self._y.clear()
        y = y_nd * self.scale
        change = y[self.change_from :].copy()
        scaled = y.copy() if self.keep else None
        y += self.offset
        if self.lid:
            y[-1] = np.where(y[-2] == 0.0, 0.0, y[-1])
        self.n += t.size
        if self.keep:
            self._kept.append((t, y, scaled))
        for c in self.consumers:
            c.add(t, y, change)

    def close(self) -> None:
        """Pass on the last node and the partial chunk."""
        if self._prev is not None:
            self._push(*self._prev)
            self._prev = None
        self._flush()

    def kept(self) -> tuple[tuple[npt.NDArray, npt.NDArray], npt.NDArray]:
        """The kept nodes ``(t, y)`` and their states times the scale, without the offset."""
        if not self._kept:
            w = self.scale.shape[0]
            return (np.empty(0), np.empty((w, 0))), np.empty((w, 0))
        t, y, c = zip(*self._kept)
        return (np.concatenate(t), np.concatenate(y, axis=1)), np.concatenate(c, axis=1)


def feed(consumers, t_pts, y_pts, change=None) -> None:
    """Pass a stored trace ``(t_pts, y_pts)`` (physical) to ``consumers`` chunk by chunk."""
    t_pts = np.atleast_1d(np.asarray(t_pts, dtype=float))
    y_pts = np.asarray(y_pts, dtype=float)
    if y_pts.ndim == 1:
        y_pts = y_pts.reshape(-1, 1)
    for a in range(0, t_pts.size, CHUNK):
        c = None if change is None else change[:, a : a + CHUNK]
        for consumer in consumers:
            consumer.add(t_pts[a : a + CHUNK], y_pts[:, a : a + CHUNK], c)


def _stag_entropies(s: EntropySolver, y: npt.NDArray) -> npt.NDArray:
    """Staggered entropies (n_stag, n) of the states ``y`` (n_state, n)."""
    return np.stack([s._stag_entropy(col) for col in y.T], axis=1)


class TableGuard:
    """First node with a cell entropy outside the EOS tables (``EntropySolver._eos_table_failure``)."""

    def __init__(self, solver: EntropySolver):
        self.s = solver
        self.t0 = None
        self.bad = None

    def add(self, t, y, change=None) -> None:
        if self.t0 is None and t.size:
            self.t0 = float(t[0])
        eos = self.s.entropy_eos
        if self.bad is not None or eos is None:
            return
        s_min, s_max = float(eos.S_min), float(eos.S_max)
        from aragog.solver.entropy_solver import _EOS_TABLE_TOL

        S = _stag_entropies(self.s, y)
        bad = ~((S >= s_min - _EOS_TABLE_TOL) & (S <= s_max + _EOS_TABLE_TOL))
        cols = np.flatnonzero(bad.any(axis=0))
        if cols.size:
            k = int(cols[0])
            node = int(np.flatnonzero(bad[:, k])[0])
            tk = float(t[k])
            where = 'at the call start, a shorter step does not help' if tk == self.t0 else ''
            self.bad = (
                tk,
                (
                    f'EOS table guard: cell {node} entropy {S[node, k]:.6g} J/kg/K outside the EOS '
                    f'tables [{s_min:.3f}, {s_max:.3f}] J/kg/K at t={tk:.6e} yr {where}'
                ).rstrip(),
            )


class FloorMin:
    """Smallest floor entropy and smallest cell entropy over the nodes (the EOS floor check)."""

    def __init__(self, solver: EntropySolver, weights: npt.NDArray):
        self.n_s, self.w = solver._n_stag, weights
        self.low = math.inf

    def add(self, t, y, change=None) -> None:
        if t.size:
            S = y[: self.n_s]
            self.low = min(self.low, float(np.min(self.w @ S)), float(S.min()))


class RegimeCheck:
    """``refuse_unmodelled_regime`` on the T_core of every node; keeps its first error."""

    def __init__(self, solver: EntropySolver, budget):
        self.row, self.budget = solver._n_stag + 1, budget
        self.error = None

    def add(self, t, y, change=None) -> None:
        if self.error is None and t.size:
            from aragog.core.regime import refuse_unmodelled_regime

            try:
                refuse_unmodelled_regime(self.budget, y[self.row])
            except Exception as exc:  # raised after the solve, never inside a CVODE callback
                self.error = exc


class EnergyQuadrature:
    """Per-call energy integrals [J] and the per-node invariant checks, chunk by chunk.

    The running form of ``EntropySolver._compute_step_energy_integrals``: per chunk the
    powers come from the JAX batch (or, without it or after it fails, from ``_step_powers``
    at each node), the trapezoid sums carry the last node of the previous chunk, and the core
    heat change and its tolerance scale are summed the same way. The checks keep the first
    failure, with the node index in the whole call.
    """

    def __init__(self, solver: EntropySolver):
        self.s = solver
        self.n = 0
        self.error: Exception | None = None
        self.batch_fn = None
        self.prev = None  # (t, powers, C_eff, d_core) of the last node
        self.sums = np.zeros(7)
        self.span_s = 0.0
        self.first = self.last = None  # (t, y, change) of the first and last node
        self.grown_max = -math.inf
        self.core_heat = 0.0
        self.core_tol = 0.0
        self.last_batch = self.used_batch = False
        self.skip = False

    def _start(self, t, y, change) -> None:
        s = self.s
        # without an entropy EOS (and no constant-property lid) the call books no energy
        self.skip = s.entropy_eos is None and not getattr(s, '_lid_const', False)
        self.rheo = s._solid_rheology()
        self.check_rheo = s._rheology_enabled
        self.batch_fn = s._cvode_step_powers_batch
        self.first = (float(t[0]), y[:, 0].copy(), None if change is None else change[:, 0])
        self.n_lid = bool(s._n_lid)
        if self.n_lid:
            from aragog import surface_lid as _lid

            self.cap = _lid.LID_CAP_FRAC * s._lid_h_top
            self.lid_lo = min(s._lid_delta_lo, float(y[-2, 0]))
        self.budget = getattr(s, '_core_module_budget', None)
        self.core_bc = getattr(s, '_core_bc', None)
        self.core = self.core_bc == 'core_module' and self.budget is not None

    def add(self, t, y, change=None) -> None:
        if not t.size:
            return
        if self.first is None:
            self._start(t, y, change)
        if self.skip:
            return
        n0 = self.n
        self.n += t.size
        self.last = (float(t[-1]), y[:, -1].copy(), None if change is None else change[:, -1])
        if self.n_lid:
            d = y[-2]
            grown = d[d > 0.0]
            if grown.size:
                self.grown_max = max(self.grown_max, float(grown.max()))
        if self.error is not None:
            return
        try:
            P = self._powers(t, y, n0)
            self._sum(t, y, change, P)
        except Exception as exc:  # raised after the solve, never inside a CVODE callback
            self.error = exc

    def _powers(self, t, y, n0) -> npt.NDArray:
        s = self.s
        counters = (s.state._pb_cache_hits, s.state._pb_cache_misses)
        try:
            P = None
            if self.batch_fn is not None:
                try:
                    P = s._step_powers_batch(t, y)
                    if P.shape != (t.size, 7):
                        raise ValueError(
                            f'Expected batch step powers shape ({t.size}, 7), got {P.shape}'
                        )
                    if not np.isfinite(P).all():
                        raise ValueError(
                            'Batch step powers returned non-finite values (NaN or Inf)'
                        )
                except Exception as exc:
                    s._warn_once(
                        'batch_powers_failed',
                        f'EntropySolver: batch step powers evaluation failed ({exc}); '
                        'falling back to numpy loop',
                    )
                    self.batch_fn = P = None
            self.last_batch = P is not None
            self.used_batch |= self.last_batch
            if P is not None:
                self._batch_checks(t, y, n0)
                return P
            rows = []
            for k in range(t.size):
                tk, yk = float(t[k]), y[:, k]
                if self.check_rheo:
                    s._check_rheology_eos_range(
                        s._stag_entropy(yk), step_idx=n0 + k, time_yr=tk
                    )
                rows.append(s._step_powers(tk, yk))
                if s._n_lid and s._lid_active:
                    s._check_lid_step(self.lid_lo, n0 + k, tk)
                if self.check_rheo:
                    s._check_rheology_step_invariants(self.rheo, step_idx=n0 + k, time_yr=tk)
            return np.array(rows)
        finally:
            s.state._pb_cache_hits, s.state._pb_cache_misses = counters

    def _batch_checks(self, t, y, n0) -> None:
        """The per-node checks of the batch path (the last-node checks run in ``result``)."""
        s = self.s
        if self.check_rheo:
            for k in range(t.size):
                s._check_rheology_eos_range(
                    s._stag_entropy(y[:, k]), step_idx=n0 + k, time_yr=float(t[k])
                )
        if not self.n_lid:
            return
        from aragog import surface_lid as _lid
        from aragog.solver.entropy_solver import _LID_CAP_SLACK

        d = y[-2]
        lo, hi = s._lid_S_range
        S_f = _stag_entropies(s, y)[-1]
        bad = np.flatnonzero((d != 0.0) & ~((S_f >= lo) & (S_f <= hi)))
        if bad.size:
            k = int(bad[0])
            raise RuntimeError(
                f'lid invariant violated at step {n0 + k}, t={t[k]:.6e} yr: entropy '
                f'S_f {S_f[k]:.6g} J/kg/K outside the lid heat table'
            )
        if s._lid_mush:  # the floor check of _check_lid_step at every node
            floor = s._lid_S_sol + _lid.LID_FLOOR_SLACK
            bad = np.flatnonzero((d != 0.0) & (S_f < floor))
            if bad.size:
                k = int(bad[0])
                raise RuntimeError(
                    f'lid invariant violated at step {n0 + k}, t={t[k]:.6e} yr: top '
                    f'cell entropy {S_f[k]:.6g} J/kg/K below the floor, solidus '
                    f'{s._lid_S_sol:.6g} J/kg/K'
                )
        cap, lo_bound = self.cap, s._lid_lo_bound(self.lid_lo)
        thin = (d != 0.0) & (d < lo_bound)
        bad = np.flatnonzero((d > cap * (1.0 + _LID_CAP_SLACK)) | thin)
        if bad.size:
            k = int(bad[0])
            raise RuntimeError(
                f'lid invariant violated at step {n0 + k}, t={t[k]:.6e} yr: thickness '
                f'{d[k]:.6g} m outside [{lo_bound:.6g}, '
                f'{cap * (1.0 + _LID_CAP_SLACK):.6g}] m (stops {self.lid_lo:.6g} and '
                f'{cap:.6g} m)'
            )

    def _sum(self, t, y, change, P) -> None:
        s = self.s
        c_eff = d_core = None
        if self.core:
            d_core = change[0] - self.first[2][0]
            c_eff = s._core_capacity(self.budget, y)
        if self.prev is not None:
            t = np.r_[self.prev[0], t]
            P = np.vstack([self.prev[1], P])
            if self.core:
                c_eff = np.r_[self.prev[2], c_eff]
                d_core = np.r_[self.prev[3], d_core]
        if t.size > 1:
            dt_s = np.diff(t) * SECS_PER_YEAR
            for j in range(7):
                self.sums[j] += float(np.sum(0.5 * (P[:-1, j] + P[1:, j]) * dt_s))
            self.span_s += float(np.sum(dt_s))
            if self.core:
                if self.budget.stratification:
                    self.core_heat += float(
                        np.sum(0.5 * (c_eff[:-1] + c_eff[1:]) * np.diff(d_core))
                    )
                self.core_tol += s._core_heat_tolerance(
                    np.r_[0.0, d_core[1:]], np.r_[0.0, c_eff[1:]]
                )
        self.prev = (t[-1], P[-1], *((c_eff[-1], d_core[-1]) if self.core else (None, None)))

    def result(self, sol) -> dict[str, float | None]:
        """The integrals of the call; raises the first failed check."""
        from aragog.solver.entropy_solver import _ZERO_ENERGY_INTEGRALS

        s = self.s
        if self.first is None or self.skip:
            return dict(_ZERO_ENERGY_INTEGRALS)
        if self.used_batch:
            s._batch_powers_calls += 1
        logger.info(
            'EntropySolver: evaluated energy powers via %s (%d nodes)',
            'JAX batch path' if self.used_batch else 'numpy loop fallback',
            self.n,
        )
        if self.n_lid:  # largest delta / cap - 1 over the accepted states with a lid
            sol.lid_cap_excess = (
                self.grown_max / self.cap - 1.0 if self.grown_max > -math.inf else math.nan
            )
        if self.error is not None:
            raise self.error
        t_last, y_last, _ = self.last
        counters = (s.state._pb_cache_hits, s.state._pb_cache_misses)
        try:
            if self.last_batch:
                s._dSdt_single(t_last, y_last)
                if s._n_lid and s._lid_active:
                    s._check_lid_step(self.lid_lo, self.n - 1, t_last)
                if self.check_rheo:
                    s._check_rheology_step_invariants(
                        self.rheo, step_idx=self.n - 1, time_yr=t_last
                    )
        finally:
            s.state._pb_cache_hits, s.state._pb_cache_misses = counters

        state_heat = s._step_heat_content(
            s._stag_entropy(sol.y[:, 0]), s._stag_entropy(sol.y[:, -1])
        )
        if s._n_lid:
            state_heat += s._lid_heat_change(sol.y[:, 0], sol.y[:, -1])
        step_dE_F_cmb = float(self.sums[1])
        denom = 4.0 * np.pi * float(s._r_basic_flat[0]) ** 2 * self.span_s
        f_cmb_step_avg = step_dE_F_cmb / denom if denom > 0.0 else None

        step_dE_core, step_core_tol = 0.0, float('nan')
        n_stag = getattr(s, '_n_stag', 0)
        t_first, y_first, c_first = self.first
        if self.core:
            d_last = self.last[2] - c_first
            if self.budget.stratification:
                shell = self.budget.shell
                step_dE_core = self.core_heat + float(
                    shell.heat_change(d_last[1 : 1 + shell.n_cells])
                )
            else:
                step_dE_core = self.budget.heat_change(float(y_first[n_stag + 1]), d_last[0])
            if t_last > t_first:  # a call that does not advance has no scale
                step_core_tol = self.core_tol
        elif self.core_bc == 'bower2018':
            C_core = getattr(s, '_core_cap', None)
            if C_core is None:
                p = s.parameters
                rho_c = float(p.mesh.core_density)
                cp_c = float(p.boundary_conditions.core_heat_capacity)
                vol_c = 4.0 / 3.0 * np.pi * float(s._r_basic_flat[0]) ** 3
                C_core = vol_c * rho_c * cp_c
            step_dE_core = float(C_core * (float(y_last[n_stag]) - float(y_first[n_stag])))

        return {
            'F_int': float(self.sums[0]),
            'F_cmb': step_dE_F_cmb,
            'F_cmb_step_avg': f_cmb_step_avg,
            'Q_radio': float(self.sums[2]),
            'Q_tidal': float(self.sums[3]),
            'Q_radio_cons': float(self.sums[4]),
            'Q_tidal_cons': float(self.sums[5]),
            'solver_residual': float(self.sums[6]),
            'state_heat': state_heat,
            'core': step_dE_core,
            'core_tol': step_core_tol,
        }
