"""The EOS domain check of a call: lookup points outside the valid window of the tables.

The windows here are set on the CI tables, so each expected distance follows from the window and
the state. The check warns by default, raises in 'raise' mode and never changes a result.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult

from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos, returns
from .test_eos_floor import CVODE_BDF
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S

pytestmark = [pytest.mark.smoke, needs_eos]

_MSG = 'EOS domain check'


def _flat_window(P_ax, lo, hi, dS):
    """A window with the same ends in every pressure column."""
    from aragog.eos.domain import PhaseWindow

    n = np.size(P_ax)
    return PhaseWindow(P=np.asarray(P_ax, float), lo=np.full(n, lo), hi=np.full(n, hi), dS=dS)


def _solver(eos, *, method='bdf', core='quasi_steady', mode=None, end_time=5.0):
    p = _build_mushy_parameters(solver_method=method, end_time=end_time)
    p.boundary_conditions.core_bc = core
    if mode is not None:
        p.solver.eos_domain_check = mode
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(eos))
    return s


def _two_states(cells):
    return OptimizeResult(t=np.array([0.0, 1.0]), y=np.column_stack([cells, cells]), status=0)


def _tight(eos):
    """Both windows only the 2 lowest S nodes of the tables: every state of a run is outside."""
    from aragog.eos.domain import TableDomain

    S_ax = eos._tables['temperature_melt']['S']
    w = _flat_window(
        eos._tables['temperature_melt']['P'], S_ax[0], S_ax[1], np.diff(S_ax).mean()
    )
    return TableDomain(melt=w, solid=w)


def _messages(caplog):
    return [r.getMessage() for r in caplog.records if _MSG in r.getMessage()]


def _masked_dir(dst, ref, rows):
    """Links to the CI tables in ``dst`` with masks valid on the S rows ``rows`` of each column."""
    for f in Path(EOS_DIR).glob('[!.]*.dat'):  # the CI archive holds ._*.dat metadata files
        if not f.name.startswith('valid_mask_'):  # the masks are written below, not linked
            (dst / f.name).symlink_to(f)
    for ph in ('melt', 'solid'):
        tbl = ref._tables[f'temperature_{ph}']
        mask = np.zeros((tbl['S'].size, tbl['P'].size), dtype=int)
        mask[rows] = 1
        np.savetxt(dst / f'valid_mask_{ph}.dat', mask, fmt='%d', header=f'{ph} mask')
    return dst


def test_mask_files_next_to_the_tables_make_a_call_warn(tmp_path, caplog):
    """Tables whose masks hold only the 2 lowest S rows: a call from a mushy start warns once,
    naming a node, its distance outside and the window."""
    eos = EntropyEOS(_masked_dir(tmp_path, entropy_eos_copy(EOS_DIR), slice(0, 2)))
    s = _solver(eos)
    with caplog.at_level(logging.WARNING, logger='fwl.aragog'):
        s.solve()
    msgs = _messages(caplog)
    assert s._solution.status == 0 and len(msgs) == 1
    assert ' node ' in msgs[0] and 'cells outside' in msgs[0] and 'window [' in msgs[0]


@pytest.mark.parametrize('core', ['quasi_steady', 'gradient'])
@pytest.mark.parametrize('method', CVODE_BDF)
def test_warn_names_the_worst_point_once_per_call_and_changes_nothing(caplog, method, core):
    """A call with every point outside writes 1 warning that names the node, its distance and
    its window; the result is identical, bit for bit, to the same call with the check off."""
    runs = {}
    for mode in ('off', 'warn'):
        eos = entropy_eos_copy(EOS_DIR)
        eos.domain = _tight(eos)
        s = _solver(eos, method=method, core=core, mode=mode)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='fwl.aragog'):
            s.solve()
        runs[mode] = (s._solution, _messages(caplog))
    (off, off_msgs), (warn, warn_msgs) = runs['off'], runs['warn']
    assert off_msgs == [] and len(warn_msgs) == 1
    assert warn.status == off.status == 0 and warn.t[-1] > warn.t[0]
    np.testing.assert_array_equal(warn.t, off.t)
    np.testing.assert_array_equal(warn.y, off.y)
    np.testing.assert_equal(warn.energy_integrals, off.energy_integrals)
    msg = warn_msgs[0]
    for part in ('node ', 'cells outside', 'window [', 'more than 5 table S cells', ' K'):
        assert part in msg


def test_raise_stops_the_call_and_holds_the_solution_of_the_call_start():
    """In 'raise' mode the call ends with EOSDomainError and the solver holds its earlier result."""
    from aragog.eos.domain import EOSDomainError

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos, mode='raise')
    s.solve()  # no domain: the check is off
    first = s._solution
    assert first.status == 0
    eos.domain = _tight(eos)
    with pytest.raises(EOSDomainError, match='cells outside the valid window'):
        s.solve()
    assert s._solution is first


def test_the_default_mode_warns():
    """A solver built from the default parameters warns."""
    eos = entropy_eos_copy(EOS_DIR)
    assert _solver(eos).parameters.solver.eos_domain_check == 'warn'


def test_only_the_top_basic_node_is_named_when_it_alone_leaves_the_window():
    """Uniform melt cells with a steep top cell: the linear top node lies beyond the melt window,
    every other point inside. The message names basic node n_stag with its distance."""
    from aragog.eos.domain import TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    P_ax, S_ax = eos._tables['temperature_melt']['P'], eos._tables['temperature_melt']['S']
    dS = float(np.diff(S_ax).mean())
    s_top = float(S_ax[-1]) - 20.0 * dS
    cells = np.full(s._n_stag, s_top - 30.0 * dS)
    cells[-1] = s_top
    assert cells.min() > np.max(eos.liquidus_entropy(s._P_basic_flat))  # all melt
    eos.domain = TableDomain(
        melt=_flat_window(P_ax, float(S_ax[0]), s_top, dS),
        solid=_flat_window(P_ax, float(S_ax[0]), float(S_ax[-1]), dS),
    )
    S_b = np.ravel(s.evaluator.mesh.entropy_at_basic_nodes(cells))
    d_top = (S_b[-1] - s_top) / dS
    assert d_top > 5.0 and np.all(S_b[:-1] <= s_top)
    msg = s._eos_domain_failure(_two_states(cells))
    assert msg.startswith(f'{_MSG}: 2 lookup points ')
    assert f'basic node {s._n_stag} ' in msg and f'{d_top:.2f} cells outside' in msg
    assert f'S={S_b[-1]:.6f} J/kg/K, read on the melt table' in msg
    assert f'window [{float(S_ax[0]):.6f}, {s_top:.6f}]' in msg


def _uniform_melt(s, eos, outside_cells, dS=100.0):
    """Melt cells at one entropy, the melt window ending ``outside_cells`` cells below it."""
    from aragog.eos.domain import TableDomain

    S_ax = eos._tables['temperature_melt']['S']
    s_c = float(np.floor(S_ax[-1] - 1000.0))
    assert s_c > np.max(eos.liquidus_entropy(s._P_basic_flat))
    w = _flat_window([0.0, 1.0e13], 0.0, s_c - outside_cells * dS, dS)
    eos.domain = TableDomain(melt=w, solid=w)
    return np.full(s._n_stag, s_c)


def test_a_point_at_the_tolerance_does_not_count():
    """At exactly 5 cells a point is inside the tolerance; just past it, every point counts."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    assert s._eos_domain_failure(_two_states(_uniform_melt(s, eos, 5.0))) is None
    msg = s._eos_domain_failure(_two_states(_uniform_melt(s, eos, 5.01)))
    assert f'{2 * (2 * s._n_stag + 1)} lookup points' in msg and '5.01 cells outside' in msg


@settings(max_examples=40, deadline=None)
@given(
    outside=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    tol=st.floats(min_value=0.5, max_value=15.0, allow_nan=False),
)
def test_the_check_fires_exactly_past_the_tolerance(outside, tol):
    """With every point ``outside`` cells out, the check fires when outside > tol, else not."""
    assume(abs(outside - tol) > 1.0e-6)
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    s.parameters.solver.eos_domain_tolerance_cells = tol
    msg = s._eos_domain_failure(_two_states(_uniform_melt(s, eos, outside)))
    assert (msg is not None) == (outside > tol)


def test_a_non_finite_cell_counts_as_outside():
    """A NaN cell entropy is outside at any tolerance and is named as not a finite point."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    cells = _uniform_melt(s, eos, 0.0)
    cells[3] = np.nan
    msg = s._eos_domain_failure(_two_states(cells))
    assert msg is not None
    assert (
        'staggered node 3 ' in msg and 'not a finite point' in msg and msg.endswith('T=nan K')
    )


@pytest.mark.parametrize('switch', ['mode off', 'no masks', 'constant properties'])
def test_the_check_is_off_without_masks_in_off_mode_and_with_constant_properties(switch):
    """With every point outside, the check returns None in each of its 3 off cases."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    cells = _uniform_melt(s, eos, 50.0)
    assert s._eos_domain_failure(_two_states(cells)) is not None
    if switch == 'mode off':
        s.parameters.solver.eos_domain_check = 'off'
    elif switch == 'no masks':
        eos.domain = None
    else:
        s.parameters.phase_mixed.const_properties = True
    assert returns(s._eos_domain_failure, _two_states(cells)) is None


def test_the_eos_reads_the_masks_next_to_its_tables(tmp_path, caplog):
    """An EOS directory with mask files gives the windows of the masks; the CI tables, which
    have no masks, give no domain and 1 info line."""
    ref = entropy_eos_copy(EOS_DIR)
    dom = EntropyEOS(_masked_dir(tmp_path, ref, slice(10, 50))).domain
    melt, solid = ref._tables['temperature_melt'], ref._tables['temperature_solid']
    np.testing.assert_array_equal(dom.melt.lo, np.full(melt['P'].size, melt['S'][10]))
    np.testing.assert_array_equal(dom.solid.hi, np.full(solid['P'].size, solid['S'][49]))
    with caplog.at_level(logging.INFO, logger='fwl.aragog'):
        assert EntropyEOS(EOS_DIR).domain is None
    assert sum('valid_mask' in r.getMessage() for r in caplog.records) == 1


def _rhs_basic(s, y):
    """Basic-node entropy that the right-hand side gives the basic phase evaluator for ``y``."""
    s._dSdt_single(0.0, np.asarray(y, dtype=float))
    return np.array(s.state._entropy_basic, dtype=float)


@pytest.mark.parametrize('core', ['energy_balance', 'core_module'])
def test_the_cmb_node_is_judged_at_the_entropy_of_the_state_gradient(core):
    """With the CMB gradient in the state, the bottom basic node is the bottom cell carried by
    that gradient, not the mesh extrapolation: a steep gradient takes it out of the window, and
    the check names it at the entropy the right-hand side reads."""
    from aragog.eos.domain import TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos, core=core)
    cells = _uniform_melt(s, eos, 0.0)
    w = eos.domain.melt
    y = np.array(s._S0, dtype=float)
    y[: s._n_stag] = cells
    y[s._n_stag] = 0.0
    assert s._eos_domain_failure(_states(y, y)) is None
    r_b = s._r_basic_flat
    y[s._n_stag] = 40.0 * w.dS / (r_b[0] - 0.5 * (r_b[0] + r_b[1]))  # 40 cells above the cells
    S_b = _rhs_basic(s, y)
    assert S_b[0] == pytest.approx(cells[0] + 40.0 * w.dS) and np.all(S_b[1:] == cells[0])
    eos.domain = TableDomain(melt=w, solid=w)
    msg = s._eos_domain_failure(_states(y, y))
    assert 'basic node 0 ' in msg and f'S={S_b[0]:.6f} J/kg/K' in msg
    assert '40.00 cells outside' in msg and msg.startswith(
        f'{_MSG}: 2 lookup points over the 2 '
    )


def test_a_steep_bottom_with_no_state_gradient_is_not_reported():
    """energy_balance with a zero CMB gradient: the mesh extrapolation of 2 steep bottom cells
    lies more than 5 cells outside, but the solver reads the bottom cell itself there, inside."""
    from aragog.eos.domain import TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos, core='energy_balance')
    cells = _uniform_melt(s, eos, 0.0)
    w = eos.domain.melt
    cells[1:] = cells[0] - 20.0 * w.dS  # the bottom cell 20 cells above the next one
    assert cells.min() > np.max(eos.liquidus_entropy(s._P_basic_flat))  # all melt
    extrap = np.ravel(s.evaluator.mesh.entropy_at_basic_nodes(cells))[0]
    hi = cells[0] + 1.0 * w.dS
    assert (extrap - hi) / w.dS > 5.0  # the mesh node would be reported
    win = _flat_window(w.P, 0.0, hi, w.dS)
    eos.domain = TableDomain(melt=win, solid=win)
    y = np.append(cells, 0.0)
    assert _rhs_basic(s, y)[0] == cells[0]
    assert s._eos_domain_failure(_states(y, y)) is None


@pytest.mark.parametrize(
    'core, bc',
    [
        ('quasi_steady', 4),
        ('energy_balance', 4),
        ('core_module', 4),
        ('gradient', 4),
        ('bower2018', 4),
        ('quasi_steady', 5),
        ('energy_balance', 5),
    ],
)
def test_the_check_reads_the_basic_entropy_of_the_right_hand_side(core, bc, monkeypatch):
    """For each core layout and both a flux and a set-temperature surface, the basic-node
    entropy that the check judges equals the one the right-hand side gives the EOS."""
    from aragog.eos import domain as dom_mod

    eos = entropy_eos_copy(EOS_DIR)
    p = _build_mushy_parameters(solver_method='bdf', end_time=5.0)
    p.boundary_conditions.core_bc = core
    p.boundary_conditions.outer_boundary_condition = bc
    p.boundary_conditions.outer_boundary_value = 2000.0 if bc == 5 else 1.0e5
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(eos))
    eos.domain = _tight(eos)
    y = np.array(s._S0, dtype=float)
    if core in ('energy_balance', 'core_module'):
        y[s._n_stag] = 3.0e-3  # a CMB gradient that moves the bottom node
    seen = []
    real = dom_mod.TableDomain.lookup_distances

    def record(self, eos_, P, S, **kw):
        seen.append(np.array(S, dtype=float))
        return real(self, eos_, P, S, **kw)

    monkeypatch.setattr(dom_mod.TableDomain, 'lookup_distances', record)
    assert returns(s._eos_domain_failure, _states(y)) is not None
    np.testing.assert_array_equal(seen[1], [_rhs_basic(s, y)])  # one row per state
    np.testing.assert_array_equal(seen[0], [s._stag_entropy(y)])
    if bc == 5:
        assert seen[1][0, -1] == s.state._surface_temperature_bc[0]


def _states(*ys, t=None):
    return OptimizeResult(
        t=np.arange(len(ys), dtype=float) if t is None else np.asarray(t, float),
        y=np.column_stack(ys),
        status=0,
    )


def test_the_window_is_read_at_the_pressure_of_each_node():
    """A window whose upper end falls with pressure: uniform melt cells are outside only at the
    deep nodes, each by the distance of the window end at its own pressure."""
    from aragog.eos.domain import PhaseWindow, TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    cells = _uniform_melt(s, eos, 0.0)
    dS = eos.domain.melt.dS
    P_s, P_b = s._P_stag_flat, s._P_basic_flat
    P_lo, P_hi = float(min(P_s.min(), P_b.min())), float(max(P_s.max(), P_b.max()))
    hi_end = np.array([cells[0] + 4.0 * dS, cells[0] - 20.0 * dS])  # at P_lo and P_hi
    w = PhaseWindow(P=np.array([P_lo, P_hi]), lo=np.zeros(2), hi=hi_end, dS=dS)
    eos.domain = TableDomain(melt=w, solid=w)
    msg = s._eos_domain_failure(_two_states(cells))
    d_s = (cells - np.interp(P_s, w.P, w.hi)) / dS
    d_b = (cells[0] - np.interp(P_b, w.P, w.hi)) / dS
    n = int((d_s > 5.0).sum() + (d_b > 5.0).sum())
    assert 0 < n < 2 * (P_s.size + P_b.size)  # some nodes inside, some outside
    k = int(np.argmax(d_b)) if d_b.max() > d_s.max() else int(np.argmax(d_s))
    kind = 'basic' if d_b.max() > d_s.max() else 'staggered'
    assert f'{2 * n} lookup points' in msg and f'{kind} node {k} ' in msg
    assert f'{max(d_b.max(), d_s.max()):.2f} cells outside' in msg


def test_the_worst_state_of_the_call_is_named_with_its_time():
    """3 output states, the middle one farthest out: the message names its time, and the count
    sums the points outside over the states."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    a = _uniform_melt(s, eos, 0.0)
    dS = eos.domain.melt.dS
    msg = s._eos_domain_failure(_states(a + 6.0 * dS, a + 9.0 * dS, a, t=[1.0, 2.5, 4.0]))
    n_pts = 2 * s._n_stag + 1
    assert f'{2 * n_pts} lookup points over the 3 output states' in msg
    assert 'worst at t=2.500000e+00 yr' in msg and '9.00 cells outside' in msg
    point = msg.split(', read on')[0]  # the point itself, before its table lookup
    assert f'S={a[0] + 9.0 * dS:.6f} J/kg/K' in point


def test_the_message_temperature_is_the_table_value_and_reads_no_range_warning():
    """T in the message is the temperature the melt table returns at the lookup, clipped to the
    table, and the check leaves the EOS range-warning counts unchanged."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    cells = _uniform_melt(s, eos, 50.0)
    S_max = float(eos._tables['temperature_melt']['S'][-1])
    cells[-1] = S_max + 500.0  # beyond the table: a range warning if read through temperature()
    before = dict(eos._range_warning_counts)
    msg = s._eos_domain_failure(_two_states(cells))
    assert eos._range_warning_counts == before
    k = int(msg.split(' node ')[1].split(' ')[0])
    kind = msg.split('worst at ')[1].split(': ')[1].split(' node')[0]
    P = (s._P_basic_flat if kind == 'basic' else s._P_stag_flat)[k]
    S = float(msg.split('table at S=')[1].split(' ')[0])
    T = eos.domain.lookup_temperature(eos, P, S, True)
    assert msg.endswith(f'T={T:.6f} K')
    tbl = eos._tables['temperature_melt']
    P_c = np.clip(P, tbl['P'][0], tbl['P'][-1])
    assert S > S_max and T == pytest.approx(float(tbl['interp']([[P_c, S_max]])[0]), rel=1e-12)


def test_a_failed_call_is_not_judged(monkeypatch, caplog):
    """A call that the table guard fails (status -1) writes no domain warning, even with every
    point outside."""
    eos = entropy_eos_copy(EOS_DIR)
    eos.domain = _tight(eos)
    s = _solver(eos)
    monkeypatch.setattr(EntropySolver, '_eos_table_failure', lambda self, sol: (sol.t[-1], 'x'))
    with caplog.at_level(logging.WARNING, logger='fwl.aragog'):
        s.solve()
    assert s._solution.status == -1 and _messages(caplog) == []


def test_raise_on_the_first_call_holds_no_solution():
    """'raise' on the first call: there is no earlier result, so the solver holds none."""
    from aragog.eos.domain import EOSDomainError

    eos = entropy_eos_copy(EOS_DIR)
    eos.domain = _tight(eos)
    s = _solver(eos, mode='raise')
    with pytest.raises(EOSDomainError):
        s.solve()
    assert s._solution is None


def test_raise_after_a_floor_re_entry_holds_the_solution_of_the_outer_call_start(monkeypatch):
    """A floor discard solves the call again; when that solve raises, the solver holds the
    result of the outer call start, not the discarded states."""
    from aragog.eos.domain import EOSDomainError

    from .test_eos_floor import _drained, _mid_cell_at, _scipy_stop_below

    eos = entropy_eos_copy(EOS_DIR)
    _scipy_stop_below(monkeypatch)
    starts, solvers, solve = [], [], EntropySolver.solve

    def counted(self, *args, **kwargs):
        self.parameters.solver.eos_domain_check = 'raise'
        self.entropy_eos.domain = _tight(self.entropy_eos)
        solvers.append(self)
        starts.append(kwargs.get('_start', 'outer'))
        return solve(self, *args, **kwargs)

    monkeypatch.setattr(EntropySolver, 'solve', counted)
    with pytest.raises(EOSDomainError):
        _drained(eos, _mid_cell_at(eos, 1e-6), method='bdf')
    assert starts == ['outer', None] and solvers[0]._solution is None


def test_an_unknown_mode_set_after_construction_is_refused():
    """A mode or tolerance set on the parameters after construction is checked at the call."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    cells = _uniform_melt(s, eos, 50.0)
    s.parameters.solver.eos_domain_check = 'error'
    with pytest.raises(ValueError, match='eos_domain_check must be one of'):
        s._eos_domain_failure(_two_states(cells))
    s.parameters.solver.eos_domain_check = 'warn'
    s.parameters.solver.eos_domain_tolerance_cells = float('nan')
    with pytest.raises(ValueError, match='finite and > 0'):
        s._eos_domain_failure(_two_states(cells))


def test_off_mode_does_not_read_a_malformed_mask(tmp_path):
    """The masks are read at the first check: with the check off, an EOS whose mask holds a value
    other than 0 and 1 loads and solves; with the check on, the first call names the fault."""
    ref = entropy_eos_copy(EOS_DIR)
    d = _masked_dir(tmp_path, ref, slice(0, 2))
    m = np.loadtxt(d / 'valid_mask_melt.dat', ndmin=2)
    m[0, 0] = 2
    (d / 'valid_mask_melt.dat').unlink()
    np.savetxt(d / 'valid_mask_melt.dat', m, fmt='%d')
    s = _solver(EntropyEOS(d), mode='off')
    s.solve()
    assert s._solution.status == 0 and 'domain' not in vars(s.entropy_eos)
    s = _solver(EntropyEOS(d))
    with pytest.raises(ValueError, match='0 and 1'):
        s.solve()
    assert getattr(s, '_solution', None) is None  # raised before the call integrated


def test_a_bad_setting_or_mask_raises_before_the_call_and_keeps_the_solution(tmp_path):
    """After a good call, a bad mode or a single mask file raises at the next call start:
    the solver keeps the solution of the good call."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    s.solve()
    first = s._solution
    s.parameters.solver.eos_domain_check = 'error'
    with pytest.raises(ValueError, match='eos_domain_check must be one of'):
        s.solve()
    assert s._solution is first
    d = _masked_dir(tmp_path, eos, slice(0, 2))
    (d / 'valid_mask_solid.dat').unlink()
    s2 = _solver(EntropyEOS(d))
    with pytest.raises(FileNotFoundError, match='valid_mask_solid.dat is missing'):
        s2.solve()
    assert getattr(s2, '_solution', None) is None


def _solid_window_case(s, eos, offset_cells):
    """Melt window over the whole melt table, solid window ending ``offset_cells`` solid cells
    below the lowest solidus entropy of the basic nodes."""
    from aragog.eos.domain import TableDomain

    m, so = eos._tables['temperature_melt'], eos._tables['temperature_solid']
    dS_s = float(np.diff(so['S']).mean())
    s_sol_min = float(np.min(eos.solidus_entropy(s._P_basic_flat)))
    melt = _flat_window(
        m['P'], float(m['S'][0]), float(m['S'][-1]), float(np.diff(m['S']).mean())
    )
    solid = _flat_window(so['P'], float(so['S'][0]), s_sol_min - offset_cells * dS_s, dS_s)
    eos.domain = TableDomain(melt=melt, solid=solid)
    return solid, s_sol_min, dS_s


def _parse(msg):
    kind, k = msg.split('worst at ')[1].split(': ')[1].split(' (r=')[0].split(' node ')
    return kind, int(k), float(msg.split('table at S=')[1].split(' ')[0])


@pytest.mark.parametrize('where', ['solid', 'mush'])
def test_a_solid_or_mush_worst_point_names_the_solid_table_lookup(where, monkeypatch):
    """With the solid window cut below the solidus and the melt window whole, the worst point
    is read on the solid table: at its own entropy for a solid cell outside the smoothing band,
    at the solidus entropy for a mush cell. The message gives that entropy, the solid window
    ends at its pressure and the solid-table temperature there."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    solid, s_sol_min, dS_s = _solid_window_case(s, eos, 12.0)
    P_s = s._P_stag_flat
    if where == 'solid':  # 0.3 of the mush width below the solidus: blend weight 0
        s_sol, s_liq = eos.solidus_entropy(P_s), eos.liquidus_entropy(P_s)
        cells = s_sol - 0.3 * (s_liq - s_sol)
        solid = _flat_window(
            eos._tables['temperature_solid']['P'], 0.0, cells.min() - 8 * dS_s, dS_s
        )
        eos.domain = type(eos.domain)(melt=eos.domain.melt, solid=solid)
    else:
        cells = 0.5 * (eos.solidus_entropy(P_s) + eos.liquidus_entropy(P_s))
    from aragog.eos.domain import TableDomain

    reads, real = [], TableDomain.lookup_temperature
    monkeypatch.setattr(
        TableDomain, 'lookup_temperature', staticmethod(lambda *a: reads.append(a) or real(*a))
    )
    msg = s._eos_domain_failure(_two_states(cells))
    assert 'read on the solid table' in msg
    kind, k, s_look = _parse(msg)
    assert (
        len(reads) == 1 and reads[0][2] == pytest.approx(s_look, abs=1e-6) and not reads[0][3]
    )
    P = (s._P_basic_flat if kind == 'basic' else P_s)[k]
    if where == 'solid':
        assert s_look == pytest.approx(float(msg.split(', S=')[1].split(' ')[0]), abs=1e-6)
        assert s_look < float(eos.solidus_entropy(np.array([P]))[0]) - 0.25 * dS_s
    else:
        assert s_look == pytest.approx(float(eos.solidus_entropy(np.array([P]))[0]), abs=1e-6)
        assert f'S={s_look:.6f}' not in msg.split(', read on')[0]  # the cell is not at S_sol
    lo, hi = solid.ends(P)
    assert f'window [{lo:.6f}, {hi:.6f}]' in msg
    tbl = eos._tables['temperature_solid']
    T = float(tbl['interp']([[np.clip(P, tbl['P'][0], tbl['P'][-1]), s_look]])[0])
    assert float(msg.split('T=')[1].split(' K')[0]) == pytest.approx(T, abs=1e-5)


def test_the_table_temperature_equals_the_eos_temperature_of_a_single_phase_point():
    """Inside a table and outside the mush, the table read of the message is the EOS
    temperature: for a melt point on the melt table, for a solid point on the solid table."""
    from aragog.eos.domain import TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    P = np.array([1.0e9, 2.0e10, 8.0e10])
    for melt, S in (
        (True, eos.liquidus_entropy(P) + 300.0),
        (False, eos.solidus_entropy(P) - 300.0),
    ):
        phi = eos.melt_fraction(P, S)
        assert np.all(phi == (1.0 if melt else 0.0))
        T_eos = eos.temperature(P, S)
        for j in range(P.size):
            T = TableDomain.lookup_temperature(eos, P[j], S[j], melt)
            assert T == pytest.approx(float(T_eos[j]), rel=1e-12)


@pytest.mark.parametrize('mode', ['warn', 'off'])
@pytest.mark.parametrize('tol', [-1.0, 0.0, float('nan'), float('inf'), 'a', True])
def test_a_bad_tolerance_raises_at_the_call_start_and_keeps_the_state(mode, tol):
    """A bad tolerance set after a good call raises when the next call starts, in any mode,
    before the call changes the solution or the CMB ledger."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    s.solve()
    first, marker = s._solution, object()
    s.cmb_ledger = marker
    s.parameters.solver.eos_domain_check = mode
    s.parameters.solver.eos_domain_tolerance_cells = tol
    with pytest.raises((TypeError, ValueError), match='eos_domain_tolerance_cells must be'):
        s.solve()
    assert s._solution is first and s.cmb_ledger is marker


def test_an_eos_class_without_a_domain_turns_the_check_off():
    """An EOS object whose class has no domain gives no domain, also with an attribute of that
    name on the instance; no EOS gives none either."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import _eos_domain_of

    assert returns(_eos_domain_of, None) is None
    assert returns(_eos_domain_of, SimpleNamespace()) is None
    assert returns(_eos_domain_of, SimpleNamespace(domain='x')) is None
    eos = entropy_eos_copy(EOS_DIR)
    eos.domain = _tight(eos)
    assert _eos_domain_of(eos) is eos.domain


def test_basic_entropy_takes_a_given_basic_profile_as_a_copy():
    """With entropy_basic given (gradient mode), the result is a float copy of it, not the mesh
    transform of the cells."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos, core='gradient')
    S_s = np.linspace(2000.0, 3000.0, s._n_stag)
    given = np.full(s._n_stag + 1, 5000.0)
    out = s.state.basic_entropy(S_s, entropy_basic=given)
    np.testing.assert_array_equal(out, given)
    assert out is not given and out.dtype == float
    assert not np.array_equal(out, np.ravel(s.evaluator.mesh.quantity_at_basic_nodes(S_s)))


def test_raise_after_a_floor_stop_at_the_call_start_holds_the_outer_call_start(monkeypatch):
    """The floor root at the call start solves the call again; when that solve raises, the
    solver holds the result of the outer call start (none on a first call)."""
    from aragog.eos.domain import EOSDomainError
    from aragog.solver import entropy_solver as es

    from .test_eos_floor import _drained

    monkeypatch.setattr(
        es, '_floor_start_level', lambda w, S, *_: (es._floor_entropy(w, S) - 1e-12, False)
    )
    real = es.solve_ivp

    def moved(fun, t_span, y0, **kwargs):
        res = real(fun, t_span, y0, **kwargs)
        if res.status == 1 and res.t[-1] == res.t[0]:
            res.y[5, -1] += 1e-3  # the state moved while the time did not
        return res

    monkeypatch.setattr(es, 'solve_ivp', moved)
    eos = entropy_eos_copy(EOS_DIR)
    starts, solvers, solve = [], [], EntropySolver.solve

    def counted(self, *args, **kwargs):
        self.parameters.solver.eos_domain_check = 'raise'
        self.entropy_eos.domain = _tight(self.entropy_eos)
        solvers.append(self)
        starts.append(kwargs.get('_start', 'outer'))
        return solve(self, *args, **kwargs)

    monkeypatch.setattr(EntropySolver, 'solve', counted)
    with pytest.raises(EOSDomainError):
        _drained(eos, float(eos.S_min) + 400.0, method='bdf', start_time=1.0e8)
    assert starts == ['outer', None] and solvers[0]._solution is None


@pytest.mark.parametrize('bad', [(math.nan, 1500.0), (1.0e9, math.nan), (1.0e9, math.inf)])
def test_the_table_temperature_of_a_non_finite_point_is_nan(bad):
    """The EOS lookup clamps a non-finite point to the table edge; the message reads NaN."""
    from aragog.eos.domain import TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    for melt in (True, False):
        assert math.isnan(TableDomain.lookup_temperature(eos, *bad, melt))


def test_the_accepted_states_of_the_energy_trace_are_judged():
    """With the energy trace of the call, each accepted state counts: a state between 2 output
    points that leaves the window is named with its time."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    a = _uniform_melt(s, eos, 0.0)
    dS = eos.domain.melt.dS
    sol = _states(a, a, t=[0.0, 2.0])
    assert s._eos_domain_failure(sol) is None
    sol['energy_trace'] = (
        np.array([0.0, 0.7, 1.3, 2.0]),
        np.column_stack([a, a, a + 8.0 * dS, a]),
    )
    msg = s._eos_domain_failure(sol)
    assert msg is not None
    n_pts = 2 * s._n_stag + 1
    assert msg.startswith(f'{_MSG}: {n_pts} lookup points over the 4 accepted states')
    assert 'worst at t=1.300000e+00 yr' in msg and '8.00 cells outside' in msg


def test_without_a_trace_the_output_states_are_judged_with_one_info_line(caplog):
    """A call with no energy trace is judged at its output states; the solver says so once."""
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    a = _uniform_melt(s, eos, 0.0)
    with caplog.at_level(logging.INFO, logger='fwl.aragog'):
        for _ in range(3):
            assert s._eos_domain_failure(_states(a, a)) is None
    assert sum('output states only' in r.getMessage() for r in caplog.records) == 1


def test_the_check_reads_the_smoothing_width_of_the_phase_evaluators(monkeypatch):
    """The check judges the phase-boundary lookups of the smoothing band with the width that the
    staggered and basic phase evaluators of the solver use."""
    from aragog.eos import domain as dom_mod

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    eos.domain = _tight(eos)
    widths = []
    real = dom_mod.TableDomain.lookup_distances

    def record(self, eos_, P, S, **kw):
        widths.append(kw.get('smooth_width', 0.0))
        return real(self, eos_, P, S, **kw)

    monkeypatch.setattr(dom_mod.TableDomain, 'lookup_distances', record)
    y = np.array(s._S0, dtype=float)
    assert s._eos_domain_failure(_states(y)) is not None
    expect = [
        float(s.state.phase_staggered._matprop_smooth_width),
        float(s.state.phase_basic._matprop_smooth_width),
    ]
    assert widths == expect and expect[0] > 0.0


def test_a_point_at_an_empty_mask_column_is_named_as_having_no_window():
    """An empty mask column at the pressure of every node: the check names the point as having
    no valid window, not as a non-finite point."""
    from aragog.eos.domain import PhaseWindow, TableDomain

    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    cells = _uniform_melt(s, eos, 0.0)
    w = eos.domain.melt
    empty = PhaseWindow(P=w.P, lo=np.full(2, np.inf), hi=np.full(2, -np.inf), dS=w.dS)
    eos.domain = TableDomain(melt=empty, solid=w)
    msg = s._eos_domain_failure(_two_states(cells))
    assert 'no valid window at this pressure' in msg and 'not a finite point' not in msg


def test_the_check_reads_the_lookup_points_of_the_jax_rhs(monkeypatch):
    """The JAX energy_balance right-hand side (the one CVODE integrates in a run) evaluates the
    phases at the staggered and basic (P, S) points that the check judges, CMB node included."""
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('equinox')
    from aragog.eos import domain as dom_mod
    from aragog.jax import phase as jphase
    from aragog.jax import solver as jsolver

    from .test_cmb_closure_ledger import _jax_args
    from .test_cmb_closure_ledger import _solver as _eb_solver

    s = _eb_solver(0.5, solver_method='cvode')
    args = _jax_args(s)
    y = np.array(s._S0, dtype=float)
    y[s._n_stag] = 3.0e-3  # a CMB gradient that moves the bottom node
    seen_jax, seen_check = [], []
    real_eval = jphase.evaluate_phase

    def eval_rec(eos_, params, P, S, *a, **k):
        seen_jax.append((np.array(P, dtype=float), np.array(S, dtype=float)))
        return real_eval(eos_, params, P, S, *a, **k)

    monkeypatch.setattr(jphase, 'evaluate_phase', eval_rec)
    monkeypatch.setattr(jsolver, 'evaluate_phase', eval_rec)
    with jax.disable_jit():
        jsolver._dSdt_energy_balance_parts(0.0, jnp.asarray(y), args)
    real_look = dom_mod.TableDomain.lookup_distances

    def look_rec(self, eos_, P, S, **kw):
        seen_check.append((np.array(P, dtype=float), np.array(S, dtype=float)))
        return real_look(self, eos_, P, S, **kw)

    monkeypatch.setattr(dom_mod.TableDomain, 'lookup_distances', look_rec)
    s.entropy_eos.domain = _tight(s.entropy_eos)
    assert s._eos_domain_failure(_states(y)) is not None
    assert len(seen_check) == 2 and len(seen_jax) >= 2
    for P_c, S_c in seen_check:
        match = [
            (P_j, S_j)
            for P_j, S_j in seen_jax
            if P_j.shape == P_c.shape and np.allclose(P_j, P_c, rtol=1e-12, atol=0.0)
        ]
        assert match, 'no JAX evaluation at the pressures of the check'
        for _, S_j in match:
            np.testing.assert_allclose(S_j, S_c[0], rtol=1e-12, atol=0.0)  # the 1 state
    assert (
        seen_check[1][1][0, 0] != seen_check[0][1][0, 0]
    )  # the CMB node is not the bottom cell


def test_a_trace_longer_than_one_block_is_judged_in_full(monkeypatch):
    """The states are judged in blocks: with 2 states per block, the worst state, the last of its
    block, is named with its time, and the count sums the points of every block."""
    import aragog.solver.entropy_solver as es_mod

    monkeypatch.setattr(es_mod, '_EOS_DOMAIN_BLOCK', 2)
    eos = entropy_eos_copy(EOS_DIR)
    s = _solver(eos)
    a = _uniform_melt(s, eos, 0.0)
    dS = eos.domain.melt.dS
    sol = _states(a, a, t=[0.0, 4.0])
    sol['energy_trace'] = (
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.column_stack([a + 6.0 * dS, a, a + 7.0 * dS, a + 12.0 * dS, a + 9.0 * dS]),
    )
    msg = s._eos_domain_failure(sol)
    assert msg is not None
    n_pts = 2 * s._n_stag + 1
    assert msg.startswith(f'{_MSG}: {4 * n_pts} lookup points over the 5 accepted states')
    assert 'worst at t=3.000000e+00 yr' in msg and '12.00 cells outside' in msg  # a block end
