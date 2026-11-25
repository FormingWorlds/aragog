"""The EOS domain check of a call: lookup points outside the valid window of the tables.

The windows here are set on the CI tables, so each expected distance follows from the window and
the state. The check warns by default, raises in 'raise' mode and never changes a result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult

from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
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
    assert 'staggered node 3 ' in msg and 'not a finite point' in msg


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
    assert s._eos_domain_failure(_two_states(cells)) is None


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
