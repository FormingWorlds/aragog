"""V2 cross-check: the sub-grid lid on a coarse mesh against a resolved mesh without the lid,
for a solid shell and for the same shell started in the mush.

The shell is ``tests/reference/surface_lid_v2_column.py``. Its resolved runs cool below the
300 K floor of the SPIDER test tables, so the cross-check data come from the PALEOS table
extended to 67 K (``tests/reference/build_surface_lid_v2_grid.py``). On the test tables the
lid run stays above the floor and is rerun against a stored pin
(``tests/reference/build_surface_lid_v2_ci_pin.py``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_legend_clear, entropy_eos_copy, opaque_legend
from tests.reference.surface_lid_v2_column import (
    LOWT_EOS_MD5,
    MELT_TIMES_YR,
    MOLTEN_TIMES_YR,
    PIN_TIMES_YR,
    TIMES_YR,
    melt_column,
    molten_column,
    molten_start,
    shell,
    solidus_crossing,
    surface_flux,
)
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(900), needs_eos]

REF = Path(__file__).resolve().parent / 'reference'
TOL = 1e-2
MELT_TOL, FLOOR = 2e-2, 1e-5  # melt start: tolerance, round-off floor of the mesh run
C3_TOL = 3e-2  # molten start
C3_BOUND = 1e-4  # top-refined 400-node run against the 800-node reference
C3_HANDOVER_TOL = 3e-4  # hand-over time on the extended table: spread up to 8e-5 at 1e-13
C3_PIN_FLUX_TOL = (
    3e-5  # flux pin of the molten shell on the test tables: spread 2.3e-6 at 1e-13
)
C3_CROSSING_TOL = 5e-4  # hand-over against the crossing without the lid: gap at most 3.5e-5


def _stored(name, times=TIMES_YR):
    with np.load(REF / name) as d:
        np.testing.assert_array_equal(d['times_yr'], times)
        return {k: d[k] for k in d.files}


def _lowt_table():
    """The extended PALEOS table named by ``ARAGOG_LOWT_EOS_DIR``, checked by md5, or None."""
    table = os.environ.get('ARAGOG_LOWT_EOS_DIR')
    if table:
        md5 = hashlib.md5((Path(table) / 'temperature_solid.dat').read_bytes()).hexdigest()
        assert md5 == LOWT_EOS_MD5
    return table


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v2_lid_matches_the_resolved_mesh(validation_plot):
    """On the extended table, a 10-node shell with the lid loses heat through its surface as a
    4000-node shell without the lid, within TOL from 300 to 5000 yr; the 10-node shell without
    the lid misses by more than 10 %. With ``ARAGOG_LOWT_EOS_DIR`` set to that table, the lid
    run is repeated here."""
    d = _stored('surface_lid_v2_grid.npz')
    assert str(d['md5']) == LOWT_EOS_MD5
    lid, grid = d['lid'], d['grid_4000']
    np.testing.assert_allclose(lid, grid, rtol=TOL)
    assert np.all(np.abs(d['coarse'] / grid - 1) > 0.1)
    assert (
        0.08 < np.min(d['coarse'] / grid) < np.max(d['coarse'] / grid) < 0.30
    )  # the page quotes this range
    table = _lowt_table()
    if table:
        live = surface_flux(10, True, TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(live, lid, rtol=1e-6)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(1 - TOL, 1 + TOL, color='C0', alpha=0.15, label=f'{TOL:.0%} band')
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=0.8)
        bound = np.abs(grid - d['grid_1000']) / grid
        ax.errorbar(
            TIMES_YR,
            np.ones_like(grid),
            yerr=bound,
            fmt='none',
            color=plt.rcParams['text.color'],
            capsize=3,
            label='grid error estimate',
        )
        ax.plot(TIMES_YR, lid / grid, 'o', color='C1', ms=5, label='10 nodes with lid')
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks([300, 1000, 3000], ['300', '1000', '3000'])
        ax.set_ylim(0.97, 1.03)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel('lid / resolved flux')
        opaque_legend(ax)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v2_resolved_mesh')


@pytest.mark.reference_pinned
def test_v2_lid_run_on_the_test_tables():
    """The V2 lid run on the SPIDER test tables equals its stored pin."""
    pin = _stored('surface_lid_v2_ci_pin.npz')['lid']
    eos = entropy_eos_copy(EOS_DIR)
    np.testing.assert_allclose(surface_flux(10, True, TIMES_YR, eos), pin, rtol=1e-6)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v2_melt_lid_matches_the_resolved_mesh(validation_plot):
    """Started in the mush, without rheology, a 10-node shell with the lid gives the surface
    flux and the crust thickness of a 4000-node shell without the lid within MELT_TOL from 100
    to 5000 yr: the lid freezes melt with its latent heat, so the case tests the Stefan jump.
    The 10-node shell without the lid misses both by more than 25 %; a round-off member of the
    1000-node run stays within FLOOR. With ``ARAGOG_LOWT_EOS_DIR`` set to the extended table,
    the lid run is repeated."""
    d = _stored('surface_lid_v2_melt.npz', MELT_TIMES_YR)
    assert str(d['md5']) == LOWT_EOS_MD5
    for q in ('flux', 'crust'):
        grid = d[f'grid_4000_{q}']
        np.testing.assert_allclose(d[f'grid_1000_p_{q}'], d[f'grid_1000_{q}'], rtol=FLOOR)
        np.testing.assert_allclose(d[f'lid_{q}'], grid, rtol=MELT_TOL)
        assert np.all(np.abs(d[f'coarse_{q}'] / grid - 1) > 0.25)
    crust = d['coarse_crust'] / d['grid_4000_crust']
    assert 4.3 < np.min(crust) < np.max(crust) < 30.5  # the page quotes this range
    table = _lowt_table()
    if table:
        live = melt_column(10, True, MELT_TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(live, (d['lid_flux'], d['lid_crust']), rtol=1e-6)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(
            1 - MELT_TOL, 1 + MELT_TOL, color='C0', alpha=0.15, label=f'{MELT_TOL:.0%} band'
        )
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=0.8)
        for q, mark, color in (('flux', 'o', 'C1'), ('crust', 's', 'C2')):
            ratio = d[f'lid_{q}'] / d[f'grid_4000_{q}']
            ax.plot(MELT_TIMES_YR, ratio, mark, color=color, ms=5, label=q)
        ax.set_xscale('log')
        ax.set_ylim(0.96, 1.04)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel('lid / resolved')
        opaque_legend(ax, loc='upper right', ncol=3)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v2_melt_resolved_mesh')


@pytest.mark.reference_pinned
def test_v2_melt_lid_run_on_the_test_tables():
    """The V2 melt lid run on the SPIDER test tables equals its stored pin."""
    pin = _stored('surface_lid_v2_ci_pin.npz')
    np.testing.assert_array_equal(pin['melt_times_yr'], MELT_TIMES_YR)
    live = melt_column(10, True, MELT_TIMES_YR, entropy_eos_copy(EOS_DIR))
    np.testing.assert_allclose(live, (pin['melt_flux'], pin['melt_crust']), rtol=1e-6)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_c3_molten_lid_matches_the_resolved_mesh(validation_plot):
    """Started molten and convecting, without rheology, a 10-node shell with the lid hands the
    lid over within the first 100 yr and then equals the 10-node shell without the lid within
    FLOOR, so this checks the hand-over and the coarse mesh after it (the hand-over criterion
    itself is tested in test_subgrid_lid_solver.py): both stay within C3_TOL of the reference,
    an 800-node shell without the lid with a top cell of 0.125 m, from 100 to 5000 yr. The
    400-node shell with a top cell of 0.25 m agrees with the reference within C3_BOUND. With
    the hand-over held off the lid misses the reference flux by more than 50 % at some time.
    The hand-over time lies within C3_CROSSING_TOL of the time at which the top cell of the stored
    10-node run without the lid cools through the solidus, a run the lid code does not touch.
    With ``ARAGOG_LOWT_EOS_DIR`` set to the extended table, the 10-node runs are repeated."""
    d = _stored('surface_lid_c3_molten.npz', MOLTEN_TIMES_YR)
    assert str(d['md5']) == LOWT_EOS_MD5
    grid = d['refined_800']
    np.testing.assert_allclose(d['refined_400'], grid, rtol=C3_BOUND)
    crossing = solidus_crossing(d['nolid_t'], d['nolid_S_top'], float(d['nolid_S_sol']))
    assert float(d['handover_yr']) == pytest.approx(crossing, rel=C3_CROSSING_TOL)
    np.testing.assert_allclose(d['lid'], d['coarse'], rtol=FLOOR)
    for run in ('lid', 'coarse'):
        np.testing.assert_allclose(d[run], grid, rtol=C3_TOL)
    assert np.max(np.abs(d['lid_kept'] / grid - 1)) > 0.5
    table = _lowt_table()
    if table:
        live, t_ho = molten_column(10, True, MOLTEN_TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(live, d['lid'], rtol=FLOOR)
        assert t_ho == pytest.approx(crossing, rel=C3_CROSSING_TOL)
        assert t_ho == pytest.approx(float(d['handover_yr']), rel=C3_HANDOVER_TOL)
        coarse, _ = molten_column(10, False, MOLTEN_TIMES_YR, entropy_eos_copy(table))
        np.testing.assert_allclose(coarse, d['coarse'], rtol=FLOOR)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(1 - C3_TOL, 1 + C3_TOL, color='C0', alpha=0.15, label=f'{C3_TOL:.0%} band')
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=0.8)
        ax.plot(
            MOLTEN_TIMES_YR, d['lid'] / grid, 'o', color='C1', ms=4, label='10 nodes with lid'
        )
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks([100, 300, 1000, 3000], ['100', '300', '1000', '3000'])
        ax.set_ylim(1 - 2 * C3_TOL, 1 + 2 * C3_TOL)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel('lid / resolved flux')
        opaque_legend(ax)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_c3_molten_resolved_mesh')


@pytest.mark.reference_pinned
def test_c3_molten_lid_run_on_the_test_tables():
    """The molten-start lid run on the SPIDER test tables equals its stored pin and hands over,
    up to 935 yr. Later the top of this shell leaves real table entries: from before 1635 yr its
    surface lies on the padded rows of the table at about 320.23 K, and from about 2334 yr a cell
    lies below the table, which the EOS guard refuses. The pin checks only that the run repeats.
    A solidus shifted by 10 J/kg/K moves these fluxes by at most 2.2e-6, inside their round-off, so a
    shifted hand-over fails only the time checks: the crossing of the solidus by the stored run
    without the lid sees a shift of 1 J/kg/K, the stored time 0.1 J/kg/K. The pin does not see
    the heat content of the merge; test_merge_is_energy_exact checks that."""
    d = _stored('surface_lid_c3_molten.npz', MOLTEN_TIMES_YR)
    np.testing.assert_array_equal(d['pin_times_yr'], PIN_TIMES_YR)
    assert np.all(np.diff(d['pin']) < -100.0)  # every row off the padded table rows
    live, t_ho = molten_column(10, True, PIN_TIMES_YR, entropy_eos_copy(EOS_DIR))
    crossing = solidus_crossing(
        d['pin_nolid_t'], d['pin_nolid_S_top'], float(d['pin_nolid_S_sol'])
    )
    assert t_ho == pytest.approx(crossing, rel=C3_CROSSING_TOL)
    np.testing.assert_allclose(live, d['pin'], rtol=C3_PIN_FLUX_TOL)
    assert t_ho == pytest.approx(float(d['pin_handover_yr']), rel=FLOOR)


def test_a_guard_cut_after_the_hand_over_keeps_its_time():
    """The molten shell on the test tables leaves the table after its hand-over: the call fails
    at the EOS guard, its kept states end with the lid handed over, and the time stays."""
    eos = entropy_eos_copy(EOS_DIR)
    s = shell(10, True, eos, 3000.0, 0.0, rheology=False, convection=True)
    s.set_initial_entropy(molten_start(s, eos))
    s.solve()
    assert s._solution.status == -1 and 'EOS table guard' in str(s._solution.message)
    assert s._solution.t[-1] > 1000.0
    out = s.get_state()
    d = _stored('surface_lid_c3_molten.npz', MOLTEN_TIMES_YR)
    assert out.lid_delta == 0.0
    assert out.lid_handover_time == pytest.approx(float(d['pin_handover_yr']), rel=FLOOR)


def _handover_times(table, end_yr, tol):
    """Hand-over times of the molten shell for the starts 0, 1e-13 and -1e-13, at the default
    solver tolerance (``tol`` None) or at ``rtol = atol = tol``."""
    times = []
    for shift in (0.0, 1e-13, -1e-13):
        eos = entropy_eos_copy(table)
        s = shell(10, True, eos, end_yr, 0.0, rheology=False, convection=True)
        if tol:
            s.parameters.solver.rtol = s.parameters.solver.atol = tol
        s.set_initial_entropy(molten_start(s, eos) * (1.0 + shift))
        s.solve()
        assert s._solution.status == 0
        times.append(float(s.get_state().lid_handover_time))
    return times


def _spread(times):
    """Relative spread of a list of hand-over times."""
    return (max(times) - min(times)) / times[0]


@pytest.mark.parametrize(
    ('name', 'end_yr', 'bounds'),
    [('test', 40.0, (FLOOR, 1e-7)), ('lowt', 12.0, (C3_HANDOVER_TOL, 1e-6))],
    ids=['test', 'lowt'],
)
def test_c3_handover_spread_at_two_solver_tolerances(name, end_yr, bounds):
    """Start shifts of 1e-13 stay within the time pin of the rerun at the default tolerance and
    within a tighter bound at rtol = atol = 1e-8. The spread is a noisy sample that depends on the
    machine, so the test does not hold the order of the 2 spreads. A tolerance without effect gives
    the same times; a start shift without effect gives 0. The printed numbers are the page's."""
    table = EOS_DIR if name == 'test' else _lowt_table()
    if not table:
        pytest.skip('ARAGOG_LOWT_EOS_DIR is not set')
    default, fine = (_handover_times(table, end_yr, tol) for tol in (None, 1e-8))
    print(
        f'{name}: default {_spread(default[:2]):.2e} {_spread(default):.2e}, '
        f'1e-8 {_spread(fine[:2]):.2e} {_spread(fine):.2e}'
    )
    assert _spread(default) <= bounds[0] and 0.0 < _spread(fine) <= bounds[1]
    assert default != fine


@pytest.mark.parametrize(
    ('name', 'solid', 'melt'),
    [('test tables', 3037.5, 3607.8), ('extended table', 3040.8, 3602.8)],
    ids=['test', 'lowt'],
)
def test_melt_density_against_the_solid_at_the_top_cell_pressure(name, solid, melt):
    """The page quotes the density sign of these tables at the top-cell pressure of its shells:
    the melt just above the liquidus is denser than the solid just below the solidus. The lid
    has no buoyancy test, so this tells only which way a crust would sink on these tables."""
    table = EOS_DIR if name == 'test tables' else _lowt_table()
    if not table:
        pytest.skip('ARAGOG_LOWT_EOS_DIR is not set')
    eos = entropy_eos_copy(table)
    P = float(np.asarray(shell(10, False, eos, 1.0, 0.0)._P_stag_flat)[-1])
    S_sol = float(np.asarray(eos.solidus_entropy(P)).item())
    S_liq = float(np.asarray(eos.liquidus_entropy(P)).item())
    assert P == pytest.approx(55.68e6, rel=1e-3)
    assert float(np.asarray(eos.density(P, S_sol - 1.0)).item()) == pytest.approx(
        solid, rel=1e-3
    )
    assert float(np.asarray(eos.density(P, S_liq + 1.0)).item()) == pytest.approx(
        melt, rel=1e-3
    )


def test_tutorial_blocks_run_and_match_page(monkeypatch, capsys, tmp_path):
    """The Python blocks of the lid tutorial run on the test tables and print the numbers of the
    page, each to a relative 1e-3."""
    import re

    path = Path(__file__).resolve().parents[1] / 'docs' / 'Tutorials' / 'surface_lid.md'
    text = path.read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.DOTALL)
    expected = ''.join(re.findall(r'```text\n(.*?)```', text, flags=re.DOTALL))
    assert blocks and expected
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', str(EOS_DIR))
    monkeypatch.chdir(tmp_path)
    ns: dict[str, object] = {'__name__': '__main__'}
    for block in blocks:
        exec(compile(block, str(path), 'exec'), ns)  # noqa: S102
    out = capsys.readouterr().out
    number = r'-?\d+\.\d+'
    words = [re.sub(number, '#', line).split() for line in expected.strip().splitlines()]
    assert [re.sub(number, '#', line).split() for line in out.strip().splitlines()] == words
    want, got = (np.array(re.findall(number, t), dtype=float) for t in (expected, out))
    assert want.size >= 10
    np.testing.assert_allclose(got, want, rtol=1e-3)
    prose = ns['EntropySolver'].from_file(filename='surface_lid.toml', eos_dir=ns['eos_dir'])
    prose.initialize()
    P = float(np.asarray(prose._P_stag_flat)[-1])
    eos = prose.entropy_eos
    assert P == pytest.approx(7.8e9, rel=5e-3)
    T_sol = float(np.asarray(eos.temperature(P, eos.solidus_entropy(P))).item())
    assert T_sol == pytest.approx(2149.0, rel=1e-3)
    assert float(np.asarray(eos.melt_fraction(P, 3050.0)).item()) == pytest.approx(
        0.36, abs=5e-3
    )
    r = np.asarray(prose.evaluator.mesh.basic.radii, dtype=float).ravel()
    assert r[-1] - r[-2] == pytest.approx(380.0e3, rel=5e-3)
