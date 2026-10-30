"""``tools/verification/compare_v5_runs.py``: config pins, run roles, floors and verdicts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'compare_v5_runs',
    Path(__file__).resolve().parents[1] / 'tools' / 'verification' / 'compare_v5_runs.py',
)
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

BRANCH, MAIN = '26.10.29.dev12+g507e408a', '26.10.22.dev8+g4ae8167d'


def _run(
    root,
    case,
    aragog,
    T_magma=1000.0,
    temp=(1.0, 2.0),
    late=None,
    t_end=1000.5,
    t_snap=None,
    extra_rows=0,
    drop=(),
    proteus=None,
    **driver,
):
    """Write a run folder: helpfile rows, PROTEUS log, 2 snapshots, and the v5b driver result.

    ``driver`` overrides the v5b result keys (reference, reference_rtol, solid_steps, s0_eps)
    and Phi_global; ``drop`` removes result keys; ``proteus`` replaces the logged PROTEUS
    commit. The last row is at ``t_end``, the last snapshot at ``t_snap`` (default ``t_end``).
    """
    pin = cmp.pins(cmp.CONFIGS[case])
    phi = driver.pop('Phi_global', 0.0 if case == 'v5b' else 1.0)
    out = root / 'output'
    (out / 'data').mkdir(parents=True)
    last = {**dict.fromkeys(cmp.COLUMNS, 1.0), 'T_magma': T_magma, 'Phi_global': phi}
    last['Time'] = t_end
    rows = [dict.fromkeys(cmp.COLUMNS, 0.0)] * (1 + extra_rows) + [last]
    lines = ['\t'.join(cmp.COLUMNS)] + ['\t'.join(str(r[k]) for k in cmp.COLUMNS) for r in rows]
    (out / 'runtime_helpfile.csv').write_text('\n'.join(lines) + '\n')
    (out / 'proteus_00.log').write_text(
        f'[ INFO  ] PROTEUS git hash  {proteus or pin["PROTEUS"]}\n'
        f'[ INFO  ] Interior module   aragog version {aragog}\n'
    )
    t_snap = t_end if t_snap is None else t_snap
    for name, T in (
        ('99p000_int.nc', late or temp),
        (f'{t_snap:.3f}'.replace('.', 'p') + '_int.nc', temp),
    ):
        with netCDF4.Dataset(out / 'data' / name, 'w') as nc:
            nc.createDimension('s', len(T))
            for k in cmp.PROFILES:
                nc.createVariable(k, 'f8', ('s',))[:] = np.asarray(T)
    if case == 'v5b':
        res = {'reference': False, 'reference_rtol': None, 'solid_steps': 3, 's0_eps': 0.0}
        res = {k: v for k, v in {**res, **driver}.items() if k not in drop}
        (root / 'v5b_result.json').write_text(json.dumps(res))
    return root


def _v5b(tmp_path, run_T=999.5, run_temp=(1.0, 1.5)):
    """REFERENCE, MAIN and RUN of a v5b case where RUN is closer to REFERENCE than MAIN."""
    ref = _run(
        tmp_path / 'ref',
        'v5b',
        BRANCH,
        1000.0,
        (1.0, 2.0),
        reference=True,
        reference_rtol=1e-10,
    )
    main = _run(tmp_path / 'main', 'v5b', MAIN, 1000.5, (1.0, 2.5))
    run = _run(tmp_path / 'run', 'v5b', BRANCH, run_T, run_temp)
    return run, main, ref


def _floor(tmp_path, run_T=(999.5, 999.5), main_T=(1000.5, 1000.5)):
    """RUN_P, RUN_M, MAIN_P and MAIN_M at s0_eps = +-1e-12 with the T_magma values given."""
    return [
        _run(tmp_path / 'rp', 'v5b', BRANCH, run_T[0], (1.0, 1.5), s0_eps=1e-12),
        _run(tmp_path / 'rm', 'v5b', BRANCH, run_T[1], (1.0, 1.5), s0_eps=-1e-12),
        _run(tmp_path / 'mp', 'v5b', MAIN, main_T[0], (1.0, 2.5), s0_eps=1e-12),
        _run(tmp_path / 'mm', 'v5b', MAIN, main_T[1], (1.0, 2.5), s0_eps=-1e-12),
    ]


def test_both_configs_record_the_proteus_and_aragog_pins(tmp_path):
    """Each case config header names a 40-digit PROTEUS commit and the Aragog commit of MAIN."""
    for case in cmp.CONFIGS:
        pin = cmp.pins(cmp.CONFIGS[case])
        assert all(len(v) == 40 and int(v, 16) >= 0 for v in pin.values())
        assert pin['Aragog'].startswith(MAIN.rsplit('+g', 1)[1])
    (tmp_path / 'c.toml').write_text('# PROTEUS ' + 'a' * 40 + '\nx = 1\n')
    with pytest.raises(SystemExit, match='needs a PROTEUS and an Aragog commit'):
        cmp.pins(tmp_path / 'c.toml')


def test_v5_passes_only_on_bitwise_equal_end_states(tmp_path):
    """Equal end states pass; one helpfile digit or one profile entry fails the case."""
    run = _run(tmp_path / 'run', 'v5', BRANCH, late=(5.0, 5.0))
    main = _run(tmp_path / 'main', 'v5', MAIN)
    res = cmp.compare('v5', run, main)
    assert res['passed'] and res['versions']['main']['Aragog'] == MAIN
    off = _run(tmp_path / 'off', 'v5', BRANCH, T_magma=1000.0 + 1e-9)
    assert cmp.compare('v5', off, main)['columns']['T_magma'] == dict(equal=False, passed=False)
    prof = _run(tmp_path / 'prof', 'v5', BRANCH, temp=(1.0, 2.0 + 1e-12))
    res = cmp.compare('v5', prof, main)
    assert not res['passed'] and not res['columns']['temp_s']['passed']


def test_v5b_compares_the_errors_against_the_reference(tmp_path):
    """RUN passes a column when its error is not larger than the error of MAIN."""
    run, main, ref = _v5b(tmp_path)
    res = cmp.compare('v5b', run, main, ref)
    assert res['passed']
    assert res['columns']['T_magma'] == dict(
        error_run=0.5, error_main=0.5, floor=0.0, passed=True
    )
    worse = _run(tmp_path / 'worse', 'v5b', BRANCH, 1000.0, (1.0, 2.75))
    res = cmp.compare('v5b', worse, main, ref)
    assert not res['passed'] and res['columns']['temp_s']['error_run'] == 0.75


def test_v5b_floor_is_the_largest_round_off_offset(tmp_path):
    """A gap smaller than the round-off offset of either code passes; a larger one fails."""
    run, main, ref = _v5b(tmp_path, run_T=1000.75)
    assert not cmp.compare('v5b', run, main, ref)['columns']['T_magma']['passed']
    floor = _floor(tmp_path, (1000.75, 1000.65), (1000.5, 1000.25))
    col = cmp.compare('v5b', run, main, ref, floor)['columns']['T_magma']
    assert col == dict(error_run=0.75, error_main=0.5, floor=0.25, passed=True)
    floor[3] = _run(tmp_path / 'mm2', 'v5b', MAIN, 1000.4, (1.0, 2.5), s0_eps=-1e-12)
    assert not cmp.compare('v5b', run, main, ref, floor)['columns']['T_magma']['passed']


@pytest.mark.parametrize(
    'role, change, failed',
    [
        ('main', dict(aragog=BRANCH), 'Aragog pin'),
        ('run', dict(aragog=MAIN), 'Aragog pin'),
        ('main', dict(aragog=MAIN + '.d20261029'), 'modified tree'),
        ('main', dict(proteus='b' * 40), 'PROTEUS commit'),
        ('reference', dict(aragog='26.10.29.dev1+gdeadbee'), 'Aragog version'),
        ('reference', dict(reference=False), 'reference'),
        ('run', dict(reference=True), 'reference'),
        ('main', dict(solid_steps=2), 'solid_steps'),
        ('run', dict(s0_eps=1e-12), 's0_eps'),
        ('run', dict(Phi_global=0.1), 'Phi_global'),
        ('reference', dict(reference_rtol=1e-9), 'reference_rtol'),
        ('reference', dict(drop=('reference_rtol',)), 'reference_rtol'),
        ('run', dict(drop=('s0_eps',)), 's0_eps'),
        ('main', dict(s0_eps=-1e-12), 's0_eps'),
        ('run', dict(drop=('reference_rtol',)), 'reference_rtol'),
        ('main', dict(drop=('reference_rtol',)), 'reference_rtol'),
        ('main', dict(aragog=BRANCH, solid_steps=2), 'Aragog pin, solid_steps'),
    ],
)
def test_v5b_refuses_a_run_that_does_not_fit_its_role(tmp_path, role, change, failed):
    """Versions, a modified tree, driver flags and the end melt fraction must fit each role.

    The message names the run and each changed check.
    """
    run, main, ref = _v5b(tmp_path)
    kw = dict(
        run=('run', 999.5, (1.0, 1.5), {}),
        main=('main', 1000.5, (1.0, 2.5), {}),
        reference=('ref', 1000.0, (1.0, 2.0), {'reference': True, 'reference_rtol': 1e-10}),
    )
    name, T, temp, driver = kw[role]
    aragog = change.pop('aragog', MAIN if role == 'main' else BRANCH)
    bad = _run(tmp_path / f'{name}_bad', 'v5b', aragog, T, temp, **{**driver, **change})
    paths = dict(run=run, main=main, reference=ref)
    paths[role] = bad
    with pytest.raises(SystemExit, match=rf'{role}: run .*; failed: {failed}$'):
        cmp.compare('v5b', paths['run'], paths['main'], paths['reference'])


@pytest.mark.parametrize(
    'change, failed',
    [
        (dict(s0_eps=0.5), 's0_eps'),
        (dict(drop=('s0_eps',)), 's0_eps'),
        (dict(s0_eps=1e-12, t_end=1001.5), 'end time and rows of run'),
        (dict(s0_eps=1e-12, extra_rows=1), 'end time and rows of run'),
    ],
)
def test_v5b_refuses_a_round_off_run_off_its_base(tmp_path, change, failed):
    """A round-off run needs s0_eps = +-1e-12 and the end time and row count of its base run."""
    run, main, ref = _v5b(tmp_path)
    floor = _floor(tmp_path)
    assert cmp.compare('v5b', run, main, ref, floor)['passed']
    floor[0] = _run(tmp_path / 'rp_bad', 'v5b', BRANCH, 999.5, (1.0, 1.5), **change)
    with pytest.raises(SystemExit, match=rf'run_p: run .*; failed: {failed}$'):
        cmp.compare('v5b', run, main, ref, floor)


def test_v5b_accepts_runs_that_fit_their_roles(tmp_path):
    """A reference at rtol 1e-10 and round-off runs at s0_eps = +-1e-12 pass the role checks."""
    run, main, ref = _v5b(tmp_path)
    floor = _floor(tmp_path)
    try:
        res = cmp.compare('v5b', run, main, ref, floor)
    except SystemExit as exc:
        pytest.fail(f'runs that fit their roles were refused: {exc}')
    assert res['passed']


@pytest.mark.parametrize('folder', ['', 'output'])
def test_versions_reads_the_driver_result_beside_the_output_folder(tmp_path, folder):
    """Given RUN or RUN/output, versions() reads the driver flags from RUN/v5b_result.json."""
    _run(tmp_path, 'v5b', BRANCH, reference=True, reference_rtol=1e-10)
    v = cmp.versions(tmp_path / folder)
    flags = {k: v.get(k) for k in ('reference', 'reference_rtol', 'solid_steps', 's0_eps')}
    assert flags == dict(reference=True, reference_rtol=1e-10, solid_steps=3, s0_eps=0.0)


def test_reference_rtol_selects_the_reference_tolerance(tmp_path):
    """A reference at rtol 1e-11 is refused by default and accepted with reference_rtol 1e-11."""
    run, main, _ = _v5b(tmp_path)
    ref11 = _run(tmp_path / 'r11', 'v5b', BRANCH, reference=True, reference_rtol=1e-11)
    with pytest.raises(SystemExit, match='reference: run .*; failed: reference_rtol$'):
        cmp.compare('v5b', run, main, ref11)
    assert cmp.compare('v5b', run, main, ref11, reference_rtol=1e-11)['passed']


def test_end_state_needs_the_last_snapshot_at_the_last_row(tmp_path):
    """The last snapshot must be within 1e-3 of the Time of the last helpfile row.

    The snapshot file name keeps 3 decimals: a row at 1000.5004 names 1000p500.
    """
    assert cmp.end_state(_run(tmp_path / 'ok', 'v5', BRANCH))['rows'] == 2
    try:
        assert cmp.end_state(_run(tmp_path / 'ok4', 'v5', BRANCH, t_end=1000.5004))['rows'] == 2
    except SystemExit as exc:
        pytest.fail(f'a row 4e-4 from its snapshot time was refused: {exc}')
    with pytest.raises(
        SystemExit, match='last snapshot at 1000.502, last helpfile Time 1000.5'
    ):
        cmp.end_state(_run(tmp_path / 'late', 'v5', BRANCH, t_snap=1000.502))


def test_runs_must_be_distinct_and_v5_must_stay_molten(tmp_path):
    """RUN given as REFERENCE is refused; a v5 run that left Phi_global = 1 is refused."""
    run, main, _ = _v5b(tmp_path)
    with pytest.raises(SystemExit, match='given once'):
        cmp.compare('v5b', run, main, run)
    molten = _run(tmp_path / 'v5r', 'v5', BRANCH, Phi_global=0.5)
    with pytest.raises(SystemExit, match='run: run .*; failed: Phi_global$'):
        cmp.compare('v5', molten, _run(tmp_path / 'v5m', 'v5', MAIN))


def test_main_writes_the_result_and_exits_1_on_a_failed_column(tmp_path, capsys):
    """main() prints the verdict, writes the JSON, and needs --reference exactly for v5b."""
    run = _run(tmp_path / 'run', 'v5', BRANCH, T_magma=2.0)
    main = _run(tmp_path / 'main', 'v5', MAIN)
    out = tmp_path / 'r.json'
    assert cmp.main(['--case', 'v5', str(run), str(main), '--output', str(out)]) == 1
    assert 'v5: FAIL' in capsys.readouterr().out
    assert json.loads(out.read_text())['columns']['T_magma']['passed'] is False
    for argv in (
        ['--case', 'v5b', str(run), str(main)],
        ['--case', 'v5', str(run), str(main), '--floor', 'a', 'b', 'c', 'd'],
    ):
        with pytest.raises(SystemExit) as exc:
            cmp.main(argv)
        assert exc.value.code == 2
