"""Regression test for the CMB per-call energy booking against a real coupled-run resume.

Aragog books the CMB heat flux for each accepted solver call into
``step_dE_F_cmb_J`` (the ``F_cmb`` quadrature slot of
``EntropySolver._compute_step_energy_integrals``). Historically that
quadrature slot's CVODE absolute tolerance was a single fixed factor
applied uniformly across every ``core_bc`` mode, unrelated to how large the
call's actual CMB flux was; when the true flux was small relative to that
fixed scale, CVODE could satisfy the (too loose) error test while still
integrating the quadrature state far from the physically-integrated flux.
The fix (``EntropySolver._assemble_atol_nd``, aragog commit ``c98c3ba``)
sizes each quadrature slot's tolerance from its own boundary flux at the
end of the previous ``solve()`` call instead.

This test drives the production PROTEUS -> Aragog call sequence
(``AragogRunner.setup_or_update_solver`` / ``run_solver``) directly in
Python, without ``proteus start`` and without AGNI/atmodeller, by replaying
a handful of real historical rows from a completed coupled run
(``item1b_armA_replay``, a resume of ``remelt_massA_floor`` from the row
immediately after its third giant impact). The atmosphere-facing boundary
condition (``F_atm``, ``T_eqm``) at each replayed step is read verbatim
from that run's frozen ``runtime_helpfile.csv``, so no atmosphere or
outgassing module needs to run; the interior call sequence is otherwise
exactly what ``proteus.interior_energetics.wrapper.run_interior`` executes
for the ``aragog`` module every coupling step.

Scope: 3 solver calls (rows 538-540 of the replay), enough to cover the
first call after a resume (a known, expected transient unrelated to this
fix -- see below) and two subsequent calls. This keeps the test to
roughly a minute (dominated by one-time PALEOS/JAX-Jacobian setup, not by
the entropy integration itself) instead of the ~31 minutes a full coupled
run of the same window takes.

Measured ratios (booked / physical) on this scenario, aragog c98c3ba (fix):
    call 0 (row 538, first call after resume): -1.79e4   (excluded, see below)
    call 1 (row 539):                           1.00000004
    call 2 (row 540):                           1.00000003
On the immediate parent commit (bb5a4a4, pre-fix), call 0 is a similarly
large transient, but call 1 is -3.72e3 instead of ~1 -- more than three
orders of magnitude off, and call 2 recovers to ~1 (this specific
scenario's defect window is narrow, one call wide). This asymmetry
(call 1 fails pre-fix, passes post-fix; call 0 and call 2 behave
similarly on both) is exactly what this test asserts.

Call 0 is excluded from the tight-tolerance assertion because the very
first ``solve()`` call on a freshly constructed ``EntropySolver`` (which a
resume always builds, discarding the previous process's solver instance)
has no prior-call flux to size its quadrature atol from; both the fixed
and the flux-scaled schemes fall back to the same fixed factor there, so
call 0's large transient is present on both commits and is not diagnostic
of this fix. Only a finite/non-NaN check is applied to call 0.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUN_DIR = Path(
    os.environ.get(
        'ARAGOG_TEST_ARMA_REPLAY_DIR',
        '/Users/timlichtenberg/git/PROTEUS-dev-1/output/item1b_armA_replay',
    )
)

try:
    import aragog.solver.entropy_solver as _entropy_solver_mod
    from proteus.config import read_config_object
    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.interior_energetics.common import Interior_t
    from proteus.interior_energetics.wrapper import get_nlevb

    _PROTEUS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _PROTEUS_IMPORT_ERROR = exc

needs_proteus = pytest.mark.skipif(
    _PROTEUS_IMPORT_ERROR is not None,
    reason=f'PROTEUS interior_energetics stack not importable: {_PROTEUS_IMPORT_ERROR}',
)
needs_fixture = pytest.mark.skipif(
    not RUN_DIR.is_dir(),
    reason=f'arm A replay fixture not found at {RUN_DIR}',
)

# Anchor row (0-based pandas index into runtime_helpfile.csv): the resume
# point, t = 1.502802e6 yr, the row immediately after the 3rd of 3 landed
# giant impacts. Matches "RESUME at helpfile row 538" (1-based row count)
# in the replay's own proteus_02.log.
ANCHOR = 537
N_STEPS = 3

# The replay's own log line this test cross-checks the standalone entropy
# restore against: "Restored entropy IC from snapshot: S_mean=10371.3 J/kg/K"
# (proteus_02.log). 0.05 matches that line's %.1f display precision.
EXPECTED_S_MEAN_AT_RESUME = 10371.3
S_MEAN_ATOL = 0.05

# Booked/physical agreement on aragog c98c3ba (the fix) measures |ratio-1|
# ~ 3e-8 to 4e-8 on calls 1 and 2 of this scenario (see module docstring).
# 1e-4 keeps four orders of magnitude of headroom over that measurement
# while still failing hard against the pre-fix call-1 ratio of -3.72e3.
RATIO_TOL = 1e-4

# sim_time (this call's actual integration endpoint) must reproduce the
# historical Time column to close to solver precision; a mismatch here
# means the standalone driver is not actually replaying this run's
# trajectory (wrong dt, wrong resume anchor, ...).
SIM_TIME_RTOL = 1e-6


def _load_hf_rows() -> list[dict]:
    df = pd.read_csv(RUN_DIR / 'runtime_helpfile.csv', sep='\t')
    return [df.iloc[i].to_dict() for i in range(ANCHOR, ANCHOR + N_STEPS + 1)]


@needs_proteus
@needs_fixture
@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_cmb_flux_booking_matches_physical_integral(tmp_path, monkeypatch):
    """Booked step_dE_F_cmb_J must match the trapezoid-integrated CMB flux.

    Drives the real ``AragogRunner`` call sequence over 3 real historical
    rows of a completed coupled run (see module docstring), then compares
    each call's booked ``step_dE_F_cmb_J`` (the CVODE-integrated quadrature
    state) against the same call's sub-step trajectory dump, trapezoid-
    integrated the same way the aragog side of that column construction
    integrates it.
    """
    # Force the per-call sub-step trajectory dump on, regardless of
    # whether ARAGOG_DUMP_STEP_TRAJ was read (or not) at aragog import
    # time; the module-level dump directory is a plain attribute the
    # dump function re-reads on every call, so patching it here (rather
    # than the environment variable, which is only read once at import)
    # works no matter what already imported this module first.
    traj_dir = tmp_path / 'traj'
    traj_dir.mkdir()
    monkeypatch.setattr(_entropy_solver_mod, '_TRAJ_DUMP_DIR', str(traj_dir))

    config = read_config_object(str(RUN_DIR / 'init_coupler.toml'))
    assert config.params.resume is True
    assert config.interior_energetics.module == 'aragog'
    assert config.interior_energetics.aragog.core_bc == 'energy_balance'

    hf_rows = _load_hf_rows()

    dirs = {
        'output': str(RUN_DIR),
        'spider_eos_dir': str(RUN_DIR / 'data' / 'spider_eos'),
    }

    interior_o = Interior_t(
        get_nlevb(config), spider_dir=None, eos_dir=config.interior_struct.eos_dir
    )
    # A resumed run always sets ic=2 (proteus.py), never 1: the fresh-run
    # IC path (ic=1) is for a t=0 start only.
    interior_o.ic = 2

    ratios = []
    for step in range(N_STEPS):
        hf_row = hf_rows[step]
        dt = float(hf_rows[step + 1]['Time']) - float(hf_rows[step]['Time'])
        # Matches AragogRunner.__init__, which sets this on every call.
        interior_o._spider_eos_dir = dirs.get('spider_eos_dir', '')

        before = set(glob.glob(str(traj_dir / 'traj_*.npz')))
        AragogRunner.setup_or_update_solver(config, hf_row, interior_o, dt, dirs)

        if step == 0:
            # Sanity gate on the standalone construction itself: if this
            # doesn't match the real run's own log line, something in the
            # config/Interior_t/resume wiring above is wrong and nothing
            # downstream should be trusted.
            s_mean = float(np.mean(interior_o._last_entropy))
            assert s_mean == pytest.approx(EXPECTED_S_MEAN_AT_RESUME, abs=S_MEAN_ATOL), (
                f'Standalone entropy restore at resume (S_mean={s_mean:.4f}) does not '
                f'match the replay run\'s own log line '
                f'(S_mean={EXPECTED_S_MEAN_AT_RESUME}); the standalone driver is not '
                'faithfully reproducing this run.'
            )

        # AragogRunner.__init__ builds a throwaway instance whose __init__
        # does nothing beyond this call sequence and the two attributes
        # below (_use_jax is False outside the research-only diffrax path);
        # constructing it directly here skips only AragogRunner's own
        # dt = compute_time_step(...) call, replaced by the real historical
        # dt read from the CSV above.
        runner = AragogRunner.__new__(AragogRunner)
        runner.aragog_solver = interior_o.aragog_solver
        runner._config = config
        runner._use_jax = False
        sim_time, output = runner.run_solver(hf_row, interior_o, dirs, write_data=False)

        expected_time = float(hf_rows[step + 1]['Time'])
        assert sim_time == pytest.approx(expected_time, rel=SIM_TIME_RTOL), (
            f'call {step}: sim_time={sim_time!r} does not reproduce the historical '
            f'Time={expected_time!r}; dt/hf_row wiring has drifted from the real run.'
        )

        after = set(glob.glob(str(traj_dir / 'traj_*.npz')))
        new_files = sorted(after - before)
        assert len(new_files) == 1, (
            f'call {step}: expected exactly one new trajectory dump, got {new_files} '
            '(a solver retry would create more than one; this replay is not expected '
            'to retry)'
        )
        z = np.load(new_files[0])
        t_yr, dt_s, P_F_cmb = z['t_yr'], z['dt_s'], z['P_F_cmb']
        assert len(t_yr) >= 2, f'call {step}: trajectory dump has < 2 accepted sub-steps'

        physical = float(np.sum(0.5 * (P_F_cmb[:-1] + P_F_cmb[1:]) * dt_s))
        booked = float(output['step_dE_F_cmb_J'])
        ratios.append((step, booked, physical))

    for step, booked, physical in ratios:
        if step == 0:
            # Known, expected transient tied to the resume itself (see
            # module docstring): both the fixed and flux-scaled atol
            # schemes fall back to the same fallback factor here, so this
            # call does not discriminate the fix. Only check the values
            # are finite, i.e. the call actually produced a result.
            assert np.isfinite(booked) and np.isfinite(physical)
            continue
        assert physical != 0.0, f'call {step}: physical CMB energy integral is exactly zero'
        ratio = booked / physical
        assert ratio == pytest.approx(1.0, abs=RATIO_TOL), (
            f'call {step}: booked/physical CMB energy ratio = {ratio!r} '
            f'(booked={booked!r} J, physical={physical!r} J), outside the '
            f'{RATIO_TOL:g} tolerance around 1.0'
        )
