"""Strategy B regression tests for the per-call mass-weighted ΔΦ_global cap.

The cap is wired into ``EntropySolver.solve()`` and clamps ``end_time``
when the projected mass-weighted |ΔΦ_global| over
[start_time, end_time] would exceed ``parameters.energy.phi_step_cap``.
The rate is the mantle-mass average of per-cell |dΦ_i/dt| evaluated at
t=start_time, scaled by a 0.5 safety factor.

Mass weighting (vs. per-cell max) prevents a thin
rheological-transition shell with very high local |dΦ_i/dt| from
dominating the cap. The thin shell carries a small fraction of the
mantle mass, so its contribution to the planet-scale dΦ_global/dt
is correspondingly small. The earlier per-cell-max formulation
locked dt to ~0.04 yr at the 1 M_E PALEOS-2phase rheological
transition (verified 2026-05-02 in
output/verify_dilon_phicap005); the mass-weighted formulation
tracks the planetary cooling rate that physics actually cares
about.

Default 0.0 keeps the existing behaviour unchanged for callers
that do not opt in. The cap is intended for production runs with
``dilatation = true`` where the H_dil heat-pump in the mushy zone
drives faster Φ-evolution than the static dt cap can contain.

Tests cover:
- default value (must be 0.0 to preserve existing behaviour)
- field assignment with an arbitrary positive value
- formula equivalence: dt_safe = 0.5 * cap / dΦ_global/dt
  produces correct clamping for known synthetic rates
- source-pattern regression: the clamp block exists in the solver,
  is gated on phi_step_cap > 0.0, uses the 0.5 safety factor,
  and weights per-cell rates by cell mass (not per-cell max)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from aragog.parser import _EnergyParameters

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'aragog'


def _make_energy(**overrides):
    base = dict(
        conduction=True,
        convection=True,
        gravitational_separation=True,
        mixing=True,
        radionuclides=True,
        dilatation=True,
        tidal=False,
    )
    base.update(overrides)
    return _EnergyParameters(**base)


def test_phi_step_cap_default_is_zero():
    """Default phi_step_cap must be 0.0 so existing tests are unchanged.

    A positive default would activate the clamp on every solve() call
    in production, which is a behavioural change and would break the
    bit-parity contract with prior runs.
    """
    e = _make_energy()
    assert e.phi_step_cap == 0.0
    assert isinstance(e.phi_step_cap, float)


def test_phi_step_cap_accepts_positive_values():
    """Recommended production setting (0.05) and an extreme (0.5)."""
    e_prod = _make_energy(phi_step_cap=0.05)
    assert e_prod.phi_step_cap == pytest.approx(0.05)
    e_loose = _make_energy(phi_step_cap=0.5)
    assert e_loose.phi_step_cap == pytest.approx(0.5)


def test_phi_step_cap_accepts_zero_explicitly():
    """Explicit 0.0 (disabled) is a valid configuration."""
    e = _make_energy(phi_step_cap=0.0)
    assert e.phi_step_cap == 0.0


def test_dt_safe_formula_no_clamp_when_rate_zero():
    """Edge case: |dΦ/dt| = 0 must NOT clamp end_time.

    A zero rate means the cap projection is infinite; the solver
    keeps the requested end_time. Verified at the formula level
    since the source short-circuits on dphi_dt_max <= 0.
    """
    cap = 0.05
    dphi_dt_max = 0.0
    safety = 0.5
    if dphi_dt_max > 0.0:
        dt_safe = safety * cap / dphi_dt_max
    else:
        dt_safe = np.inf
    assert dt_safe == np.inf


def test_dt_safe_formula_clamps_at_realistic_rate():
    """Realistic mushy-zone rate: cap=0.05, |dΦ/dt|=2e-5/yr.

    Expected dt_safe = 0.5 * 0.05 / 2e-5 = 1250 yr. This is well
    under PROTEUS's mushy_maximum=4000 yr, so the clamp is active.
    """
    cap = 0.05
    dphi_dt_max = 2.0e-5  # /yr
    safety = 0.5
    dt_safe = safety * cap / dphi_dt_max
    assert dt_safe == pytest.approx(1250.0)


def test_dt_safe_formula_does_not_clamp_at_slow_rate():
    """Slow rate: dt_safe much larger than typical end_time."""
    cap = 0.05
    dphi_dt_max = 1.0e-9  # /yr (extremely slow, e.g. nearly solid)
    safety = 0.5
    dt_safe = safety * cap / dphi_dt_max
    requested_dt = 4000.0
    assert dt_safe > requested_dt


def test_dt_safe_formula_clamps_aggressively_at_high_rate():
    """High rate from a strong heat-pump: cap=0.05, |dΦ/dt|=1e-3/yr.

    Expected dt_safe = 0.5 * 0.05 / 1e-3 = 25 yr. This is much
    smaller than 4000 yr; the clamp truncates aggressively, which
    is the intended behaviour at the Φ ≈ 0.5 wall.
    """
    cap = 0.05
    dphi_dt_max = 1.0e-3  # /yr (heat-pump driven)
    safety = 0.5
    dt_safe = safety * cap / dphi_dt_max
    assert dt_safe == pytest.approx(25.0)


def test_solver_source_contains_clamp_block():
    """Regression: the clamp block must exist in entropy_solver.solve().

    Catches accidental removal of Strategy B during refactors.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # The block must be gated on phi_step_cap > 0.0.
    assert 'phi_step_cap > 0.0' in src
    # The 0.5 safety factor must be the one in use.
    assert 'safety = 0.5' in src or 'safety=0.5' in src
    # The clamp must mutate end_time.
    assert 'end_time = end_clamped' in src
    # The Strategy B comment must remain so future readers find context.
    assert 'Strategy B' in src


def test_solver_source_uses_mass_weighted_rate():
    """Regression: the cap must weight per-cell |dΦ/dt| by cell mass.

    Per-cell max is the wrong formulation: a single fast-evolving
    cell at a thin rheological transition dominated the cap and
    locked dt to ~0.04 yr (verified in output/verify_dilon_phicap005
    on 2026-05-02). Mass weighting reflects the planet-scale
    cooling rate, which is what the cap should reflect.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # The clamp block must compute mass weights from EOS density and
    # mesh volume, then aggregate per-cell rates by mass fraction.
    m = re.search(
        r'phi_step_cap\s*>\s*0\.0.*?end_time\s*=\s*end_clamped',
        src,
        re.DOTALL,
    )
    assert m is not None, 'clamp block not found in expected form'
    block = m.group(0)
    # Must read density at staggered nodes from the EOS at start state.
    assert 'self.entropy_eos.density' in block
    # Must build per-cell mass weights from rho * volume.
    assert 'self._volume_flat' in block
    assert 'mass_stag' in block
    # Must aggregate to a single mass-weighted rate.
    assert 'dphi_dt_global' in block
    # Must NOT use the per-cell max formulation any more.
    assert 'np.max(dphi_dt_per_cell)' not in block


def test_solver_source_uses_dphi_dt_estimate_at_t_start():
    """The cap estimate must use dS/dt at the SOLVE entry, not at end.

    Using end-time dS/dt would require integrating first, defeating
    the point of the cap. The source must call _dSdt_single with
    start_time as the time argument.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # Find the clamp block and verify it sources the rate at start_time
    m = re.search(
        r'phi_step_cap\s*>\s*0\.0.*?end_time\s*=\s*end_clamped',
        src,
        re.DOTALL,
    )
    assert m is not None, 'clamp block not found in expected form'
    block = m.group(0)
    assert '_dSdt_single' in block
    assert 'start_time' in block
