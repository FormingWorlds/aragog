"""Unit tests for ``aragog.mesh._radius_for_mass_coordinate``.

The helper inverts the (monotonically increasing) mass coordinate
:math:`\\xi(r)` for the radius of each interior node on the uniform
mass grid. On a fresh initial-condition build the bracket
``[r_lo, r_hi]`` always straddles the root and the result is the
``brentq`` root. On a resumed run a loaded entropy field can shift the
EOS mass distribution so a near-boundary node's target falls just
outside ``[xi(r_lo), xi(r_hi)]``; a bare ``brentq`` raises
``ValueError`` there and aborts the resume, so the helper clamps the
node to the nearest in-domain endpoint instead.

These tests pin the normal (root-finding) branch, the two clamp
branches, the exact-endpoint edge, and the discrimination guard that a
bare ``brentq`` would raise on the clamp inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from aragog.mesh import _radius_for_mass_coordinate

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Monotonically increasing mass coordinate used throughout: xi(r) = r**3.
# Nonlinear so the recovered root is a non-trivial cube root, not an
# identity that any formula would satisfy.
R_LO = 2.0
R_HI = 10.0


def _xi(r: float) -> float:
    return r**3


@pytest.mark.unit
def test_root_branch_matches_brentq_and_is_interior():
    """Bracket straddles: return the brentq root, unchanged from the forward path.

    xi(r)=r**3, target=27 -> root=3.0, strictly inside (R_LO, R_HI). The
    result must equal the analytic cube root and the bare-brentq result
    (forward byte-identity), and must NOT coincide with either bracket
    endpoint (proving it solved rather than clamped).
    """
    xi_target = 27.0
    r = _radius_for_mass_coordinate(_xi, xi_target, R_LO, R_HI, node=1)

    # Matches the analytic root and the unchanged brentq call.
    r_brentq = brentq(lambda x: _xi(x) - xi_target, R_LO, R_HI, xtol=1.0, rtol=1e-12)
    np.testing.assert_allclose(r, 3.0, atol=1.0)  # xtol=1.0 m on the radius
    assert r == pytest.approx(r_brentq, abs=1e-9)

    # Discrimination guard: a clamp would have returned an endpoint; the
    # interior root is well away from both (|3 - 2| = 1, |3 - 10| = 7 >> xtol).
    assert abs(r - R_LO) > 0.5
    assert abs(r - R_HI) > 0.5


@pytest.mark.unit
def test_clamps_below_range_where_bare_brentq_would_raise():
    """Target below xi(R_LO): clamp to R_LO instead of aborting.

    xi(R_LO)=8, so target=4 < 8 gives f_lo=+4, f_hi=+996 (same sign). A
    bare brentq raises ValueError on this input; the helper must clamp
    to R_LO and keep the node in-domain.
    """
    xi_target = 4.0  # below xi(R_LO) = 8

    # Discrimination guard: the un-guarded call is exactly what aborted resume.
    with pytest.raises(ValueError, match='different signs'):
        brentq(lambda x: _xi(x) - xi_target, R_LO, R_HI, xtol=1.0, rtol=1e-12)

    r = _radius_for_mass_coordinate(_xi, xi_target, R_LO, R_HI, node=1)
    assert r == pytest.approx(R_LO)
    assert R_LO <= r <= R_HI


@pytest.mark.unit
def test_clamps_above_range():
    """Target above xi(R_HI): clamp to R_HI.

    xi(R_HI)=1000, so target=2000 gives f_lo=-1992, f_hi=-1000 (same
    sign, both negative). The monotone helper clamps to the upper
    endpoint, keeping the node strictly within the mesh bounds.
    """
    xi_target = 2000.0  # above xi(R_HI) = 1000
    r = _radius_for_mass_coordinate(_xi, xi_target, R_LO, R_HI, node=8)
    assert r == pytest.approx(R_HI)
    assert R_LO <= r <= R_HI


@pytest.mark.unit
def test_exact_lower_endpoint_returns_r_lo():
    """Target exactly at xi(R_LO): f_lo==0 returns R_LO without calling brentq.

    Boundary case: brentq with a zero at the bracket end is valid but the
    early return avoids relying on its endpoint handling.
    """
    xi_target = _xi(R_LO)  # == 8.0, f_lo == 0
    r = _radius_for_mass_coordinate(_xi, xi_target, R_LO, R_HI, node=1)
    assert r == pytest.approx(R_LO)


@pytest.mark.unit
def test_clamp_emits_warning(caplog):
    """The clamp path logs a warning naming the node and the out-of-range target.

    The warning is the only runtime signal that a resume hit the
    bracket-failure path, so it must fire (and only on that path).
    """
    import logging

    with caplog.at_level(logging.WARNING, logger='fwl.aragog.mesh'):
        _radius_for_mass_coordinate(_xi, 4.0, R_LO, R_HI, node=7)
    assert any('bracket failed for node 7' in rec.message for rec in caplog.records)

    # And the root branch stays silent.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.aragog.mesh'):
        _radius_for_mass_coordinate(_xi, 27.0, R_LO, R_HI, node=2)
    assert not any('bracket failed' in rec.message for rec in caplog.records)
