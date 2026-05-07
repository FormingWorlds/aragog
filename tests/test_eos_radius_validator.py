"""Validator tolerance tests for ``_validate_eos_radius_range``.

The validator checks that the EOS table's radius array stays inside
the mesh's [inner, outer] bounds. The original implementation used
strict ``<``/``>``, which trips on single-ULP drift between an EOS
table loaded from disk (frozen at the launch-time mesh) and the
freshly-recomputed Zalmoxis bounds on resume. This test set pins
the relative-tolerance behaviour of the post-fix validator:

- Bit-equal bounds pass (the resume-from-attractor case).
- Sub-mm drift well within the relative tolerance passes.
- Drift larger than the absolute floor still raises.
- Real configuration mistakes (wrong planet radius) still raise.
- Span/monotonicity checks are unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from aragog.solver.entropy_solver import _validate_eos_radius_range

pytestmark = pytest.mark.unit


def _mesh_params(eos_radius, inner, outer):
    return SimpleNamespace(
        eos_radius=np.asarray(eos_radius, dtype=float),
        inner_radius=float(inner),
        outer_radius=float(outer),
    )


def test_bit_equal_bounds_pass():
    """The resume-from-attractor case: EOS bounds match mesh exactly."""
    inner, outer = 3.407e6, 6.954e6
    er = np.linspace(inner, outer, 280)
    _validate_eos_radius_range(_mesh_params(er, inner, outer))


def test_single_ulp_drift_passes():
    """Single-ULP drift on each end (the actual exp1 failure pattern)."""
    inner, outer = 3.407e6, 6.954e6
    er = np.linspace(inner, outer, 280)
    er[0] = np.nextafter(inner, -np.inf)  # one ULP below inner
    er[-1] = np.nextafter(outer, np.inf)  # one ULP above outer
    _validate_eos_radius_range(_mesh_params(er, inner, outer))


def test_sub_mm_drift_within_tolerance_passes():
    """Drift up to ~1 mm should pass; this is well within the 1 m floor."""
    inner, outer = 3.407e6, 6.954e6
    er = np.linspace(inner - 1.0e-3, outer + 1.0e-3, 280)
    _validate_eos_radius_range(_mesh_params(er, inner, outer))


def test_drift_above_floor_raises():
    """Drift > 1 m absolute floor triggers the validator (real config error).

    Plausible-bug discrimination: the validator must not pass a
    100 m mismatch on either bound, which would be a real Zalmoxis
    re-mesh that drifted outside the solver shell rather than a
    bit-level resume artefact. Tested at both inner and outer.
    """
    inner, outer = 3.407e6, 6.954e6
    # below-inner by 100 m
    er_low = np.linspace(inner - 100.0, outer, 280)
    with pytest.raises(ValueError, match='inconsistent with mesh bounds'):
        _validate_eos_radius_range(_mesh_params(er_low, inner, outer))
    # above-outer by 100 m
    er_high = np.linspace(inner, outer + 100.0, 280)
    with pytest.raises(ValueError, match='inconsistent with mesh bounds'):
        _validate_eos_radius_range(_mesh_params(er_high, inner, outer))


def test_relative_tolerance_scales_with_span():
    """For very large planets, atol = 1e-9 × span, not the 1 m floor.

    Discriminates: the post-fix code does
    ``atol = max(1.0, 1e-9 * max(span_mesh, 1.0))``. For a 7e6 m
    mantle, atol = 1.0 m (floor wins). For a 1e15 m hypothetical
    structure, atol = 1e6 m (relative wins). This guards against
    a future regression where someone removes the floor or the
    relative term.
    """
    # mantle span 1e15 m ⇒ atol = 1e6 m; a 0.5e6 drift must pass
    inner, outer = 0.0, 1.0e15
    er = np.linspace(inner - 0.5e6, outer + 0.5e6, 280)
    _validate_eos_radius_range(_mesh_params(er, inner, outer))
    # but a 2e6 drift on the same span must raise
    er_bad = np.linspace(inner - 2.0e6, outer, 280)
    with pytest.raises(ValueError, match='inconsistent with mesh bounds'):
        _validate_eos_radius_range(_mesh_params(er_bad, inner, outer))


def test_short_span_eos_raises_unchanged():
    """The 0.75 × span_mesh check is preserved by the patch."""
    inner, outer = 3.407e6, 6.954e6
    span_mesh = outer - inner
    # span_eos = 0.5 × span_mesh, well below the 0.75 threshold
    er = np.linspace(inner, inner + 0.5 * span_mesh, 280)
    with pytest.raises(ValueError, match='inconsistent with mesh bounds'):
        _validate_eos_radius_range(_mesh_params(er, inner, outer))


def test_non_monotonic_raises_unchanged():
    """Non-monotonic radii still fail with the dedicated message."""
    inner, outer = 3.407e6, 6.954e6
    er = np.linspace(inner, outer, 280)
    er[100] = er[50]  # break monotonicity
    with pytest.raises(ValueError, match='not monotonically increasing'):
        _validate_eos_radius_range(_mesh_params(er, inner, outer))


def test_size_below_two_is_noop():
    """Edge case: a length-0 or length-1 array short-circuits cleanly."""
    inner, outer = 3.407e6, 6.954e6
    _validate_eos_radius_range(_mesh_params(np.array([]), inner, outer))
    _validate_eos_radius_range(_mesh_params(np.array([5.0e6]), inner, outer))


def test_inverted_mesh_does_not_silently_pass():
    """Physically-impossible mesh (outer < inner) should not validate as OK.

    With span_mesh = -1.0e6, the relative-tolerance term collapses
    via ``max(span_mesh, 1.0) = 1.0`` to atol = 1.0 m, so the
    bounds check still raises if er[0] is significantly outside
    the (degenerate) interval. This guards against a regression
    where atol could grow negative on an inverted mesh.
    """
    inner, outer = 6.954e6, 3.407e6  # inverted
    er = np.linspace(3.407e6, 6.954e6, 280)
    # er[-1] = 6.954e6 > outer (3.407e6) by 3.5e6 m, far above any atol
    with pytest.raises(ValueError, match='inconsistent with mesh bounds'):
        _validate_eos_radius_range(_mesh_params(er, inner, outer))
