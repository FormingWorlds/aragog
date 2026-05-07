"""Unit tests for ``aragog.output.diagnostics``.

Standalone diagnostic functions used by the PROTEUS wrapper to
post-process ``SolverOutput`` for the helpfile CSV. Failures here
silently corrupt run-summary numbers (global melt fraction,
rheological-front depth, total enthalpy) without breaking the
solver itself, so dedicated tests are essential.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from aragog.output.diagnostics import (
    melt_fraction_global,
    rheological_front,
    volume_average,
)

pytestmark = pytest.mark.unit


def _fake_mesh(N: int = 10) -> SimpleNamespace:
    """Build a minimal mesh stub with the attributes diagnostics
    needs: ``mesh.basic.volume`` (per-cell), ``total_volume``, and
    ``radii``.

    Earth-like radii so the rheological-front test has a non-trivial
    ratio that distinguishes correct (depth/R_surf) from wrong
    formulas (e.g. r/R_surf without the surface subtraction).
    """
    R_cmb, R_surf = 3.480e6, 6.371e6
    radii = np.linspace(R_cmb, R_surf, N + 1)  # basic nodes
    # Cell volumes: shells between basic nodes
    volumes = (4.0 / 3.0) * np.pi * (radii[1:] ** 3 - radii[:-1] ** 3)
    total_volume = float(volumes.sum())
    return SimpleNamespace(
        basic=SimpleNamespace(
            volume=volumes,
            total_volume=total_volume,
            radii=radii,
        ),
    )


# ---- volume_average --------------------------------------------------------


def test_volume_average_uniform_quantity_returns_input_value():
    """Property: a constant field volume-averages to its constant.

    Discriminator: a regression that summed without dividing by
    total_volume would yield total_volume * value, off by ~10^21.
    """
    mesh = _fake_mesh(N=20)
    constant_value = 7.5
    quantity = np.full_like(mesh.basic.volume, constant_value)
    out = volume_average(mesh, quantity)
    assert float(out) == pytest.approx(constant_value, rel=1e-12)


def test_volume_average_weighted_correctly_for_step_profile():
    """A two-region step profile averages to the volume-weighted
    mean of the two values. Picked region values 100 and 1; the
    mid-cell split makes the answer non-trivial (not a simple mean).

    Edge case: discriminates against a bug that uses cell counts
    instead of cell volumes.
    """
    mesh = _fake_mesh(N=10)
    quantity = np.zeros_like(mesh.basic.volume)
    quantity[:5] = 100.0
    quantity[5:] = 1.0
    expected = np.dot(quantity, mesh.basic.volume) / mesh.basic.total_volume
    out = volume_average(mesh, quantity)
    assert float(out) == pytest.approx(float(expected), rel=1e-12)
    # Also non-trivially asymmetric: the upper-half cells have
    # larger volumes (4 pi r^2 is monotonic), so the mean should
    # be biased toward 1, not 100.
    assert float(out) < 50.0


# ---- melt_fraction_global --------------------------------------------------


def test_melt_fraction_global_mixed_phase_picks_last_column():
    """For phase_mode='mixed' the function uses the last column of
    a 2D ``(nodes, times)`` array. Earlier columns must NOT affect
    the result; this discriminates against a regression that
    averaged across the time axis.
    """
    mesh = _fake_mesh(N=10)
    melt = np.zeros((mesh.basic.volume.size, 3))
    melt[:, 0] = 1.0  # earlier time, fully molten
    melt[:, 1] = 0.5
    melt[:, 2] = 0.0  # final time, fully solid
    out = melt_fraction_global(mesh, melt, phase_mode='mixed')
    assert float(out) == pytest.approx(0.0, abs=1e-12)


def test_melt_fraction_global_composite_uses_last_column_too():
    """``composite`` phase mode uses the same code path. Catches a
    regression that branched on 'mixed' alone and silently dropped
    composite-mode planets.
    """
    mesh = _fake_mesh(N=10)
    melt = np.full((mesh.basic.volume.size, 2), 0.7)
    out = melt_fraction_global(mesh, melt, phase_mode='composite')
    assert float(out) == pytest.approx(0.7, rel=1e-12)


def test_melt_fraction_global_single_phase_returns_input_unchanged():
    """For phase_mode='solid' or 'liquid' the function returns the
    input scalar unchanged (no volume averaging). Edge case: a
    regression that always volume-averaged would crash here on a
    scalar input.
    """
    mesh = _fake_mesh(N=10)
    out_solid = melt_fraction_global(mesh, 0.0, phase_mode='solid')
    out_liquid = melt_fraction_global(mesh, 1.0, phase_mode='liquid')
    assert out_solid == 0.0
    assert out_liquid == 1.0


# ---- rheological_front -----------------------------------------------------


def test_rheological_front_fully_molten_returns_inner_radius():
    """When phi_global > 0.99 (essentially magma ocean), the
    rheological front is at the inner radius (CMB), so the
    dimensionless depth is (R_surf - R_cmb) / R_surf.

    For Earth defaults: (6.371 - 3.480) / 6.371 ~ 0.4537. This
    discriminates against a regression that swapped the sign so
    the answer would be R_cmb/R_surf ~ 0.546 instead.
    """
    mesh = _fake_mesh(N=10)
    melt_basic = np.full((mesh.basic.radii.size, 1), 1.0)
    out = rheological_front(
        mesh,
        melt_basic,
        rheological_transition_melt_fraction=0.4,
        phi_global=1.0,
    )
    expected = (mesh.basic.radii[-1] - mesh.basic.radii[0]) / mesh.basic.radii[-1]
    assert float(out) == pytest.approx(float(expected), rel=1e-12)


def test_rheological_front_fully_solid_returns_zero():
    """When phi_global < 0.01 (essentially solid), the front is at
    the surface so the dimensionless depth is 0.
    """
    mesh = _fake_mesh(N=10)
    melt_basic = np.full((mesh.basic.radii.size, 1), 0.0)
    out = rheological_front(
        mesh,
        melt_basic,
        rheological_transition_melt_fraction=0.4,
        phi_global=0.005,
    )
    assert float(out) == pytest.approx(0.0, abs=1e-12)


def test_rheological_front_intermediate_finds_crossing_node():
    """In the mushy regime the front is at the node whose melt
    fraction is closest to the rheological transition value. We
    plant a stepped profile crossing 0.4 at a known basic node and
    verify the dimensionless depth matches.

    Discriminator: a regression that took ``argmax`` instead of
    ``argmin(abs(...))`` would pick the node with the LARGEST melt
    fraction, returning the wrong depth.
    """
    mesh = _fake_mesh(N=10)
    melt_basic = np.zeros((mesh.basic.radii.size, 1))
    # planted profile: 0.0 in upper half, 0.4 at index 4, 0.8 lower half
    melt_basic[:4, 0] = 0.0
    melt_basic[4, 0] = 0.4  # transition node
    melt_basic[5:, 0] = 0.8
    out = rheological_front(
        mesh,
        melt_basic,
        rheological_transition_melt_fraction=0.4,
        phi_global=0.5,
    )
    expected_r = mesh.basic.radii[4]
    expected_depth = (mesh.basic.radii[-1] - expected_r) / mesh.basic.radii[-1]
    assert float(out) == pytest.approx(float(expected_depth), rel=1e-12)
