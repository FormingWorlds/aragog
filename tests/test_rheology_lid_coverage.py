"""Tests for stagnant lid rheology closures, mesh stretching failure, and solver preconditions.

Exercises stress_closure mode validation, compute_effective_viscosity edge cases
and lid weighting, stretched_unit_grid convergence failure rejection, and
EntropySolver uninitialised state rejection.
"""

from __future__ import annotations

import numpy as np
import pytest

import aragog.rheology_lid as rlid
from aragog.mesh.stretching import stretched_unit_grid
from aragog.solver.entropy_solver import EntropySolver


@pytest.mark.unit
def test_stress_closure_contracts():
    """stress_closure validates mode and returns strain rate for local closure."""
    sr = rlid.stress_closure('local', viscous_velocity=2.0, mixing_length=100.0)
    assert np.isclose(sr, 0.01)

    with pytest.raises(ValueError, match="mixing_length must be provided when mode='local'"):
        rlid.stress_closure('local', viscous_velocity=1.0, mixing_length=None)

    with pytest.raises(ValueError, match="Unknown stress closure mode 'global'"):
        rlid.stress_closure('global', viscous_velocity=1.0)

    with pytest.raises(ValueError, match="Mode 'lid' uses boundary-layer stress closure"):
        rlid.stress_closure('lid', viscous_velocity=1.0)

    with pytest.raises(ValueError, match="Unknown stress closure mode 'unknown'"):
        rlid.stress_closure('unknown', viscous_velocity=1.0)


@pytest.mark.unit
def test_compute_effective_viscosity_contracts():
    """compute_effective_viscosity rejects invalid modes and blends with w_lid."""
    with pytest.raises(ValueError, match="Unknown stress_closure_mode 'global'"):
        rlid.compute_effective_viscosity(eta_diff=1.0e20, stress_closure_mode='global')

    with pytest.raises(
        ValueError, match="mixing_length and unyielded_velocity required for 'local' mode"
    ):
        rlid.compute_effective_viscosity(
            eta_diff=1.0e20, stress_closure_mode='local', unyielded_velocity=None
        )

    # Test w_lid blending between diff (w=0) and lid plastic yield (w=1)
    eta_diff = np.array([1.0e20, 1.0e20])
    eta_blend = rlid.compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_d=np.array([2.0e5, 2.0e5]),
        tau_y_lid=np.array([1.0e5, 1.0e5]),
        v_i=1.0,
        delta_rh=100.0,
        w_lid=np.array([0.0, 1.0]),
    )
    assert np.isclose(eta_blend[0], 1.0e20, rtol=1e-12)
    assert np.isclose(eta_blend[1], 5.0e19, rtol=1e-12)


@pytest.mark.unit
def test_mesh_stretching_convergence_failure():
    """stretched_unit_grid raises ValueError when two-sided solve fails to converge."""
    with pytest.raises(ValueError, match='two-sided mesh stretching did not converge'):
        stretched_unit_grid(5, first=1.0e-12, last=0.2)


@pytest.mark.unit
def test_entropy_solver_preconditions():
    """EntropySolver.solve raises RuntimeError when initial entropy is not set."""
    solver = EntropySolver.__new__(EntropySolver)
    with pytest.raises(RuntimeError, match='Initial entropy is not set'):
        solver.solve()
