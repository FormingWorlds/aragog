"""The solid viscosity against a published Arrhenius case with the rheology on."""

from __future__ import annotations

import math

import pytest
from scipy.constants import R

from aragog.rheology import SolidRheologyParams, compute_arrhenius_viscosity

pytestmark = [pytest.mark.unit, pytest.mark.physics_invariant, pytest.mark.reference_pinned]


def test_the_viscosity_ratio_between_1350_and_1600_c_is_the_published_one():
    """E = 300 kJ/mol and 1e19 Pa s at 1350 C (Miyazaki & Korenaga 2022, Nature, p7), with V = 0
    chosen here: the law gives exp(E/R (1/1623.15 - 1/1873.15)) = 19.43 between 1350 and 1600 C,
    and the paper says 20 times less viscous (p8)."""
    rheo = SolidRheologyParams(
        enabled=True,
        activation_energy=300.0e3,
        activation_volume=0.0,
        arrhenius_t_ref=1350.0 + 273.15,
    )

    def eta(t_celsius):
        return float(
            compute_arrhenius_viscosity(
                t_celsius + 273.15,
                0.0,
                viscosity_solid=1.0e19,
                arrhenius_t_ref=rheo.arrhenius_t_ref,
                activation_energy=rheo.activation_energy,
                activation_volume=rheo.activation_volume,
                viscosity_max_log10=rheo.viscosity_max_log10,
            )
        )

    assert eta(1350.0) == pytest.approx(1.0e19, rel=1.0e-12)
    assert eta(1350.0) / eta(1600.0) == pytest.approx(19.43, abs=0.01)
    assert eta(1350.0) / eta(1600.0) == pytest.approx(20.0, abs=1.0)
    assert eta(1350.0) / eta(1600.0) == pytest.approx(
        math.exp(300.0e3 / R * (1.0 / 1623.15 - 1.0 / 1873.15)), rel=1.0e-9
    )
