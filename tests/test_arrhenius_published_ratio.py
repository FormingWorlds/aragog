"""The solid viscosity of a built solver against a published Arrhenius case, with the rheology on."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.phase import PhaseParams, evaluate_phase
from aragog.rheology import SolidRheologyParams

from .conftest import EOS_DIR, entropy_eos_jax, needs_eos
from .test_rheology_solver_wiring import _make_abe_solid_params, _make_solver

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.physics_invariant,
    pytest.mark.reference_pinned,
    needs_eos,
]

T_1350_C, T_1600_C = 1623.15, 1873.15  # [K]


def test_the_viscosity_ratio_between_1350_and_1600_c_is_the_published_one():
    """E = 300 kJ/mol and 1e19 Pa s at 1350 C (Miyazaki & Korenaga 2022, Nature, p7), with V = 0
    chosen here. The unyielded diffusion-creep viscosity at these temperatures, in the NumPy phase
    and in the JAX phase a solver call evaluates, gives exp(E/R (1/1623.15 - 1/1873.15)) = 19.43;
    the paper says 20 times less viscous (p8)."""
    rheo = SolidRheologyParams(
        enabled=True, activation_energy=300.0e3, activation_volume=0.0, arrhenius_t_ref=T_1350_C
    )
    params = _make_abe_solid_params(rheo, number_of_nodes=12)
    params = replace(params, phase_solid=replace(params.phase_solid, viscosity=1.0e19))
    solver = _make_solver(params, identity_alpha=False)
    assert solver._rheology_enabled

    pressure = np.ravel(solver._P_stag_flat)
    eos_jax = entropy_eos_jax(EOS_DIR)
    params_jax = PhaseParams.from_parameters(params)
    assert params_jax.rheology.enabled

    def eta_and_t(temperature):
        s = np.array([solver.entropy_eos.invert_temperature(p, temperature) for p in pressure])
        solver.state.update(s, 0.0)
        phase = solver.state.phase_staggered
        jax_phase = evaluate_phase(eos_jax, params_jax, jnp.asarray(pressure), jnp.asarray(s))
        return (
            np.ravel(phase.eta_diff()),
            np.ravel(phase.temperature()),
            np.asarray(jax_phase.eta_diff),
            np.asarray(jax_phase.temperature),
        )

    eta_low, t_low, jax_eta_low, jax_t_low = eta_and_t(T_1350_C)
    eta_high, t_high, jax_eta_high, jax_t_high = eta_and_t(T_1600_C)
    # the inversion tolerance moves a cell off the target; the cell closest to both counts
    cell = np.argmin(np.abs(t_low - T_1350_C) + np.abs(t_high - T_1600_C))
    assert t_low[cell] == pytest.approx(T_1350_C, abs=0.05)
    assert t_high[cell] == pytest.approx(T_1600_C, abs=0.05)
    assert jax_t_low[cell] == pytest.approx(T_1350_C, abs=0.05)
    assert jax_t_high[cell] == pytest.approx(T_1600_C, abs=0.05)
    for low, high in ((eta_low, eta_high), (jax_eta_low, jax_eta_high)):
        assert low[cell] == pytest.approx(1.0e19, rel=1.0e-3)
        assert low[cell] / high[cell] == pytest.approx(19.43, abs=0.05)
