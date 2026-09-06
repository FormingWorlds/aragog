"""Tests for the ``separation_viscosity`` config option.

``EntropyPhaseEvaluator.relative_velocity`` (numpy) and
``aragog.jax.phase.relative_velocity`` (jax) select the gravitational-
separation drag viscosity from ``separation_viscosity``: ``'melt'``
uses the fixed single-phase liquid viscosity; ``'mixture'`` uses the
rheological-transition-blended bulk viscosity. Below phi_rheo the two
diverge sharply; well above phi_rheo the blend saturates to the melt
value and the two modes agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator

pytestmark = pytest.mark.unit

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)

from aragog.jax.phase import PhaseParams, relative_velocity  # noqa: E402


class _StubPhaseBoundaryEOS:
    """Exposes only the phase-boundary density lookup relative_velocity needs."""

    def _lookup_at_phase_boundary(self, prop, P, phase):
        assert prop == 'density'
        if phase == 'solid':
            return np.full_like(np.asarray(P, dtype=float), 4200.0)
        elif phase == 'melt':
            return np.full_like(np.asarray(P, dtype=float), 3900.0)
        raise ValueError(phase)


class _StubJaxEOS:
    """JAX counterpart of ``_StubPhaseBoundaryEOS``."""

    def _lookup_at_phase_boundary(self, field, pressure, phase):
        assert field == 'density'
        if phase == 'solid':
            return jnp.asarray(4200.0)
        elif phase == 'melt':
            return jnp.asarray(3900.0)
        raise ValueError(phase)


def _numpy_relative_velocity(mode, viscosity_val, viscosity_liquid=1e-1):
    ev = EntropyPhaseEvaluator(
        entropy_eos=_StubPhaseBoundaryEOS(),
        gravitational_acceleration=10.0,
        grain_size=1e-3,
        viscosity_liquid=viscosity_liquid,
        separation_viscosity=mode,
    )
    ev.pressure = np.array([1.0e9])
    ev._density = np.array([4000.0])
    ev._viscosity_val = np.array([viscosity_val])
    return float(ev.relative_velocity().item())


def test_modes_diverge_below_phi_rheo_and_agree_above_it():
    """Below phi_rheo the blended mixture viscosity is far from melt
    (near-solid); above phi_rheo it saturates to the melt value.
    """
    v_melt_below = _numpy_relative_velocity('melt', viscosity_val=1e21)
    v_mixture_below = _numpy_relative_velocity('mixture', viscosity_val=1e21)
    assert v_melt_below != pytest.approx(v_mixture_below, rel=1e-6)

    v_melt_above = _numpy_relative_velocity('melt', viscosity_val=1e-1)
    v_mixture_above = _numpy_relative_velocity('mixture', viscosity_val=1e-1)
    assert v_melt_above == pytest.approx(v_mixture_above, rel=1e-12)


def test_melt_mode_ignores_viscosity_val():
    """'melt' must use the fixed liquid viscosity, not the phi_rheo-blended value."""
    v_low = _numpy_relative_velocity('melt', viscosity_val=1e21, viscosity_liquid=1e-1)
    v_high = _numpy_relative_velocity('melt', viscosity_val=5e-3, viscosity_liquid=1e-1)
    assert v_low == pytest.approx(v_high, rel=1e-12)


def test_mixture_mode_ignores_viscosity_liquid():
    """'mixture' must use the phi_rheo-blended value, not the fixed liquid viscosity."""
    v_low_liquid = _numpy_relative_velocity(
        'mixture', viscosity_val=1e21, viscosity_liquid=1e-1
    )
    v_high_liquid = _numpy_relative_velocity(
        'mixture', viscosity_val=1e21, viscosity_liquid=5e-3
    )
    assert v_low_liquid == pytest.approx(v_high_liquid, rel=1e-12)


def test_numpy_rejects_invalid_separation_viscosity():
    with pytest.raises(ValueError):
        EntropyPhaseEvaluator(
            entropy_eos=_StubPhaseBoundaryEOS(),
            gravitational_acceleration=10.0,
            grain_size=1e-3,
            viscosity_liquid=1e-1,
            separation_viscosity='bogus',
        )


def test_jax_rejects_invalid_separation_viscosity():
    with pytest.raises(ValueError):
        PhaseParams(grain_size=1e-3, viscosity_liquid=1e-1, separation_viscosity='bogus')


def test_legacy_parser_rejects_invalid_separation_viscosity():
    from aragog.parser import _PhaseMixedParameters

    with pytest.raises(ValueError):
        _PhaseMixedParameters(
            latent_heat_of_fusion=4.0e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1.0e-3,
            separation_viscosity='bogus',
        )


@pytest.mark.parametrize('mode', ['melt', 'mixture'])
def test_numpy_jax_parity_in_both_modes(mode):
    """numpy and jax must select the same drag viscosity and agree bit-tight."""
    viscosity_val = 1e2

    ev = EntropyPhaseEvaluator(
        entropy_eos=_StubPhaseBoundaryEOS(),
        gravitational_acceleration=10.0,
        grain_size=1e-3,
        viscosity_liquid=1e-1,
        separation_viscosity=mode,
    )
    ev.pressure = np.array([1.0e9])
    ev._density = np.array([4000.0])
    ev._viscosity_val = np.array([viscosity_val])
    v_numpy = float(ev.relative_velocity().item())

    params = PhaseParams(grain_size=1e-3, viscosity_liquid=1e-1, separation_viscosity=mode)
    v_jax = float(
        relative_velocity(
            _StubJaxEOS(),
            params,
            P=jnp.asarray(1.0e9),
            density=jnp.asarray(4000.0),
            melt_fraction=jnp.asarray(0.5),
            gravity=jnp.asarray(10.0),
            viscosity=jnp.asarray(viscosity_val),
        )
    )

    assert v_jax == pytest.approx(v_numpy, rel=1e-10)


def _build_solver_params(separation_viscosity):
    """Build a ``Parameters`` tree for the const_properties EOS-free path.

    ``separation_viscosity=None`` omits the field from ``_PhaseMixedParameters``
    so its own default applies; any other value is passed through explicitly.
    Uses ``const_properties=True`` so ``EntropySolver`` needs no EntropyEOS
    table, keeping construction and ``initialize()`` fast enough for the
    unit tier.
    """
    from aragog.parser import (
        Parameters,
        _BoundaryConditionsParameters,
        _EnergyParameters,
        _InitialConditionParameters,
        _MeshParameters,
        _PhaseMixedParameters,
        _PhaseParameters,
        _SolverParameters,
    )

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc='quasi_steady',
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='radau',
        use_jax_jacobian=False,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=10,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm_kwargs = dict(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
        const_properties=True,
        const_rho=4000.0,
        const_Cp=1000.0,
        const_alpha=3.0e-5,
        const_cond=4.0,
        const_log10visc=2.0,
        const_T_ref=3000.0,
        const_S_ref=3000.0,
    )
    if separation_viscosity is not None:
        pm_kwargs['separation_viscosity'] = separation_viscosity
    pm = _PhaseMixedParameters(**pm_kwargs)
    sv = _SolverParameters(
        start_time=0.0, end_time=5.0, atol=1.0e-6, rtol=1.0e-6, tsurf_poststep_change=30.0
    )
    return Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        solver=sv,
    )


@pytest.mark.parametrize(
    'separation_viscosity,expected', [(None, 'melt'), ('melt', 'melt'), ('mixture', 'mixture')]
)
def test_entropy_solver_plumbs_separation_viscosity_to_evaluators(
    separation_viscosity, expected
):
    """``EntropySolver`` must pass ``separation_viscosity`` through to both evaluators.

    A ``getattr`` key typo in ``_initialize_internals`` would silently fall
    back to the default and go undetected without this test.
    """
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_solver_params(separation_viscosity)
    solver = EntropySolver(parameters, entropy_eos=None)
    solver.initialize()

    assert solver.state.phase_staggered._separation_viscosity == expected
    assert solver.state.phase_basic._separation_viscosity == expected


@pytest.mark.parametrize('separation_viscosity', ['melt', 'mixture'])
def test_entropy_solver_logs_separation_viscosity_mode(caplog, separation_viscosity):
    """``initialize()`` must log the selected drag-viscosity mode at INFO.

    Discriminator: a silently dropped or mis-formatted log call would
    remove the only run-time confirmation of which mode a coupled
    PROTEUS run actually used.
    """
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_solver_params(separation_viscosity)
    solver = EntropySolver(parameters, entropy_eos=None)
    with caplog.at_level('INFO'):
        solver.initialize()

    expected = f'Gravitational separation drag viscosity: {separation_viscosity}'
    assert expected in caplog.text
