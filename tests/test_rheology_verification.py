"""Algebraic pins for the solid-state rheology reference parameters.

Module under test: ``aragog.rheology`` (NumPy) and, when the optional JAX
stack is installed, ``aragog.jax.phase`` (the float64-parity kernels used
by the CVODE and research runners).

Invariants exercised:

- ``arrhenius_t_ref`` is the temperature at which the Arrhenius diffusion
  creep viscosity collapses to the reference viscosity ``viscosity_solid``
  (analytical limit ``eta_diff(T = t_ref, P = 0) = eta_0``).
- ``yield_stress_max`` is a hard ceiling on the Byerlee yield stress
  (boundedness ``tau_y <= yield_stress_max`` for every pressure).
- The NumPy and JAX kernels agree to float64 parity on both quantities.

The pins carry ``reference_pinned`` because each is an analytical limit or
a cross-implementation cross-check, and ``physics_invariant`` because each
asserts positivity, a fixed-point limit, or boundedness.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import compute_yield_stress, eta_diff

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Reference rheology point used across the pins. Olivine-like diffusion
# creep: E_a = 300 kJ/mol, V_a = 5 cm^3/mol, eta_0 = 1e21 Pa s at
# T_ref = 1600 K. These are the aragog defaults and let the closed-form
# exponent be evaluated by hand.
_ETA0 = 1.0e21
_E_A = 300.0e3
_V_A = 5.0e-6
_T_REF = 1600.0
_R = 8.314462618

# Byerlee reference: cohesion 50 MPa, friction 0.6, ceiling 500 MPa.
_TAU_C = 50.0e6
_TAU_MU = 0.6
_TAU_MAX = 500.0e6


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_arrhenius_viscosity_collapses_to_reference_at_t_ref():
    """At the reference temperature with no pressure term, the Arrhenius
    diffusion creep viscosity equals the reference viscosity exactly.

    This is the defining property of ``arrhenius_t_ref``: the exponent
    ``(E_a + P V_a)/(R T) - E_a/(R T_ref)`` vanishes at ``T = T_ref`` and
    ``P = 0``, so ``eta_diff = eta_0``. Away from the reference the
    viscosity must move by orders of magnitude, which is what makes the
    reference temperature a meaningful pin rather than a free offset.
    """
    eta_at_ref = eta_diff(_T_REF, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    # Analytical limit: exp(0) = 1, so eta == eta_0 to machine precision.
    assert eta_at_ref == pytest.approx(_ETA0, rel=1e-12)

    # Discrimination guard 1 (reference-temperature swap): if arrhenius_t_ref
    # were wrongly bound to the evaluation temperature, cooling to 0.9*T_ref
    # would still return eta_0. The correct law makes it far stiffer.
    cold = 0.9 * _T_REF
    eta_cold = eta_diff(cold, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    expected_cold = _ETA0 * np.exp(_E_A / _R * (1.0 / cold - 1.0 / _T_REF))
    assert eta_cold == pytest.approx(expected_cold, rel=1e-12)
    # The wrong-reference (t_ref == cold) result would be eta_0; the correct
    # one is stiffer by more than a decade. Pin that separation.
    assert eta_cold > 10.0 * _ETA0

    # Discrimination guard 2 (sign/monotonicity): heating below and above the
    # reference must bracket eta_0, and viscosity is strictly decreasing in T.
    eta_hot = eta_diff(1.1 * _T_REF, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    assert eta_hot < _ETA0 < eta_cold
    assert eta_hot > 0.0

    # Edge case (high-temperature bounded limit): as T grows without bound the
    # temperature term vanishes and, at P = 0, the exponent floors at the
    # negative constant -E_a/(R T_ref). The viscosity therefore saturates to a
    # finite, strictly positive floor eta_0 * exp(-E_a/(R T_ref)) rather than
    # collapsing to zero.
    # T = 1e15 K so the residual E_a/(R T) ~ 4e-11 term sits below rtol.
    eta_hot_limit = eta_diff(1.0e15, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    floor = _ETA0 * np.exp(-_E_A / (_R * _T_REF))
    assert eta_hot_limit == pytest.approx(floor, rel=1e-9)
    assert 0.0 < eta_hot_limit < _ETA0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_arrhenius_pressure_term_uses_reference_at_t_ref():
    """At ``T = T_ref`` the temperature term cancels, leaving a pure
    pressure activation ``eta = eta_0 exp(P V_a / (R T_ref))``.

    Pins the fact that ``arrhenius_t_ref`` sits in the temperature term
    only: raising pressure at the reference temperature must stiffen the
    mantle through ``V_a`` while the ``E_a`` contribution stays cancelled.
    """
    p_test = 100.0e9  # 100 GPa, deep-mantle scale
    eta_p = eta_diff(_T_REF, p_test, _ETA0, _E_A, _V_A, _T_REF, _R)
    expected = _ETA0 * np.exp(p_test * _V_A / (_R * _T_REF))
    assert eta_p == pytest.approx(expected, rel=1e-12)

    # Discrimination guard (missing pressure term): dropping P*V_a would give
    # eta_0. The activation-volume term raises it well above that.
    assert eta_p > 1.5 * _ETA0
    # Positivity and the P = 0 fixed point.
    assert eta_p > 0.0
    assert eta_diff(_T_REF, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R) == pytest.approx(
        _ETA0, rel=1e-12
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_yield_stress_saturates_at_maximum():
    """The Byerlee yield stress rises linearly with pressure then saturates
    at ``yield_stress_max``.

    Pins both branches of ``min(c + mu P, tau_max)``: the linear cohesion
    plus friction law below the knee, and the exact ceiling above it. The
    knee sits at ``P = (tau_max - c) / mu``.
    """
    p_knee = (_TAU_MAX - _TAU_C) / _TAU_MU  # 7.5e8 Pa

    # Below the knee: exact linear law, well clear of the ceiling.
    p_lin = 0.5 * p_knee
    tau_lin = compute_yield_stress(p_lin, _TAU_C, _TAU_MU, _TAU_MAX)
    assert tau_lin == pytest.approx(_TAU_C + _TAU_MU * p_lin, rel=1e-12)
    assert tau_lin < _TAU_MAX  # scale guard: linear branch stays sub-ceiling

    # Far above the knee: clamps to the ceiling exactly.
    p_high = 1.0e12  # 1 TPa, deep interior scale
    tau_high = compute_yield_stress(p_high, _TAU_C, _TAU_MU, _TAU_MAX)
    assert tau_high == pytest.approx(_TAU_MAX, rel=1e-12)
    # Discrimination guard: the unclamped Byerlee value at this pressure is
    # ~1200x the ceiling, so a dropped clamp would fail this by a wide margin.
    unclamped = _TAU_C + _TAU_MU * p_high
    assert unclamped > 1000.0 * _TAU_MAX

    # Boundedness invariant across a decade-spanning pressure sweep: the yield
    # stress never exceeds the ceiling and is strictly positive everywhere.
    p_sweep = np.logspace(5, 13, 40)
    tau_sweep = compute_yield_stress(p_sweep, _TAU_C, _TAU_MU, _TAU_MAX)
    assert np.all(tau_sweep <= _TAU_MAX + 1.0)  # +1 Pa float slack on 5e8
    assert np.all(tau_sweep > 0.0)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_yield_stress_knee_is_continuous():
    """The linear and saturated branches meet continuously at the knee.

    Straddling ``P = (tau_max - c)/mu`` must give the ceiling on the high
    side and just below it on the low side, with no jump across the knee.
    """
    p_knee = (_TAU_MAX - _TAU_C) / _TAU_MU
    tau_below = compute_yield_stress(0.999 * p_knee, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_at = compute_yield_stress(p_knee, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_above = compute_yield_stress(1.001 * p_knee, _TAU_C, _TAU_MU, _TAU_MAX)

    # Continuity: the value at the knee equals the ceiling and the linear
    # extrapolation from just below lands on it within the friction slope.
    assert tau_at == pytest.approx(_TAU_MAX, rel=1e-12)
    assert tau_below == pytest.approx(_TAU_C + _TAU_MU * 0.999 * p_knee, rel=1e-12)
    assert tau_above == pytest.approx(_TAU_MAX, rel=1e-12)
    # Monotone non-decreasing and bounded across the knee (no overshoot).
    assert tau_below <= tau_at <= tau_above + 1.0
    assert tau_above <= _TAU_MAX + 1.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_numpy_jax_rheology_float64_parity():
    """The NumPy and JAX rheology kernels agree to float64 parity on the
    reference viscosity and the saturated yield stress.

    Cross-implementation cross-check: the CVODE and research runners use the
    JAX kernels, the entropy solver uses the NumPy ones. A drift in either
    ``arrhenius_t_ref`` or ``yield_stress_max`` between the two backends
    would break the coupled-run parity this pins.
    """
    pytest.importorskip('jax')
    pytest.importorskip('equinox')
    import jax.numpy as jnp

    from aragog.jax.phase import compute_arrhenius_viscosity
    from aragog.jax.phase import compute_yield_stress as jax_tau

    t_test = 0.85 * _T_REF  # off the reference so the exponent is nonzero
    p_test = 40.0e9

    eta_np = eta_diff(t_test, p_test, _ETA0, _E_A, _V_A, _T_REF, _R)
    eta_jx = float(
        compute_arrhenius_viscosity(
            jnp.asarray(t_test),
            jnp.asarray(p_test),
            _ETA0,
            _E_A,
            _V_A,
            T_ref=_T_REF,
            R=_R,
        )
    )
    # rtol=1e-9: float64 exp round-off between numpy and XLA, tighter than
    # any physically meaningful viscosity difference.
    assert eta_jx == pytest.approx(eta_np, rel=1e-9)
    assert eta_jx > 0.0  # sign guard

    # Yield stress parity at a clamped pressure. The JAX kernel applies the
    # softplus-regularized ceiling via a separate min against yield_stress_max;
    # reproduce that here and confirm both saturate at the same ceiling.
    p_clamp = 1.0e12
    tau_np = compute_yield_stress(p_clamp, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_jx = float(jnp.minimum(jax_tau(jnp.asarray(p_clamp), _TAU_C, _TAU_MU), _TAU_MAX))
    assert tau_jx == pytest.approx(tau_np, rel=1e-9)
    assert tau_jx == pytest.approx(_TAU_MAX, rel=1e-9)

    # Yield stress parity in the linear branch.
    p_lin = 0.5 * (_TAU_MAX - _TAU_C) / _TAU_MU
    tau_np_lin = compute_yield_stress(p_lin, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_jx_lin = float(jnp.minimum(jax_tau(jnp.asarray(p_lin), _TAU_C, _TAU_MU), _TAU_MAX))
    assert tau_jx_lin == pytest.approx(tau_np_lin, rel=1e-9)


def test_viscosity_max_log10_clipping():
    from aragog.rheology import eta_diff, eta_eff

    T = 40.0
    P = 0.0
    eta_d = eta_diff(
        temperature=T,
        pressure=P,
        viscosity_solid=1.0e21,
        activation_energy=300e3,
        activation_volume=5e-6,
        t_ref=1600.0,
        r_gas=8.314462618,
        viscosity_max_log10=40.0,
    )

    assert eta_d == pytest.approx(1e40, rel=1e-12)

    # Check eta_eff
    eta_e = eta_eff(
        visc_diff=eta_d,
        tau_y=50e6,
        strain_rate=1e-15,
    )

    assert np.isfinite(eta_e)


def test_lid_base_mode_and_stress_closure_validation():
    """Verify validation of lid_base_mode, stress_closure_mode, and the N10 invalid combination."""
    import pytest

    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.jax.phase import PhaseParams
    from aragog.parser import _PhaseMixedParameters, _PhaseParameters

    # 1. JAX PhaseParams stress_closure_mode validation
    with pytest.raises(ValueError, match="Unknown stress_closure_mode 'invalid'"):
        PhaseParams(stress_closure_mode='invalid')

    # 2. JAX PhaseParams lid_base_mode validation
    with pytest.raises(ValueError, match="Unknown lid_base_mode 'invalid'"):
        PhaseParams(lid_base_mode='invalid')

    # 3. JAX PhaseParams rejects enabled=True with activation_energy=0 and lid_base_mode='rheological'
    with pytest.raises(ValueError, match='lid_base_mode.*activation_energy'):
        PhaseParams(enabled=True, activation_energy=0.0, lid_base_mode='rheological')

    # 4. JAX PhaseParams accepts enabled=False with activation_energy=0 and lid_base_mode='rheological'
    p = PhaseParams(enabled=False, activation_energy=0.0, lid_base_mode='rheological')
    assert p.lid_base_mode == 'rheological'

    # 5. EntropyPhaseEvaluator lid_base_mode validation
    with pytest.raises(ValueError, match="Unknown lid_base_mode 'invalid'"):
        EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=None,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.1,
            viscosity_solid=1e21,
            viscosity_liquid=1e2,
            grain_size=1e-3,
            thermal_conductivity_solid=4.0,
            thermal_conductivity_liquid=2.0,
            matprop_smooth_width=0.0,
            lid_base_mode='invalid',
        )

    # 6. EntropyPhaseEvaluator rejects enabled=True with activation_energy=0 and lid_base_mode='rheological'
    with pytest.raises(ValueError, match='lid_base_mode.*activation_energy'):
        EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=None,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.1,
            viscosity_solid=1e21,
            viscosity_liquid=1e2,
            grain_size=1e-3,
            thermal_conductivity_solid=4.0,
            thermal_conductivity_liquid=2.0,
            matprop_smooth_width=0.0,
            enabled=True,
            activation_energy=0.0,
            lid_base_mode='rheological',
        )

    # 7. Parser dataclasses validation
    mixed_kwargs = dict(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        _PhaseMixedParameters(**mixed_kwargs, lid_base_mode='invalid')

    with pytest.raises(ValueError, match="Unknown lid_base_mode 'invalid'"):
        _PhaseParameters(
            density=4000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            lid_base_mode='invalid',
        )

    with pytest.raises(ValueError, match='lid_base_mode.*activation_energy'):
        _PhaseParameters(
            density=4000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            enabled=True,
            activation_energy=0.0,
            lid_base_mode='rheological',
        )


@pytest.mark.physics_invariant
def test_eta_eff_infinite_yield_stress():
    """Verify eta_eff handles infinite yield stress without NaN evaluation."""
    from aragog.rheology import eta_eff

    visc_diff = 1.0e21
    tau_y_inf = np.inf
    sr = 1.0e-15

    # Smooth harmonic blend
    eta_smooth = eta_eff(visc_diff, tau_y_inf, sr, smooth=True)
    assert not np.isnan(eta_smooth)
    assert eta_smooth == pytest.approx(visc_diff, rel=1.0e-12)

    # Sharp min
    eta_sharp = eta_eff(visc_diff, tau_y_inf, sr, smooth=False)
    assert not np.isnan(eta_sharp)
    assert eta_sharp == pytest.approx(visc_diff, rel=1.0e-12)


@pytest.mark.physics_invariant
def test_compute_t_lid_base_zero_or_negative_activation_energy():
    """Verify compute_t_lid_base does not divide by zero when effective energy is <= 0."""
    from aragog.rheology import compute_t_lid_base

    # Zero effective energy
    t_lid = compute_t_lid_base(t_m=1600.0, p_lid=0.0, e_a=0.0, v_a=0.0)
    assert np.isfinite(t_lid)

    # Negative effective energy
    t_lid_neg = compute_t_lid_base(t_m=1600.0, p_lid=1e8, e_a=-1e5, v_a=0.0)
    assert np.isfinite(t_lid_neg)


@pytest.mark.unit
def test_entropy_phase_evaluator_yield_stress_max_and_viscosity_max():
    """Verify yield_stress_max and viscosity_max_log10 are stored and honored."""
    from aragog.config import MixedPhaseConfig, PhaseConfig
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    # Verify config classes accept all rheology fields
    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        enabled=True,
        arrhenius_t_ref=1500.0,
        yield_stress_max=200.0e6,
        viscosity_max_log10=28.0,
    )
    assert p_cfg.yield_stress_max == 200.0e6
    assert p_cfg.viscosity_max_log10 == 28.0

    with pytest.raises(TypeError, match='unexpected keyword argument'):
        MixedPhaseConfig(
            latent_heat_of_fusion=4e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1e-3,
            enabled=True,
            yield_stress_max=150.0e6,
            viscosity_max_log10=30.0,
        )

    # Evaluator in const_properties mode
    evaluator = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=None,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.1,
        viscosity_solid=1e21,
        viscosity_liquid=1e2,
        grain_size=1e-3,
        thermal_conductivity_solid=4.0,
        thermal_conductivity_liquid=2.0,
        matprop_smooth_width=0.0,
        const_properties=True,
        enabled=True,
        yield_stress_max=10.0e6,
        viscosity_max_log10=25.0,
    )
    assert evaluator._yield_stress_max == 10.0e6
    assert evaluator._viscosity_max_log10 == 25.0

    # Update with low temperature to test viscosity cap
    S = np.array([2500.0])
    P = np.array([1.0e9])  # 1 GPa: without cap tau_y = 50e6 + 0.6*1e9 = 650 MPa
    evaluator.set_pressure(P)
    evaluator.set_entropy(S)
    evaluator.update()
    assert evaluator.tau_y is not None
    assert np.all(evaluator.tau_y <= 10.0e6)
    assert evaluator.eta_diff is not None
    assert np.all(evaluator.eta_diff <= 10.0**25.0)


@pytest.mark.physics_invariant
def test_numpy_jax_crossover_sweep_parity():
    """Verify numpy and JAX Arrhenius viscosity match across the viscosity_max_log10 cap."""
    import jax
    import jax.numpy as jnp

    from aragog.jax.phase import compute_arrhenius_viscosity
    from aragog.rheology import eta_diff

    jax.config.update('jax_enable_x64', True)

    temperatures = np.linspace(400.0, 700.0, 31)
    for t_val in temperatures:
        eta_np = eta_diff(
            temperature=t_val,
            pressure=0.0,
            viscosity_solid=_ETA0,
            activation_energy=_E_A,
            activation_volume=_V_A,
            t_ref=_T_REF,
            r_gas=_R,
            viscosity_max_log10=40.0,
        )
        eta_jx = float(
            compute_arrhenius_viscosity(
                jnp.asarray(t_val),
                jnp.asarray(0.0),
                _ETA0,
                _E_A,
                _V_A,
                T_ref=_T_REF,
                R=_R,
                viscosity_max_log10=40.0,
            )
        )
        assert eta_jx == pytest.approx(eta_np, rel=1e-11)


@pytest.mark.physics_invariant
def test_compute_effective_viscosity_infinite_yield_stress_jax():
    """Verify compute_effective_viscosity handles infinite yield stress in JAX."""
    import jax.numpy as jnp

    from aragog.jax.phase import compute_effective_viscosity

    visc_diff = 1.0e21
    tau_y_inf = jnp.inf
    sr = 1.0e-15

    eta_eff_val = compute_effective_viscosity(jnp.asarray(visc_diff), tau_y_inf, sr)
    assert jnp.isfinite(eta_eff_val)
    assert float(eta_eff_val) == pytest.approx(visc_diff, rel=1.0e-12)


@pytest.mark.unit
def test_empty_pressure_yield_stress_capped():
    """Verify yield_stress_max is respected in _update_const when pressure is empty."""
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    evaluator = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=None,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.1,
        viscosity_solid=1e21,
        viscosity_liquid=1e2,
        grain_size=1e-3,
        thermal_conductivity_solid=4.0,
        thermal_conductivity_liquid=2.0,
        matprop_smooth_width=0.0,
        const_properties=True,
        enabled=True,
        yield_stress_c=600e6,
        yield_stress_max=500e6,
    )
    evaluator.set_entropy(np.array([2500.0, 2600.0]))
    evaluator.update()
    assert evaluator.tau_y is not None
    assert np.all(evaluator.tau_y == 500e6)


@pytest.mark.unit
def test_arrhenius_t_ref_positivity_validation():
    """Verify non-positive arrhenius_t_ref is rejected."""
    from aragog.config.phases import PhaseConfig
    from aragog.rheology import eta_diff

    with pytest.raises(ValueError, match='t_ref must be positive'):
        eta_diff(temperature=1600.0, pressure=0.0, t_ref=0.0)

    with pytest.raises(ValueError, match='t_ref must be positive'):
        eta_diff(temperature=1600.0, pressure=0.0, t_ref=-100.0)

    with pytest.raises(ValueError):
        PhaseConfig(
            density=3000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            arrhenius_t_ref=0.0,
        )


@pytest.mark.unit
def test_phase_config_rheological_lid_requires_positive_activation_energy():
    """Verify PhaseConfig and MixedPhaseConfig reject zero activation energy with rheological lid."""
    from aragog.config.phases import MixedPhaseConfig, PhaseConfig

    with pytest.raises(ValueError, match='requires non-zero activation_energy'):
        PhaseConfig(
            density=3000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            enabled=True,
            lid_base_mode='rheological',
            activation_energy=0.0,
        )

    with pytest.raises(TypeError, match='unexpected keyword argument'):
        MixedPhaseConfig(
            latent_heat_of_fusion=4e6,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='dummy',
            liquidus='dummy',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1e-3,
            enabled=True,
            lid_base_mode='rheological',
            activation_energy=0.0,
        )


@pytest.mark.unit
def test_water_prefactor_scalar_and_array_parity():
    """Verify water_prefactor scales viscosity linearly for scalar and array in numpy and JAX."""
    import jax.numpy as jnp

    from aragog.jax.phase import compute_arrhenius_viscosity as jax_arrhenius
    from aragog.rheology import eta_diff

    t = np.array([1600.0, 1800.0])
    p = np.array([0.0, 1e9])

    # 1. Scalar prefactor
    eta_dry = eta_diff(t, p, water_prefactor=1.0)
    eta_wet_scalar = eta_diff(t, p, water_prefactor=0.1)
    np.testing.assert_allclose(eta_wet_scalar, 0.1 * eta_dry, rtol=1e-12)

    # 2. Array prefactor
    wp_arr = np.array([0.5, 0.2])
    eta_wet_arr = eta_diff(t, p, water_prefactor=wp_arr)
    np.testing.assert_allclose(eta_wet_arr, wp_arr * eta_dry, rtol=1e-12)

    # 3. JAX parity with numpy
    t_j = jnp.asarray(t)
    p_j = jnp.asarray(p)
    eta_j_dry = jax_arrhenius(t_j, p_j, water_prefactor=1.0)
    eta_j_wet_scalar = jax_arrhenius(t_j, p_j, water_prefactor=0.1)
    eta_j_wet_arr = jax_arrhenius(t_j, p_j, water_prefactor=jnp.asarray(wp_arr))

    np.testing.assert_allclose(np.asarray(eta_j_dry), eta_dry, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(eta_j_wet_scalar), eta_wet_scalar, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(eta_j_wet_arr), eta_wet_arr, rtol=1e-12)

    # 4. Error on non-positive
    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        eta_diff(t, p, water_prefactor=0.0)
    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        eta_diff(t, p, water_prefactor=-0.5)


@pytest.mark.unit
def test_phi_visc_single_parameter_control():
    """Verify phi_visc_single controls the single-phase transition threshold."""
    from aragog.rheology import SolidRheologyParams

    # Default is 0.5
    default_p = SolidRheologyParams()
    assert default_p.phi_visc_single == 0.5

    # Valid in (0, 1)
    custom_p = SolidRheologyParams(phi_visc_single=0.7)
    assert custom_p.phi_visc_single == 0.7

    # Bounds validation when enabled
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=0.0)
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=1.0)
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=-0.1)
