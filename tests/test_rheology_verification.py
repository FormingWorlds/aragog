"""Algebraic pins for the solid-state rheology reference parameters.

Module under test: ``aragog.rheology`` and ``aragog.rheology_lid`` in NumPy
and JAX (float64) modes.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from functools import partial
from typing import Any

import numpy as np
import pytest

from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_viscosity,
    compute_stagnant_lid_state,
    compute_strain_rate_local,
    compute_yield_stress,
    viscous_mixing_length_factor,
)
from aragog.rheology_lid import (
    compute_effective_viscosity,
    stress_closure,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_ETA0 = 1.0e21
_G_HOM = 25.0
_T_SOL_0 = 1400.0
_DT_SOL_DP = 2.4e-8
_T_REF = 1600.0
_R = 8.314462618

_eta_ref = partial(
    compute_arrhenius_viscosity,
    viscosity_solid=_ETA0,
    g_homologous=_G_HOM,
    t_sol_0=_T_SOL_0,
    dt_sol_dp=_DT_SOL_DP,
    arrhenius_t_ref=_T_REF,
    r_gas=_R,
)

_TAU_C = 50.0e6
_TAU_MU = 0.6
_TAU_MAX = 500.0e6


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_arrhenius_viscosity_collapses_to_reference_at_t_ref():
    """At the reference temperature with P=0, eta_diff equals eta_0."""
    eta_at_ref = _eta_ref(_T_REF, 0.0)
    assert eta_at_ref == pytest.approx(_ETA0, rel=1e-12)

    cold = 0.9 * _T_REF
    eta_cold = _eta_ref(cold, 0.0)
    expected_cold = _ETA0 * np.exp(_G_HOM * _T_SOL_0 * (1.0 / cold - 1.0 / _T_REF))
    assert eta_cold == pytest.approx(expected_cold, rel=1e-12)
    assert eta_cold > 10.0 * _ETA0

    eta_hot = _eta_ref(1.1 * _T_REF, 0.0)
    assert eta_hot < _ETA0 < eta_cold
    assert eta_hot > 0.0

    eta_hot_limit = _eta_ref(1.0e15, 0.0)
    floor = _ETA0 * np.exp(-_G_HOM * _T_SOL_0 / _T_REF)
    assert eta_hot_limit == pytest.approx(floor, rel=1e-9)
    assert 0.0 < eta_hot_limit < _ETA0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_arrhenius_pressure_term_uses_reference_at_t_ref():
    """At T = T_ref the zero-pressure term cancels, leaving homologous pressure activation."""
    p_test = 100.0e9
    eta_p = _eta_ref(_T_REF, p_test)
    expected = _ETA0 * np.exp(_G_HOM * (_DT_SOL_DP * p_test) / _T_REF)
    assert eta_p == pytest.approx(expected, rel=1e-12)
    assert eta_p > 1.5 * _ETA0
    assert eta_p > 0.0
    assert _eta_ref(_T_REF, 0.0) == pytest.approx(_ETA0, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_yield_stress_saturates_at_maximum():
    """The Byerlee yield stress rises linearly with pressure then saturates at yield_stress_max."""
    p_knee = (_TAU_MAX - _TAU_C) / _TAU_MU

    p_lin = 0.5 * p_knee
    tau_lin = compute_yield_stress(p_lin, _TAU_C, _TAU_MU, _TAU_MAX)
    assert tau_lin == pytest.approx(_TAU_C + _TAU_MU * p_lin, rel=1e-12)
    assert tau_lin < _TAU_MAX

    p_high = 1.0e12
    tau_high = compute_yield_stress(p_high, _TAU_C, _TAU_MU, _TAU_MAX)
    assert tau_high == pytest.approx(_TAU_MAX, rel=1e-12)
    unclamped = _TAU_C + _TAU_MU * p_high
    assert unclamped > 1000.0 * _TAU_MAX

    p_sweep = np.logspace(5, 13, 40)
    tau_sweep = compute_yield_stress(p_sweep, _TAU_C, _TAU_MU, _TAU_MAX)
    assert np.all(tau_sweep <= _TAU_MAX + 1.0)
    assert np.all(tau_sweep > 0.0)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_yield_stress_knee_is_continuous():
    """The linear and saturated branches meet continuously at the knee."""
    p_knee = (_TAU_MAX - _TAU_C) / _TAU_MU
    tau_below = compute_yield_stress(0.999 * p_knee, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_at = compute_yield_stress(p_knee, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_above = compute_yield_stress(1.001 * p_knee, _TAU_C, _TAU_MU, _TAU_MAX)

    assert tau_at == pytest.approx(_TAU_MAX, rel=1e-12)
    assert tau_below == pytest.approx(_TAU_C + _TAU_MU * 0.999 * p_knee, rel=1e-12)
    assert tau_above == pytest.approx(_TAU_MAX, rel=1e-12)
    assert tau_below <= tau_at <= tau_above + 1.0
    assert tau_above <= _TAU_MAX + 1.0


def _import_jax_x64():
    """Import jax and jax.numpy with 64-bit precision enabled for in-process JAX tests."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')
    return jax, jnp


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_numpy_jax_rheology_float64_parity():
    """The NumPy and JAX rheology kernels agree to float64 parity."""
    _, jnp = _import_jax_x64()

    t_test = 0.85 * _T_REF
    p_test = 40.0e9

    eta_np = _eta_ref(t_test, p_test)
    eta_jx = float(
        compute_arrhenius_viscosity(
            jnp.asarray(t_test),
            jnp.asarray(p_test),
            viscosity_solid=_ETA0,
            g_homologous=_G_HOM,
            t_sol_0=_T_SOL_0,
            dt_sol_dp=_DT_SOL_DP,
            arrhenius_t_ref=_T_REF,
            r_gas=_R,
            xp=jnp,
        )
    )
    assert eta_jx == pytest.approx(eta_np, rel=1e-12)
    assert eta_jx > 0.0

    p_clamp = 1.0e12
    tau_np = compute_yield_stress(p_clamp, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_jx = float(
        compute_yield_stress(jnp.asarray(p_clamp), _TAU_C, _TAU_MU, _TAU_MAX, xp=jnp)
    )
    assert tau_jx == pytest.approx(tau_np, rel=1e-12)
    assert tau_jx == pytest.approx(_TAU_MAX, rel=1e-12)

    p_lin = 0.5 * (_TAU_MAX - _TAU_C) / _TAU_MU
    tau_np_lin = compute_yield_stress(p_lin, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_jx_lin = float(
        compute_yield_stress(jnp.asarray(p_lin), _TAU_C, _TAU_MU, _TAU_MAX, xp=jnp)
    )
    assert tau_jx_lin == pytest.approx(tau_np_lin, rel=1e-12)


def test_viscosity_max_log10_clipping():
    """Verify viscosity_max_log10 caps cold Arrhenius viscosity without overflow."""
    T = 40.0
    P = 0.0
    eta_d = compute_arrhenius_viscosity(
        temperature=T,
        pressure=P,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        t_sol_0=1400.0,
        dt_sol_dp=2.4e-8,
        arrhenius_t_ref=1600.0,
        r_gas=8.314462618,
        viscosity_max_log10=40.0,
    )
    assert eta_d == pytest.approx(1e40, rel=1e-12)

    eta_e = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_y=50e6,
            strain_rate=1e-15,
            stress_closure_mode='local',
        )
    )
    assert eta_e == pytest.approx(50e6 / (2.0 * 1e-15), rel=1e-12)


def test_lid_base_mode_and_stress_closure_validation():
    """Verify validation of lid_base_mode, stress_closure_mode, and invalid combinations."""
    with pytest.raises(ValueError, match="Unknown stress_closure_mode 'invalid'"):
        SolidRheologyParams(stress_closure_mode='invalid')

    with pytest.raises(ValueError, match="Unknown lid_base_mode 'invalid'"):
        SolidRheologyParams(lid_base_mode='invalid')

    with pytest.raises(ValueError, match='enabled=True requires positive g_homologous'):
        SolidRheologyParams(enabled=True, g_homologous=0.0, lid_base_mode='rheological')

    with pytest.raises(ValueError, match='enabled=True requires positive g_homologous'):
        SolidRheologyParams(enabled=True, g_homologous=0.0, lid_base_mode='fixed')

    p = SolidRheologyParams(enabled=False, g_homologous=0.0, lid_base_mode='rheological')
    assert p.lid_base_mode == 'rheological'

    with pytest.raises(ValueError, match='Unknown stress_closure_mode'):
        compute_effective_viscosity(eta_diff=1.0e21, stress_closure_mode='global')

    with pytest.raises(ValueError, match='mixing_length and unyielded_velocity required'):
        compute_effective_viscosity(eta_diff=1.0e21, stress_closure_mode='local')

    with pytest.raises(ValueError, match="not allowed when stress_closure_mode='lid'"):
        compute_effective_viscosity(
            eta_diff=1.0e21,
            eta_i=1.0e20,
            tau_d=1.0e6,
            tau_y_lid=1.0e6,
            strain_rate=1.0e-15,
            stress_closure_mode='lid',
        )

    with pytest.raises(ValueError, match="not allowed when stress_closure_mode='local'"):
        compute_effective_viscosity(
            eta_diff=1.0e21,
            tau_y=1.0e7,
            strain_rate=1.0e-15,
            tau_d=1.0e6,
            stress_closure_mode='local',
        )

    with pytest.raises(ValueError, match="eta_i is required when stress_closure_mode='lid'"):
        compute_effective_viscosity(
            eta_diff=1.0e21,
            tau_d=1.0e6,
            tau_y_lid=1.0e6,
            stress_closure_mode='lid',
        )

    with pytest.raises(ValueError, match='mixing_length must be provided'):
        stress_closure('local', viscous_velocity=1.0e-9, mixing_length=None)

    with pytest.raises(ValueError, match="Unknown stress closure mode 'global'"):
        stress_closure('global', viscous_velocity=1.0e-9, mixing_length=1.0e5)

    with pytest.raises(ValueError, match="Mode 'lid' uses boundary-layer stress closure"):
        stress_closure('lid', viscous_velocity=1.0e-9, mixing_length=1.0e5)

    with pytest.raises(ValueError, match='Unknown stress closure mode'):
        stress_closure('other', viscous_velocity=1.0e-9, mixing_length=1.0e5)

    sr = stress_closure('local', viscous_velocity=2.0e-9, mixing_length=1.0e5)
    assert sr == pytest.approx(1.0e-14, rel=1e-12, abs=0.0)
    assert compute_strain_rate_local(2.0e-9, 1.0e5) == pytest.approx(
        1.0e-14, rel=1e-12, abs=0.0
    )


@pytest.mark.physics_invariant
def test_compute_effective_viscosity_infinite_yield_stress_numpy():
    """Verify compute_effective_viscosity handles infinite yield stress without NaN evaluation."""
    visc_diff = 1.0e21
    tau_y_inf = np.inf
    sr = 1.0e-15

    eta_sharp = float(
        compute_effective_viscosity(
            eta_diff=visc_diff,
            tau_y=tau_y_inf,
            strain_rate=sr,
            stress_closure_mode='local',
        )
    )
    assert not np.isnan(eta_sharp)
    assert eta_sharp == pytest.approx(visc_diff, rel=1.0e-12)


def test_phase_config_yield_stress_max_and_viscosity_max():
    """Verify yield_stress_max and viscosity_max_log10 are stored and validated on PhaseConfig."""
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
    # Exact by construction: PhaseConfig delegates stored dataclass attributes unchanged.
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


@pytest.mark.physics_invariant
def test_numpy_jax_crossover_sweep_parity():
    """Verify numpy and JAX Arrhenius viscosity match across the viscosity_max_log10 cap."""
    _, jnp = _import_jax_x64()

    temperatures = np.linspace(400.0, 700.0, 31)
    for t_val in temperatures:
        eta_np = compute_arrhenius_viscosity(
            temperature=t_val,
            pressure=0.0,
            viscosity_solid=_ETA0,
            g_homologous=_G_HOM,
            t_sol_0=_T_SOL_0,
            dt_sol_dp=_DT_SOL_DP,
            arrhenius_t_ref=_T_REF,
            r_gas=_R,
            viscosity_max_log10=40.0,
        )
        eta_jx = float(
            compute_arrhenius_viscosity(
                jnp.asarray(t_val),
                jnp.asarray(0.0),
                viscosity_solid=_ETA0,
                g_homologous=_G_HOM,
                t_sol_0=_T_SOL_0,
                dt_sol_dp=_DT_SOL_DP,
                arrhenius_t_ref=_T_REF,
                r_gas=_R,
                viscosity_max_log10=40.0,
                xp=jnp,
            )
        )
        assert eta_jx == pytest.approx(eta_np, rel=1e-11)


@pytest.mark.physics_invariant
def test_compute_effective_viscosity_infinite_yield_stress_jax():
    """Verify aragog.rheology requires float64 on the JAX path and handles infinite yield stress."""
    _, jnp = _import_jax_x64()

    visc_diff = 1.0e21
    tau_y_inf = jnp.inf
    sr = 1.0e-15

    eta_eff_val = compute_effective_viscosity(
        jnp.asarray(visc_diff),
        tau_y=tau_y_inf,
        strain_rate=sr,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert eta_eff_val.dtype == jnp.float64
    assert jnp.isfinite(eta_eff_val)
    assert float(eta_eff_val) == pytest.approx(visc_diff, rel=1.0e-12)

    import os
    from pathlib import Path

    import aragog

    sub_env = {k: v for k, v in os.environ.items() if k != 'JAX_ENABLE_X64'}
    sub_env['PYTHONPATH'] = str(Path(aragog.__file__).parents[1])
    proc_no_jax = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import os, sys, aragog, aragog.rheology; '
                "print('jax' in sys.modules, 'JAX_ENABLE_X64' in os.environ)"
            ),
        ],
        env=sub_env,
        capture_output=True,
        text=True,
    )
    assert proc_no_jax.returncode == 0, proc_no_jax.stderr
    assert proc_no_jax.stdout.strip() == 'False False'

    proc_no_x64 = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import os, jax, jax.numpy as jnp, aragog.rheology as r\n'
                'print("env", "JAX_ENABLE_X64" in os.environ, "x64", jax.config.jax_enable_x64)\n'
                'try:\n'
                '    v = r.compute_arrhenius_viscosity(300.0, 0.0, xp=jnp)\n'
                '    print("returned", v.dtype, float(v))\n'
                'except RuntimeError as exc:\n'
                '    print("RuntimeError", exc)\n'
            ),
        ],
        env=sub_env,
        capture_output=True,
        text=True,
    )
    assert proc_no_x64.returncode == 0, proc_no_x64.stderr
    lines_no_x64 = proc_no_x64.stdout.strip().splitlines()
    assert lines_no_x64[0] == 'env False x64 False'
    assert lines_no_x64[1].startswith('RuntimeError ')
    assert 'jax_enable_x64' in lines_no_x64[1]

    proc_with_x64 = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import jax; jax.config.update("jax_enable_x64", True); '
                'import jax.numpy as jnp, aragog.rheology as r; '
                'v = r.compute_arrhenius_viscosity(300.0, 0.0, xp=jnp); '
                'print(v.dtype, float(v))'
            ),
        ],
        env=sub_env,
        capture_output=True,
        text=True,
    )
    assert proc_with_x64.returncode == 0, proc_with_x64.stderr
    dtype_str, val_str = proc_with_x64.stdout.strip().split()
    assert dtype_str == 'float64'
    assert float(val_str) == pytest.approx(1.0e40, rel=1.0e-12)


def test_arrhenius_t_ref_positivity_validation():
    """Verify non-positive arrhenius_t_ref is rejected and rejects t_ref, T_ref and R."""
    with pytest.raises(ValueError, match='arrhenius_t_ref must be positive'):
        compute_arrhenius_viscosity(temperature=1600.0, pressure=0.0, arrhenius_t_ref=0.0)

    with pytest.raises(ValueError, match='arrhenius_t_ref must be positive'):
        compute_arrhenius_viscosity(temperature=1600.0, pressure=0.0, arrhenius_t_ref=-100.0)

    for removed_alias in ('t_ref', 'T_ref', 'R'):
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            compute_arrhenius_viscosity(1600.0, 0.0, **{removed_alias: 1600.0})

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


def test_phase_config_rheological_lid_requires_positive_g_homologous():
    """Verify PhaseConfig rejects zero g_homologous with rheological lid."""
    with pytest.raises(ValueError, match='g_homologous'):
        PhaseConfig(
            density=3000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            enabled=True,
            lid_base_mode='rheological',
            g_homologous=0.0,
        )


def test_water_prefactor_scalar_and_array_parity():
    """Verify water_prefactor scales viscosity linearly for scalar and array in numpy and JAX."""
    _, jnp = _import_jax_x64()

    t = np.array([1600.0, 1800.0])
    p = np.array([0.0, 1e9])
    expected_dry = 1.0e21 * np.exp(25.0 * ((1400.0 + 2.4e-8 * p) / t - 1400.0 / 1600.0))

    eta_dry = compute_arrhenius_viscosity(t, p, water_prefactor=1.0)
    np.testing.assert_allclose(eta_dry, expected_dry, rtol=1e-12)
    eta_wet_scalar = compute_arrhenius_viscosity(t, p, water_prefactor=0.1)
    np.testing.assert_allclose(eta_wet_scalar, 0.1 * eta_dry, rtol=1e-12)

    wp_arr = np.array([0.5, 0.2])
    eta_wet_arr = compute_arrhenius_viscosity(t, p, water_prefactor=wp_arr)
    np.testing.assert_allclose(eta_wet_arr, wp_arr * eta_dry, rtol=1e-12)

    t_j = jnp.asarray(t)
    p_j = jnp.asarray(p)
    eta_j_dry = compute_arrhenius_viscosity(t_j, p_j, water_prefactor=1.0, xp=jnp)
    eta_j_wet_scalar = compute_arrhenius_viscosity(t_j, p_j, water_prefactor=0.1, xp=jnp)
    eta_j_wet_arr = compute_arrhenius_viscosity(
        t_j, p_j, water_prefactor=jnp.asarray(wp_arr), xp=jnp
    )

    np.testing.assert_allclose(np.asarray(eta_j_dry), eta_dry, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(eta_j_wet_scalar), eta_wet_scalar, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(eta_j_wet_arr), eta_wet_arr, rtol=1e-12)

    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        compute_arrhenius_viscosity(t, p, water_prefactor=0.0)
    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        compute_arrhenius_viscosity(t, p, water_prefactor=-0.5)


def test_phi_visc_single_parameter_control():
    """Verify phi_visc_single controls the single-phase transition threshold."""
    default_p = SolidRheologyParams()
    # Exact by construction: dataclass stores literal default and constructor argument.
    assert default_p.phi_visc_single == 0.5

    custom_p = SolidRheologyParams(phi_visc_single=0.7)
    assert custom_p.phi_visc_single == 0.7

    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=0.0)
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=1.0)
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=-0.1)


@pytest.mark.physics_invariant
def test_effective_viscosity_unyielded_and_yielded_limits():
    """Verify effective viscosity limits in local stress closure mode with min closure."""
    eta_d = 1.0e21
    tau_y = 1.0e8

    sr_match = tau_y / (2.0 * eta_d)
    expected_mid = eta_d * (10.0 ** (-0.01 * np.log(2.0)))
    eta_match = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_match,
        stress_closure_mode='local',
    )
    assert eta_match == pytest.approx(expected_mid, rel=1.0e-12)

    sr_low = 1.0e-18
    eta_below = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_low,
        stress_closure_mode='local',
    )
    assert eta_below == pytest.approx(eta_d, rel=1.0e-12)

    sr_high = 1.0e-10
    eta_plastic = tau_y / (2.0 * sr_high)
    eta_yielded = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_high,
        stress_closure_mode='local',
    )
    assert eta_yielded == pytest.approx(eta_plastic, rel=1.0e-12)

    _, jnp = _import_jax_x64()

    eta_match_jax = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_match,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert float(eta_match_jax) == pytest.approx(expected_mid, rel=1.0e-12)

    eta_below_jax = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_low,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert float(eta_below_jax) == pytest.approx(eta_d, rel=1.0e-12)

    eta_yielded_jax = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_high,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert float(eta_yielded_jax) == pytest.approx(eta_plastic, rel=1.0e-12)


@pytest.mark.physics_invariant
def test_moresi_solomatov_min_closure_exact_contract():
    """Verify Moresi & Solomatov (1998) min closure exact contracts."""
    eta_d = 8.3e24
    tau_y = 1.0e6

    # 1. Below yield (tau_d < tau_y): eta_eff == eta_diff.
    td_below = 4.0e5
    eta_eff_below = compute_effective_viscosity(
        eta_diff=eta_d,
        eta_i=eta_d,
        tau_d=td_below,
        tau_y_lid=tau_y,
    )
    assert eta_eff_below == pytest.approx(eta_d, rel=1.0e-14)

    # 2. Above yield with eta_i == eta_diff: eta_eff / eta_diff == tau_y / tau_d.
    td_above = 2.0e6
    eta_eff_above = compute_effective_viscosity(
        eta_diff=eta_d,
        eta_i=eta_d,
        tau_d=td_above,
        tau_y_lid=tau_y,
    )
    expected_ratio = tau_y / td_above
    assert (eta_eff_above / eta_d) == pytest.approx(expected_ratio, rel=1.0e-14)

    # 3. Interior-scaled yield cap eta_yield = eta_i * (tau_y_lid / tau_d) in a cold lid:
    eta_cold_lid = 1.0e40
    eta_interior = 1.0e21
    eta_unyielded_cold = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=4.0e5,
        tau_y_lid=1.0e6,
    )
    assert eta_unyielded_cold == pytest.approx(eta_cold_lid, rel=1.0e-14)

    eta_yielded_cold = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=1.6e6,
        tau_y_lid=1.0e6,
    )
    expected_yielded_cold = eta_interior * (1.0e6 / 1.6e6)
    assert eta_yielded_cold == pytest.approx(expected_yielded_cold, rel=1.0e-14)
    assert eta_yielded_cold < eta_interior

    # Canary: calling compute_effective_viscosity with cold-lid creep viscosity as the scale
    # (eta_i=eta_cold_lid or omitted) leaves 6.25e39 Pa s >> eta_interior and fails mobilization.
    wrong_cold_scaled = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_cold_lid,
        tau_d=1.6e6,
        tau_y_lid=1.0e6,
    )
    expected_wrong_cold = 6.25e39 * (10.0 ** (-0.01 * np.logaddexp(0.0, -np.log10(1.6) / 0.01)))
    assert wrong_cold_scaled == pytest.approx(expected_wrong_cold, rel=1.0e-14)
    with pytest.raises(AssertionError):
        assert wrong_cold_scaled < eta_interior

    # Three reference regimes (cold surface 300 K, warm surface 700 K, and low-yield lid):
    _, jnp = _import_jax_x64()
    cases = [
        (1.0e40, 1.0e21, 1.6e6, 1.0e6, 6.25e20),
        (8.30e24, 1.0e21, 1.6e6, 1.0e6, 6.25e20),
        (8.30e24, 1.0e20, 8.65e5, 1.0e4, 1.0e20 * (1.0e4 / 8.65e5)),
    ]
    for ed_c, ei_c, td_c, ty_c, exp_c in cases:
        v_np = compute_effective_viscosity(
            eta_diff=ed_c, eta_i=ei_c, tau_d=td_c, tau_y_lid=ty_c, xp=np
        )
        v_jx = float(
            compute_effective_viscosity(
                eta_diff=jnp.asarray(ed_c),
                eta_i=jnp.asarray(ei_c),
                tau_d=jnp.asarray(td_c),
                tau_y_lid=jnp.asarray(ty_c),
                xp=jnp,
            )
        )
        assert v_np == pytest.approx(exp_c, rel=1.0e-12)
        assert v_jx == pytest.approx(exp_c, rel=1.0e-12)

    # 4. Smooth log10 tanh yield transition at tau_d = tau_y (both eta_i == eta_d and eta_i < eta_d).
    eps = 1.0e-6
    eta_left = compute_effective_viscosity(
        eta_diff=eta_d,
        eta_i=eta_d,
        tau_d=tau_y - eps,
        tau_y_lid=tau_y,
    )
    eta_mid = compute_effective_viscosity(
        eta_diff=eta_d,
        eta_i=eta_d,
        tau_d=tau_y,
        tau_y_lid=tau_y,
    )
    eta_right = compute_effective_viscosity(
        eta_diff=eta_d,
        eta_i=eta_d,
        tau_d=tau_y + eps,
        tau_y_lid=tau_y,
    )
    expected_mid_equal = eta_d * (10.0 ** (-0.5 * 0.01 * np.log(2.0)))
    assert eta_left == pytest.approx(expected_mid_equal, rel=1.0e-6)
    assert eta_mid == pytest.approx(expected_mid_equal, rel=1.0e-12)
    assert eta_right == pytest.approx(expected_mid_equal, rel=1.0e-6)
    assert eta_left > eta_mid > eta_right

    eta_cold_left = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=tau_y * (1.0 - eps),
        tau_y_lid=tau_y,
    )
    eta_cold_mid = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=tau_y,
        tau_y_lid=tau_y,
    )
    eta_cold_right = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=tau_y * (1.0 + eps),
        tau_y_lid=tau_y,
    )
    expected_geom_mid = 10.0 ** (0.5 * (np.log10(eta_cold_lid) + np.log10(eta_interior)))
    assert eta_cold_mid == pytest.approx(expected_geom_mid, rel=1.0e-12)
    assert eta_cold_left > eta_cold_mid > eta_cold_right
    assert abs(np.log10(eta_cold_left) - np.log10(eta_cold_right)) < 1.0e-3


@pytest.mark.physics_invariant
def test_lid_closure_jump_at_yield_threshold():
    """Verify lid closure log10 tanh blend across tau_d = tau_y_lid, asymptotic limits, and hard-step canary."""
    import jax

    _, jnp = _import_jax_x64()

    eta_d = 1.0e26
    eta_i = 1.0e20
    tau_y = 1.0e8
    w_lid = 1.0

    # 1. Continuity and finiteness of eta_eff(x) and d(eta_eff)/dx over x in [0.8, 1.2].
    xs = np.linspace(0.8, 1.2, 81)
    grad_fn = jax.grad(
        lambda x_val: compute_effective_viscosity(
            eta_diff=jnp.asarray(eta_d),
            tau_d=x_val * jnp.asarray(tau_y),
            tau_y_lid=jnp.asarray(tau_y),
            stress_closure_mode='lid',
            w_lid=jnp.asarray(w_lid),
            eta_i=jnp.asarray(eta_i),
            xp=jnp,
        )
    )
    vals_np = np.array(
        [
            float(
                compute_effective_viscosity(
                    eta_diff=eta_d,
                    tau_d=float(xv * tau_y),
                    tau_y_lid=tau_y,
                    stress_closure_mode='lid',
                    w_lid=w_lid,
                    eta_i=eta_i,
                    xp=np,
                )
            )
            for xv in xs
        ]
    )
    grads_jx = np.array([float(grad_fn(jnp.asarray(xv))) for xv in xs])
    assert np.all(np.isfinite(vals_np))
    assert np.all(np.isfinite(grads_jx))
    assert np.all(np.diff(vals_np) < 0.0)
    assert np.all(grads_jx < 0.0)

    # At x = 1.0, w_b = 0.5 gives geometric mean sqrt(eta_d * eta_i) = 1.0e23 Pa s.
    val_mid = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_d=tau_y,
            tau_y_lid=tau_y,
            stress_closure_mode='lid',
            w_lid=w_lid,
            eta_i=eta_i,
            xp=np,
        )
    )
    assert val_mid == pytest.approx(1.0e23, rel=1.0e-12)

    # 2. Asymptotic limit checks at x = 0.9 and x = 1.1 (within 0.5% of branch over 6-decade contrast).
    val_09 = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_d=0.9 * tau_y,
            tau_y_lid=tau_y,
            stress_closure_mode='lid',
            w_lid=w_lid,
            eta_i=eta_i,
            xp=np,
        )
    )
    assert (val_09 / eta_d) == pytest.approx(0.995412852640275, rel=1.0e-12)
    assert val_09 == pytest.approx(eta_d, rel=0.01)

    val_11 = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_d=1.1 * tau_y,
            tau_y_lid=tau_y,
            stress_closure_mode='lid',
            w_lid=w_lid,
            eta_i=eta_i,
            xp=np,
        )
    )
    expected_11 = eta_i / 1.1
    assert (val_11 / expected_11) == pytest.approx(1.0046758935808797, rel=1.0e-12)
    assert val_11 == pytest.approx(expected_11, rel=0.01)

    # 3. Mutant canary: unblended hard-step mutant fails continuity bound at x = 1.0 +- 1e-4.
    def _eval_smooth(x_pt: float) -> float:
        return float(
            compute_effective_viscosity(
                eta_diff=eta_d,
                tau_d=x_pt * tau_y,
                tau_y_lid=tau_y,
                stress_closure_mode='lid',
                w_lid=w_lid,
                eta_i=eta_i,
                xp=np,
            )
        )

    def _eval_hard_mutant(x_pt: float) -> float:
        return float(eta_d if x_pt <= 1.0 else min(eta_d, eta_i / x_pt))

    smooth_step_decades = abs(
        np.log10(_eval_smooth(1.0 - 1.0e-4)) - np.log10(_eval_smooth(1.0 + 1.0e-4))
    )
    mutant_step_decades = abs(
        np.log10(_eval_hard_mutant(1.0 - 1.0e-4)) - np.log10(_eval_hard_mutant(1.0 + 1.0e-4))
    )
    assert smooth_step_decades < 0.05
    with pytest.raises(AssertionError):
        assert mutant_step_decades < 0.05


@pytest.mark.physics_invariant
def test_zero_yield_stress_cohesion_floor_prevents_zero_viscosity(monkeypatch):
    """Verify zero yield stress is floored at COHESION_FLOOR (1e-6 Pa) so eta_eff > 0."""
    import aragog.rheology as rheo_mod
    import aragog.rheology_lid as lid_mod
    from aragog.rheology_lid import COHESION_FLOOR

    assert COHESION_FLOOR == pytest.approx(1.0e-6, rel=1.0e-15, abs=0.0)
    sr = 1.0e-15
    eta_local_zero = compute_effective_viscosity(
        eta_diff=1.0e21,
        tau_y=0.0,
        strain_rate=sr,
        stress_closure_mode='local',
    )
    expected_local = COHESION_FLOOR / (2.0 * sr)
    assert eta_local_zero == pytest.approx(expected_local, rel=1.0e-12)
    assert eta_local_zero > 0.0

    eta_lid_zero = compute_effective_viscosity(
        eta_diff=1.0e21,
        eta_i=1.0e20,
        tau_d=1.0e5,
        tau_y_lid=0.0,
        stress_closure_mode='lid',
    )
    expected_lid = 1.0e20 * (COHESION_FLOOR / 1.0e5)
    assert eta_lid_zero == pytest.approx(expected_lid, rel=1.0e-12)
    assert eta_lid_zero > 0.0

    # Canary: disabling COHESION_FLOOR in the functions under test produces zero viscosity.
    monkeypatch.setattr(lid_mod, 'COHESION_FLOOR', 0.0)
    monkeypatch.setattr(rheo_mod, 'COHESION_FLOOR', 0.0)
    unfloored_local = compute_effective_viscosity(
        eta_diff=1.0e21,
        tau_y=0.0,
        strain_rate=sr,
        stress_closure_mode='local',
    )
    unfloored_lid = compute_effective_viscosity(
        eta_diff=1.0e21,
        eta_i=1.0e20,
        tau_d=1.0e5,
        tau_y_lid=0.0,
        stress_closure_mode='lid',
    )
    with pytest.raises(AssertionError):
        assert unfloored_local > 0.0
    with pytest.raises(AssertionError):
        assert unfloored_lid > 0.0


@pytest.mark.physics_invariant
def test_no_lid_zero_diagnostics_and_smooth_w_sum_transition():
    """Verify no-lid profiles zero boundary-layer diagnostics and W_sum blends smoothly."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    _, jnp = _import_jax_x64()

    n_nodes = 20
    r = np.linspace(3.48e6, 6.371e6, n_nodes)
    p = np.linspace(135.0e9, 1.0e5, n_nodes)
    t_iso = np.full(n_nodes, 2000.0)
    f_conv = np.full(n_nodes, 80.0)
    f_tot = np.full(n_nodes, 100.0)
    phi = np.zeros(n_nodes)
    params = SolidRheologyParams(enabled=True)

    st_iso_np = compute_stagnant_lid_state(
        radii=r,
        temperature=t_iso,
        pressure=p,
        convective_flux=f_conv,
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    st_iso_jx = compute_stagnant_lid_state(
        radii=jnp.asarray(r),
        temperature=jnp.asarray(t_iso),
        pressure=jnp.asarray(p),
        convective_flux=jnp.asarray(f_conv),
        total_flux=jnp.asarray(f_tot),
        solidus_temperature=None,
        melt_fraction=jnp.asarray(phi),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=jnp,
    )
    # Isothermal column has d_lid == 0.0 and remaining boundary-layer diagnostics == 0.0.
    assert float(st_iso_np['d_lid']) == 0.0
    assert float(st_iso_jx['d_lid']) == 0.0
    for key in (
        'Ra_eff',
        'Ra_rh',
        'v_m',
        'tau_d',
        'tau_d_lid',
        'tau_buoy',
        'tau_d_over_tau_buoy',
        'w_y',
        'lid_regime',
    ):
        assert float(st_iso_np[key]) == 0.0
        assert float(st_iso_jx[key]) == 0.0

    # Smooth transition of mu_i across max_k w_conv_int ~ 1e-6 for both array and scalar viscosity_mixture:
    t_grad = np.linspace(2600.0, 1200.0, n_nodes)
    visc_mix_arr = np.full(n_nodes, 1.0e18)
    params_smooth = SolidRheologyParams(enabled=True, interior_flux_fraction=0.10)
    f_low_1 = np.full(n_nodes, 2.945)
    f_low_2 = np.full(n_nodes, 2.947)
    st_arr_1 = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad,
        pressure=p,
        convective_flux=f_low_1,
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_smooth,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        viscosity_mixture=visc_mix_arr,
    )
    st_arr_2 = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad,
        pressure=p,
        convective_flux=f_low_2,
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_smooth,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        viscosity_mixture=visc_mix_arr,
    )
    mu_1 = float(st_arr_1['mu_i'])
    mu_2 = float(st_arr_2['mu_i'])
    assert 1.0e19 < mu_2 < mu_1 < float(st_arr_1['eta_solid_arrhenius'])
    assert 0.005 < abs(np.log10(mu_2) - np.log10(mu_1)) < 0.05

    st_scalar_1 = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad,
        pressure=p,
        convective_flux=f_low_1,
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_smooth,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        viscosity_mixture=1.0e18,
    )
    assert float(st_scalar_1['mu_i']) == pytest.approx(mu_1, rel=1.0e-12)

    # Inactive convection with finite viscosity_mixture (max_k w_conv_int << 1e-6 -> w_weight -> 0)
    # and with non-finite viscosity_mixture both fall back cleanly to eta_solid_arrhenius:
    params_inactive = SolidRheologyParams(enabled=True, interior_flux_fraction=0.20)
    st_inactive_finite_mix = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad,
        pressure=p,
        convective_flux=np.zeros(n_nodes),
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_inactive,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        viscosity_mixture=np.full(n_nodes, 1.0e15),
    )
    assert float(st_inactive_finite_mix['mu_i']) == pytest.approx(
        float(st_inactive_finite_mix['eta_solid_arrhenius']), rel=1.0e-8
    )

    st_inactive_nan_mix = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad,
        pressure=p,
        convective_flux=np.zeros(n_nodes),
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        viscosity_mixture=np.full(n_nodes, np.nan),
    )
    assert np.isnan(float(st_inactive_nan_mix['mu_i']))
    assert np.isnan(float(st_inactive_nan_mix['d_lid']))
    assert np.isnan(float(st_inactive_nan_mix['tau_d']))

    assert np.isnan(
        compute_effective_viscosity(
            eta_diff=1.0e21, eta_i=1.0e20, tau_d=np.nan, tau_y_lid=1.0e6
        )
    )
    assert np.isnan(
        compute_effective_viscosity(
            eta_diff=1.0e21,
            tau_y=1.0e6,
            strain_rate=np.nan,
            stress_closure_mode='local',
        )
    )


def test_rheology_and_lid_branch_coverage():
    """Verify remaining parameter bounds, module attributes, and lid branches."""
    import aragog.rheology as rheo_mod
    from aragog.rheology_lid import compute_stagnant_lid_state

    assert 'compute_stagnant_lid_state' in dir(rheo_mod)
    with pytest.raises(AttributeError, match='has no attribute'):
        _ = rheo_mod.nonexistent_attribute

    assert compute_arrhenius_viscosity(1600.0, 0.0, arrhenius_t_ref=1600.0) == pytest.approx(
        1.0e21, rel=1e-12
    )
    assert np.isnan(compute_arrhenius_viscosity(np.nan, 0.0))
    with pytest.raises(ValueError, match='pressure'):
        compute_yield_stress(-200.0e6, yield_stress_c=50.0e6, yield_stress_mu=0.6)
    with pytest.raises(ValueError, match='eta_diff'):
        compute_effective_viscosity(
            eta_diff=np.inf,
            tau_y=50.0e6,
            strain_rate=1.0e-15,
            stress_closure_mode='local',
        )
    assert float(
        compute_effective_viscosity(
            eta_diff=1.0e25,
            tau_y=50.0e6,
            strain_rate=1.0e-15,
            stress_closure_mode='local',
        )
    ) == pytest.approx(2.5e22, rel=1e-12)

    invalid_cases = [
        {'t_sol_0': 0.0},
        {'t_sol_0': -100.0},
        {'dt_sol_dp': -1.0e-8},
        {'viscosity_max_log10': 310.0},
        {'yield_stress_c': -1.0},
        {'yield_stress_mu': -0.1},
        {'yield_stress_max': 0.0},
        {'interior_flux_fraction': 0.0},
        {'lid_base_temperature': 0.0},
        {'lid_contrast_coeff': 0.0},
        {'lid_mask_width_cells': 0.0},
    ]
    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            SolidRheologyParams(enabled=True, **kwargs)

    n_nodes = 20
    r = np.linspace(3.48e6, 6.371e6, n_nodes)
    T = np.linspace(2600.0, 1200.0, n_nodes)
    P = np.linspace(135.0e9, 1.0e5, n_nodes)
    F_conv = np.full(n_nodes, 50.0)
    F_conv[-4:] = 0.0
    F_tot = np.full(n_nodes, 60.0)
    T_sol = np.linspace(3000.0, 1400.0, n_nodes)
    phi = np.zeros(n_nodes)
    params_fixed = SolidRheologyParams(
        enabled=True, lid_base_mode='fixed', lid_base_temperature=1350.0
    )
    st = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=T_sol,
        melt_fraction=phi,
        params=params_fixed,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        viscosity_mixture=1.0e20,
        density=np.full(n_nodes, 3300.0),
    )
    # Exact by construction: fixed lid_base_mode sets T_lid_iso = params.lid_base_temperature.
    assert st['T_lid_iso'] == 1350.0
    assert float(st['mu_i']) == pytest.approx(1.0e20, rel=1e-12)
    assert float(st['d_lid']) > 1.0e5
    assert float(st['w_lid'][-1]) > 0.5

    # Geometric-mean log10 blending at w_lid = 0.5 when tau_y / tau_d = 0.01:
    # eta_eff_val = 0.01 * 1e24 = 1e22 -> log_solid = 0.5 * 22 + 0.5 * 24 = 23 -> 1e23 Pa s.
    eta_blended = compute_effective_viscosity(
        eta_diff=np.array([1.0e24]),
        eta_i=1.0e24,
        tau_d=1.0e8,
        tau_y_lid=1.0e6,
        stress_closure_mode='lid',
        w_lid=np.array([0.5]),
    )
    assert float(eta_blended[0]) == pytest.approx(1.0e23, rel=1e-12)

    eta_local_vel = compute_effective_viscosity(
        eta_diff=1.0e22,
        stress_closure_mode='local',
        unyielded_velocity=2.0e-9,
        mixing_length=1.0e5,
        tau_y_profile=1.0e7,
    )
    assert eta_local_vel == pytest.approx(1.0e7 / (2.0 * 1.0e-14), rel=1e-12)
    with pytest.raises(ValueError, match='tau_y or tau_y_profile required'):
        compute_effective_viscosity(
            eta_diff=1.0e22,
            stress_closure_mode='local',
            strain_rate=1.0e-14,
        )

    with pytest.raises(TypeError, match='t_sol_0'):
        compute_stagnant_lid_state(
            radii=r,
            temperature=T,
            pressure=P,
            convective_flux=F_conv,
            total_flux=F_tot,
            solidus_temperature=2000.0,
            t_sol_0=2000.0,
            melt_fraction=phi,
            params=params_fixed,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
        )

    params_fixed_2000 = dataclasses.replace(params_fixed, t_sol_0=2000.0)
    st_scalar_sol = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=2000.0,
        melt_fraction=phi,
        params=params_fixed_2000,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    st_arr_sol = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=np.full(n_nodes, 2000.0),
        melt_fraction=phi,
        params=params_fixed_2000,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    with pytest.raises(ValueError, match='t_sol_0'):
        dataclasses.replace(params_fixed_2000, t_sol_0=0.0)
    t_i_val = float(st_scalar_sol['T_i'])
    expected_dt_rh_sol = t_i_val**2 / (params_fixed.g_homologous * 2000.0)
    expected_theta_sol = (t_i_val - T[-1]) / expected_dt_rh_sol
    assert float(st_scalar_sol['dT_rh']) == pytest.approx(expected_dt_rh_sol, rel=1e-12)
    assert float(st_scalar_sol['theta']) == pytest.approx(expected_theta_sol, rel=1e-12)
    assert float(st_scalar_sol['eta_i']) == pytest.approx(float(st_arr_sol['eta_i']), rel=1e-12)
    assert float(st_scalar_sol['tau_d']) == pytest.approx(float(st_arr_sol['tau_d']), rel=1e-12)
    assert float(st_scalar_sol['eta_contrast']) == pytest.approx(
        float(st_arr_sol['eta_contrast']), rel=1e-12
    )
    expected_eta_i_2000 = compute_arrhenius_viscosity(
        t_i_val,
        float(st_scalar_sol['P_T_i']),
        viscosity_solid=1.0e21,
        g_homologous=params_fixed.g_homologous,
        solidus_temperature=2000.0,
        t_sol_0=2000.0,
        arrhenius_t_ref=params_fixed.arrhenius_t_ref,
    )
    assert float(st_scalar_sol['eta_i']) == pytest.approx(expected_eta_i_2000, rel=1e-12)

    with pytest.raises(ValueError, match='tau_d and tau_y_lid are required'):
        compute_effective_viscosity(
            eta_diff=1.0e22,
            eta_i=1.0e21,
            tau_d=None,
            tau_y_lid=None,
            stress_closure_mode='lid',
        )


def test_validation_and_lid_base_edge_cases():
    """Verify parameter validation, non-positive T NaN return, water_prefactor log-clip, and full-lid base."""
    import math

    from aragog.rheology import stress_closure, viscous_mixing_length_factor
    from aragog.rheology_lid import compute_stagnant_lid_state

    jax, jnp = _import_jax_x64()

    # 1. enabled must be a bool.
    with pytest.raises(ValueError, match='enabled must be a bool'):
        SolidRheologyParams(enabled='False')  # type: ignore[arg-type]

    # 2. Single top-slope parameter: top_slope is not an argument.
    r = np.linspace(3.48e6, 6.371e6, 20)
    ml = np.minimum(r - r[0], r[-1] - r)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        viscous_mixing_length_factor(r, r[0], r[-1], ml, top_slope=0.5)

    # 4. T <= 0 raises ValueError in NumPy and returns NaN in JAX.
    for t_bad in (0.0, -50.0):
        with pytest.raises(ValueError, match='temperature'):
            compute_arrhenius_viscosity(t_bad, 0.0, xp=np)
        assert bool(jnp.isnan(compute_arrhenius_viscosity(t_bad, 0.0, xp=jnp)))
    assert bool(
        jnp.isnan(compute_arrhenius_viscosity(1600.0, 0.0, water_prefactor=0.0, xp=jnp))
    )
    assert bool(
        jnp.isnan(compute_arrhenius_viscosity(1600.0, 0.0, water_prefactor=-1.0, xp=jnp))
    )

    # 5. Water prefactor clipped smoothly in log space after multiplication:
    # At T=520 K, P=0, wp=0.1: unclipped log10(eta) = 39.731168 is 0.2688 decades below ceiling 40.0.
    raw_520 = 1.0e21 * 0.1 * math.exp(25.0 * 1400.0 * (1.0 / 520.0 - 1.0 / 1600.0))
    log10_520 = math.log10(raw_520)
    expected_520 = 10.0 ** (log10_520 - 0.05 * math.log1p(math.exp((log10_520 - 40.0) / 0.05)))
    assert compute_arrhenius_viscosity(520.0, 0.0, water_prefactor=0.1) == pytest.approx(
        expected_520, rel=1e-12
    )
    assert float(
        compute_arrhenius_viscosity(520.0, 0.0, water_prefactor=0.1, xp=jnp)
    ) == pytest.approx(expected_520, rel=1e-12)
    assert compute_arrhenius_viscosity(300.0, 0.0, water_prefactor=0.1) == pytest.approx(
        1.0e40, rel=1e-12
    )

    # 6. viscosity_solid, phi_rheo, and phi_width are required in compute_stagnant_lid_state.
    P = np.linspace(135.0e9, 1.0e5, 20)
    T = np.linspace(2500.0, 1200.0, 20)
    F_conv = np.full(20, 50.0)
    F_conv[-1] = 0.0
    F_tot = np.full(20, 60.0)
    phi = np.zeros(20)
    params_fixed_high = SolidRheologyParams(
        enabled=True, lid_base_mode='fixed', lid_base_temperature=5000.0
    )
    with pytest.raises(TypeError):
        compute_stagnant_lid_state(
            radii=r,
            temperature=T,
            pressure=P,
            convective_flux=F_conv,
            total_flux=F_tot,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params_fixed_high,
        )
    with pytest.raises(ValueError, match='viscosity_solid, phi_rheo, and phi_width'):
        compute_stagnant_lid_state(
            radii=r,
            temperature=T,
            pressure=P,
            convective_flux=F_conv,
            total_flux=F_tot,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params_fixed_high,
            viscosity_solid=None,  # type: ignore[arg-type]
            phi_rheo=0.4,
            phi_width=0.15,
        )

    # 7. In 'fixed' mode (compact non-convective weight) when T < lid_base_temperature=5000 K everywhere: r_lid_base sits
    # at r[-2] when nodes 0..n-2 convect, r[-1] (d_lid = 0) when all convect, and r[0] when none do.
    st_full_lid = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_fixed_high,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    assert float(st_full_lid['r_lid_base']) == pytest.approx(r[-2], rel=1e-9)
    assert float(st_full_lid['d_lid']) == pytest.approx(r[-1] - r[-2], rel=1e-4)

    st_all_conv = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=np.full_like(r, 50.0),
        total_flux=F_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_fixed_high,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    assert float(st_all_conv['r_lid_base']) == pytest.approx(r[-1], rel=1e-9)
    assert float(st_all_conv['d_lid']) == 0.0
    assert bool(st_all_conv['has_lid']) is False

    st_no_conv = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=np.zeros_like(r),
        total_flux=F_tot,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_fixed_high,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    assert float(st_no_conv['r_lid_base']) == pytest.approx(r[0], rel=1e-9)

    # 8. NaN in total_flux or temperature propagates NaN to lid diagnostics (NumPy and JAX).
    params_rheo = SolidRheologyParams(enabled=True)
    f_tot_nan = F_tot.copy()
    f_tot_nan[5] = np.nan
    t_nan = T.copy()
    t_nan[5] = np.nan
    for xp_mod in (np, jnp):
        for bad_t, bad_ftot in ((T, f_tot_nan), (t_nan, F_tot)):
            st_nan = compute_stagnant_lid_state(
                radii=xp_mod.asarray(r),
                temperature=xp_mod.asarray(bad_t),
                pressure=xp_mod.asarray(P),
                convective_flux=xp_mod.asarray(F_conv),
                total_flux=xp_mod.asarray(bad_ftot),
                solidus_temperature=None,
                melt_fraction=xp_mod.asarray(phi),
                params=params_rheo,
                unyielded_velocity=xp_mod.asarray(np.full(20, 1.0e-9)),
                viscosity_solid=1.0e21,
                phi_rheo=0.4,
                phi_width=0.15,
                xp=xp_mod,
            )
            for key in ('d_lid', 'tau_d', 'theta', 'eta_i', 'mu_i', 'v_m', 'v_i', 'lid_regime'):
                assert bool(xp_mod.isnan(st_nan[key])), (
                    f'{key} must be NaN under {xp_mod.__name__}'
                )
            assert bool(xp_mod.all(xp_mod.isnan(st_nan['w_lid'])))

    # 9. API: xp is keyword-only and last in viscous_mixing_length_factor; stress_closure takes xp.
    with pytest.raises(TypeError, match='positional arguments'):
        viscous_mixing_length_factor(r, r[0], r[-1], ml, 0.22, 1.0, 10.0, 0.17, jnp)
    fac_j = viscous_mixing_length_factor(
        jnp.asarray(r), r[0], r[-1], jnp.asarray(ml), 0.22, 1.0, 10.0, 0.17, xp=jnp
    )
    assert isinstance(fac_j, jax.Array)
    sr_j = stress_closure(
        'local', jnp.asarray([1.0e-9, 2.0e-9]), jnp.asarray([1.0e5, 1.0e5]), xp=jnp
    )
    assert isinstance(sr_j, jax.Array)
    np.testing.assert_allclose(np.asarray(sr_j), np.array([5.0e-15, 1.0e-14]), rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('lid_base_mode', ['rheological', 'fixed'])
@pytest.mark.parametrize(
    'nan_field',
    [
        'radii',
        'temperature',
        'pressure',
        'convective_flux',
        'total_flux',
        'melt_fraction',
        'density',
        'gravity',
        'thermal_expansivity',
        'thermal_diffusivity',
        'solidus_temperature',
        'unyielded_velocity',
        'viscosity_mixture',
        'viscosity_solid',
        'phi_rheo',
        'phi_width',
        'a_rh_fb',
        'C4',
    ],
)
def test_lid_state_nan_propagation_per_input(nan_field: str, lid_base_mode: str):
    """Verify NaN in each array or scalar input propagates NaN in both lid_base_modes."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    _, jnp = _import_jax_x64()
    n_nodes = 20
    base_arrays = {
        'radii': np.linspace(3.48e6, 6.371e6, n_nodes),
        'temperature': np.linspace(2600.0, 300.0, n_nodes),
        'pressure': np.linspace(135.0e9, 0.0, n_nodes),
        'convective_flux': np.where(np.linspace(3.48e6, 6.371e6, n_nodes) < 6.0e6, 100.0, 0.0),
        'total_flux': np.full(n_nodes, 100.0),
        'melt_fraction': np.zeros(n_nodes),
        'density': np.full(n_nodes, 3300.0),
        'gravity': np.full(n_nodes, 9.81),
        'thermal_expansivity': np.full(n_nodes, 3.0e-5),
        'thermal_diffusivity': np.full(n_nodes, 1.0e-6),
        'solidus_temperature': 1400.0 + 2.4e-8 * np.linspace(135.0e9, 0.0, n_nodes),
        'unyielded_velocity': np.full(n_nodes, 1.0e-9),
        'viscosity_mixture': np.full(n_nodes, 1.0e20),
    }
    scalars = {
        'viscosity_solid': 1.0e21,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
        'a_rh_fb': 1.3,
        'C4': 0.125,
    }
    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0, lid_base_mode=lid_base_mode)
    if nan_field in base_arrays:
        bad = base_arrays[nan_field].copy()
        bad[5] = np.nan
        base_arrays[nan_field] = bad
    else:
        scalars[nan_field] = float('nan')

    for xp_mod in (np, jnp):
        kw = {k: xp_mod.asarray(v) for k, v in base_arrays.items()}
        st = compute_stagnant_lid_state(
            **kw,
            **scalars,
            params=params,
            xp=xp_mod,
        )
        for key in (
            'T_i',
            'theta',
            'eta_i',
            'mu_i',
            'T_lid',
            'T_lid_iso',
            'r_lid_base',
            'P_lid_base',
            'tau_y_lid',
            'd_lid',
            'v_m',
            'v_i',
            'tau_d',
            'lid_regime',
        ):
            assert bool(xp_mod.isnan(st[key])), (
                f'Expected {key} to be NaN when {nan_field} has NaN in {lid_base_mode} under {xp_mod.__name__}'
            )
        assert bool(xp_mod.all(xp_mod.isnan(st['w_lid'])))


@pytest.mark.unit
def test_water_prefactor_and_w_lid_nan_propagation():
    """Verify NaN propagation in compute_arrhenius_viscosity, compute_effective_viscosity, and viscous_mixing_length_factor."""
    from aragog.rheology import viscous_mixing_length_factor

    _, jnp = _import_jax_x64()
    assert np.isnan(compute_arrhenius_viscosity(1600.0, 0.0, water_prefactor=np.nan))
    res_np_arr = compute_arrhenius_viscosity(
        np.array([1600.0, 1600.0]),
        np.array([0.0, 0.0]),
        water_prefactor=np.array([np.nan, 0.5]),
    )
    assert np.isnan(res_np_arr[0]) and np.isfinite(res_np_arr[1])

    assert bool(
        jnp.isnan(compute_arrhenius_viscosity(1600.0, 0.0, water_prefactor=jnp.nan, xp=jnp))
    )
    res_jx_arr = compute_arrhenius_viscosity(
        jnp.array([1600.0, 1600.0]),
        jnp.array([0.0, 0.0]),
        water_prefactor=jnp.array([jnp.nan, 0.5]),
        xp=jnp,
    )
    assert bool(jnp.isnan(res_jx_arr[0])) and bool(jnp.isfinite(res_jx_arr[1]))

    r_ml = np.linspace(3.48e6, 6.371e6, 11)
    ml_base = np.minimum(r_ml - r_ml[0], r_ml[-1] - r_ml)
    ml_nan = ml_base.copy()
    ml_nan[5] = np.nan
    r_nan = r_ml.copy()
    r_nan[5] = np.nan
    for xp_mod in (np, jnp):
        fac_ml_nan = viscous_mixing_length_factor(
            xp_mod.asarray(r_ml),
            r_ml[0],
            r_ml[-1],
            xp_mod.asarray(ml_nan),
            0.22,
            xp=xp_mod,
        )
        assert bool(xp_mod.isnan(fac_ml_nan[5]))
        assert bool(xp_mod.isfinite(fac_ml_nan[4]))

        fac_r_nan = viscous_mixing_length_factor(
            xp_mod.asarray(r_nan),
            r_ml[0],
            r_ml[-1],
            xp_mod.asarray(ml_base),
            0.22,
            xp=xp_mod,
        )
        assert bool(xp_mod.isnan(fac_r_nan[5]))

        eta_wl_nan = compute_effective_viscosity(
            eta_diff=xp_mod.asarray([1.0e24, 1.0e24]),
            eta_i=xp_mod.asarray(1.0e20),
            tau_d=xp_mod.asarray(1.0e5),
            tau_y_lid=xp_mod.asarray(1.0e6),
            w_lid=xp_mod.asarray([np.nan, 0.5]),
            stress_closure_mode='lid',
            xp=xp_mod,
        )
        assert bool(xp_mod.isnan(eta_wl_nan[0]))
        assert bool(xp_mod.isfinite(eta_wl_nan[1]))

        eta_input_nan_with_wl_zero = compute_effective_viscosity(
            eta_diff=xp_mod.asarray([1.0e24, 1.0e24]),
            eta_i=xp_mod.asarray(np.nan),
            tau_d=xp_mod.asarray(2.0e6),
            tau_y_lid=xp_mod.asarray(1.0e6),
            w_lid=xp_mod.asarray([0.0, 1.0]),
            stress_closure_mode='lid',
            xp=xp_mod,
        )
        assert bool(xp_mod.all(xp_mod.isnan(eta_input_nan_with_wl_zero)))


@pytest.mark.unit
def test_compute_stagnant_lid_state_input_validation():
    """Verify compute_stagnant_lid_state validates radii, node count, lengths, viscosity_solid, and phi_width."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    n_nodes = 10
    valid_kw = {
        'radii': np.linspace(3.48e6, 6.371e6, n_nodes),
        'temperature': np.linspace(2600.0, 300.0, n_nodes),
        'pressure': np.linspace(135.0e9, 0.0, n_nodes),
        'convective_flux': np.full(n_nodes, 80.0),
        'total_flux': np.full(n_nodes, 100.0),
        'solidus_temperature': None,
        'melt_fraction': np.zeros(n_nodes),
        'params': SolidRheologyParams(enabled=True),
        'viscosity_solid': 1.0e21,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
    }

    with pytest.raises(ValueError, match='radii must be strictly increasing'):
        compute_stagnant_lid_state(**{**valid_kw, 'radii': valid_kw['radii'][::-1]})

    dup_r = valid_kw['radii'].copy()
    dup_r[4] = dup_r[3]
    with pytest.raises(ValueError, match='radii must be strictly increasing'):
        compute_stagnant_lid_state(**{**valid_kw, 'radii': dup_r})

    for bad_n in (1, 2):
        short_kw = {
            **valid_kw,
            'radii': np.linspace(3.48e6, 6.371e6, bad_n),
            'temperature': np.linspace(2600.0, 300.0, bad_n),
            'pressure': np.linspace(135.0e9, 0.0, bad_n),
            'convective_flux': np.full(bad_n, 80.0),
            'total_flux': np.full(bad_n, 100.0),
            'melt_fraction': np.zeros(bad_n),
        }
        with pytest.raises(ValueError, match='at least 3 nodes'):
            compute_stagnant_lid_state(**short_kw)

    with pytest.raises(ValueError, match='same 1-D length as radii'):
        compute_stagnant_lid_state(**{**valid_kw, 'temperature': np.linspace(2600.0, 300.0, 9)})

    with pytest.raises(ValueError, match='same 1-D length as radii'):
        compute_stagnant_lid_state(**{**valid_kw, 'density': np.full(9, 3300.0)})

    for bad_vs in (0.0, -1.0e21, np.inf, -np.inf):
        with pytest.raises(ValueError, match='viscosity_solid must be positive'):
            compute_stagnant_lid_state(**{**valid_kw, 'viscosity_solid': bad_vs})

    for bad_pw in (0.0, -0.15, np.inf, -np.inf):
        with pytest.raises(ValueError, match='phi_width must be positive'):
            compute_stagnant_lid_state(**{**valid_kw, 'phi_width': bad_pw})

    for bad_pr in (np.inf, -np.inf, -0.1, 1.1):
        with pytest.raises(ValueError, match='phi_rheo must be'):
            compute_stagnant_lid_state(**{**valid_kw, 'phi_rheo': bad_pr})

    for bad_idx, bad_val in ((0, -np.inf), (-1, np.inf), (-1, -np.inf)):
        inf_r = valid_kw['radii'].copy()
        inf_r[bad_idx] = bad_val
        with pytest.raises(ValueError, match='radii must be strictly increasing'):
            compute_stagnant_lid_state(**{**valid_kw, 'radii': inf_r})

    for opt_name in (
        'solidus_temperature',
        'unyielded_velocity',
        'viscosity_mixture',
        'density',
        'gravity',
        'thermal_expansivity',
        'thermal_diffusivity',
    ):
        with pytest.raises(ValueError, match=rf'{opt_name}.*same 1-D length as radii'):
            compute_stagnant_lid_state(**{**valid_kw, opt_name: np.ones((5, 2), dtype=float)})


def _valid_lid_state_kwargs(n_nodes: int = 10) -> dict[str, Any]:
    """Build valid 1-D arguments for compute_stagnant_lid_state input-validation tests."""
    return {
        'radii': np.linspace(3.48e6, 6.371e6, n_nodes),
        'temperature': np.linspace(2600.0, 300.0, n_nodes),
        'pressure': np.linspace(135.0e9, 0.0, n_nodes),
        'convective_flux': np.full(n_nodes, 80.0),
        'total_flux': np.full(n_nodes, 100.0),
        'solidus_temperature': None,
        'melt_fraction': np.zeros(n_nodes),
        'params': SolidRheologyParams(enabled=True),
        'viscosity_solid': 1.0e21,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
    }


@pytest.mark.unit
def test_stagnant_lid_state_t_sol_0_inf_and_mismatch_validation():
    """Verify SolidRheologyParams rejects invalid t_sol_0 and compute_stagnant_lid_state rejects t_sol_0 kwarg."""
    valid_kw = _valid_lid_state_kwargs()
    with pytest.raises(TypeError, match='t_sol_0'):
        compute_stagnant_lid_state(**{**valid_kw, 't_sol_0': 1500.0})
    for bad_ts0 in (np.inf, -np.inf, 0.0, -100.0, np.nan):
        with pytest.raises(ValueError, match='t_sol_0'):
            SolidRheologyParams(enabled=True, t_sol_0=bad_ts0)
        with pytest.raises(ValueError, match='t_sol_0'):
            dataclasses.replace(valid_kw['params'], t_sol_0=bad_ts0)


@pytest.mark.unit
def test_stagnant_lid_state_c4_bounds_validation():
    """Verify C4 rejects non-positive and +-inf values while propagating NaN."""
    valid_kw = _valid_lid_state_kwargs()
    for bad_c4 in (0.0, -0.125, np.inf, -np.inf):
        with pytest.raises(ValueError, match='C4 must be positive and finite'):
            compute_stagnant_lid_state(**{**valid_kw, 'C4': bad_c4})
    st_nan_c4 = compute_stagnant_lid_state(**{**valid_kw, 'C4': np.nan})
    assert np.isnan(float(st_nan_c4['tau_d']))


@pytest.mark.unit
def test_stagnant_lid_state_a_rh_fb_bounds_validation():
    """Verify a_rh_fb rejects non-positive and +-inf values while propagating NaN."""
    valid_kw = _valid_lid_state_kwargs()
    for bad_arh in (0.0, -1.3, np.inf, -np.inf):
        with pytest.raises(ValueError, match='a_rh_fb must be positive and finite'):
            compute_stagnant_lid_state(**{**valid_kw, 'a_rh_fb': bad_arh})
    st_nan_arh = compute_stagnant_lid_state(**{**valid_kw, 'a_rh_fb': np.nan})
    assert np.isnan(float(st_nan_arh['tau_d']))


@pytest.mark.unit
def test_stagnant_lid_state_rejects_array_valued_scalars():
    """Verify array-valued, bool, and string scalar arguments raise ValueError instead of being accepted."""
    from aragog.rheology import _require_scalar

    for bad_scalar in (
        '1e21',
        b'1e21',
        np.array('1e21'),
        np.array(b'1e21'),
        True,
        False,
        np.bool_(True),
    ):
        with pytest.raises(ValueError, match='viscosity_solid must be a scalar'):
            _require_scalar(bad_scalar, 'viscosity_solid')

    n_nodes = 10
    valid_kw = _valid_lid_state_kwargs(n_nodes)
    for scalar_name, bad_arr in (
        ('C4', '0.125'),
        ('phi_rheo', '0.4'),
        ('viscosity_solid', '1e21'),
        ('viscosity_solid', np.array('1e21')),
        ('viscosity_solid', np.array(b'1e21')),
        ('viscosity_solid', True),
        ('phi_width', np.bool_(True)),
        ('viscosity_solid', np.full(n_nodes, 1.0e21)),
        ('phi_width', np.full(n_nodes, 0.15)),
        ('phi_rheo', np.full(n_nodes, -5.0)),
        ('a_rh_fb', np.full(n_nodes, 1.3)),
        ('C4', np.full(n_nodes, 0.125)),
    ):
        call_kw = {**valid_kw, scalar_name: bad_arr}
        with pytest.raises(ValueError, match=rf'{scalar_name} must be a scalar'):
            compute_stagnant_lid_state(**call_kw)

    assert _require_scalar(10**21, 'viscosity_solid') is None
    st_float = compute_stagnant_lid_state(**valid_kw)
    st_int = compute_stagnant_lid_state(**{**valid_kw, 'viscosity_solid': 10**21})
    for k, v_float in st_float.items():
        np.testing.assert_array_equal(st_int[k], v_float)


@pytest.mark.unit
def test_stagnant_lid_state_jitted_invalid_scalars_propagate_nan():
    """Verify jax.jit traced invalid scalar values propagate NaN through zero_or_nan."""
    valid_kw = _valid_lid_state_kwargs()
    jax, jnp = _import_jax_x64()
    j_kw = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in valid_kw.items()}

    @jax.jit
    def _jit_eval(vs, pw, pr, arh, c4):
        return compute_stagnant_lid_state(
            **{
                **j_kw,
                'viscosity_solid': vs,
                'phi_width': pw,
                'phi_rheo': pr,
                'a_rh_fb': arh,
                'C4': c4,
                'xp': jnp,
            }
        )

    st_jit_ok = _jit_eval(1.0e21, 0.15, 0.4, 1.3, 0.125)
    assert bool(jnp.isfinite(st_jit_ok['tau_d']))
    for bad_args in (
        (-1.0e21, 0.15, 0.4, 1.3, 0.125),
        (jnp.inf, 0.15, 0.4, 1.3, 0.125),
        (1.0e21, 0.0, 0.4, 1.3, 0.125),
        (1.0e21, 0.15, 1.5, 1.3, 0.125),
        (1.0e21, 0.15, 0.4, 0.0, 0.125),
        (1.0e21, 0.15, 0.4, 1.3, -0.125),
    ):
        try:
            st_jit_bad = _jit_eval(*bad_args)
        except ValueError as exc:
            pytest.fail(f'traced invalid scalar must give NaN, not raise: {exc}')
        assert not bool(st_jit_bad['has_lid'])
        for k, v in st_jit_bad.items():
            if k != 'has_lid':
                assert bool(jnp.all(jnp.isnan(v))), f'Expected NaN for {k} on {bad_args}'


@pytest.mark.unit
def test_compute_effective_viscosity_lid_inf_eta_diff_with_w_lid_one():
    """Verify w_lid=1 with eta_diff=inf raises ValueError on np and returns NaN for the inf element on JAX."""
    _, jnp = _import_jax_x64()
    with pytest.raises(ValueError, match='eta_diff'):
        compute_effective_viscosity(
            eta_diff=np.asarray([np.inf, 1.0e24]),
            eta_i=np.asarray(1.0e20),
            tau_d=np.asarray(2.0e6),
            tau_y_lid=np.asarray(1.0e6),
            w_lid=np.asarray([1.0, 1.0]),
            stress_closure_mode='lid',
            xp=np,
        )
    res_inf_jax = compute_effective_viscosity(
        eta_diff=jnp.asarray([jnp.inf, 1.0e24]),
        eta_i=jnp.asarray(1.0e20),
        tau_d=jnp.asarray(2.0e6),
        tau_y_lid=jnp.asarray(1.0e6),
        w_lid=jnp.asarray([1.0, 1.0]),
        stress_closure_mode='lid',
        xp=jnp,
    )
    assert bool(jnp.isnan(res_inf_jax[0]))
    assert float(res_inf_jax[1]) == pytest.approx(5.0e19, rel=1.0e-12)


@pytest.mark.unit
def test_compute_effective_viscosity_local_rejects_both_tau_y_and_profile():
    """Verify local stress closure mode raises ValueError when both tau_y and tau_y_profile are given."""
    with pytest.raises(ValueError, match='not both'):
        compute_effective_viscosity(
            eta_diff=1.0e21,
            strain_rate=1.0e-14,
            tau_y=1.0e7,
            tau_y_profile=1.0e3,
            stress_closure_mode='local',
        )


@pytest.mark.unit
def test_compute_effective_viscosity_w_lid_bounds_validation():
    """Verify w_lid outside [0, 1] or +-inf raises ValueError while NaN propagates."""
    for bad_wl in (
        -0.1,
        1.5,
        np.inf,
        -np.inf,
        np.array([0.5, 1.2]),
        np.array([-0.01, 0.5]),
        np.array([0.5, np.inf]),
        np.array([-np.inf, 0.5]),
    ):
        with pytest.raises(ValueError, match='w_lid must be in'):
            compute_effective_viscosity(
                eta_diff=np.array([1.0e24, 1.0e24]),
                eta_i=1.0e20,
                tau_d=2.0e6,
                tau_y_lid=1.0e6,
                w_lid=bad_wl,
                stress_closure_mode='lid',
            )
    nan_res = compute_effective_viscosity(
        eta_diff=np.array([1.0e24, 1.0e24]),
        eta_i=1.0e20,
        tau_d=2.0e6,
        tau_y_lid=1.0e6,
        w_lid=np.array([np.nan, 0.5]),
        stress_closure_mode='lid',
    )
    assert np.isnan(nan_res[0])


@pytest.mark.unit
def test_solid_rheology_params_boundary_values():
    """Verify SolidRheologyParams boundary acceptance and rejection edges."""
    # Acceptance of zero cohesion and zero friction coefficient.
    try:
        p_zero = SolidRheologyParams(enabled=False, yield_stress_c=0.0, yield_stress_mu=0.0)
    except ValueError as e:
        pytest.fail(f'yield_stress_c=0.0 and yield_stress_mu=0.0 must be accepted: {e}')
    assert p_zero.yield_stress_c == pytest.approx(0.0, abs=1e-12)
    assert p_zero.yield_stress_mu == pytest.approx(0.0, abs=1e-12)

    # Acceptance of upper bound viscosity_max_log10 = 300.0.
    try:
        p_cap = SolidRheologyParams(enabled=False, viscosity_max_log10=300.0)
    except ValueError as e:
        pytest.fail(f'viscosity_max_log10=300.0 must be accepted: {e}')
    assert p_cap.viscosity_max_log10 == pytest.approx(300.0, abs=1e-12)

    # Rejection of water_prefactor <= 0 on dataclass.
    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        SolidRheologyParams(enabled=False, water_prefactor=0.0)

    # Rejection of viscosity_max_log10 at lower open bound (20.0) and above upper bound (300.0000001).
    with pytest.raises(ValueError, match='viscosity_max_log10 must be in'):
        SolidRheologyParams(enabled=False, viscosity_max_log10=20.0)
    with pytest.raises(ValueError, match='viscosity_max_log10 must be in'):
        SolidRheologyParams(enabled=False, viscosity_max_log10=300.0000001)


@pytest.mark.unit
def test_stagnant_lid_state_rejects_2d_and_0d_core_arrays():
    """Verify compute_stagnant_lid_state rejects (N, 1) and 0-d core arrays with named ValueError."""
    n_nodes = 30
    radii = np.linspace(3480.0e3, 6370.0e3, n_nodes)
    temp = np.full(n_nodes, 1600.0)
    press = np.linspace(135.0e9, 1.0e5, n_nodes)
    f_conv = np.full(n_nodes, 100.0)
    f_tot = np.full(n_nodes, 100.0)
    phi = np.zeros(n_nodes)
    base_kwargs = {
        'radii': radii,
        'temperature': temp,
        'pressure': press,
        'convective_flux': f_conv,
        'total_flux': f_tot,
        'melt_fraction': phi,
        'solidus_temperature': None,
        'params': SolidRheologyParams(enabled=True),
        'viscosity_solid': 1.0e20,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
    }

    # 2D (N, 1) and 0-d radii
    with pytest.raises(ValueError, match='radii must be a 1-D array'):
        compute_stagnant_lid_state(**{**base_kwargs, 'radii': radii.reshape(-1, 1)})
    with pytest.raises(ValueError, match='radii must be a 1-D array'):
        compute_stagnant_lid_state(**{**base_kwargs, 'radii': np.array(6370.0e3)})

    # 2D (N, 1) and 0-d for each other core array
    for arr_name in (
        'temperature',
        'pressure',
        'convective_flux',
        'total_flux',
        'melt_fraction',
    ):
        with pytest.raises(
            ValueError, match=f'{arr_name} must have the same 1-D length as radii'
        ):
            compute_stagnant_lid_state(
                **{**base_kwargs, arr_name: base_kwargs[arr_name].reshape(-1, 1)}
            )
        with pytest.raises(
            ValueError, match=f'{arr_name} must have the same 1-D length as radii'
        ):
            compute_stagnant_lid_state(**{**base_kwargs, arr_name: np.array(100.0)})


@pytest.mark.unit
def test_viscous_mixing_length_factor_nan_scalar_inputs_propagate():
    """Verify NaN scalar inputs propagate to boundary nodes in viscous_mixing_length_factor."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)

    for kwargs in (
        {'s_top_inf': float('nan')},
        {'bottom_slope': float('nan')},
        {'theta': 10.0, 'c_theta': float('nan')},
    ):
        q = viscous_mixing_length_factor(r, r[0], r[-1], ml, **kwargs)
        assert np.isnan(q[0])
        assert np.isnan(q[-1])
        assert np.all(np.isnan(q))

    q_rin = viscous_mixing_length_factor(r, float('nan'), r[-1], ml)
    assert np.isnan(q_rin[0]) and np.isnan(q_rin[-1]) and np.all(np.isnan(q_rin))

    q_rout = viscous_mixing_length_factor(r, r[0], float('nan'), ml)
    assert np.isnan(q_rout[0]) and np.isnan(q_rout[-1]) and np.all(np.isnan(q_rout))


def _scan_ast_for_direct_rheology_imports(tree, filename: str | None = None) -> list[str]:
    """Scan an AST tree at any depth for direct imports of aragog.rheology or rheology_lid."""
    import ast
    from pathlib import Path

    def hit(name: str) -> bool:
        return name.split('.')[:2] in (['aragog', 'rheology'], ['aragog', 'rheology_lid'])

    pkg_parts: list[str] = []
    if filename:
        p_parts = Path(filename).resolve().parent.parts
        if 'aragog' in p_parts:
            idx = len(p_parts) - 1 - p_parts[::-1].index('aragog')
            pkg_parts = list(p_parts[idx:])

    aragog_names = {
        alias.asname or 'aragog'
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == 'aragog'
    } | {'aragog'}

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if hit(alias.name):
                    violations.append(f'import {alias.name}')
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if node.level > 0:
                if pkg_parts:
                    base = (
                        pkg_parts[: len(pkg_parts) - node.level + 1]
                        if node.level <= len(pkg_parts)
                        else []
                    )
                    mod = f'{".".join(base)}.{mod}' if mod else '.'.join(base)
                else:
                    mod = f'aragog.{mod}' if mod else 'aragog'
            if hit(mod):
                violations.append(f'from {mod} import ...')
            for alias in node.names:
                if hit(f'{mod}.{alias.name}'):
                    violations.append(f'from {mod} import {alias.name}')
        elif isinstance(node, ast.Attribute):
            if (
                node.attr in ('rheology', 'rheology_lid')
                and isinstance(node.value, ast.Name)
                and node.value.id in aragog_names
            ):
                violations.append(f'attribute access: aragog.{node.attr}')
        elif isinstance(node, ast.Call):
            func = node.func
            is_import = (isinstance(func, ast.Name) and func.id == '__import__') or (
                isinstance(func, ast.Attribute) and func.attr == 'import_module'
            )
            if (
                is_import
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                arg_val = node.args[0].value
                if hit(arg_val):
                    violations.append(f'import call: {arg_val}')
    return violations


@pytest.mark.unit
def test_solver_modules_do_not_direct_import_rheology_kernels():
    """Verify solver modules do not directly import aragog.rheology or aragog.rheology_lid.

    The kernels are loaded transitively through aragog.config.phases
    (SolidRheologyParams), which is allowed.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    solver_rel_paths = [
        'src/aragog/jax/phase.py',
        'src/aragog/solver/entropy_state.py',
        'src/aragog/eos/entropy_phase.py',
    ]

    for rel_path in solver_rel_paths:
        file_path = repo_root / rel_path
        tree = ast.parse(file_path.read_text(encoding='utf-8'), filename=str(file_path))
        direct_imports = _scan_ast_for_direct_rheology_imports(tree, filename=str(file_path))
        assert not direct_imports, f'{rel_path} has direct rheology imports: {direct_imports}'

    # Canary 1: synthetic direct top-level import must be caught.
    synth_tree = ast.parse('import aragog.rheology\n')
    assert _scan_ast_for_direct_rheology_imports(synth_tree), (
        'Canary 1 failed: direct top-level import was not caught'
    )

    # Canary 2: function-local import must be caught.
    local_import_code = (
        'def _lazy_probe():\n    from aragog.rheology import compute_arrhenius_viscosity\n'
    )
    assert _scan_ast_for_direct_rheology_imports(ast.parse(local_import_code)), (
        'Canary 2 failed: function-local import was not caught'
    )

    # Canary 3: dynamic importlib call must be caught.
    importlib_code = "import importlib\nmod = importlib.import_module('aragog.rheology_lid')\n"
    assert _scan_ast_for_direct_rheology_imports(ast.parse(importlib_code)), (
        'Canary 3 failed: importlib.import_module call was not caught'
    )

    for phase_file in (
        str(repo_root / 'src/aragog/jax/phase.py'),
        '/home/runner/work/aragog/aragog/src/aragog/jax/phase.py',
    ):
        # Canary 4: relative import with module ('from ..rheology import X').
        rel_code_1 = 'from ..rheology import compute_arrhenius_viscosity\n'
        assert _scan_ast_for_direct_rheology_imports(
            ast.parse(rel_code_1, filename=phase_file), filename=phase_file
        ), f'Canary 4 failed for {phase_file}: relative import from ..rheology was not caught'

        # Canary 5: relative import from parent ('from .. import rheology_lid').
        rel_code_2 = 'from .. import rheology_lid\n'
        assert _scan_ast_for_direct_rheology_imports(
            ast.parse(rel_code_2, filename=phase_file), filename=phase_file
        ), f'Canary 5 failed for {phase_file}: relative import from .. was not caught'

    # Canary 6: attribute chain access (including module alias) must be caught.
    attr_code = 'import aragog as ar\nx = ar.rheology.compute_arrhenius_viscosity\n'
    assert _scan_ast_for_direct_rheology_imports(ast.parse(attr_code)), (
        'Canary 6 failed: attribute chain aragog.rheology was not caught'
    )


@pytest.mark.unit
def test_compute_solidus_reference_values_crossings_and_paleos(tmp_path, capsys, monkeypatch):
    """Verify PALEOS defaults, multi-crossing reporting, and non-1-GPa grid slicing."""
    import importlib.util
    import re
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / 'tools'
        / 'verification'
        / 'compute_solidus_reference_values.py'
    )
    spec = importlib.util.spec_from_file_location(
        'compute_solidus_reference_values', script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.main([])
    out_default = capsys.readouterr().out
    assert 'PALEOS MgSiO3 solidus' in out_default
    assert 'T_sol(1 GPa): 1563.0407 K (1-dec: 1563.0 K, int: 1563 K)' in out_default
    assert '[1464.8000, 3066.9899, 3672.6551, 4397.9263, 4754.8271]' in out_default
    assert 'no crossings over 0..135 GPa' in out_default
    assert '-1108.1 K' in out_default
    assert (
        'Max relative linear deviation over 0..135 GPa: -35.04% (-35.0%) at 20.495 GPa (20.5 GPa)'
        in out_default
    )
    assert 'Chebyshev minimax line over 0..135 GPa: 1979.7 K + 24.37 K/GPa * P' in out_default
    assert 'max abs deviation: 514.93 K (514.9 K)' in out_default
    mm_line_default = [ln for ln in out_default.splitlines() if 'Chebyshev minimax line' in ln][
        0
    ]
    mm_match_default = re.search(
        r'exact:\s*([0-9.]+)\s*K\s*\+\s*([0-9.]+)\s*K/GPa\s*\*.*'
        r'max abs deviation:\s*([0-9.]+)\s*K\s*\(([0-9.]+)\s*K\)\s*at\s*([0-9., ]+)\s*GPa',
        mm_line_default,
    )
    assert mm_match_default is not None
    assert float(mm_match_default.group(1)) == pytest.approx(1979.73, rel=1.0e-6)
    assert float(mm_match_default.group(2)) == pytest.approx(24.3706, rel=1.0e-6)
    assert float(mm_match_default.group(3)) == pytest.approx(514.93, abs=0.01)
    assert float(mm_match_default.group(4)) == pytest.approx(514.9, abs=0.1)
    peaks_default = [float(p.strip()) for p in mm_match_default.group(5).split(',')]
    assert len(peaks_default) == 3
    for p_act, p_exp in zip(peaks_default, [0.000, 35.965, 135.000]):
        assert p_act == pytest.approx(p_exp, abs=1.0e-3)
    assert (
        'RMS: 350.64 K (350.6 K); crossings at 6.1, 98.5 GPa (all: [6.15, 98.54] GPa)'
        in out_default
    )
    assert 'Least-squares (L2) line over 0..135 GPa: 2489.3 K + 19.10 K/GPa * P' in out_default
    assert 'max abs deviation: 1024.49 K (1024.5 K) at 0.0 GPa' in out_default
    assert (
        'RMS: 239.01 K (239.0 K); crossings at 18.8, 99.9 GPa (all: [18.82, 99.89] GPa)'
        in out_default
    )

    # 1-crossing synthetic table (crossing at 45 GPa on a 30 GPa step grid).
    p_1 = np.array([0.0, 30.0, 60.0, 100.0, 135.0]) * 1.0e9
    t_lin_1 = 1400.0 + 2.4e-8 * p_1
    t_tab_1 = t_lin_1 + np.array([100.0, 50.0, -50.0, -100.0, -120.0])
    t1_file = tmp_path / 'solidus_1cross.dat'
    np.savetxt(t1_file, np.column_stack([p_1, t_tab_1]))
    c1 = mod.find_solidus_crossings(p_1, t_tab_1, 1400.0, 2.4e-8)
    assert len(c1) == 1
    assert c1[0] == pytest.approx(45.0, rel=1.0e-12)
    mod.main(['--table', str(t1_file)])
    out_1 = capsys.readouterr().out
    assert 'crossings at 45.0 GPa (all: [45.0] GPa)' in out_1

    monkeypatch.setattr(mod, 'EXPECTED_MD5', '00000000000000000000000000000000')
    with pytest.raises(SystemExit, match='MD5 mismatch'):
        mod.main(['--table', str(t1_file)])
    capsys.readouterr()
    mod.main(['--table', str(t1_file), '--allow-md5-mismatch'])
    capsys.readouterr()
    monkeypatch.setattr(mod, 'EXPECTED_MD5', None)

    t_short = tmp_path / 'solidus_short.dat'
    np.savetxt(t_short, np.column_stack([p_1[:3], t_tab_1[:3]]))
    with pytest.raises(ValueError, match='135.0e9'):
        mod.main(['--table', str(t_short)])

    # 3-crossing table on a non-1-GPa grid: roots at 0.40, 50.0 (node), and 134.0 GPa,
    # plus an excluded root at 145.0 GPa above the 135 GPa mantle bound.
    p_3_gpa = np.array([0.0, 0.2, 0.8, 25.0, 50.0, 80.0, 133.0, 140.0, 150.0])
    p_3 = p_3_gpa * 1.0e9
    t_lin_3 = 1400.0 + 24.0 * p_3_gpa
    offset_3 = np.array([-20.0, -10.0, 20.0, 100.0, 0.0, -80.0, -10.0, 60.0, -60.0])
    t_tab_3 = t_lin_3 - offset_3
    t3_file = tmp_path / 'solidus_3cross.dat'
    np.savetxt(t3_file, np.column_stack([p_3, t_tab_3]))

    c3 = mod.find_solidus_crossings(p_3, t_tab_3, 1400.0, 2.4e-8)
    assert len(c3) == 3
    assert c3[0] == pytest.approx(0.4, rel=1.0e-12)
    assert c3[1] == pytest.approx(50.0, rel=1.0e-12)
    assert c3[2] == pytest.approx(134.0, rel=1.0e-12)

    mod.main(['--table', str(t3_file), '--allow-md5-mismatch'])
    out_3 = capsys.readouterr().out
    assert '0.40 GPa' in out_3
    assert '50.0, 134.0 GPa' in out_3
    assert '[0.4, 50.0, 134.0]' in out_3

    # Canary: the old formatter dropped crossings_gpa[2:] when crossings_gpa[0] < 1.0.
    old_truncated_desc = f'crossings below 1 GPa and at {c3[1]:.1f} GPa'
    assert '134.0' not in old_truncated_desc
    assert '134.0' in mod.format_crossings_description(c3)

    # Copilot non-convex 4-point table: shifted secant gives 450 K max error,
    # while Chebyshev minimax fit via linear programming achieves 375 K.
    p_copilot_gpa = np.array([0.0, 45.0, 90.0, 135.0])
    p_copilot = p_copilot_gpa * 1.0e9
    t_copilot = np.array([1400.0, 3000.0, 3100.0, 4400.0])
    t_copilot_file = tmp_path / 'solidus_copilot_nonconvex.dat'
    np.savetxt(t_copilot_file, np.column_stack([p_copilot, t_copilot]))

    mod.main(['--table', str(t_copilot_file), '--allow-md5-mismatch'])
    out_copilot = capsys.readouterr().out
    assert 'Chebyshev minimax line over 0..135 GPa: 1775.0 K + 18.89 K/GPa * P' in out_copilot
    assert 'max abs deviation: 375.00 K (375.0 K)' in out_copilot
    mm_line_copilot = [ln for ln in out_copilot.splitlines() if 'Chebyshev minimax line' in ln][
        0
    ]
    mm_match_copilot = re.search(
        r'exact:\s*([0-9.]+)\s*K\s*\+\s*([0-9.]+)\s*K/GPa\s*\*.*'
        r'max abs deviation:\s*([0-9.]+)\s*K\s*\(([0-9.]+)\s*K\)\s*at\s*([0-9., ]+)\s*GPa',
        mm_line_copilot,
    )
    assert mm_match_copilot is not None
    assert float(mm_match_copilot.group(1)) == pytest.approx(1775.00, rel=1.0e-6)
    assert float(mm_match_copilot.group(2)) == pytest.approx(18.8889, rel=1.0e-4)
    assert float(mm_match_copilot.group(3)) == pytest.approx(375.0, abs=1.0e-6)
    peaks_copilot = [float(p.strip()) for p in mm_match_copilot.group(5).split(',')]
    assert len(peaks_copilot) == 3
    for p_act, p_exp in zip(peaks_copilot, [0.0, 45.0, 90.0]):
        assert p_act == pytest.approx(p_exp, abs=1.0e-3)
    dev_line = [ln for ln in out_copilot.splitlines() if 'max abs deviation:' in ln][0]
    err_minimax = float(dev_line.split('max abs deviation:')[1].split('K')[0].strip())
    err_copilot = float(np.max(np.abs(t_copilot - (1700.0 + 20.0 * p_copilot_gpa))))
    sec_slope = (t_copilot[-1] - t_copilot[0]) / (p_copilot_gpa[-1] - p_copilot_gpa[0])
    sec_dev = t_copilot - (t_copilot[0] + sec_slope * p_copilot_gpa)
    err_secant = float((np.max(sec_dev) - np.min(sec_dev)) / 2.0)
    assert err_minimax < err_copilot < err_secant


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_smooth_gates_and_scales():
    """Verify smooth lid weights w_d, w_T, w_act_gate, w_sum_gate and smooth scale/scale_mask."""
    import jax

    import aragog.rheology_lid as lid_mod
    from aragog.rheology_lid import compute_stagnant_lid_state

    _, jnp = _import_jax_x64()

    assert getattr(lid_mod, 'NO_LID_THICKNESS_WIDTH', None) == pytest.approx(0.25, rel=1.0e-15)
    assert getattr(lid_mod, 'NO_LID_DT_WIDTH', None) == pytest.approx(2.5e-4, rel=1.0e-15)
    assert getattr(lid_mod, 'CONV_ACTIVE_GATE_WIDTH', None) == pytest.approx(
        0.0025, rel=1.0e-15
    )
    assert getattr(lid_mod, 'CONV_SUM_GATE_WIDTH', None) == pytest.approx(2.5e-7, rel=1.0e-15)
    assert getattr(lid_mod, 'SCALE_FLOOR_WIDTH', None) == pytest.approx(0.005, rel=1.0e-15)
    assert getattr(lid_mod, 'SCALE_MASK_REL_WIDTH', None) == pytest.approx(0.005, rel=1.0e-15)

    w_lim_0 = 0.5 * (1.0 + np.tanh(-4.0))
    w_lim_2x = 0.5 * (1.0 + np.tanh(4.0))
    assert w_lim_0 == pytest.approx(3.353501304664781e-4, rel=1.0e-12)
    assert w_lim_2x == pytest.approx(1.0 - 3.353501304664781e-4, rel=1.0e-12)

    # 1. w_T transition across T_i - T_surf in [0, 2e-3 K] with fixed lid base above column temperature.
    n = 20
    r = np.linspace(3.48e6, 6.37e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    fc = np.full(n, 80.0)
    fc[-1] = 0.0
    ft = np.full(n, 100.0)
    phi = np.zeros(n)
    params_fixed = SolidRheologyParams(
        enabled=True, lid_base_mode='fixed', lid_base_temperature=2500.0
    )

    def _eval_dt(dt_surf: float):
        t_arr = np.full(n, 1600.0)
        t_arr[-1] = 1600.0 - dt_surf
        return compute_stagnant_lid_state(
            radii=r,
            temperature=t_arr,
            pressure=p,
            convective_flux=fc,
            total_flux=ft,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params_fixed,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )

    st_dt_0 = _eval_dt(0.0)
    st_dt_mid = _eval_dt(1.0e-3)
    st_dt_2x = _eval_dt(2.0e-3)
    st_dt_out = _eval_dt(1.0e-2)
    d_raw_dt = float((r[-1] - st_dt_out['r_lid_base']) * st_dt_out['w_active'])
    w_t_mid_expected = (0.5 - w_lim_0) / (1.0 - w_lim_0)
    w_t_2x_expected = (w_lim_2x - w_lim_0) / (1.0 - w_lim_0)
    assert float(st_dt_0['d_lid']) == 0.0
    assert float(st_dt_mid['d_lid']) / d_raw_dt == pytest.approx(w_t_mid_expected, rel=1.0e-6)
    assert float(st_dt_2x['d_lid']) / d_raw_dt == pytest.approx(w_t_2x_expected, rel=1.0e-6)
    # Outside band edge d* = 3.453878e-3 K (at 1e-2 K > 1e-3 + d*), d_lid matches d_raw_dt to 1e-12.
    assert float(st_dt_out['d_lid']) == pytest.approx(d_raw_dt, rel=1.0e-12)

    # Verify w_y uses raw tau_d_lid_raw so yield onset does not shift inside the w_lid_present transition band.
    tau_d_raw_mid = float(st_dt_out['tau_d_lid'])
    t_mid_arr = np.full(n, 1600.0)
    t_mid_arr[-1] = 1600.0 - 1.0e-3
    st_wy_mid = compute_stagnant_lid_state(
        radii=r,
        temperature=t_mid_arr,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(
            enabled=True,
            lid_base_mode='fixed',
            lid_base_temperature=2500.0,
            yield_stress_c=tau_d_raw_mid,
            yield_stress_mu=0.0,
            yield_stress_max=tau_d_raw_mid,
        ),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(st_wy_mid['w_y']) == pytest.approx(0.5 * w_t_mid_expected, rel=1.0e-6)

    grad_dt_fn = jax.grad(
        lambda dt_s: compute_stagnant_lid_state(
            radii=jnp.asarray(r),
            temperature=jnp.concatenate(
                [jnp.full(n - 1, 1600.0), jnp.asarray([1600.0 - dt_s])]
            ),
            pressure=jnp.asarray(p),
            convective_flux=jnp.asarray(fc),
            total_flux=jnp.asarray(ft),
            solidus_temperature=None,
            melt_fraction=jnp.asarray(phi),
            params=params_fixed,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=jnp,
        )['d_lid']
    )
    g_dt_mid = float(grad_dt_fn(jnp.asarray(1.0e-3)))
    assert np.isfinite(g_dt_mid) and g_dt_mid > 0.0

    # Hard-threshold mutant canary for T_i - T_surf > 1e-3 K:
    smooth_jump_dt = abs(
        float(_eval_dt(1.0e-3 + 1.0e-6)['d_lid']) - float(_eval_dt(1.0e-3 - 1.0e-6)['d_lid'])
    )
    hard_jump_dt = d_raw_dt
    assert smooth_jump_dt < 0.01 * d_raw_dt
    with pytest.raises(AssertionError):
        assert hard_jump_dt < 0.01 * d_raw_dt

    # 2. w_d transition across d_lid_raw = 1.0 m (at d_lid_raw = 0.75 m, w_d = 0.5*(1 + tanh(-1))).
    n100 = 100
    r100 = np.linspace(3480.0e3, 6370.0e3, n100)
    t100 = np.full(n100, 1800.0)
    t100[-1] = 1600.0
    p100 = np.linspace(135.0e9, 1.0e5, n100)
    fc100 = np.full(n100, 100.0)
    fc100[-1] = 0.0
    params_dlid = SolidRheologyParams(
        enabled=True, g_homologous=25.0, t_sol_0=1600.0, dt_sol_dp=0.0, arrhenius_t_ref=1600.0
    )
    st_d08 = compute_stagnant_lid_state(
        radii=r100,
        temperature=t100,
        pressure=p100,
        convective_flux=fc100,
        total_flux=np.full(n100, 100.0),
        solidus_temperature=None,
        melt_fraction=np.full(n100, 0.011895),
        params=params_dlid,
        viscosity_solid=1.0e20,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    d_raw_08 = float((r100[-1] - st_d08['r_lid_base']) * st_d08['w_active'])
    expected_wd_08 = (0.5 * (1.0 + np.tanh((d_raw_08 - 1.0) / 0.25)) - w_lim_0) / (
        1.0 - w_lim_0
    )
    assert float(st_d08['d_lid']) == pytest.approx(expected_wd_08 * d_raw_08, rel=1.0e-12)

    # Hard-threshold mutant canary for d_lid_raw > 1.0 m:
    st_d12 = compute_stagnant_lid_state(
        radii=r100,
        temperature=t100,
        pressure=p100,
        convective_flux=fc100,
        total_flux=np.full(n100, 100.0),
        solidus_temperature=None,
        melt_fraction=np.full(n100, 0.011870),
        params=params_dlid,
        viscosity_solid=1.0e20,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert 0.1 < float(st_d08['d_lid']) < float(st_d12['d_lid'])
    with pytest.raises(AssertionError):
        assert 0.1 < 0.0 < float(st_d12['d_lid'])

    # 3. w_act_gate (width 0.0025 at w_active = 0.05) and w_sum_gate (width 2.5e-7 at W_sum = 1e-6).
    f_at_wact_0045 = 0.05 + 0.01 * np.arctanh(2.0 * 0.045 - 1.0)
    t_cold = np.linspace(2400.0, 300.0, n)
    st_act = compute_stagnant_lid_state(
        radii=r,
        temperature=t_cold,
        pressure=p,
        convective_flux=np.full(n, f_at_wact_0045 * 100.0),
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    d_raw_act = float((r[-1] - st_act['r_lid_base']) * st_act['w_active'])
    expected_w_act_gate = 0.5 * (1.0 + np.tanh((0.045 - 0.05) / 0.0025))
    assert float(st_act['d_lid']) / d_raw_act == pytest.approx(expected_w_act_gate, rel=1.0e-5)

    # Localize convective flux on node 1 with interior_flux_fraction = 0.01 so w_active > 0.05 and W_sum = 1.25e-6.
    params_sum = SolidRheologyParams(enabled=True)
    object.__setattr__(params_sum, 'interior_flux_fraction', 0.01)
    w_c0 = 0.5 * (1.0 + np.tanh(-1.0))
    target_w_sum = 1.25e-6
    f_node1 = 0.01 + 0.01 * np.arctanh(2.0 * (w_c0 + target_w_sum) - 1.0)
    fc_sum = np.zeros(n)
    fc_sum[1] = f_node1 * 100.0
    st_sum = compute_stagnant_lid_state(
        radii=r,
        temperature=t_cold,
        pressure=p,
        convective_flux=fc_sum,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_sum,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n, 1.0e-8),
        xp=np,
    )
    # Changing CONV_SUM_GATE_WIDTH from 2.5e-7 to 2.5e-6 shifts w_sum_gate at W_sum = 1.25e-6 from tanh(1) to tanh(0.1).
    expected_w_sum_gate = (
        0.5 * (1.0 + np.tanh((target_w_sum - 1.0e-6) / 2.5e-7)) - w_lim_0
    ) / (1.0 - w_lim_0)
    expected_w_act_low = 0.5 * (1.0 + np.tanh((float(st_sum['w_active']) - 0.05) / 0.0025))
    w_w_sum = np.tanh(target_w_sum / 1.0e-6)
    d_raw_sum = float((r[-1] - st_sum['r_lid_base']) * st_sum['w_active'])
    assert float(st_sum['d_lid']) == pytest.approx(2.3550870370491588e5, rel=1.0e-12, abs=0.0)
    assert float(st_sum['d_lid']) / d_raw_sum == pytest.approx(
        w_w_sum * expected_w_act_low * expected_w_sum_gate, rel=1.0e-5, abs=0.0
    )

    # 4. Smooth temperature gradient floor (SCALE_FLOOR_WIDTH = 0.005) and scale_mask cap (SCALE_MASK_REL_WIDTH = 0.005).
    t_grad_1k = 2000.0 - np.arange(n, dtype=float) * 1.0
    st_grad_1k = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad_1k,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(
            enabled=True, lid_base_mode='fixed', lid_base_temperature=1990.5
        ),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    scale_1k = 1.0 + 0.005 * np.log(2.0)
    expected_w_hot_top = 0.5 * (1.0 + np.tanh((t_grad_1k[-1] - 1990.5) / scale_1k))
    assert float(st_grad_1k['w_lid'][-1]) == pytest.approx(
        (1.0 - expected_w_hot_top) * float(st_grad_1k['w_active']), rel=1.0e-9
    )

    # At scale == scale_cap (dT_rh == |grad T| = 10.0 K), scale_mask = scale_cap * (1 - 0.005 * ln 2).
    t_grad_10k = 2000.0 - np.arange(n, dtype=float) * 10.0
    # Choose g_homologous so dT_rh = T_i^2 / (g_hom * t_sol_0) == 10.0 K.
    st_probe = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad_10k,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True, dt_sol_dp=0.0),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    ti_val = float(st_probe['T_i'])
    g_match = (ti_val**2) / (10.0 * 1400.0)
    params_cap_match = SolidRheologyParams(
        enabled=True,
        g_homologous=g_match,
        t_sol_0=1400.0,
        dt_sol_dp=0.0,
        lid_base_mode='fixed',
        lid_base_temperature=1850.0,
    )
    st_cap_match = compute_stagnant_lid_state(
        radii=r,
        temperature=t_grad_10k,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_cap_match,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    scale_cap_10k = 10.0 * (1.0 - 0.005 * np.log(2.0))
    expected_mask_top = (
        1.0 - 0.5 * (1.0 + np.tanh((t_grad_10k[-1] - 1850.0) / scale_cap_10k))
    ) * float(st_cap_match['w_active'])
    assert float(st_cap_match['w_lid'][-1]) == pytest.approx(expected_mask_top, rel=1.0e-7)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_flux_floor_regime_slope_and_subthreshold_center():
    """Verify F_TOT_FLOOR_FRAC zero-crossing protection, LID_REGIME_LABEL_WIDTH slope, and stagnant-lid interior state parity across interior_flux_fraction.

    The 13 unchanged points are pinned to merge-base c8f0b6cd literals at rel=1e-12.
    The 7 transition points (0 < w_weight < 1) are pinned to the independent evaluation
    of the Section 1.1 geotherm endpoint-blended formulas produced by
    tools/verification/evaluate_lid_transition_reference.py
    (command: python tools/verification/evaluate_lid_transition_reference.py)
    at rel=1e-12 (just above the measured kernel difference of <= 1e-15), with merge-base
    values noted beside each.
    """
    from pathlib import Path

    import aragog.rheology_lid as lid_mod
    from aragog.rheology_lid import compute_stagnant_lid_state

    assert getattr(lid_mod, 'F_TOT_FLOOR_FRAC', None) == pytest.approx(1.0e-3, rel=1.0e-15)
    assert getattr(lid_mod, 'LID_REGIME_LABEL_WIDTH', None) == pytest.approx(0.1, rel=1.0e-15)

    n = 20
    r = np.linspace(3.48e6, 6.37e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.linspace(2600.0, 300.0, n)
    phi = np.zeros(n)

    # 1. Sign-changing F_tot with zero crossing at interior node k0 and tiny dust F_conv:
    f_tot_sign = np.linspace(-100.0, 100.0, n)
    f_tot_sign[10] = 0.0
    f_loc_rms_10 = float(np.sqrt(np.mean(f_tot_sign[8:13] ** 2)))
    f_conv_spike = np.zeros(n)
    f_conv_spike[10] = 0.025 * (1.0e-3 * f_loc_rms_10)
    st_spike = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=f_conv_spike,
        total_flux=f_tot_sign,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    # With F_TOT_FLOOR_FRAC = 1e-3, f[10] = 0.025, so w_active = 0.5*(1 + tanh((0.025 - 0.05)/0.01)).
    expected_w_active_spike = 0.5 * (1.0 + np.tanh((0.025 - 0.05) / 0.01))
    assert float(st_spike['w_active']) == pytest.approx(expected_w_active_spike, rel=1.0e-9)
    assert bool(st_spike['has_lid']) is False

    # Also verify inter-nodal sign change (|F_tot[10]| = 0.01 < 1e-3 * F_loc_rms[10] = 0.01543 W/m^2)
    # activates F_TOT_FLOOR_FRAC: f[10] = 0.048 < 0.05 when floored, vs 0.0741 > 0.05 unfloored.
    f_tot_internodal = np.linspace(-100.0, 100.0, n)
    f_tot_internodal[9:11] = np.array([-0.01, 0.01])
    f_loc_rms_in = float(np.sqrt(np.mean(f_tot_internodal[8:13] ** 2)))
    f_conv_in = np.zeros(n)
    f_conv_in[10] = 0.048 * (1.0e-3 * f_loc_rms_in)
    st_internodal = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=f_conv_in,
        total_flux=f_tot_internodal,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    smax2 = getattr(lid_mod, '_softmax2')
    f_floor_in = float(smax2(1.0e-3 * f_loc_rms_in, 1.0e-12, 0.01 * 1.0e-12, np))
    f_safe_10 = float(smax2(abs(f_tot_internodal[10]), f_floor_in, 0.01 * f_floor_in, np))
    expected_w_active_in = 0.5 * (1.0 + np.tanh((f_conv_in[10] / f_safe_10 - 0.05) / 0.01))
    assert float(st_internodal['w_active']) == pytest.approx(expected_w_active_in, rel=1.0e-9)
    assert float(st_internodal['w_active']) < 0.45

    # 2. Mid-slope pin of lid_regime at tau_d = tau_y_lid * exp(+-0.1):
    fc_active = np.full(n, 80.0)
    fc_active[-4:] = 0.0
    ft_active = np.full(n, 100.0)
    st_base = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc_active,
        total_flux=ft_active,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(
            enabled=True, yield_stress_c=0.0, yield_stress_mu=0.0, yield_stress_max=1.0e6
        ),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    tau_d_val = float(st_base['tau_d_lid'])
    for sign, ratio_target in ((-1.0, np.exp(-0.1)), (1.0, np.exp(0.1))):
        ty_target = tau_d_val / ratio_target
        st_reg = compute_stagnant_lid_state(
            radii=r,
            temperature=t,
            pressure=p,
            convective_flux=fc_active,
            total_flux=ft_active,
            solidus_temperature=None,
            melt_fraction=phi,
            params=SolidRheologyParams(
                enabled=True,
                yield_stress_c=ty_target,
                yield_stress_mu=0.0,
                yield_stress_max=ty_target,
            ),
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        expected_wy = 0.5 * (1.0 + np.tanh((ratio_target - 1.0) / 0.1))
        assert float(st_reg['w_y']) == pytest.approx(expected_wy, rel=1.0e-9)
        assert float(st_reg['lid_regime']) == pytest.approx(
            float(st_reg['w_active']) * (1.0 + expected_wy), rel=1.0e-6
        )

    # 3. Sub-threshold and low-flux parity of eta_i, T_i, and mu_i across interior_flux_fraction
    # in {0.05, 0.10, 0.20, 0.50} with geotherm endpoint blending.
    n_l1 = 50
    r_l1 = np.linspace(3.48e6, 6.371e6, n_l1)
    t_l1 = np.linspace(2600.0, 1400.0, n_l1)
    p_l1 = np.linspace(135.0e9, 0.0, n_l1)
    phi_l1 = np.zeros(n_l1)
    visc_mix_l1 = np.full(n_l1, 1.0e15)

    # 13 points outside transition bands pinned to merge-base c8f0b6cd literals:
    merge_base_literals = {
        (0.05, 0.0): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.05, 1.0e-3): (1000000065827473.5, 1424.4909098970506, 1000000065827473.5),
        (0.05, 0.03): (1000000000000004.1, 1424.490907753382, 1000000000000004.1),
        (0.10, 0.0): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.20, 0.0): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.20, 6.00843e-5): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.20, 1.0e-4): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.20, 1.0e-3): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.50, 0.0): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.50, 6.00843e-5): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.50, 1.0e-4): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.50, 1.0e-3): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
        (0.50, 0.03): (6.046716893870374e30, 2575.5102040816328, 6.046716893870374e30),
    }

    # 7 transition points (0 < w_weight < 1) evaluated independently by
    # tools/verification/evaluate_lid_transition_reference.py from Section 1.1 formulas:
    transition_derivation_literals = {
        (0.05, 6.00843e-5): (
            2.3060133389415036e20,
            1856.5546317909584,
            2.3060133389415036e20,
        ),  # c8f0b6cd merge base: 8.9648269867330201e20, 1856.5546317909584, 8.9648269867330201e20
        (0.05, 1.0e-4): (
            3.2116823890682131e17,
            1626.7229387915370,
            3.2116823890682131e17,
        ),  # c8f0b6cd merge base: 5.4401391641927213e17, 1626.7229387915370, 5.4401391641927213e17
        (0.10, 6.00843e-5): (
            6.0370307137675392e30,
            2575.4671827559978,
            6.0370307137675392e30,
        ),  # c8f0b6cd merge base: 6.0389133587052112e30, 2575.4671827559978, 6.0389133587052112e30
        (0.10, 1.0e-4): (
            6.0305401114385923e30,
            2575.4383157027542,
            6.0305401114385923e30,
        ),  # c8f0b6cd merge base: 6.0336828502715910e30, 2575.4383157027542, 6.0336828502715910e30
        (0.10, 1.0e-3): (
            5.8718073494424216e30,
            2574.7224307551792,
            5.8718073494424216e30,
        ),  # c8f0b6cd merge base: 5.9054014562120670e30, 2574.7224307551792, 5.9054014562120670e30
        (0.10, 0.03): (
            1.0036004104618392e18,
            1667.4062468015593,
            1.0036004104618392e18,
        ),  # c8f0b6cd merge base: 1.9932572310324982e18, 1667.4062468015593, 1.9932572310324982e18
        (0.20, 0.03): (
            6.0467159147630670e30,
            2575.5101979531364,
            6.0467159147630670e30,
        ),  # c8f0b6cd merge base: 6.0467165336094990e30, 2575.5101979531364, 6.0467165336094990e30
    }
    for (f0_cfg, f_test), (exp_eta_i, exp_t_i, exp_mu_i) in {
        **merge_base_literals,
        **transition_derivation_literals,
    }.items():
        st_sub = compute_stagnant_lid_state(
            radii=r_l1,
            temperature=t_l1,
            pressure=p_l1,
            convective_flux=np.full(n_l1, f_test),
            total_flux=np.ones(n_l1),
            solidus_temperature=None,
            melt_fraction=phi_l1,
            params=SolidRheologyParams(enabled=True, interior_flux_fraction=f0_cfg),
            viscosity_solid=1.0e21,
            viscosity_mixture=visc_mix_l1,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        assert float(st_sub['eta_i']) == pytest.approx(exp_eta_i, rel=1.0e-12)
        assert float(st_sub['T_i']) == pytest.approx(exp_t_i, rel=1.0e-12)
        assert float(st_sub['mu_i']) == pytest.approx(exp_mu_i, rel=1.0e-12)

    # 4. A3-2 check on checkpoint_profile_175kyr.npz across all 4 interior_flux_fraction values:
    # outside transition bands, outputs match hard reference to 1e-12.
    ckpt_path = Path(__file__).parent / 'reference' / 'checkpoint_profile_175kyr.npz'
    ckpt = np.load(ckpt_path)
    for f0_ckpt in (0.05, 0.10, 0.20, 0.50):
        st_ckpt = compute_stagnant_lid_state(
            radii=ckpt['r_basic'],
            temperature=ckpt['T_basic'],
            pressure=ckpt['P_basic'],
            convective_flux=ckpt['F_conv'],
            total_flux=ckpt['F_tot'],
            solidus_temperature=None,
            melt_fraction=ckpt['phi_basic'],
            params=SolidRheologyParams(enabled=True, interior_flux_fraction=f0_ckpt),
            unyielded_velocity=ckpt['v_unyielded'],
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        assert float(st_ckpt['eta_i']) == pytest.approx(2.760705302477158e22, rel=1.0e-12)
        assert float(st_ckpt['T_i']) == pytest.approx(1423.0148338372128, rel=1.0e-12)
        assert float(st_ckpt['mu_i']) == pytest.approx(2.760705302477158e22, rel=1.0e-12)
        assert float(st_ckpt['d_lid']) == pytest.approx(42134.621325017884, rel=1.0e-12)
        assert float(st_ckpt['tau_d']) == pytest.approx(1299363.3191059122, rel=1.0e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_boundary_weights_smooth_max_and_no_fallback():
    """Verify _smooth_max and _boundary_weights continuity across two-node peak swap and uniform invariance."""
    import inspect

    import jax

    import aragog.rheology_lid as lid_mod
    from aragog.rheology_lid import _boundary_weights

    _, jnp = _import_jax_x64()

    assert getattr(lid_mod, 'SMOOTH_MAX_REL_WIDTH', None) == pytest.approx(0.02, rel=1.0e-15)
    smooth_max = getattr(lid_mod, '_smooth_max')

    # 1. Uniform positive array returns c identically for any N.
    for n_len in (5, 20, 100):
        u_arr = np.full(n_len, 3.5e-7)
        assert float(smooth_max(u_arr, np)) == pytest.approx(3.5e-7, rel=1.0e-15)
        w_t, w_s, w_b = _boundary_weights(u_arr, np)
        assert float(w_b) == pytest.approx(float(np.tanh(3.5e-7 / 1.0e-6)), rel=1.0e-12)
        assert float(np.sum(w_t)) == pytest.approx(1.0, rel=1.0e-12)
        assert float(w_s) > 0.0

    # 2. Two-node peak crossing at delta = 0 has continuous derivative and value < hard max at tie.
    def _eval_w_base(delta_val):
        arr = jnp.array([0.0, 5.0e-7 - delta_val, 5.0e-7 + delta_val, 0.0])
        return _boundary_weights(arr, jnp)[2]

    w_tie = float(_eval_w_base(jnp.asarray(0.0)))
    assert w_tie == pytest.approx(float(np.tanh(5.0e-7 / 1.0e-6)), rel=1.0e-12)
    w_near = float(_eval_w_base(jnp.asarray(5.0e-9)))
    v_arr_near = np.array([0.0, 5.0e-7 - 5.0e-9, 5.0e-7 + 5.0e-9, 0.0])
    v_rms_near = float(np.sqrt(np.mean(v_arr_near**2)))
    sigma_near = 0.02 * v_rms_near
    u_near = (v_arr_near - np.max(v_arr_near)) / sigma_near
    smax_expected = float(
        np.max(v_arr_near)
        + np.sum(np.exp(u_near) * (v_arr_near - np.max(v_arr_near))) / np.sum(np.exp(u_near))
    )
    assert w_near == pytest.approx(float(np.tanh(smax_expected / 1.0e-6)), rel=1.0e-12)

    grad_w = jax.grad(_eval_w_base)
    g_left = float(grad_w(jnp.asarray(-1.0e-17)))
    g_zero = float(grad_w(jnp.asarray(0.0)))
    g_right = float(grad_w(jnp.asarray(1.0e-17)))
    assert np.isfinite(g_zero) and abs(g_zero) < 1.0e-6
    assert abs(g_right - g_left) < 1.0e-2

    # 3. Verify compute_stagnant_lid_state calls _boundary_weights(w_Ti, xp, ...) without a uniform fallback.
    src_text = inspect.getsource(lid_mod.compute_stagnant_lid_state)
    assert '_boundary_weights(w_Ti, xp' in src_text
    assert 'xp.ones_like(r) / r.shape[0]' not in src_text


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_invariants(monkeypatch):
    """Verify runtime assertions in compute_stagnant_lid_state for weights in [0, 1] and finite eta_i."""
    import aragog.rheology_lid as lid_mod
    from aragog.rheology_lid import compute_stagnant_lid_state

    n = 20
    r = np.linspace(3.48e6, 6.37e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.linspace(2600.0, 300.0, n)
    fc = np.full(n, 80.0)
    fc[-4:] = 0.0
    ft = np.full(n, 100.0)
    phi = np.zeros(n)
    kw = dict(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )

    st = compute_stagnant_lid_state(**kw)
    assert np.all((st['w_lid'] >= 0.0) & (st['w_lid'] <= 1.0))
    assert 0.0 <= float(st['w_active']) <= 1.0
    assert np.isfinite(float(st['eta_i']))

    # 1. Forcing w_lid_present outside [0, 1] while eta_i remains finite triggers the w_lid_present range assert.
    monkeypatch.setattr(lid_mod, 'NO_LID_THICKNESS_WIDTH', np.nan)
    with pytest.raises(AssertionError):
        compute_stagnant_lid_state(**kw)
    monkeypatch.setattr(lid_mod, 'NO_LID_THICKNESS_WIDTH', 0.25)

    # 2. Forcing non-finite eta_i triggers assert xp.all(xp.isfinite(eta_i)).
    monkeypatch.setattr(lid_mod, 'compute_arrhenius_viscosity', lambda *a, **k: np.inf)
    with pytest.raises(AssertionError):
        compute_stagnant_lid_state(**kw)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_smooth_gates_and_scales_conv_sum_gate():
    """Verify w_sum_gate across the W_sum = 1e-6 +/- 2.5e-7 gate band on gated lid outputs."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    n = 20
    r = np.linspace(3.48e6, 6.37e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t_cold = np.linspace(2400.0, 300.0, n)
    ft = np.full(n, 100.0)
    phi = np.zeros(n)

    # Set interior_flux_fraction = 0.01 so w_conv_0 > 0.05 (w_act_gate = 1) while W_sum sits in 1e-6 +/- 2.5e-7.
    params_sum = SolidRheologyParams(enabled=True)
    object.__setattr__(params_sum, 'interior_flux_fraction', 0.01)
    w_c0 = 0.5 * (1.0 + np.tanh(-1.0))

    fc_lo = np.zeros(n)
    fc_lo[1] = (0.01 + 0.01 * np.arctanh(2.0 * (w_c0 + 0.75e-6) - 1.0)) * 100.0
    st_lo = compute_stagnant_lid_state(
        radii=r,
        temperature=t_cold,
        pressure=p,
        convective_flux=fc_lo,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_sum,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n, 1.0e-8),
        xp=np,
    )

    fc_hi = np.zeros(n)
    fc_hi[1] = (0.01 + 0.01 * np.arctanh(2.0 * (w_c0 + 1.25e-6) - 1.0)) * 100.0
    st_hi = compute_stagnant_lid_state(
        radii=r,
        temperature=t_cold,
        pressure=p,
        convective_flux=fc_hi,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params_sum,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        unyielded_velocity=np.full(n, 1.0e-8),
        xp=np,
    )

    assert float(st_lo['d_lid']) == pytest.approx(2.3806335181320017e4, rel=1.0e-12, abs=0.0)
    assert float(st_lo['tau_d']) == pytest.approx(2.7067907640183691e7, rel=1.0e-12, abs=0.0)
    assert float(st_hi['d_lid']) == pytest.approx(2.3550870370491588e5, rel=1.0e-12, abs=0.0)
    assert float(st_hi['tau_d']) == pytest.approx(2.6777443028467780e8, rel=1.0e-12, abs=0.0)
    assert float(st_lo['d_lid']) < float(st_hi['d_lid'])
    assert float(st_lo['tau_d']) < float(st_hi['tau_d'])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_flux_floor_zero_crossing_width():
    """Verify w_zero_cross activates on local F_tot zero crossings and leaves positive interior profiles untouched."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    n = 20
    r = np.linspace(3.48e6, 6.37e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.linspace(2600.0, 300.0, n)
    phi = np.zeros(n)

    # 1. Sign-changing F_tot (d_zero = 0): 5-node local interior RMS floor F_tot_floor = 1.0e-3 * F_loc_rms.
    ft_cross = np.linspace(-100.0, 100.0, n)
    ft_cross[10] = 0.0
    f_pad20 = np.concatenate([ft_cross[1:2], ft_cross[1:-1], ft_cross[-2:-1]])
    f_loc_rms_10 = float(
        np.sqrt(
            0.2
            * (
                f_pad20[8] ** 2
                + f_pad20[9] ** 2
                + f_pad20[10] ** 2
                + f_pad20[11] ** 2
                + f_pad20[12] ** 2
            )
        )
    )
    fc_cross = np.zeros(n)
    fc_cross[10] = 0.045 * (1.0e-3 * f_loc_rms_10)
    st_cross = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc_cross,
        total_flux=ft_cross,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(st_cross['w_active']) == pytest.approx(0.268941421369995, rel=1.0e-12, abs=0.0)
    assert float(st_cross['d_lid']) == pytest.approx(3.617717821797071e5, rel=1.0e-12, abs=0.0)
    assert float(st_cross['tau_d']) == pytest.approx(2.0853767364515128e9, rel=1.0e-12, abs=0.0)

    # 2. Strictly positive F_tot above the 0.0014 * F_loc_rms floor (F_tot[10] = 20.0 on 100.0):
    # F_tot_safe[10] = 20.0 bitwise and f[10] = 0.08.
    ft_near = np.full(n, 100.0)
    ft_near[10] = 20.0
    fc_near = np.zeros(n)
    fc_near[10] = 0.08 * ft_near[10]
    st_near = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc_near,
        total_flux=ft_near,
        solidus_temperature=None,
        melt_fraction=phi,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(st_near['w_active']) == pytest.approx(0.9975273768433652, rel=1.0e-12, abs=0.0)
    assert float(st_near['d_lid']) == pytest.approx(1.3418433466118898e6, rel=1.0e-12, abs=0.0)
    assert float(st_near['tau_d']) == pytest.approx(7.734845659125156e9, rel=1.0e-12, abs=0.0)
    assert float(st_near['w_active']) > float(st_cross['w_active'])
    assert float(st_near['d_lid']) > float(st_cross['d_lid'])
    assert float(st_near['tau_d']) > float(st_cross['tau_d'])

    # 3. Zero total flux at CMB (F_tot[0] = 0) or surface (F_tot[-1] = 0) leaves interior T_i and d_lid untouched.
    n40 = 40
    r40 = np.linspace(3.48e6, 6.371e6, n40)
    p40 = np.linspace(135.0e9, 1.0e5, n40)
    t40 = np.linspace(3000.0, 1500.0, n40)
    phi40 = np.zeros(n40)
    for b_idx in (0, -1):
        ft_b0 = np.full(n40, 100.0)
        ft_b0[1:10] = 3000.0
        ft_b0[b_idx] = 0.0
        fc_b0 = 0.99 * ft_b0
        fc_b0[0] = 0.0
        fc_b0[-5:] = 0.0
        st_b0 = compute_stagnant_lid_state(
            radii=r40,
            temperature=t40,
            pressure=p40,
            convective_flux=fc_b0,
            total_flux=ft_b0,
            solidus_temperature=None,
            melt_fraction=phi40,
            params=SolidRheologyParams(enabled=True),
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        assert float(st_b0['T_i']) == pytest.approx(1692.3094384588371, rel=1.0e-12, abs=0.0)
        assert float(st_b0['d_lid']) == pytest.approx(129026.7742045084, rel=1.0e-12, abs=0.0)
        assert float(st_b0['tau_d']) == pytest.approx(2404798.925410915, rel=1.0e-12, abs=0.0)

    # 4. Positive interior total flux with a 1e4 step contrast (F_tot[:32] = 1, F_tot[32:] = 1e4) does not floor deep nodes.
    ft_steep = np.ones(n40)
    ft_steep[32:] = 1.0e4
    fc_steep = np.zeros(n40)
    fc_steep[:20] = 1.0
    st_steep = compute_stagnant_lid_state(
        radii=r40,
        temperature=t40,
        pressure=p40,
        convective_flux=fc_steep,
        total_flux=ft_steep,
        solidus_temperature=None,
        melt_fraction=phi40,
        params=SolidRheologyParams(enabled=True),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(st_steep['w_active']) == pytest.approx(1.0, rel=1.0e-12, abs=0.0)
    assert float(st_steep['d_lid']) == pytest.approx(1221182.1912324484, rel=1.0e-12, abs=0.0)
    assert float(st_steep['tau_d']) == pytest.approx(25121340.963971354, rel=1.0e-12, abs=0.0)

    # 5. Continuity when one neighbour moves through zero (F[25] in [+1e-3, ..., -1e-3]) and dips of 1, 2, 3 nodes.
    n30 = 30
    r30 = np.linspace(3.48e6, 6.371e6, n30)
    p30 = np.linspace(135.0e9, 1.0e5, n30)
    t30 = np.linspace(3500.0, 1600.0, n30)
    phi30 = np.zeros(n30)
    params30 = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    for eps in (1.0e-3, 1.0e-6, 1.0e-12, 0.0, -1.0e-12, -1.0e-6, -1.0e-3):
        ft_nb = np.full(n30, 100.0)
        ft_nb[25] = eps
        ft_nb[26] = 0.5
        fc_nb = 0.8 * np.abs(ft_nb)
        fc_nb[25] = 0.0
        fc_nb[27:] = 0.0
        st_nb = compute_stagnant_lid_state(
            radii=r30,
            temperature=t30,
            pressure=p30,
            convective_flux=fc_nb,
            total_flux=ft_nb,
            solidus_temperature=None,
            melt_fraction=phi30,
            params=params30,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            unyielded_velocity=np.full(n30, 1.0e-9),
            xp=np,
        )
        assert float(st_nb['T_i']) == pytest.approx(1796.5576729592096, rel=1.0e-11, abs=0.0)
        assert float(st_nb['eta_i']) == pytest.approx(9.68874084044789e21, rel=1.0e-10, abs=0.0)
        assert float(st_nb['d_lid']) == pytest.approx(93552.92084194534, rel=1.0e-11, abs=0.0)
        assert float(st_nb['tau_d']) == pytest.approx(1101349.6769275344, rel=1.0e-11, abs=0.0)

    for sl in (slice(20, 21), slice(20, 22), slice(19, 22)):
        ft_dip = np.full(n30, 100.0)
        ft_dip[sl] = 1.0e-6
        fc_dip = np.zeros(n30)
        fc_dip[sl] = 0.5 * ft_dip[sl]
        st_dip = compute_stagnant_lid_state(
            radii=r30,
            temperature=t30,
            pressure=p30,
            convective_flux=fc_dip,
            total_flux=ft_dip,
            solidus_temperature=None,
            melt_fraction=phi30,
            params=params30,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            unyielded_velocity=np.full(n30, 1.0e-9),
            xp=np,
        )
        assert bool(st_dip['has_lid']) is False
        assert float(st_dip['w_active']) < 1.0e-4
        assert float(st_dip['d_lid']) == 0.0
        assert float(st_dip['tau_d']) == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_w_T_negative_dt_lid_zero():
    """Verify w_T and all gated lid outputs are identically 0 for dT_lid <= 0 in rheological and fixed modes."""
    from aragog.rheology_lid import compute_stagnant_lid_state

    n = 30
    r = np.linspace(3.48e6, 6.371e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    ft = np.full(n, 0.05)
    fc = np.full(n, 0.04)
    fc[-1] = 0.0
    phi = np.zeros(n)
    gated_keys = (
        'd_lid',
        'Ra_eff',
        'Ra_rh',
        'v_m',
        'tau_d',
        'tau_d_lid',
        'tau_buoy',
        'tau_d_over_tau_buoy',
        'w_y',
        'lid_regime',
    )
    for dt_lid in (-100.0, -1.0e-2, -3.0e-3, -1.0e-3, -1.0e-4):
        t_col = np.full(n, 1600.0)
        t_col[-1] = 1600.0 - dt_lid
        for params_mode in (
            SolidRheologyParams(
                enabled=True,
                stress_closure_mode='lid',
                lid_base_mode='fixed',
                lid_base_temperature=1600.0,
            ),
            SolidRheologyParams(
                enabled=True,
                stress_closure_mode='lid',
                lid_base_mode='rheological',
            ),
        ):
            t_run = t_col.copy()
            st = compute_stagnant_lid_state(
                radii=r,
                temperature=t_run,
                pressure=p,
                convective_flux=fc,
                total_flux=ft,
                solidus_temperature=None,
                melt_fraction=phi,
                params=params_mode,
                viscosity_solid=1.0e21,
                phi_rheo=0.4,
                phi_width=0.15,
                unyielded_velocity=np.full(n, 1.0e-9),
                xp=np,
            )
            actual_dt = float(st['T_i']) - float(t_run[-1])
            np.testing.assert_allclose(actual_dt, dt_lid, rtol=1.0e-10, atol=1.0e-12)
            assert bool(st['has_lid']) is False
            for k in gated_keys:
                assert float(st[k]) == 0.0, (
                    f'Expected {k} == 0.0 at dT_lid={dt_lid} in mode={params_mode.lid_base_mode}, got {float(st[k])}'
                )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_w_T_and_d_lid_grad_continuity_across_zero():
    """Verify jax.grad continuity of w_T and d_lid across dT_lid = 0 via ratio test."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)

    n = 20
    r = jnp.linspace(3.48e6, 6.371e6, n)
    p = jnp.linspace(135.0e9, 1.0e5, n)
    fc = jnp.full(n, 0.04)
    fc = fc.at[-1].set(0.0)
    ft = jnp.full(n, 0.05)
    phi = jnp.zeros(n)
    v_un = jnp.full(n, 1.0e-9)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=2000.0,
    )

    def _lid_state(dt_lid):
        t_run = jnp.full(n, 1600.0)
        t_run = t_run.at[-1].set(1600.0 - dt_lid)
        return compute_stagnant_lid_state(
            radii=r,
            temperature=t_run,
            pressure=p,
            convective_flux=fc,
            total_flux=ft,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params,
            unyielded_velocity=v_un,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=jnp,
        )

    def eval_w_lid_present(dt_lid):
        st = _lid_state(dt_lid)
        d_raw = (r[-1] - st['r_lid_base']) * st['w_active']
        return st['d_lid'] / d_raw

    def eval_d_lid(dt_lid):
        st = _lid_state(dt_lid)
        return st['d_lid']

    grad_w_lid_present = jax.grad(eval_w_lid_present)
    grad_d_lid = jax.grad(eval_d_lid)

    steps = (1.0e-6, 1.0e-7, 1.0e-8)
    diffs_w_lid_present = []
    diffs_d_lid = []
    for step in steps:
        h = jnp.asarray(step)
        diffs_w_lid_present.append(
            abs(float(grad_w_lid_present(h)) - float(grad_w_lid_present(-h)))
        )
        diffs_d_lid.append(abs(float(grad_d_lid(h)) - float(grad_d_lid(-h))))

    # Under smooth C1 onset, derivative jump scales as O(h). Ratio falls by ~10x per decade.
    assert diffs_w_lid_present[1] < 0.2 * diffs_w_lid_present[0]
    assert diffs_w_lid_present[2] < 0.2 * diffs_w_lid_present[1]
    assert diffs_d_lid[1] < 0.2 * diffs_d_lid[0]
    assert diffs_d_lid[2] < 0.2 * diffs_d_lid[1]


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_w_T_positive_part_subthreshold_values():
    """Verify w_T values in 0 < dT_lid <= 8 NO_LID_DT_WIDTH against hand-computed literals."""
    import aragog.rheology_lid as lid_mod

    n = 20
    r = np.linspace(3.48e6, 6.371e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    fc = np.full(n, 0.04)
    fc[-1] = 0.0
    ft = np.full(n, 0.05)
    phi = np.zeros(n)
    v_un = np.full(n, 1.0e-9)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=2000.0,
    )
    w_width = lid_mod.NO_LID_DT_WIDTH

    literals = (
        (0.1, 2.227177990575e-05, 3.3888271425e00),
        (0.5, 5.152758399282e-04, 7.8403287012e01),
        (1.0, 2.125477577685e-03, 3.2340819353e02),
        (2.0, 1.765655210878e-02, 2.6865837972e03),
        (3.0, 1.189074449219e-01, 1.8092706488e04),
        (4.0, 4.998322686756e-01, 7.6053425723e04),
        (5.0, 8.807570898524e-01, 1.3401414457e05),
        (6.0, 9.820077563366e-01, 1.4942023282e05),
        (7.0, 9.975265473707e-01, 1.5178153939e05),
        (8.0, 9.996645373721e-01, 1.5210685145e05),
    )

    for factor, w_T_exp, d_lid_exp in literals:
        dt_lid = factor * w_width
        t_run = np.full(n, 1600.0)
        t_run[-1] = 1600.0 - dt_lid
        st = compute_stagnant_lid_state(
            radii=r,
            temperature=t_run,
            pressure=p,
            convective_flux=fc,
            total_flux=ft,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params,
            unyielded_velocity=v_un,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        np.testing.assert_allclose(float(st['d_lid']), d_lid_exp, rtol=1.0e-7)
        d_raw = (r[-1] - float(st['r_lid_base'])) * float(st['w_active'])
        w_T_actual = float(st['d_lid']) / d_raw
        np.testing.assert_allclose(w_T_actual, w_T_exp, rtol=1.0e-7)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_local_rms_floor_dip_widths_4_5_6():
    """Verify 5-node moving RMS floor response to positive dips of width 4, 5, and 6 nodes."""
    n = 25
    r = np.linspace(3.48e6, 6.371e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.linspace(2600.0, 1600.0, n)
    phi = np.zeros(n)
    v_un = np.full(n, 1.0e-9)
    params = SolidRheologyParams(enabled=True)

    expected_w_active = {
        4: 1.110329e-04,
        5: 1.000000e00,
        6: 1.000000e00,
    }

    for width, exp_w_act in expected_w_active.items():
        ft = np.full(n, 100.0)
        fc = np.zeros(n)
        start = 12 - width // 2
        ft[start : start + width] = 1.0e-3
        fc[start : start + width] = 2.0e-4
        st = compute_stagnant_lid_state(
            radii=r,
            temperature=t,
            pressure=p,
            convective_flux=fc,
            total_flux=ft,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params,
            unyielded_velocity=v_un,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        np.testing.assert_allclose(float(st['w_active']), exp_w_act, rtol=1.0e-5)
        assert bool(st['has_lid']) is (width >= 5)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_local_rms_floor_boundary_window_rule():
    """Verify local RMS floor excludes boundary nodes for N = 3, 4, 5, 20."""
    params = SolidRheologyParams(enabled=True)

    for n in (3, 4, 5, 20):
        r = np.linspace(3.48e6, 6.37e6, n)
        p = np.linspace(135.0e9, 1.0e5, n)
        t = np.linspace(2600.0, 300.0, n)
        phi = np.zeros(n)
        v_un = np.full(n, 1.0e-9)

        ft_base = np.full(n, 1.0)
        fc = np.zeros(n)
        fc[1] = 0.1

        st_base = compute_stagnant_lid_state(
            radii=r,
            temperature=t,
            pressure=p,
            convective_flux=fc,
            total_flux=ft_base,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params,
            unyielded_velocity=v_un,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )

        for ft_cmb in (0.0, 10000.0):
            for ft_surf in (0.0, 10000.0):
                ft_mod = ft_base.copy()
                ft_mod[0] = ft_cmb
                ft_mod[-1] = ft_surf
                st_mod = compute_stagnant_lid_state(
                    radii=r,
                    temperature=t,
                    pressure=p,
                    convective_flux=fc,
                    total_flux=ft_mod,
                    solidus_temperature=None,
                    melt_fraction=phi,
                    params=params,
                    unyielded_velocity=v_un,
                    viscosity_solid=1.0e21,
                    phi_rheo=0.4,
                    phi_width=0.15,
                    xp=np,
                )
                assert float(st_mod['T_i']) == float(st_base['T_i'])
                assert float(st_mod['d_lid']) == float(st_base['d_lid'])
                assert float(st_mod['w_active']) == float(st_base['w_active'])


def test_stagnant_lid_boundary_window_lead_probe_n3():
    """Verify N = 3 boundary-window rule preserves convective activity and lid."""
    r = np.linspace(3.48e6, 6.37e6, 3)
    p = np.linspace(135.0e9, 1.0e5, 3)
    t = np.linspace(2600.0, 300.0, 3)
    phi = np.zeros(3)
    v_un = np.full(3, 1.0e-9)
    params = SolidRheologyParams(enabled=True)

    ft = np.array([1000.0, 1.0e-3, 1000.0])
    fc = np.array([0.0, 0.5e-3, 0.0])

    st = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        unyielded_velocity=v_un,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert bool(st['has_lid']) is True
    assert float(st['w_active']) > 0.5
    assert float(st['d_lid']) > 0.0


def test_compute_f_tot_safe_boundary_window_closed_form():
    """Verify _compute_f_tot_safe closed-form boundary window values for N = 3 and N = 20."""
    from aragog.rheology_lid import _compute_f_tot_safe

    # N = 3: boundary window isolates node 1 from outer 1000 W/m^2
    ft3 = np.array([1000.0, 1.0e-3, 1000.0])
    safe3 = _compute_f_tot_safe(ft3)
    assert float(safe3[1]) == pytest.approx(1.0e-3, rel=1.0e-5)

    # N = 20: boundary window isolates node 1 from CMB 10000 W/m^2
    ft20 = np.ones(20)
    ft20[0] = 1.0e4
    safe20 = _compute_f_tot_safe(ft20)
    assert float(safe20[1]) == pytest.approx(1.0, rel=1.0e-5)


def test_stagnant_lid_local_rms_floor_p5_profile_unfloored_at_50_wm2():
    """Verify p5 profile with 50 W/m2 at nodes 10-11 keeps convective fraction unfloored at main value."""
    from aragog.rheology_lid import _compute_f_tot_safe

    n = 40
    ft = np.full(n, 100.0)
    ft[:10] = 1.0e4
    ft[10:12] = 50.0
    fc = 0.99 * np.maximum(ft, 0.0)
    fc[-5:] = 0.0

    f_tot_safe = _compute_f_tot_safe(ft)
    f_10 = float(np.clip(fc[10] / f_tot_safe[10], 0.0, 1.0))
    assert f_10 == pytest.approx(0.99, rel=1.0e-3)


def test_stagnant_lid_local_rms_floor_p5_profile_floored_at_1e_minus_3_wm2():
    """Verify p5 profile with 1e-3 W/m2 at nodes 10-11 is floored far below main's 0.99."""
    from aragog.rheology_lid import _compute_f_tot_safe

    n = 40
    ft = np.full(n, 100.0)
    ft[:10] = 1.0e4
    ft[10:12] = 1.0e-3
    fc = 0.99 * np.maximum(ft, 0.0)
    fc[-5:] = 0.0

    f_tot_safe = _compute_f_tot_safe(ft)
    f_10 = float(np.clip(fc[10] / f_tot_safe[10], 0.0, 1.0))
    assert f_10 < 0.01
    assert f_10 == pytest.approx(0.99 * 1.0e-3 / (1.0e-3 * 6324.7134), rel=1.0e-3)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_ti_jax_gradient_zero_at_saturated_flux_fraction():
    """Verify jax.grad of T_i and eta_i equals finite differences (0) at saturated f == 1 nodes.

    Raising F_conv at a saturated node leaves f at the cap and lowering it leaves
    f_int_max at 1 through the other tied nodes, so finite differences are zero; a [0, 1]
    clip passes half the gradient at the cap and the max tie then gives T_i a large gradient.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from scipy.special import erf

    jax.config.update('jax_enable_x64', True)
    params = SolidRheologyParams(enabled=True)
    from aragog.rheology_lid import _compute_f_tot_safe

    # N = 40 erf profile where F_conv equals F_tot to round-off over deep interior nodes.
    n40 = 40
    dlt = 1.5e5
    r40 = np.linspace(3.48e6, 6.371e6, n40)
    z40 = r40[-1] - r40
    t40 = 300.0 + 1300.0 * erf(z40 / dlt)
    p40 = 4000.0 * 9.81 * z40
    ft40 = 0.08 * np.ones(n40) + 0.02 * (r40 - r40[0]) / (r40[-1] - r40[0])
    fc40 = ft40 * (1.0 - np.exp(-((z40 / dlt) ** 2)))
    phi40 = np.zeros(n40)

    def eval_40(fc_arr, key, xp):
        st = compute_stagnant_lid_state(
            radii=r40,
            temperature=t40,
            pressure=p40,
            convective_flux=fc_arr,
            total_flux=ft40,
            solidus_temperature=None,
            melt_fraction=phi40,
            params=params,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=xp,
        )
        return st[key]

    node = 13
    f_raw_val = float(fc40[node] / _compute_f_tot_safe(ft40, np)[node])
    assert f_raw_val >= 1.0
    t_i_40_0 = float(eval_40(fc40, 'T_i', np))
    eta_i_40_0 = float(eval_40(fc40, 'eta_i', np))
    for h_rel in (1.0e-3, 1.0e-5, 1.0e-7):
        step = h_rel * fc40[node]
        fc_plus = fc40.copy()
        fc_plus[node] += step
        fc_minus = fc40.copy()
        fc_minus[node] -= step
        fd_right_ti = (float(eval_40(fc_plus, 'T_i', np)) - t_i_40_0) / step
        fd_left_ti = (t_i_40_0 - float(eval_40(fc_minus, 'T_i', np))) / step
        assert fd_right_ti == 0.0
        assert fd_left_ti == 0.0
        fd_right_eta = (float(eval_40(fc_plus, 'eta_i', np)) - eta_i_40_0) / step
        fd_left_eta = (eta_i_40_0 - float(eval_40(fc_minus, 'eta_i', np))) / step
        assert fd_right_eta == 0.0
        assert fd_left_eta == 0.0

    ad_ti_40 = np.asarray(jax.grad(lambda x: eval_40(x, 'T_i', jnp))(jnp.asarray(fc40)))
    ad_eta_40 = np.asarray(jax.grad(lambda x: eval_40(x, 'eta_i', jnp))(jnp.asarray(fc40)))
    assert np.all(np.isfinite(ad_ti_40))
    assert np.all(np.isfinite(ad_eta_40))
    assert ad_ti_40[node] == 0.0
    assert ad_eta_40[node] == 0.0

    # N = 10 profile with interior nodes 1..6 saturated at f == 1 and node 7 at f == 0.9.
    n = 10
    r = np.linspace(3.48e6, 6.371e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.linspace(2600.0, 300.0, n)
    ft = np.full(n, 0.1)
    fc = ft.copy()
    fc[7:] = (0.09, 0.05, 0.0)
    phi = np.zeros(n)

    def t_i(fc_arr, xp):
        st = compute_stagnant_lid_state(
            radii=r,
            temperature=t,
            pressure=p,
            convective_flux=fc_arr,
            total_flux=ft,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=xp,
        )
        return st['T_i']

    t_i_0 = float(t_i(fc, np))
    for k in (2, 3, 4):
        for h in (1.0e-3, -1.0e-3):
            fc_h = fc.copy()
            fc_h[k] += h
            assert abs(float(t_i(fc_h, np)) - t_i_0) < 1.0e-9
    grad = np.asarray(jax.grad(lambda x: t_i(x, jnp))(jnp.asarray(fc)))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad[1:7])) == 0.0
    assert abs(grad[7]) > 1.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_geotherm_interior_state_and_monotone_subthreshold_band():
    """Verify P_T_i stays on the geotherm and eta_solid_arrhenius(s), eta_i(s), and theta(s) are monotone across s in [0, 1e-3]."""
    from pathlib import Path

    ckpt_path = Path(__file__).parent / 'reference' / 'checkpoint_profile_175kyr.npz'
    d = np.load(ckpt_path)
    r = np.asarray(d['r_basic'], dtype=float)
    t = np.asarray(d['T_basic'], dtype=float)
    p = np.asarray(d['P_basic'], dtype=float)
    phi = np.asarray(d['phi_basic'], dtype=float)
    fc = np.asarray(d['F_conv'], dtype=float)
    ft = np.asarray(d['F_tot'], dtype=float)

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        interior_flux_fraction=0.05,
        g_homologous=25.0,
        t_sol_0=1400.0,
    )
    s_grid = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 6.00843e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3]
    log_eta_arr_vals = []
    log_eta_i_vals = []
    theta_vals = []

    for s_val in s_grid:
        st = compute_stagnant_lid_state(
            radii=r,
            temperature=t,
            pressure=p,
            convective_flux=fc * s_val,
            total_flux=ft,
            solidus_temperature=None,
            melt_fraction=phi,
            params=params,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        t_i = float(st['T_i'])
        p_t_i = float(st['P_T_i'])
        r_i = float(st['r_i'])
        p_geo = float(np.interp(t_i, t[::-1], p[::-1]))
        r_geo = float(np.interp(t_i, t[::-1], r[::-1]))
        # Geotherm pressure and radius agree to within the endpoint discretization offset.
        assert abs(p_t_i - p_geo) < 1.0e6
        assert abs(r_i - r_geo) < 50.0
        log_eta_arr_vals.append(np.log10(float(st['eta_solid_arrhenius'])))
        log_eta_i_vals.append(np.log10(float(st['eta_i'])))
        theta_vals.append(float(st['theta']))

    # Monotone across the sub-threshold band (no 0.82-decade chord dip at s = 1e-4).
    diffs_arr = np.diff(log_eta_arr_vals)
    diffs_eta_i = np.diff(log_eta_i_vals)
    diffs_theta = np.diff(theta_vals)
    assert np.all(diffs_arr >= -1.0e-14) or np.all(diffs_arr <= 1.0e-14)
    assert np.all(diffs_eta_i >= -1.0e-14) or np.all(diffs_eta_i <= 1.0e-14)
    assert np.all(diffs_theta <= 1.0e-14)

    # Affine geotherm P(T) = a * T + b: P_T_i matches the linear chord sum to machine precision.
    n_aff = 40
    r_aff = np.linspace(3.48e6, 6.371e6, n_aff)
    t_aff = np.linspace(3000.0, 1200.0, n_aff)
    p_aff = np.linspace(120.0e9, 0.0, n_aff)
    st_aff = compute_stagnant_lid_state(
        radii=r_aff,
        temperature=t_aff,
        pressure=p_aff,
        convective_flux=np.full(n_aff, 1.0e-4),
        total_flux=np.ones(n_aff),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_aff),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    p_aff_expected = 120.0e9 * (float(st_aff['T_i']) - 1200.0) / (3000.0 - 1200.0)
    r_aff_expected = 3.48e6 + (6.371e6 - 3.48e6) * (3000.0 - float(st_aff['T_i'])) / (
        3000.0 - 1200.0
    )
    assert float(st_aff['P_T_i']) == pytest.approx(p_aff_expected, rel=1.0e-14)
    assert float(st_aff['r_i']) == pytest.approx(r_aff_expected, rel=1.0e-14)

    # Discriminating non-affine geotherm P(T) = a * (T - T_surf)**2:
    # P_T_i strictly differs from the linear chord sum (Lens 2 F7 item 3).
    p_quad = 120.0e9 * ((t_aff - 1200.0) / (3000.0 - 1200.0)) ** 2
    st_quad = compute_stagnant_lid_state(
        radii=r_aff,
        temperature=t_aff,
        pressure=p_quad,
        convective_flux=np.full(n_aff, 1.0e-4),
        total_flux=np.ones(n_aff),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_aff),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    t_i_quad = float(st_quad['T_i'])
    p_geo_quad = float(np.interp(t_i_quad, t_aff[::-1], p_quad[::-1]))
    p_linear_chord = 120.0e9 * (t_i_quad - 1200.0) / (3000.0 - 1200.0)
    assert float(st_quad['P_T_i']) == pytest.approx(p_geo_quad, rel=1.0e-9)
    assert abs(float(st_quad['P_T_i']) - p_linear_chord) > 1.0e10

    # Discriminating test for is_mono_T tolerance and dT_ep_nz threshold at 1e-12 (Lens 2 F4a):
    # 1. is_mono_T: sub-threshold step (1e-13) takes fallback branch; super-threshold step (1e-11) takes geotherm branch.
    t_sub = t_aff.copy()
    t_sub[21:] = t_aff[21:] + (t_sub[20] - t_sub[21]) - 1.0e-13
    st_sub_mono = compute_stagnant_lid_state(
        radii=r_aff,
        temperature=t_sub,
        pressure=p_quad,
        convective_flux=np.full(n_aff, 1.0e-4),
        total_flux=np.ones(n_aff),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_aff),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    t_sup = t_aff.copy()
    t_sup[21:] = t_aff[21:] + (t_sup[20] - t_sup[21]) - 1.0e-11
    st_sup_mono = compute_stagnant_lid_state(
        radii=r_aff,
        temperature=t_sup,
        pressure=p_quad,
        convective_flux=np.full(n_aff, 1.0e-4),
        total_flux=np.ones(n_aff),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_aff),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert abs(float(st_sub_mono['P_T_i']) - float(st_sup_mono['P_T_i'])) > 1.0e9

    # 2. dT_ep_nz: across sub-threshold (1e-13) vs super-threshold (1e-11) temperature differences
    fc_ep = np.zeros(n_aff)
    fc_ep[1] = 6.00843e-5
    fc_ep[2] = 6.00843e-5
    t_ep_sub = t_aff.copy()
    t_ep_sub[2] = t_ep_sub[1] - 1.0e-13
    t_ep_sub[3:] = t_ep_sub[2] - np.linspace(10.0, 1000.0, n_aff - 3)
    st_ep_sub = compute_stagnant_lid_state(
        radii=r_aff,
        temperature=t_ep_sub,
        pressure=p_quad,
        convective_flux=fc_ep,
        total_flux=np.ones(n_aff),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_aff),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    t_ep_sup = t_aff.copy()
    t_ep_sup[2] = t_ep_sup[1] - 1.0e-11
    t_ep_sup[3:] = t_ep_sup[2] - np.linspace(10.0, 1000.0, n_aff - 3)
    st_ep_sup = compute_stagnant_lid_state(
        radii=r_aff,
        temperature=t_ep_sup,
        pressure=p_quad,
        convective_flux=fc_ep,
        total_flux=np.ones(n_aff),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_aff),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert abs(float(st_ep_sub['theta']) - float(st_ep_sup['theta'])) > 0.01

    # Curved geotherm where endpoint blending pushes unclipped P_T_i > max(P) and r_i < r[0]:
    # clipping keeps P_T_i inside [min(P), max(P)] and r_i inside [r[0], r[-1]].
    r_clip = np.array([3.0e6, 3.01e6, 3.02e6, 5.98e6, 5.99e6, 6.0e6])
    t_clip = np.array([3000.0, 2900.0, 2500.0, 2100.0, 1300.0, 1000.0])
    p_clip = np.array([100.0e9, 99.0e9, 98.0e9, 1.0e9, 0.5e9, 0.0])
    fc_clip = np.array([0.0, 1.5e-5, 0.0, 0.0, 0.9 * 1.5e-5, 0.0])
    st_clip = compute_stagnant_lid_state(
        radii=r_clip,
        temperature=t_clip,
        pressure=p_clip,
        convective_flux=fc_clip,
        total_flux=np.ones(6),
        solidus_temperature=None,
        melt_fraction=np.zeros(6),
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(np.min(p_clip)) <= float(st_clip['P_T_i']) <= float(np.max(p_clip))
    assert float(r_clip[0]) <= float(st_clip['r_i']) <= float(r_clip[-1])
    assert float(st_clip['P_T_i']) == pytest.approx(float(np.max(p_clip)), rel=1.0e-14)
    assert float(st_clip['r_i']) == pytest.approx(float(r_clip[0]), rel=1.0e-14)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_fixed_lid_base_mode_excludes_convecting_nodes():
    """Verify fixed lid-base mode weights both w_hot_iso and w_hot_iso_mask by w_nonconv."""
    _, jnp = _import_jax_x64()

    n = 50
    r = np.linspace(3.48e6, 6.370e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.full(n, 1500.0)
    t[45:] = np.linspace(1500.0, 300.0, 5)
    fc = np.full(n, 50.0)
    fc[45:] = 0.0
    ft = np.full(n, 60.0)
    phi = np.zeros(n)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=1800.0,
        interior_flux_fraction=0.05,
        g_homologous=25.0,
        t_sol_0=1400.0,
        dt_sol_dp=0.0,
    )
    for xp_mod in (np, jnp):
        st = compute_stagnant_lid_state(
            radii=xp_mod.asarray(r),
            temperature=xp_mod.asarray(t),
            pressure=xp_mod.asarray(p),
            convective_flux=xp_mod.asarray(fc),
            total_flux=xp_mod.asarray(ft),
            solidus_temperature=None,
            melt_fraction=xp_mod.asarray(phi),
            params=params,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=xp_mod,
        )
        w_lid = np.asarray(st['w_lid'], dtype=float)
        assert np.max(w_lid[:45]) == 0.0
        assert np.sum(w_lid[:45]) == 0.0
        assert np.all(w_lid[45:] > 0.999)
        assert float(st['lid_cell_count']) == pytest.approx(
            float(np.sum(w_lid[45:])), rel=1.0e-12
        )
        assert float(st['lid_cell_count']) == pytest.approx(4.999911573009957, rel=1.0e-12)
        assert float(st['r_lid_base']) == pytest.approx(6078786.231143484, rel=1.0e-12)
        assert float(st['d_lid']) == pytest.approx(291213.7688565161, rel=1.0e-12)
        d_geom = float(r[-1] - r[0]) * float(st['tau_d']) / float(st['tau_d_lid'])
        assert d_geom == pytest.approx(385333.5273929512, rel=1.0e-12)
        assert float(st['tau_d_lid']) == pytest.approx(5715108.044708197, rel=1.0e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_d_geom_floor_tau_d_lid_yield_and_d_lid_raw():
    """Verify d_geom floor bounds tau_d_lid on incipient lids, w_y compares tau_d_lid with tau_y_lid, and d_lid_raw is returned."""
    for n in (3, 10, 50, 100):
        r = np.linspace(3.48e6, 6.371e6, n)
        p = np.linspace(135.0e9, 1.0e5, n)
        t = np.linspace(2600.0, 1200.0, n)
        fc = np.full(n, 0.08)
        fc[-1] = 0.0
        ft = np.full(n, 0.10)
        for phi_val in (0.0, 0.015, 0.020, 0.025):
            st = compute_stagnant_lid_state(
                radii=r,
                temperature=t,
                pressure=p,
                convective_flux=fc,
                total_flux=ft,
                solidus_temperature=None,
                melt_fraction=np.full(n, phi_val),
                params=SolidRheologyParams(enabled=True, stress_closure_mode='lid'),
                viscosity_solid=1.0e21,
                phi_rheo=0.4,
                phi_width=0.15,
                xp=np,
            )
            assert 'd_lid_raw' in st
            assert len(st) == 32
            d_lid = float(st['d_lid'])
            d_lid_raw = float(st['d_lid_raw'])
            w_act = float(st['w_active'])
            assert 0.0 <= d_lid <= d_lid_raw * (1.0 + 1.0e-12)
            assert d_lid_raw <= float(r[-1] - r[0]) * w_act * (1.0 + 1.0e-12)

            dr_top = float(r[-1] - r[-2])
            delta_rh = float(st['delta_rh'])
            d_floor = max(delta_rh, dr_top)
            d_conv = float(r[-1] - r[0])
            tau_d = float(st['tau_d'])
            tau_d_lid = float(st['tau_d_lid'])
            if tau_d > 0.0:
                assert tau_d_lid / tau_d <= (d_conv / d_floor) * (1.0 + 1.0e-12)

    # Sub-threshold thermal contrast (0 < T_i - T_surf < 1.0e-3 K): has_lid is False, d_lid_raw > d_lid > 0.
    r_sub = np.linspace(3.48e6, 6.371e6, 10)
    p_sub = np.zeros(10)
    t_sub = np.full(10, 2000.0)
    t_sub[-1] = 2000.0 - 0.0005
    fc_sub = np.full(10, 0.08)
    fc_sub[-1] = 0.0
    st_sub = compute_stagnant_lid_state(
        radii=r_sub,
        temperature=t_sub,
        pressure=p_sub,
        convective_flux=fc_sub,
        total_flux=np.full(10, 0.10),
        solidus_temperature=None,
        melt_fraction=np.zeros(10),
        params=SolidRheologyParams(
            enabled=True,
            interior_flux_fraction=0.05,
            lid_base_mode='fixed',
            lid_base_temperature=1999.9998,
        ),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert bool(st_sub['has_lid']) is False
    assert 0.0 < float(st_sub['d_lid']) < float(st_sub['d_lid_raw'])


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_saturating_activation_enthalpy_limits(validation_plot) -> None:
    """Verify P_decay -> inf and V_0 = 0 limits of saturating activation enthalpy."""
    import math

    jax, jnp = _import_jax_x64()
    from aragog.rheology import compute_arrhenius_enthalpy

    p_gpa = np.linspace(0.0, 1000.0, 201)
    p_pa = p_gpa * 1.0e9
    e_a = 300.0e3

    # Limit P_decay = inf: bitwise equality with E_a + V_0 * P for Tackley et al. (2013)
    # Table 2 upper mantle (V_0 = 5.00 cm^3/mol) and Armann & Tackley (2012) (V_0 = 6.00 cm^3/mol)
    for v_0 in (5.0e-6, 6.0e-6):
        h_lin_exact = e_a + v_0 * p_pa
        h_inf_np = compute_arrhenius_enthalpy(
            p_pa,
            activation_energy=e_a,
            activation_volume=v_0,
            activation_volume_decay_pressure=math.inf,
            xp=np,
        )
        h_inf_jax = np.asarray(
            compute_arrhenius_enthalpy(
                jnp.asarray(p_pa),
                activation_energy=e_a,
                activation_volume=v_0,
                activation_volume_decay_pressure=math.inf,
                xp=jnp,
            )
        )
        np.testing.assert_array_equal(h_inf_np, h_lin_exact)
        np.testing.assert_array_equal(h_inf_jax, h_lin_exact)

        # Canary: perturbing 1 entry by 1 ulp fails bitwise equality
        h_bad = h_inf_np.copy()
        h_bad[100] = np.nextafter(h_bad[100], math.inf)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(h_bad, h_lin_exact)

    # Analytical Lagrange remainder bound: 0 <= H_lin(P) - H_int(P) <= V_0 * P^2 / (2 * P_decay)
    v0_ref = 5.0e-6
    h_lin_ref = e_a + v0_ref * p_pa
    p_decay_curves_gpa = (200.0, 1100.0, 1610.0)
    curves_kj: dict[float, np.ndarray] = {}
    for pd_gpa in p_decay_curves_gpa:
        pd_pa = pd_gpa * 1.0e9
        h_int_np = compute_arrhenius_enthalpy(
            p_pa,
            activation_energy=e_a,
            activation_volume=v0_ref,
            activation_volume_decay_pressure=pd_pa,
            xp=np,
        )
        h_int_jax = np.asarray(
            compute_arrhenius_enthalpy(
                jnp.asarray(p_pa),
                activation_energy=e_a,
                activation_volume=v0_ref,
                activation_volume_decay_pressure=pd_pa,
                xp=jnp,
            )
        )
        np.testing.assert_allclose(h_int_np, h_int_jax, rtol=1.0e-15, atol=0.0)
        dev = h_lin_ref - h_int_np
        bound = v0_ref * (p_pa**2) / (2.0 * pd_pa)
        assert np.all(dev >= 0.0)
        assert np.all(dev <= bound * (1.0 + 1.0e-14))
        assert np.all(h_int_np <= e_a + v0_ref * pd_pa * (1.0 + 1.0e-15))
        curves_kj[pd_gpa] = h_int_np / 1.0e3

    # Tackley et al. (2013) secant form H_sec(P) = E_0 + P * V_0 * exp(-P / P_decay) diverges above P_decay
    pd_200_pa = 200.0e9
    h_sec_200_kj = (e_a + p_pa * v0_ref * np.exp(-p_pa / pd_200_pa)) / 1.0e3
    assert curves_kj[200.0][-1] > h_sec_200_kj[-1] + 500.0

    # Physical discrimination canaries: secant form and wrong-sign exponential violate Lagrange bounds
    dev_sec_200 = h_lin_ref[1:] - (e_a + p_pa[1:] * v0_ref * np.exp(-p_pa[1:] / pd_200_pa))
    bound_200 = v0_ref * (p_pa[1:] ** 2) / (2.0 * pd_200_pa)
    assert np.any(dev_sec_200 > bound_200 * 1.1)
    h_wrong_sign = e_a + v0_ref * pd_200_pa * np.expm1(p_pa[1:] / pd_200_pa)
    assert np.all(h_lin_ref[1:] - h_wrong_sign < 0.0)

    # Verify compute_arrhenius_viscosity 0-ulp bitwise equality at P_decay = inf and V_0 = 0
    temp_arr = np.linspace(1600.0, 3500.0, p_pa.size)
    eta_lin_np = compute_arrhenius_viscosity(
        temp_arr,
        p_pa,
        viscosity_solid=1.0e21,
        activation_energy=e_a,
        activation_volume=v0_ref,
        activation_volume_decay_pressure=math.inf,
        xp=np,
    )
    eta_lin_default_np = compute_arrhenius_viscosity(
        temp_arr,
        p_pa,
        viscosity_solid=1.0e21,
        activation_energy=e_a,
        activation_volume=v0_ref,
        xp=np,
    )
    np.testing.assert_array_equal(eta_lin_np, eta_lin_default_np)

    # Verify JAX autodiff dH/dP = V_0 * exp(-P / P_decay) for V_0 > 0 across Taylor and exp branches
    p_grad_test = np.array([0.0, 100.0, 1000.0, 1.0e9, 50.0e9, 135.0e9])
    for pd_pa in (200.0e9, 1100.0e9, 1610.0e9):
        grad_v0_pos = jax.grad(
            lambda p_s: jnp.sum(
                compute_arrhenius_enthalpy(
                    p_s,
                    activation_energy=e_a,
                    activation_volume=v0_ref,
                    activation_volume_decay_pressure=pd_pa,
                    xp=jnp,
                )
            )
        )
        dh_dp_pos = np.asarray(grad_v0_pos(jnp.asarray(p_grad_test)))
        expected_dh_dp = v0_ref * np.exp(-p_grad_test / pd_pa)
        np.testing.assert_allclose(dh_dp_pos, expected_dh_dp, rtol=1.0e-12, atol=0.0)

    # Sweep P_decay across [100, 1e6] GPa at P = 135 GPa (Earth CMB)
    p_cmb = 135.0e9
    p_decay_sweep_gpa = np.logspace(2.0, 6.0, 41)
    dev_cmb_kj = np.empty_like(p_decay_sweep_gpa)
    bound_cmb_kj = np.empty_like(p_decay_sweep_gpa)
    h_lin_cmb = e_a + v0_ref * p_cmb
    for idx, pd_gpa in enumerate(p_decay_sweep_gpa):
        pd_pa = float(pd_gpa) * 1.0e9
        h_val = float(
            compute_arrhenius_enthalpy(
                p_cmb,
                activation_energy=e_a,
                activation_volume=v0_ref,
                activation_volume_decay_pressure=pd_pa,
                xp=np,
            )
        )
        dev_cmb_kj[idx] = (h_lin_cmb - h_val) / 1.0e3
        bound_cmb_kj[idx] = (v0_ref * (p_cmb**2) / (2.0 * pd_pa)) / 1.0e3
    assert np.all(dev_cmb_kj > 0.0)
    assert np.all(dev_cmb_kj <= bound_cmb_kj * (1.0 + 1.0e-14))

    # Limit V_0 = 0: bitwise equality H(P) == E_a and dH/dP == 0.0 across P_decay in {200, 1100, 1610, inf} GPa
    expected_v0_zero = np.full_like(p_pa, e_a)
    eta_iso_ref_np = compute_arrhenius_viscosity(
        temp_arr,
        p_pa,
        viscosity_solid=1.0e21,
        activation_energy=e_a,
        activation_volume=0.0,
        activation_volume_decay_pressure=math.inf,
        xp=np,
    )
    eta_iso_ref_jax = np.asarray(
        compute_arrhenius_viscosity(
            jnp.asarray(temp_arr),
            jnp.asarray(p_pa),
            viscosity_solid=1.0e21,
            activation_energy=e_a,
            activation_volume=0.0,
            activation_volume_decay_pressure=math.inf,
            xp=jnp,
        )
    )
    for pd_pa in (200.0e9, 1100.0e9, 1610.0e9, math.inf):
        h_v0_np = compute_arrhenius_enthalpy(
            p_pa,
            activation_energy=e_a,
            activation_volume=0.0,
            activation_volume_decay_pressure=pd_pa,
            xp=np,
        )
        h_v0_jax = np.asarray(
            compute_arrhenius_enthalpy(
                jnp.asarray(p_pa),
                activation_energy=e_a,
                activation_volume=0.0,
                activation_volume_decay_pressure=pd_pa,
                xp=jnp,
            )
        )
        np.testing.assert_array_equal(h_v0_np, expected_v0_zero)
        np.testing.assert_array_equal(h_v0_jax, expected_v0_zero)

        eta_v0_np = compute_arrhenius_viscosity(
            temp_arr,
            p_pa,
            viscosity_solid=1.0e21,
            activation_energy=e_a,
            activation_volume=0.0,
            activation_volume_decay_pressure=pd_pa,
            xp=np,
        )
        eta_v0_jax = np.asarray(
            compute_arrhenius_viscosity(
                jnp.asarray(temp_arr),
                jnp.asarray(p_pa),
                viscosity_solid=1.0e21,
                activation_energy=e_a,
                activation_volume=0.0,
                activation_volume_decay_pressure=pd_pa,
                xp=jnp,
            )
        )
        np.testing.assert_array_equal(eta_v0_np, eta_iso_ref_np)
        np.testing.assert_array_equal(eta_v0_jax, eta_iso_ref_jax)

        grad_fn = jax.grad(
            lambda p_s: jnp.sum(
                compute_arrhenius_enthalpy(
                    p_s,
                    activation_energy=e_a,
                    activation_volume=0.0,
                    activation_volume_decay_pressure=pd_pa,
                    xp=jnp,
                )
            )
        )
        dh_dp = np.asarray(grad_fn(jnp.asarray(p_pa)))
        np.testing.assert_array_equal(dh_dp, np.zeros_like(p_pa))

    def figure():
        import matplotlib.pyplot as plt

        sl = slice(None, None, 8)
        sl_pd = slice(None, None, 2)
        plt.rcParams.update(
            {
                'axes.labelsize': 11,
                'xtick.labelsize': 9.5,
                'ytick.labelsize': 9.5,
                'legend.fontsize': 7.5,
            }
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))
        ax1.plot(p_gpa[sl], (h_lin_ref / 1.0e3)[sl], label='P_decay = inf')
        ax1.plot(p_gpa[sl], curves_kj[1610.0][sl], '--', label='1610 GPa')
        ax1.plot(p_gpa[sl], curves_kj[1100.0][sl], '-.', label='1100 GPa')
        ax1.plot(p_gpa[sl], curves_kj[200.0][sl], ':', label='200 GPa (int)')
        ax1.plot(p_gpa[sl], h_sec_200_kj[sl], '--', label='200 GPa (sec)')
        ax1.plot(p_gpa[sl], (expected_v0_zero / 1.0e3)[sl], '-', label='V_0 = 0')
        ax1.set(
            xlabel='(a) P [GPa]',
            ylabel='H(P) [kJ/mol]',
            xticks=[0, 500, 1000],
            yticks=[1000, 3000, 5000],
        )
        ax1.legend()

        ax2.plot(
            np.log10(p_decay_sweep_gpa[sl_pd]),
            np.log10(dev_cmb_kj[sl_pd]),
            label='dH = H_lin - H_int',
        )
        ax2.plot(
            np.log10(p_decay_sweep_gpa[sl_pd]),
            np.log10(bound_cmb_kj[sl_pd]),
            '--',
            label='Bound V_0 P^2 / (2 P_decay)',
        )
        ax2.set(
            xlabel='(b) log10 P_decay [GPa] (P = 135 GPa)',
            ylabel='log10 dH(135 GPa) [kJ/mol]',
            xticks=[2, 4, 6],
            yticks=[-1, 1, 3],
        )
        ax2.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'ssc_wp4_activation_law_limits')
