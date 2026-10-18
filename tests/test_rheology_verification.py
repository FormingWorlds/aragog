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
    eta_match = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_match,
        stress_closure_mode='local',
    )
    assert eta_match == pytest.approx(eta_d, rel=1.0e-12)

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
    assert float(eta_match_jax) == pytest.approx(eta_d, rel=1.0e-12)

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
    td_below = 8.645e5
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
        tau_d=8.645e5,
        tau_y_lid=1.0e6,
    )
    assert eta_unyielded_cold == pytest.approx(eta_cold_lid, rel=1.0e-14)

    eta_yielded_cold = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=1.2e6,
        tau_y_lid=1.0e6,
    )
    expected_yielded_cold = eta_interior * (1.0e6 / 1.2e6)
    assert eta_yielded_cold == pytest.approx(expected_yielded_cold, rel=1.0e-14)
    assert eta_yielded_cold < eta_interior

    # Canary: calling compute_effective_viscosity with cold-lid creep viscosity as the scale
    # (eta_i=eta_cold_lid or omitted) leaves 8.333e39 Pa s >> eta_interior and fails mobilization.
    wrong_cold_scaled = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_cold_lid,
        tau_d=1.2e6,
        tau_y_lid=1.0e6,
    )
    assert wrong_cold_scaled == pytest.approx(8.333333333333334e39, rel=1.0e-14)
    with pytest.raises(AssertionError):
        assert wrong_cold_scaled < eta_interior

    # Three reference regimes (cold surface 300 K, warm surface 700 K, and low-yield lid):
    _, jnp = _import_jax_x64()
    cases = [
        (1.0e40, 1.0e21, 1.2e6, 1.0e6, 8.333333333333334e20),
        (8.30e24, 1.0e21, 1.2e6, 1.0e6, 8.333333333333334e20),
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

    # 4. Yield threshold transition at tau_d = tau_y (both eta_i == eta_d and eta_i < eta_d).
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
    assert eta_left == pytest.approx(eta_d, rel=1.0e-14)
    assert eta_mid == pytest.approx(eta_d, rel=1.0e-14)
    assert eta_right == pytest.approx(eta_d, rel=1.0e-11)

    eta_cold_left = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=tau_y - eps,
        tau_y_lid=tau_y,
    )
    eta_cold_right = compute_effective_viscosity(
        eta_diff=eta_cold_lid,
        eta_i=eta_interior,
        tau_d=tau_y + eps,
        tau_y_lid=tau_y,
    )
    assert eta_cold_left == pytest.approx(eta_cold_lid, rel=1.0e-14)
    assert eta_cold_right == pytest.approx(eta_interior, rel=1.0e-11)


@pytest.mark.physics_invariant
def test_lid_closure_jump_at_yield_threshold():
    """Verify lid closure jumps at tau_d = tau_y_lid from eta_diff below/at yield to eta_i * tau_y/tau_d above."""
    _, jnp = _import_jax_x64()

    eta_d = 1.0e26
    eta_i = 1.0e20
    tau_y = 1.0e8
    w_lid = 1.0

    # Below yield (tau_d / tau_y = 0.999999) and at yield (tau_d / tau_y = 1.0): eta_eff == eta_diff (1.0e26).
    for ratio_td_ty in (0.999999, 1.0):
        td = ratio_td_ty * tau_y
        val_np = float(
            compute_effective_viscosity(
                eta_diff=eta_d,
                tau_d=td,
                tau_y_lid=tau_y,
                stress_closure_mode='lid',
                w_lid=w_lid,
                eta_i=eta_i,
                xp=np,
            )
        )
        val_jx = float(
            compute_effective_viscosity(
                eta_diff=jnp.asarray(eta_d),
                tau_d=jnp.asarray(td),
                tau_y_lid=jnp.asarray(tau_y),
                stress_closure_mode='lid',
                w_lid=jnp.asarray(w_lid),
                eta_i=jnp.asarray(eta_i),
                xp=jnp,
            )
        )
        assert val_np == pytest.approx(1.0e26, rel=1.0e-14)
        assert val_jx == pytest.approx(1.0e26, rel=1.0e-14)

    # Immediately above yield (tau_d / tau_y = 1.000001): eta_eff == eta_i * (tau_y / tau_d) = 1.0e20 / 1.000001.
    td_above = 1.000001 * tau_y
    expected_above = 1.0e20 / 1.000001
    val_above_np = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_d=td_above,
            tau_y_lid=tau_y,
            stress_closure_mode='lid',
            w_lid=w_lid,
            eta_i=eta_i,
            xp=np,
        )
    )
    val_above_jx = float(
        compute_effective_viscosity(
            eta_diff=jnp.asarray(eta_d),
            tau_d=jnp.asarray(td_above),
            tau_y_lid=jnp.asarray(tau_y),
            stress_closure_mode='lid',
            w_lid=jnp.asarray(w_lid),
            eta_i=jnp.asarray(eta_i),
            xp=jnp,
        )
    )
    assert val_above_np == pytest.approx(expected_above, rel=1.0e-14)
    assert val_above_jx == pytest.approx(expected_above, rel=1.0e-14)
    assert val_above_np == pytest.approx(1.0e20, rel=2.0e-6)
    assert (1.0e26 / val_above_np) == pytest.approx(1.0e6 * 1.000001, rel=1.0e-14)

    # Canary: cold-lid scaling eta_diff * (tau_y / tau_d) = 1.0e26 / 1.000001 fails the interior-scaled contract.
    wrong_cold_above = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_d=td_above,
            tau_y_lid=tau_y,
            stress_closure_mode='lid',
            w_lid=w_lid,
            eta_i=eta_d,
            xp=np,
        )
    )
    with pytest.raises(AssertionError):
        assert wrong_cold_above == pytest.approx(expected_above, rel=1.0e-3)


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
    # if_lid sets isothermal no-lid diagnostics to literal 0.0.
    for key in (
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
    # Exact by construction: xp.maximum(0.0, ...) clamps negative Byerlee stress to literal 0.0.
    assert compute_yield_stress(-200.0e6, yield_stress_c=50.0e6, yield_stress_mu=0.6) == 0.0
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
        t_sol_0=1400.0,
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

    with pytest.raises(ValueError, match='t_sol_0 must be provided when solidus_temperature'):
        compute_stagnant_lid_state(
            radii=r,
            temperature=T,
            pressure=P,
            convective_flux=F_conv,
            total_flux=F_tot,
            solidus_temperature=2000.0,
            t_sol_0=None,
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
        t_sol_0=2000.0,
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
        t_sol_0=2000.0,
        melt_fraction=phi,
        params=params_fixed_2000,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    with pytest.raises(ValueError, match='t_sol_0'):
        compute_stagnant_lid_state(
            radii=r,
            temperature=T,
            pressure=P,
            convective_flux=F_conv,
            total_flux=F_tot,
            solidus_temperature=np.full(n_nodes, 2000.0),
            t_sol_0=2000.0,
            melt_fraction=phi,
            params=dataclasses.replace(params_fixed, t_sol_0=2200.0),
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
        )
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

    # 5. Water prefactor clipped in log space after multiplication:
    # At T=520 K, P=0, wp=0.1: unclipped is 1e21 * 0.1 * exp(25 * 1400 * (1/520 - 1/1600)) = 5.424869e39 Pa s.
    expected_520 = 1.0e21 * 0.1 * math.exp(25.0 * 1400.0 * (1.0 / 520.0 - 1.0 / 1600.0))
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

    # 7. When W_base_sum -> 0 (entire mantle below lid_base_temperature=5000 K),
    # r_lid_base goes to r[0] (CMB) and d_lid equals full mantle thickness r[-1] - r[0].
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
    assert float(st_full_lid['r_lid_base']) == pytest.approx(r[0], rel=1e-9)
    assert float(st_full_lid['d_lid']) == pytest.approx(r[-1] - r[0], rel=1e-4)

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
        't_sol_0',
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
        't_sol_0': 1400.0,
        'a_rh_fb': 1.3,
        'C4': 0.125,
    }
    if nan_field in base_arrays:
        bad = base_arrays[nan_field].copy()
        bad[5] = np.nan
        base_arrays[nan_field] = bad
    else:
        scalars[nan_field] = float('nan')

    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0, lid_base_mode=lid_base_mode)
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
        extra = {'t_sol_0': 1400.0} if opt_name == 'solidus_temperature' else {}
        with pytest.raises(ValueError, match=rf'{opt_name}.*same 1-D length as radii'):
            compute_stagnant_lid_state(
                **{**valid_kw, **extra, opt_name: np.ones((5, 2), dtype=float)}
            )


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
    """Verify t_sol_0 rejects +-inf and mismatched values while propagating NaN."""
    valid_kw = _valid_lid_state_kwargs()
    for bad_ts0 in (np.inf, -np.inf, 1500.0):
        with pytest.raises(ValueError, match='t_sol_0'):
            compute_stagnant_lid_state(**{**valid_kw, 't_sol_0': bad_ts0})
    st_nan_ts0 = compute_stagnant_lid_state(**{**valid_kw, 't_sol_0': np.nan})
    assert np.isnan(float(st_nan_ts0['tau_d']))


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
    from aragog.rheology_lid import _scalar_or_nan

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
            _scalar_or_nan(bad_scalar, 'viscosity_solid')

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
        ('t_sol_0', np.full(n_nodes, 1400.0)),
    ):
        with pytest.raises(ValueError, match=rf'{scalar_name} must be a scalar'):
            compute_stagnant_lid_state(**{**valid_kw, scalar_name: bad_arr})

    assert _scalar_or_nan(10**21, 'viscosity_solid') == 1.0e21
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
    def _jit_eval(vs, pw, pr, arh, c4, ts0):
        return compute_stagnant_lid_state(
            **{
                **j_kw,
                'viscosity_solid': vs,
                'phi_width': pw,
                'phi_rheo': pr,
                'a_rh_fb': arh,
                'C4': c4,
                't_sol_0': ts0,
                'xp': jnp,
            }
        )

    st_jit_ok = _jit_eval(1.0e21, 0.15, 0.4, 1.3, 0.125, 1400.0)
    assert bool(jnp.isfinite(st_jit_ok['tau_d']))
    for bad_args in (
        (-1.0e21, 0.15, 0.4, 1.3, 0.125, 1400.0),
        (jnp.inf, 0.15, 0.4, 1.3, 0.125, 1400.0),
        (1.0e21, 0.0, 0.4, 1.3, 0.125, 1400.0),
        (1.0e21, 0.15, 1.5, 1.3, 0.125, 1400.0),
        (1.0e21, 0.15, 0.4, 0.0, 0.125, 1400.0),
        (1.0e21, 0.15, 0.4, 1.3, -0.125, 1400.0),
        (1.0e21, 0.15, 0.4, 1.3, 0.125, 1500.0),
        (1.0e21, 0.15, 0.4, 1.3, 0.125, jnp.inf),
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
