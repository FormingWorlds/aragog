"""Unit and property tests for the standalone 0-D stagnant-lid reference helper.

This module tests the independent 0-D parameterized stagnant-lid helper in
``tests/reference/stagnant_lid_0d.py`` formulated
from Moresi & Solomatov (1998), Solomatov & Moresi (2000), Foley & Bercovici (2014),
and Thiriet et al. (2019). It verifies import isolation from ``aragog.*``,
analytic limits of activation enthalpy and Arrhenius viscosity, stagnant-lid
boundary-layer scalings, the ``0.5**(1/3)`` yield-ceiling lid-thinning law,
the steady-state equilibrium and first-law energy balance of the 0-D cooling ODE,
and Hypothesis property and input-validation invariants.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.reference.stagnant_lid_0d import (
    StagnantLid0DParams,
    compute_0d_activation_enthalpy,
    compute_0d_interior_viscosity,
    evaluate_stagnant_lid_0d,
    integrate_stagnant_lid_0d,
    load_aragog_0d_params,
    softmin2_scalar,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / 'tests' / 'reference' / 'stagnant_lid_0d.py'
RHEOLOGY_PY = REPO_ROOT / 'src' / 'aragog' / 'rheology.py'
ABE_SOLID_TOML = REPO_ROOT / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'


def _assert_raises_value_error(fn: Any, *args: Any, match: str, **kwargs: Any) -> None:
    """Call ``fn`` and assert via ``AssertionError`` that it raises ``ValueError`` matching ``match``."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        assert isinstance(exc, ValueError), (
            f'Expected ValueError containing {match!r}, got {type(exc).__name__}: {exc}'
        )
        assert match in str(exc), f'Expected {match!r} in ValueError message, got {str(exc)!r}'
    else:
        raise AssertionError(
            f'Expected ValueError containing {match!r}, but call returned normally'
        )


def _find_forbidden_aragog_imports(source_text: str) -> list[str]:
    """Scan Python source AST for any static or dynamic import of ``aragog``."""
    tree = ast.parse(source_text)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'aragog' or alias.name.startswith('aragog.'):
                    violations.append(f'import {alias.name}')
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod == 'aragog' or mod.startswith('aragog.'):
                violations.append(f'from {mod} import ...')
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in ('__import__', 'import_module')
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith('aragog')
            ):
                violations.append(f'{node.func.id}({node.args[0].value!r})')
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'import_module'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith('aragog')
            ):
                violations.append(f'import_module({node.args[0].value!r})')
    return violations


@functools.lru_cache(maxsize=1)
def _cached_base_params() -> StagnantLid0DParams:
    """Parse and cache the default active-solid 0-D parameter bundle once per process."""
    return load_aragog_0d_params(
        abe_solid_toml=ABE_SOLID_TOML,
        viscosity_solid=1.0e18,
        t_surf=300.0,
        p_i_depth_m=100.0e3,
        a_rh_fb=1.3,
        c4_fb=0.125,
        a_nu=0.5954345503010264,
        yield_blend_ratio_width=0.025,
    )


def _make_base_params(**overrides: Any) -> StagnantLid0DParams:
    """Build the active-solid 0-D parameter set from repo config files with optional overrides."""
    base = _cached_base_params()
    if not overrides:
        return base
    return dataclasses.replace(base, **overrides)


def test_stagnant_lid_0d_helper_import_isolation() -> None:
    """Verify that ``tests/reference/stagnant_lid_0d.py`` imports no ``aragog`` modules."""
    source_text = HELPER_PATH.read_text(encoding='utf-8')
    assert _find_forbidden_aragog_imports(source_text) == []

    # Negative canary: verify the AST scanner catches both static and dynamic aragog imports.
    canary_src = (
        'import aragog.rheology\n'
        'from aragog.rheology_lid import compute_stagnant_lid_state\n'
        'import importlib\n'
        "importlib.import_module('aragog.solver')\n"
        'from importlib import import_module\n'
        "import_module('aragog.rheology')\n"
        "__import__('aragog.mesh')\n"
    )
    canary_hits = _find_forbidden_aragog_imports(canary_src)
    assert len(canary_hits) == 5

    # Subprocess check: importing the helper in a fresh interpreter loads zero aragog modules.
    code = (
        'import sys\n'
        'import tests.reference.stagnant_lid_0d as m\n'
        'aragog_mods = [k for k in sys.modules if k == "aragog" or k.startswith("aragog.")]\n'
        'assert not aragog_mods, f"Loaded aragog modules: {aragog_mods}"\n'
        'assert hasattr(m, "evaluate_stagnant_lid_0d")\n'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f'Subprocess failed: {proc.stderr}'


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_stagnant_lid_0d_activation_and_viscosity_limits() -> None:
    """Verify analytic limits of 0-D activation enthalpy and Arrhenius viscosity.

    Pins the homologous activation enthalpy law ``H(P) = g_hom * R * T_sol(P)``
    and the saturating integral law ``H(P) = E_a + V_0 * P_decay * (1 - exp(-P / P_decay))``
    (Moresi & Solomatov 1998 Eq. 6, p. 671; Thiriet et al. 2019 Eq. 10, p. 140),
    together with the reference normalization and softmin2 ceiling of ``eta_i``.
    """
    params = _make_base_params()
    p_i = 4.1202e9

    # 1. Homologous law at P = 0 and P = 4.1202 GPa.
    h_0 = compute_0d_activation_enthalpy(0.0, params)
    h_pi = compute_0d_activation_enthalpy(p_i, params)
    assert h_0 == pytest.approx(291006.19163, rel=1e-12)
    assert h_pi == pytest.approx(311560.5409572102, rel=1e-12)
    expected_dh = params.g_homologous * params.r_gas * params.dt_sol_dp * p_i
    assert (h_pi - h_0) == pytest.approx(expected_dh, rel=1e-12)

    # 2. Saturating integral law limits: P_decay = +inf, V_0 = 0, P = 0, and P >> P_decay.
    e_a = 3.0e5
    v_0 = 4.0e-6
    p_decay = 10.0e9
    params_lin = _make_base_params(
        activation_energy=e_a,
        activation_volume=v_0,
        activation_volume_decay_p=math.inf,
    )
    params_v0_zero = _make_base_params(
        activation_energy=e_a,
        activation_volume=0.0,
        activation_volume_decay_p=p_decay,
    )
    params_sat = _make_base_params(
        activation_energy=e_a,
        activation_volume=v_0,
        activation_volume_decay_p=p_decay,
    )

    assert compute_0d_activation_enthalpy(p_i, params_lin) == e_a + v_0 * p_i
    assert compute_0d_activation_enthalpy(p_i, params_v0_zero) == e_a
    assert compute_0d_activation_enthalpy(0.0, params_sat) == e_a

    h_sat = compute_0d_activation_enthalpy(p_i, params_sat)
    h_lin = compute_0d_activation_enthalpy(p_i, params_lin)
    lagrange_bound = 0.5 * v_0 * (p_i**2) / p_decay
    assert 0.0 < (h_lin - h_sat) < lagrange_bound
    assert compute_0d_activation_enthalpy(1.0e12, params_sat) == pytest.approx(
        e_a + v_0 * p_decay, rel=1e-12
    )
    # Large finite decay pressure (P / P_decay = 4.1202e-17) retains full linear V_0 * P accuracy via expm1.
    params_large_pdec = _make_base_params(
        activation_energy=e_a,
        activation_volume=v_0,
        activation_volume_decay_p=1.0e26,
    )
    assert compute_0d_activation_enthalpy(p_i, params_large_pdec) == pytest.approx(
        e_a + v_0 * p_i, rel=1e-15
    )

    # Discrimination guard: wrong secant law E_a + V_0 * P * exp(-P / P_decay) differs from integral law.
    h_wrong_secant = e_a + v_0 * p_i * math.exp(-p_i / p_decay)
    assert abs(h_sat - h_wrong_secant) > 2.0e3

    # 3. Softmin2 branches: d < -40 (unclipped), d == 0 (midpoint), and d > 40 (saturated ceiling).
    assert softmin2_scalar(18.0, 40.0, 0.05) == 18.0
    assert softmin2_scalar(40.0, 18.0, 0.05) == 18.0
    assert softmin2_scalar(25.0, 25.0, 0.05) == pytest.approx(
        25.0 - 0.05 * math.log(2.0), rel=1e-14
    )

    # 4. Viscosity at (T_ref, P=0) equals viscosity_solid * water_prefactor.
    eta_ref, log10_eta_ref = compute_0d_interior_viscosity(params.arrhenius_t_ref, 0.0, params)
    assert eta_ref == pytest.approx(params.viscosity_solid * params.water_prefactor, rel=1e-12)
    assert log10_eta_ref == pytest.approx(18.0, rel=1e-12)

    params_wet = _make_base_params(water_prefactor=0.25)
    eta_wet, _ = compute_0d_interior_viscosity(params_wet.arrhenius_t_ref, 0.0, params_wet)
    assert eta_wet == pytest.approx(0.25 * params.viscosity_solid, rel=1e-12)

    # At T_mid where log10_raw(T_mid, 0) == viscosity_max_log10 == 18.0, softmin2 is at d == 0.
    params_mid_ceil = _make_base_params(viscosity_solid=1.0e17, viscosity_max_log10=18.0)
    t_mid_ceil = 1.0 / (
        1.0 / params.arrhenius_t_ref + math.log(10.0) / (params.g_homologous * params.t_sol_0)
    )
    _, log10_at_ceil = compute_0d_interior_viscosity(t_mid_ceil, 0.0, params_mid_ceil)
    assert log10_at_ceil == pytest.approx(18.0 - 0.05 * math.log(2.0), rel=1e-12)
    params_wide_ceil = _make_base_params(
        viscosity_solid=1.0e17, viscosity_max_log10=18.0, viscosity_ceiling_width=0.10
    )
    _, log10_at_wide = compute_0d_interior_viscosity(t_mid_ceil, 0.0, params_wide_ceil)
    assert log10_at_wide == pytest.approx(18.0 - 0.10 * math.log(2.0), rel=1e-12)

    # Cold temperature saturates onto viscosity_max_log10.
    params_low_ceil = _make_base_params(viscosity_max_log10=24.0)
    eta_cold, log10_cold = compute_0d_interior_viscosity(600.0, p_i, params_low_ceil)
    assert log10_cold == pytest.approx(24.0, rel=1e-14)
    assert eta_cold == pytest.approx(1.0e24, rel=1e-13)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_stagnant_lid_0d_reference_states_and_scalings() -> None:
    """Verify 0-D stagnant-lid reference states, power-law exponents, and lid-base modes.

    Pins the algebraic 0-D stagnant-lid states at ``T_i = 1500, 1600, 1800, 2000 K``
    from Foley & Bercovici (2014, Eqs. 17-18, 26-28), Solomatov & Moresi (2000,
    Eqs. 8-12), and Thiriet et al. (2019, Eqs. 1, 6, 15, 18), and checks exact
    power-law exponents, closure prefactor sensitivities, and mode branches.
    """
    params = _make_base_params()

    pinned = {
        1500.0: {
            'eta_i_Pa_s': 2.234077163005132e19,
            'dT_rh_K': 60.04464118923616,
            'theta': 19.985130666666667,
            'T_lid_iso_K': 1367.9017893836804,
            'Ra_i': 23237568.003321808,
            'Nu_unyielded': 3.132960597491228,
            'delta_l_unyielded_m': 319186.90608517936,
            'd_lid_unyielded_m': 284050.2234635032,
            'delta_rh_m': 20762.585185535932,
            'v_m_m_s': 1.5679735652105906e-9,
            'tau_d_lid_Pa': 246644.68779640345,
            'F_top_spherical_W_m2': 0.01436773415197379,
        },
        1600.0: {
            'eta_i_Pa_s': 4.688323238079306e18,
            'dT_rh_K': 68.31745841975314,
            'theta': 19.0288109375,
            'T_lid_iso_K': 1449.7015914765432,
            'Ra_i': 119959156.27831261,
            'Nu_unyielded': 5.780467499936093,
            'delta_l_unyielded_m': 172996.38826981653,
            'd_lid_unyielded_m': 152995.55608730926,
            'delta_rh_m': 11818.673562390659,
            'v_m_m_s': 4.8390946459204725e-9,
            'tau_d_lid_Pa': 296573.8412269606,
            'F_top_spherical_W_m2': 0.029336596693294092,
        },
        1800.0: {
            'eta_i_Pa_s': 3.4744268752828806e17,
            'dT_rh_K': 86.46428331250007,
            'theta': 17.348203703703703,
            'T_lid_iso_K': 1609.7785767125,
            'Ra_i': 1867736819.0319602,
            'Nu_unyielded': 16.32800935392247,
            'delta_l_unyielded_m': 61244.45291059135,
            'd_lid_unyielded_m': 53477.78157651337,
            'delta_rh_m': 4589.396697409715,
            'v_m_m_s': 3.2091527821813407e-8,
            'tau_d_lid_Pa': 416994.3608205534,
            'F_top_spherical_W_m2': 0.09714571829480309,
        },
        2000.0: {
            'eta_i_Pa_s': 4.3328701992612024e16,
            'dT_rh_K': 106.74602878086428,
            'theta': 15.925651,
            'T_lid_iso_K': 1765.1587366820986,
            'Ra_i': 16973868733.141434,
            'Nu_unyielded': 38.19132435393112,
            'delta_l_unyielded_m': 26183.95713991698,
            'd_lid_unyielded_m': 22566.855037917045,
            'delta_rh_m': 2137.378514818143,
            'v_m_m_s': 1.4795808857500208e-7,
            'tau_d_lid_Pa': 568163.5226965582,
            'F_top_spherical_W_m2': 0.2587811131335361,
        },
    }

    for t_i, expected in pinned.items():
        st_0d = evaluate_stagnant_lid_0d(t_i, params)
        for key, val in expected.items():
            assert st_0d[key] == pytest.approx(val, rel=1e-12), (
                f'Mismatch at T_i={t_i}, key={key}'
            )

    # Exact power-law scalings under a 8x reduction of viscosity_solid (Ra_i -> 8 * Ra_i at fixed theta).
    st_base = evaluate_stagnant_lid_0d(1600.0, params)
    assert st_base['T_i_K'] == pytest.approx(1600.0, rel=1e-14)
    assert st_base['T_surf_K'] == pytest.approx(300.0, rel=1e-14)
    assert st_base['H_0_J_mol'] == pytest.approx(291006.19163, rel=1e-12)
    assert st_base['H_Pi_J_mol'] == pytest.approx(311560.5409572102, rel=1e-12)
    assert st_base['log10_eta_i'] == pytest.approx(18.67101754662667, rel=1e-12)
    assert st_base['Ra_rh'] == pytest.approx(8195304.671112291, rel=1e-12)
    assert st_base['tau_d_Pa'] == pytest.approx(45374.4797594682, rel=1e-12)
    assert st_base['P_lid_Pa'] == pytest.approx(6303722901.909316, rel=1e-12)
    assert st_base['F_top_planar_W_m2'] == pytest.approx(0.030058430999667686, rel=1e-12)
    st_8x = evaluate_stagnant_lid_0d(1600.0, _make_base_params(viscosity_solid=1.0e18 / 8.0))
    assert st_8x['Ra_i'] / st_base['Ra_i'] == pytest.approx(8.0, rel=1e-12)
    assert st_8x['Nu_unyielded'] / st_base['Nu_unyielded'] == pytest.approx(2.0, rel=1e-12)
    assert st_8x['delta_l_unyielded_m'] / st_base['delta_l_unyielded_m'] == pytest.approx(
        0.5, rel=1e-12
    )
    assert st_8x['v_m_m_s'] / st_base['v_m_m_s'] == pytest.approx(4.0, rel=1e-12)

    # Discrimination guard against wrong Ra^(1/4) Nusselt exponent (8^(1/4) = 1.68179 != 2.0).
    assert abs(st_8x['Nu_unyielded'] / st_base['Nu_unyielded'] - 8.0**0.25) > 0.3

    # Conductive saturation branch: at viscosity_solid = 1e21 Pa s, Nu < 1 and delta_l_unyielded == d.
    st_stiff = evaluate_stagnant_lid_0d(1600.0, _make_base_params(viscosity_solid=1.0e21))
    assert st_stiff['Nu_unyielded'] == pytest.approx(0.5780467499936093, rel=1e-12)
    assert st_stiff['delta_l_unyielded_m'] == pytest.approx(1.0e6, rel=1e-12)

    # Incipient-lid branch: when theta < a_lid, lid_frac == 0, d_lid == 0, and d_geom == delta_rh.
    st_incipient = evaluate_stagnant_lid_0d(1600.0, _make_base_params(t_surf=1500.0))
    assert st_incipient['theta'] < params.lid_contrast_coeff
    assert st_incipient['lid_frac'] == 0.0
    assert st_incipient['d_lid_unyielded_m'] == 0.0
    expected_tau_d_lid_inc = (
        2.0 * st_incipient['eta_i_Pa_s'] * st_incipient['v_m_m_s'] / st_incipient['delta_rh_m']
    )
    assert st_incipient['tau_d_lid_Pa'] == pytest.approx(expected_tau_d_lid_inc, rel=1e-12)

    # Fixed lid-base isotherm mode: T_lid_iso = T_base and lid_frac = (T_base - T_surf) / (T_i - T_surf).
    params_fixed = _make_base_params(lid_base_mode='fixed', lid_base_temperature=1400.0)
    st_fixed = evaluate_stagnant_lid_0d(1600.0, params_fixed, p_i=2.0e9, p_lid=1.0e9)
    assert st_fixed['P_i_Pa'] == pytest.approx(2.0e9, rel=1e-12)
    assert st_fixed['P_lid_Pa'] == pytest.approx(1.0e9, rel=1e-12)
    assert st_fixed['T_lid_iso_K'] == pytest.approx(1400.0, rel=1e-12)
    assert st_fixed['lid_frac'] == pytest.approx((1400.0 - 300.0) / (1600.0 - 300.0), rel=1e-12)

    # Fixed mode when T_i <= T_base clamps lid_frac at 1.0.
    st_fixed_cold = evaluate_stagnant_lid_0d(1350.0, params_fixed)
    assert st_fixed_cold['lid_frac'] == pytest.approx(1.0, rel=1e-12)

    # Closure prefactor sensitivity checks (a_nu, a_rh_fb, c4_fb).
    st_2anu = evaluate_stagnant_lid_0d(1600.0, _make_base_params(a_nu=2.0 * params.a_nu))
    assert st_2anu['Nu_unyielded'] / st_base['Nu_unyielded'] == pytest.approx(2.0, rel=1e-12)
    assert st_2anu['delta_l_unyielded_m'] / st_base['delta_l_unyielded_m'] == pytest.approx(
        0.5, rel=1e-12
    )
    st_2arh = evaluate_stagnant_lid_0d(1600.0, _make_base_params(a_rh_fb=2.0 * params.a_rh_fb))
    assert st_2arh['Ra_rh'] / st_base['Ra_rh'] == pytest.approx(2.0, rel=1e-12)
    assert st_2arh['delta_rh_m'] / st_base['delta_rh_m'] == pytest.approx(2.0, rel=1e-12)
    assert st_2arh['v_m_m_s'] / st_base['v_m_m_s'] == pytest.approx(
        2.0 ** (2.0 / 3.0), rel=1e-12
    )
    st_2c4 = evaluate_stagnant_lid_0d(1600.0, _make_base_params(c4_fb=2.0 * params.c4_fb))
    assert st_2c4['v_m_m_s'] / st_base['v_m_m_s'] == pytest.approx(2.0, rel=1e-12)
    assert st_2c4['tau_d_Pa'] / st_base['tau_d_Pa'] == pytest.approx(2.0, rel=1e-12)
    assert st_2c4['tau_d_lid_Pa'] / st_base['tau_d_lid_Pa'] == pytest.approx(2.0, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_stagnant_lid_0d_yield_ceiling_thinning_limit() -> None:
    """Verify Moresi & Solomatov (1998) ``(tau_y / tau_d_lid)**(1/3)`` yield-ceiling lid thinning.

    Checks the unyielded plateau (``tau_y_lid >= 2 * tau_d_lid``), the transition
    midpoint (``tau_y_lid == tau_d_lid`` -> ``w_b == 0.5``), the analytic ``0.5**(1/3)``
    lid thinning at ``tau_y_lid = 0.5 * tau_d_lid`` (Moresi & Solomatov 1998 Eqs. 13-15,
    pp. 672, 676), Byerlee cohesion+friction vs cap regimes, and the zero-yield floor.
    """
    params = _make_base_params()
    st_unyielded_default = evaluate_stagnant_lid_0d(1600.0, params)
    tau_d_lid = st_unyielded_default['tau_d_lid_Pa']
    assert tau_d_lid == pytest.approx(296573.8412269606, rel=1e-12)
    assert st_unyielded_default['w_b_yield'] == 0.0
    assert st_unyielded_default['thin_factor'] == 1.0

    # Unyielded at tau_y = 2 * tau_d_lid vs yield-thinned at tau_y = 0.5 * tau_d_lid.
    st_high = evaluate_stagnant_lid_0d(
        1600.0, _make_base_params(yield_stress_max=2.0 * tau_d_lid)
    )
    st_half = evaluate_stagnant_lid_0d(
        1600.0, _make_base_params(yield_stress_max=0.5 * tau_d_lid)
    )
    assert st_high['w_b_yield'] == pytest.approx(0.0, abs=1e-12)
    assert st_high['thin_factor'] == pytest.approx(1.0, rel=1e-12)
    assert st_half['w_b_yield'] == pytest.approx(1.0, rel=1e-12)
    assert st_half['thin_factor'] == pytest.approx(0.5 ** (1.0 / 3.0), rel=1e-12)

    ratio_dlid = st_half['d_lid_eff_m'] / st_high['d_lid_eff_m']
    ratio_deltal = st_half['delta_l_eff_m'] / st_high['delta_l_eff_m']
    assert ratio_dlid == pytest.approx(0.5 ** (1.0 / 3.0), rel=1e-12)
    assert ratio_deltal == pytest.approx(0.5 ** (1.0 / 3.0), rel=1e-12)
    assert st_half['F_top_planar_W_m2'] / st_high['F_top_planar_W_m2'] == pytest.approx(
        2.0 ** (1.0 / 3.0), rel=1e-12
    )
    assert st_half['F_top_spherical_W_m2'] > st_high['F_top_spherical_W_m2']

    # Discrimination guards: linear ratio 0.5 and square-root ratio 0.5**0.5 fall outside tolerance.
    assert abs(ratio_dlid - 0.5) > 0.28
    assert abs(ratio_dlid - math.sqrt(0.5)) > 0.08

    # Transition midpoint: tau_y_lid == tau_d_lid gives stress_ratio == 1 and w_b == 0.5.
    st_mid = evaluate_stagnant_lid_0d(1600.0, _make_base_params(yield_stress_max=tau_d_lid))
    assert st_mid['stress_ratio'] == pytest.approx(1.0, rel=1e-12)
    assert st_mid['w_b_yield'] == pytest.approx(0.5, rel=1e-12)
    assert st_mid['thin_factor'] == pytest.approx(1.0, rel=1e-12)

    # Yield transition width sensitivity at stress_ratio == 1.025.
    tau_y_1025 = tau_d_lid / 1.025
    st_w025 = evaluate_stagnant_lid_0d(
        1600.0, _make_base_params(yield_stress_max=tau_y_1025, yield_blend_ratio_width=0.025)
    )
    st_w050 = evaluate_stagnant_lid_0d(
        1600.0, _make_base_params(yield_stress_max=tau_y_1025, yield_blend_ratio_width=0.050)
    )
    assert st_w025['w_b_yield'] == pytest.approx(0.5 * (1.0 + math.tanh(1.0)), rel=1e-12)
    assert st_w050['w_b_yield'] == pytest.approx(0.5 * (1.0 + math.tanh(0.5)), rel=1e-12)

    # Power-law scaling across tau_y / tau_d_lid in {0.1, 0.2}:
    for frac in (0.1, 0.2):
        st_f = evaluate_stagnant_lid_0d(
            1600.0, _make_base_params(yield_stress_max=frac * tau_d_lid)
        )
        assert st_f['d_lid_eff_m'] / st_high['d_lid_eff_m'] == pytest.approx(
            frac ** (1.0 / 3.0), rel=1e-12
        )

    # Byerlee linear cohesion + friction branch (when c + mu * P_lid < tau_max).
    params_byerlee = _make_base_params(
        yield_stress_c=1.0e5,
        yield_stress_mu=1.0e-5,
        yield_stress_max=1.0e9,
    )
    st_byerlee = evaluate_stagnant_lid_0d(1600.0, params_byerlee, p_lid=2.0e7)
    assert st_byerlee['tau_y_lid_Pa'] == pytest.approx(1.0e5 + 1.0e-5 * 2.0e7, rel=1e-12)

    # Zero Byerlee cohesion and friction exercises yield_stress_floor in stress_ratio and thin_factor.
    params_zero_yield = _make_base_params(
        yield_stress_c=0.0,
        yield_stress_mu=0.0,
        yield_stress_floor=10.0,
    )
    st_zero_y = evaluate_stagnant_lid_0d(1600.0, params_zero_yield)
    expected_zero_thin = (10.0 / tau_d_lid) ** (1.0 / 3.0)
    assert st_zero_y['tau_y_lid_Pa'] == 0.0
    assert st_zero_y['stress_ratio'] == pytest.approx(tau_d_lid / 10.0, rel=1e-12)
    assert st_zero_y['thin_factor'] == pytest.approx(expected_zero_thin, rel=1e-12)
    assert st_zero_y['d_lid_eff_m'] == pytest.approx(
        st_zero_y['d_lid_unyielded_m'] * expected_zero_thin, rel=1e-12
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_stagnant_lid_0d_ode_steady_state_limit() -> None:
    """Verify the analytic steady-state equilibrium and relaxation stability of the 0-D ODE.

    When basal and internal heat sources balance surface heat loss at ``T_i_eq = 1600 K``
    (``A_CMB * F_bot + M_m * H_rad == A_surf * F_top(T_i_eq)``, Thiriet et al. 2019
    Eq. 12, p. 140), starting at ``T_i_eq`` holds ``T_i(t)`` and ``d_lid(t)`` invariant,
    while perturbations above or below ``T_i_eq`` relax monotonically toward ``T_i_eq``.
    """
    params = _make_base_params()
    t_eq = 1600.0
    st_eq = evaluate_stagnant_lid_0d(t_eq, params)
    a_surf = 4.0 * math.pi * (params.r_surf**2)
    a_cmb = 4.0 * math.pi * (params.r_cmb**2)
    v_mantle = (4.0 / 3.0) * math.pi * (params.r_surf**3 - params.r_cmb**3)
    m_mantle = params.rho * v_mantle

    q_out_eq = a_surf * st_eq['F_top_spherical_W_m2']
    f_bot_eq = 0.4 * q_out_eq / a_cmb
    h_rad_eq = 0.6 * q_out_eq / m_mantle

    # Fast 1-second diagnostic and rate check before long-timescale integration.
    sol_1s = integrate_stagnant_lid_0d(
        1680.0,
        [0.0, 1.0],
        params,
        f_bot=f_bot_eq,
        h_rad=h_rad_eq,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
    )
    assert sol_1s['A_surf_m2'] == pytest.approx(a_surf, rel=1e-12)
    assert sol_1s['A_cmb_m2'] == pytest.approx(a_cmb, rel=1e-12)
    assert sol_1s['M_mantle_kg'] == pytest.approx(m_mantle, rel=1e-12)
    assert sol_1s['C_eff_J_K'] == pytest.approx(m_mantle * params.c_p, rel=1e-12)
    assert sol_1s['Q_in_W'][0] == pytest.approx(q_out_eq, rel=1e-12)
    st_1680 = evaluate_stagnant_lid_0d(1680.0, params)
    expected_dt_dt = (q_out_eq - a_surf * st_1680['F_top_spherical_W_m2']) / (
        m_mantle * params.c_p
    )
    assert sol_1s['dT_i_dt_K_s'][0] == pytest.approx(expected_dt_dt, rel=1e-12)
    assert (sol_1s['T_i_K'][1] - 1680.0) == pytest.approx(expected_dt_dt, rel=1e-6)

    sol_1s_ad = integrate_stagnant_lid_0d(
        1680.0,
        [0.0, 1.0],
        params,
        f_bot=f_bot_eq,
        h_rad=h_rad_eq,
        conduction_geometry='spherical',
        heat_capacity_mode='adiabatic',
    )
    assert sol_1s_ad['C_eff_J_K'] == pytest.approx(1.9107356339947686e27, rel=1e-12)

    sec_per_yr = 365.25 * 86400.0
    t_eval_s = np.linspace(0.0, 1.0e9 * sec_per_yr, 9)

    sol_eq = integrate_stagnant_lid_0d(
        t_eq,
        t_eval_s,
        params,
        f_bot=f_bot_eq,
        h_rad=h_rad_eq,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
    )
    np.testing.assert_allclose(sol_eq['T_i_K'], t_eq, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        sol_eq['d_lid_eff_m'], st_eq['d_lid_eff_m'], rtol=1e-12, atol=1e-6
    )
    np.testing.assert_allclose(sol_eq['dT_i_dt_K_s'], 0.0, atol=1e-22)

    # Hot perturbation (1680 K) cools monotonically toward T_eq.
    sol_hot = integrate_stagnant_lid_0d(
        1680.0,
        t_eval_s,
        params,
        f_bot=f_bot_eq,
        h_rad=h_rad_eq,
    )
    assert np.all(np.diff(sol_hot['T_i_K']) < 0.0)
    assert t_eq < sol_hot['T_i_K'][-1] < 1640.0

    # Cold perturbation (1530 K) warms monotonically toward T_eq.
    sol_cold = integrate_stagnant_lid_0d(
        1530.0,
        t_eval_s,
        params,
        f_bot=f_bot_eq,
        h_rad=h_rad_eq,
    )
    assert np.all(np.diff(sol_cold['T_i_K']) > 0.0)
    assert 1545.0 < sol_cold['T_i_K'][-1] < t_eq

    # Discrimination guard: without heat sources, 1600 K cools by > 110 K over 1 Gyr.
    sol_unheated = integrate_stagnant_lid_0d(t_eq, t_eval_s, params)
    assert (t_eq - sol_unheated['T_i_K'][-1]) > 110.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_stagnant_lid_0d_ode_energy_balance() -> None:
    """Verify thermal energy balance of the 0-D cooling ODE.

    Checks trajectory gradient consistency against dT_i/dt and integral
    (C_eff * (T_i(t_end) - T_i(0)) == integral(Q_in - Q_out, dt)) energy
    conservation across spherical and planar conduction geometries and isothermal
    and adiabatic heat capacities, and pins the 1 Gyr cooling endpoints (Thiriet
    et al. 2019 Eqs. 10-12, 18).
    """
    params = _make_base_params()
    sec_per_yr = 365.25 * 86400.0
    t_eval_s = (np.linspace(0.0, 1.0, 401) ** 2) * (1.0e9 * sec_per_yr)

    # 1. Spherical isothermal secular cooling from 1800 K (1499.995025 K).
    sol_sph_iso = integrate_stagnant_lid_0d(
        1800.0,
        t_eval_s,
        params,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
    )
    assert sol_sph_iso['T_i_K'][-1] == pytest.approx(1499.9950247662127, rel=1e-9)
    assert sol_sph_iso['d_lid_eff_m'][-1] == pytest.approx(284059.4911176443, rel=1e-9)

    # 2. Planar isothermal secular cooling from 1800 K (1496.551907 K).
    sol_pla_iso = integrate_stagnant_lid_0d(
        1800.0,
        t_eval_s,
        params,
        conduction_geometry='planar',
        heat_capacity_mode='isothermal',
    )
    assert sol_pla_iso['T_i_K'][-1] == pytest.approx(1496.5519067125895, rel=1e-9)
    assert sol_pla_iso['d_lid_eff_m'][-1] == pytest.approx(290559.8713038643, rel=1e-9)
    assert sol_pla_iso['T_i_K'][-1] < sol_sph_iso['T_i_K'][-1]

    # 3. Spherical adiabatic-heat-capacity secular cooling from 1800 K (1505.908642 K).
    sol_sph_ad = integrate_stagnant_lid_0d(
        1800.0,
        t_eval_s,
        params,
        conduction_geometry='spherical',
        heat_capacity_mode='adiabatic',
    )
    assert sol_sph_ad['T_i_K'][-1] == pytest.approx(1505.9086416016703, rel=1e-9)
    assert sol_sph_ad['d_lid_eff_m'][-1] == pytest.approx(273292.02730344643, rel=1e-9)
    assert sol_sph_ad['C_eff_J_K'] > sol_sph_iso['C_eff_J_K']
    assert sol_sph_ad['T_i_K'][-1] > sol_sph_iso['T_i_K'][-1]

    # 4. Mixed basal + radiogenic heating energy conservation check.
    sol_mixed = integrate_stagnant_lid_0d(
        1650.0,
        t_eval_s,
        params,
        f_bot=0.012,
        h_rad=4.0e-12,
        conduction_geometry='spherical',
        heat_capacity_mode='isothermal',
    )

    for sol in (sol_sph_iso, sol_pla_iso, sol_sph_ad, sol_mixed):
        assert sol['t_s'].shape == sol['T_i_K'].shape == t_eval_s.shape
        c_eff = sol['C_eff_J_K']
        q_net = sol['Q_in_W'] - sol['Q_out_W']
        # Energy balance against trajectory gradient and integral heat content change.
        dt_dt_num = np.gradient(sol['T_i_K'], sol['t_s'], edge_order=2)
        np.testing.assert_allclose(
            dt_dt_num[40:-1],
            sol['dT_i_dt_K_s'][40:-1],
            rtol=1e-4,
            atol=1e-19,
        )
        # Integral energy balance over [0, 1 Gyr] via trapezoidal rule on 401 nodes.
        delta_e_stored = c_eff * (sol['T_i_K'][-1] - sol['T_i_K'][0])
        delta_e_integrated = float(np.trapezoid(q_net, sol['t_s']))
        assert delta_e_stored == pytest.approx(delta_e_integrated, rel=1e-5)

    # Discrimination guard: using A_surf instead of A_CMB on f_bot breaks Q_in by (1 - f^2) * A_surf * f_bot.
    wrong_q_in = sol_mixed['A_surf_m2'] * 0.012 + sol_mixed['M_mantle_kg'] * 4.0e-12
    assert abs(sol_mixed['Q_in_W'][0] - wrong_q_in) / sol_mixed['Q_in_W'][0] > 0.08


@pytest.mark.physics_invariant
@settings(max_examples=40, deadline=None)
@given(
    t_i=st.floats(min_value=1200.0, max_value=2400.0, allow_nan=False, allow_infinity=False),
    dt_step=st.floats(min_value=25.0, max_value=150.0, allow_nan=False, allow_infinity=False),
    log10_eta_solid=st.floats(
        min_value=16.0, max_value=22.0, allow_nan=False, allow_infinity=False
    ),
    tau_max_mpa=st.floats(
        min_value=0.05, max_value=800.0, allow_nan=False, allow_infinity=False
    ),
    a_lid=st.floats(min_value=1.0, max_value=3.5, allow_nan=False, allow_infinity=False),
    c_byerlee_pa=st.floats(
        min_value=1.0e4, max_value=5.0e7, allow_nan=False, allow_infinity=False
    ),
)
def test_stagnant_lid_0d_hypothesis_properties_and_monotonicity(
    t_i: float,
    dt_step: float,
    log10_eta_solid: float,
    tau_max_mpa: float,
    a_lid: float,
    c_byerlee_pa: float,
) -> None:
    """Verify physical bounds and monotonicities of the 0-D stagnant-lid model via Hypothesis."""
    eta_solid = 10.0**log10_eta_solid
    tau_max = tau_max_mpa * 1.0e6
    params = _make_base_params(
        viscosity_solid=eta_solid,
        viscosity_max_log10=45.0,
        yield_stress_max=tau_max,
        lid_contrast_coeff=a_lid,
    )

    st_1 = evaluate_stagnant_lid_0d(t_i, params)
    st_2 = evaluate_stagnant_lid_0d(t_i + dt_step, params)
    d_mantle = params.r_surf - params.r_cmb

    # 1. Positivity and geometric ordering bounds.
    assert st_1['eta_i_Pa_s'] > 0.0
    assert st_1['Ra_i'] > 0.0
    assert st_1['Ra_rh'] > 0.0
    assert st_1['Nu_unyielded'] > 0.0
    assert 0.0 < st_1['delta_l_unyielded_m'] <= d_mantle
    assert 0.0 <= st_1['lid_frac'] <= 1.0
    assert 0.0 <= st_1['d_lid_unyielded_m'] <= st_1['delta_l_unyielded_m']
    assert 0.0 <= st_1['w_b_yield'] <= 1.0
    assert 0.0 < st_1['thin_factor'] <= 1.0
    assert 0.0 <= st_1['d_lid_eff_m'] <= st_1['d_lid_unyielded_m'] + 1.0e-9
    assert 0.0 < st_1['F_top_spherical_W_m2'] <= st_1['F_top_planar_W_m2']

    # 2. Temperature monotonicity: hotter interior has lower viscosity, higher Ra_i, Nu, and v_m.
    assert st_2['eta_i_Pa_s'] < st_1['eta_i_Pa_s']
    assert st_2['Ra_i'] > st_1['Ra_i']
    assert st_2['Nu_unyielded'] > st_1['Nu_unyielded']
    assert st_2['v_m_m_s'] > st_1['v_m_m_s']
    if st_1['Nu_unyielded'] > 1.0:
        assert st_2['delta_l_unyielded_m'] < st_1['delta_l_unyielded_m']
        assert st_2['F_top_planar_W_m2'] > st_1['F_top_planar_W_m2']
        assert st_2['F_top_spherical_W_m2'] > st_1['F_top_spherical_W_m2']

    # 3. Yield-ceiling and Byerlee frictional monotonicity: stronger yield envelope never thins lid further.
    params_stronger = _make_base_params(
        viscosity_solid=eta_solid,
        viscosity_max_log10=45.0,
        yield_stress_max=2.0 * tau_max,
        lid_contrast_coeff=a_lid,
    )
    st_stronger = evaluate_stagnant_lid_0d(t_i, params_stronger)
    assert st_stronger['d_lid_eff_m'] >= st_1['d_lid_eff_m'] - 1.0e-9

    params_byerlee_1 = _make_base_params(
        viscosity_solid=eta_solid,
        viscosity_max_log10=45.0,
        yield_stress_c=c_byerlee_pa,
        yield_stress_mu=1.0e-5,
        yield_stress_max=1.0e11,
        lid_contrast_coeff=a_lid,
    )
    params_byerlee_2 = _make_base_params(
        viscosity_solid=eta_solid,
        viscosity_max_log10=45.0,
        yield_stress_c=2.0 * c_byerlee_pa,
        yield_stress_mu=2.0e-5,
        yield_stress_max=1.0e11,
        lid_contrast_coeff=a_lid,
    )
    st_b1 = evaluate_stagnant_lid_0d(t_i, params_byerlee_1)
    st_b2 = evaluate_stagnant_lid_0d(t_i, params_byerlee_2)
    assert st_b2['tau_y_lid_Pa'] > st_b1['tau_y_lid_Pa']
    assert st_b2['d_lid_eff_m'] >= st_b1['d_lid_eff_m'] - 1.0e-9


_POS_PARAM_FIELDS = (
    'r_cmb',
    'rho',
    'c_p',
    'alpha',
    'k_cond',
    'gravity',
    't_surf',
    'viscosity_solid',
    'r_gas',
    'g_homologous',
    't_sol_0',
    'arrhenius_t_ref',
    'viscosity_ceiling_width',
    'water_prefactor',
    'lid_contrast_coeff',
    'lid_base_temperature',
    'a_rh_fb',
    'c4_fb',
    'a_nu',
    'yield_stress_max',
    'yield_stress_floor',
    'yield_blend_ratio_width',
)
_NONNEG_PARAM_FIELDS = (
    'p_i_ref',
    'dt_sol_dp',
    'activation_volume',
    'yield_stress_c',
    'yield_stress_mu',
)


@settings(max_examples=25, deadline=None)
@given(
    bad_pos=st.one_of(
        st.just(float('nan')),
        st.just(float('inf')),
        st.just(float('-inf')),
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    ),
    bad_nonneg=st.one_of(
        st.just(float('nan')),
        st.just(float('inf')),
        st.just(float('-inf')),
        st.floats(max_value=-1.0e-12, allow_nan=False, allow_infinity=False),
    ),
)
def _sweep_invalid_floats(bad_pos: float, bad_nonneg: float) -> None:
    """Property check that positive and non-negative parameter fields reject invalid floats."""
    for pos_field in _POS_PARAM_FIELDS:
        _assert_raises_value_error(
            _make_base_params,
            match=pos_field,
            **{pos_field: bad_pos},
        )
    for nonneg_field in _NONNEG_PARAM_FIELDS:
        _assert_raises_value_error(
            _make_base_params,
            match=nonneg_field,
            **{nonneg_field: bad_nonneg},
        )


@pytest.mark.physics_invariant
def test_stagnant_lid_0d_invalid_inputs_rejected() -> None:
    """Verify that invalid parameters, booleans, non-finite values, and domain errors raise ValueError."""
    params = _make_base_params()

    # Boolean values are rejected even though isinstance(True, int) is True in Python.
    for b_field in (*_POS_PARAM_FIELDS, *_NONNEG_PARAM_FIELDS):
        _assert_raises_value_error(_make_base_params, match=b_field, **{b_field: False})
    _assert_raises_value_error(compute_0d_activation_enthalpy, True, params, match='pressure')
    _assert_raises_value_error(compute_0d_interior_viscosity, True, 0.0, params, match='t_i')

    # Domain ordering and mode checks on StagnantLid0DParams.
    _assert_raises_value_error(_make_base_params, match='r_surf', r_surf=params.r_cmb)
    _assert_raises_value_error(_make_base_params, match='p_i_ref', p_i_ref=-1.0)
    _assert_raises_value_error(_make_base_params, match='dt_sol_dp', dt_sol_dp=-1.0e-9)
    _assert_raises_value_error(
        _make_base_params, match='viscosity_max_log10', viscosity_max_log10=18.0
    )
    _assert_raises_value_error(
        _make_base_params, match='viscosity_max_log10', viscosity_max_log10=300.1
    )
    _assert_raises_value_error(
        _make_base_params, match='activation_energy', activation_energy=0.0
    )
    _assert_raises_value_error(
        _make_base_params, match='activation_volume', activation_volume=-1.0e-6
    )
    _assert_raises_value_error(
        _make_base_params, match='activation_volume_decay_p', activation_volume_decay_p=0.0
    )
    _assert_raises_value_error(
        _make_base_params, match='lid_base_mode', lid_base_mode='invalid'
    )
    _assert_raises_value_error(
        _make_base_params,
        match='lid_base_temperature',
        lid_base_mode='fixed',
        lid_base_temperature=params.t_surf,
    )
    _assert_raises_value_error(_make_base_params, match='yield_stress_c', yield_stress_c=-1.0)
    _assert_raises_value_error(_make_base_params, match='yield_stress_mu', yield_stress_mu=-0.1)

    # Runtime argument validation on helper functions.
    _assert_raises_value_error(softmin2_scalar, 1.0, 2.0, 0.0, match='width')
    _assert_raises_value_error(compute_0d_activation_enthalpy, -1.0, params, match='pressure')
    _assert_raises_value_error(
        compute_0d_interior_viscosity, params.t_surf, 0.0, params, match='t_i'
    )
    _assert_raises_value_error(compute_0d_interior_viscosity, 1600.0, -1.0, params, match='p_i')
    _assert_raises_value_error(evaluate_stagnant_lid_0d, params.t_surf, params, match='t_i')
    _assert_raises_value_error(evaluate_stagnant_lid_0d, 1600.0, params, p_i=-1.0, match='p_i')
    _assert_raises_value_error(
        evaluate_stagnant_lid_0d, 1600.0, params, p_lid=-1.0, match='p_lid'
    )

    # ODE integrator input validation.
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, params.t_surf, [0.0, 1.0], params, match='t_i_init'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [0.0], params, match='t_eval_s'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [-1.0, 1.0], params, match='t_eval_s'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [0.0, 1.0, 1.0], params, match='strictly increasing'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [0.0, 1.0], params, f_bot=-0.1, match='f_bot'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [0.0, 1.0], params, h_rad=-1.0, match='h_rad'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d,
        1600.0,
        [0.0, 1.0],
        params,
        conduction_geometry='cylindrical',
        match='conduction_geometry',
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d,
        1600.0,
        [0.0, 1.0],
        params,
        heat_capacity_mode='unknown',
        match='heat_capacity_mode',
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [0.0, 1.0], params, rtol=0.0, match='rtol'
    )
    _assert_raises_value_error(
        integrate_stagnant_lid_0d, 1600.0, [0.0, 1.0], params, atol=0.0, match='atol'
    )

    # Boundary acceptance checks: non-negative fields accept 0.0 and strict bounds accept +eps / 300.0.
    p_zero = _make_base_params(
        r_surf=params.r_cmb + 1.0,
        p_i_ref=0.0,
        dt_sol_dp=0.0,
        viscosity_max_log10=math.log10(params.viscosity_solid) + 0.01,
        activation_energy=1.0,
        activation_volume=0.0,
        lid_base_mode='fixed',
        lid_base_temperature=params.t_surf + 1.0,
        yield_stress_c=0.0,
        yield_stress_mu=0.0,
    )
    assert p_zero.p_i_ref == 0.0
    p_max_ceil = _make_base_params(viscosity_max_log10=300.0)
    assert p_max_ceil.viscosity_max_log10 == 300.0
    compute_0d_activation_enthalpy(0.0, p_zero)
    compute_0d_interior_viscosity(params.t_surf + 0.1, 0.0, p_zero)
    evaluate_stagnant_lid_0d(params.t_surf + 0.1, p_zero, p_i=0.0, p_lid=0.0)
    integrate_stagnant_lid_0d(
        params.t_surf + 0.1,
        [0.0, 1.0],
        params,
        f_bot=0.0,
        h_rad=0.0,
    )

    # Hypothesis property sweep across positive and non-negative parameter fields.
    _sweep_invalid_floats()


def test_integrate_stagnant_lid_0d_raises_on_solver_failure(monkeypatch) -> None:
    """integrate_stagnant_lid_0d raises RuntimeError when solve_ivp fails."""
    from scipy.integrate._ivp.ivp import OdeResult

    import tests.reference.stagnant_lid_0d as ref_mod

    params = _make_base_params()
    t_eval_s = np.array([0.0, 1.0e6])

    orig_solve_ivp = ref_mod.solve_ivp

    def mock_solve_ivp(*args, **kwargs):
        res = orig_solve_ivp(*args, **kwargs)
        return OdeResult(
            t=res.t,
            y=res.y,
            status=-1,
            message='Step size below machine precision',
            success=False,
        )

    monkeypatch.setattr(ref_mod, 'solve_ivp', mock_solve_ivp)
    with pytest.raises(RuntimeError, match='solve_ivp integration failed'):
        integrate_stagnant_lid_0d(1800.0, t_eval_s, params)


def test_stagnant_lid_0d_constants_parity_with_solver_defaults() -> None:
    """Verify parity between stagnant_lid_0d explicit constants and aragog.rheology defaults."""
    import aragog.rheology as rheo_mod
    import tests.reference.stagnant_lid_0d as ref_mod
    from aragog.rheology import SolidRheologyParams

    assert ref_mod.R_GAS_DEFAULT == rheo_mod.R_GAS
    assert ref_mod.COHESION_FLOOR_DEFAULT == rheo_mod.COHESION_FLOOR
    assert (
        ref_mod.VISCOSITY_CEILING_LOG10_WIDTH_DEFAULT == rheo_mod.VISCOSITY_CEILING_LOG10_WIDTH
    )

    defaults = SolidRheologyParams()
    assert ref_mod.G_HOMOLOGOUS_DEFAULT == defaults.g_homologous
    assert ref_mod.T_SOL_0_DEFAULT == defaults.t_sol_0
    assert ref_mod.DT_SOL_DP_DEFAULT == defaults.dt_sol_dp
    assert ref_mod.ARRHENIUS_T_REF_DEFAULT == defaults.arrhenius_t_ref
    assert ref_mod.VISCOSITY_MAX_LOG10_DEFAULT == defaults.viscosity_max_log10
    assert ref_mod.WATER_PREFACTOR_DEFAULT == defaults.water_prefactor
    assert ref_mod.ACTIVATION_ENERGY_DEFAULT == defaults.activation_energy
    assert ref_mod.ACTIVATION_VOLUME_DEFAULT == defaults.activation_volume
    assert (
        ref_mod.ACTIVATION_VOLUME_DECAY_PRESSURE_DEFAULT
        == defaults.activation_volume_decay_pressure
    )
    assert ref_mod.YIELD_STRESS_C_DEFAULT == defaults.yield_stress_c
    assert ref_mod.YIELD_STRESS_MU_DEFAULT == defaults.yield_stress_mu
    assert ref_mod.YIELD_STRESS_MAX_DEFAULT == defaults.yield_stress_max
    assert ref_mod.LID_BASE_MODE_DEFAULT == defaults.lid_base_mode
    assert ref_mod.LID_CONTRAST_COEFF_DEFAULT == defaults.lid_contrast_coeff
    assert ref_mod.LID_BASE_TEMPERATURE_DEFAULT == defaults.lid_base_temperature
