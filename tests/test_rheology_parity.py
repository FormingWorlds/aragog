"""Parity tests between NumPy and JAX rheology kernel implementations.

Validates machine-precision agreement across physical regimes and ensures
autodiff reversibility without ConcretizationTypeError.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_viscosity,
    compute_effective_viscosity,
    compute_stagnant_lid_state,
    compute_yield_stress,
)

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
jnp = pytest.importorskip('jax.numpy')


def _lid_state(xp, **fields):
    """Evaluate compute_stagnant_lid_state with every array input converted to ``xp``."""
    conv = {k: xp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in fields.items()}
    conv.setdefault('phi_rheo', 0.4)
    conv.setdefault('phi_width', 0.15)
    return compute_stagnant_lid_state(xp=xp, **conv)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_arrhenius_viscosity_parity_float64():
    """Verify homologous Arrhenius viscosity matches analytical values and JAX parity to 1e-12."""
    t_vals = np.linspace(1400.0, 3000.0, 50)
    p_vals = np.linspace(0.0, 135.0e9, 50)
    t_sol_vals = 1400.0 + 2.4e-8 * p_vals - 5.0e-20 * p_vals**2

    eta_np = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )

    eta_jx = compute_arrhenius_viscosity(
        jnp.array(t_vals),
        jnp.array(p_vals),
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=jnp.array(t_sol_vals),
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=jnp,
    )

    np.testing.assert_allclose(np.array(eta_jx), eta_np, rtol=1.0e-12, atol=1.0e-30)

    # When solidus_temperature is given and t_sol_0 is None, raise ValueError:
    with pytest.raises(ValueError, match='t_sol_0'):
        compute_arrhenius_viscosity(
            t_vals,
            p_vals,
            viscosity_solid=1.0e21,
            g_homologous=25.0,
            solidus_temperature=t_sol_vals,
            t_sol_0=None,
            arrhenius_t_ref=1600.0,
            viscosity_max_log10=40.0,
            xp=np,
        )

    # Scalar, full array, and sliced array solidus_temperature give the same viscosity:
    eta_sliced = compute_arrhenius_viscosity(
        t_vals[10:20],
        p_vals[10:20],
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=t_sol_vals[10:20],
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    np.testing.assert_allclose(eta_sliced, eta_np[10:20], rtol=1.0e-12, atol=1.0e-30)

    eta_scalar_2000 = compute_arrhenius_viscosity(
        2000.0,
        10.0e9,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=2000.0,
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    eta_arr_2000 = compute_arrhenius_viscosity(
        np.full(5, 2000.0),
        np.linspace(0.0, 20.0e9, 5),
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=np.full(5, 2000.0),
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    np.testing.assert_allclose(eta_arr_2000, eta_scalar_2000, rtol=1.0e-12, atol=1.0e-30)

    # Exact hand-computed analytical verification:
    expected = np.minimum(
        1.0e21 * np.exp(25.0 * (t_sol_vals / t_vals - 1400.0 / 1600.0)),
        10.0**40.0,
    )
    np.testing.assert_allclose(eta_np, expected, rtol=1.0e-12, atol=1.0e-30)

    # Canary 1: negative g_homologous raises ValueError; wrong g_homologous (14.0) fails rtol=1.0e-12.
    with pytest.raises(ValueError, match='g_homologous'):
        compute_arrhenius_viscosity(
            t_vals,
            p_vals,
            viscosity_solid=1.0e21,
            g_homologous=-25.0,
            solidus_temperature=t_sol_vals,
            t_sol_0=1400.0,
            arrhenius_t_ref=1600.0,
            viscosity_max_log10=40.0,
            xp=np,
        )
    eta_wrong_g = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=14.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=1400.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eta_wrong_g, eta_np, rtol=1.0e-12, atol=1.0e-30)

    # Canary 2: t_sol_0 <= 0 raises ValueError on NumPy, and a wrong positive t_sol_0 fails parity.
    with pytest.raises(ValueError, match='t_sol_0'):
        compute_arrhenius_viscosity(
            t_vals,
            p_vals,
            viscosity_solid=1.0e21,
            g_homologous=25.0,
            solidus_temperature=t_sol_vals,
            t_sol_0=0.0,
            arrhenius_t_ref=1600.0,
            viscosity_max_log10=40.0,
            xp=np,
        )
    eta_wrong_t_sol_0 = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=t_sol_vals,
        t_sol_0=1000.0,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eta_wrong_t_sol_0, eta_np, rtol=1.0e-12, atol=1.0e-30)

    # Canary 3: ignoring solidus_temperature and falling back to linear solidus must fail.
    eta_linear_fallback = compute_arrhenius_viscosity(
        t_vals,
        p_vals,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        solidus_temperature=None,
        t_sol_0=1400.0,
        dt_sol_dp=2.4e-8,
        arrhenius_t_ref=1600.0,
        viscosity_max_log10=40.0,
        xp=np,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eta_linear_fallback, eta_np, rtol=1.0e-12, atol=1.0e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_yield_stress_parity_float64():
    """Verify yield stress evaluates identically in NumPy and JAX to 1e-12."""
    p_vals = np.linspace(0.0, 135.0e9, 50)

    tau_np = compute_yield_stress(
        p_vals,
        yield_stress_c=50.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
        xp=np,
    )

    tau_jx = compute_yield_stress(
        jnp.array(p_vals),
        yield_stress_c=50.0e6,
        yield_stress_mu=0.6,
        yield_stress_max=500.0e6,
        xp=jnp,
    )

    np.testing.assert_allclose(np.array(tau_jx), tau_np, rtol=1.0e-12, atol=1.0e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'regime',
    ['unyielded', 'yielding', 'mobile', 'no_lid', 'mushy', 'inactive_convection'],
)
def test_lid_state_and_effective_viscosity_parity(regime: str):
    """Verify lid diagnostics and effective viscosity match between NumPy and JAX."""
    n_nodes = 40
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)

    if regime == 'unyielded':
        temperature = np.linspace(3000.0, 1800.0, n_nodes)
        conv_flux = np.linspace(0.0, 50.0, n_nodes)
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(0.0, 1.0e-10, n_nodes)
        v_unyielded[-8:] = 0.0
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 2500.0e6
    elif regime == 'yielding':
        temperature = np.linspace(3200.0, 1600.0, n_nodes)
        conv_flux = np.linspace(0.0, 200.0, n_nodes)
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(0.0, 1.0e-9, n_nodes)
        v_unyielded[-8:] = 0.0
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 23.2e6
    elif regime == 'mobile':
        temperature = np.linspace(3400.0, 1400.0, n_nodes)
        conv_flux = np.linspace(0.0, 1000.0, n_nodes)
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(0.0, 1.0e-8, n_nodes)
        v_unyielded[-8:] = 0.0
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 2.0e6
    elif regime == 'no_lid':
        temperature = np.full(n_nodes, 2200.0)
        conv_flux = np.full(n_nodes, 100.0)
        v_unyielded = np.full(n_nodes, 1.0e-9)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 100.0e6
    elif regime == 'mushy':
        temperature = np.linspace(3200.0, 2000.0, n_nodes)
        conv_flux = np.linspace(10.0, 500.0, n_nodes)
        conv_flux[-8:] = 0.0
        v_unyielded = np.linspace(1.0e-11, 1.0e-9, n_nodes)
        v_unyielded[-8:] = 0.0
        melt_frac = np.linspace(0.6, 0.1, n_nodes)
        tau_y_max = 50.0e6
    elif regime == 'inactive_convection':
        temperature = np.linspace(3000.0, 1800.0, n_nodes)
        conv_flux = np.zeros(n_nodes)
        v_unyielded = np.zeros(n_nodes)
        melt_frac = np.zeros(n_nodes)
        tau_y_max = 500.0e6
    else:
        raise ValueError(f'Unknown regime: {regime}')

    tot_flux = conv_flux + 10.0
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_max=tau_y_max,
    )

    lid_inputs = dict(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=tot_flux,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=v_unyielded,
        viscosity_solid=1.0e21,
    )
    state_np = _lid_state(np, **lid_inputs)
    state_jx = _lid_state(jnp, **lid_inputs)

    for key in [
        'T_i',
        'P_T_i',
        'dT_rh',
        'theta',
        'd_lid',
        'd_lid_raw',
        'v_i',
        'v_m',
        'tau_d',
        'tau_d_lid',
        'tau_buoy',
        'tau_d_over_tau_buoy',
        'Ra_eff',
        'tau_y_lid',
        'w_active',
        'w_y',
        'lid_regime',
    ]:
        val_np = float(state_np[key])
        val_jx = float(state_jx[key])
        atol = 1.0e-20 if key in ('v_i', 'v_m') else 1.0e-12
        np.testing.assert_allclose(
            val_jx,
            val_np,
            rtol=1.0e-9,
            atol=atol,
            err_msg=f'Mismatch in diagnostic {key} for regime {regime}',
        )

    if regime in ('unyielded', 'yielding', 'mobile'):
        assert state_np['d_lid'] > 1.0e4, f'{regime} must form stagnant lid > 10 km'
        assert state_np['w_active'] > 0.5, f'{regime} must have active lid closure'
        ratio = float(state_np['tau_d_lid']) / float(state_np['tau_y_lid'])
        if regime == 'unyielded':
            assert ratio < 0.01
            assert float(state_np['w_y']) < 1.0e-4
            assert float(state_np['lid_regime']) == pytest.approx(1.0, rel=1.0e-4)
        elif regime == 'yielding':
            assert 0.95 < ratio < 1.0
            assert 0.2 < float(state_np['w_y']) < 0.5
            assert 1.2 < float(state_np['lid_regime']) < 1.5
        elif regime == 'mobile':
            assert ratio > 1.1
            assert float(state_np['w_y']) > 0.99
            assert float(state_np['lid_regime']) > 1.99
    elif regime in ('no_lid', 'mushy'):
        # Non-lid profiles have d_lid == 0.0 and remaining boundary-layer diagnostics == 0.0.
        assert state_np['d_lid'] == 0.0, f'{regime} must suppress stagnant lid formation'
        assert state_np['lid_regime'] == 0.0, f'{regime} must have lid_regime == 0'
        assert float(state_np['w_y']) == 0.0
        assert float(state_np['tau_d']) == 0.0
        assert float(state_np['tau_d_lid']) == 0.0
        assert float(state_np['tau_buoy']) == 0.0
        assert float(state_np['tau_d_over_tau_buoy']) == 0.0
        assert float(state_np['v_m']) == 0.0
        assert float(state_np['Ra_eff']) == 0.0
        assert float(state_np['Ra_rh']) == 0.0
    elif regime == 'inactive_convection':
        # Zero convective flux gives w_weight == 0.0, setting inactive-convection diagnostics to literal 0.0.
        assert state_np['d_lid'] == 0.0
        assert state_np['tau_d'] == 0.0

    np.testing.assert_allclose(
        np.array(state_jx['w_lid']),
        state_np['w_lid'],
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg=f'Mismatch in w_lid for regime {regime}',
    )

    eta_diff_np = compute_arrhenius_viscosity(
        temperature,
        pressure,
        viscosity_solid=1.0e21,
        g_homologous=params.g_homologous,
        t_sol_0=params.t_sol_0,
        dt_sol_dp=params.dt_sol_dp,
        arrhenius_t_ref=params.arrhenius_t_ref,
        xp=np,
    )
    eta_diff_jx = jnp.array(eta_diff_np)

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        eta_i=state_np['eta_i'],
        tau_d=state_np['tau_d_lid'],
        tau_y_lid=state_np['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )

    eta_eff_jx = compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        eta_i=state_jx['eta_i'],
        tau_d=state_jx['tau_d_lid'],
        tau_y_lid=state_jx['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
        xp=jnp,
    )

    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg=f'Mismatch in eta_eff for regime {regime}',
    )
    if regime == 'mobile':
        assert np.any(eta_eff_np < eta_diff_np)

    if regime == 'unyielded':
        t_sol_1d = 2200.0 + 2.4e-8 * pressure - 5.0e-20 * pressure**2
        params_1d = dataclasses.replace(params, t_sol_0=2200.0)
        st_sol_np = compute_stagnant_lid_state(
            radii=radii,
            temperature=temperature,
            pressure=pressure,
            convective_flux=conv_flux,
            total_flux=tot_flux,
            solidus_temperature=t_sol_1d,
            melt_fraction=melt_frac,
            params=params_1d,
            unyielded_velocity=v_unyielded,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=np,
        )
        st_sol_jx = compute_stagnant_lid_state(
            radii=jnp.array(radii),
            temperature=jnp.array(temperature),
            pressure=jnp.array(pressure),
            convective_flux=jnp.array(conv_flux),
            total_flux=jnp.array(tot_flux),
            solidus_temperature=jnp.array(t_sol_1d),
            melt_fraction=jnp.array(melt_frac),
            params=params_1d,
            unyielded_velocity=jnp.array(v_unyielded),
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=jnp,
        )
        assert float(st_sol_np['d_lid']) > 1.0e5
        assert float(st_sol_np['tau_d']) > 0.0
        for key in ['theta', 'dT_rh', 'd_lid', 'tau_d', 'eta_i']:
            np.testing.assert_allclose(
                float(st_sol_jx[key]),
                float(st_sol_np[key]),
                rtol=1.0e-9,
                atol=1.0e-12,
            )


@pytest.mark.unit
def test_jax_jacrev_closure_differentiability():
    """Verify JAX reverse-mode autodiff through the full lid and yield closure."""
    n_nodes = 20
    radii = jnp.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = jnp.linspace(135.0e9, 1.0e5, n_nodes)
    v_unyielded = jnp.linspace(0.0, 1.0e-8, n_nodes).at[-5:].set(0.0)
    conv_flux = jnp.linspace(0.0, 200.0, n_nodes).at[-5:].set(0.0)
    tot_flux = jnp.full(n_nodes, 200.0)
    melt_frac = jnp.zeros(n_nodes)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_max=1.0e6,
    )

    def _eval_closure(T_profile):
        lid_st = compute_stagnant_lid_state(
            radii=radii,
            temperature=T_profile,
            pressure=pressure,
            convective_flux=conv_flux,
            total_flux=tot_flux,
            solidus_temperature=None,
            melt_fraction=melt_frac,
            params=params,
            unyielded_velocity=v_unyielded,
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
            xp=jnp,
        )
        eta_d = compute_arrhenius_viscosity(
            T_profile,
            pressure,
            viscosity_solid=1.0e21,
            g_homologous=params.g_homologous,
            t_sol_0=params.t_sol_0,
            dt_sol_dp=params.dt_sol_dp,
            arrhenius_t_ref=params.arrhenius_t_ref,
            xp=jnp,
        )
        eta_eff = compute_effective_viscosity(
            eta_diff=eta_d,
            eta_i=lid_st['eta_i'],
            tau_d=lid_st['tau_d'],
            tau_y_lid=lid_st['tau_y_lid'],
            stress_closure_mode='lid',
            w_lid=lid_st['w_lid'],
            xp=jnp,
        )
        return lid_st, eta_d, eta_eff

    def _loss(T_profile):
        _, _, eta_eff = _eval_closure(T_profile)
        return jnp.sum(jnp.log10(eta_eff))

    T_init = jnp.linspace(3400.0, 1400.0, n_nodes)
    lid_st_init, eta_d_init, eta_eff_init = _eval_closure(T_init)
    assert float(lid_st_init['d_lid']) > 0.0
    assert float(lid_st_init['tau_d']) > float(lid_st_init['tau_y_lid']) > 0.0
    assert float(jnp.max(lid_st_init['w_lid'])) > 0.9
    assert jnp.any(eta_eff_init < eta_d_init)

    grad_val = jax.grad(_loss)(T_init)
    grad_tau_d = jax.grad(lambda T_p: _eval_closure(T_p)[0]['tau_d'])(T_init)

    assert grad_val.shape == (n_nodes,)
    assert jnp.all(jnp.isfinite(grad_val))
    assert float(jnp.linalg.norm(grad_val)) > 0.0
    assert grad_tau_d.shape == (n_nodes,)
    assert jnp.all(jnp.isfinite(grad_tau_d))
    assert float(jnp.linalg.norm(grad_tau_d)) > 0.0

    jac_log_eta = jax.jit(jax.jacrev(lambda T_p: jnp.log10(_eval_closure(T_p)[2])))(T_init)
    assert jac_log_eta.shape == (n_nodes, n_nodes)
    assert jnp.all(jnp.isfinite(jac_log_eta))
    np.testing.assert_allclose(
        np.asarray(jnp.sum(jac_log_eta, axis=0)),
        np.asarray(grad_val),
        rtol=1.0e-8,
        atol=1.0e-11,
    )

    for T_reg, f_reg, phi_reg in (
        (T_init, conv_flux, melt_frac),
        (jnp.full(n_nodes, 1600.0), conv_flux, melt_frac),
        (T_init, jnp.zeros(n_nodes), melt_frac),
        (T_init, conv_flux, jnp.full(n_nodes, 0.8)),
    ):
        jac_reg = jax.jit(
            jax.jacrev(
                lambda T_p, f_c=f_reg, phi_m=phi_reg: jnp.stack(
                    [
                        compute_stagnant_lid_state(
                            radii=radii,
                            temperature=T_p,
                            pressure=pressure,
                            convective_flux=f_c,
                            total_flux=tot_flux,
                            solidus_temperature=None,
                            melt_fraction=phi_m,
                            params=params,
                            unyielded_velocity=v_unyielded,
                            viscosity_solid=1.0e21,
                            phi_rheo=0.4,
                            phi_width=0.15,
                            xp=jnp,
                        )[k]
                        for k in ('T_i', 'theta', 'd_lid', 'v_m', 'tau_d', 'lid_regime')
                    ]
                )
            )
        )(T_reg)
        assert jac_reg.shape == (6, n_nodes)
        assert jnp.all(jnp.isfinite(jac_reg))


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_checkpoint_profile_175kyr_paleos_solidus_consistency():
    """Verify 175 kyr fixture T_basic and phi_basic come from one PALEOS melting curve.

    In ``tests/reference/checkpoint_profile_175kyr.npz`` (built and verified at rtol 1e-12 by
    ``tests/reference/build_checkpoint_profile_175kyr.py``), ``r_basic``, ``P_basic``,
    ``F_conv``, ``F_tot``, ``v_unyielded``, ``T_basic[98:100]``, and ``phi_basic[4:100]``
    are 175 kyr run values; ``T_basic[4:98]`` is mapped from ``phi_basic[4:98]`` by the
    PALEOS lever rule, ``T_basic[0:4]`` preserves ``T_old[0:4] - T_old[4]`` above node 4,
    and ``phi_basic[0:4]`` is recomputed from ``T_basic[0:4]`` by the PALEOS lever rule.
    """
    import importlib.util

    builder_path = os.path.join(
        os.path.dirname(__file__), 'reference', 'build_checkpoint_profile_175kyr.py'
    )
    spec = importlib.util.spec_from_file_location(
        'build_checkpoint_profile_175kyr', builder_path
    )
    assert spec is not None and spec.loader is not None
    builder_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder_mod)
    assert builder_mod.main(['--check']) == 0
    offline_rebuilt = builder_mod.build_fixture_arrays(source_commit='0' * 40)
    np.testing.assert_allclose(
        offline_rebuilt['T_basic'],
        np.load(builder_path.replace('build_', '').replace('.py', '.npz'))['T_basic'],
        rtol=1.0e-12,
        atol=0.0,
    )
    assert f'{float(offline_rebuilt["T_basic"][98]):.2f} K' in builder_mod.__doc__
    assert f'{float(offline_rebuilt["T_basic"][99]):.2f} K' in builder_mod.__doc__

    ckpt_path = os.path.join(
        os.path.dirname(__file__), 'reference', 'checkpoint_profile_175kyr.npz'
    )
    data = np.load(ckpt_path)
    p_gpa = data['P_basic'] * 1.0e-9
    t_basic = data['T_basic']
    phi_basic = data['phi_basic']
    p0_gpa = 2.551686137257537
    t_liq = np.where(
        p_gpa < p0_gpa,
        1831.0 * (1.0 + p_gpa / 4.6) ** 0.33,
        6000.0 * (np.maximum(p_gpa, 0.0) / 140.0) ** 0.26,
    )
    t_sol = 0.8 * t_liq

    np.testing.assert_array_equal(
        phi_basic > 0.0,
        t_basic > t_sol,
        err_msg='phi_basic > 0 must hold if and only if T_basic > T_sol(P_basic)',
    )
    phi_expected = np.clip((t_basic - t_sol) / (t_liq - t_sol), 0.0, 1.0)
    np.testing.assert_allclose(phi_basic, phi_expected, rtol=1.0e-12, atol=1.0e-12)
    # Canary: shifting T_sol by +100 K at the uppermost molten node breaks the equivalence.
    t_sol_shifted = t_sol.copy()
    t_sol_shifted[97] += 100.0
    assert not np.array_equal(phi_basic > 0.0, t_basic > t_sol_shifted)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stagnant_lid_checkpoint_profile_parity():
    """Verify stagnant lid state on 175 kyr checkpoint profile matches between NumPy and JAX.

    Fixture ``tests/reference/checkpoint_profile_175kyr.npz`` (built by
    ``tests/reference/build_checkpoint_profile_175kyr.py``) is a 100-node Earth-sized
    magma-ocean solidification snapshot at ``t = 175 kyr`` where ``r_basic`` (3480 to
    6371 km), ``P_basic`` (145.5 to 0.71 GPa), ``F_conv``, ``F_tot``, ``v_unyielded``,
    ``T_basic[98:100]``, and ``phi_basic[4:100]`` are 175 kyr run values, while
    ``T_basic[4:98]`` is mapped from ``phi_basic[4:98]`` by the PALEOS lever rule,
    ``T_basic[0:4]`` preserves ``T_old[0:4] - T_old[4]`` above node 4, and
    ``phi_basic[0:4]`` is recomputed from ``T_basic[0:4]`` by the PALEOS lever rule
    (``T_basic`` 5040.9 to 452.6 K, ``phi_basic`` 0.159 at the CMB to 0.0 at the surface)
    with ``T_basic[-1] = 452.559`` K and ``f[-1] = F_conv[-1] / |F_tot[-1]| = 0.08007493``.
    Because the interior temperature selector ``w_Ti`` is centred at ``0.9 * f_int_max``
    (``0.9`` on this profile) and excludes the boundary nodes, both the default
    ``interior_flux_fraction = 0.05`` and ``0.10`` give ``T_i = 1423.015`` K,
    ``eta_i = 2.760705e+22`` Pa s, ``d_lid = 42134.62`` m, ``max(w_lid) = 1.000000``,
    ``tau_d = 1.299363e+06`` Pa, and ``tau_y_lid = 5.000000e+08`` Pa.
    """
    # Evaluates under linear solidus fallback (table-free) with solidus_temperature=None.
    ckpt_path = os.path.join(
        os.path.dirname(__file__), 'reference', 'checkpoint_profile_175kyr.npz'
    )
    assert os.path.exists(ckpt_path), (
        f'Missing reference checkpoint profile fixture: {ckpt_path}'
    )

    data = np.load(ckpt_path)
    r_basic = data['r_basic']
    T_basic = data['T_basic']
    P_basic = data['P_basic']
    F_conv = data['F_conv']
    F_tot = data['F_tot']
    phi_basic = data['phi_basic']
    v_unyielded = data['v_unyielded']

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_max=500.0e6,
        interior_flux_fraction=0.10,
    )
    visc_solid = 1.0e21

    lid_inputs = dict(
        radii=r_basic,
        temperature=T_basic,
        pressure=P_basic,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=None,
        melt_fraction=phi_basic,
        params=params,
        unyielded_velocity=v_unyielded,
        viscosity_solid=visc_solid,
    )
    state_np = _lid_state(np, **lid_inputs)
    state_jx = _lid_state(jnp, **lid_inputs)

    assert float(state_np['T_i']) == pytest.approx(1423.014834, rel=1.0e-6)
    assert float(state_np['eta_i']) == pytest.approx(2.760705e22, rel=1.0e-5)
    assert float(state_np['d_lid']) == pytest.approx(42134.62, rel=1.0e-5)
    assert float(np.max(state_np['w_lid'])) == pytest.approx(1.000000, rel=1.0e-5)
    assert float(state_np['tau_d']) == pytest.approx(1.299363e6, rel=1.0e-5)
    assert float(state_np['tau_y_lid']) == pytest.approx(5.0e8, rel=1.0e-12)

    for key in [
        'eta_i',
        'mu_i',
        'tau_d',
        'd_lid',
        'theta',
        'v_i',
        'v_m',
        'tau_buoy',
        'tau_d_over_tau_buoy',
        'Ra_eff',
        'delta_rh',
    ]:
        atol = 1.0e-20 if key in ('v_i', 'v_m') else 1.0e-12
        np.testing.assert_allclose(
            float(state_np[key]),
            float(state_jx[key]),
            rtol=1.0e-9,
            atol=atol,
            err_msg=f'Mismatch in diagnostic {key} on checkpoint profile',
        )

    assert np.max(state_np['w_lid']) > 0.5, (
        'NumPy state must form stagnant lid with w_lid > 0.5'
    )
    assert np.max(state_jx['w_lid']) > 0.5, 'JAX state must form stagnant lid with w_lid > 0.5'
    np.testing.assert_allclose(
        np.array(state_jx['w_lid']),
        state_np['w_lid'],
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in w_lid on checkpoint profile',
    )

    eta_diff_np = compute_arrhenius_viscosity(
        T_basic,
        P_basic,
        viscosity_solid=visc_solid,
        g_homologous=params.g_homologous,
        t_sol_0=params.t_sol_0,
        dt_sol_dp=params.dt_sol_dp,
        arrhenius_t_ref=params.arrhenius_t_ref,
        xp=np,
    )
    eta_diff_jx = jnp.array(eta_diff_np)

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        eta_i=state_np['eta_i'],
        tau_d=state_np['tau_d'],
        tau_y_lid=state_np['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )
    eta_eff_jx = compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        eta_i=state_jx['eta_i'],
        tau_d=state_jx['tau_d'],
        tau_y_lid=state_jx['tau_y_lid'],
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
        xp=jnp,
    )
    np.testing.assert_array_equal(
        eta_eff_np,
        eta_diff_np,
        err_msg='Effective viscosity must match diffusion creep in sub-yield lid',
    )
    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in eta_eff on checkpoint profile (sub-yield)',
    )

    eta_eff_np_yield = compute_effective_viscosity(
        eta_diff=eta_diff_np,
        eta_i=state_np['eta_i'],
        tau_d=state_np['tau_d'],
        tau_y_lid=0.5e6,
        stress_closure_mode='lid',
        w_lid=state_np['w_lid'],
        xp=np,
    )
    eta_eff_jx_yield = compute_effective_viscosity(
        eta_diff=eta_diff_jx,
        eta_i=state_jx['eta_i'],
        tau_d=state_jx['tau_d'],
        tau_y_lid=0.5e6,
        stress_closure_mode='lid',
        w_lid=state_jx['w_lid'],
        xp=jnp,
    )
    assert np.any(eta_eff_np_yield < eta_diff_np), (
        'Effective viscosity must fall below diffusion creep in yielding lid'
    )
    np.testing.assert_allclose(
        np.array(eta_eff_jx_yield),
        eta_eff_np_yield,
        rtol=1.0e-9,
        atol=1.0e-12,
        err_msg='Mismatch in eta_eff on checkpoint profile (yielding)',
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_convecting_layer_viscosity_mixture_parity():
    """Verify mu_i evaluated from mixture viscosity matches between NumPy and JAX."""
    params = SolidRheologyParams()
    r = np.linspace(5.371e6, 6.371e6, 30)
    T = np.linspace(3500.0, 1400.0, 30)
    P = np.linspace(1.3e11, 1.0e5, 30)
    conv_flux = np.ones(30) * 1.0e5
    conv_flux[-4:] = 0.0
    tot_flux = np.ones(30) * 1.0e5
    phi = np.zeros(30)
    phi[:20] = 0.35

    visc_mix_np = np.full(30, 1.0e11)
    visc_mix_np[-4:] = 1.0e23

    base = dict(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=conv_flux,
        total_flux=tot_flux,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
    )

    # 1. Fallback when viscosity_mixture is None: mu_i == arrhenius(T_i, P_T_i).
    st_none_np = _lid_state(np, **base, viscosity_mixture=None)
    st_none_jx = _lid_state(jnp, **base, viscosity_mixture=None)
    np.testing.assert_allclose(float(st_none_np['mu_i']), float(st_none_jx['mu_i']), rtol=1e-12)
    assert float(st_none_np['mu_i']) == pytest.approx(
        float(st_none_np['eta_solid_arrhenius']), rel=1.0e-12
    )

    # 2. When viscosity_mixture is provided: mu_i is evaluated from the mixture viscosity.
    st_mix_np = _lid_state(np, **base, viscosity_mixture=visc_mix_np)
    st_mix_jx = _lid_state(jnp, **base, viscosity_mixture=visc_mix_np)
    np.testing.assert_allclose(float(st_mix_np['mu_i']), float(st_mix_jx['mu_i']), rtol=1e-12)
    np.testing.assert_allclose(float(st_mix_np['tau_d']), float(st_mix_jx['tau_d']), rtol=1e-12)
    np.testing.assert_allclose(
        float(st_mix_np['Ra_eff']), float(st_mix_jx['Ra_eff']), rtol=1e-12
    )
    np.testing.assert_allclose(float(st_mix_np['mu_i']), 1.0e11, rtol=1e-12)
    assert not np.isclose(st_mix_np['mu_i'], st_none_np['mu_i'])
    assert st_none_np['mu_i'] / st_mix_np['mu_i'] > 1.0e5
    assert st_none_jx['mu_i'] / st_mix_jx['mu_i'] > 1.0e5

    # 3. Fallback when convective layer is inactive (W_sum << 1e-6 -> w_weight -> 0).
    inactive_base = dict(
        base,
        convective_flux=np.zeros(30),
        params=dataclasses.replace(params, interior_flux_fraction=0.20),
    )
    st_inactive_np = _lid_state(np, **inactive_base, viscosity_mixture=visc_mix_np)
    st_inactive_jx = _lid_state(jnp, **inactive_base, viscosity_mixture=visc_mix_np)
    np.testing.assert_allclose(
        float(st_inactive_np['mu_i']), float(st_inactive_jx['mu_i']), rtol=1e-12
    )
    np.testing.assert_allclose(
        float(st_inactive_np['mu_i']),
        float(st_inactive_np['eta_solid_arrhenius']),
        rtol=1e-8,
    )
    assert float(st_inactive_np['mu_i']) > 1.0e5 * float(st_mix_np['mu_i'])


@pytest.mark.unit
def test_local_stress_closure_min_form_parity():
    """Verify local mode minimum closure parity between NumPy and JAX."""
    eta_diff = np.array([1.0e21, 1.0e22, 1.0e23, 1.0e24])
    tau_y = np.array([1.0e7, 1.0e7, 1.0e7, 1.0e7])
    sr = np.array([1.0e-18, 1.0e7 / (2.0 * 1.0e22), 1.0e-10, 0.0])

    eta_eff_np = compute_effective_viscosity(
        eta_diff=eta_diff,
        tau_y=tau_y,
        strain_rate=sr,
        stress_closure_mode='local',
        xp=np,
    )
    eta_eff_jx = compute_effective_viscosity(
        eta_diff=jnp.array(eta_diff),
        tau_y=jnp.array(tau_y),
        strain_rate=jnp.array(sr),
        stress_closure_mode='local',
        xp=jnp,
    )

    np.testing.assert_allclose(
        np.array(eta_eff_jx),
        eta_eff_np,
        rtol=1.0e-12,
        err_msg='Mismatch in local mode eta_eff between NumPy and JAX',
    )
    assert eta_eff_np[0] == pytest.approx(eta_diff[0], rel=1.0e-12)
    assert eta_eff_np[1] == pytest.approx(
        eta_diff[1] * (10.0 ** (-0.01 * np.log(2.0))), rel=1.0e-12
    )
    expected_eta_y = tau_y[2] / (2.0 * sr[2])
    assert eta_eff_np[2] == pytest.approx(expected_eta_y, rel=1.0e-12)
    assert eta_eff_np[3] == pytest.approx(eta_diff[3], rel=1.0e-12)
