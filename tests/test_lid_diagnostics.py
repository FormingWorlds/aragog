"""Verification tests for stagnant lid physical diagnostics.

Validates representative interior temperature T_i, physical lid thickness
d_lid, solidus and rheological front clipping, and Frank-Kamenetskii
contrast parameter theta.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from aragog.rheology import (
    R_GAS,
    SolidRheologyParams,
    compute_stagnant_lid_state,
)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_lid_base_solidus_clipping():
    """Verify that lid base does not penetrate below the solidus."""
    n_nodes = 50
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    # Hot surface above solidus
    temperature = np.linspace(3200.0, 1800.0, n_nodes)
    solidus = np.linspace(3000.0, 1500.0, n_nodes)
    conv_flux = np.linspace(10.0, 200.0, n_nodes)
    total_flux = conv_flux + 10.0
    melt_frac = np.zeros(n_nodes)

    # High fixed lid base temperature above solidus near surface
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=1900.0,  # Hotter than surface
    )

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=solidus,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        xp=np,
    )

    # 1. Molten surface: when surface temperature exceeds solidus, lid formation is suppressed
    assert state['d_lid'] < 1.0, (
        f'Expected d_lid ~ 0 when surface is above solidus, got {state["d_lid"]}'
    )
    assert state['lid_regime'] < 1.0e-3, (
        f'Expected regime ~ 0 (no lid), got {state["lid_regime"]}'
    )

    # 2. Subsolidus surface with hot isotherm: lid base must be clipped by solidus
    temp_cold = np.linspace(3000.0, 1000.0, n_nodes)
    solidus_prof = np.linspace(2500.0, 1400.0, n_nodes)
    flux_conv = 100.0 * np.sin(np.pi * (radii - radii[0]) / (radii[-1] - radii[0]))

    params_hot = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=2000.0,
    )
    state_clipped = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=solidus_prof,
        melt_fraction=melt_frac,
        params=params_hot,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        xp=np,
    )
    state_unclipped = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params_hot,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        xp=np,
    )

    # Solidus clipping must strictly reduce lid thickness to keep lid subsolidus
    assert state_clipped['d_lid'] < state_unclipped['d_lid'] - 1.0e4
    assert state_clipped['T_lid'] < state_unclipped['T_lid']


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_lid_base_rheological_front_clipping():
    """Verify that lid base is clipped by the rheological front (phi = 0.4)."""
    n_nodes = 50
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    temperature = np.linspace(3200.0, 1400.0, n_nodes)
    conv_flux = np.linspace(10.0, 200.0, n_nodes)
    total_flux = conv_flux + 10.0
    # Shallow melt pond / mush where melt fraction > 0.4 near top
    melt_frac = np.zeros(n_nodes)
    melt_frac[-5:] = 0.5

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
    )

    state = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        xp=np,
    )

    # Melt fraction > 0.4 at top cells forces lid base to top cells
    assert state['d_lid'] < 0.2 * (radii[-1] - radii[0])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_lid_base_parameterized_phi_rheo():
    """Verify lid base clipping respects parameterized phi_rheo."""
    n_nodes = 50
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    temp_cold = np.linspace(3000.0, 1000.0, n_nodes)
    flux_conv = 100.0 * np.sin(np.pi * (radii - radii[0]) / (radii[-1] - radii[0]))
    solidus_prof = np.linspace(2500.0, 1400.0, n_nodes)

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=2000.0,
    )

    melt_frac = np.zeros(n_nodes)
    melt_frac[35:45] = 0.5

    state_default = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=solidus_prof,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        xp=np,
    )
    state_high = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=solidus_prof,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        phi_rheo=0.6,
        xp=np,
    )
    assert state_default['d_lid'] < state_high['d_lid']


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_lid_base_without_solidus_governed_by_solidus_melt_threshold():
    """Verify lid base without solidus profile is governed by SOLIDUS_MELT_FRACTION_THRESHOLD."""
    n_nodes = 50
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    temp_cold = np.linspace(3000.0, 1000.0, n_nodes)
    flux_conv = 100.0 * np.sin(np.pi * (radii - radii[0]) / (radii[-1] - radii[0]))

    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
        lid_base_temperature=2000.0,
    )

    melt_frac = np.zeros(n_nodes)
    melt_frac[35:45] = 0.5

    state_04 = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        xp=np,
    )
    state_06 = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        phi_rheo=0.6,
        xp=np,
    )
    state_09 = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=None,
        melt_fraction=melt_frac,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        phi_rheo=0.9,
        xp=np,
    )
    assert state_04['d_lid'] == pytest.approx(state_06['d_lid'], rel=1e-12)
    assert state_04['d_lid'] == pytest.approx(state_09['d_lid'], rel=1e-12)

    # Melt fraction below SOLIDUS_MELT_FRACTION_THRESHOLD (0.01) leaves lid unclipped
    melt_frac_sub = np.zeros(n_nodes)
    melt_frac_sub[35:45] = 0.005
    state_sub = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_cold,
        pressure=pressure,
        convective_flux=flux_conv,
        total_flux=flux_conv + 10.0,
        solidus_temperature=None,
        melt_fraction=melt_frac_sub,
        params=params,
        unyielded_velocity=np.full(n_nodes, 1.0e-9),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        xp=np,
    )
    assert state_sub['d_lid'] > state_04['d_lid']


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_frank_kamenetskii_contrast_scaling():
    """Verify theta scales monotonically with surface temperature drop."""
    n_nodes = 40
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    conv_flux = np.linspace(200.0, 0.0, n_nodes)
    total_flux = np.full(n_nodes, 210.0)
    melt_frac = np.zeros(n_nodes)
    params = SolidRheologyParams(enabled=True, stress_closure_mode='lid')

    thetas = []
    t_surfs = [2200.0, 1800.0, 1400.0, 1000.0, 600.0, 300.0]
    for ts in t_surfs:
        temp = np.linspace(3200.0, ts, n_nodes)
        st = compute_stagnant_lid_state(
            radii=radii,
            temperature=temp,
            pressure=pressure,
            convective_flux=conv_flux,
            total_flux=total_flux,
            solidus_temperature=None,
            melt_fraction=melt_frac,
            params=params,
            unyielded_velocity=np.full(n_nodes, 1.0e-9),
            viscosity_solid=1.0e21,
            xp=np,
        )
        thetas.append(st['theta'])

    thetas = np.array(thetas)
    diffs = np.diff(thetas)
    assert np.all(diffs > 0.0), 'Theta must increase as surface cools'
    assert thetas[0] < 9.0, 'Hot surface must yield small Frank-Kamenetskii parameter'
    assert thetas[-1] > 20.0, 'Cold surface must yield large Frank-Kamenetskii parameter'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_frank_kamenetskii_theta_definition_activation_energy():
    """Verify theta matches Foley & Bercovici (2014) definition using activation energy E."""
    n_nodes = 30
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    pressure = np.linspace(135.0e9, 1.0e5, n_nodes)
    temperature = np.linspace(3000.0, 1500.0, n_nodes)
    conv_flux = np.linspace(100.0, 0.0, n_nodes)
    total_flux = np.full(n_nodes, 120.0)
    params = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        activation_energy=300.0e3,
        activation_volume=1.0e-5,
    )
    st = compute_stagnant_lid_state(
        radii=radii,
        temperature=temperature,
        pressure=pressure,
        convective_flux=conv_flux,
        total_flux=total_flux,
        solidus_temperature=None,
        melt_fraction=np.zeros(n_nodes),
        params=params,
        xp=np,
    )
    expected_theta = (params.activation_energy * (st['T_i'] - temperature[-1])) / (
        R_GAS * st['T_i'] ** 2
    )
    assert st['theta'] == pytest.approx(expected_theta, rel=1.0e-12)


@pytest.mark.unit
def test_low_theta_regime_warning(caplog):
    """Verify warning log when Frank-Kamenetskii parameter theta is below 9."""
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_entropy_solver_const_properties_smoke import (
        _build_const_properties_parameters,
    )

    params = _build_const_properties_parameters(n_nodes=15, end_time=1.0)
    params.phase_solid.rheology = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    params.initial_condition.surface_temperature = 2500.0
    params.initial_condition.basal_temperature = 2600.0

    solver = EntropySolver(params, entropy_eos=None)
    solver.initialize()

    with caplog.at_level(logging.WARNING):
        solver.set_initial_entropy(3050.0)
        solver.solve()

    assert solver.get_state().theta < 9.0
    assert any('below 9.0' in r.message for r in caplog.records)


@pytest.mark.unit
def test_continuous_lid_regime_output():
    """Verify that lid_regime in SolverOutput is continuous and not rounded."""
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_entropy_solver_const_properties_smoke import (
        _build_const_properties_parameters,
    )

    params = _build_const_properties_parameters(n_nodes=20, end_time=1.0)
    params.phase_solid.rheology = SolidRheologyParams(
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='rheological',
        yield_stress_c=1e12,
        yield_stress_max=1e12,
    )

    solver = EntropySolver(params, entropy_eos=None)
    solver.initialize()
    phi_arr = np.zeros(20)
    phi_arr[-1] = 0.012
    solver.state.phase_basic.melt_fraction = lambda: phi_arr
    solver.state.phase_staggered.melt_fraction = lambda: np.zeros(19)

    s_init = np.linspace(3000.0, 1200.0, 19)
    solver.set_initial_entropy(s_init)
    solver.solve()
    output = solver.get_state()

    assert 0.05 < output.lid_regime < 0.95, (
        f'lid_regime must be continuous float; got {output.lid_regime}'
    )
    assert not float(output.lid_regime).is_integer(), (
        f'lid_regime must not be rounded to integer; got {output.lid_regime}'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_lid_velocity_scale_continuity():
    """Verify convective velocity scale v_i is continuous across front shifts."""
    params = SolidRheologyParams(
        enabled=True,
        lid_base_mode='rheological',
        activation_energy=300e3,
        activation_volume=5e-6,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
    )

    r = np.linspace(3.4e6, 6.371e6, 60)
    T = 3000.0 - 1500.0 * ((r - r[0]) / (r[-1] - r[0])) ** 2
    P = 1.3e11 * (1.0 - (r - r[0]) / (r[-1] - r[0]))
    v = 1e-9 * np.sin(np.pi * (r - r[0]) / (r[-1] - r[0]))

    rc_vals = np.linspace(5.0e6, 5.5e6, 200)
    v_is = []
    for rc in rc_vals:
        f_conv = 0.5 * (1.0 + np.tanh((rc - r) / 50e3))
        out = compute_stagnant_lid_state(
            r,
            T,
            P,
            f_conv,
            np.ones_like(r),
            None,
            np.zeros_like(r),
            params,
            unyielded_velocity=v,
            viscosity_solid=1e21,
        )
        v_is.append(out['v_i'])

    v_is = np.array(v_is)
    rel_jumps = np.abs(np.diff(v_is)) / np.maximum(v_is[:-1], 1e-30)
    assert np.max(rel_jumps) < 2e-4
    assert np.all(v_is > 0.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_lid_velocity_scale_deep_velocity_insensitivity():
    """Convective velocity scale and lid regime are insensitive to deep mantle velocities."""
    params = SolidRheologyParams(
        enabled=True,
        lid_base_mode='rheological',
        activation_energy=300e3,
        activation_volume=5e-6,
        arrhenius_t_ref=1600.0,
        interior_flux_fraction=0.05,
    )

    r = np.linspace(3.4e6, 6.371e6, 100)
    x = (r - r[0]) / (r[-1] - r[0])
    T = 3000.0 - 1500.0 * x**2
    P = 1.3e11 * (1.0 - x)
    v0 = 1.4e-9 * np.ones_like(r)
    f_conv = 0.5 * (1.0 + np.tanh((6.25e6 - r) / 50e3))

    out_base = compute_stagnant_lid_state(
        r,
        T,
        P,
        f_conv,
        np.ones_like(r),
        None,
        np.zeros_like(r),
        params,
        unyielded_velocity=v0,
        viscosity_solid=1e21,
    )

    v_i_base = float(out_base['v_i'])
    w_upper = np.asarray(out_base['w_upper'])

    assert v_i_base > 0.0

    # Perturb every node singly across velocity perturbations
    for j in range(len(r)):
        for v_pert_val in [1e-8, 1e-5, 1e-3]:
            v_pert = v0.copy()
            v_pert[j] = v_pert_val
            out_pert = compute_stagnant_lid_state(
                r,
                T,
                P,
                f_conv,
                np.ones_like(r),
                None,
                np.zeros_like(r),
                params,
                unyielded_velocity=v_pert,
                viscosity_solid=1e21,
            )
            v_i_pert = float(out_pert['v_i'])
            dv_i = abs(v_i_pert - v_i_base)
            bound = float(w_upper[j]) * v_pert_val * (1.0 + 1e-6) + 1e-15
            assert dv_i <= bound, (
                f'Node {j} (r={r[j]:.3e}, w_upper={w_upper[j]:.3e}) perturbation {v_pert_val} '
                f'violated bound: dv_i={dv_i:.4e} > bound={bound:.4e}'
            )
            if w_upper[j] == 0.0:
                assert dv_i == 0.0
