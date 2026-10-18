"""Verification that SolidRheologyParams is the single owner of rheology parameters.

Exercises the contract that SolidRheologyParams defines all solid rheology
parameters, and that PhaseConfig matches SolidRheologyParams field-by-field at defaults.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.rheology import (
    R_GAS,
    SolidRheologyParams,
    compute_arrhenius_viscosity,
    compute_stagnant_lid_state,
    merge_rheology,
)

pytestmark = pytest.mark.unit

EXPECTED_FIELDS = {
    'enabled',
    'g_homologous',
    't_sol_0',
    'dt_sol_dp',
    'arrhenius_t_ref',
    'viscosity_max_log10',
    'water_prefactor',
    'yield_stress_c',
    'yield_stress_mu',
    'yield_stress_max',
    'stress_closure_mode',
    'interior_flux_fraction',
    'lid_base_mode',
    'lid_base_temperature',
    'lid_contrast_coeff',
    'lid_mask_width_cells',
    'phi_visc_single',
    'mlt_bottom_slope',
    's_top_inf',
    'c_theta',
}


def test_solid_rheology_params_has_expected_fields():
    """Verify SolidRheologyParams declares all expected fields and no dead fields."""
    import aragog.rheology as rheo_mod

    field_names = {f.name for f in dataclasses.fields(SolidRheologyParams)}
    assert field_names == EXPECTED_FIELDS
    assert not hasattr(rheo_mod, 'eta_eff')
    assert not hasattr(rheo_mod, 'eta_diff')


def test_phase_config_rheology_defaults_match():
    """Verify PhaseConfig defaults match SolidRheologyParams field-by-field and reflect overrides."""
    default_rheo = SolidRheologyParams()
    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
    )
    assert p_cfg.rheology == default_rheo
    # Exact by construction: delegate_rheology forwards each stored field attribute unchanged.
    for name in EXPECTED_FIELDS:
        assert hasattr(p_cfg, name) and getattr(p_cfg, name) == getattr(default_rheo, name)

    p_cfg_custom = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        g_homologous=19.0,
        yield_stress_c=35.0e6,
        water_prefactor=0.25,
    )
    assert p_cfg_custom.rheology != default_rheo
    assert p_cfg_custom.g_homologous == pytest.approx(19.0)
    assert p_cfg_custom.yield_stress_c == pytest.approx(35.0e6)
    assert p_cfg_custom.water_prefactor == pytest.approx(0.25)


def test_phase_config_requires_viscosity():
    """Verify PhaseConfig requires viscosity and rejects None."""
    with pytest.raises(TypeError, match='viscosity'):
        PhaseConfig(
            density=4000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
        )

    with pytest.raises(ValueError, match='viscosity must not be None'):
        PhaseConfig(
            density=4000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=None,  # type: ignore[arg-type]
        )


def test_mixed_phase_config_has_no_rheology_fields():
    """Verify MixedPhaseConfig has no rheology fields."""
    mixed_cfg = MixedPhaseConfig(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1e-3,
    )
    for name in EXPECTED_FIELDS:
        assert not hasattr(mixed_cfg, name)


def test_custom_solid_rheology_params_propagation():
    """Verify non-default SolidRheologyParams propagates through PhaseConfig and merge_rheology."""
    custom_rheo = SolidRheologyParams(
        enabled=True,
        g_homologous=18.0,
        t_sol_0=1450.0,
        dt_sol_dp=2.0e-8,
        water_prefactor=0.5,
        yield_stress_c=40e6,
        stress_closure_mode='local',
        lid_base_mode='fixed',
        lid_base_temperature=1350.0,
        phi_visc_single=0.6,
        s_top_inf=0.18,
        c_theta=0.20,
    )

    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom_rheo,
    )
    # Exact by construction: PhaseConfig stores custom_rheo and delegates its attributes.
    assert p_cfg.rheology == custom_rheo
    assert p_cfg.g_homologous == 18.0
    assert p_cfg.t_sol_0 == 1450.0
    assert p_cfg.dt_sol_dp == 2.0e-8
    assert p_cfg.water_prefactor == 0.5
    assert p_cfg.phi_visc_single == 0.6
    assert p_cfg.s_top_inf == 0.18
    assert p_cfg.c_theta == 0.20


def test_solid_rheology_params_unconditional_bounds_validation():
    """Verify bounds validation runs unconditionally even when enabled=False."""
    with pytest.raises(ValueError, match='phi_visc_single'):
        SolidRheologyParams(enabled=False, phi_visc_single=-0.1)

    with pytest.raises(ValueError, match='g_homologous'):
        SolidRheologyParams(enabled=False, g_homologous=-1.0)

    with pytest.raises(ValueError, match='t_sol_0'):
        SolidRheologyParams(enabled=False, t_sol_0=0.0)

    with pytest.raises(ValueError, match='dt_sol_dp'):
        SolidRheologyParams(enabled=False, dt_sol_dp=-1.0e-9)

    with pytest.raises(ValueError, match='viscosity_max_log10'):
        SolidRheologyParams(enabled=False, viscosity_max_log10=15.0)


def test_phase_config_merge_precedence():
    """Verify PhaseConfig preserves custom rheology when flat kwargs are provided."""
    custom = SolidRheologyParams(
        enabled=True,
        g_homologous=20.0,
        t_sol_0=1500.0,
        yield_stress_c=40e6,
        water_prefactor=0.5,
    )
    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom,
        lid_contrast_coeff=3.0,
    )
    # Exact by construction: merge_rheology copies dataclass fields without arithmetic.
    assert p_cfg.rheology.enabled is True
    assert p_cfg.rheology.g_homologous == 20.0
    assert p_cfg.rheology.t_sol_0 == 1500.0
    assert p_cfg.rheology.yield_stress_c == 40e6
    assert p_cfg.rheology.water_prefactor == 0.5
    assert p_cfg.rheology.lid_contrast_coeff == 3.0
    assert p_cfg.lid_contrast_coeff == 3.0


def test_arrhenius_viscosity_water_prefactor_ceiling_numpy_and_jax():
    """Verify water_prefactor scaling in compute_arrhenius_viscosity and compute_stagnant_lid_state."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = pytest.importorskip('jax.numpy')

    T = 650.0
    P = 0.0
    cap_log10 = 35.0
    expected_cap = 10.0**cap_log10
    expected_dry_650 = 1.0e21 * np.exp(25.0 * (1400.0 / 650.0 - 1400.0 / 1600.0))
    assert expected_dry_650 < expected_cap < 100.0 * expected_dry_650

    res_dry_np = compute_arrhenius_viscosity(
        T, P, viscosity_solid=1e21, viscosity_max_log10=cap_log10, water_prefactor=1.0
    )
    assert res_dry_np == pytest.approx(expected_dry_650, rel=1.0e-12)
    assert res_dry_np < expected_cap

    res_np = compute_arrhenius_viscosity(
        T, P, viscosity_solid=1e21, viscosity_max_log10=cap_log10, water_prefactor=100.0
    )
    assert res_np == pytest.approx(expected_cap, rel=1.0e-12)
    assert res_np <= expected_cap

    res_dry_jax = compute_arrhenius_viscosity(
        jnp.asarray(T),
        jnp.asarray(P),
        viscosity_solid=1e21,
        viscosity_max_log10=cap_log10,
        water_prefactor=1.0,
        xp=jnp,
    )
    assert float(res_dry_jax) == pytest.approx(expected_dry_650, rel=1.0e-12)
    assert float(res_dry_jax) < expected_cap

    res_jax = compute_arrhenius_viscosity(
        jnp.asarray(T),
        jnp.asarray(P),
        viscosity_solid=1e21,
        viscosity_max_log10=cap_log10,
        water_prefactor=100.0,
        xp=jnp,
    )
    assert float(res_jax) == pytest.approx(expected_cap, rel=1.0e-12)
    assert float(res_jax) <= expected_cap

    # Unclipped hand-computed verification at T = 1800 K, P = 10 GPa, g_hom = 25:
    t_unclipped = 1800.0
    p_unclipped = 10.0e9
    t_sol_p = 1400.0 + 2.4e-8 * p_unclipped
    expected_unclipped = 1.0e21 * 0.1 * np.exp(25.0 * (t_sol_p / t_unclipped - 1400.0 / 1600.0))
    eta_wet_np = compute_arrhenius_viscosity(
        t_unclipped,
        p_unclipped,
        viscosity_solid=1.0e21,
        g_homologous=25.0,
        t_sol_0=1400.0,
        dt_sol_dp=2.4e-8,
        arrhenius_t_ref=1600.0,
        r_gas=R_GAS,
        water_prefactor=0.1,
        xp=np,
    )
    assert eta_wet_np == pytest.approx(expected_unclipped, rel=1.0e-12)

    # Verify compute_stagnant_lid_state forwards params.water_prefactor into eta_i and eta_surf.
    n_nodes = 20
    radii = np.linspace(3.48e6, 6.371e6, n_nodes)
    temp_prof = np.linspace(2400.0, 1400.0, n_nodes)
    press_prof = np.linspace(100.0e9, 0.0, n_nodes)
    f_conv = np.full(n_nodes, 50.0)
    f_conv[-4:] = 0.0
    f_tot = np.full(n_nodes, 60.0)
    phi_prof = np.zeros(n_nodes)

    st_dry = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_prof,
        pressure=press_prof,
        convective_flux=f_conv,
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi_prof,
        params=SolidRheologyParams(enabled=True, water_prefactor=1.0),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    st_wet = compute_stagnant_lid_state(
        radii=radii,
        temperature=temp_prof,
        pressure=press_prof,
        convective_flux=f_conv,
        total_flux=f_tot,
        solidus_temperature=None,
        melt_fraction=phi_prof,
        params=SolidRheologyParams(enabled=True, water_prefactor=0.1),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(st_wet['eta_i']) == pytest.approx(0.1 * float(st_dry['eta_i']), rel=1.0e-12)
    assert float(st_wet['mu_i']) == pytest.approx(0.1 * float(st_dry['mu_i']), rel=1.0e-12)
    assert float(st_wet['eta_contrast']) == pytest.approx(
        float(st_dry['eta_contrast']), rel=1.0e-12
    )
    assert float(st_wet['tau_d']) == pytest.approx(
        (0.1 ** (1.0 / 3.0)) * float(st_dry['tau_d']), rel=1.0e-12
    )


def test_solid_rheology_params_rejects_nan():
    """Verify SolidRheologyParams rejects NaN for all numeric parameters."""
    nan = float('nan')
    for f in dataclasses.fields(SolidRheologyParams):
        if f.type in ('float', float):
            with pytest.raises(ValueError):
                SolidRheologyParams(**{f.name: nan})


def test_merge_precedence_explicit_default_value():
    """Verify explicit keyword argument matching default overrides base rheology while keeping others."""
    custom = SolidRheologyParams(
        enabled=True, g_homologous=18.0, t_sol_0=1500.0, yield_stress_c=40.0e6
    )
    merged = merge_rheology(custom, {'g_homologous': 25.0, 't_sol_0': 1400.0, 'enabled': False})
    # Exact by construction: merge_rheology replaces dataclass fields with the override values.
    assert merged.g_homologous == 25.0
    assert merged.t_sol_0 == 1400.0
    assert merged.enabled is False
    assert merged.yield_stress_c == pytest.approx(40.0e6)

    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom,
        g_homologous=25.0,
        t_sol_0=1400.0,
        enabled=False,
    )
    assert p_cfg.g_homologous == 25.0
    assert p_cfg.t_sol_0 == 1400.0
    assert p_cfg.enabled is False
    assert p_cfg.yield_stress_c == pytest.approx(40.0e6)


def _collect_nav_strings(mkdocs_text: str) -> set[str]:
    """Collect leaf target strings from the top-level nav block of mkdocs.yml."""
    out: set[str] = set()
    in_nav = False
    for raw_line in mkdocs_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if not raw_line[0].isspace():
            if stripped == 'nav:':
                in_nav = True
                continue
            if in_nav:
                break
        elif in_nav and ': ' in stripped:
            _, rhs = stripped.split(': ', 1)
            val = rhs.strip().strip('\'"')
            if val:
                out.add(val)
    return out


def test_rheology_docs_and_mkdocs_nav_registration(monkeypatch):
    """Verify docs page parameter table, API stubs, and mkdocs.yml nav entries match this tip."""
    import sys
    from pathlib import Path

    monkeypatch.setitem(sys.modules, 'yaml', None)

    repo_root = Path(__file__).resolve().parent.parent
    nav_leaves = _collect_nav_strings((repo_root / 'mkdocs.yml').read_text(encoding='utf-8'))
    for nav_path in (
        'Explanations/solid_state_convection.md',
        'Reference/api/aragog.rheology.md',
        'Reference/api/aragog.rheology_lid.md',
    ):
        assert nav_path in nav_leaves
        assert (repo_root / 'docs' / nav_path).is_file()

    doc_text = (repo_root / 'docs' / 'Explanations' / 'solid_state_convection.md').read_text(
        encoding='utf-8'
    )
    table_defaults: dict[str, str] = {}
    for line in doc_text.splitlines():
        parts = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(parts) == 3 and parts[0].startswith('`') and parts[0].endswith('`'):
            field_key = parts[0].strip('`')
            if field_key != 'field':
                table_defaults[field_key] = parts[2]

    default_rheo = SolidRheologyParams()
    expected_map = {
        f.name: f'`{getattr(default_rheo, f.name)!r}`'
        for f in dataclasses.fields(SolidRheologyParams)
    }
    assert table_defaults == expected_map
    # Verify physical scaling numbers referenced in docs are computed from code formulas
    r_gas = 8.314462618
    dt_rh_val = (r_gas * 1600.0**2) / 300.0e3
    tau_buoy_val = 3300.0 * 9.81 * 3.0e-5 * dt_rh_val * 30791.0646
    tau_buoy_str = rf'{tau_buoy_val * 1e-6:.2f}\text{{ MPa}}'
    assert tau_buoy_str in doc_text
    assert r'\tau_d / \tau_\text{buoy} \approx 0.150' in doc_text

    # Verify computed reference temperatures, erf offsets, and solidus offset ratio in docs
    assert rf'{dt_rh_val:.2f}\text{{ K}}' in doc_text

    ckpt_path = repo_root / 'tests' / 'reference' / 'checkpoint_profile_175kyr.npz'
    if ckpt_path.is_file():
        from aragog.rheology import compute_stagnant_lid_state

        data = np.load(ckpt_path)
        st_ckpt = compute_stagnant_lid_state(
            radii=data['r_basic'],
            temperature=data['T_basic'],
            pressure=data['P_basic'],
            convective_flux=data['F_conv'],
            total_flux=data['F_tot'],
            solidus_temperature=None,
            melt_fraction=data['phi_basic'],
            params=SolidRheologyParams(enabled=True),
            unyielded_velocity=data['v_unyielded'],
            viscosity_solid=1.0e21,
            phi_rheo=0.4,
            phi_width=0.15,
        )
        assert rf'{float(st_ckpt["T_i"]):.1f}\text{{ K}}' in doc_text

        f_ckpt = np.clip(
            np.maximum(data['F_conv'], 0.0) / np.maximum(np.abs(data['F_tot']), 1.0e-12),
            0.0,
            1.0,
        )
        idx_f09 = int(np.where(f_ckpt[:-1] >= 0.9)[0][-1])
        assert rf'{float(data["T_basic"][idx_f09 - 1]):.1f}\text{{ K}}' in doc_text

    import math

    z_over_delta_005 = math.sqrt(-math.log(0.95))
    t_005 = 300.0 + 1300.0 * math.erf(z_over_delta_005)
    ratio_005 = (1600.0 - t_005) / dt_rh_val
    assert rf'{ratio_005:.1f}\,\Delta T_\text{{rh}}' in doc_text

    z_over_delta_09 = math.sqrt(-math.log(0.1))
    t_09 = 300.0 + 1300.0 * math.erf(z_over_delta_09)
    ratio_09 = (1600.0 - t_09) / dt_rh_val
    assert rf'{ratio_09:.3f}\,\Delta T_\text{{rh}}' in doc_text

    ratio_sol_ref = math.exp(
        default_rheo.g_homologous * (1429.0 - default_rheo.t_sol_0) / 1600.0
    )
    assert rf'{ratio_sol_ref:.2f}\times' in doc_text

    d_mantle = 6370.0e3 - 3480.0e3
    d_lid_cond = 7.0 * (d_mantle / 49.0)
    theta_ref = (1600.0 - 300.0) / dt_rh_val
    delta_rh_cond = d_lid_cond / theta_ref
    tau_buoy_cond = 3300.0 * 9.81 * 3.0e-5 * dt_rh_val * delta_rh_cond
    assert rf'{tau_buoy_cond * 1e-6:.2f}\text{{ MPa}}' in doc_text

    rheo_api = (repo_root / 'docs' / 'Reference' / 'api' / 'aragog.rheology.md').read_text(
        encoding='utf-8'
    )
    lid_api = (repo_root / 'docs' / 'Reference' / 'api' / 'aragog.rheology_lid.md').read_text(
        encoding='utf-8'
    )
    assert '`stress_closure`' not in rheo_api
    assert '`merge_rheology`' in rheo_api
    assert '`delegate_rheology`' in rheo_api
    assert '`stress_closure`' in lid_api


def test_kernel_signatures_use_none_sentinels_and_match_dataclass_defaults(monkeypatch):
    """Verify kernel signatures use None sentinels that resolve dynamically to SolidRheologyParams."""
    import inspect

    from aragog.rheology import (
        compute_arrhenius_enthalpy,
        compute_mlt_top_slope,
        compute_yield_stress,
        viscous_mixing_length_factor,
    )

    expected_none_params = {
        compute_mlt_top_slope: ('s_top_inf', 'c_theta'),
        viscous_mixing_length_factor: ('s_top_inf', 'bottom_slope', 'c_theta'),
        compute_arrhenius_enthalpy: ('g_homologous', 't_sol_0', 'dt_sol_dp'),
        compute_arrhenius_viscosity: (
            'viscosity_solid',
            'g_homologous',
            't_sol_0',
            'dt_sol_dp',
            'arrhenius_t_ref',
            'viscosity_max_log10',
            'water_prefactor',
        ),
        compute_yield_stress: ('yield_stress_c', 'yield_stress_mu', 'yield_stress_max'),
    }
    for fn, param_names in expected_none_params.items():
        sig = inspect.signature(fn)
        for p_name in param_names:
            assert sig.parameters[p_name].default is None, (
                f'{fn.__name__} parameter {p_name} must default to None'
            )

    defaults = SolidRheologyParams()
    assert compute_arrhenius_enthalpy(10.0e9) == pytest.approx(
        compute_arrhenius_enthalpy(
            10.0e9,
            g_homologous=defaults.g_homologous,
            t_sol_0=defaults.t_sol_0,
            dt_sol_dp=defaults.dt_sol_dp,
        ),
        rel=1.0e-12,
    )
    assert compute_arrhenius_viscosity(1500.0, 10.0e9) == pytest.approx(
        compute_arrhenius_viscosity(
            1500.0,
            10.0e9,
            viscosity_solid=1.0e21,
            g_homologous=defaults.g_homologous,
            t_sol_0=defaults.t_sol_0,
            dt_sol_dp=defaults.dt_sol_dp,
            arrhenius_t_ref=defaults.arrhenius_t_ref,
            viscosity_max_log10=defaults.viscosity_max_log10,
            water_prefactor=defaults.water_prefactor,
        ),
        rel=1.0e-12,
    )
    assert compute_yield_stress(10.0e9) == pytest.approx(
        compute_yield_stress(
            10.0e9,
            yield_stress_c=defaults.yield_stress_c,
            yield_stress_mu=defaults.yield_stress_mu,
            yield_stress_max=defaults.yield_stress_max,
        ),
        rel=1.0e-12,
    )
    r_grid = np.linspace(3.48e6, 6.371e6, 7)
    ml_grid = np.minimum(r_grid - r_grid[0], r_grid[-1] - r_grid)
    np.testing.assert_allclose(
        viscous_mixing_length_factor(r_grid, r_grid[0], r_grid[-1], ml_grid),
        viscous_mixing_length_factor(
            r_grid,
            r_grid[0],
            r_grid[-1],
            ml_grid,
            s_top_inf=defaults.s_top_inf,
            bottom_slope=defaults.mlt_bottom_slope,
        ),
        rtol=1.0e-12,
    )

    monkeypatch.setattr(SolidRheologyParams, 'g_homologous', 30.0)
    assert compute_arrhenius_enthalpy(0.0) == pytest.approx(
        30.0 * R_GAS * defaults.t_sol_0, rel=1.0e-12
    )
    monkeypatch.setattr(SolidRheologyParams, 'yield_stress_c', 80.0e6)
    assert compute_yield_stress(0.0) == pytest.approx(80.0e6, rel=1.0e-12)


def test_stagnant_lid_state_rejects_mismatched_t_sol_0():
    """Verify compute_stagnant_lid_state raises ValueError when t_sol_0 and params.t_sol_0 differ."""
    n_nodes = 10
    r = np.linspace(3.48e6, 6.371e6, n_nodes)
    t = np.linspace(2600.0, 300.0, n_nodes)
    p = np.linspace(135.0e9, 0.0, n_nodes)
    params = SolidRheologyParams(enabled=True, t_sol_0=1400.0)
    for sol_t in (None, np.full(n_nodes, 2000.0)):
        with pytest.raises(ValueError, match='t_sol_0'):
            compute_stagnant_lid_state(
                radii=r,
                temperature=t,
                pressure=p,
                convective_flux=np.full(n_nodes, 80.0),
                total_flux=np.full(n_nodes, 100.0),
                solidus_temperature=sol_t,
                t_sol_0=2000.0,
                melt_fraction=np.zeros(n_nodes),
                params=params,
                viscosity_solid=1.0e21,
                phi_rheo=0.4,
                phi_width=0.15,
            )


def test_softmax_velocity_scale_constant_and_phase_config_roundtrip(monkeypatch):
    """Verify SOFTMAX_VELOCITY_SCALE constant in rheology_lid and PhaseConfig attrs round-trip."""
    import attrs

    import aragog.rheology_lid as lid_mod

    assert hasattr(lid_mod, 'SOFTMAX_VELOCITY_SCALE')
    assert 'SOFTMAX_VELOCITY_SCALE' in lid_mod.__all__
    assert lid_mod.SOFTMAX_VELOCITY_SCALE == pytest.approx(3.17e-12, rel=1.0e-12)

    n_nodes = 20
    r = np.linspace(3.48e6, 6.371e6, n_nodes)
    t = np.linspace(2600.0, 300.0, n_nodes)
    p = np.linspace(135.0e9, 0.0, n_nodes)
    f_conv = np.full(n_nodes, 80.0)
    f_conv[-4:] = 0.0
    v_uny = np.linspace(1.0e-10, 1.0e-8, n_nodes)
    kw = dict(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=f_conv,
        total_flux=np.full(n_nodes, 100.0),
        solidus_temperature=None,
        melt_fraction=np.zeros(n_nodes),
        params=SolidRheologyParams(enabled=True),
        unyielded_velocity=v_uny,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
    )
    st_default = compute_stagnant_lid_state(**kw)
    monkeypatch.setattr(lid_mod, 'SOFTMAX_VELOCITY_SCALE', 3.17e-9)
    st_mutated = compute_stagnant_lid_state(**kw)
    assert abs(float(st_mutated['v_i']) - float(st_default['v_i'])) > 1.0e-9

    pc = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        g_homologous=22.0,
        yield_stress_c=40.0e6,
    )
    evolved = attrs.evolve(
        pc,
        viscosity=2.0e21,
        rheology=dataclasses.replace(pc.rheology, enabled=True, g_homologous=19.0),
    )
    assert evolved.viscosity == pytest.approx(2.0e21)
    assert evolved.g_homologous == pytest.approx(19.0)
    assert evolved.rheology.enabled is True
    assert PhaseConfig(**attrs.asdict(pc)) == pc
    assert (
        PhaseConfig(**{**attrs.asdict(pc), 'rheology': dataclasses.asdict(pc.rheology)}) == pc
    )
    with pytest.raises(AttributeError):
        pc.g_homologous = 30.0  # type: ignore[misc]


@pytest.mark.unit
def test_solid_rheology_params_bounds_covers_every_numeric_field():
    """Verify _BOUNDS table covers every numeric field in SolidRheologyParams."""
    from aragog.rheology import _BOUNDS

    bounds_fields = {entry[0] for entry in _BOUNDS}
    for field_info in dataclasses.fields(SolidRheologyParams):
        if field_info.type in ('float', float):
            assert field_info.name in bounds_fields, (
                f'Numeric field {field_info.name} is missing from _BOUNDS'
            )


@pytest.mark.unit
def test_arrhenius_viscosity_lower_exponent_clip_active():
    """Verify arrhenius viscosity clips exponent to -700 rather than -100."""
    from aragog.rheology import compute_arrhenius_viscosity

    # Setting arrhenius_t_ref = 1.0 K forces exponent = g_hom * (1 - t_sol_0 / arrhenius_t_ref) = -1399
    # which exercises the lower clip -700.0.
    eta = compute_arrhenius_viscosity(
        temperature=1400.0,
        pressure=0.0,
        viscosity_solid=1.0e20,
        g_homologous=1.0,
        t_sol_0=1400.0,
        dt_sol_dp=0.0,
        arrhenius_t_ref=1.0,
    )
    expected_eta = 1.0e20 * np.exp(-700.0)
    assert float(eta) == pytest.approx(expected_eta, rel=1.0e-12)

    # A mutant clipping at -100.0 yields ~3.72e-24, failing this assertion.
    mutant_eta = 1.0e20 * np.exp(-100.0)
    assert abs(float(eta) - mutant_eta) / mutant_eta > 0.99


@pytest.mark.unit
def test_arrhenius_enthalpy_exact_r_gas_codata_2018():
    """Verify compute_arrhenius_enthalpy uses CODATA 2018 exact R_GAS."""
    from aragog.rheology import R_GAS, compute_arrhenius_enthalpy

    assert R_GAS == pytest.approx(8.314462618, abs=1.0e-12)

    # Check absolute H at P=0 Pa with t_sol_0 = 1400.0 K and g_hom = 25.0
    h_linear = compute_arrhenius_enthalpy(
        pressure=0.0, g_homologous=25.0, t_sol_0=1400.0, dt_sol_dp=0.0
    )
    expected_h = 25.0 * 8.314462618 * 1400.0
    assert float(h_linear) == pytest.approx(expected_h, rel=1.0e-12)

    # Assert mutant R_GAS = 8.314 fails
    mutant_h = 25.0 * 8.314 * 1400.0
    assert abs(float(h_linear) - mutant_h) > 15.0

    # Check table enthalpy values from r15728072 solidus
    t_sol_table = np.array([1380.6038, 2475.9161, 2875.9474, 3673.7136, 4232.1499])
    h_g25 = compute_arrhenius_enthalpy(
        pressure=np.array([0.0, 25.0e9, 50.0e9, 100.0e9, 135.0e9]),
        g_homologous=25.0,
        solidus_temperature=t_sol_table,
        t_sol_0=1380.6038,
    )
    expected_h_g25 = 25.0 * 8.314462618 * t_sol_table
    np.testing.assert_allclose(h_g25, expected_h_g25, rtol=1.0e-12)
