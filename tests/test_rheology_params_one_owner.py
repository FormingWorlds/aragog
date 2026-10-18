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
        assert getattr(p_cfg, name) == getattr(default_rheo, name)

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

    for banned in (
        "as `'tau_y_profile'`",
        'added separately',
        'not yet connected',
        'pending calibration',
        'clips the effective lid base temperature',
        'rather than inside the rheological sublayer',
        '2.21 \\Delta T_\\text{rh}',
        '0.416 \\Delta T_\\text{rh}',
    ):
        assert banned not in doc_text
    assert '1.55\\text{ MPa}' in doc_text
    assert '0.150' in doc_text
    assert '300 + 1300\\,\\mathrm{erf}(z / \\delta)' in doc_text
    assert '1300\\text{ K}' in doc_text
    assert '13.7\\,\\Delta T_\\text{rh}' in doc_text
    assert '0.584\\,\\Delta T_\\text{rh}' in doc_text
    assert '70.95\\text{ K}' in doc_text
    assert '1423.0\\text{ K}' in doc_text
    assert '1736.9\\text{ K}' in doc_text

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
