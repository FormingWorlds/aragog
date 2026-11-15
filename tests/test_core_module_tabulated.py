"""The ``profile_source = 'paleos'`` settings of the core module and the solver budget cache."""

from __future__ import annotations

import math
import pathlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.core.module import (
    build_core_module_budget,
    core_module_cache_key,
    load_core_arrays,
    split_core_module_params,
)
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.profiles_tabulated import TabulatedCoreProfiles
from tests.conftest import PALEOS_CUT as CUT
from tests.conftest import entropy_eos_copy, needs_eos, write_core

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]

R_CMB = 3.480e6  # the CMB of the smoke-test mesh


@pytest.fixture
def paleos_params(tmp_path):
    """Settings of a PALEOS core on the smoke-test geometry."""
    core, _ = write_core(tmp_path / 'core.npz')
    return {
        'profile_source': 'paleos',
        'core_liquid_table': str(CUT),
        'core_profile_file': str(core),
        'ds_fusion': 172.8,
        'icn_width': 10.0,
    }


def test_paleos_source_builds_a_tabulated_budget(paleos_params):
    """The factory builds the tabulated profile from the files with c_p 840 and a 10 K grid;
    the Gaussian keys that a config surface carries are accepted and unused."""
    params = {**paleos_params, 'rho_cen': 1.0, 'alpha': 9.9, 'c_p': 1.0, 'p_cen': 4e11}
    b = build_core_module_budget(params, r_cmb=R_CMB, p_cmb_fallback=136e9)
    assert isinstance(b.profiles, TabulatedCoreProfiles)
    assert b.tabulated
    assert b.profiles.c_p_override == 840.0
    assert float(b.profiles.t_grid[1] - b.profiles.t_grid[0]) == pytest.approx(10.0, rel=0.01)
    table_cp = build_core_module_budget(
        {**paleos_params, 'core_cp_override': 'table', 'core_t_cmb_step': 25.0},
        r_cmb=R_CMB,
        p_cmb_fallback=136e9,
    )
    assert table_cp.profiles.c_p_override is None
    ratio = float(table_cp.secular_capacity(t_cmb=4000.0)) / float(
        b.secular_capacity(t_cmb=4000.0)
    )
    assert 0.5 < ratio < 0.8  # PALEOS liquid c_p at the CMB over 840


def test_the_mass_and_radius_of_the_arrays_are_checked(paleos_params):
    """``m_core`` must equal the arrays' mass to 1e-10 and the mesh CMB their last radius."""
    m = load_core_arrays(paleos_params['core_profile_file'])['m'][-1]
    build_core_module_budget(paleos_params, r_cmb=R_CMB, p_cmb_fallback=1.0, m_core=m)
    with pytest.raises(ValueError, match='core arrays hold'):
        build_core_module_budget(
            paleos_params, r_cmb=R_CMB, p_cmb_fallback=1.0, m_core=1.01 * m
        )
    build_core_module_budget(paleos_params, r_cmb=R_CMB + 0.9, p_cmb_fallback=1.0)
    with pytest.raises(ValueError, match='the mesh CMB'):
        build_core_module_budget(paleos_params, r_cmb=R_CMB + 2.0, p_cmb_fallback=1.0)


def test_core_arrays_load_with_their_scalars(paleos_params):
    """0-d arrays come back as floats, a string array as str, the profile arrays as arrays."""
    core = load_core_arrays(paleos_params['core_profile_file'])
    assert core['M_ic'] == 0.0 and isinstance(core['M_ic'], float)
    assert core['w_layers'] == '{"outer": {"S": 0.0}}'
    assert core['r'].shape == (101,) and core['r'][-1] == R_CMB


def test_core_array_file_errors(tmp_path):
    """A missing file and a file that is not .npz are refused."""
    with pytest.raises(FileNotFoundError):
        load_core_arrays(tmp_path / 'absent.npz')
    with pytest.raises(ValueError, match='must be an .npz'):
        load_core_arrays(tmp_path / 'core.dat')


@pytest.mark.parametrize(
    ('change', 'msg'),
    [
        ({'profile_source': 'tabulated'}, 'profile_source must be one of'),
        ({'core_profile_file': None}, r"needs \['core_profile_file'\]"),
        ({'core_liquid_table': None, 'core_profile_file': None}, 'needs'),
        ({'core_liquid_table': 3.0}, 'must be a file path'),
        ({'core_profile_file': ''}, r"needs \['core_profile_file'\]|must be a file path"),
        ({'stratification': True, 'k_core': 40.0}, 'does not support stratification'),
        ({'profile_source': 'gaussian'}, "read only with profile_source = 'paleos'"),
    ],
)
def test_bad_tabulated_settings_are_refused_before_the_build(paleos_params, change, msg):
    """Each fault is refused by the parameter split, which runs before any profile build."""
    with pytest.raises(ValueError, match=msg):
        split_core_module_params({**paleos_params, **change})


@settings(max_examples=40, deadline=None)
@given(
    key=st.sampled_from(['core_cp_override', 'core_t_cmb_step']),
    bad=st.sampled_from([0.0, -1.0, math.nan, math.inf, -math.inf, True, False, 'x', [840.0]]),
)
def test_a_bad_override_or_grid_step_is_refused(key, bad):
    """A TOML bool, a string other than 'table', zero, negative, NaN, inf or a list is
    refused for the c_p override and the grid step."""
    params = {
        'profile_source': 'paleos',
        'core_liquid_table': str(CUT),
        'core_profile_file': 'core.npz',
        key: bad,
    }
    with pytest.raises(ValueError, match=f'{key} must be a positive number'):
        split_core_module_params(params)


def test_the_gaussian_default_ignores_the_tabulated_values():
    """With no source (or None, as an attrs surface writes it) the Gaussian path builds and
    the c_p override and grid step of a config surface are ignored."""
    for source in (None, 'gaussian'):
        params = {'profile_source': source, 'core_cp_override': 840.0, 'core_t_cmb_step': None}
        b = build_core_module_budget(params, r_cmb=R_CMB, p_cmb_fallback=136e9)
        assert isinstance(b.profiles, GaussianCoreProfiles)
        assert not b.tabulated


def test_the_cache_key_follows_the_file_content(paleos_params, tmp_path):
    """Rewriting the core file at the same path with new arrays changes the key; rewriting
    it with the same bytes keeps it; the Gaussian key has no file part."""
    key = core_module_cache_key(paleos_params, R_CMB, 136e9)
    path = pathlib.Path(paleos_params['core_profile_file'])
    data = path.read_bytes()
    path.write_bytes(data)
    assert core_module_cache_key(paleos_params, R_CMB, 136e9) == key
    write_core(path, rho_cen=12400.0)
    assert core_module_cache_key(paleos_params, R_CMB, 136e9) != key
    assert core_module_cache_key({'rho_cen': 1.0}, R_CMB, 136e9)[-1] == ()


@needs_eos
def test_a_solver_reset_rebuilds_the_budget_when_the_core_file_changes(paleos_params):
    """A solver reset keeps the budget while the core file holds the same bytes and rebuilds
    it when a structure update rewrites the file at the same path."""
    from tests.test_entropy_solver_core_module_smoke import _build

    solver = _build('core_module', entropy_eos_copy(), dict(paleos_params), s_init='driven')
    budget = solver._core_module_budget
    assert budget.tabulated
    path = pathlib.Path(paleos_params['core_profile_file'])
    path.write_bytes(path.read_bytes())
    solver.reset()
    assert solver._core_module_budget is budget
    write_core(path, rho_cen=12400.0)
    solver.reset()
    assert solver._core_module_budget is not budget
    assert solver._core_module_budget.profiles.rho_cen == pytest.approx(12400.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_core_module_diagnostics_report_the_calibration_fraction(paleos_params):
    """A PALEOS core reports the mass fraction outside the Luo et al. (2024) window: an Earth
    core at 5000 K (liquid, centre near 7300 K) lies below its 8000 K floor everywhere
    (fraction 1); a Gaussian core keeps its diagnostic keys unchanged."""
    from aragog.core.module import CoreModule

    b = build_core_module_budget(paleos_params, r_cmb=R_CMB, p_cmb_fallback=136e9)
    diag = CoreModule(b, t_cmb=5000.0).diagnostics()
    assert diag['core_eos_outside_calibration'] == pytest.approx(1.0, abs=1e-15)
    assert 7000.0 < diag['t_cen'] < 8000.0
    assert not diag['inner_core_present']
    assert diag['effective_capacity'] == pytest.approx(
        float(b.secular_capacity(t_cmb=5000.0)), rel=1e-12
    )
    gauss = build_core_module_budget({}, r_cmb=R_CMB, p_cmb_fallback=136e9)
    keys = set(CoreModule(gauss, t_cmb=4000.0).diagnostics())
    assert keys == {
        't_cmb',
        't_cen',
        'r_icb',
        'nucleation_factor',
        'effective_capacity',
        'inner_core_present',
        'fully_frozen',
        'regime',
    }


@pytest.mark.parametrize('key', ['m_core', 'r_cmb'])
def test_nan_mass_or_radius_is_refused(paleos_params, key):
    """A NaN core mass or mesh CMB radius fails the checks against the arrays."""
    kw = {'r_cmb': R_CMB, 'p_cmb_fallback': 136e9, key: math.nan}
    with pytest.raises(ValueError, match='core arrays hold|the mesh CMB'):
        build_core_module_budget(paleos_params, **kw)


@pytest.mark.parametrize('bad', [False, 0, 1.0, ['paleos']])
def test_a_profile_source_that_is_not_a_name_is_refused(bad):
    """Only None and the names in PROFILE_SOURCES select a source; a bool or a number is
    refused, not read as the Gaussian default."""
    with pytest.raises(ValueError, match='profile_source must be one of'):
        split_core_module_params({'profile_source': bad})


def test_core_module_refuses_a_start_outside_the_grid(paleos_params):
    """CoreModule refuses a CMB temperature outside the tabulated isentropes at the start."""
    from aragog.core.module import CoreModule

    b = build_core_module_budget(paleos_params, r_cmb=R_CMB, p_cmb_fallback=136e9)
    with pytest.raises(ValueError, match='outside the tabulated isentropes'):
        CoreModule(b, t_cmb=1500.0)


def test_a_mesh_pressure_far_from_the_arrays_is_logged(paleos_params, caplog):
    """A CMB pressure of the arrays more than 1 % from the mesh value is logged once as a
    warning naming both; within 1 % nothing is logged."""
    with caplog.at_level('WARNING', logger='fwl.aragog.core.module'):
        build_core_module_budget(paleos_params, r_cmb=R_CMB, p_cmb_fallback=136e9)
        assert not caplog.records
        build_core_module_budget(paleos_params, r_cmb=R_CMB, p_cmb_fallback=100e9)
    assert any('CMB pressure' in r.getMessage() for r in caplog.records)
