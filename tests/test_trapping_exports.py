"""Phase-boundary exports at the basic nodes: porosity, phase densities and gravity."""

from __future__ import annotations

import dataclasses

import netCDF4 as nc
import numpy as np
import pytest

from aragog.eos.entropy_phase import mobility_function
from aragog.solver.entropy_solver import EntropySolver
from tests.conftest import EOS_DIR, needs_eos
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters
from tests.test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S

pytestmark = pytest.mark.unit

EXPORTS = ('porosity_b', 'rho_solid_b', 'rho_melt_b', 'g_b')
D = 1.0e-3  # grain size [m]


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def _solve(eos, S0, end_time=1.0, **phase_mixed):
    p = _build_mushy_parameters(solver_method='cvode', n_nodes=16, end_time=end_time)
    p.phase_mixed = dataclasses.replace(p.phase_mixed, **phase_mixed)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(S0)
    s.solve()
    return s


def _final_y(s):
    return float(s._solution.t[-1]), np.asarray(s._solution.y[:, -1], dtype=float)


@pytest.mark.parametrize(
    'phi,expected',
    [
        (0.001, D**2 * 0.001**2 / (0.999**2 * 1000.0)),  # Blake-Kozeny-Carman
        (0.4, D**2 * 0.4**4.5 * 5.0 / 7.0),  # Rumpf-Gupte
        (0.99, D**2 * 2.0 / 9.0),  # Stokes settling
    ],
)
def test_mobility_function_reduces_to_each_regime(phi, expected):
    """Far from the blends ``F`` is the closed form of its regime, for scalar and array input."""
    assert mobility_function(phi, D) == pytest.approx(expected, rel=1e-3)
    np.testing.assert_allclose(mobility_function(np.array([phi]), D), [expected], rtol=1e-3)


def test_mobility_function_rises_to_a_bounded_stokes_plateau():
    """``F`` is non-negative and rises up to porosity 0.805, where the Rumpf-Gupte tail
    overshoots the Stokes value by 4.4 percent, and ends at the Stokes value."""
    phi = np.linspace(0.0, 1.0, 200001)
    F = mobility_function(phi, D)
    stokes = D**2 * 2.0 / 9.0
    assert np.all(F >= 0.0) and np.all(np.diff(F[phi <= 0.805]) > 0.0)
    assert np.max(F) == pytest.approx(1.0437 * stokes, rel=1e-4)
    assert F[-1] == pytest.approx(stokes, rel=1e-3)


@needs_eos
@pytest.mark.parametrize('mode', ['melt', 'mixture'])
def test_relative_velocity_is_built_from_the_exported_quantities(shared_eos, mode):
    """``v_rel = |rho_l - rho_s| g F(porosity) / eta`` with the evaluator's public pieces."""
    s = _solve(shared_eos, _pick_mushy_S(shared_eos), separation_viscosity=mode)
    ph = s.state.phase_basic
    rho_s, rho_l = ph.phase_boundary_densities()
    eta = ph._viscosity_val if mode == 'mixture' else ph._visc_liquid
    F = mobility_function(ph.porosity(), ph._grain_size)
    expected = np.sqrt((rho_l - rho_s) ** 2 + 1e-12) * ph.gravitational_acceleration() * F / eta
    np.testing.assert_array_equal(ph.relative_velocity(), expected)


@needs_eos
def test_exports_are_the_final_state_and_reach_the_netcdf(shared_eos, tmp_path):
    """The four arrays describe the returned final state, not the last RHS call, and are written."""
    s = _solve(shared_eos, _pick_mushy_S(shared_eos))
    t, y = _final_y(s)
    s._dSdt_single(t, 1.01 * y)  # leave the cached state away from the final one
    out = s.get_state()
    s._dSdt_single(t, y)
    ph = s.state.phase_basic
    P_b = np.asarray(ph.pressure).ravel()
    np.testing.assert_array_equal(out.porosity_b, np.ravel(ph.porosity()))
    lookup = shared_eos._lookup_at_phase_boundary
    np.testing.assert_array_equal(out.rho_solid_b, np.ravel(lookup('density', P_b, 'solid')))
    np.testing.assert_array_equal(out.rho_melt_b, np.ravel(lookup('density', P_b, 'melt')))
    g = s.parameters.mesh.gravitational_acceleration
    np.testing.assert_array_equal(out.g_b, np.full(out.r_basic.shape, abs(g)))
    assert np.all((out.porosity_b > 0.0) & (out.porosity_b < 1.0))
    raw = (out.rho_solid_b - out.rho_basic) / (out.rho_solid_b - out.rho_melt_b)
    mush = (raw > 0.05) & (raw < 0.95)
    assert np.any(mush)
    np.testing.assert_allclose(out.porosity_b[mush], raw[mush], rtol=0.0, atol=1e-5)
    assert out.rho_solid_b[0] > out.rho_melt_b[0]

    out.to_netcdf(tmp_path / 'o.nc')
    with nc.Dataset(tmp_path / 'o.nc') as ds:
        for name in EXPORTS:
            assert ds[name].dimensions == ('basic',)
            np.testing.assert_array_equal(ds[name][:], getattr(out, name))


@needs_eos
def test_porosity_is_near_zero_in_a_solid_column(shared_eos):
    s = _solve(shared_eos, 1800.0)
    out = s.get_state()
    assert np.max(out.phi_basic) == 0.0
    assert np.max(out.porosity_b) < 1e-3 and np.min(out.porosity_b) >= -2.5e-7


@needs_eos
def test_dense_melt_keeps_the_sign_of_the_density_contrast(shared_eos, monkeypatch):
    """Where the melt is denser the exports keep ``rho_solid_b < rho_melt_b`` and porosity is small."""
    s = _solve(shared_eos, _pick_mushy_S(shared_eos))
    ph = s.state.phase_basic
    P_b = np.asarray(ph.pressure).ravel()
    P_mid = np.median(P_b)
    deep = P_b > P_mid
    real = shared_eos._lookup_at_phase_boundary

    def swapped(prop, P, phase):
        other = {'solid': 'melt', 'melt': 'solid'}[phase]
        return np.where(np.asarray(P) > P_mid, real(prop, P, other), real(prop, P, phase))

    monkeypatch.setattr(shared_eos, '_lookup_at_phase_boundary', swapped)
    real_contrast = real('density', P_b, 'solid') - real('density', P_b, 'melt')
    out = s.get_state()
    contrast = out.rho_solid_b - out.rho_melt_b
    np.testing.assert_array_equal(contrast, np.where(deep, -real_contrast, real_contrast))
    assert np.all(contrast[deep] < 0.0)
    assert np.max(out.porosity_b[deep]) < 1e-3


def test_const_properties_exports():
    """Without phase contrast: porosity one, both densities ``const_rho``, gravity the mesh value."""
    p = _build_const_properties_parameters(n_nodes=12, end_time=1.0)
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    s.set_initial_entropy(3000.0)
    s.solve()
    out = s.get_state()
    n = out.r_basic.shape
    np.testing.assert_array_equal(out.porosity_b, np.ones(n))
    np.testing.assert_array_equal(out.rho_solid_b, np.full(n, 4000.0))
    np.testing.assert_array_equal(out.rho_melt_b, np.full(n, 4000.0))
    np.testing.assert_array_equal(out.g_b, np.full(n, abs(p.mesh.gravitational_acceleration)))
