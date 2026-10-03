"""Phase-boundary exports at the basic nodes: porosity, phase densities and gravity."""

from __future__ import annotations

import dataclasses

import netCDF4 as nc
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from aragog.eos.entropy_phase import EntropyPhaseEvaluator, mobility_function
from aragog.solver.entropy_solver import EntropySolver
from tests.conftest import EOS_DIR, needs_eos
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters
from tests.test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S

pytestmark = pytest.mark.unit

D = 1.0e-3  # grain size [m]


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def _solve(eos, S0, mesh=None, **phase_mixed):
    p = _build_mushy_parameters(solver_method='cvode', n_nodes=16, end_time=1.0)
    p.phase_mixed = dataclasses.replace(p.phase_mixed, **phase_mixed)
    for k, v in (mesh or {}).items():
        setattr(p.mesh, k, v)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(S0)
    s.solve()
    return s


def _bkc(phi):
    return D**2 * phi**2 / ((1.0 - phi) ** 2 * 1000.0)


def _rg(phi):
    return D**2 * phi**4.5 * 5.0 / 7.0


def _blend(lo, hi, phi, centre, width):
    w = 0.5 * (1.0 + np.tanh((phi - centre) / width))
    return (1.0 - w) * lo + w * hi


@pytest.mark.parametrize(
    'phi,expected',
    [
        (0.001, _bkc(0.001)),  # Blake-Kozeny-Carman
        (0.4, _rg(0.4)),  # Rumpf-Gupte
        (0.99, D**2 * 2.0 / 9.0),  # Stokes settling
        # Inside each blend, where the two regimes differ, so the blend centre is pinned.
        (0.05, _blend(_bkc(0.05), _rg(0.05), 0.05, 0.0769452, 0.02)),
        (0.75, _blend(_rg(0.75), D**2 * 2.0 / 9.0, 0.75, 0.771462, 0.05)),
    ],
)
def test_mobility_function_reduces_to_each_regime(phi, expected):
    """``F`` is the closed form of each regime away from the blends and their tanh blend inside."""
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
    t, y = s._solution.t[-1], s._solution.y[:, -1]
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
        for name in ('porosity_b', 'rho_solid_b', 'rho_melt_b', 'g_b'):
            assert ds[name].dimensions == ('basic',)
            np.testing.assert_array_equal(ds[name][:], getattr(out, name))


@needs_eos
def test_porosity_is_the_clip_value_at_the_solidus(shared_eos):
    """Where the node density equals ``rho_s`` (at the solidus) porosity is the clip value 4.9975e-4."""
    s = _solve(shared_eos, 1800.0)
    out = s.get_state()
    assert np.max(out.phi_basic) == 0.0 and np.min(out.porosity_b) >= -2.5e-7
    ph = s.state.phase_basic
    ph.entropy = shared_eos.solidus_entropy(np.asarray(ph.pressure))
    ph.update()
    np.testing.assert_allclose(
        ph.porosity(), 1.0 - (0.9995 + np.sqrt(0.9995**2 + 1e-6)) / 2, rtol=1e-6
    )


@pytest.mark.parametrize(
    'rho_s,rho_l,rho,expected',
    [
        (3300.0, 2800.0, 3150.0, 0.3),  # mush: (rho_s - rho) / (rho_s - rho_l)
        (3300.0, 2800.0, 4.0e5, -2.5e-7),  # far below raw 0: the lower clip limit
        (3000.0, 3200.0, 2999.7, 0.3),  # denser melt: denominator floored at 1 kg/m^3
        (3000.0, 3200.0, 2990.0, 1.0),
        (3000.0, 3200.0, 3010.0, 0.0),
    ],
)
def test_porosity_formula_and_its_dense_melt_band(rho_s, rho_l, rho, expected):
    """The documented porosity: the density ratio, and with denser melt a 1 kg/m^3 wide band."""
    ph = EntropyPhaseEvaluator(entropy_eos=None, gravitational_acceleration=10.0)
    ph._density = np.array([rho])
    np.testing.assert_allclose(ph._porosity_from(rho_s, rho_l), [expected], rtol=0.0, atol=1e-5)


@needs_eos
def test_dense_melt_keeps_the_sign_of_the_density_contrast(shared_eos, monkeypatch):
    """Where the melt is denser the exports keep ``rho_solid_b < rho_melt_b``; a mushy node
    there is denser than ``rho_solid_b``, so its porosity sits at the lower clip limit."""
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


@needs_eos
def test_gravity_export_follows_an_external_profile(shared_eos):
    """``g_b`` is the external gravity profile interpolated to the basic nodes, not a scalar."""
    r = np.linspace(3.480e6, 6.371e6, 50)
    g = 9.0 + 2.0 * ((r[-1] - r) / (r[-1] - r[0])) ** 3
    out = _solve(
        shared_eos, _pick_mushy_S(shared_eos), mesh={'eos_radius': r, 'eos_gravity': g}
    ).get_state()
    np.testing.assert_allclose(out.g_b, PchipInterpolator(r, g)(out.r_basic), rtol=1e-12)


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
