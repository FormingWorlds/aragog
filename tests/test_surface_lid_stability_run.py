"""The density sign and the Rayleigh-Taylor time scales of the lid in the solver output."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import Julian_year

from aragog import surface_lid as sl
from aragog.rheology import R_GAS
from tests.conftest import assert_legend_clear, entropy_eos_copy, opaque_legend
from tests.reference.surface_lid_v2_column import LOWT_EOS_MD5, molten_start, shell
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos
from tools.verification.run_surface_lid_case import _install_factory
from tools.verification.run_surface_lid_stability import FIELDS, MOLTEN_TIMES_YR

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(600), needs_eos]

REF = Path(__file__).resolve().parent / 'reference' / 'surface_lid_stability.json'
TIMES = ('lid_L', 'lid_tau_rt', 'lid_tau_th', 'lid_L_crit', 'lid_rt_flag')
P_SHELL = 55.7e6  # top-cell pressure [Pa] of the shell
T_HOT = 1350.0  # start of a solid column above the fixed lid base at 1300 K


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


def _lowt_table():
    """The extended PALEOS table named by ``ARAGOG_LOWT_EOS_DIR``, checked by md5, or None."""
    table = os.environ.get('ARAGOG_LOWT_EOS_DIR')
    if table:
        md5 = hashlib.md5((Path(table) / 'temperature_solid.dat').read_bytes()).hexdigest()
        assert md5 == LOWT_EOS_MD5
    return table


def _hot_shell(eos, factory=False, end=5.0, **kw):
    """The solid shell above its fixed lid base, so the base is colder than the fluid; solved."""
    s = shell(10, True, eos, end, **kw)
    s.set_initial_entropy(
        np.array([eos.entropy_at_temperature(float(P), T_HOT) for P in s._P_stag_flat])
    )
    if factory:
        s.parameters.energy.solver_method = 'cvode'
        _install_factory(s, EOS_DIR)
    s.solve()
    assert s._solution.status == 0
    return s


def test_a_solid_lid_base_colder_than_the_fluid_gives_each_field(eos):
    """With the fixed lid base 50 K below the fluid, the base is denser than the fluid and each
    output field follows from the lid state, in the units of its description: the viscosity at
    the base from the top-cell viscosity law, the conduction time with the solid conductivity
    and the density and heat capacity of the base."""
    s = _hot_shell(eos)
    o, ev = s.get_state(), s._lid_last
    P = s._lid_P
    rho_b = float(np.asarray(eos.density(P, ev.S_b)).item())
    rho_f = float(np.asarray(eos.density(P, ev.S_f)).item())
    assert ev.S_b < ev.S_f
    assert o.lid_drho == pytest.approx(rho_b - rho_f, rel=1e-14)
    assert o.lid_drho > 0.0  # colder is denser: a sign flip fails here
    ph = s.state.phase_staggered
    T_sol = float(np.ravel(ph._solidus_temperature)[-1])
    H = s._solid_rheology().g_homologous * R_GAS * T_sol
    alpha = ev.surf.q_b / s._lid_k
    L = R_GAS * ev.T_b**2 / (H * alpha)
    assert o.lid_L == pytest.approx(L, rel=1e-12)
    cp = float(np.asarray(eos.heat_capacity(P, ev.S_b)).item())
    kappa = float(ph._k_solid) / (rho_b * cp)
    assert o.lid_tau_th == pytest.approx(L * L / kappa / Julian_year, rel=1e-12)
    g = abs(float(np.ravel(ph.gravitational_acceleration())[-1]))
    eta0 = o.lid_tau_rt * Julian_year * sl.LID_RT_PSI0 * o.lid_drho * g * L
    assert o.lid_L_crit == pytest.approx(
        (eta0 * kappa / (o.lid_drho * g)) ** (1 / 3), rel=1e-12
    )
    T_f = float(np.ravel(ph.temperature())[-1])
    eta_f = float(np.ravel(ph._eta_diff)[-1])
    assert eta0 == pytest.approx(eta_f * math.exp(H / R_GAS * (1 / ev.T_b - 1 / T_f)), rel=1e-9)
    assert o.lid_rt_flag == (1.0 if o.lid_tau_th > o.lid_tau_rt else 0.0)


def test_the_activation_energy_law_sets_the_layer(eos):
    """With ``activation_energy`` set, the layer takes that enthalpy, not the homologous one."""
    from dataclasses import replace

    s = _hot_shell(eos)
    base = s.get_state()
    H = (
        s._solid_rheology().g_homologous
        * R_GAS
        * float(np.ravel(s.state.phase_staggered._solidus_temperature)[-1])
    )
    rheo = replace(s.parameters.phase_solid.rheology, activation_energy=0.5 * H)
    s.parameters.phase_solid.rheology = rheo
    s._lid_rheo = rheo
    other = s._lid_stability(s._lid_last)
    assert other['lid_L'] == pytest.approx(2.0 * base.lid_L, rel=1e-9)


def test_the_jax_path_gives_the_same_fields(eos):
    """A solve on the JAX right-hand side reports the same fields as one on the NumPy path, within
    the 2 integrations: at most 7.6e-6 here."""
    a, b = _hot_shell(eos).get_state(), _hot_shell(eos, factory=True).get_state()
    assert a.lid_drho > 0.0
    assert b.lid_delta != a.lid_delta  # bitwise equal means the JAX path did not run
    for name in FIELDS:
        assert getattr(b, name) == pytest.approx(getattr(a, name), rel=1e-4), name


def test_without_the_rheology_only_the_density_difference_is_written(eos, tmp_path):
    """No creep law exists without the solid rheology, so the time scales stay NaN. The density
    difference stays, in the netCDF snapshot too, negative for a lid on melt on these tables."""
    import netCDF4

    s = shell(10, True, eos, 0.1, rheology=False, convection=True)
    s.set_initial_entropy(np.asarray(molten_start(s, eos)))
    s.solve()
    o, ev = s.get_state(), s._lid_last
    expected = ev.rho - float(np.asarray(eos.density(s._lid_P, ev.S_f)).item())
    assert o.lid_drho == pytest.approx(expected, rel=1e-14)
    assert o.lid_drho < 0.0
    assert all(math.isnan(getattr(o, k)) for k in TIMES)
    o.to_netcdf(tmp_path / 'on.nc', time=1.0)
    with netCDF4.Dataset(tmp_path / 'on.nc') as ds:
        assert float(ds['lid_drho'][...]) == pytest.approx(o.lid_drho, rel=1e-12)
        assert ds['lid_tau_rt'].units == 'yr'
        assert np.ma.is_masked(ds['lid_tau_rt'][...]) or math.isnan(
            float(ds['lid_tau_rt'][...])
        )


def test_with_the_lid_off_or_handed_over_every_field_is_nan(eos, tmp_path):
    """With the lid off no field exists, also not in the netCDF snapshot; after a hand-over the
    thickness is 0 and the new fields are NaN."""
    import netCDF4

    from tests.test_subgrid_lid_solver import _cold_top_solver

    off = shell(10, False, eos, 0.1)
    off.set_initial_entropy(
        np.array([eos.entropy_at_temperature(float(P), T_HOT) for P in off._P_stag_flat])
    )
    off.solve()
    o = off.get_state()
    assert all(math.isnan(getattr(o, k)) for k in FIELDS)
    o.to_netcdf(tmp_path / 'off.nc', time=1.0)
    with netCDF4.Dataset(tmp_path / 'off.nc') as ds:
        assert 'lid_drho' not in ds.variables
    s = _cold_top_solver(eos)
    s.solve()
    h = s.get_state()
    assert h.lid_delta == pytest.approx(0.0, abs=0)
    assert all(math.isnan(getattr(h, k)) for k in FIELDS)


def test_a_hidden_lid_has_no_time_scales(eos):
    """A lid with weight 0 is a hidden state: the density difference is written, the times are
    not, even when the base is denser than the fluid."""
    s = _hot_shell(eos)
    s.get_state()
    ev = s._lid_last
    assert ev.rates.w > 0.0
    assert math.isfinite(s._lid_stability(ev)['lid_L'])
    from dataclasses import replace

    hidden = replace(ev, rates=replace(ev.rates, w=0.0))
    out = s._lid_stability(hidden)
    assert out == {'lid_drho': pytest.approx(s._lid_stability(ev)['lid_drho'], rel=1e-15)}


def test_a_constant_property_lid_has_no_density_difference():
    """A column of one density gives a difference of exactly 0, so the times stay NaN."""
    from tests.test_subgrid_lid_const import _solved

    o = _solved().get_state()
    assert o.lid_drho == pytest.approx(0.0, abs=0)
    assert all(math.isnan(getattr(o, k)) for k in TIMES)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('table', 'expected'),
    [
        (
            'test',
            {'liquidus': -570.4, 'mid': -260.7, 'solid_300': 40.7, 'solid_600': 77.5},
        ),
        (
            'lowt',
            {'liquidus': -562.0, 'mid': -257.2, 'solid_300': 40.7, 'solid_600': 77.4},
        ),
    ],
)
def test_the_sign_of_the_density_difference_per_table(table, expected):
    """At the top-cell pressure of the shell, on these tables, a base at the solidus is lighter
    than melt or mush below it, and a solid base is denser than warmer solid only thermally.
    New tables that flip a sign fail here. The extended table needs ``ARAGOG_LOWT_EOS_DIR``."""
    folder = EOS_DIR if table == 'test' else _lowt_table()
    if not folder:
        pytest.skip('ARAGOG_LOWT_EOS_DIR is not set')
    eos = entropy_eos_copy(folder)
    S_sol = float(np.asarray(eos.solidus_entropy(P_SHELL)).item())
    S_liq = float(np.asarray(eos.liquidus_entropy(P_SHELL)).item())

    def drho(S_b, S_f):
        rho = (float(np.asarray(eos.density(P_SHELL, S)).item()) for S in (S_b, S_f))
        return next(rho) - next(rho)

    got = {
        'liquidus': drho(S_sol, S_liq),
        'mid': drho(S_sol, 0.5 * (S_sol + S_liq)),
        'solid_300': drho(S_sol - 300.0, S_sol - 50.0),
        'solid_600': drho(S_sol - 600.0, S_sol - 50.0),
    }
    assert got == pytest.approx(expected, abs=0.06)


@pytest.mark.physics_invariant
def test_s4_molten_shell_on_the_test_tables():
    """S4, again by a run: a molten shell with the solid rheology on the test tables has a lid
    lighter than the melt, flag NaN, as long as the top cell is melt."""
    eos, y, t_prev = entropy_eos_copy(EOS_DIR), None, 0.0
    for t in MOLTEN_TIMES_YR[:4]:
        s = shell(10, True, eos, float(t), t_prev, rheology=True, convection=True)
        if y is not None:
            s.set_initial_lid(float(y[-2]), float(y[-1]))
        s.set_initial_entropy(np.asarray(molten_start(s, eos) if y is None else y[: s._n_stag]))
        s.solve()
        y, t_prev = np.asarray(s._solution.y[:, -1]), float(t)
        o = s.get_state()
        assert o.phi_stag[-1] == pytest.approx(1.0, abs=1e-6)
        assert -600.0 < o.lid_drho < -400.0
        assert math.isnan(o.lid_rt_flag)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_s4_stored_runs_on_the_extended_table(validation_plot):
    """S4 on the stored runs of ``tools/verification/run_surface_lid_stability.py`` on the extended
    table. In the hand-over column the lid-base entropy is capped at the fluid entropy, so the
    difference is 0 in 718 of 812 rows and the flag is NaN there; in the other 94, at most 18.3
    kg/m^3, the growth time stays at least 41 times above the conduction time, so no row sets
    the flag. In the molten shell the lid is lighter than the melt, by 417 to 547 kg/m^3."""
    d = json.loads(REF.read_text())
    assert d['meta']['md5'] == LOWT_EOS_MD5
    solid = {k: np.array(v, dtype=float) for k, v in d['solid'].items()}
    assert (d['solid']['rows'], d['solid']['rows_denser'], d['solid']['rows_flag_1']) == (
        812,
        94,
        0,
    )
    dense = solid['lid_drho'] > 0.0
    assert np.all(solid['lid_drho'] >= 0.0)
    assert np.all(np.isnan(solid['lid_rt_flag'][~dense]))
    assert np.all(solid['lid_rt_flag'][dense] == 0.0)
    assert solid['lid_drho'].max() == pytest.approx(18.26, abs=0.5)
    assert 2.1e5 < solid['t'][dense].min() and solid['t'][dense].max() < 6.9e5
    assert np.min(solid['lid_tau_rt'][dense] / solid['lid_tau_th'][dense]) > 41.0
    ratio = solid['lid_L'][dense] / solid['delta'][dense]
    assert 0.1 < np.max(ratio) < 0.7  # Eq. 7 of the paper asks for L well below the thickness
    melt = {k: np.array(v, dtype=float) for k, v in d['molten'].items()}
    early = melt['t'] < 5.0
    assert np.all((-550.0 < melt['lid_drho'][early]) & (melt['lid_drho'][early] < -410.0))
    assert np.all(np.isnan(melt['lid_rt_flag']))

    def figure():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=False)
        ax = axes[0]
        ax.plot(melt['t'], melt['lid_drho'])
        ax.set_xscale('log')
        ax.set_xlabel('time, molten shell [yr]')
        ax.set_ylabel(r'$\Delta\rho$ [kg/m$^3$]')
        ax = axes[1]
        t = solid['t'][dense]
        ax.plot(t, solid['lid_tau_rt'][dense], label=r'$\tau_{RT}$')
        ax.plot(t, solid['lid_tau_th'][dense], label=r'$\tau_{th}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(top=ax.get_ylim()[1] * 300)
        ax.set_xlabel('time, solid column [yr]')
        ax.set_ylabel('time scale [yr]')
        opaque_legend(ax, loc='upper center', ncols=2)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_rt_runs')
