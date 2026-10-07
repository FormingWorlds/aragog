"""The resolved stable layer of the core: a conducting shell with convective mixing over the
convecting core, in its analytic limits and its heat balance."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erfc

from aragog.core import GaussianCoreProfiles
from aragog.core.layer import CoreShell
from aragog.core.stratification import _q_ad

pytestmark = pytest.mark.unit
MYR = 1e6 * 365.25 * 86400.0
T_C, K_CORE = 4400.0, 130.0


def _shell(**kw):
    profiles = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=139e9,
        alpha=1.25e-5,
        c_p=840.0,
        pressure_mode='quadrature',
    )
    return CoreShell(profiles, K_CORE, **kw)


def _evolve(shell, q_cmb, t_end):
    """Shell temperatures after ``t_end`` [s] under ``q_cmb`` with the convecting core held."""
    rate = jax.jit(lambda y: shell.rates(y, T_C, q_cmb)[0])
    jac = jax.jit(jax.jacfwd(lambda y: shell.rates(y, T_C, q_cmb)[0]))
    y0 = np.asarray(shell.adiabatic_profile(T_C))
    sol = solve_ivp(
        lambda _, y: np.asarray(rate(y)),
        (0.0, t_end),
        y0,
        'BDF',
        jac=lambda _, y: np.asarray(jac(y)),
        rtol=1e-9,
        atol=1e-9,
    )
    assert sol.success
    return sol.y[:, -1], y0


@pytest.mark.physics_invariant
def test_the_shell_heat_changes_by_the_flows_at_its_faces():
    shell, t = _shell(), np.linspace(4380.0, 4500.0, 64) + 0.3 * np.arange(64) ** 1.5
    rates, q_base = shell.rates(t, T_C, 8e12, heating=1e-11)
    content = float(shell.profiles.c_p * np.sum(np.asarray(shell.mass) * np.asarray(rates)))
    expected = float(q_base) - 8e12 + 1e-11 * float(np.sum(shell.mass))
    assert content == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.reference_pinned
@pytest.mark.parametrize('t_myr', [0.03, 0.1])
def test_a_young_layer_follows_the_erfc_solution_of_a_half_space(t_myr):
    """A fixed flow deficit Q_k - q at the top of a conducting half space raises the
    temperature by (2 dF / k) sqrt(kappa t) ierfc(z / 2 sqrt(kappa t)) (Carslaw and Jaeger,
    the planar limit of Greenwood et al. 2021, eq. 27); the cells also cool at the rate the
    divergence of the adiabatic conduction sets. Earlier the top cell limits the match, later
    the curvature of the sphere. Weak mixing isolates the conduction: with the convecting
    core held, the sink would make the cells below the front mix."""
    shell = _shell(k_mix=1e-3)
    p, t = shell.profiles, t_myr * MYR
    q_k = float(_q_ad(p, K_CORE, p.r_cmb, T_C))
    t_shell, y0 = _evolve(shell, 8e12, t)
    sink = np.asarray(shell.rates(y0, T_C, q_k)[0]) * t
    length = np.sqrt(K_CORE / (float(p.density(p.r_cmb)) * p.c_p) * t)
    x = (p.r_cmb - np.asarray(shell.r_cells)) / (2.0 * length)
    ierfc = np.exp(-(x**2)) / np.sqrt(np.pi) - x * erfc(x)
    deficit = (q_k - 8e12) / (4.0 * np.pi * p.r_cmb**2)
    expected = 2.0 * deficit / K_CORE * length * ierfc + sink
    near = x < 3.0
    assert np.max(np.abs(t_shell - y0 - expected)[near]) < 1e-2 * expected[-1]


@pytest.mark.reference_pinned
@pytest.mark.parametrize('q_tw', [8, 12])
def test_a_steady_layer_reaches_the_quasi_static_depth(q_tw):
    """With the convecting core held, the layer settles where the conducted adiabatic flow
    equals the CMB flow."""
    shell = _shell()
    p = shell.profiles
    t_shell, _ = _evolve(shell, q_tw * 1e12, 30e3 * MYR)
    r_s = brentq(lambda r: float(_q_ad(p, K_CORE, r, T_C)) - q_tw * 1e12, 1e5, p.r_cmb)
    depth = p.r_cmb - float(shell.layer_base(t_shell, T_C))
    assert depth == pytest.approx(p.r_cmb - r_s, rel=0.05)


@pytest.mark.physics_invariant
def test_mixing_keeps_a_superadiabatic_shell_on_the_adiabat():
    """Above the conducted adiabatic flow the shell mixes: the departure from the adiabat has a
    gradient below 1e-3 of the adiabatic one, and no layer forms."""
    shell = _shell()
    t_shell, y0 = _evolve(shell, 20e12, 1.0 * MYR)
    theta = t_shell - y0
    gradient = np.abs(np.diff(theta) / np.diff(np.asarray(shell.r_cells)))
    adiabatic = 2.0 * shell.profiles.r_cmb * T_C / shell.profiles.d_scale**2
    assert gradient.max() < 1e-3 * adiabatic
    assert float(shell.layer_base(t_shell, T_C)) == pytest.approx(shell.profiles.r_cmb, abs=1e3)


def test_the_shell_refuses_a_bad_geometry():
    with pytest.raises(ValueError, match='base_fraction'):
        _shell(base_fraction=1.0)
    with pytest.raises(ValueError, match='n_cells'):
        _shell(n_cells=1)
