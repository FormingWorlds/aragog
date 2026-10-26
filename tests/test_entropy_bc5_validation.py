"""Validation of the prescribed surface temperature (outer boundary condition 5) against
closed-form conduction solutions, on the const_properties path (k 4 W/m/K, rho c_p 4e6 J/m3/K,
so kappa = 1e-6 m2/s) with convection off and a uniform radius mesh."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Julian_year
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.sparse import diags

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(300)]

K, RHO, CP, T_REF, S_REF = 4.0, 4000.0, 1000.0, 3000.0, 3000.0  # the builder's const_properties
KAPPA = K / (RHO * CP)
R_OUT = 6.371e6


def _entropy(t):
    return S_REF + CP * np.log(np.asarray(t) / T_REF)


def _temperature(s):
    return T_REF * np.exp((np.asarray(s) - S_REF) / CP)


def _solver(n_cells, r_in, t_surf, heating=0.0, convection=False, profile='nearest_boundary'):
    """Zero CMB flux, uniform radius mesh, uniform heating [W/kg], no kappa_h floor; with
    convection the viscosity is 1e21 Pa s."""
    p = _build_const_properties_parameters(n_nodes=n_cells + 1)
    b = p.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, t_surf
    b.inner_boundary_condition, b.inner_boundary_value = 2, 0.0
    p.mesh.inner_radius, p.mesh.outer_radius, p.mesh.mass_coordinates = r_in, R_OUT, False
    p.energy.convection, p.energy.kappah_floor = convection, 0.0
    p.mesh.mixing_length_profile, p.phase_mixed.const_log10visc = profile, 21.0
    if heating:
        p.energy.tidal, p.energy.tidal_array = True, [heating]
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    return s


def _rhs(s):
    return lambda t, y: np.asarray(s.dSdt(t, y)).ravel()


# Case 1: steady heated shell, Turcotte & Schubert (2014), section 4.9, eq. 4.40.
R_IN_1, T_S_1 = 3.48e6, 1000.0
H_1 = 3.4e-13  # [W/kg]; rho H = 1.36e-9 W/m3 gives a 990 K rise at the CMB


def _shell_exact(r):
    q = RHO * H_1
    return (
        T_S_1 + q * (R_OUT**2 - r**2) / (6 * K) + q * R_IN_1**3 * (1 / R_OUT - 1 / r) / (3 * K)
    )


def _steady_shell(n_cells):
    """Root of the right-hand side, scaled by the heating term H / T per year."""
    s = _solver(n_cells, R_IN_1, T_S_1, heating=H_1)
    r = np.asarray(s._r_stag_flat)
    scale = H_1 / _shell_exact(r) * Julian_year
    sol = root(lambda y: _rhs(s)(0.0, y) / scale, _entropy(_shell_exact(r)), tol=1e-14)
    assert np.max(np.abs(_rhs(s)(0.0, sol.x) / scale)) < 1e-9
    return s, r, _temperature(sol.x)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_steady_heated_shell_matches_the_closed_form(validation_plot):
    """Uniform heating, zero flux at r_in and T_s at r_out: T(r) = T_s + rho H (r_out^2 - r^2)
    / (6k) + rho H r_in^3 (1/r_out - 1/r) / (3k) (Turcotte and Schubert 2014, eq. 4.40 with
    dT/dr = 0 at r_in). At steady state the surface flux is rho H (r_out^3 - r_in^3) / (3 r_out^2)
    for every N, since the cell volumes are exact shell volumes, so the flux checks the energy
    balance; the temperature error falls as N^-2, and a full-cell face drops the order to 1."""
    f_exact = RHO * H_1 * (R_OUT**3 - R_IN_1**3) / (3 * R_OUT**2)
    rise = _shell_exact(R_IN_1) - T_S_1
    errors, runs = [], {}
    for n in (50, 100, 200):
        s, r, t = _steady_shell(n)
        f_top = float(np.ravel(s.state.heat_flux)[-1])
        assert f_top == pytest.approx(f_exact, rel=1e-8)
        errors.append(np.max(np.abs(t - _shell_exact(r))) / rise)
        runs[n] = (r, t)
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(np.abs(orders - 2.0) <= 0.2), orders

    def figure():
        import matplotlib.pyplot as plt

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 4.8))
        r, t = runs[50]
        rr = np.linspace(R_IN_1, R_OUT, 60)
        top.plot(rr / 1e3, _shell_exact(rr), label='eq. 4.40')
        top.plot(r / 1e3, t, 'o', ms=3, label='Aragog')
        top.set(xlabel='r [km]', ylabel='T [K]')
        top.legend()
        n = np.array([50, 100, 200])
        bottom.plot(n, n**2 * np.asarray(errors), 'o-')
        bottom.set(xlabel='N', ylabel='N$^2$ error', ylim=(0, 1.3 * n[-1] ** 2 * errors[-1]))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'bc5_steady_heated_shell')


# Case 2: cooling below a fixed surface temperature, Turcotte & Schubert (2014), eq. 4.116.
D_2, N_2, T_I_2, T_S_2 = 25e3, 400, 1100.0, 1000.0
T_WINDOW = np.geomspace(5.3e10, 7.5e12, 16) / Julian_year  # [yr]


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_cooling_below_a_fixed_surface_temperature_follows_the_half_space_flux(validation_plot):
    """From a uniform T_i, the surface flux is k (T_i - T_s) / sqrt(pi kappa t) (Turcotte and
    Schubert 2014, eq. 4.116) to 1 % over 2.2 decades: the half-cell error 0.063 (h/delta)^2 is
    4.7e-3 at the window start (delta = sqrt(kappa t) = 3.7 h), the curvature of the shell
    lowers the flux by sqrt(pi kappa t) / r_out, at most 7.6e-4, and the bottom at 9 delta stays
    unseen. The signed error follows the sum of the two estimates to 5e-4."""
    s = _solver(N_2, R_OUT - D_2, T_S_2)
    y0 = np.full(N_2, _entropy(T_I_2))
    band = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(N_2, N_2))
    sol = solve_ivp(
        _rhs(s),
        (0.0, T_WINDOW[-1]),
        y0,
        method='BDF',
        t_eval=T_WINDOW,
        rtol=1e-8,
        atol=1e-8,
        jac_sparsity=band,
    )
    assert sol.success, sol.message
    flux = []
    for t, y in zip(sol.t, sol.y.T):
        _rhs(s)(t, y)
        flux.append(float(np.ravel(s.state.heat_flux)[-1]))
    flux = np.asarray(flux)
    seconds = sol.t * Julian_year
    exact = K * (T_I_2 - T_S_2) / np.sqrt(np.pi * KAPPA * seconds)
    error = flux / exact - 1.0
    assert np.max(np.abs(error)) < 1e-2, error
    # the error follows the estimate 0.063 (h/delta)^2 - sqrt(pi kappa t)/r_out to 6.7e-5; a
    # full-cell face misses it by 9e-3
    h = D_2 / N_2
    estimate = 0.063 * h**2 / (KAPPA * seconds) - np.sqrt(np.pi * KAPPA * seconds) / R_OUT
    assert np.max(np.abs(error - estimate)) < 5e-4

    def figure():
        import matplotlib.pyplot as plt

        fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(5.6, 4.8))
        top.loglog(seconds, exact, label='eq. 4.116')
        top.loglog(seconds, flux, 'o', ms=3, label='Aragog')
        top.set(ylabel='F [W/m$^2$]')
        top.legend()
        bottom.semilogx(seconds, error, '.', ms=4)
        for sign in (1, -1):
            bottom.axhline(sign * 1e-2, color=plt.rcParams['text.color'], ls='--')
        bottom.set(xlabel='Time [s]', ylabel='Rel. error')
        for ax in (top, bottom):
            ax.minorticks_off()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'bc5_half_space_cooling')


# Case 3: the heated shell of case 1 with mixing-length convection, grid convergence.
R_FIX = np.array([3.6e6, 4.5e6, 5.5e6, 6.2e6, 6.3e6])  # [m], radii of the compared temperatures


def _steady_convecting_shell(n_cells, profile, guess):
    """Steady state by BDF to 1e13 yr; convection with log10 viscosity 21 Pa s."""
    s = _solver(n_cells, R_IN_1, T_S_1, heating=H_1, convection=True, profile=profile)
    r = np.asarray(s._r_stag_flat)
    band = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n_cells, n_cells))
    sol = solve_ivp(
        _rhs(s), (0.0, 1e13), guess(r), method='BDF', rtol=1e-10, atol=1e-10, jac_sparsity=band
    )
    assert sol.success, sol.message
    y = sol.y[:, -1]
    scale = H_1 / _shell_exact(r) * Julian_year
    assert np.max(np.abs(_rhs(s)(0.0, y) / scale)) < 1e-6
    return s, r, _temperature(y)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_convecting_shell_converges_with_second_order(validation_plot):
    """With mixing-length convection (nearest_boundary and constant mixing length) the steady
    surface flux still equals the total heating over the surface area; the mean temperature and
    the temperature at 5 fixed radii converge with order 2.0 +- 0.4 between N = 100, 200 and 400,
    and the N = 400 values lie within 1e-3 of the rise from their Richardson extrapolation. The
    surface boundary layer (depth to 90 % of the mid-mantle rise) spans at least 8 cells."""
    from scipy.interpolate import CubicSpline

    f_exact = RHO * H_1 * (R_OUT**3 - R_IN_1**3) / (3 * R_OUT**2)
    curves = {}
    for profile in ('nearest_boundary', 'constant'):
        guess, rows = (lambda r: _entropy(_shell_exact(r))), []
        for n in (100, 200, 400):
            s, r, t = _steady_convecting_shell(n, profile, guess)
            assert float(np.ravel(s.state.heat_flux)[-1]) == pytest.approx(f_exact, rel=1e-8)
            vol = np.ravel(s.evaluator.mesh.basic.volume)
            rows.append(np.r_[np.sum(t * vol) / np.sum(vol), CubicSpline(r, t)(R_FIX)])
            guess = lambda rr, r=r, t=t: np.interp(rr, r, _entropy(t))  # noqa: E731
        mid = t[np.argmin(np.abs(r - 0.5 * (R_IN_1 + R_OUT)))]
        rise = t.max() - T_S_1
        depth = R_OUT - r
        layer = depth[
            (t - T_S_1 >= 0.9 * (mid - T_S_1)) & (depth < 0.5 * (R_OUT - R_IN_1))
        ].min()
        assert layer / ((R_OUT - R_IN_1) / 400) >= 8.0
        c1, c2, c4 = rows
        orders = np.log2(np.abs(c1 - c2) / np.abs(c2 - c4))
        assert np.all(np.abs(orders - 2.0) <= 0.4), orders
        assert np.max(np.abs(c4 - c2) / 3.0) / rise < 1e-3
        curves[profile] = (
            r,
            t,
            [abs(c[0] - (c4[0] + (c4[0] - c2[0]) / 3.0)) / rise for c in rows],
        )

    def figure():
        import matplotlib.pyplot as plt

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 4.8))
        n = np.array([100, 200, 400])
        for profile, (r, t, err) in curves.items():
            (line,) = top.plot(r / 1e3, t, label=profile.replace('_', ' '))
            bottom.plot(n, n**2 * np.asarray(err), 'o-', color=line.get_color())
        top.set(xlabel='r [km]', ylabel='T [K]')
        top.legend()
        bottom.set(xlabel='N', ylabel='N$^2$ error', ylim=(0, None))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'bc5_convecting_shell_convergence')
