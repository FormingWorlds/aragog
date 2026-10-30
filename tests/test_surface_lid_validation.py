"""Validation of the sub-grid lid closure against analytical solutions.

The cases drive the lid functions of ``aragog.surface_lid`` (the closure the solver calls each
right-hand side) through time with fixed material properties: the Neumann solution of crust
growth over melt, half-space cooling of a solid and the quasi-steady crust under a fixed flux.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erf

from aragog import surface_lid as sl
from tests.conftest import assert_legend_clear

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

RHO, CP, K = 3000.0, 1200.0, 4.0
KAPPA = K / (RHO * CP)
T_B, T_S = 1400.0, 300.0
YR = 3.15576e7


def _neumann_lambda(St):
    return brentq(lambda x: x * np.exp(x * x) * erf(x) - St / np.sqrt(np.pi), 1e-9, 10.0)


HV = sl.LidEnthalpy.linear(RHO, CP, T_B, CP * T_B)


def _energy(delta, T_m):
    return delta * float(HV.H(T_m))


def _evolve(y0, t0, t_out, *, dh, F_b, surface):
    """Integrate the lid states (delta, E_lid) with the lid functions; ``surface(delta, E)``
    returns the LidSurface. Returns delta, the surface solution and the base rates at t_out."""
    h_b = CP * T_B

    def state(y):
        surf = surface(y[0], y[1])
        rates = sl.lid_rates(
            delta=y[0],
            E_lid=y[1],
            surf=surf,
            F_b=F_b,
            F_grid=0.0,
            w=1.0,
            rho=RHO,
            H_f=RHO * (h_b + dh),
            T_b=T_B,
            H_b=RHO * h_b,
            hv=HV,
            h_top=1e12,
            delta_lo=1e-9,
            heating=0.0,
        )
        return surf, rates

    def rhs(t, y):
        return [state(y)[1].ddelta, state(y)[1].dE]

    sol = solve_ivp(
        rhs,
        (t0, t_out[-1]),
        y0,
        method='Radau',
        t_eval=t_out,
        rtol=1e-10,
        atol=[1e-12 * y0[0], 1e-12 * abs(y0[1])],
    )
    assert sol.status == 0
    return sol.y[0], [state(sol.y[:, i])[0] for i in range(sol.t.size)]


def _dirichlet(delta, E):
    return sl.fixed_surface_state(delta, E, T_B, K, T_S, 0.0, HV)


T_V1 = np.array([1e8, 1e10, 1e12])


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v1_crust_growth_follows_neumann(validation_plot):
    """V1: a lid over melt at its base temperature, no base flux and a fixed surface temperature
    grows as the Neumann solution delta = 2 lambda sqrt(kappa t), lambda exp(lambda^2) erf(lambda)
    = St / sqrt(pi) (Turcotte and Schubert 2014, section 4.18, eqs. 4.136 and 4.141, pp. 197-198),
    within 1 % at 1e8, 1e10 and 1e12 s for St = 0.1, 1 and 3.381. A lid started at 1 m on the
    linear profile approaches the same law. The linear-profile growth law sqrt(2 St kappa t)
    misses by more than the tolerance at each St (by 1.6, 14 and 37 %)."""
    results = {}
    for St in (0.1, 1.0, 3.381):
        lam = _neumann_lambda(St)
        dh = CP * (T_B - T_S) / St
        t0 = 1e6
        d0 = 2 * lam * np.sqrt(KAPPA * t0)
        T_m0 = T_S + sl.shape(lam**2)[2] * (T_B - T_S)
        E0 = _energy(d0, T_m0)
        d, _ = _evolve([d0, E0], t0, T_V1, dh=dh, F_b=0.0, surface=_dirichlet)
        ref = 2 * lam * np.sqrt(KAPPA * T_V1)
        np.testing.assert_allclose(d, ref, rtol=1e-2)
        # the similarity profile holds the Neumann growth exactly: 2.4e-12 in the run
        np.testing.assert_allclose(d, ref, rtol=1e-8)
        linear = np.sqrt(2 * St * KAPPA * T_V1)
        assert np.all(np.abs(linear / ref - 1) > 1e-2)
        t_lin = np.geomspace(1e5, 1e12, 30)
        E_lin = _energy(1.0, 0.5 * (T_S + T_B))
        d_lin, _ = _evolve([1.0, E_lin], 0.0, t_lin, dh=dh, F_b=0.0, surface=_dirichlet)
        err = np.abs(d_lin / (2 * lam * np.sqrt(KAPPA * t_lin)) - 1)
        assert err[-1] < 1e-2 and err[-1] < err[5]
        results[St] = (lam, d, d_lin, t_lin)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        for i, (St, (lam, d, d_lin, t_lin)) in enumerate(results.items()):
            c = f'C{i}'
            ax.axhline(2 * lam, color=c, lw=1.0, label=f'Neumann, St = {St:g}')
            ax.plot(T_V1 / YR, d / np.sqrt(KAPPA * T_V1), 'o', color=c, ms=6)
            ax.plot(t_lin / YR, d_lin / np.sqrt(KAPPA * t_lin), '--', color=c, lw=1.2)
        ink = plt.rcParams['text.color']
        ax.plot([], [], 'o', color=ink, ms=6, label='lid, similarity start')
        ax.plot([], [], '--', color=ink, lw=1.2, label='lid, linear start')
        ax.set_xscale('log')
        ax.set_xlabel('time (yr)')
        ax.set_ylabel(r'$\delta / \sqrt{\kappa t}$')
        ax.legend(fontsize=8)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v1_neumann_growth')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v2_half_space_cooling(validation_plot):
    """V2: without latent heat (the enthalpy jump at its floor) the lid is a cooling half space:
    the surface flux follows k (T_b - T_s) / sqrt(pi kappa t) (Turcotte and Schubert 2014,
    section 4.15, eq. 4.116, p. 189) within 1 % from three lid-diffusion times of the start on.
    Twice that flux, the most likely slip in the denominator, misses by far."""
    St = CP * (T_B - T_S) / sl.LID_DH_MIN
    lam = _neumann_lambda(St)
    t0 = 1e6
    d0 = 2 * lam * np.sqrt(KAPPA * t0)
    T_m0 = T_S + sl.shape(lam**2)[2] * (T_B - T_S)
    E0 = _energy(d0, T_m0)
    t = np.geomspace(3 * t0, 1e12, 25)
    _, surf = _evolve([d0, E0], t0, t, dh=0.0, F_b=0.0, surface=_dirichlet)
    F = np.array([s.F_s for s in surf])
    ref = K * (T_B - T_S) / np.sqrt(np.pi * KAPPA * t)
    np.testing.assert_allclose(F, ref, rtol=1e-2)
    assert np.all(np.abs(F / (2 * ref) - 1) > 0.4)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.axhspan(0.99, 1.01, color='C0', alpha=0.15, label='1 % band')
        ax.axhline(1.0, color='C0', lw=1.0)
        ax.semilogx(t / YR, F / ref, 'o', color='C1', ms=5, label='lid surface flux')
        ax.set_ylim(0.98, 1.02)
        ax.set_xlabel('time (yr)')
        ax.set_ylabel(r'$F_s \,/\, (k \Delta T / \sqrt{\pi \kappa t})$')
        ax.legend(fontsize=8)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v2_half_space')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_v3_quasi_steady_crust(validation_plot):
    """V3: under a fixed base flux and a grey surface the lid settles at the steady crust,
    delta = k (T_b - T_s) / F_b with grey(T_s) = F_b, on the linear profile (s = 0): within
    1e-4 after ten relaxation times rho dh delta^2 / (k (T_b - T_s)). A lid without the latent
    term in the growth rate would relax to the same depth, so the check is on the end state."""
    F_b, T_eq, dh = 100.0, 255.0, 4.0e5
    T_s_eq = (F_b / Stefan_Boltzmann + T_eq**4) ** 0.25
    d_eq = K * (T_B - T_s_eq) / F_b
    tau = max(RHO * dh * d_eq**2 / (K * (T_B - T_s_eq)), d_eq**2 / KAPPA)
    t = np.geomspace(1e-3 * tau, 10 * tau, 20)
    E0 = _energy(10.0, 0.5 * (T_B + T_S))

    def grey(delta, E):
        return sl.surface_state(delta, E, T_B, K, 1.0, T_eq, 0.0, HV)

    d, surf = _evolve([10.0, E0], 0.0, t, dh=dh, F_b=F_b, surface=grey)
    assert d[-1] == pytest.approx(d_eq, rel=1e-4)
    assert abs(surf[-1].s) < 1e-3
    assert surf[-1].q_b == pytest.approx(F_b, rel=1e-4)
    assert abs(d[0] / d_eq - 1) > 0.5

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.semilogx(t / tau, d / d_eq, '-', color='C0', label=r'$\delta / \delta_\mathrm{eq}$')
        ax.semilogx(
            t / tau, [s.q_b / F_b for s in surf], '--', color='C1', label=r'$q_b / F_b$'
        )
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=0.8)
        ax.set_xlabel('time / relaxation time')
        ax.set_ylabel('ratio to steady state')
        ax.legend(fontsize=8)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_v3_steady_crust')
