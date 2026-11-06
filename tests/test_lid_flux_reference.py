"""The flux lid against a resolved crust: 1-D conduction with a moving base, constant properties.

The reference itself is checked against the constant-flux half-space solution.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from aragog import surface_lid as sl

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(900)]

RHO, CP, K, T_B, DW = 3000.0, 1200.0, 4.0, 1500.0, 3.5e9
KAPPA = K / (RHO * CP)
YR = 3.15576e7
HV = sl.LidEnthalpy.linear(RHO, CP, T_B, 0.0)


def _lid(d0, F_atm, F_b, T_s0, times):
    """``(delta, T_s)`` of the 2 lid states at ``times`` [yr]."""

    def rhs(_t, y):
        surf = sl.flux_surface_state(y[0], y[1], T_B, K, F_atm, sl.LID_DELTA_FLOOR, HV)
        r = sl.lid_rates(
            delta=y[0],
            E_lid=y[1],
            surf=surf,
            F_b=F_b,
            F_grid=F_atm,
            w=sl.flux_weight(F_b, T_B, surf.T_s, K, 1.0, 255.0, 1.0),
            rho=RHO,
            H_f=DW,
            T_b=T_B,
            H_b=0.0,
            hv=HV,
            h_top=1.0e7,
            delta_lo=0.5,
            heating=0.0,
        )
        return [r.ddelta, r.dE]

    E0 = d0 * float(HV.profile_heat(0.0, T_s0, T_B))
    t = np.asarray(times) * YR
    sol = solve_ivp(
        rhs, (0.0, t[-1]), [d0, E0], t_eval=t, method='Radau', rtol=1e-8, atol=[1e-4, 1.0]
    )
    assert sol.status == 0
    surf = (
        sl.flux_surface_state(d, E, T_B, K, F_atm, sl.LID_DELTA_FLOOR, HV)
        for d, E in zip(sol.y[0], sol.y[1])
    )
    return sol.y[0], np.array([v.T_s for v in surf])


def _resolved(d0, F_atm, F_b, T_s0, times, n=200):
    """``(delta, T_s)`` of the conduction equation on ``xi = z / delta`` with a Stefan base."""
    dx = 1.0 / n
    xi = (np.arange(n) + 0.5) * dx
    pattern = lil_matrix((n + 1, n + 1))
    for i in range(n):
        pattern[i, max(i - 1, 0) : min(i + 2, n)] = 1
        pattern[i, n] = 1
    pattern[n, n - 1] = pattern[n, n] = 1

    def surface(y):
        return y[0] - F_atm * y[-1] * dx / (2.0 * K)

    def rhs(_t, y):
        T, d = y[:-1], y[-1]
        dd = (K * (T_B - T[-1]) / (d * dx / 2.0) - F_b) / DW
        edge = np.concatenate(([surface(y)], 0.5 * (T[1:] + T[:-1]), [T_B]))
        grad = np.concatenate(
            (
                [(T[0] - surface(y)) / (dx / 2.0)],
                (T[1:] - T[:-1]) / dx,
                [(T_B - T[-1]) / (dx / 2.0)],
            )
        )
        rate = KAPPA / d**2 * np.diff(grad) / dx + xi * (dd / d) * np.diff(edge) / dx
        return np.append(rate, dd)

    t = np.asarray(times) * YR
    y0 = np.append(T_s0 + (T_B - T_s0) * xi, d0)
    sol = solve_ivp(
        rhs,
        (0.0, t[-1]),
        y0,
        t_eval=t,
        method='BDF',
        rtol=1e-7,
        atol=1e-6,
        jac_sparsity=pattern.tocsr(),
    )
    assert sol.status == 0
    return sol.y[-1], np.array([surface(y) for y in sol.y.T])


def test_reference_follows_the_half_space_after_a_flux_step():
    """``dT_s = (2 dF / k) sqrt(kappa t / pi)`` before the base feels the step."""
    F0, F1, d0 = 1.0, 3.0, 1000.0
    T_s0 = T_B - F0 * d0 / K
    times = np.array([1.0, 10.0, 100.0, 1000.0])
    _, T_s = _resolved(d0, F1, F0, T_s0, times, n=800)
    want = T_s0 - 2.0 * (F1 - F0) / K * np.sqrt(KAPPA * times * YR / np.pi)
    np.testing.assert_allclose(T_s, want, atol=0.1)


@pytest.mark.parametrize(
    'F_atm, F_b, times',
    [(1.5, 1.0, [1.0e4, 1.0e5, 3.0e5]), (0.5, 1.0, [1.0e4, 1.0e5]), (1.0, 1.0, [1.0e5])],
    ids=['growth', 'melt-back', 'steady'],
)
def test_lid_thickness_and_surface_follow_the_resolved_crust(F_atm, F_b, times):
    """Thickness change within 5 % and surface temperature within 2 % of the resolved crust."""
    d0 = 1000.0
    T_s0 = T_B - F_atm * d0 / K
    d_lid, T_lid = _lid(d0, F_atm, F_b, T_s0, times)
    d_ref, T_ref = _resolved(d0, F_atm, F_b, T_s0, times)
    np.testing.assert_allclose(d_lid - d0, d_ref - d0, rtol=0.05, atol=1.0e-3)
    np.testing.assert_allclose(T_lid, T_ref, rtol=0.02)
