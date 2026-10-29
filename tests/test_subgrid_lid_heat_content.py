"""The lid heat content is the EOS enthalpy of the lid profile.

A lid state built as the EOS integral of a known profile, ``E = int_0^delta H(T(z)) dz`` with
``H(S) = int_{S_sol}^S rho T dS`` at the top-cell pressure (the per-volume enthalpy of the
merge), must give back that profile: its shape parameter and its surface temperature.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import erf, erfi

from aragog import surface_lid as sl
from tests.conftest import entropy_eos_copy
from tests.reference.surface_lid_v2_column import T_I, shell
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(300), needs_eos]

LOWT = os.environ.get('ARAGOG_LOWT_EOS_DIR')
TABLES = [
    pytest.param(
        LOWT, 255.0, id='extended', marks=pytest.mark.skipif(not LOWT, reason='no table')
    ),
    pytest.param(str(EOS_DIR), 450.0, id='test_tables'),
]


def _profile(s, x):
    if s > 0:
        return erf(np.sqrt(s) * x) / erf(np.sqrt(s))
    if s < 0:
        return erfi(np.sqrt(-s) * x) / erfi(np.sqrt(-s))
    return x


@pytest.mark.physics_invariant
@pytest.mark.parametrize('shape', [-2.0, 0.0, 1.5])
@pytest.mark.parametrize(('table', 'T_eq'), TABLES)
def test_lid_state_gives_back_its_eos_profile(table, T_eq, shape):
    eos = entropy_eos_copy(table)
    s = shell(10, True, eos, 1.0, T_eq=T_eq)
    P, S_sol = s._lid_P, s._lid_S_sol
    delta, k, T_b = 50.0, s._lid_k, T_I
    f0 = sl.shape(shape)[0]
    d = np.hypot(delta, sl.LID_DELTA_FLOOR)
    T_s = brentq(
        lambda T: Stefan_Boltzmann * (T**4 - T_eq**4) - k * (T_b - T) * f0 / d,
        T_eq,
        T_b,
        xtol=1e-12,
    )

    # H(S) on a dense entropy grid by 8-point Gauss-Legendre per interval, then a cubic spline
    S_lo = eos.entropy_at_temperature(P, T_eq - 1.0)
    edges = np.linspace(S_lo, S_sol, 4001)
    g8, w8 = np.polynomial.legendre.leggauss(8)
    half = 0.5 * np.diff(edges)[:, None]
    nodes = (0.5 * (edges[1:] + edges[:-1]))[:, None] + half * g8
    Sn, Pn = nodes.ravel(), np.full(nodes.size, P)
    f = np.asarray(eos.density(Pn, Sn) * eos.temperature(Pn, Sn)).ravel()
    cum = np.concatenate([[0.0], np.cumsum((half * w8 * f.reshape(nodes.shape)).sum(1))])
    H_of_S = CubicSpline(edges, cum - cum[-1])

    def H(T):
        return float(H_of_S(eos.entropy_at_temperature(P, T)))

    xg, wg = np.polynomial.legendre.leggauss(50)
    x = np.concatenate([(xg + 1 + 2 * j) / 8 for j in range(4)])
    E = delta * np.dot(
        np.tile(wg, 4) / 8, [H(T_s + (T_b - T_s) * _profile(shape, v)) for v in x]
    )
    S0 = np.array([eos.entropy_at_temperature(float(p), T_I) for p in s._P_stag_flat])
    seen = {}
    evaluate = s._lid_evaluate
    s._lid_evaluate = lambda entropy: seen.setdefault('ev', evaluate(entropy))
    s.dSdt(0.0, np.concatenate([S0, [delta, E]]))
    surf = seen['ev'].surf
    assert surf.clipped == 0
    assert surf.s == pytest.approx(shape, abs=1e-6)
    assert surf.T_s == pytest.approx(T_s, rel=1e-8)
