"""Verification test B2: Newton vs Brentq root on skin balance."""

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann as SIGMA
from scipy.optimize import brentq

from aragog.surface_skin import skin_temperature

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

def test_skin_temperature_matches_brentq_to_1e12():
    rng = np.random.default_rng(12345)
    n_cases = 1000
    T_top = rng.uniform(200.0, 4000.0, n_cases)
    T_eq = rng.uniform(50.0, 2000.0, n_cases)
    G = 10.0 ** rng.uniform(-5.0, 3.0, n_cases)
    eps = rng.uniform(0.1, 1.0, n_cases)

    T_newton = skin_temperature(T_top, G, eps, T_eq, SIGMA)
    for i in range(n_cases):
        tt, ge, ep, te = T_top[i], G[i], eps[i], T_eq[i]
        f = lambda T: ep * SIGMA * (T**4 - te**4) - ge * (tt - T)
        root = brentq(f, 0.0, max(tt, te), xtol=1e-13, rtol=1e-15)
        assert abs(T_newton[i] - root) / root < 1e-12
