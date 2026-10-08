"""The steady-layer solve from 12 starts that differ at the rounding level."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from aragog.core.stratification import _q_ad
from tests import test_core_layer as base

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('q_tw', [8, 12])
def test_the_steady_layer_solve_holds_for_perturbed_starts(q_tw, seed, monkeypatch):
    shell = base._shell()
    start = np.asarray(shell.adiabatic_profile(base.T_C))
    noise = 1e-12 * np.random.default_rng(seed).standard_normal(start.size) * (seed > 0)
    monkeypatch.setattr(shell, 'adiabatic_profile', lambda t_c: start * (1 + noise))
    t_shell, _ = base._evolve(shell, q_tw * 1e12, 30e3 * base.MYR)
    p = shell.profiles
    r_s = brentq(
        lambda r: float(_q_ad(p, base.K_CORE, r, base.T_C)) - q_tw * 1e12, 1e5, p.r_cmb
    )
    depth = p.r_cmb - float(shell.layer_base(t_shell, base.T_C))
    assert depth / (p.r_cmb - r_s) - 1 == pytest.approx(
        {8: -0.02961, 12: -0.00832}[q_tw], abs=1e-4
    )
