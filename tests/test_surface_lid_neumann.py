"""Validation of the sub-grid lid closure against the Neumann solution of crust growth over melt.

The case drives the lid functions of ``aragog.surface_lid`` through time with the fixed material
properties and helpers of ``tests/test_surface_lid_validation.py``; it runs about 2 minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog import surface_lid as sl
from tests.conftest import assert_legend_clear
from tests.test_surface_lid_validation import (
    CP,
    KAPPA,
    T_B,
    T_S,
    YR,
    _dirichlet,
    _energy,
    _evolve,
    _neumann_lambda,
)

pytestmark = [pytest.mark.slow, pytest.mark.timeout(600)]

T_V1 = np.array([1e8, 1e10, 1e12])


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
