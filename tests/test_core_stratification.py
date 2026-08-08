"""Unit tests for ``aragog.core.stratification``.

The diagnostics' contract is the conductive-matching identity: the layer
base returned by the solve must carry exactly the imposed heat flow along
the adiabat, the thickness must vanish for superadiabatic flow and grow
monotonically as the flow drops, and the adiabatic ratio must sit on the
printed model-2 heat-flow anchor (ADR = 1 at 15 TW).
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.core import CoreEnergyBudget, CoreEntropyBudget, GaussianCoreProfiles
from aragog.core.melting import QuadraticMeltingCurve
from aragog.core.stratification import adiabatic_ratio, stratification_depth

pytestmark = pytest.mark.unit

T_C = 4180.0


@pytest.fixture(scope='module')
def ent():
    """Nimmo model-2 entropy budget (Qk = 15.0 TW at 4180 K)."""
    prof = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=136e9,
        alpha=1.25e-5,
        c_p=840.0,
        pressure_mode='labrosse',
    )
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    budget = CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0)
    return CoreEntropyBudget(budget, k_core=130.0)


@pytest.mark.physics_invariant
def test_adr_anchors_on_the_adiabatic_heat_flow(ent):
    """ADR = 1 exactly at the model-2 adiabatic flow (14.97 TW), above one
    for superadiabatic flow, below for subadiabatic."""
    qk = float(ent.adiabatic_heat_flow(T_C))
    assert float(adiabatic_ratio(ent, T_C, qk)) == pytest.approx(1.0, rel=1e-12)
    assert float(adiabatic_ratio(ent, T_C, 17e12)) > 1.0
    assert float(adiabatic_ratio(ent, T_C, 10e12)) < 1.0


@pytest.mark.physics_invariant
def test_depth_satisfies_conductive_matching_and_monotonicity(ent):
    """The returned layer base conducts exactly the imposed flow along the
    adiabat (the defining identity, checked independently), the thickness
    vanishes for superadiabatic flow, spans the core for zero flow, and
    grows monotonically as the heat flow falls."""
    p = ent.budget.profiles
    qk = float(ent.adiabatic_heat_flow(T_C))

    for frac in (0.3, 0.6, 0.9):
        q = frac * qk
        depth = float(stratification_depth(ent, T_C, q))
        r_s = p.r_cmb - depth
        # Defining identity, computed independently of the solver.
        q_ad_at_base = float(
            4.0
            * np.pi
            * r_s**2
            * 130.0
            * (2.0 * r_s * float(p.adiabat(r_s, T_C)) / p.d_scale**2)
        )
        assert q_ad_at_base == pytest.approx(q, rel=1e-9)
        assert 0.0 < depth < p.r_cmb

    assert float(stratification_depth(ent, T_C, 1.05 * qk)) == 0.0
    assert float(stratification_depth(ent, T_C, 0.0)) == pytest.approx(p.r_cmb)

    depths = [float(stratification_depth(ent, T_C, f * qk)) for f in (0.9, 0.6, 0.3, 0.1)]
    assert all(np.diff(depths) > 0.0)  # less flow, deeper stratification
    # Discrimination: at 30% of the adiabatic flow the layer is a
    # substantial fraction of the core, not a boundary sliver.
    assert depths[2] / p.r_cmb > 0.2
