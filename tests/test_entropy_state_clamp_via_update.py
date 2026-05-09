"""End-to-end coverage of the EntropyState Cp-clamp warnings.

Companion file to ``tests/test_entropy_state_clamp_warnings.py``, which
exercises the throttling helper ``EntropyState._maybe_warn_cp_floor``
directly with stubbed phase evaluators. This file drives the SAME
warning code paths through the full ``EntropyState.update()`` call chain
on a real PALEOS-style EOS, so the production callsites inside
``update()`` (around the conduction Cp >= 100 J/kg/K guard and the MLT
Cp >= 1 J/kg/K guard) are covered. Without this file the helper is
covered but the wiring from ``update()`` into it is not.

These tests are gated on the SPIDER P-S table directory being present
(``ARAGOG_TEST_EOS_DIR`` env var, or the in-repo CI default). They
mirror the gating convention used by ``tests/test_entropy_verification.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest

EOS_DIR = Path(
    os.environ.get(
        'ARAGOG_TEST_EOS_DIR',
        '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
    )
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)

LOGGER_NAME = 'fwl.aragog.solver.entropy_state'

pytestmark = pytest.mark.unit


def _build_state_for_update(
    eos,
    *,
    conduction: bool = True,
    convection: bool = True,
):
    """Build a real-EOS-backed ``EntropyState`` configured for an
    end-to-end ``update()`` call.

    Mirrors ``test_entropy_verification.make_mesh`` + ``make_state`` but
    inlined here so the file is self-contained (no cross-test imports).
    """
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    N = 30
    R_cmb, R_surf = 3480e3, 6371e3
    P_cmb, P_surf = 135e9, 1e5

    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_stag = np.linspace(P_cmb, P_surf, N)
    P_basic = np.interp(r_basic, r_stag, P_stag)

    class _Mesh:
        pass

    class _Sub:
        pass

    mesh = _Mesh()
    mesh.basic = _Sub()
    mesh.staggered = _Sub()
    mesh.basic.radii = r_basic
    mesh.staggered.radii = r_stag
    mesh.basic.area = 4.0 * np.pi * r_basic**2
    mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.minimum(r_basic - R_cmb, R_surf - r_basic)
    mesh.basic.mixing_length = np.maximum(ml, 1.0)
    mesh.basic.mixing_length_squared = mesh.basic.mixing_length**2
    mesh.basic.mixing_length_cubed = mesh.basic.mixing_length**3
    mesh.basic.pressure = P_basic
    mesh.staggered.pressure = P_stag
    mesh.basic.mass_radii = r_basic
    mesh.staggered.mass_radii = r_stag
    mesh.dxidr = np.ones_like(r_basic)

    def quantity_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[0], out[-1] = q[0], q[-1]
        out[1:-1] = 0.5 * (q[:-1] + q[1:])
        return out

    def d_dr_at_basic_nodes(q):
        q = np.asarray(q).flatten()
        out = np.zeros(N + 1)
        out[1:-1] = np.diff(q) / dr
        out[0], out[-1] = out[1], out[-2]
        return out

    mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
    mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes
    mesh.dr = dr
    mesh.N = N

    phase_stag = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(mesh.staggered.pressure)
    phase_basic = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_basic.set_pressure(mesh.basic.pressure)

    class _Eval:
        pass

    evaluator = _Eval()
    evaluator.mesh = mesh

    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=conduction,
        convection=convection,
    )
    return state, mesh, phase_basic


@needs_eos
def test_update_with_healthy_EOS_emits_no_clamp_warnings(caplog):
    """End-to-end ``update()`` on a real PALEOS-style EOS with both
    conduction and convection enabled. The production callsites for
    ``_maybe_warn_cp_floor`` (conduction Cp >= 100 and MLT Cp >= 1)
    must execute on every RHS evaluation but stay silent on healthy
    Cp values. This is the negative regression that catches a
    sign-flip in the trigger or a misnamed flag attribute that would
    fire the warning every step.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    state, _mesh, _ = _build_state_for_update(eos)
    S0 = np.linspace(3400.0, 3000.0, _mesh.N)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state.update(S0, time=0.0)

    clamp_records = [
        r for r in caplog.records if 'EntropyState' in r.message and 'Cp' in r.message
    ]
    assert clamp_records == [], (
        f'unexpected Cp-clamp warnings on healthy EOS: {[r.message for r in clamp_records]}'
    )
    assert state._cp_floor_warned is False
    assert state._cp_mlt_floor_warned is False


@needs_eos
def test_update_fires_conduction_and_mlt_cp_warnings_when_Cp_below_floor(caplog):
    """End-to-end ``update()`` with a monkey-patched ``heat_capacity``
    that returns 0.5 J/kg/K on every basic node. Both production
    callsites of ``_maybe_warn_cp_floor`` (conduction Cp >= 100 and
    MLT Cp >= 1) must fire on the FIRST update() call. Throttling is
    verified by a second update() that emits no further warnings even
    though the same low-Cp condition still holds.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    state, mesh, phase_basic = _build_state_for_update(eos)
    S0 = np.linspace(3400.0, 3000.0, mesh.N)

    # Monkey-patch the basic-node heat_capacity AFTER the state is
    # constructed; the patched lambda shadows the bound method on the
    # instance and is called by every state.update() RHS evaluation.
    n_basic = mesh.basic.radii.size
    phase_basic.heat_capacity = lambda: np.full(n_basic, 0.5)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state.update(S0, time=0.0)
        cond_first = [r for r in caplog.records if 'EntropyState conduction' in r.message]
        mlt_first = [r for r in caplog.records if 'EntropyState MLT' in r.message]
        # Second call: throttle must hold; no new warnings emitted.
        n_after_first = len(caplog.records)
        state.update(S0, time=0.0)
        new_records = caplog.records[n_after_first:]

    assert len(cond_first) == 1, f'conduction warning fired {len(cond_first)} times'
    assert len(mlt_first) == 1, f'MLT warning fired {len(mlt_first)} times'
    assert state._cp_floor_warned is True
    assert state._cp_mlt_floor_warned is True
    # Throttle held across the second update().
    new_clamp_records = [
        r for r in new_records if 'EntropyState' in r.message and 'Cp' in r.message
    ]
    assert new_clamp_records == [], (
        f'throttle leaked on second update(): {[r.message for r in new_clamp_records]}'
    )
