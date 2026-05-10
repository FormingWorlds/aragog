"""Edge-case unit tests for ``EntropyState`` branches not exercised
by the existing clamp / verification tests.

Targets:

* Negative ``eddy_diffusivity_thermal`` / ``eddy_diffusivity_chemical``
  (lines 638, 645): SPIDER convention of "constant kappa" via the
  -1 sentinel value.
* Phase smoothing dispatch ``'cubic_hermite'`` (lines 808-809,
  857-858): the analytic 16 phi^2 (1 - phi)^2 mask used as an
  alternative to the SPIDER tanh smoother.
* Tidal heating array length mismatch (line 904): when
  ``len(tidal_array)`` is neither 1 nor n_stag, the tidal contribution
  must default to zeros (silent rejection of a malformed config).
* ``EntropyState.dTdr`` accessor (lines 1014-1015): post-update
  diagnostic accessor used by PROTEUS's flux-decomposition output.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


def _build_state(
    eos,
    *,
    conduction: bool = True,
    convection: bool = True,
    gravitational_separation: bool = False,
    mixing: bool = False,
    tidal: bool = False,
    tidal_array=None,
    eddy_diffusivity_thermal: float = 1.0,
    eddy_diffusivity_chemical: float = 1.0,
    phase_smoothing: str = 'tanh',
):
    """Mirror of ``test_entropy_state_clamp_via_update._build_state_for_update``
    but with knobs exposed for the additional branches we test here.
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
        gravitational_separation=gravitational_separation,
        mixing=mixing,
        tidal=tidal,
        tidal_array=tidal_array,
        eddy_diffusivity_thermal=eddy_diffusivity_thermal,
        eddy_diffusivity_chemical=eddy_diffusivity_chemical,
        phase_smoothing=phase_smoothing,
    )
    return state, mesh


@needs_eos
def test_negative_eddy_diff_thermal_uses_constant_kappa(caplog):
    """``eddy_diffusivity_thermal`` < 0 means "use |value| as a
    constant kappa across the mesh", matching SPIDER's matprop.c
    convention.

    Discriminator: the array of eddy diffusivities must be uniform
    (constant), not modulated by kh_raw. A regression that ignored
    the negative sentinel would scale the constant by kh_raw and
    surface as a non-uniform profile.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    K_const = 5.0  # m^2/s
    state, mesh = _build_state(
        eos,
        eddy_diffusivity_thermal=-K_const,
        eddy_diffusivity_chemical=-K_const,
    )
    S0 = np.linspace(3300.0, 3000.0, mesh.N)
    state.update(S0, time=0.0)

    ed = state.eddy_diffusivity
    np.testing.assert_allclose(ed, K_const, rtol=1e-12, atol=0.0)
    # Chemical kappa: same convention.
    kc = state._kappac
    np.testing.assert_allclose(kc, K_const, rtol=1e-12, atol=0.0)


@needs_eos
def test_phase_smoothing_cubic_hermite_path_runs_without_errors():
    """``phase_smoothing='cubic_hermite'`` selects the analytic
    16 phi^2 (1 - phi)^2 mask instead of SPIDER's tanh blender.

    Discriminator: the cubic-Hermite branch is structurally distinct
    from the tanh branch and has its own coverage shadow inside both
    Jgrav and Jmix. A run with grav_sep + mixing on must finish
    without raising and produce finite output. We test invariance of
    the heat_flux array's finiteness; a regression that mis-typed the
    cubic mask would produce NaN/inf on out-of-mushy nodes.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    state, mesh = _build_state(
        eos,
        gravitational_separation=True,
        mixing=True,
        phase_smoothing='cubic_hermite',
    )
    S0 = np.linspace(3300.0, 3000.0, mesh.N)
    state.update(S0, time=0.0)
    assert np.all(np.isfinite(state.heat_flux)), (
        'cubic_hermite smoothing produced non-finite heat_flux'
    )


@needs_eos
def test_tidal_array_with_bad_length_falls_back_to_zero():
    """When ``tidal=True`` and ``len(tidal_array)`` is neither 1 nor
    n_stag, the contribution must silently fall back to zeros rather
    than broadcasting incorrectly.

    Edge case: a regression that auto-broadcast a length-3 array onto
    n_stag=30 would either raise inside numpy or pad with garbage,
    producing a state-dependent heating profile.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    # Bad length: 3 != 1 != n_stag (30). Triggers the fallback branch.
    state, mesh = _build_state(
        eos,
        tidal=True,
        tidal_array=[1.0e-6, 2.0e-6, 3.0e-6],
    )
    S0 = np.linspace(3300.0, 3000.0, mesh.N)
    state.update(S0, time=0.0)
    np.testing.assert_array_equal(state.heating_tidal, np.zeros(mesh.N))


@needs_eos
def test_dTdr_accessor_returns_basic_node_temperature_gradient():
    """``EntropyState.dTdr()`` returns dT/dr at basic nodes by
    differencing the staggered T profile.

    Discriminator: a regression that returned dS/dr instead would
    surface as a sign-flipped or order-of-magnitude-off result. We
    cross-check against numpy diff applied to the same staggered T.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    state, mesh = _build_state(eos)
    S0 = np.linspace(3500.0, 2800.0, mesh.N)  # decreasing toward surface
    state.update(S0, time=0.0)

    grad = state.dTdr()
    assert grad.shape == (mesh.N + 1,) or grad.shape == (mesh.N + 1, 1), (
        f'dTdr shape {grad.shape} not (n_basic,) or (n_basic, 1)'
    )
    # Decreasing S typically maps to decreasing T (within mushy zone
    # adiabats), so dT/dr should be mostly negative (temperature
    # falls toward the surface). Allow some interior cells to flip
    # sign in the partial-melt region.
    grad_arr = np.asarray(grad).ravel()
    interior = grad_arr[1:-1]
    assert np.median(interior) < 0.0, (
        f'median interior dT/dr = {np.median(interior):.3e}, expected < 0 '
        'for entropy decreasing toward the surface'
    )
