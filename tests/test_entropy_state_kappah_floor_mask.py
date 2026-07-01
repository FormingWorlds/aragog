"""Regression tests for the convective-mask gating of the kappa_h floor on
the numpy (scipy Radau/BDF + parity-harness) solver path.

The phi-modulated kappa_h floor (``kappah_floor``) must act ONLY in
convectively-unstable cells (dS/dr < 0). A stably-stratified, just-frozen
mushy cell at the crystallisation front must NOT receive a floored eddy
diffusivity: the signed convective flux ``rho*T*kappa_h*(-dS/dr)`` would
otherwise inject a spurious sign-flipped flux that pins the cell
sub-solidus (the T_cmb-cliff / front-inversion artifact). SPIDER carries
no kappa_h floor, so floor = 0 in stratified layers is the
SPIDER-consistent limit.

The numpy floor lives in the source helper
``aragog.solver.entropy_state.apply_kappah_floor``, which
``EntropyState.update`` calls with
``f_floor = tanh_weight(phi, phi_rheo, phi_width)`` and
``is_convective = self._dSdr < 0``. The helper computes
``kappah_floor * f_floor * is_convective`` and floors the raw eddy
diffusivity with ``np.maximum``. The companion file
``test_jax_kappah_floor_mask.py`` covers the JAX twin
(``aragog.jax.phase.compute_mlt``).

Strategy. Two complementary checks are used:

* The ``test_numpy_floor_*_matches_jax`` and ``..._strict_stable_vs_convecting``
  checks call the REAL source helper ``apply_kappah_floor`` (via the thin
  ``_numpy_floor`` wrapper that only derives ``f_floor`` and
  ``is_convective``), using the already-verified production JAX
  ``compute_mlt`` as the ground-truth floored kappa_h in a controlled
  stratified-mush state. They assert term-for-term parity at floor = 10 and
  run in every environment (no EOS data required), so a regression that drops
  the convective mask inside ``apply_kappah_floor`` fails them. They are the
  discriminating gated-vs-ungated contrast on the numpy formula.
* ``test_numpy_entropy_state_floor_gated_end_to_end`` exercises the real
  ``EntropyState.update`` numpy MLT block end-to-end when SPIDER P-S tables
  are available; it is skipped otherwise.

Module under test: ``aragog.solver.entropy_state`` (numpy MLT floor block).

Testing standards: ../docs/How-to/build_tests.md. Physics context:
../docs/Explanations/heat_transport.md, ../docs/Explanations/mixing_length.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt
from aragog.solver.entropy_state import apply_kappah_floor
from aragog.utilities import tanh_weight

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Shared synthetic state (mirrors test_jax_kappah_floor_mask.py so the two
# paths are exercised against the SAME controlled stratified-mush state).
# ---------------------------------------------------------------------------

_KAPPAH_FLOOR = 10.0
_PHI_RHEO = 0.4
_PHI_WIDTH = 0.15
_PHI_MUSH = 0.40  # mushy cell sits at the rheological transition (phi_rheo)
_PARAMS = dict(
    kappah_floor=_KAPPAH_FLOOR,
    phi_rheo=_PHI_RHEO,
    phi_width=_PHI_WIDTH,
    eddy_diff_thermal=1.0,
)


def _mesh(n_basic: int) -> MeshArrays:
    """Synthetic mesh with a short mixing length (100 m) so the raw
    (un-floored) MLT diffusivity sits far below ``kappah_floor``, making the
    floor's presence/absence resolvable rather than masked by a large raw
    value."""
    n = n_basic - 1
    r_basic = jnp.linspace(3.5e6, 6.371e6, n_basic)
    r_stag = 0.5 * (r_basic[1:] + r_basic[:-1])
    P_basic = jnp.linspace(120e9, 1e9, n_basic)
    P_stag = 0.5 * (P_basic[1:] + P_basic[:-1])
    return MeshArrays(
        d_dr_matrix=jnp.zeros((n_basic, n)),
        quantity_matrix=jnp.zeros((n_basic, n)),
        area=jnp.ones(n_basic),
        volume=jnp.ones(n_basic),
        radii_basic=r_basic,
        radii_stag=r_stag,
        mixing_length=jnp.ones(n_basic) * 1.0e2,
        mixing_length_sq=jnp.ones(n_basic) * 1.0e4,
        mixing_length_cu=jnp.ones(n_basic) * 1.0e6,
        P_stag=P_stag,
        P_basic=P_basic,
        dP_dr_basic=jnp.gradient(P_basic, r_basic),
        gravity=jnp.full(n_basic, 9.81),
    )


def _phase(n_basic: int, melt_fraction) -> PhaseProperties:
    """Minimal PhaseProperties with physically sane convection-relevant
    fields, matching the JAX-path companion fixture."""
    ones = jnp.ones(n_basic)
    return PhaseProperties(
        temperature=ones * 4000.0,
        density=ones * 4500.0,
        heat_capacity=ones * 1000.0,
        thermal_expansivity=ones * 3.0e-5,
        dTdPs=ones * 1.0e-8,
        melt_fraction=jnp.asarray(melt_fraction, dtype=jnp.float64),
        viscosity=ones * 1.0e3,
        kinematic_viscosity=ones * 1.0,
        thermal_conductivity=ones * 4.0,
        latent_heat=ones * 4.0e5,
        capacitance=ones * 4500.0 * 4000.0,
    )


def _numpy_floor(kh_raw: np.ndarray, phi: np.ndarray, dSdr: np.ndarray) -> np.ndarray:
    """Apply the numpy-path kappa_h floor by calling the REAL source helper
    ``aragog.solver.entropy_state.apply_kappah_floor``.

    The masked-multiply that gates the floor to convecting cells
    (``kappah_floor * f_floor * is_convective`` then ``max(kappa_h_raw, .)``)
    is the source function under test; only the ``f_floor`` and
    ``is_convective`` derivations are reproduced here, from the same
    ``tanh_weight`` symbol and the same ``dSdr < 0`` definition the source
    uses. A regression that drops the ``is_convective`` mask inside
    ``apply_kappah_floor`` therefore fails these parity checks.
    """
    f_floor = tanh_weight(phi, _PHI_RHEO, _PHI_WIDTH)
    is_convective = dSdr < 0.0
    return apply_kappah_floor(kh_raw, _KAPPAH_FLOOR, f_floor, is_convective)


# ---------------------------------------------------------------------------
# Floor-formula parity against the verified JAX reference (no EOS needed).
# ---------------------------------------------------------------------------


def test_numpy_floor_off_in_stable_mushy_cell_matches_jax():
    """A stably-stratified mushy cell (dS/dr >= 0, phi at phi_rheo) must NOT
    receive the numpy kappa_h floor; the floor is gated to convecting cells.

    The production JAX ``compute_mlt`` is the verified reference: its floored
    output in this state is the ground truth the numpy floor must reproduce.
    Discriminating: with kappah_floor = 10 and a meaningfully active
    f_floor(phi=0.4) ~ 0.5, the UN-gated numpy floor would set kappa_h ~ 5 in
    the stable cell, whereas the gated numpy floor leaves it at the raw (~0)
    MLT value, more than an order of magnitude below the floor and matching
    the JAX reference.
    """
    n_basic = 6
    k = 3  # interior cell; idx 0/1 are tied by the SPIDER CMB copy
    ph = _phase(n_basic, [0.6, 0.6, 0.6, _PHI_MUSH, 0.6, 0.6])
    mesh = _mesh(n_basic)
    params = PhaseParams(**_PARAMS)

    # Convecting (dS/dr < 0) everywhere except a single stable cell at k.
    dSdr_jax = jnp.full(n_basic, -1.0e-6).at[k].set(+1.0e-6)
    kh_jax, _ = compute_mlt(dSdr_jax, ph, mesh, params)

    # Numpy path: reproduce the raw (pre-floor) kappa_h from the JAX run by
    # reading it off the convecting reference where no floor is active is not
    # possible (it is floored there), so drive the numpy floor with the SAME
    # raw kappa_h the JAX path produced in the STABLE cell, where the JAX
    # floor is gated off and kh therefore equals the raw value.
    dSdr_np = np.asarray(dSdr_jax)
    phi_np = np.asarray(ph.melt_fraction)
    kh_raw_np = np.asarray(kh_jax).copy()  # raw == floored in the stable cell
    kh_np = _numpy_floor(kh_raw_np, phi_np, dSdr_np)

    f_floor_k = float(tanh_weight(np.asarray(_PHI_MUSH), _PHI_RHEO, _PHI_WIDTH))
    kh_floor_k = _KAPPAH_FLOOR * f_floor_k  # what the UN-gated floor would impose
    # The floor must be meaningfully active at phi_rheo for the test to bite.
    assert f_floor_k > 0.4
    # Gated numpy floor: stable cell keeps its raw (near-zero) value, NOT the
    # floor, and matches the verified JAX reference at that cell.
    assert float(kh_np[k]) < 0.1 * kh_floor_k, (
        f'stable mushy cell received the numpy floor: kappa_h={float(kh_np[k])} '
        f'vs un-gated floor {kh_floor_k}'
    )
    np.testing.assert_allclose(kh_np[k], float(kh_jax[k]), rtol=1e-12, atol=0.0)


def test_numpy_floor_preserved_in_convecting_mushy_cell_matches_jax():
    """The numpy floor must be PRESERVED where it belongs: a convecting mushy
    cell (dS/dr < 0, phi at phi_rheo) still receives the floor, so MLT does
    not numerically freeze where physical convection is expected.

    Discriminating: with a short mixing length the raw diffusivity is below
    the floor, so the convecting cell's numpy kappa_h is pinned AT the floor
    ``kappah_floor * f_floor`` and matches the verified JAX reference, far
    above the stable cell's ~0 in the companion test.
    """
    n_basic = 6
    k = 3
    ph = _phase(n_basic, [0.6, 0.6, 0.6, _PHI_MUSH, 0.6, 0.6])
    mesh = _mesh(n_basic)
    params = PhaseParams(**_PARAMS)

    dSdr_jax = jnp.full(n_basic, -1.0e-6)  # convecting everywhere, including k
    kh_jax, _ = compute_mlt(dSdr_jax, ph, mesh, params)

    # Reconstruct the numpy raw kappa_h (pre-floor) so the numpy floor is
    # applied to a genuine raw value, not the already-floored JAX output.
    # The raw value is below the floor here (short mixing length), so we take
    # a small sub-floor raw and confirm the floor lifts it to the JAX value.
    dSdr_np = np.asarray(dSdr_jax)
    phi_np = np.asarray(ph.melt_fraction)
    kh_raw_np = np.full(n_basic, 1.0e-3)  # raw MLT << floor in this regime
    kh_np = _numpy_floor(kh_raw_np, phi_np, dSdr_np)

    kh_floor_k = _KAPPAH_FLOOR * float(
        tanh_weight(np.asarray(_PHI_MUSH), _PHI_RHEO, _PHI_WIDTH)
    )
    # Convecting numpy cell is pinned at the floor and equals the JAX value.
    assert float(kh_np[k]) == pytest.approx(kh_floor_k, rel=1e-6), (
        f'convecting mushy cell lost the numpy floor: kappa_h={float(kh_np[k])} '
        f'!= floor {kh_floor_k}'
    )
    np.testing.assert_allclose(kh_np[k], float(kh_jax[k]), rtol=1e-6, atol=0.0)


def test_numpy_floor_gating_is_a_strict_stable_vs_convecting_contrast():
    """Direct contrast on the numpy formula: at the SAME phi and floor,
    flipping a single cell from convecting to stable must drop its kappa_h
    from the floor to its raw ~0. This is the discriminating signature that
    distinguishes the gated numpy floor from an un-gated one (which would give
    the floor in BOTH cases).
    """
    n_basic = 6
    k = 3
    phi_np = np.array([0.6, 0.6, 0.6, _PHI_MUSH, 0.6, 0.6])
    kh_raw = np.full(n_basic, 1.0e-3)  # raw MLT << floor

    kh_conv = _numpy_floor(kh_raw, phi_np, np.full(n_basic, -1.0e-6))
    dSdr_stable = np.full(n_basic, -1.0e-6)
    dSdr_stable[k] = +1.0e-6
    kh_stable = _numpy_floor(kh_raw, phi_np, dSdr_stable)

    kh_floor_k = _KAPPAH_FLOOR * float(
        tanh_weight(np.asarray(_PHI_MUSH), _PHI_RHEO, _PHI_WIDTH)
    )
    # Convecting: at the floor. Stable: collapses to the raw ~0. The contrast
    # is the whole point of the gating; an un-gated floor would make them
    # equal.
    assert float(kh_conv[k]) == pytest.approx(kh_floor_k, rel=1e-6)
    assert float(kh_stable[k]) == pytest.approx(1.0e-3, rel=1e-12)
    assert float(kh_stable[k]) < 0.1 * float(kh_conv[k])


# ---------------------------------------------------------------------------
# End-to-end numpy EntropyState.update floor (needs SPIDER P-S EOS tables).
# ---------------------------------------------------------------------------

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


def _build_state(eos, *, kappah_floor: float):
    """Mirror of ``test_entropy_state_branches._build_state`` with the
    kappa_h floor enabled, returning the state plus the mesh."""
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
        conduction=True,
        convection=True,
        eddy_diffusivity_thermal=1.0,
        kappah_floor=kappah_floor,
    )
    return state, mesh


@needs_eos
def test_numpy_entropy_state_floor_gated_end_to_end():
    """End-to-end numpy MLT floor: a mushy basic cell held at phi ~ phi_rheo
    receives the kappa_h floor only when it is convecting (dS/dr < 0), and
    keeps its raw (sub-floor) eddy diffusivity when stably stratified.

    Exercises the real ``EntropyState.update`` numpy floor block via a
    controlled ``dSdr`` argument (gradient-mode), with the cell's melt
    fraction pinned to phi_rheo by placing its staggered entropy at the
    lever-rule midpoint between the EOS solidus and liquidus entropies.

    Discriminating: with kappah_floor = 10 and f_floor(phi_rheo) ~ 0.5, the
    convecting solve floors the basic node at ~5 m^2/s, while the stable solve
    leaves the SAME node at its raw MLT value (orders of magnitude smaller).
    An un-gated floor would floor both, collapsing the contrast.
    """
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    state, mesh = _build_state(eos, kappah_floor=_KAPPAH_FLOOR)

    # Put the WHOLE staggered profile in the mushy window at phi = phi_rheo so
    # every basic node (an average of adjacent staggered cells) is mushy too.
    S_sol = np.asarray(eos.solidus_entropy(mesh.staggered.pressure)).ravel()
    S_liq = np.asarray(eos.liquidus_entropy(mesh.staggered.pressure)).ravel()
    S0 = S_sol + _PHI_MUSH * (S_liq - S_sol)

    n_basic = mesh.N + 1
    k = 5  # interior basic node, clear of the CMB copy (idx 0<-1)

    # Convecting everywhere: floor active at every mushy basic node.
    state.update(S0, time=0.0, dSdr=np.full(n_basic, -1.0e-6))
    kh_conv = np.asarray(state.eddy_diffusivity).ravel().copy()
    phi_basic = np.asarray(state.phase_basic.melt_fraction()).ravel()

    # Stable at cell k only: floor must switch off there, stay on elsewhere.
    dSdr_stable = np.full(n_basic, -1.0e-6)
    dSdr_stable[k] = +1.0e-6
    state.update(S0, time=0.0, dSdr=dSdr_stable)
    kh_stable = np.asarray(state.eddy_diffusivity).ravel().copy()

    # The basic node must actually be mushy for the phi-modulated floor to
    # bite; otherwise f_floor ~ 0 and the test would not discriminate.
    f_floor_k = float(tanh_weight(phi_basic[k], _PHI_RHEO, _PHI_WIDTH))
    assert 0.2 < phi_basic[k] < 0.95, f'basic node {k} not mushy: phi={phi_basic[k]}'
    assert f_floor_k > 0.1, f'f_floor inactive at node {k}: {f_floor_k}'

    kh_floor_k = _KAPPAH_FLOOR * f_floor_k
    # Convecting: pinned at (or above) the phi-modulated floor.
    assert kh_conv[k] >= kh_floor_k * (1.0 - 1e-9), (
        f'convecting mushy node lost the floor: kappa_h={kh_conv[k]} < floor {kh_floor_k}'
    )
    # Stable: floor gated off, node sits well below the floor it would
    # otherwise receive. This is the gated-vs-ungated discriminator.
    assert kh_stable[k] < 0.1 * kh_floor_k, (
        f'stable mushy node still received the floor: kappa_h={kh_stable[k]} '
        f'vs floor {kh_floor_k}'
    )
    assert kh_stable[k] < 0.1 * kh_conv[k]
