"""Regression tests for the convective-mask gating of the kappa_h floor.

The phi-modulated kappa_h floor (``params.kappah_floor``) must act ONLY in
convectively-unstable cells (dS/dr < 0). A stably-stratified, just-frozen
mushy cell at the crystallisation front must NOT receive a floored eddy
diffusivity: the signed convective flux ``rho*T*kappa_h*(-dS/dr)`` would
otherwise inject a spurious sign-flipped flux that pins the cell sub-solidus
(the T_cmb-cliff / front-inversion artifact). SPIDER carries no kappa_h
floor, so floor = 0 in stratified layers is the SPIDER-consistent limit.

Module under test: ``aragog.jax.phase.compute_mlt``.

Testing standards: ../docs/How-to/build_tests.md. Physics context:
../docs/Explanations/heat_transport.md, ../docs/Explanations/mixing_length.md.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt
from aragog.utilities import tanh_weight

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _mesh(n_basic: int) -> MeshArrays:
    """Synthetic mesh with a short mixing length so the raw (un-floored) MLT
    diffusivity sits below the floor, making the floor's effect resolvable."""
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
    """Minimal PhaseProperties with physically sane convection-relevant fields."""
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


_PARAMS = dict(kappah_floor=10.0, phi_rheo=0.4, phi_width=0.15, eddy_diff_thermal=1.0)


def test_kappah_floor_off_in_stable_mushy_cell():
    """A stably-stratified mushy cell (dS/dr >= 0, phi at phi_rheo) must NOT
    receive the kappa_h floor; the floor is gated to convecting cells.

    Discriminating: with kappah_floor = 10 and a meaningfully active
    f_floor(phi=0.4), the un-gated floor would set kappa_h ~ kappah_floor *
    f_floor in the stable cell. The gated floor leaves it at the raw (~0)
    MLT value, an order of magnitude below the floor.
    """
    n_basic = 6
    k = 3  # interior cell; idx 0/1 are tied by the SPIDER CMB copy
    ph = _phase(n_basic, [0.6, 0.6, 0.6, 0.40, 0.6, 0.6])
    mesh = _mesh(n_basic)
    params = PhaseParams(**_PARAMS)

    # Convecting (dS/dr < 0) everywhere except a single stable cell at k.
    dSdr = jnp.full(n_basic, -1.0e-6).at[k].set(+1.0e-6)
    kh, _ = compute_mlt(dSdr, ph, mesh, params)

    f_floor_k = float(tanh_weight(jnp.asarray(0.40), 0.4, 0.15))
    kh_floor_k = 10.0 * f_floor_k  # what the UN-gated floor would impose
    # The floor must be meaningfully active at phi_rheo for the test to bite.
    assert f_floor_k > 0.4
    # Gated: the stable cell keeps its raw (near-zero) MLT value, NOT the floor.
    assert float(kh[k]) < 0.1 * kh_floor_k, (
        f'stable mushy cell received the floor: kappa_h={float(kh[k])} vs '
        f'un-gated floor {kh_floor_k}'
    )


def test_kappah_floor_preserved_in_convecting_mushy_cell():
    """The fix must PRESERVE the floor where it belongs: a convecting mushy
    cell (dS/dr < 0, phi at phi_rheo) still receives the floor, so MLT does
    not numerically freeze where physical convection is expected.

    Discriminating: with a short mixing length the raw diffusivity is below
    the floor, so the convecting cell's kappa_h is pinned AT the floor, far
    above the stable cell's ~0 in the companion test.
    """
    n_basic = 6
    k = 3
    ph = _phase(n_basic, [0.6, 0.6, 0.6, 0.40, 0.6, 0.6])
    mesh = _mesh(n_basic)
    params = PhaseParams(**_PARAMS)

    dSdr = jnp.full(n_basic, -1.0e-6)  # convecting everywhere, including k
    kh, _ = compute_mlt(dSdr, ph, mesh, params)

    kh_floor_k = 10.0 * float(tanh_weight(jnp.asarray(0.40), 0.4, 0.15))
    assert float(kh[k]) == pytest.approx(kh_floor_k, rel=1e-6), (
        f'convecting mushy cell lost the floor: kappa_h={float(kh[k])} != floor {kh_floor_k}'
    )


def test_floor_gating_is_a_strict_stable_vs_convecting_contrast():
    """Direct contrast: at the SAME phi and floor, flipping a single cell from
    convecting to stable must drop its kappa_h from the floor to ~0. This is
    the discriminating signature that distinguishes the gated floor from the
    un-gated one (which would give the floor in BOTH cases)."""
    n_basic = 6
    k = 3
    ph = _phase(n_basic, [0.6, 0.6, 0.6, 0.40, 0.6, 0.6])
    mesh = _mesh(n_basic)
    params = PhaseParams(**_PARAMS)

    kh_conv, _ = compute_mlt(jnp.full(n_basic, -1.0e-6), ph, mesh, params)
    kh_stable, _ = compute_mlt(jnp.full(n_basic, -1.0e-6).at[k].set(+1.0e-6), ph, mesh, params)

    kh_floor_k = 10.0 * float(tanh_weight(jnp.asarray(0.40), 0.4, 0.15))
    # Convecting: at the floor. Stable: collapses to ~0. The ratio is the
    # whole point of the gating, an un-gated floor would make them equal.
    assert float(kh_conv[k]) == pytest.approx(kh_floor_k, rel=1e-6)
    assert float(kh_stable[k]) < 0.1 * float(kh_conv[k])
