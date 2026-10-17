"""Regression tests for the floor-mask gating of the kappa_h floor.

The phi-modulated kappa_h floor (``params.kappah_floor``) must act ONLY in
cells with dS/dr < D (D = ``params.convective_threshold``). A mushy
cell with a stable gradient above D, such as a just-frozen cell at the
crystallisation front, must NOT receive a floored eddy diffusivity: the
signed convective flux ``rho*T*kappa_h*(-dS/dr)`` would otherwise inject a
spurious sign-flipped flux that pins the cell sub-solidus (the T_cmb-cliff /
front-inversion artifact). SPIDER carries no kappa_h
floor, so floor = 0 in stratified layers is the SPIDER-consistent limit.

Module under test: ``aragog.jax.phase.compute_mlt``.

Testing standards: ../docs/How-to/build_tests.md. Physics context:
../docs/Explanations/heat_transport.md, ../docs/Explanations/mixing_length.md.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt
from aragog.utilities import tanh_weight

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30, method='signal')]


def _mesh(n_basic: int, ml: float = 1.0e2) -> MeshArrays:
    """Synthetic mesh; the default short mixing length keeps the raw
    (un-floored) MLT diffusivity below the floor, so the floor is resolvable."""
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
        mixing_length=jnp.full(n_basic, ml),
        mixing_length_sq=jnp.full(n_basic, ml**2),
        mixing_length_cu=jnp.full(n_basic, ml**3),
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
    """A stably-stratified mushy cell (dS/dr above D, phi at phi_rheo) must NOT
    receive the kappa_h floor; the floor is gated to cells with dS/dr < D.

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
    dSdr = jnp.full(n_basic, -1.0e-6).at[k].set(+1.0e-4)
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
    kh_stable, _ = compute_mlt(jnp.full(n_basic, -1.0e-6).at[k].set(+1.0e-4), ph, mesh, params)

    kh_floor_k = 10.0 * float(tanh_weight(jnp.asarray(0.40), 0.4, 0.15))
    # Convecting: at the floor. Stable: collapses to ~0. The ratio is the
    # whole point of the gating, an un-gated floor would make them equal.
    assert float(kh_conv[k]) == pytest.approx(kh_floor_k, rel=1e-6)
    assert float(kh_stable[k]) < 0.1 * float(kh_conv[k])


# Basic-node entropy gradients [J/kg/K/m] across D = 1e-6 (index 0 is the CMB copy of 1).
_DSDR_D = jnp.array([-1.0e-4, -1.0e-4, -1.0e-7, -1.0e-31, 0.0, 1.0e-31, 5.0e-7, 2.0e-6, 1.0e-4])
# kappa_h on _DSDR_D (mixing length 1e5 m, phi = 0.4) from compute_mlt of aragog 26.10.10,
# whose switch is dS/dr < 0.
_KH_SIGN_SWITCH = [857758.707329748, 857758.707329748, 27124.711980037686, 5.0, 0, 0, 0, 0, 0]


def test_stable_node_below_threshold_gets_the_floor_only():
    """A node with 0 <= dS/dr < D gets the phase-modulated floor and no
    mixing-length eddy diffusivity; a node above D gets neither.

    Discriminating: under the dS/dr < 0 switch the nodes at 0, 1e-31 and
    5e-7 get kappa_h = 0 (``_KH_SIGN_SWITCH``); with the drive in the band the
    node at 5e-7 would get about 6e4 m^2/s.
    """
    n = _DSDR_D.size
    ph = _phase(n, [0.4] * n)
    kh, kc = compute_mlt(_DSDR_D, ph, _mesh(n, ml=1.0e5), PhaseParams(**_PARAMS))
    kh_floor = 10.0 * float(tanh_weight(jnp.asarray(0.4), 0.4, 0.15))

    np.testing.assert_allclose(np.asarray(kh[4:7]), kh_floor, rtol=1e-12)
    assert np.all(np.asarray(kc[4:7]) == 0.0)
    assert float(kh[7]) == 0.0 and float(kh[8]) == 0.0


def test_zero_threshold_is_the_sign_switch():
    """D = 0 gives the kappa_h of the dS/dr < 0 switch on every node."""
    n = _DSDR_D.size
    ph = _phase(n, [0.4] * n)
    params = PhaseParams(**_PARAMS, convective_threshold=0.0)
    kh, _ = compute_mlt(_DSDR_D, ph, _mesh(n, ml=1.0e5), params)
    np.testing.assert_allclose(np.asarray(kh), _KH_SIGN_SWITCH, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('D', [1.0e-6, 3.0e-7, 3.0e-6])
def test_threshold_is_strict_at_D(D):
    """A node just below D gets the floor (and no chemical diffusivity) and a node
    at D does not, for the default and two other values of D; the floor mask is
    ``dS/dr < D``, not ``<=``."""
    x = jnp.array([-1.0e-4, -1.0e-4, np.nextafter(D, 0.0), D, np.nextafter(D, 1.0), -1.0e-4])
    params = PhaseParams(**_PARAMS, convective_threshold=D)
    kh, kc = compute_mlt(x, _phase(x.size, [0.4] * x.size), _mesh(x.size, ml=1.0e5), params)
    kh_floor = 10.0 * float(tanh_weight(jnp.asarray(0.4), 0.4, 0.15))
    assert float(kh[2]) == pytest.approx(kh_floor, rel=1e-12) and float(kc[2]) == 0.0
    assert float(kh[3]) == 0.0 and float(kh[4]) == 0.0 and float(kc[3]) == 0.0


def test_cmb_node_follows_the_node_above_across_0_and_D():
    """The CMB node takes kappa_h of the node above it, whatever its own gradient. Just
    below dS/dr = 0 that node has the kappa_h and kappa_c of a gradient D, from 0 to
    below D the floor alone and no kappa_c, from D on neither."""
    D = 1.0e-6
    params = PhaseParams(**_PARAMS, convective_threshold=D)
    ph, mesh = _phase(4, [0.4] * 4), _mesh(4, ml=1.0e5)
    kh_floor = 10.0 * float(tanh_weight(jnp.asarray(0.4), 0.4, 0.15))
    # -1e-300, not the subnormal nextafter(0, -1), which XLA flushes to zero on the CPU.
    xs = (-1.0e-300, 0.0, np.nextafter(D, 0.0), D)
    for cmb in (-1.0e-4, 1.0e-4):
        out = [compute_mlt(jnp.array([cmb, x1, -D, -1.0e-4]), ph, mesh, params) for x1 in xs]
        assert all(float(kh[0]) == float(kh[1]) for kh, _ in out)
        (kh, kc), band, edge, (kh_d, kc_d) = out
        assert float(kh[1]) == pytest.approx(float(kh[2]), rel=1e-12)
        assert float(kh[1]) > 1.0e3 * kh_floor
        assert float(kc[1]) == pytest.approx(float(kc[2]), rel=1e-12)
        for kh_b, kc_b in (band, edge):
            assert (
                float(kh_b[1]) == pytest.approx(kh_floor, rel=1e-12) and float(kc_b[1]) == 0.0
            )
        assert float(kh_d[1]) == 0.0 and float(kc_d[1]) == 0.0


def test_unstable_node_gets_at_least_the_drive_of_D():
    """An unstable node with |dS/dr| < D takes the kappa_h and kappa_c of a gradient
    D; a node with |dS/dr| > D keeps its own value, and D = 0 keeps every value.

    Discriminating: without the minimum drive the node at -1e-7 has 2.7e4 m^2/s
    (``_KH_SIGN_SWITCH``) and the node at -1e-31 has 5 (the floor).
    """
    x = jnp.array([-1.0e-4, -1.0e-4, -1.0e-6, -1.0e-7, -1.0e-31])
    ph, mesh = _phase(x.size, [0.4] * x.size), _mesh(x.size, ml=1.0e5)
    kh, kc = compute_mlt(x, ph, mesh, PhaseParams(**_PARAMS, convective_threshold=1.0e-6))
    np.testing.assert_allclose(np.asarray(kh[3:]), float(kh[2]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(kc[3:]), float(kc[2]), rtol=1e-12)
    assert float(kh[0]) == pytest.approx(_KH_SIGN_SWITCH[0], rel=1e-12)
    kh0, _ = compute_mlt(x, ph, mesh, PhaseParams(**_PARAMS, convective_threshold=0.0))
    assert float(kh0[3]) == pytest.approx(_KH_SIGN_SWITCH[2], rel=1e-12)
