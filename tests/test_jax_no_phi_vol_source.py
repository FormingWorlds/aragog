"""JAX-side negative regression: no explicit Φ_vol source on heating.

The Soucasse §1.2 form

    Phi_vol = rho · g · (1/rho_m - 1/rho_s) · (j_cm + j_gm)

is NOT added on top of the JAX ``compute_fluxes`` heating array.
Volumetric work is already implicit in the divergence of the
Δh-weighted mass-flux contributions to ``flux_out.heat_flux`` (chain
rule on Δh = Δu + P·Δv with hydrostatic ∂P/∂r = -ρg). Adding it
explicitly would double-count (Bower 2018 §3, SPIDER energy.c). The
JAX RHS is the production CHILI integration path under
``solver_method='cvode'`` + ``use_jax_jacobian=True``, so a separate
JAX-side regression is required even when the numpy path is
covered in ``test_entropy_verification.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

jax.config.update('jax_enable_x64', True)

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

pytestmark = pytest.mark.unit


def _build_synthetic_mesh(N=24, R_cmb=3.48e6, R_surf=6.371e6, P_cmb=135.0e9, P_surf=1.0e5):
    """Synthetic Earth-like mesh + JAX MeshArrays for compute_fluxes calls."""
    from aragog.jax.phase import MeshArrays

    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    P_stag = np.linspace(P_cmb, P_surf, N)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_basic = np.interp(r_basic, r_stag, P_stag)
    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_cmb, R_surf - r_basic), 1.0)

    d_dr_mat = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr_mat[i, i - 1] = -1.0 / dr[i - 1]
        d_dr_mat[i, i] = 1.0 / dr[i - 1]
    d_dr_mat[0, :] = d_dr_mat[1, :]
    d_dr_mat[-1, :] = d_dr_mat[-2, :]

    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5

    g = 9.81
    mesh_jax = MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr_mat),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(area),
        volume=jnp.asarray(volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.asarray(P_stag),
        P_basic=jnp.asarray(P_basic),
        gravity=jnp.full(r_basic.shape, g),
        gravity_stag=jnp.full(r_stag.shape, g),
    )
    return r_stag, P_stag, mesh_jax


@needs_eos
def test_no_phi_vol_added_in_mushy_zone_jax():
    """Mushy regime where the historical Φ_vol would have been large.

    Drives a non-trivial entropy gradient across the mushy band so
    j_grav ~ 1e-3 kg/m²/s and j_mix ~ 1e-3 kg/m²/s; pre-deletion the
    explicit Soucasse §1.2 source would have produced
    H_dil ~ 1e-8 W/kg here. We assert max|heating| < 1e-15 W/kg with
    no radio or tidal contribution wired in, which discriminates
    against any regression that re-introduces even a fractional copy
    of Φ_vol on the JAX path.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    r_stag, P_stag, mesh_jax = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)

    # Mushy IC: between solidus and liquidus, with a slight gradient
    # to drive convective instability so jgrav and jmix are non-trivial.
    S_sol = np.asarray(eos_jax.solidus_entropy(jnp.asarray(P_stag)))
    S_liq = np.asarray(eos_jax.liquidus_entropy(jnp.asarray(P_stag)))
    S_init = 0.45 * S_sol + 0.55 * S_liq
    S_init = S_init - 5.0 * np.linspace(0.0, 1.0, S_init.size)

    params = PhaseParams(grav_sep=True, mixing=True, grain_size=0.1)
    heating_in = jnp.zeros(S_init.size)
    out = compute_fluxes(
        jnp.asarray(S_init),
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    H_jax = np.asarray(out.heating).ravel()

    max_abs_jax = float(np.max(np.abs(H_jax)))
    assert max_abs_jax < 1e-15, (
        f'JAX heating array contains a non-zero contribution '
        f'(max|H|={max_abs_jax:.3e} W/kg) where only radio/tidal '
        f'should be present; an explicit Φ_vol source has been re-introduced.'
    )


@needs_eos
def test_radio_only_heating_in_mushy_zone_jax():
    """In the mushy regime, the only heating contribution is the
    radio array passed in. Output must equal input exactly when
    grav_sep and mixing are on but no other source is wired.

    Discriminates against a regression that re-introduces an
    unconditional Φ_vol: such a regression would add ~1e-8 W/kg in
    the mushy zone, breaking the radio-only equality.
    """
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import PhaseParams, compute_fluxes

    r_stag, P_stag, mesh_jax = _build_synthetic_mesh(N=24)
    eos_jax = EntropyEOS_JAX(EOS_DIR)
    S_sol = np.asarray(eos_jax.solidus_entropy(jnp.asarray(P_stag)))
    S_liq = np.asarray(eos_jax.liquidus_entropy(jnp.asarray(P_stag)))
    S_init = 0.45 * S_sol + 0.55 * S_liq
    S_init = S_init - 5.0 * np.linspace(0.0, 1.0, S_init.size)

    params = PhaseParams(grav_sep=True, mixing=True, grain_size=0.1)
    radio_value = 1.234e-9  # per-cell uniform [W/kg]
    heating_in = jnp.full(S_init.size, radio_value)
    out = compute_fluxes(
        jnp.asarray(S_init),
        0.0,
        eos_jax,
        params,
        mesh_jax,
        heating_in,
    )
    H_out = np.asarray(out.heating)
    H_in = np.asarray(heating_in)
    # Output must equal input exactly; no Φ_vol contribution.
    np.testing.assert_allclose(H_out, H_in, rtol=0.0, atol=1e-30)
    # Also discriminate against radio being silently dropped: max|H|
    # must equal the radio value.
    assert float(np.max(np.abs(H_out))) == pytest.approx(radio_value, rel=1e-12), (
        f'Expected heating == {radio_value} (radio uniform), got '
        f'max|H|={float(np.max(np.abs(H_out))):.3e}'
    )
