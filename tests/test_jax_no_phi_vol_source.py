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


# ---- Schema-level negative regressions ------------------------------------
#
# These tests do NOT need the JAX runtime, the EOS tables, or any
# numerical computation. They simply assert that the dilatation
# surface (config field, SolverOutput attribute, JAX heating path)
# stays deleted. They run fast and discriminate against the most
# obvious regressions: someone re-adding the field "for backward
# compatibility" without realising it would re-introduce the
# 2x-overshoot heat-pump quasi-equilibrium documented in
# finding_2026_05_03_phi_vol_double_count_verdict.md.


def test_solver_output_has_no_Q_dil_total_field():
    """SolverOutput must not expose ``Q_dil_total`` (or any Φ_vol-
    derived total power). The PROTEUS wrapper used to read this
    field; readers that try ``getattr(out, 'Q_dil_total')`` should
    fail loudly so the regression is caught at PR review.
    """
    from aragog.solver.entropy_solver import SolverOutput

    field_names = {f.name for f in SolverOutput.__dataclass_fields__.values()}
    forbidden = {'Q_dil_total', 'Q_dil_W', 'F_dil', 'F_dil_W'}
    found = forbidden & field_names
    assert not found, (
        f'SolverOutput re-introduced dilatation fields: {sorted(found)}. '
        f'See aragog 0948279 / dcd7f37 for the deletion rationale.'
    )


def test_solver_output_has_no_step_dE_Q_dil_J_field():
    """The per-call energy integral list must not include
    step_dE_Q_dil_J. Only the four physical integrals
    (F_int, F_cmb, Q_radio, Q_tidal) are valid sources.

    Discriminator: a regression that re-added Q_dil to
    _compute_step_energy_integrals and exposed it on SolverOutput
    would silently restore the +200 to +2000 E_residual_frac
    plateau seen in the 7-cell energy-diagnostic matrix.
    """
    from aragog.solver.entropy_solver import SolverOutput

    field_names = {f.name for f in SolverOutput.__dataclass_fields__.values()}
    assert 'step_dE_Q_dil_J' not in field_names, (
        'SolverOutput re-introduced step_dE_Q_dil_J; '
        'the dilatation per-call integral is gone for a reason.'
    )
    # Also assert the four valid integrals are still there so a
    # regression that deletes them all (instead of just dilatation)
    # is caught.
    expected_integrals = {
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_J',
        'step_dE_Q_tidal_J',
    }
    missing = expected_integrals - field_names
    assert not missing, f'SolverOutput is missing required energy integrals: {sorted(missing)}'


def test_energy_parameters_does_not_accept_dilatation_kwarg():
    """The legacy ``dilatation`` boolean was removed from
    ``_EnergyParameters`` in aragog c8bc611. Any TOML still passing
    it must fail at parser construction so the silent no-op is
    impossible.

    Edge case: include several other plausibly-related keywords
    (``Phi_vol``, ``Phi_vol_active``) in the same regression sweep
    so a regression that re-introduces ANY dilatation alias is
    caught.
    """
    from aragog.parser import _EnergyParameters

    base_kwargs = dict(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    for forbidden in ('dilatation', 'Phi_vol', 'Phi_vol_active', 'phi_vol'):
        with pytest.raises(TypeError, match='unexpected keyword'):
            _EnergyParameters(**base_kwargs, **{forbidden: False})  # type: ignore[arg-type]
